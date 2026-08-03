"""V20e tangent-constrained, six-fold response-fitting smoke.

V20e is the follow-up to the V20d boundary-collapse no-go.  It starts from
the same fixed response ``w0=(0, 1, 0)`` but solves the natural quadratic in
the exact tangent cone, follows the complete feasible ray, and chooses a
single convex fraction with leave-one-fit-family-out validation.  The two
held Reed/Sundial roles remain a historically reused diagnostic and cannot
authorize a fresh-family, serving, compression, speed, or fidelity claim.

The report is write-once 0600 scalar/hash-only JSON.  No raw gradients,
teacher rows, logits, H4 tensors, targets, coordinates, or provider sidecar is
serialized.
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

from . import complete_h4_fisher_tangent_response as _core
from .complete_h4_autonomous_residual import _tensor_sha256 as _provider_tensor_sha256
from .complete_h4_fisher_continuous_transfer import (
    AutonomousCompleteH4FisherContinuousTransferProvider,
    build_autonomous_complete_h4_fisher_continuous_constant_control,
    build_autonomous_complete_h4_fisher_continuous_transfer,
    fisher_continuous_bilinear_box_max_abs,
    fisher_continuous_bilinear_corner_values,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_continuous_response_smoke as _v20c
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as _v20b
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from . import gemma3_l3_l4_complete_h4_natural_response_smoke as _v20d
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_tangent_response_smoke",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V20D_OUTPUT = _v20d.DEFAULT_OUTPUT
_V20D_LOGICAL_SHA256 = "8876f73ac56774d8c3cb621952999565988116a4a88379a13988dd6ee04cc78b"
_V20D_FILE_SHA256 = "8f6c7e6bc18141e85c0e44e022a81ba5c0c1ae7edc3b29557ea6a37d304f79f1"
_V20D_CLASSIFICATION = "natural_response_pair_smoke_failed"

DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-tangent-response-pair-smoke-"
    "r16-k256-a-fit16-dev-v20e.json"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_tangent_response_smoke.v20e"
_FORMAT_VERSION = 21
_REPORT_DOMAIN = b"fisher-graph:tangent-response-smoke-report:v20e\0"
_SOURCE_DOMAIN = b"fisher-graph:tangent-response-smoke-source:v20e\0"
_INITIAL_DOMAIN = b"fisher-graph:tangent-response-initial-vjp:v20e\0"
_INITIAL_EXECUTION_DOMAIN = b"fisher-graph:tangent-response-initial-execution:v20e\0"
_PROVIDER_SEED_DOMAIN = b"fisher-graph:tangent-response-provider-seed:v20e\0"
_MANIFEST_DOMAIN = b"fisher-graph:tangent-response-cv-provider-manifest:v20e\0"
_CV_EXECUTION_DOMAIN = b"fisher-graph:tangent-response-cv-execution:v20e\0"
_CV_FOLD_DOMAIN = b"fisher-graph:tangent-response-cv-fold:v20e\0"
_SELECTION_DOMAIN = b"fisher-graph:tangent-response-two-law-selection:v20e\0"
_FINAL_TRACE_DOMAIN = b"fisher-graph:tangent-response-final-trace:v20e\0"
_PROVIDER_RECEIPT_DOMAIN = b"fisher-graph:tangent-response-provider:v20e\0"
_PROVIDER_BUNDLE_DOMAIN = b"fisher-graph:tangent-response-provider-bundle:v20e\0"
_HELD_EXECUTION_DOMAIN = b"fisher-graph:tangent-response-held-execution:v20e\0"
_ROLE_EVIDENCE_DOMAIN = b"fisher-graph:tangent-response-role-evidence:v20e\0"
_QUALIFICATION_DOMAIN = b"fisher-graph:tangent-response-qualification:v20e\0"

_INITIAL_WEIGHTS = _core.TANGENT_RESPONSE_INITIAL_WEIGHTS
_FRACTIONS = _core.TANGENT_RESPONSE_FRACTIONS
_POSITIVE_FRACTIONS = tuple(value for value in _FRACTIONS if value > 0.0)
_LAWS = _core.TANGENT_RESPONSE_LAWS
_ARMS = _core.TANGENT_RESPONSE_ARMS
_PROMPTS_PER_FAMILY = 2
_FIT_FAMILY_COUNT = 6
_FAMILY_COUNT = 8
_RANK = 256
_CONDITIONAL_RANK = 16

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20e_six_fold_tangent_constrained_response_fit",
    "scientific_status": "development_only_reused_a16_pair",
    "prerequisite": "exact_immutable_failed_v20d_pair_smoke",
    "initial_weights": _INITIAL_WEIGHTS,
    "response_laws": _LAWS,
    "tangent_solver": "exact_four_face_active_set_natural_quadratic",
    "ray": "complete_feasible_ray_to_opposite_bilinear_box_face",
    "radial_projection_used": False,
    "fraction_ladder": _FRACTIONS,
    "cross_validation": "six_fold_leave_one_fit_family_out",
    "beta_zero": "same_law_initial_vjp_execution_reused_exactly",
    "gradient_bank": (
        "one_hash_only_six_family_bank_per_law_frozen_before_all_14_direction_solves"
    ),
    "gradient_bank_lineage": (
        "outer_initial_evidence_commits_bank_and_every_direction_binds_outer_evidence"
    ),
    "provider_manifest": "all_120_logical_slots_frozen_before_any_cv_capability",
    "cv_capabilities": "twelve_law_fold_restricted_capabilities",
    "positive_fraction_scores": "nine_fractions_times_two_omitted_examples",
    "selection": "macro_improvement_and_at_least_four_of_six_folds",
    "all_six_refit": "selected_fraction_replayed_without_objective_rescore",
    "post_cv_all_six_exact_rescore_performed": False,
    "held_arms": _ARMS,
    "held_barrier": "two_fits_and_seven_providers_frozen_before_capability",
    "fresh_family_disjoint_claim_authorized": False,
    "serving_claim_authorized": False,
    "compression_claim_authorized": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(_FIXED_PROTOCOL, domain=_SOURCE_DOMAIN)


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


def _hashed(payload: Mapping[str, object], *, domain: bytes) -> dict[str, object]:
    selected = dict(payload)
    selected["artifact_sha256"] = _v14._sha256(selected, domain=domain)
    _v14._scalar_report(selected)
    return selected


def _validate_hashed(
    value: Mapping[str, object], *, domain: bytes, label: str
) -> dict[str, object]:
    selected = dict(value)
    payload = {key: item for key, item in selected.items() if key != "artifact_sha256"}
    if selected.get("artifact_sha256") != _v14._sha256(payload, domain=domain):
        raise ValueError(f"{label} artifact hash differs")
    _v14._scalar_report(selected)
    return selected


def _fraction_key(value: float) -> str:
    selected = float(value)
    if selected not in _FRACTIONS:
        raise ValueError("V20e fraction differs from the frozen ladder")
    return selected.hex()


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or not _v20b._is_under_local_runs(output):
        raise ValueError("V20e output must be JSON under .local-runs")
    if _v20b._same_destination(output, _V20D_OUTPUT):
        raise ValueError("V20e must preserve the immutable V20d report")
    return output


def _load_authenticated_v20d_source(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Authenticate V20d and its V20b endpoint before model construction."""

    _v20b._secure_stat(_V20D_OUTPUT, label="pinned V20d report")
    if _v14._file_sha256(_V20D_OUTPUT) != _V20D_FILE_SHA256:
        raise RuntimeError("pinned V20d report file hash drifted")
    report = _v20d._load_existing_report(_V20D_OUTPUT)
    if (
        report.get("report_sha256") != _V20D_LOGICAL_SHA256
        or report.get("classification") != _V20D_CLASSIFICATION
        or report.get("passed") is not False
        or report.get("fit_stage_authorized_for_held_scoring") is not True
        or report.get("next_full_reused_panel_screen_authorized") is not False
        or report.get("fresh_family_disjoint_claim_authorized") is not False
        or report.get("serving_authorized") is not False
        or report.get("compression_claim") is not False
        or report.get("candidate") is not None
        or report.get("provider_sidecar") is not None
    ):
        raise RuntimeError("pinned V20d scientific authority differs")
    logical = dict(report)
    logical.pop("report_sha256", None)
    if _v14._sha256(logical, domain=_v20d._REPORT_DOMAIN) != _V20D_LOGICAL_SHA256:
        raise RuntimeError("pinned V20d logical hash drifted")
    source20d = _mapping(report.get("source"), label="V20d source")
    fit20d = _mapping(report.get("two_fit_bundle_receipt"), label="V20d fit bundle")
    providers20d = _mapping(
        report.get("provider_bundle_receipt"), label="V20d provider bundle"
    )
    source_payload = {
        "path": _V20D_OUTPUT.as_posix(),
        "report_logical_sha256": _V20D_LOGICAL_SHA256,
        "report_file_sha256": _V20D_FILE_SHA256,
        "classification": _V20D_CLASSIFICATION,
        "passed": False,
        "v20d_source_artifact_sha256": _sha(
            source20d.get("artifact_sha256"), label="V20d source"
        ),
        "v20d_two_fit_bundle_artifact_sha256": _sha(
            fit20d.get("artifact_sha256"), label="V20d fit bundle"
        ),
        "v20d_provider_bundle_artifact_sha256": _sha(
            providers20d.get("artifact_sha256"), label="V20d provider bundle"
        ),
        "v20c_report_logical_sha256": _sha(
            source20d.get("report_logical_sha256"), label="V20c logical report"
        ),
        "v20c_report_file_sha256": _sha(
            source20d.get("report_file_sha256"), label="V20c file"
        ),
        "v20b_pair_fragment_sha256": _sha(
            source20d.get("v20b_pair_fragment_sha256"), label="V20b pair fragment"
        ),
        "endpoint_fit_artifact_sha256": _sha(
            _mapping(report.get("shared_fit_receipt"), label="endpoint fit").get(
                "artifact_sha256"
            ),
            label="endpoint fit",
        ),
        "coordinate_trace_artifact_sha256": _sha(
            _mapping(
                report.get("coordinate_trace_receipt"), label="coordinate trace"
            ).get("artifact_sha256"),
            label="coordinate trace",
        ),
        "authenticated_before_model_work": True,
        "v20d_learned_weights_used_as_initialization": False,
    }
    source = _hashed(source_payload, domain=_SOURCE_DOMAIN)
    _authenticated_v20c_source, authenticated_v20c, fragment = (
        _v20d._load_authenticated_v20c_source()
    )
    for field in (
        "panel_receipt",
        "shared_fit_receipt",
        "fit_training_evidence",
        "coordinate_trace_receipt",
    ):
        if _v14._canonical_json_bytes(report.get(field)) != _v14._canonical_json_bytes(
            authenticated_v20c.get(field)
        ):
            raise RuntimeError("V20d endpoint lineage differs from immutable V20c")
    if (
        _v14._canonical_json_bytes(fragment.get("shared_fit_receipt"))
        != _v14._canonical_json_bytes(report.get("shared_fit_receipt"))
        or _v14._canonical_json_bytes(fragment.get("fit_training_evidence"))
        != _v14._canonical_json_bytes(report.get("fit_training_evidence"))
    ):
        raise RuntimeError("V20d endpoint evidence differs from frozen V20b pair")
    return source, dict(report), dict(fragment)


def _source_sha256s(source: Mapping[str, object]) -> dict[str, str]:
    return {
        "v20d_report_logical_sha256": _sha(
            source.get("report_logical_sha256"), label="V20d logical report"
        ),
        "v20d_report_file_sha256": _sha(
            source.get("report_file_sha256"), label="V20d report file"
        ),
        "v20d_source_receipt_sha256": _sha(
            source.get("artifact_sha256"), label="V20d source receipt"
        ),
        "v20d_two_fit_bundle_artifact_sha256": _sha(
            source.get("v20d_two_fit_bundle_artifact_sha256"),
            label="V20d fit bundle",
        ),
        "v20b_pair_fragment_sha256": _sha(
            source.get("v20b_pair_fragment_sha256"), label="V20b pair fragment"
        ),
    }


def _weight_tensor(value: Sequence[float]) -> Tensor:
    weight = torch.tensor(tuple(float(item) for item in value), dtype=torch.float64)
    if weight.shape != (3,) or not bool(torch.isfinite(weight).all()):
        raise ValueError("V20e response weight geometry differs")
    return weight


def _build_response_provider(
    workspace: object,
    *,
    law: str,
    response_weight: Sequence[float],
    polarity: int,
    evidence_sha256: str,
) -> AutonomousCompleteH4FisherContinuousTransferProvider:
    weight = _weight_tensor(response_weight)
    if fisher_continuous_bilinear_box_max_abs(weight) > 1.0:
        raise ValueError("V20e response provider escaped the exact bilinear box")
    return build_autonomous_complete_h4_fisher_continuous_transfer(
        workspace.base_provider,
        workspace.proposal_provider,
        response_law=law,
        response_source="direct",
        response_weight=weight,
        polarity=polarity,
        transfer_protocol_sha256=_core.TANGENT_RESPONSE_PROTOCOL_SHA256,
        transfer_evidence_sha256=evidence_sha256,
        signed_log_kappa=_core.TANGENT_RESPONSE_KAPPA,
    )


def _core_work(*, held_scoring_executed: bool) -> dict[str, object]:
    core = _core.tangent_response_work_accounting(
        collection_forward_count=32,
        collection_backward_count=16,
        endpoint_forward_count=12,
        endpoint_backward_count=12,
        endpoint_local_contraction_count=12,
        endpoint_teacher_access_count=12,
        law_count=2,
        fit_prompt_count=12,
        cv_fold_count=6,
        validation_prompts_per_fold=2,
        fraction_count=10,
        fraction_zero_vjp_reused=True,
        held_role_count=2,
        held_arm_count=7,
        held_prompts_per_role=2,
        held_scoring_executed=held_scoring_executed,
    )
    result: dict[str, object] = {
        **core,
        "law_initial_vjp_forward_count": core["law_fit_gradient_forward_count"],
        "law_initial_vjp_backward_count": core["law_fit_gradient_backward_count"],
        "law_initial_local_contraction_count": core[
            "law_fit_gradient_local_contraction_count"
        ],
        "unique_response_gradient_row_count": 24,
        "tangent_qp_and_ray_solve_count": 14,
        "logical_cv_candidate_count": 120,
        "positive_cv_candidate_count": 108,
        "cv_prompt_score_forward_count": 216,
        "reused_beta_zero_prompt_score_count": 24,
        "held_exact_arm_score_count": core["held_score_count"],
        "provider_only_runtime_trace_count": 136 if held_scoring_executed else 122,
        "total_capability_count_including_endpoint_reconstruction": (
            17 if held_scoring_executed else 15
        ),
        "full_model_forward_count": core["total_forward_count"],
        "full_suffix_backward_traversal_count": core["total_backward_count"],
        "local_head_autograd_contraction_count": core[
            "total_local_contraction_count"
        ],
        "total_autograd_grad_call_count": core[
            "total_backward_or_local_gradient_call_count"
        ],
        "teacher_capability_access_count": core["teacher_access_count"],
        "post_cast_h4_hash_check_count": core["teacher_access_count"],
        "supervised_full_vocab_logits_hash_check_count": core[
            "teacher_access_count"
        ],
        "radial_projection_call_count": 0,
        "post_cv_all_six_exact_rescore_count": 0,
        "provider_only_runtime_traces_excluded_from_full_model_forward_count": True,
    }
    return result


def _work_accounting(*, held_scoring_executed: bool) -> dict[str, object]:
    planned = _core_work(held_scoring_executed=True)
    observed = _core_work(held_scoring_executed=held_scoring_executed)
    if (
        planned["full_model_forward_count"] != 312
        or planned["teacher_capability_access_count"] != 280
        or observed["full_model_forward_count"]
        != (312 if held_scoring_executed else 284)
        or observed["teacher_capability_access_count"]
        != (280 if held_scoring_executed else 252)
    ):
        raise RuntimeError("V20e exact work accounting differs")
    return {
        "planned_full_budget": planned,
        "observed_stage_counters": observed,
        "observed_matches_planned_full_budget": held_scoring_executed,
    }


@dataclass(slots=True)
class _InitialBank:
    law: str
    provider: AutonomousCompleteH4FisherContinuousTransferProvider
    training_records: tuple[object, ...]
    gradients_by_family: dict[str, dict[str, tuple[float, float, float]]]
    objectives_by_family: dict[str, dict[str, float]]
    h4_sha256s: dict[str, str]
    logits_sha256s: dict[str, str]
    execution_sha256s: dict[str, str]
    gradient_bank_receipt: dict[str, object]
    evidence: dict[str, object]


@dataclass(slots=True)
class _ManifestLive:
    directions: dict[str, dict[str, dict[str, object]]]
    rays: dict[str, dict[str, dict[str, object]]]
    providers: dict[
        str,
        dict[str, dict[float, AutonomousCompleteH4FisherContinuousTransferProvider]],
    ]
    receipt: dict[str, object]


@dataclass(slots=True)
class _CVLive:
    law: str
    receipt: dict[str, object]
    fold_evidence: tuple[dict[str, object], ...]


@dataclass(slots=True)
class _FinalLive:
    law: str
    direction: dict[str, object]
    ray: dict[str, object]
    provider: AutonomousCompleteH4FisherContinuousTransferProvider
    trace_evidence: dict[str, object]
    candidate: dict[str, object]
    fit: dict[str, object]


class _ControlledTerminal(RuntimeError):
    """A scientifically expected no-go that must still publish a report."""

    def __init__(self, *, stage: str, evidence: Mapping[str, object]) -> None:
        super().__init__(f"V20e controlled terminal at {stage}")
        self.stage = stage
        self.evidence = dict(evidence)


def _initial_execution_sha256(
    *, law: str, provider_sha256: str, example_id: str, family_id: str,
    objective: float, h4_sha256: str, logits_sha256: str,
) -> str:
    return _v14._sha256(
        {
            "response_law": law,
            "phase": "all_six_initial_vjp_beta_zero",
            "provider_artifact_sha256": provider_sha256,
            "example_id": example_id,
            "family_id": family_id,
            "objective": objective,
            "post_cast_h4_sha256": h4_sha256,
            "supervised_full_vocab_logits_sha256": logits_sha256,
        },
        domain=_INITIAL_EXECUTION_DOMAIN,
    )


def _collect_initial_bank(
    context: object,
    workspace: object,
    teacher_vault: object,
    *,
    law: str,
    source: Mapping[str, object],
) -> _InitialBank:
    if law not in _LAWS:
        raise ValueError("V20e initial law differs")
    training = _v20b._ordered_records(workspace.training_records)
    families = tuple(sorted({record.sequence.family_id for record in training}))
    if (
        len(training) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or len(families) != _FIT_FAMILY_COUNT
        or set(families) & set(_v20c._FROZEN_EXCLUDED)
    ):
        raise RuntimeError("V20e initial gradient complement differs")
    provider_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "source_artifact_sha256": source["artifact_sha256"],
            "endpoint_fit_artifact_sha256": workspace.fit_receipt["artifact_sha256"],
            "response_law": law,
            "weights": _INITIAL_WEIGHTS,
            "scope": "all_six_shared_beta_zero",
            "held_rows_used": False,
        },
        domain=_PROVIDER_SEED_DOMAIN,
    )
    provider = _build_response_provider(
        workspace,
        law=law,
        response_weight=_INITIAL_WEIGHTS,
        polarity=1,
        evidence_sha256=provider_seed,
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
            raise RuntimeError("V20e initial objective capture drifted")
        gradient = _v20d._local_response_weight_gradient(
            provider, record.sequence, h4_gradient
        )
        family = record.sequence.family_id
        example = record.sequence.example_id
        values = tuple(float(item) for item in gradient.tolist())
        gradients[family][example] = values  # type: ignore[assignment]
        objectives[family][example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        gradient_hashes[example] = _v14._tensor_sha256(gradient)
        execution_hashes[example] = _initial_execution_sha256(
            law=law,
            provider_sha256=provider.artifact_sha256,
            example_id=example,
            family_id=family,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
        )
        del model_inputs, teacher, execution, h4_gradient, gradient
    capability_receipt = capability.receipt()
    expected_ids = tuple(record.sequence.example_id for record in training)
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=expected_ids,
        expected_family_count=_FIT_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label=f"V20e {law} initial VJP capability",
    )
    gradient_bank_receipt = _core.build_tangent_response_gradient_bank_receipt(
        source_artifact_sha256s=_source_sha256s(source),
        family_ids=tuple(sorted((*families, *_v20c._FROZEN_EXCLUDED))),
        excluded_family_ids=_v20c._FROZEN_EXCLUDED,
        fit_gradients_by_family=gradients,
        base_provider_artifact_sha256=workspace.base_provider.artifact_sha256,
        proposal_provider_artifact_sha256=(
            workspace.proposal_provider.artifact_sha256
        ),
        response_law=law,
        held_objectives_or_gradients_used=False,
    )
    core_gradient_row_hashes = {
        str(example): _sha(row_hash, label="core gradient row")
        for family_summary in _mapping(
            gradient_bank_receipt["family_gradient_summaries_by_family"],
            label="gradient-bank family summaries",
        ).values()
        for example, row_hash in _mapping(
            _mapping(
                family_summary, label="gradient-bank family summary"
            )["example_gradient_sha256s"],
            label="gradient-bank row hashes",
        ).items()
    }
    evidence = _hashed(
        {
            "response_law": law,
            "source_artifact_sha256": source["artifact_sha256"],
            "endpoint_fit_artifact_sha256": workspace.fit_receipt["artifact_sha256"],
            "provider_artifact_sha256": provider.artifact_sha256,
            "provider_transfer_evidence_sha256": provider.transfer_evidence_sha256,
            "initial_weights": _INITIAL_WEIGHTS,
            "initial_weight_tensor_sha256": _provider_tensor_sha256(
                _weight_tensor(_INITIAL_WEIGHTS)
            ),
            "training_family_ids": families,
            "training_sequence_sha256s": {
                record.sequence.example_id: record.sequence.artifact_sha256
                for record in training
            },
            "training_example_family_ids": {
                record.sequence.example_id: record.sequence.family_id
                for record in training
            },
            "initial_objectives_by_family": objectives,
            "gradient_sha256s": dict(sorted(gradient_hashes.items())),
            "core_gradient_row_sha256s": dict(
                sorted(core_gradient_row_hashes.items())
            ),
            "gradient_bank_receipt": gradient_bank_receipt,
            "gradient_bank_frozen_before_direction_solves": True,
            "direction_solve_count_at_gradient_bank_freeze": 0,
            "gradient_tensor_and_core_row_hashes_collected_from_same_live_tensor": True,
            "unique_empirical_fisher_gradient_row_count": len(training),
            "empirical_fisher_outer_product_evaluation_count": len(training),
            "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
            "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
            "execution_receipt_sha256s": dict(sorted(execution_hashes.items())),
            "capability_receipt": capability_receipt,
            "full_suffix_vjp_count": len(training),
            "local_response_autograd_contraction_count": len(training),
            "beta_zero_exact_execution_reusable_by_all_six_folds": True,
            "all_initial_executions_finite": True,
            "all_initial_executions_exact": True,
            "held_family_ids": _v20c._FROZEN_EXCLUDED,
            "held_data_or_objectives_used": False,
            "raw_gradients_h4_logits_targets_or_tensors_serialized": False,
        },
        domain=_INITIAL_DOMAIN,
    )
    return _InitialBank(
        law=law,
        provider=provider,
        training_records=training,
        gradients_by_family=gradients,
        objectives_by_family=objectives,
        h4_sha256s=h4_hashes,
        logits_sha256s=logits_hashes,
        execution_sha256s=execution_hashes,
        gradient_bank_receipt=gradient_bank_receipt,
        evidence=evidence,
    )


def _build_cv_provider_manifest(
    workspace: object,
    *,
    source: Mapping[str, object],
    family_ids: Sequence[str],
    banks: Mapping[str, _InitialBank],
) -> _ManifestLive:
    """Construct and freeze all 120 logical slots before CV capability issue."""

    if set(banks) != set(_LAWS):
        raise ValueError("V20e initial banks differ")
    fit_families = tuple(
        family for family in sorted(family_ids) if family not in _v20c._FROZEN_EXCLUDED
    )
    if len(fit_families) != _FIT_FAMILY_COUNT:
        raise ValueError("V20e fit family geometry differs")
    directions: dict[str, dict[str, dict[str, object]]] = {}
    rays: dict[str, dict[str, dict[str, object]]] = {}
    providers: dict[
        str,
        dict[str, dict[float, AutonomousCompleteH4FisherContinuousTransferProvider]],
    ] = {}
    slots: dict[str, dict[str, dict[str, object]]] = {}
    # Freeze all twelve direction/ray receipts first.  A zero ray is a valid
    # scientific no-go, but positive provider slots would be fictitious, so it
    # terminates before any CV capability is issued.
    for law in _LAWS:
        bank = banks[law]
        directions[law] = {}
        rays[law] = {}
        providers[law] = {}
        slots[law] = {}
        for validation_family in fit_families:
            direction = (
                _core.build_tangent_response_direction_from_gradient_bank_receipt(
                gradient_bank_receipt=bank.gradient_bank_receipt,
                gradient_evidence_sha256=str(bank.evidence["artifact_sha256"]),
                validation_family_id=validation_family,
                )
            )
            ray = _core.build_tangent_response_ray_receipt(
                direction_receipt=direction
            )
            directions[law][validation_family] = direction
            rays[law][validation_family] = ray
    degenerate = tuple(
        {
            "response_law": law,
            "validation_family_id": family,
            "direction_artifact_sha256": directions[law][family][
                "artifact_sha256"
            ],
            "ray_artifact_sha256": rays[law][family]["artifact_sha256"],
            "strict_descent_direction": directions[law][family][
                "strict_descent_direction"
            ],
            "direction_degenerate": rays[law][family]["direction_degenerate"],
        }
        for law in _LAWS
        for family in fit_families
        if (
            rays[law][family].get("direction_degenerate") is True
            or directions[law][family].get("strict_descent_direction") is not True
        )
    )
    if degenerate:
        raise _ControlledTerminal(
            stage="cv_direction_preflight",
            evidence=_hashed(
                {
                    "source_artifact_sha256": source["artifact_sha256"],
                    "direction_receipts_by_law_and_fold": directions,
                    "ray_receipts_by_law_and_fold": rays,
                    "degenerate_rows": degenerate,
                    "provider_manifest_created": False,
                    "cv_capability_count": 0,
                    "held_capability_count": 0,
                    "radial_projection_used": False,
                    "classification": "tangent_direction_preflight_failed",
                },
                domain=_MANIFEST_DOMAIN,
            ),
        )
    for law in _LAWS:
        bank = banks[law]
        for validation_family in fit_families:
            direction = directions[law][validation_family]
            ray = rays[law][validation_family]
            providers[law][validation_family] = {}
            slots[law][validation_family] = {}
            for fraction in _FRACTIONS:
                proposal = _core.tangent_response_fraction_proposal(
                    direction_receipt=direction,
                    ray_receipt=ray,
                    fraction=fraction,
                )
                weights = tuple(float(item) for item in proposal["weights"])
                if fraction == 0.0:
                    provider = bank.provider
                    seed = bank.provider.transfer_evidence_sha256
                else:
                    seed = _v14._sha256(
                        {
                            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                            "source_artifact_sha256": source["artifact_sha256"],
                            "response_law": law,
                            "validation_family_id": validation_family,
                            "direction_artifact_sha256": direction["artifact_sha256"],
                            "ray_artifact_sha256": ray["artifact_sha256"],
                            "fraction": fraction,
                            "weights": weights,
                            "radial_projection_used": False,
                            "held_rows_used": False,
                        },
                        domain=_PROVIDER_SEED_DOMAIN,
                    )
                    provider = _build_response_provider(
                        workspace,
                        law=law,
                        response_weight=weights,
                        polarity=1,
                        evidence_sha256=seed,
                    )
                if tuple(float(item) for item in provider.response_weight.tolist()) != weights:
                    raise RuntimeError("V20e provider weights differ from certified ray")
                providers[law][validation_family][fraction] = provider
                slots[law][validation_family][_fraction_key(fraction)] = {
                    "fraction": fraction,
                    "weights": weights,
                    "weight_tensor_sha256": _provider_tensor_sha256(
                        provider.response_weight
                    ),
                    "box_certificate": fisher_continuous_bilinear_box_max_abs(
                        provider.response_weight
                    ),
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "provider_transfer_evidence_sha256": seed,
                    "direction_artifact_sha256": direction["artifact_sha256"],
                    "ray_artifact_sha256": ray["artifact_sha256"],
                    "radial_projection_used": False,
                }
    positive_hashes = [
        str(slots[law][family][_fraction_key(fraction)]["provider_artifact_sha256"])
        for law in _LAWS
        for family in fit_families
        for fraction in _POSITIVE_FRACTIONS
    ]
    zero_by_law = {
        law: {
            str(slots[law][family][_fraction_key(0.0)]["provider_artifact_sha256"])
            for family in fit_families
        }
        for law in _LAWS
    }
    if len(positive_hashes) != 108 or len(set(positive_hashes)) != 108:
        raise RuntimeError("V20e positive CV providers are not artifact-distinct")
    if any(len(values) != 1 for values in zero_by_law.values()):
        raise RuntimeError("V20e beta-zero provider was not shared within its law")
    if len({next(iter(values)) for values in zero_by_law.values()}) != 2:
        raise RuntimeError("V20e law-specific beta-zero providers are not distinct")
    receipt = _hashed(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.TANGENT_RESPONSE_PROTOCOL_SHA256,
            "source_artifact_sha256": source["artifact_sha256"],
            "law_order": _LAWS,
            "fold_order": fit_families,
            "fraction_order": _FRACTIONS,
            "direction_artifact_sha256s_by_law_and_fold": {
                law: {
                    family: directions[law][family]["artifact_sha256"]
                    for family in fit_families
                }
                for law in _LAWS
            },
            "ray_artifact_sha256s_by_law_and_fold": {
                law: {
                    family: rays[law][family]["artifact_sha256"]
                    for family in fit_families
                }
                for law in _LAWS
            },
            "provider_slots_by_law_and_fold": slots,
            "logical_provider_slot_count": 120,
            "positive_provider_artifact_count": 108,
            "positive_provider_hashes_unique": True,
            "beta_zero_provider_artifact_sha256s_by_law": {
                law: next(iter(zero_by_law[law])) for law in _LAWS
            },
            "beta_zero_provider_reused_across_six_folds_per_law": True,
            "all_slots_frozen_before_first_cv_capability": True,
            "cv_capability_count_at_freeze": 0,
            "radial_projection_used": False,
            "held_data_or_objectives_used": False,
            "raw_tensors_or_provider_sidecars_serialized": False,
        },
        domain=_MANIFEST_DOMAIN,
    )
    return _ManifestLive(
        directions=directions,
        rays=rays,
        providers=providers,
        receipt=receipt,
    )


def _trace_provider(
    provider: object, records: Sequence[object], *, arm: str
) -> tuple[dict[str, object], dict[str, Tensor]]:
    transient = _v20c._response_runtime_trace(provider, records, arm=arm)
    gains = {
        str(key): value
        for key, value in _mapping(
            transient.get("_transient_gain_values"), label="response gain values"
        ).items()
        if isinstance(value, Tensor)
    }
    trace = _v20c._strip_transient_trace(transient)
    expected_ids = {record.sequence.example_id for record in records}
    if set(gains) != expected_ids:
        raise RuntimeError("V20e response trace example geometry differs")
    return trace, gains


def _cv_execution_sha256(
    *, manifest_sha256: str, law: str, validation_family_id: str,
    fraction: float, provider_sha256: str, example_id: str, objective: float,
    h4_sha256: str, logits_sha256: str, trace_sha256: str,
) -> str:
    return _v14._sha256(
        {
            "manifest_artifact_sha256": manifest_sha256,
            "response_law": law,
            "validation_family_id": validation_family_id,
            "fraction": fraction,
            "provider_artifact_sha256": provider_sha256,
            "example_id": example_id,
            "objective": objective,
            "post_cast_h4_sha256": h4_sha256,
            "supervised_full_vocab_logits_sha256": logits_sha256,
            "response_trace_sha256": trace_sha256,
        },
        domain=_CV_EXECUTION_DOMAIN,
    )


def _score_cv_fold(
    context: object,
    teacher_vault: object,
    *,
    law: str,
    validation_family_id: str,
    bank: _InitialBank,
    manifest: _ManifestLive,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Score one omitted fit family after the full manifest is frozen."""

    selected_records = _v20b._ordered_records(
        tuple(
            record
            for record in bank.training_records
            if record.sequence.family_id == validation_family_id
        )
    )
    if len(selected_records) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20e CV fold prompt geometry differs")
    direction = manifest.directions[law][validation_family_id]
    ray = manifest.rays[law][validation_family_id]
    providers = manifest.providers[law][validation_family_id]
    manifest_sha = str(manifest.receipt["artifact_sha256"])
    # This is the first teacher capability created after the complete manifest.
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in selected_records),
        # This is an inner LOFO validation fold over the six authorized fit
        # families, not an outer Reed/Sundial held-family boundary.  The vault's
        # held_family_id parameter means "forbid this family entirely", so the
        # validation rows must use the ordinary fit capability here.
        held_family_id=None,
    )
    candidate_receipts: list[dict[str, object]] = []
    persisted: dict[str, dict[str, object]] = {}
    baseline_h4 = {
        record.sequence.example_id: bank.h4_sha256s[record.sequence.example_id]
        for record in selected_records
    }
    baseline_logits = {
        record.sequence.example_id: bank.logits_sha256s[record.sequence.example_id]
        for record in selected_records
    }
    for fraction in _FRACTIONS:
        provider = providers[fraction]
        trace, _gains = _trace_provider(
            provider,
            selected_records,
            arm=f"cv_{law}_{validation_family_id}_{_fraction_key(fraction)}",
        )
        objectives: dict[str, float] = {}
        h4_hashes: dict[str, str] = {}
        logits_hashes: dict[str, str] = {}
        execution_hashes: dict[str, str] = {}
        if fraction == 0.0:
            for record in selected_records:
                example = record.sequence.example_id
                objectives[example] = bank.objectives_by_family[
                    validation_family_id
                ][example]
                h4_hashes[example] = bank.h4_sha256s[example]
                logits_hashes[example] = bank.logits_sha256s[example]
                execution_hashes[example] = bank.execution_sha256s[example]
            beta_zero_reused = True
        else:
            for record in selected_records:
                model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
                    context, record
                )
                teacher = capability.get(
                    record.sequence.example_id,
                    family_id=record.sequence.family_id,
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
                objectives[example] = score
                h4_hashes[example] = h4_sha
                logits_hashes[example] = logits_sha
                execution_hashes[example] = _cv_execution_sha256(
                    manifest_sha256=manifest_sha,
                    law=law,
                    validation_family_id=validation_family_id,
                    fraction=fraction,
                    provider_sha256=provider.artifact_sha256,
                    example_id=example,
                    objective=score,
                    h4_sha256=h4_sha,
                    logits_sha256=logits_sha,
                    trace_sha256=str(trace["artifact_sha256"]),
                )
                del model_inputs, teacher, execution
            beta_zero_reused = False
        changed = h4_hashes != baseline_h4 or logits_hashes != baseline_logits
        evidence = _hashed(
            {
                "manifest_artifact_sha256": manifest_sha,
                "response_law": law,
                "validation_family_id": validation_family_id,
                "fraction": fraction,
                "provider_artifact_sha256": provider.artifact_sha256,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(logits_hashes.items())
                ),
                "execution_receipt_sha256s": dict(sorted(execution_hashes.items())),
                "response_trace": trace,
                "execution_changed_from_beta_zero": changed,
                "beta_zero_reused_from_same_law_initial_vjp": beta_zero_reused,
                "exact_finite_execution": True,
                "manifest_frozen_before_capability": True,
                "held_data_or_objectives_used": False,
                "raw_tensors_logits_targets_or_coordinates_serialized": False,
            },
            domain=_CV_EXECUTION_DOMAIN,
        )
        candidate = _core.build_tangent_response_cv_candidate(
            direction_receipt=direction,
            ray_receipt=ray,
            fraction=fraction,
            provider_artifact_sha256=provider.artifact_sha256,
            validation_example_ids=tuple(objectives),
            validation_objectives_by_example=objectives,
            validation_execution_receipt_sha256s_by_example=execution_hashes,
            execution_evidence_sha256=str(evidence["artifact_sha256"]),
            finite=trace.get("finite") is True,
            pointwise_trust_passed=trace.get("pointwise_trust_passed") is True,
            rank_is_16=trace.get("endpoint_conditional_ranks_are_16") is True,
            execution_exact=True,
            execution_changed_from_baseline=changed,
            held_objectives_used=False,
        )
        candidate_receipts.append(candidate)
        persisted[_fraction_key(fraction)] = {
            "candidate_receipt": candidate,
            "execution_evidence": evidence,
        }
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(
            record.sequence.example_id for record in selected_records
        ),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_POSITIVE_FRACTIONS),
        label=f"V20e {law} {validation_family_id} CV capability",
    )
    fold = _hashed(
        {
            "manifest_artifact_sha256": manifest_sha,
            "response_law": law,
            "validation_family_id": validation_family_id,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "ray_artifact_sha256": ray["artifact_sha256"],
            "fraction_order": _FRACTIONS,
            "candidate_execution_evidence_by_fraction": persisted,
            "capability_receipt": capability_receipt,
            "positive_fraction_accesses_per_example": len(_POSITIVE_FRACTIONS),
            "beta_zero_teacher_access_count": 0,
            "beta_zero_execution_reused_from_initial_vjp": True,
            "manifest_frozen_before_capability": True,
            "held_data_or_objectives_used": False,
            "raw_tensors_logits_targets_or_coordinates_serialized": False,
        },
        domain=_CV_FOLD_DOMAIN,
    )
    return tuple(candidate_receipts), fold


def _run_cross_validation(
    context: object,
    teacher_vault: object,
    *,
    banks: Mapping[str, _InitialBank],
    manifest: _ManifestLive,
) -> dict[str, _CVLive]:
    """Run both laws only after all 120 provider slots have been frozen."""

    if manifest.receipt.get("all_slots_frozen_before_first_cv_capability") is not True:
        raise PermissionError("V20e provider manifest is not frozen")
    result: dict[str, _CVLive] = {}
    for law in _LAWS:
        fold_order = tuple(sorted(manifest.directions[law]))
        all_candidates: list[dict[str, object]] = []
        fold_evidence: list[dict[str, object]] = []
        for validation_family in fold_order:
            candidates, evidence = _score_cv_fold(
                context,
                teacher_vault,
                law=law,
                validation_family_id=validation_family,
                bank=banks[law],
                manifest=manifest,
            )
            all_candidates.extend(candidates)
            fold_evidence.append(evidence)
        receipt = _core.build_tangent_response_cv_receipt(
            direction_receipts=tuple(
                manifest.directions[law][family] for family in fold_order
            ),
            ray_receipts=tuple(manifest.rays[law][family] for family in fold_order),
            candidates=tuple(all_candidates),
        )
        result[law] = _CVLive(
            law=law,
            receipt=receipt,
            fold_evidence=tuple(fold_evidence),
        )
    return result


def _build_selection_bundle(
    cv: Mapping[str, _CVLive], *, manifest: Mapping[str, object]
) -> dict[str, object]:
    if set(cv) != set(_LAWS):
        raise ValueError("V20e CV selections differ")
    return _hashed(
        {
            "manifest_artifact_sha256": manifest["artifact_sha256"],
            "law_order": _LAWS,
            "cv_receipt_artifact_sha256s_by_law": {
                law: cv[law].receipt["artifact_sha256"] for law in _LAWS
            },
            "selected_fractions_by_law": {
                law: cv[law].receipt["selected_fraction"] for law in _LAWS
            },
            "both_law_selections_frozen_before_either_all_six_provider": True,
            "all_six_provider_count_at_freeze": 0,
            "held_capability_count_at_freeze": 0,
            "post_cv_all_six_exact_rescore_performed": False,
            "outer_held_objectives_used": False,
        },
        domain=_SELECTION_DOMAIN,
    )


def _initial_gain_hashes_from_cv(
    cv: _CVLive,
) -> dict[str, str]:
    return _initial_gain_hashes_from_fold_evidence(cv.fold_evidence)


def _initial_gain_hashes_from_fold_evidence(
    fold_evidence: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for fold in fold_evidence:
        by_fraction = _mapping(
            fold.get("candidate_execution_evidence_by_fraction"),
            label="CV fold candidates",
        )
        zero = _mapping(by_fraction.get(_fraction_key(0.0)), label="beta-zero row")
        evidence = _mapping(zero.get("execution_evidence"), label="beta-zero evidence")
        trace = _mapping(evidence.get("response_trace"), label="beta-zero trace")
        for example, value in _mapping(
            trace.get("response_gain_sha256s"), label="beta-zero gain hashes"
        ).items():
            key = str(example)
            if key in result:
                raise RuntimeError("V20e beta-zero CV trace example was duplicated")
            result[key] = _sha(value, label="beta-zero response gain")
    if len(result) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20e beta-zero CV traces do not cover all fit examples")
    return result


def _build_all_six_direction_preflight(
    workspace: object,
    *,
    source: Mapping[str, object],
    family_ids: Sequence[str],
    banks: Mapping[str, _InitialBank],
    selection_bundle: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    directions: dict[str, dict[str, object]] = {}
    rays: dict[str, dict[str, object]] = {}
    for law in _LAWS:
        directions[law] = (
            _core.build_tangent_response_direction_from_gradient_bank_receipt(
            gradient_bank_receipt=banks[law].gradient_bank_receipt,
            gradient_evidence_sha256=str(banks[law].evidence["artifact_sha256"]),
            validation_family_id=None,
            )
        )
        rays[law] = _core.build_tangent_response_ray_receipt(
            direction_receipt=directions[law]
        )
    degenerate = tuple(
        law
        for law in _LAWS
        if rays[law].get("direction_degenerate") is True
        or directions[law].get("strict_descent_direction") is not True
    )
    if degenerate:
        raise _ControlledTerminal(
            stage="all_six_direction_preflight",
            evidence=_hashed(
                {
                    "selection_bundle_artifact_sha256": selection_bundle[
                        "artifact_sha256"
                    ],
                    "direction_receipts_by_law": directions,
                    "ray_receipts_by_law": rays,
                    "degenerate_laws": degenerate,
                    "all_six_provider_created": False,
                    "post_cv_all_six_exact_rescore_performed": False,
                    "held_capability_count": 0,
                    "radial_projection_used": False,
                    "classification": "all_six_tangent_direction_failed",
                },
                domain=_FINAL_TRACE_DOMAIN,
            ),
        )
    return directions, rays


def _build_final_fit(
    workspace: object,
    *,
    law: str,
    source: Mapping[str, object],
    family_ids: Sequence[str],
    bank: _InitialBank,
    cv: _CVLive,
    selection_bundle: Mapping[str, object],
    direction: Mapping[str, object],
    ray: Mapping[str, object],
) -> _FinalLive:
    """Replay the OOF-selected fraction on all six families without NLL rescore."""

    if (
        selection_bundle.get(
            "both_law_selections_frozen_before_either_all_six_provider"
        )
        is not True
        or selection_bundle.get("post_cv_all_six_exact_rescore_performed") is not False
    ):
        raise PermissionError("V20e two-law selection bundle is not frozen")
    direction = _core.validate_tangent_response_direction_receipt(
        direction, expected_source_artifact_sha256s=_source_sha256s(source)
    )
    ray = _core.validate_tangent_response_ray_receipt(
        ray, direction_receipt=direction
    )
    if direction.get("response_law") != law or direction.get("validation_family_id") is not None:
        raise ValueError("V20e all-six direction binding differs")
    selected_fraction = float(cv.receipt["selected_fraction"])
    proposal = _core.tangent_response_fraction_proposal(
        direction_receipt=direction,
        ray_receipt=ray,
        fraction=selected_fraction,
    )
    weights = tuple(float(item) for item in proposal["weights"])
    if selected_fraction == 0.0:
        provider = bank.provider
        provider_seed = bank.provider.transfer_evidence_sha256
    else:
        provider_seed = _v14._sha256(
            {
                "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                "source_artifact_sha256": source["artifact_sha256"],
                "selection_bundle_artifact_sha256": selection_bundle[
                    "artifact_sha256"
                ],
                "response_law": law,
                "scope": "all_six_provider_only_trace",
                "direction_artifact_sha256": direction["artifact_sha256"],
                "ray_artifact_sha256": ray["artifact_sha256"],
                "selected_fraction": selected_fraction,
                "weights": weights,
                "radial_projection_used": False,
                "post_cv_all_six_exact_rescore_performed": False,
            },
            domain=_PROVIDER_SEED_DOMAIN,
        )
        provider = _build_response_provider(
            workspace,
            law=law,
            response_weight=weights,
            polarity=1,
            evidence_sha256=provider_seed,
        )
    trace, gains = _trace_provider(
        provider,
        bank.training_records,
        arm=f"final_fit_support_{law}",
    )
    initial_gain_hashes = _initial_gain_hashes_from_cv(cv)
    selected_gain_hashes = {
        str(key): _sha(value, label="selected response gain")
        for key, value in _mapping(
            trace.get("response_gain_sha256s"), label="selected response gains"
        ).items()
    }
    changed_from_initial = selected_gain_hashes != initial_gain_hashes
    records_by_family: dict[str, list[object]] = {}
    for record in bank.training_records:
        records_by_family.setdefault(record.sequence.family_id, []).append(record)
    example_ids_by_family: dict[str, tuple[str, ...]] = {}
    trace_receipts_by_family: dict[str, dict[str, str]] = {}
    gain_trace_sha_by_family: dict[str, str] = {}
    gain_min_by_family: dict[str, float] = {}
    gain_max_by_family: dict[str, float] = {}
    gain_distinct_by_family: dict[str, int] = {}
    for family, records in sorted(records_by_family.items()):
        example_ids = tuple(sorted(record.sequence.example_id for record in records))
        values = torch.cat(tuple(gains[example].reshape(-1) for example in example_ids))
        example_ids_by_family[family] = example_ids
        trace_receipts_by_family[family] = {
            example: _v14._sha256(
                {
                    "selection_bundle_artifact_sha256": selection_bundle[
                        "artifact_sha256"
                    ],
                    "response_law": law,
                    "family_id": family,
                    "example_id": example,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "provider_trace_artifact_sha256": trace["artifact_sha256"],
                    "response_gain_sha256": selected_gain_hashes[example],
                    "objective_rescored": False,
                },
                domain=_FINAL_TRACE_DOMAIN,
            )
            for example in example_ids
        }
        gain_trace_sha_by_family[family] = _v14._sha256(
            {
                "response_law": law,
                "family_id": family,
                "provider_artifact_sha256": provider.artifact_sha256,
                "response_gain_sha256s": {
                    example: selected_gain_hashes[example]
                    for example in example_ids
                },
            },
            domain=_FINAL_TRACE_DOMAIN,
        )
        gain_min_by_family[family] = float(values.min())
        gain_max_by_family[family] = float(values.max())
        gain_distinct_by_family[family] = int(torch.unique(values).numel())
    trace_evidence = _hashed(
        {
            "selection_bundle_artifact_sha256": selection_bundle["artifact_sha256"],
            "response_law": law,
            "selected_fraction": selected_fraction,
            "selected_weights": weights,
            "provider_artifact_sha256": provider.artifact_sha256,
            "provider_transfer_evidence_sha256": provider_seed,
            "provider_trace": trace,
            "initial_response_gain_sha256s": dict(sorted(initial_gain_hashes.items())),
            "selected_response_gain_sha256s": dict(
                sorted(selected_gain_hashes.items())
            ),
            "provider_trace_receipt_sha256s_by_family": trace_receipts_by_family,
            "gain_trace_sha256s_by_family": gain_trace_sha_by_family,
            "gain_min_by_family": gain_min_by_family,
            "gain_max_by_family": gain_max_by_family,
            "gain_distinct_count_by_family": gain_distinct_by_family,
            "provider_trace_changed_from_initial": changed_from_initial,
            "provider_only_trace": True,
            "teacher_capability_created": False,
            "exact_fit_objective_rescored": False,
            "post_cast_h4_or_logits_created": False,
            "radial_projection_used": False,
            "raw_response_or_modal_tensors_serialized": False,
        },
        domain=_FINAL_TRACE_DOMAIN,
    )
    candidate = _core.build_tangent_response_final_candidate_receipt(
        cv_receipt=cv.receipt,
        direction_receipt=direction,
        ray_receipt=ray,
        selected_provider_artifact_sha256=provider.artifact_sha256,
        fit_support_example_ids_by_family=example_ids_by_family,
        fit_support_provider_trace_receipt_sha256s_by_family=(
            trace_receipts_by_family
        ),
        fit_support_gain_trace_sha256s_by_family=gain_trace_sha_by_family,
        fit_support_gain_min_by_family=gain_min_by_family,
        fit_support_gain_max_by_family=gain_max_by_family,
        fit_support_gain_distinct_count_by_family=gain_distinct_by_family,
        provider_trace_evidence_sha256=str(trace_evidence["artifact_sha256"]),
        provider_trace_finite=trace.get("finite") is True,
        pointwise_trust_passed=trace.get("pointwise_trust_passed") is True,
        rank_is_16=trace.get("endpoint_conditional_ranks_are_16") is True,
        provider_trace_exact=True,
        provider_trace_changed_from_initial=changed_from_initial,
        final_exact_fit_objectives_used=False,
        outer_held_objectives_used=False,
    )
    fit = _core.build_tangent_response_fit_receipt(
        cv_receipt=cv.receipt,
        final_direction_receipt=direction,
        final_ray_receipt=ray,
        final_candidate_receipt=candidate,
    )
    return _FinalLive(
        law=law,
        direction=direction,
        ray=ray,
        provider=provider,
        trace_evidence=trace_evidence,
        candidate=candidate,
        fit=fit,
    )


def _fit_stage_authorized(
    fit_bundle: Mapping[str, object], finals: Mapping[str, _FinalLive]
) -> bool:
    return bool(
        fit_bundle.get("both_fits_authorized") is True
        and fit_bundle.get("held_score_authorized") is True
        and set(finals) == set(_LAWS)
        and all(
            finals[law].fit.get("learned_candidate_authorized") is True
            and finals[law].trace_evidence.get("provider_trace_changed_from_initial")
            is True
            for law in _LAWS
        )
    )


def _provider_receipt(
    provider: object,
    *,
    arm: str,
    fit_bundle_sha256: str,
    selected_fit_sha256: str | None,
) -> dict[str, object]:
    if arm not in _ARMS:
        raise ValueError("V20e provider arm differs")
    metadata = dict(getattr(provider, "metadata")())
    _v14._scalar_report(metadata)
    continuous = isinstance(
        provider, AutonomousCompleteH4FisherContinuousTransferProvider
    )
    weight = (
        tuple(float(item) for item in provider.response_weight.tolist())
        if continuous
        else None
    )
    corners = (
        tuple(
            float(item)
            for item in fisher_continuous_bilinear_corner_values(
                provider.response_weight
            )
        )
        if continuous
        else None
    )
    box = max(abs(item) for item in corners) if corners is not None else None
    return _hashed(
        {
            "arm": arm,
            "fit_bundle_artifact_sha256": _sha(
                fit_bundle_sha256, label="V20e fit bundle"
            ),
            "selected_fit_artifact_sha256": (
                _sha(selected_fit_sha256, label="V20e selected fit")
                if selected_fit_sha256 is not None
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
            "response_law": provider.response_law if continuous else "base",
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
            "radial_projection_used": False,
            "provider_sidecar_serialized": False,
        },
        domain=_PROVIDER_RECEIPT_DOMAIN,
    )


def _build_frozen_held_providers(
    workspace: object,
    *,
    banks: Mapping[str, _InitialBank],
    finals: Mapping[str, _FinalLive],
    fit_bundle: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    if not _fit_stage_authorized(fit_bundle, finals):
        raise PermissionError("V20e fit bundle did not authorize held providers")
    signed = finals["signed_log"]
    linear = finals["linear"]
    bundle_sha = str(fit_bundle["artifact_sha256"])
    providers: dict[str, object] = {
        "base": workspace.base_provider,
        "constant_plus_one": (
            build_autonomous_complete_h4_fisher_continuous_constant_control(
                workspace.base_provider,
                workspace.proposal_provider,
                alpha=1,
                transfer_protocol_sha256=_core.TANGENT_RESPONSE_PROTOCOL_SHA256,
                transfer_evidence_sha256=bundle_sha,
            )
        ),
        "fixed_signed_log": banks["signed_log"].provider,
        "fixed_linear": banks["linear"].provider,
        "learned_signed_log": signed.provider,
        "learned_linear": linear.provider,
        "learned_signed_log_sign_flip": _build_response_provider(
            workspace,
            law="signed_log",
            response_weight=tuple(float(item) for item in signed.fit["selected_weights"]),
            polarity=-1,
            evidence_sha256=signed.provider.transfer_evidence_sha256,
        ),
    }
    if set(providers) != set(_ARMS) or len(
        {getattr(value, "artifact_sha256") for value in providers.values()}
    ) != len(_ARMS):
        raise RuntimeError("V20e held providers are not seven artifact-distinct arms")
    selected_fits = {
        "learned_signed_log": str(signed.fit["artifact_sha256"]),
        "learned_signed_log_sign_flip": str(signed.fit["artifact_sha256"]),
        "learned_linear": str(linear.fit["artifact_sha256"]),
    }
    receipts = {
        arm: _provider_receipt(
            providers[arm],
            arm=arm,
            fit_bundle_sha256=bundle_sha,
            selected_fit_sha256=selected_fits.get(arm),
        )
        for arm in _ARMS
    }
    bundle = _hashed(
        {
            "fit_bundle_artifact_sha256": bundle_sha,
            "base_provider_artifact_sha256": workspace.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": (
                workspace.proposal_provider.artifact_sha256
            ),
            "arm_order": _ARMS,
            "provider_artifact_sha256s": {
                arm: receipts[arm]["provider_artifact_sha256"] for arm in _ARMS
            },
            "provider_receipt_artifact_sha256s": {
                arm: receipts[arm]["artifact_sha256"] for arm in _ARMS
            },
            "learned_signed_log_fit_artifact_sha256": signed.fit[
                "artifact_sha256"
            ],
            "learned_linear_fit_artifact_sha256": linear.fit["artifact_sha256"],
            "all_seven_providers_frozen_before_held_capability": True,
            "held_capability_count_at_freeze": 0,
            "post_cv_all_six_exact_rescore_performed": False,
            "radial_projection_used": False,
            "provider_sidecar_or_raw_tensor_serialized": False,
        },
        domain=_PROVIDER_BUNDLE_DOMAIN,
    )
    _validate_provider_bundle(
        receipts,
        provider_bundle=bundle,
        fit_bundle=fit_bundle,
        banks=banks,
        finals=finals,
        initial_evidence_by_law={law: banks[law].evidence for law in _LAWS},
        final_trace_evidence_by_law={
            law: finals[law].trace_evidence for law in _LAWS
        },
    )
    return providers, receipts, bundle


def _validate_provider_bundle(
    receipts: Mapping[str, Mapping[str, object]],
    *,
    provider_bundle: Mapping[str, object],
    fit_bundle: Mapping[str, object],
    banks: Mapping[str, _InitialBank] | None = None,
    finals: Mapping[str, _FinalLive] | None = None,
    initial_evidence_by_law: Mapping[str, Mapping[str, object]] | None = None,
    final_trace_evidence_by_law: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    if set(receipts) != set(_ARMS):
        raise ValueError("V20e provider receipt arms differ")
    normalized = {
        arm: _validate_hashed(
            _mapping(receipts[arm], label=f"{arm} provider receipt"),
            domain=_PROVIDER_RECEIPT_DOMAIN,
            label=f"{arm} provider receipt",
        )
        for arm in _ARMS
    }
    bundle = _validate_hashed(
        provider_bundle,
        domain=_PROVIDER_BUNDLE_DOMAIN,
        label="V20e provider bundle",
    )
    fit_sha = _sha(fit_bundle.get("artifact_sha256"), label="V20e fit bundle")
    base_sha = _sha(
        fit_bundle.get("base_provider_artifact_sha256"), label="base provider"
    ) if "base_provider_artifact_sha256" in fit_bundle else str(
        _mapping(fit_bundle["fit_receipts_by_law"], label="fit receipts")[
            "signed_log"
        ]["final_direction_receipt"]["base_provider_artifact_sha256"]
    )
    proposal_sha = str(
        _mapping(fit_bundle["fit_receipts_by_law"], label="fit receipts")[
            "signed_log"
        ]["final_direction_receipt"]["proposal_provider_artifact_sha256"]
    )
    expected_law_polarity = {
        "base": ("base", 0),
        "constant_plus_one": ("linear", 1),
        "fixed_signed_log": ("signed_log", 1),
        "fixed_linear": ("linear", 1),
        "learned_signed_log": ("signed_log", 1),
        "learned_linear": ("linear", 1),
        "learned_signed_log_sign_flip": ("signed_log", -1),
    }
    fits = _mapping(fit_bundle["fit_receipts_by_law"], label="V20e fits")
    expected_sources = {
        "base": "base_zero",
        "constant_plus_one": "constant",
        "fixed_signed_log": "direct",
        "fixed_linear": "direct",
        "learned_signed_log": "direct",
        "learned_linear": "direct",
        "learned_signed_log_sign_flip": "direct",
    }
    expected_selected_fits = {
        "base": None,
        "constant_plus_one": None,
        "fixed_signed_log": None,
        "fixed_linear": None,
        "learned_signed_log": fits["signed_log"]["artifact_sha256"],
        "learned_linear": fits["linear"]["artifact_sha256"],
        "learned_signed_log_sign_flip": fits["signed_log"]["artifact_sha256"],
    }
    for arm, receipt in normalized.items():
        law, polarity = expected_law_polarity[arm]
        continuous = arm != "base"
        _sha(receipt.get("provider_artifact_sha256"), label=f"{arm} provider")
        _sha(receipt.get("provider_metadata_sha256"), label=f"{arm} metadata")
        if continuous:
            _sha(
                receipt.get("transfer_evidence_sha256"),
                label=f"{arm} transfer evidence",
            )
        raw_weight = receipt.get("response_weight")
        if continuous:
            weights = tuple(float(item) for item in _sequence(raw_weight, label=f"{arm} weights"))
            tensor = _weight_tensor(weights)
            corners = tuple(
                float(item)
                for item in fisher_continuous_bilinear_corner_values(tensor)
            )
            box = max(abs(item) for item in corners)
            tensor_hash = _provider_tensor_sha256(tensor)
        else:
            weights = None
            corners = None
            box = None
            tensor_hash = None
        if (
            receipt.get("arm") != arm
            or receipt.get("fit_bundle_artifact_sha256") != fit_sha
            or receipt.get("base_provider_artifact_sha256") != base_sha
            or receipt.get("proposal_provider_artifact_sha256")
            != (proposal_sha if continuous else None)
            or receipt.get("response_law") != law
            or receipt.get("polarity") != polarity
            or receipt.get("response_source") != expected_sources[arm]
            or receipt.get("selected_fit_artifact_sha256")
            != expected_selected_fits[arm]
            or receipt.get("transfer_protocol_sha256")
            != (_core.TANGENT_RESPONSE_PROTOCOL_SHA256 if continuous else None)
            or receipt.get("signed_log_kappa")
            != (_core.TANGENT_RESPONSE_KAPPA if continuous else None)
            or receipt.get("analysis_only") is not continuous
            or not isinstance(receipt.get("prepared_float_scalar_count"), int)
            or int(receipt.get("prepared_float_scalar_count", 0)) <= 0
            or not isinstance(
                receipt.get("logical_macs_per_token_upper_bound"), int
            )
            or int(receipt.get("logical_macs_per_token_upper_bound", 0)) <= 0
            or receipt.get("response_weight_sha256") != tensor_hash
            or (
                tuple(receipt.get("bilinear_corner_values", ()))
                if continuous
                else receipt.get("bilinear_corner_values")
            )
            != corners
            or receipt.get("bilinear_box_max_abs") != box
            or receipt.get("global_bilinear_box_feasible") is not True
            or receipt.get("rank") != _RANK
            or receipt.get("conditional_rank") != _CONDITIONAL_RANK
            or receipt.get("radial_projection_used") is not False
            or receipt.get("provider_sidecar_serialized") is not False
        ):
            raise ValueError(f"V20e {arm} provider semantics differ")
        if not continuous and (
            receipt.get("provider_artifact_sha256") != base_sha
            or receipt.get("transfer_evidence_sha256") is not None
            or receipt.get("response_weight") is not None
        ):
            raise ValueError("V20e base provider semantics differ")
    if (
        normalized["learned_signed_log"]["provider_artifact_sha256"]
        != fits["signed_log"]["selected_provider_artifact_sha256"]
        or normalized["learned_linear"]["provider_artifact_sha256"]
        != fits["linear"]["selected_provider_artifact_sha256"]
    ):
        raise ValueError("V20e learned providers differ from final fits")
    if normalized["constant_plus_one"]["transfer_evidence_sha256"] != fit_sha:
        raise ValueError("V20e constant control evidence differs from fit bundle")
    expected_weights = {
        "constant_plus_one": (0.0, 0.0, 0.0),
        "fixed_signed_log": _INITIAL_WEIGHTS,
        "fixed_linear": _INITIAL_WEIGHTS,
        "learned_signed_log": tuple(float(item) for item in fits["signed_log"]["selected_weights"]),
        "learned_linear": tuple(float(item) for item in fits["linear"]["selected_weights"]),
        "learned_signed_log_sign_flip": tuple(float(item) for item in fits["signed_log"]["selected_weights"]),
    }
    if any(
        tuple(normalized[arm]["response_weight"]) != expected
        for arm, expected in expected_weights.items()
    ):
        raise ValueError("V20e held provider weights differ from fitted semantics")
    if len(
        {str(normalized[arm]["provider_artifact_sha256"]) for arm in _ARMS}
    ) != len(_ARMS):
        raise ValueError("V20e held provider artifacts are not distinct")
    initial_provider_hashes = _mapping(
        fit_bundle.get("initial_provider_artifact_sha256s_by_law"),
        label="initial provider hashes",
    )
    if (
        normalized["fixed_signed_log"]["provider_artifact_sha256"]
        != initial_provider_hashes["signed_log"]
        or normalized["fixed_linear"]["provider_artifact_sha256"]
        != initial_provider_hashes["linear"]
    ):
        raise ValueError("V20e fixed providers differ from fit initial providers")
    positive = normalized["learned_signed_log"]
    mirror = normalized["learned_signed_log_sign_flip"]
    for field in (
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "response_weight",
        "response_weight_sha256",
        "bilinear_corner_values",
        "bilinear_box_max_abs",
        "response_law",
        "signed_log_kappa",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "selected_fit_artifact_sha256",
        "rank",
        "conditional_rank",
        "prepared_float_scalar_count",
        "logical_macs_per_token_upper_bound",
        "analysis_only",
    ):
        if positive[field] != mirror[field]:
            raise ValueError("V20e learned mirror changes more than polarity")
    expected_bundle_payload = {
        "fit_bundle_artifact_sha256": fit_sha,
        "base_provider_artifact_sha256": base_sha,
        "proposal_provider_artifact_sha256": proposal_sha,
        "arm_order": _ARMS,
        "provider_artifact_sha256s": {
            arm: normalized[arm]["provider_artifact_sha256"] for arm in _ARMS
        },
        "provider_receipt_artifact_sha256s": {
            arm: normalized[arm]["artifact_sha256"] for arm in _ARMS
        },
        "learned_signed_log_fit_artifact_sha256": fits["signed_log"][
            "artifact_sha256"
        ],
        "learned_linear_fit_artifact_sha256": fits["linear"]["artifact_sha256"],
        "all_seven_providers_frozen_before_held_capability": True,
        "held_capability_count_at_freeze": 0,
        "post_cv_all_six_exact_rescore_performed": False,
        "radial_projection_used": False,
        "provider_sidecar_or_raw_tensor_serialized": False,
    }
    expected_bundle = _hashed(expected_bundle_payload, domain=_PROVIDER_BUNDLE_DOMAIN)
    if _v14._canonical_json_bytes(bundle) != _v14._canonical_json_bytes(
        expected_bundle
    ):
        raise ValueError("V20e provider bundle receipt differs")
    if banks is not None and (
        normalized["fixed_signed_log"]["provider_artifact_sha256"]
        != banks["signed_log"].provider.artifact_sha256
        or normalized["fixed_linear"]["provider_artifact_sha256"]
        != banks["linear"].provider.artifact_sha256
    ):
        raise ValueError("V20e fixed providers differ from initial banks")
    if finals is not None and (
        normalized["learned_signed_log"]["provider_artifact_sha256"]
        != finals["signed_log"].provider.artifact_sha256
        or normalized["learned_linear"]["provider_artifact_sha256"]
        != finals["linear"].provider.artifact_sha256
    ):
        raise ValueError("V20e learned providers differ from live finals")
    if initial_evidence_by_law is not None and (
        normalized["fixed_signed_log"]["transfer_evidence_sha256"]
        != initial_evidence_by_law["signed_log"][
            "provider_transfer_evidence_sha256"
        ]
        or normalized["fixed_linear"]["transfer_evidence_sha256"]
        != initial_evidence_by_law["linear"]["provider_transfer_evidence_sha256"]
    ):
        raise ValueError("V20e fixed provider evidence differs from initial banks")
    if final_trace_evidence_by_law is not None and (
        normalized["learned_signed_log"]["transfer_evidence_sha256"]
        != final_trace_evidence_by_law["signed_log"][
            "provider_transfer_evidence_sha256"
        ]
        or normalized["learned_signed_log_sign_flip"]["transfer_evidence_sha256"]
        != final_trace_evidence_by_law["signed_log"][
            "provider_transfer_evidence_sha256"
        ]
        or normalized["learned_linear"]["transfer_evidence_sha256"]
        != final_trace_evidence_by_law["linear"][
            "provider_transfer_evidence_sha256"
        ]
    ):
        raise ValueError("V20e learned provider evidence differs from final traces")
    return normalized, bundle


def _held_execution_sha256(
    *, fit_bundle_sha256: str, provider_bundle_sha256: str, arm: str,
    provider_sha256: str, outer_family_id: str, scored_family_id: str,
    h4_sha256s: Mapping[str, str], logits_sha256s: Mapping[str, str],
    trace_sha256: str, objective: float,
) -> str:
    return _v14._sha256(
        {
            "fit_bundle_artifact_sha256": fit_bundle_sha256,
            "provider_bundle_artifact_sha256": provider_bundle_sha256,
            "arm": arm,
            "provider_artifact_sha256": provider_sha256,
            "outer_held_family_id": outer_family_id,
            "scored_inner_family_id": scored_family_id,
            "post_cast_h4_sha256s": dict(sorted(h4_sha256s.items())),
            "supervised_full_vocab_logits_sha256s": dict(
                sorted(logits_sha256s.items())
            ),
            "response_trace_sha256": trace_sha256,
            "objective": objective,
        },
        domain=_HELD_EXECUTION_DOMAIN,
    )


def _score_held_arm(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    arm: str,
    outer_family_id: str,
    fit_bundle_sha256: str,
    provider_bundle_sha256: str,
    baseline_hashes: tuple[Mapping[str, str], Mapping[str, str]] | None,
) -> tuple[dict[str, object], tuple[dict[str, str], dict[str, str]], dict[str, Tensor]]:
    ordered = _v20b._ordered_records(records)
    families = {record.sequence.family_id for record in ordered}
    if arm not in _ARMS or len(ordered) != 2 or len(families) != 1:
        raise RuntimeError("V20e held arm geometry differs")
    scored_family = next(iter(families))
    trace, gains = _trace_provider(provider, ordered, arm=arm)
    joined = torch.cat(tuple(value.reshape(-1) for value in gains.values()))
    gain_min = float(joined.min())
    gain_max = float(joined.max())
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
        raise RuntimeError("V20e held arm family objective differs")
    changed = bool(
        baseline_hashes is not None
        and (
            h4_hashes != dict(baseline_hashes[0])
            or logits_hashes != dict(baseline_hashes[1])
        )
    )
    execution_sha = _held_execution_sha256(
        fit_bundle_sha256=fit_bundle_sha256,
        provider_bundle_sha256=provider_bundle_sha256,
        arm=arm,
        provider_sha256=provider.artifact_sha256,
        outer_family_id=outer_family_id,
        scored_family_id=scored_family,
        h4_sha256s=h4_hashes,
        logits_sha256s=logits_hashes,
        trace_sha256=str(trace["artifact_sha256"]),
        objective=objective,
    )
    result = {
        "arm": arm,
        "objective": objective,
        "prompt_objectives": dict(sorted(prompt_scores.items())),
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_receipt_sha256": execution_sha,
        "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
        "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
        "response_trace": trace,
        "response_gain_min_on_held_support": gain_min,
        "response_gain_max_on_held_support": gain_max,
        "response_gain_range_on_held_support": gain_max - gain_min,
        "response_gain_nonconstant_on_held_support": gain_max > gain_min,
        "execution_changed_from_base": changed,
    }
    _v14._scalar_report(result)
    return result, (dict(h4_hashes), dict(logits_hashes)), gains


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
        or provider_bundle.get("all_seven_providers_frozen_before_held_capability")
        is not True
        or set(providers) != set(_ARMS)
        or set(provider_receipts) != set(_ARMS)
    ):
        raise PermissionError("V20e held capability barrier is not satisfied")
    selected_records = _v20b._ordered_records(
        tuple(
            record
            for record in records
            if record.sequence.family_id == scored_family_id
        )
    )
    if len(selected_records) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20e held role prompt geometry differs")
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in selected_records),
        held_family_id=outer_family_id,
    )
    raw: dict[str, dict[str, object]] = {}
    transient_gains: dict[str, dict[str, Tensor]] = {}
    base, base_hashes, gains = _score_held_arm(
        context,
        selected_records,
        capability,
        provider=providers["base"],
        arm="base",
        outer_family_id=outer_family_id,
        fit_bundle_sha256=str(fit_bundle["artifact_sha256"]),
        provider_bundle_sha256=str(provider_bundle["artifact_sha256"]),
        baseline_hashes=None,
    )
    raw["base"] = base
    transient_gains["base"] = gains
    for arm in _ARMS:
        if arm == "base":
            continue
        row, _hashes, gains = _score_held_arm(
            context,
            selected_records,
            capability,
            provider=providers[arm],
            arm=arm,
            outer_family_id=outer_family_id,
            fit_bundle_sha256=str(fit_bundle["artifact_sha256"]),
            provider_bundle_sha256=str(provider_bundle["artifact_sha256"]),
            baseline_hashes=base_hashes,
        )
        raw[arm] = row
        transient_gains[arm] = gains
    if set(transient_gains["learned_signed_log"]) != set(
        transient_gains["learned_signed_log_sign_flip"]
    ) or any(
        not torch.equal(
            transient_gains["learned_signed_log_sign_flip"][example],
            -transient_gains["learned_signed_log"][example],
        )
        for example in transient_gains["learned_signed_log"]
    ):
        raise RuntimeError("V20e learned signed-log mirror is not exact")
    arm_semantics = {
        "base": ("base", 0),
        "constant_plus_one": ("constant", 1),
        "fixed_signed_log": ("signed_log", 1),
        "fixed_linear": ("linear", 1),
        "learned_signed_log": ("signed_log", 1),
        "learned_linear": ("linear", 1),
        "learned_signed_log_sign_flip": ("signed_log", -1),
    }
    selected_weight_hashes = _mapping(
        fit_bundle.get("selected_weight_sha256s_by_law"),
        label="selected fit weight hashes",
    )
    held_weight_hashes = {
        "learned_signed_log": selected_weight_hashes["signed_log"],
        "learned_signed_log_sign_flip": selected_weight_hashes["signed_log"],
        "learned_linear": selected_weight_hashes["linear"],
    }
    scores: list[dict[str, object]] = []
    for arm in _ARMS:
        row = raw[arm]
        trace = _mapping(row["response_trace"], label=f"{arm} held trace")
        law, polarity = arm_semantics[arm]
        scores.append(
            _core.build_tangent_response_held_arm_score(
                fit_bundle_receipt=fit_bundle,
                outer_held_family_id=outer_family_id,
                held_family_id=scored_family_id,
                arm=arm,
                response_law=law,
                response_polarity=polarity,
                response_weight_sha256=str(
                    held_weight_hashes.get(arm)
                    or provider_receipts[arm].get("response_weight_sha256")
                    or _provider_tensor_sha256(torch.zeros(3, dtype=torch.float64))
                ),
                objective=float(row["objective"]),
                provider_artifact_sha256=str(row["provider_artifact_sha256"]),
                execution_receipt_sha256=str(row["execution_receipt_sha256"]),
                finite=trace.get("finite") is True,
                pointwise_trust_passed=trace.get("pointwise_trust_passed") is True,
                rank_is_16=trace.get("endpoint_conditional_ranks_are_16") is True,
                execution_changed_from_base=bool(row["execution_changed_from_base"]),
                response_nonconstant=bool(
                    row["response_gain_nonconstant_on_held_support"]
                ),
            )
        )
    role = _core.build_tangent_response_held_role_receipt(
        fit_bundle_receipt=fit_bundle, arm_scores=scores
    )
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(
            record.sequence.example_id for record in selected_records
        ),
        expected_family_count=1,
        expected_held_family_id=outer_family_id,
        expected_accesses_per_example=len(_ARMS),
        label="V20e held role capability",
    )
    evidence = _hashed(
        {
            "fit_bundle_artifact_sha256": fit_bundle["artifact_sha256"],
            "provider_bundle_artifact_sha256": provider_bundle["artifact_sha256"],
            "outer_held_family_id": outer_family_id,
            "scored_inner_family_id": scored_family_id,
            "capability_receipt": capability_receipt,
            "arm_execution_evidence": raw,
            "learned_signed_log_gain_nonconstant_on_held_support": raw[
                "learned_signed_log"
            ]["response_gain_nonconstant_on_held_support"],
            "learned_signed_log_mirror_exact_negative": True,
            "both_fits_and_all_seven_providers_frozen_before_capability": True,
            "historically_reused_reed_sundial_diagnostic_only": True,
        },
        domain=_ROLE_EVIDENCE_DOMAIN,
    )
    return role, evidence


def _pair_qualification(
    *,
    fit_bundle: Mapping[str, object],
    provider_bundle: Mapping[str, object],
    roles: Sequence[Mapping[str, object]],
    role_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    core = _core.build_tangent_response_pair_qualification(
        fit_bundle_receipt=fit_bundle, roles=roles
    )
    evidence = tuple(
        _validate_hashed(
            _mapping(value, label="V20e role evidence"),
            domain=_ROLE_EVIDENCE_DOMAIN,
            label="V20e role evidence",
        )
        for value in role_evidence
    )
    expected_outer = set(str(item) for item in fit_bundle["excluded_family_ids"])
    if (
        len(evidence) != 2
        or {str(value["outer_held_family_id"]) for value in evidence}
        != expected_outer
        or any(
            value.get("fit_bundle_artifact_sha256")
            != fit_bundle.get("artifact_sha256")
            or value.get("provider_bundle_artifact_sha256")
            != provider_bundle.get("artifact_sha256")
            or value.get("learned_signed_log_mirror_exact_negative") is not True
            or value.get(
                "both_fits_and_all_seven_providers_frozen_before_capability"
            )
            is not True
            for value in evidence
        )
    ):
        raise ValueError("V20e reciprocal role evidence differs")
    gates = {
        "core_pair_qualification_passed": core["passed"] is True,
        "learned_signed_log_nonconstant_on_both_roles": all(
            value.get("learned_signed_log_gain_nonconstant_on_held_support")
            is True
            for value in evidence
        ),
        "learned_signed_log_mirror_exact_negative_on_both_roles": True,
        "provider_barrier_bound_to_both_roles": True,
    }
    return _hashed(
        {
            "fit_bundle_artifact_sha256": fit_bundle["artifact_sha256"],
            "provider_bundle_artifact_sha256": provider_bundle["artifact_sha256"],
            "core_pair_qualification": core,
            "role_evidence_artifact_sha256s": tuple(
                value["artifact_sha256"] for value in evidence
            ),
            "runtime_and_barrier_gates": gates,
            "passed": all(gates.values()),
            "scientific_scope": "development_only_historically_reused_a16_pair",
            "fresh_family_disjoint_claim_authorized": False,
        },
        domain=_QUALIFICATION_DOMAIN,
    )


def _stage_work(*, held_scoring_executed: bool, terminal_stage: str | None) -> dict[str, object]:
    planned = _core_work(held_scoring_executed=True)
    if terminal_stage == "cv_direction_preflight":
        observed = {
            **_core_work(held_scoring_executed=False),
            "cv_positive_fraction_score_count": 0,
            "cv_validation_forward_count": 0,
            "tangent_qp_solve_count": 12,
            "total_forward_count": 68,
            "teacher_access_count": 36,
            "logical_cv_candidate_count": 0,
            "positive_cv_candidate_count": 0,
            "cv_prompt_score_forward_count": 0,
            "reused_beta_zero_prompt_score_count": 0,
            "tangent_qp_and_ray_solve_count": 12,
            "provider_only_runtime_trace_count": 0,
            "total_capability_count_including_endpoint_reconstruction": 3,
            "full_model_forward_count": 68,
            "teacher_capability_access_count": 36,
            "post_cast_h4_hash_check_count": 36,
            "supervised_full_vocab_logits_hash_check_count": 36,
        }
    elif terminal_stage == "all_six_direction_preflight":
        observed = {
            **_core_work(held_scoring_executed=False),
            "provider_only_runtime_trace_count": 120,
        }
    else:
        observed = _core_work(held_scoring_executed=held_scoring_executed)
    return {
        "planned_full_budget": planned,
        "observed_stage_counters": observed,
        "observed_matches_planned_full_budget": (
            held_scoring_executed and terminal_stage is None
        ),
        "terminal_stage": terminal_stage,
    }


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    v20d_report: Mapping[str, object],
    workspace: object,
    coordinate_trace: Mapping[str, object],
    banks: Mapping[str, _InitialBank],
    manifest: _ManifestLive | None,
    cv: Mapping[str, _CVLive],
    selection_bundle: Mapping[str, object] | None,
    finals: Mapping[str, _FinalLive],
    fit_bundle: Mapping[str, object] | None,
    provider_receipts: Mapping[str, Mapping[str, object]] | None,
    provider_bundle: Mapping[str, object] | None,
    roles: Sequence[Mapping[str, object]],
    role_evidence: Sequence[Mapping[str, object]],
    qualification: Mapping[str, object] | None,
    terminal: _ControlledTerminal | None,
) -> dict[str, object]:
    held_executed = provider_bundle is not None
    fit_authorized = bool(
        fit_bundle is not None and _fit_stage_authorized(fit_bundle, finals)
    )
    if held_executed != fit_authorized:
        raise RuntimeError("V20e held execution did not follow the fit-stage gate")
    terminal_stage = terminal.stage if terminal is not None else None
    if held_executed:
        if qualification is None or len(roles) != 2 or len(role_evidence) != 2:
            raise RuntimeError("V20e held report evidence is incomplete")
        passed = qualification.get("passed") is True
        classification = (
            "tangent_response_pair_smoke_passed"
            if passed
            else "tangent_response_pair_smoke_failed"
        )
    else:
        passed = False
        if terminal_stage == "cv_direction_preflight":
            classification = "tangent_direction_preflight_failed"
        elif terminal_stage == "all_six_direction_preflight":
            classification = "all_six_tangent_direction_failed"
        else:
            classification = "fit_only_tangent_response_failed"
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "artifact": output.as_posix(),
        "experiment_stage": "A16_development_only_reused_pair_tangent_response",
        "scientific_status": "development_only_historically_reused_a16_pair",
        "fixed_protocol": dict(_FIXED_PROTOCOL),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.TANGENT_RESPONSE_PROTOCOL_SHA256,
        "source": dict(source),
        "source_pair_diagnostic": dict(v20d_report["source_pair_diagnostic"]),
        "panel_receipt": dict(v20d_report["panel_receipt"]),
        "shared_fit_receipt": dict(workspace.fit_receipt),
        "fit_training_evidence": dict(workspace.fit_training_evidence),
        "coordinate_trace_receipt": dict(coordinate_trace),
        "initial_vjp_evidence_by_law": {
            law: banks[law].evidence for law in _LAWS
        },
        "cv_provider_manifest_receipt": (
            manifest.receipt if manifest is not None else None
        ),
        "cv_receipts_by_law": {
            law: cv[law].receipt for law in _LAWS if law in cv
        },
        "cv_fold_execution_evidence_by_law": {
            law: cv[law].fold_evidence for law in _LAWS if law in cv
        },
        "two_law_selection_bundle_receipt": (
            dict(selection_bundle) if selection_bundle is not None else None
        ),
        "all_six_direction_receipts_by_law": {
            law: finals[law].direction for law in _LAWS if law in finals
        },
        "all_six_ray_receipts_by_law": {
            law: finals[law].ray for law in _LAWS if law in finals
        },
        "final_provider_trace_evidence_by_law": {
            law: finals[law].trace_evidence for law in _LAWS if law in finals
        },
        "tangent_response_fit_receipts_by_law": {
            law: finals[law].fit for law in _LAWS if law in finals
        },
        "two_fit_bundle_receipt": (
            dict(fit_bundle) if fit_bundle is not None else None
        ),
        "fit_stage_authorized_for_held_scoring": fit_authorized,
        "provider_receipts": (
            {arm: dict(provider_receipts[arm]) for arm in _ARMS}
            if provider_receipts is not None
            else {}
        ),
        "provider_bundle_receipt": (
            dict(provider_bundle) if provider_bundle is not None else None
        ),
        "roles": tuple(dict(role) for role in roles),
        "role_execution_evidence": tuple(dict(value) for value in role_evidence),
        "pair_qualification": (
            dict(qualification) if qualification is not None else None
        ),
        "controlled_terminal_evidence": (
            dict(terminal.evidence) if terminal is not None else None
        ),
        "held_scoring_executed": held_executed,
        "classification": classification,
        "passed": passed,
        "next_fresh_family_validation_authorized": passed,
        "tangent_constrained_natural_response_fit_implemented": True,
        "six_fold_leave_one_fit_family_out_selection_implemented": True,
        "all_120_provider_slots_frozen_before_cv_capability": (
            manifest is not None
        ),
        "radial_projection_used": False,
        "post_cv_all_six_exact_rescore_performed": False,
        "fresh_family_disjoint_claim_authorized": False,
        "held_fidelity_claim": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "end_to_end_parameter_or_flop_claim": False,
        "candidate": None,
        "provider_sidecar": None,
        "raw_tensors_logits_gradients_targets_or_coordinates_serialized": False,
        "work_accounting": _stage_work(
            held_scoring_executed=held_executed,
            terminal_stage=terminal_stage,
        ),
    }
    _v14._scalar_report(report)
    return report


_REPORT_KEYS = {
    "schema", "format_version", "artifact", "experiment_stage",
    "scientific_status", "fixed_protocol", "runner_protocol_sha256",
    "core_protocol_sha256", "source", "source_pair_diagnostic",
    "panel_receipt", "shared_fit_receipt", "fit_training_evidence",
    "coordinate_trace_receipt", "initial_vjp_evidence_by_law",
    "cv_provider_manifest_receipt", "cv_receipts_by_law",
    "cv_fold_execution_evidence_by_law", "two_law_selection_bundle_receipt",
    "all_six_direction_receipts_by_law", "all_six_ray_receipts_by_law",
    "final_provider_trace_evidence_by_law",
    "tangent_response_fit_receipts_by_law", "two_fit_bundle_receipt",
    "fit_stage_authorized_for_held_scoring", "provider_receipts",
    "provider_bundle_receipt", "roles", "role_execution_evidence",
    "pair_qualification", "controlled_terminal_evidence",
    "held_scoring_executed", "classification", "passed",
    "next_fresh_family_validation_authorized",
    "tangent_constrained_natural_response_fit_implemented",
    "six_fold_leave_one_fit_family_out_selection_implemented",
    "all_120_provider_slots_frozen_before_cv_capability",
    "radial_projection_used", "post_cv_all_six_exact_rescore_performed",
    "fresh_family_disjoint_claim_authorized", "held_fidelity_claim",
    "serving_authorized", "compression_claim", "speed_or_latency_claim",
    "end_to_end_parameter_or_flop_claim", "candidate", "provider_sidecar",
    "raw_tensors_logits_gradients_targets_or_coordinates_serialized",
    "work_accounting", "report_sha256",
}


def _validate_top_level_claims(
    selected: Mapping[str, object], *, output: Path, manifest_created: bool
) -> None:
    if (
        set(selected) != _REPORT_KEYS
        or selected.get("schema") != _SCHEMA
        or selected.get("format_version") != _FORMAT_VERSION
        or selected.get("artifact") != output.as_posix()
        or selected.get("experiment_stage")
        != "A16_development_only_reused_pair_tangent_response"
        or selected.get("scientific_status")
        != "development_only_historically_reused_a16_pair"
        or _v14._canonical_json_bytes(selected.get("fixed_protocol"))
        != _v14._canonical_json_bytes(_FIXED_PROTOCOL)
        or selected.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or selected.get("core_protocol_sha256")
        != _core.TANGENT_RESPONSE_PROTOCOL_SHA256
        or selected.get("tangent_constrained_natural_response_fit_implemented")
        is not True
        or selected.get("six_fold_leave_one_fit_family_out_selection_implemented")
        is not True
        or selected.get("all_120_provider_slots_frozen_before_cv_capability")
        is not manifest_created
        or selected.get("radial_projection_used") is not False
        or selected.get("post_cv_all_six_exact_rescore_performed") is not False
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
    ):
        raise ValueError("V20e top-level authority or claim boundary differs")


def _validate_response_trace(
    value: Mapping[str, object], *, expected_provider_sha256: str,
    expected_family_ids: Sequence[str], expected_example_ids: Sequence[str],
) -> dict[str, object]:
    trace = _validate_hashed(
        value,
        domain=_v20c._RESPONSE_TRACE_DOMAIN,
        label="V20e response trace",
    )
    gains = _mapping(trace.get("response_gain_sha256s"), label="response gains")
    if (
        trace.get("provider_artifact_sha256") != expected_provider_sha256
        or tuple(trace.get("scored_family_ids", ())) != tuple(sorted(expected_family_ids))
        or set(gains) != set(expected_example_ids)
        or trace.get("finite") is not True
        or trace.get("raw_response_or_modal_tensors_serialized") is not False
    ):
        raise ValueError("V20e response trace semantics differ")
    for value in gains.values():
        _sha(value, label="response gain hash")
    return trace


def _validate_initial_evidence(
    value: Mapping[str, object],
    *,
    law: str,
    source: Mapping[str, object],
    endpoint_fit: Mapping[str, object],
) -> dict[str, object]:
    evidence = _validate_hashed(
        value, domain=_INITIAL_DOMAIN, label=f"V20e {law} initial VJP evidence"
    )
    family_map = {
        str(example): str(family)
        for example, family in _mapping(
            evidence.get("training_example_family_ids"),
            label="initial example families",
        ).items()
    }
    example_ids = tuple(sorted(family_map))
    families = tuple(sorted(set(family_map.values())))
    capability = _mapping(evidence.get("capability_receipt"), label="initial capability")
    _v20b._validate_capability_receipt(
        capability,
        expected_example_ids=example_ids,
        expected_family_count=_FIT_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label=f"V20e {law} initial VJP capability",
    )
    objectives_by_family = _mapping(
        evidence.get("initial_objectives_by_family"), label="initial objectives"
    )
    objectives = {
        str(example): float(score)
        for family, rows in objectives_by_family.items()
        for example, score in _mapping(rows, label="initial family objectives").items()
    }
    h4 = _mapping(evidence.get("post_cast_h4_sha256s"), label="initial H4 hashes")
    logits = _mapping(
        evidence.get("supervised_full_vocab_logits_sha256s"),
        label="initial logit hashes",
    )
    executions = _mapping(
        evidence.get("execution_receipt_sha256s"), label="initial executions"
    )
    gradients = _mapping(evidence.get("gradient_sha256s"), label="initial gradients")
    core_gradient_rows = _mapping(
        evidence.get("core_gradient_row_sha256s"),
        label="initial core gradient rows",
    )
    sequences = _mapping(
        evidence.get("training_sequence_sha256s"), label="initial sequences"
    )
    gradient_bank = _core.validate_tangent_response_gradient_bank_receipt(
        _mapping(
            evidence.get("gradient_bank_receipt"),
            label="initial gradient bank",
        ),
        expected_source_artifact_sha256s=_source_sha256s(source),
    )
    bank_summaries = _mapping(
        gradient_bank["family_gradient_summaries_by_family"],
        label="initial gradient-bank family summaries",
    )
    expected_example_ids_by_family = {
        family: tuple(
            sorted(
                example
                for example, example_family in family_map.items()
                if example_family == family
            )
        )
        for family in families
    }
    bank_example_ids_by_family = {
        str(family): tuple(str(example) for example in examples)
        for family, examples in _mapping(
            gradient_bank["fit_example_ids_by_family"],
            label="initial gradient-bank example IDs",
        ).items()
    }
    objective_example_ids_by_family = {
        str(family): tuple(
            sorted(
                str(example)
                for example in _mapping(
                    rows, label="initial family objective rows"
                )
            )
        )
        for family, rows in objectives_by_family.items()
    }
    expected_core_gradient_rows = {
        str(example): _sha(row_hash, label="initial bank row")
        for family_summary in bank_summaries.values()
        for example, row_hash in _mapping(
            _mapping(
                family_summary, label="initial bank family summary"
            )["example_gradient_sha256s"],
            label="initial bank row hashes",
        ).items()
    }
    if (
        bank_example_ids_by_family != expected_example_ids_by_family
        or objective_example_ids_by_family != expected_example_ids_by_family
        or any(
            len(expected_example_ids_by_family[family]) != _PROMPTS_PER_FAMILY
            for family in families
        )
    ):
        raise ValueError(
            f"V20e {law} initial gradient-bank or objective family grouping differs"
        )
    if (
        evidence.get("response_law") != law
        or evidence.get("source_artifact_sha256") != source.get("artifact_sha256")
        or evidence.get("endpoint_fit_artifact_sha256")
        != endpoint_fit.get("artifact_sha256")
        or tuple(evidence.get("initial_weights", ())) != _INITIAL_WEIGHTS
        or evidence.get("initial_weight_tensor_sha256")
        != _provider_tensor_sha256(_weight_tensor(_INITIAL_WEIGHTS))
        or tuple(evidence.get("training_family_ids", ())) != families
        or tuple(evidence.get("held_family_ids", ())) != _v20c._FROZEN_EXCLUDED
        or len(example_ids) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or any(set(mapping) != set(example_ids) for mapping in (objectives, h4, logits, executions, gradients, sequences))
        or set(core_gradient_rows) != set(example_ids)
        or dict(core_gradient_rows) != expected_core_gradient_rows
        or gradient_bank.get("response_law") != law
        or tuple(gradient_bank.get("family_ids", ()))
        != tuple(sorted((*families, *_v20c._FROZEN_EXCLUDED)))
        or tuple(gradient_bank.get("fit_family_ids", ())) != families
        or gradient_bank.get("base_provider_artifact_sha256")
        != endpoint_fit.get("base_provider_artifact_sha256")
        or gradient_bank.get("proposal_provider_artifact_sha256")
        != endpoint_fit.get("proposal_provider_artifact_sha256")
        or evidence.get("gradient_bank_frozen_before_direction_solves") is not True
        or evidence.get("direction_solve_count_at_gradient_bank_freeze") != 0
        or evidence.get(
            "gradient_tensor_and_core_row_hashes_collected_from_same_live_tensor"
        )
        is not True
        or evidence.get("unique_empirical_fisher_gradient_row_count")
        != len(example_ids)
        or evidence.get("empirical_fisher_outer_product_evaluation_count")
        != len(example_ids)
        or evidence.get("full_suffix_vjp_count") != len(example_ids)
        or evidence.get("local_response_autograd_contraction_count")
        != len(example_ids)
        or evidence.get("beta_zero_exact_execution_reusable_by_all_six_folds")
        is not True
        or evidence.get("all_initial_executions_finite") is not True
        or evidence.get("all_initial_executions_exact") is not True
        or evidence.get("held_data_or_objectives_used") is not False
        or evidence.get("raw_gradients_h4_logits_targets_or_tensors_serialized")
        is not False
    ):
        raise ValueError(f"V20e {law} initial VJP evidence semantics differ")
    provider_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "source_artifact_sha256": source["artifact_sha256"],
            "endpoint_fit_artifact_sha256": endpoint_fit["artifact_sha256"],
            "response_law": law,
            "weights": _INITIAL_WEIGHTS,
            "scope": "all_six_shared_beta_zero",
            "held_rows_used": False,
        },
        domain=_PROVIDER_SEED_DOMAIN,
    )
    if evidence.get("provider_transfer_evidence_sha256") != provider_seed:
        raise ValueError("V20e initial provider evidence seed differs")
    for example in example_ids:
        family = family_map[example]
        expected = _initial_execution_sha256(
            law=law,
            provider_sha256=str(evidence["provider_artifact_sha256"]),
            example_id=example,
            family_id=family,
            objective=objectives[example],
            h4_sha256=_sha(h4[example], label="initial H4"),
            logits_sha256=_sha(logits[example], label="initial logits"),
        )
        if executions[example] != expected:
            raise ValueError("V20e initial execution receipt differs")
        _sha(gradients[example], label="initial gradient")
        _sha(sequences[example], label="initial sequence")
    return evidence


def _validate_manifest(
    value: Mapping[str, object], *, expected_source_sha256: str | None = None
) -> dict[str, object]:
    manifest = _validate_hashed(
        value, domain=_MANIFEST_DOMAIN, label="V20e CV provider manifest"
    )
    slots_by_law = _mapping(
        manifest.get("provider_slots_by_law_and_fold"), label="manifest slots"
    )
    direction_maps = _mapping(
        manifest.get("direction_artifact_sha256s_by_law_and_fold"),
        label="manifest direction hashes",
    )
    ray_maps = _mapping(
        manifest.get("ray_artifact_sha256s_by_law_and_fold"),
        label="manifest ray hashes",
    )
    folds = tuple(manifest.get("fold_order", ()))
    if (
        manifest.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or manifest.get("core_protocol_sha256")
        != _core.TANGENT_RESPONSE_PROTOCOL_SHA256
        or (
            expected_source_sha256 is not None
            and manifest.get("source_artifact_sha256") != expected_source_sha256
        )
        or
        tuple(manifest.get("law_order", ())) != _LAWS
        or len(folds) != _FIT_FAMILY_COUNT
        or tuple(manifest.get("fraction_order", ())) != _FRACTIONS
        or set(slots_by_law) != set(_LAWS)
        or manifest.get("logical_provider_slot_count") != 120
        or manifest.get("positive_provider_artifact_count") != 108
        or manifest.get("positive_provider_hashes_unique") is not True
        or manifest.get("beta_zero_provider_reused_across_six_folds_per_law")
        is not True
        or manifest.get("all_slots_frozen_before_first_cv_capability") is not True
        or manifest.get("cv_capability_count_at_freeze") != 0
        or manifest.get("radial_projection_used") is not False
        or manifest.get("held_data_or_objectives_used") is not False
        or manifest.get("raw_tensors_or_provider_sidecars_serialized") is not False
    ):
        raise ValueError("V20e CV provider manifest authority differs")
    positive: list[str] = []
    zero: dict[str, set[str]] = {law: set() for law in _LAWS}
    for law in _LAWS:
        law_slots = _mapping(slots_by_law[law], label=f"{law} manifest folds")
        law_directions = _mapping(direction_maps[law], label="law direction hashes")
        law_rays = _mapping(ray_maps[law], label="law ray hashes")
        if (
            set(law_slots) != set(folds)
            or set(law_directions) != set(folds)
            or set(law_rays) != set(folds)
        ):
            raise ValueError("V20e manifest fold geometry differs")
        for family in folds:
            direction_sha = _sha(law_directions[family], label="manifest direction")
            ray_sha = _sha(law_rays[family], label="manifest ray")
            rows = _mapping(law_slots[family], label="manifest fraction rows")
            if set(rows) != {_fraction_key(value) for value in _FRACTIONS}:
                raise ValueError("V20e manifest fraction geometry differs")
            for fraction in _FRACTIONS:
                row = _mapping(rows[_fraction_key(fraction)], label="manifest slot")
                weights = tuple(float(item) for item in row.get("weights", ()))
                # The live provider records the Torch boundary calculation.
                # Python ``fsum`` can differ by one ULP on an exactly feasible
                # face, so replay must authenticate the same implementation
                # while independently requiring the exact core certificate to
                # remain inside the box.
                runtime_box_certificate = fisher_continuous_bilinear_box_max_abs(
                    _weight_tensor(weights)
                )
                exact_box_certificate = _core.bilinear_box_certificate(weights)
                if (
                    row.get("fraction") != fraction
                    or len(weights) != 3
                    or row.get("box_certificate")
                    != runtime_box_certificate
                    or float(row["box_certificate"]) > 1.0
                    or exact_box_certificate > 1.0
                    or row.get("weight_tensor_sha256")
                    != _provider_tensor_sha256(_weight_tensor(weights))
                    or row.get("direction_artifact_sha256") != direction_sha
                    or row.get("ray_artifact_sha256") != ray_sha
                    or row.get("radial_projection_used") is not False
                ):
                    raise ValueError("V20e manifest slot semantics differ")
                provider_sha = _sha(
                    row.get("provider_artifact_sha256"), label="manifest provider"
                )
                _sha(
                    row.get("provider_transfer_evidence_sha256"),
                    label="manifest provider evidence",
                )
                _sha(row.get("weight_tensor_sha256"), label="manifest weight")
                if fraction == 0.0:
                    if weights != _INITIAL_WEIGHTS:
                        raise ValueError("V20e beta-zero manifest weights differ")
                    zero[law].add(provider_sha)
                else:
                    expected_seed = _v14._sha256(
                        {
                            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                            "source_artifact_sha256": manifest[
                                "source_artifact_sha256"
                            ],
                            "response_law": law,
                            "validation_family_id": family,
                            "direction_artifact_sha256": direction_sha,
                            "ray_artifact_sha256": ray_sha,
                            "fraction": fraction,
                            "weights": weights,
                            "radial_projection_used": False,
                            "held_rows_used": False,
                        },
                        domain=_PROVIDER_SEED_DOMAIN,
                    )
                    if row.get("provider_transfer_evidence_sha256") != expected_seed:
                        raise ValueError("V20e manifest provider seed differs")
                    positive.append(provider_sha)
    if len(positive) != 108 or len(set(positive)) != 108 or any(
        len(values) != 1 for values in zero.values()
    ):
        raise ValueError("V20e manifest provider uniqueness differs")
    expected_zero = {law: next(iter(zero[law])) for law in _LAWS}
    if manifest.get("beta_zero_provider_artifact_sha256s_by_law") != expected_zero:
        raise ValueError("V20e manifest beta-zero provider binding differs")
    return manifest


def _validate_cv_candidate_manifest_weight_binding(
    candidate: Mapping[str, object], manifest_slot: Mapping[str, object]
) -> None:
    """Bind one weight vector while respecting each receipt's arithmetic.

    Core candidates record the exact Python corner certificate; live provider
    slots record the Torch certificate.  Both are deterministic and feasible,
    but they may straddle a one-ULP rounding boundary.
    """

    candidate_weights = tuple(
        float(item) for item in candidate.get("weights", ())
    )
    manifest_weights = tuple(
        float(item) for item in manifest_slot.get("weights", ())
    )
    if (
        len(candidate_weights) != 3
        or candidate_weights != manifest_weights
        or candidate.get("box_certificate")
        != _core.bilinear_box_certificate(candidate_weights)
        or manifest_slot.get("box_certificate")
        != fisher_continuous_bilinear_box_max_abs(
            _weight_tensor(manifest_weights)
        )
    ):
        raise ValueError("V20e CV candidate/manifest weight binding differs")


def _validate_cv_fold_evidence(
    value: Mapping[str, object],
    *,
    law: str,
    cv_receipt: Mapping[str, object],
    manifest: Mapping[str, object],
    initial: Mapping[str, object],
) -> dict[str, object]:
    fold = _validate_hashed(
        value, domain=_CV_FOLD_DOMAIN, label=f"V20e {law} CV fold evidence"
    )
    family = _identifier(
        fold.get("validation_family_id"), label="CV validation family"
    )
    candidates_by_family = _mapping(
        cv_receipt.get("candidate_receipts_by_validation_family"),
        label="CV candidate receipts",
    )
    if family not in candidates_by_family:
        raise ValueError("V20e CV fold family is absent from the CV receipt")
    core_candidates = {
        float(item["fraction"]): _mapping(item, label="core CV candidate")
        for item in _sequence(
            candidates_by_family[family], label="core CV fold candidates"
        )
        if isinstance(item, Mapping)
    }
    rows = _mapping(
        fold.get("candidate_execution_evidence_by_fraction"),
        label="CV execution rows",
    )
    manifest_slot_rows = _mapping(
        _mapping(
            _mapping(
                manifest.get("provider_slots_by_law_and_fold"),
                label="manifest slots",
            )[law],
            label="manifest law slots",
        )[family],
        label="manifest fold slots",
    )
    manifest_direction = _sha(
        _mapping(
            _mapping(
                manifest.get("direction_artifact_sha256s_by_law_and_fold"),
                label="manifest directions",
            )[law],
            label="manifest law directions",
        )[family],
        label="manifest fold direction",
    )
    manifest_ray = _sha(
        _mapping(
            _mapping(
                manifest.get("ray_artifact_sha256s_by_law_and_fold"),
                label="manifest rays",
            )[law],
            label="manifest law rays",
        )[family],
        label="manifest fold ray",
    )
    cv_direction = _sha(
        _mapping(
            cv_receipt.get(
                "direction_artifact_sha256s_by_validation_family"
            ),
            label="CV direction hashes",
        )[family],
        label="CV fold direction",
    )
    cv_ray = _sha(
        _mapping(
            cv_receipt.get("ray_artifact_sha256s_by_validation_family"),
            label="CV ray hashes",
        )[family],
        label="CV fold ray",
    )
    if (
        fold.get("manifest_artifact_sha256") != manifest.get("artifact_sha256")
        or fold.get("response_law") != law
        or fold.get("direction_artifact_sha256") != manifest_direction
        or fold.get("direction_artifact_sha256") != cv_direction
        or fold.get("ray_artifact_sha256") != manifest_ray
        or fold.get("ray_artifact_sha256") != cv_ray
        or tuple(fold.get("fraction_order", ())) != _FRACTIONS
        or set(rows) != {_fraction_key(item) for item in _FRACTIONS}
        or set(core_candidates) != set(_FRACTIONS)
        or fold.get("positive_fraction_accesses_per_example")
        != len(_POSITIVE_FRACTIONS)
        or fold.get("beta_zero_teacher_access_count") != 0
        or fold.get("beta_zero_execution_reused_from_initial_vjp") is not True
        or fold.get("manifest_frozen_before_capability") is not True
        or fold.get("held_data_or_objectives_used") is not False
        or fold.get("raw_tensors_logits_targets_or_coordinates_serialized")
        is not False
    ):
        raise ValueError("V20e CV fold authority differs")
    zero_core = core_candidates[0.0]
    example_ids = tuple(zero_core["validation_example_ids"])
    capability = _mapping(fold.get("capability_receipt"), label="CV capability")
    _v20b._validate_capability_receipt(
        capability,
        expected_example_ids=example_ids,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_POSITIVE_FRACTIONS),
        label=f"V20e {law} {family} CV capability",
    )
    initial_objectives = _mapping(
        _mapping(
            initial.get("initial_objectives_by_family"), label="initial objectives"
        ).get(family),
        label="initial fold objectives",
    )
    initial_h4 = _mapping(initial.get("post_cast_h4_sha256s"), label="initial H4")
    initial_logits = _mapping(
        initial.get("supervised_full_vocab_logits_sha256s"), label="initial logits"
    )
    initial_exec = _mapping(
        initial.get("execution_receipt_sha256s"), label="initial executions"
    )
    base_h4: dict[str, str] = {}
    base_logits: dict[str, str] = {}
    for fraction in _FRACTIONS:
        wrapper = _mapping(rows[_fraction_key(fraction)], label="CV evidence wrapper")
        candidate = _mapping(wrapper.get("candidate_receipt"), label="CV candidate")
        evidence = _validate_hashed(
            _mapping(wrapper.get("execution_evidence"), label="CV execution evidence"),
            domain=_CV_EXECUTION_DOMAIN,
            label="V20e CV execution evidence",
        )
        core_candidate = core_candidates[fraction]
        manifest_slot = _mapping(
            manifest_slot_rows[_fraction_key(fraction)], label="manifest slot"
        )
        _validate_cv_candidate_manifest_weight_binding(candidate, manifest_slot)
        if _v14._canonical_json_bytes(candidate) != _v14._canonical_json_bytes(
            core_candidate
        ):
            raise ValueError("V20e fold candidate differs from CV receipt")
        provider_sha = _sha(
            evidence.get("provider_artifact_sha256"), label="CV provider"
        )
        trace = _validate_response_trace(
            _mapping(evidence.get("response_trace"), label="CV response trace"),
            expected_provider_sha256=provider_sha,
            expected_family_ids=(family,),
            expected_example_ids=example_ids,
        )
        objectives = {
            str(key): float(item)
            for key, item in _mapping(
                evidence.get("objectives_by_example"), label="CV objectives"
            ).items()
        }
        h4 = {
            str(key): _sha(item, label="CV H4")
            for key, item in _mapping(
                evidence.get("post_cast_h4_sha256s"), label="CV H4 hashes"
            ).items()
        }
        logits = {
            str(key): _sha(item, label="CV logits")
            for key, item in _mapping(
                evidence.get("supervised_full_vocab_logits_sha256s"),
                label="CV logit hashes",
            ).items()
        }
        executions = {
            str(key): _sha(item, label="CV execution")
            for key, item in _mapping(
                evidence.get("execution_receipt_sha256s"),
                label="CV execution hashes",
            ).items()
        }
        if (
            evidence.get("manifest_artifact_sha256") != manifest["artifact_sha256"]
            or evidence.get("response_law") != law
            or evidence.get("validation_family_id") != family
            or evidence.get("fraction") != fraction
            or evidence.get("provider_artifact_sha256")
            != manifest_slot.get("provider_artifact_sha256")
            or set(objectives) != set(example_ids)
            or set(h4) != set(example_ids)
            or set(logits) != set(example_ids)
            or set(executions) != set(example_ids)
            or evidence.get("exact_finite_execution") is not True
            or evidence.get("manifest_frozen_before_capability") is not True
            or evidence.get("held_data_or_objectives_used") is not False
            or evidence.get("raw_tensors_logits_targets_or_coordinates_serialized")
            is not False
            or candidate.get("execution_evidence_sha256")
            != evidence.get("artifact_sha256")
            or candidate.get("provider_artifact_sha256") != provider_sha
            or candidate.get("direction_artifact_sha256")
            != manifest_slot.get("direction_artifact_sha256")
            or candidate.get("ray_artifact_sha256")
            != manifest_slot.get("ray_artifact_sha256")
            or candidate.get("finite") is not (trace.get("finite") is True)
            or candidate.get("pointwise_trust_passed")
            is not (trace.get("pointwise_trust_passed") is True)
            or candidate.get("rank_is_16")
            is not (trace.get("endpoint_conditional_ranks_are_16") is True)
            or candidate.get("execution_exact")
            is not (evidence.get("exact_finite_execution") is True)
            or candidate.get("validation_objectives_by_example") != objectives
            or candidate.get("validation_execution_receipt_sha256s_by_example")
            != executions
        ):
            raise ValueError("V20e CV execution binding differs")
        if fraction == 0.0:
            expected_objectives = {
                example: float(initial_objectives[example]) for example in example_ids
            }
            if (
                objectives != expected_objectives
                or provider_sha != initial.get("provider_artifact_sha256")
                or manifest_slot.get("provider_transfer_evidence_sha256")
                != initial.get("provider_transfer_evidence_sha256")
                or h4 != {example: initial_h4[example] for example in example_ids}
                or logits
                != {example: initial_logits[example] for example in example_ids}
                or executions
                != {example: initial_exec[example] for example in example_ids}
                or evidence.get("beta_zero_reused_from_same_law_initial_vjp")
                is not True
                or evidence.get("execution_changed_from_beta_zero") is not False
            ):
                raise ValueError("V20e beta-zero reuse differs")
            base_h4 = h4
            base_logits = logits
        else:
            expected_executions = {
                example: _cv_execution_sha256(
                    manifest_sha256=str(manifest["artifact_sha256"]),
                    law=law,
                    validation_family_id=family,
                    fraction=fraction,
                    provider_sha256=provider_sha,
                    example_id=example,
                    objective=objectives[example],
                    h4_sha256=h4[example],
                    logits_sha256=logits[example],
                    trace_sha256=str(trace["artifact_sha256"]),
                )
                for example in example_ids
            }
            changed = h4 != base_h4 or logits != base_logits
            if (
                executions != expected_executions
                or evidence.get("beta_zero_reused_from_same_law_initial_vjp")
                is not False
                or evidence.get("execution_changed_from_beta_zero") is not changed
                or candidate.get("execution_changed_from_baseline") is not changed
            ):
                raise ValueError("V20e positive CV execution differs")
    return fold


def _validate_selection_bundle(
    value: Mapping[str, object], *, cv: Mapping[str, Mapping[str, object]],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    selected = _validate_hashed(
        value, domain=_SELECTION_DOMAIN, label="V20e two-law selection bundle"
    )
    expected = _build_selection_bundle(
        {
            law: _CVLive(law=law, receipt=dict(cv[law]), fold_evidence=())
            for law in _LAWS
        },
        manifest=manifest,
    )
    if _v14._canonical_json_bytes(selected) != _v14._canonical_json_bytes(expected):
        raise ValueError("V20e two-law selection bundle differs")
    return selected


def _validate_cv_stage(
    selected: Mapping[str, object],
    *,
    initial: Mapping[str, Mapping[str, object]],
    source: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    manifest = _validate_manifest(
        _mapping(selected.get("cv_provider_manifest_receipt"), label="V20e provider manifest"),
        expected_source_sha256=(
            str(source["artifact_sha256"]) if source is not None else None
        ),
    )
    zero_hashes = _mapping(
        manifest.get("beta_zero_provider_artifact_sha256s_by_law"),
        label="manifest beta-zero providers",
    )
    manifest_slots = _mapping(
        manifest.get("provider_slots_by_law_and_fold"), label="manifest slots"
    )
    for law in _LAWS:
        if zero_hashes[law] != initial[law].get("provider_artifact_sha256"):
            raise ValueError("V20e manifest beta-zero provider differs from initial bank")
        for fold_rows in _mapping(
            manifest_slots[law], label="manifest law folds"
        ).values():
            zero = _mapping(fold_rows, label="manifest fold slots")[
                _fraction_key(0.0)
            ]
            if zero["provider_transfer_evidence_sha256"] != initial[law].get(
                "provider_transfer_evidence_sha256"
            ):
                raise ValueError("V20e manifest beta-zero evidence differs from initial bank")
    raw_cv = _mapping(selected.get("cv_receipts_by_law"), label="V20e CV receipts")
    raw_fold = _mapping(
        selected.get("cv_fold_execution_evidence_by_law"),
        label="V20e CV fold evidence",
    )
    if set(raw_cv) != set(_LAWS) or set(raw_fold) != set(_LAWS):
        raise ValueError("V20e CV law geometry differs")
    cv = {
        law: _core.validate_tangent_response_cv_receipt(
            _mapping(raw_cv[law], label=f"{law} CV receipt")
        )
        for law in _LAWS
    }
    for law in _LAWS:
        rows = tuple(
            _validate_cv_fold_evidence(
                _mapping(item, label=f"{law} CV fold evidence"),
                law=law,
                cv_receipt=cv[law],
                manifest=manifest,
                initial=initial[law],
            )
            for item in _sequence(raw_fold[law], label=f"{law} CV folds")
        )
        if len(rows) != _FIT_FAMILY_COUNT or {
            row["validation_family_id"] for row in rows
        } != set(cv[law]["fold_order"]):
            raise ValueError("V20e CV fold evidence coverage differs")
        manifest_directions = _mapping(
            manifest["direction_artifact_sha256s_by_law_and_fold"],
            label="manifest directions",
        )[law]
        manifest_rays = _mapping(
            manifest["ray_artifact_sha256s_by_law_and_fold"],
            label="manifest rays",
        )[law]
        if (
            cv[law]["direction_artifact_sha256s_by_validation_family"]
            != manifest_directions
            or cv[law]["ray_artifact_sha256s_by_validation_family"]
            != manifest_rays
        ):
            raise ValueError("V20e CV receipts differ from provider manifest")
    selection = _validate_selection_bundle(
        _mapping(
            selected.get("two_law_selection_bundle_receipt"),
            label="V20e selection bundle",
        ),
        cv=cv,
        manifest=manifest,
    )
    return manifest, cv, selection


def _validate_cv_panel_lineage(
    *, manifest: Mapping[str, object], cv: Mapping[str, Mapping[str, object]],
    family_ids: Sequence[str], fit_family_ids: Sequence[str],
    source: Mapping[str, object],
    endpoint_fit: Mapping[str, object],
    initial: Mapping[str, Mapping[str, object]],
) -> None:
    families = tuple(sorted(family_ids))
    fit = tuple(sorted(fit_family_ids))
    if tuple(manifest.get("fold_order", ())) != fit:
        raise ValueError("V20e manifest fold order differs from authenticated panel")
    for law in _LAWS:
        receipt = cv[law]
        directions = {
            str(family): _mapping(item, label="CV direction endpoint binding")
            for family, item in _mapping(
                receipt.get("direction_receipts_by_validation_family"),
                label="CV directions",
            ).items()
        }
        for validation_family, direction in directions.items():
            _validate_direction_against_initial_bank(
                direction,
                initial_evidence=initial[law],
                validation_family_id=validation_family,
            )
        if (
            receipt.get("source_artifact_sha256s") != _source_sha256s(source)
            or
            tuple(receipt.get("family_ids", ())) != families
            or tuple(receipt.get("excluded_family_ids", ()))
            != tuple(sorted(_v20c._FROZEN_EXCLUDED))
            or tuple(receipt.get("fit_family_ids", ())) != fit
            or tuple(receipt.get("fold_order", ())) != fit
            or any(
                direction.get("base_provider_artifact_sha256")
                != endpoint_fit.get("base_provider_artifact_sha256")
                or direction.get("proposal_provider_artifact_sha256")
                != endpoint_fit.get("proposal_provider_artifact_sha256")
                or direction.get("gradient_evidence_sha256")
                != initial[law].get("artifact_sha256")
                for direction in directions.values()
            )
        ):
            raise ValueError("V20e CV lineage differs from authenticated panel")


def _validate_direction_against_initial_bank(
    direction: Mapping[str, object], *,
    initial_evidence: Mapping[str, object],
    validation_family_id: str | None,
) -> dict[str, object]:
    bank = _core.validate_tangent_response_gradient_bank_receipt(
        _mapping(
            initial_evidence.get("gradient_bank_receipt"),
            label="initial gradient bank",
        )
    )
    expected = (
        _core.build_tangent_response_direction_from_gradient_bank_receipt(
            gradient_bank_receipt=bank,
            gradient_evidence_sha256=_sha(
                initial_evidence.get("artifact_sha256"),
                label="initial evidence",
            ),
            validation_family_id=validation_family_id,
        )
    )
    if _v14._canonical_json_bytes(direction) != _v14._canonical_json_bytes(
        expected
    ):
        raise ValueError("V20e direction differs from its frozen initial bank")
    return expected


def _canonical_bank_from_fold_directions(
    directions: Mapping[str, Mapping[str, object]],
    *,
    expected_family_ids: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, dict[str, str]]]:
    expected = tuple(sorted(expected_family_ids))
    ids: dict[str, tuple[str, ...]] = {}
    hashes: dict[str, dict[str, str]] = {}
    for family in expected:
        observations: list[tuple[tuple[str, ...], dict[str, str]]] = []
        for direction in directions.values():
            raw_ids = _mapping(
                direction.get("fit_example_ids_by_family"),
                label="fold direction example IDs",
            )
            if family not in raw_ids:
                continue
            family_ids = tuple(str(item) for item in raw_ids[family])
            raw_hashes = _mapping(
                _mapping(
                    direction.get("example_gradient_sha256s_by_family"),
                    label="fold direction gradient hashes",
                )[family],
                label="family gradient hashes",
            )
            family_hashes = {
                example: _sha(raw_hashes[example], label="fold gradient")
                for example in family_ids
            }
            observations.append((family_ids, family_hashes))
        if not observations or any(
            _v14._canonical_json_bytes(value)
            != _v14._canonical_json_bytes(observations[0])
            for value in observations[1:]
        ):
            raise ValueError("V20e overlapping fold gradient bank differs")
        ids[family], hashes[family] = observations[0]
    flat = tuple(example for family in expected for example in ids[family])
    if (
        any(len(ids[family]) != _PROMPTS_PER_FAMILY for family in expected)
        or len(flat) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or len(set(flat)) != len(flat)
    ):
        raise ValueError("V20e canonical fold gradient bank geometry differs")
    return ids, hashes


def _validate_all_six_bank_against_cv(
    direction: Mapping[str, object], *, cv_receipt: Mapping[str, object],
    initial_evidence: Mapping[str, object],
) -> None:
    _validate_direction_against_initial_bank(
        direction,
        initial_evidence=initial_evidence,
        validation_family_id=None,
    )
    fold_directions = {
        str(family): _mapping(item, label="CV fold direction")
        for family, item in _mapping(
            cv_receipt.get("direction_receipts_by_validation_family"),
            label="CV fold directions",
        ).items()
    }
    expected_ids, expected_hashes = _canonical_bank_from_fold_directions(
        fold_directions,
        expected_family_ids=tuple(cv_receipt["fit_family_ids"]),
    )
    if (
        direction.get("fit_example_ids_by_family") != expected_ids
        or direction.get("example_gradient_sha256s_by_family") != expected_hashes
    ):
        raise ValueError("V20e all-six direction differs from canonical CV bank")


def _validate_cv_direction_terminal(
    value: Mapping[str, object], *, source: Mapping[str, object],
    endpoint_fit: Mapping[str, object], family_ids: Sequence[str],
    fit_family_ids: Sequence[str], initial: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    terminal = _validate_hashed(
        value,
        domain=_MANIFEST_DOMAIN,
        label="V20e direction preflight terminal",
    )
    directions_raw = _mapping(
        terminal.get("direction_receipts_by_law_and_fold"),
        label="terminal directions",
    )
    rays_raw = _mapping(
        terminal.get("ray_receipts_by_law_and_fold"), label="terminal rays"
    )
    if set(directions_raw) != set(_LAWS) or set(rays_raw) != set(_LAWS):
        raise ValueError("V20e direction terminal law geometry differs")
    recomputed: list[dict[str, object]] = []
    all_folds: set[str] | None = None
    for law in _LAWS:
        law_directions = _mapping(directions_raw[law], label="terminal law directions")
        law_rays = _mapping(rays_raw[law], label="terminal law rays")
        normalized_law_directions: dict[str, dict[str, object]] = {}
        if len(law_directions) != 6 or set(law_directions) != set(law_rays):
            raise ValueError("V20e direction terminal fold geometry differs")
        if all_folds is None:
            all_folds = set(str(item) for item in law_directions)
        elif set(law_directions) != all_folds:
            raise ValueError("V20e direction terminal folds differ by law")
        for family in sorted(law_directions):
            direction = _core.validate_tangent_response_direction_receipt(
                _mapping(law_directions[family], label="terminal direction"),
                expected_source_artifact_sha256s=_source_sha256s(source),
            )
            _validate_direction_against_initial_bank(
                direction,
                initial_evidence=initial[law],
                validation_family_id=str(family),
            )
            normalized_law_directions[str(family)] = direction
            ray = _core.validate_tangent_response_ray_receipt(
                _mapping(law_rays[family], label="terminal ray"),
                direction_receipt=direction,
            )
            if direction["response_law"] != law or direction["validation_family_id"] != family:
                raise ValueError("V20e direction terminal lineage differs")
            if (
                tuple(direction["family_ids"]) != tuple(sorted(family_ids))
                or tuple(direction["excluded_family_ids"])
                != tuple(sorted(_v20c._FROZEN_EXCLUDED))
                or tuple(direction["fit_family_ids"])
                != tuple(sorted(fit_family_ids))
                or direction["gradient_evidence_sha256"]
                != initial[law].get("artifact_sha256")
                or
                direction["base_provider_artifact_sha256"]
                != endpoint_fit.get("base_provider_artifact_sha256")
                or direction["proposal_provider_artifact_sha256"]
                != endpoint_fit.get("proposal_provider_artifact_sha256")
            ):
                raise ValueError("V20e direction terminal endpoints differ")
            if ray["direction_degenerate"] is True or direction["strict_descent_direction"] is not True:
                recomputed.append(
                    {
                        "response_law": law,
                        "validation_family_id": family,
                        "direction_artifact_sha256": direction["artifact_sha256"],
                        "ray_artifact_sha256": ray["artifact_sha256"],
                        "strict_descent_direction": direction[
                            "strict_descent_direction"
                        ],
                        "direction_degenerate": ray["direction_degenerate"],
                    }
                )
        canonical_ids, _canonical_hashes = _canonical_bank_from_fold_directions(
            normalized_law_directions,
            expected_family_ids=fit_family_ids,
        )
        expected_ids = {
            family: tuple(
                sorted(
                    example
                    for example, example_family in _mapping(
                        initial[law]["training_example_family_ids"],
                        label="initial example families",
                    ).items()
                    if example_family == family
                )
            )
            for family in fit_family_ids
        }
        if canonical_ids != expected_ids:
            raise ValueError("V20e terminal fold bank differs from initial VJP rows")
    if (
        all_folds != set(fit_family_ids)
        or
        terminal.get("source_artifact_sha256") != source.get("artifact_sha256")
        or _v14._canonical_json_bytes(terminal.get("degenerate_rows"))
        != _v14._canonical_json_bytes(tuple(recomputed))
        or not recomputed
        or terminal.get("provider_manifest_created") is not False
        or terminal.get("cv_capability_count") != 0
        or terminal.get("held_capability_count") != 0
        or terminal.get("radial_projection_used") is not False
        or terminal.get("classification") != "tangent_direction_preflight_failed"
    ):
        raise ValueError("V20e direction terminal authority differs")
    return terminal


def _validate_all_six_terminal(
    value: Mapping[str, object], *, source: Mapping[str, object],
    selection: Mapping[str, object], endpoint_fit: Mapping[str, object],
    family_ids: Sequence[str], fit_family_ids: Sequence[str],
    initial: Mapping[str, Mapping[str, object]],
    cv: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    terminal = _validate_hashed(
        value,
        domain=_FINAL_TRACE_DOMAIN,
        label="V20e all-six direction terminal",
    )
    directions_raw = _mapping(
        terminal.get("direction_receipts_by_law"), label="all-six directions"
    )
    rays_raw = _mapping(terminal.get("ray_receipts_by_law"), label="all-six rays")
    if set(directions_raw) != set(_LAWS) or set(rays_raw) != set(_LAWS):
        raise ValueError("V20e all-six terminal law geometry differs")
    degenerate: list[str] = []
    for law in _LAWS:
        direction = _core.validate_tangent_response_direction_receipt(
            _mapping(directions_raw[law], label="all-six direction"),
            expected_source_artifact_sha256s=_source_sha256s(source),
        )
        ray = _core.validate_tangent_response_ray_receipt(
            _mapping(rays_raw[law], label="all-six ray"),
            direction_receipt=direction,
        )
        if direction["response_law"] != law or direction["validation_family_id"] is not None:
            raise ValueError("V20e all-six terminal lineage differs")
        _validate_all_six_bank_against_cv(
            direction,
            cv_receipt=cv[law],
            initial_evidence=initial[law],
        )
        if (
            tuple(direction["family_ids"]) != tuple(sorted(family_ids))
            or tuple(direction["excluded_family_ids"])
            != tuple(sorted(_v20c._FROZEN_EXCLUDED))
            or tuple(direction["fit_family_ids"])
            != tuple(sorted(fit_family_ids))
            or direction["gradient_evidence_sha256"]
            != initial[law].get("artifact_sha256")
            or
            direction["base_provider_artifact_sha256"]
            != endpoint_fit.get("base_provider_artifact_sha256")
            or direction["proposal_provider_artifact_sha256"]
            != endpoint_fit.get("proposal_provider_artifact_sha256")
        ):
            raise ValueError("V20e all-six terminal endpoints differ")
        if ray["direction_degenerate"] is True or direction["strict_descent_direction"] is not True:
            degenerate.append(law)
    if (
        terminal.get("selection_bundle_artifact_sha256")
        != selection.get("artifact_sha256")
        or tuple(terminal.get("degenerate_laws", ())) != tuple(degenerate)
        or not degenerate
        or terminal.get("all_six_provider_created") is not False
        or terminal.get("post_cv_all_six_exact_rescore_performed") is not False
        or terminal.get("held_capability_count") != 0
        or terminal.get("radial_projection_used") is not False
        or terminal.get("classification") != "all_six_tangent_direction_failed"
    ):
        raise ValueError("V20e all-six terminal authority differs")
    return terminal


def _validate_final_trace_evidence(
    value: Mapping[str, object],
    *,
    law: str,
    selection: Mapping[str, object],
    fit: Mapping[str, object],
    expected_initial_gain_sha256s: Mapping[str, str],
    source: Mapping[str, object],
    initial_evidence: Mapping[str, object],
) -> dict[str, object]:
    evidence = _validate_hashed(
        value, domain=_FINAL_TRACE_DOMAIN, label=f"V20e {law} final trace"
    )
    candidate = _mapping(fit.get("final_candidate_receipt"), label="final candidate")
    provider_sha = _sha(
        evidence.get("provider_artifact_sha256"), label="final provider"
    )
    trace = _validate_response_trace(
        _mapping(evidence.get("provider_trace"), label="final provider trace"),
        expected_provider_sha256=provider_sha,
        expected_family_ids=tuple(candidate["fit_support_family_ids"]),
        expected_example_ids=tuple(
            example
            for values in _mapping(
                candidate["fit_support_example_ids_by_family"],
                label="fit support IDs",
            ).values()
            for example in values
        ),
    )
    initial_gains = {
        str(key): _sha(item, label="initial gain")
        for key, item in _mapping(
            evidence.get("initial_response_gain_sha256s"), label="initial gains"
        ).items()
    }
    selected_gains = {
        str(key): _sha(item, label="selected gain")
        for key, item in _mapping(
            evidence.get("selected_response_gain_sha256s"), label="selected gains"
        ).items()
    }
    trace_gains = {
        str(key): _sha(item, label="trace selected gain")
        for key, item in _mapping(
            trace["response_gain_sha256s"], label="trace selected gains"
        ).items()
    }
    changed = initial_gains != selected_gains
    selected_fraction = float(fit["selected_fraction"])
    if selected_fraction == 0.0:
        expected_provider_seed = initial_evidence[
            "provider_transfer_evidence_sha256"
        ]
    else:
        expected_provider_seed = _v14._sha256(
            {
                "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                "source_artifact_sha256": source["artifact_sha256"],
                "selection_bundle_artifact_sha256": selection[
                    "artifact_sha256"
                ],
                "response_law": law,
                "scope": "all_six_provider_only_trace",
                "direction_artifact_sha256": fit[
                    "final_direction_artifact_sha256"
                ],
                "ray_artifact_sha256": fit["final_ray_artifact_sha256"],
                "selected_fraction": selected_fraction,
                "weights": tuple(float(item) for item in fit["selected_weights"]),
                "radial_projection_used": False,
                "post_cv_all_six_exact_rescore_performed": False,
            },
            domain=_PROVIDER_SEED_DOMAIN,
        )
    if (
        evidence.get("selection_bundle_artifact_sha256")
        != selection.get("artifact_sha256")
        or evidence.get("response_law") != law
        or evidence.get("provider_transfer_evidence_sha256")
        != expected_provider_seed
        or evidence.get("selected_fraction") != fit.get("selected_fraction")
        or tuple(evidence.get("selected_weights", ()))
        != tuple(fit.get("selected_weights", ()))
        or candidate.get("selected_provider_artifact_sha256") != provider_sha
        or candidate.get("provider_trace_evidence_sha256")
        != evidence.get("artifact_sha256")
        or initial_gains != dict(expected_initial_gain_sha256s)
        or set(initial_gains) != set(selected_gains)
        or selected_gains != trace_gains
        or evidence.get("provider_trace_changed_from_initial") is not changed
        or candidate.get("provider_trace_changed_from_initial") is not changed
        or candidate.get("provider_trace_finite")
        is not (trace.get("finite") is True)
        or candidate.get("pointwise_trust_passed")
        is not (trace.get("pointwise_trust_passed") is True)
        or candidate.get("rank_is_16")
        is not (trace.get("endpoint_conditional_ranks_are_16") is True)
        or candidate.get("provider_trace_exact") is not True
        or evidence.get("provider_only_trace") is not True
        or evidence.get("teacher_capability_created") is not False
        or evidence.get("exact_fit_objective_rescored") is not False
        or evidence.get("post_cast_h4_or_logits_created") is not False
        or evidence.get("radial_projection_used") is not False
        or evidence.get("raw_response_or_modal_tensors_serialized") is not False
    ):
        raise ValueError(f"V20e {law} final provider trace authority differs")
    ids_by_family = _mapping(
        candidate.get("fit_support_example_ids_by_family"), label="fit support IDs"
    )
    receipts = _mapping(
        evidence.get("provider_trace_receipt_sha256s_by_family"),
        label="final trace receipts",
    )
    gain_trace_hashes = _mapping(
        evidence.get("gain_trace_sha256s_by_family"),
        label="final family gain trace hashes",
    )
    if (
        candidate.get("fit_support_provider_trace_receipt_sha256s_by_family")
        != receipts
        or candidate.get("fit_support_gain_trace_sha256s_by_family")
        != gain_trace_hashes
        or candidate.get("fit_support_gain_min_by_family")
        != evidence.get("gain_min_by_family")
        or candidate.get("fit_support_gain_max_by_family")
        != evidence.get("gain_max_by_family")
        or candidate.get("fit_support_gain_distinct_count_by_family")
        != evidence.get("gain_distinct_count_by_family")
    ):
        raise ValueError("V20e final fit-support summary differs")
    for family, raw_ids in ids_by_family.items():
        ids = tuple(raw_ids)
        family_receipts = _mapping(receipts[family], label="family trace receipts")
        if set(family_receipts) != set(ids):
            raise ValueError("V20e final provider trace example geometry differs")
        for example in ids:
            expected = _v14._sha256(
                {
                    "selection_bundle_artifact_sha256": selection[
                        "artifact_sha256"
                    ],
                    "response_law": law,
                    "family_id": family,
                    "example_id": example,
                    "provider_artifact_sha256": provider_sha,
                    "provider_trace_artifact_sha256": trace["artifact_sha256"],
                    "response_gain_sha256": selected_gains[example],
                    "objective_rescored": False,
                },
                domain=_FINAL_TRACE_DOMAIN,
            )
            if family_receipts[example] != expected:
                raise ValueError("V20e final provider trace receipt differs")
        expected_gain_trace = _v14._sha256(
            {
                "response_law": law,
                "family_id": family,
                "provider_artifact_sha256": provider_sha,
                "response_gain_sha256s": {
                    example: selected_gains[example] for example in ids
                },
            },
            domain=_FINAL_TRACE_DOMAIN,
        )
        if gain_trace_hashes[family] != expected_gain_trace:
            raise ValueError("V20e final family gain trace hash differs")
    return evidence


def _validate_role_evidence(
    value: Mapping[str, object],
    *,
    role: Mapping[str, object],
    provider_receipts: Mapping[str, Mapping[str, object]],
    provider_bundle: Mapping[str, object],
    fit_bundle: Mapping[str, object],
    expected_example_ids: Sequence[str],
) -> dict[str, object]:
    evidence = _validate_hashed(
        value, domain=_ROLE_EVIDENCE_DOMAIN, label="V20e held role evidence"
    )
    outer = _identifier(evidence.get("outer_held_family_id"), label="outer family")
    held = _identifier(evidence.get("scored_inner_family_id"), label="held family")
    raw_by_arm = _mapping(
        evidence.get("arm_execution_evidence"), label="held arm evidence"
    )
    score_by_arm = {
        str(item["arm"]): _mapping(item, label="held arm score")
        for item in _sequence(role.get("arm_scores"), label="held arm scores")
        if isinstance(item, Mapping)
    }
    base = _mapping(raw_by_arm.get("base"), label="base held evidence")
    base_h4 = dict(_mapping(base.get("post_cast_h4_sha256s"), label="base H4"))
    base_logits = dict(
        _mapping(
            base.get("supervised_full_vocab_logits_sha256s"), label="base logits"
        )
    )
    selected_weight_hashes = _mapping(
        fit_bundle.get("selected_weight_sha256s_by_law"),
        label="selected fit weight hashes",
    )
    learned_weight_hashes = {
        "learned_signed_log": selected_weight_hashes["signed_log"],
        "learned_signed_log_sign_flip": selected_weight_hashes["signed_log"],
        "learned_linear": selected_weight_hashes["linear"],
    }
    example_ids = tuple(sorted(base_h4))
    if example_ids != tuple(sorted(expected_example_ids)) or len(example_ids) != _PROMPTS_PER_FAMILY:
        raise ValueError("V20e held role differs from the historical prompt pair")
    capability = _mapping(evidence.get("capability_receipt"), label="held capability")
    _v20b._validate_capability_receipt(
        capability,
        expected_example_ids=example_ids,
        expected_family_count=1,
        expected_held_family_id=outer,
        expected_accesses_per_example=len(_ARMS),
        label="V20e held role capability",
    )
    if (
        set(raw_by_arm) != set(_ARMS)
        or set(score_by_arm) != set(_ARMS)
        or role.get("outer_held_family_id") != outer
        or role.get("held_family_id") != held
        or evidence.get("fit_bundle_artifact_sha256")
        != fit_bundle.get("artifact_sha256")
        or evidence.get("provider_bundle_artifact_sha256")
        != provider_bundle.get("artifact_sha256")
        or evidence.get("learned_signed_log_mirror_exact_negative") is not True
        or evidence.get(
            "both_fits_and_all_seven_providers_frozen_before_capability"
        )
        is not True
        or evidence.get("historically_reused_reed_sundial_diagnostic_only")
        is not True
    ):
        raise ValueError("V20e held role evidence authority differs")
    for arm in _ARMS:
        raw = _mapping(raw_by_arm[arm], label=f"{arm} held execution")
        score = score_by_arm[arm]
        provider_sha = str(provider_receipts[arm]["provider_artifact_sha256"])
        trace = _validate_response_trace(
            _mapping(raw.get("response_trace"), label=f"{arm} held trace"),
            expected_provider_sha256=provider_sha,
            expected_family_ids=(held,),
            expected_example_ids=example_ids,
        )
        objectives = {
            str(key): float(item)
            for key, item in _mapping(
                raw.get("prompt_objectives"), label="held prompt objectives"
            ).items()
        }
        h4 = {
            str(key): _sha(item, label="held H4")
            for key, item in _mapping(
                raw.get("post_cast_h4_sha256s"), label="held H4 hashes"
            ).items()
        }
        logits = {
            str(key): _sha(item, label="held logits")
            for key, item in _mapping(
                raw.get("supervised_full_vocab_logits_sha256s"),
                label="held logit hashes",
            ).items()
        }
        objective = math.fsum(objectives.values()) / len(objectives)
        changed = arm != "base" and (h4 != base_h4 or logits != base_logits)
        expected_execution = _held_execution_sha256(
            fit_bundle_sha256=str(fit_bundle["artifact_sha256"]),
            provider_bundle_sha256=str(provider_bundle["artifact_sha256"]),
            arm=arm,
            provider_sha256=provider_sha,
            outer_family_id=outer,
            scored_family_id=held,
            h4_sha256s=h4,
            logits_sha256s=logits,
            trace_sha256=str(trace["artifact_sha256"]),
            objective=objective,
        )
        nonconstant = float(raw["response_gain_range_on_held_support"]) > 0.0
        expected_gain_range = float(raw["response_gain_max_on_held_support"]) - float(
            raw["response_gain_min_on_held_support"]
        )
        if (
            set(objectives) != set(example_ids)
            or set(h4) != set(example_ids)
            or set(logits) != set(example_ids)
            or raw.get("arm") != arm
            or raw.get("provider_artifact_sha256") != provider_sha
            or raw.get("execution_receipt_sha256") != expected_execution
            or float(raw.get("objective", -1.0)) != objective
            or raw.get("execution_changed_from_base") is not changed
            or raw.get("response_gain_nonconstant_on_held_support") is not nonconstant
            or raw.get("response_gain_range_on_held_support")
            != expected_gain_range
            or score.get("objective") != objective
            or score.get("provider_artifact_sha256") != provider_sha
            or score.get("response_weight_sha256")
            != (
                learned_weight_hashes.get(arm)
                or provider_receipts[arm].get("response_weight_sha256")
                or _provider_tensor_sha256(torch.zeros(3, dtype=torch.float64))
            )
            or score.get("execution_receipt_sha256") != expected_execution
            or score.get("execution_changed_from_base") is not changed
            or score.get("response_nonconstant") is not nonconstant
            or score.get("finite") is not (trace.get("finite") is True)
            or score.get("pointwise_trust_passed")
            is not (trace.get("pointwise_trust_passed") is True)
            or score.get("rank_is_16")
            is not (trace.get("endpoint_conditional_ranks_are_16") is True)
        ):
            raise ValueError(f"V20e {arm} held execution binding differs")
    return evidence


def _validate_report(
    value: Mapping[str, object],
    *,
    output: Path,
    authenticated_source: Mapping[str, object],
    authenticated_v20d: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(value)
    source = _validate_hashed(
        _mapping(selected.get("source"), label="V20e source"),
        domain=_SOURCE_DOMAIN,
        label="V20e source",
    )
    if _v14._canonical_json_bytes(source) != _v14._canonical_json_bytes(
        authenticated_source
    ):
        raise ValueError("V20e authenticated source differs")
    panel = _v20b._core.validate_nested_microstep_panel_receipt(
        _mapping(selected.get("panel_receipt"), label="V20e panel")
    )
    endpoint_fit = _v20b._core.validate_nested_microstep_fit_receipt(
        _mapping(selected.get("shared_fit_receipt"), label="V20e endpoint fit"),
        panel_receipt=panel,
    )
    _v20b._validate_fit_training_evidence(
        selected.get("fit_training_evidence"), fit_receipt=endpoint_fit
    )
    for field in (
        "source_pair_diagnostic",
        "panel_receipt",
        "shared_fit_receipt",
        "fit_training_evidence",
        "coordinate_trace_receipt",
    ):
        if _v14._canonical_json_bytes(selected.get(field)) != _v14._canonical_json_bytes(
            authenticated_v20d.get(field)
        ):
            raise ValueError("V20e report drifted from immutable V20d endpoint evidence")
    authenticated_sequence_sha256s: dict[str, str] = {}
    authenticated_coordinate_families: dict[str, str] = {}
    for raw_row in _sequence(
        _mapping(
            selected.get("coordinate_trace_receipt"),
            label="authenticated coordinate trace",
        ).get("sequence_rows"),
        label="authenticated coordinate sequence rows",
    ):
        row = _mapping(raw_row, label="authenticated coordinate sequence row")
        example = _identifier(
            row.get("example_id"), label="authenticated coordinate example"
        )
        if example in authenticated_sequence_sha256s:
            raise ValueError("V20e authenticated coordinate example is duplicated")
        authenticated_sequence_sha256s[example] = _sha(
            row.get("sequence_artifact_sha256"),
            label="authenticated coordinate sequence",
        )
        authenticated_coordinate_families[example] = _identifier(
            row.get("family_id"), label="authenticated coordinate family"
        )
    raw_initial = _mapping(
        selected.get("initial_vjp_evidence_by_law"), label="V20e initial evidence"
    )
    if set(raw_initial) != set(_LAWS):
        raise ValueError("V20e initial law geometry differs")
    initial = {
        law: _validate_initial_evidence(
            _mapping(raw_initial[law], label=f"{law} initial evidence"),
            law=law,
            source=source,
            endpoint_fit=endpoint_fit,
        )
        for law in _LAWS
    }
    if len({row["artifact_sha256"] for row in initial.values()}) != 2 or len(
        {row["provider_artifact_sha256"] for row in initial.values()}
    ) != 2:
        raise ValueError("V20e law-specific initial VJP banks are not distinct")
    panel_families = tuple(
        sorted(
            _mapping(
                panel.get("family_prompt_sha256s"),
                label="authenticated panel families",
            )
        )
    )
    fit_families = tuple(
        family for family in panel_families if family not in _v20c._FROZEN_EXCLUDED
    )
    authenticated_example_families = {
        str(example): str(family)
        for example, family in _mapping(
            _mapping(
                selected.get("fit_training_evidence"),
                label="authenticated fit training evidence",
            ).get("example_family_ids"),
            label="authenticated fit example families",
        ).items()
    }
    initial_family_maps = {
        law: dict(
            _mapping(
                initial[law]["training_example_family_ids"],
                label=f"{law} initial family map",
            )
        )
        for law in _LAWS
    }
    initial_sequence_maps = {
        law: dict(
            _mapping(
                initial[law]["training_sequence_sha256s"],
                label=f"{law} initial sequence map",
            )
        )
        for law in _LAWS
    }
    if (
        len(panel_families) != _FAMILY_COUNT
        or len(fit_families) != _FIT_FAMILY_COUNT
        or any(
            tuple(initial[law]["training_family_ids"]) != fit_families
            or initial_family_maps[law] != authenticated_example_families
            or any(
                tuple(initial_family_maps[law].values()).count(family)
                != _PROMPTS_PER_FAMILY
                for family in fit_families
            )
            for law in _LAWS
        )
        or initial_family_maps["signed_log"] != initial_family_maps["linear"]
        or initial_sequence_maps["signed_log"] != initial_sequence_maps["linear"]
        or authenticated_coordinate_families != authenticated_example_families
        or any(
            initial_sequence_maps[law] != authenticated_sequence_sha256s
            for law in _LAWS
        )
    ):
        raise ValueError("V20e initial banks differ from authenticated fit rows")

    terminal_raw = selected.get("controlled_terminal_evidence")
    terminal_stage: str | None = None
    manifest_raw = selected.get("cv_provider_manifest_receipt")
    if terminal_raw is not None and manifest_raw is not None:
        manifest, cv, selection = _validate_cv_stage(
            selected, initial=initial, source=source
        )
        _validate_cv_panel_lineage(
            manifest=manifest,
            cv=cv,
            family_ids=panel_families,
            fit_family_ids=fit_families,
            source=source,
            endpoint_fit=endpoint_fit,
            initial=initial,
        )
        _validate_all_six_terminal(
            _mapping(terminal_raw, label="V20e all-six controlled terminal"),
            source=source,
            selection=selection,
            endpoint_fit=endpoint_fit,
            family_ids=panel_families,
            fit_family_ids=fit_families,
            initial=initial,
            cv=cv,
        )
        if (
            selected.get("all_six_direction_receipts_by_law") != {}
            or selected.get("all_six_ray_receipts_by_law") != {}
            or selected.get("final_provider_trace_evidence_by_law") != {}
            or selected.get("tangent_response_fit_receipts_by_law") != {}
            or selected.get("two_fit_bundle_receipt") is not None
            or selected.get("fit_stage_authorized_for_held_scoring") is not False
            or selected.get("provider_receipts") != {}
            or selected.get("provider_bundle_receipt") is not None
            or tuple(selected.get("roles", ()))
            or tuple(selected.get("role_execution_evidence", ()))
            or selected.get("pair_qualification") is not None
            or selected.get("held_scoring_executed") is not False
            or selected.get("classification") != "all_six_tangent_direction_failed"
            or selected.get("passed") is not False
            or selected.get("next_fresh_family_validation_authorized") is not False
            or selected.get("all_120_provider_slots_frozen_before_cv_capability")
            is not True
            or selected.get("work_accounting")
            != _stage_work(
                held_scoring_executed=False,
                terminal_stage="all_six_direction_preflight",
            )
        ):
            raise ValueError("V20e all-six controlled terminal authority differs")
        _validate_top_level_claims(selected, output=output, manifest_created=True)
        _v14._scalar_report(selected)
        return selected
    if terminal_raw is not None and manifest_raw is None:
        _validate_cv_direction_terminal(
            _mapping(terminal_raw, label="V20e controlled terminal"),
            source=source,
            endpoint_fit=endpoint_fit,
            family_ids=panel_families,
            fit_family_ids=fit_families,
            initial=initial,
        )
        classification = "tangent_direction_preflight_failed"
        terminal_stage = "cv_direction_preflight"
        if (
            selected.get("cv_receipts_by_law") != {}
            or selected.get("cv_fold_execution_evidence_by_law") != {}
            or selected.get("two_law_selection_bundle_receipt") is not None
            or selected.get("all_six_direction_receipts_by_law") != {}
            or selected.get("all_six_ray_receipts_by_law") != {}
            or selected.get("final_provider_trace_evidence_by_law") != {}
            or selected.get("tangent_response_fit_receipts_by_law") != {}
            or selected.get("two_fit_bundle_receipt") is not None
            or selected.get("fit_stage_authorized_for_held_scoring") is not False
            or selected.get("provider_receipts") != {}
            or selected.get("provider_bundle_receipt") is not None
            or tuple(selected.get("roles", ()))
            or tuple(selected.get("role_execution_evidence", ()))
            or selected.get("pair_qualification") is not None
            or selected.get("held_scoring_executed") is not False
        ):
            raise ValueError("V20e early terminal contains later-stage authority")
        fit_bundle = None
        fit_authorized = False
        held_executed = False
        passed = False
    else:
        if terminal_raw is not None:
            raise ValueError("V20e nonterminal report carries terminal evidence")
        manifest, cv, selection = _validate_cv_stage(
            selected, initial=initial, source=source
        )
        _validate_cv_panel_lineage(
            manifest=manifest,
            cv=cv,
            family_ids=panel_families,
            fit_family_ids=fit_families,
            source=source,
            endpoint_fit=endpoint_fit,
            initial=initial,
        )
        raw_fits = _mapping(
            selected.get("tangent_response_fit_receipts_by_law"),
            label="V20e final fits",
        )
        raw_directions = _mapping(
            selected.get("all_six_direction_receipts_by_law"),
            label="V20e all-six directions",
        )
        raw_rays = _mapping(
            selected.get("all_six_ray_receipts_by_law"),
            label="V20e all-six rays",
        )
        raw_traces = _mapping(
            selected.get("final_provider_trace_evidence_by_law"),
            label="V20e final traces",
        )
        if any(set(rows) != set(_LAWS) for rows in (raw_fits, raw_directions, raw_rays, raw_traces)):
            raise ValueError("V20e final fit law geometry differs")
        fits: dict[str, dict[str, object]] = {}
        for law in _LAWS:
            fits[law] = _core.validate_tangent_response_fit_receipt(
                _mapping(raw_fits[law], label=f"{law} final fit")
            )
            final_direction = _mapping(
                fits[law]["final_direction_receipt"],
                label=f"{law} all-six direction",
            )
            _validate_all_six_bank_against_cv(
                final_direction,
                cv_receipt=cv[law],
                initial_evidence=initial[law],
            )
            if (
                _v14._canonical_json_bytes(fits[law]["final_direction_receipt"])
                != _v14._canonical_json_bytes(raw_directions[law])
                or _v14._canonical_json_bytes(fits[law]["final_ray_receipt"])
                != _v14._canonical_json_bytes(raw_rays[law])
                or _v14._canonical_json_bytes(fits[law]["cv_receipt"])
                != _v14._canonical_json_bytes(cv[law])
                or final_direction.get("source_artifact_sha256s")
                != _source_sha256s(source)
                or final_direction.get("base_provider_artifact_sha256")
                != endpoint_fit.get("base_provider_artifact_sha256")
                or final_direction.get("proposal_provider_artifact_sha256")
                != endpoint_fit.get("proposal_provider_artifact_sha256")
                or final_direction.get("gradient_evidence_sha256")
                != initial[law].get("artifact_sha256")
            ):
                raise ValueError("V20e final fit receipt lineage differs")
            _validate_final_trace_evidence(
                _mapping(raw_traces[law], label=f"{law} final trace"),
                law=law,
                selection=selection,
                fit=fits[law],
                source=source,
                initial_evidence=initial[law],
                expected_initial_gain_sha256s=(
                    _initial_gain_hashes_from_fold_evidence(
                        tuple(
                            _mapping(item, label=f"{law} CV fold evidence")
                            for item in _sequence(
                                _mapping(
                                    selected.get(
                                        "cv_fold_execution_evidence_by_law"
                                    ),
                                    label="CV fold evidence by law",
                                )[law],
                                label=f"{law} CV fold evidence",
                            )
                        )
                    )
                ),
            )
        fit_bundle = _core.validate_tangent_response_two_fit_bundle_receipt(
            _mapping(
                selected.get("two_fit_bundle_receipt"), label="V20e fit bundle"
            )
        )
        if any(
            _v14._canonical_json_bytes(fits[law])
            != _v14._canonical_json_bytes(fit_bundle["fit_receipts_by_law"][law])
            for law in _LAWS
        ):
            raise ValueError("V20e final fits differ from two-fit bundle")
        fit_authorized = bool(fit_bundle["held_score_authorized"] is True)
        held_executed = selected.get("held_scoring_executed") is True
        terminal_stage = None
        if held_executed:
            if not fit_authorized:
                raise ValueError("V20e held scoring bypassed the fit gate")
            provider_receipts = {
                str(arm): _mapping(row, label=f"{arm} provider receipt")
                for arm, row in _mapping(
                    selected.get("provider_receipts"), label="V20e providers"
                ).items()
            }
            provider_receipts, provider_bundle = _validate_provider_bundle(
                provider_receipts,
                provider_bundle=_mapping(
                    selected.get("provider_bundle_receipt"),
                    label="V20e provider bundle",
                ),
                fit_bundle=fit_bundle,
                initial_evidence_by_law=initial,
                final_trace_evidence_by_law={
                    law: _mapping(raw_traces[law], label=f"{law} final trace")
                    for law in _LAWS
                },
            )
            roles = tuple(
                _core.validate_tangent_response_held_role_receipt(
                    _mapping(item, label="V20e held role"),
                    fit_bundle_receipt=fit_bundle,
                )
                for item in _sequence(selected.get("roles"), label="V20e roles")
            )
            evidence_rows = tuple(
                _validate_role_evidence(
                    _mapping(item, label="V20e role evidence"),
                    role=roles[index],
                    provider_receipts=provider_receipts,
                    provider_bundle=provider_bundle,
                    fit_bundle=fit_bundle,
                    expected_example_ids=tuple(
                        sorted(
                            _mapping(
                                _mapping(
                                    next(
                                        row
                                        for row in _sequence(
                                            authenticated_v20d.get(
                                                "role_execution_evidence"
                                            ),
                                            label="V20d role evidence",
                                        )
                                        if isinstance(row, Mapping)
                                        and row.get("outer_held_family_id")
                                        == roles[index]["outer_held_family_id"]
                                    ).get("arm_execution_evidence"),
                                    label="V20d held arm evidence",
                                )["base"],
                                label="V20d base held evidence",
                            )["post_cast_h4_sha256s"]
                        )
                    ),
                )
                for index, item in enumerate(
                    _sequence(
                        selected.get("role_execution_evidence"),
                        label="V20e role evidence",
                    )
                )
            )
            if len(roles) != 2 or len(evidence_rows) != 2:
                raise ValueError("V20e reciprocal held evidence differs")
            expected_qualification = _pair_qualification(
                fit_bundle=fit_bundle,
                provider_bundle=provider_bundle,
                roles=roles,
                role_evidence=evidence_rows,
            )
            if _v14._canonical_json_bytes(
                selected.get("pair_qualification")
            ) != _v14._canonical_json_bytes(expected_qualification):
                raise ValueError("V20e pair qualification differs")
            passed = expected_qualification["passed"] is True
            classification = (
                "tangent_response_pair_smoke_passed"
                if passed
                else "tangent_response_pair_smoke_failed"
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
                raise ValueError("V20e fit terminal contains held authority")
            passed = False
            classification = "fit_only_tangent_response_failed"
    _validate_top_level_claims(
        selected,
        output=output,
        manifest_created=terminal_stage != "cv_direction_preflight",
    )
    if (
        selected.get("schema") != _SCHEMA
        or selected.get("format_version") != _FORMAT_VERSION
        or selected.get("artifact") != output.as_posix()
        or selected.get("experiment_stage")
        != "A16_development_only_reused_pair_tangent_response"
        or selected.get("scientific_status")
        != "development_only_historically_reused_a16_pair"
        or _v14._canonical_json_bytes(selected.get("fixed_protocol"))
        != _v14._canonical_json_bytes(_FIXED_PROTOCOL)
        or selected.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or selected.get("core_protocol_sha256")
        != _core.TANGENT_RESPONSE_PROTOCOL_SHA256
        or selected.get("fit_stage_authorized_for_held_scoring")
        is not fit_authorized
        or selected.get("held_scoring_executed") is not held_executed
        or selected.get("classification") != classification
        or selected.get("passed") is not passed
        or selected.get("next_fresh_family_validation_authorized") is not passed
        or selected.get("tangent_constrained_natural_response_fit_implemented")
        is not True
        or selected.get("six_fold_leave_one_fit_family_out_selection_implemented")
        is not True
        or selected.get("all_120_provider_slots_frozen_before_cv_capability")
        is not (terminal_stage != "cv_direction_preflight")
        or selected.get("radial_projection_used") is not False
        or selected.get("post_cv_all_six_exact_rescore_performed") is not False
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
        != _stage_work(
            held_scoring_executed=held_executed, terminal_stage=terminal_stage
        )
    ):
        raise ValueError("V20e report authority or claim boundary differs")
    _v14._scalar_report(selected)
    return selected


def _load_existing_report(output: Path) -> dict[str, object]:
    selected = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20e report",
    )
    source, v20d_report, _fragment = _load_authenticated_v20d_source()
    return _validate_report(
        selected,
        output=output,
        authenticated_source=source,
        authenticated_v20d=v20d_report,
    )


def run_gemma3_l3_l4_complete_h4_tangent_response_smoke(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or authenticate the V20e tangent-response diagnostic."""

    destination = _validate_output(output)
    if destination.exists():
        return _load_existing_report(destination)

    # Both immutable prerequisites authenticate before model construction.
    source, v20d_report, fragment = _load_authenticated_v20d_source()
    prerequisite, _v20a_payload, _v20a_folds = (
        _v20b._load_authenticated_v20a_artifact()
    )
    panel_receipt = dict(
        _mapping(v20d_report.get("panel_receipt"), label="V20d panel receipt")
    )
    family_ids = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20d panel family prompts",
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
            raise RuntimeError("V20e live family order differs from V20d")
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
        if _v14._canonical_json_bytes(coordinate_trace) != _v14._canonical_json_bytes(
            v20d_report.get("coordinate_trace_receipt")
        ):
            raise RuntimeError("V20e fit coordinates drifted from immutable V20d")

        banks = {
            law: _collect_initial_bank(
                context,
                workspace,
                teacher_vault,
                law=law,
                source=source,
            )
            for law in _LAWS
        }
        manifest: _ManifestLive | None = None
        cv: dict[str, _CVLive] = {}
        selection: dict[str, object] | None = None
        finals: dict[str, _FinalLive] = {}
        fit_bundle: dict[str, object] | None = None
        provider_receipts: dict[str, dict[str, object]] | None = None
        provider_bundle: dict[str, object] | None = None
        roles: list[dict[str, object]] = []
        role_evidence: list[dict[str, object]] = []
        qualification: dict[str, object] | None = None
        terminal: _ControlledTerminal | None = None
        try:
            manifest = _build_cv_provider_manifest(
                workspace,
                source=source,
                family_ids=family_ids,
                banks=banks,
            )
            cv = _run_cross_validation(
                context,
                teacher_vault,
                banks=banks,
                manifest=manifest,
            )
            selection = _build_selection_bundle(cv, manifest=manifest.receipt)
            all_six_directions, all_six_rays = _build_all_six_direction_preflight(
                workspace,
                source=source,
                family_ids=family_ids,
                banks=banks,
                selection_bundle=selection,
            )
            # Both all-six directions/rays exist before either final provider.
            finals = {
                law: _build_final_fit(
                    workspace,
                    law=law,
                    source=source,
                    family_ids=family_ids,
                    bank=banks[law],
                    cv=cv[law],
                    selection_bundle=selection,
                    direction=all_six_directions[law],
                    ray=all_six_rays[law],
                )
                for law in _LAWS
            }
            fit_bundle = _core.build_tangent_response_two_fit_bundle_receipt(
                signed_log_fit_receipt=finals["signed_log"].fit,
                linear_fit_receipt=finals["linear"].fit,
            )
            if _fit_stage_authorized(fit_bundle, finals):
                providers, provider_receipts, provider_bundle = (
                    _build_frozen_held_providers(
                        workspace,
                        banks=banks,
                        finals=finals,
                        fit_bundle=fit_bundle,
                    )
                )
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
                    roles=roles,
                    role_evidence=role_evidence,
                )
        except _ControlledTerminal as expected_terminal:
            terminal = expected_terminal
            if expected_terminal.stage == "cv_direction_preflight":
                manifest = None
                cv = {}
                selection = None
            finals = {}
            fit_bundle = None
            provider_receipts = None
            provider_bundle = None
            roles = []
            role_evidence = []
            qualification = None

        report = _build_report(
            output=destination,
            source=source,
            v20d_report=v20d_report,
            workspace=workspace,
            coordinate_trace=coordinate_trace,
            banks=banks,
            manifest=manifest,
            cv=cv,
            selection_bundle=selection,
            finals=finals,
            fit_bundle=fit_bundle,
            provider_receipts=provider_receipts,
            provider_bundle=provider_bundle,
            roles=roles,
            role_evidence=role_evidence,
            qualification=qualification,
            terminal=terminal,
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
            label="V20e report",
        )
    except FileExistsError:
        return _load_existing_report(destination)
    return _load_existing_report(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only V20e tangent-constrained response smoke"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tangent_response_smoke(
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
