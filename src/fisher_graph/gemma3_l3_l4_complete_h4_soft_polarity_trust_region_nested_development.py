"""V20g trust-region nested development for a smooth polarity router.

The adaptive experiment is deliberately confined to the historically reused
A16 development panel and binds the failed V20f report as its hypothesis
source.  Each outer fold reconstructs the exact V20a seven-family endpoint,
fits a box-logit-normalized four-scalar continuous router without the held
family, freezes the learned provider and three controls, and only then opens a
capability for the two held prompts.  A successful eight-fold screen triggers
a separate all-family endpoint and router refit; Calibration-B is merely marked
eligible after that final provider receipt is frozen and is never loaded or
tokenized here.

Reports and resumable fold fragments are write-once, mode-0600, scalar/hash
only JSON.  Provider tensors, prompts, token IDs, logits, activations, and raw
gradients are never serialized.
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

from . import complete_h4_fisher_soft_polarity_trust_region_fit as _core
from .complete_h4_fisher_soft_polarity import (
    FISHER_SOFT_POLARITY_ETA_COUNT,
    AutonomousCompleteH4FisherSoftPolarityProvider,
    build_autonomous_complete_h4_fisher_soft_polarity,
    build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control,
    fisher_soft_polarity_box_certificate,
    fisher_soft_polarity_modal_terms,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as _v20b
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from . import gemma3_l3_l4_complete_h4_tangent_response_smoke as _v20e
from . import gemma3_l3_l4_complete_h4_soft_polarity_nested_development as _v20f
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-trust-region-nested-"
    "r16-k256-a-fit16-dev-v20g.json"
)

_V20B_OUTPUT = _v20b.DEFAULT_OUTPUT
_V20B_LOGICAL_SHA256 = (
    "bb45e535074608c5feb877fbceb3342809d872f41ff1776851be656de1b0403b"
)
_V20B_FILE_SHA256 = (
    "42060cc4f4dffbb11ea1203518138a27b46bcd4f483623d4d7874da083a97214"
)
_V20E_OUTPUT = _v20e.DEFAULT_OUTPUT
_V20E_LOGICAL_SHA256 = (
    "bd3b91407c09cd37d961a93c21255a0a10d3b3f2f5db5e890b36be6b7ab2400d"
)
_V20E_FILE_SHA256 = (
    "d4e76a829067fe3945e2a787dc02003b9519e90cdbcb95bc3d1395c0a4d86ebb"
)
_V20F_OUTPUT = _v20f.DEFAULT_OUTPUT
_V20F_LOGICAL_SHA256 = (
    "c718108aac095da5f6f633859843f1fe5d3486c0ca6638ecfc002bf2b533b744"
)
_V20F_FILE_SHA256 = (
    "4c82874a972d8651518481420727af245557d5c4783d8a61f28543d10253201d"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_trust_region_nested.v20g"
_FORMAT_VERSION = 23
_FOLD_SCHEMA = "fisher_graph.complete_h4_soft_polarity_trust_region_outer_fold.v20g"
_FINAL_SCHEMA = "fisher_graph.complete_h4_soft_polarity_trust_region_final_refit.v20g"
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-trust-region-nested-report:v20g\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-trust-region-nested-source:v20g\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-trust-region-nested-fold:v20g\0"
_FINAL_DOMAIN = b"fisher-graph:soft-polarity-trust-region-nested-final:v20g\0"
_FIT_EXECUTION_DOMAIN = b"fisher-graph:soft-polarity-trust-region-fit-execution:v20g\0"
_HELD_EXECUTION_DOMAIN = b"fisher-graph:soft-polarity-trust-region-held-execution:v20g\0"
_PROVIDER_MANIFEST_DOMAIN = b"fisher-graph:soft-polarity-trust-region-provider-manifest:v20g\0"
_ENDPOINT_DOMAIN = b"fisher-graph:soft-polarity-trust-region-all-family-endpoint:v20g\0"

_FAMILY_COUNT = 8
_PROMPTS_PER_FAMILY = 2
_CONDITIONAL_RANK = 16
_ARMS = ("base", "fixed_plus", "fixed_minus", "soft_router")

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20g_eight_fold_soft_polarity_trust_region_nested_development",
    "scientific_status": (
        "adaptive_after_pinned_V20f_historically_reused_A16_development_only"
    ),
    "outer_validation": "leave_one_whole_family_out_eight_folds",
    "outer_endpoint": "exact_authenticated_V20a_seven_family_base_and_first_Adam",
    "router": "four_scalar_tanh_over_1_c1_c2_c1c2_times_fixed_signed_log_c2_envelope",
    "router_initial_eta": (0.0, 0.0, 0.0, 0.0),
    "router_fisher": "family_equal_prompt_gradient_OPG_at_eta_zero",
    "router_selection": (
        "exact_box_logit_normalized_natural_direction_tau_2^-7_through_2_"
        "exact_family_equal_teacher_KL"
    ),
    "schedule_basis": (
        "pinned_V20f_all_eight_folds_selected_zero_after_the_first_"
        "2^-8_candidate_overshot"
    ),
    "trust_region": (
        "max_abs_eta_dot_1_c1_c2_c1c2_at_most_tau_for_the_full_"
        "minus1_plus1_coordinate_box"
    ),
    "alpha_field_semantics": "dimensionless_box_logit_trust_radius_tau",
    "held_arms": _ARMS,
    "fixed_minus_role": "diagnostic_opposite_global_polarity_not_an_authorization_gate",
    "held_barrier": "all_four_providers_frozen_before_held_capability",
    "aggregate_gate": (
        "soft_macro_below_base_and_fixed_plus_envelope_and_at_least_6_of_8_"
        "wins_against_each_with_finite_trusted_nonconstant_execution"
    ),
    "final_refit": (
        "all_eight_family_endpoint_then_all_eight_eta_refit_only_after_CV_pass_"
        "with_nonzero_nonconstant_final_provider_required_for_calibration_B_eligibility"
    ),
    "calibration_b": "eligibility_only_after_final_provider_hash_freeze_never_opened",
    "family_ids_enter_runtime_numeric_rule": False,
    "hard_routing_or_inference_conditionals": False,
    "serving_claim_authorized": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
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


def _identifier(value: object, *, label: str) -> str:
    return _v14._identifier(value, label=label)


def _sha(value: object, *, label: str) -> str:
    return _v19._sha256_identifier(value, label=label)


def _hashed(payload: Mapping[str, object], *, domain: bytes) -> dict[str, object]:
    selected = dict(payload)
    selected["artifact_sha256"] = _v14._sha256(selected, domain=domain)
    _v14._scalar_report(selected)
    return selected


def _validate_hashed(
    value: Mapping[str, object], *, domain: bytes, label: str
) -> dict[str, object]:
    selected = dict(value)
    artifact = _sha(selected.pop("artifact_sha256", None), label=f"{label} artifact")
    if artifact != _v14._sha256(selected, domain=domain):
        raise ValueError(f"{label} artifact hash differs")
    selected["artifact_sha256"] = artifact
    _v14._scalar_report(selected)
    return selected


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or not _v20b._is_under_local_runs(destination):
        raise ValueError("V20g output must be JSON under .local-runs")
    protected = (_V20B_OUTPUT, _V20E_OUTPUT, _V20F_OUTPUT, _v20a.DEFAULT_OUTPUT)
    if any(_v20b._same_destination(destination, item) for item in protected):
        raise ValueError("V20g must preserve immutable prerequisite reports")
    return destination


def _family_suffix(family_id: str) -> str:
    return _v14._sha256(
        {"family_id": _identifier(family_id, label="fold family")},
        domain=_FOLD_DOMAIN,
    )[:20]


def _fold_path(output: Path | str, family_id: str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    return destination.with_name(
        f"{destination.stem}.fold-{_family_suffix(family_id)}.json"
    )


def _final_path(output: Path | str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    return destination.with_name(f"{destination.stem}.final-refit.json")


@dataclass(slots=True)
class _FitLive:
    provider: AutonomousCompleteH4FisherSoftPolarityProvider
    receipt: dict[str, object]
    training_evidence: dict[str, object]


@dataclass(slots=True)
class _EndpointLive:
    training_records: tuple[object, ...]
    base_provider: object
    proposal_provider: object
    receipt: dict[str, object]
    evidence: dict[str, object]


def _load_prerequisites() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """Authenticate the complete V20f failure authority before Gemma."""

    prerequisite, folds, v20f_source = _v20f._load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"),
            label="V20g inherited nested panel receipt",
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20g inherited bridge binding",
    )
    if _v14._file_sha256(_V20F_OUTPUT) != _V20F_FILE_SHA256:
        raise RuntimeError("pinned V20f report file hash drifted")
    v20f = _v20f._load_existing_report(
        _V20F_OUTPUT,
        source=v20f_source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_folds=folds,
    )
    qualification = _mapping(
        v20f.get("oof_qualification"), label="pinned V20f OOF qualification"
    )
    selected_alphas = {
        _identifier(family, label="pinned V20f selected family"): float(alpha)
        for family, alpha in _mapping(
            qualification.get("selected_alphas_by_family"),
            label="pinned V20f selected alphas",
        ).items()
    }
    if (
        v20f.get("report_sha256") != _V20F_LOGICAL_SHA256
        or v20f.get("classification")
        != "soft_polarity_oof_failed_rollback_to_base"
        or v20f.get("passed") is not False
        or v20f.get("rollback_to_base") is not True
        or v20f.get("all_eight_outer_folds_completed") is not True
        or v20f.get("all_eight_family_refit_completed") is not False
        or v20f.get("final_refit") is not None
        or v20f.get("calibration_b_opened") is not False
        or set(selected_alphas) != set(folds)
        or any(alpha != 0.0 for alpha in selected_alphas.values())
    ):
        raise RuntimeError("pinned V20f trust-region schedule authority differs")
    source = _hashed(
        {
            "v20f_report_sha256": _V20F_LOGICAL_SHA256,
            "v20f_file_sha256": _V20F_FILE_SHA256,
            "v20f_source_receipt_sha256": _sha(
                v20f_source.get("artifact_sha256"),
                label="pinned V20f source receipt",
            ),
            "v20f_fold_fragment_sha256s_by_family": dict(
                sorted(
                    _mapping(
                        v20f.get("fold_fragment_sha256s_by_family"),
                        label="pinned V20f fold fragments",
                    ).items()
                )
            ),
            "v20f_selected_alphas_by_family": dict(sorted(selected_alphas.items())),
            "adaptation": (
                "normalize_each_V20f_natural_direction_by_its_exact_box_"
                "corner_logit_max_then_search_tau_2^-7_through_2"
            ),
            "authenticated_before_model_construction": True,
            "historically_reused_A16_only": True,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
        },
        domain=_SOURCE_DOMAIN,
    )
    return prerequisite, folds, source


def _eta_tensor(value: Sequence[float] | Tensor) -> Tensor:
    selected = (
        value.detach().to(device="cpu", dtype=torch.float64)
        if isinstance(value, Tensor)
        else torch.tensor(tuple(float(item) for item in value), dtype=torch.float64)
    )
    if (
        selected.shape != (FISHER_SOFT_POLARITY_ETA_COUNT,)
        or not bool(torch.isfinite(selected).all())
    ):
        raise ValueError("V20g eta must contain four finite float64 scalars")
    return selected.contiguous()


def _local_eta_gradient(
    provider: AutonomousCompleteH4FisherSoftPolarityProvider,
    sequence: object,
    h4_gradient: Tensor,
) -> Tensor:
    """Contract exact suffix dKL/dH4 through only the four router scalars."""

    if (
        not isinstance(h4_gradient, Tensor)
        or h4_gradient.shape != (1, *sequence.base_h4.shape)
        or not h4_gradient.is_floating_point()
        or not bool(torch.isfinite(h4_gradient).all())
    ):
        raise ValueError("V20g suffix H4 gradient geometry differs")
    parent = _training_parent_modal(provider.parent_provider, sequence)
    coordinates = provider.bounded_coordinates(parent)
    eta = provider.eta.detach().clone().requires_grad_(True)
    delta = fisher_soft_polarity_modal_terms(
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
        eta,
        trust_fraction=provider.trust_fraction,
    )[-1]
    decoded = delta @ provider.parent_provider.output_decoder
    suffix = h4_gradient[0].detach().to(device="cpu", dtype=torch.float64)
    gradient = torch.autograd.grad(
        (decoded * suffix).sum(), eta, retain_graph=False, create_graph=False
    )[0]
    if (
        gradient.shape != (FISHER_SOFT_POLARITY_ETA_COUNT,)
        or not bool(torch.isfinite(gradient).all())
    ):
        raise RuntimeError("V20g local eta gradient became invalid")
    return gradient.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _execution_sha256(
    *,
    phase: str,
    outer_family_id: str | None,
    provider_artifact_sha256: str,
    example_id: str,
    family_id: str,
    objective: float,
    h4_sha256: str,
    logits_sha256: str,
    evidence_sha256: str,
    execution_domain: bytes | None = None,
) -> str:
    domain = (
        execution_domain
        if execution_domain is not None
        else (
            _HELD_EXECUTION_DOMAIN
            if phase.startswith("held_")
            else _FIT_EXECUTION_DOMAIN
        )
    )
    if not isinstance(domain, bytes) or not domain:
        raise ValueError("V20g execution domain must be nonempty bytes")
    return _v14._sha256(
        {
            "phase": phase,
            "outer_held_family_id": outer_family_id,
            "provider_artifact_sha256": provider_artifact_sha256,
            "example_id": example_id,
            "family_id": family_id,
            "objective": float(objective),
            "post_cast_h4_sha256": h4_sha256,
            "supervised_full_vocab_logits_sha256": logits_sha256,
            "evidence_sha256": evidence_sha256,
        },
        domain=domain,
    )


def _provider_trace(
    provider: object,
    records: Sequence[object],
    *,
    arm: str,
    artifact_domain: bytes = _PROVIDER_MANIFEST_DOMAIN,
) -> dict[str, object]:
    if not isinstance(artifact_domain, bytes) or not artifact_domain:
        raise ValueError("V20g trace artifact domain must be nonempty bytes")
    ordered = _v20b._ordered_records(records)
    sequences = tuple(record.sequence for record in ordered)
    runtime = _v19._held_runtime_diagnostics(provider, sequences)
    gain_hashes: dict[str, str] = {}
    values: list[Tensor] = []
    for sequence in sequences:
        parent = _training_parent_modal(provider.parent_provider, sequence)
        coordinates = provider.bounded_coordinates(parent)
        if arm == "base":
            gain = torch.zeros(
                coordinates.shape[:-1], dtype=torch.float64, device=coordinates.device
            )
        else:
            gain = provider.response_gain(coordinates)
        support = sequence.support_mask.to(gain.device)
        selected = gain[support].detach().to(device="cpu", dtype=torch.float64)
        if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
            raise RuntimeError("V20g provider gain trace is empty or nonfinite")
        gain_hashes[sequence.example_id] = _v14._tensor_sha256(selected)
        values.append(selected.reshape(-1))
    joined = torch.cat(values)
    payload = {
        "arm": arm,
        "provider_artifact_sha256": provider.artifact_sha256,
        "scored_family_ids": tuple(sorted({record.sequence.family_id for record in ordered})),
        "response_gain_sha256s": dict(sorted(gain_hashes.items())),
        "response_gain_min": float(joined.min()),
        "response_gain_max": float(joined.max()),
        "response_gain_distinct_count": int(torch.unique(joined).numel()),
        "response_gain_nonconstant": bool(float(joined.max()) > float(joined.min())),
        "runtime_receipt_sha256": runtime["receipt_sha256"],
        "finite": True,
        "pointwise_trust_passed": runtime["pointwise_trust_passed"] is True,
        "max_bounded_direction_to_parent_norm_ratio": runtime[
            "max_bounded_direction_to_parent_norm_ratio"
        ],
        "max_emitted_delta_to_parent_norm_ratio": runtime[
            "max_emitted_delta_to_parent_norm_ratio"
        ],
        "endpoint_conditional_ranks_are_16": int(provider.conditional_rank)
        == _CONDITIONAL_RANK,
        "raw_response_or_modal_tensors_serialized": False,
    }
    return _hashed(payload, domain=artifact_domain)


def _provider_receipt(provider: object, *, arm: str) -> dict[str, object]:
    metadata = _mapping(provider.metadata(), label=f"{arm} provider metadata")
    payload = {
        "arm": arm,
        "provider_artifact_sha256": _sha(
            provider.artifact_sha256, label=f"{arm} provider artifact"
        ),
        "provider_metadata_sha256": _v14._sha256(
            metadata, domain=_PROVIDER_MANIFEST_DOMAIN
        ),
        "base_provider_artifact_sha256": (
            provider.artifact_sha256
            if arm == "base"
            else metadata["base_provider_artifact_sha256"]
        ),
        "proposal_provider_artifact_sha256": (
            None if arm == "base" else metadata["proposal_provider_artifact_sha256"]
        ),
        "eta_sha256": metadata.get("eta_sha256"),
        "transfer_evidence_sha256": metadata.get("transfer_evidence_sha256"),
        "rank": int(provider.rank),
        "conditional_rank": int(provider.conditional_rank),
        "prepared_float_scalar_count": int(provider.prepared_float_scalar_count),
        "logical_macs_per_token_upper_bound": int(
            provider.logical_macs_per_token_upper_bound
        ),
        "analysis_only": arm != "base",
    }
    return _hashed(payload, domain=_PROVIDER_MANIFEST_DOMAIN)


def _freeze_held_providers(
    endpoint: _EndpointLive,
    fit: _FitLive,
    *,
    outer_family_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    evidence = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "outer_held_family_id": outer_family_id,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "fit_receipt_sha256": fit.receipt["artifact_sha256"],
            "scope": "four_arm_held_manifest_before_capability",
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )
    providers: dict[str, object] = {
        "base": endpoint.base_provider,
        "fixed_plus": build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
            endpoint.base_provider,
            endpoint.proposal_provider,
            polarity=1,
            transfer_protocol_sha256=_core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            transfer_evidence_sha256=evidence,
        ),
        "fixed_minus": build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
            endpoint.base_provider,
            endpoint.proposal_provider,
            polarity=-1,
            transfer_protocol_sha256=_core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            transfer_evidence_sha256=evidence,
        ),
        "soft_router": fit.provider,
    }
    if set(providers) != set(_ARMS):
        raise RuntimeError("V20g held provider arm geometry differs")
    receipts = {arm: _provider_receipt(providers[arm], arm=arm) for arm in _ARMS}
    provider_hashes = {
        arm: str(receipts[arm]["provider_artifact_sha256"]) for arm in _ARMS
    }
    if len(set(provider_hashes.values())) != len(_ARMS):
        raise RuntimeError("V20g held provider artifacts are not distinct")
    manifest = _hashed(
        {
            "outer_held_family_id": outer_family_id,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "fit_receipt_sha256": fit.receipt["artifact_sha256"],
            "arm_order": _ARMS,
            "provider_artifact_sha256s": provider_hashes,
            "provider_receipts": receipts,
            "all_four_providers_frozen_before_held_capability": True,
            "held_capability_count_at_freeze": 0,
            "held_objectives_or_teacher_rows_used": False,
            "provider_sidecar_or_raw_tensor_serialized": False,
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )
    return providers, manifest


def _outer_endpoint(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    family_ids: Sequence[str],
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
    authenticated_v20a_fold: Mapping[str, object],
) -> _EndpointLive:
    outer = _identifier(outer_family_id, label="outer family")
    workspace = _v20b._fit_endpoint_from_scratch(
        context,
        records,
        teacher_vault,
        excluded_family_ids=(outer,),
        panel_receipt=panel_receipt,
        outer_fit=True,
    )
    binding = _mapping(
        authenticated_v20a_fold.get("endpoint_binding"),
        label="authenticated V20a endpoint binding",
    )
    expected_training = tuple(family for family in sorted(family_ids) if family != outer)
    if (
        authenticated_v20a_fold.get("held_family_id") != outer
        or binding.get("held_family_id") != outer
        or workspace.excluded_family_ids != (outer,)
        or tuple(sorted({row.sequence.family_id for row in workspace.training_records}))
        != expected_training
        or workspace.base_provider.artifact_sha256
        != binding.get("base_provider_artifact_sha256")
        or workspace.proposal_provider.artifact_sha256
        != binding.get("proposal_provider_artifact_sha256")
    ):
        raise RuntimeError("V20g outer endpoint differs from authenticated V20a")
    evidence = _hashed(
        {
            "kind": "seven_family_outer_endpoint_fit_evidence",
            "outer_held_family_id": outer,
            "endpoint_fit_receipt": dict(workspace.fit_receipt),
            "endpoint_fit_training_evidence": dict(
                workspace.fit_training_evidence
            ),
            "raw_endpoint_gradients_logits_h4_or_tensors_serialized": False,
        },
        domain=_ENDPOINT_DOMAIN,
    )
    receipt = _hashed(
        {
            "kind": "seven_family_outer_endpoint",
            "outer_held_family_id": outer,
            "training_family_ids": expected_training,
            "training_example_ids": tuple(
                row.sequence.example_id for row in workspace.training_records
            ),
            "panel_receipt_sha256": panel_receipt["artifact_sha256"],
            "v20a_endpoint_binding_sha256": _v14._sha256(
                binding, domain=_ENDPOINT_DOMAIN
            ),
            "base_provider_artifact_sha256": workspace.base_provider.artifact_sha256,
            "proposal_provider_artifact_sha256": workspace.proposal_provider.artifact_sha256,
            "endpoint_fit_receipt_sha256": workspace.fit_receipt["artifact_sha256"],
            "endpoint_fit_training_evidence_sha256": evidence[
                "artifact_sha256"
            ],
            "held_family_absent_from_endpoint_fit": True,
            "authenticated_before_router_fit": True,
            "raw_endpoint_tensors_serialized": False,
        },
        domain=_ENDPOINT_DOMAIN,
    )
    return _EndpointLive(
        training_records=workspace.training_records,
        base_provider=workspace.base_provider,
        proposal_provider=workspace.proposal_provider,
        receipt=receipt,
        evidence=evidence,
    )


def _all_family_endpoint(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    family_ids: Sequence[str],
    panel_receipt: Mapping[str, object],
) -> _EndpointLive:
    """Fit the exact checkpoint-zero/first-Adam endpoint on all A16 rows."""

    training = _v20b._ordered_records(records)
    families = tuple(sorted(family_ids))
    if (
        len(training) != _FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or tuple(sorted({row.sequence.family_id for row in training})) != families
    ):
        raise RuntimeError("V20g all-family endpoint training geometry differs")
    sequences = tuple(row.sequence for row in training)
    parent = _v19._fit_parent(
        sequences, bridge_binding_sha256=context.bridge.bridge_binding_sha256
    )
    start = _v19._fit_v18_start(
        sequences, parent=parent, coordinate_objective="reverse_vjp_fisher"
    )
    state0 = _v19._initial_joint_state(start)
    base = _v19._provisional_provider(
        start,
        state0,
        held_family_id=None,
        coordinate_objective="reverse_vjp_fisher",
        checkpoint=0,
    )
    capability = teacher_vault.capability(
        tuple(row.sequence.example_id for row in training), held_family_id=None
    )
    endpoint_gradients: list[object] = []
    objectives: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
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
            context.adapter, model_inputs, objective=objective, h4_head=base
        )
        score, h4_sha, logits_sha = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=base.artifact_sha256,
        )
        if len(captured) != 1 or captured[0] != score:
            raise RuntimeError("V20g all-family endpoint objective capture drifted")
        endpoint_gradients.append(
            _v19._local_ste_parameter_gradients(
                base, state0, record.sequence, h4_gradient
            )
        )
        example = record.sequence.example_id
        objectives[example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        del model_inputs, teacher, execution, h4_gradient
    zero = _v19._zero_state(state0)
    state1, _moments = _v19._adam_step(
        state0,
        _v19._mean_gradient(endpoint_gradients),
        _v19._AdamMoments(first=zero, second=zero, step=0),
    )
    proposal = _v19._provisional_provider(
        start,
        state1,
        held_family_id=None,
        coordinate_objective="reverse_vjp_fisher",
        checkpoint=1,
    )
    for provider in (base, proposal):
        _v19._validate_joint_provider(
            provider,
            start_provider=start,
            pedal_mode="conditional",
            expected_family_count=_FAMILY_COUNT,
        )
    flags = tuple(_v20a._runtime_flags(provider, training) for provider in (base, proposal))
    if any(
        row.get("finite") is not True
        or row.get("pointwise_trust_passed") is not True
        or row.get("rank_is_16") is not True
        for row in flags
    ):
        raise RuntimeError("V20g all-family endpoint health failed")
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(row.sequence.example_id for row in training),
        expected_family_count=_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label="V20g all-family endpoint capability",
    )
    evidence = _hashed(
        {
            "kind": "all_eight_family_endpoint_fit_evidence",
            "panel_receipt_sha256": panel_receipt["artifact_sha256"],
            "training_family_ids": families,
            "training_example_family_ids": {
                row.sequence.example_id: row.sequence.family_id for row in training
            },
            "base_objectives_by_example": dict(sorted(objectives.items())),
            "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
            "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
            "capability_receipt": capability_receipt,
            "base_runtime_flags": flags[0],
            "proposal_runtime_flags": flags[1],
            "full_suffix_vjp_count": len(training),
            "local_endpoint_autograd_contraction_count": len(training),
            "raw_gradients_logits_h4_or_tensors_serialized": False,
        },
        domain=_ENDPOINT_DOMAIN,
    )
    receipt = _hashed(
        {
            "kind": "all_eight_family_endpoint",
            "panel_receipt_sha256": panel_receipt["artifact_sha256"],
            "training_family_ids": families,
            "training_example_ids": tuple(row.sequence.example_id for row in training),
            "parent_provider_artifact_sha256": parent.artifact_sha256,
            "start_provider_artifact_sha256": start.artifact_sha256,
            "base_provider_artifact_sha256": base.artifact_sha256,
            "proposal_provider_artifact_sha256": proposal.artifact_sha256,
            "fit_evidence_sha256": evidence["artifact_sha256"],
            "finite_trusted_rank16": True,
            "raw_endpoint_tensors_serialized": False,
        },
        domain=_ENDPOINT_DOMAIN,
    )
    return _EndpointLive(
        training_records=training,
        base_provider=base,
        proposal_provider=proposal,
        receipt=receipt,
        evidence=evidence,
    )


def _score_exact_provider(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    phase: str,
    outer_family_id: str | None,
    evidence_sha256: str,
    execution_domain: bytes | None = None,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    """Execute one frozen provider exactly once on every selected prompt."""

    ordered = _v20b._ordered_records(records)
    objectives: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
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
        objectives[example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        execution_hashes[example] = _execution_sha256(
            phase=phase,
            outer_family_id=outer_family_id,
            provider_artifact_sha256=provider.artifact_sha256,
            example_id=example,
            family_id=record.sequence.family_id,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
            evidence_sha256=evidence_sha256,
            execution_domain=execution_domain,
        )
        del model_inputs, teacher, execution
    return objectives, h4_hashes, logits_hashes, execution_hashes


def _values_by_family(
    values: Mapping[str, Any], records: Sequence[object]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    expected: set[str] = set()
    for record in _v20b._ordered_records(records):
        example = record.sequence.example_id
        expected.add(example)
        if example not in values:
            raise ValueError("V20g family map omitted an example")
        result.setdefault(record.sequence.family_id, {})[example] = values[example]
    if set(values) != expected:
        raise ValueError("V20g family map contains an unknown example")
    return {family: dict(sorted(rows.items())) for family, rows in sorted(result.items())}


def _direction_vector(receipt: Mapping[str, object]) -> tuple[float, ...]:
    for key in ("natural_direction", "direction", "direction_values", "eta_direction"):
        value = receipt.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            selected = tuple(float(item) for item in value)
            if len(selected) == FISHER_SOFT_POLARITY_ETA_COUNT and all(
                math.isfinite(item) for item in selected
            ):
                return selected
    raise ValueError("V20g direction receipt omitted the finite four-vector")


def _fit_seed_sha256(
    endpoint: _EndpointLive,
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str | None,
) -> str:
    """Rebuild the V20g eta-zero fit seed from authenticated fold authority."""

    all_families = tuple(
        sorted(_identifier(item, label="fit seed family") for item in all_family_ids)
    )
    outer = (
        None
        if outer_family_id is None
        else _identifier(outer_family_id, label="fit seed outer family")
    )
    if (
        len(all_families) != _FAMILY_COUNT
        or len(set(all_families)) != _FAMILY_COUNT
        or (outer is not None and outer not in all_families)
    ):
        raise ValueError("V20g fit seed family geometry differs")
    endpoint_receipt = _mapping(
        endpoint.receipt, label="fit seed endpoint receipt"
    )
    endpoint_evidence = _mapping(
        endpoint.evidence, label="fit seed endpoint evidence"
    )
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt.get("artifact_sha256"),
                label="fit seed endpoint receipt artifact",
            ),
            "endpoint_evidence_sha256": _sha(
                endpoint_evidence.get("artifact_sha256"),
                label="fit seed endpoint evidence artifact",
            ),
            "all_development_family_ids": all_families,
            "outer_held_family_id": outer,
            "eta": (0.0,) * FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )


def _build_radius_provider_ladder(
    endpoint: _EndpointLive,
    *,
    direction_receipt: Mapping[str, object],
    fit_seed_sha256: str,
    outer_family_id: str | None,
    alpha_ladder: Sequence[float],
    eta_zero: AutonomousCompleteH4FisherSoftPolarityProvider | None = None,
) -> tuple[
    dict[float, AutonomousCompleteH4FisherSoftPolarityProvider],
    dict[float, str],
]:
    """Deterministically materialize an inherited V20g trust-radius ladder.

    Every requested radius must belong to V20g's authenticated 0..2 ladder.
    Later rungs must use their own transfer protocol and evidence domain for
    extension radii rather than silently extending V20g's fixed protocol.
    """

    _core.validate_soft_polarity_direction_receipt(direction_receipt)
    direction = _direction_vector(direction_receipt)
    direction_sha256 = _sha(
        direction_receipt.get("artifact_sha256"),
        label="radius direction artifact",
    )
    fit_seed = _sha(fit_seed_sha256, label="radius fit seed")
    outer = (
        None
        if outer_family_id is None
        else _identifier(outer_family_id, label="radius outer family")
    )
    if direction_receipt.get("held_family_id") != outer:
        raise ValueError("V20g radius direction held family differs")
    if not isinstance(alpha_ladder, Sequence) or isinstance(
        alpha_ladder, (str, bytes)
    ):
        raise TypeError("V20g alpha ladder must be a sequence")
    alphas: list[float] = []
    for value in alpha_ladder:
        if isinstance(value, bool):
            raise ValueError("V20g alpha ladder must contain finite radii")
        alpha = float(value)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("V20g alpha ladder must contain finite radii")
        alphas.append(0.0 if alpha == 0.0 else alpha)
    if not alphas or any(
        right <= left for left, right in zip(alphas, alphas[1:])
    ):
        raise ValueError("V20g alpha ladder must be strictly increasing")
    selected_alphas = tuple(alphas)
    if not set(selected_alphas).issubset(
        {float(item) for item in _core.SOFT_POLARITY_FIT_ALPHAS}
    ):
        raise ValueError("V20g alpha ladder exceeds its authenticated protocol")
    if eta_zero is not None and 0.0 not in selected_alphas:
        raise ValueError("V20g eta-zero provider is outside the alpha ladder")

    zero_provider = eta_zero
    if 0.0 in selected_alphas and zero_provider is None:
        zero_provider = build_autonomous_complete_h4_fisher_soft_polarity(
            endpoint.base_provider,
            endpoint.proposal_provider,
            eta=torch.zeros(FISHER_SOFT_POLARITY_ETA_COUNT, dtype=torch.float64),
            transfer_protocol_sha256=_core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            transfer_evidence_sha256=fit_seed,
        )
    if zero_provider is not None:
        zero_eta = _eta_tensor(zero_provider.eta)
        if (
            bool(torch.count_nonzero(zero_eta))
            or zero_provider.base_provider.artifact_sha256
            != endpoint.base_provider.artifact_sha256
            or zero_provider.proposal_provider.artifact_sha256
            != endpoint.proposal_provider.artifact_sha256
            or zero_provider.transfer_protocol_sha256
            != _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256
            or zero_provider.transfer_evidence_sha256 != fit_seed
        ):
            raise ValueError("V20g eta-zero provider authority differs")

    providers: dict[float, AutonomousCompleteH4FisherSoftPolarityProvider] = {}
    provider_seeds: dict[float, str] = {}
    for alpha in selected_alphas:
        eta = tuple(alpha * item for item in direction)
        seed = _v14._sha256(
            {
                "fit_seed_sha256": fit_seed,
                "direction_artifact_sha256": direction_sha256,
                "alpha": alpha,
                "eta": eta,
                "outer_held_family_id": outer,
                "held_rows_used": False,
            },
            domain=_FIT_EXECUTION_DOMAIN,
        )
        provider_seeds[alpha] = seed
        if alpha == 0.0:
            if zero_provider is None:  # pragma: no cover - guarded above
                raise RuntimeError("V20g eta-zero provider was not constructed")
            providers[alpha] = zero_provider
        else:
            providers[alpha] = build_autonomous_complete_h4_fisher_soft_polarity(
                endpoint.base_provider,
                endpoint.proposal_provider,
                eta=_eta_tensor(eta),
                transfer_protocol_sha256=_core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
                transfer_evidence_sha256=seed,
            )
    return providers, provider_seeds


def _fit_soft_router(
    context: object,
    teacher_vault: object,
    *,
    endpoint: _EndpointLive,
    all_family_ids: Sequence[str],
    outer_family_id: str | None,
) -> _FitLive:
    """Fit the box-normalized natural-gradient ray and exact radius ladder."""

    training = _v20b._ordered_records(endpoint.training_records)
    training_families = tuple(sorted({row.sequence.family_id for row in training}))
    all_families = tuple(sorted(_identifier(item, label="fit family") for item in all_family_ids))
    outer = (
        None
        if outer_family_id is None
        else _identifier(outer_family_id, label="router outer family")
    )
    expected = tuple(family for family in all_families if family != outer)
    if (
        len(all_families) != _FAMILY_COUNT
        or training_families != expected
        or len(training) != len(expected) * _PROMPTS_PER_FAMILY
    ):
        raise RuntimeError("V20g router training complement differs")

    fit_seed = _fit_seed_sha256(
        endpoint,
        all_family_ids=all_families,
        outer_family_id=outer,
    )
    eta_zero = build_autonomous_complete_h4_fisher_soft_polarity(
        endpoint.base_provider,
        endpoint.proposal_provider,
        eta=torch.zeros(FISHER_SOFT_POLARITY_ETA_COUNT, dtype=torch.float64),
        transfer_protocol_sha256=_core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        transfer_evidence_sha256=fit_seed,
    )
    capability = teacher_vault.capability(
        tuple(row.sequence.example_id for row in training), held_family_id=outer
    )
    gradients: dict[str, tuple[float, ...]] = {}
    gradient_hashes: dict[str, str] = {}
    objectives: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
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
            context.adapter, model_inputs, objective=objective, h4_head=eta_zero
        )
        score, h4_sha, logits_sha = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=eta_zero.artifact_sha256,
        )
        if len(captured) != 1 or captured[0] != score:
            raise RuntimeError("V20g eta-zero objective capture drifted")
        gradient = _local_eta_gradient(eta_zero, record.sequence, h4_gradient)
        example = record.sequence.example_id
        gradients[example] = tuple(float(item) for item in gradient.tolist())
        gradient_hashes[example] = _v14._tensor_sha256(gradient)
        objectives[example] = score
        h4_hashes[example] = h4_sha
        logits_hashes[example] = logits_sha
        execution_hashes[example] = _execution_sha256(
            phase="fit_eta_zero_vjp",
            outer_family_id=outer,
            provider_artifact_sha256=eta_zero.artifact_sha256,
            example_id=example,
            family_id=record.sequence.family_id,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
            evidence_sha256=fit_seed,
        )
        del model_inputs, teacher, execution, h4_gradient, gradient

    gradient_rows = _values_by_family(gradients, training)
    objective_rows_zero = _values_by_family(objectives, training)
    gradient_evidence = _hashed(
        {
            "outer_held_family_id": outer,
            "training_family_ids": training_families,
            "training_example_family_ids": {
                row.sequence.example_id: row.sequence.family_id for row in training
            },
            "eta_zero_provider_artifact_sha256": eta_zero.artifact_sha256,
            "eta_zero_objectives_by_family": objective_rows_zero,
            "eta_gradient_sha256s": dict(sorted(gradient_hashes.items())),
            "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
            "supervised_full_vocab_logits_sha256s": dict(
                sorted(logits_hashes.items())
            ),
            "eta_zero_execution_sha256s": dict(sorted(execution_hashes.items())),
            "full_suffix_vjp_count": len(training),
            "local_eta_autograd_contraction_count": len(training),
            "family_equal_prompt_gradient_OPG": True,
            "held_family_absent": outer is None or outer not in training_families,
            "raw_gradients_logits_h4_teacher_rows_or_tensors_serialized": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    direction = _core.build_soft_polarity_direction_receipt(
        source_artifact_sha256s={
            "endpoint_receipt": str(endpoint.receipt["artifact_sha256"]),
            "endpoint_evidence": str(endpoint.evidence["artifact_sha256"]),
            "gradient_evidence": str(gradient_evidence["artifact_sha256"]),
        },
        all_development_family_ids=all_families,
        held_family_id=outer,
        gradient_rows_by_family=gradient_rows,
        gradient_evidence_sha256=str(gradient_evidence["artifact_sha256"]),
    )
    alphas = tuple(float(value) for value in _core.SOFT_POLARITY_FIT_ALPHAS)
    if not alphas or alphas[0] != 0.0 or any(
        not math.isfinite(value) or value < 0.0 for value in alphas
    ):
        raise RuntimeError("V20g fixed trust-radius ladder differs")

    # Every training candidate is built and hash-bound before the first
    # positive-radius exact score.  The held family is still absent here.
    providers, provider_seeds = _build_radius_provider_ladder(
        endpoint,
        direction_receipt=direction,
        fit_seed_sha256=fit_seed,
        outer_family_id=outer,
        alpha_ladder=alphas,
        eta_zero=eta_zero,
    )
    provider_manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "alpha_order": alphas,
            "provider_artifact_sha256s": {
                str(alpha): providers[alpha].artifact_sha256 for alpha in alphas
            },
            "all_alpha_candidates_frozen_before_positive_scoring": True,
            "held_capability_count": 0,
            "raw_provider_tensors_serialized": False,
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )

    candidate_receipts: list[dict[str, object]] = []
    candidate_evidence: dict[str, dict[str, object]] = {}
    candidate_traces = {
        alpha: _provider_trace(
            providers[alpha], training, arm=f"fit_alpha_{alpha.hex()}"
        )
        for alpha in alphas
    }
    for alpha in alphas:
        provider = providers[alpha]
        if alpha == 0.0:
            candidate_values = dict(objectives)
            candidate_h4 = dict(h4_hashes)
            candidate_logits = dict(logits_hashes)
            candidate_execution = dict(execution_hashes)
        else:
            (
                candidate_values,
                candidate_h4,
                candidate_logits,
                candidate_execution,
            ) = _score_exact_provider(
                context,
                training,
                capability,
                provider=provider,
                phase="fit_positive_alpha",
                outer_family_id=outer,
                evidence_sha256=provider_seeds[alpha],
            )
        macro, family_means = _v19._family_equal_mean(candidate_values, training)
        by_family = _values_by_family(candidate_values, training)
        execution_receipt = _v14._sha256(
            {
                "provider_manifest_sha256": provider_manifest["artifact_sha256"],
                "provider_artifact_sha256": provider.artifact_sha256,
                "alpha": alpha,
                "execution_sha256s": dict(sorted(candidate_execution.items())),
                "family_equal_objective": macro,
            },
            domain=_FIT_EXECUTION_DOMAIN,
        )
        receipt = _core.build_soft_polarity_candidate_receipt(
            direction_receipt=direction,
            alpha=alpha,
            exact_train_objectives_by_family=by_family,
            execution_receipt_sha256=execution_receipt,
            exact_execution=True,
        )
        candidate_receipts.append(receipt)
        candidate_evidence[str(alpha)] = _hashed(
            {
                "outer_held_family_id": outer,
                "alpha": alpha,
                "provider_artifact_sha256": provider.artifact_sha256,
                "provider_manifest_sha256": provider_manifest["artifact_sha256"],
                "candidate_receipt_sha256": receipt["artifact_sha256"],
                "response_trace": candidate_traces[alpha],
                "family_equal_objective": macro,
                "family_mean_objectives": family_means,
                "objectives_by_family": by_family,
                "post_cast_h4_sha256s": dict(sorted(candidate_h4.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(candidate_logits.items())
                ),
                "execution_sha256s": dict(sorted(candidate_execution.items())),
                "execution_receipt_sha256": execution_receipt,
                "eta_zero_execution_reused": alpha == 0.0,
                "exact_execution": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_FIT_EXECUTION_DOMAIN,
        )
    selected_candidate = min(
        candidate_receipts,
        key=lambda item: (
            float(item["family_equal_train_objective"]),
            float(item["alpha"]),
            str(item["artifact_sha256"]),
        ),
    )
    selected_alpha = float(selected_candidate["alpha"])
    selected_provider = providers[selected_alpha]
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(row.sequence.example_id for row in training),
        expected_family_count=len(training_families),
        expected_held_family_id=outer,
        expected_accesses_per_example=len(alphas),
        label="V20g router-fit capability",
    )
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "training_family_ids": training_families,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "gradient_evidence_sha256": gradient_evidence["artifact_sha256"],
            "direction_receipt": direction,
            "candidate_receipts": tuple(candidate_receipts),
            "provider_manifest_sha256": provider_manifest["artifact_sha256"],
            "selected_alpha": selected_alpha,
            "selected_eta": tuple(float(item) for item in selected_provider.eta.tolist()),
            "selected_candidate_artifact_sha256": candidate_receipts[
                alphas.index(selected_alpha)
            ]["artifact_sha256"],
            "selected_provider_artifact_sha256": selected_provider.artifact_sha256,
            "selection_rule": (
                "exact_family_equal_teacher_KL_then_smaller_alpha_then_eta_hash_"
                "with_float64_floor_rollback"
            ),
            "selection_frozen_before_held_scores": True,
            "held_family_used_for_fit_or_selection": False,
            "box_certificate": fisher_soft_polarity_box_certificate(
                selected_provider.eta
            ),
            "raw_eta_provider_or_gradient_tensors_serialized": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    evidence = _hashed(
        {
            "fit_receipt_sha256": receipt["artifact_sha256"],
            "gradient_evidence": gradient_evidence,
            "candidate_provider_manifest": provider_manifest,
            "candidate_evidence": candidate_evidence,
            "capability_receipt": capability_receipt,
            "exact_candidate_execution_count": len(training) * len(alphas),
            "full_suffix_vjp_count": len(training),
            "held_family_rows_used": False,
            "raw_prompts_token_ids_logits_h4_gradients_or_teacher_rows_serialized": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    return _FitLive(
        provider=selected_provider,
        receipt=receipt,
        training_evidence=evidence,
    )


def _score_held_fold(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    endpoint: _EndpointLive,
    fit: _FitLive,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Freeze every arm, then perform the sole held-family score schedule."""

    outer = _identifier(outer_family_id, label="held outer family")
    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20g held fold prompt geometry differs")
    providers, manifest = _freeze_held_providers(
        endpoint, fit, outer_family_id=outer
    )
    if (
        manifest.get("all_four_providers_frozen_before_held_capability") is not True
        or manifest.get("held_capability_count_at_freeze") != 0
    ):
        raise PermissionError("V20g held provider freeze barrier is not satisfied")

    # Runtime traces consume no teacher rows and are frozen before capability.
    traces = {
        arm: _provider_trace(providers[arm], held, arm=arm) for arm in _ARMS
    }
    trace_bundle_sha = _v14._sha256(
        {arm: traces[arm]["artifact_sha256"] for arm in _ARMS},
        domain=_HELD_EXECUTION_DOMAIN,
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in held), held_family_id=None
    )
    objectives_by_arm: dict[str, dict[str, float]] = {}
    evidence_by_arm: dict[str, dict[str, object]] = {}
    execution_receipts: dict[str, str] = {}
    for arm in _ARMS:
        seed = _v14._sha256(
            {
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "arm": arm,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "outer_held_family_id": outer,
            },
            domain=_HELD_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=providers[arm],
                phase=f"held_{arm}",
                outer_family_id=outer,
                evidence_sha256=seed,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {outer}:
            raise RuntimeError("V20g held objective family geometry differs")
        execution_receipt = _v14._sha256(
            {
                "seed_sha256": seed,
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "objective": macro,
            },
            domain=_HELD_EXECUTION_DOMAIN,
        )
        execution_receipts[arm] = execution_receipt
        objectives_by_arm[arm] = objectives
        evidence_by_arm[arm] = _hashed(
            {
                "arm": arm,
                "outer_held_family_id": outer,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "response_trace": traces[arm],
                "objective": macro,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(logits_hashes.items())
                ),
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "execution_receipt_sha256": execution_receipt,
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_HELD_EXECUTION_DOMAIN,
        )
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(record.sequence.example_id for record in held),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20g held capability",
    )
    selected_trace = traces["soft_router"]
    all_healthy = all(
        traces[arm]["finite"] is True
        and traces[arm]["pointwise_trust_passed"] is True
        and traces[arm]["endpoint_conditional_ranks_are_16"] is True
        for arm in _ARMS
    )
    fold = _core.build_soft_polarity_fold_receipt(
        direction_receipt=_mapping(
            fit.receipt.get("direction_receipt"), label="soft fit direction"
        ),
        candidate_receipts=tuple(
            _mapping(item, label="soft fit candidate")
            for item in _sequence(
                fit.receipt.get("candidate_receipts"),
                label="soft fit candidates",
            )
        ),
        held_objectives_by_arm=objectives_by_arm,
        held_execution_receipt_sha256s_by_arm=execution_receipts,
        held_trace_evidence_sha256=str(selected_trace["artifact_sha256"]),
        response_gain_min=float(selected_trace["response_gain_min"]),
        response_gain_max=float(selected_trace["response_gain_max"]),
        response_gain_distinct_count=int(
            selected_trace["response_gain_distinct_count"]
        ),
        finite=all_healthy,
        pointwise_trust_passed=(
            selected_trace["pointwise_trust_passed"] is True
        ),
        exact_execution=True,
    )
    if (
        fold.get("selected_candidate_artifact_sha256")
        != fit.receipt.get("selected_candidate_artifact_sha256")
    ):
        raise RuntimeError("V20g held fold selection drifted after freeze")
    return fold, manifest, _hashed(
        {
            "outer_held_family_id": outer,
            "provider_manifest_sha256": manifest["artifact_sha256"],
            "arm_evidence": evidence_by_arm,
            "capability_receipt": capability_receipt,
            "all_four_providers_frozen_before_held_capability": True,
            "held_family_used_for_fit_or_selection": False,
            "exact_held_execution_count": len(_ARMS) * len(held),
            "raw_prompts_token_ids_logits_h4_or_teacher_rows_serialized": False,
        },
        domain=_HELD_EXECUTION_DOMAIN,
    )


_FOLD_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "source_artifact_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "fit_receipt",
        "fit_training_evidence",
        "provider_manifest",
        "held_evidence",
        "fold_receipt",
        "fixed_schedule_completed",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _validate_fit_bundle(
    fit_receipt: object,
    fit_evidence: object,
    *,
    outer_family_id: str | None,
    endpoint_receipt: Mapping[str, object] | None = None,
    endpoint_evidence: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    fit = _validate_hashed(
        _mapping(fit_receipt, label="fit receipt"),
        domain=_FIT_EXECUTION_DOMAIN,
        label="V20g fit receipt",
    )
    evidence = _validate_hashed(
        _mapping(fit_evidence, label="fit evidence"),
        domain=_FIT_EXECUTION_DOMAIN,
        label="V20g fit evidence",
    )
    direction = _mapping(fit.get("direction_receipt"), label="fit direction")
    _core.validate_soft_polarity_direction_receipt(direction)
    candidates = tuple(
        _mapping(item, label="fit candidate")
        for item in _sequence(fit.get("candidate_receipts"), label="fit candidates")
    )
    if len(candidates) != len(_core.SOFT_POLARITY_FIT_ALPHAS):
        raise ValueError("V20g fit alpha ladder is incomplete")
    candidates_by_alpha: dict[float, Mapping[str, object]] = {}
    for candidate in candidates:
        _core.validate_soft_polarity_candidate_receipt(
            candidate, direction_receipt=direction
        )
        alpha = float(candidate["alpha"])
        if alpha in candidates_by_alpha:
            raise ValueError("V20g fit alpha ladder contains a duplicate")
        candidates_by_alpha[alpha] = candidate
    expected_alphas = tuple(float(item) for item in _core.SOFT_POLARITY_FIT_ALPHAS)
    if set(candidates_by_alpha) != set(expected_alphas):
        raise ValueError("V20g fit alpha ladder differs")
    selected = min(
        candidates,
        key=lambda item: (
            float(item["family_equal_train_objective"]),
            float(item["alpha"]),
            str(item["artifact_sha256"]),
        ),
    )
    gradient_evidence = _validate_hashed(
        _mapping(evidence.get("gradient_evidence"), label="fit gradient evidence"),
        domain=_FIT_EXECUTION_DOMAIN,
        label="V20g fit gradient evidence",
    )
    provider_manifest = _validate_hashed(
        _mapping(
            evidence.get("candidate_provider_manifest"),
            label="fit candidate provider manifest",
        ),
        domain=_PROVIDER_MANIFEST_DOMAIN,
        label="V20g fit candidate provider manifest",
    )
    provider_hashes = _mapping(
        provider_manifest.get("provider_artifact_sha256s"),
        label="fit candidate provider hashes",
    )
    alpha_keys = tuple(str(alpha) for alpha in expected_alphas)
    if set(provider_hashes) != set(alpha_keys):
        raise ValueError("V20g fit candidate provider ladder differs")
    for key in alpha_keys:
        _sha(provider_hashes[key], label=f"fit provider {key}")
    candidate_evidence = _mapping(
        evidence.get("candidate_evidence"), label="fit candidate evidence"
    )
    if set(candidate_evidence) != set(alpha_keys):
        raise ValueError("V20g fit candidate evidence ladder differs")
    training_families = tuple(
        _identifier(item, label="fit training family")
        for item in _sequence(
            direction.get("training_family_ids"), label="fit training families"
        )
    )
    training_ids_by_family = _mapping(
        direction.get("training_example_ids_by_family"),
        label="fit training example ids",
    )
    training_example_ids = tuple(
        _identifier(example, label="fit training example")
        for family in training_families
        for example in _sequence(
            training_ids_by_family[family],
            label=f"fit training examples for {family}",
        )
    )
    gradient_example_families = {
        _identifier(example, label="fit gradient example"): _identifier(
            family, label="fit gradient family"
        )
        for example, family in _mapping(
            gradient_evidence.get("training_example_family_ids"),
            label="fit gradient example families",
        ).items()
    }
    for field, label in (
        ("eta_gradient_sha256s", "eta gradient"),
        ("post_cast_h4_sha256s", "fit H4"),
        ("supervised_full_vocab_logits_sha256s", "fit logits"),
        ("eta_zero_execution_sha256s", "eta-zero execution"),
    ):
        hashes = _mapping(gradient_evidence.get(field), label=f"{label} hashes")
        if set(hashes) != set(training_example_ids):
            raise ValueError("V20g fit gradient execution geometry differs")
        for item in hashes.values():
            _sha(item, label=label)
    _v20b._validate_capability_receipt(
        evidence.get("capability_receipt"),
        expected_example_ids=training_example_ids,
        expected_family_count=len(training_families),
        expected_held_family_id=outer_family_id,
        expected_accesses_per_example=len(expected_alphas),
        label="V20g resumed router-fit capability",
    )
    if (
        gradient_evidence.get("outer_held_family_id") != outer_family_id
        or tuple(gradient_evidence.get("training_family_ids", ()))
        != training_families
        or set(gradient_example_families) != set(training_example_ids)
        or any(
            gradient_example_families[example] != family
            for family in training_families
            for example in _sequence(
                training_ids_by_family[family],
                label=f"fit examples for {family}",
            )
        )
        or gradient_evidence.get("full_suffix_vjp_count")
        != len(training_example_ids)
        or gradient_evidence.get("local_eta_autograd_contraction_count")
        != len(training_example_ids)
        or gradient_evidence.get("family_equal_prompt_gradient_OPG") is not True
        or gradient_evidence.get("held_family_absent") is not True
        or evidence.get("exact_candidate_execution_count")
        != len(training_example_ids) * len(expected_alphas)
        or evidence.get("full_suffix_vjp_count") != len(training_example_ids)
        or evidence.get("held_family_rows_used") is not False
    ):
        raise ValueError("V20g fit capability or gradient boundary differs")
    all_families = tuple(
        _identifier(item, label="fit development family")
        for item in _sequence(
            direction.get("all_development_family_ids"),
            label="fit development families",
        )
    )
    sources = _mapping(
        direction.get("source_artifact_sha256s"), label="fit direction sources"
    )
    fit_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": sources["endpoint_receipt"],
            "endpoint_evidence_sha256": sources["endpoint_evidence"],
            "all_development_family_ids": all_families,
            "outer_held_family_id": outer_family_id,
            "eta": (0.0,) * FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    validated_candidate_evidence: dict[str, dict[str, object]] = {}
    for alpha in expected_alphas:
        key = str(alpha)
        candidate = candidates_by_alpha[alpha]
        row = _validate_hashed(
            _mapping(candidate_evidence[key], label=f"fit candidate evidence {key}"),
            domain=_FIT_EXECUTION_DOMAIN,
            label=f"V20g fit candidate evidence {key}",
        )
        validated_candidate_evidence[key] = row
        objectives_by_family = {
            str(family): {
                _identifier(example, label=f"fit candidate {key} example"): float(
                    objective
                )
                for example, objective in _mapping(
                    rows, label=f"fit candidate {key} family objectives"
                ).items()
            }
            for family, rows in _mapping(
                row.get("objectives_by_family"),
                label=f"fit candidate {key} objectives",
            ).items()
        }
        if set(objectives_by_family) != set(training_families) or any(
            tuple(sorted(objectives_by_family[family]))
            != tuple(training_ids_by_family[family])
            for family in training_families
        ):
            raise ValueError("V20g fit candidate prompt geometry differs")
        flattened_objectives = {
            example: objective
            for family in training_families
            for example, objective in objectives_by_family[family].items()
        }
        if not all(math.isfinite(item) for item in flattened_objectives.values()):
            raise ValueError("V20g fit candidate objective is nonfinite")
        candidate_h4 = _mapping(
            row.get("post_cast_h4_sha256s"),
            label=f"fit candidate {key} H4 hashes",
        )
        candidate_logits = _mapping(
            row.get("supervised_full_vocab_logits_sha256s"),
            label=f"fit candidate {key} logits hashes",
        )
        candidate_executions = _mapping(
            row.get("execution_sha256s"),
            label=f"fit candidate {key} executions",
        )
        if (
            set(candidate_h4) != set(training_example_ids)
            or set(candidate_logits) != set(training_example_ids)
            or set(candidate_executions) != set(training_example_ids)
        ):
            raise ValueError("V20g fit candidate execution geometry differs")
        trace = _validate_hashed(
            _mapping(row.get("response_trace"), label=f"fit candidate {key} trace"),
            domain=_PROVIDER_MANIFEST_DOMAIN,
            label=f"V20g fit candidate {key} trace",
        )
        trace_gain_hashes = _mapping(
            trace.get("response_gain_sha256s"),
            label=f"fit candidate {key} response-gain hashes",
        )
        if set(trace_gain_hashes) != set(training_example_ids):
            raise ValueError("V20g fit candidate trace geometry differs")
        for item in trace_gain_hashes.values():
            _sha(item, label=f"fit candidate {key} response gain")
        gain_min = float(trace.get("response_gain_min"))
        gain_max = float(trace.get("response_gain_max"))
        gain_distinct = trace.get("response_gain_distinct_count")
        gain_nonconstant = bool(
            math.isfinite(gain_min)
            and math.isfinite(gain_max)
            and type(gain_distinct) is int
            and gain_distinct >= 2
            and gain_min < gain_max
        )
        provider_seed = _v14._sha256(
            {
                "fit_seed_sha256": fit_seed,
                "direction_artifact_sha256": direction["artifact_sha256"],
                "alpha": alpha,
                "eta": tuple(float(item) for item in candidate["eta"]),
                "outer_held_family_id": outer_family_id,
                "held_rows_used": False,
            },
            domain=_FIT_EXECUTION_DOMAIN,
        )
        execution_seed = fit_seed if alpha == 0.0 else provider_seed
        expected_executions = {
            example: _execution_sha256(
                phase=("fit_eta_zero_vjp" if alpha == 0.0 else "fit_positive_alpha"),
                outer_family_id=outer_family_id,
                provider_artifact_sha256=str(provider_hashes[key]),
                example_id=example,
                family_id=gradient_example_families[example],
                objective=objective,
                h4_sha256=_sha(
                    candidate_h4[example], label=f"fit candidate {key} H4"
                ),
                logits_sha256=_sha(
                    candidate_logits[example],
                    label=f"fit candidate {key} logits",
                ),
                evidence_sha256=execution_seed,
            )
            for example, objective in flattened_objectives.items()
        }
        family_means = {
            family: math.fsum(objectives_by_family[family].values())
            / len(objectives_by_family[family])
            for family in training_families
        }
        macro = math.fsum(family_means.values()) / len(family_means)
        expected_execution_receipt = _v14._sha256(
            {
                "provider_manifest_sha256": provider_manifest["artifact_sha256"],
                "provider_artifact_sha256": provider_hashes[key],
                "alpha": alpha,
                "execution_sha256s": dict(sorted(expected_executions.items())),
                "family_equal_objective": macro,
            },
            domain=_FIT_EXECUTION_DOMAIN,
        )
        rebuilt = _core.build_soft_polarity_candidate_receipt(
            direction_receipt=direction,
            alpha=alpha,
            exact_train_objectives_by_family=objectives_by_family,
            execution_receipt_sha256=_sha(
                row.get("execution_receipt_sha256"),
                label=f"fit candidate {key} execution",
            ),
            exact_execution=True,
        )
        if (
            row.get("outer_held_family_id") != outer_family_id
            or row.get("alpha") != alpha
            or row.get("provider_artifact_sha256") != provider_hashes[key]
            or row.get("provider_manifest_sha256")
            != provider_manifest.get("artifact_sha256")
            or row.get("candidate_receipt_sha256")
            != candidate.get("artifact_sha256")
            or rebuilt.get("artifact_sha256") != candidate.get("artifact_sha256")
            or trace.get("arm") != f"fit_alpha_{alpha.hex()}"
            or trace.get("provider_artifact_sha256") != provider_hashes[key]
            or tuple(trace.get("scored_family_ids", ())) != training_families
            or trace.get("response_gain_nonconstant") is not gain_nonconstant
            or not (-1.0 <= gain_min <= gain_max <= 1.0)
            or trace.get("finite") is not True
            or trace.get("pointwise_trust_passed") is not True
            or trace.get("endpoint_conditional_ranks_are_16") is not True
            or dict(candidate_executions)
            != dict(sorted(expected_executions.items()))
            or row.get("execution_receipt_sha256")
            != expected_execution_receipt
            or row.get("family_equal_objective") != macro
            or _v14._canonical_json_bytes(row.get("family_mean_objectives"))
            != _v14._canonical_json_bytes(family_means)
            or row.get("eta_zero_execution_reused") is not (alpha == 0.0)
            or row.get("exact_execution") is not True
        ):
            raise ValueError("V20g fit candidate execution binding differs")
    if _v14._canonical_json_bytes(
        gradient_evidence.get("eta_zero_objectives_by_family")
    ) != _v14._canonical_json_bytes(
        validated_candidate_evidence[str(expected_alphas[0])].get(
            "objectives_by_family"
        )
    ):
        raise ValueError("V20g eta-zero objective evidence differs")
    zero_row = validated_candidate_evidence[str(expected_alphas[0])]
    if (
        gradient_evidence.get("eta_zero_provider_artifact_sha256")
        != provider_hashes[str(expected_alphas[0])]
        or _v14._canonical_json_bytes(
            gradient_evidence.get("post_cast_h4_sha256s")
        )
        != _v14._canonical_json_bytes(zero_row.get("post_cast_h4_sha256s"))
        or _v14._canonical_json_bytes(
            gradient_evidence.get("supervised_full_vocab_logits_sha256s")
        )
        != _v14._canonical_json_bytes(
            zero_row.get("supervised_full_vocab_logits_sha256s")
        )
        or _v14._canonical_json_bytes(
            gradient_evidence.get("eta_zero_execution_sha256s")
        )
        != _v14._canonical_json_bytes(zero_row.get("execution_sha256s"))
    ):
        raise ValueError("V20g eta-zero execution evidence differs")

    selected_alpha = float(selected["alpha"])
    selected_key = str(selected_alpha)
    selected_eta = tuple(
        float(item)
        for item in _sequence(fit.get("selected_eta"), label="fit selected eta")
    )
    expected_box_certificate = fisher_soft_polarity_box_certificate(
        _eta_tensor(selected_eta)
    )
    if (
        fit.get("outer_held_family_id") != outer_family_id
        or direction.get("held_family_id") != outer_family_id
        or fit.get("selected_alpha") != selected_alpha
        or selected_eta
        != tuple(float(item) for item in _sequence(
            selected.get("eta"), label="selected candidate eta"
        ))
        or fit.get("selected_candidate_artifact_sha256")
        != selected.get("artifact_sha256")
        or fit.get("selected_provider_artifact_sha256")
        != provider_hashes[selected_key]
        or fit.get("gradient_evidence_sha256")
        != gradient_evidence.get("artifact_sha256")
        or direction.get("gradient_evidence_sha256")
        != gradient_evidence.get("artifact_sha256")
        or fit.get("provider_manifest_sha256")
        != provider_manifest.get("artifact_sha256")
        or provider_manifest.get("outer_held_family_id") != outer_family_id
        or provider_manifest.get("direction_artifact_sha256")
        != direction.get("artifact_sha256")
        or tuple(float(item) for item in _sequence(
            provider_manifest.get("alpha_order"), label="fit manifest alpha order"
        ))
        != expected_alphas
        or provider_manifest.get(
            "all_alpha_candidates_frozen_before_positive_scoring"
        )
        is not True
        or provider_manifest.get("held_capability_count") != 0
        or fit.get("selection_frozen_before_held_scores") is not True
        or fit.get("held_family_used_for_fit_or_selection") is not False
        or _v14._canonical_json_bytes(fit.get("box_certificate"))
        != _v14._canonical_json_bytes(expected_box_certificate)
        or evidence.get("fit_receipt_sha256") != fit.get("artifact_sha256")
    ):
        raise ValueError("V20g fit selection or evidence binding differs")
    if endpoint_receipt is not None or endpoint_evidence is not None:
        if endpoint_receipt is None or endpoint_evidence is None:
            raise ValueError("V20g fit endpoint bundle is partial")
        sources = _mapping(
            direction.get("source_artifact_sha256s"), label="fit direction sources"
        )
        if (
            fit.get("endpoint_receipt_sha256")
            != endpoint_receipt.get("artifact_sha256")
            or sources.get("endpoint_receipt")
            != endpoint_receipt.get("artifact_sha256")
            or sources.get("endpoint_evidence")
            != endpoint_evidence.get("artifact_sha256")
        ):
            raise ValueError("V20g fit endpoint lineage differs")
    return fit, evidence


def _execute_outer_fold(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    family_ids: Sequence[str],
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
    authenticated_v20a_fold: Mapping[str, object],
) -> dict[str, object]:
    endpoint = _outer_endpoint(
        context,
        records,
        teacher_vault,
        family_ids=family_ids,
        outer_family_id=outer_family_id,
        panel_receipt=panel_receipt,
        authenticated_v20a_fold=authenticated_v20a_fold,
    )
    fit = _fit_soft_router(
        context,
        teacher_vault,
        endpoint=endpoint,
        all_family_ids=family_ids,
        outer_family_id=outer_family_id,
    )
    fold, manifest, held_evidence = _score_held_fold(
        context,
        records,
        teacher_vault,
        outer_family_id=outer_family_id,
        endpoint=endpoint,
        fit=fit,
    )
    return {
        "endpoint_receipt": endpoint.receipt,
        "endpoint_evidence": endpoint.evidence,
        "fit_receipt": fit.receipt,
        "fit_training_evidence": fit.training_evidence,
        "provider_manifest": manifest,
        "held_evidence": held_evidence,
        "fold_receipt": fold,
    }


def _fold_payload(
    live: Mapping[str, object],
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
) -> dict[str, object]:
    return {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": output.resolve(strict=False).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "outer_held_family_id": outer_family_id,
        "endpoint_receipt": dict(
            _mapping(live.get("endpoint_receipt"), label="live endpoint receipt")
        ),
        "endpoint_evidence": dict(
            _mapping(live.get("endpoint_evidence"), label="live endpoint evidence")
        ),
        "fit_receipt": dict(
            _mapping(live.get("fit_receipt"), label="live fit receipt")
        ),
        "fit_training_evidence": dict(
            _mapping(live.get("fit_training_evidence"), label="live fit evidence")
        ),
        "provider_manifest": dict(
            _mapping(live.get("provider_manifest"), label="live provider manifest")
        ),
        "held_evidence": dict(
            _mapping(live.get("held_evidence"), label="live held evidence")
        ),
        "fold_receipt": dict(
            _mapping(live.get("fold_receipt"), label="live fold receipt")
        ),
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
    }


def _validate_outer_endpoint_bundle(
    endpoint: Mapping[str, object],
    endpoint_evidence: Mapping[str, object],
    *,
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    authenticated_v20a_fold: Mapping[str, object],
) -> None:
    """Re-bind a resumed endpoint to the authenticated V20a fold and A16 panel."""

    outer = _identifier(outer_family_id, label="outer endpoint family")
    family_rows = _mapping(
        panel_receipt.get("family_prompt_sha256s"),
        label="outer endpoint panel families",
    )
    expected_families = tuple(sorted(str(item) for item in family_rows if item != outer))
    binding = _mapping(
        authenticated_v20a_fold.get("endpoint_binding"),
        label="outer endpoint authenticated V20a binding",
    )
    nested_fit = _v20b._core.validate_nested_microstep_fit_receipt(
        _mapping(
            endpoint_evidence.get("endpoint_fit_receipt"),
            label="outer endpoint nested fit",
        ),
        panel_receipt=panel_receipt,
    )
    nested_evidence = _v20b._validate_fit_training_evidence(
        endpoint_evidence.get("endpoint_fit_training_evidence"),
        fit_receipt=nested_fit,
    )
    example_families = _mapping(
        nested_evidence.get("example_family_ids"),
        label="outer endpoint example families",
    )
    endpoint_examples = tuple(
        _identifier(item, label="outer endpoint example")
        for item in _sequence(
            endpoint.get("training_example_ids"),
            label="outer endpoint training examples",
        )
    )
    if (
        authenticated_v20a_fold.get("held_family_id") != outer
        or binding.get("held_family_id") != outer
        or endpoint.get("kind") != "seven_family_outer_endpoint"
        or endpoint_evidence.get("kind")
        != "seven_family_outer_endpoint_fit_evidence"
        or endpoint.get("outer_held_family_id") != outer
        or endpoint_evidence.get("outer_held_family_id") != outer
        or endpoint.get("panel_receipt_sha256")
        != panel_receipt.get("artifact_sha256")
        or tuple(endpoint.get("training_family_ids", ())) != expected_families
        or tuple(nested_fit.get("training_family_ids", ())) != expected_families
        or tuple(nested_fit.get("excluded_family_ids", ())) != (outer,)
        or set(endpoint_examples) != set(example_families)
        or len(endpoint_examples) != len(expected_families) * _PROMPTS_PER_FAMILY
        or set(example_families.values()) != set(expected_families)
        or endpoint.get("base_provider_artifact_sha256")
        != nested_fit.get("base_provider_artifact_sha256")
        or endpoint.get("proposal_provider_artifact_sha256")
        != nested_fit.get("proposal_provider_artifact_sha256")
        or endpoint.get("base_provider_artifact_sha256")
        != binding.get("base_provider_artifact_sha256")
        or endpoint.get("proposal_provider_artifact_sha256")
        != binding.get("proposal_provider_artifact_sha256")
        or endpoint.get("endpoint_fit_receipt_sha256")
        != nested_fit.get("artifact_sha256")
        or endpoint.get("endpoint_fit_training_evidence_sha256")
        != endpoint_evidence.get("artifact_sha256")
        or endpoint.get("v20a_endpoint_binding_sha256")
        != _v14._sha256(binding, domain=_ENDPOINT_DOMAIN)
        or endpoint.get("held_family_absent_from_endpoint_fit") is not True
        or endpoint.get("authenticated_before_router_fit") is not True
    ):
        raise ValueError("V20g outer endpoint lineage differs")


def _validate_held_bundle(
    manifest: Mapping[str, object],
    held_evidence: Mapping[str, object],
    fold: Mapping[str, object],
    *,
    endpoint: Mapping[str, object],
    fit: Mapping[str, object],
    outer_family_id: str,
) -> None:
    """Reconstruct held execution commitments across all four frozen arms."""

    outer = _identifier(outer_family_id, label="held bundle family")
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s"),
        label="held provider hashes",
    )
    provider_receipts_raw = _mapping(
        manifest.get("provider_receipts"), label="held provider receipts"
    )
    if set(provider_hashes) != set(_ARMS) or set(provider_receipts_raw) != set(_ARMS):
        raise ValueError("V20g held provider arm geometry differs")
    provider_receipts = {
        arm: _validate_hashed(
            _mapping(provider_receipts_raw[arm], label=f"held {arm} provider receipt"),
            domain=_PROVIDER_MANIFEST_DOMAIN,
            label=f"V20g held {arm} provider receipt",
        )
        for arm in _ARMS
    }
    for arm in _ARMS:
        if (
            provider_receipts[arm].get("arm") != arm
            or provider_receipts[arm].get("provider_artifact_sha256")
            != provider_hashes[arm]
        ):
            raise ValueError("V20g held provider receipt binding differs")
    base_hash = endpoint.get("base_provider_artifact_sha256")
    proposal_hash = endpoint.get("proposal_provider_artifact_sha256")
    direction = _mapping(fit.get("direction_receipt"), label="held fit direction")
    direction_sources = _mapping(
        direction.get("source_artifact_sha256s"), label="held fit sources"
    )
    all_families = tuple(
        _identifier(item, label="held fit development family")
        for item in _sequence(
            direction.get("all_development_family_ids"),
            label="held fit development families",
        )
    )
    fit_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": direction_sources["endpoint_receipt"],
            "endpoint_evidence_sha256": direction_sources["endpoint_evidence"],
            "all_development_family_ids": all_families,
            "outer_held_family_id": outer,
            "eta": (0.0,) * FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    selected_eta = tuple(
        float(item)
        for item in _sequence(fit.get("selected_eta"), label="held selected eta")
    )
    selected_provider_seed = _v14._sha256(
        {
            "fit_seed_sha256": fit_seed,
            "direction_artifact_sha256": direction["artifact_sha256"],
            "alpha": float(fit["selected_alpha"]),
            "eta": selected_eta,
            "outer_held_family_id": outer,
            "held_rows_used": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    held_control_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "outer_held_family_id": outer,
            "endpoint_receipt_sha256": endpoint["artifact_sha256"],
            "fit_receipt_sha256": fit["artifact_sha256"],
            "scope": "four_arm_held_manifest_before_capability",
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )
    if (
        manifest.get("outer_held_family_id") != outer
        or manifest.get("endpoint_receipt_sha256")
        != endpoint.get("artifact_sha256")
        or manifest.get("fit_receipt_sha256") != fit.get("artifact_sha256")
        or tuple(manifest.get("arm_order", ())) != _ARMS
        or manifest.get("all_four_providers_frozen_before_held_capability")
        is not True
        or manifest.get("held_capability_count_at_freeze") != 0
        or manifest.get("held_objectives_or_teacher_rows_used") is not False
        or provider_hashes["base"] != base_hash
        or provider_hashes["soft_router"]
        != fit.get("selected_provider_artifact_sha256")
        or provider_receipts["base"].get("base_provider_artifact_sha256")
        != base_hash
        or provider_receipts["base"].get("proposal_provider_artifact_sha256")
        is not None
        or provider_receipts["soft_router"].get("eta_sha256")
        != _mapping(
            fit.get("box_certificate"), label="held fit box certificate"
        ).get("eta_sha256")
        or provider_receipts["soft_router"].get("transfer_evidence_sha256")
        != (
            fit_seed
            if float(fit["selected_alpha"]) == 0.0
            else selected_provider_seed
        )
        or any(
            provider_receipts[arm].get("transfer_evidence_sha256")
            != held_control_seed
            for arm in ("fixed_plus", "fixed_minus")
        )
        or any(
            provider_receipts[arm].get("base_provider_artifact_sha256")
            != base_hash
            or provider_receipts[arm].get("proposal_provider_artifact_sha256")
            != proposal_hash
            for arm in ("fixed_plus", "fixed_minus", "soft_router")
        )
    ):
        raise ValueError("V20g held provider manifest lineage differs")

    arm_evidence_raw = _mapping(
        held_evidence.get("arm_evidence"), label="held arm evidence"
    )
    if set(arm_evidence_raw) != set(_ARMS):
        raise ValueError("V20g held evidence arm geometry differs")
    arm_evidence: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, object]] = {}
    objectives_by_arm: dict[str, dict[str, float]] = {}
    execution_receipts: dict[str, str] = {}
    for arm in _ARMS:
        arm_row = _validate_hashed(
            _mapping(arm_evidence_raw[arm], label=f"held {arm} evidence"),
            domain=_HELD_EXECUTION_DOMAIN,
            label=f"V20g held {arm} evidence",
        )
        trace = _validate_hashed(
            _mapping(arm_row.get("response_trace"), label=f"held {arm} trace"),
            domain=_PROVIDER_MANIFEST_DOMAIN,
            label=f"V20g held {arm} trace",
        )
        objectives = {
            _identifier(example, label=f"held {arm} example"): float(value)
            for example, value in _mapping(
                arm_row.get("objectives_by_example"),
                label=f"held {arm} objectives",
            ).items()
        }
        if not objectives or not all(math.isfinite(item) for item in objectives.values()):
            raise ValueError("V20g held objectives are empty or nonfinite")
        h4_hashes = _mapping(
            arm_row.get("post_cast_h4_sha256s"), label=f"held {arm} H4 hashes"
        )
        logits_hashes = _mapping(
            arm_row.get("supervised_full_vocab_logits_sha256s"),
            label=f"held {arm} logits hashes",
        )
        executions = _mapping(
            arm_row.get("execution_sha256s"), label=f"held {arm} executions"
        )
        if set(h4_hashes) != set(objectives) or set(logits_hashes) != set(objectives) or set(executions) != set(objectives):
            raise ValueError("V20g held execution example geometry differs")
        arm_evidence[arm] = arm_row
        traces[arm] = trace
        objectives_by_arm[arm] = objectives
        execution_receipts[arm] = _sha(
            arm_row.get("execution_receipt_sha256"),
            label=f"held {arm} execution receipt",
        )
        if (
            arm_row.get("arm") != arm
            or arm_row.get("outer_held_family_id") != outer
            or arm_row.get("provider_artifact_sha256") != provider_hashes[arm]
            or arm_row.get("provider_manifest_sha256")
            != manifest.get("artifact_sha256")
            or trace.get("arm") != arm
            or trace.get("provider_artifact_sha256") != provider_hashes[arm]
            or tuple(trace.get("scored_family_ids", ())) != (outer,)
            or arm_row.get("exact_execution") is not True
            or arm_row.get("finite") is not True
        ):
            raise ValueError("V20g held arm lineage differs")

    trace_bundle_sha = _v14._sha256(
        {arm: traces[arm]["artifact_sha256"] for arm in _ARMS},
        domain=_HELD_EXECUTION_DOMAIN,
    )
    for arm in _ARMS:
        arm_row = arm_evidence[arm]
        objectives = objectives_by_arm[arm]
        h4_hashes = _mapping(
            arm_row["post_cast_h4_sha256s"], label=f"held {arm} H4 hashes"
        )
        logits_hashes = _mapping(
            arm_row["supervised_full_vocab_logits_sha256s"],
            label=f"held {arm} logits hashes",
        )
        executions = _mapping(
            arm_row["execution_sha256s"], label=f"held {arm} executions"
        )
        seed = _v14._sha256(
            {
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "arm": arm,
                "provider_artifact_sha256": provider_hashes[arm],
                "outer_held_family_id": outer,
            },
            domain=_HELD_EXECUTION_DOMAIN,
        )
        expected_executions = {
            example: _execution_sha256(
                phase=f"held_{arm}",
                outer_family_id=outer,
                provider_artifact_sha256=str(provider_hashes[arm]),
                example_id=example,
                family_id=outer,
                objective=objective,
                h4_sha256=_sha(h4_hashes[example], label=f"held {arm} H4"),
                logits_sha256=_sha(
                    logits_hashes[example], label=f"held {arm} logits"
                ),
                evidence_sha256=seed,
            )
            for example, objective in objectives.items()
        }
        macro = math.fsum(objectives.values()) / len(objectives)
        expected_receipt = _v14._sha256(
            {
                "seed_sha256": seed,
                "execution_sha256s": dict(sorted(expected_executions.items())),
                "objective": macro,
            },
            domain=_HELD_EXECUTION_DOMAIN,
        )
        if (
            dict(executions) != dict(sorted(expected_executions.items()))
            or arm_row.get("objective") != macro
            or execution_receipts[arm] != expected_receipt
        ):
            raise ValueError("V20g held exact execution commitment differs")

    held_ids = tuple(sorted(objectives_by_arm["base"]))
    if len(held_ids) != _PROMPTS_PER_FAMILY or any(
        tuple(sorted(objectives_by_arm[arm])) != held_ids for arm in _ARMS
    ):
        raise ValueError("V20g held arm prompt geometry differs")
    _v20b._validate_capability_receipt(
        held_evidence.get("capability_receipt"),
        expected_example_ids=held_ids,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20g resumed held capability",
    )
    selected_trace = traces["soft_router"]
    all_healthy = all(
        traces[arm].get("finite") is True
        and traces[arm].get("pointwise_trust_passed") is True
        and traces[arm].get("endpoint_conditional_ranks_are_16") is True
        for arm in _ARMS
    )
    rebuilt_fold = _core.build_soft_polarity_fold_receipt(
        direction_receipt=_mapping(fit["direction_receipt"], label="held direction"),
        candidate_receipts=tuple(
            _mapping(item, label="held candidate")
            for item in _sequence(fit["candidate_receipts"], label="held candidates")
        ),
        held_objectives_by_arm=objectives_by_arm,
        held_execution_receipt_sha256s_by_arm=execution_receipts,
        held_trace_evidence_sha256=str(selected_trace["artifact_sha256"]),
        response_gain_min=float(selected_trace["response_gain_min"]),
        response_gain_max=float(selected_trace["response_gain_max"]),
        response_gain_distinct_count=int(
            selected_trace["response_gain_distinct_count"]
        ),
        finite=all_healthy,
        pointwise_trust_passed=(
            selected_trace.get("pointwise_trust_passed") is True
        ),
        exact_execution=True,
    )
    if (
        held_evidence.get("outer_held_family_id") != outer
        or held_evidence.get("provider_manifest_sha256")
        != manifest.get("artifact_sha256")
        or held_evidence.get("all_four_providers_frozen_before_held_capability")
        is not True
        or held_evidence.get("held_family_used_for_fit_or_selection") is not False
        or held_evidence.get("exact_held_execution_count")
        != len(_ARMS) * _PROMPTS_PER_FAMILY
        or rebuilt_fold.get("artifact_sha256") != fold.get("artifact_sha256")
    ):
        raise ValueError("V20g held evidence or fold binding differs")


def _validate_fold_fragment(
    value: Mapping[str, object],
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20a_fold: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(value)
    if set(selected) != _FOLD_FRAGMENT_KEYS:
        raise ValueError("V20g fold fragment fields differ")
    outer = _identifier(outer_family_id, label="expected fold family")
    if (
        selected.get("schema") != _FOLD_SCHEMA
        or selected.get("format_version") != _FORMAT_VERSION
        or selected.get("target_output")
        != output.resolve(strict=False).as_posix()
        or selected.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or selected.get("core_protocol_sha256")
        != _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256
        or selected.get("source_artifact_sha256")
        != source.get("artifact_sha256")
        or selected.get("panel_receipt_sha256")
        != panel_receipt.get("artifact_sha256")
        or selected.get("bridge_binding_sha256") != bridge_binding_sha256
        or selected.get("outer_held_family_id") != outer
        or selected.get("fixed_schedule_completed") is not True
        or selected.get("candidate") is not None
        or selected.get("provider_sidecar") is not None
    ):
        raise ValueError("V20g fold fragment authority differs")
    endpoint = _validate_hashed(
        _mapping(selected.get("endpoint_receipt"), label="fold endpoint"),
        domain=_ENDPOINT_DOMAIN,
        label="V20g fold endpoint",
    )
    endpoint_evidence = _validate_hashed(
        _mapping(selected.get("endpoint_evidence"), label="fold endpoint evidence"),
        domain=_ENDPOINT_DOMAIN,
        label="V20g fold endpoint evidence",
    )
    _validate_outer_endpoint_bundle(
        endpoint,
        endpoint_evidence,
        panel_receipt=panel_receipt,
        outer_family_id=outer,
        authenticated_v20a_fold=authenticated_v20a_fold,
    )
    fit, _fit_evidence = _validate_fit_bundle(
        selected.get("fit_receipt"),
        selected.get("fit_training_evidence"),
        outer_family_id=outer,
        endpoint_receipt=endpoint,
        endpoint_evidence=endpoint_evidence,
    )
    manifest = _validate_hashed(
        _mapping(selected.get("provider_manifest"), label="fold manifest"),
        domain=_PROVIDER_MANIFEST_DOMAIN,
        label="V20g fold manifest",
    )
    held_evidence = _validate_hashed(
        _mapping(selected.get("held_evidence"), label="fold held evidence"),
        domain=_HELD_EXECUTION_DOMAIN,
        label="V20g fold held evidence",
    )
    fold = _mapping(selected.get("fold_receipt"), label="fold core receipt")
    direction = _mapping(fit.get("direction_receipt"), label="fold direction")
    candidates = tuple(
        _mapping(item, label="fold candidate")
        for item in _sequence(fit.get("candidate_receipts"), label="fold candidates")
    )
    _core.validate_soft_polarity_fold_receipt(
        fold, direction_receipt=direction, candidate_receipts=candidates
    )
    _validate_held_bundle(
        manifest,
        held_evidence,
        fold,
        endpoint=endpoint,
        fit=fit,
        outer_family_id=outer,
    )
    if (
        endpoint.get("outer_held_family_id") != outer
        or endpoint_evidence.get("artifact_sha256")
        != endpoint.get("endpoint_fit_training_evidence_sha256")
        or manifest.get("outer_held_family_id") != outer
        or held_evidence.get("outer_held_family_id") != outer
        or fold.get("held_family_id") != outer
        or fold.get("selected_candidate_artifact_sha256")
        != fit.get("selected_candidate_artifact_sha256")
    ):
        raise ValueError("V20g fold fragment lineage differs")
    _v14._scalar_report(selected)
    return selected


def _publish_fold_fragment(
    payload: Mapping[str, object], *, output: Path, outer_family_id: str
) -> dict[str, object]:
    return _v20b._publish_scalar_fragment(
        payload,
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20g outer-fold fragment",
    )


def _load_fold_fragment(
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20a_fold: Mapping[str, object],
) -> dict[str, object]:
    selected = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20g outer-fold fragment",
    )
    return _validate_fold_fragment(
        selected,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        outer_family_id=outer_family_id,
        bridge_binding_sha256=bridge_binding_sha256,
        authenticated_v20a_fold=authenticated_v20a_fold,
    )


_FINAL_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "source_artifact_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "oof_qualification_sha256",
        "endpoint_receipt",
        "endpoint_evidence",
        "fit_receipt",
        "fit_training_evidence",
        "final_provider_receipt",
        "final_provider_trace",
        "final_provider_freeze",
        "all_eight_family_refit_completed",
        "final_provider_frozen_before_calibration_b_eligibility",
        "final_provider_qualifies_for_calibration_b",
        "calibration_b_manifest_read",
        "calibration_b_tokenized",
        "calibration_b_scored",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _execute_final_refit(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    family_ids: Sequence[str],
    panel_receipt: Mapping[str, object],
    oof_qualification: Mapping[str, object],
) -> dict[str, object]:
    if oof_qualification.get("full_refit_authorized") is not True:
        raise PermissionError("V20g all-family refit requires passing OOF authority")
    endpoint = _all_family_endpoint(
        context,
        records,
        teacher_vault,
        family_ids=family_ids,
        panel_receipt=panel_receipt,
    )
    fit = _fit_soft_router(
        context,
        teacher_vault,
        endpoint=endpoint,
        all_family_ids=family_ids,
        outer_family_id=None,
    )
    provider_receipt = _provider_receipt(fit.provider, arm="soft_router")
    provider_trace = _provider_trace(fit.provider, records, arm="soft_router")
    selected_candidate = next(
        _mapping(item, label="final selected candidate")
        for item in _sequence(
            fit.receipt.get("candidate_receipts"), label="final fit candidates"
        )
        if item.get("artifact_sha256")
        == fit.receipt.get("selected_candidate_artifact_sha256")
    )
    provider_qualifies = bool(
        float(fit.receipt["selected_alpha"]) > 0.0
        and selected_candidate.get("execution_changed_from_base") is True
        and provider_trace.get("finite") is True
        and provider_trace.get("pointwise_trust_passed") is True
        and provider_trace.get("endpoint_conditional_ranks_are_16") is True
        and provider_trace.get("response_gain_nonconstant") is True
        and -1.0 <= float(provider_trace["response_gain_min"])
        <= float(provider_trace["response_gain_max"])
        <= 1.0
    )
    freeze = _hashed(
        {
            "oof_qualification_sha256": oof_qualification["artifact_sha256"],
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "fit_receipt_sha256": fit.receipt["artifact_sha256"],
            "selected_alpha": fit.receipt["selected_alpha"],
            "selected_eta": fit.receipt["selected_eta"],
            "selected_candidate_artifact_sha256": fit.receipt[
                "selected_candidate_artifact_sha256"
            ],
            "final_provider_artifact_sha256": fit.provider.artifact_sha256,
            "final_provider_receipt_sha256": provider_receipt["artifact_sha256"],
            "final_provider_trace_sha256": provider_trace["artifact_sha256"],
            "final_provider_qualifies_for_calibration_b": provider_qualifies,
            "all_eight_development_families_used": True,
            "provider_frozen_before_calibration_b_eligibility": True,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
            "calibration_b_scored": False,
            "raw_provider_tensors_serialized": False,
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )
    return {
        "endpoint_receipt": endpoint.receipt,
        "endpoint_evidence": endpoint.evidence,
        "fit_receipt": fit.receipt,
        "fit_training_evidence": fit.training_evidence,
        "final_provider_receipt": provider_receipt,
        "final_provider_trace": provider_trace,
        "final_provider_freeze": freeze,
    }


def _final_payload(
    live: Mapping[str, object],
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    oof_qualification: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": _FINAL_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": output.resolve(strict=False).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "oof_qualification_sha256": oof_qualification["artifact_sha256"],
        "endpoint_receipt": dict(
            _mapping(live.get("endpoint_receipt"), label="final endpoint receipt")
        ),
        "endpoint_evidence": dict(
            _mapping(live.get("endpoint_evidence"), label="final endpoint evidence")
        ),
        "fit_receipt": dict(
            _mapping(live.get("fit_receipt"), label="final fit receipt")
        ),
        "fit_training_evidence": dict(
            _mapping(live.get("fit_training_evidence"), label="final fit evidence")
        ),
        "final_provider_receipt": dict(
            _mapping(
                live.get("final_provider_receipt"),
                label="final provider receipt",
            )
        ),
        "final_provider_trace": dict(
            _mapping(
                live.get("final_provider_trace"),
                label="final provider trace",
            )
        ),
        "final_provider_freeze": dict(
            _mapping(live.get("final_provider_freeze"), label="final provider freeze")
        ),
        "all_eight_family_refit_completed": True,
        "final_provider_frozen_before_calibration_b_eligibility": True,
        "final_provider_qualifies_for_calibration_b": bool(
            _mapping(
                live.get("final_provider_freeze"), label="final provider freeze"
            ).get("final_provider_qualifies_for_calibration_b")
        ),
        "calibration_b_manifest_read": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "candidate": None,
        "provider_sidecar": None,
    }


def _validate_all_family_endpoint_bundle(
    endpoint: Mapping[str, object],
    endpoint_evidence: Mapping[str, object],
    *,
    panel_receipt: Mapping[str, object],
) -> None:
    families = tuple(
        sorted(
            str(item)
            for item in _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="final endpoint panel families",
            )
        )
    )
    example_families = {
        _identifier(example, label="final endpoint example"): _identifier(
            family, label="final endpoint example family"
        )
        for example, family in _mapping(
            endpoint_evidence.get("training_example_family_ids"),
            label="final endpoint example families",
        ).items()
    }
    endpoint_examples = tuple(
        _identifier(item, label="final endpoint receipt example")
        for item in _sequence(
            endpoint.get("training_example_ids"),
            label="final endpoint training examples",
        )
    )
    base_objectives = {
        _identifier(example, label="final endpoint objective example"): float(value)
        for example, value in _mapping(
            endpoint_evidence.get("base_objectives_by_example"),
            label="final endpoint objectives",
        ).items()
    }
    h4_hashes = _mapping(
        endpoint_evidence.get("post_cast_h4_sha256s"),
        label="final endpoint H4 hashes",
    )
    logits_hashes = _mapping(
        endpoint_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="final endpoint logits hashes",
    )
    if (
        set(base_objectives) != set(endpoint_examples)
        or set(h4_hashes) != set(endpoint_examples)
        or set(logits_hashes) != set(endpoint_examples)
        or not all(math.isfinite(item) for item in base_objectives.values())
    ):
        raise ValueError("V20g all-family endpoint execution geometry differs")
    for item in h4_hashes.values():
        _sha(item, label="final endpoint H4")
    for item in logits_hashes.values():
        _sha(item, label="final endpoint logits")
    capability = _v20b._validate_capability_receipt(
        endpoint_evidence.get("capability_receipt"),
        expected_example_ids=tuple(sorted(example_families)),
        expected_family_count=_FAMILY_COUNT,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label="V20g final endpoint capability",
    )
    del capability
    # Runtime flags are each scalar mappings, not a row sequence.  Validate
    # their exact health predicates while leaving their hashed diagnostics in
    # the enclosing endpoint-evidence commitment.
    base_flags = _mapping(
        endpoint_evidence.get("base_runtime_flags"),
        label="final endpoint base runtime flags",
    )
    proposal_flags = _mapping(
        endpoint_evidence.get("proposal_runtime_flags"),
        label="final endpoint proposal runtime flags",
    )
    if (
        len(families) != _FAMILY_COUNT
        or endpoint.get("kind") != "all_eight_family_endpoint"
        or endpoint_evidence.get("kind")
        != "all_eight_family_endpoint_fit_evidence"
        or endpoint.get("panel_receipt_sha256")
        != panel_receipt.get("artifact_sha256")
        or endpoint_evidence.get("panel_receipt_sha256")
        != panel_receipt.get("artifact_sha256")
        or tuple(endpoint.get("training_family_ids", ())) != families
        or tuple(endpoint_evidence.get("training_family_ids", ())) != families
        or set(endpoint_examples) != set(example_families)
        or len(endpoint_examples) != _FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or set(example_families.values()) != set(families)
        or any(
            tuple(example_families.values()).count(family)
            != _PROMPTS_PER_FAMILY
            for family in families
        )
        or endpoint.get("fit_evidence_sha256")
        != endpoint_evidence.get("artifact_sha256")
        or endpoint.get("finite_trusted_rank16") is not True
        or endpoint_evidence.get("full_suffix_vjp_count")
        != _FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or endpoint_evidence.get("local_endpoint_autograd_contraction_count")
        != _FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or any(
            row.get("finite") is not True
            or row.get("pointwise_trust_passed") is not True
            or row.get("rank_is_16") is not True
            for row in (base_flags, proposal_flags)
        )
    ):
        raise ValueError("V20g all-family endpoint lineage differs")
    _sha(endpoint.get("base_provider_artifact_sha256"), label="final endpoint base")
    _sha(
        endpoint.get("proposal_provider_artifact_sha256"),
        label="final endpoint proposal",
    )


def _validate_final_fragment(
    value: Mapping[str, object],
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    oof_qualification: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(value)
    if set(selected) != _FINAL_FRAGMENT_KEYS:
        raise ValueError("V20g final-refit fragment fields differ")
    if (
        oof_qualification.get("full_refit_authorized") is not True
        or selected.get("schema") != _FINAL_SCHEMA
        or selected.get("format_version") != _FORMAT_VERSION
        or selected.get("target_output")
        != output.resolve(strict=False).as_posix()
        or selected.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or selected.get("core_protocol_sha256")
        != _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256
        or selected.get("source_artifact_sha256")
        != source.get("artifact_sha256")
        or selected.get("panel_receipt_sha256")
        != panel_receipt.get("artifact_sha256")
        or selected.get("bridge_binding_sha256") != bridge_binding_sha256
        or selected.get("oof_qualification_sha256")
        != oof_qualification.get("artifact_sha256")
        or selected.get("all_eight_family_refit_completed") is not True
        or selected.get("final_provider_frozen_before_calibration_b_eligibility")
        is not True
        or type(selected.get("final_provider_qualifies_for_calibration_b")) is not bool
        or any(
            selected.get(name) is not False
            for name in (
                "calibration_b_manifest_read",
                "calibration_b_tokenized",
                "calibration_b_scored",
            )
        )
        or selected.get("candidate") is not None
        or selected.get("provider_sidecar") is not None
    ):
        raise ValueError("V20g final-refit authority differs")
    endpoint = _validate_hashed(
        _mapping(selected.get("endpoint_receipt"), label="final endpoint"),
        domain=_ENDPOINT_DOMAIN,
        label="V20g final endpoint",
    )
    endpoint_evidence = _validate_hashed(
        _mapping(selected.get("endpoint_evidence"), label="final endpoint evidence"),
        domain=_ENDPOINT_DOMAIN,
        label="V20g final endpoint evidence",
    )
    _validate_all_family_endpoint_bundle(
        endpoint,
        endpoint_evidence,
        panel_receipt=panel_receipt,
    )
    endpoint_examples = tuple(
        _identifier(item, label="final fragment endpoint example")
        for item in _sequence(
            endpoint.get("training_example_ids"),
            label="final fragment endpoint examples",
        )
    )
    fit, fit_evidence = _validate_fit_bundle(
        selected.get("fit_receipt"),
        selected.get("fit_training_evidence"),
        outer_family_id=None,
        endpoint_receipt=endpoint,
        endpoint_evidence=endpoint_evidence,
    )
    provider = _validate_hashed(
        _mapping(selected.get("final_provider_receipt"), label="final provider"),
        domain=_PROVIDER_MANIFEST_DOMAIN,
        label="V20g final provider",
    )
    trace = _validate_hashed(
        _mapping(selected.get("final_provider_trace"), label="final provider trace"),
        domain=_PROVIDER_MANIFEST_DOMAIN,
        label="V20g final provider trace",
    )
    freeze = _validate_hashed(
        _mapping(selected.get("final_provider_freeze"), label="final freeze"),
        domain=_PROVIDER_MANIFEST_DOMAIN,
        label="V20g final freeze",
    )
    selected_candidate = next(
        _mapping(item, label="final frozen selected candidate")
        for item in _sequence(
            fit.get("candidate_receipts"), label="final frozen candidates"
        )
        if item.get("artifact_sha256")
        == fit.get("selected_candidate_artifact_sha256")
    )
    trace_gain_hashes = _mapping(
        trace.get("response_gain_sha256s"), label="final response-gain hashes"
    )
    if set(trace_gain_hashes) != set(endpoint_examples):
        raise ValueError("V20g final response-gain trace geometry differs")
    for item in trace_gain_hashes.values():
        _sha(item, label="final response-gain trace")
    gain_min = float(trace.get("response_gain_min"))
    gain_max = float(trace.get("response_gain_max"))
    distinct = trace.get("response_gain_distinct_count")
    trace_nonconstant = bool(
        math.isfinite(gain_min)
        and math.isfinite(gain_max)
        and type(distinct) is int
        and distinct >= 2
        and gain_min < gain_max
    )
    if trace.get("response_gain_nonconstant") is not trace_nonconstant:
        raise ValueError("V20g final response-gain health differs")
    qualifies = bool(
        float(fit["selected_alpha"]) > 0.0
        and selected_candidate.get("execution_changed_from_base") is True
        and trace.get("finite") is True
        and trace.get("pointwise_trust_passed") is True
        and trace.get("endpoint_conditional_ranks_are_16") is True
        and trace_nonconstant
        and -1.0 <= gain_min <= gain_max <= 1.0
    )
    final_direction = _mapping(
        fit.get("direction_receipt"), label="final fit direction"
    )
    final_sources = _mapping(
        final_direction.get("source_artifact_sha256s"), label="final fit sources"
    )
    final_families = tuple(
        _identifier(item, label="final fit development family")
        for item in _sequence(
            final_direction.get("all_development_family_ids"),
            label="final fit development families",
        )
    )
    final_fit_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": final_sources["endpoint_receipt"],
            "endpoint_evidence_sha256": final_sources["endpoint_evidence"],
            "all_development_family_ids": final_families,
            "outer_held_family_id": None,
            "eta": (0.0,) * FISHER_SOFT_POLARITY_ETA_COUNT,
            "held_rows_used": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    final_eta = tuple(
        float(item)
        for item in _sequence(fit.get("selected_eta"), label="final selected eta")
    )
    final_provider_seed = _v14._sha256(
        {
            "fit_seed_sha256": final_fit_seed,
            "direction_artifact_sha256": final_direction["artifact_sha256"],
            "alpha": float(fit["selected_alpha"]),
            "eta": final_eta,
            "outer_held_family_id": None,
            "held_rows_used": False,
        },
        domain=_FIT_EXECUTION_DOMAIN,
    )
    if (
        endpoint.get("kind") != "all_eight_family_endpoint"
        or endpoint_evidence.get("artifact_sha256")
        != endpoint.get("fit_evidence_sha256")
        or fit.get("outer_held_family_id") is not None
        or provider.get("arm") != "soft_router"
        or provider.get("provider_artifact_sha256")
        != fit.get("selected_provider_artifact_sha256")
        or provider.get("base_provider_artifact_sha256")
        != endpoint.get("base_provider_artifact_sha256")
        or provider.get("proposal_provider_artifact_sha256")
        != endpoint.get("proposal_provider_artifact_sha256")
        or provider.get("eta_sha256")
        != _mapping(
            fit.get("box_certificate"), label="final fit box certificate"
        ).get("eta_sha256")
        or provider.get("transfer_evidence_sha256")
        != (
            final_fit_seed
            if float(fit["selected_alpha"]) == 0.0
            else final_provider_seed
        )
        or trace.get("arm") != "soft_router"
        or trace.get("provider_artifact_sha256")
        != provider.get("provider_artifact_sha256")
        or tuple(trace.get("scored_family_ids", ()))
        != tuple(endpoint.get("training_family_ids", ()))
        or freeze.get("provider_frozen_before_calibration_b_eligibility")
        is not True
        or freeze.get("oof_qualification_sha256")
        != oof_qualification.get("artifact_sha256")
        or freeze.get("endpoint_receipt_sha256")
        != endpoint.get("artifact_sha256")
        or freeze.get("selected_alpha") != fit.get("selected_alpha")
        or tuple(freeze.get("selected_eta", ()))
        != tuple(fit.get("selected_eta", ()))
        or freeze.get("selected_candidate_artifact_sha256")
        != fit.get("selected_candidate_artifact_sha256")
        or freeze.get("final_provider_artifact_sha256")
        != provider.get("provider_artifact_sha256")
        or freeze.get("final_provider_receipt_sha256")
        != provider.get("artifact_sha256")
        or freeze.get("final_provider_trace_sha256")
        != trace.get("artifact_sha256")
        or freeze.get("final_provider_qualifies_for_calibration_b") is not qualifies
        or selected.get("final_provider_qualifies_for_calibration_b") is not qualifies
        or freeze.get("fit_receipt_sha256") != fit.get("artifact_sha256")
        or fit_evidence.get("fit_receipt_sha256") != fit.get("artifact_sha256")
    ):
        raise ValueError("V20g final-refit lineage differs")
    _v14._scalar_report(selected)
    return selected


def _publish_final_fragment(
    payload: Mapping[str, object], *, output: Path
) -> dict[str, object]:
    return _v20b._publish_scalar_fragment(
        payload,
        path=_final_path(output),
        domain=_FINAL_DOMAIN,
        hash_key="fragment_sha256",
        label="V20g all-family final-refit fragment",
    )


def _load_final_fragment(
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    oof_qualification: Mapping[str, object],
) -> dict[str, object]:
    selected = _v20b._load_scalar_fragment(
        path=_final_path(output),
        domain=_FINAL_DOMAIN,
        hash_key="fragment_sha256",
        label="V20g all-family final-refit fragment",
    )
    return _validate_final_fragment(
        selected,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        oof_qualification=oof_qualification,
    )


def _runner_work_accounting(
    *,
    fold_fragments: Mapping[str, Mapping[str, object]],
    full_refit_performed: bool,
) -> dict[str, object]:
    directions = []
    candidates = []
    folds = []
    for family in sorted(fold_fragments):
        fragment = fold_fragments[family]
        fit = _mapping(fragment.get("fit_receipt"), label="work fit receipt")
        directions.append(
            _mapping(fit.get("direction_receipt"), label="work direction")
        )
        candidates.extend(
            _mapping(item, label="work candidate")
            for item in _sequence(
                fit.get("candidate_receipts"), label="work candidates"
            )
        )
        folds.append(
            _mapping(fragment.get("fold_receipt"), label="work fold")
        )
    core = _core.soft_polarity_work_accounting(
        direction_receipts=directions,
        candidate_receipts=candidates,
        fold_receipts=folds,
        full_refit_performed=full_refit_performed,
    )
    alpha_count = len(_core.SOFT_POLARITY_FIT_ALPHAS)
    authority_forwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * 2
    authority_backwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY
    outer_rows = _FAMILY_COUNT * (_FAMILY_COUNT - 1) * _PROMPTS_PER_FAMILY
    outer_endpoint_vjp = outer_rows
    outer_router_vjp = outer_rows
    outer_candidate_forwards = outer_rows * (alpha_count - 1)
    outer_held_forwards = (
        _FAMILY_COUNT * _PROMPTS_PER_FAMILY * len(_ARMS)
    )
    final_rows = _FAMILY_COUNT * _PROMPTS_PER_FAMILY if full_refit_performed else 0
    final_endpoint_vjp = final_rows
    final_router_vjp = final_rows
    final_candidate_forwards = final_rows * (alpha_count - 1)
    outer_fit_trace_rows = outer_rows * alpha_count
    outer_held_trace_rows = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * len(_ARMS)
    final_fit_trace_rows = final_rows * alpha_count
    final_freeze_trace_rows = final_rows
    outer_endpoint_health_rows = outer_rows * 2
    final_endpoint_health_rows = final_rows * 2
    canonical_forwards = (
        authority_forwards
        + outer_endpoint_vjp
        + outer_router_vjp
        + outer_candidate_forwards
        + outer_held_forwards
        + final_endpoint_vjp
        + final_router_vjp
        + final_candidate_forwards
    )
    teacher_capability_accesses = (
        outer_endpoint_vjp
        + outer_router_vjp
        + outer_candidate_forwards
        + outer_held_forwards
        + final_endpoint_vjp
        + final_router_vjp
        + final_candidate_forwards
    )
    return {
        "accounting_scope": "canonical_one_shot_schedule",
        "resume_attempt_overhead_included": False,
        "resume_attempt_count_persisted": False,
        "resume_note": (
            "each partial resume repeats A16 authority collection; actual cumulative "
            "work requires an external attempt ledger"
        ),
        "core_logical_accounting": core,
        "authenticated_A16_collection_forward_count": authority_forwards,
        "authenticated_A16_collection_suffix_backward_count": authority_backwards,
        "outer_endpoint_vjp_forward_backward_count": outer_endpoint_vjp,
        "outer_router_vjp_forward_backward_count": outer_router_vjp,
        "outer_positive_candidate_exact_forward_count": outer_candidate_forwards,
        "outer_held_four_arm_exact_forward_count": outer_held_forwards,
        "full_refit_performed": full_refit_performed,
        "full_endpoint_vjp_forward_backward_count": final_endpoint_vjp,
        "full_router_vjp_forward_backward_count": final_router_vjp,
        "full_positive_candidate_exact_forward_count": final_candidate_forwards,
        "outer_fit_provider_trace_sequence_count": outer_fit_trace_rows,
        "outer_held_provider_trace_sequence_count": outer_held_trace_rows,
        "full_fit_provider_trace_sequence_count": final_fit_trace_rows,
        "full_freeze_provider_trace_sequence_count": final_freeze_trace_rows,
        "outer_endpoint_health_sequence_count": outer_endpoint_health_rows,
        "full_endpoint_health_sequence_count": final_endpoint_health_rows,
        "total_local_provider_trace_sequence_count": (
            outer_fit_trace_rows
            + outer_held_trace_rows
            + final_fit_trace_rows
            + final_freeze_trace_rows
        ),
        "runtime_diagnostic_modal_pass_count": (
            outer_fit_trace_rows
            + outer_held_trace_rows
            + final_fit_trace_rows
            + final_freeze_trace_rows
        ),
        "response_gain_trace_pass_count": (
            outer_fit_trace_rows
            + outer_held_trace_rows
            + final_fit_trace_rows
            + final_freeze_trace_rows
        ),
        "endpoint_runtime_health_pass_count": (
            outer_endpoint_health_rows + final_endpoint_health_rows
        ),
        "canonical_one_shot_total_model_forward_count": canonical_forwards,
        "total_model_forward_count": canonical_forwards,
        "canonical_teacher_capability_access_count": teacher_capability_accesses,
        "total_suffix_backward_count": (
            authority_backwards
            + outer_endpoint_vjp
            + outer_router_vjp
            + final_endpoint_vjp
            + final_router_vjp
        ),
        "total_local_autograd_contraction_count": (
            outer_endpoint_vjp
            + outer_router_vjp
            + final_endpoint_vjp
            + final_router_vjp
        ),
        "calibration_b_forward_or_tokenization_count": 0,
    }


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    fold_fragments: Mapping[str, Mapping[str, object]],
    oof_qualification: Mapping[str, object],
    final_fragment: Mapping[str, object] | None,
) -> dict[str, object]:
    fold_receipts = tuple(
        dict(
            _mapping(
                fold_fragments[family].get("fold_receipt"),
                label="report fold receipt",
            )
        )
        for family in sorted(fold_fragments)
    )
    _core.validate_soft_polarity_oof_qualification(
        oof_qualification, fold_receipts=fold_receipts
    )
    gate = oof_qualification.get("full_refit_authorized") is True
    if gate is not (final_fragment is not None):
        raise ValueError("V20g report final-refit gate differs")
    final_summary: dict[str, object] | None = None
    if final_fragment is not None:
        final_summary = {
            "fragment_sha256": final_fragment["fragment_sha256"],
            "endpoint_receipt": final_fragment["endpoint_receipt"],
            "fit_receipt": final_fragment["fit_receipt"],
            "final_provider_receipt": final_fragment["final_provider_receipt"],
            "final_provider_trace": final_fragment["final_provider_trace"],
            "final_provider_freeze": final_fragment["final_provider_freeze"],
            "final_provider_qualifies_for_calibration_b": final_fragment[
                "final_provider_qualifies_for_calibration_b"
            ],
        }
    final_qualifies = bool(
        final_summary is not None
        and final_summary["final_provider_qualifies_for_calibration_b"] is True
    )
    eligible = bool(gate and final_summary is not None and final_qualifies)
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "artifact": {
            "path": output.resolve(strict=False).as_posix(),
            "role": "historically_reused_A16_soft_polarity_trust_region_nested_development",
        },
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
        "fixed_protocol": _FIXED_PROTOCOL,
        "source_receipt": dict(source),
        "panel_receipt": dict(panel_receipt),
        "bridge_binding_sha256": bridge_binding_sha256,
        "fold_fragment_sha256s_by_family": {
            family: fold_fragments[family]["fragment_sha256"]
            for family in sorted(fold_fragments)
        },
        "fold_receipts": fold_receipts,
        "oof_qualification": dict(oof_qualification),
        "final_refit": final_summary,
        "classification": (
            "soft_polarity_trust_region_oof_passed_final_provider_frozen"
            if eligible
            else (
                "soft_polarity_trust_region_oof_passed_final_refit_rolled_back_to_base"
                if gate
                else "soft_polarity_trust_region_oof_failed_rollback_to_base"
            )
        ),
        "passed": eligible,
        "oof_passed": gate,
        "rollback_to_base": not eligible,
        "all_eight_outer_folds_completed": len(fold_receipts) == _FAMILY_COUNT,
        "all_eight_family_refit_completed": final_summary is not None,
        "final_provider_frozen": final_summary is not None,
        "final_provider_qualifies_for_calibration_b": final_qualifies,
        "calibration_b_eligibility_gate_passed": eligible,
        "calibration_b_eligible": eligible,
        "calibration_b_authorized": False,
        "calibration_b_manifest_read": False,
        "calibration_b_opened": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "validation_opened": False,
        "test_opened": False,
        "fresh_family_disjoint_scoring_performed": False,
        "serving_claim_authorized": False,
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
        "fixed_minus_is_diagnostic_only": True,
        "conditional_router_dominance_over_both_fixed_signs_claimed": False,
        "candidate": None,
        "provider_sidecar": None,
        "work_accounting": _runner_work_accounting(
            fold_fragments=fold_fragments,
            full_refit_performed=final_summary is not None,
        ),
        "integrity": {
            "prerequisites_authenticated_before_model_construction": True,
            "all_fold_fragments_mode_0600_hash_authenticated": True,
            "all_candidates_selected_without_outer_held_family": True,
            "all_four_arms_frozen_before_each_held_capability": True,
            "final_refit_started_only_after_complete_OOF_gate": True,
            "final_provider_frozen_before_calibration_b_eligibility": True,
            "raw_prompts_tokens_logits_h4_gradients_or_provider_tensors_serialized": False,
        },
    }
    _v14._scalar_report(report)
    return report


def _load_existing_report(
    output: Path,
    *,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    authenticated_folds: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20g report",
    )
    families = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="report panel families",
            )
        )
    )
    folds = {
        family: _load_fold_fragment(
            output=output,
            source=source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20a_fold=_mapping(
                authenticated_folds.get(family),
                label="report authenticated V20a fold",
            ),
        )
        for family in families
    }
    fold_receipts = tuple(
        _mapping(folds[family]["fold_receipt"], label="report fold")
        for family in families
    )
    qualification = _core.build_soft_polarity_oof_qualification(
        fold_receipts=fold_receipts
    )
    final: dict[str, object] | None = None
    if qualification.get("full_refit_authorized") is True:
        final = _load_final_fragment(
            output=output,
            source=source,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding_sha256,
            oof_qualification=qualification,
        )
    elif _final_path(output).exists():
        raise ValueError("V20g failed OOF report has a stale final-refit fragment")
    rebuilt = _build_report(
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        fold_fragments=folds,
        oof_qualification=qualification,
        final_fragment=final,
    )
    supplied = dict(value)
    report_sha = supplied.pop("report_sha256", None)
    if (
        _v14._canonical_json_bytes(supplied)
        != _v14._canonical_json_bytes(rebuilt)
        or report_sha
        != _v14._sha256(rebuilt, domain=_REPORT_DOMAIN)
    ):
        raise ValueError("V20g report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or resume the eight-fold development screen and gated final refit."""

    destination = _validate_output(output)
    # Historical authorities always authenticate before any model object is
    # constructed.  Existing reports are also re-bound to those authorities.
    prerequisite, authenticated_folds, source = _load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"),
            label="V20g nested panel receipt",
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20g authenticated bridge",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            source=source,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_folds=authenticated_folds,
        )

    family_ids = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20g development families",
            )
        )
    )
    if (
        len(family_ids) != _FAMILY_COUNT
        or set(authenticated_folds) != set(family_ids)
    ):
        raise RuntimeError("V20g authenticated family geometry differs")

    # A complete failed-OOF campaign (or a passing campaign whose final-refit
    # fragment already exists) can be authenticated and published without
    # rebuilding Gemma or recollecting A16 authority.  This is especially
    # important after an aggregation-only interruption: completed expensive
    # folds remain the sole scoring authority.
    if all(_fold_path(destination, family).exists() for family in family_ids):
        completed_fragments = {
            family: _load_fold_fragment(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                outer_family_id=family,
                bridge_binding_sha256=bridge_binding,
                authenticated_v20a_fold=authenticated_folds[family],
            )
            for family in family_ids
        }
        completed_folds = tuple(
            _mapping(
                completed_fragments[family]["fold_receipt"],
                label="V20g completed fold",
            )
            for family in family_ids
        )
        completed_qualification = _core.build_soft_polarity_oof_qualification(
            fold_receipts=completed_folds
        )
        completed_final: dict[str, object] | None = None
        can_finish_without_model = (
            completed_qualification.get("full_refit_authorized") is not True
        )
        if completed_qualification.get("full_refit_authorized") is True:
            if _final_path(destination).exists():
                completed_final = _load_final_fragment(
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    bridge_binding_sha256=bridge_binding,
                    oof_qualification=completed_qualification,
                )
                can_finish_without_model = True
        elif _final_path(destination).exists():
            raise ValueError("V20g failed OOF campaign has a final-refit fragment")
        if can_finish_without_model:
            completed_report = _build_report(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                bridge_binding_sha256=bridge_binding,
                fold_fragments=completed_fragments,
                oof_qualification=completed_qualification,
                final_fragment=completed_final,
            )
            try:
                _v20b._publish_scalar_fragment(
                    completed_report,
                    path=destination,
                    domain=_REPORT_DOMAIN,
                    hash_key="report_sha256",
                    label="V20g report",
                )
            except FileExistsError:
                pass
            return _load_existing_report(
                destination,
                source=source,
                panel_receipt=panel_receipt,
                bridge_binding_sha256=bridge_binding,
                authenticated_folds=authenticated_folds,
            )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        if context.bridge.bridge_binding_sha256 != bridge_binding:
            raise RuntimeError("V20g live bridge differs from authenticated V20a")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != family_ids:
            raise RuntimeError("V20g live family order differs from authenticated A16")
        fragments: dict[str, dict[str, object]] = {}
        for family in family_ids:
            path = _fold_path(destination, family)
            if path.exists():
                fragments[family] = _load_fold_fragment(
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    outer_family_id=family,
                    bridge_binding_sha256=bridge_binding,
                    authenticated_v20a_fold=authenticated_folds[family],
                )
                continue
            live = _execute_outer_fold(
                context,
                records,
                teacher_vault,
                family_ids=family_ids,
                outer_family_id=family,
                panel_receipt=panel_receipt,
                authenticated_v20a_fold=authenticated_folds[family],
            )
            payload = _fold_payload(
                live,
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                bridge_binding_sha256=bridge_binding,
                outer_family_id=family,
            )
            try:
                _publish_fold_fragment(
                    payload, output=destination, outer_family_id=family
                )
            except FileExistsError:
                pass
            fragments[family] = _load_fold_fragment(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                outer_family_id=family,
                bridge_binding_sha256=bridge_binding,
                authenticated_v20a_fold=authenticated_folds[family],
            )

        fold_receipts = tuple(
            _mapping(fragments[family]["fold_receipt"], label="V20g fold")
            for family in family_ids
        )
        qualification = _core.build_soft_polarity_oof_qualification(
            fold_receipts=fold_receipts
        )
        final: dict[str, object] | None = None
        if qualification.get("full_refit_authorized") is True:
            if _final_path(destination).exists():
                final = _load_final_fragment(
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    bridge_binding_sha256=bridge_binding,
                    oof_qualification=qualification,
                )
            else:
                final_live = _execute_final_refit(
                    context,
                    records,
                    teacher_vault,
                    family_ids=family_ids,
                    panel_receipt=panel_receipt,
                    oof_qualification=qualification,
                )
                payload = _final_payload(
                    final_live,
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    bridge_binding_sha256=bridge_binding,
                    oof_qualification=qualification,
                )
                try:
                    _publish_final_fragment(payload, output=destination)
                except FileExistsError:
                    pass
                final = _load_final_fragment(
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    bridge_binding_sha256=bridge_binding,
                    oof_qualification=qualification,
                )
        elif _final_path(destination).exists():
            raise ValueError("V20g failed OOF campaign has a final-refit fragment")
        report = _build_report(
            output=destination,
            source=source,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            fold_fragments=fragments,
            oof_qualification=qualification,
            final_fragment=final,
        )
    finally:
        context.validate_immutable_inputs()
        context.close()

    try:
        _v20b._publish_scalar_fragment(
            report,
            path=destination,
            domain=_REPORT_DOMAIN,
            hash_key="report_sha256",
            label="V20g report",
        )
    except FileExistsError:
        pass
    return _load_existing_report(
        destination,
        source=source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_folds=authenticated_folds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the adaptive A16 V20g box-normalized soft-polarity trust-region screen"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development(
        output=arguments.output,
        cache_dir=arguments.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
