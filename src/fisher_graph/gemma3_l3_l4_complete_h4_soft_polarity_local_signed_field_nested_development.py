"""V20p nested local signed-field campaign for Gemma 3.

V20p is the predeclared follow-up to the completed V20o development failure.
It retains the authenticated V20m response and reflection protocol but replaces
one prompt-static signed scalar by a continuous token-local field::

    s(c) = clamp(b + a * psi(c), -1, 1)

The objective-free library contains 24 adaptive fields over ``c1``, ``c2``,
``c1*c2``, and the frozen source projection, plus exact ``-1``, ``0``, and
``+1`` anchors.  In each outer fold the V20m 7x19 response screen is replayed,
then all 7x27 field providers and traces are frozen before any field capability
is opened.  Selection uses family-equal exact float64 full-vocabulary
``KL(teacher || candidate)`` on the seven inner-held families.  The outer
family is inaccessible until the field is fixed and all nine outer arms and
traces are frozen.

This is historically reused A16 development evidence.  It grants no fresh
validation, Calibration-B, compression, fidelity, speed, or serving claim.
Fold fragments are atomic mode-0600 scalar/hash artifacts and complete runs
replay without constructing Gemma.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as _v20b
from . import gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development as _v20g
from . import gemma3_l3_l4_complete_h4_soft_polarity_reflection_nested_development as _v20i
from . import gemma3_l3_l4_complete_h4_soft_polarity_signed_stack_nested_development as _v20l
from . import gemma3_l3_l4_complete_h4_soft_polarity_simplex_response_nested_development as _v20m
from . import gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development as _v20o
from . import complete_h4_fisher_soft_polarity_local_signed_field_fit as _field_fit
from . import complete_h4_fisher_soft_polarity_reflection_fit as _reflection
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .complete_h4_fisher_soft_polarity_local_signed_field import (
    AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider,
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
    build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field,
    fisher_soft_polarity_local_signed_field_direction_sha256,
    fisher_soft_polarity_local_signed_field_feature,
    fisher_soft_polarity_local_signed_field_projection,
    fisher_soft_polarity_local_signed_field_signed_scalar,
    validate_fisher_soft_polarity_local_signed_field_provider_evidence,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_local_signed_field_nested_development",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-local-signed-field-"
    "nested-r16-k256-a-fit16-dev-v20p.json"
)

_V20O_OUTPUT = _v20o.DEFAULT_OUTPUT
_V20O_LOGICAL_SHA256 = "d9d050db5135535ebab39a1568f16e4e1e1ca7d0676fd781db8ad657c4b8e83d"
_V20O_FILE_SHA256 = "7b2a65c528d115824addf718b8daa5eba4bfa033baaa6a88688ae20c64b3fc17"
_V20O_SOURCE_SHA256 = "758410440278eba087898a2e8ed76d465c68c81781f68f00367b6991554d9bd5"
_V20O_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": "ca533b6b6f64ea04e4d29d6e9d8a11300637130b84730e88aab0e8d456a37046",
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": "691feb9cb023e6f16994fc2a361bde80815c0cb6f3559384142969fae7aea1bb",
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": "6f26662d2fbff6ff27c5f7631a639818c80c24124644ef89228b12ba55e36a4d",
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": "ee4aa7d87a72f0d9d13c1957234dd913501b5fa4f3f3a2880754a8be41b1d9e7",
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": "ba5d11b7c53393945f5ce18e61768c8d650742c3256610e96b352161d78b0ad8",
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": "cb2dfb2dd7d396e8f8214b7e414fba2d941a8de1c77d08b465c5aa92519f6e82",
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": "14f47ae41f02940d8f54e35e479321aa7f66d184df5955cf65f3f691e5cf3474",
    "structured-strong-v9-calibration_a-varve-lamination-v9": "399086b76445085e9591ac600c38031736a3b548103796f8d264f48eab6321ef",
}

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_local_signed_field_nested.v20p"
_FOLD_SCHEMA = "fisher_graph.complete_h4_soft_polarity_local_signed_field_nested_outer_fold.v20p"
_FORMAT_VERSION = 32
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-nested-report:v20p\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-nested-source:v20p\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-nested-fold:v20p\0"
_INNER_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-inner:v20p\0"
_MANIFEST_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-manifest:v20p\0"
_EXECUTION_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-execution:v20p\0"
_PROVIDER_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-provider:v20p\0"
_TRACE_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-trace:v20p\0"
_DECISION_DOMAIN = b"fisher-graph:soft-polarity-local-signed-field-decision:v20p\0"

_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_PROMPTS_PER_FAMILY = 2
_FIELD_CANDIDATE_COUNT = 27
_INNER_PROVIDER_COUNT_PER_OUTER = 322
_FIELD_LIBRARY = tuple(_field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_LIBRARY)
_FIELD_CANDIDATE_IDS = tuple(
    _field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_CANDIDATE_IDS
)
_FIELD_LADDER_RECEIPT = _field_fit.build_soft_polarity_local_signed_field_ladder_receipt()
_FIELD_LADDER_SHA256 = str(_FIELD_LADDER_RECEIPT["artifact_sha256"])
if len(_FIELD_LIBRARY) != _FIELD_CANDIDATE_COUNT:
    raise RuntimeError("V20p field library must contain exactly 27 candidates")

_ARMS = (
    "base",
    "fixed_plus",
    "fixed_minus",
    "matched_linear_reflected",
    "matched_v20l_boundary_reflected",
    "same_simplex_response_unreflected",
    "local_signed_field_reflected",
    "simplex_response_reflected_exact_mirror",
    "matched_v20m_simplex_reflected",
)
_PRIMARY_ARM = "local_signed_field_reflected"

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20p_nested_local_signed_field",
    "scientific_status": "historically_reused_A16_development_after_completed_v20o_failure",
    "fresh_validation_claim": False,
    "v20o_motivated_hypothesis": "global_signed_scalar_collapsed_to_exact_v20m_endpoint_in_all_eight_outer_folds",
    "field_formula": "s(c)=clamp(b+a*psi(c),-1,1)",
    "field_library_receipt_sha256": _FIELD_LADDER_SHA256,
    "field_candidate_count": _FIELD_CANDIDATE_COUNT,
    "response_selection": "reproduce_V20m_seven_times_nineteen_inner_response_screen_per_outer_fold",
    "inner_freeze_barrier": "freeze_all_seven_times_twenty_seven_field_providers_and_traces_before_any_field_capability",
    "inner_objective": "family_equal_token_mean_exact_float64_full_vocabulary_KL_teacher_to_candidate",
    "outer_validation": "eight_leave_one_whole_development_family_out_folds",
    "outer_freeze_barrier": "freeze_all_nine_providers_and_traces_before_outer_capability",
    "outer_arms": _ARMS,
    "primary_gate": "candidate_macro_below_base_and_fixed_plus_with_at_least_six_of_eight_strict_wins_each",
    "mechanism_gate": "candidate_macro_below_unreflected_mirror_linear_v20l_and_v20m_with_at_least_five_of_eight_strict_wins_each",
    "field_evidence_gate": "adaptive_at_least_six_nonconstant_at_least_six_at_least_one_held_field_crosses_zero_and_candidate_distinct_from_each_anchor_at_least_six",
    "endpoint_identity_gate": "all_inner_minus_one_zero_plus_one_exact_outputs_replay_against_authenticated_V20o_and_V20m_anchors",
    "expected_work": {
        "model_forwards": 5440,
        "teacher_accesses": 5408,
        "suffix_backwards": 128,
        "local_autograd_contractions": 112,
        "inner_providers": 2576,
        "inner_providers_per_outer": 322,
        "outer_providers": 72,
        "final_refit": 0,
    },
    "failure_policy": "rollback_to_base_no_claims_no_B",
    "calibration_b_eligible": False,
    "compression_claim_authorized": False,
    "fidelity_claim_authorized": False,
    "speed_claim_authorized": False,
    "serving_authorized": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(_FIXED_PROTOCOL, domain=_SOURCE_DOMAIN)
_TRANSFER_PROTOCOL_SHA256 = _v14._sha256(
    {
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "field_fit_protocol_sha256": _field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
        "field_provider_protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "operation": "V20p_domain_separated_local_signed_field_materialization",
        "held_rows_used": False,
    },
    domain=_PROVIDER_DOMAIN,
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _has_exact_arm_keys(*values: Mapping[str, object]) -> bool:
    """Check canonical JSON arm maps without relying on object key order."""

    expected = set(_ARMS)
    return all(set(value) == expected for value in values)


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty canonical identifier")
    return value


def _sha(value: object, *, label: str) -> str:
    return _v20o._sha(value, label=label)


def _hashed(payload: Mapping[str, object], *, domain: bytes) -> dict[str, object]:
    result = dict(payload)
    result["artifact_sha256"] = _v14._sha256(result, domain=domain)
    return result


def _validate_hashed(value: Mapping[str, object], *, domain: bytes, label: str) -> None:
    supplied = dict(_mapping(value, label=label))
    artifact = _sha(supplied.pop("artifact_sha256", None), label=f"{label} artifact")
    if artifact != _v14._sha256(supplied, domain=domain):
        raise ValueError(f"{label} artifact hash drifted")


def _validate_output(path: Path | str) -> Path:
    destination = _v20o._validate_output(path)
    protected = {
        _V20O_OUTPUT.resolve(strict=False),
        *(
            _v20o._fold_path(_V20O_OUTPUT, family).resolve(strict=False)
            for family in sorted(_V20O_FOLD_SHA256S)
        ),
    }
    if destination in protected:
        raise ValueError("V20p output must preserve immutable V20o authority")
    return destination


def _fold_path(output: Path | str, family_id: str) -> Path:
    return _v20o._fold_path(output, family_id)


@dataclass(slots=True)
class _Authorities:
    prerequisite: dict[str, object]
    authenticated_v20a_folds: dict[str, dict[str, object]]
    v20g_report: dict[str, object]
    authenticated_v20g_folds: dict[str, dict[str, object]]
    v20i_report: dict[str, object]
    authenticated_v20i_folds: dict[str, dict[str, object]]
    authenticated_v20l_folds: dict[str, dict[str, object]]
    v20m_report: dict[str, object]
    authenticated_v20m_folds: dict[str, dict[str, object]]
    v20o_report: dict[str, object]
    authenticated_v20o_folds: dict[str, dict[str, object]]
    source: dict[str, object]


def _load_prerequisites() -> _Authorities:
    """Authenticate the complete V20o result and lineage before Gemma exists."""

    if not all((_V20O_LOGICAL_SHA256, _V20O_FILE_SHA256, _V20O_SOURCE_SHA256)):
        raise RuntimeError("V20p is fail-closed until all V20o authority pins exist")
    (
        prerequisite,
        authenticated_v20a_folds,
        v20g_report,
        authenticated_v20g_folds,
        v20i_report,
        authenticated_v20i_folds,
        _v20l_report,
        authenticated_v20l_folds,
        v20m_report,
        authenticated_v20m_folds,
        v20n_report,
        authenticated_v20n_folds,
        v20o_parent_source,
    ) = _v20o._load_prerequisites()
    panel = dict(_mapping(prerequisite.get("nested_panel_receipt"), label="V20p panel"))
    bridge = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20p bridge binding",
    )
    if _v14._file_sha256(_V20O_OUTPUT) != _V20O_FILE_SHA256:
        raise RuntimeError("pinned V20o report file hash drifted")
    v20o_report = _v20o._load_existing_report(
        _V20O_OUTPUT,
        source=v20o_parent_source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        v20m_report=v20m_report,
        v20n_report=v20n_report,
        panel_receipt=panel,
        bridge_binding_sha256=bridge,
        authenticated_v20g_folds=authenticated_v20g_folds,
        authenticated_v20i_folds=authenticated_v20i_folds,
        authenticated_v20l_folds=authenticated_v20l_folds,
        authenticated_v20m_folds=authenticated_v20m_folds,
        authenticated_v20n_folds=authenticated_v20n_folds,
    )
    observed = {
        _identifier(key, label="V20p V20o fold family"): _sha(
            item, label="V20p V20o fold hash"
        )
        for key, item in _mapping(
            v20o_report.get("fold_fragment_sha256s_by_family"),
            label="V20p V20o fold hashes",
        ).items()
    }
    decision = _mapping(v20o_report.get("decision"), label="V20p V20o decision")
    if (
        v20o_report.get("report_sha256") != _V20O_LOGICAL_SHA256
        or _mapping(v20o_report.get("source_receipt"), label="V20p V20o source").get("artifact_sha256") != _V20O_SOURCE_SHA256
        or observed != _V20O_FOLD_SHA256S
        or v20o_report.get("all_eight_outer_folds_completed") is not True
        or decision.get("integrity_passed") is not True
        or decision.get("selected_interior_signed_scalar_count") != 0
        or set(_mapping(decision.get("selected_signed_scalar_by_family"), label="V20p V20o selections").values()) != {1.0}
        or v20o_report.get("final_refit") is not None
        or v20o_report.get("calibration_b_opened") is not False
    ):
        raise RuntimeError("pinned completed V20o authority differs")
    families = tuple(sorted(_V20O_FOLD_SHA256S))
    authenticated_v20o_folds = {
        family: _v20o._load_fold_fragment(
            output=_V20O_OUTPUT,
            source=v20o_parent_source,
            panel_receipt=panel,
            outer_family_id=family,
            bridge_binding_sha256=bridge,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
            authenticated_v20i_fold=authenticated_v20i_folds[family],
            authenticated_v20l_fold=authenticated_v20l_folds[family],
            authenticated_v20m_fold=authenticated_v20m_folds[family],
            authenticated_v20n_fold=authenticated_v20n_folds[family],
        )
        for family in families
    }
    if {
        family: fold["fragment_sha256"]
        for family, fold in authenticated_v20o_folds.items()
    } != _V20O_FOLD_SHA256S:
        raise RuntimeError("pinned V20o fold authority differs")
    inherited = {
        key: value
        for key, value in v20o_parent_source.items()
        if key != "artifact_sha256"
    }
    source = _hashed(
        {
            **inherited,
            "v20o_parent_source_receipt_sha256": v20o_parent_source["artifact_sha256"],
            "v20o_report_sha256": _V20O_LOGICAL_SHA256,
            "v20o_file_sha256": _V20O_FILE_SHA256,
            "v20o_source_receipt_sha256": _V20O_SOURCE_SHA256,
            "v20o_fold_fragment_sha256s_by_family": dict(sorted(_V20O_FOLD_SHA256S.items())),
            "v20o_classification": v20o_report.get("classification"),
            "v20o_selected_signed_scalar_by_family": dict(decision["selected_signed_scalar_by_family"]),
            "local_signed_field_fit_protocol_sha256": _field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
            "local_signed_field_provider_protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
            "local_signed_field_ladder_receipt_sha256": _FIELD_LADDER_SHA256,
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "authenticated_before_model_construction": True,
            "historically_reused_A16_only": True,
            "fresh_validation_claim": False,
            "held_scores_used_before_response_or_field_freeze": False,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
        },
        domain=_SOURCE_DOMAIN,
    )
    return _Authorities(
        dict(prerequisite),
        authenticated_v20a_folds,
        dict(v20g_report),
        authenticated_v20g_folds,
        dict(v20i_report),
        authenticated_v20i_folds,
        authenticated_v20l_folds,
        dict(v20m_report),
        authenticated_v20m_folds,
        dict(v20o_report),
        authenticated_v20o_folds,
        source,
    )


def _response_tuple(value: object) -> tuple[float, float, float]:
    return _v20o._response_tuple(value)


def _response_key(value: object) -> str:
    return _v20o._response_key(value)


def _selected_direction(receipt: Mapping[str, object]) -> tuple[float, float, float, float]:
    return _v20o._selected_direction(receipt)


def _field_spec(index: int) -> tuple[str, float, float]:
    if type(index) is not int or not 0 <= index < len(_FIELD_LIBRARY):
        raise ValueError("V20p field candidate index is outside the frozen library")
    feature, bias, slope = _FIELD_LIBRARY[index]
    return str(feature), float(bias), float(slope)


def _provider_feature_id(feature_id: str) -> int:
    try:
        return tuple(_field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_IDS).index(
            feature_id
        )
    except ValueError as error:
        raise ValueError("V20p field feature differs from the frozen fitter") from error


def _field_transfer_seed(
    *,
    endpoint_receipt_sha256: str,
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    candidate_id: str,
    feature_id: str,
    field_bias: float,
    field_slope: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> str:
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt_sha256, label="V20p field endpoint receipt"
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256, label="V20p field direction"
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256, label="V20p reflection fit"
            ),
            "response": response,
            "candidate_id": candidate_id,
            "feature_id": feature_id,
            "field_bias": field_bias,
            "field_bias_hex": field_bias.hex(),
            "field_slope": field_slope,
            "field_slope_hex": field_slope.hex(),
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "role": role,
            "held_rows_used": False,
        },
        domain=_PROVIDER_DOMAIN,
    )


def _field_seed(
    endpoint: _v20g._EndpointLive,
    *,
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    candidate_id: str,
    feature_id: str,
    field_bias: float,
    field_slope: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> str:
    return _field_transfer_seed(
        endpoint_receipt_sha256=str(endpoint.receipt["artifact_sha256"]),
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        candidate_id=candidate_id,
        feature_id=feature_id,
        field_bias=field_bias,
        field_slope=field_slope,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )


def _inner_execution_seed(
    *,
    manifest_sha256: str,
    outer_family_id: str,
    inner_family_id: str,
    candidate_id: str,
    provider_artifact_sha256: str,
) -> str:
    return _v14._sha256(
        {
            "manifest_sha256": _sha(
                manifest_sha256, label="V20p inner execution manifest"
            ),
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "candidate_id": candidate_id,
            "provider_artifact_sha256": _sha(
                provider_artifact_sha256,
                label="V20p inner execution provider",
            ),
            "all_189_field_providers_frozen": True,
        },
        domain=_EXECUTION_DOMAIN,
    )


def _materialize_field_provider(
    endpoint: _v20g._EndpointLive,
    *,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    candidate_id: str,
    feature_id: str,
    field_bias: float,
    field_slope: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> tuple[AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider, str]:
    seed = _field_seed(
        endpoint,
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        candidate_id=candidate_id,
        feature_id=feature_id,
        field_bias=field_bias,
        field_slope=field_slope,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )
    radius, shrink_mass, polarity_bias = response
    provider = build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field(
        endpoint.base_provider,
        endpoint.proposal_provider,
        direction=torch.tensor(tuple(direction), dtype=torch.float64),
        radius=radius,
        shrink_mass=shrink_mass,
        polarity_bias=polarity_bias,
        field_bias=field_bias,
        field_slope=field_slope,
        feature_id=_provider_feature_id(feature_id),
        transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=seed,
    )
    return provider, seed


def _field_provider_receipt(
    provider: AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider,
    *,
    role: str,
    candidate_id: str,
    candidate_index: int,
    feature_id: str,
    field_bias: float,
    field_slope: float,
    response: tuple[float, float, float],
) -> dict[str, object]:
    payload = provider.artifact_payload()
    metadata = provider.metadata()
    validated = validate_fisher_soft_polarity_local_signed_field_provider_evidence(
        payload, metadata
    )
    if validated.artifact_sha256 != provider.artifact_sha256:
        raise RuntimeError("V20p provider evidence differs from the live provider")
    runtime = provider.runtime_provider
    if (
        runtime.field_bias != field_bias
        or runtime.field_slope != field_slope
        or runtime.feature_id != _provider_feature_id(feature_id)
        or (runtime.radius, runtime.shrink_mass, runtime.polarity_bias) != response
    ):
        raise RuntimeError("V20p provider differs from its frozen candidate")
    return _hashed(
        {
            "role": role,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "feature_id": feature_id,
            "field_bias": field_bias,
            "field_bias_hex": field_bias.hex(),
            "field_slope": field_slope,
            "field_slope_hex": field_slope.hex(),
            "source_response": response,
            "provider_artifact_sha256": provider.artifact_sha256,
            "runtime_provider_artifact_sha256": runtime.artifact_sha256,
            "provider_payload": payload,
            "provider_metadata": metadata,
            "provider_metadata_sha256": _v14._sha256(metadata, domain=_PROVIDER_DOMAIN),
            "rank": int(provider.rank),
            "conditional_rank": int(provider.conditional_rank),
            "prepared_float_scalar_count": int(provider.prepared_float_scalar_count),
            "logical_macs_per_token_upper_bound": int(
                provider.logical_macs_per_token_upper_bound
            ),
            "lineage_wrapper_not_inference_executor": True,
            "raw_provider_tensors_serialized": False,
            "analysis_only": True,
        },
        domain=_PROVIDER_DOMAIN,
    )


def _validate_field_provider_receipt(
    value: Mapping[str, object], *, expected_role: str | None = None
) -> None:
    receipt = _mapping(value, label="V20p field provider receipt")
    _validate_hashed(receipt, domain=_PROVIDER_DOMAIN, label="V20p field provider receipt")
    index = receipt.get("candidate_index")
    if type(index) is not int:
        raise ValueError("V20p provider candidate index must be an integer")
    feature_id, bias, slope = _field_spec(index)
    candidate_id = _FIELD_CANDIDATE_IDS[index]
    response = _response_tuple(receipt.get("source_response"))
    if (
        (expected_role is not None and receipt.get("role") != expected_role)
        or receipt.get("candidate_id") != candidate_id
        or receipt.get("feature_id") != feature_id
        or type(receipt.get("field_bias")) is not float
        or receipt.get("field_bias") != bias
        or receipt.get("field_bias_hex") != bias.hex()
        or type(receipt.get("field_slope")) is not float
        or receipt.get("field_slope") != slope
        or receipt.get("field_slope_hex") != slope.hex()
        or receipt.get("analysis_only") is not True
        or receipt.get("raw_provider_tensors_serialized") is not False
    ):
        raise ValueError("V20p provider candidate semantics differ")
    payload = _mapping(receipt.get("provider_payload"), label="V20p provider payload")
    metadata = _mapping(receipt.get("provider_metadata"), label="V20p provider metadata")
    validated = validate_fisher_soft_polarity_local_signed_field_provider_evidence(
        payload, metadata
    )
    runtime_payload = _mapping(
        payload.get("compiled_runtime_provider_payload"),
        label="V20p runtime provider payload",
    )
    if (
        validated.artifact_sha256 != receipt.get("provider_artifact_sha256")
        or payload.get("compiled_runtime_provider_artifact_sha256")
        != receipt.get("runtime_provider_artifact_sha256")
        or _v14._sha256(metadata, domain=_PROVIDER_DOMAIN)
        != receipt.get("provider_metadata_sha256")
        or receipt.get("raw_provider_tensors_serialized") is not False
        or receipt.get("lineage_wrapper_not_inference_executor") is not True
        or runtime_payload.get("feature_name") != feature_id
        or runtime_payload.get("feature_id") != _provider_feature_id(feature_id)
        or runtime_payload.get("field_bias") != bias
        or runtime_payload.get("field_bias_hex") != bias.hex()
        or runtime_payload.get("field_slope") != slope
        or runtime_payload.get("field_slope_hex") != slope.hex()
        or (
            runtime_payload.get("radius"),
            runtime_payload.get("shrink_mass"),
            runtime_payload.get("polarity_bias"),
        )
        != response
        or receipt.get("rank") != metadata.get("rank")
        or receipt.get("conditional_rank") != metadata.get("conditional_rank")
        or receipt.get("prepared_float_scalar_count")
        != metadata.get("prepared_float_scalar_count")
        or receipt.get("logical_macs_per_token_upper_bound")
        != metadata.get("logical_macs_per_token_upper_bound")
    ):
        raise ValueError("V20p provider receipt binding differs")


def _field_trace(
    provider: AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider,
    records: Sequence[object],
    *,
    role: str,
) -> dict[str, object]:
    base = _v20o._provider_trace(provider, records, role=role)
    scalar_hashes: dict[str, str] = {}
    values: list[Tensor] = []
    runtime = provider.runtime_provider
    for record in _v20b._ordered_records(records):
        sequence = record.sequence
        parent = _training_parent_modal(provider.parent_provider, sequence)
        coordinates = provider.bounded_coordinates(parent)
        flat = coordinates.reshape(-1, 2).to(dtype=torch.float64)
        projection = fisher_soft_polarity_local_signed_field_projection(
            flat, runtime.direction.to(device=flat.device, dtype=torch.float64)
        )
        feature = fisher_soft_polarity_local_signed_field_feature(
            flat, projection, runtime.feature_id
        )
        scalar = fisher_soft_polarity_local_signed_field_signed_scalar(
            feature, runtime.field_bias, runtime.field_slope
        ).reshape(coordinates.shape[:-1])
        support = sequence.support_mask.to(scalar.device)
        selected = scalar[support].detach().to(device="cpu", dtype=torch.float64)
        if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
            raise RuntimeError("V20p local scalar trace is empty or nonfinite")
        scalar_hashes[sequence.example_id] = _v14._tensor_sha256(selected)
        values.append(selected.reshape(-1))
    joined = torch.cat(values)
    payload = dict(base)
    payload.pop("artifact_sha256", None)
    payload.update(
        {
            "local_signed_scalar_sha256s": dict(sorted(scalar_hashes.items())),
            "local_signed_scalar_min": float(joined.min()),
            "local_signed_scalar_max": float(joined.max()),
            "local_signed_scalar_distinct_count": int(torch.unique(joined).numel()),
            "local_signed_scalar_nonconstant": bool(float(joined.max()) > float(joined.min())),
            "local_signed_scalar_has_negative": bool(float(joined.min()) < 0.0),
            "local_signed_scalar_has_positive": bool(float(joined.max()) > 0.0),
            "local_signed_scalar_inside_closed_unit_interval": bool(
                float(joined.min()) >= -1.0 and float(joined.max()) <= 1.0
            ),
            "raw_local_scalar_tensors_serialized": False,
        }
    )
    return _hashed(payload, domain=_TRACE_DOMAIN)


def _validate_field_trace_semantics(value: Mapping[str, object]) -> None:
    trace = _mapping(value, label="V20p local field trace")
    lower = trace.get("local_signed_scalar_min")
    upper = trace.get("local_signed_scalar_max")
    distinct = trace.get("local_signed_scalar_distinct_count")
    if (
        type(lower) is not float
        or type(upper) is not float
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not -1.0 <= lower <= upper <= 1.0
        or type(distinct) is not int
        or distinct < 1
    ):
        raise ValueError("V20p local field trace scalar statistics differ")
    nonconstant = upper > lower
    if (
        trace.get("local_signed_scalar_inside_closed_unit_interval") is not True
        or trace.get("local_signed_scalar_nonconstant") is not nonconstant
        or trace.get("local_signed_scalar_has_negative")
        is not (lower < 0.0)
        or trace.get("local_signed_scalar_has_positive")
        is not (upper > 0.0)
        or (distinct == 1) is not (not nonconstant)
        or (nonconstant and distinct < 2)
        or trace.get("raw_local_scalar_tensors_serialized") is not False
    ):
        raise ValueError("V20p local field trace derived flags differ")
    hashes = _mapping(
        trace.get("local_signed_scalar_sha256s"),
        label="V20p local field scalar hashes",
    )
    if not hashes:
        raise ValueError("V20p local field trace has no example commitments")
    for digest in hashes.values():
        _sha(digest, label="V20p local field scalar hash")


def _runtime_provider(provider: object) -> object:
    if isinstance(provider, AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider):
        return provider.runtime_provider
    runtime = getattr(provider, "runtime_provider", None)
    return runtime if runtime is not None else provider


def _score_provider(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    phase: str,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
    evidence_sha256: str,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    return _v20o._score_exact_provider(
        context,
        records,
        capability,
        provider=_runtime_provider(provider),
        phase=phase,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
        evidence_sha256=evidence_sha256,
        domain=_EXECUTION_DOMAIN,
    )


def _outer_execution_seed(
    *,
    manifest_sha256: str,
    outer_family_id: str,
    arm: str,
    provider_artifact_sha256: str,
) -> str:
    if arm not in _ARMS:
        raise ValueError("V20p outer execution seed arm differs")
    return _v14._sha256(
        {
            "manifest_sha256": _sha(
                manifest_sha256, label="V20p outer seed manifest"
            ),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20p outer seed family"
            ),
            "arm": arm,
            "provider_artifact_sha256": _sha(
                provider_artifact_sha256,
                label="V20p outer seed provider",
            ),
            "all_nine_outer_providers_frozen": True,
        },
        domain=_EXECUTION_DOMAIN,
    )


def _anchor_candidate_id(value: float) -> str:
    expected = {24: -1.0, 25: 0.0, 26: 1.0}
    for index, anchor in expected.items():
        if value == anchor and _field_spec(index) == ("source_z", anchor, 0.0):
            return _FIELD_CANDIDATE_IDS[index]
    raise ValueError("V20p exact anchor differs from the frozen field library")


def _exact_bundle_equal(
    objectives: Mapping[str, object],
    h4_hashes: Mapping[str, object],
    logits_hashes: Mapping[str, object],
    authority: Mapping[str, object],
) -> bool:
    return (
        dict(objectives)
        == dict(_mapping(authority.get("objectives_by_example"), label="anchor objectives"))
        and dict(h4_hashes)
        == dict(_mapping(authority.get("post_cast_h4_sha256s"), label="anchor H4 hashes"))
        and dict(logits_hashes)
        == dict(
            _mapping(
                authority.get("supervised_full_vocab_logits_sha256s"),
                label="anchor logits hashes",
            )
        )
    )


def _fit_inner_local_field(
    context: object,
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Reproduce V20m, freeze 7x27 fields, exact-score, then select."""

    outer = _identifier(outer_family_id, label="V20p inner outer family")
    v20m_inner, response_selection = _v20o._fit_inner_response(
        context,
        endpoint,
        source_direction_receipt,
        teacher_vault,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
    )
    inherited_selection = _mapping(
        authenticated_v20m_fold.get("response_selection_receipt"),
        label="V20p authenticated V20m response selection",
    )
    for key in (
        "objectives_by_inner_family_and_response",
        "family_equal_objective_by_response",
        "simplex_response_selection_receipt",
        "selected_response",
    ):
        if _v14._canonical_json_bytes(response_selection.get(key)) != _v14._canonical_json_bytes(
            inherited_selection.get(key)
        ):
            raise RuntimeError(f"V20p V20m response reproduction differs at {key}")
    response = _response_tuple(response_selection["selected_response"])
    response_key = _response_key(response)
    training = _v20b._ordered_records(endpoint.training_records)
    inner_evidence = _mapping(
        v20m_inner.get("inner_evidence_by_family"),
        label="V20p V20m inner evidence",
    )
    families = tuple(sorted(inner_evidence))
    if len(families) != _INNER_FAMILY_COUNT or outer in families:
        raise RuntimeError("V20p inner family geometry differs")

    providers: dict[str, dict[str, AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider]] = {}
    traces: dict[str, dict[str, dict[str, object]]] = {}
    receipts: dict[str, dict[str, dict[str, object]]] = {}
    seeds: dict[str, dict[str, str]] = {}
    held_by_family: dict[str, tuple[object, ...]] = {}
    directions: dict[str, tuple[float, float, float, float]] = {}
    reflection_fits: dict[str, Mapping[str, object]] = {}

    # Freeze every provider and trace before the first field capability exists.
    for family in families:
        family_evidence = _mapping(inner_evidence[family], label="V20p inner family")
        fit = _mapping(
            family_evidence.get("reflection_fit_receipt"),
            label="V20p inner reflection fit",
        )
        reflection_fits[family] = fit
        directions[family] = _selected_direction(fit)
        held = _v20b._ordered_records(
            tuple(
                record
                for record in training
                if record.sequence.family_id == family
            )
        )
        if len(held) != _PROMPTS_PER_FAMILY:
            raise RuntimeError("V20p inner prompt geometry differs")
        held_by_family[family] = held
        providers[family] = {}
        traces[family] = {}
        receipts[family] = {}
        seeds[family] = {}
        for index, candidate_id in enumerate(_FIELD_CANDIDATE_IDS):
            feature_id, bias, slope = _field_spec(index)
            provider, seed = _materialize_field_provider(
                endpoint,
                direction=directions[family],
                direction_artifact_sha256=_sha(
                    fit.get("selected_variant_artifact_sha256"),
                    label="V20p inner selected direction",
                ),
                reflection_fit_sha256=_sha(
                    fit.get("artifact_sha256"), label="V20p inner reflection fit"
                ),
                response=response,
                candidate_id=candidate_id,
                feature_id=feature_id,
                field_bias=bias,
                field_slope=slope,
                outer_family_id=outer,
                inner_family_id=family,
                role="inner_local_signed_field_candidate",
            )
            providers[family][candidate_id] = provider
            seeds[family][candidate_id] = seed
            traces[family][candidate_id] = _field_trace(
                provider,
                held,
                role=f"inner_{family}_{candidate_id}",
            )
            receipts[family][candidate_id] = _field_provider_receipt(
                provider,
                role="inner_local_signed_field_candidate",
                candidate_id=candidate_id,
                candidate_index=index,
                feature_id=feature_id,
                field_bias=bias,
                field_slope=slope,
                response=response,
            )
    artifacts = tuple(
        providers[family][candidate_id].artifact_sha256
        for family in families
        for candidate_id in _FIELD_CANDIDATE_IDS
    )
    if len(artifacts) != 7 * 27 or len(set(artifacts)) != len(artifacts):
        raise RuntimeError("V20p 7x27 provider artifacts are not globally distinct")
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "inner_family_order": families,
            "source_response": response,
            "field_ladder_receipt": _FIELD_LADDER_RECEIPT,
            "field_ladder_receipt_sha256": _FIELD_LADDER_SHA256,
            "candidate_order": _FIELD_CANDIDATE_IDS,
            "provider_artifact_sha256s_by_inner_family_and_candidate": {
                family: {
                    candidate_id: providers[family][candidate_id].artifact_sha256
                    for candidate_id in _FIELD_CANDIDATE_IDS
                }
                for family in families
            },
            "runtime_provider_artifact_sha256s_by_inner_family_and_candidate": {
                family: {
                    candidate_id: providers[family][candidate_id].runtime_provider.artifact_sha256
                    for candidate_id in _FIELD_CANDIDATE_IDS
                }
                for family in families
            },
            "provider_transfer_evidence_sha256s_by_inner_family_and_candidate": seeds,
            "provider_receipts_by_inner_family_and_candidate": receipts,
            "trace_sha256s_by_inner_family_and_candidate": {
                family: {
                    candidate_id: traces[family][candidate_id]["artifact_sha256"]
                    for candidate_id in _FIELD_CANDIDATE_IDS
                }
                for family in families
            },
            "all_seven_times_twenty_seven_providers_frozen_before_any_field_capability": True,
            "all_seven_times_twenty_seven_traces_frozen_before_any_field_capability": True,
            "field_capability_count_at_freeze": 0,
            "field_objectives_or_teacher_rows_used_at_freeze": False,
            "outer_held_family_used": False,
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized": False,
        },
        domain=_MANIFEST_DOMAIN,
    )

    objective_rows: dict[str, dict[str, float]] = {}
    scored_evidence: dict[str, dict[str, dict[str, object]]] = {}
    endpoint_anchors: dict[str, dict[str, bool]] = {}
    v20o_signed = _mapping(
        authenticated_v20o_fold.get("signed_continuum_selection_receipt"),
        label="V20p authenticated V20o signed selection",
    )
    v20o_missing = _mapping(
        v20o_signed.get("missing_anchor_evidence_by_family_and_anchor"),
        label="V20p authenticated V20o missing anchors",
    )
    for family in families:
        held = held_by_family[family]
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        objective_rows[family] = {}
        scored_evidence[family] = {}
        exact_bundles: dict[str, tuple[dict[str, float], dict[str, str], dict[str, str]]] = {}
        for candidate_id in _FIELD_CANDIDATE_IDS:
            provider = providers[family][candidate_id]
            seed = _inner_execution_seed(
                manifest_sha256=str(manifest["artifact_sha256"]),
                outer_family_id=outer,
                inner_family_id=family,
                candidate_id=candidate_id,
                provider_artifact_sha256=provider.artifact_sha256,
            )
            objectives, h4_hashes, logits_hashes, execution_hashes = _score_provider(
                context,
                held,
                capability,
                provider=provider,
                phase="inner_local_signed_field_exact_score",
                outer_family_id=outer,
                inner_family_id=family,
                role="inner_local_signed_field_candidate",
                evidence_sha256=seed,
            )
            macro, family_scores = _v19._family_equal_mean(objectives, held)
            if set(family_scores) != {family}:
                raise RuntimeError("V20p field score family geometry differs")
            objective_rows[family][candidate_id] = macro
            exact_bundles[candidate_id] = (objectives, h4_hashes, logits_hashes)
            scored_evidence[family][candidate_id] = _hashed(
                {
                    "outer_held_family_id": outer,
                    "inner_held_family_id": family,
                    "candidate_id": candidate_id,
                    "provider_artifact_sha256": provider.artifact_sha256,
                    "runtime_provider_artifact_sha256": provider.runtime_provider.artifact_sha256,
                    "manifest_sha256": manifest["artifact_sha256"],
                    "execution_seed_sha256": seed,
                    "response_trace": traces[family][candidate_id],
                    "objective": macro,
                    "objectives_by_example": dict(sorted(objectives.items())),
                    "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                    "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
                    "execution_sha256s": dict(sorted(execution_hashes.items())),
                    "all_field_candidates_frozen_before_score": True,
                    "outer_family_absent_from_fit_and_score": True,
                    "exact_float64_full_vocabulary_teacher_kl": True,
                    "raw_logits_h4_teacher_or_local_scalar_tensors_serialized": False,
                },
                domain=_EXECUTION_DOMAIN,
            )
        capability_receipt = capability.receipt()
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=tuple(record.sequence.example_id for record in held),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=_FIELD_CANDIDATE_COUNT,
            label="V20p inner field capability",
        )
        plus_authority = _mapping(
            _mapping(
                _mapping(inner_evidence[family], label="V20p inner response family").get("response_evidence"),
                label="V20p inner response evidence",
            ).get(response_key),
            label="V20p selected V20m response authority",
        )
        missing_family = _mapping(v20o_missing.get(family), label="V20p V20o anchor family")
        endpoint_anchors[family] = {}
        for anchor, authority in (
            (-1.0, _mapping(missing_family.get("signed_minus_one"), label="V20p minus anchor")),
            (0.0, _mapping(missing_family.get("signed_zero"), label="V20p zero anchor")),
            (1.0, plus_authority),
        ):
            candidate_id = _anchor_candidate_id(anchor)
            objectives, h4_hashes, logits_hashes = exact_bundles[candidate_id]
            endpoint_anchors[family][candidate_id] = _exact_bundle_equal(
                objectives, h4_hashes, logits_hashes, authority
            )
            if not endpoint_anchors[family][candidate_id]:
                raise RuntimeError("V20p local field endpoint output anchor differs")

    all_families = tuple(sorted((*families, outer)))
    fit_receipt = _field_fit.build_soft_polarity_local_signed_field_fit_receipt(
        ladder_receipt=_FIELD_LADDER_RECEIPT,
        all_development_family_ids=all_families,
        outer_held_family_id=outer,
        exact_objectives_by_family_and_candidate=objective_rows,
    )
    selected_index = int(fit_receipt["selected_candidate_index"])
    selected_feature, selected_bias, selected_slope = _field_spec(selected_index)
    if (
        fit_receipt.get("selected_candidate_id") != _FIELD_CANDIDATE_IDS[selected_index]
        or fit_receipt.get("selected_feature_id") != selected_feature
        or float(fit_receipt.get("selected_b")) != selected_bias
        or float(fit_receipt.get("selected_a")) != selected_slope
    ):
        raise RuntimeError("V20p selected field differs from the frozen library")
    selection = _hashed(
        {
            "outer_held_family_id": outer,
            "source_response": response,
            "v20m_response_selection_receipt_sha256": response_selection["artifact_sha256"],
            "field_ladder_receipt": _FIELD_LADDER_RECEIPT,
            "field_provider_manifest": manifest,
            "field_evidence_by_inner_family_and_candidate": scored_evidence,
            "exact_objectives_by_inner_family_and_candidate": objective_rows,
            "core_fit_receipt": fit_receipt,
            "selected_candidate_id": fit_receipt["selected_candidate_id"],
            "selected_candidate_index": selected_index,
            "selected_feature_id": selected_feature,
            "selected_b": selected_bias,
            "selected_a": selected_slope,
            "selected_adaptive": bool(fit_receipt["selected_adaptive"]),
            "endpoint_exact_output_anchor_by_inner_family_and_candidate": endpoint_anchors,
            "all_three_endpoint_identities_exact_on_all_seven_inner_families": all(
                all(row.values()) for row in endpoint_anchors.values()
            ),
            "all_189_field_providers_and_traces_frozen_before_any_field_capability": True,
            "exact_additional_inner_execution_count": 7 * 27 * 2,
            "outer_held_family_used_for_fit_or_selection": False,
            "final_refit_or_calibration_b_used": False,
            "raw_provider_prompt_token_logit_h4_teacher_or_local_scalar_tensors_serialized": False,
        },
        domain=_INNER_DOMAIN,
    )
    return v20m_inner, response_selection, selection


def _freeze_outer_providers(
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    held_records: Sequence[object],
    *,
    selected_response: tuple[float, float, float],
    selected_candidate_id: str,
    selected_candidate_index: int,
    selected_feature_id: str,
    selected_b: float,
    selected_a: float,
    outer_family_id: str,
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    """Freeze the exact V20o controls plus one V20p field candidate."""

    outer = _identifier(outer_family_id, label="V20p outer provider family")
    controls, v20o_manifest, control_traces = _v20o._freeze_outer_providers(
        endpoint,
        source_direction_receipt,
        outer_reflection_fit,
        held_records,
        selected_response=selected_response,
        selected_signed_scalar=1.0,
        outer_family_id=outer,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    reflected = _selected_direction(outer_reflection_fit)
    candidate, candidate_seed = _materialize_field_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=_sha(
            outer_reflection_fit.get("selected_variant_artifact_sha256"),
            label="V20p outer selected direction",
        ),
        reflection_fit_sha256=_sha(
            outer_reflection_fit.get("artifact_sha256"),
            label="V20p outer reflection fit",
        ),
        response=selected_response,
        candidate_id=selected_candidate_id,
        feature_id=selected_feature_id,
        field_bias=selected_b,
        field_slope=selected_a,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_local_signed_field_reflected",
    )
    providers: dict[str, object] = {
        "base": controls["base"],
        "fixed_plus": controls["fixed_plus"],
        "fixed_minus": controls["fixed_minus"],
        "matched_linear_reflected": controls["matched_linear_reflected"],
        "matched_v20l_boundary_reflected": controls[
            "matched_v20l_boundary_reflected"
        ],
        "same_simplex_response_unreflected": controls[
            "same_simplex_response_unreflected"
        ],
        "local_signed_field_reflected": candidate,
        "simplex_response_reflected_exact_mirror": controls[
            "simplex_response_reflected_exact_mirror"
        ],
        "matched_v20m_simplex_reflected": controls[
            "matched_v20m_simplex_reflected"
        ],
    }
    if tuple(providers) != _ARMS or len(
        {provider.artifact_sha256 for provider in providers.values()}
    ) != len(_ARMS):
        raise RuntimeError("V20p outer provider artifacts are not distinct")
    control_receipts = _mapping(
        v20o_manifest.get("provider_receipts"),
        label="V20p V20o control receipts",
    )
    candidate_receipt = _field_provider_receipt(
        candidate,
        role=_PRIMARY_ARM,
        candidate_id=selected_candidate_id,
        candidate_index=selected_candidate_index,
        feature_id=selected_feature_id,
        field_bias=selected_b,
        field_slope=selected_a,
        response=selected_response,
    )
    candidate_trace = _field_trace(
        candidate, held_records, role=_PRIMARY_ARM
    )
    receipts = {
        arm: (
            candidate_receipt
            if arm == _PRIMARY_ARM
            else dict(
                _mapping(control_receipts[arm], label=f"V20p {arm} receipt")
            )
        )
        for arm in _ARMS
    }
    traces = {
        arm: candidate_trace if arm == _PRIMARY_ARM else control_traces[arm]
        for arm in _ARMS
    }
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "source_direction_receipt_sha256": source_direction_receipt[
                "artifact_sha256"
            ],
            "outer_reflection_fit_receipt_sha256": outer_reflection_fit[
                "artifact_sha256"
            ],
            "selected_response": selected_response,
            "selected_candidate_id": selected_candidate_id,
            "selected_candidate_index": selected_candidate_index,
            "selected_feature_id": selected_feature_id,
            "selected_b": selected_b,
            "selected_a": selected_a,
            "arm_order": _ARMS,
            "provider_artifact_sha256s": {
                arm: providers[arm].artifact_sha256 for arm in _ARMS
            },
            "runtime_provider_artifact_sha256s": {
                arm: _runtime_provider(providers[arm]).artifact_sha256
                for arm in _ARMS
            },
            "provider_receipts": receipts,
            "response_trace_sha256s": {
                arm: traces[arm]["artifact_sha256"] for arm in _ARMS
            },
            "candidate_transfer_evidence_sha256": candidate_seed,
            "v20o_control_manifest_sha256": v20o_manifest["artifact_sha256"],
            "matched_v20l_source_fold_sha256": authenticated_v20l_fold[
                "fragment_sha256"
            ],
            "matched_v20m_source_fold_sha256": authenticated_v20m_fold[
                "fragment_sha256"
            ],
            "all_nine_providers_frozen_before_outer_capability": True,
            "all_nine_traces_frozen_before_outer_capability": True,
            "outer_capability_count_at_freeze": 0,
            "outer_objectives_or_teacher_rows_used_at_freeze": False,
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized": False,
        },
        domain=_MANIFEST_DOMAIN,
    )
    return providers, manifest, traces


def _score_outer_arms(
    context: object,
    endpoint: _v20g._EndpointLive,
    records: Sequence[object],
    teacher_vault: object,
    source_direction_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    *,
    selected_response: tuple[float, float, float],
    field_selection: Mapping[str, object],
    outer_family_id: str,
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20p outer score family")
    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20p outer prompt geometry differs")
    candidate_id = _identifier(
        field_selection.get("selected_candidate_id"),
        label="V20p selected candidate",
    )
    candidate_index = int(field_selection["selected_candidate_index"])
    feature_id, bias, slope = _field_spec(candidate_index)
    if (
        candidate_id != _FIELD_CANDIDATE_IDS[candidate_index]
        or field_selection.get("selected_feature_id") != feature_id
        or float(field_selection["selected_b"]) != bias
        or float(field_selection["selected_a"]) != slope
    ):
        raise RuntimeError("V20p outer field differs from the inner selection")
    providers, manifest, traces = _freeze_outer_providers(
        endpoint,
        source_direction_receipt,
        outer_reflection_fit,
        held,
        selected_response=selected_response,
        selected_candidate_id=candidate_id,
        selected_candidate_index=candidate_index,
        selected_feature_id=feature_id,
        selected_b=bias,
        selected_a=slope,
        outer_family_id=outer,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in held),
        held_family_id=None,
    )
    results: dict[
        str,
        tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]],
    ] = {}
    seeds: dict[str, str] = {}
    for arm in _ARMS:
        seeds[arm] = _outer_execution_seed(
            manifest_sha256=str(manifest["artifact_sha256"]),
            outer_family_id=outer,
            arm=arm,
            provider_artifact_sha256=providers[arm].artifact_sha256,
        )
        results[arm] = _score_provider(
            context,
            held,
            capability,
            provider=providers[arm],
            phase="outer_held_local_signed_field_score",
            outer_family_id=outer,
            inner_family_id=None,
            role=arm,
            evidence_sha256=seeds[arm],
        )
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(record.sequence.example_id for record in held),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20p outer capability",
    )
    objective_by_arm: dict[str, float] = {}
    arm_evidence: dict[str, dict[str, object]] = {}
    prior_arms = _mapping(
        _mapping(
            authenticated_v20o_fold.get("held_evidence"),
            label="V20p authenticated V20o held evidence",
        ).get("arm_evidence"),
        label="V20p authenticated V20o arms",
    )
    control_anchor_by_arm: dict[str, bool] = {}
    for arm in _ARMS:
        objectives, h4_hashes, logits_hashes, execution_hashes = results[arm]
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {outer}:
            raise RuntimeError("V20p outer score family geometry differs")
        objective_by_arm[arm] = macro
        if arm != _PRIMARY_ARM:
            control_anchor_by_arm[arm] = _exact_bundle_equal(
                objectives,
                h4_hashes,
                logits_hashes,
                _mapping(prior_arms.get(arm), label=f"V20p prior {arm}"),
            )
            if not control_anchor_by_arm[arm]:
                raise RuntimeError(f"V20p {arm} control output differs from V20o")
        arm_evidence[arm] = _hashed(
            {
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "runtime_provider_artifact_sha256": _runtime_provider(
                    providers[arm]
                ).artifact_sha256,
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "execution_seed_sha256": seeds[arm],
                "response_trace": traces[arm],
                "objective": macro,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(sorted(logits_hashes.items())),
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "lineage_wrapper_not_inference_executor": arm == _PRIMARY_ARM,
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_EXECUTION_DOMAIN,
        )
    candidate_result = results[_PRIMARY_ARM]
    exact_distinct_by_anchor = {
        "signed_minus_one": not (
            candidate_result[1]
            == results["simplex_response_reflected_exact_mirror"][1]
            and candidate_result[2]
            == results["simplex_response_reflected_exact_mirror"][2]
        ),
        "signed_zero": not (
            candidate_result[1] == results["fixed_plus"][1]
            and candidate_result[2] == results["fixed_plus"][2]
        ),
        "signed_plus_one": not (
            candidate_result[1] == results["matched_v20m_simplex_reflected"][1]
            and candidate_result[2] == results["matched_v20m_simplex_reflected"][2]
        ),
    }
    candidate_trace = traces[_PRIMARY_ARM]
    field_nonconstant = candidate_trace.get("local_signed_scalar_nonconstant") is True
    field_has_negative = candidate_trace.get("local_signed_scalar_has_negative") is True
    field_has_positive = candidate_trace.get("local_signed_scalar_has_positive") is True
    all_runtime_health = all(
        _mapping(arm_evidence[arm].get("response_trace"), label="V20p arm trace").get(
            "pointwise_trust_passed"
        )
        is True
        for arm in _ARMS
    )
    held_evidence = _hashed(
        {
            "outer_held_family_id": outer,
            "outer_manifest_sha256": manifest["artifact_sha256"],
            "arm_evidence": arm_evidence,
            "capability_receipt": capability_receipt,
            "control_exact_output_anchor_by_arm": control_anchor_by_arm,
            "all_eight_inherited_control_exact_output_anchors_passed": all(
                control_anchor_by_arm.values()
            ),
            "candidate_exact_output_distinct_by_anchor": exact_distinct_by_anchor,
            "candidate_field_nonconstant": field_nonconstant,
            "candidate_field_has_negative": field_has_negative,
            "candidate_field_has_positive": field_has_positive,
            "all_nine_providers_and_traces_frozen_before_outer_capability": True,
            "exact_outer_execution_count": len(_ARMS) * len(held),
            "outer_family_used_for_fit_or_selection": False,
            "raw_prompts_tokens_logits_h4_teacher_or_local_scalar_tensors_serialized": False,
        },
        domain=_EXECUTION_DOMAIN,
    )
    fold_receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "arm_order": _ARMS,
            "selected_response": selected_response,
            "selected_candidate_id": candidate_id,
            "selected_candidate_index": candidate_index,
            "selected_feature_id": feature_id,
            "selected_b": bias,
            "selected_a": slope,
            "selected_adaptive": bool(field_selection["selected_adaptive"]),
            "held_objective_by_arm": objective_by_arm,
            "candidate_field_nonconstant": field_nonconstant,
            "candidate_field_has_negative": field_has_negative,
            "candidate_field_has_positive": field_has_positive,
            "candidate_exact_output_distinct_by_anchor": exact_distinct_by_anchor,
            "all_inner_endpoint_exact_output_anchors_passed": field_selection.get(
                "all_three_endpoint_identities_exact_on_all_seven_inner_families"
            )
            is True,
            "all_inherited_control_exact_output_anchors_passed": all(
                control_anchor_by_arm.values()
            ),
            "all_runtime_health_passed": all_runtime_health,
            "selection_frozen_before_outer_score": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_execution": True,
        },
        domain=_DECISION_DOMAIN,
    )
    return manifest, held_evidence, fold_receipt


@dataclass(slots=True)
class _FoldLive:
    endpoint: _v20g._EndpointLive
    inner_receipt: dict[str, object]
    outer_reflection_fit: dict[str, object]
    response_selection: dict[str, object]
    field_selection: dict[str, object]
    provider_manifest: dict[str, object]
    held_evidence: dict[str, object]
    fold_receipt: dict[str, object]


def _execute_outer_fold(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    family_ids: Sequence[str],
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
    authenticated_v20a_fold: Mapping[str, object],
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> _FoldLive:
    outer = _identifier(outer_family_id, label="V20p outer family")
    if authenticated_v20o_fold.get("fragment_sha256") != _V20O_FOLD_SHA256S.get(
        outer
    ):
        raise RuntimeError("V20p authenticated V20o fold differs")
    endpoint = _v20g._outer_endpoint(
        context,
        records,
        teacher_vault,
        family_ids=family_ids,
        outer_family_id=outer,
        panel_receipt=panel_receipt,
        authenticated_v20a_fold=authenticated_v20a_fold,
    )
    if (
        _v14._canonical_json_bytes(endpoint.receipt)
        != _v14._canonical_json_bytes(authenticated_v20g_fold.get("endpoint_receipt"))
        or _v14._canonical_json_bytes(endpoint.evidence)
        != _v14._canonical_json_bytes(authenticated_v20g_fold.get("endpoint_evidence"))
    ):
        raise RuntimeError("V20p reconstructed endpoint differs from V20g")
    fit = _mapping(authenticated_v20g_fold.get("fit_receipt"), label="V20p V20g fit")
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20p source direction"
    )
    if source_direction.get("held_family_id") != outer:
        raise RuntimeError("V20p source direction held family differs")
    inner, response_selection, field_selection = _fit_inner_local_field(
        context,
        endpoint,
        source_direction,
        teacher_vault,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
        authenticated_v20o_fold=authenticated_v20o_fold,
    )
    outer_reflection_fit = _reflection.build_soft_polarity_reflection_fit_receipt(
        direction_receipt=source_direction
    )
    _v20o._validate_v20i_reflection_lineage(
        inner_receipt=inner,
        outer_reflection_fit=outer_reflection_fit,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    manifest, held_evidence, fold_receipt = _score_outer_arms(
        context,
        endpoint,
        records,
        teacher_vault,
        source_direction,
        outer_reflection_fit,
        selected_response=_response_tuple(response_selection["selected_response"]),
        field_selection=field_selection,
        outer_family_id=outer,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
        authenticated_v20o_fold=authenticated_v20o_fold,
    )
    return _FoldLive(
        endpoint,
        inner,
        dict(outer_reflection_fit),
        dict(response_selection),
        field_selection,
        manifest,
        held_evidence,
        fold_receipt,
    )


_FOLD_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "field_fit_protocol_sha256",
        "field_provider_protocol_sha256",
        "field_ladder_receipt_sha256",
        "source_artifact_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "v20g_fold_fragment_sha256",
        "v20i_fold_fragment_sha256",
        "v20l_fold_fragment_sha256",
        "v20m_fold_fragment_sha256",
        "v20o_fold_fragment_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "inner_receipt",
        "outer_reflection_fit_receipt",
        "response_selection_receipt",
        "field_selection_receipt",
        "provider_manifest",
        "held_evidence",
        "fold_receipt",
        "fixed_schedule_completed",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _fold_payload(
    live: _FoldLive,
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="V20p fold family")
    selection = live.field_selection
    candidate = {
        "candidate_id": selection["selected_candidate_id"],
        "candidate_index": selection["selected_candidate_index"],
        "feature_id": selection["selected_feature_id"],
        "b": selection["selected_b"],
        "a": selection["selected_a"],
        "adaptive": selection["selected_adaptive"],
        "analysis_only": True,
    }
    payload = {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "field_fit_protocol_sha256": _field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
        "field_provider_protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "field_ladder_receipt_sha256": _FIELD_LADDER_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold["fragment_sha256"],
        "v20i_fold_fragment_sha256": authenticated_v20i_fold["fragment_sha256"],
        "v20l_fold_fragment_sha256": authenticated_v20l_fold["fragment_sha256"],
        "v20m_fold_fragment_sha256": authenticated_v20m_fold["fragment_sha256"],
        "v20o_fold_fragment_sha256": authenticated_v20o_fold["fragment_sha256"],
        "outer_held_family_id": outer,
        "endpoint_receipt": live.endpoint.receipt,
        "endpoint_evidence": live.endpoint.evidence,
        "inner_receipt": live.inner_receipt,
        "outer_reflection_fit_receipt": live.outer_reflection_fit,
        "response_selection_receipt": live.response_selection,
        "field_selection_receipt": live.field_selection,
        "provider_manifest": live.provider_manifest,
        "held_evidence": live.held_evidence,
        "fold_receipt": live.fold_receipt,
        "fixed_schedule_completed": True,
        "candidate": candidate,
        "provider_sidecar": None,
    }
    return payload


def _validate_field_selection(
    value: Mapping[str, object],
    *,
    outer_family_id: str,
    response_selection: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> None:
    selection = _mapping(value, label="V20p field selection")
    _validate_hashed(selection, domain=_INNER_DOMAIN, label="V20p field selection")
    if (
        selection.get("outer_held_family_id") != outer_family_id
        or selection.get("source_response")
        != response_selection.get("selected_response")
        or selection.get("v20m_response_selection_receipt_sha256")
        != response_selection.get("artifact_sha256")
        or selection.get("outer_held_family_used_for_fit_or_selection") is not False
        or selection.get("all_189_field_providers_and_traces_frozen_before_any_field_capability") is not True
        or selection.get("exact_additional_inner_execution_count") != 378
        or selection.get("all_three_endpoint_identities_exact_on_all_seven_inner_families") is not True
    ):
        raise ValueError("V20p field selection boundary differs")
    manifest = _mapping(selection.get("field_provider_manifest"), label="V20p inner manifest")
    _validate_hashed(manifest, domain=_MANIFEST_DOMAIN, label="V20p inner manifest")
    if (
        manifest.get("field_ladder_receipt_sha256") != _FIELD_LADDER_SHA256
        or _v14._canonical_json_bytes(manifest.get("field_ladder_receipt"))
        != _v14._canonical_json_bytes(_FIELD_LADDER_RECEIPT)
        or tuple(manifest.get("candidate_order", ())) != _FIELD_CANDIDATE_IDS
        or manifest.get("outer_held_family_id") != outer_family_id
        or manifest.get("source_response") != selection.get("source_response")
        or tuple(manifest.get("inner_family_order", ()))
        != tuple(sorted(receipt_rows := _mapping(
            manifest.get("provider_receipts_by_inner_family_and_candidate"),
            label="V20p inner provider receipts",
        )))
        or manifest.get("all_seven_times_twenty_seven_providers_frozen_before_any_field_capability") is not True
        or manifest.get("field_capability_count_at_freeze") != 0
    ):
        raise ValueError("V20p inner provider freeze barrier differs")
    if len(receipt_rows) != 7:
        raise ValueError("V20p provider receipt family count differs")
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s_by_inner_family_and_candidate"),
        label="V20p inner provider hashes",
    )
    runtime_hashes = _mapping(
        manifest.get("runtime_provider_artifact_sha256s_by_inner_family_and_candidate"),
        label="V20p inner runtime hashes",
    )
    trace_hashes = _mapping(
        manifest.get("trace_sha256s_by_inner_family_and_candidate"),
        label="V20p inner trace hashes",
    )
    seeds = _mapping(
        manifest.get("provider_transfer_evidence_sha256s_by_inner_family_and_candidate"),
        label="V20p inner transfer hashes",
    )
    if not (
        set(provider_hashes)
        == set(runtime_hashes)
        == set(trace_hashes)
        == set(seeds)
        == set(receipt_rows)
    ):
        raise ValueError("V20p manifest family maps differ")
    all_provider_artifacts: list[str] = []
    all_runtime_artifacts: list[str] = []
    v20m_inner_receipt = _mapping(
        authenticated_v20m_fold.get("inner_receipt"),
        label="V20p authenticated V20m inner receipt",
    )
    v20m_inner_evidence = _mapping(
        v20m_inner_receipt.get("inner_evidence_by_family"),
        label="V20p authenticated V20m inner evidence",
    )
    endpoint_receipt_sha256 = _sha(
        _mapping(
            authenticated_v20m_fold.get("endpoint_receipt"),
            label="V20p authenticated V20m endpoint receipt",
        ).get("artifact_sha256"),
        label="V20p authenticated V20m endpoint receipt",
    )
    source_response = _response_tuple(selection.get("source_response"))
    for family, row_value in receipt_rows.items():
        row = _mapping(row_value, label=f"V20p {family} provider row")
        if tuple(row) != _FIELD_CANDIDATE_IDS:
            raise ValueError("V20p provider candidate order differs")
        provider_row = _mapping(provider_hashes[family], label="V20p provider hash row")
        runtime_row = _mapping(runtime_hashes[family], label="V20p runtime hash row")
        trace_row = _mapping(trace_hashes[family], label="V20p trace hash row")
        seed_row = _mapping(seeds[family], label="V20p seed row")
        if not (
            tuple(provider_row)
            == tuple(runtime_row)
            == tuple(trace_row)
            == tuple(seed_row)
            == _FIELD_CANDIDATE_IDS
        ):
            raise ValueError("V20p manifest candidate maps differ")
        for candidate_id, raw_receipt in row.items():
            receipt = _mapping(raw_receipt, label="V20p provider receipt")
            _validate_field_provider_receipt(
                receipt, expected_role="inner_local_signed_field_candidate"
            )
            if (
                receipt.get("candidate_id") != candidate_id
                or receipt.get("provider_artifact_sha256")
                != provider_row[candidate_id]
                or receipt.get("runtime_provider_artifact_sha256")
                != runtime_row[candidate_id]
            ):
                raise ValueError("V20p manifest provider receipt cross-binding differs")
            seed = _sha(seed_row[candidate_id], label="V20p provider transfer seed")
            runtime_payload = _mapping(
                _mapping(
                    receipt.get("provider_payload"), label="V20p field provider payload"
                ).get("compiled_runtime_provider_payload"),
                label="V20p field runtime payload",
            )
            reflection_fit = _mapping(
                _mapping(v20m_inner_evidence[family], label="V20p V20m family").get(
                    "reflection_fit_receipt"
                ),
                label="V20p V20m reflection fit",
            )
            feature_id, field_bias, field_slope = _field_spec(
                int(receipt["candidate_index"])
            )
            expected_transfer_seed = _field_transfer_seed(
                endpoint_receipt_sha256=endpoint_receipt_sha256,
                direction_artifact_sha256=_sha(
                    reflection_fit.get("selected_variant_artifact_sha256"),
                    label="V20p inner selected direction",
                ),
                reflection_fit_sha256=_sha(
                    reflection_fit.get("artifact_sha256"),
                    label="V20p inner reflection fit",
                ),
                response=source_response,
                candidate_id=candidate_id,
                feature_id=feature_id,
                field_bias=field_bias,
                field_slope=field_slope,
                outer_family_id=outer_family_id,
                inner_family_id=family,
                role="inner_local_signed_field_candidate",
            )
            expected_direction_sha = fisher_soft_polarity_local_signed_field_direction_sha256(
                torch.tensor(_selected_direction(reflection_fit), dtype=torch.float64)
            )
            if (
                seed != expected_transfer_seed
                or runtime_payload.get("direction_sha256") != expected_direction_sha
                or runtime_payload.get("transfer_evidence_sha256") != seed
                or (
                    runtime_payload.get("radius"),
                    runtime_payload.get("shrink_mass"),
                    runtime_payload.get("polarity_bias"),
                )
                != source_response
            ):
                raise ValueError("V20p inner runtime lineage differs")
            all_provider_artifacts.append(str(provider_row[candidate_id]))
            all_runtime_artifacts.append(str(runtime_row[candidate_id]))
    if (
        len(set(all_provider_artifacts)) != 7 * 27
        or len(set(all_runtime_artifacts)) != 7 * 27
    ):
        raise ValueError("V20p inner provider/runtime artifacts are not globally unique")
    evidence_rows = _mapping(
        selection.get("field_evidence_by_inner_family_and_candidate"),
        label="V20p inner field evidence",
    )
    if set(evidence_rows) != set(receipt_rows):
        raise ValueError("V20p evidence family geometry differs")
    for family, row_value in evidence_rows.items():
        row = _mapping(row_value, label=f"V20p {family} evidence row")
        if tuple(row) != _FIELD_CANDIDATE_IDS:
            raise ValueError("V20p evidence candidate order differs")
        for evidence in row.values():
            selected = _mapping(evidence, label="V20p candidate evidence")
            _validate_hashed(selected, domain=_EXECUTION_DOMAIN, label="V20p candidate evidence")
            trace = _mapping(selected.get("response_trace"), label="V20p candidate trace")
            _validate_hashed(trace, domain=_TRACE_DOMAIN, label="V20p candidate trace")
            _validate_field_trace_semantics(trace)
    objective_rows = _mapping(
        selection.get("exact_objectives_by_inner_family_and_candidate"),
        label="V20p exact objective rows",
    )
    if set(objective_rows) != set(evidence_rows):
        raise ValueError("V20p objective family geometry differs")
    for family, row_value in evidence_rows.items():
        evidence_row = _mapping(row_value, label="V20p evidence row")
        objective_row = _mapping(objective_rows[family], label="V20p objective row")
        if tuple(objective_row) != _FIELD_CANDIDATE_IDS:
            raise ValueError("V20p objective candidate order differs")
        for candidate_id, evidence_value in evidence_row.items():
            evidence = _mapping(evidence_value, label="V20p candidate evidence")
            provider_receipt = _mapping(
                _mapping(receipt_rows[family], label="V20p receipt row")[candidate_id],
                label="V20p candidate provider receipt",
            )
            trace = _mapping(evidence.get("response_trace"), label="V20p trace")
            if (
                evidence.get("outer_held_family_id") != outer_family_id
                or evidence.get("inner_held_family_id") != family
                or evidence.get("candidate_id") != candidate_id
                or evidence.get("provider_artifact_sha256")
                != provider_receipt.get("provider_artifact_sha256")
                or evidence.get("runtime_provider_artifact_sha256")
                != provider_receipt.get("runtime_provider_artifact_sha256")
                or evidence.get("manifest_sha256") != manifest.get("artifact_sha256")
                or not isinstance(evidence.get("execution_seed_sha256"), str)
                or evidence.get("objective") != objective_row[candidate_id]
                or trace.get("artifact_sha256")
                != _mapping(trace_hashes[family], label="V20p trace row")[candidate_id]
                or evidence.get("all_field_candidates_frozen_before_score") is not True
                or evidence.get("outer_family_absent_from_fit_and_score") is not True
            ):
                raise ValueError("V20p candidate evidence cross-binding differs")
            objectives = _mapping(
                evidence.get("objectives_by_example"),
                label="V20p candidate objectives",
            )
            h4_hashes = _mapping(
                evidence.get("post_cast_h4_sha256s"), label="V20p candidate H4"
            )
            logits_hashes = _mapping(
                evidence.get("supervised_full_vocab_logits_sha256s"),
                label="V20p candidate logits",
            )
            execution_hashes = _mapping(
                evidence.get("execution_sha256s"),
                label="V20p candidate execution hashes",
            )
            if not (
                len(objectives) == 2
                and set(objectives)
                == set(h4_hashes)
                == set(logits_hashes)
                == set(execution_hashes)
                and math.fsum(float(item) for item in objectives.values())
                / len(objectives)
                == float(evidence["objective"])
            ):
                raise ValueError("V20p exact objective bundle differs")
            execution_seed = _sha(
                evidence.get("execution_seed_sha256"),
                label="V20p execution seed",
            )
            expected_execution_seed = _inner_execution_seed(
                manifest_sha256=str(manifest["artifact_sha256"]),
                outer_family_id=outer_family_id,
                inner_family_id=family,
                candidate_id=candidate_id,
                provider_artifact_sha256=str(
                    provider_receipt["provider_artifact_sha256"]
                ),
            )
            if execution_seed != expected_execution_seed:
                raise ValueError("V20p candidate execution seed differs")
            runtime_artifact = str(evidence["runtime_provider_artifact_sha256"])
            for example in objectives:
                expected_execution = _v20o._execution_sha256(
                    phase="inner_local_signed_field_exact_score",
                    outer_family_id=outer_family_id,
                    inner_family_id=family,
                    role="inner_local_signed_field_candidate",
                    provider_artifact_sha256=runtime_artifact,
                    example_id=example,
                    family_id=family,
                    objective=float(objectives[example]),
                    h4_sha256=str(h4_hashes[example]),
                    logits_sha256=str(logits_hashes[example]),
                    evidence_sha256=execution_seed,
                    domain=_EXECUTION_DOMAIN,
                )
                if execution_hashes[example] != expected_execution:
                    raise ValueError("V20p candidate execution commitment differs")
    all_families = tuple(sorted((*objective_rows.keys(), outer_family_id)))
    core = _mapping(selection.get("core_fit_receipt"), label="V20p core fit")
    _field_fit.validate_soft_polarity_local_signed_field_fit_receipt(
        core,
        ladder_receipt=_FIELD_LADDER_RECEIPT,
        all_development_family_ids=all_families,
        outer_held_family_id=outer_family_id,
        exact_objectives_by_family_and_candidate=objective_rows,
    )
    for key, core_key in (
        ("selected_candidate_id", "selected_candidate_id"),
        ("selected_candidate_index", "selected_candidate_index"),
        ("selected_feature_id", "selected_feature_id"),
        ("selected_b", "selected_b"),
        ("selected_a", "selected_a"),
        ("selected_adaptive", "selected_adaptive"),
    ):
        if selection.get(key) != core.get(core_key):
            raise ValueError(f"V20p field selection {key} differs from core fit")
    anchors = _mapping(
        selection.get("endpoint_exact_output_anchor_by_inner_family_and_candidate"),
        label="V20p endpoint anchor map",
    )
    if set(anchors) != set(evidence_rows):
        raise ValueError("V20p endpoint anchor family geometry differs")
    v20m_inner = _mapping(
        _mapping(
            authenticated_v20m_fold.get("inner_receipt"),
            label="V20p V20m inner receipt",
        ).get("inner_evidence_by_family"),
        label="V20p V20m inner evidence",
    )
    v20o_missing = _mapping(
        _mapping(
            authenticated_v20o_fold.get("signed_continuum_selection_receipt"),
            label="V20p V20o signed selection",
        ).get("missing_anchor_evidence_by_family_and_anchor"),
        label="V20p V20o missing anchors",
    )
    response_key = _response_key(response_selection["selected_response"])
    for family in evidence_rows:
        row = _mapping(anchors[family], label="V20p endpoint anchor row")
        expected_ids = tuple(_anchor_candidate_id(value) for value in (-1.0, 0.0, 1.0))
        if tuple(row) != expected_ids:
            raise ValueError("V20p endpoint anchor candidate order differs")
        v20m_response = _mapping(
            _mapping(
                _mapping(v20m_inner[family], label="V20p V20m family").get("response_evidence"),
                label="V20p V20m response evidence",
            ).get(response_key),
            label="V20p V20m selected response",
        )
        missing = _mapping(v20o_missing[family], label="V20p V20o anchor family")
        authorities = (
            _mapping(missing.get("signed_minus_one"), label="V20p minus authority"),
            _mapping(missing.get("signed_zero"), label="V20p zero authority"),
            v20m_response,
        )
        for candidate_id, authority in zip(expected_ids, authorities, strict=True):
            evidence = _mapping(
                _mapping(evidence_rows[family], label="V20p evidence row")[candidate_id],
                label="V20p endpoint evidence",
            )
            replayed = _exact_bundle_equal(
                _mapping(evidence.get("objectives_by_example"), label="V20p endpoint objectives"),
                _mapping(evidence.get("post_cast_h4_sha256s"), label="V20p endpoint H4"),
                _mapping(evidence.get("supervised_full_vocab_logits_sha256s"), label="V20p endpoint logits"),
                authority,
            )
            if row[candidate_id] is not replayed or not replayed:
                raise ValueError("V20p endpoint exact-output replay differs")


def _validate_fold_fragment(
    value: Mapping[str, object],
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> None:
    fold = _mapping(value, label="V20p fold fragment")
    if set(fold) != _FOLD_KEYS:
        raise ValueError("V20p fold fragment key set differs")
    supplied = dict(fold)
    fragment_sha = _sha(supplied.pop("fragment_sha256"), label="V20p fragment")
    if fragment_sha != _v14._sha256(supplied, domain=_FOLD_DOMAIN):
        raise ValueError("V20p fold fragment hash differs")
    expected = {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "field_fit_protocol_sha256": _field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
        "field_provider_protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "field_ladder_receipt_sha256": _FIELD_LADDER_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold["fragment_sha256"],
        "v20i_fold_fragment_sha256": authenticated_v20i_fold["fragment_sha256"],
        "v20l_fold_fragment_sha256": authenticated_v20l_fold["fragment_sha256"],
        "v20m_fold_fragment_sha256": authenticated_v20m_fold["fragment_sha256"],
        "v20o_fold_fragment_sha256": authenticated_v20o_fold["fragment_sha256"],
        "outer_held_family_id": outer_family_id,
        "fixed_schedule_completed": True,
        "provider_sidecar": None,
    }
    for key, expected_value in expected.items():
        if fold.get(key) != expected_value:
            raise ValueError(f"V20p fold {key} differs")
    if (
        _v14._canonical_json_bytes(fold.get("endpoint_receipt"))
        != _v14._canonical_json_bytes(authenticated_v20g_fold.get("endpoint_receipt"))
        or _v14._canonical_json_bytes(fold.get("endpoint_evidence"))
        != _v14._canonical_json_bytes(authenticated_v20g_fold.get("endpoint_evidence"))
        or _v14._canonical_json_bytes(fold.get("response_selection_receipt"))
        != _v14._canonical_json_bytes(authenticated_v20o_fold.get("response_selection_receipt"))
        or _v14._canonical_json_bytes(fold.get("inner_receipt"))
        != _v14._canonical_json_bytes(authenticated_v20o_fold.get("inner_receipt"))
        or _v14._canonical_json_bytes(fold.get("outer_reflection_fit_receipt"))
        != _v14._canonical_json_bytes(
            authenticated_v20o_fold.get("outer_reflection_fit_receipt")
        )
    ):
        raise ValueError("V20p inherited endpoint, inner fit, or response differs")
    _validate_field_selection(
        _mapping(fold.get("field_selection_receipt"), label="V20p field selection"),
        outer_family_id=outer_family_id,
        response_selection=_mapping(
            fold.get("response_selection_receipt"),
            label="V20p response selection",
        ),
        authenticated_v20m_fold=authenticated_v20m_fold,
        authenticated_v20o_fold=authenticated_v20o_fold,
    )
    manifest = _mapping(fold.get("provider_manifest"), label="V20p outer manifest")
    _validate_hashed(manifest, domain=_MANIFEST_DOMAIN, label="V20p outer manifest")
    prior_manifest = _mapping(
        authenticated_v20o_fold.get("provider_manifest"),
        label="V20p V20o provider manifest",
    )
    endpoint_receipt = _mapping(
        fold.get("endpoint_receipt"), label="V20p outer endpoint receipt"
    )
    source_direction = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_receipt"),
            label="V20p authenticated V20g fit",
        ).get("direction_receipt"),
        label="V20p authenticated V20g source direction",
    )
    outer_reflection_fit = _mapping(
        fold.get("outer_reflection_fit_receipt"),
        label="V20p outer reflection fit",
    )
    if (
        manifest.get("outer_held_family_id") != outer_family_id
        or manifest.get("endpoint_receipt_sha256")
        != endpoint_receipt.get("artifact_sha256")
        or manifest.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or manifest.get("outer_reflection_fit_receipt_sha256")
        != outer_reflection_fit.get("artifact_sha256")
        or manifest.get("matched_v20l_source_fold_sha256")
        != authenticated_v20l_fold.get("fragment_sha256")
        or manifest.get("matched_v20m_source_fold_sha256")
        != authenticated_v20m_fold.get("fragment_sha256")
        or manifest.get("v20o_control_manifest_sha256")
        != prior_manifest.get("artifact_sha256")
        or tuple(manifest.get("arm_order", ())) != _ARMS
        or manifest.get("all_nine_providers_frozen_before_outer_capability") is not True
        or manifest.get("all_nine_traces_frozen_before_outer_capability") is not True
        or manifest.get("outer_capability_count_at_freeze") != 0
        or manifest.get("outer_objectives_or_teacher_rows_used_at_freeze") is not False
        or manifest.get(
            "raw_provider_prompt_token_logit_h4_or_teacher_tensors_serialized"
        )
        is not False
    ):
        raise ValueError("V20p outer freeze barrier differs")
    outer_receipts = _mapping(
        manifest.get("provider_receipts"), label="V20p outer receipts"
    )
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s"), label="V20p outer provider hashes"
    )
    runtime_hashes = _mapping(
        manifest.get("runtime_provider_artifact_sha256s"),
        label="V20p outer runtime hashes",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s"), label="V20p outer trace hashes"
    )
    if not _has_exact_arm_keys(
        outer_receipts,
        provider_hashes,
        runtime_hashes,
        trace_hashes,
    ):
        raise ValueError("V20p outer manifest arm maps differ")
    if (
        len(set(provider_hashes.values())) != 9
        or len(set(runtime_hashes.values())) != 9
    ):
        raise ValueError("V20p outer provider/runtime artifacts are not unique")
    candidate_receipt = _mapping(
        outer_receipts.get(_PRIMARY_ARM), label="V20p candidate provider receipt"
    )
    _validate_field_provider_receipt(
        candidate_receipt,
        expected_role=_PRIMARY_ARM,
    )
    selection = _mapping(fold.get("field_selection_receipt"), label="V20p selection")
    if (
        manifest.get("selected_response") != selection.get("source_response")
        or manifest.get("selected_candidate_id") != selection.get("selected_candidate_id")
        or manifest.get("selected_candidate_index") != selection.get("selected_candidate_index")
        or manifest.get("selected_feature_id") != selection.get("selected_feature_id")
        or manifest.get("selected_b") != selection.get("selected_b")
        or manifest.get("selected_a") != selection.get("selected_a")
        or candidate_receipt.get("candidate_id") != selection.get("selected_candidate_id")
        or candidate_receipt.get("candidate_index") != selection.get("selected_candidate_index")
        or candidate_receipt.get("provider_artifact_sha256")
        != provider_hashes[_PRIMARY_ARM]
        or candidate_receipt.get("runtime_provider_artifact_sha256")
        != runtime_hashes[_PRIMARY_ARM]
    ):
        raise ValueError("V20p outer manifest selection binding differs")
    candidate_seed = _sha(
        manifest.get("candidate_transfer_evidence_sha256"),
        label="V20p outer candidate transfer seed",
    )
    candidate_feature, candidate_bias, candidate_slope = _field_spec(
        int(selection["selected_candidate_index"])
    )
    expected_candidate_seed = _field_transfer_seed(
        endpoint_receipt_sha256=_sha(
            endpoint_receipt.get("artifact_sha256"),
            label="V20p outer endpoint receipt",
        ),
        direction_artifact_sha256=_sha(
            outer_reflection_fit.get("selected_variant_artifact_sha256"),
            label="V20p outer selected direction",
        ),
        reflection_fit_sha256=_sha(
            outer_reflection_fit.get("artifact_sha256"),
            label="V20p outer reflection fit",
        ),
        response=_response_tuple(selection.get("source_response")),
        candidate_id=str(selection["selected_candidate_id"]),
        feature_id=candidate_feature,
        field_bias=candidate_bias,
        field_slope=candidate_slope,
        outer_family_id=outer_family_id,
        inner_family_id=None,
        role="outer_local_signed_field_reflected",
    )
    candidate_runtime_payload = _mapping(
        _mapping(
            candidate_receipt.get("provider_payload"),
            label="V20p outer candidate payload",
        ).get("compiled_runtime_provider_payload"),
        label="V20p outer candidate runtime payload",
    )
    expected_outer_direction_sha = fisher_soft_polarity_local_signed_field_direction_sha256(
        torch.tensor(
            _selected_direction(
                _mapping(
                    fold.get("outer_reflection_fit_receipt"),
                    label="V20p outer reflection fit",
                )
            ),
            dtype=torch.float64,
        )
    )
    if (
        candidate_seed != expected_candidate_seed
        or candidate_runtime_payload.get("direction_sha256")
        != expected_outer_direction_sha
        or candidate_runtime_payload.get("transfer_evidence_sha256")
        != candidate_seed
        or (
            candidate_runtime_payload.get("radius"),
            candidate_runtime_payload.get("shrink_mass"),
            candidate_runtime_payload.get("polarity_bias"),
        )
        != _response_tuple(selection.get("source_response"))
    ):
        raise ValueError("V20p outer candidate runtime lineage differs")
    prior_receipts = _mapping(
        prior_manifest.get("provider_receipts"), label="V20p prior outer receipts"
    )
    prior_provider_hashes = _mapping(
        prior_manifest.get("provider_artifact_sha256s"),
        label="V20p prior provider hashes",
    )
    prior_runtime_hashes = _mapping(
        prior_manifest.get("runtime_provider_artifact_sha256s"),
        label="V20p prior runtime hashes",
    )
    prior_trace_hashes = _mapping(
        prior_manifest.get("response_trace_sha256s"),
        label="V20p prior trace hashes",
    )
    for arm in _ARMS:
        if arm == _PRIMARY_ARM:
            continue
        if (
            _v14._canonical_json_bytes(outer_receipts[arm])
            != _v14._canonical_json_bytes(prior_receipts[arm])
            or provider_hashes[arm] != prior_provider_hashes[arm]
            or runtime_hashes[arm] != prior_runtime_hashes[arm]
            or trace_hashes[arm] != prior_trace_hashes[arm]
        ):
            raise ValueError(f"V20p inherited outer control {arm} differs")
    held = _mapping(fold.get("held_evidence"), label="V20p held evidence")
    _validate_hashed(held, domain=_EXECUTION_DOMAIN, label="V20p held evidence")
    arms = _mapping(held.get("arm_evidence"), label="V20p arms")
    if (
        not _has_exact_arm_keys(arms)
        or held.get("outer_held_family_id") != outer_family_id
        or held.get("outer_manifest_sha256") != manifest.get("artifact_sha256")
        or held.get("all_nine_providers_and_traces_frozen_before_outer_capability")
        is not True
        or held.get("outer_family_used_for_fit_or_selection") is not False
        or held.get("exact_outer_execution_count") != 18
    ):
        raise ValueError("V20p held evidence geometry differs")
    prior_arms = _mapping(
        _mapping(
            authenticated_v20o_fold.get("held_evidence"),
            label="V20p prior held evidence",
        ).get("arm_evidence"),
        label="V20p prior arm evidence",
    )
    control_flags = _mapping(
        held.get("control_exact_output_anchor_by_arm"),
        label="V20p control anchor flags",
    )
    if set(control_flags) != set(_ARMS) - {_PRIMARY_ARM}:
        raise ValueError("V20p control anchor order differs")
    for arm, evidence in arms.items():
        selected = _mapping(evidence, label=f"V20p {arm} evidence")
        _validate_hashed(selected, domain=_EXECUTION_DOMAIN, label=f"V20p {arm} evidence")
        trace = _mapping(selected.get("response_trace"), label=f"V20p {arm} trace")
        if (
            selected.get("outer_held_family_id") != outer_family_id
            or selected.get("arm") != arm
            or selected.get("provider_artifact_sha256") != provider_hashes[arm]
            or selected.get("runtime_provider_artifact_sha256") != runtime_hashes[arm]
            or selected.get("outer_manifest_sha256") != manifest.get("artifact_sha256")
            or trace.get("artifact_sha256") != trace_hashes[arm]
            or selected.get("exact_execution") is not True
            or selected.get("finite") is not True
        ):
            raise ValueError(f"V20p held arm {arm} binding differs")
        objectives = _mapping(
            selected.get("objectives_by_example"), label=f"V20p {arm} objectives"
        )
        h4_hashes = _mapping(
            selected.get("post_cast_h4_sha256s"), label=f"V20p {arm} H4"
        )
        logits_hashes = _mapping(
            selected.get("supervised_full_vocab_logits_sha256s"),
            label=f"V20p {arm} logits",
        )
        execution_hashes = _mapping(
            selected.get("execution_sha256s"), label=f"V20p {arm} executions"
        )
        execution_seed = _sha(
            selected.get("execution_seed_sha256"),
            label=f"V20p {arm} execution seed",
        )
        expected_seed = _outer_execution_seed(
            manifest_sha256=str(manifest["artifact_sha256"]),
            outer_family_id=outer_family_id,
            arm=arm,
            provider_artifact_sha256=str(provider_hashes[arm]),
        )
        if not (
            execution_seed == expected_seed
            and len(objectives) == 2
            and set(objectives)
            == set(h4_hashes)
            == set(logits_hashes)
            == set(execution_hashes)
            and math.fsum(float(item) for item in objectives.values())
            / len(objectives)
            == float(selected["objective"])
        ):
            raise ValueError(f"V20p held arm {arm} objective bundle differs")
        for example in objectives:
            expected_execution = _v20o._execution_sha256(
                phase="outer_held_local_signed_field_score",
                outer_family_id=outer_family_id,
                inner_family_id=None,
                role=arm,
                provider_artifact_sha256=str(runtime_hashes[arm]),
                example_id=example,
                family_id=outer_family_id,
                objective=float(objectives[example]),
                h4_sha256=str(h4_hashes[example]),
                logits_sha256=str(logits_hashes[example]),
                evidence_sha256=execution_seed,
                domain=_EXECUTION_DOMAIN,
            )
            if execution_hashes[example] != expected_execution:
                raise ValueError(f"V20p held arm {arm} execution commitment differs")
        if arm == _PRIMARY_ARM:
            _validate_hashed(
                trace,
                domain=_TRACE_DOMAIN,
                label="V20p candidate trace",
            )
            _validate_field_trace_semantics(trace)
        else:
            replayed = _exact_bundle_equal(
                _mapping(selected.get("objectives_by_example"), label="V20p control objectives"),
                _mapping(selected.get("post_cast_h4_sha256s"), label="V20p control H4"),
                _mapping(selected.get("supervised_full_vocab_logits_sha256s"), label="V20p control logits"),
                _mapping(prior_arms[arm], label="V20p prior control evidence"),
            )
            if control_flags[arm] is not replayed or not replayed:
                raise ValueError(f"V20p control exact-output replay {arm} differs")
    if held.get("all_eight_inherited_control_exact_output_anchors_passed") is not all(
        bool(value) for value in control_flags.values()
    ):
        raise ValueError("V20p aggregate control anchor flag differs")
    base_examples = tuple(
        _mapping(arms["base"], label="V20p base evidence")
        .get("objectives_by_example", {})
        .keys()
    )
    _v20b._validate_capability_receipt(
        _mapping(held.get("capability_receipt"), label="V20p outer capability"),
        expected_example_ids=base_examples,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=9,
        label="V20p outer capability",
    )
    receipt = _mapping(fold.get("fold_receipt"), label="V20p fold receipt")
    _validate_hashed(receipt, domain=_DECISION_DOMAIN, label="V20p fold receipt")
    candidate = _mapping(fold.get("candidate"), label="V20p candidate summary")
    candidate_trace = _mapping(
        _mapping(arms[_PRIMARY_ARM], label="V20p candidate evidence").get("response_trace"),
        label="V20p candidate trace",
    )
    derived_objectives = {
        arm: float(_mapping(arms[arm], label=f"V20p {arm} evidence")["objective"])
        for arm in _ARMS
    }
    candidate_h4 = _mapping(
        _mapping(arms[_PRIMARY_ARM], label="V20p candidate evidence").get("post_cast_h4_sha256s"),
        label="V20p candidate H4",
    )
    candidate_logits = _mapping(
        _mapping(arms[_PRIMARY_ARM], label="V20p candidate evidence").get("supervised_full_vocab_logits_sha256s"),
        label="V20p candidate logits",
    )
    derived_distinct = {}
    for anchor, arm in (
        ("signed_minus_one", "simplex_response_reflected_exact_mirror"),
        ("signed_zero", "fixed_plus"),
        ("signed_plus_one", "matched_v20m_simplex_reflected"),
    ):
        anchor_evidence = _mapping(arms[arm], label="V20p anchor evidence")
        derived_distinct[anchor] = not (
            dict(candidate_h4)
            == dict(_mapping(anchor_evidence.get("post_cast_h4_sha256s"), label="V20p anchor H4"))
            and dict(candidate_logits)
            == dict(_mapping(anchor_evidence.get("supervised_full_vocab_logits_sha256s"), label="V20p anchor logits"))
        )
    derived_runtime_health = all(
        _mapping(
            _mapping(arms[arm], label="V20p arm evidence").get("response_trace"),
            label="V20p arm trace",
        ).get("pointwise_trust_passed")
        is True
        for arm in _ARMS
    )
    if (
        candidate.get("candidate_id") != selection.get("selected_candidate_id")
        or candidate.get("candidate_index") != selection.get("selected_candidate_index")
        or candidate.get("feature_id") != selection.get("selected_feature_id")
        or candidate.get("b") != selection.get("selected_b")
        or candidate.get("a") != selection.get("selected_a")
        or candidate.get("adaptive") != selection.get("selected_adaptive")
        or candidate.get("analysis_only") is not True
        or tuple(receipt.get("arm_order", ())) != _ARMS
        or receipt.get("outer_held_family_id") != outer_family_id
        or receipt.get("selected_response") != selection.get("source_response")
        or receipt.get("selected_candidate_id") != selection.get("selected_candidate_id")
        or receipt.get("selected_candidate_index") != selection.get("selected_candidate_index")
        or receipt.get("selected_feature_id") != selection.get("selected_feature_id")
        or receipt.get("selected_b") != selection.get("selected_b")
        or receipt.get("selected_a") != selection.get("selected_a")
        or receipt.get("selected_adaptive") != selection.get("selected_adaptive")
        or receipt.get("held_objective_by_arm") != derived_objectives
        or receipt.get("candidate_field_nonconstant")
        is not (candidate_trace.get("local_signed_scalar_nonconstant") is True)
        or receipt.get("candidate_field_has_negative")
        is not (candidate_trace.get("local_signed_scalar_has_negative") is True)
        or receipt.get("candidate_field_has_positive")
        is not (candidate_trace.get("local_signed_scalar_has_positive") is True)
        or receipt.get("candidate_exact_output_distinct_by_anchor") != derived_distinct
        or receipt.get("all_inner_endpoint_exact_output_anchors_passed")
        is not selection.get("all_three_endpoint_identities_exact_on_all_seven_inner_families")
        or receipt.get("all_inherited_control_exact_output_anchors_passed")
        is not held.get("all_eight_inherited_control_exact_output_anchors_passed")
        or receipt.get("all_runtime_health_passed") is not derived_runtime_health
        or receipt.get("selection_frozen_before_outer_score") is not True
        or receipt.get("outer_family_used_for_fit_or_selection") is not False
        or receipt.get("exact_execution") is not True
    ):
        raise ValueError("V20p candidate summary or derived fold receipt differs")


def _publish_fold_fragment(
    payload: Mapping[str, object], *, output: Path | str, outer_family_id: str
) -> None:
    _v20b._publish_scalar_fragment(
        payload,
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20p local signed-field outer fold",
    )


def _load_fold_fragment(
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
    authenticated_v20l_fold: Mapping[str, object],
    authenticated_v20m_fold: Mapping[str, object],
    authenticated_v20o_fold: Mapping[str, object],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20p local signed-field outer fold",
    )
    _validate_fold_fragment(
        value,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        outer_family_id=outer_family_id,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20i_fold=authenticated_v20i_fold,
        authenticated_v20l_fold=authenticated_v20l_fold,
        authenticated_v20m_fold=authenticated_v20m_fold,
        authenticated_v20o_fold=authenticated_v20o_fold,
    )
    return dict(value)


def _aggregate_decision(
    fold_fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    families = tuple(sorted(fold_fragments))
    if len(families) != _FAMILY_COUNT:
        raise ValueError("V20p decision requires exactly eight outer folds")
    receipts = {
        family: _mapping(
            fold_fragments[family].get("fold_receipt"),
            label=f"V20p {family} fold receipt",
        )
        for family in families
    }
    scores = {
        family: {
            arm: float(value)
            for arm, value in _mapping(
                receipts[family].get("held_objective_by_arm"),
                label=f"V20p {family} held scores",
            ).items()
        }
        for family in families
    }
    if not _has_exact_arm_keys(*scores.values()):
        raise ValueError("V20p decision arm geometry differs")
    macro = {
        arm: math.fsum(scores[family][arm] for family in families) / len(families)
        for arm in _ARMS
    }
    references = (
        "base",
        "fixed_plus",
        "same_simplex_response_unreflected",
        "simplex_response_reflected_exact_mirror",
        "matched_linear_reflected",
        "matched_v20l_boundary_reflected",
        "matched_v20m_simplex_reflected",
    )
    wins = {
        arm: sum(
            scores[family][_PRIMARY_ARM] < scores[family][arm]
            for family in families
        )
        for arm in references
    }
    primary = (
        macro[_PRIMARY_ARM] < macro["base"]
        and macro[_PRIMARY_ARM] < macro["fixed_plus"]
        and wins["base"] >= 6
        and wins["fixed_plus"] >= 6
    )
    mechanism_refs = (
        "same_simplex_response_unreflected",
        "simplex_response_reflected_exact_mirror",
        "matched_linear_reflected",
        "matched_v20l_boundary_reflected",
        "matched_v20m_simplex_reflected",
    )
    mechanism = all(
        macro[_PRIMARY_ARM] < macro[arm] and wins[arm] >= 5
        for arm in mechanism_refs
    )
    adaptive = {
        family: receipts[family].get("selected_adaptive") is True
        for family in families
    }
    nonconstant = {
        family: receipts[family].get("candidate_field_nonconstant") is True
        for family in families
    }
    negative = {
        family: receipts[family].get("candidate_field_has_negative") is True
        for family in families
    }
    positive = {
        family: receipts[family].get("candidate_field_has_positive") is True
        for family in families
    }
    crosses_zero = {
        family: negative[family] and positive[family] for family in families
    }
    distinct = {
        anchor: {
            family: _mapping(
                receipts[family].get("candidate_exact_output_distinct_by_anchor"),
                label="V20p anchor distinctness",
            ).get(anchor)
            is True
            for family in families
        }
        for anchor in ("signed_minus_one", "signed_zero", "signed_plus_one")
    }
    distinct_counts = {
        anchor: sum(row.values()) for anchor, row in distinct.items()
    }
    field_evidence = (
        sum(adaptive.values()) >= 6
        and sum(nonconstant.values()) >= 6
        and any(negative.values())
        and any(positive.values())
        and any(crosses_zero.values())
        and all(count >= 6 for count in distinct_counts.values())
    )
    endpoint_identity = all(
        receipts[family].get("all_inner_endpoint_exact_output_anchors_passed")
        is True
        and receipts[family].get(
            "all_inherited_control_exact_output_anchors_passed"
        )
        is True
        for family in families
    )
    runtime_health = all(
        receipts[family].get("all_runtime_health_passed") is True
        and receipts[family].get("selection_frozen_before_outer_score") is True
        and receipts[family].get("outer_family_used_for_fit_or_selection") is False
        and receipts[family].get("exact_execution") is True
        for family in families
    )
    integrity = endpoint_identity and runtime_health
    passed = primary and mechanism and field_evidence and integrity
    return _hashed(
        {
            "family_ids": families,
            "held_objective_by_family_and_arm": scores,
            "macro_objective_by_arm": macro,
            "candidate_win_count_by_reference_arm": wins,
            "strict_win_comparison": True,
            "primary_development_gate_passed": primary,
            "mechanism_gate_passed": mechanism,
            "selected_adaptive_by_family": adaptive,
            "selected_adaptive_count": sum(adaptive.values()),
            "held_field_nonconstant_by_family": nonconstant,
            "held_field_nonconstant_count": sum(nonconstant.values()),
            "held_field_has_negative_by_family": negative,
            "held_field_has_negative_count": sum(negative.values()),
            "held_field_has_positive_by_family": positive,
            "held_field_has_positive_count": sum(positive.values()),
            "held_field_crosses_zero_by_family": crosses_zero,
            "held_field_crosses_zero_count": sum(crosses_zero.values()),
            "candidate_exact_output_distinct_by_anchor_and_family": distinct,
            "candidate_exact_output_distinct_count_by_anchor": distinct_counts,
            "local_signed_field_evidence_gate_passed": field_evidence,
            "endpoint_identity_gate_passed": endpoint_identity,
            "runtime_health_gate_passed": runtime_health,
            "integrity_passed": integrity,
            "development_oof_passed": passed,
        },
        domain=_DECISION_DOMAIN,
    )


def _runner_work_accounting() -> dict[str, object]:
    return {
        "accounting_scope": "canonical_one_shot_schedule",
        "canonical_model_forward_count": 5440,
        "total_model_forward_count": 5440,
        "canonical_teacher_access_count": 5408,
        "total_teacher_access_count": 5408,
        "canonical_suffix_backward_count": 128,
        "total_suffix_backward_count": 128,
        "canonical_local_autograd_contraction_count": 112,
        "total_local_autograd_contraction_count": 112,
        "live_authority_collection_model_forward_count": 32,
        "live_authority_collection_suffix_backward_count": 16,
        "endpoint_reconstruction_model_forward_count": 112,
        "endpoint_reconstruction_suffix_backward_count": 112,
        "endpoint_reconstruction_local_autograd_contraction_count": 112,
        "inner_original_response_model_forward_count": 2128,
        "inner_local_signed_field_model_forward_count": 3024,
        "inner_conditional_leave_one_family_out_model_forward_count": 5152,
        "outer_held_model_forward_count": 144,
        "simplex_response_candidate_count": 1064,
        "local_signed_field_candidate_count": 1512,
        "inner_provider_candidate_count": 2576,
        "inner_providers_and_traces_staged_global_count": 2576,
        "inner_providers_and_traces_staged_per_outer_fold": 322,
        "inner_response_trace_example_count": 2128,
        "inner_local_signed_field_trace_example_count": 3024,
        "outer_arm_provider_count": 72,
        "endpoint_health_trace_example_count": 112,
        "reflection_fit_count": 64,
        "masked_fisher_solve_count": 56,
        "all_eight_final_refit_model_forward_count": 0,
        "calibration_b_forward_or_tokenization_count": 0,
        "inner_endpoint_retrained_per_fold": False,
        "resume_and_authentication_overhead_excluded": True,
    }


def _build_report(
    *,
    output: Path | str,
    authorities: _Authorities,
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    fold_fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    decision = _aggregate_decision(fold_fragments)
    passed = decision["development_oof_passed"] is True
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "field_fit_protocol_sha256": _field_fit.SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_PROTOCOL_SHA256,
        "field_provider_protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "field_ladder_receipt_sha256": _FIELD_LADDER_SHA256,
        "fixed_protocol": _FIXED_PROTOCOL,
        "source_receipt": authorities.source,
        "panel_receipt": dict(panel_receipt),
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20o_authority": {
            "report_sha256": _V20O_LOGICAL_SHA256,
            "file_sha256": _V20O_FILE_SHA256,
            "source_receipt_sha256": _V20O_SOURCE_SHA256,
            "fold_fragment_sha256s_by_family": dict(sorted(_V20O_FOLD_SHA256S.items())),
            "classification": authorities.v20o_report.get("classification"),
            "passed": authorities.v20o_report.get("passed"),
            "motivated_local_field_class": True,
        },
        "fold_fragment_sha256s_by_family": {
            family: fold_fragments[family]["fragment_sha256"]
            for family in sorted(fold_fragments)
        },
        "fold_receipts_by_family": {
            family: fold_fragments[family]["fold_receipt"]
            for family in sorted(fold_fragments)
        },
        "decision": decision,
        "work_accounting": _runner_work_accounting(),
        "all_eight_outer_folds_completed": len(fold_fragments) == 8,
        "primary_development_gate_passed": decision[
            "primary_development_gate_passed"
        ],
        "mechanism_gate_passed": decision["mechanism_gate_passed"],
        "local_signed_field_evidence_gate_passed": decision[
            "local_signed_field_evidence_gate_passed"
        ],
        "development_oof_passed": passed,
        "passed": passed,
        "classification": (
            "soft_polarity_local_signed_field_nested_oof_passed"
            if passed
            else "soft_polarity_local_signed_field_nested_oof_failed_rollback_to_base"
        ),
        "rollback_to_base": not passed,
        "next_rung": (
            "freeze_success_then_seek_fresh_family_disjoint_validation"
            if passed
            else "stop_local_field_class_no_token_JVP_regression_in_this_rung"
        ),
        "historically_reused_A16_only": True,
        "fresh_family_disjoint_scoring_performed": False,
        "fresh_validation_claim_authorized": False,
        "final_refit": None,
        "full_refit_performed": False,
        "calibration_b_eligible": False,
        "calibration_b_opened": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "compression_claim_authorized": False,
        "fidelity_claim_authorized": False,
        "speed_claim_authorized": False,
        "serving_claim_authorized": False,
        "candidate": None,
        "provider_sidecar": None,
        "artifact": None,
    }
    _v14._scalar_report(report)
    return report


def _load_existing_report(
    output: Path,
    *,
    authorities: _Authorities,
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20p local signed-field nested report",
    )
    families = tuple(sorted(authorities.authenticated_v20o_folds))
    folds = {
        family: _load_fold_fragment(
            output=output,
            source=authorities.source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20g_fold=authorities.authenticated_v20g_folds[family],
            authenticated_v20i_fold=authorities.authenticated_v20i_folds[family],
            authenticated_v20l_fold=authorities.authenticated_v20l_folds[family],
            authenticated_v20m_fold=authorities.authenticated_v20m_folds[family],
            authenticated_v20o_fold=authorities.authenticated_v20o_folds[family],
        )
        for family in families
    }
    rebuilt = _build_report(
        output=output,
        authorities=authorities,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        fold_fragments=folds,
    )
    supplied = dict(value)
    logical = supplied.pop("report_sha256", None)
    if (
        _v14._canonical_json_bytes(supplied)
        != _v14._canonical_json_bytes(rebuilt)
        or logical != _v14._sha256(rebuilt, domain=_REPORT_DOMAIN)
    ):
        raise ValueError("V20p report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_local_signed_field_nested_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run, resume, or model-free replay the fixed V20p campaign."""

    destination = _validate_output(output)
    authorities = _load_prerequisites()
    panel = dict(
        _mapping(
            authorities.prerequisite.get("nested_panel_receipt"),
            label="V20p panel receipt",
        )
    )
    bridge = _sha(
        authorities.prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20p bridge binding",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            authorities=authorities,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
        )
    families = tuple(sorted(authorities.authenticated_v20o_folds))
    if (
        len(families) != 8
        or set(authorities.authenticated_v20a_folds) != set(families)
        or set(authorities.authenticated_v20g_folds) != set(families)
        or set(authorities.authenticated_v20i_folds) != set(families)
        or set(authorities.authenticated_v20l_folds) != set(families)
        or set(authorities.authenticated_v20m_folds) != set(families)
        or set(
            _mapping(panel.get("family_prompt_sha256s"), label="V20p panel families")
        )
        != set(families)
    ):
        raise RuntimeError("V20p authenticated family geometry differs")
    if all(_fold_path(destination, family).exists() for family in families):
        folds = {
            family: _load_fold_fragment(
                output=destination,
                source=authorities.source,
                panel_receipt=panel,
                outer_family_id=family,
                bridge_binding_sha256=bridge,
                authenticated_v20g_fold=authorities.authenticated_v20g_folds[family],
                authenticated_v20i_fold=authorities.authenticated_v20i_folds[family],
                authenticated_v20l_fold=authorities.authenticated_v20l_folds[family],
                authenticated_v20m_fold=authorities.authenticated_v20m_folds[family],
                authenticated_v20o_fold=authorities.authenticated_v20o_folds[family],
            )
            for family in families
        }
        report = _build_report(
            output=destination,
            authorities=authorities,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
            fold_fragments=folds,
        )
        try:
            _v20b._publish_scalar_fragment(
                report,
                path=destination,
                domain=_REPORT_DOMAIN,
                hash_key="report_sha256",
                label="V20p local signed-field nested report",
            )
        except FileExistsError:
            pass
        return _load_existing_report(
            destination,
            authorities=authorities,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
        )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        if context.bridge.bridge_binding_sha256 != bridge:
            raise RuntimeError("V20p live bridge differs from authenticated authority")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=authorities.prerequisite
        )
        if tuple(live_families) != families:
            raise RuntimeError("V20p live family order differs from A16 authority")
        fragments: dict[str, dict[str, object]] = {}
        for family in families:
            if _fold_path(destination, family).exists():
                fragments[family] = _load_fold_fragment(
                    output=destination,
                    source=authorities.source,
                    panel_receipt=panel,
                    outer_family_id=family,
                    bridge_binding_sha256=bridge,
                    authenticated_v20g_fold=authorities.authenticated_v20g_folds[family],
                    authenticated_v20i_fold=authorities.authenticated_v20i_folds[family],
                    authenticated_v20l_fold=authorities.authenticated_v20l_folds[family],
                    authenticated_v20m_fold=authorities.authenticated_v20m_folds[family],
                    authenticated_v20o_fold=authorities.authenticated_v20o_folds[family],
                )
                continue
            live = _execute_outer_fold(
                context,
                records,
                teacher_vault,
                family_ids=families,
                outer_family_id=family,
                panel_receipt=panel,
                authenticated_v20a_fold=authorities.authenticated_v20a_folds[family],
                authenticated_v20g_fold=authorities.authenticated_v20g_folds[family],
                authenticated_v20i_fold=authorities.authenticated_v20i_folds[family],
                authenticated_v20l_fold=authorities.authenticated_v20l_folds[family],
                authenticated_v20m_fold=authorities.authenticated_v20m_folds[family],
                authenticated_v20o_fold=authorities.authenticated_v20o_folds[family],
            )
            payload = _fold_payload(
                live,
                output=destination,
                source=authorities.source,
                panel_receipt=panel,
                bridge_binding_sha256=bridge,
                outer_family_id=family,
                authenticated_v20g_fold=authorities.authenticated_v20g_folds[family],
                authenticated_v20i_fold=authorities.authenticated_v20i_folds[family],
                authenticated_v20l_fold=authorities.authenticated_v20l_folds[family],
                authenticated_v20m_fold=authorities.authenticated_v20m_folds[family],
                authenticated_v20o_fold=authorities.authenticated_v20o_folds[family],
            )
            try:
                _publish_fold_fragment(
                    payload, output=destination, outer_family_id=family
                )
            except FileExistsError:
                pass
            fragments[family] = _load_fold_fragment(
                output=destination,
                source=authorities.source,
                panel_receipt=panel,
                outer_family_id=family,
                bridge_binding_sha256=bridge,
                authenticated_v20g_fold=authorities.authenticated_v20g_folds[family],
                authenticated_v20i_fold=authorities.authenticated_v20i_folds[family],
                authenticated_v20l_fold=authorities.authenticated_v20l_folds[family],
                authenticated_v20m_fold=authorities.authenticated_v20m_folds[family],
                authenticated_v20o_fold=authorities.authenticated_v20o_folds[family],
            )
        report = _build_report(
            output=destination,
            authorities=authorities,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
            fold_fragments=fragments,
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
            label="V20p local signed-field nested report",
        )
    except FileExistsError:
        pass
    return _load_existing_report(
        destination,
        authorities=authorities,
        panel_receipt=panel,
        bridge_binding_sha256=bridge,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the V20p nested local signed-field development screen"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_soft_polarity_local_signed_field_nested_development(
        output=args.output, cache_dir=args.cache_dir
    )
    print(_v14._canonical_json_bytes(report).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
