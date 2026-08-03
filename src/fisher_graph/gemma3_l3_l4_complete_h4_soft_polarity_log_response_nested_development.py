"""V20k nested log-response development campaign for Gemma 3.

V20k authenticates the completed V20j rollback and its V20g/V20i lineage
before model construction.  It keeps V20i's six-family masked Fisher directions and
training-only CVaR-2 reflection fit, but replaces the single linear radius
with a bounded odd log response

``q(z) = tanh(r*((1-lambda)*z + lambda*asinh(4*z)/4))``.

The frozen eleven-pair ladder contains a zero anchor, two matched-linear
controls, and eight gently sublinear alternatives.  All seven-by-eleven providers and traces
are frozen before any inner-held capability is opened, and one pair is chosen
by family-equal conditional leave-one-family-out token-mean exact float64
full-vocabulary KL(teacher||candidate).  The endpoint is
fixed across those seven inner scores: the held inner family is excluded from
direction/reflection fitting, but not from endpoint fitting.  This is therefore
response selection conditional on the seven-family endpoint, not fully
nested model cross-validation.  Seven outer mechanism arms are then frozen
before the genuinely endpoint-disjoint outer family is scored.  This remains
reused-A16 development evidence: there is no all-eight refit or Calibration-B
access.  Reports and resumable fold fragments contain scalar/hash evidence
only and are mode 0600.
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

from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as _v20b
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development
    as _v20g,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_reflection_nested_development
    as _v20i,
)
from . import (
    gemma3_l3_l4_complete_h4_soft_polarity_confidence_nested_development
    as _v20j,
)
from . import complete_h4_fisher_soft_polarity_reflection_fit as _reflection
from . import complete_h4_fisher_soft_polarity_log_response_fit as _log_response_fit
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .complete_h4_fisher_soft_polarity import (
    build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control,
)
from .complete_h4_fisher_soft_polarity_log_response import (
    AutonomousCompleteH4FisherSoftPolarityLogResponseProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_log_response,
    fisher_soft_polarity_log_response_box_certificate,
    fisher_soft_polarity_log_response_direction_sha256,
    validate_fisher_soft_polarity_log_response_provider_evidence,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_log_response_nested_development",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-log-response-nested-"
    "r16-k256-a-fit16-dev-v20k.json"
)

_V20J_OUTPUT = _v20j.DEFAULT_OUTPUT
_V20J_LOGICAL_SHA256 = (
    "2dd3be4a6bf3a30596bdecbf760d8e8fb6c518cdeaf87918000be2b1e3cfd28b"
)
_V20J_FILE_SHA256 = (
    "dea76949766ee93c48154f398fb3ac776287354bbcd628fc699cd6975537c2ce"
)
_V20J_SOURCE_SHA256 = (
    "eb9ed4076957e3a5986dacfe9a1b5a3886dabc359a015b20a41ed4f19916c36d"
)
_V20J_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "04d662e3e8d62e4a67a72f95bd2ffa8c00c8ad5cfe0825d186f33d8eef4c9c03"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "cd93bb8d652b53b3c56440e816a0905ae2bfcf962e6e66a5bd77402b18b66ecb"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "5c539e4a674cf122d957444b266ed6ff79691dc9dd6745bb6547ca293f90d60b"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "508e03d845615e0d3217a0a658238328a66c50ea20094e7b5f0fa2704c7dc3c8"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "a81d998dbbf3f0e15dbcc8f07c6f1da5c38a2631cb39c84aa9b444e4617918f2"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "a47dd589d3d7fde7aee42f4aae9a619a5ae7dcdd41db6b8196a21adf379e52d3"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "329233151edb0f813d6821009fd304f850f9dbbb289c5fb5c634c5b15cfc0e98"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "761e8dd07097962103304be364e98a584f8b1bcb7f3d2f7220b964b60e41f9b0"
    ),
}

_V20I_OUTPUT = _v20i.DEFAULT_OUTPUT
_V20I_LOGICAL_SHA256 = (
    "14618913ff620c67000213aa765f45d8587e8ba4dcb0f8dcb634d7ab3490ecdd"
)
_V20I_FILE_SHA256 = (
    "cc044795b3ad3e243eb2b091c17fc491aedbb8f905f92000c4c3f95002c8f356"
)
_V20I_SOURCE_SHA256 = (
    "39f6808053e8121294414db5807a7e3716a35c0b77404157eb0c4b282ca593a1"
)
_V20I_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "e2cd0922a260c198a3cf1fcc27ac5acb1029c7bffd5c60ff9044ff61a1a14324"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "aa19f13d016d410d6716fc2837b6285ed3e556fe34a400870d30411b80814891"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "1dc93c8a8ad1ebeb8a73d98ab77601503f048ab64ec9f928eb1b0b68c8973da3"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "5def19b3c650ed171a6162677e7c0f18ce77a91bfdcd1be561520e40e0c86818"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "8af8692ab94f50d306af7adc3048a53415301da7010537514562b63ab880b990"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "3f467552e75b33d6eeb79c4c7a1393bc390760b533a18e08d529a3acae761fc8"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "f7182ed7b8202443d06cfa87d5d15b6ffc4e947c63a7d5a6a4204647c9165ef0"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "f1776c84fa0b0f275616bce60346bca90aeff708f07bc7658be52c6e4e97ff46"
    ),
}

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_log_response_nested.v20k"
)
_FOLD_SCHEMA = (
    "fisher_graph.complete_h4_soft_polarity_log_response_nested_outer_fold.v20k"
)
_FORMAT_VERSION = 27
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-log_response-nested-report:v20k\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-log_response-nested-source:v20k\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-log_response-nested-fold:v20k\0"
_INNER_FIT_DOMAIN = b"fisher-graph:soft-polarity-log_response-inner-fit:v20k\0"
_INNER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-log_response-inner-manifest:v20k\0"
)
_INNER_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-log_response-inner-execution:v20k\0"
)
_RESPONSE_SELECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-log_response-response-selection:v20k\0"
)
_OUTER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-log_response-outer-manifest:v20k\0"
)
_OUTER_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-log_response-outer-execution:v20k\0"
)
_PROVIDER_DOMAIN = b"fisher-graph:soft-polarity-log_response-provider:v20k\0"
_TRACE_DOMAIN = b"fisher-graph:soft-polarity-log_response-trace:v20k\0"
_DECISION_DOMAIN = b"fisher-graph:soft-polarity-log_response-decision:v20k\0"

_FAMILY_COUNT = 8
_PROMPTS_PER_FAMILY = 2
_INNER_FAMILY_COUNT = 7
_INNER_TRAINING_FAMILY_COUNT = 6
_CONDITIONAL_RANK = 16
_RESPONSES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.125, 0.0),
    (0.125, 0.25),
    (0.125, 0.5),
    (0.125, 0.75),
    (0.125, 1.0),
    (0.25, 0.0),
    (0.25, 0.25),
    (0.25, 0.5),
    (0.25, 0.75),
    (0.25, 1.0),
)
_RESPONSE_KEYS = tuple(
    f"radius={radius.hex()};mix={mix.hex()}" for radius, mix in _RESPONSES
)
if _RESPONSES != tuple(_log_response_fit.SOFT_POLARITY_LOG_RESPONSE_LADDER):
    raise RuntimeError("V20k runner response ladder differs from core protocol")
_LOG_RESPONSE_LADDER_RECEIPT = (
    _log_response_fit.build_soft_polarity_log_response_ladder_receipt()
)
_LOG_RESPONSE_LADDER_RECEIPT_SHA256 = str(
    _LOG_RESPONSE_LADDER_RECEIPT["artifact_sha256"]
)
_ARMS = (
    "base",
    "fixed_plus",
    "fixed_minus",
    "matched_linear_reflected",
    "same_response_unreflected",
    "log_response_reflected",
    "log_response_reflected_exact_mirror",
)
_PRIMARY_ARM = "log_response_reflected"
_PROVIDER_RECEIPT_KEYS = frozenset(
    {
        "role",
        "response",
        "response_key",
        "radius",
        "mix",
        "direction",
        "direction_box_corner_scores",
        "box_certificate",
        "provider_artifact_sha256",
        "provider_metadata",
        "provider_metadata_sha256",
        "provider_payload",
        "transfer_protocol_sha256",
        "transfer_evidence_sha256",
        "rank",
        "conditional_rank",
        "prepared_float_scalar_count",
        "logical_macs_per_token_upper_bound",
        "analysis_only",
        "raw_provider_tensors_serialized",
        "artifact_sha256",
    }
)

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20k_nested_reflection_and_odd_monotone_log_response_fit",
    "scientific_status": (
        "posthoc_after_v20j_reused_A16_development_hypothesis_only"
    ),
    "source": (
        "pinned_V20j_report_and_folds_plus_V20i_reflections_and_"
        "V20g_gradient_Fisher_summaries"
    ),
    "outer_validation": "eight_leave_one_whole_development_family_out_folds",
    "inner_validation": (
        "seven_conditional_leave_one_outer_training_family_out_exact_"
        "log_response_folds_on_one_fixed_seven_family_endpoint"
    ),
    "inner_endpoint_scope": (
        "fixed_endpoint_fit_on_all_seven_outer_training_families_not_retrained_"
        "per_inner_fold"
    ),
    "inner_held_family_used_for_endpoint_fit": True,
    "inner_held_family_used_for_direction_or_reflection_fit": False,
    "inner_claim_scope": (
        "conditional_response_LOFO_not_fully_nested_model_cross_validation"
    ),
    "inner_direction": (
        "six_family_masked_Fisher_natural_direction_then_training_only_CVaR2_"
        "one_coordinate_reflection"
    ),
    "response_order": _RESPONSES,
    "log_response_ladder_receipt_sha256": _LOG_RESPONSE_LADDER_RECEIPT_SHA256,
    "log_response_ladder_receipt_constructed_before_provider_freeze": True,
    "response_selection": (
        "family_equal_mean_of_seven_inner_OOF_token_mean_exact_float64_"
        "full_vocabulary_KL_teacher_to_candidate_then_smaller_mix_then_"
        "smaller_radius_then_fixed_index_then_artifact_sha256"
    ),
    "log_response_formula": (
        "tanh(radius_times_open_paren_one_minus_mix_times_z_plus_mix_times_"
        "asinh_4z_over_4_close_paren)"
    ),
    "log_response_constraints": (
        "radius_in_zero_one_fourth_mix_in_zero_one_odd_monotone_bounded"
    ),
    "inner_freeze_barrier": (
        "all_seven_times_eleven_providers_and_traces_before_any_inner_capability"
    ),
    "outer_arms": _ARMS,
    "outer_freeze_barrier": "all_seven_providers_and_traces_before_outer_capability",
    "primary_gate": (
        "candidate_macro_below_base_and_fixed_plus_and_at_least_six_of_eight_"
        "wins_against_each"
    ),
    "mechanism_gate": (
        "candidate_macro_below_same_response_unreflected_with_five_wins_"
        "below_exact_mirror_with_six_wins_and_below_matched_linear_with_five_wins"
    ),
    "positive_changed_gate": (
        "all_selected_radius_positive_and_candidate_changed_exact"
    ),
    "curvature_evidence_gate": "at_least_five_selected_mix_positive",
    "fixed_minus": "diagnostic_only",
    "matched_linear_reflected_control": True,
    "all_eight_final_refit_in_this_rung": False,
    "calibration_b_eligible": False,
    "serving_authorized": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(_FIXED_PROTOCOL, domain=_SOURCE_DOMAIN)
_TRANSFER_PROTOCOL_SHA256 = _v14._sha256(
    {
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "reflection_fit_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "log_response_fit_protocol_sha256": (
            _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "operation": "V20k_domain_separated_log_response_provider_materialization",
        "held_rows_used": False,
    },
    domain=_PROVIDER_DOMAIN,
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _identifier(value: object, *, label: str) -> str:
    return _v20i._identifier(value, label=label)


def _sha(value: object, *, label: str) -> str:
    return _v20i._sha(value, label=label)


def _hashed(payload: Mapping[str, object], *, domain: bytes) -> dict[str, object]:
    return _v20i._hashed(payload, domain=domain)


def _validate_hashed(
    value: Mapping[str, object], *, domain: bytes, label: str
) -> Mapping[str, object]:
    return _v20i._validate_hashed(value, domain=domain, label=label)


def _response_pair(value: object) -> tuple[float, float]:
    raw = _sequence(value, label="V20k response pair")
    if len(raw) != 2:
        raise ValueError("V20k response must contain exactly radius and mix")
    if any(type(item) not in (int, float) for item in raw):
        raise ValueError("V20k response values must be JSON numbers")
    selected = tuple(float(item) for item in raw)
    if any(
        not math.isfinite(item)
        or item < 0.0
        or (item == 0.0 and math.copysign(1.0, item) < 0.0)
        for item in selected
    ) or selected not in _RESPONSES:
        raise ValueError("V20k response is outside the fixed ladder")
    return selected[0], selected[1]


def _response_key(value: object) -> str:
    radius, mix = _response_pair(value)
    return f"radius={radius.hex()};mix={mix.hex()}"


def _response_order(value: object) -> tuple[tuple[float, float], ...]:
    return tuple(
        _response_pair(item)
        for item in _sequence(value, label="V20k response order")
    )


def _validate_v20i_reflection_lineage(
    *,
    inner_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
) -> None:
    """Bind every reused reflection decision to the pinned V20i authority."""

    inherited_outer = _mapping(
        authenticated_v20i_fold.get("outer_reflection_fit_receipt"),
        label="V20k inherited V20i outer reflection fit",
    )
    if _v14._canonical_json_bytes(outer_reflection_fit) != (
        _v14._canonical_json_bytes(inherited_outer)
    ):
        raise ValueError("V20k outer reflection lineage differs from pinned V20i")

    current_inner = _mapping(
        inner_receipt.get("inner_evidence_by_family"),
        label="V20k current inner reflection evidence",
    )
    inherited_inner_receipt = _mapping(
        authenticated_v20i_fold.get("inner_receipt"),
        label="V20k inherited V20i inner receipt",
    )
    inherited_inner = _mapping(
        inherited_inner_receipt.get("inner_evidence_by_family"),
        label="V20k inherited V20i inner reflection evidence",
    )
    if set(current_inner) != set(inherited_inner):
        raise ValueError("V20k inner reflection family lineage differs from V20i")
    for family in sorted(current_inner):
        current = _mapping(
            current_inner[family], label="V20k current inner reflection family"
        )
        inherited = _mapping(
            inherited_inner[family], label="V20k inherited inner reflection family"
        )
        for field in ("masked_direction_receipt", "reflection_fit_receipt"):
            if _v14._canonical_json_bytes(current.get(field)) != (
                _v14._canonical_json_bytes(inherited.get(field))
            ):
                raise ValueError(
                    f"V20k {field} lineage differs from pinned V20i"
                )


def _validate_output(path: Path | str) -> Path:
    destination = Path(path).resolve(strict=False)
    local_root = _LOCAL_ROOT.resolve(strict=False)
    protected = {
        candidate.resolve(strict=False)
        for candidate in (
            _v20g.DEFAULT_OUTPUT,
            _V20I_OUTPUT,
            _V20J_OUTPUT,
            *getattr(_v20g, "_PROTECTED_PREREQUISITE_PATHS", ()),
        )
    }
    if destination in protected:
        raise ValueError("V20k output must preserve immutable prerequisite artifacts")
    if destination.parent != local_root:
        raise ValueError("V20k output must remain directly under .local-runs")
    return destination


def _family_suffix(family_id: str) -> str:
    return _v20i._family_suffix(family_id)


def _fold_path(output: Path | str, family_id: str) -> Path:
    destination = _validate_output(output)
    return destination.with_name(
        f"{destination.stem}.fold-{_family_suffix(family_id)}.json"
    )


@dataclass(slots=True)
class _FoldLive:
    endpoint: _v20g._EndpointLive
    inner_receipt: dict[str, object]
    outer_reflection_fit: dict[str, object]
    response_selection: dict[str, object]
    provider_manifest: dict[str, object]
    held_evidence: dict[str, object]
    fold_receipt: dict[str, object]


def _load_prerequisites() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """Authenticate V20j and its complete V20g/V20i lineage before construction."""

    (
        prerequisite,
        authenticated_v20a_folds,
        v20g_report,
        authenticated_v20g_folds,
        v20i_report,
        authenticated_v20i_folds,
        v20j_source,
    ) = _v20j._load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"),
            label="V20k inherited panel receipt",
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20k inherited bridge binding",
    )
    if _v14._file_sha256(_V20J_OUTPUT) != _V20J_FILE_SHA256:
        raise RuntimeError("pinned V20j report file hash drifted")
    v20j_report = _v20j._load_existing_report(
        _V20J_OUTPUT,
        source=v20j_source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_v20g_folds=authenticated_v20g_folds,
        authenticated_v20i_folds=authenticated_v20i_folds,
    )
    observed_fold_hashes = {
        _identifier(family, label="V20k V20j fold family"): _sha(
            value, label="V20k V20j fold hash"
        )
        for family, value in _mapping(
            v20j_report.get("fold_fragment_sha256s_by_family"),
            label="V20k V20j fold hashes",
        ).items()
    }
    if (
        v20j_report.get("report_sha256") != _V20J_LOGICAL_SHA256
        or v20j_report.get("classification")
        != "soft_polarity_confidence_nested_oof_failed_rollback_to_base"
        or v20j_report.get("development_oof_passed") is not False
        or v20j_report.get("primary_development_gate_passed") is not False
        or v20j_report.get("mechanism_gate_passed") is not False
        or _mapping(
            v20j_report.get("decision"), label="V20k V20j decision"
        ).get("integrity_passed")
        is not True
        or v20j_report.get("passed") is not False
        or v20j_report.get("rollback_to_base") is not True
        or v20j_report.get("final_refit") is not None
        or v20j_report.get("calibration_b_opened") is not False
        or _mapping(
            v20j_report.get("source_receipt"), label="V20k V20j source"
        ).get("artifact_sha256")
        != _V20J_SOURCE_SHA256
        or observed_fold_hashes != _V20J_FOLD_SHA256S
    ):
        raise RuntimeError("pinned V20j development authority differs")

    families = tuple(sorted(_V20J_FOLD_SHA256S))
    authenticated_v20j_folds = {
        family: _v20j._load_fold_fragment(
            output=_V20J_OUTPUT,
            source=v20j_source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
            authenticated_v20i_fold=authenticated_v20i_folds[family],
        )
        for family in families
    }
    if {
        family: fragment["fragment_sha256"]
        for family, fragment in authenticated_v20j_folds.items()
    } != _V20J_FOLD_SHA256S:
        raise RuntimeError("pinned V20j fold authority differs")
    source = _hashed(
        {
            "v20g_report_sha256": v20g_report["report_sha256"],
            "v20g_fold_fragment_sha256s_by_family": {
                family: authenticated_v20g_folds[family]["fragment_sha256"]
                for family in families
            },
            "v20i_report_sha256": _V20I_LOGICAL_SHA256,
            "v20i_file_sha256": _V20I_FILE_SHA256,
            "v20i_source_receipt_sha256": _V20I_SOURCE_SHA256,
            "v20i_fold_fragment_sha256s_by_family": dict(
                sorted(_V20I_FOLD_SHA256S.items())
            ),
            "v20j_report_sha256": _V20J_LOGICAL_SHA256,
            "v20j_file_sha256": _V20J_FILE_SHA256,
            "v20j_source_receipt_sha256": _V20J_SOURCE_SHA256,
            "v20j_fold_fragment_sha256s_by_family": dict(
                sorted(_V20J_FOLD_SHA256S.items())
            ),
            "reflection_fit_protocol_sha256": (
                _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
            ),
            "masked_direction_protocol_sha256": (
                _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
            ),
            "log_response_fit_protocol_sha256": (
                _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_FIT_PROTOCOL_SHA256
            ),
            "response_order": _RESPONSES,
            "exact_objective_kind": (
                "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
            ),
            "authenticated_before_model_construction": True,
            "historically_reused_A16_only": True,
            "held_scores_used_before_direction_or_response_freeze": False,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
        },
        domain=_SOURCE_DOMAIN,
    )
    return (
        prerequisite,
        authenticated_v20a_folds,
        dict(v20g_report),
        authenticated_v20g_folds,
        dict(v20i_report),
        authenticated_v20i_folds,
        source,
    )


def _selected_direction(
    reflection_fit: Mapping[str, object],
) -> tuple[float, float, float, float]:
    if reflection_fit.get("selected_variant_available") is not True:
        raise RuntimeError("V20k reflection fit has no admissible direction")
    raw = tuple(
        float(item)
        for item in _sequence(
            reflection_fit.get("selected_normalized_direction"),
            label="V20k selected reflection direction",
        )
    )
    if len(raw) != 4 or not all(math.isfinite(item) for item in raw):
        raise RuntimeError("V20k reflection direction is not a finite four-vector")
    return raw  # type: ignore[return-value]


def _unreflected_direction(
    direction_receipt: Mapping[str, object],
) -> tuple[float, float, float, float]:
    raw = tuple(
        float(item)
        for item in _sequence(
            direction_receipt.get("natural_direction"),
            label="V20k unreflected direction",
        )
    )
    if len(raw) != 4 or not all(math.isfinite(item) for item in raw):
        raise RuntimeError("V20k unreflected direction is not finite")
    return raw  # type: ignore[return-value]


def _box_corner_scores(values: Sequence[float] | Tensor) -> tuple[float, ...]:
    direction = tuple(float(item) for item in _v20g._eta_tensor(values).tolist())
    return tuple(
        direction[0]
        + direction[1] * c1
        + direction[2] * c2
        + direction[3] * c1 * c2
        for c1, c2 in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    )


def _provider_seed(
    *,
    endpoint_receipt_sha256: str,
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float],
    direction: Sequence[float],
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> str:
    radius, mix = _response_pair(response)
    selected_direction = tuple(float(item) for item in direction)
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt_sha256, label="V20k provider endpoint"
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256, label="V20k provider direction"
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256, label="V20k provider reflection fit"
            ),
            "response": (radius, mix),
            "response_key": _response_key((radius, mix)),
            "direction": selected_direction,
            "direction_box_corner_scores": _box_corner_scores(selected_direction),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20k provider outer family"
            ),
            "inner_held_family_id": inner_family_id,
            "role": role,
            "held_rows_used": False,
        },
        domain=_PROVIDER_DOMAIN,
    )


def _materialize_provider(
    endpoint: _v20g._EndpointLive,
    *,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float],
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> tuple[AutonomousCompleteH4FisherSoftPolarityLogResponseProvider, str]:
    radius, mix = _response_pair(response)
    seed = _provider_seed(
        endpoint_receipt_sha256=str(endpoint.receipt["artifact_sha256"]),
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        direction=direction,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )
    provider = build_autonomous_complete_h4_fisher_soft_polarity_log_response(
        endpoint.base_provider,
        endpoint.proposal_provider,
        direction=_v20g._eta_tensor(direction),
        radius=radius,
        mix=mix,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=seed,
    )
    return provider, seed


def _provider_receipt(
    provider: object,
    *,
    role: str,
    response: tuple[float, float] | None = None,
    direction: Sequence[float] | None = None,
) -> dict[str, object]:
    metadata = _mapping(provider.metadata(), label=f"V20k {role} metadata")
    provider_payload: Mapping[str, object] | None = None
    selected_direction: tuple[float, ...] | None = None
    corners: tuple[float, ...] | None = None
    radius: float | None = None
    mix: float | None = None
    if isinstance(
        provider, AutonomousCompleteH4FisherSoftPolarityLogResponseProvider
    ):
        provider_payload = provider.artifact_payload()
        selected_direction = tuple(float(item) for item in provider.direction.tolist())
        corners = _box_corner_scores(selected_direction)
        if response is None or direction is None:
            raise ValueError(
                "V20k log_response provider receipt needs response and direction"
            )
        radius, mix = _response_pair(response)
        expected = tuple(float(item) for item in direction)
        if selected_direction != expected:
            raise RuntimeError("V20k provider differs from its frozen direction")
        if (
            float(provider.radius) != radius
            or float(provider.mix) != mix
        ):
            raise RuntimeError("V20k provider coefficients differ from response")
        bound = max(abs(item) for item in corners)
        if abs(bound - 1.0) > 1.0e-12:
            raise RuntimeError("V20k provider direction is not box normalized")
    payload = {
        "role": role,
        "response": response,
        "response_key": (
            None if response is None else _response_key(response)
        ),
        "radius": radius,
        "mix": mix,
        "direction": None if direction is None else tuple(float(x) for x in direction),
        "direction_box_corner_scores": corners,
        "box_certificate": (
            None
            if not isinstance(
                provider, AutonomousCompleteH4FisherSoftPolarityLogResponseProvider
            )
            else fisher_soft_polarity_log_response_box_certificate(
                provider.direction,
                radius=float(provider.radius),
                mix=float(provider.mix),
            )
        ),
        "provider_artifact_sha256": _sha(
            provider.artifact_sha256, label=f"V20k {role} provider artifact"
        ),
        "provider_metadata": dict(metadata),
        "provider_metadata_sha256": _v14._sha256(
            metadata, domain=_PROVIDER_DOMAIN
        ),
        "provider_payload": (
            None if provider_payload is None else dict(provider_payload)
        ),
        "transfer_protocol_sha256": metadata.get("transfer_protocol_sha256"),
        "transfer_evidence_sha256": metadata.get("transfer_evidence_sha256"),
        "rank": int(provider.rank),
        "conditional_rank": int(provider.conditional_rank),
        "prepared_float_scalar_count": int(provider.prepared_float_scalar_count),
        "logical_macs_per_token_upper_bound": int(
            provider.logical_macs_per_token_upper_bound
        ),
        "analysis_only": role != "base",
        "raw_provider_tensors_serialized": False,
    }
    return _hashed(payload, domain=_PROVIDER_DOMAIN)


def _strict_receipt_integer(
    value: Mapping[str, object], key: str, *, label: str
) -> int:
    selected = value.get(key)
    if type(selected) is not int or selected < 0:
        raise ValueError(f"{label} {key} must be a nonnegative integer")
    return selected


def _expected_provider_accounting_from_v20i(
    authenticated_v20i_fold: Mapping[str, object], *, role: str
) -> tuple[int, int, int, int]:
    manifest = _mapping(
        authenticated_v20i_fold.get("provider_manifest"),
        label="V20k inherited V20i provider manifest",
    )
    receipts = _mapping(
        manifest.get("provider_receipts"),
        label="V20k inherited V20i provider receipts",
    )
    reference_role = {
        "base": "base",
        "fixed_plus": "fixed_plus",
        "fixed_minus": "fixed_minus",
    }.get(role, "fixed_plus")
    authority = _mapping(
        receipts.get(reference_role),
        label=f"V20k inherited V20i {reference_role} provider receipt",
    )
    rank = _strict_receipt_integer(authority, "rank", label="V20i provider")
    conditional_rank = _strict_receipt_integer(
        authority, "conditional_rank", label="V20i provider"
    )
    prepared = _strict_receipt_integer(
        authority, "prepared_float_scalar_count", label="V20i provider"
    )
    macs = _strict_receipt_integer(
        authority,
        "logical_macs_per_token_upper_bound",
        label="V20i provider",
    )
    if role not in ("base", "fixed_plus", "fixed_minus"):
        # The fixed signed-log axis control stores three response scalars.
        # LogResponse stores four direction values plus radius and mix, and
        # adds one dense projection MAC; the asinh mixture remains scalar
        # arithmetic outside the dense-MAC total.
        prepared += 3
        macs += 1
    return rank, conditional_rank, prepared, macs


def _validate_provider_receipt_evidence(
    receipt: Mapping[str, object],
    *,
    expected_role: str,
    expected_provider_artifact_sha256: str,
    expected_endpoint_receipt: Mapping[str, object],
    expected_bridge_binding_sha256: str,
    authenticated_v20i_fold: Mapping[str, object],
    expected_response: tuple[float, float] | None = None,
    expected_direction: Sequence[float] | None = None,
    expected_transfer_evidence_sha256: str | None = None,
) -> None:
    """Replay one scalar/hash provider claim against independent authority."""

    if set(receipt) != _PROVIDER_RECEIPT_KEYS:
        raise ValueError("V20k provider receipt key set differs")
    provider_artifact = _sha(
        expected_provider_artifact_sha256,
        label="V20k expected provider artifact",
    )
    if (
        receipt.get("role") != expected_role
        or receipt.get("provider_artifact_sha256") != provider_artifact
        or receipt.get("analysis_only") is not (expected_role != "base")
        or receipt.get("raw_provider_tensors_serialized") is not False
    ):
        raise ValueError("V20k provider receipt identity differs")

    metadata = _mapping(
        receipt.get("provider_metadata"), label="V20k provider metadata"
    )
    metadata_sha = _sha(
        receipt.get("provider_metadata_sha256"),
        label="V20k provider metadata hash",
    )
    if (
        _v14._sha256(metadata, domain=_PROVIDER_DOMAIN) != metadata_sha
        or metadata.get("artifact_sha256") != provider_artifact
    ):
        raise ValueError("V20k provider metadata authentication differs")

    receipt_accounting = tuple(
        _strict_receipt_integer(receipt, key, label="V20k provider receipt")
        for key in (
            "rank",
            "conditional_rank",
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
        )
    )
    metadata_accounting = tuple(
        _strict_receipt_integer(metadata, key, label="V20k provider metadata")
        for key in (
            "rank",
            "conditional_rank",
            "prepared_float_scalar_count",
            "logical_macs_per_token_upper_bound",
        )
    )
    expected_accounting = _expected_provider_accounting_from_v20i(
        authenticated_v20i_fold, role=expected_role
    )
    if (
        receipt_accounting != metadata_accounting
        or receipt_accounting != expected_accounting
    ):
        raise ValueError("V20k provider accounting differs from pinned V20i")

    if metadata.get("bridge_binding_sha256") not in (
        None,
        expected_bridge_binding_sha256,
    ):
        raise ValueError("V20k provider bridge metadata differs")

    log_response_role = expected_response is not None or expected_direction is not None
    if log_response_role:
        if expected_response is None or expected_direction is None:
            raise ValueError("V20k log_response provider expectation is incomplete")
        response = _response_pair(expected_response)
        direction = tuple(float(item) for item in expected_direction)
        if len(direction) != 4 or not all(math.isfinite(item) for item in direction):
            raise ValueError("V20k expected log_response direction differs")
        payload = _mapping(
            receipt.get("provider_payload"), label="V20k log_response provider payload"
        )
        validated = validate_fisher_soft_polarity_log_response_provider_evidence(
            payload, metadata
        )
        if _v14._canonical_json_bytes(
            validated.metadata.get("box_certificate")
        ) != _v14._canonical_json_bytes(receipt.get("box_certificate")):
            raise ValueError("V20k log_response provider box certificate differs")
        expected_direction_sha = fisher_soft_polarity_log_response_direction_sha256(
            _v20g._eta_tensor(direction)
        )
        endpoint_base = _sha(
            expected_endpoint_receipt.get("base_provider_artifact_sha256"),
            label="V20k endpoint base provider",
        )
        endpoint_proposal = _sha(
            expected_endpoint_receipt.get("proposal_provider_artifact_sha256"),
            label="V20k endpoint proposal provider",
        )
        expected_transfer = _sha(
            expected_transfer_evidence_sha256,
            label="V20k log_response transfer evidence",
        )
        expected_bindings = {
            "bridge_binding_sha256": expected_bridge_binding_sha256,
            "base_provider_artifact_sha256": endpoint_base,
            "proposal_provider_artifact_sha256": endpoint_proposal,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "transfer_evidence_sha256": expected_transfer,
            "direction_sha256": expected_direction_sha,
            "radius": response[0],
            "mix": response[1],
        }
        for key, expected in expected_bindings.items():
            if validated.payload.get(key) != expected:
                raise ValueError(f"V20k log_response provider {key} differs")
        for key in (
            "parent_provider_artifact_sha256",
            "start_provider_artifact_sha256",
        ):
            inherited = expected_endpoint_receipt.get(key)
            if inherited is not None and validated.payload.get(key) != inherited:
                raise ValueError(f"V20k log_response provider {key} differs")
        if validated.artifact_sha256 != provider_artifact:
            raise ValueError("V20k log_response provider artifact replay differs")
    else:
        if receipt.get("provider_payload") is not None:
            raise ValueError("V20k non-log_response provider serialized a payload")
        if expected_role in ("fixed_plus", "fixed_minus"):
            expected_transfer = _sha(
                expected_transfer_evidence_sha256,
                label="V20k fixed-control transfer evidence",
            )
            expected_bindings = {
                "base_provider_artifact_sha256": _sha(
                    expected_endpoint_receipt.get(
                        "base_provider_artifact_sha256"
                    ),
                    label="V20k fixed-control endpoint base",
                ),
                "proposal_provider_artifact_sha256": _sha(
                    expected_endpoint_receipt.get(
                        "proposal_provider_artifact_sha256"
                    ),
                    label="V20k fixed-control endpoint proposal",
                ),
                "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
                "transfer_evidence_sha256": expected_transfer,
            }
            for key, expected in expected_bindings.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"V20k fixed-control {key} differs")


def _provider_trace(
    provider: object, records: Sequence[object], *, role: str
) -> dict[str, object]:
    return _v20g._provider_trace(
        provider,
        records,
        arm="base" if role == "base" else role,
        artifact_domain=_TRACE_DOMAIN,
    )


def _execution_sha256(
    *,
    phase: str,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
    provider_artifact_sha256: str,
    example_id: str,
    family_id: str,
    objective: float,
    h4_sha256: str,
    logits_sha256: str,
    evidence_sha256: str,
    domain: bytes,
) -> str:
    return _v14._sha256(
        {
            "phase": phase,
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "role": role,
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


def _score_exact_provider(
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
    domain: bytes,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    objectives: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    execution_hashes: dict[str, str] = {}
    for record in _v20b._ordered_records(records):
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
            inner_family_id=inner_family_id,
            role=role,
            provider_artifact_sha256=provider.artifact_sha256,
            example_id=example,
            family_id=record.sequence.family_id,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
            evidence_sha256=evidence_sha256,
            domain=domain,
        )
        del model_inputs, teacher, execution
    return objectives, h4_hashes, logits_hashes, execution_hashes


def _freeze_inner_providers(
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    records: Sequence[object],
    *,
    outer_family_id: str,
) -> tuple[
    dict[
        str,
        dict[
            tuple[float, float],
            AutonomousCompleteH4FisherSoftPolarityLogResponseProvider,
        ],
    ],
    dict[str, object],
    dict[str, dict[tuple[float, float], dict[str, object]]],
    dict[str, dict[str, object]],
]:
    """Freeze all 7x11 inner providers and traces before any capability."""

    outer = _identifier(outer_family_id, label="V20k inner outer family")
    ordered = _v20b._ordered_records(records)
    training_families = tuple(
        sorted({record.sequence.family_id for record in ordered})
    )
    if (
        len(training_families) != _INNER_FAMILY_COUNT
        or outer in training_families
        or tuple(source_direction_receipt.get("training_family_ids", ()))
        != training_families
    ):
        raise RuntimeError("V20k inner family geometry differs")

    providers: dict[
        str,
        dict[
            tuple[float, float],
            AutonomousCompleteH4FisherSoftPolarityLogResponseProvider,
        ],
    ] = {}
    traces: dict[str, dict[tuple[float, float], dict[str, object]]] = {}
    fits: dict[str, dict[str, object]] = {}
    provider_hashes: dict[str, dict[str, str]] = {}
    trace_hashes: dict[str, dict[str, str]] = {}
    provider_receipts: dict[str, dict[str, dict[str, object]]] = {}
    transfer_evidence: dict[str, dict[str, str]] = {}

    for inner in training_families:
        masked = _reflection.build_soft_polarity_masked_direction_receipt(
            source_direction_receipt=source_direction_receipt,
            excluded_training_family_id=inner,
        )
        reflection_fit = _reflection.build_soft_polarity_reflection_fit_receipt(
            direction_receipt=masked
        )
        selected = _selected_direction(reflection_fit)
        selected_artifact = _sha(
            reflection_fit.get("selected_variant_artifact_sha256"),
            label="V20k inner selected reflection variant",
        )
        held_records = tuple(
            record for record in ordered if record.sequence.family_id == inner
        )
        if len(held_records) != _PROMPTS_PER_FAMILY:
            raise RuntimeError("V20k inner-held prompt geometry differs")

        providers[inner] = {}
        traces[inner] = {}
        provider_hashes[inner] = {}
        trace_hashes[inner] = {}
        provider_receipts[inner] = {}
        transfer_evidence[inner] = {}
        for response in _RESPONSES:
            key = _response_key(response)
            provider, seed = _materialize_provider(
                endpoint,
                direction=selected,
                direction_artifact_sha256=selected_artifact,
                reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
                response=response,
                outer_family_id=outer,
                inner_family_id=inner,
                role="inner_reflected_response_candidate",
            )
            providers[inner][response] = provider
            radius, mix = response
            traces[inner][response] = _provider_trace(
                provider,
                held_records,
                role=(
                    f"inner_{inner}_radius_{radius.hex()}_mix_{mix.hex()}"
                ),
            )
            provider_hashes[inner][key] = provider.artifact_sha256
            trace_hashes[inner][key] = str(
                traces[inner][response]["artifact_sha256"]
            )
            provider_receipts[inner][key] = _provider_receipt(
                provider,
                role="inner_reflected_response_candidate",
                response=response,
                direction=selected,
            )
            transfer_evidence[inner][key] = seed
        fits[inner] = {
            "masked_direction_receipt": masked,
            "reflection_fit_receipt": reflection_fit,
            "selected_variant_artifact_sha256": selected_artifact,
            "selected_normalized_direction": selected,
            "inner_held_family_id": inner,
            "inner_training_family_ids": tuple(
                family for family in training_families if family != inner
            ),
        }

    flat_hashes = tuple(
        provider_hashes[inner][_response_key(response)]
        for inner in training_families
        for response in _RESPONSES
    )
    if len(flat_hashes) != _INNER_FAMILY_COUNT * len(_RESPONSES) or len(
        set(flat_hashes)
    ) != len(flat_hashes):
        raise RuntimeError("V20k inner provider artifacts are not all distinct")
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "inner_family_order": training_families,
            "response_order": _RESPONSES,
            "log_response_ladder_receipt_sha256": (
                _LOG_RESPONSE_LADDER_RECEIPT_SHA256
            ),
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "source_direction_receipt_sha256": source_direction_receipt[
                "artifact_sha256"
            ],
            "masked_direction_receipt_sha256s_by_inner_family": {
                inner: fits[inner]["masked_direction_receipt"]["artifact_sha256"]
                for inner in training_families
            },
            "reflection_fit_receipt_sha256s_by_inner_family": {
                inner: fits[inner]["reflection_fit_receipt"]["artifact_sha256"]
                for inner in training_families
            },
            "selected_variant_artifact_sha256s_by_inner_family": {
                inner: fits[inner]["selected_variant_artifact_sha256"]
                for inner in training_families
            },
            "provider_artifact_sha256s_by_inner_family_and_response": provider_hashes,
            "provider_transfer_evidence_sha256s_by_inner_family_and_response": (
                transfer_evidence
            ),
            "provider_receipts_by_inner_family_and_response": provider_receipts,
            "response_trace_sha256s_by_inner_family_and_response": trace_hashes,
            "all_seven_times_eleven_providers_frozen_before_any_inner_capability": True,
            "all_seven_times_eleven_traces_frozen_before_any_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "inner_objectives_or_teacher_rows_used_at_freeze": False,
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_INNER_MANIFEST_DOMAIN,
    )
    return providers, manifest, traces, fits


def _aggregate_response_selection(
    inner_evidence_by_family: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    families = tuple(sorted(inner_evidence_by_family))
    if len(families) != _INNER_FAMILY_COUNT:
        raise ValueError("V20k response selection requires seven inner OOF families")
    outer_ids = {
        _identifier(
            inner_evidence_by_family[family].get("outer_held_family_id"),
            label="V20k response selection outer family",
        )
        for family in families
    }
    if len(outer_ids) != 1 or next(iter(outer_ids)) in families:
        raise ValueError("V20k response selection outer family geometry differs")
    outer = next(iter(outer_ids))
    all_families = tuple(sorted((*families, outer)))
    objectives_by_response: dict[str, float] = {}
    aggregate_artifacts: dict[str, str] = {}
    objectives_by_family_and_response: dict[str, dict[str, float]] = {}
    for family in families:
        raw = _mapping(
            inner_evidence_by_family[family].get("objective_by_response"),
            label="V20k inner objective ladder",
        )
        if set(raw) != set(_RESPONSE_KEYS):
            raise ValueError("V20k inner objective response geometry differs")
        objectives_by_family_and_response[family] = {
            key: float(raw[key]) for key in _RESPONSE_KEYS
        }
    for response in _RESPONSES:
        key = _response_key(response)
        objectives_by_response[key] = math.fsum(
            objectives_by_family_and_response[family][key] for family in families
        ) / len(families)
        aggregate_artifacts[key] = _v14._sha256(
            {
                "response": response,
                "inner_family_order": families,
                "inner_evidence_sha256s": {
                    family: inner_evidence_by_family[family]["artifact_sha256"]
                    for family in families
                },
                "family_objectives": {
                    family: objectives_by_family_and_response[family][key]
                    for family in families
                },
                "family_equal_objective": objectives_by_response[key],
            },
            domain=_RESPONSE_SELECTION_DOMAIN,
        )
    ladder_receipt = dict(_LOG_RESPONSE_LADDER_RECEIPT)
    exact_by_family_and_candidate = {
        family: {
            candidate_id: objectives_by_family_and_response[family][
                _response_key(response)
            ]
            for candidate_id, response in zip(
                _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_CANDIDATE_IDS,
                _RESPONSES,
                strict=True,
            )
        }
        for family in families
    }
    core_selection = (
        _log_response_fit.build_soft_polarity_log_response_inner_oof_selection_receipt(
            ladder_receipt=ladder_receipt,
            all_development_family_ids=all_families,
            outer_held_family_id=outer,
            exact_objectives_by_family_and_candidate=exact_by_family_and_candidate,
        )
    )
    selected = (
        float(core_selection["selected_r"]),
        float(core_selection["selected_lambda"]),
    )
    return _hashed(
        {
            "inner_family_order": families,
            "response_order": _RESPONSES,
            "objectives_by_inner_family_and_response": (
                objectives_by_family_and_response
            ),
            "family_equal_objective_by_response": objectives_by_response,
            "aggregate_artifact_sha256_by_response": aggregate_artifacts,
            "log_response_fit_protocol_sha256": (
                _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_FIT_PROTOCOL_SHA256
            ),
            "log_response_ladder_receipt": ladder_receipt,
            "log_response_selection_receipt": core_selection,
            "selection_rule": (
                "minimum_inner_OOF_family_equal_token_mean_exact_float64_full_"
                "vocabulary_KL_teacher_to_candidate_then_smaller_mix_then_"
                "smaller_radius_then_fixed_index_then_candidate_artifact_sha256"
            ),
            "selected_response": selected,
            "selected_family_equal_objective": objectives_by_response[
                _response_key(selected)
            ],
            "selected_aggregate_artifact_sha256": aggregate_artifacts[
                _response_key(selected)
            ],
            "all_inner_providers_frozen_before_any_inner_score": True,
            "same_family_used_for_direction_fit_and_inner_score": False,
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "inner_claim_scope": (
                "conditional_response_LOFO_not_fully_nested_model_cross_"
                "validation"
            ),
            "outer_held_family_used_for_selection": False,
        },
        domain=_RESPONSE_SELECTION_DOMAIN,
    )


def _fit_inner_response(
    context: object,
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20k inner-response outer family")
    training = _v20b._ordered_records(endpoint.training_records)
    providers, manifest, traces, fits = _freeze_inner_providers(
        endpoint,
        source_direction_receipt,
        training,
        outer_family_id=outer,
    )
    if (
        manifest.get(
            "all_seven_times_eleven_providers_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get(
            "all_seven_times_eleven_traces_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get("inner_capability_count_at_freeze") != 0
        or manifest.get("inner_objectives_or_teacher_rows_used_at_freeze")
        is not False
    ):
        raise PermissionError("V20k inner freeze barrier is not satisfied")

    inner_evidence: dict[str, dict[str, object]] = {}
    gradient_evidence = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_training_evidence"),
            label="V20k inherited fit evidence",
        ).get("gradient_evidence"),
        label="V20k inherited gradient evidence",
    )
    eta_zero_objectives = _mapping(
        gradient_evidence.get("eta_zero_objectives_by_family"),
        label="V20k inherited eta-zero objectives",
    )
    eta_zero_h4 = _mapping(
        gradient_evidence.get("post_cast_h4_sha256s"),
        label="V20k inherited eta-zero H4 hashes",
    )
    eta_zero_logits = _mapping(
        gradient_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="V20k inherited eta-zero logits hashes",
    )
    for inner in tuple(manifest["inner_family_order"]):
        held = _v20b._ordered_records(
            tuple(record for record in training if record.sequence.family_id == inner)
        )
        trace_bundle_sha = _v14._sha256(
            {
                _response_key(response): traces[inner][response]["artifact_sha256"]
                for response in _RESPONSES
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in held),
            held_family_id=outer,
        )
        objective_by_response: dict[str, float] = {}
        evidence_by_response: dict[str, dict[str, object]] = {}
        for response in _RESPONSES:
            key = _response_key(response)
            seed = _v14._sha256(
                {
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "trace_bundle_sha256": trace_bundle_sha,
                    "outer_held_family_id": outer,
                    "inner_held_family_id": inner,
                    "response": response,
                    "provider_artifact_sha256": providers[inner][
                        response
                    ].artifact_sha256,
                    "all_inner_candidates_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            objectives, h4_hashes, logits_hashes, execution_hashes = (
                _score_exact_provider(
                    context,
                    held,
                    capability,
                    provider=providers[inner][response],
                    phase="inner_conditional_leave_one_family_out_response_score",
                    outer_family_id=outer,
                    inner_family_id=inner,
                    role="inner_reflected_response_candidate",
                    evidence_sha256=seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
            )
            macro, family_scores = _v19._family_equal_mean(objectives, held)
            if set(family_scores) != {inner}:
                raise RuntimeError("V20k inner score family geometry differs")
            objective_by_response[key] = macro
            evidence_by_response[key] = _hashed(
                {
                    "outer_held_family_id": outer,
                    "inner_held_family_id": inner,
                    "response": response,
                    "provider_artifact_sha256": providers[inner][
                        response
                    ].artifact_sha256,
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "response_trace": traces[inner][response],
                    "objective": macro,
                    "objectives_by_example": dict(sorted(objectives.items())),
                    "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                    "supervised_full_vocab_logits_sha256s": dict(
                        sorted(logits_hashes.items())
                    ),
                    "execution_sha256s": dict(sorted(execution_hashes.items())),
                    "exact_execution": True,
                    "finite": True,
                    "inner_family_absent_from_direction_and_reflection_fit": True,
                    "outer_family_absent_from_endpoint_direction_and_score": True,
                    "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
        capability_receipt = capability.receipt()
        _v20b._validate_capability_receipt(
            capability_receipt,
            expected_example_ids=tuple(
                record.sequence.example_id for record in held
            ),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=len(_RESPONSES),
            label="V20k inner-held capability",
        )
        zero = evidence_by_response[_response_key((0.0, 0.0))]
        zero_objectives = _mapping(
            zero.get("objectives_by_example"),
            label="V20k inner eta-zero objectives",
        )
        zero_h4 = _mapping(
            zero.get("post_cast_h4_sha256s"),
            label="V20k inner eta-zero H4 hashes",
        )
        zero_logits = _mapping(
            zero.get("supervised_full_vocab_logits_sha256s"),
            label="V20k inner eta-zero logits hashes",
        )
        expected_zero_objectives = _mapping(
            eta_zero_objectives.get(inner),
            label="V20k inherited family eta-zero objectives",
        )
        zero_anchor = (
            dict(zero_objectives) == dict(expected_zero_objectives)
            and dict(zero_h4)
            == {
                example: eta_zero_h4[example]
                for example in sorted(expected_zero_objectives)
            }
            and dict(zero_logits)
            == {
                example: eta_zero_logits[example]
                for example in sorted(expected_zero_objectives)
            }
        )
        if not zero_anchor:
            raise RuntimeError("V20k inner eta-zero output anchor differs from V20g")
        inner_evidence[inner] = _hashed(
            {
                "outer_held_family_id": outer,
                "inner_held_family_id": inner,
                "inner_training_family_ids": fits[inner][
                    "inner_training_family_ids"
                ],
                "masked_direction_receipt": fits[inner][
                    "masked_direction_receipt"
                ],
                "reflection_fit_receipt": fits[inner]["reflection_fit_receipt"],
                "selected_variant_artifact_sha256": fits[inner][
                    "selected_variant_artifact_sha256"
                ],
                "response_order": _RESPONSES,
                "objective_by_response": objective_by_response,
                "response_evidence": evidence_by_response,
                "capability_receipt": capability_receipt,
                "exact_execution_count": len(_RESPONSES) * len(held),
                "zero_response_exact_v20g_eta_zero_output_anchor": zero_anchor,
                "all_inner_candidates_frozen_before_capability": True,
                "held_family_used_for_direction_or_reflection_fit": False,
                "held_family_used_for_endpoint_fit": True,
                "endpoint_retrained_without_held_inner_family": False,
                "raw_prompts_tokens_logits_h4_gradients_or_teacher_rows_"
                "serialized": False,
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )

    selection = _aggregate_response_selection(inner_evidence)
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "source_direction_receipt_sha256": source_direction_receipt[
                "artifact_sha256"
            ],
            "inner_provider_manifest": manifest,
            "inner_evidence_by_family": inner_evidence,
            "response_selection_receipt": selection,
            "inner_family_order": manifest["inner_family_order"],
            "response_order": _RESPONSES,
            "all_inner_fits_and_providers_frozen_before_any_inner_capability": True,
            "exact_inner_execution_count": (
                _INNER_FAMILY_COUNT * len(_RESPONSES) * _PROMPTS_PER_FAMILY
            ),
            "inner_endpoint_retrained_per_fold": False,
            "inner_held_family_used_for_endpoint_fit": True,
            "inner_claim_scope": (
                "conditional_response_LOFO_not_fully_nested_model_cross_"
                "validation"
            ),
            "outer_held_family_used_for_fit_or_selection": False,
            "raw_provider_gradient_logits_h4_or_teacher_tensors_serialized": False,
        },
        domain=_INNER_FIT_DOMAIN,
    )
    return receipt, selection


def _freeze_outer_providers(
    endpoint: _v20g._EndpointLive,
    source_direction_receipt: Mapping[str, object],
    outer_reflection_fit: Mapping[str, object],
    held_records: Sequence[object],
    *,
    selected_response: tuple[float, float],
    outer_family_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    outer = _identifier(outer_family_id, label="V20k outer provider family")
    response = _response_pair(selected_response)
    linear_response = (response[0], 0.0)
    reflected = _selected_direction(outer_reflection_fit)
    unreflected = _unreflected_direction(source_direction_receipt)
    mirror = tuple(-item for item in reflected)
    selected_variant_artifact = _sha(
        outer_reflection_fit.get("selected_variant_artifact_sha256"),
        label="V20k outer reflection variant",
    )
    source_direction_artifact = _sha(
        source_direction_receipt.get("artifact_sha256"),
        label="V20k outer unreflected direction",
    )
    reflection_fit_artifact = _sha(
        outer_reflection_fit.get("artifact_sha256"),
        label="V20k outer reflection fit",
    )

    reflected_provider, reflected_seed = _materialize_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_log_response_reflected",
    )
    unreflected_provider, unreflected_seed = _materialize_provider(
        endpoint,
        direction=unreflected,
        direction_artifact_sha256=source_direction_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_same_response_unreflected",
    )
    mirror_provider, mirror_seed = _materialize_provider(
        endpoint,
        direction=mirror,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_log_response_reflected_exact_mirror",
    )
    linear_provider, linear_seed = _materialize_provider(
        endpoint,
        direction=reflected,
        direction_artifact_sha256=selected_variant_artifact,
        reflection_fit_sha256=reflection_fit_artifact,
        response=linear_response,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_matched_linear_reflected",
    )
    control_seed = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "outer_held_family_id": outer,
            "reflection_fit_sha256": reflection_fit_artifact,
            "selected_response": response,
            "role": "outer_fixed_controls",
            "held_rows_used": False,
        },
        domain=_OUTER_MANIFEST_DOMAIN,
    )
    providers: dict[str, object] = {
        "base": endpoint.base_provider,
        "fixed_plus": (
            build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
                endpoint.base_provider,
                endpoint.proposal_provider,
                polarity=1,
                transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
                transfer_evidence_sha256=control_seed,
            )
        ),
        "fixed_minus": (
            build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
                endpoint.base_provider,
                endpoint.proposal_provider,
                polarity=-1,
                transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
                transfer_evidence_sha256=control_seed,
            )
        ),
        "matched_linear_reflected": linear_provider,
        "same_response_unreflected": unreflected_provider,
        "log_response_reflected": reflected_provider,
        "log_response_reflected_exact_mirror": mirror_provider,
    }
    if tuple(providers) != _ARMS or len(
        {provider.artifact_sha256 for provider in providers.values()}
    ) != len(_ARMS):
        raise RuntimeError("V20k outer provider arm artifacts are not distinct")

    receipts = {
        "base": _provider_receipt(providers["base"], role="base"),
        "fixed_plus": _provider_receipt(
            providers["fixed_plus"], role="fixed_plus"
        ),
        "fixed_minus": _provider_receipt(
            providers["fixed_minus"], role="fixed_minus"
        ),
        "matched_linear_reflected": _provider_receipt(
            providers["matched_linear_reflected"],
            role="matched_linear_reflected",
            response=linear_response,
            direction=reflected,
        ),
        "same_response_unreflected": _provider_receipt(
            providers["same_response_unreflected"],
            role="same_response_unreflected",
            response=response,
            direction=unreflected,
        ),
        "log_response_reflected": _provider_receipt(
            providers["log_response_reflected"],
            role="log_response_reflected",
            response=response,
            direction=reflected,
        ),
        "log_response_reflected_exact_mirror": _provider_receipt(
            providers["log_response_reflected_exact_mirror"],
            role="log_response_reflected_exact_mirror",
            response=response,
            direction=mirror,
        ),
    }
    traces = {
        arm: _provider_trace(providers[arm], held_records, role=arm)
        for arm in _ARMS
    }
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "source_direction_receipt_sha256": source_direction_artifact,
            "outer_reflection_fit_receipt_sha256": reflection_fit_artifact,
            "selected_variant_artifact_sha256": selected_variant_artifact,
            "selected_response": response,
            "matched_linear_response": linear_response,
            "arm_order": _ARMS,
            "provider_artifact_sha256s": {
                arm: providers[arm].artifact_sha256 for arm in _ARMS
            },
            "provider_receipts": receipts,
            "response_trace_sha256s": {
                arm: traces[arm]["artifact_sha256"] for arm in _ARMS
            },
            "soft_provider_transfer_evidence_sha256s": {
                "matched_linear_reflected": linear_seed,
                "same_response_unreflected": unreflected_seed,
                "log_response_reflected": reflected_seed,
                "log_response_reflected_exact_mirror": mirror_seed,
            },
            "fixed_control_transfer_evidence_sha256": control_seed,
            "all_seven_providers_frozen_before_outer_capability": True,
            "all_seven_traces_frozen_before_outer_capability": True,
            "outer_capability_count_at_freeze": 0,
            "outer_objectives_or_teacher_rows_used_at_freeze": False,
            "raw_provider_or_response_tensors_serialized": False,
        },
        domain=_OUTER_MANIFEST_DOMAIN,
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
    selected_response: tuple[float, float],
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20k held outer family")
    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20k outer-held prompt geometry differs")
    providers, manifest, traces = _freeze_outer_providers(
        endpoint,
        source_direction_receipt,
        outer_reflection_fit,
        held,
        selected_response=selected_response,
        outer_family_id=outer,
    )
    if (
        manifest.get("all_seven_providers_frozen_before_outer_capability") is not True
        or manifest.get("all_seven_traces_frozen_before_outer_capability") is not True
        or manifest.get("outer_capability_count_at_freeze") != 0
        or manifest.get("outer_objectives_or_teacher_rows_used_at_freeze")
        is not False
    ):
        raise PermissionError("V20k outer freeze barrier is not satisfied")

    trace_bundle_sha = _v14._sha256(
        {arm: traces[arm]["artifact_sha256"] for arm in _ARMS},
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in held), held_family_id=None
    )
    objective_by_arm: dict[str, float] = {}
    evidence_by_arm: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        seed = _v14._sha256(
            {
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "all_outer_arms_frozen": True,
            },
            domain=_OUTER_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=providers[arm],
                phase="outer_family_disjoint_mechanism_score",
                outer_family_id=outer,
                inner_family_id=None,
                role=arm,
                evidence_sha256=seed,
                domain=_OUTER_EXECUTION_DOMAIN,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {outer}:
            raise RuntimeError("V20k outer score family geometry differs")
        objective_by_arm[arm] = macro
        evidence_by_arm[arm] = _hashed(
            {
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": providers[arm].artifact_sha256,
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "response_trace": traces[arm],
                "objective": macro,
                "objectives_by_example": dict(sorted(objectives.items())),
                "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
                "supervised_full_vocab_logits_sha256s": dict(
                    sorted(logits_hashes.items())
                ),
                "execution_sha256s": dict(sorted(execution_hashes.items())),
                "exact_execution": True,
                "finite": True,
                "raw_logits_h4_teacher_rows_or_tensors_serialized": False,
            },
            domain=_OUTER_EXECUTION_DOMAIN,
        )
    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(record.sequence.example_id for record in held),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20k outer-held capability",
    )
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20k inherited V20g held evidence",
        ).get("arm_evidence"),
        label="V20k inherited V20g held arms",
    )
    control_anchors: dict[str, bool] = {}
    for arm in ("base", "fixed_plus", "fixed_minus"):
        inherited = _mapping(
            inherited_arms.get(arm), label=f"V20k inherited V20g {arm} arm"
        )
        current = evidence_by_arm[arm]
        control_anchors[arm] = (
            float(current["objective"]) == float(inherited["objective"])
            and dict(
                _mapping(
                    current.get("objectives_by_example"),
                    label=f"V20k current {arm} objectives",
                )
            )
            == dict(
                _mapping(
                    inherited.get("objectives_by_example"),
                    label=f"V20k inherited {arm} objectives",
                )
            )
            and dict(
                _mapping(
                    current.get("post_cast_h4_sha256s"),
                    label=f"V20k current {arm} H4 hashes",
                )
            )
            == dict(
                _mapping(
                    inherited.get("post_cast_h4_sha256s"),
                    label=f"V20k inherited {arm} H4 hashes",
                )
            )
            and dict(
                _mapping(
                    current.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20k current {arm} logits hashes",
                )
            )
            == dict(
                _mapping(
                    inherited.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20k inherited {arm} logits hashes",
                )
            )
        )
    if not all(control_anchors.values()):
        raise RuntimeError("V20k outer control output anchor differs from V20g")
    base_logits = _mapping(
        evidence_by_arm["base"].get("supervised_full_vocab_logits_sha256s"),
        label="V20k base output hashes",
    )
    candidate_logits = _mapping(
        evidence_by_arm[_PRIMARY_ARM].get(
            "supervised_full_vocab_logits_sha256s"
        ),
        label="V20k candidate output hashes",
    )
    candidate_changed = any(
        candidate_logits[example] != base_logits[example] for example in base_logits
    )
    health = all(
        evidence_by_arm[arm]["finite"] is True
        and traces[arm]["finite"] is True
        and traces[arm]["pointwise_trust_passed"] is True
        and traces[arm]["endpoint_conditional_ranks_are_16"] is True
        for arm in _ARMS
    )
    held_evidence = _hashed(
        {
            "outer_held_family_id": outer,
            "outer_manifest_sha256": manifest["artifact_sha256"],
            "arm_evidence": evidence_by_arm,
            "capability_receipt": capability_receipt,
            "all_seven_providers_and_traces_frozen_before_outer_capability": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_outer_execution_count": len(_ARMS) * len(held),
            "v20g_control_output_anchors": control_anchors,
            "all_v20g_control_output_anchors_passed": all(
                control_anchors.values()
            ),
            "raw_prompts_tokens_logits_h4_or_teacher_rows_serialized": False,
        },
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    fold_receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "selected_response": _response_pair(selected_response),
            "selected_response_key": _response_key(selected_response),
            "selected_variant_id": outer_reflection_fit["selected_variant_id"],
            "selected_variant_artifact_sha256": outer_reflection_fit[
                "selected_variant_artifact_sha256"
            ],
            "arm_order": _ARMS,
            "held_objective_by_arm": objective_by_arm,
            "candidate_provider_artifact_sha256": manifest[
                "provider_artifact_sha256s"
            ][_PRIMARY_ARM],
            "base_provider_artifact_sha256": manifest[
                "provider_artifact_sha256s"
            ]["base"],
            "candidate_provider_distinct_from_base": (
                manifest["provider_artifact_sha256s"][_PRIMARY_ARM]
                != manifest["provider_artifact_sha256s"]["base"]
            ),
            "candidate_exact_execution_changed_from_base": candidate_changed,
            "selected_radius_positive": _response_pair(selected_response)[0] > 0.0,
            "selected_mix_positive": _response_pair(
                selected_response
            )[1]
            > 0.0,
            "all_runtime_health_passed": health,
            "all_v20g_control_output_anchors_passed": all(
                control_anchors.values()
            ),
            "selection_frozen_before_outer_score": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_execution": True,
        },
        domain=_DECISION_DOMAIN,
    )
    return manifest, held_evidence, fold_receipt


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
) -> _FoldLive:
    outer = _identifier(outer_family_id, label="V20k outer family")
    endpoint = _v20g._outer_endpoint(
        context,
        records,
        teacher_vault,
        family_ids=family_ids,
        outer_family_id=outer,
        panel_receipt=panel_receipt,
        authenticated_v20a_fold=authenticated_v20a_fold,
    )
    inherited_endpoint = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20k inherited endpoint receipt",
    )
    inherited_evidence = _mapping(
        authenticated_v20g_fold.get("endpoint_evidence"),
        label="V20k inherited endpoint evidence",
    )
    if (
        _v14._canonical_json_bytes(endpoint.receipt)
        != _v14._canonical_json_bytes(inherited_endpoint)
        or _v14._canonical_json_bytes(endpoint.evidence)
        != _v14._canonical_json_bytes(inherited_evidence)
    ):
        raise RuntimeError("V20k reconstructed endpoint differs from pinned V20g")
    fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"),
        label="V20k inherited V20g fit receipt",
    )
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20k inherited V20g direction"
    )
    _v20g._core.validate_soft_polarity_direction_receipt(source_direction)
    if source_direction.get("held_family_id") != outer:
        raise RuntimeError("V20k inherited direction held family differs")

    inner_receipt, response_selection = _fit_inner_response(
        context,
        endpoint,
        source_direction,
        teacher_vault,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
    )
    outer_reflection_fit = _reflection.build_soft_polarity_reflection_fit_receipt(
        direction_receipt=source_direction
    )
    _validate_v20i_reflection_lineage(
        inner_receipt=inner_receipt,
        outer_reflection_fit=outer_reflection_fit,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    selected_response = _response_pair(
        response_selection["selected_response"]
    )
    provider_manifest, held_evidence, fold_receipt = _score_outer_arms(
        context,
        endpoint,
        records,
        teacher_vault,
        source_direction,
        outer_reflection_fit,
        selected_response=selected_response,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
    )
    return _FoldLive(
        endpoint=endpoint,
        inner_receipt=inner_receipt,
        outer_reflection_fit=outer_reflection_fit,
        response_selection=response_selection,
        provider_manifest=provider_manifest,
        held_evidence=held_evidence,
        fold_receipt=fold_receipt,
    )


_FOLD_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "masked_direction_protocol_sha256",
        "log_response_fit_protocol_sha256",
        "exact_objective_kind",
        "source_artifact_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "v20g_fold_fragment_sha256",
        "v20i_fold_fragment_sha256",
        "v20j_fold_fragment_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "inner_receipt",
        "outer_reflection_fit_receipt",
        "response_selection_receipt",
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
) -> dict[str, object]:
    return {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "masked_direction_protocol_sha256": (
            _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
        ),
        "log_response_fit_protocol_sha256": (
            _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "exact_objective_kind": (
            "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
        ),
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold[
            "fragment_sha256"
        ],
        "v20i_fold_fragment_sha256": authenticated_v20i_fold[
            "fragment_sha256"
        ],
        "v20j_fold_fragment_sha256": source[
            "v20j_fold_fragment_sha256s_by_family"
        ][outer_family_id],
        "outer_held_family_id": outer_family_id,
        "endpoint_receipt": live.endpoint.receipt,
        "endpoint_evidence": live.endpoint.evidence,
        "inner_receipt": live.inner_receipt,
        "outer_reflection_fit_receipt": live.outer_reflection_fit,
        "response_selection_receipt": live.response_selection,
        "provider_manifest": live.provider_manifest,
        "held_evidence": live.held_evidence,
        "fold_receipt": live.fold_receipt,
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
    }


def _validate_exact_score_bundle(
    evidence: Mapping[str, object],
    *,
    expected_example_ids: Sequence[str],
    label: str,
) -> tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]]:
    examples = tuple(
        sorted(
            _identifier(item, label=f"{label} example")
            for item in expected_example_ids
        )
    )
    if len(examples) != _PROMPTS_PER_FAMILY or len(set(examples)) != len(examples):
        raise ValueError(f"{label} example geometry differs")
    objectives = {
        _identifier(example, label=f"{label} objective example"): float(value)
        for example, value in _mapping(
            evidence.get("objectives_by_example"),
            label=f"{label} objectives",
        ).items()
    }
    hashes: list[dict[str, str]] = []
    for field, field_label in (
        ("post_cast_h4_sha256s", "H4"),
        ("supervised_full_vocab_logits_sha256s", "logits"),
        ("execution_sha256s", "execution"),
    ):
        values = {
            _identifier(example, label=f"{label} {field_label} example"): _sha(
                value, label=f"{label} {field_label} hash"
            )
            for example, value in _mapping(
                evidence.get(field), label=f"{label} {field_label} hashes"
            ).items()
        }
        hashes.append(values)
    h4_hashes, logits_hashes, execution_hashes = hashes
    if (
        set(objectives) != set(examples)
        or set(h4_hashes) != set(examples)
        or set(logits_hashes) != set(examples)
        or set(execution_hashes) != set(examples)
        or not all(math.isfinite(value) for value in objectives.values())
    ):
        raise ValueError(f"{label} exact output geometry differs")
    macro = math.fsum(objectives[example] for example in examples) / len(examples)
    if not math.isfinite(macro) or float(evidence.get("objective", math.nan)) != macro:
        raise ValueError(f"{label} exact objective replay differs")
    return objectives, h4_hashes, logits_hashes, execution_hashes


def _validate_inner_receipt(
    value: Mapping[str, object],
    *,
    source_direction: Mapping[str, object],
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
) -> Mapping[str, object]:
    receipt = _validate_hashed(
        value, domain=_INNER_FIT_DOMAIN, label="V20k inner receipt"
    )
    endpoint_receipt = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20k inherited endpoint receipt",
    )
    inherited_bridge = authenticated_v20g_fold.get("bridge_binding_sha256")
    if inherited_bridge is None:
        inherited_bridge = endpoint_receipt.get("bridge_binding_sha256")
    if inherited_bridge is None:
        inherited_bridge = source_direction.get("bridge_binding_sha256")
    expected_bridge_binding = _sha(
        inherited_bridge,
        label="V20k inherited bridge binding",
    )
    if (
        receipt.get("outer_held_family_id") != outer_family_id
        or receipt.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or receipt.get(
            "all_inner_fits_and_providers_frozen_before_any_inner_capability"
        )
        is not True
        or _response_order(receipt.get("response_order", ())) != _RESPONSES
        or int(receipt.get("exact_inner_execution_count", -1))
        != _INNER_FAMILY_COUNT * len(_RESPONSES) * _PROMPTS_PER_FAMILY
        or receipt.get("inner_endpoint_retrained_per_fold") is not False
        or receipt.get("inner_held_family_used_for_endpoint_fit") is not True
        or receipt.get("inner_claim_scope")
        != "conditional_response_LOFO_not_fully_nested_model_cross_validation"
        or receipt.get("outer_held_family_used_for_fit_or_selection") is not False
        or receipt.get("raw_provider_gradient_logits_h4_or_teacher_tensors_serialized")
        is not False
    ):
        raise ValueError("V20k inner receipt boundary differs")
    manifest = _validate_hashed(
        _mapping(
            receipt.get("inner_provider_manifest"),
            label="V20k inner provider manifest",
        ),
        domain=_INNER_MANIFEST_DOMAIN,
        label="V20k inner provider manifest",
    )
    families = tuple(
        _identifier(item, label="V20k inner family")
        for item in _sequence(
            manifest.get("inner_family_order"), label="V20k inner family order"
        )
    )
    source_families = tuple(
        _identifier(item, label="V20k source training family")
        for item in _sequence(
            source_direction.get("training_family_ids"),
            label="V20k source training families",
        )
    )
    training_ids_by_family = _mapping(
        source_direction.get("training_example_ids_by_family"),
        label="V20k source training example ids",
    )
    if (
        len(families) != _INNER_FAMILY_COUNT
        or len(set(families)) != len(families)
        or families != source_families
        or manifest.get("outer_held_family_id") != outer_family_id
        or _response_order(manifest.get("response_order", ())) != _RESPONSES
        or manifest.get("log_response_ladder_receipt_sha256")
        != _LOG_RESPONSE_LADDER_RECEIPT_SHA256
        or manifest.get("endpoint_receipt_sha256")
        != endpoint_receipt.get("artifact_sha256")
        or manifest.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or manifest.get(
            "all_seven_times_eleven_providers_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get(
            "all_seven_times_eleven_traces_frozen_before_any_inner_capability"
        )
        is not True
        or manifest.get("inner_capability_count_at_freeze") != 0
        or manifest.get("inner_objectives_or_teacher_rows_used_at_freeze")
        is not False
        or manifest.get("inner_endpoint_retrained_per_fold") is not False
        or manifest.get("inner_held_family_used_for_endpoint_fit") is not True
        or manifest.get("raw_provider_or_response_tensors_serialized") is not False
    ):
        raise ValueError("V20k inner manifest freeze geometry differs")
    if tuple(receipt.get("inner_family_order", ())) != families:
        raise ValueError("V20k inner receipt family order differs")
    masked_hashes = _mapping(
        manifest.get("masked_direction_receipt_sha256s_by_inner_family"),
        label="V20k inner masked receipt hashes",
    )
    fit_hashes = _mapping(
        manifest.get("reflection_fit_receipt_sha256s_by_inner_family"),
        label="V20k inner reflection fit hashes",
    )
    variant_hashes = _mapping(
        manifest.get("selected_variant_artifact_sha256s_by_inner_family"),
        label="V20k inner selected variant hashes",
    )
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s_by_inner_family_and_response"),
        label="V20k inner provider hashes",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts_by_inner_family_and_response"),
        label="V20k inner provider receipts",
    )
    transfer_hashes = _mapping(
        manifest.get(
            "provider_transfer_evidence_sha256s_by_inner_family_and_response"
        ),
        label="V20k inner provider transfer hashes",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s_by_inner_family_and_response"),
        label="V20k inner trace hashes",
    )
    family_set = set(families)
    if any(
        set(values) != family_set
        for values in (
            masked_hashes,
            fit_hashes,
            variant_hashes,
            provider_hashes,
            provider_receipts,
            transfer_hashes,
            trace_hashes,
        )
    ):
        raise ValueError("V20k inner manifest family bindings differ")
    gradient_evidence = _mapping(
        _mapping(
            authenticated_v20g_fold.get("fit_training_evidence"),
            label="V20k inherited fit evidence",
        ).get("gradient_evidence"),
        label="V20k inherited gradient evidence",
    )
    inherited_zero_objectives = _mapping(
        gradient_evidence.get("eta_zero_objectives_by_family"),
        label="V20k inherited eta-zero objectives",
    )
    inherited_zero_h4 = _mapping(
        gradient_evidence.get("post_cast_h4_sha256s"),
        label="V20k inherited eta-zero H4 hashes",
    )
    inherited_zero_logits = _mapping(
        gradient_evidence.get("supervised_full_vocab_logits_sha256s"),
        label="V20k inherited eta-zero logits hashes",
    )
    if set(inherited_zero_objectives) != family_set:
        raise ValueError("V20k inherited eta-zero family geometry differs")
    raw_inner = _mapping(
        receipt.get("inner_evidence_by_family"),
        label="V20k inner evidence map",
    )
    if set(raw_inner) != set(families):
        raise ValueError("V20k inner evidence family geometry differs")
    validated_inner: dict[str, Mapping[str, object]] = {}
    for family in families:
        evidence = _validate_hashed(
            _mapping(raw_inner[family], label="V20k inner family evidence"),
            domain=_INNER_EXECUTION_DOMAIN,
            label="V20k inner family evidence",
        )
        masked = _mapping(
            evidence.get("masked_direction_receipt"),
            label="V20k masked direction receipt",
        )
        _reflection.validate_soft_polarity_masked_direction_receipt(
            masked,
            source_direction_receipt=source_direction,
            expected_excluded_training_family_id=family,
        )
        reflection_fit = _mapping(
            evidence.get("reflection_fit_receipt"),
            label="V20k inner reflection fit",
        )
        _reflection.validate_soft_polarity_reflection_fit_receipt(
            reflection_fit, direction_receipt=masked
        )
        expected_inner_training = tuple(item for item in families if item != family)
        expected_examples = tuple(
            _identifier(item, label="V20k inner expected example")
            for item in _sequence(
                training_ids_by_family.get(family),
                label="V20k inner expected examples",
            )
        )
        objectives = _mapping(
            evidence.get("objective_by_response"),
            label="V20k inner objectives",
        )
        response_evidence = _mapping(
            evidence.get("response_evidence"),
            label="V20k inner response evidence",
        )
        if (
            set(objectives) != set(_RESPONSE_KEYS)
            or set(response_evidence) != set(_RESPONSE_KEYS)
            or evidence.get("outer_held_family_id") != outer_family_id
            or evidence.get("inner_held_family_id") != family
            or tuple(evidence.get("inner_training_family_ids", ()))
            != expected_inner_training
            or masked.get("artifact_sha256") != masked_hashes[family]
            or reflection_fit.get("artifact_sha256") != fit_hashes[family]
            or evidence.get("selected_variant_artifact_sha256")
            != variant_hashes[family]
            or evidence.get("held_family_used_for_direction_or_reflection_fit")
            is not False
            or evidence.get("held_family_used_for_endpoint_fit") is not True
            or evidence.get("endpoint_retrained_without_held_inner_family")
            is not False
            or evidence.get("all_inner_candidates_frozen_before_capability")
            is not True
            or evidence.get("zero_response_exact_v20g_eta_zero_output_anchor")
            is not True
            or int(evidence.get("exact_execution_count", -1))
            != len(_RESPONSES) * _PROMPTS_PER_FAMILY
            or evidence.get(
                "raw_prompts_tokens_logits_h4_gradients_or_teacher_rows_serialized"
            )
            is not False
        ):
            raise ValueError("V20k inner evidence schedule differs")
        family_provider_hashes = _mapping(
            provider_hashes[family], label="V20k inner family provider hashes"
        )
        family_provider_receipts = _mapping(
            provider_receipts[family], label="V20k inner family provider receipts"
        )
        family_transfer_hashes = _mapping(
            transfer_hashes[family],
            label="V20k inner family provider transfer hashes",
        )
        family_trace_hashes = _mapping(
            trace_hashes[family], label="V20k inner family trace hashes"
        )
        if any(
            set(values) != set(_RESPONSE_KEYS)
            for values in (
                family_provider_hashes,
                family_provider_receipts,
                family_transfer_hashes,
                family_trace_hashes,
            )
        ):
            raise ValueError("V20k inner manifest response bindings differ")
        trace_bundle_sha = _v14._sha256(
            {
                _response_key(response): family_trace_hashes[_response_key(response)]
                for response in _RESPONSES
            },
            domain=_INNER_EXECUTION_DOMAIN,
        )
        selected_direction = _selected_direction(reflection_fit)
        for response in _RESPONSES:
            key = _response_key(response)
            arm = _validate_hashed(
                _mapping(
                    response_evidence[key], label="V20k inner response arm evidence"
                ),
                domain=_INNER_EXECUTION_DOMAIN,
                label="V20k inner response arm evidence",
            )
            trace = _validate_hashed(
                _mapping(arm.get("response_trace"), label="V20k inner trace"),
                domain=_TRACE_DOMAIN,
                label="V20k inner trace",
            )
            provider_receipt = _validate_hashed(
                _mapping(
                    family_provider_receipts[key],
                    label="V20k inner provider receipt",
                ),
                domain=_PROVIDER_DOMAIN,
                label="V20k inner provider receipt",
            )
            score_bundle = _validate_exact_score_bundle(
                arm,
                expected_example_ids=expected_examples,
                label=f"V20k inner {family} response {key}",
            )
            arm_objectives, arm_h4, arm_logits, arm_executions = score_bundle
            provider_artifact = _sha(
                family_provider_hashes[key], label="V20k inner provider hash"
            )
            transfer_artifact = _sha(
                family_transfer_hashes[key],
                label="V20k inner provider transfer hash",
            )
            expected_transfer_artifact = _provider_seed(
                endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
                direction_artifact_sha256=str(
                    reflection_fit["selected_variant_artifact_sha256"]
                ),
                reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
                response=response,
                direction=selected_direction,
                outer_family_id=outer_family_id,
                inner_family_id=family,
                role="inner_reflected_response_candidate",
            )
            _validate_provider_receipt_evidence(
                provider_receipt,
                expected_role="inner_reflected_response_candidate",
                expected_provider_artifact_sha256=provider_artifact,
                expected_endpoint_receipt=endpoint_receipt,
                expected_bridge_binding_sha256=expected_bridge_binding,
                authenticated_v20i_fold=authenticated_v20i_fold,
                expected_response=response,
                expected_direction=selected_direction,
                expected_transfer_evidence_sha256=expected_transfer_artifact,
            )
            trace_artifact = _sha(
                family_trace_hashes[key], label="V20k inner trace hash"
            )
            seed = _v14._sha256(
                {
                    "inner_manifest_sha256": manifest["artifact_sha256"],
                    "trace_bundle_sha256": trace_bundle_sha,
                    "outer_held_family_id": outer_family_id,
                    "inner_held_family_id": family,
                    "response": response,
                    "provider_artifact_sha256": provider_artifact,
                    "all_inner_candidates_frozen": True,
                },
                domain=_INNER_EXECUTION_DOMAIN,
            )
            expected_executions = {
                example: _execution_sha256(
                    phase="inner_conditional_leave_one_family_out_response_score",
                    outer_family_id=outer_family_id,
                    inner_family_id=family,
                    role="inner_reflected_response_candidate",
                    provider_artifact_sha256=provider_artifact,
                    example_id=example,
                    family_id=family,
                    objective=arm_objectives[example],
                    h4_sha256=arm_h4[example],
                    logits_sha256=arm_logits[example],
                    evidence_sha256=seed,
                    domain=_INNER_EXECUTION_DOMAIN,
                )
                for example in expected_examples
            }
            response_gain_hashes = _mapping(
                trace.get("response_gain_sha256s"),
                label="V20k inner response gain hashes",
            )
            for value in response_gain_hashes.values():
                _sha(value, label="V20k inner response gain hash")
            expected_corners = _box_corner_scores(selected_direction)
            expected_certificate = fisher_soft_polarity_log_response_box_certificate(
                _v20g._eta_tensor(selected_direction),
                radius=response[0],
                mix=response[1],
            )
            expected_trace_arm = (
                f"inner_{family}_radius_{response[0].hex()}_mix_{response[1].hex()}"
            )
            if (
                _response_pair(arm.get("response")) != response
                or float(arm.get("objective", math.nan))
                != float(objectives[key])
                or arm.get("provider_artifact_sha256") != provider_artifact
                or arm.get("inner_manifest_sha256")
                != manifest.get("artifact_sha256")
                or trace.get("artifact_sha256") != trace_artifact
                or trace.get("provider_artifact_sha256") != provider_artifact
                or trace.get("arm") != expected_trace_arm
                or tuple(trace.get("scored_family_ids", ())) != (family,)
                or set(response_gain_hashes) != set(expected_examples)
                or provider_receipt.get("provider_artifact_sha256")
                != provider_artifact
                or transfer_artifact != expected_transfer_artifact
                or provider_receipt.get("transfer_protocol_sha256")
                != _TRANSFER_PROTOCOL_SHA256
                or provider_receipt.get("transfer_evidence_sha256")
                != transfer_artifact
                or provider_receipt.get("role")
                != "inner_reflected_response_candidate"
                or _response_pair(provider_receipt.get("response"))
                != response
                or provider_receipt.get("response_key") != key
                or float(
                    provider_receipt.get("radius", math.nan)
                )
                != response[0]
                or float(
                    provider_receipt.get("mix", math.nan)
                )
                != response[1]
                or tuple(provider_receipt.get("direction", ()))
                != selected_direction
                or tuple(
                    provider_receipt.get("direction_box_corner_scores", ())
                )
                != expected_corners
                or _v14._canonical_json_bytes(
                    _mapping(
                        provider_receipt.get("box_certificate"),
                        label="V20k log_response box certificate",
                    )
                )
                != _v14._canonical_json_bytes(expected_certificate)
                or int(provider_receipt.get("conditional_rank", -1))
                != _CONDITIONAL_RANK
                or provider_receipt.get("analysis_only") is not True
                or provider_receipt.get("raw_provider_tensors_serialized")
                is not False
                or arm_executions != expected_executions
                or arm.get("exact_execution") is not True
                or arm.get("finite") is not True
                or arm.get("raw_logits_h4_teacher_rows_or_tensors_serialized")
                is not False
                or trace.get("finite") is not True
                or trace.get("pointwise_trust_passed") is not True
                or trace.get("endpoint_conditional_ranks_are_16") is not True
                or trace.get("raw_response_or_modal_tensors_serialized")
                is not False
            ):
                raise ValueError("V20k inner response evidence differs")
        _v20b._validate_capability_receipt(
            evidence.get("capability_receipt"),
            expected_example_ids=expected_examples,
            expected_family_count=1,
            expected_held_family_id=outer_family_id,
            expected_accesses_per_example=len(_RESPONSES),
            label="V20k resumed inner-held capability",
        )
        zero = _mapping(
            response_evidence[_response_key((0.0, 0.0))],
            label="V20k inner zero-response evidence",
        )
        expected_zero_objectives = {
            _identifier(example, label="V20k inherited zero objective example"): float(
                objective
            )
            for example, objective in _mapping(
                inherited_zero_objectives[family],
                label="V20k inherited family zero objectives",
            ).items()
        }
        zero_anchor = (
            dict(
                _mapping(
                    zero.get("objectives_by_example"),
                    label="V20k zero objectives",
                )
            )
            == expected_zero_objectives
            and dict(
                _mapping(
                    zero.get("post_cast_h4_sha256s"),
                    label="V20k zero H4 hashes",
                )
            )
            == {
                example: inherited_zero_h4[example]
                for example in expected_examples
            }
            and dict(
                _mapping(
                    zero.get("supervised_full_vocab_logits_sha256s"),
                    label="V20k zero logits hashes",
                )
            )
            == {
                example: inherited_zero_logits[example]
                for example in expected_examples
            }
        )
        if (
            not zero_anchor
            or evidence.get("zero_response_exact_v20g_eta_zero_output_anchor")
            is not zero_anchor
        ):
            raise ValueError("V20k inner zero-response V20g output anchor differs")
        validated_inner[family] = evidence
    selection = _aggregate_response_selection(validated_inner)
    persisted_selection = _mapping(
        receipt.get("response_selection_receipt"),
        label="V20k persisted response selection",
    )
    if _v14._canonical_json_bytes(selection) != _v14._canonical_json_bytes(
        persisted_selection
    ):
        raise ValueError("V20k inner response selection replay differs")
    return receipt


def _validate_fold_fragment(
    value: Mapping[str, object],
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20g_fold: Mapping[str, object],
    authenticated_v20i_fold: Mapping[str, object],
) -> None:
    fragment = _mapping(value, label="V20k fold fragment")
    if set(fragment) != _FOLD_FRAGMENT_KEYS:
        raise ValueError("V20k fold fragment key set differs")
    outer = _identifier(outer_family_id, label="V20k fold outer family")
    expected_header = {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "masked_direction_protocol_sha256": (
            _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
        ),
        "log_response_fit_protocol_sha256": (
            _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "exact_objective_kind": (
            "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
        ),
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold[
            "fragment_sha256"
        ],
        "v20i_fold_fragment_sha256": authenticated_v20i_fold[
            "fragment_sha256"
        ],
        "v20j_fold_fragment_sha256": source[
            "v20j_fold_fragment_sha256s_by_family"
        ][outer],
        "outer_held_family_id": outer,
    }
    if any(fragment.get(key) != expected for key, expected in expected_header.items()):
        raise ValueError("V20k fold fragment header differs")
    if (
        fragment.get("fixed_schedule_completed") is not True
        or fragment.get("candidate") is not None
        or fragment.get("provider_sidecar") is not None
    ):
        raise ValueError("V20k fold scalar-only boundary differs")
    for key in ("endpoint_receipt", "endpoint_evidence"):
        if _v14._canonical_json_bytes(fragment.get(key)) != _v14._canonical_json_bytes(
            authenticated_v20g_fold.get(key)
        ):
            raise ValueError("V20k fold endpoint lineage differs")

    fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"),
        label="V20k validation V20g fit",
    )
    source_direction = _mapping(
        fit.get("direction_receipt"), label="V20k validation source direction"
    )
    inner = _validate_inner_receipt(
        _mapping(fragment.get("inner_receipt"), label="V20k inner receipt"),
        source_direction=source_direction,
        outer_family_id=outer,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    response_selection = _validate_hashed(
        _mapping(
            fragment.get("response_selection_receipt"),
            label="V20k response selection",
        ),
        domain=_RESPONSE_SELECTION_DOMAIN,
        label="V20k response selection",
    )
    if _v14._canonical_json_bytes(response_selection) != _v14._canonical_json_bytes(
        inner.get("response_selection_receipt")
    ):
        raise ValueError("V20k duplicated response selection differs")

    reflection_fit = _mapping(
        fragment.get("outer_reflection_fit_receipt"),
        label="V20k outer reflection fit",
    )
    _reflection.validate_soft_polarity_reflection_fit_receipt(
        reflection_fit, direction_receipt=source_direction
    )
    _validate_v20i_reflection_lineage(
        inner_receipt=inner,
        outer_reflection_fit=reflection_fit,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    manifest = _validate_hashed(
        _mapping(
            fragment.get("provider_manifest"), label="V20k outer manifest"
        ),
        domain=_OUTER_MANIFEST_DOMAIN,
        label="V20k outer manifest",
    )
    endpoint_receipt = _mapping(
        authenticated_v20g_fold.get("endpoint_receipt"),
        label="V20k outer inherited endpoint receipt",
    )
    if (
        tuple(manifest.get("arm_order", ())) != _ARMS
        or manifest.get("outer_held_family_id") != outer
        or _response_pair(manifest.get("selected_response"))
        != _response_pair(response_selection["selected_response"])
        or _response_pair(manifest.get("matched_linear_response"))
        != (
            _response_pair(response_selection["selected_response"])[0],
            0.0,
        )
        or manifest.get("endpoint_receipt_sha256")
        != endpoint_receipt.get("artifact_sha256")
        or manifest.get("source_direction_receipt_sha256")
        != source_direction.get("artifact_sha256")
        or manifest.get("outer_reflection_fit_receipt_sha256")
        != reflection_fit.get("artifact_sha256")
        or manifest.get("selected_variant_artifact_sha256")
        != reflection_fit.get("selected_variant_artifact_sha256")
        or manifest.get("all_seven_providers_frozen_before_outer_capability")
        is not True
        or manifest.get("all_seven_traces_frozen_before_outer_capability") is not True
        or manifest.get("outer_capability_count_at_freeze") != 0
        or manifest.get("outer_objectives_or_teacher_rows_used_at_freeze")
        is not False
        or manifest.get("raw_provider_or_response_tensors_serialized") is not False
    ):
        raise ValueError("V20k outer provider manifest differs")
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s"),
        label="V20k outer provider hashes",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts"),
        label="V20k outer provider receipts",
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s"),
        label="V20k outer trace hashes",
    )
    soft_transfer_hashes = _mapping(
        manifest.get("soft_provider_transfer_evidence_sha256s"),
        label="V20k outer soft provider transfer hashes",
    )
    expected_soft_transfer_arms = (
        "matched_linear_reflected",
        "same_response_unreflected",
        "log_response_reflected",
        "log_response_reflected_exact_mirror",
    )
    if set(soft_transfer_hashes) != set(expected_soft_transfer_arms):
        raise ValueError("V20k outer soft transfer arm bindings differ")
    fixed_control_transfer_hash = _sha(
        manifest.get("fixed_control_transfer_evidence_sha256"),
        label="V20k outer fixed-control transfer hash",
    )
    if any(
        set(values) != set(_ARMS)
        for values in (provider_hashes, provider_receipts, trace_hashes)
    ):
        raise ValueError("V20k outer manifest arm bindings differ")
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20k inherited V20g held evidence",
        ).get("arm_evidence"),
        label="V20k inherited V20g held arms",
    )
    if not all(arm in inherited_arms for arm in ("base", "fixed_plus", "fixed_minus")):
        raise ValueError("V20k inherited V20g control arms differ")
    expected_examples = tuple(
        sorted(
            _identifier(example, label="V20k outer expected example")
            for example in _mapping(
                _mapping(
                    inherited_arms["base"],
                    label="V20k inherited V20g base arm",
                ).get("objectives_by_example"),
                label="V20k inherited V20g base objectives",
            )
        )
    )
    held = _validate_hashed(
        _mapping(fragment.get("held_evidence"), label="V20k held evidence"),
        domain=_OUTER_EXECUTION_DOMAIN,
        label="V20k held evidence",
    )
    arms = _mapping(held.get("arm_evidence"), label="V20k held arm evidence")
    if (
        set(arms) != set(_ARMS)
        or held.get("outer_held_family_id") != outer
        or held.get("outer_manifest_sha256") != manifest.get("artifact_sha256")
        or held.get("all_seven_providers_and_traces_frozen_before_outer_capability")
        is not True
        or held.get("outer_family_used_for_fit_or_selection") is not False
        or held.get("all_v20g_control_output_anchors_passed") is not True
        or int(held.get("exact_outer_execution_count", -1))
        != len(_ARMS) * _PROMPTS_PER_FAMILY
        or held.get("raw_prompts_tokens_logits_h4_or_teacher_rows_serialized")
        is not False
    ):
        raise ValueError("V20k outer held schedule differs")
    objectives: dict[str, float] = {}
    exact_outputs: dict[
        str, tuple[dict[str, float], dict[str, str], dict[str, str]]
    ] = {}
    trace_bundle_sha = _v14._sha256(
        {arm: trace_hashes[arm] for arm in _ARMS},
        domain=_OUTER_EXECUTION_DOMAIN,
    )
    selected_response = _response_pair(
        response_selection["selected_response"]
    )
    matched_linear_response = (selected_response[0], 0.0)
    selected_direction = _selected_direction(reflection_fit)
    unreflected_direction = _unreflected_direction(source_direction)
    mirror_direction = tuple(-item for item in selected_direction)
    expected_soft_transfers = {
        "matched_linear_reflected": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=matched_linear_response,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_matched_linear_reflected",
        ),
        "same_response_unreflected": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(source_direction["artifact_sha256"]),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            direction=unreflected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_same_response_unreflected",
        ),
        "log_response_reflected": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            direction=selected_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_log_response_reflected",
        ),
        "log_response_reflected_exact_mirror": _provider_seed(
            endpoint_receipt_sha256=str(manifest["endpoint_receipt_sha256"]),
            direction_artifact_sha256=str(
                reflection_fit["selected_variant_artifact_sha256"]
            ),
            reflection_fit_sha256=str(reflection_fit["artifact_sha256"]),
            response=selected_response,
            direction=mirror_direction,
            outer_family_id=outer,
            inner_family_id=None,
            role="outer_log_response_reflected_exact_mirror",
        ),
    }
    expected_soft_directions = {
        "matched_linear_reflected": selected_direction,
        "same_response_unreflected": unreflected_direction,
        "log_response_reflected": selected_direction,
        "log_response_reflected_exact_mirror": mirror_direction,
    }
    expected_soft_responses = {
        "matched_linear_reflected": matched_linear_response,
        "same_response_unreflected": selected_response,
        "log_response_reflected": selected_response,
        "log_response_reflected_exact_mirror": selected_response,
    }
    expected_fixed_control_transfer = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": manifest["endpoint_receipt_sha256"],
            "outer_held_family_id": outer,
            "reflection_fit_sha256": reflection_fit["artifact_sha256"],
            "selected_response": selected_response,
            "role": "outer_fixed_controls",
            "held_rows_used": False,
        },
        domain=_OUTER_MANIFEST_DOMAIN,
    )
    if (
        {
            arm: _sha(
                soft_transfer_hashes[arm],
                label=f"V20k {arm} soft transfer hash",
            )
            for arm in expected_soft_transfer_arms
        }
        != expected_soft_transfers
        or fixed_control_transfer_hash != expected_fixed_control_transfer
    ):
        raise ValueError("V20k outer provider transfer lineage differs")
    health_by_arm: dict[str, bool] = {}
    for arm in _ARMS:
        evidence = _validate_hashed(
            _mapping(arms[arm], label=f"V20k {arm} arm evidence"),
            domain=_OUTER_EXECUTION_DOMAIN,
            label=f"V20k {arm} arm evidence",
        )
        trace = _validate_hashed(
            _mapping(evidence.get("response_trace"), label=f"V20k {arm} trace"),
            domain=_TRACE_DOMAIN,
            label=f"V20k {arm} trace",
        )
        provider_receipt = _validate_hashed(
            _mapping(
                provider_receipts[arm],
                label=f"V20k {arm} provider receipt",
            ),
            domain=_PROVIDER_DOMAIN,
            label=f"V20k {arm} provider receipt",
        )
        score_bundle = _validate_exact_score_bundle(
            evidence,
            expected_example_ids=expected_examples,
            label=f"V20k outer {arm}",
        )
        arm_objectives, arm_h4, arm_logits, arm_executions = score_bundle
        provider_artifact = _sha(
            provider_hashes[arm], label=f"V20k {arm} provider hash"
        )
        _validate_provider_receipt_evidence(
            provider_receipt,
            expected_role=arm,
            expected_provider_artifact_sha256=provider_artifact,
            expected_endpoint_receipt=endpoint_receipt,
            expected_bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20i_fold=authenticated_v20i_fold,
            expected_response=(
                expected_soft_responses[arm]
                if arm in expected_soft_transfer_arms
                else None
            ),
            expected_direction=(
                expected_soft_directions[arm]
                if arm in expected_soft_transfer_arms
                else None
            ),
            expected_transfer_evidence_sha256=(
                expected_soft_transfers[arm]
                if arm in expected_soft_transfer_arms
                else (
                    expected_fixed_control_transfer
                    if arm in ("fixed_plus", "fixed_minus")
                    else None
                )
            ),
        )
        trace_artifact = _sha(
            trace_hashes[arm], label=f"V20k {arm} trace hash"
        )
        seed = _v14._sha256(
            {
                "outer_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "outer_held_family_id": outer,
                "arm": arm,
                "provider_artifact_sha256": provider_artifact,
                "all_outer_arms_frozen": True,
            },
            domain=_OUTER_EXECUTION_DOMAIN,
        )
        expected_executions = {
            example: _execution_sha256(
                phase="outer_family_disjoint_mechanism_score",
                outer_family_id=outer,
                inner_family_id=None,
                role=arm,
                provider_artifact_sha256=provider_artifact,
                example_id=example,
                family_id=outer,
                objective=arm_objectives[example],
                h4_sha256=arm_h4[example],
                logits_sha256=arm_logits[example],
                evidence_sha256=seed,
                domain=_OUTER_EXECUTION_DOMAIN,
            )
            for example in expected_examples
        }
        response_gain_hashes = _mapping(
            trace.get("response_gain_sha256s"),
            label=f"V20k {arm} response gain hashes",
        )
        for value in response_gain_hashes.values():
            _sha(value, label=f"V20k {arm} response gain hash")
        health_by_arm[arm] = bool(
            evidence.get("finite") is True
            and trace.get("finite") is True
            and trace.get("pointwise_trust_passed") is True
            and trace.get("endpoint_conditional_ranks_are_16") is True
        )
        if (
            evidence.get("arm") != arm
            or evidence.get("provider_artifact_sha256") != provider_artifact
            or evidence.get("outer_manifest_sha256")
            != manifest.get("artifact_sha256")
            or trace.get("artifact_sha256") != trace_artifact
            or trace.get("provider_artifact_sha256") != provider_artifact
            or trace.get("arm") != arm
            or tuple(trace.get("scored_family_ids", ())) != (outer,)
            or set(response_gain_hashes) != set(expected_examples)
            or provider_receipt.get("provider_artifact_sha256")
            != provider_artifact
            or provider_receipt.get("role") != arm
            or provider_receipt.get("raw_provider_tensors_serialized")
            is not False
            or int(provider_receipt.get("conditional_rank", -1))
            != _CONDITIONAL_RANK
            or provider_receipt.get("analysis_only") is not (arm != "base")
            or (
                arm in expected_soft_transfer_arms
                and (
                    provider_receipt.get("transfer_protocol_sha256")
                    != _TRANSFER_PROTOCOL_SHA256
                    or provider_receipt.get("transfer_evidence_sha256")
                    != expected_soft_transfers[arm]
                    or tuple(provider_receipt.get("direction", ()))
                    != expected_soft_directions[arm]
                    or _response_pair(provider_receipt.get("response"))
                    != expected_soft_responses[arm]
                    or provider_receipt.get("response_key")
                    != _response_key(expected_soft_responses[arm])
                    or float(
                        provider_receipt.get("radius", math.nan)
                    )
                    != expected_soft_responses[arm][0]
                    or float(
                        provider_receipt.get("mix", math.nan)
                    )
                    != expected_soft_responses[arm][1]
                    or tuple(
                        provider_receipt.get("direction_box_corner_scores", ())
                    )
                    != _box_corner_scores(expected_soft_directions[arm])
                    or _v14._canonical_json_bytes(
                        _mapping(
                            provider_receipt.get("box_certificate"),
                            label=f"V20k {arm} log_response certificate",
                        )
                    )
                    != _v14._canonical_json_bytes(
                        fisher_soft_polarity_log_response_box_certificate(
                            _v20g._eta_tensor(expected_soft_directions[arm]),
                            radius=expected_soft_responses[arm][0],
                            mix=expected_soft_responses[arm][1],
                        )
                    )
                )
            )
            or (
                arm in ("fixed_plus", "fixed_minus")
                and (
                    provider_receipt.get("transfer_protocol_sha256")
                    != _TRANSFER_PROTOCOL_SHA256
                    or provider_receipt.get("transfer_evidence_sha256")
                    != expected_fixed_control_transfer
                )
            )
            or arm_executions != expected_executions
            or evidence.get("exact_execution") is not True
            or evidence.get("raw_logits_h4_teacher_rows_or_tensors_serialized")
            is not False
            or trace.get("raw_response_or_modal_tensors_serialized") is not False
            or not health_by_arm[arm]
        ):
            raise ValueError("V20k outer arm health differs")
        objectives[arm] = float(evidence["objective"])
        exact_outputs[arm] = (arm_objectives, arm_h4, arm_logits)
    _v20b._validate_capability_receipt(
        held.get("capability_receipt"),
        expected_example_ids=expected_examples,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_ARMS),
        label="V20k resumed outer-held capability",
    )
    control_anchors: dict[str, bool] = {}
    for arm in ("base", "fixed_plus", "fixed_minus"):
        inherited = _mapping(
            inherited_arms[arm], label=f"V20k inherited V20g {arm} arm"
        )
        current_objectives, current_h4, current_logits = exact_outputs[arm]
        control_anchors[arm] = bool(
            objectives[arm] == float(inherited.get("objective", math.nan))
            and current_objectives
            == {
                _identifier(example, label=f"V20k inherited {arm} example"): float(
                    objective
                )
                for example, objective in _mapping(
                    inherited.get("objectives_by_example"),
                    label=f"V20k inherited {arm} objectives",
                ).items()
            }
            and current_h4
            == dict(
                _mapping(
                    inherited.get("post_cast_h4_sha256s"),
                    label=f"V20k inherited {arm} H4 hashes",
                )
            )
            and current_logits
            == dict(
                _mapping(
                    inherited.get("supervised_full_vocab_logits_sha256s"),
                    label=f"V20k inherited {arm} logits hashes",
                )
            )
        )
    persisted_control_anchors = _mapping(
        held.get("v20g_control_output_anchors"),
        label="V20k persisted control anchors",
    )
    if (
        not all(control_anchors.values())
        or dict(persisted_control_anchors) != control_anchors
        or held.get("all_v20g_control_output_anchors_passed")
        is not all(control_anchors.values())
    ):
        raise ValueError("V20k outer V20g control output anchor differs")
    base_logits = exact_outputs["base"][2]
    candidate_logits = exact_outputs[_PRIMARY_ARM][2]
    candidate_changed = any(
        candidate_logits[example] != base_logits[example]
        for example in expected_examples
    )
    candidate_distinct = (
        provider_hashes[_PRIMARY_ARM] != provider_hashes["base"]
    )
    all_healthy = all(health_by_arm.values())
    fold = _validate_hashed(
        _mapping(fragment.get("fold_receipt"), label="V20k fold receipt"),
        domain=_DECISION_DOMAIN,
        label="V20k fold receipt",
    )
    if (
        fold.get("outer_held_family_id") != outer
        or tuple(fold.get("arm_order", ())) != _ARMS
        or dict(fold.get("held_objective_by_arm", {})) != objectives
        or _response_pair(fold.get("selected_response"))
        != selected_response
        or fold.get("selected_response_key")
        != _response_key(selected_response)
        or fold.get("selected_variant_artifact_sha256")
        != reflection_fit.get("selected_variant_artifact_sha256")
        or fold.get("selected_variant_id")
        != reflection_fit.get("selected_variant_id")
        or fold.get("candidate_provider_artifact_sha256")
        != provider_hashes[_PRIMARY_ARM]
        or fold.get("base_provider_artifact_sha256")
        != provider_hashes["base"]
        or fold.get("candidate_provider_distinct_from_base")
        is not candidate_distinct
        or fold.get("candidate_exact_execution_changed_from_base")
        is not candidate_changed
        or fold.get("selected_radius_positive")
        is not (selected_response[0] > 0.0)
        or fold.get("selected_mix_positive")
        is not (selected_response[1] > 0.0)
        or fold.get("all_runtime_health_passed") is not all_healthy
        or fold.get("all_v20g_control_output_anchors_passed")
        is not all(control_anchors.values())
        or fold.get("selection_frozen_before_outer_score") is not True
        or fold.get("outer_family_used_for_fit_or_selection") is not False
        or fold.get("exact_execution") is not True
    ):
        raise ValueError("V20k fold decision receipt differs")


def _publish_fold_fragment(
    payload: Mapping[str, object], *, output: Path | str, outer_family_id: str
) -> dict[str, object]:
    path = _fold_path(output, outer_family_id)
    _v20b._publish_scalar_fragment(
        payload,
        path=path,
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20k log_response fold fragment",
    )
    return _v20b._load_scalar_fragment(
        path=path,
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20k log_response fold fragment",
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
) -> dict[str, object]:
    fragment = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20k log_response fold fragment",
    )
    _validate_fold_fragment(
        fragment,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        outer_family_id=outer_family_id,
        bridge_binding_sha256=bridge_binding_sha256,
        authenticated_v20g_fold=authenticated_v20g_fold,
        authenticated_v20i_fold=authenticated_v20i_fold,
    )
    return fragment


def _fold_receipt_map(
    fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {
        family: _mapping(
            fragments[family].get("fold_receipt"), label="V20k aggregate fold"
        )
        for family in sorted(fragments)
    }


def _aggregate_decision(
    fold_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    families = tuple(sorted(fold_receipts))
    if len(families) != _FAMILY_COUNT:
        raise ValueError("V20k decision requires all eight outer folds")
    scores: dict[str, dict[str, float]] = {}
    responses: dict[str, tuple[float, float]] = {}
    variants: dict[str, str] = {}
    changed: dict[str, bool] = {}
    health: dict[str, bool] = {}
    anchors: dict[str, bool] = {}
    for family in families:
        fold = fold_receipts[family]
        if tuple(fold.get("arm_order", ())) != _ARMS:
            raise ValueError("V20k aggregate arm order differs")
        raw_scores = _mapping(
            fold.get("held_objective_by_arm"), label="V20k aggregate arm scores"
        )
        if set(raw_scores) != set(_ARMS):
            raise ValueError("V20k aggregate arm geometry differs")
        scores[family] = {arm: float(raw_scores[arm]) for arm in _ARMS}
        if not all(math.isfinite(value) for value in scores[family].values()):
            raise ValueError("V20k aggregate score became nonfinite")
        responses[family] = _response_pair(fold["selected_response"])
        variants[family] = str(fold["selected_variant_id"])
        changed[family] = (
            fold.get("candidate_provider_distinct_from_base") is True
            and fold.get("candidate_exact_execution_changed_from_base") is True
        )
        health[family] = fold.get("all_runtime_health_passed") is True
        anchors[family] = (
            fold.get("all_v20g_control_output_anchors_passed") is True
        )
    macro = {
        arm: math.fsum(scores[family][arm] for family in families) / len(families)
        for arm in _ARMS
    }

    def wins(reference: str) -> int:
        return sum(
            scores[family][_PRIMARY_ARM] < scores[family][reference]
            for family in families
        )

    wins_vs = {
        arm: wins(arm)
        for arm in (
            "base",
            "fixed_plus",
            "fixed_minus",
            "matched_linear_reflected",
            "same_response_unreflected",
            "log_response_reflected_exact_mirror",
        )
    }
    positive_changed = all(
        responses[family][0] > 0.0 and changed[family]
        for family in families
    )
    positive_mix_count = sum(
        responses[family][1] > 0.0 for family in families
    )
    curvature_evidence = positive_mix_count >= 5
    integrity = all(health.values()) and all(anchors.values())
    primary_gate = (
        integrity
        and positive_changed
        and macro[_PRIMARY_ARM] < macro["base"]
        and macro[_PRIMARY_ARM] < macro["fixed_plus"]
        and wins_vs["base"] >= 6
        and wins_vs["fixed_plus"] >= 6
    )
    mechanism_gate = (
        integrity
        and positive_changed
        and curvature_evidence
        and macro[_PRIMARY_ARM] < macro["same_response_unreflected"]
        and wins_vs["same_response_unreflected"] >= 5
        and macro[_PRIMARY_ARM] < macro["log_response_reflected_exact_mirror"]
        and wins_vs["log_response_reflected_exact_mirror"] >= 6
        and macro[_PRIMARY_ARM] < macro["matched_linear_reflected"]
        and wins_vs["matched_linear_reflected"] >= 5
    )
    passed = primary_gate and mechanism_gate
    return _hashed(
        {
            "family_ids": families,
            "selected_response_by_family": responses,
            "selected_variant_id_by_family": variants,
            "held_objective_by_family_and_arm": scores,
            "macro_objective_by_arm": macro,
            "candidate_win_count_by_reference_arm": wins_vs,
            "candidate_changed_exact_by_family": changed,
            "runtime_health_by_family": health,
            "control_output_anchor_by_family": anchors,
            "all_selected_radii_positive_and_candidates_changed_exact": (
                positive_changed
            ),
            "selected_mix_positive_by_family": {
                family: responses[family][1] > 0.0 for family in families
            },
            "selected_mix_positive_count": positive_mix_count,
            "curvature_evidence_gate_passed": curvature_evidence,
            "integrity_passed": integrity,
            "primary_development_gate_passed": primary_gate,
            "mechanism_gate_passed": mechanism_gate,
            "development_oof_passed": passed,
            "fixed_minus_diagnostic_only": True,
            "strict_win_comparison": True,
        },
        domain=_DECISION_DOMAIN,
    )


def _runner_work_accounting() -> dict[str, object]:
    """Return the fixed canonical V20k one-shot schedule.

    Authentication and resume attempts are deliberately excluded.  V20k
    consumes serialized V20g Fisher/gradient summaries for its 56 masked
    solves, but live authority collection and endpoint reconstruction retain
    the same V20i backward/contraction accounting.
    """

    authority_forwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * 2
    endpoint_forwards = (
        _FAMILY_COUNT * (_FAMILY_COUNT - 1) * _PROMPTS_PER_FAMILY
    )
    inner_forwards = (
        _FAMILY_COUNT
        * _INNER_FAMILY_COUNT
        * _PROMPTS_PER_FAMILY
        * len(_RESPONSES)
    )
    outer_forwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY * len(_ARMS)
    total_forwards = (
        authority_forwards + endpoint_forwards + inner_forwards + outer_forwards
    )
    authority_backwards = _FAMILY_COUNT * _PROMPTS_PER_FAMILY
    total_backwards = authority_backwards + endpoint_forwards
    teacher_accesses = endpoint_forwards + inner_forwards + outer_forwards
    if (
        total_forwards != 1488
        or total_backwards != 128
        or endpoint_forwards != 112
        or inner_forwards != 1232
        or outer_forwards != 112
        or teacher_accesses != 1456
    ):
        raise RuntimeError("V20k canonical work schedule drifted")
    return {
        "accounting_scope": "canonical_one_shot_schedule",
        "resume_and_authentication_overhead_excluded": True,
        "live_authority_collection_model_forward_count": authority_forwards,
        "endpoint_reconstruction_model_forward_count": endpoint_forwards,
        "inner_conditional_leave_one_family_out_model_forward_count": (
            inner_forwards
        ),
        "inner_endpoint_retrained_per_fold": False,
        "outer_held_model_forward_count": outer_forwards,
        "canonical_model_forward_count": total_forwards,
        "total_model_forward_count": total_forwards,
        "canonical_teacher_access_count": teacher_accesses,
        "total_teacher_access_count": teacher_accesses,
        "live_authority_collection_suffix_backward_count": authority_backwards,
        "endpoint_reconstruction_suffix_backward_count": endpoint_forwards,
        "canonical_suffix_backward_count": total_backwards,
        "total_suffix_backward_count": total_backwards,
        "endpoint_reconstruction_local_autograd_contraction_count": (
            endpoint_forwards
        ),
        "canonical_local_autograd_contraction_count": endpoint_forwards,
        "total_local_autograd_contraction_count": endpoint_forwards,
        "masked_fisher_solve_count": (
            _FAMILY_COUNT * _INNER_FAMILY_COUNT
        ),
        "reflection_fit_count": (
            _FAMILY_COUNT * (_INNER_FAMILY_COUNT + 1)
        ),
        "reflection_variant_receipt_count": (
            _FAMILY_COUNT * (_INNER_FAMILY_COUNT + 1) * 5
        ),
        "log_response_candidate_count": (
            _FAMILY_COUNT * _INNER_FAMILY_COUNT * len(_RESPONSES)
        ),
        "inner_provider_candidate_count": (
            _FAMILY_COUNT * _INNER_FAMILY_COUNT * len(_RESPONSES)
        ),
        "outer_arm_provider_count": _FAMILY_COUNT * len(_ARMS),
        "inner_response_trace_example_count": inner_forwards,
        "outer_response_trace_example_count": outer_forwards,
        "endpoint_health_trace_example_count": endpoint_forwards,
        "all_eight_final_refit_model_forward_count": 0,
        "calibration_b_forward_or_tokenization_count": 0,
    }


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    v20g_report: Mapping[str, object],
    v20i_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    fold_fragments: Mapping[str, Mapping[str, object]],
    decision: Mapping[str, object] | None = None,
) -> dict[str, object]:
    folds = _fold_receipt_map(fold_fragments)
    aggregate = (
        _aggregate_decision(folds)
        if decision is None
        else _validate_hashed(
            decision, domain=_DECISION_DOMAIN, label="V20k aggregate decision"
        )
    )
    families = tuple(
        _identifier(item, label="V20k report family")
        for item in _sequence(
            aggregate.get("family_ids"), label="V20k report families"
        )
    )
    if (
        len(families) != _FAMILY_COUNT
        or set(fold_fragments) != set(families)
        or set(folds) != set(families)
    ):
        raise RuntimeError("V20k report requires all eight authenticated folds")
    replayed = _aggregate_decision(folds)
    if _v14._canonical_json_bytes(replayed) != _v14._canonical_json_bytes(
        aggregate
    ):
        raise ValueError("V20k supplied decision differs from fold replay")
    passed = aggregate.get("development_oof_passed") is True
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": (
            _reflection.SOFT_POLARITY_REFLECTION_FIT_PROTOCOL_SHA256
        ),
        "masked_direction_protocol_sha256": (
            _reflection.SOFT_POLARITY_MASKED_DIRECTION_PROTOCOL_SHA256
        ),
        "log_response_fit_protocol_sha256": (
            _log_response_fit.SOFT_POLARITY_LOG_RESPONSE_FIT_PROTOCOL_SHA256
        ),
        "exact_objective_kind": (
            "token_mean_exact_float64_full_vocabulary_kl_teacher_to_candidate"
        ),
        "fixed_protocol": _FIXED_PROTOCOL,
        "source_receipt": dict(source),
        "v20g_authority": {
            "report_sha256": v20g_report.get("report_sha256"),
            "classification": v20g_report.get("classification"),
            "passed": v20g_report.get("passed"),
            "rollback_to_base": v20g_report.get("rollback_to_base"),
            "fold_fragment_sha256s_by_family": {
                family: source["v20g_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20i_authority": {
            "report_sha256": v20i_report.get("report_sha256"),
            "classification": v20i_report.get("classification"),
            "development_oof_passed": v20i_report.get("development_oof_passed"),
            "primary_development_gate_passed": v20i_report.get(
                "primary_development_gate_passed"
            ),
            "mechanism_gate_passed": v20i_report.get("mechanism_gate_passed"),
            "passed": v20i_report.get("passed"),
            "rollback_to_base": v20i_report.get("rollback_to_base"),
            "fold_fragment_sha256s_by_family": {
                family: source["v20i_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "v20j_authority": {
            "report_sha256": source["v20j_report_sha256"],
            "file_sha256": source["v20j_file_sha256"],
            "source_receipt_sha256": source["v20j_source_receipt_sha256"],
            "classification": (
                "soft_polarity_confidence_nested_oof_failed_rollback_to_base"
            ),
            "development_oof_passed": False,
            "primary_development_gate_passed": False,
            "mechanism_gate_passed": False,
            "passed": False,
            "rollback_to_base": True,
            "fold_fragment_sha256s_by_family": {
                family: source["v20j_fold_fragment_sha256s_by_family"][family]
                for family in families
            },
        },
        "panel_receipt": dict(panel_receipt),
        "bridge_binding_sha256": bridge_binding_sha256,
        "fold_fragment_sha256s_by_family": {
            family: fold_fragments[family]["fragment_sha256"]
            for family in families
        },
        "fold_receipts_by_family": {
            family: dict(folds[family]) for family in families
        },
        "decision": dict(aggregate),
        "classification": (
            "soft_polarity_log_response_nested_oof_passed_fresh_shadow_eligible"
            if passed
            else "soft_polarity_log_response_nested_oof_failed_rollback_to_base"
        ),
        "passed": passed,
        "development_oof_passed": passed,
        "primary_development_gate_passed": (
            aggregate.get("primary_development_gate_passed") is True
        ),
        "mechanism_gate_passed": aggregate.get("mechanism_gate_passed") is True,
        "all_eight_outer_folds_completed": True,
        "all_eight_final_refit_completed": False,
        "full_refit_performed": False,
        "final_refit_authorized_for_next_fresh_shadow": passed,
        "fresh_family_disjoint_shadow_eligible": passed,
        "fresh_family_disjoint_scoring_performed": False,
        "final_refit": None,
        "final_provider_frozen": False,
        "rollback_to_base": not passed,
        "calibration_b_eligibility_gate_passed": False,
        "calibration_b_eligible": False,
        "calibration_b_authorized": False,
        "calibration_b_manifest_read": False,
        "calibration_b_opened": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "validation_opened": False,
        "test_opened": False,
        "serving_claim_authorized": False,
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
        "fixed_minus_is_diagnostic_only": True,
        "candidate": None,
        "provider_sidecar": None,
        "next_rung": (
            "fresh_family_disjoint_shadow_then_all_eight_refit"
            if passed
            else "revise_reflection_or_response_transfer_then_repeat_nested_OOF"
        ),
        "work_accounting": _runner_work_accounting(),
        "integrity": {
            "V20g_V20i_V20j_reports_and_all_fragments_authenticated_before_model_"
            "construction": True,
            "all_56_masked_directions_use_only_six_training_family_summaries": True,
            "inner_response_selection_is_conditional_on_each_fixed_seven_"
            "family_endpoint_not_full_inner_model_cross_validation": True,
            "all_eight_outer_held_families_absent_from_endpoint_direction_"
            "reflection_and_response_selection": True,
            "all_77_inner_providers_and_traces_frozen_before_each_outer_fold_"
            "inner_scoring": True,
            "all_seven_outer_providers_and_traces_frozen_before_each_outer_"
            "capability": True,
            "all_inner_zero_response_V20g_eta_zero_output_anchors_passed": True,
            "all_outer_base_fixed_plus_fixed_minus_V20g_output_anchors_"
            "passed": True,
            "no_all_eight_refit_or_calibration_b_access_in_this_rung": True,
            "raw_prompts_tokens_logits_h4_gradients_or_provider_tensors_"
            "serialized": False,
        },
        "artifact": None,
    }
    _v14._scalar_report(report)
    return report


def _load_existing_report(
    output: Path,
    *,
    source: Mapping[str, object],
    v20g_report: Mapping[str, object],
    v20i_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    authenticated_v20g_folds: Mapping[str, Mapping[str, object]],
    authenticated_v20i_folds: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20k log_response nested report",
    )
    families = tuple(sorted(authenticated_v20g_folds))
    if set(authenticated_v20i_folds) != set(families):
        raise ValueError("V20k report authority family geometry differs")
    folds = {
        family: _load_fold_fragment(
            output=output,
            source=source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
            authenticated_v20i_fold=authenticated_v20i_folds[family],
        )
        for family in families
    }
    rebuilt = _build_report(
        output=output,
        source=source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        fold_fragments=folds,
    )
    supplied = dict(value)
    report_sha = supplied.pop("report_sha256", None)
    if (
        _v14._canonical_json_bytes(supplied)
        != _v14._canonical_json_bytes(rebuilt)
        or report_sha != _v14._sha256(rebuilt, domain=_REPORT_DOMAIN)
    ):
        raise ValueError("V20k report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_log_response_nested_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or resume the V20k nested reflection/response development screen."""

    destination = _validate_output(output)
    (
        prerequisite,
        authenticated_v20a_folds,
        v20g_report,
        authenticated_v20g_folds,
        v20i_report,
        authenticated_v20i_folds,
        source,
    ) = _load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"), label="V20k panel receipt"
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20k bridge binding",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
            authenticated_v20i_folds=authenticated_v20i_folds,
        )

    family_ids = tuple(sorted(authenticated_v20g_folds))
    if (
        len(family_ids) != _FAMILY_COUNT
        or set(authenticated_v20a_folds) != set(family_ids)
        or set(authenticated_v20i_folds) != set(family_ids)
        or set(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20k panel families",
            )
        )
        != set(family_ids)
    ):
        raise RuntimeError("V20k authenticated family geometry differs")

    # The final aggregation is model-free.  Completed fold fragments remain
    # authoritative after an interruption and must never trigger a second
    # Gemma construction or a hidden all-eight refit.
    if all(_fold_path(destination, family).exists() for family in family_ids):
        completed = {
            family: _load_fold_fragment(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                outer_family_id=family,
                bridge_binding_sha256=bridge_binding,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
            )
            for family in family_ids
        }
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            fold_fragments=completed,
        )
        try:
            _v20b._publish_scalar_fragment(
                report,
                path=destination,
                domain=_REPORT_DOMAIN,
                hash_key="report_sha256",
                label="V20k log_response nested report",
            )
        except FileExistsError:
            pass
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
            authenticated_v20i_folds=authenticated_v20i_folds,
        )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        if context.bridge.bridge_binding_sha256 != bridge_binding:
            raise RuntimeError("V20k live bridge differs from authenticated authority")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != family_ids:
            raise RuntimeError("V20k live family order differs from authenticated A16")
        fragments: dict[str, dict[str, object]] = {}
        for family in family_ids:
            if _fold_path(destination, family).exists():
                fragments[family] = _load_fold_fragment(
                    output=destination,
                    source=source,
                    panel_receipt=panel_receipt,
                    outer_family_id=family,
                    bridge_binding_sha256=bridge_binding,
                    authenticated_v20g_fold=authenticated_v20g_folds[family],
                    authenticated_v20i_fold=authenticated_v20i_folds[family],
                )
                continue
            live = _execute_outer_fold(
                context,
                records,
                teacher_vault,
                family_ids=family_ids,
                outer_family_id=family,
                panel_receipt=panel_receipt,
                authenticated_v20a_fold=authenticated_v20a_folds[family],
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
            )
            payload = _fold_payload(
                live,
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                bridge_binding_sha256=bridge_binding,
                outer_family_id=family,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
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
                authenticated_v20g_fold=authenticated_v20g_folds[family],
                authenticated_v20i_fold=authenticated_v20i_folds[family],
            )
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            v20i_report=v20i_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
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
            label="V20k log_response nested report",
        )
    except FileExistsError:
        pass
    return _load_existing_report(
        destination,
        source=source,
        v20g_report=v20g_report,
        v20i_report=v20i_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_v20g_folds=authenticated_v20g_folds,
        authenticated_v20i_folds=authenticated_v20i_folds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V20k nested six-family Fisher reflection and exact-response "
            "development screen"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = (
        run_gemma3_l3_l4_complete_h4_soft_polarity_log_response_nested_development(
            output=arguments.output,
            cache_dir=arguments.cache_dir,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
