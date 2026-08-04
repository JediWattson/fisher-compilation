"""V20q nested token-VJP compilation of the existing V20p local field.

V20q does not add a serving feature.  It measures post-cast H4 coefficient
secants at smooth ``s=+/-0.5`` charts, contracts exact token teacher-KL VJPs
with those two directions, and compiles the resulting Fisher moments back
into the same V20p ``(feature_id, field_bias, field_slope)`` executor.

All feature/ridge/step decisions are six-family-to-one inner OOF decisions.
The outer family is inaccessible until the selected specification has been
refit on all seven training families and its provider has been frozen.  Raw
prompts, tokens, logits, H4 tensors, secants, and gradients are never written
to fold or report artifacts.

This remains historically reused A16 development evidence.  It authorizes no
fresh validation, Calibration-B, compression, fidelity, speed, or serving
claim.
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
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from . import gemma3_l3_l4_complete_h4_soft_polarity_trust_region_nested_development as _v20g
from . import gemma3_l3_l4_complete_h4_soft_polarity_local_signed_field_nested_development as _v20p
from . import gemma3_l3_l4_graph_organized_svd_shadow_runtime as _shadow_runtime
from . import complete_h4_fisher_soft_polarity_token_vjp_fit as _token_fit
from . import complete_h4_fisher_soft_polarity_token_vjp_protocol as _token_protocol
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .complete_h4_fisher_soft_polarity_local_signed_field import (
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
    AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field,
)
from .complete_h4_fisher_soft_polarity_token_vjp_fit import (
    SoftPolarityTokenVJPPromptRecord,
    aggregate_soft_polarity_token_vjp_records,
    contract_soft_polarity_token_h4_vjps,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)
from .gemma3_l3_l4_soft_polarity_token_vjp_compiler import (
    SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
    SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
    build_selected_teacher_grid,
    build_soft_polarity_post_cast_h4_secants,
    soft_polarity_post_cast_h4_secant_stability,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_token_vjp_nested_development",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-token-vjp-"
    "nested-r16-k256-a-fit16-dev-v20q.json"
)

_V20P_OUTPUT = _v20p.DEFAULT_OUTPUT
_V20P_LOGICAL_SHA256 = "e4ea102253a52922c26ac4c3939e604fb9a17568e47ca05cf81b7471de6351d0"
_V20P_FILE_SHA256 = "508b026b59b410552920ee8b62c8c42ea55eccd2e76cc3a4502b76b445a3ca4b"
_V20P_SOURCE_SHA256 = "37cd3d743305b65fdd85b0924e71aee9fa7567e0b2cced176bba6d2cfb29bcbf"
_V20P_FOLD_SHA256S = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": "363cc43e9f7395456145000fd068e97bdd2369383b1ab88254a5c55adf253bc5",
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": "8542283678cf1e23c856e040ac44d98124a226485ca3a430d04a54fc3b37d38b",
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": "73d4377fb22eb7afa6d2b2655ec57e91bafa64febfacb3540e4d658bb23fa96a",
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": "af56f6bafb40c2ba47213dac692b6f3f59fc020217ebee5565e02aa5b42762b8",
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": "4ee1377fbe01f68ca46f1d655d96c256ae8c8d9eb3aca4a68724968dab36f6f0",
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": "03ed3dac76919fd832f983bd6719790d088a65d72c7aa71e90afcbb1b86f6718",
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": "071a177fd83ae77f7e3b6617825872e6de2d64beaa80d7a4be08d6d476d14a70",
    "structured-strong-v9-calibration_a-varve-lamination-v9": "1011b2a8593c78092c80d007b13ba037ef755a794f778cd6de49b27054aaa28a",
}

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_token_vjp_nested.v20q"
_FOLD_SCHEMA = "fisher_graph.complete_h4_soft_polarity_token_vjp_outer_fold.v20q"
_FORMAT_VERSION = 33
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-nested-report:v20q\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-nested-source:v20q\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-nested-fold:v20q\0"
_PROVIDER_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-provider:v20q\0"
_EXECUTION_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-execution:v20q\0"
_COLLECTION_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-collection:v20q\0"
_DECISION_DOMAIN = b"fisher-graph:soft-polarity-token-vjp-decision:v20q\0"

_FAMILY_COUNT = 8
_INNER_FAMILY_COUNT = 7
_PROMPTS_PER_FAMILY = 2
_FEATURES = ("c1", "c2", "c1_times_c2", "source_z")
_SEED_SIGNS = (-1, 1)
_SEED_ABS_B = 0.5
_VJP_CHUNK_SIZE = 8
_DERIVATIVE_CONVENTION = (
    "reverse_token_teacher_KL_VJP_contracted_with_primary_post_cast_"
    "central_H4_secants_at_smooth_constant_seed"
)
_TOKEN_VJP_PROTOCOL_RECEIPT = (
    _token_protocol.build_soft_polarity_token_vjp_protocol_receipt()
)
_TOKEN_VJP_PROTOCOL_RECEIPT_SHA256 = str(
    _TOKEN_VJP_PROTOCOL_RECEIPT["artifact_sha256"]
)
_CANDIDATE_SPEC_BY_ID = {
    str(row[0]): {
        "candidate_id": str(row[0]),
        "role": str(row[1]),
        "feature_id": row[2],
        "seed_sign": row[3],
        "seed_b": row[4],
        "seed_a": row[5],
        "ridge": row[6],
        "alpha": row[7],
        "fixed_b": row[8],
        "fixed_a": row[9],
    }
    for row in _token_protocol.SOFT_POLARITY_TOKEN_VJP_CANDIDATE_LIBRARY
}
if (
    _FEATURES != tuple(_token_protocol.SOFT_POLARITY_TOKEN_VJP_FEATURE_IDS)
    or _SEED_SIGNS
    != tuple(_token_protocol.SOFT_POLARITY_TOKEN_VJP_SEED_SIGNS)
    or SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP
    != _token_protocol.SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP
    or SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP
    != _token_protocol.SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP
):
    raise RuntimeError("V20q runner and frozen token-VJP policy differ")

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20q_nested_token_vjp_continuous_local_field",
    "scientific_status": "historically_reused_A16_development_after_completed_v20p_failure",
    "runtime_provider": "exact_unchanged_v20p_local_signed_field",
    "token_vjp_policy_receipt_sha256": _TOKEN_VJP_PROTOCOL_RECEIPT_SHA256,
    "feature_order": _FEATURES,
    "smooth_seed_biases": (-0.5, 0.5),
    "smooth_seed_slope": 0.0,
    "primary_post_cast_secant_half_step": SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
    "audit_post_cast_secant_half_step": SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
    "secant_stability_minimum_cosine": 0.99,
    "secant_stability_norm_ratio_interval": (0.80, 1.25),
    "derivative_authority": "exact_reverse_token_teacher_KL_H4_VJP_float64_objective",
    "compiler_fit": "mean_KL_natural_gradient_using_family_prompt_token_equal_OPG",
    "degenerate_direction_policy": "abort_entire_campaign_fail_closed",
    "secant_endpoint_binding": (
        "each_perturbed_provider_and_post_cast_H4_endpoint_hash_bound"
    ),
    "token_vjp_scalar_receipt": (
        "float64_objective_chunk_teacher_grid_supervision_and_H4_head_bound"
    ),
    "inner_validation": "seven_leave_one_whole_family_out_folds_fit_six_score_one",
    "outer_boundary": "one_of_eight_development_families_inaccessible_until_final_provider_freeze",
    "primary_gate": (
        "candidate_macro_strictly_below_v20p_incumbent_and_strictly_wins_"
        "at_least_five_of_eight_outer_families"
    ),
    "continuous_fit_gate": (
        "at_least_six_of_eight_select_nonzero_token_vjp_fit_with_inner_OOF_"
        "mean_strictly_below_v20p_incumbent"
    ),
    "exact_output_difference_gate": (
        "candidate_differs_from_v20p_incumbent_on_at_least_six_of_eight_"
        "outer_families"
    ),
    "derivative_gate": (
        "all_deployed_token_vjp_fits_have_stable_secants_and_negative_"
        "predicted_directional_derivative"
    ),
    "integrity_gate": (
        "all_outer_providers_frozen_before_score_no_outer_fit_or_selection_"
        "exact_execution_pointwise_trust_and_authenticated_v20p_controls"
    ),
    "raw_tensor_serialization": False,
    "failure_policy": "rollback_to_base_no_claims_no_B",
    "fresh_validation_claim": False,
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
        "v20p_provider_protocol_sha256": FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
        "operation": "V20q_fit_only_materialization_into_unchanged_V20p_runtime",
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
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be a canonical nonempty identifier")
    return value


def _sha(value: object, *, label: str) -> str:
    return _v20p._sha(value, label=label)


def _hashed(payload: Mapping[str, object], *, domain: bytes) -> dict[str, object]:
    result = dict(payload)
    result["artifact_sha256"] = _v14._sha256(result, domain=domain)
    return result


def _validate_hashed(
    value: Mapping[str, object], *, domain: bytes, label: str
) -> None:
    supplied = dict(_mapping(value, label=label))
    artifact = _sha(
        supplied.pop("artifact_sha256", None), label=f"{label} artifact"
    )
    if artifact != _v14._sha256(supplied, domain=domain):
        raise ValueError(f"{label} artifact hash differs")


def _validate_output(path: Path | str) -> Path:
    destination = _v20p._validate_output(path)
    protected = {
        _V20P_OUTPUT.resolve(strict=False),
        *(
            _v20p._fold_path(_V20P_OUTPUT, family).resolve(strict=False)
            for family in sorted(_V20P_FOLD_SHA256S)
        ),
    }
    if destination in protected:
        raise ValueError("V20q output must preserve immutable V20p authority")
    return destination


def _fold_path(output: Path | str, family_id: str) -> Path:
    return _v20p._fold_path(output, family_id)


@dataclass(slots=True)
class _Authorities:
    parent: object
    v20p_report: dict[str, object]
    authenticated_v20p_folds: dict[str, dict[str, object]]
    source: dict[str, object]


def _load_prerequisites() -> _Authorities:
    """Authenticate completed V20p and its full ancestry before Gemma exists."""

    parent = _v20p._load_prerequisites()
    panel = dict(
        _mapping(parent.prerequisite.get("nested_panel_receipt"), label="V20q panel")
    )
    bridge = _sha(
        parent.prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20q bridge binding",
    )
    if _v14._file_sha256(_V20P_OUTPUT) != _V20P_FILE_SHA256:
        raise RuntimeError("pinned V20p report file hash drifted")
    report = _v20p._load_existing_report(
        _V20P_OUTPUT,
        authorities=parent,
        panel_receipt=panel,
        bridge_binding_sha256=bridge,
    )
    observed = {
        _identifier(key, label="V20q V20p fold family"): _sha(
            value, label="V20q V20p fold hash"
        )
        for key, value in _mapping(
            report.get("fold_fragment_sha256s_by_family"),
            label="V20q V20p fold hashes",
        ).items()
    }
    decision = _mapping(report.get("decision"), label="V20q V20p decision")
    if (
        report.get("report_sha256") != _V20P_LOGICAL_SHA256
        or _mapping(report.get("source_receipt"), label="V20q V20p source").get(
            "artifact_sha256"
        )
        != _V20P_SOURCE_SHA256
        or observed != _V20P_FOLD_SHA256S
        or report.get("all_eight_outer_folds_completed") is not True
        or decision.get("integrity_passed") is not True
        or report.get("development_oof_passed") is not False
        or report.get("rollback_to_base") is not True
        or report.get("calibration_b_opened") is not False
        or report.get("final_refit") is not None
    ):
        raise RuntimeError("pinned completed V20p authority differs")
    families = tuple(sorted(_V20P_FOLD_SHA256S))
    folds = {
        family: _v20p._load_fold_fragment(
            output=_V20P_OUTPUT,
            source=parent.source,
            panel_receipt=panel,
            outer_family_id=family,
            bridge_binding_sha256=bridge,
            authenticated_v20g_fold=parent.authenticated_v20g_folds[family],
            authenticated_v20i_fold=parent.authenticated_v20i_folds[family],
            authenticated_v20l_fold=parent.authenticated_v20l_folds[family],
            authenticated_v20m_fold=parent.authenticated_v20m_folds[family],
            authenticated_v20o_fold=parent.authenticated_v20o_folds[family],
        )
        for family in families
    }
    if {
        family: fold["fragment_sha256"] for family, fold in folds.items()
    } != _V20P_FOLD_SHA256S:
        raise RuntimeError("pinned V20p fold authority differs")
    source = _hashed(
        {
            "schema": "fisher_graph.soft_polarity_token_vjp_source.v20q",
            "v20p_report_sha256": _V20P_LOGICAL_SHA256,
            "v20p_file_sha256": _V20P_FILE_SHA256,
            "v20p_source_receipt_sha256": _V20P_SOURCE_SHA256,
            "v20p_fold_fragment_sha256s_by_family": dict(
                sorted(_V20P_FOLD_SHA256S.items())
            ),
            "v20p_classification": report.get("classification"),
            "v20p_integrity_passed": decision.get("integrity_passed"),
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "transfer_protocol_sha256": _TRANSFER_PROTOCOL_SHA256,
            "authenticated_before_model_construction": True,
            "historically_reused_A16_only": True,
            "fresh_validation_claim": False,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
        },
        domain=_SOURCE_DOMAIN,
    )
    return _Authorities(parent, dict(report), folds, source)


def _feature_index(feature_id: str) -> int:
    feature = _identifier(feature_id, label="V20q feature")
    try:
        return _FEATURES.index(feature)
    except ValueError as error:
        raise ValueError("V20q feature is outside the frozen set") from error


def _provider_seed(
    *,
    endpoint_sha256: str,
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    logical_candidate_id: str,
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
            "endpoint_receipt_sha256": _sha(endpoint_sha256, label="V20q endpoint"),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256, label="V20q direction"
            ),
            "reflection_fit_sha256": _sha(
                reflection_fit_sha256, label="V20q reflection fit"
            ),
            "response": response,
            "logical_candidate_id": _identifier(
                logical_candidate_id, label="V20q logical candidate"
            ),
            "feature_id": _FEATURES[_feature_index(feature_id)],
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


def _materialize_provider(
    endpoint: object,
    *,
    direction: Sequence[float],
    direction_artifact_sha256: str,
    reflection_fit_sha256: str,
    response: tuple[float, float, float],
    logical_candidate_id: str,
    feature_id: str,
    field_bias: float,
    field_slope: float,
    outer_family_id: str,
    inner_family_id: str | None,
    role: str,
) -> tuple[AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider, str]:
    seed = _provider_seed(
        endpoint_sha256=str(endpoint.receipt["artifact_sha256"]),
        direction_artifact_sha256=direction_artifact_sha256,
        reflection_fit_sha256=reflection_fit_sha256,
        response=response,
        logical_candidate_id=logical_candidate_id,
        feature_id=feature_id,
        field_bias=field_bias,
        field_slope=field_slope,
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        role=role,
    )
    provider = build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field(
        endpoint.base_provider,
        endpoint.proposal_provider,
        direction=torch.tensor(tuple(direction), dtype=torch.float64),
        radius=response[0],
        shrink_mass=response[1],
        polarity_bias=response[2],
        field_bias=field_bias,
        field_slope=field_slope,
        feature_id=_feature_index(feature_id),
        transfer_protocol_sha256=_TRANSFER_PROTOCOL_SHA256,
        transfer_evidence_sha256=seed,
    )
    return provider, seed


def _training_field_correction(
    provider: AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider,
    sequence: object,
) -> Tensor:
    """Replay V20p modal generation from the authenticated training trace."""

    parent = _training_parent_modal(provider.parent_provider, sequence)
    coordinates = provider.bounded_coordinates(parent)
    delta = provider.terms_from_parent(parent, coordinates)[-1]
    modal = (parent + delta).contiguous()
    support = sequence.support_mask.to(device=modal.device)
    modal = modal.masked_fill((~support).unsqueeze(-1), 0.0)
    correction = modal.to(dtype=torch.float64) @ provider.parent_provider.output_decoder.to(
        device=modal.device, dtype=torch.float64
    )
    correction = correction.masked_fill((~support).unsqueeze(-1), 0.0).contiguous()
    if not bool(torch.isfinite(correction[support]).all()) or bool(
        (correction[~support] != 0.0).any()
    ):
        raise RuntimeError("V20q training-field correction is invalid")
    return correction


def _canonical_supervised_grid(indices: Tensor) -> Tensor:
    if (
        not isinstance(indices, Tensor)
        or indices.ndim != 1
        or indices.dtype != torch.int64
        or indices.shape[0] <= 0
    ):
        raise ValueError("V20q supervised positions must be nonempty int64 [N]")
    positions = indices.detach().to(device="cpu").contiguous()
    if bool((positions < 0).any()) or (
        positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all())
    ):
        raise ValueError("V20q supervised positions must be canonical")
    return torch.stack((torch.zeros_like(positions), positions), dim=1).contiguous()


class _V20qCachedTeacherCapability:
    """V20q-local logical view over one authenticated legacy capability.

    The legacy capability authenticates every authorized row at issuance.  V20q
    then reuses those exact tensor objects for its many logical candidate reads
    without re-hashing the full row on every read.  ``receipt()`` delegates back
    to the legacy capability once more, so any mutation is still detected before
    a phase can publish its access receipt.

    The synthesized receipt is byte-for-byte compatible with the legacy receipt
    that would have resulted from the same logical access sequence.  This keeps
    fold tensors, candidates, gates, and persisted receipt schemas unchanged.
    """

    __slots__ = (
        "_accesses",
        "_completion_integrity_check_count",
        "_completion_integrity_check_passed",
        "_families",
        "_finalized",
        "_issuance_receipt",
        "_legacy",
        "_phase",
        "_rows",
    )

    def __init__(
        self,
        *,
        phase: str,
        legacy_capability: object,
        rows: Mapping[str, Tensor],
        families: Mapping[str, str],
        issuance_receipt: Mapping[str, object],
    ) -> None:
        self._phase = _identifier(phase, label="V20q teacher capability phase")
        self._legacy = legacy_capability
        self._rows = dict(rows)
        self._families = dict(families)
        self._issuance_receipt = dict(issuance_receipt)
        self._accesses: list[str] = []
        self._completion_integrity_check_count = 0
        self._completion_integrity_check_passed = False
        self._finalized = False
        if not self._rows or set(self._rows) != set(self._families):
            raise ValueError("V20q cached teacher rows/families differ")
        if self._issuance_receipt.get("access_count") != len(self._rows):
            raise RuntimeError("V20q teacher issuance access count differs")
        issuance_counts = _mapping(
            self._issuance_receipt.get("per_example_access_counts"),
            label="V20q teacher issuance counts",
        )
        if dict(issuance_counts) != {key: 1 for key in sorted(self._rows)}:
            raise RuntimeError("V20q teacher issuance row coverage differs")

    def get(self, example_id: str, *, family_id: str) -> Tensor:
        if self._finalized:
            raise RuntimeError("V20q teacher capability is already finalized")
        example = _identifier(example_id, label="V20q teacher example")
        family = _identifier(family_id, label="V20q teacher family")
        if example not in self._rows or self._families.get(example) != family:
            raise PermissionError("teacher row is outside this V20q capability")
        self._accesses.append(example)
        return self._rows[example]

    def receipt(self) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("V20q teacher capability was finalized twice")
        self._completion_integrity_check_count += 1
        legacy_receipt = self._legacy.receipt()
        if _v14._canonical_json_bytes(legacy_receipt) != _v14._canonical_json_bytes(
            self._issuance_receipt
        ):
            raise RuntimeError("V20q teacher capability changed after issuance")
        counts = {
            example: self._accesses.count(example) for example in sorted(self._rows)
        }
        receipt = {
            **self._issuance_receipt,
            "access_count": len(self._accesses),
            "per_example_access_counts": counts,
        }
        self._completion_integrity_check_passed = True
        self._finalized = True
        return receipt

    def phase_access_accounting(self) -> dict[str, object]:
        return {
            "phase": self._phase,
            "authorized_example_count": len(self._rows),
            "physical_legacy_teacher_row_fetch_count": len(self._rows),
            "logical_teacher_row_access_count": len(self._accesses),
            "per_example_logical_access_counts": {
                example: self._accesses.count(example)
                for example in sorted(self._rows)
            },
            "issuance_integrity_check_count": 1,
            "completion_integrity_check_count": (
                self._completion_integrity_check_count
            ),
            "per_logical_read_full_row_rehash_count": 0,
            "completion_integrity_check_passed": (
                self._completion_integrity_check_passed
            ),
            "persisted_capability_receipt_schema_unchanged": True,
        }


def _issue_v20q_cached_teacher_capability(
    teacher_vault: object,
    records: Sequence[object],
    *,
    held_family_id: str | None,
    phase: str,
) -> _V20qCachedTeacherCapability:
    ordered = _v20b._ordered_records(records)
    if not ordered:
        raise ValueError("V20q teacher capability records are empty")
    examples = tuple(record.sequence.example_id for record in ordered)
    families = {
        record.sequence.example_id: record.sequence.family_id for record in ordered
    }
    if len(set(examples)) != len(examples) or set(examples) != set(families):
        raise ValueError("V20q teacher capability record ownership differs")
    legacy = teacher_vault.capability(examples, held_family_id=held_family_id)
    rows = {
        example: legacy.get(example, family_id=families[example])
        for example in examples
    }
    issuance_receipt = legacy.receipt()
    _v20b._validate_capability_receipt(
        issuance_receipt,
        expected_example_ids=examples,
        expected_family_count=len(set(families.values())),
        expected_held_family_id=held_family_id,
        expected_accesses_per_example=1,
        label=f"V20q {phase} teacher issuance",
    )
    return _V20qCachedTeacherCapability(
        phase=phase,
        legacy_capability=legacy,
        rows=rows,
        families=families,
        issuance_receipt=issuance_receipt,
    )


def _finalize_v20q_cached_teacher_capability(
    capability: _V20qCachedTeacherCapability,
    *,
    expected_example_ids: Sequence[str],
    expected_family_count: int,
    expected_held_family_id: str | None,
    expected_accesses_per_example: int,
    label: str,
) -> dict[str, object]:
    if not isinstance(capability, _V20qCachedTeacherCapability):
        raise TypeError("V20q teacher capability wrapper differs")
    receipt = capability.receipt()
    validated = _v20b._validate_capability_receipt(
        receipt,
        expected_example_ids=expected_example_ids,
        expected_family_count=expected_family_count,
        expected_held_family_id=expected_held_family_id,
        expected_accesses_per_example=expected_accesses_per_example,
        label=label,
    )
    accounting = capability.phase_access_accounting()
    expected_logical = len(tuple(expected_example_ids)) * expected_accesses_per_example
    if (
        accounting["authorized_example_count"] != len(tuple(expected_example_ids))
        or accounting["physical_legacy_teacher_row_fetch_count"]
        != len(tuple(expected_example_ids))
        or accounting["logical_teacher_row_access_count"] != expected_logical
        or accounting["issuance_integrity_check_count"] != 1
        or accounting["completion_integrity_check_count"] != 1
        or accounting["per_logical_read_full_row_rehash_count"] != 0
        or accounting["completion_integrity_check_passed"] is not True
        or accounting["persisted_capability_receipt_schema_unchanged"] is not True
    ):
        raise RuntimeError("V20q teacher phase/access accounting differs")
    return validated


@dataclass(slots=True)
class _ChartCollection:
    feature_id: str
    seed_sign: int
    prompt_records: tuple[SoftPolarityTokenVJPPromptRecord, ...]
    receipt: dict[str, object]


def _chart_id(feature_id: str, seed_sign: int) -> str:
    return f"{feature_id}__seed_{'neg' if seed_sign < 0 else 'pos'}"


def _collect_chart_records(
    context: object,
    endpoint: object,
    records: Sequence[object],
    capability: object,
    *,
    outer_family_id: str,
    reflection_fit: Mapping[str, object],
    response: tuple[float, float, float],
    feature_id: str,
    seed_sign: int,
) -> _ChartCollection:
    if seed_sign not in _SEED_SIGNS:
        raise ValueError("V20q seed sign must be -1 or +1")
    outer = _identifier(outer_family_id, label="V20q chart outer family")
    feature = _FEATURES[_feature_index(feature_id)]
    direction = _v20p._selected_direction(reflection_fit)
    direction_artifact = _sha(
        reflection_fit.get("selected_variant_artifact_sha256"),
        label="V20q selected direction",
    )
    reflection_sha = _sha(
        reflection_fit.get("artifact_sha256"), label="V20q reflection fit"
    )
    chart = _chart_id(feature, seed_sign)
    center_b = seed_sign * _SEED_ABS_B
    center_a = 0.0

    specifications = {
        "center": (center_b, center_a),
        "primary_bias_minus": (
            center_b - SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
            center_a,
        ),
        "primary_bias_plus": (
            center_b + SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
            center_a,
        ),
        "primary_slope_minus": (
            center_b,
            center_a - SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
        ),
        "primary_slope_plus": (
            center_b,
            center_a + SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
        ),
        "audit_bias_minus": (
            center_b - SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
            center_a,
        ),
        "audit_bias_plus": (
            center_b + SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
            center_a,
        ),
        "audit_slope_minus": (
            center_b,
            center_a - SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
        ),
        "audit_slope_plus": (
            center_b,
            center_a + SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
        ),
    }
    providers: dict[str, AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider] = {}
    provider_seeds: dict[str, str] = {}
    for role, (bias, slope) in specifications.items():
        providers[role], provider_seeds[role] = _materialize_provider(
            endpoint,
            direction=direction,
            direction_artifact_sha256=direction_artifact,
            reflection_fit_sha256=reflection_sha,
            response=response,
            logical_candidate_id=f"{chart}__{role}",
            feature_id=feature,
            field_bias=bias,
            field_slope=slope,
            outer_family_id=outer,
            inner_family_id=None,
            role="inner_token_vjp_chart",
        )
    if len({provider.artifact_sha256 for provider in providers.values()}) != 9:
        raise RuntimeError("V20q chart provider artifacts are not distinct")

    prompt_records: list[SoftPolarityTokenVJPPromptRecord] = []
    prompt_evidence: dict[str, dict[str, object]] = {}
    for record in _v20b._ordered_records(records):
        sequence = record.sequence
        model_inputs, supervised_positions, _targets = _v20a._verified_model_inputs(
            context, record
        )
        teacher_rows = capability.get(
            sequence.example_id, family_id=sequence.family_id
        )
        support_positions = sequence.support_mask.index_select(
            0, supervised_positions.detach().to(device="cpu")
        )
        if not bool(support_positions.any()):
            raise RuntimeError("V20q prompt has no supervised token on H4 support")
        selected_positions = supervised_positions.detach().to(device="cpu")[
            support_positions
        ].contiguous()
        selected_rows = teacher_rows.detach().to(device="cpu")[
            support_positions
        ].contiguous()
        supervised_grid = _canonical_supervised_grid(selected_positions)
        input_ids = _mapping(model_inputs, label="V20q model inputs").get("input_ids")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise RuntimeError("V20q model inputs lack a token grid")
        teacher_grid, teacher_grid_receipt = build_selected_teacher_grid(
            selected_rows,
            supervised_grid,
            batch_size=int(input_ids.shape[0]),
            sequence_length=int(input_ids.shape[1]),
        )
        vjp = context.bridge.execute_h4_token_teacher_kl_vjps(
            context.adapter,
            model_inputs,
            teacher_logits=teacher_grid.to(device=input_ids.device),
            supervised_indices=supervised_grid.to(device=input_ids.device),
            vjp_chunk_size=_VJP_CHUNK_SIZE,
            h4_head=providers["center"].runtime_provider,
            objective_dtype=torch.float64,
        )
        vjp.validate_integrity()
        runtime_teacher_grid_sha = _shadow_runtime._runtime_tensor_sha256(
            teacher_grid
        )
        runtime_supervised_indices_sha = _shadow_runtime._runtime_tensor_sha256(
            supervised_grid
        )
        if (
            vjp.teacher_logits_sha256 != runtime_teacher_grid_sha
            or vjp.h4_head_sha256
            != providers["center"].runtime_provider.artifact_sha256
            or vjp.vjp_chunk_size != _VJP_CHUNK_SIZE
            or vjp.objective_dtype != str(torch.float64)
            or vjp.backward_call_count
            != (vjp.token_count + _VJP_CHUNK_SIZE - 1) // _VJP_CHUNK_SIZE
            or not torch.equal(
                vjp.supervised_indices.detach().to(device="cpu"),
                supervised_grid,
            )
        ):
            raise RuntimeError("V20q token teacher-KL VJP authority differs")
        vjp_scalar_receipt = _hashed(
            {
                "token_teacher_kl_vjp_artifact_sha256": vjp.artifact_sha256,
                "execution_artifact_sha256": vjp.execution.artifact_sha256,
                "teacher_grid_receipt_sha256": teacher_grid_receipt.artifact_sha256,
                "teacher_grid_runtime_sha256": runtime_teacher_grid_sha,
                "supervised_indices_runtime_sha256": runtime_supervised_indices_sha,
                "teacher_logits_shape": vjp.teacher_logits_shape,
                "h4_head_runtime_provider_sha256": vjp.h4_head_sha256,
                "vjp_chunk_size": vjp.vjp_chunk_size,
                "backward_call_count": vjp.backward_call_count,
                "token_count": vjp.token_count,
                "model_forward_count": vjp.model_forward_count,
                "objective_dtype": vjp.objective_dtype,
                "teacher_grid_runtime_hash_replay_exact": True,
                "supervised_indices_replay_exact": True,
                "raw_teacher_logit_h4_or_gradient_tensors_serialized": False,
            },
            domain=_COLLECTION_DOMAIN,
        )
        reference_h4 = sequence.base_h4.unsqueeze(0)
        support_mask = sequence.support_mask.unsqueeze(0)
        corrections = {
            role: _training_field_correction(provider, sequence).unsqueeze(0)
            for role, provider in providers.items()
        }
        primary_center, primary_tangents, primary_receipt = (
            build_soft_polarity_post_cast_h4_secants(
                reference_h4=reference_h4,
                center_correction=corrections["center"],
                bias_minus_correction=corrections["primary_bias_minus"],
                bias_plus_correction=corrections["primary_bias_plus"],
                slope_minus_correction=corrections["primary_slope_minus"],
                slope_plus_correction=corrections["primary_slope_plus"],
                support_mask=support_mask,
                half_step=SOFT_POLARITY_TOKEN_VJP_PRIMARY_SECANT_HALF_STEP,
                reference_provider_sha256=providers["center"].artifact_sha256,
                bias_minus_provider_sha256=providers[
                    "primary_bias_minus"
                ].artifact_sha256,
                bias_plus_provider_sha256=providers[
                    "primary_bias_plus"
                ].artifact_sha256,
                slope_minus_provider_sha256=providers[
                    "primary_slope_minus"
                ].artifact_sha256,
                slope_plus_provider_sha256=providers[
                    "primary_slope_plus"
                ].artifact_sha256,
            )
        )
        audit_center, audit_tangents, audit_receipt = (
            build_soft_polarity_post_cast_h4_secants(
                reference_h4=reference_h4,
                center_correction=corrections["center"],
                bias_minus_correction=corrections["audit_bias_minus"],
                bias_plus_correction=corrections["audit_bias_plus"],
                slope_minus_correction=corrections["audit_slope_minus"],
                slope_plus_correction=corrections["audit_slope_plus"],
                support_mask=support_mask,
                half_step=SOFT_POLARITY_TOKEN_VJP_AUDIT_SECANT_HALF_STEP,
                reference_provider_sha256=providers["center"].artifact_sha256,
                bias_minus_provider_sha256=providers[
                    "audit_bias_minus"
                ].artifact_sha256,
                bias_plus_provider_sha256=providers[
                    "audit_bias_plus"
                ].artifact_sha256,
                slope_minus_provider_sha256=providers[
                    "audit_slope_minus"
                ].artifact_sha256,
                slope_plus_provider_sha256=providers[
                    "audit_slope_plus"
                ].artifact_sha256,
            )
        )
        stability = soft_polarity_post_cast_h4_secant_stability(
            primary_tangents, audit_tangents
        )
        if stability["passed"] is not True:
            raise RuntimeError("V20q post-cast H4 secant stability gate failed")
        if (
            not torch.equal(primary_center, audit_center)
            or not torch.equal(
                primary_center.to(
                    device=vjp.execution.candidate_h4.device,
                    dtype=vjp.execution.candidate_h4.dtype,
                ),
                vjp.execution.candidate_h4,
            )
            or vjp.execution.h4_head_sha256
            != providers["center"].runtime_provider.artifact_sha256
        ):
            raise RuntimeError("V20q center post-cast H4 replay differs")
        q = contract_soft_polarity_token_h4_vjps(
            token_h4_gradients=vjp.h4_gradients.detach().to(
                device="cpu", dtype=torch.float64
            ),
            local_h4_tangents=primary_tangents.detach().to(
                device="cpu", dtype=torch.float64
            ),
            canonical_support_mask=support_mask.to(device="cpu"),
            supervised_indices=supervised_grid.to(device="cpu"),
        )
        token_kl = vjp.token_kl_divergences.detach().to(
            device="cpu", dtype=torch.float64
        ).contiguous()
        record_receipt = SoftPolarityTokenVJPPromptRecord(
            feature_id=feature,
            family_id=sequence.family_id,
            example_id=sequence.example_id,
            reference_b=float(center_b),
            reference_a=0.0,
            derivative_convention=_DERIVATIVE_CONVENTION,
            derivative_artifact_sha256s=tuple(
                sorted(
                    {
                        teacher_grid_receipt.artifact_sha256,
                        vjp.artifact_sha256,
                        primary_receipt.artifact_sha256,
                        audit_receipt.artifact_sha256,
                    }
                )
            ),
            token_teacher_kl=token_kl,
            token_parameter_gradients=q,
        )
        prompt_records.append(record_receipt)
        prompt_evidence[sequence.example_id] = {
            "family_id": sequence.family_id,
            "prompt_record": record_receipt.metadata(),
            "teacher_grid_receipt": teacher_grid_receipt.metadata(),
            "token_teacher_kl_vjp_receipt": vjp_scalar_receipt,
            "primary_secant_receipt": primary_receipt.metadata(),
            "audit_secant_receipt": audit_receipt.metadata(),
            "secant_stability": stability,
            "center_post_cast_h4_replay_exact": True,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        }
        del model_inputs, teacher_rows, teacher_grid, vjp, corrections, q, token_kl

    ordered_records = tuple(
        sorted(prompt_records, key=lambda item: (item.family_id, item.example_id))
    )
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "chart_id": chart,
            "feature_id": feature,
            "seed_sign": seed_sign,
            "seed_bias": float(center_b),
            "seed_bias_hex": float(center_b).hex(),
            "seed_slope": 0.0,
            "provider_artifact_sha256s_by_role": {
                role: providers[role].artifact_sha256 for role in sorted(providers)
            },
            "runtime_provider_artifact_sha256s_by_role": {
                role: providers[role].runtime_provider.artifact_sha256
                for role in sorted(providers)
            },
            "provider_transfer_evidence_sha256s_by_role": dict(
                sorted(provider_seeds.items())
            ),
            "prompt_record_artifact_sha256s": tuple(
                record.artifact_sha256 for record in ordered_records
            ),
            "prompt_evidence_by_example": dict(sorted(prompt_evidence.items())),
            "prompt_count": len(ordered_records),
            "family_count": len({record.family_id for record in ordered_records}),
            "all_primary_and_audit_secant_stability_gates_passed": True,
            "all_center_post_cast_h4_replays_exact": True,
            "outer_family_absent": all(
                record.family_id != outer for record in ordered_records
            ),
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_COLLECTION_DOMAIN,
    )
    if (
        len(ordered_records) != _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or receipt["family_count"] != _INNER_FAMILY_COUNT
        or receipt["outer_family_absent"] is not True
    ):
        raise RuntimeError("V20q chart collection geometry differs")
    return _ChartCollection(feature, seed_sign, ordered_records, receipt)


def _collect_all_charts(
    context: object,
    endpoint: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    reflection_fit: Mapping[str, object],
    response: tuple[float, float, float],
) -> tuple[dict[tuple[str, int], _ChartCollection], dict[str, object]]:
    outer = _identifier(outer_family_id, label="V20q collection outer family")
    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id != outer)
    )
    families = tuple(sorted({record.sequence.family_id for record in held}))
    if (
        len(held) != _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or len(families) != _INNER_FAMILY_COUNT
        or outer in families
    ):
        raise RuntimeError("V20q chart training-panel geometry differs")
    capability = _issue_v20q_cached_teacher_capability(
        teacher_vault,
        held,
        held_family_id=outer,
        phase="token_vjp_chart_collection",
    )
    result: dict[tuple[str, int], _ChartCollection] = {}
    for feature in _FEATURES:
        for sign in _SEED_SIGNS:
            result[(feature, sign)] = _collect_chart_records(
                context,
                endpoint,
                held,
                capability,
                outer_family_id=outer,
                reflection_fit=reflection_fit,
                response=response,
                feature_id=feature,
                seed_sign=sign,
            )
    capability_receipt = _finalize_v20q_cached_teacher_capability(
        capability,
        expected_example_ids=tuple(record.sequence.example_id for record in held),
        expected_family_count=_INNER_FAMILY_COUNT,
        expected_held_family_id=outer,
        expected_accesses_per_example=len(_FEATURES) * len(_SEED_SIGNS),
        label="V20q token-VJP chart capability",
    )
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "training_family_ids": families,
            "chart_order": tuple(_chart_id(feature, sign) for feature in _FEATURES for sign in _SEED_SIGNS),
            "chart_receipt_sha256s": {
                _chart_id(feature, sign): result[(feature, sign)].receipt[
                    "artifact_sha256"
                ]
                for feature in _FEATURES
                for sign in _SEED_SIGNS
            },
            "chart_receipts_by_id": {
                _chart_id(feature, sign): result[(feature, sign)].receipt
                for feature in _FEATURES
                for sign in _SEED_SIGNS
            },
            "token_vjp_capability_receipt": capability_receipt,
            "all_eight_charts_collected_before_inner_exact_KL_capability": True,
            "outer_family_absent_from_all_chart_records": True,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_COLLECTION_DOMAIN,
    )
    return result, receipt


def _execution_seed(
    *,
    provider_manifest_sha256: str,
    outer_family_id: str,
    inner_family_id: str | None,
    logical_candidate_id: str,
    provider_artifact_sha256: str,
    phase: str,
) -> str:
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "provider_manifest_sha256": _sha(
                provider_manifest_sha256, label="V20q execution manifest"
            ),
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "logical_candidate_id": logical_candidate_id,
            "provider_artifact_sha256": _sha(
                provider_artifact_sha256, label="V20q execution provider"
            ),
            "phase": phase,
            "provider_frozen_before_teacher_capability": True,
        },
        domain=_EXECUTION_DOMAIN,
    )


def _score_exact_provider(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider,
    phase: str,
    outer_family_id: str,
    inner_family_id: str | None,
    logical_candidate_id: str,
    evidence_sha256: str,
) -> tuple[float, dict[str, object]]:
    objectives, h4_hashes, logits_hashes, execution_hashes = (
        _v20p._v20o._score_exact_provider(
            context,
            records,
            capability,
            provider=provider.runtime_provider,
            phase=phase,
            outer_family_id=outer_family_id,
            inner_family_id=inner_family_id,
            role=logical_candidate_id,
            evidence_sha256=evidence_sha256,
            domain=_EXECUTION_DOMAIN,
        )
    )
    macro, family_scores = _v19._family_equal_mean(objectives, records)
    expected_families = {record.sequence.family_id for record in records}
    if set(family_scores) != expected_families or len(expected_families) != 1:
        raise RuntimeError("V20q exact candidate score family geometry differs")
    receipt = _hashed(
        {
            "phase": phase,
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "logical_candidate_id": logical_candidate_id,
            "provider_artifact_sha256": provider.artifact_sha256,
            "runtime_provider_artifact_sha256": provider.runtime_provider.artifact_sha256,
            "evidence_sha256": evidence_sha256,
            "objective": macro,
            "objectives_by_example": dict(sorted(objectives.items())),
            "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
            "supervised_full_vocab_logits_sha256s": dict(
                sorted(logits_hashes.items())
            ),
            "execution_sha256s": dict(sorted(execution_hashes.items())),
            "exact_float64_full_vocabulary_teacher_KL": True,
            "raw_teacher_logit_h4_or_token_tensors_serialized": False,
        },
        domain=_EXECUTION_DOMAIN,
    )
    return macro, receipt


def _training_secant_summary(
    collection: _ChartCollection,
    *,
    training_family_ids: Sequence[str],
) -> tuple[str, str, dict[str, object]]:
    families = tuple(sorted(_identifier(value, label="V20q fit family") for value in training_family_ids))
    if len(families) not in (6, 7) or len(set(families)) != len(families):
        raise ValueError("V20q secant summary requires six or seven families")
    prompt_evidence = _mapping(
        collection.receipt.get("prompt_evidence_by_example"),
        label="V20q chart prompt evidence",
    )
    selected_ids = tuple(
        record.example_id
        for record in collection.prompt_records
        if record.family_id in families
    )
    if not selected_ids or {
        record.family_id
        for record in collection.prompt_records
        if record.example_id in selected_ids
    } != set(families):
        raise ValueError("V20q secant summary family coverage differs")
    primary_hashes: list[str] = []
    audit_hashes: list[str] = []
    cosines: list[tuple[float, float]] = []
    ratios: list[tuple[float, float]] = []
    for example_id in sorted(selected_ids):
        evidence = _mapping(
            prompt_evidence.get(example_id), label="V20q prompt secant evidence"
        )
        primary_hashes.append(
            _sha(
                _mapping(
                    evidence.get("primary_secant_receipt"),
                    label="V20q primary secant receipt",
                ).get("artifact_sha256"),
                label="V20q primary secant artifact",
            )
        )
        audit_hashes.append(
            _sha(
                _mapping(
                    evidence.get("audit_secant_receipt"),
                    label="V20q audit secant receipt",
                ).get("artifact_sha256"),
                label="V20q audit secant artifact",
            )
        )
        stability = _mapping(
            evidence.get("secant_stability"), label="V20q secant stability"
        )
        cosine = tuple(
            float(value)
            for value in _sequence(
                stability.get("cosine_by_parameter"), label="V20q secant cosines"
            )
        )
        ratio = tuple(
            float(value)
            for value in _sequence(
                stability.get("audit_to_primary_norm_ratio_by_parameter"),
                label="V20q secant norm ratios",
            )
        )
        if len(cosine) != 2 or len(ratio) != 2 or stability.get("passed") is not True:
            raise RuntimeError("V20q prompt secant stability evidence differs")
        cosines.append((cosine[0], cosine[1]))
        ratios.append((ratio[0], ratio[1]))
    minimum_cosines = tuple(min(row[index] for row in cosines) for index in range(2))
    # Preserve the ratio furthest from one on each axis; it is the conservative
    # single scalar checked again by the tensor-free policy boundary.
    worst_ratios = tuple(
        max((row[index] for row in ratios), key=lambda value: abs(math.log(value)))
        for index in range(2)
    )
    passed = all(value >= 0.99 for value in minimum_cosines) and all(
        0.80 <= value <= 1.25 for value in worst_ratios
    )
    if not passed:
        raise RuntimeError("V20q aggregate secant stability gate failed")
    primary_set_sha = _v14._sha256(
        {
            "chart_receipt_sha256": collection.receipt["artifact_sha256"],
            "training_family_ids": families,
            "primary_secant_receipt_sha256s": tuple(sorted(primary_hashes)),
        },
        domain=_COLLECTION_DOMAIN,
    )
    audit_set_sha = _v14._sha256(
        {
            "chart_receipt_sha256": collection.receipt["artifact_sha256"],
            "training_family_ids": families,
            "audit_secant_receipt_sha256s": tuple(sorted(audit_hashes)),
        },
        domain=_COLLECTION_DOMAIN,
    )
    return primary_set_sha, audit_set_sha, {
        "cosine_by_parameter": minimum_cosines,
        "audit_to_primary_norm_ratio_by_parameter": worst_ratios,
        "passed": True,
        "all_prompt_primary_and_audit_stability_gates_passed": True,
        "raw_secant_tensors_serialized": False,
    }


def _scalar_fit_output(
    collection: _ChartCollection,
    *,
    training_family_ids: Sequence[str],
    held_family_id: str,
    ridge_multiplier: float,
) -> tuple[object, object, dict[str, object], dict[str, object]]:
    families = tuple(sorted(training_family_ids))
    records = tuple(
        record for record in collection.prompt_records if record.family_id in families
    )
    aggregate = aggregate_soft_polarity_token_vjp_records(
        records, held_family_id=held_family_id
    )
    if tuple(aggregate.training_family_ids) != families:
        raise RuntimeError("V20q scalar fit family firewall differs")
    typed_direction = _token_fit.build_soft_polarity_token_vjp_natural_direction(
        aggregate, ridge_multiplier=float(ridge_multiplier)
    )
    if typed_direction.no_op:
        raise RuntimeError(
            f"V20q natural direction is degenerate: {typed_direction.no_op_reason}"
        )
    pure_direction = (
        _token_protocol.build_soft_polarity_token_vjp_natural_direction_output(
            aggregate_metadata=aggregate.metadata(),
            mean_gradient=tuple(
                float(value) for value in aggregate.mean_parameter_gradient_tensor()
            ),
            gradient_gram=tuple(
                tuple(float(value) for value in row)
                for row in aggregate.gradient_gram_tensor()
            ),
            ridge_multiplier=float(ridge_multiplier),
        )
    )
    if (
        not math.isclose(
            float(pure_direction["direction_b"]),
            typed_direction.direction_b,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(pure_direction["direction_a"]),
            typed_direction.direction_a,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
        or not math.isclose(
            float(pure_direction["predicted_derivative"]),
            typed_direction.predicted_derivative,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    ):
        raise RuntimeError("V20q tensor and scalar natural-direction authorities differ")
    primary_sha, audit_sha, stability = _training_secant_summary(
        collection, training_family_ids=families
    )
    scalar = _token_protocol.build_soft_polarity_token_vjp_scalar_fit_output(
        direction_metadata=pure_direction,
        aggregate_metadata=aggregate.metadata(),
        primary_secant_receipt_sha256=primary_sha,
        audit_secant_receipt_sha256=audit_sha,
        secant_stability=stability,
    )
    return aggregate, typed_direction, pure_direction, scalar


def _candidate_coefficients(
    candidate_id: str,
    *,
    scalar_fit_outputs: Mapping[tuple[str, int, float], Mapping[str, object]],
    incumbent_feature_id: str,
    incumbent_b: float,
    incumbent_a: float,
) -> tuple[str, float, float, Mapping[str, object] | None]:
    spec = _CANDIDATE_SPEC_BY_ID.get(candidate_id)
    if spec is None:
        raise ValueError("V20q candidate is outside the frozen library")
    role = spec["role"]
    if role == "token_vjp_fit":
        key = (
            str(spec["feature_id"]),
            int(spec["seed_sign"]),
            float(spec["ridge"]),
        )
        fit = _mapping(
            scalar_fit_outputs.get(key), label="V20q scalar fit output"
        )
        alpha = float(spec["alpha"])
        return (
            str(spec["feature_id"]),
            float(fit["reference_b"]) + alpha * float(fit["direction_b"]),
            float(fit["reference_a"]) + alpha * float(fit["direction_a"]),
            fit,
        )
    if role == "v20p_incumbent":
        return incumbent_feature_id, incumbent_b, incumbent_a, None
    return (
        str(spec["feature_id"]),
        float(spec["fixed_b"]),
        float(spec["fixed_a"]),
        None,
    )


@dataclass(slots=True)
class _InnerFamilyResult:
    candidate_receipts: dict[str, dict[str, object]]
    exact_objectives: dict[str, float]
    receipt: dict[str, object]


def _fit_and_score_inner_family(
    context: object,
    endpoint: object,
    records: Sequence[object],
    teacher_vault: object,
    charts: Mapping[tuple[str, int], _ChartCollection],
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str,
    inner_family_id: str,
    reflection_fit: Mapping[str, object],
    response: tuple[float, float, float],
    incumbent_feature_id: str,
    incumbent_b: float,
    incumbent_a: float,
    incumbent_fit_receipt_sha256: str,
) -> _InnerFamilyResult:
    all_families = tuple(sorted(all_family_ids))
    outer = _identifier(outer_family_id, label="V20q inner outer family")
    inner = _identifier(inner_family_id, label="V20q inner held family")
    training_families = tuple(
        family for family in all_families if family not in (outer, inner)
    )
    if (
        len(all_families) != _FAMILY_COUNT
        or len(training_families) != 6
        or outer == inner
    ):
        raise RuntimeError("V20q six/one family split differs")

    scalar_fit_outputs: dict[tuple[str, int, float], dict[str, object]] = {}
    direction_evidence: dict[str, dict[str, object]] = {}
    for feature in _FEATURES:
        for sign in _SEED_SIGNS:
            collection = charts[(feature, sign)]
            for ridge in _token_protocol.SOFT_POLARITY_TOKEN_VJP_RIDGE_LADDER:
                aggregate, typed_direction, pure_direction, scalar = _scalar_fit_output(
                    collection,
                    training_family_ids=training_families,
                    held_family_id=inner,
                    ridge_multiplier=float(ridge),
                )
                key = (feature, sign, float(ridge))
                scalar_fit_outputs[key] = scalar
                direction_evidence[
                    f"{_chart_id(feature, sign)}__ridge_{float(ridge).hex()}"
                ] = {
                    "aggregate": aggregate.metadata(),
                    "typed_natural_direction": typed_direction.metadata(),
                    "scalar_natural_direction": pure_direction,
                    "scalar_fit_output": scalar,
                    "training_family_ids": training_families,
                    "inner_held_family_id": inner,
                    "outer_held_family_id": outer,
                    "raw_fit_tensors_serialized": False,
                }

    direction = _v20p._selected_direction(reflection_fit)
    direction_artifact = _sha(
        reflection_fit.get("selected_variant_artifact_sha256"),
        label="V20q inner selected direction",
    )
    reflection_sha = _sha(
        reflection_fit.get("artifact_sha256"), label="V20q inner reflection fit"
    )
    held_records = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == inner)
    )
    if len(held_records) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20q inner held prompt geometry differs")

    providers: dict[
        str, AutonomousCompleteH4FisherSoftPolarityLocalSignedFieldProvider
    ] = {}
    provider_seeds: dict[str, str] = {}
    candidate_receipts: dict[str, dict[str, object]] = {}
    traces: dict[str, dict[str, object]] = {}
    coefficients: dict[str, tuple[str, float, float]] = {}
    for candidate_id in _token_protocol.SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS:
        feature, bias, slope, scalar = _candidate_coefficients(
            candidate_id,
            scalar_fit_outputs=scalar_fit_outputs,
            incumbent_feature_id=incumbent_feature_id,
            incumbent_b=incumbent_b,
            incumbent_a=incumbent_a,
        )
        provider, seed = _materialize_provider(
            endpoint,
            direction=direction,
            direction_artifact_sha256=direction_artifact,
            reflection_fit_sha256=reflection_sha,
            response=response,
            logical_candidate_id=candidate_id,
            feature_id=feature,
            field_bias=bias,
            field_slope=slope,
            outer_family_id=outer,
            inner_family_id=inner,
            role="inner_exact_KL_candidate",
        )
        kwargs: dict[str, object] = {}
        role = _CANDIDATE_SPEC_BY_ID[candidate_id]["role"]
        if role == "token_vjp_fit":
            kwargs["scalar_fit_output"] = scalar
        elif role == "v20p_incumbent":
            kwargs.update(
                incumbent_feature_id=incumbent_feature_id,
                incumbent_b=incumbent_b,
                incumbent_a=incumbent_a,
                incumbent_fit_receipt_sha256=incumbent_fit_receipt_sha256,
            )
        candidate_receipt = (
            _token_protocol.build_soft_polarity_token_vjp_candidate_receipt(
                protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
                all_development_family_ids=all_families,
                outer_held_family_id=outer,
                inner_held_family_id=inner,
                candidate_id=candidate_id,
                candidate_provider_sha256=provider.artifact_sha256,
                **kwargs,
            )
        )
        if (
            candidate_receipt.get("feature_id") != feature
            or float(candidate_receipt["b"]).hex() != bias.hex()
            or float(candidate_receipt["a"]).hex() != slope.hex()
        ):
            raise RuntimeError("V20q policy and materialized candidate differ")
        providers[candidate_id] = provider
        provider_seeds[candidate_id] = seed
        candidate_receipts[candidate_id] = candidate_receipt
        coefficients[candidate_id] = (feature, bias, slope)
        traces[candidate_id] = _v20p._field_trace(
            provider, held_records, role=f"inner_{inner}_{candidate_id}"
        )
    candidate_order = tuple(_token_protocol.SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS)
    if (
        tuple(providers) != candidate_order
        or len(providers) != 174
        or len({provider.artifact_sha256 for provider in providers.values()})
        != len(providers)
    ):
        raise RuntimeError("V20q inner candidate provider freeze differs")
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "inner_held_family_id": inner,
            "training_family_ids": training_families,
            "candidate_order": candidate_order,
            "candidate_count": len(candidate_order),
            "direction_evidence_by_chart_and_ridge": dict(
                sorted(direction_evidence.items())
            ),
            "candidate_provider_artifact_sha256s": {
                candidate_id: providers[candidate_id].artifact_sha256
                for candidate_id in candidate_order
            },
            "candidate_runtime_provider_artifact_sha256s": {
                candidate_id: providers[candidate_id].runtime_provider.artifact_sha256
                for candidate_id in candidate_order
            },
            "candidate_transfer_evidence_sha256s": provider_seeds,
            "candidate_receipt_sha256s": {
                candidate_id: candidate_receipts[candidate_id]["artifact_sha256"]
                for candidate_id in candidate_order
            },
            "candidate_trace_sha256s": {
                candidate_id: traces[candidate_id]["artifact_sha256"]
                for candidate_id in candidate_order
            },
            "candidate_coefficients_feature_b_a": coefficients,
            "all_174_logical_candidates_and_traces_frozen_before_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "outer_family_used_for_fit_or_selection": False,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_PROVIDER_DOMAIN,
    )

    capability = _issue_v20q_cached_teacher_capability(
        teacher_vault,
        held_records,
        held_family_id=outer,
        phase="inner_held_exact_KL",
    )
    objectives: dict[str, float] = {}
    execution_receipts: dict[str, dict[str, object]] = {}
    for candidate_id in candidate_order:
        seed = _execution_seed(
            provider_manifest_sha256=str(manifest["artifact_sha256"]),
            outer_family_id=outer,
            inner_family_id=inner,
            logical_candidate_id=candidate_id,
            provider_artifact_sha256=providers[candidate_id].artifact_sha256,
            phase="inner_held_exact_KL",
        )
        objective, receipt = _score_exact_provider(
            context,
            held_records,
            capability,
            provider=providers[candidate_id],
            phase="inner_held_exact_KL",
            outer_family_id=outer,
            inner_family_id=inner,
            logical_candidate_id=candidate_id,
            evidence_sha256=seed,
        )
        objectives[candidate_id] = objective
        execution_receipts[candidate_id] = receipt
    capability_receipt = _finalize_v20q_cached_teacher_capability(
        capability,
        expected_example_ids=tuple(
            record.sequence.example_id for record in held_records
        ),
        expected_family_count=1,
        expected_held_family_id=outer,
        expected_accesses_per_example=len(candidate_order),
        label="V20q inner exact-KL capability",
    )
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "inner_held_family_id": inner,
            "training_family_ids": training_families,
            "provider_manifest": manifest,
            "candidate_receipts": candidate_receipts,
            "exact_objective_by_candidate": objectives,
            "execution_receipts_by_candidate": execution_receipts,
            "capability_receipt": capability_receipt,
            "selection_not_yet_performed": True,
            "all_candidates_frozen_before_inner_held_exact_KL": True,
            "outer_family_used_for_fit_score_or_selection": False,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_EXECUTION_DOMAIN,
    )
    return _InnerFamilyResult(candidate_receipts, objectives, receipt)


def _fit_inner_oof(
    context: object,
    endpoint: object,
    records: Sequence[object],
    teacher_vault: object,
    charts: Mapping[tuple[str, int], _ChartCollection],
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str,
    reflection_fit: Mapping[str, object],
    response: tuple[float, float, float],
    authenticated_v20p_fold: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    families = tuple(sorted(all_family_ids))
    outer = _identifier(outer_family_id, label="V20q OOF outer family")
    inner_families = tuple(family for family in families if family != outer)
    selection = _mapping(
        authenticated_v20p_fold.get("field_selection_receipt"),
        label="V20q V20p incumbent selection",
    )
    incumbent_feature = _identifier(
        selection.get("selected_feature_id"), label="V20q incumbent feature"
    )
    incumbent_b = float(selection["selected_b"])
    incumbent_a = float(selection["selected_a"])
    incumbent_fit_sha = _sha(
        _mapping(
            selection.get("core_fit_receipt"), label="V20q incumbent fit receipt"
        ).get("artifact_sha256"),
        label="V20q incumbent fit artifact",
    )
    results: dict[str, _InnerFamilyResult] = {}
    for inner in inner_families:
        results[inner] = _fit_and_score_inner_family(
            context,
            endpoint,
            records,
            teacher_vault,
            charts,
            all_family_ids=families,
            outer_family_id=outer,
            inner_family_id=inner,
            reflection_fit=reflection_fit,
            response=response,
            incumbent_feature_id=incumbent_feature,
            incumbent_b=incumbent_b,
            incumbent_a=incumbent_a,
            incumbent_fit_receipt_sha256=incumbent_fit_sha,
        )
    candidate_receipts = {
        family: results[family].candidate_receipts for family in inner_families
    }
    objectives = {
        family: results[family].exact_objectives for family in inner_families
    }
    selection_receipt = (
        _token_protocol.build_soft_polarity_token_vjp_inner_oof_selection_receipt(
            protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
            all_development_family_ids=families,
            outer_held_family_id=outer,
            candidate_receipts_by_inner_family=candidate_receipts,
            exact_objectives_by_inner_family_and_candidate=objectives,
        )
    )
    _token_protocol.validate_soft_polarity_token_vjp_inner_oof_selection_receipt(
        selection_receipt,
        protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
        all_development_family_ids=families,
        outer_held_family_id=outer,
        candidate_receipts_by_inner_family=candidate_receipts,
        exact_objectives_by_inner_family_and_candidate=objectives,
    )
    receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "inner_family_order": inner_families,
            "incumbent_feature_id": incumbent_feature,
            "incumbent_b": incumbent_b,
            "incumbent_b_hex": incumbent_b.hex(),
            "incumbent_a": incumbent_a,
            "incumbent_a_hex": incumbent_a.hex(),
            "incumbent_fit_receipt_sha256": incumbent_fit_sha,
            "inner_family_receipts": {
                family: results[family].receipt for family in inner_families
            },
            "selection_receipt": selection_receipt,
            "selected_candidate_id": selection_receipt["selected_candidate_id"],
            "all_seven_six_to_one_folds_completed": True,
            "outer_family_used_for_fit_score_or_selection": False,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_DECISION_DOMAIN,
    )
    return selection_receipt, receipt


@dataclass(slots=True)
class _OuterResult:
    final_refit_receipt: dict[str, object]
    provider_manifest: dict[str, object]
    held_evidence: dict[str, object]
    fold_receipt: dict[str, object]


def _refit_and_score_outer(
    context: object,
    endpoint: object,
    records: Sequence[object],
    teacher_vault: object,
    charts: Mapping[tuple[str, int], _ChartCollection],
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str,
    reflection_fit: Mapping[str, object],
    response: tuple[float, float, float],
    selection_receipt: Mapping[str, object],
    authenticated_v20p_fold: Mapping[str, object],
) -> _OuterResult:
    families = tuple(sorted(all_family_ids))
    outer = _identifier(outer_family_id, label="V20q final outer family")
    training_families = tuple(family for family in families if family != outer)
    selected_id = _identifier(
        selection_receipt.get("selected_candidate_id"),
        label="V20q selected candidate",
    )
    spec = _CANDIDATE_SPEC_BY_ID[selected_id]
    incumbent_selection = _mapping(
        authenticated_v20p_fold.get("field_selection_receipt"),
        label="V20q final incumbent selection",
    )
    incumbent_feature = _identifier(
        incumbent_selection.get("selected_feature_id"),
        label="V20q final incumbent feature",
    )
    incumbent_b = float(incumbent_selection["selected_b"])
    incumbent_a = float(incumbent_selection["selected_a"])
    incumbent_fit_sha = _sha(
        _mapping(
            incumbent_selection.get("core_fit_receipt"),
            label="V20q final incumbent fit",
        ).get("artifact_sha256"),
        label="V20q final incumbent fit artifact",
    )

    scalar_fit: dict[str, object] | None = None
    refit_evidence: dict[str, object] | None = None
    if spec["role"] == "token_vjp_fit":
        collection = charts[(str(spec["feature_id"]), int(spec["seed_sign"]))]
        aggregate, typed_direction, pure_direction, scalar_fit = _scalar_fit_output(
            collection,
            training_family_ids=training_families,
            held_family_id=outer,
            ridge_multiplier=float(spec["ridge"]),
        )
        alpha = float(spec["alpha"])
        feature = str(spec["feature_id"])
        bias = float(scalar_fit["reference_b"]) + alpha * float(
            scalar_fit["direction_b"]
        )
        slope = float(scalar_fit["reference_a"]) + alpha * float(
            scalar_fit["direction_a"]
        )
        refit_evidence = {
            "aggregate": aggregate.metadata(),
            "typed_natural_direction": typed_direction.metadata(),
            "scalar_natural_direction": pure_direction,
            "scalar_fit_output": scalar_fit,
            "selected_alpha": alpha,
            "reused_all_seven_chart_VJPs_without_new_teacher_access": True,
            "raw_fit_tensors_serialized": False,
        }
    elif spec["role"] == "v20p_incumbent":
        feature, bias, slope = incumbent_feature, incumbent_b, incumbent_a
    else:
        feature = str(spec["feature_id"])
        bias = float(spec["fixed_b"])
        slope = float(spec["fixed_a"])

    direction = _v20p._selected_direction(reflection_fit)
    provider, provider_seed = _materialize_provider(
        endpoint,
        direction=direction,
        direction_artifact_sha256=_sha(
            reflection_fit.get("selected_variant_artifact_sha256"),
            label="V20q final selected direction",
        ),
        reflection_fit_sha256=_sha(
            reflection_fit.get("artifact_sha256"),
            label="V20q final reflection fit",
        ),
        response=response,
        logical_candidate_id=selected_id,
        feature_id=feature,
        field_bias=bias,
        field_slope=slope,
        outer_family_id=outer,
        inner_family_id=None,
        role="outer_token_vjp_candidate",
    )
    final_kwargs: dict[str, object] = {}
    if spec["role"] == "token_vjp_fit":
        final_kwargs["scalar_fit_output"] = scalar_fit
    elif spec["role"] == "v20p_incumbent":
        final_kwargs.update(
            incumbent_feature_id=incumbent_feature,
            incumbent_b=incumbent_b,
            incumbent_a=incumbent_a,
            incumbent_fit_receipt_sha256=incumbent_fit_sha,
        )
    final_refit = _token_protocol.build_soft_polarity_token_vjp_all_seven_refit_receipt(
        protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
        selection_receipt=selection_receipt,
        all_development_family_ids=families,
        outer_held_family_id=outer,
        final_candidate_provider_sha256=provider.artifact_sha256,
        **final_kwargs,
    )
    _token_protocol.validate_soft_polarity_token_vjp_all_seven_refit_receipt(
        final_refit,
        protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
        selection_receipt=selection_receipt,
        all_development_family_ids=families,
        outer_held_family_id=outer,
    )
    if (
        final_refit.get("feature_id") != feature
        or float(final_refit["b"]).hex() != bias.hex()
        or float(final_refit["a"]).hex() != slope.hex()
        or final_refit.get("provider_frozen_before_outer_held_objective")
        is not True
    ):
        raise RuntimeError("V20q final policy/provider coefficients differ")

    held_records = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held_records) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20q outer held prompt geometry differs")
    trace = _v20p._field_trace(provider, held_records, role="token_vjp_reflected")
    manifest = _hashed(
        {
            "outer_held_family_id": outer,
            "selected_candidate_id": selected_id,
            "selected_candidate_spec": spec,
            "feature_id": feature,
            "b": bias,
            "b_hex": bias.hex(),
            "a": slope,
            "a_hex": slope.hex(),
            "provider_artifact_sha256": provider.artifact_sha256,
            "runtime_provider_artifact_sha256": provider.runtime_provider.artifact_sha256,
            "provider_transfer_evidence_sha256": provider_seed,
            "field_trace": trace,
            "final_refit_receipt": final_refit,
            "refit_evidence": refit_evidence,
            "provider_and_trace_frozen_before_outer_capability": True,
            "outer_capability_count_at_freeze": 0,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_PROVIDER_DOMAIN,
    )
    capability = _issue_v20q_cached_teacher_capability(
        teacher_vault,
        held_records,
        held_family_id=None,
        phase="outer_held_exact_KL",
    )
    execution_seed = _execution_seed(
        provider_manifest_sha256=str(manifest["artifact_sha256"]),
        outer_family_id=outer,
        inner_family_id=None,
        logical_candidate_id=selected_id,
        provider_artifact_sha256=provider.artifact_sha256,
        phase="outer_held_exact_KL",
    )
    candidate_objective, candidate_execution = _score_exact_provider(
        context,
        held_records,
        capability,
        provider=provider,
        phase="outer_held_exact_KL",
        outer_family_id=outer,
        inner_family_id=None,
        logical_candidate_id=selected_id,
        evidence_sha256=execution_seed,
    )
    capability_receipt = _finalize_v20q_cached_teacher_capability(
        capability,
        expected_example_ids=tuple(
            record.sequence.example_id for record in held_records
        ),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=1,
        label="V20q outer exact-KL capability",
    )

    inherited_fold_receipt = _mapping(
        authenticated_v20p_fold.get("fold_receipt"),
        label="V20q inherited V20p fold receipt",
    )
    inherited_scores = {
        key: float(value)
        for key, value in _mapping(
            inherited_fold_receipt.get("held_objective_by_arm"),
            label="V20q inherited V20p held scores",
        ).items()
    }
    incumbent_score = inherited_scores["local_signed_field_reflected"]
    prior_candidate = _mapping(
        _mapping(
            authenticated_v20p_fold.get("held_evidence"),
            label="V20q inherited V20p held evidence",
        ).get("arm_evidence"),
        label="V20q inherited V20p arm evidence",
    )["local_signed_field_reflected"]
    prior_candidate = _mapping(
        prior_candidate, label="V20q inherited V20p candidate evidence"
    )
    candidate_differs = (
        _mapping(
            candidate_execution.get("post_cast_h4_sha256s"),
            label="V20q candidate H4 hashes",
        )
        != _mapping(
            prior_candidate.get("post_cast_h4_sha256s"),
            label="V20q incumbent H4 hashes",
        )
        or _mapping(
            candidate_execution.get("supervised_full_vocab_logits_sha256s"),
            label="V20q candidate logits hashes",
        )
        != _mapping(
            prior_candidate.get("supervised_full_vocab_logits_sha256s"),
            label="V20q incumbent logits hashes",
        )
    )
    held_evidence = _hashed(
        {
            "outer_held_family_id": outer,
            "selected_candidate_id": selected_id,
            "candidate_objective": candidate_objective,
            "candidate_execution": candidate_execution,
            "candidate_provider_manifest": manifest,
            "outer_capability_receipt": capability_receipt,
            "inherited_v20p_held_objective_by_arm": inherited_scores,
            "v20p_incumbent_objective": incumbent_score,
            "candidate_strictly_beats_v20p_incumbent": candidate_objective
            < incumbent_score,
            "candidate_exact_output_differs_from_v20p_incumbent": candidate_differs,
            "provider_frozen_before_outer_held_objective": True,
            "outer_held_objective_used_for_adaptation": False,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_EXECUTION_DOMAIN,
    )
    selected_aggregate = _mapping(
        _mapping(
            selection_receipt.get("aggregate_by_candidate"),
            label="V20q OOF candidate aggregates",
        ).get(selected_id),
        label="V20q selected OOF aggregate",
    )
    incumbent_aggregate = _mapping(
        _mapping(
            selection_receipt.get("aggregate_by_candidate"),
            label="V20q OOF candidate aggregates",
        ).get(_token_protocol.SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID),
        label="V20q incumbent OOF aggregate",
    )
    selected_role = str(spec["role"])
    fold_receipt = _hashed(
        {
            "outer_held_family_id": outer,
            "selected_candidate_id": selected_id,
            "selected_candidate_role": selected_role,
            "feature_id": feature,
            "b": bias,
            "b_hex": bias.hex(),
            "a": slope,
            "a_hex": slope.hex(),
            "selected_nonzero_continuous_candidate": selected_role
            == "token_vjp_fit",
            "selected_inner_oof_mean": float(
                selected_aggregate["family_equal_exact_kl"]
            ),
            "incumbent_inner_oof_mean": float(
                incumbent_aggregate["family_equal_exact_kl"]
            ),
            "selected_inner_oof_mean_beats_incumbent": float(
                selected_aggregate["family_equal_exact_kl"]
            )
            < float(incumbent_aggregate["family_equal_exact_kl"]),
            "candidate_objective": candidate_objective,
            "inherited_v20p_held_objective_by_arm": inherited_scores,
            "candidate_strictly_beats_v20p_incumbent": candidate_objective
            < incumbent_score,
            "candidate_exact_output_differs_from_v20p_incumbent": candidate_differs,
            "candidate_field_nonconstant": trace[
                "local_signed_scalar_nonconstant"
            ],
            "candidate_field_has_negative": trace[
                "local_signed_scalar_has_negative"
            ],
            "candidate_field_has_positive": trace[
                "local_signed_scalar_has_positive"
            ],
            "all_secant_stability_gates_passed": selected_role
            != "token_vjp_fit"
            or (
                refit_evidence is not None
                and scalar_fit is not None
                and scalar_fit.get("secant_stability_passed") is True
                and charts[
                    (str(spec["feature_id"]), int(spec["seed_sign"]))
                ].receipt.get(
                    "all_primary_and_audit_secant_stability_gates_passed"
                )
                is True
            ),
            "deployed_direction_has_negative_predicted_derivative": selected_role
            != "token_vjp_fit"
            or float(scalar_fit["predicted_derivative"]) < 0.0,
            "candidate_pointwise_trust_passed": trace.get(
                "pointwise_trust_passed"
            )
            is True,
            "provider_frozen_before_outer_score": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_execution": True,
        },
        domain=_DECISION_DOMAIN,
    )
    return _OuterResult(final_refit, manifest, held_evidence, fold_receipt)


@dataclass(slots=True)
class _FoldLive:
    endpoint: object
    chart_collection_receipt: dict[str, object]
    inner_oof_receipt: dict[str, object]
    selection_receipt: dict[str, object]
    outer_result: _OuterResult


def _execute_outer_fold(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    authorities: _Authorities,
    family_ids: Sequence[str],
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
) -> _FoldLive:
    """Execute one sealed V20q outer fold from authenticated V20p controls."""

    families = tuple(sorted(family_ids))
    outer = _identifier(outer_family_id, label="V20q outer fold family")
    if len(families) != _FAMILY_COUNT or outer not in families:
        raise RuntimeError("V20q outer family geometry differs")
    v20p_fold = authorities.authenticated_v20p_folds[outer]
    if v20p_fold.get("fragment_sha256") != _V20P_FOLD_SHA256S.get(outer):
        raise RuntimeError("V20q authenticated V20p fold differs")
    parent = authorities.parent
    endpoint = _v20g._outer_endpoint(
        context,
        records,
        teacher_vault,
        family_ids=families,
        outer_family_id=outer,
        panel_receipt=panel_receipt,
        authenticated_v20a_fold=parent.authenticated_v20a_folds[outer],
    )
    if (
        _v14._canonical_json_bytes(endpoint.receipt)
        != _v14._canonical_json_bytes(v20p_fold.get("endpoint_receipt"))
        or _v14._canonical_json_bytes(endpoint.evidence)
        != _v14._canonical_json_bytes(v20p_fold.get("endpoint_evidence"))
    ):
        raise RuntimeError("V20q reconstructed endpoint differs from V20p")
    reflection_fit = dict(
        _mapping(
            v20p_fold.get("outer_reflection_fit_receipt"),
            label="V20q inherited reflection fit",
        )
    )
    response_selection = _mapping(
        v20p_fold.get("response_selection_receipt"),
        label="V20q inherited response selection",
    )
    response = _v20p._response_tuple(response_selection.get("selected_response"))
    charts, chart_receipt = _collect_all_charts(
        context,
        endpoint,
        records,
        teacher_vault,
        outer_family_id=outer,
        reflection_fit=reflection_fit,
        response=response,
    )
    selection, inner_oof = _fit_inner_oof(
        context,
        endpoint,
        records,
        teacher_vault,
        charts,
        all_family_ids=families,
        outer_family_id=outer,
        reflection_fit=reflection_fit,
        response=response,
        authenticated_v20p_fold=v20p_fold,
    )
    outer_result = _refit_and_score_outer(
        context,
        endpoint,
        records,
        teacher_vault,
        charts,
        all_family_ids=families,
        outer_family_id=outer,
        reflection_fit=reflection_fit,
        response=response,
        selection_receipt=selection,
        authenticated_v20p_fold=v20p_fold,
    )
    return _FoldLive(
        endpoint,
        chart_receipt,
        inner_oof,
        selection,
        outer_result,
    )


_FOLD_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "token_vjp_protocol_receipt",
        "source_artifact_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "v20p_fold_fragment_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "inherited_v20p_outer_reflection_fit_receipt",
        "inherited_v20p_response_selection_receipt",
        "inherited_v20p_field_selection_receipt",
        "chart_collection_receipt",
        "inner_oof_receipt",
        "selection_receipt",
        "final_refit_receipt",
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
    authenticated_v20p_fold: Mapping[str, object],
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="V20q fold family")
    final = live.outer_result.final_refit_receipt
    candidate = {
        "candidate_id": final["selected_candidate_id"],
        "candidate_role": _CANDIDATE_SPEC_BY_ID[
            str(final["selected_candidate_id"])
        ]["role"],
        "feature_id": final["feature_id"],
        "b": final["b"],
        "a": final["a"],
        "analysis_only": True,
    }
    return {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "token_vjp_protocol_receipt": _TOKEN_VJP_PROTOCOL_RECEIPT,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20p_fold_fragment_sha256": authenticated_v20p_fold["fragment_sha256"],
        "outer_held_family_id": outer,
        "endpoint_receipt": live.endpoint.receipt,
        "endpoint_evidence": live.endpoint.evidence,
        "inherited_v20p_outer_reflection_fit_receipt": authenticated_v20p_fold[
            "outer_reflection_fit_receipt"
        ],
        "inherited_v20p_response_selection_receipt": authenticated_v20p_fold[
            "response_selection_receipt"
        ],
        "inherited_v20p_field_selection_receipt": authenticated_v20p_fold[
            "field_selection_receipt"
        ],
        "chart_collection_receipt": live.chart_collection_receipt,
        "inner_oof_receipt": live.inner_oof_receipt,
        "selection_receipt": live.selection_receipt,
        "final_refit_receipt": final,
        "provider_manifest": live.outer_result.provider_manifest,
        "held_evidence": live.outer_result.held_evidence,
        "fold_receipt": live.outer_result.fold_receipt,
        "fixed_schedule_completed": True,
        "candidate": candidate,
        "provider_sidecar": None,
    }


def _validate_chart_collection_receipt(
    value: Mapping[str, object], *, outer_family_id: str
) -> None:
    receipt = _mapping(value, label="V20q chart collection receipt")
    _validate_hashed(
        receipt, domain=_COLLECTION_DOMAIN, label="V20q chart collection receipt"
    )
    expected_ids = tuple(
        _chart_id(feature, sign)
        for feature in _FEATURES
        for sign in _SEED_SIGNS
    )
    charts = _mapping(
        receipt.get("chart_receipts_by_id"), label="V20q chart receipts"
    )
    chart_hashes = _mapping(
        receipt.get("chart_receipt_sha256s"), label="V20q chart hashes"
    )
    if (
        receipt.get("outer_held_family_id") != outer_family_id
        or tuple(receipt.get("chart_order", ())) != expected_ids
        or set(charts) != set(expected_ids)
        or set(chart_hashes) != set(expected_ids)
        or receipt.get(
            "all_eight_charts_collected_before_inner_exact_KL_capability"
        )
        is not True
        or receipt.get("outer_family_absent_from_all_chart_records") is not True
    ):
        raise ValueError("V20q chart collection geometry differs")
    for chart_id in expected_ids:
        chart = _mapping(charts[chart_id], label=f"V20q chart {chart_id}")
        _validate_hashed(chart, domain=_COLLECTION_DOMAIN, label=f"V20q chart {chart_id}")
        if chart.get("artifact_sha256") != chart_hashes[chart_id]:
            raise ValueError("V20q chart hash binding differs")
        evidence = _mapping(
            chart.get("prompt_evidence_by_example"),
            label=f"V20q chart {chart_id} prompt evidence",
        )
        if len(evidence) != _INNER_FAMILY_COUNT * _PROMPTS_PER_FAMILY:
            raise ValueError("V20q chart prompt evidence geometry differs")
        for example_id, row_value in evidence.items():
            _identifier(example_id, label="V20q chart example")
            row = _mapping(row_value, label="V20q chart prompt row")
            vjp_receipt = _mapping(
                row.get("token_teacher_kl_vjp_receipt"),
                label="V20q token VJP scalar receipt",
            )
            _validate_hashed(
                vjp_receipt,
                domain=_COLLECTION_DOMAIN,
                label="V20q token VJP scalar receipt",
            )
            if (
                vjp_receipt.get("vjp_chunk_size") != _VJP_CHUNK_SIZE
                or vjp_receipt.get("objective_dtype") != str(torch.float64)
                or vjp_receipt.get("teacher_grid_runtime_hash_replay_exact")
                is not True
                or vjp_receipt.get("supervised_indices_replay_exact") is not True
                or row.get("center_post_cast_h4_replay_exact") is not True
            ):
                raise ValueError("V20q token VJP scalar authority differs")


def _execution_objective(
    value: Mapping[str, object], *, label: str
) -> tuple[float, tuple[str, ...]]:
    """Replay one exact-KL scalar from its per-example execution evidence."""

    receipt = _mapping(value, label=label)
    _validate_hashed(receipt, domain=_EXECUTION_DOMAIN, label=label)
    objectives = {
        _identifier(example_id, label=f"{label} example"): float(objective)
        for example_id, objective in _mapping(
            receipt.get("objectives_by_example"),
            label=f"{label} objectives by example",
        ).items()
    }
    example_ids = tuple(sorted(objectives))
    h4_hashes = _mapping(
        receipt.get("post_cast_h4_sha256s"), label=f"{label} H4 hashes"
    )
    logits_hashes = _mapping(
        receipt.get("supervised_full_vocab_logits_sha256s"),
        label=f"{label} logits hashes",
    )
    execution_hashes = _mapping(
        receipt.get("execution_sha256s"), label=f"{label} execution hashes"
    )
    if (
        not example_ids
        or set(h4_hashes) != set(example_ids)
        or set(logits_hashes) != set(example_ids)
        or set(execution_hashes) != set(example_ids)
        or any(not math.isfinite(value) or value < 0.0 for value in objectives.values())
        or receipt.get("exact_float64_full_vocabulary_teacher_KL") is not True
        or receipt.get("raw_teacher_logit_h4_or_token_tensors_serialized")
        is not False
    ):
        raise ValueError(f"{label} exact objective geometry differs")
    for hashes, kind in (
        (h4_hashes, "H4"),
        (logits_hashes, "logits"),
        (execution_hashes, "execution"),
    ):
        for example_id in example_ids:
            _sha(hashes[example_id], label=f"{label} {kind} {example_id}")
    objective = math.fsum(objectives[example_id] for example_id in example_ids) / len(
        example_ids
    )
    supplied = float(receipt.get("objective"))
    if not math.isfinite(supplied) or supplied.hex() != objective.hex():
        raise ValueError(f"{label} objective does not replay from examples")
    return objective, example_ids


def _selection_inputs(
    inner_oof_receipt: Mapping[str, object],
) -> tuple[
    dict[str, Mapping[str, Mapping[str, object]]],
    dict[str, Mapping[str, float]],
]:
    rows = _mapping(
        inner_oof_receipt.get("inner_family_receipts"),
        label="V20q inner family receipts",
    )
    candidate_rows: dict[str, Mapping[str, Mapping[str, object]]] = {}
    objective_rows: dict[str, Mapping[str, float]] = {}
    for family, row_value in rows.items():
        inner = _mapping(row_value, label=f"V20q inner fold {family}")
        _validate_hashed(
            inner, domain=_EXECUTION_DOMAIN, label=f"V20q inner fold {family}"
        )
        manifest = _mapping(
            inner.get("provider_manifest"), label="V20q inner provider manifest"
        )
        _validate_hashed(
            manifest, domain=_PROVIDER_DOMAIN, label="V20q inner provider manifest"
        )
        executions = _mapping(
            inner.get("execution_receipts_by_candidate"),
            label="V20q inner execution receipts",
        )
        candidates = _mapping(
            inner.get("candidate_receipts"),
            label="V20q inner candidate receipts",
        )
        objectives = {
            str(candidate_id): float(objective)
            for candidate_id, objective in _mapping(
                inner.get("exact_objective_by_candidate"),
                label="V20q inner exact objectives",
            ).items()
        }
        candidate_order = tuple(_token_protocol.SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS)
        outer = _identifier(
            inner.get("outer_held_family_id"), label="V20q inner outer family"
        )
        held = _identifier(
            inner.get("inner_held_family_id"), label="V20q inner held family"
        )
        provider_hashes = _mapping(
            manifest.get("candidate_provider_artifact_sha256s"),
            label="V20q inner provider hashes",
        )
        runtime_hashes = _mapping(
            manifest.get("candidate_runtime_provider_artifact_sha256s"),
            label="V20q inner runtime provider hashes",
        )
        candidate_hashes = _mapping(
            manifest.get("candidate_receipt_sha256s"),
            label="V20q inner candidate receipt hashes",
        )
        if (
            str(family) != held
            or tuple(manifest.get("candidate_order", ())) != candidate_order
            or manifest.get("candidate_count") != len(candidate_order)
            or manifest.get("outer_held_family_id") != outer
            or manifest.get("inner_held_family_id") != held
            or set(candidates) != set(candidate_order)
            or set(executions) != set(candidate_order)
            or set(objectives) != set(candidate_order)
            or set(provider_hashes) != set(candidate_order)
            or set(runtime_hashes) != set(candidate_order)
            or set(candidate_hashes) != set(candidate_order)
            or manifest.get(
                "all_174_logical_candidates_and_traces_frozen_before_inner_capability"
            )
            is not True
            or manifest.get("inner_capability_count_at_freeze") != 0
            or manifest.get("outer_family_used_for_fit_or_selection") is not False
            or inner.get("selection_not_yet_performed") is not True
            or inner.get("all_candidates_frozen_before_inner_held_exact_KL")
            is not True
            or inner.get("outer_family_used_for_fit_score_or_selection") is not False
        ):
            raise ValueError("V20q inner execution geometry differs")
        expected_example_ids: tuple[str, ...] | None = None
        for candidate_id in candidate_order:
            candidate = _mapping(
                candidates[candidate_id],
                label=f"V20q inner candidate {candidate_id}",
            )
            execution = _mapping(
                executions[candidate_id],
                label=f"V20q inner execution {candidate_id}",
            )
            provider_sha = _sha(
                provider_hashes[candidate_id],
                label=f"V20q inner provider {candidate_id}",
            )
            runtime_sha = _sha(
                runtime_hashes[candidate_id],
                label=f"V20q inner runtime provider {candidate_id}",
            )
            observed_objective, example_ids = _execution_objective(
                execution, label=f"V20q inner execution {candidate_id}"
            )
            expected_evidence = _execution_seed(
                provider_manifest_sha256=str(manifest["artifact_sha256"]),
                outer_family_id=outer,
                inner_family_id=held,
                logical_candidate_id=candidate_id,
                provider_artifact_sha256=provider_sha,
                phase="inner_held_exact_KL",
            )
            if expected_example_ids is None:
                expected_example_ids = example_ids
            if (
                example_ids != expected_example_ids
                or candidate.get("candidate_id") != candidate_id
                or candidate.get("candidate_provider_sha256") != provider_sha
                or candidate.get("artifact_sha256") != candidate_hashes[candidate_id]
                or execution.get("phase") != "inner_held_exact_KL"
                or execution.get("outer_held_family_id") != outer
                or execution.get("inner_held_family_id") != held
                or execution.get("logical_candidate_id") != candidate_id
                or execution.get("provider_artifact_sha256") != provider_sha
                or execution.get("runtime_provider_artifact_sha256") != runtime_sha
                or execution.get("evidence_sha256") != expected_evidence
                or float(objectives[candidate_id]).hex()
                != observed_objective.hex()
            ):
                raise ValueError("V20q inner execution semantic binding differs")
        candidate_rows[str(family)] = candidates
        objective_rows[str(family)] = objectives
    return candidate_rows, objective_rows


def _expected_held_evidence(
    *,
    outer_family_id: str,
    selected_candidate_id: str,
    manifest: Mapping[str, object],
    held_evidence: Mapping[str, object],
    authenticated_v20p_fold: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild outer evidence from execution and authenticated V20p controls."""

    outer = _identifier(outer_family_id, label="V20q held outer family")
    selected_id = _identifier(
        selected_candidate_id, label="V20q held selected candidate"
    )
    manifest_row = _mapping(manifest, label="V20q held provider manifest")
    held = _mapping(held_evidence, label="V20q held evidence")
    execution = _mapping(
        held.get("candidate_execution"), label="V20q outer candidate execution"
    )
    objective, _ = _execution_objective(
        execution, label="V20q outer candidate execution"
    )
    provider_sha = _sha(
        manifest_row.get("provider_artifact_sha256"),
        label="V20q outer provider",
    )
    runtime_sha = _sha(
        manifest_row.get("runtime_provider_artifact_sha256"),
        label="V20q outer runtime provider",
    )
    expected_execution_seed = _execution_seed(
        provider_manifest_sha256=str(manifest_row["artifact_sha256"]),
        outer_family_id=outer,
        inner_family_id=None,
        logical_candidate_id=selected_id,
        provider_artifact_sha256=provider_sha,
        phase="outer_held_exact_KL",
    )
    if (
        execution.get("phase") != "outer_held_exact_KL"
        or execution.get("outer_held_family_id") != outer
        or execution.get("inner_held_family_id") is not None
        or execution.get("logical_candidate_id") != selected_id
        or execution.get("provider_artifact_sha256") != provider_sha
        or execution.get("runtime_provider_artifact_sha256") != runtime_sha
        or execution.get("evidence_sha256") != expected_execution_seed
    ):
        raise ValueError("V20q outer execution semantic binding differs")

    inherited_fold_receipt = _mapping(
        authenticated_v20p_fold.get("fold_receipt"),
        label="V20q authenticated V20p fold receipt",
    )
    inherited_scores = {
        str(arm): float(score)
        for arm, score in _mapping(
            inherited_fold_receipt.get("held_objective_by_arm"),
            label="V20q authenticated V20p held scores",
        ).items()
    }
    if set(inherited_scores) != set(_v20p._ARMS):
        raise ValueError("V20q authenticated V20p score geometry differs")
    incumbent_score = inherited_scores["local_signed_field_reflected"]
    prior_candidate = _mapping(
        _mapping(
            _mapping(
                authenticated_v20p_fold.get("held_evidence"),
                label="V20q authenticated V20p held evidence",
            ).get("arm_evidence"),
            label="V20q authenticated V20p arm evidence",
        ).get("local_signed_field_reflected"),
        label="V20q authenticated V20p incumbent evidence",
    )
    candidate_differs = (
        _mapping(
            execution.get("post_cast_h4_sha256s"),
            label="V20q outer candidate H4 hashes",
        )
        != _mapping(
            prior_candidate.get("post_cast_h4_sha256s"),
            label="V20q authenticated incumbent H4 hashes",
        )
        or _mapping(
            execution.get("supervised_full_vocab_logits_sha256s"),
            label="V20q outer candidate logits hashes",
        )
        != _mapping(
            prior_candidate.get("supervised_full_vocab_logits_sha256s"),
            label="V20q authenticated incumbent logits hashes",
        )
    )
    capability = _mapping(
        held.get("outer_capability_receipt"),
        label="V20q outer capability receipt",
    )
    return _hashed(
        {
            "outer_held_family_id": outer,
            "selected_candidate_id": selected_id,
            "candidate_objective": objective,
            "candidate_execution": execution,
            "candidate_provider_manifest": manifest_row,
            "outer_capability_receipt": capability,
            "inherited_v20p_held_objective_by_arm": inherited_scores,
            "v20p_incumbent_objective": incumbent_score,
            "candidate_strictly_beats_v20p_incumbent": objective
            < incumbent_score,
            "candidate_exact_output_differs_from_v20p_incumbent": candidate_differs,
            "provider_frozen_before_outer_held_objective": True,
            "outer_held_objective_used_for_adaptation": False,
            "raw_teacher_logit_h4_secant_gradient_or_token_tensors_serialized": False,
        },
        domain=_EXECUTION_DOMAIN,
    )


def _expected_fold_receipt(
    *,
    outer_family_id: str,
    selection_receipt: Mapping[str, object],
    final_refit_receipt: Mapping[str, object],
    chart_collection_receipt: Mapping[str, object],
    manifest: Mapping[str, object],
    held_evidence: Mapping[str, object],
    trace: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild every decision-driving fold scalar from nested evidence."""

    outer = _identifier(outer_family_id, label="V20q receipt outer family")
    selection = _mapping(selection_receipt, label="V20q receipt selection")
    final_refit = _mapping(final_refit_receipt, label="V20q receipt final refit")
    charts = _mapping(
        chart_collection_receipt, label="V20q receipt chart collection"
    )
    manifest_row = _mapping(manifest, label="V20q receipt manifest")
    held = _mapping(held_evidence, label="V20q receipt held evidence")
    trace_row = _mapping(trace, label="V20q receipt field trace")
    selected_id = _identifier(
        final_refit.get("selected_candidate_id"),
        label="V20q receipt selected candidate",
    )
    if selected_id != selection.get("selected_candidate_id"):
        raise ValueError("V20q receipt selection and final refit differ")
    spec = _CANDIDATE_SPEC_BY_ID[selected_id]
    selected_role = str(spec["role"])
    aggregates = _mapping(
        selection.get("aggregate_by_candidate"),
        label="V20q receipt candidate aggregates",
    )
    selected_aggregate = _mapping(
        aggregates.get(selected_id), label="V20q receipt selected aggregate"
    )
    incumbent_aggregate = _mapping(
        aggregates.get(_token_protocol.SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID),
        label="V20q receipt incumbent aggregate",
    )
    scalar_fit: Mapping[str, object] | None = None
    all_secants_stable = True
    direction_descends = True
    if selected_role == "token_vjp_fit":
        scalar_fit = _mapping(
            final_refit.get("scalar_fit_output"),
            label="V20q receipt selected scalar fit",
        )
        chart_id = _chart_id(str(spec["feature_id"]), int(spec["seed_sign"]))
        selected_chart = _mapping(
            _mapping(
                charts.get("chart_receipts_by_id"),
                label="V20q receipt charts",
            ).get(chart_id),
            label="V20q receipt selected chart",
        )
        all_secants_stable = (
            scalar_fit.get("secant_stability_passed") is True
            and selected_chart.get(
                "all_primary_and_audit_secant_stability_gates_passed"
            )
            is True
        )
        direction_descends = float(scalar_fit["predicted_derivative"]) < 0.0
    if (
        final_refit.get("provider_frozen_before_outer_held_objective") is not True
        or final_refit.get("outer_held_objective_consumed") is not False
        or manifest_row.get("provider_and_trace_frozen_before_outer_capability")
        is not True
        or manifest_row.get("outer_capability_count_at_freeze") != 0
        or held.get("provider_frozen_before_outer_held_objective") is not True
        or held.get("outer_held_objective_used_for_adaptation") is not False
    ):
        raise ValueError("V20q outer freeze boundary differs")
    bias = float(final_refit["b"])
    slope = float(final_refit["a"])
    candidate_objective = float(held["candidate_objective"])
    inherited_scores = {
        str(arm): float(score)
        for arm, score in _mapping(
            held.get("inherited_v20p_held_objective_by_arm"),
            label="V20q receipt inherited scores",
        ).items()
    }
    incumbent_score = inherited_scores["local_signed_field_reflected"]
    return _hashed(
        {
            "outer_held_family_id": outer,
            "selected_candidate_id": selected_id,
            "selected_candidate_role": selected_role,
            "feature_id": final_refit["feature_id"],
            "b": bias,
            "b_hex": bias.hex(),
            "a": slope,
            "a_hex": slope.hex(),
            "selected_nonzero_continuous_candidate": selected_role
            == "token_vjp_fit",
            "selected_inner_oof_mean": float(
                selected_aggregate["family_equal_exact_kl"]
            ),
            "incumbent_inner_oof_mean": float(
                incumbent_aggregate["family_equal_exact_kl"]
            ),
            "selected_inner_oof_mean_beats_incumbent": float(
                selected_aggregate["family_equal_exact_kl"]
            )
            < float(incumbent_aggregate["family_equal_exact_kl"]),
            "candidate_objective": candidate_objective,
            "inherited_v20p_held_objective_by_arm": inherited_scores,
            "candidate_strictly_beats_v20p_incumbent": candidate_objective
            < incumbent_score,
            "candidate_exact_output_differs_from_v20p_incumbent": held[
                "candidate_exact_output_differs_from_v20p_incumbent"
            ],
            "candidate_field_nonconstant": trace_row[
                "local_signed_scalar_nonconstant"
            ],
            "candidate_field_has_negative": trace_row[
                "local_signed_scalar_has_negative"
            ],
            "candidate_field_has_positive": trace_row[
                "local_signed_scalar_has_positive"
            ],
            "all_secant_stability_gates_passed": all_secants_stable,
            "deployed_direction_has_negative_predicted_derivative": direction_descends,
            "candidate_pointwise_trust_passed": trace_row.get(
                "pointwise_trust_passed"
            )
            is True,
            "provider_frozen_before_outer_score": True,
            "outer_family_used_for_fit_or_selection": False,
            "exact_execution": True,
        },
        domain=_DECISION_DOMAIN,
    )


def _validate_held_evidence_semantics(
    held_evidence: Mapping[str, object],
    *,
    outer_family_id: str,
    selected_candidate_id: str,
    manifest: Mapping[str, object],
    authenticated_v20p_fold: Mapping[str, object],
) -> None:
    expected = _expected_held_evidence(
        outer_family_id=outer_family_id,
        selected_candidate_id=selected_candidate_id,
        manifest=manifest,
        held_evidence=held_evidence,
        authenticated_v20p_fold=authenticated_v20p_fold,
    )
    if _v14._canonical_json_bytes(held_evidence) != _v14._canonical_json_bytes(
        expected
    ):
        raise ValueError("V20q held evidence does not replay from nested evidence")


def _validate_fold_receipt_semantics(
    receipt: Mapping[str, object],
    *,
    outer_family_id: str,
    selection_receipt: Mapping[str, object],
    final_refit_receipt: Mapping[str, object],
    chart_collection_receipt: Mapping[str, object],
    manifest: Mapping[str, object],
    held_evidence: Mapping[str, object],
    trace: Mapping[str, object],
) -> None:
    expected = _expected_fold_receipt(
        outer_family_id=outer_family_id,
        selection_receipt=selection_receipt,
        final_refit_receipt=final_refit_receipt,
        chart_collection_receipt=chart_collection_receipt,
        manifest=manifest,
        held_evidence=held_evidence,
        trace=trace,
    )
    if _v14._canonical_json_bytes(receipt) != _v14._canonical_json_bytes(expected):
        raise ValueError("V20q fold receipt does not replay from nested evidence")


def _validate_fold_fragment(
    value: Mapping[str, object],
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
    all_family_ids: Sequence[str],
    authenticated_v20p_fold: Mapping[str, object],
) -> None:
    fold = _mapping(value, label="V20q fold fragment")
    if set(fold) != _FOLD_KEYS:
        raise ValueError("V20q fold fragment key set differs")
    supplied = dict(fold)
    fragment_sha = _sha(
        supplied.pop("fragment_sha256", None), label="V20q fold fragment"
    )
    if fragment_sha != _v14._sha256(supplied, domain=_FOLD_DOMAIN):
        raise ValueError("V20q fold fragment hash differs")
    expected = {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20p_fold_fragment_sha256": authenticated_v20p_fold["fragment_sha256"],
        "outer_held_family_id": outer_family_id,
        "fixed_schedule_completed": True,
        "provider_sidecar": None,
    }
    for key, expected_value in expected.items():
        if fold.get(key) != expected_value:
            raise ValueError(f"V20q fold {key} differs")
    _token_protocol.validate_soft_polarity_token_vjp_protocol_receipt(
        _mapping(
            fold.get("token_vjp_protocol_receipt"),
            label="V20q token VJP protocol",
        )
    )
    for key in (
        "endpoint_receipt",
        "endpoint_evidence",
        "outer_reflection_fit_receipt",
        "response_selection_receipt",
        "field_selection_receipt",
    ):
        fold_key = (
            key
            if key.startswith("endpoint_")
            else f"inherited_v20p_{key}"
        )
        if _v14._canonical_json_bytes(fold.get(fold_key)) != _v14._canonical_json_bytes(
            authenticated_v20p_fold.get(key)
        ):
            raise ValueError(f"V20q inherited {key} differs")
    _validate_chart_collection_receipt(
        _mapping(
            fold.get("chart_collection_receipt"),
            label="V20q chart collection",
        ),
        outer_family_id=outer_family_id,
    )
    inner_oof = _mapping(fold.get("inner_oof_receipt"), label="V20q inner OOF")
    _validate_hashed(inner_oof, domain=_DECISION_DOMAIN, label="V20q inner OOF")
    candidate_rows, objective_rows = _selection_inputs(inner_oof)
    selection = _mapping(
        fold.get("selection_receipt"), label="V20q OOF selection"
    )
    _token_protocol.validate_soft_polarity_token_vjp_inner_oof_selection_receipt(
        selection,
        protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
        all_development_family_ids=all_family_ids,
        outer_held_family_id=outer_family_id,
        candidate_receipts_by_inner_family=candidate_rows,
        exact_objectives_by_inner_family_and_candidate=objective_rows,
    )
    if (
        inner_oof.get("selection_receipt") != selection
        or inner_oof.get("selected_candidate_id")
        != selection.get("selected_candidate_id")
    ):
        raise ValueError("V20q inner OOF selection binding differs")
    final_refit = _mapping(
        fold.get("final_refit_receipt"), label="V20q final refit"
    )
    _token_protocol.validate_soft_polarity_token_vjp_all_seven_refit_receipt(
        final_refit,
        protocol_receipt=_TOKEN_VJP_PROTOCOL_RECEIPT,
        selection_receipt=selection,
        all_development_family_ids=all_family_ids,
        outer_held_family_id=outer_family_id,
    )
    manifest = _mapping(
        fold.get("provider_manifest"), label="V20q outer provider manifest"
    )
    _validate_hashed(
        manifest, domain=_PROVIDER_DOMAIN, label="V20q outer provider manifest"
    )
    trace = _mapping(manifest.get("field_trace"), label="V20q outer field trace")
    _validate_hashed(trace, domain=_v20p._TRACE_DOMAIN, label="V20q outer field trace")
    _v20p._validate_field_trace_semantics(trace)
    held = _mapping(fold.get("held_evidence"), label="V20q held evidence")
    receipt = _mapping(fold.get("fold_receipt"), label="V20q fold receipt")
    _validate_hashed(held, domain=_EXECUTION_DOMAIN, label="V20q held evidence")
    _validate_hashed(receipt, domain=_DECISION_DOMAIN, label="V20q fold receipt")
    _validate_held_evidence_semantics(
        held,
        outer_family_id=outer_family_id,
        selected_candidate_id=str(final_refit["selected_candidate_id"]),
        manifest=manifest,
        authenticated_v20p_fold=authenticated_v20p_fold,
    )
    _validate_fold_receipt_semantics(
        receipt,
        outer_family_id=outer_family_id,
        selection_receipt=selection,
        final_refit_receipt=final_refit,
        chart_collection_receipt=_mapping(
            fold.get("chart_collection_receipt"),
            label="V20q chart collection",
        ),
        manifest=manifest,
        held_evidence=held,
        trace=trace,
    )
    if (
        manifest.get("final_refit_receipt") != final_refit
        or held.get("candidate_provider_manifest") != manifest
        or held.get("selected_candidate_id") != final_refit.get("selected_candidate_id")
        or receipt.get("selected_candidate_id")
        != final_refit.get("selected_candidate_id")
        or float(receipt["b"]).hex() != float(final_refit["b"]).hex()
        or float(receipt["a"]).hex() != float(final_refit["a"]).hex()
        or receipt.get("feature_id") != final_refit.get("feature_id")
        or receipt.get("candidate_pointwise_trust_passed") is not True
        or receipt.get("provider_frozen_before_outer_score") is not True
        or receipt.get("outer_family_used_for_fit_or_selection") is not False
        or receipt.get("exact_execution") is not True
    ):
        raise ValueError("V20q final provider or fold summary binding differs")
    inherited_scores = _mapping(
        _mapping(
            authenticated_v20p_fold.get("fold_receipt"),
            label="V20q authenticated V20p fold receipt",
        ).get("held_objective_by_arm"),
        label="V20q authenticated V20p held scores",
    )
    if _v14._canonical_json_bytes(
        held.get("inherited_v20p_held_objective_by_arm")
    ) != _v14._canonical_json_bytes(inherited_scores):
        raise ValueError("V20q inherited V20p control scores differ")
    candidate = _mapping(fold.get("candidate"), label="V20q candidate")
    spec = _CANDIDATE_SPEC_BY_ID[str(final_refit["selected_candidate_id"])]
    expected_candidate = {
        "candidate_id": final_refit["selected_candidate_id"],
        "candidate_role": spec["role"],
        "feature_id": final_refit["feature_id"],
        "b": final_refit["b"],
        "a": final_refit["a"],
        "analysis_only": True,
    }
    if candidate != expected_candidate:
        raise ValueError("V20q candidate summary differs")


def _publish_fold_fragment(
    payload: Mapping[str, object], *, output: Path | str, outer_family_id: str
) -> None:
    _v20b._publish_scalar_fragment(
        payload,
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20q token-VJP outer fold",
    )


def _load_fold_fragment(
    *,
    output: Path | str,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
    all_family_ids: Sequence[str],
    authenticated_v20p_fold: Mapping[str, object],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20q token-VJP outer fold",
    )
    _validate_fold_fragment(
        value,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding_sha256,
        outer_family_id=outer_family_id,
        all_family_ids=all_family_ids,
        authenticated_v20p_fold=authenticated_v20p_fold,
    )
    return dict(value)


def _aggregate_decision(
    fold_fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen V20q eight-family development gates."""

    families = tuple(sorted(fold_fragments))
    if len(families) != _FAMILY_COUNT:
        raise ValueError("V20q decision requires exactly eight outer folds")
    receipts = {
        family: _mapping(
            fold_fragments[family].get("fold_receipt"),
            label=f"V20q {family} fold receipt",
        )
        for family in families
    }
    inherited = {
        family: {
            str(arm): float(score)
            for arm, score in _mapping(
                receipts[family].get("inherited_v20p_held_objective_by_arm"),
                label=f"V20q {family} inherited scores",
            ).items()
        }
        for family in families
    }
    if any(set(row) != set(_v20p._ARMS) for row in inherited.values()):
        raise ValueError("V20q inherited V20p arm geometry differs")
    scores = {
        family: {
            "token_vjp_candidate": float(receipts[family]["candidate_objective"]),
            **inherited[family],
        }
        for family in families
    }
    systems = ("token_vjp_candidate", *_v20p._ARMS)
    macro = {
        system: math.fsum(scores[family][system] for family in families)
        / len(families)
        for system in systems
    }
    references = tuple(_v20p._ARMS)
    wins = {
        reference: sum(
            scores[family]["token_vjp_candidate"]
            < scores[family][reference]
            for family in families
        )
        for reference in references
    }
    incumbent = "local_signed_field_reflected"
    primary = (
        macro["token_vjp_candidate"] < macro[incumbent]
        and wins[incumbent] >= 5
    )
    continuous = {
        family: receipts[family].get("selected_nonzero_continuous_candidate")
        is True
        for family in families
    }
    inner_better = {
        family: receipts[family].get(
            "selected_inner_oof_mean_beats_incumbent"
        )
        is True
        for family in families
    }
    continuous_fit = (
        sum(continuous.values()) >= 6
        and sum(
            continuous[family] and inner_better[family] for family in families
        )
        >= 6
    )
    output_differs = {
        family: receipts[family].get(
            "candidate_exact_output_differs_from_v20p_incumbent"
        )
        is True
        for family in families
    }
    difference_gate = sum(output_differs.values()) >= 6
    derivative = all(
        (
            not continuous[family]
            or (
                receipts[family].get("all_secant_stability_gates_passed")
                is True
                and receipts[family].get(
                    "deployed_direction_has_negative_predicted_derivative"
                )
                is True
            )
        )
        for family in families
    )
    pointwise = all(
        receipts[family].get("candidate_pointwise_trust_passed") is True
        for family in families
    )
    runtime_health = all(
        receipts[family].get("provider_frozen_before_outer_score") is True
        and receipts[family].get("outer_family_used_for_fit_or_selection")
        is False
        and receipts[family].get("exact_execution") is True
        for family in families
    )
    integrity = derivative and pointwise and runtime_health
    passed = primary and continuous_fit and difference_gate and integrity
    return _hashed(
        {
            "family_ids": families,
            "held_objective_by_family_and_system": scores,
            "macro_objective_by_system": macro,
            "candidate_strict_win_count_by_v20p_arm": wins,
            "strict_win_comparison": True,
            "primary_candidate_vs_v20p_incumbent_gate_passed": primary,
            "selected_nonzero_continuous_by_family": continuous,
            "selected_nonzero_continuous_count": sum(continuous.values()),
            "selected_inner_oof_mean_beats_incumbent_by_family": inner_better,
            "selected_continuous_and_inner_better_count": sum(
                continuous[family] and inner_better[family]
                for family in families
            ),
            "continuous_fit_gate_passed": continuous_fit,
            "candidate_exact_output_differs_by_family": output_differs,
            "candidate_exact_output_differs_count": sum(output_differs.values()),
            "exact_output_difference_gate_passed": difference_gate,
            "all_deployed_fit_directions_descending_and_secants_stable": derivative,
            "all_candidate_pointwise_trust_passed": pointwise,
            "runtime_health_gate_passed": runtime_health,
            "inherited_v20p_integrity_passed": True,
            "integrity_passed": integrity,
            "development_oof_passed": passed,
        },
        domain=_DECISION_DOMAIN,
    )


def _runner_work_accounting(
    fold_fragments: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    measured_backward_calls: int | None = None
    if fold_fragments is not None:
        measured_backward_calls = 0
        for fold in fold_fragments.values():
            collection = _mapping(
                fold.get("chart_collection_receipt"),
                label="V20q work chart collection",
            )
            for chart_value in _mapping(
                collection.get("chart_receipts_by_id"),
                label="V20q work charts",
            ).values():
                chart = _mapping(chart_value, label="V20q work chart")
                for row_value in _mapping(
                    chart.get("prompt_evidence_by_example"),
                    label="V20q work prompt evidence",
                ).values():
                    row = _mapping(row_value, label="V20q work prompt")
                    vjp = _mapping(
                        row.get("token_teacher_kl_vjp_receipt"),
                        label="V20q work VJP receipt",
                    )
                    measured_backward_calls += int(vjp["backward_call_count"])
    return {
        "accounting_scope": "canonical_one_shot_schedule",
        "canonical_model_forward_count": 20544,
        "total_model_forward_count": 20544,
        "canonical_teacher_access_count": 20512,
        "total_teacher_access_count": 20512,
        "live_authority_collection_model_forward_count": 32,
        "endpoint_reconstruction_model_forward_count": 112,
        "token_vjp_chart_model_forward_count": 896,
        "inner_exact_candidate_model_forward_count": 19488,
        "outer_held_candidate_model_forward_count": 16,
        "inner_logical_candidate_count": 9744,
        "logical_candidates_per_inner_fold": 174,
        "inner_folds_per_outer_fold": 7,
        "token_vjp_chart_count": 64,
        "token_vjp_prompt_record_count": 896,
        "measured_token_vjp_backward_call_count": measured_backward_calls,
        "runtime_provider_parameter_count_delta_vs_v20p": 0,
        "runtime_provider_logical_macs_delta_vs_v20p": 0,
        "compiler_only_fisher_vjp_work_excluded_from_inference": True,
        "all_eight_final_refit_model_forward_count": 0,
        "calibration_b_forward_or_tokenization_count": 0,
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
        "token_vjp_protocol_receipt": _TOKEN_VJP_PROTOCOL_RECEIPT,
        "fixed_protocol": _FIXED_PROTOCOL,
        "source_receipt": authorities.source,
        "panel_receipt": dict(panel_receipt),
        "bridge_binding_sha256": bridge_binding_sha256,
        "v20p_authority": {
            "report_sha256": _V20P_LOGICAL_SHA256,
            "file_sha256": _V20P_FILE_SHA256,
            "source_receipt_sha256": _V20P_SOURCE_SHA256,
            "fold_fragment_sha256s_by_family": dict(
                sorted(_V20P_FOLD_SHA256S.items())
            ),
            "classification": authorities.v20p_report.get("classification"),
            "development_oof_passed": False,
            "rollback_to_base": True,
            "integrity_passed": True,
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
        "work_accounting": _runner_work_accounting(fold_fragments),
        "all_eight_outer_folds_completed": len(fold_fragments) == _FAMILY_COUNT,
        "primary_development_gate_passed": decision[
            "primary_candidate_vs_v20p_incumbent_gate_passed"
        ],
        "continuous_fit_gate_passed": decision["continuous_fit_gate_passed"],
        "exact_output_difference_gate_passed": decision[
            "exact_output_difference_gate_passed"
        ],
        "integrity_gate_passed": decision["integrity_passed"],
        "development_oof_passed": passed,
        "passed": passed,
        "classification": (
            "soft_polarity_token_vjp_continuous_refit_nested_oof_passed"
            if passed
            else "soft_polarity_token_vjp_continuous_refit_failed_rollback_to_base"
        ),
        "rollback_to_base": not passed,
        "next_rung": (
            "freeze_success_then_seek_fresh_family_disjoint_shadow_validation"
            if passed
            else "stop_token_vjp_local_field_refit_class"
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
        label="V20q token-VJP nested report",
    )
    families = tuple(sorted(authorities.authenticated_v20p_folds))
    folds = {
        family: _load_fold_fragment(
            output=output,
            source=authorities.source,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding_sha256,
            outer_family_id=family,
            all_family_ids=families,
            authenticated_v20p_fold=authorities.authenticated_v20p_folds[family],
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
        raise ValueError("V20q report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_token_vjp_nested_development(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run, resume, or model-free replay the frozen V20q campaign."""

    destination = _validate_output(output)
    authorities = _load_prerequisites()
    parent = authorities.parent
    panel = dict(
        _mapping(
            parent.prerequisite.get("nested_panel_receipt"),
            label="V20q panel receipt",
        )
    )
    bridge = _sha(
        parent.prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20q bridge binding",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            authorities=authorities,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
        )
    families = tuple(sorted(authorities.authenticated_v20p_folds))
    if (
        len(families) != _FAMILY_COUNT
        or set(parent.authenticated_v20a_folds) != set(families)
        or set(
            _mapping(
                panel.get("family_prompt_sha256s"),
                label="V20q panel families",
            )
        )
        != set(families)
    ):
        raise RuntimeError("V20q authenticated family geometry differs")
    if all(_fold_path(destination, family).exists() for family in families):
        folds = {
            family: _load_fold_fragment(
                output=destination,
                source=authorities.source,
                panel_receipt=panel,
                bridge_binding_sha256=bridge,
                outer_family_id=family,
                all_family_ids=families,
                authenticated_v20p_fold=authorities.authenticated_v20p_folds[
                    family
                ],
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
                label="V20q token-VJP nested report",
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
            raise RuntimeError("V20q live bridge differs from authenticated authority")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=parent.prerequisite
        )
        if tuple(live_families) != families:
            raise RuntimeError("V20q live family order differs from A16 authority")
        fragments: dict[str, dict[str, object]] = {}
        for family in families:
            if _fold_path(destination, family).exists():
                fragments[family] = _load_fold_fragment(
                    output=destination,
                    source=authorities.source,
                    panel_receipt=panel,
                    bridge_binding_sha256=bridge,
                    outer_family_id=family,
                    all_family_ids=families,
                    authenticated_v20p_fold=authorities.authenticated_v20p_folds[
                        family
                    ],
                )
                continue
            live = _execute_outer_fold(
                context,
                records,
                teacher_vault,
                authorities=authorities,
                family_ids=families,
                outer_family_id=family,
                panel_receipt=panel,
            )
            payload = _fold_payload(
                live,
                output=destination,
                source=authorities.source,
                panel_receipt=panel,
                bridge_binding_sha256=bridge,
                outer_family_id=family,
                authenticated_v20p_fold=authorities.authenticated_v20p_folds[
                    family
                ],
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
                bridge_binding_sha256=bridge,
                outer_family_id=family,
                all_family_ids=families,
                authenticated_v20p_fold=authorities.authenticated_v20p_folds[
                    family
                ],
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
            label="V20q token-VJP nested report",
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
        description="Run the V20q nested continuous token-VJP development screen"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = (
        run_gemma3_l3_l4_complete_h4_soft_polarity_token_vjp_nested_development(
            output=args.output, cache_dir=args.cache_dir
        )
    )
    print(_v14._canonical_json_bytes(report).decode("ascii"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
