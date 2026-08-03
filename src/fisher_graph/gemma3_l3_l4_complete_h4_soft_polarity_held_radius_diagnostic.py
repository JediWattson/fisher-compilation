"""V20h frozen held-radius diagnostic for the V20g polarity direction.

V20h does not fit or select a new model.  It authenticates the complete failed
V20g campaign, reconstructs each seven-family endpoint, and re-materializes the
already fitted Fisher direction at a fixed radius ladder.  Every provider and
runtime trace is frozen before the two outer-held examples become accessible.
The resulting curves diagnose whether V20g failed because its direction did not
transfer or because its training-only radius selector overshot.

The held-family oracle is descriptive only.  This runner cannot authorize a
refit, Calibration-B, serving, compression, or speed claims.  Reports and
resumable fold fragments are write-once, mode-0600, scalar/hash-only JSON.
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
from .complete_h4_fisher_conditional_residual import _training_parent_modal
from .complete_h4_fisher_soft_polarity import (
    AutonomousCompleteH4FisherSoftPolarityProvider,
    build_autonomous_complete_h4_fisher_soft_polarity,
    fisher_soft_polarity_box_certificate,
)
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_soft_polarity_held_radius_diagnostic",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-soft-polarity-held-radius-"
    "r16-k256-a-fit16-dev-v20h.json"
)

_V20G_OUTPUT = _v20g.DEFAULT_OUTPUT
_V20G_LOGICAL_SHA256 = (
    "b11db35a813cbc83718201a585c933bac6d95f6b3c5e64c8d9c820df3d76df09"
)
_V20G_FILE_SHA256 = (
    "5898a828173a73802e626cdb0bb16f085d0d7e189ddf6810a138a944d31c1094"
)
_V20G_SOURCE_SHA256 = (
    "9317a2dffc4a0378bfd0febfaf4f10c5a5609e10d03f7a9aa9659d39991b4ab0"
)
_V20G_FOLD_SHA256S: dict[str, str] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": (
        "72e0613801e1ec55b09cdc08850722051a7708d8796b3f18a0a70c9b1b53e900"
    ),
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": (
        "7221f56fe09e6bdd623b8877d028adf094de03290433fbbdf2215ba232aebeb8"
    ),
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": (
        "3ece847dd6a24730409367bf96dcec0b6c04ba62d8b07fc273401066606edb74"
    ),
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": (
        "396ae55cba220a96513153bc3050e857c54b7218cf52dbba5506dea46d7b9ae3"
    ),
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": (
        "77c7da2f91380f146896ff6a956034c806854a5d53c660bcb69d7ca0ab21fb4e"
    ),
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": (
        "b2fdb87e2016cbedb1333ce6e3518b94a12d7b18bc8f9b5e8cac5da0f927b765"
    ),
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": (
        "97472c5ae333f9818a919ae469ff1574f517aa14dfc4a185a8d29b8b925970a9"
    ),
    "structured-strong-v9-calibration_a-varve-lamination-v9": (
        "4a26b3e618f5292997ba372abac5803419d71329e018cb44b090bd53bb72fbbe"
    ),
}
_V20G_SELECTED_RADII: dict[str, float] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": 1.0,
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": 2.0,
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": 2.0,
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": 2.0,
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": 2.0,
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": 1.0,
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": 2.0,
    "structured-strong-v9-calibration_a-varve-lamination-v9": 1.0,
}
_PRECOMMITTED_CVAR2_RADII: dict[str, float] = {
    "structured-strong-v9-calibration_a-alpine-fir-ring-density-v9": 0.5,
    "structured-strong-v9-calibration_a-cave-pearl-layering-v9": 1.0,
    "structured-strong-v9-calibration_a-kiln-brick-thermal-face-v9": 1.0,
    "structured-strong-v9-calibration_a-obsidian-hydration-rim-v9": 1.0,
    "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9": 2.0,
    "structured-strong-v9-calibration_a-shell-midden-stratigraphy-v9": 1.0,
    "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9": 1.0,
    "structured-strong-v9-calibration_a-varve-lamination-v9": 0.5,
}

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_soft_polarity_held_radius_diagnostic.v20h"
_FORMAT_VERSION = 24
_FOLD_SCHEMA = "fisher_graph.complete_h4_soft_polarity_held_radius_outer_fold.v20h"
_REPORT_DOMAIN = b"fisher-graph:soft-polarity-held-radius-diagnostic-report:v20h\0"
_SOURCE_DOMAIN = b"fisher-graph:soft-polarity-held-radius-diagnostic-source:v20h\0"
_FOLD_DOMAIN = b"fisher-graph:soft-polarity-held-radius-diagnostic-fold:v20h\0"
_HELD_EXECUTION_DOMAIN = (
    b"fisher-graph:soft-polarity-held-radius-diagnostic-held-execution:v20h\0"
)
_DIAGNOSTIC_DOMAIN = (
    b"fisher-graph:soft-polarity-held-radius-diagnostic-receipt:v20h\0"
)
_PROVIDER_MANIFEST_DOMAIN = (
    b"fisher-graph:soft-polarity-held-radius-diagnostic-provider-manifest:v20h\0"
)
_EXTENSION_PROVIDER_DOMAIN = (
    b"fisher-graph:soft-polarity-held-radius-diagnostic-extension-provider:v20h\0"
)
_SELECTION_DOMAIN = (
    b"fisher-graph:soft-polarity-held-radius-diagnostic-training-selector:v20h\0"
)

_FAMILY_COUNT = 8
_PROMPTS_PER_FAMILY = 2
_CONDITIONAL_RANK = 16
_INHERITED_RADII = tuple(float(value) for value in _v20g._core.SOFT_POLARITY_FIT_ALPHAS)
_RADII = (*_INHERITED_RADII, 4.0, 8.0)
_RADIUS_KEYS = tuple(str(value) for value in _RADII)

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "v20h_frozen_held_radius_direction_diagnostic",
    "scientific_status": "post_hoc_historically_reused_A16_development_diagnostic_only",
    "hypothesis_source": "exact_pinned_failed_V20g_normalized_Fisher_direction",
    "radius_order": _RADII,
    "inherited_radius_providers": "V20g_candidates_0_through_2_byte_identical",
    "extension_radius_providers": "same_direction_at_tau_4_and_8_frozen_in_V20h",
    "saturation_endpoint": "tau_8_has_tanh_tau_approximately_0.999999775",
    "held_barrier": "all_twelve_providers_and_traces_frozen_before_held_capability",
    "execution": "exactly_once_per_radius_per_held_example_no_early_stop",
    "anchors": "tau_0_matches_V20g_base_and_V20g_selected_tau_matches_V20g_soft",
    "training_only_selector_diagnostic": (
        "precommitted_CVaR2_mean_of_two_largest_family_deltas_tie_smaller_tau"
    ),
    "held_oracle": "descriptive_only_never_used_for_selection_or_authorization",
    "final_refit_authorized": False,
    "calibration_b_eligible": False,
    "serving_claim_authorized": False,
    "compression_claim_authorized": False,
    "speed_claim_authorized": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(_FIXED_PROTOCOL, domain=_SOURCE_DOMAIN)
_EXTENSION_TRANSFER_PROTOCOL_SHA256 = _v14._sha256(
    {
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "operation": "materialize_same_authenticated_direction_at_frozen_tau_4_or_8",
        "fit_or_selection": False,
        "held_rows_used": False,
    },
    domain=_EXTENSION_PROVIDER_DOMAIN,
)


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


def _radius_key(value: float) -> str:
    radius = float(value)
    if radius not in _RADII:
        raise ValueError("V20h radius is outside the frozen ladder")
    return str(radius)


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".json" or not _v20b._is_under_local_runs(destination):
        raise ValueError("V20h output must be JSON under .local-runs")
    protected = (
        _V20G_OUTPUT,
        _v20g._V20F_OUTPUT,
        _v20g._V20E_OUTPUT,
        _v20g._V20B_OUTPUT,
        _v20a.DEFAULT_OUTPUT,
    )
    if any(_v20b._same_destination(destination, item) for item in protected):
        raise ValueError("V20h must preserve immutable prerequisite reports")
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


@dataclass(slots=True)
class _FoldLive:
    endpoint: _v20g._EndpointLive
    provider_manifest: dict[str, object]
    held_evidence: dict[str, object]
    diagnostic_receipt: dict[str, object]


def _cvar2_selection(fit_receipt: Mapping[str, object]) -> dict[str, object]:
    """Select a radius using only the two worst training-family deltas."""

    candidates = tuple(
        _mapping(item, label="V20g CVaR-2 candidate")
        for item in _sequence(
            fit_receipt.get("candidate_receipts"), label="V20g CVaR-2 candidates"
        )
    )
    by_radius = {float(item["alpha"]): item for item in candidates}
    if set(by_radius) != set(_INHERITED_RADII):
        raise ValueError("V20h CVaR-2 candidate ladder differs from V20g")
    base = _mapping(
        by_radius[0.0].get("family_mean_train_objectives"),
        label="V20h CVaR-2 base family objectives",
    )
    rows: list[dict[str, object]] = []
    for radius in _INHERITED_RADII:
        family_scores = _mapping(
            by_radius[radius].get("family_mean_train_objectives"),
            label=f"V20h CVaR-2 radius {radius} family objectives",
        )
        if set(family_scores) != set(base):
            raise ValueError("V20h CVaR-2 training-family geometry differs")
        deltas = {
            family: float(family_scores[family]) - float(base[family])
            for family in sorted(base)
        }
        if not all(math.isfinite(value) for value in deltas.values()):
            raise ValueError("V20h CVaR-2 delta became nonfinite")
        worst = tuple(sorted(deltas.values(), reverse=True)[:2])
        rows.append(
            {
                "radius": radius,
                "cvar2_delta": float(sum(worst) / len(worst)),
                "worst_two_deltas": worst,
                "candidate_artifact_sha256": _sha(
                    by_radius[radius].get("artifact_sha256"),
                    label="V20h CVaR-2 candidate",
                ),
            }
        )
    selected = min(
        rows,
        key=lambda item: (
            float(item["cvar2_delta"]),
            float(item["radius"]),
            str(item["candidate_artifact_sha256"]),
        ),
    )
    return _hashed(
        {
            "outer_held_family_id": fit_receipt.get("outer_held_family_id"),
            "radius_order": _INHERITED_RADII,
            "score_by_radius": {
                str(row["radius"]): row["cvar2_delta"] for row in rows
            },
            "selected_radius": selected["radius"],
            "selected_candidate_artifact_sha256": selected[
                "candidate_artifact_sha256"
            ],
            "rule": "mean_two_largest_training_family_deltas_then_smaller_radius_then_candidate_hash",
            "held_rows_used": False,
        },
        domain=_SELECTION_DOMAIN,
    )


def _load_prerequisites() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, object],
]:
    """Authenticate V20g and every fold before any live model construction."""

    prerequisite, authenticated_v20a_folds, v20g_source = _v20g._load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"),
            label="V20h inherited panel receipt",
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20h inherited bridge binding",
    )
    if _v14._file_sha256(_V20G_OUTPUT) != _V20G_FILE_SHA256:
        raise RuntimeError("pinned V20g report file hash drifted")
    v20g = _v20g._load_existing_report(
        _V20G_OUTPUT,
        source=v20g_source,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_folds=authenticated_v20a_folds,
    )
    qualification = _mapping(
        v20g.get("oof_qualification"), label="pinned V20g OOF qualification"
    )
    fragment_hashes = {
        _identifier(family, label="pinned V20g fragment family"): _sha(
            value, label="pinned V20g fragment"
        )
        for family, value in _mapping(
            v20g.get("fold_fragment_sha256s_by_family"),
            label="pinned V20g fragment hashes",
        ).items()
    }
    selected_radii = {
        _identifier(family, label="pinned V20g selected family"): float(value)
        for family, value in _mapping(
            qualification.get("selected_alphas_by_family"),
            label="pinned V20g selected radii",
        ).items()
    }
    if (
        v20g.get("report_sha256") != _V20G_LOGICAL_SHA256
        or v20g.get("classification")
        != "soft_polarity_trust_region_oof_failed_rollback_to_base"
        or v20g.get("passed") is not False
        or v20g.get("rollback_to_base") is not True
        or v20g.get("all_eight_outer_folds_completed") is not True
        or v20g.get("all_eight_family_refit_completed") is not False
        or v20g.get("final_refit") is not None
        or v20g.get("calibration_b_opened") is not False
        or v20g.get("calibration_b_tokenized") is not False
        or v20g.get("calibration_b_scored") is not False
        or _mapping(v20g.get("source_receipt"), label="pinned V20g source").get(
            "artifact_sha256"
        )
        != _V20G_SOURCE_SHA256
        or fragment_hashes != _V20G_FOLD_SHA256S
        or selected_radii != _V20G_SELECTED_RADII
    ):
        raise RuntimeError("pinned V20g diagnostic authority differs")

    families = tuple(sorted(_V20G_FOLD_SHA256S))
    folds = {
        family: _v20g._load_fold_fragment(
            output=_V20G_OUTPUT,
            source=v20g_source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20a_fold=authenticated_v20a_folds[family],
        )
        for family in families
    }
    cvar_receipts = {
        family: _cvar2_selection(
            _mapping(folds[family].get("fit_receipt"), label="V20g fit receipt")
        )
        for family in families
    }
    observed_cvar = {
        family: float(cvar_receipts[family]["selected_radius"])
        for family in families
    }
    if observed_cvar != _PRECOMMITTED_CVAR2_RADII:
        raise RuntimeError("precommitted CVaR-2 schedule differs from pinned V20g")
    source = _hashed(
        {
            "v20g_report_sha256": _V20G_LOGICAL_SHA256,
            "v20g_file_sha256": _V20G_FILE_SHA256,
            "v20g_source_receipt_sha256": _V20G_SOURCE_SHA256,
            "v20g_fold_fragment_sha256s_by_family": dict(
                sorted(_V20G_FOLD_SHA256S.items())
            ),
            "v20g_selected_radii_by_family": dict(
                sorted(_V20G_SELECTED_RADII.items())
            ),
            "precommitted_cvar2_radii_by_family": dict(
                sorted(_PRECOMMITTED_CVAR2_RADII.items())
            ),
            "precommitted_cvar2_receipt_sha256s_by_family": {
                family: cvar_receipts[family]["artifact_sha256"]
                for family in families
            },
            "radius_order": _RADII,
            "authenticated_before_model_construction": True,
            "held_radius_scores_used_for_provider_selection": False,
            "historically_reused_A16_only": True,
            "calibration_b_manifest_read": False,
            "calibration_b_tokenized": False,
        },
        domain=_SOURCE_DOMAIN,
    )
    return prerequisite, authenticated_v20a_folds, dict(v20g), folds, source


def _box_corner_logits(eta: Sequence[float] | Tensor) -> tuple[float, ...]:
    values = tuple(float(item) for item in _v20g._eta_tensor(eta).tolist())
    return tuple(
        values[0] + values[1] * c1 + values[2] * c2 + values[3] * c1 * c2
        for c1, c2 in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    )


def _inherited_fit_seed_sha256(
    endpoint_receipt: Mapping[str, object],
    endpoint_evidence: Mapping[str, object],
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str,
) -> str:
    families = tuple(
        sorted(
            _identifier(item, label="V20h inherited fit-seed family")
            for item in all_family_ids
        )
    )
    outer = _identifier(outer_family_id, label="V20h inherited fit-seed outer")
    if len(families) != _FAMILY_COUNT or outer not in families:
        raise ValueError("V20h inherited fit-seed family geometry differs")
    return _v14._sha256(
        {
            "runner_protocol_sha256": _v20g._RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _v20g._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256,
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt.get("artifact_sha256"),
                label="V20h inherited endpoint receipt",
            ),
            "endpoint_evidence_sha256": _sha(
                endpoint_evidence.get("artifact_sha256"),
                label="V20h inherited endpoint evidence",
            ),
            "all_development_family_ids": families,
            "outer_held_family_id": outer,
            "eta": (0.0,) * 4,
            "held_rows_used": False,
        },
        domain=_v20g._FIT_EXECUTION_DOMAIN,
    )


def _extension_provider_seed_sha256(
    *,
    endpoint_receipt_sha256: str,
    direction_artifact_sha256: str,
    radius: float,
    eta: Sequence[float],
    outer_family_id: str,
) -> str:
    return _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "extension_transfer_protocol_sha256": (
                _EXTENSION_TRANSFER_PROTOCOL_SHA256
            ),
            "endpoint_receipt_sha256": _sha(
                endpoint_receipt_sha256, label="V20h extension endpoint"
            ),
            "direction_artifact_sha256": _sha(
                direction_artifact_sha256, label="V20h extension direction"
            ),
            "radius": float(radius),
            "eta": tuple(float(item) for item in eta),
            "outer_held_family_id": _identifier(
                outer_family_id, label="V20h extension outer family"
            ),
            "held_rows_used": False,
            "fit_or_selection_performed": False,
        },
        domain=_EXTENSION_PROVIDER_DOMAIN,
    )


def _provider_receipt(
    provider: AutonomousCompleteH4FisherSoftPolarityProvider,
    *,
    radius: float,
    direction: Sequence[float],
    inherited_from_v20g: bool,
) -> dict[str, object]:
    eta = tuple(float(item) for item in provider.eta.tolist())
    expected = tuple(float(radius) * float(item) for item in direction)
    if eta != expected:
        raise RuntimeError("V20h provider eta differs from the frozen Fisher ray")
    corners = _box_corner_logits(eta)
    bound = max(abs(value) for value in corners)
    tolerance = max(1e-15, abs(float(radius)) * 1e-14)
    if abs(bound - float(radius)) > tolerance:
        raise RuntimeError("V20h provider violates the exact box-logit radius")
    metadata = _mapping(provider.metadata(), label="V20h provider metadata")
    return _hashed(
        {
            "radius": float(radius),
            "eta": eta,
            "eta_sha256": _v14._tensor_sha256(provider.eta),
            "box_corner_logits": corners,
            "box_logit_max_abs": bound,
            "box_logit_bound": float(radius),
            "global_box_certificate": fisher_soft_polarity_box_certificate(provider.eta),
            "provider_artifact_sha256": _sha(
                provider.artifact_sha256, label="V20h provider artifact"
            ),
            "provider_metadata_sha256": _v14._sha256(
                metadata, domain=_PROVIDER_MANIFEST_DOMAIN
            ),
            "transfer_protocol_sha256": _sha(
                provider.transfer_protocol_sha256,
                label="V20h provider transfer protocol",
            ),
            "transfer_evidence_sha256": _sha(
                provider.transfer_evidence_sha256,
                label="V20h provider transfer evidence",
            ),
            "rank": int(provider.rank),
            "conditional_rank": int(provider.conditional_rank),
            "prepared_float_scalar_count": int(provider.prepared_float_scalar_count),
            "logical_macs_per_token_upper_bound": int(
                provider.logical_macs_per_token_upper_bound
            ),
            "inherited_provider_identity_from_v20g": inherited_from_v20g,
            "fit_or_selection_performed": False,
            "analysis_only": True,
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )


def _provider_trace(
    provider: AutonomousCompleteH4FisherSoftPolarityProvider,
    records: Sequence[object],
    *,
    radius: float,
) -> dict[str, object]:
    ordered = _v20b._ordered_records(records)
    sequences = tuple(record.sequence for record in ordered)
    runtime = _v19._held_runtime_diagnostics(provider, sequences)
    gain_hashes: dict[str, str] = {}
    values: list[Tensor] = []
    for sequence in sequences:
        parent = _training_parent_modal(provider.parent_provider, sequence)
        coordinates = provider.bounded_coordinates(parent)
        gain = provider.response_gain(coordinates)
        support = sequence.support_mask.to(gain.device)
        selected = gain[support].detach().to(device="cpu", dtype=torch.float64)
        if selected.numel() == 0 or not bool(torch.isfinite(selected).all()):
            raise RuntimeError("V20h provider gain trace is empty or nonfinite")
        gain_hashes[sequence.example_id] = _v14._tensor_sha256(selected)
        values.append(selected.reshape(-1))
    joined = torch.cat(values)
    distinct = int(torch.unique(joined).numel())
    nonconstant = bool(float(joined.max()) > float(joined.min()))
    if float(radius) == 0.0:
        if distinct != 1 or nonconstant:
            raise RuntimeError("V20h zero-radius response is not identically zero")
    elif distinct <= 1 or not nonconstant:
        raise RuntimeError("V20h positive-radius response is constant")
    payload = {
        "radius": float(radius),
        "provider_artifact_sha256": provider.artifact_sha256,
        "scored_family_ids": tuple(
            sorted({record.sequence.family_id for record in ordered})
        ),
        "response_gain_sha256s": dict(sorted(gain_hashes.items())),
        "response_gain_min": float(joined.min()),
        "response_gain_max": float(joined.max()),
        "response_gain_distinct_count": distinct,
        "response_gain_nonconstant": nonconstant,
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
    return _hashed(payload, domain=_PROVIDER_MANIFEST_DOMAIN)


def _execution_sha256(
    *,
    phase: str,
    outer_family_id: str,
    provider_artifact_sha256: str,
    example_id: str,
    family_id: str,
    objective: float,
    h4_sha256: str,
    logits_sha256: str,
    evidence_sha256: str,
) -> str:
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
        domain=_HELD_EXECUTION_DOMAIN,
    )


def _score_exact_provider(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: AutonomousCompleteH4FisherSoftPolarityProvider,
    radius: float,
    outer_family_id: str,
    evidence_sha256: str,
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
            phase=f"held_radius_tau_{float(radius).hex()}",
            outer_family_id=outer_family_id,
            provider_artifact_sha256=provider.artifact_sha256,
            example_id=example,
            family_id=record.sequence.family_id,
            objective=score,
            h4_sha256=h4_sha,
            logits_sha256=logits_sha,
            evidence_sha256=evidence_sha256,
        )
        del model_inputs, teacher, execution
    return objectives, h4_hashes, logits_hashes, execution_hashes


def _materialize_radius_ladder(
    endpoint: _v20g._EndpointLive,
    v20g_fold: Mapping[str, object],
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str,
) -> tuple[
    dict[float, AutonomousCompleteH4FisherSoftPolarityProvider],
    dict[float, str],
]:
    fit = _mapping(v20g_fold.get("fit_receipt"), label="V20h inherited fit")
    evidence = _mapping(
        v20g_fold.get("fit_training_evidence"), label="V20h inherited fit evidence"
    )
    direction_receipt = _mapping(
        fit.get("direction_receipt"), label="V20h inherited direction"
    )
    direction = _v20g._direction_vector(direction_receipt)
    outer = _identifier(outer_family_id, label="V20h provider outer family")
    fit_seed = _v20g._fit_seed_sha256(
        endpoint, all_family_ids=all_family_ids, outer_family_id=outer
    )
    if fit_seed != _inherited_fit_seed_sha256(
        endpoint.receipt,
        endpoint.evidence,
        all_family_ids=all_family_ids,
        outer_family_id=outer,
    ):
        raise RuntimeError("V20h inherited V20g fit-seed reconstruction differs")
    providers, _candidate_seeds = _v20g._build_radius_provider_ladder(
        endpoint,
        direction_receipt=direction_receipt,
        fit_seed_sha256=fit_seed,
        outer_family_id=outer,
        alpha_ladder=_INHERITED_RADII,
    )
    transfer_evidence = {
        radius: _sha(
            providers[radius].transfer_evidence_sha256,
            label=f"V20h inherited transfer evidence {radius}",
        )
        for radius in _INHERITED_RADII
    }
    if transfer_evidence[0.0] != fit_seed or any(
        transfer_evidence[radius] != _candidate_seeds[radius]
        for radius in _INHERITED_RADII[1:]
    ):
        raise RuntimeError("V20h inherited provider transfer evidence differs")
    for radius in _RADII[len(_INHERITED_RADII) :]:
        eta = tuple(radius * item for item in direction)
        seed = _extension_provider_seed_sha256(
            endpoint_receipt_sha256=str(endpoint.receipt["artifact_sha256"]),
            direction_artifact_sha256=str(direction_receipt["artifact_sha256"]),
            radius=radius,
            eta=eta,
            outer_family_id=outer,
        )
        providers[radius] = build_autonomous_complete_h4_fisher_soft_polarity(
            endpoint.base_provider,
            endpoint.proposal_provider,
            eta=_v20g._eta_tensor(eta),
            transfer_protocol_sha256=_EXTENSION_TRANSFER_PROTOCOL_SHA256,
            transfer_evidence_sha256=seed,
        )
        transfer_evidence[radius] = seed

    inherited_manifest = _mapping(
        _mapping(
            evidence.get("candidate_provider_manifest"),
            label="V20h inherited provider manifest",
        ).get("provider_artifact_sha256s"),
        label="V20h inherited provider hashes",
    )
    observed = {
        str(radius): providers[radius].artifact_sha256 for radius in _INHERITED_RADII
    }
    if observed != dict(inherited_manifest):
        raise RuntimeError("V20h rebuilt V20g provider identity differs")
    return providers, transfer_evidence


def _freeze_radius_providers(
    endpoint: _v20g._EndpointLive,
    v20g_fold: Mapping[str, object],
    held_records: Sequence[object],
    *,
    all_family_ids: Sequence[str],
    outer_family_id: str,
) -> tuple[
    dict[float, AutonomousCompleteH4FisherSoftPolarityProvider],
    dict[str, object],
    dict[float, dict[str, object]],
]:
    fit = _mapping(v20g_fold.get("fit_receipt"), label="V20h freeze fit")
    direction_receipt = _mapping(
        fit.get("direction_receipt"), label="V20h freeze direction"
    )
    direction = _v20g._direction_vector(direction_receipt)
    providers, transfer_evidence = _materialize_radius_ladder(
        endpoint,
        v20g_fold,
        all_family_ids=all_family_ids,
        outer_family_id=outer_family_id,
    )
    receipts = {
        _radius_key(radius): _provider_receipt(
            providers[radius],
            radius=radius,
            direction=direction,
            inherited_from_v20g=radius in _INHERITED_RADII,
        )
        for radius in _RADII
    }
    traces = {
        radius: _provider_trace(providers[radius], held_records, radius=radius)
        for radius in _RADII
    }
    cvar = _cvar2_selection(fit)
    manifest = _hashed(
        {
            "outer_held_family_id": outer_family_id,
            "endpoint_receipt_sha256": endpoint.receipt["artifact_sha256"],
            "v20g_fit_receipt_sha256": fit["artifact_sha256"],
            "v20g_direction_receipt_sha256": direction_receipt["artifact_sha256"],
            "radius_order": _RADII,
            "provider_artifact_sha256s": {
                _radius_key(radius): providers[radius].artifact_sha256
                for radius in _RADII
            },
            "provider_transfer_evidence_sha256s": {
                _radius_key(radius): transfer_evidence[radius]
                for radius in _RADII
            },
            "provider_receipts": receipts,
            "response_trace_sha256s": {
                _radius_key(radius): traces[radius]["artifact_sha256"]
                for radius in _RADII
            },
            "precommitted_cvar2_receipt": cvar,
            "v20g_selected_radius": float(fit["selected_alpha"]),
            "all_twelve_providers_frozen_before_held_capability": True,
            "all_twelve_traces_frozen_before_held_capability": True,
            "held_capability_count_at_freeze": 0,
            "held_objectives_or_teacher_rows_used": False,
            "held_oracle_used_for_selection": False,
            "raw_provider_tensors_serialized": False,
        },
        domain=_PROVIDER_MANIFEST_DOMAIN,
    )
    return providers, manifest, traces


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
) -> _FoldLive:
    outer = _identifier(outer_family_id, label="V20h outer family")
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
        label="V20h inherited endpoint receipt",
    )
    inherited_endpoint_evidence = _mapping(
        authenticated_v20g_fold.get("endpoint_evidence"),
        label="V20h inherited endpoint evidence",
    )
    if (
        _v14._canonical_json_bytes(endpoint.receipt)
        != _v14._canonical_json_bytes(inherited_endpoint)
        or _v14._canonical_json_bytes(endpoint.evidence)
        != _v14._canonical_json_bytes(inherited_endpoint_evidence)
    ):
        raise RuntimeError("V20h reconstructed endpoint differs from pinned V20g")

    held = _v20b._ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    if len(held) != _PROMPTS_PER_FAMILY:
        raise RuntimeError("V20h held fold prompt geometry differs")
    providers, manifest, traces = _freeze_radius_providers(
        endpoint,
        authenticated_v20g_fold,
        held,
        all_family_ids=family_ids,
        outer_family_id=outer,
    )
    if (
        manifest.get("all_twelve_providers_frozen_before_held_capability") is not True
        or manifest.get("all_twelve_traces_frozen_before_held_capability") is not True
        or manifest.get("held_capability_count_at_freeze") != 0
        or manifest.get("held_objectives_or_teacher_rows_used") is not False
    ):
        raise PermissionError("V20h provider freeze barrier is not satisfied")

    trace_bundle_sha = _v14._sha256(
        {
            _radius_key(radius): traces[radius]["artifact_sha256"]
            for radius in _RADII
        },
        domain=_HELD_EXECUTION_DOMAIN,
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in held), held_family_id=None
    )
    objective_by_radius: dict[str, float] = {}
    evidence_by_radius: dict[str, dict[str, object]] = {}
    for radius in _RADII:
        key = _radius_key(radius)
        seed = _v14._sha256(
            {
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "trace_bundle_sha256": trace_bundle_sha,
                "radius": radius,
                "provider_artifact_sha256": providers[radius].artifact_sha256,
                "outer_held_family_id": outer,
                "all_radii_frozen": True,
            },
            domain=_HELD_EXECUTION_DOMAIN,
        )
        objectives, h4_hashes, logits_hashes, execution_hashes = (
            _score_exact_provider(
                context,
                held,
                capability,
                provider=providers[radius],
                radius=radius,
                outer_family_id=outer,
                evidence_sha256=seed,
            )
        )
        macro, family_scores = _v19._family_equal_mean(objectives, held)
        if set(family_scores) != {outer}:
            raise RuntimeError("V20h held objective family geometry differs")
        objective_by_radius[key] = macro
        evidence_by_radius[key] = _hashed(
            {
                "outer_held_family_id": outer,
                "radius": radius,
                "provider_artifact_sha256": providers[radius].artifact_sha256,
                "provider_manifest_sha256": manifest["artifact_sha256"],
                "response_trace": traces[radius],
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
            domain=_HELD_EXECUTION_DOMAIN,
        )

    capability_receipt = capability.receipt()
    _v20b._validate_capability_receipt(
        capability_receipt,
        expected_example_ids=tuple(record.sequence.example_id for record in held),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_RADII),
        label="V20h held-radius capability",
    )

    inherited_held = _mapping(
        authenticated_v20g_fold.get("held_evidence"),
        label="V20h inherited held evidence",
    )
    inherited_arms = _mapping(
        inherited_held.get("arm_evidence"), label="V20h inherited held arms"
    )
    v20g_base = _mapping(inherited_arms.get("base"), label="V20h V20g base arm")
    v20g_soft = _mapping(
        inherited_arms.get("soft_router"), label="V20h V20g soft arm"
    )
    selected_radius = float(
        _mapping(
            authenticated_v20g_fold.get("fit_receipt"), label="V20h inherited fit"
        )["selected_alpha"]
    )
    zero_evidence = evidence_by_radius[_radius_key(0.0)]
    selected_evidence = evidence_by_radius[_radius_key(selected_radius)]

    def _output_anchor_matches(
        observed: Mapping[str, object], expected: Mapping[str, object]
    ) -> bool:
        return (
            float(observed["objective"]) == float(expected["objective"])
            and observed.get("objectives_by_example")
            == expected.get("objectives_by_example")
            and observed.get("post_cast_h4_sha256s")
            == expected.get("post_cast_h4_sha256s")
            and observed.get("supervised_full_vocab_logits_sha256s")
            == expected.get("supervised_full_vocab_logits_sha256s")
        )

    zero_anchor = _output_anchor_matches(zero_evidence, v20g_base)
    selected_anchor = _output_anchor_matches(selected_evidence, v20g_soft)
    if not zero_anchor or not selected_anchor:
        raise RuntimeError("V20h output anchor differs from pinned V20g")

    health_passed = all(
        trace.get("finite") is True
        and trace.get("pointwise_trust_passed") is True
        and trace.get("endpoint_conditional_ranks_are_16") is True
        for trace in traces.values()
    )
    if not health_passed:
        raise RuntimeError("V20h held-radius runtime health failed")
    oracle_radius = min(
        _RADII,
        key=lambda radius: (
            objective_by_radius[_radius_key(radius)],
            radius,
            providers[radius].artifact_sha256,
        ),
    )
    base_objective = objective_by_radius[_radius_key(0.0)]
    improving = tuple(
        radius
        for radius in _RADII[1:]
        if objective_by_radius[_radius_key(radius)] < base_objective
    )
    smaller_improving = tuple(radius for radius in improving if radius < selected_radius)
    selected_objective = objective_by_radius[_radius_key(selected_radius)]
    if not improving:
        diagnosis = "positive_direction_failed_to_transfer"
    elif selected_objective >= base_objective and smaller_improving:
        diagnosis = "training_selected_radius_overshot_transfer_window"
    elif oracle_radius > selected_radius and objective_by_radius[
        _radius_key(oracle_radius)
    ] < selected_objective:
        diagnosis = "training_selected_radius_undershot_transfer_window"
    else:
        diagnosis = "direction_transfers_at_current_or_nearby_radius"

    cvar = _mapping(
        manifest.get("precommitted_cvar2_receipt"),
        label="V20h precommitted CVaR-2 receipt",
    )
    cvar_radius = float(cvar["selected_radius"])
    held_evidence = _hashed(
        {
            "outer_held_family_id": outer,
            "provider_manifest_sha256": manifest["artifact_sha256"],
            "trace_bundle_sha256": trace_bundle_sha,
            "radius_evidence": evidence_by_radius,
            "capability_receipt": capability_receipt,
            "all_twelve_providers_and_traces_frozen_before_held_capability": True,
            "held_family_used_for_fit_or_selection": False,
            "held_oracle_used_for_selection": False,
            "exact_held_execution_count": len(_RADII) * len(held),
            "raw_prompts_token_ids_logits_h4_or_teacher_rows_serialized": False,
        },
        domain=_HELD_EXECUTION_DOMAIN,
    )
    diagnostic = _hashed(
        {
            "outer_held_family_id": outer,
            "radius_order": _RADII,
            "held_objective_by_radius": objective_by_radius,
            "delta_from_base_by_radius": {
                key: objective_by_radius[key] - base_objective
                for key in _RADIUS_KEYS
            },
            "v20g_base_objective": base_objective,
            "v20g_fixed_plus_objective": float(
                _mapping(
                    inherited_arms.get("fixed_plus"),
                    label="V20h V20g fixed-plus arm",
                )["objective"]
            ),
            "v20g_selected_radius": selected_radius,
            "v20g_selected_objective": selected_objective,
            "precommitted_cvar2_radius": cvar_radius,
            "precommitted_cvar2_objective": objective_by_radius[
                _radius_key(cvar_radius)
            ],
            "diagnostic_oracle_radius": oracle_radius,
            "diagnostic_oracle_objective": objective_by_radius[
                _radius_key(oracle_radius)
            ],
            "diagnostic_oracle_used_for_selection": False,
            "positive_improving_radii": improving,
            "smaller_than_v20g_selected_improving_radii": smaller_improving,
            "direction_rescuable_by_some_positive_radius": bool(improving),
            "diagnosis": diagnosis,
            "tau_zero_exact_V20g_base_output_anchor": zero_anchor,
            "V20g_selected_tau_exact_soft_output_anchor": selected_anchor,
            "health_passed": health_passed,
            "provider_manifest_sha256": manifest["artifact_sha256"],
            "held_evidence_sha256": held_evidence["artifact_sha256"],
            "passed": False,
            "full_refit_authorized": False,
            "calibration_b_eligible": False,
            "rollback_to_base": True,
        },
        domain=_DIAGNOSTIC_DOMAIN,
    )
    return _FoldLive(
        endpoint=endpoint,
        provider_manifest=manifest,
        held_evidence=held_evidence,
        diagnostic_receipt=diagnostic,
    )


_FOLD_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "source_artifact_sha256",
        "v20g_report_sha256",
        "v20g_fold_fragment_sha256",
        "panel_receipt_sha256",
        "bridge_binding_sha256",
        "outer_held_family_id",
        "endpoint_receipt",
        "endpoint_evidence",
        "provider_manifest",
        "held_evidence",
        "diagnostic_receipt",
        "fixed_schedule_completed",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _fold_payload(
    live: _FoldLive,
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    outer_family_id: str,
    authenticated_v20g_fold: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": _FOLD_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": str(_validate_output(output).resolve(strict=False)),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "source_artifact_sha256": source["artifact_sha256"],
        "v20g_report_sha256": _V20G_LOGICAL_SHA256,
        "v20g_fold_fragment_sha256": authenticated_v20g_fold[
            "fragment_sha256"
        ],
        "panel_receipt_sha256": panel_receipt["artifact_sha256"],
        "bridge_binding_sha256": bridge_binding_sha256,
        "outer_held_family_id": outer_family_id,
        "endpoint_receipt": live.endpoint.receipt,
        "endpoint_evidence": live.endpoint.evidence,
        "provider_manifest": live.provider_manifest,
        "held_evidence": live.held_evidence,
        "diagnostic_receipt": live.diagnostic_receipt,
        "fixed_schedule_completed": True,
        "candidate": None,
        "provider_sidecar": None,
    }


def _validate_fold_fragment(
    value: Mapping[str, object],
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20g_fold: Mapping[str, object],
) -> dict[str, object]:
    selected = dict(value)
    if set(selected) != _FOLD_FRAGMENT_KEYS:
        raise ValueError("V20h fold fragment key set differs")
    outer = _identifier(outer_family_id, label="V20h fold outer family")
    if (
        selected.get("schema") != _FOLD_SCHEMA
        or selected.get("format_version") != _FORMAT_VERSION
        or selected.get("target_output")
        != str(_validate_output(output).resolve(strict=False))
        or selected.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or selected.get("source_artifact_sha256") != source.get("artifact_sha256")
        or selected.get("v20g_report_sha256") != _V20G_LOGICAL_SHA256
        or selected.get("v20g_fold_fragment_sha256")
        != authenticated_v20g_fold.get("fragment_sha256")
        or selected.get("panel_receipt_sha256")
        != panel_receipt.get("artifact_sha256")
        or selected.get("bridge_binding_sha256") != bridge_binding_sha256
        or selected.get("outer_held_family_id") != outer
        or selected.get("fixed_schedule_completed") is not True
        or selected.get("candidate") is not None
        or selected.get("provider_sidecar") is not None
    ):
        raise ValueError("V20h fold fragment authority differs")

    endpoint = _mapping(
        selected.get("endpoint_receipt"), label="V20h fold endpoint receipt"
    )
    endpoint_evidence = _mapping(
        selected.get("endpoint_evidence"), label="V20h fold endpoint evidence"
    )
    if (
        _v14._canonical_json_bytes(endpoint)
        != _v14._canonical_json_bytes(
            _mapping(
                authenticated_v20g_fold.get("endpoint_receipt"),
                label="V20h pinned endpoint receipt",
            )
        )
        or _v14._canonical_json_bytes(endpoint_evidence)
        != _v14._canonical_json_bytes(
            _mapping(
                authenticated_v20g_fold.get("endpoint_evidence"),
                label="V20h pinned endpoint evidence",
            )
        )
    ):
        raise ValueError("V20h fold endpoint lineage differs")

    manifest = _validate_hashed(
        _mapping(selected.get("provider_manifest"), label="V20h fold manifest"),
        domain=_PROVIDER_MANIFEST_DOMAIN,
        label="V20h provider manifest",
    )
    held = _validate_hashed(
        _mapping(selected.get("held_evidence"), label="V20h fold held evidence"),
        domain=_HELD_EXECUTION_DOMAIN,
        label="V20h held evidence",
    )
    diagnostic = _validate_hashed(
        _mapping(
            selected.get("diagnostic_receipt"), label="V20h fold diagnostic"
        ),
        domain=_DIAGNOSTIC_DOMAIN,
        label="V20h diagnostic receipt",
    )
    radii = tuple(
        float(item)
        for item in _sequence(manifest.get("radius_order"), label="V20h radii")
    )
    if radii != _RADII:
        raise ValueError("V20h frozen radius order differs")
    pinned_fit = _mapping(
        authenticated_v20g_fold.get("fit_receipt"), label="V20h pinned fit"
    )
    pinned_direction_receipt = _mapping(
        pinned_fit.get("direction_receipt"), label="V20h pinned direction"
    )
    pinned_direction = _v20g._direction_vector(pinned_direction_receipt)
    pinned_all_families = tuple(
        _identifier(item, label="V20h pinned direction family")
        for item in _sequence(
            pinned_direction_receipt.get("all_development_family_ids"),
            label="V20h pinned direction families",
        )
    )
    expected_fit_seed = _inherited_fit_seed_sha256(
        endpoint,
        endpoint_evidence,
        all_family_ids=pinned_all_families,
        outer_family_id=outer,
    )
    provider_hashes = _mapping(
        manifest.get("provider_artifact_sha256s"), label="V20h provider hashes"
    )
    transfer_evidence = _mapping(
        manifest.get("provider_transfer_evidence_sha256s"),
        label="V20h provider transfer evidence",
    )
    provider_receipts = _mapping(
        manifest.get("provider_receipts"), label="V20h provider receipts"
    )
    trace_hashes = _mapping(
        manifest.get("response_trace_sha256s"), label="V20h trace hashes"
    )
    if any(
        set(mapping) != set(_RADIUS_KEYS)
        for mapping in (
            provider_hashes,
            transfer_evidence,
            provider_receipts,
            trace_hashes,
        )
    ):
        raise ValueError("V20h provider ladder manifest is incomplete")
    for radius in _RADII:
        key = _radius_key(radius)
        _sha(provider_hashes[key], label=f"V20h provider {key}")
        _sha(
            transfer_evidence[key],
            label=f"V20h provider transfer evidence {key}",
        )
        receipt = _validate_hashed(
            _mapping(provider_receipts[key], label=f"V20h provider receipt {key}"),
            domain=_PROVIDER_MANIFEST_DOMAIN,
            label=f"V20h provider receipt {key}",
        )
        eta = tuple(radius * item for item in pinned_direction)
        if radius == 0.0:
            expected_transfer_evidence = expected_fit_seed
        elif radius in _INHERITED_RADII:
            expected_transfer_evidence = _v14._sha256(
                {
                    "fit_seed_sha256": expected_fit_seed,
                    "direction_artifact_sha256": pinned_direction_receipt[
                        "artifact_sha256"
                    ],
                    "alpha": radius,
                    "eta": eta,
                    "outer_held_family_id": outer,
                    "held_rows_used": False,
                },
                domain=_v20g._FIT_EXECUTION_DOMAIN,
            )
        else:
            expected_transfer_evidence = _extension_provider_seed_sha256(
                endpoint_receipt_sha256=str(endpoint["artifact_sha256"]),
                direction_artifact_sha256=str(
                    pinned_direction_receipt["artifact_sha256"]
                ),
                radius=radius,
                eta=eta,
                outer_family_id=outer,
            )
        if (
            float(receipt.get("radius", -1.0)) != radius
            or tuple(
                float(item)
                for item in _sequence(
                    receipt.get("eta"), label=f"V20h provider eta {key}"
                )
            )
            != tuple(radius * item for item in pinned_direction)
            or receipt.get("provider_artifact_sha256") != provider_hashes[key]
            or receipt.get("transfer_evidence_sha256") != transfer_evidence[key]
            or transfer_evidence[key] != expected_transfer_evidence
            or receipt.get("transfer_protocol_sha256")
            != (
                _v20g._core.SOFT_POLARITY_FIT_PROTOCOL_SHA256
                if radius in _INHERITED_RADII
                else _EXTENSION_TRANSFER_PROTOCOL_SHA256
            )
            or float(receipt.get("box_logit_bound", -1.0)) != radius
            or abs(float(receipt.get("box_logit_max_abs", -1.0)) - radius)
            > max(1e-15, radius * 1e-14)
        ):
            raise ValueError("V20h provider receipt radius binding differs")
    pinned_fit_evidence = _mapping(
        authenticated_v20g_fold.get("fit_training_evidence"),
        label="V20h pinned fit evidence",
    )
    pinned_provider_hashes = _mapping(
        _mapping(
            pinned_fit_evidence.get("candidate_provider_manifest"),
            label="V20h pinned provider manifest",
        ).get("provider_artifact_sha256s"),
        label="V20h pinned provider hashes",
    )
    observed_inherited_hashes = {
        _radius_key(radius): provider_hashes[_radius_key(radius)]
        for radius in _INHERITED_RADII
    }
    if observed_inherited_hashes != dict(pinned_provider_hashes):
        raise ValueError("V20h inherited provider identity differs from pinned V20g")
    cvar = _validate_hashed(
        _mapping(
            manifest.get("precommitted_cvar2_receipt"),
            label="V20h manifest CVaR-2 receipt",
        ),
        domain=_SELECTION_DOMAIN,
        label="V20h CVaR-2 receipt",
    )
    if (
        manifest.get("outer_held_family_id") != outer
        or manifest.get("endpoint_receipt_sha256") != endpoint.get("artifact_sha256")
        or manifest.get("v20g_fit_receipt_sha256")
        != pinned_fit.get("artifact_sha256")
        or manifest.get("v20g_direction_receipt_sha256")
        != pinned_direction_receipt.get("artifact_sha256")
        or manifest.get("all_twelve_providers_frozen_before_held_capability")
        is not True
        or manifest.get("all_twelve_traces_frozen_before_held_capability") is not True
        or manifest.get("held_capability_count_at_freeze") != 0
        or manifest.get("held_objectives_or_teacher_rows_used") is not False
        or manifest.get("held_oracle_used_for_selection") is not False
        or float(cvar.get("selected_radius"))
        != _PRECOMMITTED_CVAR2_RADII[outer]
        or cvar.get("artifact_sha256")
        != _mapping(
            source.get("precommitted_cvar2_receipt_sha256s_by_family"),
            label="V20h source CVaR-2 receipts",
        ).get(outer)
    ):
        raise ValueError("V20h provider freeze authority differs")

    radius_evidence = _mapping(
        held.get("radius_evidence"), label="V20h held radius evidence"
    )
    if set(radius_evidence) != set(_RADIUS_KEYS):
        raise ValueError("V20h held radius evidence is incomplete")
    objectives: dict[str, float] = {}
    for radius in _RADII:
        key = _radius_key(radius)
        evidence = _validate_hashed(
            _mapping(radius_evidence[key], label=f"V20h held radius {key}"),
            domain=_HELD_EXECUTION_DOMAIN,
            label=f"V20h held radius {key}",
        )
        trace = _validate_hashed(
            _mapping(evidence.get("response_trace"), label=f"V20h trace {key}"),
            domain=_PROVIDER_MANIFEST_DOMAIN,
            label=f"V20h trace {key}",
        )
        objective = float(evidence.get("objective", math.nan))
        if (
            not math.isfinite(objective)
            or float(evidence.get("radius", -1.0)) != radius
            or evidence.get("outer_held_family_id") != outer
            or evidence.get("provider_artifact_sha256") != provider_hashes[key]
            or evidence.get("provider_manifest_sha256")
            != manifest.get("artifact_sha256")
            or evidence.get("exact_execution") is not True
            or evidence.get("finite") is not True
            or trace.get("artifact_sha256") != trace_hashes[key]
            or trace.get("finite") is not True
            or trace.get("pointwise_trust_passed") is not True
            or trace.get("endpoint_conditional_ranks_are_16") is not True
        ):
            raise ValueError("V20h held radius evidence differs")
        objectives[key] = objective
    v20g_fold_receipt = _mapping(
        authenticated_v20g_fold.get("fold_receipt"),
        label="V20h pinned V20g fold receipt",
    )
    held_example_ids = tuple(
        _identifier(item, label="V20h held example")
        for item in _sequence(
            v20g_fold_receipt.get("held_example_ids"),
            label="V20h held example ids",
        )
    )
    _v20b._validate_capability_receipt(
        _mapping(
            held.get("capability_receipt"), label="V20h held capability receipt"
        ),
        expected_example_ids=held_example_ids,
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=len(_RADII),
        label="V20h held capability",
    )
    inherited_arms = _mapping(
        _mapping(
            authenticated_v20g_fold.get("held_evidence"),
            label="V20h pinned held evidence",
        ).get("arm_evidence"),
        label="V20h pinned held arms",
    )
    base_objective = float(
        _mapping(inherited_arms.get("base"), label="V20h pinned base")["objective"]
    )
    selected_radius = float(
        _mapping(
            authenticated_v20g_fold.get("fit_receipt"), label="V20h pinned fit"
        )["selected_alpha"]
    )
    selected_objective = float(
        _mapping(
            inherited_arms.get("soft_router"), label="V20h pinned soft router"
        )["objective"]
    )
    zero_radius_evidence = _mapping(
        radius_evidence[_radius_key(0.0)], label="V20h zero-radius evidence"
    )
    selected_radius_evidence = _mapping(
        radius_evidence[_radius_key(selected_radius)],
        label="V20h selected-radius evidence",
    )
    pinned_base_evidence = _mapping(
        inherited_arms.get("base"), label="V20h pinned base evidence"
    )
    pinned_soft_evidence = _mapping(
        inherited_arms.get("soft_router"), label="V20h pinned soft evidence"
    )
    pinned_fixed_plus_objective = float(
        _mapping(
            inherited_arms.get("fixed_plus"), label="V20h pinned fixed-plus evidence"
        )["objective"]
    )

    def _replay_anchor_matches(
        observed: Mapping[str, object], expected: Mapping[str, object]
    ) -> bool:
        return (
            float(observed.get("objective", math.nan))
            == float(expected.get("objective", math.nan))
            and observed.get("objectives_by_example")
            == expected.get("objectives_by_example")
            and observed.get("post_cast_h4_sha256s")
            == expected.get("post_cast_h4_sha256s")
            and observed.get("supervised_full_vocab_logits_sha256s")
            == expected.get("supervised_full_vocab_logits_sha256s")
        )

    zero_anchor = _replay_anchor_matches(zero_radius_evidence, pinned_base_evidence)
    selected_anchor = _replay_anchor_matches(
        selected_radius_evidence, pinned_soft_evidence
    )
    cvar_radius = float(cvar["selected_radius"])
    oracle_radius = min(
        _RADII,
        key=lambda radius: (objectives[_radius_key(radius)], radius),
    )
    improving = tuple(
        radius
        for radius in _RADII[1:]
        if objectives[_radius_key(radius)] < base_objective
    )
    smaller_improving = tuple(
        radius for radius in improving if radius < selected_radius
    )
    if not improving:
        expected_diagnosis = "positive_direction_failed_to_transfer"
    elif selected_objective >= base_objective and smaller_improving:
        expected_diagnosis = "training_selected_radius_overshot_transfer_window"
    elif oracle_radius > selected_radius and objectives[
        _radius_key(oracle_radius)
    ] < selected_objective:
        expected_diagnosis = "training_selected_radius_undershot_transfer_window"
    else:
        expected_diagnosis = "direction_transfers_at_current_or_nearby_radius"
    if (
        held.get("outer_held_family_id") != outer
        or held.get("provider_manifest_sha256") != manifest.get("artifact_sha256")
        or held.get(
            "all_twelve_providers_and_traces_frozen_before_held_capability"
        )
        is not True
        or held.get("held_family_used_for_fit_or_selection") is not False
        or held.get("held_oracle_used_for_selection") is not False
        or held.get("exact_held_execution_count")
        != len(_RADII) * _PROMPTS_PER_FAMILY
        or diagnostic.get("outer_held_family_id") != outer
        or tuple(float(item) for item in diagnostic.get("radius_order", ()))
        != _RADII
        or diagnostic.get("held_objective_by_radius") != objectives
        or float(diagnostic.get("v20g_base_objective")) != base_objective
        or float(diagnostic.get("v20g_selected_radius")) != selected_radius
        or float(diagnostic.get("v20g_selected_objective")) != selected_objective
        or float(diagnostic.get("v20g_fixed_plus_objective"))
        != pinned_fixed_plus_objective
        or float(diagnostic.get("precommitted_cvar2_radius")) != cvar_radius
        or float(diagnostic.get("precommitted_cvar2_objective"))
        != objectives[_radius_key(cvar_radius)]
        or float(diagnostic.get("diagnostic_oracle_radius")) != oracle_radius
        or float(diagnostic.get("diagnostic_oracle_objective"))
        != objectives[_radius_key(oracle_radius)]
        or tuple(diagnostic.get("positive_improving_radii", ())) != improving
        or tuple(
            diagnostic.get("smaller_than_v20g_selected_improving_radii", ())
        )
        != smaller_improving
        or diagnostic.get("direction_rescuable_by_some_positive_radius")
        is not bool(improving)
        or diagnostic.get("diagnosis") != expected_diagnosis
        or not zero_anchor
        or not selected_anchor
        or diagnostic.get("tau_zero_exact_V20g_base_output_anchor") is not True
        or diagnostic.get("V20g_selected_tau_exact_soft_output_anchor") is not True
        or diagnostic.get("health_passed") is not True
        or diagnostic.get("diagnostic_oracle_used_for_selection") is not False
        or diagnostic.get("passed") is not False
        or diagnostic.get("full_refit_authorized") is not False
        or diagnostic.get("calibration_b_eligible") is not False
        or diagnostic.get("rollback_to_base") is not True
    ):
        raise ValueError("V20h diagnostic receipt authority differs")
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
        label="V20h held-radius fold fragment",
    )


def _load_fold_fragment(
    *,
    output: Path,
    source: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    outer_family_id: str,
    bridge_binding_sha256: str,
    authenticated_v20g_fold: Mapping[str, object],
) -> dict[str, object]:
    selected = _v20b._load_scalar_fragment(
        path=_fold_path(output, outer_family_id),
        domain=_FOLD_DOMAIN,
        hash_key="fragment_sha256",
        label="V20h held-radius fold fragment",
    )
    return _validate_fold_fragment(
        selected,
        output=output,
        source=source,
        panel_receipt=panel_receipt,
        outer_family_id=outer_family_id,
        bridge_binding_sha256=bridge_binding_sha256,
        authenticated_v20g_fold=authenticated_v20g_fold,
    )


def _diagnostic_from_fragment(fragment: Mapping[str, object]) -> Mapping[str, object]:
    value = fragment.get("diagnostic_receipt")
    return _mapping(value, label="V20h aggregate fold diagnostic") if value is not None else fragment


def _aggregate_diagnostic(
    fold_fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate fixed held curves without granting selection authority."""

    if not fold_fragments:
        raise ValueError("V20h aggregate requires at least one fold")
    families = tuple(sorted(_identifier(item, label="V20h aggregate family") for item in fold_fragments))
    diagnostics = {
        family: _diagnostic_from_fragment(fold_fragments[family])
        for family in families
    }
    objectives_by_family: dict[str, dict[str, float]] = {}
    for family in families:
        diagnostic = diagnostics[family]
        if diagnostic.get("outer_held_family_id") != family:
            raise ValueError("V20h aggregate family lineage differs")
        order = tuple(
            float(item)
            for item in _sequence(
                diagnostic.get("radius_order"), label="V20h aggregate radius order"
            )
        )
        curve = {
            str(key): float(value)
            for key, value in _mapping(
                diagnostic.get("held_objective_by_radius"),
                label="V20h aggregate held curve",
            ).items()
        }
        if order != _RADII or set(curve) != set(_RADIUS_KEYS) or not all(
            math.isfinite(value) for value in curve.values()
        ):
            raise ValueError("V20h aggregate held curve differs")
        objectives_by_family[family] = {
            key: curve[key] for key in _RADIUS_KEYS
        }

    macro_by_radius = {
        key: float(
            sum(objectives_by_family[family][key] for family in families)
            / len(families)
        )
        for key in _RADIUS_KEYS
    }
    base_by_family = {
        family: objectives_by_family[family][_radius_key(0.0)]
        for family in families
    }
    fixed_plus_by_family = {
        family: float(diagnostics[family]["v20g_fixed_plus_objective"])
        for family in families
    }
    fixed_plus_macro = float(
        sum(fixed_plus_by_family.values()) / len(families)
    )
    base_macro = macro_by_radius[_radius_key(0.0)]
    delta_by_family = {
        family: {
            key: objectives_by_family[family][key] - base_by_family[family]
            for key in _RADIUS_KEYS
        }
        for family in families
    }
    slopes_by_family = {
        family: {
            _radius_key(right): (
                objectives_by_family[family][_radius_key(right)]
                - objectives_by_family[family][_radius_key(left)]
            )
            / (right - left)
            for left, right in zip(_RADII, _RADII[1:])
        }
        for family in families
    }

    oracle_radius_by_family: dict[str, float] = {}
    for family in families:
        oracle_radius_by_family[family] = min(
            _RADII,
            key=lambda radius: (
                objectives_by_family[family][_radius_key(radius)], radius
            ),
        )
    global_best = min(
        _RADII, key=lambda radius: (macro_by_radius[_radius_key(radius)], radius)
    )
    v20g_schedule = {
        family: float(diagnostics[family]["v20g_selected_radius"])
        for family in families
    }
    cvar_schedule = {
        family: float(diagnostics[family]["precommitted_cvar2_radius"])
        for family in families
    }
    if any(radius not in _RADII for radius in (*v20g_schedule.values(), *cvar_schedule.values())):
        raise ValueError("V20h aggregate schedule contains an unknown radius")

    def _schedule_scores(schedule: Mapping[str, float]) -> dict[str, float]:
        return {
            family: objectives_by_family[family][_radius_key(schedule[family])]
            for family in families
        }

    v20g_scores = _schedule_scores(v20g_schedule)
    cvar_scores = _schedule_scores(cvar_schedule)
    oracle_scores = {
        family: objectives_by_family[family][
            _radius_key(oracle_radius_by_family[family])
        ]
        for family in families
    }
    global_scores = {
        family: objectives_by_family[family][_radius_key(global_best)]
        for family in families
    }

    def _macro(scores: Mapping[str, float]) -> float:
        return float(sum(scores[family] for family in families) / len(families))

    def _wins(scores: Mapping[str, float], reference: Mapping[str, float]) -> int:
        return sum(scores[family] < reference[family] for family in families)

    v20g_regret = {
        family: v20g_scores[family] - oracle_scores[family] for family in families
    }
    cvar_regret = {
        family: cvar_scores[family] - oracle_scores[family] for family in families
    }
    direction_rescuable = tuple(
        family
        for family in families
        if any(
            objectives_by_family[family][_radius_key(radius)]
            < base_by_family[family]
            for radius in _RADII[1:]
        )
    )
    unrescuable = tuple(family for family in families if family not in direction_rescuable)
    overshoot = tuple(
        family
        for family in families
        if diagnostics[family].get("diagnosis")
        == "training_selected_radius_overshot_transfer_window"
    )
    health = all(diagnostics[family].get("health_passed") is True for family in families)
    anchors = all(
        diagnostics[family].get("tau_zero_exact_V20g_base_output_anchor") is True
        and diagnostics[family].get("V20g_selected_tau_exact_soft_output_anchor") is True
        for family in families
    )
    real_campaign = len(families) == _FAMILY_COUNT
    cvar_macro = _macro(cvar_scores)
    cvar_base_wins = _wins(cvar_scores, base_by_family)
    cvar_plus_wins = _wins(cvar_scores, fixed_plus_by_family)
    oracle_macro = _macro(oracle_scores)
    oracle_base_wins = _wins(oracle_scores, base_by_family)
    oracle_plus_wins = _wins(oracle_scores, fixed_plus_by_family)
    global_macro = _macro(global_scores)
    global_base_wins = _wins(global_scores, base_by_family)
    global_plus_wins = _wins(global_scores, fixed_plus_by_family)
    return _hashed(
        {
            "family_ids": families,
            "radius_order": _RADII,
            "family_objectives_by_radius": objectives_by_family,
            "macro_objective_by_radius": macro_by_radius,
            "delta_from_base_by_family_and_radius": delta_by_family,
            "finite_difference_slopes_by_family": slopes_by_family,
            "tau_4_to_8_objective_change_by_family": {
                family: objectives_by_family[family][_radius_key(8.0)]
                - objectives_by_family[family][_radius_key(4.0)]
                for family in families
            },
            "base_macro_objective": base_macro,
            "fixed_plus_macro_objective": fixed_plus_macro,
            "diagnostic_global_best_radius": global_best,
            "diagnostic_global_best_macro_objective": global_macro,
            "diagnostic_global_best_vs_base_win_count": global_base_wins,
            "diagnostic_global_best_vs_fixed_plus_win_count": global_plus_wins,
            "diagnostic_oracle_radius_by_family": oracle_radius_by_family,
            "diagnostic_oracle_macro_objective": oracle_macro,
            "diagnostic_oracle_vs_base_win_count": oracle_base_wins,
            "diagnostic_oracle_vs_fixed_plus_win_count": oracle_plus_wins,
            "diagnostic_oracle_boundary_count": sum(
                radius in (0.0, 8.0) for radius in oracle_radius_by_family.values()
            ),
            "diagnostic_oracle_used_for_selection": False,
            "v20g_selected_radius_by_family": v20g_schedule,
            "v20g_selected_schedule_macro_objective": _macro(v20g_scores),
            "v20g_selected_vs_base_win_count": _wins(v20g_scores, base_by_family),
            "v20g_selected_vs_fixed_plus_win_count": _wins(
                v20g_scores, fixed_plus_by_family
            ),
            "v20g_selected_regret_by_family": v20g_regret,
            "precommitted_cvar2_radius_by_family": cvar_schedule,
            "precommitted_cvar2_macro_objective": cvar_macro,
            "precommitted_cvar2_vs_base_win_count": cvar_base_wins,
            "precommitted_cvar2_vs_fixed_plus_win_count": cvar_plus_wins,
            "precommitted_cvar2_regret_by_family": cvar_regret,
            "precommitted_cvar2_mean_regret": float(
                sum(cvar_regret.values()) / len(families)
            ),
            "precommitted_cvar2_max_regret": max(cvar_regret.values()),
            "direction_rescuable_family_count": len(direction_rescuable),
            "direction_rescuable_family_ids": direction_rescuable,
            "direction_unrescuable_family_ids": unrescuable,
            "radius_overshoot_family_ids": overshoot,
            "single_global_tau_sufficient_for_gate": bool(
                real_campaign
                and health
                and global_macro < base_macro
                and global_macro < fixed_plus_macro
                and global_base_wins >= 6
                and global_plus_wins >= 6
            ),
            "diagnostic_oracle_capacity_sufficient": bool(
                real_campaign
                and health
                and oracle_macro < base_macro
                and oracle_macro < fixed_plus_macro
                and oracle_base_wins >= 6
                and oracle_plus_wins >= 6
            ),
            "precommitted_cvar2_development_gate_passed": bool(
                real_campaign
                and health
                and cvar_macro < base_macro
                and cvar_macro < fixed_plus_macro
                and cvar_base_wins >= 6
                and cvar_plus_wins >= 6
            ),
            "all_output_anchors_passed": anchors,
            "all_runtime_health_passed": health,
            "integrity_passed": bool(real_campaign and anchors and health),
            "held_oracle_or_global_radius_authorized": False,
            "fresh_family_disjoint_validation_required": True,
        },
        domain=_DIAGNOSTIC_DOMAIN,
    )


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    v20g_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    fold_fragments: Mapping[str, Mapping[str, object]],
    diagnostic: Mapping[str, object] | None = None,
) -> dict[str, object]:
    aggregate = (
        _aggregate_diagnostic(fold_fragments)
        if diagnostic is None
        else _validate_hashed(
            diagnostic,
            domain=_DIAGNOSTIC_DOMAIN,
            label="V20h aggregate diagnostic",
        )
    )
    families = tuple(
        _identifier(item, label="V20h report family")
        for item in _sequence(
            aggregate.get("family_ids"), label="V20h report families"
        )
    )
    if (
        len(families) != _FAMILY_COUNT
        or set(fold_fragments) != set(families)
        or aggregate.get("integrity_passed") is not True
    ):
        raise RuntimeError("V20h report requires eight authenticated healthy folds")
    cvar_gate = aggregate.get("precommitted_cvar2_development_gate_passed") is True
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": str(_validate_output(output).resolve(strict=False)),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "fixed_protocol": _FIXED_PROTOCOL,
        "source_receipt": dict(source),
        "v20g_authority": {
            "report_sha256": _V20G_LOGICAL_SHA256,
            "file_sha256": _V20G_FILE_SHA256,
            "source_receipt_sha256": _V20G_SOURCE_SHA256,
            "fold_fragment_sha256s_by_family": dict(
                sorted(_V20G_FOLD_SHA256S.items())
            ),
            "classification": v20g_report.get("classification"),
            "passed": v20g_report.get("passed"),
            "rollback_to_base": v20g_report.get("rollback_to_base"),
            "final_refit": v20g_report.get("final_refit"),
        },
        "panel_receipt": dict(panel_receipt),
        "bridge_binding_sha256": bridge_binding_sha256,
        "fold_fragment_sha256s_by_family": {
            family: fold_fragments[family]["fragment_sha256"]
            for family in families
        },
        "fold_diagnostic_receipts_by_family": {
            family: fold_fragments[family]["diagnostic_receipt"]
            for family in families
        },
        "diagnostic": dict(aggregate),
        "classification": "soft_polarity_held_radius_diagnostic_complete",
        "diagnostic_complete": True,
        "diagnostic_integrity_passed": True,
        "all_eight_outer_folds_completed": True,
        "precommitted_cvar2_development_gate_passed": cvar_gate,
        "next_rung": (
            "fresh_family_disjoint_shadow_of_precommitted_cvar2_selector"
            if cvar_gate
            else "revise_Fisher_direction_transfer_before_radius_selection"
        ),
        "passed": False,
        "rollback_to_base": True,
        "full_refit_authorized": False,
        "final_refit": None,
        "final_provider_frozen": False,
        "calibration_b_eligibility_gate_passed": False,
        "calibration_b_eligible": False,
        "calibration_b_authorized": False,
        "calibration_b_manifest_read": False,
        "calibration_b_opened": False,
        "calibration_b_tokenized": False,
        "calibration_b_scored": False,
        "fresh_family_disjoint_scoring_performed": False,
        "validation_opened": False,
        "test_opened": False,
        "held_oracle_used_for_selection": False,
        "serving_claim_authorized": False,
        "compression_claim_authorized": False,
        "speed_claim_authorized": False,
        "work_accounting": {
            "canonical_incremental_model_forward_count": 336,
            "live_authority_collection_model_forward_count": 32,
            "endpoint_reconstruction_model_forward_count": 112,
            "held_radius_model_forward_count": 192,
            "canonical_incremental_suffix_backward_count": 128,
            "canonical_incremental_local_autograd_contraction_count": 112,
            "canonical_incremental_teacher_access_count": 304,
            "candidate_provider_and_certificate_count": 96,
            "positive_radius_candidate_count": 88,
            "held_radius_score_count": 192,
            "held_response_trace_example_count": 192,
            "endpoint_health_trace_example_count": 224,
            "new_Fisher_solve_count": 0,
            "new_router_VJP_count": 0,
            "canonical_V20g_plus_V20h_model_forward_count": 1664,
            "canonical_V20g_plus_V20h_suffix_backward_count": 368,
            "canonical_V20g_plus_V20h_local_autograd_contraction_count": 336,
            "resume_and_authentication_overhead_excluded": True,
        },
        "artifact": None,
        "candidate": None,
        "provider_sidecar": None,
        "integrity": {
            "V20g_report_and_all_fragments_authenticated_before_model_construction": True,
            "V20g_normalized_direction_reused_without_refit": True,
            "ten_inherited_provider_hashes_matched_per_fold": True,
            "all_twelve_providers_and_traces_frozen_before_each_held_capability": True,
            "tau_zero_and_V20g_selected_tau_output_anchors_passed": True,
            "held_oracle_is_descriptive_only": True,
            "raw_prompts_tokens_logits_h4_gradients_or_provider_tensors_serialized": False,
        },
    }
    _v14._scalar_report(report)
    return report


def _load_existing_report(
    output: Path,
    *,
    source: Mapping[str, object],
    v20g_report: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
    authenticated_v20g_folds: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20h held-radius report",
    )
    families = tuple(sorted(authenticated_v20g_folds))
    folds = {
        family: _load_fold_fragment(
            output=output,
            source=source,
            panel_receipt=panel_receipt,
            outer_family_id=family,
            bridge_binding_sha256=bridge_binding_sha256,
            authenticated_v20g_fold=authenticated_v20g_folds[family],
        )
        for family in families
    }
    rebuilt = _build_report(
        output=output,
        source=source,
        v20g_report=v20g_report,
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
        raise ValueError("V20h report reconstruction differs")
    return dict(value)


def run_gemma3_l3_l4_complete_h4_soft_polarity_held_radius_diagnostic(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or resume the diagnostic without granting downstream authority."""

    destination = _validate_output(output)
    (
        prerequisite,
        authenticated_v20a_folds,
        v20g_report,
        authenticated_v20g_folds,
        source,
    ) = _load_prerequisites()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"), label="V20h panel receipt"
        )
    )
    bridge_binding = _sha(
        prerequisite.get("authenticated_bridge_binding_sha256"),
        label="V20h bridge binding",
    )
    if destination.exists():
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
        )
    family_ids = tuple(sorted(authenticated_v20g_folds))
    if (
        len(family_ids) != _FAMILY_COUNT
        or set(authenticated_v20a_folds) != set(family_ids)
        or set(_mapping(panel_receipt.get("family_prompt_sha256s"), label="V20h panel families"))
        != set(family_ids)
    ):
        raise RuntimeError("V20h authenticated family geometry differs")

    if all(_fold_path(destination, family).exists() for family in family_ids):
        folds = {
            family: _load_fold_fragment(
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                outer_family_id=family,
                bridge_binding_sha256=bridge_binding,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
            )
            for family in family_ids
        }
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            fold_fragments=folds,
        )
        try:
            _v20b._publish_scalar_fragment(
                report,
                path=destination,
                domain=_REPORT_DOMAIN,
                hash_key="report_sha256",
                label="V20h held-radius report",
            )
        except FileExistsError:
            pass
        return _load_existing_report(
            destination,
            source=source,
            v20g_report=v20g_report,
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge_binding,
            authenticated_v20g_folds=authenticated_v20g_folds,
        )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        if context.bridge.bridge_binding_sha256 != bridge_binding:
            raise RuntimeError("V20h live bridge differs from authenticated V20g")
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != family_ids:
            raise RuntimeError("V20h live family order differs from authenticated A16")
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
            )
            payload = _fold_payload(
                live,
                output=destination,
                source=source,
                panel_receipt=panel_receipt,
                bridge_binding_sha256=bridge_binding,
                outer_family_id=family,
                authenticated_v20g_fold=authenticated_v20g_folds[family],
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
            )
        report = _build_report(
            output=destination,
            source=source,
            v20g_report=v20g_report,
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
            label="V20h held-radius report",
        )
    except FileExistsError:
        pass
    return _load_existing_report(
        destination,
        source=source,
        v20g_report=v20g_report,
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge_binding,
        authenticated_v20g_folds=authenticated_v20g_folds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the V20h frozen held-radius diagnostic over the pinned V20g Fisher direction"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_soft_polarity_held_radius_diagnostic(
        output=arguments.output,
        cache_dir=arguments.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
