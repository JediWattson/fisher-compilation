"""V20c fixed-law continuous-response smoke on one frozen V20b pair.

This runner is deliberately smaller than a new validation ladder.  It
authenticates the completed (failed) V20b nested report, reconstructs exactly
the predeclared reed/sundial six-family endpoint pair, chooses one raw bounded
Fisher coordinate from those six families only, and freezes five arms before
creating either held-role teacher capability::

    base, constant +1, raw-linear, signed-log(kappa=9), -signed-log(kappa=9)

Both reciprocal roles are then scored by exact full-suffix execution.  The
artifact is development evidence on the reused A16 panel: it cannot authorize
serving, compression, or a fresh family-disjoint fidelity claim.  Only scalar
values and SHA-256 receipts are persisted; endpoint/provider tensors remain
transient and no provider sidecar is written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from . import complete_h4_fisher_continuous_response as _core
from .complete_h4_autonomous_residual import (
    _tensor_sha256 as _provider_tensor_sha256,
)
from .complete_h4_fisher_conditional_pedal import _training_parent_modal
from .complete_h4_fisher_continuous_transfer import (
    AutonomousCompleteH4FisherContinuousTransferProvider,
    build_autonomous_complete_h4_fisher_continuous_axis_response,
    build_autonomous_complete_h4_fisher_continuous_constant_control,
)
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
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
    "run_gemma3_l3_l4_complete_h4_continuous_response_smoke",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V20B_OUTPUT = _v20b.DEFAULT_OUTPUT
_V20B_LOGICAL_SHA256 = (
    "bb45e535074608c5feb877fbceb3342809d872f41ff1776851be656de1b0403b"
)
_V20B_FILE_SHA256 = (
    "42060cc4f4dffbb11ea1203518138a27b46bcd4f483623d4d7874da083a97214"
)
_V20B_CLASSIFICATION = "nested_inner_selection_failed"

_REED = "structured-strong-v9-calibration_a-reed-boat-fiber-strain-v9"
_SUNDIAL = "structured-strong-v9-calibration_a-sundial-gnomon-survey-v9"
_FROZEN_EXCLUDED = tuple(sorted((_REED, _SUNDIAL)))
_FROZEN_PAIR_KEY = (
    "01a308b42728255e2e5b246ef84bd0a16fb0341e313787c42792913df85252c6"
)
_FROZEN_PAIR_FRAGMENT_SHA256 = (
    "2e972a7efe4eb162a8fcd6e84015814bcc69ff81c5e71624a0004a0da5c9dc35"
)

DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-continuous-response-pair-smoke-"
    "r16-k256-a-fit16-dev-v20c.json"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4.complete_h4_continuous_response_smoke.v20c"
_FORMAT_VERSION = 20
_REPORT_DOMAIN = b"fisher-graph:continuous-response-smoke-report:v20c\0"
_SOURCE_DOMAIN = b"fisher-graph:continuous-response-smoke-source:v20c\0"
_COORDINATE_TRACE_DOMAIN = (
    b"fisher-graph:continuous-response-smoke-coordinate-trace:v20c\0"
)
_LAW_EVIDENCE_DOMAIN = b"fisher-graph:continuous-response-smoke-law:v20c\0"
_PROVIDER_RECEIPT_DOMAIN = (
    b"fisher-graph:continuous-response-smoke-provider:v20c\0"
)
_RESPONSE_TRACE_DOMAIN = (
    b"fisher-graph:continuous-response-smoke-response-trace:v20c\0"
)
_EXECUTION_DOMAIN = b"fisher-graph:continuous-response-smoke-execution:v20c\0"
_ROLE_EVIDENCE_DOMAIN = (
    b"fisher-graph:continuous-response-smoke-role-evidence:v20c\0"
)
_PAIR_QUALIFICATION_DOMAIN = (
    b"fisher-graph:continuous-response-smoke-qualification:v20c\0"
)

_ARMS = tuple(_core.CONTINUOUS_RESPONSE_ARMS)
_PROMPTS_PER_FAMILY = 2
_FIT_FAMILY_COUNT = 6
_EXPECTED_FAMILY_COUNT = 8
_CONDITIONAL_RANK = 16
_PROVIDER_RECEIPT_KEYS = {
    "arm",
    "provider_artifact_sha256",
    "provider_metadata_sha256",
    "base_provider_artifact_sha256",
    "proposal_provider_artifact_sha256",
    "response_weight_sha256",
    "response_source",
    "response_law",
    "polarity",
    "signed_log_kappa",
    "transfer_protocol_sha256",
    "transfer_evidence_sha256",
    "rank",
    "conditional_rank",
    "prepared_float_scalar_count",
    "logical_macs_per_token_upper_bound",
    "analysis_only",
    "artifact_sha256",
}
_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "frozen_v20b_pair_fixed_continuous_response_smoke",
    "scientific_status": "development_only_reused_a16",
    "source": "exact_failed_v20b_report_and_frozen_pair_fragment",
    "excluded_pair": _FROZEN_EXCLUDED,
    "coordinate_selection": (
        "fit_only_family_equal_variance_lowest_coordinate_index_tie"
    ),
    "coordinate_runtime_application": (
        "raw_bounded_coordinate_no_center_scale_clamp_or_prompt_label"
    ),
    "arms": _ARMS,
    "signed_log_kappa": _core.CONTINUOUS_RESPONSE_KAPPA,
    "held_barrier": "all_laws_and_providers_frozen_before_role_capabilities",
    "score_source": "exact_finite_full_suffix_execution",
    "pair_smoke_gate": (
        "signed_log_at_least_1pct_family_equal_macro_improvement_improves_both_"
        "roles_worst_regression_at_most_2pct_and_beats_constant_linear_and_"
        "mirror_in_macro_with_all_runtime_health_checks"
    ),
    "required_family_equal_macro_improvement": 0.01,
    "maximum_worst_role_regression": 0.02,
    "empirical_fisher_response_weight_fit": False,
    "raw_tensor_or_provider_sidecar": False,
    "serving_or_compression_claim": False,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(
    _FIXED_PROTOCOL,
    domain=_LAW_EVIDENCE_DOMAIN,
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
        raise ValueError("V20c output must be JSON under .local-runs")
    if _v20b._same_destination(output, _V20B_OUTPUT):
        raise ValueError("V20c must preserve the write-once V20b report")
    return output


def _source_sha256s() -> dict[str, str]:
    return {
        "v20b_report_logical_sha256": _V20B_LOGICAL_SHA256,
        "v20b_report_file_sha256": _V20B_FILE_SHA256,
        "v20b_pair_fragment_sha256": _FROZEN_PAIR_FRAGMENT_SHA256,
        "v20b_pair_key": _FROZEN_PAIR_KEY,
    }


def _validate_frozen_pair_fragment(
    fragment: Mapping[str, object],
    *,
    family_ids: Sequence[str],
) -> dict[str, object]:
    """Fail closed unless the source is exactly the declared reciprocal pair."""

    families = tuple(sorted(_identifier(item, label="A16 family") for item in family_ids))
    if len(families) != _EXPECTED_FAMILY_COUNT or len(set(families)) != len(families):
        raise ValueError("V20c source family geometry differs")
    fit = _mapping(fragment.get("shared_fit_receipt"), label="V20b shared fit")
    excluded = tuple(fragment.get("excluded_family_ids", ()))
    training = tuple(fit.get("training_family_ids", ()))
    expected_training = tuple(item for item in families if item not in _FROZEN_EXCLUDED)
    panels = tuple(
        _mapping(item, label="V20b directed panel")
        for item in _sequence(fragment.get("directed_panels"), label="V20b directed panels")
    )
    roles = {
        (
            panel.get("outer_held_family_id"),
            panel.get("inner_held_family_id"),
        )
        for panel in panels
    }
    if (
        fragment.get("fragment_sha256") != _FROZEN_PAIR_FRAGMENT_SHA256
        or fragment.get("pair_key") != _FROZEN_PAIR_KEY
        or excluded != _FROZEN_EXCLUDED
        or fit.get("fit_key") != _FROZEN_PAIR_KEY
        or tuple(fit.get("excluded_family_ids", ())) != _FROZEN_EXCLUDED
        or tuple(sorted(training)) != expected_training
        or len(training) != _FIT_FAMILY_COUNT
        or set(training) & set(_FROZEN_EXCLUDED)
        or roles
        != {
            (_REED, _SUNDIAL),
            (_SUNDIAL, _REED),
        }
    ):
        raise RuntimeError("V20c frozen V20b pair authority differs")
    return dict(fragment)


def _joint_one_source_diagnostic(
    fragment: Mapping[str, object],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for raw in _sequence(fragment.get("directed_panels"), label="source panels"):
        panel = _mapping(raw, label="source panel")
        baseline = _mapping(panel.get("baseline"), label="source baseline")
        selected = tuple(
            _mapping(value, label="source candidate")
            for value in _sequence(
                panel.get("positive_candidates"), label="source candidates"
            )
            if isinstance(value, Mapping)
            and value.get("path") == "joint"
            and value.get("alpha") == 1.0
        )
        if len(selected) != 1:
            raise RuntimeError("V20c source joint +1 candidate differs")
        candidate = selected[0]
        base_objective = float(baseline["objective"])
        candidate_objective = float(candidate["objective"])
        rows.append(
            {
                "outer_held_family_id": panel["outer_held_family_id"],
                "scored_inner_family_id": panel["inner_held_family_id"],
                "base_objective": base_objective,
                "joint_plus_one_objective": candidate_objective,
                "joint_plus_one_relative_improvement": (
                    base_objective - candidate_objective
                )
                / base_objective,
            }
        )
    rows.sort(key=lambda value: str(value["outer_held_family_id"]))
    if len(rows) != 2 or sum(row["joint_plus_one_relative_improvement"] > 0.0 for row in rows) != 1:
        raise RuntimeError("V20c source pair no longer has divergent +1 outcomes")
    return {
        "source": "authenticated_v20b_exact_finite_scores",
        "reciprocal_roles": tuple(rows),
        "joint_plus_one_role_outcomes_diverged": True,
    }


def _load_authenticated_v20b_source(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Authenticate exact V20b report and pair before any model creation."""

    _v20b._secure_stat(_V20B_OUTPUT, label="pinned V20b report")
    observed_file = _v14._file_sha256(_V20B_OUTPUT)
    if observed_file != _V20B_FILE_SHA256:
        raise RuntimeError("pinned V20b report file hash drifted")
    try:
        payload = json.loads(_V20B_OUTPUT.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("pinned V20b report is unreadable") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("pinned V20b report is not a mapping")
    logical = dict(payload)
    logical.pop("report_sha256", None)
    if (
        payload.get("schema") != _v20b._SCHEMA
        or payload.get("format_version") != _v20b._FORMAT_VERSION
        or payload.get("report_sha256") != _V20B_LOGICAL_SHA256
        or _v14._sha256(logical, domain=_v20b._REPORT_DOMAIN)
        != _V20B_LOGICAL_SHA256
        or payload.get("classification") != _V20B_CLASSIFICATION
        or payload.get("passed") is not False
        or payload.get("held_fidelity_claim") is not False
        or payload.get("serving_authorized") is not False
        or payload.get("compression_claim") is not False
        or payload.get("candidate") is not None
        or payload.get("provider_sidecar") is not None
    ):
        raise RuntimeError("pinned V20b report authority differs")
    panel = _mapping(payload.get("panel_receipt"), label="V20b panel receipt")
    family_rows = _mapping(
        panel.get("family_prompt_sha256s"), label="V20b panel families"
    )
    families = tuple(sorted(str(item) for item in family_rows))
    fragment = _v20b._load_pair_fragment(
        output=_V20B_OUTPUT,
        panel_binding_sha256=_sha(
            payload.get("panel_binding_sha256"), label="V20b panel binding"
        ),
        excluded_family_ids=_FROZEN_EXCLUDED,
        panel_receipt=panel,
    )
    frozen = _validate_frozen_pair_fragment(fragment, family_ids=families)
    pair_rows = _mapping(
        payload.get("pair_fragment_sha256s"), label="V20b pair fragment hashes"
    )
    if pair_rows.get(_FROZEN_PAIR_KEY) != _FROZEN_PAIR_FRAGMENT_SHA256:
        raise RuntimeError("pinned V20b report does not bind the frozen pair")
    source_payload = {
        "path": _V20B_OUTPUT.as_posix(),
        "report_logical_sha256": _V20B_LOGICAL_SHA256,
        "report_file_sha256": _V20B_FILE_SHA256,
        "classification": _V20B_CLASSIFICATION,
        "passed": False,
        "pair_key": _FROZEN_PAIR_KEY,
        "pair_fragment_sha256": _FROZEN_PAIR_FRAGMENT_SHA256,
        "excluded_family_ids": _FROZEN_EXCLUDED,
        "base_provider_artifact_sha256": frozen["shared_fit_receipt"][
            "base_provider_artifact_sha256"
        ],
        "proposal_provider_artifact_sha256": frozen["shared_fit_receipt"][
            "proposal_provider_artifact_sha256"
        ],
        "authenticated_before_model_work": True,
    }
    source = {
        **source_payload,
        "artifact_sha256": _v14._sha256(source_payload, domain=_SOURCE_DOMAIN),
    }
    _v14._scalar_report(source)
    return source, dict(payload), frozen


def _fit_coordinates_by_family(
    training_records: Sequence[object],
    *,
    base_provider: object,
    family_ids: Sequence[str],
    excluded_family_ids: Sequence[str],
) -> tuple[dict[str, tuple[tuple[float, float], ...]], dict[str, object]]:
    """Return transient raw fit coordinates plus a scalar/hash-only receipt."""

    families = tuple(sorted(_identifier(item, label="fit family") for item in family_ids))
    excluded = tuple(sorted(_identifier(item, label="excluded family") for item in excluded_family_ids))
    expected = tuple(item for item in families if item not in excluded)
    if len(families) != _EXPECTED_FAMILY_COUNT or len(excluded) != 2 or expected != tuple(
        sorted(getattr(base_provider, "fit_family_ids"))
    ):
        raise ValueError("V20c coordinate family scope differs")
    grouped: dict[str, list[tuple[float, float]]] = {family: [] for family in expected}
    per_sequence: list[dict[str, object]] = []
    for record in _v20b._ordered_records(training_records):
        sequence = getattr(record, "sequence")
        family = _identifier(sequence.family_id, label="coordinate record family")
        if family not in grouped or family in excluded:
            raise PermissionError("held family reached V20c coordinate statistics")
        parent = _training_parent_modal(base_provider.parent_provider, sequence)
        coordinates = base_provider.bounded_coordinates(parent)
        support = sequence.support_mask.to(coordinates.device)
        selected = coordinates[support].detach().to(device="cpu", dtype=torch.float64)
        if (
            selected.ndim != 2
            or selected.shape[0] == 0
            or selected.shape[1] != 2
            or not bool(torch.isfinite(selected).all())
            or bool((selected.abs() >= 1.0).any())
        ):
            raise RuntimeError("V20c fit coordinate trace differs")
        grouped[family].extend(
            (float(row[0]), float(row[1])) for row in selected.tolist()
        )
        per_sequence.append(
            {
                "example_id": sequence.example_id,
                "family_id": family,
                "sequence_artifact_sha256": sequence.artifact_sha256,
                "support_row_count": int(selected.shape[0]),
                "bounded_coordinate_sha256": _v14._tensor_sha256(selected),
            }
        )
    if (
        set(grouped) != set(expected)
        or any(not rows for rows in grouped.values())
        or len(per_sequence) != _FIT_FAMILY_COUNT * _PROMPTS_PER_FAMILY
        or {row["family_id"] for row in per_sequence} & set(excluded)
    ):
        raise RuntimeError("V20c coordinate trace did not use exact six-family fit complement")
    frozen = {family: tuple(grouped[family]) for family in expected}
    core_statistics = _core.fit_coordinate_statistics(
        tuple(frozen[family] for family in expected)
    )
    trace_payload = {
        "scope": "six_fit_families_only_before_held_capabilities",
        "coordinate_source_family_ids": expected,
        "excluded_family_ids": excluded,
        "sequence_rows": tuple(per_sequence),
        "coordinate_family_row_counts": {
            family: len(frozen[family]) for family in expected
        },
        "fit_coordinate_rows_sha256": core_statistics[
            "fit_coordinate_rows_sha256"
        ],
        "raw_coordinates_serialized": False,
        "family_ids_used_only_to_define_equal_weight_fit_groups": True,
        "family_id_values_enter_numeric_response": False,
        "objectives_used_to_select_axis": False,
    }
    trace = {
        **trace_payload,
        "artifact_sha256": _v14._sha256(
            trace_payload, domain=_COORDINATE_TRACE_DOMAIN
        ),
    }
    _v14._scalar_report(trace)
    return frozen, trace


def _validate_coordinate_trace_law_binding(
    coordinate_trace: Mapping[str, object],
    *,
    roles: Sequence[Mapping[str, object]],
    fit_receipt: Mapping[str, object],
) -> None:
    trace_payload = {
        key: value
        for key, value in coordinate_trace.items()
        if key != "artifact_sha256"
    }
    if _sha(
        coordinate_trace.get("artifact_sha256"), label="coordinate trace receipt"
    ) != _v14._sha256(trace_payload, domain=_COORDINATE_TRACE_DOMAIN):
        raise ValueError("V20c coordinate trace receipt hash differs")
    trace_hash = _sha(
        coordinate_trace.get("fit_coordinate_rows_sha256"),
        label="coordinate trace grouped rows",
    )
    fit_families = tuple(fit_receipt.get("training_family_ids", ()))
    if (
        tuple(coordinate_trace.get("coordinate_source_family_ids", ()))
        != fit_families
        or tuple(coordinate_trace.get("excluded_family_ids", ()))
        != _FROZEN_EXCLUDED
        or coordinate_trace.get("raw_coordinates_serialized") is not False
    ):
        raise ValueError("V20c coordinate trace fit scope differs")
    for role in roles:
        law = _mapping(role.get("law_receipt"), label="coordinate-bound law")
        statistics = _mapping(
            law.get("coordinate_statistics"), label="coordinate-bound statistics"
        )
        if (
            statistics.get("fit_coordinate_rows_sha256") != trace_hash
            or tuple(law.get("coordinate_source_family_ids", ())) != fit_families
            or tuple(law.get("excluded_family_ids", ())) != _FROZEN_EXCLUDED
        ):
            raise ValueError("V20c law coordinate trace binding differs")


def _provider_receipt(provider: object, *, arm: str) -> dict[str, object]:
    metadata = dict(getattr(provider, "metadata")())
    _v14._scalar_report(metadata)
    continuous = isinstance(
        provider, AutonomousCompleteH4FisherContinuousTransferProvider
    )
    payload = {
        "arm": arm,
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
        "response_weight_sha256": (
            metadata["response_weight_sha256"] if continuous else None
        ),
        "response_source": metadata["response_source"] if continuous else "base_zero",
        "response_law": metadata["response_law"] if continuous else "base_zero",
        "polarity": int(metadata["polarity"]) if continuous else 0,
        "signed_log_kappa": (
            float(metadata["signed_log_kappa"]) if continuous else None
        ),
        "transfer_protocol_sha256": (
            metadata["transfer_protocol_sha256"] if continuous else None
        ),
        "transfer_evidence_sha256": (
            metadata["transfer_evidence_sha256"] if continuous else None
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
    }
    result = {
        **payload,
        "artifact_sha256": _v14._sha256(
            payload, domain=_PROVIDER_RECEIPT_DOMAIN
        ),
    }
    _v14._scalar_report(result)
    return result


def _expected_response_weight_sha256(
    *, arm: str, selected_coordinate_index: int
) -> str | None:
    if arm == "base":
        return None
    weight = torch.zeros(3, dtype=torch.float64)
    if arm != "constant_plus_one":
        weight[selected_coordinate_index] = 1.0
    return _provider_tensor_sha256(weight)


def _validate_provider_receipt_semantics(
    provider_receipts: Mapping[str, Mapping[str, object]],
    *,
    selected_coordinate_index: int,
    law_evidence_sha256: str,
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
) -> dict[str, dict[str, object]]:
    """Rebuild every arm's transparent continuous-provider semantics."""

    if selected_coordinate_index not in {0, 1} or set(provider_receipts) != set(
        _ARMS
    ):
        raise ValueError("V20c provider semantic geometry differs")
    base_sha = _sha(base_provider_artifact_sha256, label="semantic base provider")
    proposal_sha = _sha(
        proposal_provider_artifact_sha256, label="semantic proposal provider"
    )
    evidence_sha = _sha(law_evidence_sha256, label="semantic law evidence")
    expected = {
        "base": ("base_zero", "base_zero", 0, None),
        "signed_log": ("direct", "signed_log", 1, _core.CONTINUOUS_RESPONSE_KAPPA),
        "constant_plus_one": (
            "constant",
            "linear",
            1,
            _core.CONTINUOUS_RESPONSE_KAPPA,
        ),
        "signed_log_sign_flip": (
            "direct",
            "signed_log",
            -1,
            _core.CONTINUOUS_RESPONSE_KAPPA,
        ),
        "linear": ("direct", "linear", 1, _core.CONTINUOUS_RESPONSE_KAPPA),
    }
    result: dict[str, dict[str, object]] = {}
    provider_shas: set[str] = set()
    for arm in _ARMS:
        receipt = dict(_mapping(provider_receipts[arm], label=f"{arm} provider receipt"))
        source, law, polarity, kappa = expected[arm]
        if set(receipt) != _PROVIDER_RECEIPT_KEYS:
            raise ValueError(f"V20c {arm} provider receipt fields differ")
        receipt_payload = {
            key: value for key, value in receipt.items() if key != "artifact_sha256"
        }
        provider_sha = _sha(
            receipt.get("provider_artifact_sha256"), label=f"{arm} provider artifact"
        )
        expected_semantics = {
            "arm": arm,
            "response_weight_sha256": _expected_response_weight_sha256(
                arm=arm, selected_coordinate_index=selected_coordinate_index
            ),
            "response_source": source,
            "response_law": law,
            "polarity": polarity,
            "signed_log_kappa": kappa,
            "base_provider_artifact_sha256": base_sha,
            "proposal_provider_artifact_sha256": (
                None if arm == "base" else proposal_sha
            ),
            "transfer_protocol_sha256": (
                None
                if arm == "base"
                else _core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256
            ),
            "transfer_evidence_sha256": (
                None if arm == "base" else evidence_sha
            ),
            "analysis_only": arm != "base",
            "rank": 256,
            "conditional_rank": _CONDITIONAL_RANK,
            "prepared_float_scalar_count": (
                377_608 if arm == "base" else 394_515
            ),
            "logical_macs_per_token_upper_bound": (
                541_187 if arm == "base" else 565_769
            ),
        }
        mismatches = [
            key
            for key, expected_value in expected_semantics.items()
            if receipt.get(key) != expected_value
        ]
        _sha(
            receipt.get("provider_metadata_sha256"),
            label=f"{arm} provider metadata",
        )
        if _sha(
            receipt.get("artifact_sha256"), label=f"{arm} provider receipt"
        ) != _v14._sha256(receipt_payload, domain=_PROVIDER_RECEIPT_DOMAIN):
            mismatches.append("artifact_sha256")
        if arm == "base" and provider_sha != base_sha:
            mismatches.append("provider_artifact_sha256")
        if mismatches:
            raise ValueError(
                f"V20c {arm} provider semantics differ: {','.join(mismatches)}"
            )
        provider_shas.add(provider_sha)
        result[arm] = receipt
    if len(provider_shas) != len(_ARMS):
        raise ValueError("V20c provider artifacts are not arm-distinct")
    return result


def _validate_provider_execution_bindings(
    provider_receipts: Mapping[str, Mapping[str, object]],
    *,
    roles: Sequence[Mapping[str, object]],
    role_evidence: Sequence[Mapping[str, object]],
    selected_coordinate_index: int,
    law_evidence_sha256: str,
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    fit_receipt_sha256: str,
) -> None:
    receipts = _validate_provider_receipt_semantics(
        provider_receipts,
        selected_coordinate_index=selected_coordinate_index,
        law_evidence_sha256=law_evidence_sha256,
        base_provider_artifact_sha256=base_provider_artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider_artifact_sha256,
    )
    role_rows = tuple(roles)
    evidence_rows = tuple(role_evidence)
    if len(role_rows) != 2 or len(evidence_rows) != 2:
        raise ValueError("V20c provider binding role geometry differs")
    evidence_by_role = {
        (
            row.get("outer_held_family_id"),
            row.get("scored_inner_family_id"),
        ): row
        for row in evidence_rows
    }
    if len(evidence_by_role) != 2:
        raise ValueError("V20c provider binding evidence is duplicated")
    for role in role_rows:
        outer = str(role["outer_held_family_id"])
        held = str(role["held_family_id"])
        evidence = _mapping(
            evidence_by_role.get((outer, held)), label="V20c role execution evidence"
        )
        evidence_payload = {
            key: value for key, value in evidence.items() if key != "artifact_sha256"
        }
        if _sha(
            evidence.get("artifact_sha256"), label="V20c role evidence receipt"
        ) != _v14._sha256(evidence_payload, domain=_ROLE_EVIDENCE_DOMAIN):
            raise ValueError("V20c role execution evidence hash differs")
        capability = _mapping(
            evidence.get("capability_receipt"), label="V20c role capability"
        )
        arm_evidence = _mapping(
            evidence.get("arm_execution_evidence"), label="V20c arm evidence"
        )
        scores = {
            str(score["arm"]): _mapping(score, label="V20c core arm score")
            for score in _sequence(role.get("arm_scores"), label="V20c arm scores")
            if isinstance(score, Mapping)
        }
        if set(arm_evidence) != set(_ARMS) or set(scores) != set(_ARMS):
            raise ValueError("V20c provider binding arm geometry differs")
        base_execution = _mapping(
            arm_evidence["base"], label="V20c base persisted execution"
        )
        base_h4 = dict(
            _mapping(base_execution.get("post_cast_h4_sha256s"), label="base H4")
        )
        base_logits = dict(
            _mapping(
                base_execution.get("supervised_full_vocab_logits_sha256s"),
                label="base logits",
            )
        )
        example_ids: set[str] | None = None
        for arm in _ARMS:
            receipt = receipts[arm]
            execution = _mapping(
                arm_evidence[arm], label=f"{arm} persisted execution"
            )
            trace = _mapping(
                execution.get("response_trace"), label=f"{arm} response trace"
            )
            trace_payload = {
                key: value for key, value in trace.items() if key != "artifact_sha256"
            }
            if _sha(
                trace.get("artifact_sha256"), label=f"{arm} response trace"
            ) != _v14._sha256(trace_payload, domain=_RESPONSE_TRACE_DOMAIN):
                raise ValueError(f"V20c {arm} response trace hash differs")
            score = scores[arm]
            objective = float(execution["objective"])
            if not math.isfinite(objective) or objective <= 0.0:
                raise ValueError("V20c persisted objective is invalid")
            current_ids = set(
                str(key)
                for key in _mapping(
                    execution.get("post_cast_h4_sha256s"),
                    label=f"{arm} H4 hashes",
                )
            )
            logits_ids = set(
                str(key)
                for key in _mapping(
                    execution.get("supervised_full_vocab_logits_sha256s"),
                    label=f"{arm} logits hashes",
                )
            )
            gain_ids = set(
                str(key)
                for key in _mapping(
                    trace.get("response_gain_sha256s"),
                    label=f"{arm} response gain hashes",
                )
            )
            if logits_ids != current_ids or gain_ids != current_ids:
                raise ValueError("V20c response/H4/logits example binding differs")
            derived_changed = arm != "base" and (
                dict(execution["post_cast_h4_sha256s"]) != base_h4
                or dict(execution["supervised_full_vocab_logits_sha256s"])
                != base_logits
            )
            if example_ids is None:
                example_ids = current_ids
            elif current_ids != example_ids:
                raise ValueError("V20c role arm example geometry differs")
            expected_execution = _execution_receipt(
                arm=arm,
                provider_artifact_sha256=str(
                    receipt["provider_artifact_sha256"]
                ),
                fit_receipt_sha256=fit_receipt_sha256,
                outer_family_id=outer,
                scored_family_id=held,
                h4_sha256s={
                    str(key): _sha(value, label=f"{arm} H4 hash")
                    for key, value in _mapping(
                        execution.get("post_cast_h4_sha256s"),
                        label=f"{arm} H4 hashes",
                    ).items()
                },
                logits_sha256s={
                    str(key): _sha(value, label=f"{arm} logits hash")
                    for key, value in _mapping(
                        execution.get("supervised_full_vocab_logits_sha256s"),
                        label=f"{arm} logits hashes",
                    ).items()
                },
                response_trace_sha256=_sha(
                    trace.get("artifact_sha256"), label=f"{arm} response trace"
                ),
                objective=objective,
            )
            if (
                execution.get("arm") != arm
                or execution.get("provider_artifact_sha256")
                != receipt["provider_artifact_sha256"]
                or trace.get("arm") != arm
                or trace.get("provider_artifact_sha256")
                != receipt["provider_artifact_sha256"]
                or trace.get("artifact_sha256")
                != score.get("response_trace_sha256")
                or execution.get("execution_receipt_sha256")
                != expected_execution
                or score.get("execution_receipt_sha256") != expected_execution
                or score.get("arm") != arm
                or float(score.get("objective", float("nan"))) != objective
                or execution.get("execution_changed_from_base") != derived_changed
                or score.get("execution_changed_from_base") != derived_changed
                or score.get("finite") != trace.get("finite")
                or score.get("pointwise_trust_passed")
                != trace.get("pointwise_trust_passed")
                or score.get("rank_is_16")
                != trace.get("endpoint_conditional_ranks_are_16")
                or tuple(trace.get("scored_family_ids", ())) != (held,)
            ):
                raise ValueError(f"V20c {arm} provider execution binding differs")
        if (
            example_ids is None
            or len(example_ids) != _PROMPTS_PER_FAMILY
            or set(capability.get("per_example_access_counts", ())) != example_ids
            or capability.get("access_count") != len(_ARMS) * _PROMPTS_PER_FAMILY
            or any(
                value != len(_ARMS)
                for value in capability["per_example_access_counts"].values()
            )
            or capability.get("held_family_id") != outer
            or capability.get("authorized_example_count") != _PROMPTS_PER_FAMILY
            or capability.get("authorized_family_count") != 1
            or _sha(
                capability.get("artifact_sha256"),
                label="V20c live-issued capability",
            )
            != capability.get("artifact_sha256")
            or capability.get("held_family_capability_excluded") is not True
            or capability.get("teacher_rows_consumed_only_through_capability")
            is not True
            or evidence.get("signed_log_mirror_exact") is not True
            or evidence.get("law_and_all_providers_frozen_before_capability")
            is not True
        ):
            raise ValueError("V20c role capability execution binding differs")


def _build_frozen_providers(
    workspace: object,
    *,
    selected_coordinate_index: int,
    law_evidence_sha256: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if selected_coordinate_index not in {0, 1}:
        raise ValueError("V20c selected coordinate index differs")
    base = workspace.base_provider
    proposal = workspace.proposal_provider
    common = {
        "transfer_protocol_sha256": _core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256,
        "transfer_evidence_sha256": law_evidence_sha256,
    }
    providers: dict[str, object] = {
        "base": base,
        "constant_plus_one": (
            build_autonomous_complete_h4_fisher_continuous_constant_control(
                base, proposal, alpha=1, **common
            )
        ),
        "linear": build_autonomous_complete_h4_fisher_continuous_axis_response(
            base,
            proposal,
            coordinate_index=selected_coordinate_index,
            response_law="linear",
            polarity=1,
            **common,
        ),
        "signed_log": build_autonomous_complete_h4_fisher_continuous_axis_response(
            base,
            proposal,
            coordinate_index=selected_coordinate_index,
            response_law="signed_log",
            polarity=1,
            signed_log_kappa=_core.CONTINUOUS_RESPONSE_KAPPA,
            **common,
        ),
        "signed_log_sign_flip": (
            build_autonomous_complete_h4_fisher_continuous_axis_response(
                base,
                proposal,
                coordinate_index=selected_coordinate_index,
                response_law="signed_log",
                polarity=-1,
                signed_log_kappa=_core.CONTINUOUS_RESPONSE_KAPPA,
                **common,
            )
        ),
    }
    if set(providers) != set(_ARMS):
        raise RuntimeError("V20c provider arm geometry differs")
    receipts = {
        arm: _provider_receipt(providers[arm], arm=arm) for arm in _ARMS
    }
    if len({receipt["provider_artifact_sha256"] for receipt in receipts.values()}) != len(_ARMS):
        raise RuntimeError("V20c provider artifacts are not arm-distinct")
    return providers, receipts


def _response_runtime_trace(
    provider: object,
    records: Sequence[object],
    *,
    arm: str,
) -> dict[str, object]:
    sequences = tuple(getattr(record, "sequence") for record in records)
    runtime = _v19._held_runtime_diagnostics(provider, sequences)
    gains: dict[str, str] = {}
    mirror_values: dict[str, Tensor] = {}
    for sequence in sorted(sequences, key=lambda value: value.artifact_sha256):
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
            raise RuntimeError("V20c response trace is empty or nonfinite")
        gains[sequence.example_id] = _v14._tensor_sha256(selected)
        mirror_values[sequence.example_id] = selected
    payload = {
        "arm": arm,
        "provider_artifact_sha256": provider.artifact_sha256,
        "scored_family_ids": tuple(sorted({sequence.family_id for sequence in sequences})),
        "response_gain_sha256s": dict(sorted(gains.items())),
        "runtime_receipt_sha256": runtime["receipt_sha256"],
        "finite": True,
        "pointwise_trust_passed": runtime["pointwise_trust_passed"],
        "max_bounded_direction_to_parent_norm_ratio": runtime[
            "max_bounded_direction_to_parent_norm_ratio"
        ],
        "max_emitted_delta_to_parent_norm_ratio": runtime[
            "max_emitted_delta_to_parent_norm_ratio"
        ],
        "endpoint_conditional_ranks_are_16": (
            int(provider.conditional_rank) == _CONDITIONAL_RANK
        ),
        "raw_response_or_modal_tensors_serialized": False,
    }
    trace = {
        **payload,
        "artifact_sha256": _v14._sha256(payload, domain=_RESPONSE_TRACE_DOMAIN),
    }
    _v14._scalar_report(trace)
    # Transient values are returned separately for the exact mirror assertion.
    trace["_transient_gain_values"] = mirror_values
    return trace


def _strip_transient_trace(trace: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in trace.items() if not key.startswith("_transient_")}


def _assert_exact_mirror(
    positive_trace: Mapping[str, object], mirror_trace: Mapping[str, object]
) -> None:
    positive = _mapping(
        positive_trace.get("_transient_gain_values"), label="positive gains"
    )
    mirror = _mapping(
        mirror_trace.get("_transient_gain_values"), label="mirror gains"
    )
    if set(positive) != set(mirror) or any(
        not torch.equal(mirror[key], -positive[key]) for key in positive
    ):
        raise RuntimeError("V20c signed-log mirror is not exact")


def _execution_receipt(
    *,
    arm: str,
    provider_artifact_sha256: str,
    fit_receipt_sha256: str,
    outer_family_id: str,
    scored_family_id: str,
    h4_sha256s: Mapping[str, str],
    logits_sha256s: Mapping[str, str],
    response_trace_sha256: str,
    objective: float,
) -> str:
    return _v14._sha256(
        {
            "arm": arm,
            "provider_artifact_sha256": provider_artifact_sha256,
            "fit_receipt_sha256": fit_receipt_sha256,
            "outer_held_family_id": outer_family_id,
            "scored_inner_family_id": scored_family_id,
            "post_cast_h4_sha256s": dict(sorted(h4_sha256s.items())),
            "supervised_full_vocab_logits_sha256s": dict(
                sorted(logits_sha256s.items())
            ),
            "response_trace_sha256": response_trace_sha256,
            "objective": float(objective),
        },
        domain=_EXECUTION_DOMAIN,
    )


def _score_exact_arm(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    arm: str,
    outer_family_id: str,
    fit_receipt_sha256: str,
    baseline_hashes: tuple[Mapping[str, str], Mapping[str, str]] | None,
) -> tuple[dict[str, object], tuple[dict[str, str], dict[str, str]]]:
    ordered = _v20b._ordered_records(records)
    families = {record.sequence.family_id for record in ordered}
    if len(ordered) != _PROMPTS_PER_FAMILY or len(families) != 1:
        raise RuntimeError("V20c score role must contain exactly one family")
    scored_family = next(iter(families))
    trace_with_values = _response_runtime_trace(provider, ordered, arm=arm)
    trace = _strip_transient_trace(trace_with_values)
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
        score, h4_hash, logits_hash = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=provider.artifact_sha256,
        )
        prompt_scores[record.sequence.example_id] = score
        h4_hashes[record.sequence.example_id] = h4_hash
        logits_hashes[record.sequence.example_id] = logits_hash
        del model_inputs, teacher, execution
    objective, family_scores = _v19._family_equal_mean(prompt_scores, ordered)
    if set(family_scores) != {scored_family}:
        raise RuntimeError("V20c score family objective geometry differs")
    changed = False
    if baseline_hashes is not None:
        changed = h4_hashes != dict(baseline_hashes[0]) or logits_hashes != dict(
            baseline_hashes[1]
        )
    receipt = _execution_receipt(
        arm=arm,
        provider_artifact_sha256=provider.artifact_sha256,
        fit_receipt_sha256=fit_receipt_sha256,
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
        "provider_artifact_sha256": provider.artifact_sha256,
        "execution_receipt_sha256": receipt,
        "post_cast_h4_sha256s": dict(sorted(h4_hashes.items())),
        "supervised_full_vocab_logits_sha256s": dict(
            sorted(logits_hashes.items())
        ),
        "response_trace": trace,
        "execution_changed_from_base": changed,
    }
    _v14._scalar_report(result)
    # Keep the transient trace available only to the caller for mirror proof.
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
    law_receipt: Mapping[str, object],
    fit_receipt_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    selected_records = _v20b._ordered_records(
        tuple(
            record
            for record in records
            if record.sequence.family_id == scored_family_id
        )
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in selected_records),
        held_family_id=outer_family_id,
    )
    evidence: dict[str, dict[str, object]] = {}
    baseline, baseline_hashes = _score_exact_arm(
        context,
        selected_records,
        capability,
        provider=providers["base"],
        arm="base",
        outer_family_id=outer_family_id,
        fit_receipt_sha256=fit_receipt_sha256,
        baseline_hashes=None,
    )
    evidence["base"] = baseline
    for arm in _ARMS:
        if arm == "base":
            continue
        row, _hashes = _score_exact_arm(
            context,
            selected_records,
            capability,
            provider=providers[arm],
            arm=arm,
            outer_family_id=outer_family_id,
            fit_receipt_sha256=fit_receipt_sha256,
            baseline_hashes=baseline_hashes,
        )
        evidence[arm] = row
    _assert_exact_mirror(
        _mapping(evidence["signed_log"]["_transient_trace"], label="signed trace"),
        _mapping(
            evidence["signed_log_sign_flip"]["_transient_trace"],
            label="mirror trace",
        ),
    )
    arm_scores: list[dict[str, object]] = []
    persisted_evidence: dict[str, dict[str, object]] = {}
    for arm in _ARMS:
        raw = evidence[arm]
        trace = _mapping(raw["response_trace"], label=f"{arm} response trace")
        score = _core.build_continuous_response_arm_score(
            law_receipt=law_receipt,
            arm=arm,
            objective=float(raw["objective"]),
            execution_receipt_sha256=str(raw["execution_receipt_sha256"]),
            response_trace_sha256=str(trace["artifact_sha256"]),
            finite=trace.get("finite") is True,
            pointwise_trust_passed=trace.get("pointwise_trust_passed") is True,
            rank_is_16=trace.get("endpoint_conditional_ranks_are_16") is True,
            execution_changed_from_base=bool(raw["execution_changed_from_base"]),
        )
        arm_scores.append(score)
        persisted_evidence[arm] = {
            key: value for key, value in raw.items() if not key.startswith("_transient_")
        }
    role = _core.build_continuous_response_role_receipt(
        law_receipt=law_receipt, arm_scores=arm_scores
    )
    capability_receipt = capability.receipt()
    expected_accesses = len(_ARMS)
    if (
        capability_receipt.get("held_family_id") != outer_family_id
        or capability_receipt.get("authorized_family_count") != 1
        or capability_receipt.get("authorized_example_count") != _PROMPTS_PER_FAMILY
        or capability_receipt.get("access_count")
        != expected_accesses * _PROMPTS_PER_FAMILY
        or any(
            value != expected_accesses
            for value in capability_receipt["per_example_access_counts"].values()
        )
    ):
        raise RuntimeError("V20c held capability access geometry differs")
    role_evidence_payload = {
        "outer_held_family_id": outer_family_id,
        "scored_inner_family_id": scored_family_id,
        "capability_receipt": capability_receipt,
        "arm_execution_evidence": persisted_evidence,
        "signed_log_mirror_exact": True,
        "law_and_all_providers_frozen_before_capability": True,
    }
    role_evidence = {
        **role_evidence_payload,
        "artifact_sha256": _v14._sha256(
            role_evidence_payload, domain=_ROLE_EVIDENCE_DOMAIN
        ),
    }
    _v14._scalar_report(role_evidence)
    return role, role_evidence


def _pair_qualification(roles: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = tuple(_core.validate_continuous_response_role_receipt(role) for role in roles)
    if len(selected) != 2 or len({role["held_family_id"] for role in selected}) != 2:
        raise ValueError("V20c pair qualification requires two reciprocal roles")
    objectives = {
        arm: tuple(float(role["arm_objectives"][arm]) for role in selected)
        for arm in _ARMS
    }
    macros = {arm: math.fsum(values) / len(values) for arm, values in objectives.items()}
    signed = objectives["signed_log"]
    base = objectives["base"]
    mirror = objectives["signed_log_sign_flip"]
    base_improvements = tuple((b - s) / b for b, s in zip(base, signed))
    macro_improvement = (macros["base"] - macros["signed_log"]) / macros["base"]
    worst_improvement = min(base_improvements)
    numerical_floor = 1.0e-12
    health = all(
        score["finite"] is True
        and score["pointwise_trust_passed"] is True
        and score["rank_is_16"] is True
        and score["score_source"] == "exact_finite_execution"
        and score["predicted_only"] is False
        and (score["arm"] == "base" or score["execution_changed_from_base"] is True)
        for role in selected
        for score in role["arm_scores"]
    )
    gates = {
        "signed_log_family_equal_macro_improvement_at_least_1pct": (
            macro_improvement >= 0.01
        ),
        "signed_log_improves_both_roles": all(value > numerical_floor for value in base_improvements),
        "worst_role_regression_at_most_2pct": worst_improvement >= -0.02,
        "signed_log_beats_constant_plus_one_macro": (
            macros["constant_plus_one"] - macros["signed_log"] > numerical_floor
        ),
        "signed_log_beats_linear_macro": (
            macros["linear"] - macros["signed_log"] > numerical_floor
        ),
        "signed_log_beats_mirror_macro": (
            macros["signed_log_sign_flip"] - macros["signed_log"]
            > numerical_floor
        ),
        "signed_log_beats_mirror_in_both_roles": all(
            mirror_value - selected_value > numerical_floor
            for mirror_value, selected_value in zip(mirror, signed)
        ),
        "all_arms_finite_trusted_endpoint_rank16_changed_exact": health,
    }
    payload = {
        "aggregation": "family_equal_two_reciprocal_roles",
        "role_artifact_sha256s": tuple(role["artifact_sha256"] for role in selected),
        "arm_macro_objectives": macros,
        "signed_log_family_equal_macro_relative_improvement": macro_improvement,
        "required_family_equal_macro_relative_improvement": 0.01,
        "signed_log_relative_improvements_by_role": base_improvements,
        "worst_role_relative_improvement": worst_improvement,
        "maximum_worst_role_regression": 0.02,
        "gates": gates,
        "passed": all(gates.values()),
        "scope": "bounded_pair_smoke_not_seven_role_or_fresh_validation",
    }
    return {
        **payload,
        "artifact_sha256": _v14._sha256(
            payload, domain=_PAIR_QUALIFICATION_DOMAIN
        ),
    }


def _work_accounting() -> dict[str, object]:
    breakdown = {
        "collection_native_source_forwards": 16,
        "collection_base_vjp_forwards": 16,
        "pair_endpoint_reconstruction_forwards": 12,
        "held_exact_arm_score_forwards": 2 * len(_ARMS) * _PROMPTS_PER_FAMILY,
        "collection_base_vjp_backwards": 16,
        "pair_endpoint_reconstruction_backwards": 12,
        "pair_endpoint_local_head_contractions": 12,
    }
    forward_count = sum(value for key, value in breakdown.items() if "forwards" in key)
    backward_count = sum(value for key, value in breakdown.items() if "backwards" in key)
    local_count = breakdown["pair_endpoint_local_head_contractions"]
    work = {
        "full_model_forward_count": forward_count,
        "full_suffix_backward_traversal_count": backward_count,
        "local_head_autograd_contraction_count": local_count,
        "total_autograd_grad_call_count": backward_count + local_count,
        "teacher_capability_access_count": 12
        + 2 * len(_ARMS) * _PROMPTS_PER_FAMILY,
        "post_cast_h4_hash_check_count": 12
        + 2 * len(_ARMS) * _PROMPTS_PER_FAMILY,
        "supervised_full_vocab_logits_hash_check_count": 12
        + 2 * len(_ARMS) * _PROMPTS_PER_FAMILY,
        "physical_pair_endpoint_fit_count": 1,
        "held_role_count": 2,
        "exact_arm_score_count": 2 * len(_ARMS),
        "empirical_fisher_response_weight_fit_count": 0,
        "breakdown": breakdown,
    }
    if (
        forward_count != 64
        or backward_count != 28
        or local_count != 12
        or work["teacher_capability_access_count"] != 32
    ):
        raise RuntimeError("V20c work accounting differs")
    return work


def _build_report(
    *,
    output: Path,
    source: Mapping[str, object],
    source_pair_diagnostic: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    workspace: object,
    coordinate_trace: Mapping[str, object],
    law_evidence_sha256: str,
    provider_receipts: Mapping[str, Mapping[str, object]],
    roles: Sequence[Mapping[str, object]],
    role_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    qualification = _pair_qualification(roles)
    fit_receipt = dict(workspace.fit_receipt)
    fit_evidence = dict(workspace.fit_training_evidence)
    work = _work_accounting()
    selected_indices = {
        int(role["law_receipt"]["coordinate_statistics"]["selected_coordinate_index"])
        for role in roles
    }
    if len(selected_indices) != 1:
        raise RuntimeError("V20c persisted roles selected different coordinates")
    _validate_coordinate_trace_law_binding(
        coordinate_trace,
        roles=roles,
        fit_receipt=fit_receipt,
    )
    _validate_provider_execution_bindings(
        provider_receipts,
        roles=roles,
        role_evidence=role_evidence,
        selected_coordinate_index=next(iter(selected_indices)),
        law_evidence_sha256=law_evidence_sha256,
        base_provider_artifact_sha256=str(
            fit_receipt["base_provider_artifact_sha256"]
        ),
        proposal_provider_artifact_sha256=str(
            fit_receipt["proposal_provider_artifact_sha256"]
        ),
        fit_receipt_sha256=str(fit_receipt["artifact_sha256"]),
    )
    report = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "artifact": output.as_posix(),
        "experiment_stage": "A16_development_only_reused_panel_pair_smoke",
        "scientific_status": "development_only_not_fresh_family_disjoint_validation",
        "fixed_protocol": dict(_FIXED_PROTOCOL),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256,
        "source": dict(source),
        "source_pair_diagnostic": dict(source_pair_diagnostic),
        "panel_receipt": dict(panel_receipt),
        "shared_fit_receipt": fit_receipt,
        "fit_training_evidence": fit_evidence,
        "coordinate_trace_receipt": dict(coordinate_trace),
        "law_evidence_sha256": law_evidence_sha256,
        "provider_receipts": {
            arm: dict(provider_receipts[arm]) for arm in _ARMS
        },
        "roles": tuple(dict(role) for role in roles),
        "role_execution_evidence": tuple(dict(row) for row in role_evidence),
        "pair_qualification": qualification,
        "classification": (
            "fixed_continuous_response_pair_smoke_passed"
            if qualification["passed"] is True
            else "fixed_continuous_response_pair_smoke_failed"
        ),
        "passed": qualification["passed"] is True,
        "next_full_reused_panel_screen_authorized": qualification["passed"] is True,
        "empirical_fisher_response_weight_fit_implemented": False,
        "empirical_fisher_response_weight_fit_status": (
            "deferred_until_fixed_law_pair_smoke_result"
        ),
        "fresh_family_disjoint_claim_authorized": False,
        "held_fidelity_claim": False,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "end_to_end_parameter_or_flop_claim": False,
        "candidate": None,
        "provider_sidecar": None,
        "raw_tensors_logits_gradients_or_coordinates_serialized": False,
        "work_accounting": work,
    }
    _v14._scalar_report(report)
    _validate_report_lineage_and_claims(report)
    return report


def _validate_report_lineage_and_claims(
    value: Mapping[str, object],
    *,
    authenticated_v20b_report: Mapping[str, object] | None = None,
    authenticated_pair_fragment: Mapping[str, object] | None = None,
) -> None:
    """Rebuild the pinned scientific lineage and all negative claim flags."""

    panel = _v20b._core.validate_nested_microstep_panel_receipt(
        _mapping(value.get("panel_receipt"), label="V20c report panel")
    )
    fit = _v20b._core.validate_nested_microstep_fit_receipt(
        _mapping(value.get("shared_fit_receipt"), label="V20c report shared fit"),
        panel_receipt=panel,
    )
    _v20b._validate_fit_training_evidence(
        value.get("fit_training_evidence"), fit_receipt=fit
    )
    source = _mapping(value.get("source"), label="V20c report source")
    source_payload = {
        key: item for key, item in source.items() if key != "artifact_sha256"
    }
    base_sha = str(fit["base_provider_artifact_sha256"])
    proposal_sha = str(fit["proposal_provider_artifact_sha256"])
    coordinate_trace = _mapping(
        value.get("coordinate_trace_receipt"), label="V20c report coordinate trace"
    )
    expected_law_evidence = _v14._sha256(
        {
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "source_artifact_sha256": source.get("artifact_sha256"),
            "fit_receipt_sha256": fit["artifact_sha256"],
            "coordinate_trace_sha256": coordinate_trace.get("artifact_sha256"),
            "signed_log_kappa": _core.CONTINUOUS_RESPONSE_KAPPA,
            "signed_log_lambda": 1.0,
            "linear_lambda": 1.0,
        },
        domain=_LAW_EVIDENCE_DOMAIN,
    )
    families = tuple(panel["family_ids"])
    roles = tuple(
        _mapping(item, label="V20c lineage role")
        for item in _sequence(value.get("roles"), label="V20c lineage roles")
    )
    observed_roles = {
        (role.get("outer_held_family_id"), role.get("held_family_id"))
        for role in roles
    }
    if (
        value.get("schema") != _SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or _v14._canonical_json_bytes(value.get("fixed_protocol"))
        != _v14._canonical_json_bytes(_FIXED_PROTOCOL)
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256")
        != _core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256
        or source.get("artifact_sha256")
        != _v14._sha256(source_payload, domain=_SOURCE_DOMAIN)
        or source.get("path") != _V20B_OUTPUT.as_posix()
        or source.get("report_logical_sha256") != _V20B_LOGICAL_SHA256
        or source.get("report_file_sha256") != _V20B_FILE_SHA256
        or source.get("classification") != _V20B_CLASSIFICATION
        or source.get("passed") is not False
        or source.get("pair_key") != _FROZEN_PAIR_KEY
        or source.get("pair_fragment_sha256") != _FROZEN_PAIR_FRAGMENT_SHA256
        or tuple(source.get("excluded_family_ids", ())) != _FROZEN_EXCLUDED
        or source.get("base_provider_artifact_sha256") != base_sha
        or source.get("proposal_provider_artifact_sha256") != proposal_sha
        or source.get("authenticated_before_model_work") is not True
        or value.get("law_evidence_sha256") != expected_law_evidence
        or fit.get("fit_key") != _FROZEN_PAIR_KEY
        or tuple(fit.get("excluded_family_ids", ())) != _FROZEN_EXCLUDED
        or tuple(fit.get("training_family_ids", ()))
        != tuple(item for item in families if item not in _FROZEN_EXCLUDED)
        or fit.get("rank") != 256
        or fit.get("conditional_rank") != _CONDITIONAL_RANK
        or fit.get("finite") is not True
        or fit.get("pointwise_trust_passed") is not True
        or observed_roles != {(_REED, _SUNDIAL), (_SUNDIAL, _REED)}
        or len(roles) != 2
        or value.get("experiment_stage")
        != "A16_development_only_reused_panel_pair_smoke"
        or value.get("scientific_status")
        != "development_only_not_fresh_family_disjoint_validation"
        or value.get("empirical_fisher_response_weight_fit_implemented") is not False
        or value.get("fresh_family_disjoint_claim_authorized") is not False
        or value.get("held_fidelity_claim") is not False
        or value.get("serving_authorized") is not False
        or value.get("compression_claim") is not False
        or value.get("speed_or_latency_claim") is not False
        or value.get("end_to_end_parameter_or_flop_claim") is not False
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
        or value.get("raw_tensors_logits_gradients_or_coordinates_serialized")
        is not False
    ):
        raise ValueError("V20c report lineage or claim boundary differs")
    for role in roles:
        law = _core.validate_continuous_response_law_receipt(
            _mapping(role.get("law_receipt"), label="V20c lineage law"),
            expected_v20b_source_sha256s=_source_sha256s(),
            expected_base_provider_artifact_sha256=base_sha,
            expected_proposal_provider_artifact_sha256=proposal_sha,
        )
        if (
            tuple(law.get("family_ids", ())) != families
            or tuple(law.get("excluded_family_ids", ())) != _FROZEN_EXCLUDED
            or law.get("lambda_fit_evidence_sha256")
            != value.get("law_evidence_sha256")
            or law.get("held_data_used_for_statistics_or_fit") is not False
            or law.get("objectives_used_for_coordinate_selection") is not False
            or law.get("family_labels_used_for_coordinate_selection") is not False
        ):
            raise ValueError("V20c report law lineage differs")
    if authenticated_v20b_report is not None:
        if _v14._canonical_json_bytes(panel) != _v14._canonical_json_bytes(
            authenticated_v20b_report.get("panel_receipt")
        ):
            raise ValueError("V20c report panel differs from pinned V20b")
    if authenticated_pair_fragment is not None:
        if (
            _v14._canonical_json_bytes(fit)
            != _v14._canonical_json_bytes(
                authenticated_pair_fragment.get("shared_fit_receipt")
            )
            or _v14._canonical_json_bytes(value.get("fit_training_evidence"))
            != _v14._canonical_json_bytes(
                authenticated_pair_fragment.get("fit_training_evidence")
            )
        ):
            raise ValueError("V20c report endpoint evidence differs from pinned V20b")


def _load_existing_report(output: Path) -> dict[str, object]:
    value = _v20b._load_scalar_fragment(
        path=output,
        domain=_REPORT_DOMAIN,
        hash_key="report_sha256",
        label="V20c report",
    )
    _validate_report_lineage_and_claims(value)
    roles = tuple(
        _core.validate_continuous_response_role_receipt(
            _mapping(item, label="V20c persisted role")
        )
        for item in _sequence(value.get("roles"), label="V20c persisted roles")
    )
    qualification = _pair_qualification(roles)
    fit_receipt = _mapping(
        value.get("shared_fit_receipt"), label="V20c persisted shared fit"
    )
    _v20b._validate_fit_training_evidence(
        value.get("fit_training_evidence"), fit_receipt=fit_receipt
    )
    provider_receipts = {
        str(key): _mapping(item, label=f"persisted {key} provider receipt")
        for key, item in _mapping(
            value.get("provider_receipts"), label="V20c persisted providers"
        ).items()
    }
    role_evidence = tuple(
        _mapping(item, label="V20c persisted role execution evidence")
        for item in _sequence(
            value.get("role_execution_evidence"),
            label="V20c persisted role execution evidence",
        )
    )
    selected_indices = {
        int(role["law_receipt"]["coordinate_statistics"]["selected_coordinate_index"])
        for role in roles
    }
    if len(selected_indices) != 1:
        raise ValueError("V20c existing coordinate selection differs")
    _validate_coordinate_trace_law_binding(
        _mapping(
            value.get("coordinate_trace_receipt"),
            label="V20c persisted coordinate trace",
        ),
        roles=roles,
        fit_receipt=fit_receipt,
    )
    _validate_provider_execution_bindings(
        provider_receipts,
        roles=roles,
        role_evidence=role_evidence,
        selected_coordinate_index=next(iter(selected_indices)),
        law_evidence_sha256=_sha(
            value.get("law_evidence_sha256"), label="persisted law evidence"
        ),
        base_provider_artifact_sha256=str(
            fit_receipt["base_provider_artifact_sha256"]
        ),
        proposal_provider_artifact_sha256=str(
            fit_receipt["proposal_provider_artifact_sha256"]
        ),
        fit_receipt_sha256=str(fit_receipt["artifact_sha256"]),
    )
    if (
        value.get("schema") != _SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("artifact") != output.as_posix()
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256")
        != _core.CONTINUOUS_RESPONSE_PROTOCOL_SHA256
        or value.get("fresh_family_disjoint_claim_authorized") is not False
        or value.get("serving_authorized") is not False
        or value.get("compression_claim") is not False
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
        or value.get("raw_tensors_logits_gradients_or_coordinates_serialized")
        is not False
        or value.get("empirical_fisher_response_weight_fit_implemented") is not False
        or value.get("work_accounting") != _work_accounting()
        or _v14._canonical_json_bytes(value.get("pair_qualification"))
        != _v14._canonical_json_bytes(qualification)
        or value.get("passed") != (qualification["passed"] is True)
        or value.get("next_full_reused_panel_screen_authorized")
        != (qualification["passed"] is True)
        or value.get("classification")
        != (
            "fixed_continuous_response_pair_smoke_passed"
            if qualification["passed"] is True
            else "fixed_continuous_response_pair_smoke_failed"
        )
    ):
        raise ValueError("V20c existing report authority differs")
    source = _mapping(value.get("source"), label="V20c persisted source")
    if (
        source.get("report_logical_sha256") != _V20B_LOGICAL_SHA256
        or source.get("report_file_sha256") != _V20B_FILE_SHA256
        or source.get("classification") != _V20B_CLASSIFICATION
        or source.get("passed") is not False
        or source.get("pair_key") != _FROZEN_PAIR_KEY
        or source.get("pair_fragment_sha256") != _FROZEN_PAIR_FRAGMENT_SHA256
        or tuple(source.get("excluded_family_ids", ())) != _FROZEN_EXCLUDED
        or source.get("authenticated_before_model_work") is not True
    ):
        raise ValueError("V20c existing source authority differs")
    _authenticated_source, authenticated_report, authenticated_fragment = (
        _load_authenticated_v20b_source()
    )
    _validate_report_lineage_and_claims(
        value,
        authenticated_v20b_report=authenticated_report,
        authenticated_pair_fragment=authenticated_fragment,
    )
    return value


def run_gemma3_l3_l4_complete_h4_continuous_response_smoke(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or authenticate the fixed V20c reciprocal-pair smoke."""

    destination = _validate_output(output)
    if destination.exists():
        return _load_existing_report(destination)

    # Exact failed-source authentication intentionally precedes model creation.
    source, v20b_report, fragment = _load_authenticated_v20b_source()
    source_pair_diagnostic = _joint_one_source_diagnostic(fragment)
    panel_receipt = dict(
        _mapping(v20b_report.get("panel_receipt"), label="V20b panel receipt")
    )
    families = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="V20b panel family prompts",
            )
        )
    )

    # The V20a prerequisite must also authenticate before model construction;
    # the exact V20b report above cannot substitute for this independent
    # collection/panel authority check.
    prerequisite, _v20a_payload, _v20a_folds = (
        _v20b._load_authenticated_v20a_artifact()
    )
    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records, teacher_vault, live_families = _v20b._collect_live_fit_authority(
            context, prerequisite=prerequisite
        )
        if tuple(live_families) != families:
            raise RuntimeError("V20c live family order differs from V20b")
        workspace = _v20b._reconstruct_pair_workspace(
            context,
            records,
            teacher_vault,
            fragment=fragment,
            panel_receipt=panel_receipt,
        )
        grouped_coordinates, coordinate_trace = _fit_coordinates_by_family(
            workspace.training_records,
            base_provider=workspace.base_provider,
            family_ids=families,
            excluded_family_ids=_FROZEN_EXCLUDED,
        )
        law_evidence_sha256 = _v14._sha256(
            {
                "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
                "source_artifact_sha256": source["artifact_sha256"],
                "fit_receipt_sha256": workspace.fit_receipt["artifact_sha256"],
                "coordinate_trace_sha256": coordinate_trace["artifact_sha256"],
                "signed_log_kappa": _core.CONTINUOUS_RESPONSE_KAPPA,
                "signed_log_lambda": 1.0,
                "linear_lambda": 1.0,
            },
            domain=_LAW_EVIDENCE_DOMAIN,
        )
        fit_families = tuple(workspace.fit_receipt["training_family_ids"])
        law_by_role: dict[tuple[str, str], dict[str, object]] = {}
        for outer, held in ((_REED, _SUNDIAL), (_SUNDIAL, _REED)):
            law_by_role[(outer, held)] = _core.build_continuous_response_law_receipt(
                v20b_source_sha256s=_source_sha256s(),
                family_ids=families,
                outer_held_family_id=outer,
                held_family_id=held,
                coordinate_source_family_ids=fit_families,
                fit_coordinates_by_family=grouped_coordinates,
                base_provider_artifact_sha256=workspace.base_provider.artifact_sha256,
                proposal_provider_artifact_sha256=(
                    workspace.proposal_provider.artifact_sha256
                ),
                lambda_fit_evidence_sha256=law_evidence_sha256,
                signed_log_lambda=1.0,
                linear_lambda=1.0,
            )
        selected_indices = {
            int(law["coordinate_statistics"]["selected_coordinate_index"])
            for law in law_by_role.values()
        }
        if len(selected_indices) != 1:
            raise RuntimeError("V20c reciprocal roles selected different coordinates")
        providers, provider_receipts = _build_frozen_providers(
            workspace,
            selected_coordinate_index=next(iter(selected_indices)),
            law_evidence_sha256=law_evidence_sha256,
        )

        # No held-role teacher capability exists before this point.
        roles: list[dict[str, object]] = []
        role_evidence: list[dict[str, object]] = []
        for outer, held in ((_REED, _SUNDIAL), (_SUNDIAL, _REED)):
            role, evidence = _score_reciprocal_role(
                context,
                records,
                teacher_vault,
                outer_family_id=outer,
                scored_family_id=held,
                providers=providers,
                law_receipt=law_by_role[(outer, held)],
                fit_receipt_sha256=str(workspace.fit_receipt["artifact_sha256"]),
            )
            roles.append(role)
            role_evidence.append(evidence)
        report = _build_report(
            output=destination,
            source=source,
            source_pair_diagnostic=source_pair_diagnostic,
            panel_receipt=panel_receipt,
            workspace=workspace,
            coordinate_trace=coordinate_trace,
            law_evidence_sha256=law_evidence_sha256,
            provider_receipts=provider_receipts,
            roles=roles,
            role_evidence=role_evidence,
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
            label="V20c report",
        )
    except FileExistsError:
        # A concurrent identical runner may have won the immutable publish.
        # Never overwrite it; authenticate the complete artifact instead.
        return _load_existing_report(destination)
    return _load_existing_report(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only V20c fixed continuous-response pair smoke"
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_continuous_response_smoke(
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
