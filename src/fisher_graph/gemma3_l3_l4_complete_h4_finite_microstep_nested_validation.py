"""V20b true nested family-disjoint finite-microstep validation.

V20a proved only that a fractional first V19 Adam proposal improves the rows
used to choose it.  This rung removes that optimism with an exhaustive 8 x 7
nested design.  For each outer-held family, the other seven families are used
only through seven inner leave-one-family-out roles.  Every inner role scores
the complete 3 x 7 positive path/alpha grid, one policy is selected from the
seven-family macro, and its exact signed mirror is then scored.  The selected
policy is refit on all seven non-outer families and frozen, together with its
provider hashes, before a capability that can read the outer family exists.

The 56 directed inner roles share 28 independently fitted six-family
endpoints.  Expensive phases are protected by authenticated write-once 0600
scalar/hash-only fragments: one per unordered pair, one per outer inner
selection, one global selection lock, one per outer score, and a report-ready
checkpoint.  No provider tensor is serialized.  The final report remains a
validation artifact and makes no serving, compression, speed, parameter, or
FLOP claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import torch
from torch import Tensor

from .complete_h4_fisher_finite_microstep import (
    FisherFiniteMicrostepReceipt,
    build_autonomous_complete_h4_fisher_finite_microstep,
)
from . import complete_h4_fisher_nested_microstep as _core
from . import gemma3_l3_l4_complete_h4_autonomous_residual_development as _v14
from . import gemma3_l3_l4_complete_h4_finite_joint_pedal_development as _v19
from . import gemma3_l3_l4_complete_h4_finite_microstep_preflight as _v20a
from . import gemma3_l3_l4_complete_h4_fisher_pedal_development as _v18
from .gemma3_l3_l4_complete_h4_rank320_basis_materialization import (
    prepare_complete_h4_rank320_live_context,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_finite_microstep_nested_validation_report",
    "run_gemma3_l3_l4_complete_h4_finite_microstep_nested_validation",
    "build_parser",
    "main",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_V20A_OUTPUT = _v20a.DEFAULT_OUTPUT
_V20A_LOGICAL_SHA256 = (
    "255ba898d823d983bf1f3122796032f5001f204760b49fddc168fb13311aa84e"
)
_V20A_FILE_SHA256 = (
    "318e05f1643df19a674dbc0f36f7da05c65204b9e1e0561f5e37a6273dd355da"
)
_V20A_CLASSIFICATION = "finite_microstep_preflight_passed_for_nested_validation"

DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-complete-h4-finite-microstep-nested-validation-"
    "r16-k256-a-fit16-dev-v20b.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4.complete_h4_finite_microstep_"
    "nested_family_validation.v20b"
)
_FORMAT_VERSION = 20
_REPORT_DOMAIN = b"fisher-graph:finite-microstep-nested-report:v20b\0"
_RUNNER_PROTOCOL_DOMAIN = b"fisher-graph:finite-microstep-nested-runner:v20b\0"
_PANEL_BINDING_DOMAIN = b"fisher-graph:finite-microstep-nested-panel:v20b\0"
_PAIR_EVIDENCE_DOMAIN = b"fisher-graph:finite-microstep-nested-pair-evidence:v20b\0"
_EXECUTION_DOMAIN = b"fisher-graph:finite-microstep-nested-execution:v20b\0"
_FIT_EXECUTION_DOMAIN = (
    b"fisher-graph:finite-microstep-nested-fit-execution:v20b\0"
)
_ACCESS_ACCOUNTING_DOMAIN = (
    b"fisher-graph:finite-microstep-nested-access-accounting:v20b\0"
)
_PAIR_FRAGMENT_DOMAIN = b"fisher-graph:finite-microstep-nested-pair-fragment:v20b\0"
_INNER_FRAGMENT_DOMAIN = b"fisher-graph:finite-microstep-nested-inner-fragment:v20b\0"
_SELECTION_LOCK_DOMAIN = b"fisher-graph:finite-microstep-nested-selection-lock:v20b\0"
_OUTER_FRAGMENT_DOMAIN = b"fisher-graph:finite-microstep-nested-outer-fragment:v20b\0"
_REPORT_READY_DOMAIN = b"fisher-graph:finite-microstep-nested-report-ready:v20b\0"
_FULL_FIT_DOMAIN = b"fisher-graph:finite-microstep-nested-full-fit:v20b\0"

_PAIR_FRAGMENT_SCHEMA = "fisher_graph.complete_h4.nested_pair_fragment.v20b"
_INNER_FRAGMENT_SCHEMA = "fisher_graph.complete_h4.nested_inner_fragment.v20b"
_SELECTION_LOCK_SCHEMA = "fisher_graph.complete_h4.nested_selection_lock.v20b"
_OUTER_FRAGMENT_SCHEMA = "fisher_graph.complete_h4.nested_outer_fragment.v20b"
_REPORT_READY_SCHEMA = "fisher_graph.complete_h4.nested_report_ready.v20b"

_EXPECTED_FAMILIES = 8
_PROMPTS_PER_FAMILY = 2
_PAIR_COUNT = 28
_INNER_ROLE_COUNT = 56
_PARENT_RANK = 256
_CONDITIONAL_RANK = 16

_PATHS = tuple(_core.NESTED_MICROSTEP_PATHS)
_POSITIVE_ALPHAS = tuple(_core.NESTED_MICROSTEP_POSITIVE_ALPHAS)
_PATH_ORDER = {path: index for index, path in enumerate(_PATHS)}

_FIXED_PROTOCOL: dict[str, object] = {
    "protocol": "true_8x7_nested_family_disjoint_finite_microstep",
    "prerequisite": "authenticated_and_rebuilt_write_once_V20a",
    "outer_roles": 8,
    "inner_roles_per_outer": 7,
    "shared_six_family_endpoint_fits": 28,
    "shared_pair_rule": "one_fit_per_unordered_excluded_family_pair",
    "positive_grid": "all_three_paths_times_all_seven_nonzero_alphas",
    "positive_candidate_count_per_inner_role": 21,
    "selection_objective": "family_equal_macro_across_seven_inner_held_families",
    "selection_tie_break": "objective_then_smaller_alpha_then_path_order",
    "inner_direction_control": "exact_negative_of_outer_selected_policy_in_all_seven_roles",
    "outer_refit": "fresh_seven_family_checkpoint_zero_and_first_Adam_endpoints",
    "selection_lock": "all_eight_choices_and_outer_provider_hashes_before_outer_capability",
    "outer_schedule": "fixed_all_or_none_baseline_positive_mirror_for_all_eight_families",
    "outer_outcome_adaptivity": False,
    "persisted_execution_evidence": (
        "parameter_h4_logits_full_microstep_and_capability_receipts"
    ),
    "teacher_access_accounting": (
        "recomputed_from_authenticated_vault_capabilities_and_execution_hashes"
    ),
    "candidate_or_provider_sidecar": False,
    "claims_fail_closed": True,
    "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
}
_RUNNER_PROTOCOL_SHA256 = _v14._sha256(
    _FIXED_PROTOCOL,
    domain=_RUNNER_PROTOCOL_DOMAIN,
)


def _is_under_local_runs(path: Path) -> bool:
    return ".local-runs" in path.resolve(strict=False).parts


def _same_destination(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_output(path: Path | str) -> Path:
    output = Path(path)
    if output.suffix != ".json" or not _is_under_local_runs(output):
        raise ValueError("V20b output must be JSON under .local-runs")
    if _same_destination(output, _V20A_OUTPUT):
        raise ValueError("V20b must preserve the write-once V20a artifact")
    return output


def _identifier(value: object, *, label: str) -> str:
    return _v14._identifier(value, label=label)


def _sha(value: object, *, label: str) -> str:
    return _v19._sha256_identifier(value, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} mapping is missing")
    return value


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} sequence is missing")
    return tuple(value)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _secure_stat(path: Path, *, label: str) -> os.stat_result:
    selected = path.lstat()
    if (
        not stat.S_ISREG(selected.st_mode)
        or path.is_symlink()
        or stat.S_IMODE(selected.st_mode) != 0o600
        or selected.st_nlink != 1
        or selected.st_uid != os.getuid()
    ):
        raise RuntimeError(f"{label} is unsafe")
    return selected


def _publish_scalar_fragment(
    payload: Mapping[str, object],
    *,
    path: Path,
    domain: bytes,
    hash_key: str,
    label: str,
) -> dict[str, object]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {label}")
    value = dict(payload)
    if hash_key in value:
        raise ValueError(f"{label} payload already contains its hash")
    _v14._scalar_report(value)
    value[hash_key] = _v14._sha256(value, domain=domain)
    reservation = _v14._reserve_outputs((path,))
    stage: Path | None = None
    try:
        stage = _v14._stage_json(value, path)
        _secure_stat(stage, label=f"staged {label}")
        reservation.publish((stage,))
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)
    _secure_stat(path, label=f"published {label}")
    return value


def _load_scalar_fragment(
    *,
    path: Path,
    domain: bytes,
    hash_key: str,
    label: str,
) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{label} is absent")
    _secure_stat(path, label=label)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(value, Mapping) or hash_key not in value:
        raise ValueError(f"{label} fields differ")
    payload = {key: item for key, item in value.items() if key != hash_key}
    supplied = _sha(value.get(hash_key), label=f"{label} hash")
    if _v14._sha256(payload, domain=domain) != supplied:
        raise ValueError(f"{label} hash drifted")
    _v14._scalar_report(payload)
    return dict(value)


def _fragment_stem(output: Path | str) -> str:
    return _validate_output(output).resolve(strict=False).stem


def _pair_fragment_path(output: Path | str, pair_key: str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    key = _sha(pair_key, label="V20b pair key")
    return destination.with_name(f"{destination.stem}.pair-{key[:20]}.json")


def _inner_fragment_path(output: Path | str, outer_family_id: str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    outer = _identifier(outer_family_id, label="V20b outer family")
    suffix = _v14._sha256({"outer": outer}, domain=_INNER_FRAGMENT_DOMAIN)[:20]
    return destination.with_name(f"{destination.stem}.inner-{suffix}.json")


def _selection_lock_path(output: Path | str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    return destination.with_name(f"{destination.stem}.selection-lock.json")


def _outer_fragment_path(output: Path | str, outer_family_id: str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    outer = _identifier(outer_family_id, label="V20b outer family")
    suffix = _v14._sha256({"outer": outer}, domain=_OUTER_FRAGMENT_DOMAIN)[:20]
    return destination.with_name(f"{destination.stem}.outer-{suffix}.json")


def _report_ready_path(output: Path | str) -> Path:
    destination = _validate_output(output).resolve(strict=False)
    return destination.with_name(f"{destination.stem}.report-ready.json")


def _load_authenticated_v20a_artifact(
) -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    """Authenticate and independently rebuild V20a before any model work."""

    _secure_stat(_V20A_OUTPUT, label="pinned V20a report")
    if _v14._file_sha256(_V20A_OUTPUT) != _V20A_FILE_SHA256:
        raise RuntimeError("pinned V20a report file hash drifted")
    try:
        payload = json.loads(_V20A_OUTPUT.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("pinned V20a report is unreadable") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("pinned V20a report is not a mapping")
    logical_payload = dict(payload)
    logical_payload.pop("report_sha256", None)
    if (
        payload.get("schema") != _v20a._SCHEMA
        or payload.get("format_version") != 20
        or payload.get("report_sha256") != _V20A_LOGICAL_SHA256
        or _v14._sha256(logical_payload, domain=_v20a._REPORT_DOMAIN)
        != _V20A_LOGICAL_SHA256
        or payload.get("classification") != _V20A_CLASSIFICATION
        or payload.get("passed") is not True
        or payload.get("nested_v20b_authorized") is not True
        or payload.get("candidate") is not None
        or payload.get("provider_sidecar") is not None
        or _V20A_OUTPUT.with_suffix(".provider.pt").exists()
    ):
        raise RuntimeError("pinned V20a authority differs")
    rebuilt = _v20a.build_finite_microstep_preflight_report(
        artifact_path=_V20A_OUTPUT,
        panel=_mapping(payload.get("panel"), label="V20a panel"),
        bridge_binding_sha256=_sha(
            payload.get("bridge_binding_sha256"), label="V20a bridge"
        ),
        prerequisite=_mapping(payload.get("prerequisite"), label="V20a prerequisite"),
        fit_collection=_mapping(payload.get("fit_collection"), label="V20a collection"),
        sentinel=_mapping(payload.get("sentinel"), label="V20a sentinel"),
        folds=_sequence(payload.get("folds"), label="V20a folds"),  # type: ignore[arg-type]
        work=_mapping(payload.get("work_accounting"), label="V20a work"),
        integrity=_mapping(payload.get("integrity"), label="V20a integrity"),
    )
    rebuilt["report_sha256"] = _v14._sha256(rebuilt, domain=_v20a._REPORT_DOMAIN)
    if _v14._canonical_json_bytes(rebuilt) != _v14._canonical_json_bytes(payload):
        raise RuntimeError("pinned V20a report did not independently rebuild")

    folds = _sequence(payload.get("folds"), label="V20a folds")
    if len(folds) != _EXPECTED_FAMILIES:
        raise RuntimeError("pinned V20a fold geometry differs")
    prompt_hashes: dict[str, tuple[str, ...]] = {}
    fold_by_family: dict[str, dict[str, object]] = {}
    for raw in folds:
        fold = _mapping(raw, label="V20a fold")
        held = _identifier(fold.get("held_family_id"), label="V20a held family")
        ownership = _mapping(
            fold.get("ownership_receipt"), label="V20a ownership receipt"
        )
        hashes = tuple(
            _sha(item, label="V20a held prompt")
            for item in _sequence(
                ownership.get("held_sequence_sha256s"),
                label="V20a held prompt hashes",
            )
        )
        if len(hashes) != _PROMPTS_PER_FAMILY:
            raise RuntimeError("V20a held prompt geometry differs")
        prompt_hashes[held] = hashes
        fold_by_family[held] = dict(fold)
    panel_receipt = _core.build_nested_microstep_panel_receipt(prompt_hashes)
    prerequisite = {
        "path": _V20A_OUTPUT.as_posix(),
        "format_version": 20,
        "report_sha256": _V20A_LOGICAL_SHA256,
        "file_sha256": _V20A_FILE_SHA256,
        "classification": _V20A_CLASSIFICATION,
        "passed": True,
        "nested_v20b_authorized": True,
        "report_rebuilt_before_model_work": True,
        "candidate": None,
        "provider_sidecar": None,
        "authenticated_panel": dict(_mapping(payload.get("panel"), label="V20a panel")),
        "authenticated_bridge_binding_sha256": _sha(
            payload.get("bridge_binding_sha256"), label="V20a bridge"
        ),
        "authenticated_fit_collection": dict(
            _mapping(payload.get("fit_collection"), label="V20a collection")
        ),
        "nested_panel_receipt": panel_receipt,
    }
    _v14._scalar_report(prerequisite)
    return prerequisite, dict(payload), fold_by_family


def _panel_binding_sha256(
    *,
    panel_receipt: Mapping[str, object],
    bridge_binding_sha256: str,
) -> str:
    return _v14._sha256(
        {
            "panel_receipt": dict(panel_receipt),
            "bridge_binding_sha256": _sha(
                bridge_binding_sha256, label="V20b bridge binding"
            ),
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
            "v20a_report_sha256": _V20A_LOGICAL_SHA256,
            "v20a_file_sha256": _V20A_FILE_SHA256,
        },
        domain=_PANEL_BINDING_DOMAIN,
    )


def _validate_live_authority(
    *,
    prerequisite: Mapping[str, object],
    context: object,
    records: Sequence[object],
) -> tuple[str, ...]:
    panel = _mapping(
        prerequisite.get("authenticated_panel"), label="authenticated V20a panel"
    )
    if _v14._canonical_json_bytes(dict(panel)) != _v14._canonical_json_bytes(
        dict(getattr(context, "panel_receipt"))
    ):
        raise RuntimeError("live panel differs from authenticated V20a")
    bridge = _sha(
        getattr(getattr(context, "bridge"), "bridge_binding_sha256", None),
        label="live V20b bridge",
    )
    if bridge != prerequisite.get("authenticated_bridge_binding_sha256"):
        raise RuntimeError("live bridge differs from authenticated V20a")
    families = tuple(
        sorted(
            {
                _identifier(
                    getattr(getattr(record, "sequence"), "family_id", None),
                    label="live family",
                )
                for record in records
            }
        )
    )
    if len(records) != _EXPECTED_FAMILIES * _PROMPTS_PER_FAMILY or len(
        families
    ) != _EXPECTED_FAMILIES:
        raise RuntimeError("live V20b family geometry differs")
    panel_receipt = _mapping(
        prerequisite.get("nested_panel_receipt"), label="nested panel receipt"
    )
    family_rows = _mapping(
        panel_receipt.get("family_prompt_sha256s"),
        label="nested panel family prompts",
    )
    if set(families) != set(family_rows):
        raise RuntimeError("live families differ from authenticated nested panel")
    for family in families:
        current = tuple(
            sorted(
                _sha(
                    getattr(getattr(record, "sequence"), "artifact_sha256", None),
                    label="live training sequence",
                )
                for record in records
                if record.sequence.family_id == family
            )
        )
        expected = tuple(sorted(str(item) for item in family_rows[family]))
        if current != expected:
            raise RuntimeError("live prompt receipts differ from authenticated V20a")
    return families


@dataclass(slots=True)
class _EndpointWorkspace:
    excluded_family_ids: tuple[str, ...]
    training_records: tuple[object, ...]
    base_provider: object
    proposal_provider: object
    fit_receipt: dict[str, object]
    fit_training_evidence: dict[str, object]


@dataclass(slots=True)
class _ScoreEvidence:
    summary: dict[str, object]
    parameter_sha256s: dict[str, str]
    h4_sha256s: dict[str, str]
    logits_sha256s: dict[str, str]
    execution_receipt_sha256: str = ""
    execution_change: dict[str, object] | None = None
    microstep_receipt: dict[str, object] | None = None


def _ordered_records(records: Sequence[object]) -> tuple[object, ...]:
    return tuple(
        sorted(
            records,
            key=lambda value: (
                value.sequence.family_id,
                value.sequence.example_id,
            ),
        )
    )


def _endpoint_evidence_sha256(
    *,
    excluded_family_ids: Sequence[str],
    records: Sequence[object],
    panel_receipt: Mapping[str, object],
) -> str:
    return _v14._sha256(
        {
            "excluded_family_ids": tuple(sorted(excluded_family_ids)),
            "training_family_ids": tuple(
                sorted({record.sequence.family_id for record in records})
            ),
            "training_record_receipt_sha256s": tuple(
                record.receipt_sha256 for record in _ordered_records(records)
            ),
            "training_sequence_sha256s": tuple(
                record.sequence.artifact_sha256 for record in _ordered_records(records)
            ),
            "panel_receipt_sha256": panel_receipt["artifact_sha256"],
            "coordinate_objective": "reverse_vjp_fisher",
            "checkpoint_count": 2,
        },
        domain=_PAIR_EVIDENCE_DOMAIN,
    )


def _fit_endpoint_from_scratch(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    excluded_family_ids: Sequence[str],
    panel_receipt: Mapping[str, object],
    outer_fit: bool,
) -> _EndpointWorkspace:
    """Fit checkpoint zero/one from the exact family complement.

    Pair endpoints always use ``held_family_id=None`` in V19's provisional
    provider so reciprocal directed roles produce one identical artifact.
    Outer endpoints use their single held family as in V19/V20a.
    """

    excluded = tuple(sorted(_identifier(item, label="excluded family") for item in excluded_family_ids))
    expected_excluded = 1 if outer_fit else 2
    if len(excluded) != expected_excluded or len(set(excluded)) != expected_excluded:
        raise ValueError("V20b endpoint excluded-family geometry differs")
    training = _ordered_records(
        tuple(record for record in records if record.sequence.family_id not in excluded)
    )
    expected_families = _EXPECTED_FAMILIES - expected_excluded
    if (
        len(training) != expected_families * _PROMPTS_PER_FAMILY
        or len({record.sequence.family_id for record in training}) != expected_families
        or any(record.sequence.family_id in excluded for record in training)
    ):
        raise RuntimeError("V20b endpoint training complement differs")
    sequences = tuple(record.sequence for record in training)
    parent = _v19._fit_parent(
        sequences,
        bridge_binding_sha256=context.bridge.bridge_binding_sha256,
    )
    _v18._validate_parent(parent, expected_fit_family_count=expected_families)
    start = _v19._fit_v18_start(
        sequences,
        parent=parent,
        coordinate_objective="reverse_vjp_fisher",
    )
    _v18._validate_child(
        start,
        coordinate_objective="reverse_vjp_fisher",
        pedal_mode="conditional",
        expected_parent_artifact_sha256=parent.artifact_sha256,
        expected_fit_family_count=expected_families,
    )
    authorized = tuple(record.sequence.example_id for record in training)
    capability = teacher_vault.capability(
        authorized,
        held_family_id=excluded[0] if outer_fit else None,
    )
    state0 = _v19._initial_joint_state(start)
    base = _v19._provisional_provider(
        start,
        state0,
        held_family_id=excluded[0] if outer_fit else None,
        coordinate_objective="reverse_vjp_fisher",
        checkpoint=0,
    )
    gradients: list[object] = []
    prompt_scores: dict[str, float] = {}
    h4_sha256s: dict[str, str] = {}
    logits_sha256s: dict[str, str] = {}
    for record in training:
        model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
            context, record
        )
        teacher = capability.get(
            record.sequence.example_id,
            family_id=record.sequence.family_id,
        )
        objective, captured = _v19._teacher_kl_objective(
            teacher, supervised_indices
        )
        execution, h4_gradient = context.bridge.execute_h4_vjp(
            context.adapter,
            model_inputs,
            objective=objective,
            h4_head=base,
        )
        score, h4_sha256, logits_sha256 = _v20a._execution_hashes_and_score(
            execution=execution,
            record=record,
            teacher=teacher,
            supervised_indices=supervised_indices,
            provider_artifact_sha256=base.artifact_sha256,
        )
        if len(captured) != 1 or score != captured[0]:
            raise RuntimeError("V20b checkpoint-zero score replay drifted")
        prompt_scores[record.sequence.example_id] = score
        h4_sha256s[record.sequence.example_id] = h4_sha256
        logits_sha256s[record.sequence.example_id] = logits_sha256
        gradients.append(
            _v19._local_ste_parameter_gradients(
                base,
                state0,
                record.sequence,
                h4_gradient,
            )
        )
        del model_inputs, teacher, execution, h4_gradient
    zero = _v19._zero_state(state0)
    state1, _moments = _v19._adam_step(
        state0,
        _v19._mean_gradient(gradients),
        _v19._AdamMoments(first=zero, second=zero, step=0),
    )
    proposal = _v19._provisional_provider(
        start,
        state1,
        held_family_id=excluded[0] if outer_fit else None,
        coordinate_objective="reverse_vjp_fisher",
        checkpoint=1,
    )
    for provider in (base, proposal):
        _v19._validate_joint_provider(
            provider,
            start_provider=start,
            pedal_mode="conditional",
            expected_family_count=expected_families,
        )
    fit_evidence = _endpoint_evidence_sha256(
        excluded_family_ids=excluded,
        records=training,
        panel_receipt=panel_receipt,
    )
    flags = tuple(_v20a._runtime_flags(provider, training) for provider in (base, proposal))
    finite = all(flag.get("finite") is True and flag.get("rank_is_16") is True for flag in flags)
    trust = all(flag.get("pointwise_trust_passed") is True for flag in flags)
    if outer_fit:
        fit_receipt = _core.build_nested_microstep_outer_fit_receipt(
            panel_receipt=panel_receipt,
            outer_held_family_id=excluded[0],
            base_provider_artifact_sha256=base.artifact_sha256,
            proposal_provider_artifact_sha256=proposal.artifact_sha256,
            fit_protocol_sha256=_RUNNER_PROTOCOL_SHA256,
            fit_evidence_sha256=fit_evidence,
            rank=_PARENT_RANK,
            conditional_rank=_CONDITIONAL_RANK,
            finite=finite,
            pointwise_trust_passed=trust,
        )
    else:
        fit_receipt = _core.build_nested_microstep_shared_fit_receipt(
            panel_receipt=panel_receipt,
            excluded_family_ids=excluded,
            base_provider_artifact_sha256=base.artifact_sha256,
            proposal_provider_artifact_sha256=proposal.artifact_sha256,
            fit_protocol_sha256=_RUNNER_PROTOCOL_SHA256,
            fit_evidence_sha256=fit_evidence,
            rank=_PARENT_RANK,
            conditional_rank=_CONDITIONAL_RANK,
            finite=finite,
            pointwise_trust_passed=trust,
        )
    family_mean, family_scores = _v19._family_equal_mean(prompt_scores, training)
    if not math.isfinite(family_mean) or len(family_scores) != expected_families:
        raise RuntimeError("V20b endpoint fit objective differs")
    capability_receipt = capability.receipt()
    if (
        capability_receipt.get("access_count") != len(training)
        or any(count != 1 for count in capability_receipt["per_example_access_counts"].values())
    ):
        raise RuntimeError("V20b endpoint capability access geometry differs")
    fit_training_payload = {
        "fit_receipt_sha256": fit_receipt["artifact_sha256"],
        "provider_artifact_sha256": base.artifact_sha256,
        "parameter_sha256s": _v20a._parameter_sha256s(base),
        "example_family_ids": {
            record.sequence.example_id: record.sequence.family_id
            for record in training
        },
        "post_cast_h4_sha256s": dict(sorted(h4_sha256s.items())),
        "supervised_full_vocab_logits_sha256s": dict(
            sorted(logits_sha256s.items())
        ),
        "capability_receipt": capability_receipt,
        "raw_tensors_or_logits_serialized": False,
    }
    fit_training_evidence = {
        **fit_training_payload,
        "execution_receipt_sha256": _v14._sha256(
            fit_training_payload,
            domain=_FIT_EXECUTION_DOMAIN,
        ),
    }
    _validate_fit_training_evidence(
        fit_training_evidence,
        fit_receipt=fit_receipt,
    )
    return _EndpointWorkspace(
        excluded_family_ids=excluded,
        training_records=training,
        base_provider=base,
        proposal_provider=proposal,
        fit_receipt=fit_receipt,
        fit_training_evidence=fit_training_evidence,
    )


def _score_execution_receipt(
    *,
    provider_artifact_sha256: str,
    fit_receipt_sha256: str,
    scored_family_id: str,
    h4_sha256s: Mapping[str, str],
    logits_sha256s: Mapping[str, str],
) -> str:
    return _v14._sha256(
        {
            "provider_artifact_sha256": provider_artifact_sha256,
            "fit_receipt_sha256": fit_receipt_sha256,
            "scored_family_id": scored_family_id,
            "post_cast_h4_sha256s": dict(sorted(h4_sha256s.items())),
            "supervised_full_vocab_logits_sha256s": dict(
                sorted(logits_sha256s.items())
            ),
        },
        domain=_EXECUTION_DOMAIN,
    )


def _score_provider(
    context: object,
    records: Sequence[object],
    capability: object,
    *,
    provider: object,
    fit_receipt_sha256: str,
    baseline: _ScoreEvidence | None,
    path: str | None,
    alpha: float | None,
    microstep_receipt: Mapping[str, object] | None,
) -> _ScoreEvidence:
    ordered = _ordered_records(records)
    families = {record.sequence.family_id for record in ordered}
    if len(ordered) != _PROMPTS_PER_FAMILY or len(families) != 1:
        raise RuntimeError("V20b score role must contain exactly one family")
    family = next(iter(families))
    prompt_scores: dict[str, float] = {}
    h4_hashes: dict[str, str] = {}
    logits_hashes: dict[str, str] = {}
    for record in ordered:
        model_inputs, supervised_indices, _targets = _v20a._verified_model_inputs(
            context, record
        )
        teacher = capability.get(
            record.sequence.example_id,
            family_id=record.sequence.family_id,
        )
        execution = context.bridge.execute(
            context.adapter,
            model_inputs,
            h4_head=provider,
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
    if set(family_scores) != {family}:
        raise RuntimeError("V20b scored family objective geometry differs")
    parameters = _v20a._parameter_sha256s(provider)
    flags = _v20a._runtime_flags(provider, ordered)
    execution_receipt = _score_execution_receipt(
        provider_artifact_sha256=provider.artifact_sha256,
        fit_receipt_sha256=fit_receipt_sha256,
        scored_family_id=family,
        h4_sha256s=h4_hashes,
        logits_sha256s=logits_hashes,
    )
    if baseline is None:
        if microstep_receipt is not None:
            raise ValueError("V20b baseline cannot carry microstep provenance")
        change = None
        summary = _core.build_nested_microstep_baseline_score(
            objective=objective,
            fit_receipt_sha256=fit_receipt_sha256,
            provider_artifact_sha256=provider.artifact_sha256,
            execution_receipt_sha256=execution_receipt,
            finite=flags.get("finite") is True,
            pointwise_trust_passed=flags.get("pointwise_trust_passed") is True,
            rank_is_16=flags.get("rank_is_16") is True,
        )
    else:
        if path not in _PATH_ORDER or alpha is None or microstep_receipt is None:
            raise ValueError("V20b candidate score metadata is incomplete")
        receipt = FisherFiniteMicrostepReceipt.from_metadata(microstep_receipt)
        change = _v20a.detect_execution_change(
            base_parameter_sha256s=baseline.parameter_sha256s,
            candidate_parameter_sha256s=parameters,
            base_h4_sha256s=baseline.h4_sha256s,
            candidate_h4_sha256s=h4_hashes,
            base_logits_sha256s=baseline.logits_sha256s,
            candidate_logits_sha256s=logits_hashes,
        )
        summary = _core.build_nested_microstep_candidate_score(
            path=path,
            alpha=alpha,
            objective=objective,
            fit_receipt_sha256=fit_receipt_sha256,
            provider_artifact_sha256=provider.artifact_sha256,
            microstep_receipt_sha256=receipt.artifact_sha256,
            execution_change_receipt_sha256=change["receipt_sha256"],
            execution_changed=change["execution_changed"],
            finite=flags.get("finite") is True,
            pointwise_trust_passed=flags.get("pointwise_trust_passed") is True,
            rank_is_16=flags.get("rank_is_16") is True,
        )
    return _ScoreEvidence(
        summary=summary,
        parameter_sha256s=parameters,
        h4_sha256s=h4_hashes,
        logits_sha256s=logits_hashes,
        execution_receipt_sha256=execution_receipt,
        execution_change=change,
        microstep_receipt=None if microstep_receipt is None else dict(microstep_receipt),
    )


def _microstep_provider(
    workspace: _EndpointWorkspace,
    *,
    path: str,
    alpha: float,
    role_evidence: str,
) -> tuple[object, dict[str, object]]:
    result = build_autonomous_complete_h4_fisher_finite_microstep(
        workspace.base_provider,
        workspace.proposal_provider,
        microstep_path=path,
        alpha=alpha,
        microstep_protocol_sha256=_core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        microstep_evidence_sha256=role_evidence,
    )
    return result.provider, result.receipt.metadata()


def _pair_candidate_evidence_sha256(
    workspace: _EndpointWorkspace,
    *,
    path: str,
    alpha: float,
) -> str:
    return _pair_candidate_evidence_from_fit(
        workspace.fit_receipt,
        path=path,
        alpha=alpha,
    )


def _pair_candidate_evidence_from_fit(
    fit_receipt: Mapping[str, object],
    *,
    path: str,
    alpha: float,
) -> str:
    return _v14._sha256(
        {
            "pair_key": fit_receipt["fit_key"],
            "fit_receipt_sha256": fit_receipt["artifact_sha256"],
            "base_provider_artifact_sha256": fit_receipt[
                "base_provider_artifact_sha256"
            ],
            "proposal_provider_artifact_sha256": fit_receipt[
                "proposal_provider_artifact_sha256"
            ],
            "path": path,
            "alpha": alpha,
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        },
        domain=_PAIR_EVIDENCE_DOMAIN,
    )


def _outer_candidate_evidence_sha256(
    workspace: _EndpointWorkspace,
    *,
    outer_family_id: str,
    path: str,
    alpha: float,
    selection_artifact_sha256: str,
) -> str:
    return _outer_candidate_evidence_from_fit(
        workspace.fit_receipt,
        outer_family_id=outer_family_id,
        path=path,
        alpha=alpha,
        selection_artifact_sha256=selection_artifact_sha256,
    )


def _outer_candidate_evidence_from_fit(
    fit_receipt: Mapping[str, object],
    *,
    outer_family_id: str,
    path: str,
    alpha: float,
    selection_artifact_sha256: str,
) -> str:
    return _v14._sha256(
        {
            "outer_held_family_id": outer_family_id,
            "fit_receipt_sha256": fit_receipt["artifact_sha256"],
            "base_provider_artifact_sha256": fit_receipt[
                "base_provider_artifact_sha256"
            ],
            "proposal_provider_artifact_sha256": fit_receipt[
                "proposal_provider_artifact_sha256"
            ],
            "path": path,
            "alpha": alpha,
            "selection_artifact_sha256": selection_artifact_sha256,
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        },
        domain=_PAIR_EVIDENCE_DOMAIN,
    )


def _raw_hash_mapping(value: object, *, label: str) -> dict[str, str]:
    selected = _mapping(value, label=label)
    if not selected:
        raise ValueError(f"{label} cannot be empty")
    return {
        _identifier(key, label=f"{label} key"): _sha(item, label=f"{label} hash")
        for key, item in selected.items()
    }


_CAPABILITY_RECEIPT_KEYS = frozenset(
    {
        "artifact_sha256",
        "held_family_id",
        "authorized_example_count",
        "authorized_family_count",
        "access_count",
        "per_example_access_counts",
        "held_family_capability_excluded",
        "teacher_rows_consumed_only_through_capability",
    }
)


def _validate_capability_receipt(
    value: object,
    *,
    expected_example_ids: Sequence[str],
    expected_family_count: int,
    expected_held_family_id: str | None,
    expected_accesses_per_example: int,
    label: str,
) -> dict[str, object]:
    receipt = _mapping(value, label=label)
    if set(receipt) != _CAPABILITY_RECEIPT_KEYS:
        raise ValueError(f"{label} fields differ")
    examples = tuple(
        sorted(_identifier(item, label=f"{label} example") for item in expected_example_ids)
    )
    if len(examples) != len(set(examples)):
        raise ValueError(f"{label} example ownership differs")
    held = (
        None
        if expected_held_family_id is None
        else _identifier(expected_held_family_id, label=f"{label} held family")
    )
    counts = _mapping(
        receipt.get("per_example_access_counts"), label=f"{label} access counts"
    )
    normalized_counts: dict[str, int] = {}
    for key, item in counts.items():
        example = _identifier(key, label=f"{label} count example")
        if type(item) is not int or item < 0:
            raise TypeError(f"{label} access count differs")
        normalized_counts[example] = item
    expected_counts = {item: expected_accesses_per_example for item in examples}
    if (
        normalized_counts != expected_counts
        or receipt.get("held_family_id") != held
        or receipt.get("authorized_example_count") != len(examples)
        or receipt.get("authorized_family_count") != expected_family_count
        or receipt.get("access_count")
        != len(examples) * expected_accesses_per_example
        or receipt.get("held_family_capability_excluded") is not (held is not None)
        or receipt.get("teacher_rows_consumed_only_through_capability") is not True
    ):
        raise ValueError(f"{label} geometry differs")
    _sha(receipt.get("artifact_sha256"), label=f"{label} artifact")
    return {**receipt, "per_example_access_counts": normalized_counts}


_FIT_TRAINING_EVIDENCE_KEYS = frozenset(
    {
        "fit_receipt_sha256",
        "provider_artifact_sha256",
        "parameter_sha256s",
        "example_family_ids",
        "post_cast_h4_sha256s",
        "supervised_full_vocab_logits_sha256s",
        "capability_receipt",
        "raw_tensors_or_logits_serialized",
        "execution_receipt_sha256",
    }
)


def _validate_fit_training_evidence(
    value: object,
    *,
    fit_receipt: Mapping[str, object],
) -> dict[str, object]:
    evidence = _mapping(value, label="V20b fit training evidence")
    if set(evidence) != _FIT_TRAINING_EVIDENCE_KEYS:
        raise ValueError("V20b fit training evidence fields differ")
    fit_sha = _sha(fit_receipt.get("artifact_sha256"), label="fit evidence receipt")
    if (
        evidence.get("fit_receipt_sha256") != fit_sha
        or evidence.get("provider_artifact_sha256")
        != fit_receipt.get("base_provider_artifact_sha256")
        or evidence.get("raw_tensors_or_logits_serialized") is not False
    ):
        raise ValueError("V20b fit training provider binding differs")
    parameters = _raw_hash_mapping(
        evidence.get("parameter_sha256s"), label="fit training parameters"
    )
    if len(parameters) != 4:
        raise ValueError("V20b fit training parameter geometry differs")
    families = {
        _identifier(key, label="fit training example"): _identifier(
            item, label="fit training family"
        )
        for key, item in _mapping(
            evidence.get("example_family_ids"), label="fit example families"
        ).items()
    }
    h4 = _raw_hash_mapping(
        evidence.get("post_cast_h4_sha256s"), label="fit training H4"
    )
    logits = _raw_hash_mapping(
        evidence.get("supervised_full_vocab_logits_sha256s"),
        label="fit training logits",
    )
    expected_training_families = tuple(fit_receipt.get("training_family_ids", ()))
    if (
        set(families) != set(h4)
        or set(h4) != set(logits)
        or len(families) != int(fit_receipt.get("fit_prompt_count", -1))
        or set(families.values()) != set(expected_training_families)
        or any(
            tuple(families.values()).count(family) != _PROMPTS_PER_FAMILY
            for family in expected_training_families
        )
    ):
        raise ValueError("V20b fit training execution geometry differs")
    excluded = tuple(fit_receipt.get("excluded_family_ids", ()))
    held = excluded[0] if fit_receipt.get("kind") == "outer_full" else None
    capability = _validate_capability_receipt(
        evidence.get("capability_receipt"),
        expected_example_ids=tuple(families),
        expected_family_count=len(expected_training_families),
        expected_held_family_id=held,
        expected_accesses_per_example=1,
        label="fit training capability receipt",
    )
    payload = {
        key: item
        for key, item in evidence.items()
        if key != "execution_receipt_sha256"
    }
    if _v14._sha256(payload, domain=_FIT_EXECUTION_DOMAIN) != _sha(
        evidence.get("execution_receipt_sha256"), label="fit execution receipt"
    ):
        raise ValueError("V20b fit training execution receipt drifted")
    return {
        **evidence,
        "parameter_sha256s": parameters,
        "example_family_ids": families,
        "post_cast_h4_sha256s": h4,
        "supervised_full_vocab_logits_sha256s": logits,
        "capability_receipt": capability,
    }


_SCORE_EVIDENCE_KEYS = frozenset(
    {
        "score",
        "parameter_sha256s",
        "post_cast_h4_sha256s",
        "supervised_full_vocab_logits_sha256s",
        "execution_receipt_sha256",
        "execution_change",
        "microstep_receipt",
    }
)


def _score_evidence_payload(value: _ScoreEvidence) -> dict[str, object]:
    return {
        "score": dict(value.summary),
        "parameter_sha256s": dict(value.parameter_sha256s),
        "post_cast_h4_sha256s": dict(value.h4_sha256s),
        "supervised_full_vocab_logits_sha256s": dict(value.logits_sha256s),
        "execution_receipt_sha256": value.execution_receipt_sha256,
        "execution_change": (
            None if value.execution_change is None else dict(value.execution_change)
        ),
        "microstep_receipt": (
            None if value.microstep_receipt is None else dict(value.microstep_receipt)
        ),
    }


def _validate_score_evidence(
    value: object,
    *,
    fit_receipt: Mapping[str, object],
    scored_family_id: str,
    baseline: Mapping[str, object] | None,
    expected_microstep_evidence_sha256: str | None,
) -> dict[str, object]:
    evidence = _mapping(value, label="V20b score evidence")
    if set(evidence) != _SCORE_EVIDENCE_KEYS:
        raise ValueError("V20b score evidence fields differ")
    fit_sha = _sha(fit_receipt.get("artifact_sha256"), label="score evidence fit")
    family = _identifier(scored_family_id, label="score evidence family")
    parameters = _raw_hash_mapping(
        evidence.get("parameter_sha256s"), label="score evidence parameters"
    )
    h4 = _raw_hash_mapping(
        evidence.get("post_cast_h4_sha256s"), label="score evidence H4"
    )
    logits = _raw_hash_mapping(
        evidence.get("supervised_full_vocab_logits_sha256s"),
        label="score evidence logits",
    )
    if len(parameters) != 4 or len(h4) != _PROMPTS_PER_FAMILY or set(h4) != set(logits):
        raise ValueError("V20b score evidence hash geometry differs")
    execution_receipt = _score_execution_receipt(
        provider_artifact_sha256=str(
            _mapping(evidence.get("score"), label="score evidence score").get(
                "provider_artifact_sha256"
            )
        ),
        fit_receipt_sha256=fit_sha,
        scored_family_id=family,
        h4_sha256s=h4,
        logits_sha256s=logits,
    )
    if evidence.get("execution_receipt_sha256") != execution_receipt:
        raise ValueError("V20b score execution receipt drifted")
    score_value = _mapping(evidence.get("score"), label="score evidence score")
    if baseline is None:
        score = _core.validate_nested_microstep_baseline_score(score_value)
        if (
            evidence.get("execution_change") is not None
            or evidence.get("microstep_receipt") is not None
            or score.get("fit_receipt_sha256") != fit_sha
            or score.get("provider_artifact_sha256")
            != fit_receipt.get("base_provider_artifact_sha256")
            or score.get("execution_receipt_sha256") != execution_receipt
        ):
            raise ValueError("V20b baseline execution evidence drifted")
    else:
        score = _core.validate_nested_microstep_candidate_score(score_value)
        base = _mapping(baseline, label="candidate baseline evidence")
        receipt_metadata = _mapping(
            evidence.get("microstep_receipt"), label="candidate microstep receipt"
        )
        receipt = FisherFiniteMicrostepReceipt.from_metadata(receipt_metadata)
        expected_evidence = _sha(
            expected_microstep_evidence_sha256,
            label="candidate expected microstep evidence",
        )
        if (
            score.get("fit_receipt_sha256") != fit_sha
            or receipt.base_provider_artifact_sha256
            != fit_receipt.get("base_provider_artifact_sha256")
            or receipt.proposal_provider_artifact_sha256
            != fit_receipt.get("proposal_provider_artifact_sha256")
            or receipt.selected_provider_artifact_sha256
            != score.get("provider_artifact_sha256")
            or receipt.microstep_path != score.get("path")
            or receipt.alpha != score.get("alpha")
            or receipt.microstep_protocol_sha256
            != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
            or receipt.microstep_evidence_sha256 != expected_evidence
            or receipt.artifact_sha256 != score.get("microstep_receipt_sha256")
            or dict(receipt.selected_tensor_sha256s) != parameters
            or receipt.rank != _PARENT_RANK
            or receipt.conditional_rank != _CONDITIONAL_RANK
        ):
            raise ValueError("V20b candidate microstep binding differs")
        change = _v20a.detect_execution_change(
            base_parameter_sha256s=_mapping(
                base.get("parameter_sha256s"), label="candidate base parameters"
            ),
            candidate_parameter_sha256s=parameters,
            base_h4_sha256s=_mapping(
                base.get("post_cast_h4_sha256s"), label="candidate base H4"
            ),
            candidate_h4_sha256s=h4,
            base_logits_sha256s=_mapping(
                base.get("supervised_full_vocab_logits_sha256s"),
                label="candidate base logits",
            ),
            candidate_logits_sha256s=logits,
        )
        if (
            _v14._canonical_json_bytes(change)
            != _v14._canonical_json_bytes(evidence.get("execution_change"))
            or score.get("execution_change_receipt_sha256")
            != change.get("receipt_sha256")
            or score.get("execution_changed") is not change.get("execution_changed")
        ):
            raise ValueError("V20b candidate execution-change evidence drifted")
    return {
        **evidence,
        "score": score,
        "parameter_sha256s": parameters,
        "post_cast_h4_sha256s": h4,
        "supervised_full_vocab_logits_sha256s": logits,
    }


def _role_panel(
    *,
    outer_family_id: str,
    inner_family_id: str,
    fit_receipt_sha256: str,
    baseline: _ScoreEvidence,
    positive_candidates: Sequence[_ScoreEvidence],
    capability_receipt: Mapping[str, object],
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="pair panel outer family")
    inner = _identifier(inner_family_id, label="pair panel inner family")
    evidence_rows = tuple(_score_evidence_payload(row) for row in positive_candidates)
    candidates = tuple(dict(row.summary) for row in positive_candidates)
    expected = {(path, alpha) for path in _PATHS for alpha in _POSITIVE_ALPHAS}
    observed = {
        (str(row.get("path")), float(row.get("alpha", float("nan"))))
        for row in candidates
    }
    if len(candidates) != len(expected) or observed != expected:
        raise ValueError("V20b directed panel positive grid differs")
    payload = {
        "outer_held_family_id": outer,
        "inner_held_family_id": inner,
        "fit_receipt_sha256": _sha(
            fit_receipt_sha256, label="directed panel fit receipt"
        ),
        "baseline": dict(baseline.summary),
        "baseline_parameter_sha256s": dict(baseline.parameter_sha256s),
        "baseline_h4_sha256s": dict(baseline.h4_sha256s),
        "baseline_logits_sha256s": dict(baseline.logits_sha256s),
        "positive_candidates": candidates,
        "positive_candidate_evidence": evidence_rows,
        "capability_receipt": dict(capability_receipt),
        "outer_rows_consumed": False,
        "matched_negative_deferred_until_outer_policy_selection": True,
    }
    _v14._scalar_report(payload)
    return payload


def _evaluate_pair_matrix(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    workspace: _EndpointWorkspace,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    left, right = workspace.excluded_family_ids
    role_specs = ((left, right), (right, left))
    role_state: dict[
        tuple[str, str],
        tuple[object, tuple[object, ...], _ScoreEvidence, list[_ScoreEvidence]],
    ] = {}
    for outer, inner in role_specs:
        inner_records = _ordered_records(
            tuple(record for record in records if record.sequence.family_id == inner)
        )
        capability = teacher_vault.capability(
            tuple(record.sequence.example_id for record in inner_records),
            held_family_id=outer,
        )
        baseline = _score_provider(
            context,
            inner_records,
            capability,
            provider=workspace.base_provider,
            fit_receipt_sha256=str(workspace.fit_receipt["artifact_sha256"]),
            baseline=None,
            path=None,
            alpha=None,
            microstep_receipt=None,
        )
        role_state[(outer, inner)] = (capability, inner_records, baseline, [])

    for path in _PATHS:
        for alpha in _POSITIVE_ALPHAS:
            evidence = _pair_candidate_evidence_sha256(
                workspace,
                path=path,
                alpha=alpha,
            )
            provider, receipt = _microstep_provider(
                workspace,
                path=path,
                alpha=alpha,
                role_evidence=evidence,
            )
            for role in role_specs:
                capability, inner_records, baseline, candidates = role_state[role]
                scored = _score_provider(
                    context,
                    inner_records,
                    capability,
                    provider=provider,
                    fit_receipt_sha256=str(
                        workspace.fit_receipt["artifact_sha256"]
                    ),
                    baseline=baseline,
                    path=path,
                    alpha=alpha,
                    microstep_receipt=receipt,
                )
                candidates.append(scored)

    panels: list[dict[str, object]] = []
    for role in role_specs:
        capability, _records, baseline, candidates = role_state[role]
        receipt = capability.receipt()
        expected_accesses = 1 + len(_PATHS) * len(_POSITIVE_ALPHAS)
        if (
            receipt.get("held_family_id") != role[0]
            or receipt.get("authorized_family_count") != 1
            or receipt.get("authorized_example_count") != _PROMPTS_PER_FAMILY
            or receipt.get("access_count") != expected_accesses * _PROMPTS_PER_FAMILY
            or any(
                count != expected_accesses
                for count in receipt["per_example_access_counts"].values()
            )
        ):
            raise RuntimeError("V20b inner positive capability accesses differ")
        panels.append(
            _role_panel(
                outer_family_id=role[0],
                inner_family_id=role[1],
                fit_receipt_sha256=str(workspace.fit_receipt["artifact_sha256"]),
                baseline=baseline,
                positive_candidates=candidates,
                capability_receipt=receipt,
            )
        )
    return panels[0], panels[1]


_PAIR_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "v20a_report_sha256",
        "v20a_file_sha256",
        "panel_binding_sha256",
        "pair_key",
        "excluded_family_ids",
        "shared_fit_receipt",
        "fit_training_evidence",
        "directed_panels",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _validate_directed_panel(
    value: object,
    *,
    shared_fit_receipt: Mapping[str, object],
) -> dict[str, object]:
    panel = _mapping(value, label="V20b directed panel")
    expected_keys = {
        "outer_held_family_id",
        "inner_held_family_id",
        "fit_receipt_sha256",
        "baseline",
        "baseline_parameter_sha256s",
        "baseline_h4_sha256s",
        "baseline_logits_sha256s",
        "positive_candidates",
        "positive_candidate_evidence",
        "capability_receipt",
        "outer_rows_consumed",
        "matched_negative_deferred_until_outer_policy_selection",
    }
    if set(panel) != expected_keys:
        raise ValueError("V20b directed panel fields differ")
    outer = _identifier(panel.get("outer_held_family_id"), label="directed outer")
    inner = _identifier(panel.get("inner_held_family_id"), label="directed inner")
    if outer == inner:
        raise ValueError("V20b directed panel outer/inner families overlap")
    fit_sha = _sha(panel.get("fit_receipt_sha256"), label="directed fit receipt")
    if fit_sha != shared_fit_receipt.get("artifact_sha256"):
        raise ValueError("V20b directed panel fit binding differs")
    baseline_evidence = _validate_score_evidence(
        {
            "score": panel.get("baseline"),
            "parameter_sha256s": panel.get("baseline_parameter_sha256s"),
            "post_cast_h4_sha256s": panel.get("baseline_h4_sha256s"),
            "supervised_full_vocab_logits_sha256s": panel.get(
                "baseline_logits_sha256s"
            ),
            "execution_receipt_sha256": _mapping(
                panel.get("baseline"), label="directed baseline"
            ).get("execution_receipt_sha256"),
            "execution_change": None,
            "microstep_receipt": None,
        },
        fit_receipt=shared_fit_receipt,
        scored_family_id=inner,
        baseline=None,
        expected_microstep_evidence_sha256=None,
    )
    baseline = _mapping(baseline_evidence.get("score"), label="directed baseline")
    candidates = _sequence(panel.get("positive_candidates"), label="directed positives")
    expected = {(path, alpha) for path in _PATHS for alpha in _POSITIVE_ALPHAS}
    evidence_rows = _sequence(
        panel.get("positive_candidate_evidence"),
        label="directed positive evidence",
    )
    if len(evidence_rows) != len(candidates):
        raise ValueError("V20b directed positive evidence geometry differs")
    evidence_by_key: dict[str, dict[str, object]] = {}
    for raw in evidence_rows:
        row = _mapping(raw, label="directed positive evidence")
        score = _mapping(row.get("score"), label="directed evidence score")
        key = _identifier(score.get("key"), label="directed evidence key")
        if key in evidence_by_key:
            raise ValueError("V20b directed positive evidence is duplicated")
        candidate_path = str(score.get("path"))
        candidate_alpha = _finite(
            score.get("alpha"), label="directed evidence alpha"
        )
        evidence_by_key[key] = _validate_score_evidence(
            row,
            fit_receipt=shared_fit_receipt,
            scored_family_id=inner,
            baseline=baseline_evidence,
            expected_microstep_evidence_sha256=(
                _pair_candidate_evidence_from_fit(
                    shared_fit_receipt,
                    path=candidate_path,
                    alpha=candidate_alpha,
                )
            ),
        )
    observed: set[tuple[str, float]] = set()
    for raw in candidates:
        candidate = _core.validate_nested_microstep_candidate_score(
            _mapping(raw, label="directed positive")
        )
        path = str(candidate.get("path"))
        alpha = _finite(candidate.get("alpha"), label="directed positive alpha")
        if path not in _PATH_ORDER or alpha not in _POSITIVE_ALPHAS:
            raise ValueError("V20b directed candidate key differs")
        if candidate.get("fit_receipt_sha256") != fit_sha:
            raise ValueError("V20b directed candidate fit binding differs")
        evidence_row = evidence_by_key.get(str(candidate["key"]))
        if evidence_row is None or _v14._canonical_json_bytes(
            evidence_row["score"]
        ) != _v14._canonical_json_bytes(candidate):
            raise ValueError("V20b directed positive evidence binding differs")
        observed.add((path, alpha))
    if len(candidates) != len(expected) or observed != expected:
        raise ValueError("V20b directed positive grid is not exhaustive")
    capability = _validate_capability_receipt(
        panel.get("capability_receipt"),
        expected_example_ids=tuple(
            _mapping(
                baseline_evidence.get("post_cast_h4_sha256s"),
                label="directed baseline H4",
            )
        ),
        expected_family_count=1,
        expected_held_family_id=outer,
        expected_accesses_per_example=1 + len(expected),
        label="directed capability receipt",
    )
    if (
        panel.get("outer_rows_consumed") is not False
        or panel.get("matched_negative_deferred_until_outer_policy_selection") is not True
    ):
        raise ValueError("V20b directed capability or phase boundary differs")
    return {
        **panel,
        "baseline": dict(baseline),
        "positive_candidates": tuple(
            evidence_by_key[str(row["key"])]["score"] for row in candidates
        ),
        "positive_candidate_evidence": tuple(
            evidence_by_key[str(row["key"])] for row in candidates
        ),
        "capability_receipt": capability,
    }


def _publish_pair_fragment(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    workspace: _EndpointWorkspace,
    directed_panels: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    pair_key = _sha(workspace.fit_receipt.get("fit_key"), label="pair fit key")
    payload = {
        "schema": _PAIR_FRAGMENT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        "v20a_report_sha256": _V20A_LOGICAL_SHA256,
        "v20a_file_sha256": _V20A_FILE_SHA256,
        "panel_binding_sha256": _sha(
            panel_binding_sha256, label="pair panel binding"
        ),
        "pair_key": pair_key,
        "excluded_family_ids": workspace.excluded_family_ids,
        "shared_fit_receipt": workspace.fit_receipt,
        "fit_training_evidence": workspace.fit_training_evidence,
        "directed_panels": tuple(dict(row) for row in directed_panels),
        "candidate": None,
        "provider_sidecar": None,
    }
    return _publish_scalar_fragment(
        payload,
        path=_pair_fragment_path(output, pair_key),
        domain=_PAIR_FRAGMENT_DOMAIN,
        hash_key="fragment_sha256",
        label="V20b pair fragment",
    )


def _load_pair_fragment(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    excluded_family_ids: Sequence[str],
    panel_receipt: Mapping[str, object],
) -> dict[str, object]:
    excluded = tuple(sorted(_identifier(item, label="pair excluded family") for item in excluded_family_ids))
    if len(excluded) != 2 or len(set(excluded)) != 2:
        raise ValueError("V20b pair fragment excluded geometry differs")
    pair_key = _core.nested_microstep_fit_pair_key(*excluded)
    value = _load_scalar_fragment(
        path=_pair_fragment_path(output, pair_key),
        domain=_PAIR_FRAGMENT_DOMAIN,
        hash_key="fragment_sha256",
        label="V20b pair fragment",
    )
    if set(value) != _PAIR_FRAGMENT_KEYS:
        raise ValueError("V20b pair fragment fields differ")
    if (
        value.get("schema") != _PAIR_FRAGMENT_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("target_output") != _validate_output(output).as_posix()
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256") != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
        or value.get("v20a_report_sha256") != _V20A_LOGICAL_SHA256
        or value.get("v20a_file_sha256") != _V20A_FILE_SHA256
        or value.get("panel_binding_sha256") != panel_binding_sha256
        or value.get("pair_key") != pair_key
        or tuple(value.get("excluded_family_ids", ())) != excluded
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
    ):
        raise ValueError("V20b pair fragment authority differs")
    fit = _core.validate_nested_microstep_fit_receipt(
        _mapping(value.get("shared_fit_receipt"), label="shared fit receipt"),
        panel_receipt=panel_receipt,
    )
    if (
        fit.get("fit_key") != pair_key
        or tuple(fit.get("excluded_family_ids", ())) != excluded
        or fit.get("panel_artifact_sha256") is None
    ):
        raise ValueError("V20b shared fit receipt binding differs")
    fit_training_evidence = _validate_fit_training_evidence(
        value.get("fit_training_evidence"),
        fit_receipt=fit,
    )
    panels = _sequence(value.get("directed_panels"), label="directed pair panels")
    if len(panels) != 2:
        raise ValueError("V20b pair fragment must contain two directed panels")
    validated = tuple(
        _validate_directed_panel(panel, shared_fit_receipt=fit) for panel in panels
    )
    observed = {
        (panel["outer_held_family_id"], panel["inner_held_family_id"])
        for panel in validated
    }
    if observed != {(excluded[0], excluded[1]), (excluded[1], excluded[0])}:
        raise ValueError("V20b reciprocal directed panel geometry differs")
    # Provider artifacts are shared across reciprocal roles at each grid key.
    by_role = {
        (panel["outer_held_family_id"], panel["inner_held_family_id"]): {
            (row["path"], row["alpha"]): (
                row["provider_artifact_sha256"],
                row["microstep_receipt_sha256"],
            )
            for row in panel["positive_candidates"]
        }
        for panel in validated
    }
    if len(set(tuple(sorted(items.items())) for items in by_role.values())) != 1:
        raise ValueError("V20b reciprocal roles did not share candidate providers")
    return {
        **value,
        "fit_training_evidence": fit_training_evidence,
        "directed_panels": validated,
    }


def _panel_candidate_map(
    panel: Mapping[str, object],
) -> dict[tuple[str, float], Mapping[str, object]]:
    return {
        (str(row["path"]), float(row["alpha"])): row
        for row in _sequence(
            panel.get("positive_candidates"), label="inner panel positives"
        )
        if isinstance(row, Mapping)
    }


def _select_outer_policy(
    *,
    outer_family_id: str,
    panels: Sequence[Mapping[str, object]],
    panel_receipt: Mapping[str, object],
    shared_fit_receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Run the core pre-mirror selector over exactly seven directed panels."""

    outer = _identifier(outer_family_id, label="selector outer family")
    rows = tuple(panels)
    if len(rows) != _EXPECTED_FAMILIES - 1:
        raise ValueError("V20b selector requires seven inner roles")
    inner_ids = tuple(
        _identifier(row.get("inner_held_family_id"), label="selector inner family")
        for row in rows
    )
    if (
        len(set(inner_ids)) != _EXPECTED_FAMILIES - 1
        or outer in set(inner_ids)
        or any(row.get("outer_held_family_id") != outer for row in rows)
    ):
        raise ValueError("V20b selector role ownership differs")
    fit_rows: list[Mapping[str, object]] = []
    roles: list[dict[str, object]] = []
    for panel in rows:
        fit_sha = _sha(panel.get("fit_receipt_sha256"), label="selector fit")
        fit = shared_fit_receipts.get(fit_sha)
        if fit is None:
            raise ValueError("V20b selector panel references an unknown shared fit")
        fit_rows.append(fit)
        roles.append(
            _build_inner_role(
                panel_receipt=panel_receipt,
                shared_fit_receipt=fit,
                panel=panel,
                matched_negative=None,
            )
        )
    return _core.select_nested_microstep_inner_candidate(
        panel_receipt=panel_receipt,
        shared_fit_receipts=tuple(
            shared_fit_receipts[key] for key in sorted(shared_fit_receipts)
        ),
        outer_held_family_id=outer,
        inner_roles=tuple(roles),
        require_mirrors=False,
    )


def _score_inner_mirror(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    workspace: _EndpointWorkspace,
    panel: Mapping[str, object],
    path: str,
    alpha: float,
) -> dict[str, object]:
    outer = _identifier(panel.get("outer_held_family_id"), label="mirror outer")
    inner = _identifier(panel.get("inner_held_family_id"), label="mirror inner")
    inner_records = _ordered_records(
        tuple(record for record in records if record.sequence.family_id == inner)
    )
    capability = teacher_vault.capability(
        tuple(record.sequence.example_id for record in inner_records),
        held_family_id=outer,
    )
    baseline = _ScoreEvidence(
        summary=dict(_mapping(panel.get("baseline"), label="mirror baseline")),
        parameter_sha256s=_raw_hash_mapping(
            panel.get("baseline_parameter_sha256s"), label="mirror baseline parameters"
        ),
        h4_sha256s=_raw_hash_mapping(
            panel.get("baseline_h4_sha256s"), label="mirror baseline H4"
        ),
        logits_sha256s=_raw_hash_mapping(
            panel.get("baseline_logits_sha256s"), label="mirror baseline logits"
        ),
    )
    signed_alpha = -float(alpha)
    evidence = _pair_candidate_evidence_sha256(
        workspace,
        path=path,
        alpha=signed_alpha,
    )
    provider, receipt_metadata = _microstep_provider(
        workspace,
        path=path,
        alpha=signed_alpha,
        role_evidence=evidence,
    )
    scored = _score_provider(
        context,
        inner_records,
        capability,
        provider=provider,
        fit_receipt_sha256=str(workspace.fit_receipt["artifact_sha256"]),
        baseline=baseline,
        path=path,
        alpha=signed_alpha,
        microstep_receipt=receipt_metadata,
    )
    receipt = capability.receipt()
    if (
        receipt.get("held_family_id") != outer
        or receipt.get("authorized_family_count") != 1
        or receipt.get("access_count") != _PROMPTS_PER_FAMILY
        or any(count != 1 for count in receipt["per_example_access_counts"].values())
    ):
        raise RuntimeError("V20b inner mirror capability accesses differ")
    return {
        "outer_held_family_id": outer,
        "inner_held_family_id": inner,
        "fit_receipt_sha256": workspace.fit_receipt["artifact_sha256"],
        "score_evidence": _score_evidence_payload(scored),
        "capability_receipt": receipt,
    }


def _build_inner_role(
    *,
    panel_receipt: Mapping[str, object],
    shared_fit_receipt: Mapping[str, object],
    panel: Mapping[str, object],
    matched_negative: Mapping[str, object] | None,
) -> dict[str, object]:
    return _core.build_nested_microstep_inner_role(
        panel_receipt=panel_receipt,
        shared_fit_receipt=shared_fit_receipt,
        outer_held_family_id=str(panel["outer_held_family_id"]),
        inner_held_family_id=str(panel["inner_held_family_id"]),
        baseline=_mapping(panel.get("baseline"), label="inner role baseline"),
        positive_candidates=_sequence(
            panel.get("positive_candidates"), label="inner role positives"
        ),
        matched_negative=matched_negative,
    )


_INNER_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "v20a_report_sha256",
        "v20a_file_sha256",
        "panel_binding_sha256",
        "outer_held_family_id",
        "pair_fragment_sha256s",
        "selection_preview",
        "inner_roles",
        "matched_negative_evidence",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _publish_inner_fragment(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    outer_family_id: str,
    pair_fragments: Sequence[Mapping[str, object]],
    inner_roles: Sequence[Mapping[str, object]],
    matched_negative_evidence: Sequence[Mapping[str, object]],
    selection_preview: Mapping[str, object],
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="inner fragment outer")
    pair_hashes = {
        str(fragment["pair_key"]): _sha(
            fragment.get("fragment_sha256"), label="pair fragment hash"
        )
        for fragment in pair_fragments
    }
    if len(pair_hashes) != _EXPECTED_FAMILIES - 1:
        raise ValueError("V20b inner fragment pair ownership differs")
    payload = {
        "schema": _INNER_FRAGMENT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        "v20a_report_sha256": _V20A_LOGICAL_SHA256,
        "v20a_file_sha256": _V20A_FILE_SHA256,
        "panel_binding_sha256": _sha(
            panel_binding_sha256, label="inner panel binding"
        ),
        "outer_held_family_id": outer,
        "pair_fragment_sha256s": dict(sorted(pair_hashes.items())),
        "selection_preview": dict(selection_preview),
        "inner_roles": tuple(dict(role) for role in inner_roles),
        "matched_negative_evidence": tuple(
            dict(row) for row in matched_negative_evidence
        ),
        "candidate": None,
        "provider_sidecar": None,
    }
    return _publish_scalar_fragment(
        payload,
        path=_inner_fragment_path(output, outer),
        domain=_INNER_FRAGMENT_DOMAIN,
        hash_key="fragment_sha256",
        label="V20b inner-selection fragment",
    )


def _load_inner_fragment(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    outer_family_id: str,
    pair_fragments: Mapping[str, Mapping[str, object]],
    panel_receipt: Mapping[str, object],
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="inner fragment outer")
    value = _load_scalar_fragment(
        path=_inner_fragment_path(output, outer),
        domain=_INNER_FRAGMENT_DOMAIN,
        hash_key="fragment_sha256",
        label="V20b inner-selection fragment",
    )
    if set(value) != _INNER_FRAGMENT_KEYS:
        raise ValueError("V20b inner-selection fragment fields differ")
    if (
        value.get("schema") != _INNER_FRAGMENT_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("target_output") != _validate_output(output).as_posix()
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256") != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
        or value.get("v20a_report_sha256") != _V20A_LOGICAL_SHA256
        or value.get("v20a_file_sha256") != _V20A_FILE_SHA256
        or value.get("panel_binding_sha256") != panel_binding_sha256
        or value.get("outer_held_family_id") != outer
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
    ):
        raise ValueError("V20b inner-selection fragment authority differs")
    expected_pairs = {
        key: fragment["fragment_sha256"]
        for key, fragment in pair_fragments.items()
        if outer in tuple(fragment["excluded_family_ids"])
    }
    supplied_pairs = dict(
        _mapping(value.get("pair_fragment_sha256s"), label="inner pair hashes")
    )
    if supplied_pairs != dict(sorted(expected_pairs.items())) or len(supplied_pairs) != 7:
        raise ValueError("V20b inner fragment pair binding differs")
    roles = _sequence(value.get("inner_roles"), label="inner fragment roles")
    if len(roles) != 7:
        raise ValueError("V20b inner fragment role geometry differs")
    panels: list[Mapping[str, object]] = []
    rebuilt_roles: list[dict[str, object]] = []
    for raw_role in roles:
        role = _mapping(raw_role, label="inner fragment role")
        inner = _identifier(role.get("inner_held_family_id"), label="inner role family")
        pair_key = _core.nested_microstep_fit_pair_key(outer, inner)
        fragment = pair_fragments.get(pair_key)
        if fragment is None:
            raise ValueError("V20b inner role references an unknown pair")
        fit = _mapping(fragment.get("shared_fit_receipt"), label="inner shared fit")
        panel = next(
            (
                item
                for item in fragment["directed_panels"]
                if item["outer_held_family_id"] == outer
                and item["inner_held_family_id"] == inner
            ),
            None,
        )
        if panel is None:
            raise ValueError("V20b inner role has no directed source panel")
        rebuilt = _build_inner_role(
            panel_receipt=panel_receipt,
            shared_fit_receipt=fit,
            panel=panel,
            matched_negative=_mapping(
                role.get("matched_negative"), label="inner matched negative"
            )
            if role.get("matched_negative") is not None
            else None,
        )
        if _v14._canonical_json_bytes(rebuilt) != _v14._canonical_json_bytes(role):
            raise ValueError("V20b inner role did not rebuild")
        panels.append(panel)
        rebuilt_roles.append(rebuilt)
    shared_by_sha = {
        str(fragment["shared_fit_receipt"]["artifact_sha256"]): fragment[
            "shared_fit_receipt"
        ]
        for fragment in pair_fragments.values()
    }
    preview = _select_outer_policy(
        outer_family_id=outer,
        panels=panels,
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared_by_sha,
    )
    if _v14._canonical_json_bytes(preview) != _v14._canonical_json_bytes(
        value.get("selection_preview")
    ):
        raise ValueError("V20b inner selection preview drifted")
    selected = preview.get("selected")
    raw_mirror_evidence = _sequence(
        value.get("matched_negative_evidence"),
        label="inner matched-negative evidence",
    )
    evidence_by_inner: dict[str, dict[str, object]] = {}
    for raw in raw_mirror_evidence:
        record = _mapping(raw, label="inner matched-negative evidence row")
        if set(record) != {
            "outer_held_family_id",
            "inner_held_family_id",
            "fit_receipt_sha256",
            "score_evidence",
            "capability_receipt",
        }:
            raise ValueError("V20b inner matched-negative evidence fields differ")
        inner = _identifier(
            record.get("inner_held_family_id"), label="mirror evidence inner"
        )
        if record.get("outer_held_family_id") != outer or inner in evidence_by_inner:
            raise ValueError("V20b inner matched-negative evidence ownership differs")
        pair_key = _core.nested_microstep_fit_pair_key(outer, inner)
        pair_fragment = pair_fragments.get(pair_key)
        if pair_fragment is None:
            raise ValueError("V20b mirror evidence references an unknown pair")
        fit = _mapping(
            pair_fragment.get("shared_fit_receipt"), label="mirror evidence fit"
        )
        if record.get("fit_receipt_sha256") != fit.get("artifact_sha256"):
            raise ValueError("V20b mirror evidence fit binding differs")
        source_panel = _panel_for_role(
            pair_fragment,
            outer_family_id=outer,
            inner_family_id=inner,
        )
        base_evidence = {
            "score": source_panel.get("baseline"),
            "parameter_sha256s": source_panel.get("baseline_parameter_sha256s"),
            "post_cast_h4_sha256s": source_panel.get("baseline_h4_sha256s"),
            "supervised_full_vocab_logits_sha256s": source_panel.get(
                "baseline_logits_sha256s"
            ),
            "execution_receipt_sha256": _mapping(
                source_panel.get("baseline"), label="mirror baseline"
            ).get("execution_receipt_sha256"),
            "execution_change": None,
            "microstep_receipt": None,
        }
        score_evidence = _mapping(
            record.get("score_evidence"), label="mirror score evidence"
        )
        score = _mapping(score_evidence.get("score"), label="mirror evidence score")
        path = str(score.get("path"))
        alpha = _finite(score.get("alpha"), label="mirror evidence alpha")
        validated_score = _validate_score_evidence(
            score_evidence,
            fit_receipt=fit,
            scored_family_id=inner,
            baseline=base_evidence,
            expected_microstep_evidence_sha256=_pair_candidate_evidence_from_fit(
                fit,
                path=path,
                alpha=alpha,
            ),
        )
        capability = _validate_capability_receipt(
            record.get("capability_receipt"),
            expected_example_ids=tuple(
                _mapping(
                    validated_score.get("post_cast_h4_sha256s"),
                    label="mirror score H4",
                )
            ),
            expected_family_count=1,
            expected_held_family_id=outer,
            expected_accesses_per_example=1,
            label="inner mirror capability receipt",
        )
        evidence_by_inner[inner] = {
            **record,
            "score_evidence": validated_score,
            "capability_receipt": capability,
        }
    for role in rebuilt_roles:
        mirror = role.get("matched_negative")
        inner = str(role["inner_held_family_id"])
        evidence = evidence_by_inner.get(inner)
        if selected is None:
            if mirror is not None or evidence is not None:
                raise ValueError("V20b baseline selection emitted a mirror")
        elif (
            not isinstance(mirror, Mapping)
            or evidence is None
            or mirror.get("path") != selected["path"]
            or mirror.get("alpha") != -float(selected["alpha"])
            or _v14._canonical_json_bytes(mirror)
            != _v14._canonical_json_bytes(
                _mapping(
                    evidence["score_evidence"], label="validated mirror evidence"
                ).get("score")
            )
        ):
            raise ValueError("V20b inner mirror does not match frozen policy")
    expected_mirror_count = 0 if selected is None else _EXPECTED_FAMILIES - 1
    if len(evidence_by_inner) != expected_mirror_count:
        raise ValueError("V20b inner mirror evidence schedule differs")
    return {
        **value,
        "inner_roles": tuple(rebuilt_roles),
        "matched_negative_evidence": tuple(
            evidence_by_inner[key] for key in sorted(evidence_by_inner)
        ),
    }


def _build_selection_receipt(
    *,
    panel_receipt: Mapping[str, object],
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    shared = tuple(
        _mapping(fragment.get("shared_fit_receipt"), label="selection shared fit")
        for _, fragment in sorted(pair_fragments.items())
    )
    roles = tuple(
        role
        for _, fragment in sorted(inner_fragments.items())
        for role in _sequence(fragment.get("inner_roles"), label="selection inner roles")
    )
    if len(shared) != _PAIR_COUNT or len(roles) != _INNER_ROLE_COUNT:
        raise ValueError("V20b complete nested selection geometry differs")
    return _core.build_nested_microstep_selection_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=shared,
        inner_roles=roles,
    )


def _selection_preview_by_outer(
    inner_fragments: Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    return {
        outer: _mapping(
            fragment.get("selection_preview"), label="selection preview"
        )
        for outer, fragment in inner_fragments.items()
    }


def _selection_item_by_outer(
    selection_receipt: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    rows = _sequence(
        selection_receipt.get("outer_selections"), label="core outer selections"
    )
    selected = {
        _identifier(row.get("outer_held_family_id"), label="core selection outer"): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(selected) != _EXPECTED_FAMILIES:
        raise ValueError("V20b core selection outer geometry differs")
    return selected


def _prepare_outer_refit(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
    selection_preview: Mapping[str, object],
    selection_item: Mapping[str, object],
    v20a_fold: Mapping[str, object],
) -> tuple[_EndpointWorkspace, dict[str, object], object, object]:
    outer = _identifier(outer_family_id, label="outer refit family")
    selected = _mapping(
        selection_preview.get("selected"), label="outer selected policy"
    )
    path = str(selected.get("path"))
    alpha = _finite(selected.get("alpha"), label="outer selected alpha")
    if path not in _PATH_ORDER or alpha not in _POSITIVE_ALPHAS:
        raise ValueError("V20b outer selected policy differs")
    workspace = _fit_endpoint_from_scratch(
        context,
        records,
        teacher_vault,
        excluded_family_ids=(outer,),
        panel_receipt=panel_receipt,
        outer_fit=True,
    )
    endpoint = _mapping(v20a_fold.get("endpoint_binding"), label="V20a outer endpoint")
    if (
        workspace.base_provider.artifact_sha256
        != endpoint.get("base_provider_artifact_sha256")
        or workspace.proposal_provider.artifact_sha256
        != endpoint.get("proposal_provider_artifact_sha256")
    ):
        raise RuntimeError("V20b outer refit endpoints differ from authenticated V20a")
    selection_sha = _sha(
        selection_item.get("artifact_sha256"), label="outer core selection"
    )
    positive_evidence = _outer_candidate_evidence_sha256(
        workspace,
        outer_family_id=outer,
        path=path,
        alpha=alpha,
        selection_artifact_sha256=selection_sha,
    )
    positive, positive_receipt = _microstep_provider(
        workspace,
        path=path,
        alpha=alpha,
        role_evidence=positive_evidence,
    )
    negative_evidence = _outer_candidate_evidence_sha256(
        workspace,
        outer_family_id=outer,
        path=path,
        alpha=-alpha,
        selection_artifact_sha256=selection_sha,
    )
    negative, negative_receipt = _microstep_provider(
        workspace,
        path=path,
        alpha=-alpha,
        role_evidence=negative_evidence,
    )
    flags = {
        "baseline": _v20a._runtime_flags(workspace.base_provider, workspace.training_records),
        "positive": _v20a._runtime_flags(positive, workspace.training_records),
        "mirror": _v20a._runtime_flags(negative, workspace.training_records),
    }
    if any(
        value.get("finite") is not True
        or value.get("pointwise_trust_passed") is not True
        or value.get("rank_is_16") is not True
        for value in flags.values()
    ):
        raise RuntimeError("V20b outer frozen provider failed finite/trust/rank")
    receipt = {
        "outer_held_family_id": outer,
        "selection_artifact_sha256": selection_sha,
        "selected_path": path,
        "selected_alpha": alpha,
        "outer_fit_receipt": workspace.fit_receipt,
        "fit_training_evidence": workspace.fit_training_evidence,
        "base_provider_artifact_sha256": workspace.base_provider.artifact_sha256,
        "proposal_provider_artifact_sha256": workspace.proposal_provider.artifact_sha256,
        "positive_provider_artifact_sha256": positive.artifact_sha256,
        "positive_microstep_receipt_sha256": positive_receipt["artifact_sha256"],
        "positive_microstep_receipt": positive_receipt,
        "mirror_provider_artifact_sha256": negative.artifact_sha256,
        "mirror_microstep_receipt_sha256": negative_receipt["artifact_sha256"],
        "mirror_microstep_receipt": negative_receipt,
        "runtime_flags": flags,
        "v20a_base_endpoint_sha256": endpoint["base_provider_artifact_sha256"],
        "v20a_proposal_endpoint_sha256": endpoint[
            "proposal_provider_artifact_sha256"
        ],
        "outer_scoring_capability_created": False,
    }
    receipt["artifact_sha256"] = _v14._sha256(
        receipt, domain=_FULL_FIT_DOMAIN
    )
    return workspace, receipt, positive, negative


_SELECTION_LOCK_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "v20a_report_sha256",
        "v20a_file_sha256",
        "panel_binding_sha256",
        "bridge_binding_sha256",
        "pair_fragment_sha256s",
        "inner_fragment_sha256s",
        "selection_receipt",
        "outer_refits",
        "outer_schedule_authorized",
        "resume_overhead",
        "candidate",
        "provider_sidecar",
        "selection_lock_sha256",
    }
)


def _selection_gate_passed(selection_receipt: Mapping[str, object]) -> bool:
    for key in (
        "passed",
        "inner_selection_passed",
        "outer_scoring_authorized",
    ):
        value = selection_receipt.get(key)
        if type(value) is bool:
            return bool(value)
    # A strict fallback for early core revisions: every outer selection must
    # name a positive policy and pass its local directional gate.
    rows = _sequence(
        selection_receipt.get("outer_selections"), label="selection gate rows"
    )
    return len(rows) == _EXPECTED_FAMILIES and all(
        isinstance(row, Mapping)
        and row.get("selected") is not None
        and row.get("passed") is True
        for row in rows
    )


def _publish_selection_lock(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    bridge_binding_sha256: str,
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    selection_receipt: Mapping[str, object],
    outer_refits: Sequence[Mapping[str, object]],
    pair_endpoint_reconstruction_count: int,
) -> dict[str, object]:
    authorized = _selection_gate_passed(selection_receipt)
    if authorized and len(outer_refits) != _EXPECTED_FAMILIES:
        raise ValueError("passing V20b selection must freeze all eight outer refits")
    if not authorized and outer_refits:
        raise ValueError("failed V20b selection cannot freeze partial outer refits")
    payload = {
        "schema": _SELECTION_LOCK_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        "v20a_report_sha256": _V20A_LOGICAL_SHA256,
        "v20a_file_sha256": _V20A_FILE_SHA256,
        "panel_binding_sha256": _sha(
            panel_binding_sha256, label="selection-lock panel binding"
        ),
        "bridge_binding_sha256": _sha(
            bridge_binding_sha256, label="selection-lock bridge"
        ),
        "pair_fragment_sha256s": {
            key: fragment["fragment_sha256"]
            for key, fragment in sorted(pair_fragments.items())
        },
        "inner_fragment_sha256s": {
            key: fragment["fragment_sha256"]
            for key, fragment in sorted(inner_fragments.items())
        },
        "selection_receipt": dict(selection_receipt),
        "outer_refits": tuple(dict(row) for row in outer_refits),
        "outer_schedule_authorized": authorized,
        "resume_overhead": _resume_overhead(
            pair_endpoint_reconstruction_count=pair_endpoint_reconstruction_count,
            outer_endpoint_reconstruction_count=0,
        ),
        "candidate": None,
        "provider_sidecar": None,
    }
    return _publish_scalar_fragment(
        payload,
        path=_selection_lock_path(output),
        domain=_SELECTION_LOCK_DOMAIN,
        hash_key="selection_lock_sha256",
        label="V20b selection lock",
    )


def _authenticated_v20a_endpoint_map(
    authenticated_v20a_folds: Mapping[str, Mapping[str, object]],
    *,
    panel_receipt: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    families = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="authenticated V20a endpoint panel",
            )
        )
    )
    if (
        len(families) != _EXPECTED_FAMILIES
        or set(authenticated_v20a_folds) != set(families)
    ):
        raise ValueError("authenticated V20a endpoint family geometry differs")
    result: dict[str, dict[str, str]] = {}
    for family in families:
        fold = _mapping(
            authenticated_v20a_folds.get(family),
            label="authenticated V20a fold",
        )
        if fold.get("held_family_id") != family:
            raise ValueError("authenticated V20a endpoint fold ownership differs")
        endpoint = _mapping(
            fold.get("endpoint_binding"),
            label="authenticated V20a endpoint binding",
        )
        if endpoint.get("held_family_id") != family:
            raise ValueError("authenticated V20a endpoint ownership differs")
        result[family] = {
            "base_provider_artifact_sha256": _sha(
                endpoint.get("base_provider_artifact_sha256"),
                label="authenticated V20a base endpoint",
            ),
            "proposal_provider_artifact_sha256": _sha(
                endpoint.get("proposal_provider_artifact_sha256"),
                label="authenticated V20a proposal endpoint",
            ),
        }
    return result


def _validate_outer_refit_receipt(
    value: object,
    *,
    selection_item: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    authenticated_v20a_endpoint: Mapping[str, object],
) -> dict[str, object]:
    row = _mapping(value, label="V20b outer refit")
    expected = {
        "outer_held_family_id",
        "selection_artifact_sha256",
        "selected_path",
        "selected_alpha",
        "outer_fit_receipt",
        "fit_training_evidence",
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "positive_provider_artifact_sha256",
        "positive_microstep_receipt_sha256",
        "positive_microstep_receipt",
        "mirror_provider_artifact_sha256",
        "mirror_microstep_receipt_sha256",
        "mirror_microstep_receipt",
        "runtime_flags",
        "v20a_base_endpoint_sha256",
        "v20a_proposal_endpoint_sha256",
        "outer_scoring_capability_created",
        "artifact_sha256",
    }
    if set(row) != expected:
        raise ValueError("V20b outer refit fields differ")
    outer = _identifier(row.get("outer_held_family_id"), label="outer refit family")
    if outer != selection_item.get("outer_held_family_id"):
        raise ValueError("V20b outer refit selection ownership differs")
    if row.get("selection_artifact_sha256") != selection_item.get("artifact_sha256"):
        raise ValueError("V20b outer refit selection binding differs")
    path = str(row.get("selected_path"))
    alpha = _finite(row.get("selected_alpha"), label="outer refit alpha")
    if path not in _PATH_ORDER or alpha not in _POSITIVE_ALPHAS:
        raise ValueError("V20b outer refit policy differs")
    selected_policy = _mapping(
        selection_item.get("selected"), label="outer selected policy"
    )
    selection_path = _identifier(
        selected_policy.get("path"), label="outer selection path"
    )
    selection_alpha = _finite(
        selected_policy.get("alpha"), label="outer selection alpha"
    )
    if path != selection_path or alpha != selection_alpha:
        raise ValueError("V20b outer refit policy differs from frozen selection")
    if "key" in selected_policy and selected_policy.get(
        "key"
    ) != _core.nested_microstep_candidate_key(path, alpha):
        raise ValueError("V20b outer refit key differs from frozen selection")

    authenticated_base = _sha(
        authenticated_v20a_endpoint.get("base_provider_artifact_sha256"),
        label="authenticated V20a outer base endpoint",
    )
    authenticated_proposal = _sha(
        authenticated_v20a_endpoint.get("proposal_provider_artifact_sha256"),
        label="authenticated V20a outer proposal endpoint",
    )
    if (
        row.get("base_provider_artifact_sha256") != authenticated_base
        or row.get("v20a_base_endpoint_sha256") != authenticated_base
        or row.get("proposal_provider_artifact_sha256") != authenticated_proposal
        or row.get("v20a_proposal_endpoint_sha256") != authenticated_proposal
    ):
        raise ValueError("V20b outer refit differs from authenticated V20a endpoints")
    fit = _core.validate_nested_microstep_fit_receipt(
        _mapping(row.get("outer_fit_receipt"), label="outer fit receipt"),
        panel_receipt=panel_receipt,
    )
    if (
        fit.get("kind") != "outer_full"
        or tuple(fit.get("excluded_family_ids", ())) != (outer,)
        or fit.get("base_provider_artifact_sha256")
        != row.get("base_provider_artifact_sha256")
        or fit.get("proposal_provider_artifact_sha256")
        != row.get("proposal_provider_artifact_sha256")
    ):
        raise ValueError("V20b outer fit provider binding differs")
    fit_training_evidence = _validate_fit_training_evidence(
        row.get("fit_training_evidence"),
        fit_receipt=fit,
    )
    for key in (
        "base_provider_artifact_sha256",
        "proposal_provider_artifact_sha256",
        "positive_provider_artifact_sha256",
        "positive_microstep_receipt_sha256",
        "mirror_provider_artifact_sha256",
        "mirror_microstep_receipt_sha256",
        "v20a_base_endpoint_sha256",
        "v20a_proposal_endpoint_sha256",
    ):
        _sha(row.get(key), label=f"outer refit {key}")
    if (
        row.get("base_provider_artifact_sha256")
        != row.get("v20a_base_endpoint_sha256")
        or row.get("proposal_provider_artifact_sha256")
        != row.get("v20a_proposal_endpoint_sha256")
        or row.get("outer_scoring_capability_created") is not False
    ):
        raise ValueError("V20b outer refit V20a or phase binding differs")
    for prefix, signed_alpha in (("positive", alpha), ("mirror", -alpha)):
        receipt = FisherFiniteMicrostepReceipt.from_metadata(
            _mapping(
                row.get(f"{prefix}_microstep_receipt"),
                label=f"outer {prefix} microstep receipt",
            )
        )
        expected_evidence = _outer_candidate_evidence_from_fit(
            fit,
            outer_family_id=outer,
            path=path,
            alpha=signed_alpha,
            selection_artifact_sha256=str(row["selection_artifact_sha256"]),
        )
        if (
            receipt.base_provider_artifact_sha256
            != row.get("base_provider_artifact_sha256")
            or receipt.proposal_provider_artifact_sha256
            != row.get("proposal_provider_artifact_sha256")
            or receipt.selected_provider_artifact_sha256
            != row.get(f"{prefix}_provider_artifact_sha256")
            or receipt.microstep_path != path
            or receipt.alpha != signed_alpha
            or receipt.microstep_protocol_sha256
            != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
            or receipt.microstep_evidence_sha256 != expected_evidence
            or receipt.artifact_sha256
            != row.get(f"{prefix}_microstep_receipt_sha256")
            or receipt.rank != _PARENT_RANK
            or receipt.conditional_rank != _CONDITIONAL_RANK
        ):
            raise ValueError(f"V20b outer {prefix} microstep binding differs")
    flags = _mapping(row.get("runtime_flags"), label="outer runtime flags")
    if set(flags) != {"baseline", "positive", "mirror"}:
        raise ValueError("V20b outer refit runtime flag geometry differs")
    for item in flags.values():
        selected = _mapping(item, label="outer runtime flag")
        if any(
            selected.get(key) is not True
            for key in ("finite", "pointwise_trust_passed", "rank_is_16")
        ):
            raise ValueError("V20b frozen outer provider failed runtime gates")
    payload = {key: item for key, item in row.items() if key != "artifact_sha256"}
    if _v14._sha256(payload, domain=_FULL_FIT_DOMAIN) != _sha(
        row.get("artifact_sha256"), label="outer refit artifact"
    ):
        raise ValueError("V20b outer refit hash drifted")
    return {**row, "outer_fit_receipt": fit, "fit_training_evidence": fit_training_evidence}


def _load_selection_lock(
    *,
    output: Path | str,
    panel_binding_sha256: str | None,
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    panel_receipt: Mapping[str, object],
    authenticated_v20a_folds: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = _load_scalar_fragment(
        path=_selection_lock_path(output),
        domain=_SELECTION_LOCK_DOMAIN,
        hash_key="selection_lock_sha256",
        label="V20b selection lock",
    )
    if set(value) != _SELECTION_LOCK_KEYS:
        raise ValueError("V20b selection-lock fields differ")
    binding = _sha(value.get("panel_binding_sha256"), label="selection panel binding")
    if panel_binding_sha256 is not None and binding != panel_binding_sha256:
        raise ValueError("V20b selection-lock live panel binding differs")
    if (
        value.get("schema") != _SELECTION_LOCK_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("target_output") != _validate_output(output).as_posix()
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256") != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
        or value.get("v20a_report_sha256") != _V20A_LOGICAL_SHA256
        or value.get("v20a_file_sha256") != _V20A_FILE_SHA256
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
    ):
        raise ValueError("V20b selection-lock authority differs")
    lock_overhead = _mapping(
        value.get("resume_overhead"), label="selection-lock resume overhead"
    )
    if (
        lock_overhead.get("outer_endpoint_reconstruction_count") != 0
        or lock_overhead.get("excluded_from_canonical_scientific_work") is not True
    ):
        raise ValueError("V20b selection-lock resume overhead differs")
    rebuilt_lock_overhead = _resume_overhead(
        pair_endpoint_reconstruction_count=int(
            lock_overhead.get("pair_endpoint_reconstruction_count", -1)
        ),
        outer_endpoint_reconstruction_count=0,
    )
    if _v14._canonical_json_bytes(rebuilt_lock_overhead) != _v14._canonical_json_bytes(
        lock_overhead
    ):
        raise ValueError("V20b selection-lock resume overhead arithmetic drifted")
    expected_pair_hashes = {
        key: fragment["fragment_sha256"]
        for key, fragment in sorted(pair_fragments.items())
    }
    expected_inner_hashes = {
        key: fragment["fragment_sha256"]
        for key, fragment in sorted(inner_fragments.items())
    }
    if (
        dict(_mapping(value.get("pair_fragment_sha256s"), label="selection pair hashes"))
        != expected_pair_hashes
        or dict(
            _mapping(value.get("inner_fragment_sha256s"), label="selection inner hashes")
        )
        != expected_inner_hashes
    ):
        raise ValueError("V20b selection-lock fragment binding differs")
    rebuilt_selection = _build_selection_receipt(
        panel_receipt=panel_receipt,
        pair_fragments=pair_fragments,
        inner_fragments=inner_fragments,
    )
    if _v14._canonical_json_bytes(rebuilt_selection) != _v14._canonical_json_bytes(
        value.get("selection_receipt")
    ):
        raise ValueError("V20b selection-lock core selection drifted")
    authorized = _selection_gate_passed(rebuilt_selection)
    if value.get("outer_schedule_authorized") is not authorized:
        raise ValueError("V20b selection-lock outer authorization drifted")
    refits = _sequence(value.get("outer_refits"), label="selection outer refits")
    selections = _selection_item_by_outer(rebuilt_selection)
    authenticated_endpoints = _authenticated_v20a_endpoint_map(
        authenticated_v20a_folds,
        panel_receipt=panel_receipt,
    )
    if authorized:
        if len(refits) != _EXPECTED_FAMILIES:
            raise ValueError("V20b selection lock omitted an outer refit")
        validated = tuple(
            _validate_outer_refit_receipt(
                row,
                selection_item=selections[
                    _identifier(row.get("outer_held_family_id"), label="outer refit")
                ],
                panel_receipt=panel_receipt,
                authenticated_v20a_endpoint=authenticated_endpoints[
                    _identifier(
                        row.get("outer_held_family_id"), label="outer refit endpoint"
                    )
                ],
            )
            for row in refits
            if isinstance(row, Mapping)
        )
        if len({row["outer_held_family_id"] for row in validated}) != 8:
            raise ValueError("V20b outer refit ownership differs")
    else:
        if refits:
            raise ValueError("failed V20b selection lock contains outer refits")
        validated = ()
    return {
        **value,
        "selection_receipt": rebuilt_selection,
        "outer_refits": validated,
    }


def _outer_refit_map(
    selection_lock: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    return {
        _identifier(row.get("outer_held_family_id"), label="outer refit map"): row
        for row in _sequence(
            selection_lock.get("outer_refits"), label="selection outer refits"
        )
        if isinstance(row, Mapping)
    }


def _outer_capability_after_selection_lock(
    teacher_vault: object,
    authorized_example_ids: Sequence[str],
    *,
    output: Path | str,
    expected_selection_lock: Mapping[str, object],
    expected_outer_family_id: str,
    expected_refit_receipt: Mapping[str, object],
    panel_binding_sha256: str,
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    panel_receipt: Mapping[str, object],
    authenticated_v20a_folds: Mapping[str, Mapping[str, object]],
) -> object:
    """Reauthenticate the complete lock, then issue one outer capability."""

    disk_lock = _load_selection_lock(
        output=output,
        panel_binding_sha256=panel_binding_sha256,
        pair_fragments=pair_fragments,
        inner_fragments=inner_fragments,
        panel_receipt=panel_receipt,
        authenticated_v20a_folds=authenticated_v20a_folds,
    )
    expected_lock_sha = _sha(
        expected_selection_lock.get("selection_lock_sha256"),
        label="expected V20b selection lock",
    )
    if disk_lock.get("selection_lock_sha256") != expected_lock_sha:
        raise ValueError("V20b selection lock was replaced before outer capability")
    if disk_lock.get("outer_schedule_authorized") is not True:
        raise ValueError("V20b selection lock does not authorize outer scoring")

    outer = _identifier(expected_outer_family_id, label="expected outer family")
    family_prompts = _mapping(
        panel_receipt.get("family_prompt_sha256s"),
        label="outer capability panel families",
    )
    if len(family_prompts) != _EXPECTED_FAMILIES or outer not in family_prompts:
        raise ValueError("V20b outer capability panel geometry differs")
    refits = _outer_refit_map(disk_lock)
    if set(refits) != set(family_prompts):
        raise ValueError("V20b outer capability requires all eight locked refits")
    disk_refit = refits[outer]
    if _v14._canonical_json_bytes(disk_refit) != _v14._canonical_json_bytes(
        expected_refit_receipt
    ):
        raise ValueError("V20b outer capability refit differs from frozen expectation")
    if _v14._canonical_json_bytes(disk_lock) != _v14._canonical_json_bytes(
        expected_selection_lock
    ):
        raise ValueError("V20b selection lock contents changed before outer capability")

    return teacher_vault.capability(
        tuple(authorized_example_ids),
        held_family_id=None,
    )


def _score_outer_family(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    outer_family_id: str,
    workspace: _EndpointWorkspace,
    positive_provider: object,
    mirror_provider: object,
    refit_receipt: Mapping[str, object],
    selection_item: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    output: Path | str,
    selection_lock: Mapping[str, object],
    panel_binding_sha256: str,
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    authenticated_v20a_folds: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Create the outer-row capability only after the selection lock exists."""

    outer = _identifier(outer_family_id, label="outer score family")
    outer_records = _ordered_records(
        tuple(record for record in records if record.sequence.family_id == outer)
    )
    capability = _outer_capability_after_selection_lock(
        teacher_vault,
        tuple(record.sequence.example_id for record in outer_records),
        output=output,
        expected_selection_lock=selection_lock,
        expected_outer_family_id=outer,
        expected_refit_receipt=refit_receipt,
        panel_binding_sha256=panel_binding_sha256,
        pair_fragments=pair_fragments,
        inner_fragments=inner_fragments,
        panel_receipt=panel_receipt,
        authenticated_v20a_folds=authenticated_v20a_folds,
    )
    fit_sha = _sha(
        _mapping(refit_receipt.get("outer_fit_receipt"), label="outer fit").get(
            "artifact_sha256"
        ),
        label="outer fit receipt",
    )
    baseline = _score_provider(
        context,
        outer_records,
        capability,
        provider=workspace.base_provider,
        fit_receipt_sha256=fit_sha,
        baseline=None,
        path=None,
        alpha=None,
        microstep_receipt=None,
    )
    path = str(refit_receipt["selected_path"])
    alpha = float(refit_receipt["selected_alpha"])
    positive = _score_provider(
        context,
        outer_records,
        capability,
        provider=positive_provider,
        fit_receipt_sha256=fit_sha,
        baseline=baseline,
        path=path,
        alpha=alpha,
        microstep_receipt=_mapping(
            refit_receipt.get("positive_microstep_receipt"),
            label="outer positive microstep receipt",
        ),
    )
    mirror = _score_provider(
        context,
        outer_records,
        capability,
        provider=mirror_provider,
        fit_receipt_sha256=fit_sha,
        baseline=baseline,
        path=path,
        alpha=-alpha,
        microstep_receipt=_mapping(
            refit_receipt.get("mirror_microstep_receipt"),
            label="outer mirror microstep receipt",
        ),
    )
    outer_score = _core.build_nested_microstep_outer_score(
        panel_receipt=panel_receipt,
        selection=selection_item,
        full_fit_receipt=_mapping(
            refit_receipt.get("outer_fit_receipt"), label="outer fit receipt"
        ),
        baseline=baseline.summary,
        selected_positive=positive.summary,
        matched_negative=mirror.summary,
    )
    receipt = capability.receipt()
    if (
        receipt.get("held_family_id") is not None
        or receipt.get("authorized_family_count") != 1
        or receipt.get("authorized_example_count") != _PROMPTS_PER_FAMILY
        or receipt.get("access_count") != 3 * _PROMPTS_PER_FAMILY
        or any(count != 3 for count in receipt["per_example_access_counts"].values())
    ):
        raise RuntimeError("V20b outer capability schedule differs")
    return (
        outer_score,
        receipt,
        {
            "baseline": _score_evidence_payload(baseline),
            "selected_positive": _score_evidence_payload(positive),
            "matched_negative": _score_evidence_payload(mirror),
        },
    )


_OUTER_FRAGMENT_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "v20a_report_sha256",
        "v20a_file_sha256",
        "selection_lock_sha256",
        "outer_held_family_id",
        "outer_refit_artifact_sha256",
        "outer_score",
        "score_evidence",
        "capability_receipt",
        "fixed_schedule_completed",
        "resume_outer_endpoint_reconstructed",
        "candidate",
        "provider_sidecar",
        "fragment_sha256",
    }
)


def _publish_outer_fragment(
    *,
    output: Path | str,
    selection_lock: Mapping[str, object],
    outer_family_id: str,
    refit_receipt: Mapping[str, object],
    outer_score: Mapping[str, object],
    score_evidence: Mapping[str, object],
    capability_receipt: Mapping[str, object],
    resume_outer_endpoint_reconstructed: bool,
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="outer fragment family")
    payload = {
        "schema": _OUTER_FRAGMENT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": _validate_output(output).as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        "v20a_report_sha256": _V20A_LOGICAL_SHA256,
        "v20a_file_sha256": _V20A_FILE_SHA256,
        "selection_lock_sha256": _sha(
            selection_lock.get("selection_lock_sha256"), label="outer selection lock"
        ),
        "outer_held_family_id": outer,
        "outer_refit_artifact_sha256": _sha(
            refit_receipt.get("artifact_sha256"), label="outer refit artifact"
        ),
        "outer_score": dict(outer_score),
        "score_evidence": dict(score_evidence),
        "capability_receipt": dict(capability_receipt),
        "fixed_schedule_completed": True,
        "resume_outer_endpoint_reconstructed": bool(
            resume_outer_endpoint_reconstructed
        ),
        "candidate": None,
        "provider_sidecar": None,
    }
    return _publish_scalar_fragment(
        payload,
        path=_outer_fragment_path(output, outer),
        domain=_OUTER_FRAGMENT_DOMAIN,
        hash_key="fragment_sha256",
        label="V20b outer-score fragment",
    )


def _load_outer_fragment(
    *,
    output: Path | str,
    selection_lock: Mapping[str, object],
    outer_family_id: str,
    panel_receipt: Mapping[str, object],
) -> dict[str, object]:
    outer = _identifier(outer_family_id, label="outer fragment family")
    value = _load_scalar_fragment(
        path=_outer_fragment_path(output, outer),
        domain=_OUTER_FRAGMENT_DOMAIN,
        hash_key="fragment_sha256",
        label="V20b outer-score fragment",
    )
    if set(value) != _OUTER_FRAGMENT_KEYS:
        raise ValueError("V20b outer-score fragment fields differ")
    if (
        value.get("schema") != _OUTER_FRAGMENT_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("target_output") != _validate_output(output).as_posix()
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256") != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
        or value.get("v20a_report_sha256") != _V20A_LOGICAL_SHA256
        or value.get("v20a_file_sha256") != _V20A_FILE_SHA256
        or value.get("selection_lock_sha256")
        != selection_lock.get("selection_lock_sha256")
        or value.get("outer_held_family_id") != outer
        or value.get("fixed_schedule_completed") is not True
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
    ):
        raise ValueError("V20b outer-score fragment authority differs")
    if type(value.get("resume_outer_endpoint_reconstructed")) is not bool:
        raise TypeError("V20b outer resume reconstruction flag differs")
    refits = _outer_refit_map(selection_lock)
    refit = refits.get(outer)
    if (
        refit is None
        or value.get("outer_refit_artifact_sha256") != refit.get("artifact_sha256")
    ):
        raise ValueError("V20b outer-score refit binding differs")
    selections = _selection_item_by_outer(
        _mapping(selection_lock.get("selection_receipt"), label="selection receipt")
    )
    rebuilt = _core.build_nested_microstep_outer_score(
        panel_receipt=panel_receipt,
        selection=selections[outer],
        full_fit_receipt=_mapping(
            refit.get("outer_fit_receipt"), label="outer fit receipt"
        ),
        baseline=_mapping(
            _mapping(value.get("outer_score"), label="outer score").get("baseline"),
            label="outer baseline",
        ),
        selected_positive=_mapping(
            _mapping(value.get("outer_score"), label="outer score").get(
                "selected_positive"
            ),
            label="outer positive",
        ),
        matched_negative=_mapping(
            _mapping(value.get("outer_score"), label="outer score").get(
                "matched_negative"
            ),
            label="outer mirror",
        ),
    )
    if _v14._canonical_json_bytes(rebuilt) != _v14._canonical_json_bytes(
        value.get("outer_score")
    ):
        raise ValueError("V20b outer score did not rebuild")
    evidence = _mapping(value.get("score_evidence"), label="outer score evidence")
    if set(evidence) != {"baseline", "selected_positive", "matched_negative"}:
        raise ValueError("V20b outer score evidence fields differ")
    fit = _mapping(refit.get("outer_fit_receipt"), label="outer evidence fit")
    baseline_evidence = _validate_score_evidence(
        evidence.get("baseline"),
        fit_receipt=fit,
        scored_family_id=outer,
        baseline=None,
        expected_microstep_evidence_sha256=None,
    )
    path = str(refit["selected_path"])
    alpha = float(refit["selected_alpha"])
    positive_evidence = _validate_score_evidence(
        evidence.get("selected_positive"),
        fit_receipt=fit,
        scored_family_id=outer,
        baseline=baseline_evidence,
        expected_microstep_evidence_sha256=_outer_candidate_evidence_from_fit(
            fit,
            outer_family_id=outer,
            path=path,
            alpha=alpha,
            selection_artifact_sha256=str(refit["selection_artifact_sha256"]),
        ),
    )
    mirror_evidence = _validate_score_evidence(
        evidence.get("matched_negative"),
        fit_receipt=fit,
        scored_family_id=outer,
        baseline=baseline_evidence,
        expected_microstep_evidence_sha256=_outer_candidate_evidence_from_fit(
            fit,
            outer_family_id=outer,
            path=path,
            alpha=-alpha,
            selection_artifact_sha256=str(refit["selection_artifact_sha256"]),
        ),
    )
    for name, score_evidence in (
        ("baseline", baseline_evidence),
        ("selected_positive", positive_evidence),
        ("matched_negative", mirror_evidence),
    ):
        if _v14._canonical_json_bytes(score_evidence["score"]) != _v14._canonical_json_bytes(
            rebuilt[name]
        ):
            raise ValueError("V20b outer score evidence binding differs")
    capability = _validate_capability_receipt(
        value.get("capability_receipt"),
        expected_example_ids=tuple(
            _mapping(
                baseline_evidence.get("post_cast_h4_sha256s"),
                label="outer baseline H4",
            )
        ),
        expected_family_count=1,
        expected_held_family_id=None,
        expected_accesses_per_example=3,
        label="outer capability receipt",
    )
    return {
        **value,
        "outer_score": rebuilt,
        "score_evidence": {
            "baseline": baseline_evidence,
            "selected_positive": positive_evidence,
            "matched_negative": mirror_evidence,
        },
        "capability_receipt": capability,
    }


def _authenticated_teacher_access_accounting(
    *,
    prerequisite: Mapping[str, object],
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    selection_lock: Mapping[str, object],
    outer_fragments: Mapping[str, Mapping[str, object]],
    work_accounting: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild every claimed teacher access from persisted execution evidence."""

    collection = _mapping(
        prerequisite.get("authenticated_fit_collection"),
        label="access accounting fit collection",
    )
    vault = _mapping(
        collection.get("teacher_vault_receipt"),
        label="access accounting teacher vault",
    )
    teacher_hashes = _raw_hash_mapping(
        vault.get("teacher_row_sha256s"), label="access accounting teacher rows"
    )
    example_families: dict[str, str] = {}
    pending_capabilities: list[
        tuple[Mapping[str, object], dict[str, str], str]
    ] = []
    categories = {
        "pair_training": 0,
        "inner_positive_scoring": 0,
        "inner_mirror_scoring": 0,
        "outer_refit_training": 0,
        "outer_scoring": 0,
    }
    h4_count = 0
    logits_count = 0

    def bind_families(values: Mapping[str, str], *, label: str) -> None:
        for example, family in values.items():
            previous = example_families.setdefault(example, family)
            if previous != family:
                raise ValueError(f"{label} family ownership drifted")

    def add(
        *,
        category: str,
        capability: Mapping[str, object],
        families: Mapping[str, str],
        evidence_h4_count: int,
        evidence_logits_count: int,
        label: str,
    ) -> None:
        nonlocal h4_count, logits_count
        access_count = capability.get("access_count")
        if type(access_count) is not int or access_count < 0:
            raise TypeError(f"{label} access count differs")
        if access_count != evidence_h4_count or access_count != evidence_logits_count:
            raise ValueError(f"{label} execution evidence count differs")
        bind_families(families, label=label)
        pending_capabilities.append((capability, dict(families), label))
        categories[category] += access_count
        h4_count += evidence_h4_count
        logits_count += evidence_logits_count

    for fragment in pair_fragments.values():
        fit_evidence = _mapping(
            fragment.get("fit_training_evidence"), label="pair fit evidence"
        )
        fit_families = {
            str(key): str(value)
            for key, value in _mapping(
                fit_evidence.get("example_family_ids"),
                label="pair fit example families",
            ).items()
        }
        add(
            category="pair_training",
            capability=_mapping(
                fit_evidence.get("capability_receipt"),
                label="pair fit capability",
            ),
            families=fit_families,
            evidence_h4_count=len(
                _mapping(
                    fit_evidence.get("post_cast_h4_sha256s"),
                    label="pair fit H4",
                )
            ),
            evidence_logits_count=len(
                _mapping(
                    fit_evidence.get("supervised_full_vocab_logits_sha256s"),
                    label="pair fit logits",
                )
            ),
            label="pair fit capability",
        )
        for panel in _sequence(
            fragment.get("directed_panels"), label="access directed panels"
        ):
            directed = _mapping(panel, label="access directed panel")
            inner = _identifier(
                directed.get("inner_held_family_id"),
                label="access directed inner",
            )
            baseline_ids = tuple(
                _mapping(
                    directed.get("baseline_h4_sha256s"),
                    label="access directed baseline H4",
                )
            )
            evidence_rows = _sequence(
                directed.get("positive_candidate_evidence"),
                label="access directed positives",
            )
            positive_h4 = sum(
                len(
                    _mapping(
                        _mapping(row, label="access positive evidence").get(
                            "post_cast_h4_sha256s"
                        ),
                        label="access positive H4",
                    )
                )
                for row in evidence_rows
            )
            positive_logits = sum(
                len(
                    _mapping(
                        _mapping(row, label="access positive evidence").get(
                            "supervised_full_vocab_logits_sha256s"
                        ),
                        label="access positive logits",
                    )
                )
                for row in evidence_rows
            )
            add(
                category="inner_positive_scoring",
                capability=_mapping(
                    directed.get("capability_receipt"),
                    label="inner positive capability",
                ),
                families={example: inner for example in baseline_ids},
                evidence_h4_count=len(baseline_ids) + positive_h4,
                evidence_logits_count=len(
                    _mapping(
                        directed.get("baseline_logits_sha256s"),
                        label="access directed baseline logits",
                    )
                )
                + positive_logits,
                label="inner positive capability",
            )

    for fragment in inner_fragments.values():
        for raw in _sequence(
            fragment.get("matched_negative_evidence"),
            label="access inner mirrors",
        ):
            record = _mapping(raw, label="access inner mirror")
            inner = _identifier(
                record.get("inner_held_family_id"), label="access mirror inner"
            )
            evidence = _mapping(
                record.get("score_evidence"), label="access mirror score evidence"
            )
            h4 = _mapping(
                evidence.get("post_cast_h4_sha256s"), label="access mirror H4"
            )
            logits = _mapping(
                evidence.get("supervised_full_vocab_logits_sha256s"),
                label="access mirror logits",
            )
            add(
                category="inner_mirror_scoring",
                capability=_mapping(
                    record.get("capability_receipt"),
                    label="inner mirror capability",
                ),
                families={str(example): inner for example in h4},
                evidence_h4_count=len(h4),
                evidence_logits_count=len(logits),
                label="inner mirror capability",
            )

    for raw in _sequence(
        selection_lock.get("outer_refits"), label="access outer refits"
    ):
        refit = _mapping(raw, label="access outer refit")
        fit_evidence = _mapping(
            refit.get("fit_training_evidence"), label="outer refit fit evidence"
        )
        families = {
            str(key): str(value)
            for key, value in _mapping(
                fit_evidence.get("example_family_ids"),
                label="outer refit example families",
            ).items()
        }
        add(
            category="outer_refit_training",
            capability=_mapping(
                fit_evidence.get("capability_receipt"),
                label="outer refit capability",
            ),
            families=families,
            evidence_h4_count=len(
                _mapping(
                    fit_evidence.get("post_cast_h4_sha256s"),
                    label="outer refit H4",
                )
            ),
            evidence_logits_count=len(
                _mapping(
                    fit_evidence.get("supervised_full_vocab_logits_sha256s"),
                    label="outer refit logits",
                )
            ),
            label="outer refit capability",
        )

    for fragment in outer_fragments.values():
        outer = _identifier(
            fragment.get("outer_held_family_id"), label="access outer family"
        )
        score_evidence = _mapping(
            fragment.get("score_evidence"), label="access outer score evidence"
        )
        evidence_rows = tuple(
            _mapping(score_evidence.get(key), label=f"access outer {key}")
            for key in ("baseline", "selected_positive", "matched_negative")
        )
        ids = tuple(
            _mapping(
                evidence_rows[0].get("post_cast_h4_sha256s"),
                label="access outer baseline H4",
            )
        )
        add(
            category="outer_scoring",
            capability=_mapping(
                fragment.get("capability_receipt"),
                label="outer score capability",
            ),
            families={str(example): outer for example in ids},
            evidence_h4_count=sum(
                len(
                    _mapping(
                        row.get("post_cast_h4_sha256s"),
                        label="access outer score H4",
                    )
                )
                for row in evidence_rows
            ),
            evidence_logits_count=sum(
                len(
                    _mapping(
                        row.get("supervised_full_vocab_logits_sha256s"),
                        label="access outer score logits",
                    )
                )
                for row in evidence_rows
            ),
            label="outer score capability",
        )

    if set(example_families) != set(teacher_hashes):
        raise ValueError("V20b teacher evidence does not cover the authenticated vault")
    family_count = len(set(example_families.values()))
    if (
        vault.get("example_count") != len(teacher_hashes)
        or vault.get("family_count") != family_count
        or vault.get("source_rows_cached_in_native_dtype_on_cpu") is not True
        or vault.get("float64_teacher_log_probabilities_or_probabilities_cached")
        is not False
        or _v14._sha256(
            {
                "example_family_ids": dict(sorted(example_families.items())),
                "teacher_row_sha256s": dict(sorted(teacher_hashes.items())),
            },
            domain=_v19._TEACHER_VAULT_DOMAIN,
        )
        != _sha(vault.get("artifact_sha256"), label="authenticated teacher vault")
    ):
        raise ValueError("V20b authenticated teacher vault receipt drifted")
    for capability, families, label in pending_capabilities:
        selected_hashes = {
            example: teacher_hashes[example] for example in sorted(families)
        }
        expected_artifact = _v14._sha256(
            {
                "authorized_example_ids": tuple(sorted(families)),
                "authorized_family_ids": tuple(sorted(set(families.values()))),
                "held_family_id": capability.get("held_family_id"),
                "teacher_row_sha256s": selected_hashes,
            },
            domain=_v19._CAPABILITY_DOMAIN,
        )
        if expected_artifact != capability.get("artifact_sha256"):
            raise ValueError(f"{label} artifact drifted")

    total = sum(categories.values())
    expected_total = work_accounting.get("teacher_capability_access_count")
    if (
        total != expected_total
        or h4_count != work_accounting.get("post_cast_h4_hash_check_count")
        or logits_count
        != work_accounting.get("supervised_full_vocab_logits_hash_check_count")
    ):
        raise ValueError("V20b authenticated teacher-access accounting drifted")
    expected_categories = {
        "pair_training": _PAIR_COUNT * 12,
        "inner_positive_scoring": _INNER_ROLE_COUNT
        * (1 + len(_PATHS) * len(_POSITIVE_ALPHAS))
        * _PROMPTS_PER_FAMILY,
        "inner_mirror_scoring": int(
            work_accounting.get("inner_matched_negative_score_count", -1)
        )
        * _PROMPTS_PER_FAMILY,
        "outer_refit_training": int(
            work_accounting.get("physical_outer_full_fit_count", -1)
        )
        * 14,
        "outer_scoring": (
            int(work_accounting.get("outer_baseline_score_count", -1))
            + int(work_accounting.get("outer_selected_positive_score_count", -1))
            + int(work_accounting.get("outer_matched_negative_score_count", -1))
        )
        * _PROMPTS_PER_FAMILY,
    }
    if categories != expected_categories:
        raise ValueError("V20b teacher-access phase accounting drifted")
    receipt: dict[str, object] = {
        "teacher_vault_artifact_sha256": vault["artifact_sha256"],
        "capability_receipt_count": len(pending_capabilities),
        "phase_access_counts": categories,
        "teacher_capability_access_count": total,
        "post_cast_h4_hash_check_count": h4_count,
        "supervised_full_vocab_logits_hash_check_count": logits_count,
        "all_capability_artifacts_recomputed": True,
        "all_accesses_bound_to_execution_hash_evidence": True,
        "raw_tensors_or_logits_serialized": False,
    }
    receipt["artifact_sha256"] = _v14._sha256(
        receipt,
        domain=_ACCESS_ACCOUNTING_DOMAIN,
    )
    return receipt


def _resume_overhead(
    *,
    pair_endpoint_reconstruction_count: int,
    outer_endpoint_reconstruction_count: int,
) -> dict[str, object]:
    for label, value in (
        ("pair", pair_endpoint_reconstruction_count),
        ("outer", outer_endpoint_reconstruction_count),
    ):
        if type(value) is not int or value < 0:
            raise TypeError(f"V20b {label} reconstruction count differs")
    pair_prompts = pair_endpoint_reconstruction_count * 12
    outer_prompts = outer_endpoint_reconstruction_count * 14
    return {
        "excluded_from_canonical_scientific_work": True,
        "pair_endpoint_reconstruction_count": pair_endpoint_reconstruction_count,
        "outer_endpoint_reconstruction_count": outer_endpoint_reconstruction_count,
        "extra_full_model_forward_count": pair_prompts + outer_prompts,
        "extra_full_suffix_backward_traversal_count": pair_prompts + outer_prompts,
        "extra_local_head_autograd_contraction_count": pair_prompts + outer_prompts,
        "extra_teacher_capability_access_count": pair_prompts + outer_prompts,
        "extra_post_cast_h4_hash_check_count": pair_prompts + outer_prompts,
        "extra_supervised_full_vocab_logits_hash_check_count": (
            pair_prompts + outer_prompts
        ),
        "reason": "authenticated_resume_provider_reconstruction_only",
    }


def build_finite_microstep_nested_validation_report(
    *,
    artifact_path: Path | str,
    prerequisite: Mapping[str, object],
    panel_receipt: Mapping[str, object],
    panel_binding_sha256: str,
    bridge_binding_sha256: str,
    pair_fragments: Sequence[Mapping[str, object]],
    inner_fragments: Sequence[Mapping[str, object]],
    selection_lock: Mapping[str, object],
    outer_fragments: Sequence[Mapping[str, object]],
    resume_overhead: Mapping[str, object],
    integrity: Mapping[str, object],
) -> dict[str, object]:
    """Independently rebuild the scalar/hash-only V20b result."""

    destination = _validate_output(artifact_path)
    authenticated, _payload, authenticated_v20a_folds = (
        _load_authenticated_v20a_artifact()
    )
    if _v14._canonical_json_bytes(authenticated) != _v14._canonical_json_bytes(
        prerequisite
    ):
        raise ValueError("V20b prerequisite differs from authenticated V20a")
    expected_panel = _mapping(
        authenticated.get("nested_panel_receipt"), label="authenticated nested panel"
    )
    if _v14._canonical_json_bytes(expected_panel) != _v14._canonical_json_bytes(
        panel_receipt
    ):
        raise ValueError("V20b nested panel receipt differs")
    bridge = _sha(bridge_binding_sha256, label="V20b report bridge")
    if bridge != authenticated.get("authenticated_bridge_binding_sha256"):
        raise ValueError("V20b report bridge differs from V20a")
    expected_binding = _panel_binding_sha256(
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge,
    )
    if panel_binding_sha256 != expected_binding:
        raise ValueError("V20b panel binding drifted")
    pairs = tuple(pair_fragments)
    inners = tuple(inner_fragments)
    if len(pairs) != _PAIR_COUNT or len(inners) != _EXPECTED_FAMILIES:
        raise ValueError("V20b report checkpoint geometry differs")
    pair_map = {
        _sha(row.get("pair_key"), label="report pair key"): row for row in pairs
    }
    inner_map = {
        _identifier(row.get("outer_held_family_id"), label="report inner outer"): row
        for row in inners
    }
    if len(pair_map) != _PAIR_COUNT or len(inner_map) != _EXPECTED_FAMILIES:
        raise ValueError("V20b report checkpoint ownership differs")
    families = tuple(
        sorted(
            _mapping(
                panel_receipt.get("family_prompt_sha256s"),
                label="report nested panel families",
            )
        )
    )
    disk_pairs = _pair_fragment_map(
        output=destination,
        panel_binding_sha256=panel_binding_sha256,
        families=families,
        panel_receipt=panel_receipt,
    )
    if _v14._canonical_json_bytes(disk_pairs) != _v14._canonical_json_bytes(pair_map):
        raise ValueError("V20b report pair inputs differ from write-once fragments")
    disk_inners = _inner_fragment_map(
        output=destination,
        panel_binding_sha256=panel_binding_sha256,
        families=families,
        pair_fragments=disk_pairs,
        panel_receipt=panel_receipt,
    )
    if _v14._canonical_json_bytes(disk_inners) != _v14._canonical_json_bytes(inner_map):
        raise ValueError("V20b report inner inputs differ from write-once fragments")
    disk_lock = _load_selection_lock(
        output=destination,
        panel_binding_sha256=panel_binding_sha256,
        pair_fragments=disk_pairs,
        inner_fragments=disk_inners,
        panel_receipt=panel_receipt,
        authenticated_v20a_folds=authenticated_v20a_folds,
    )
    if _v14._canonical_json_bytes(disk_lock) != _v14._canonical_json_bytes(
        selection_lock
    ):
        raise ValueError("V20b report selection input differs from write-once lock")
    disk_outers: dict[str, dict[str, object]] = {}
    for outer in families:
        path = _outer_fragment_path(destination, outer)
        if path.exists():
            disk_outers[outer] = _load_outer_fragment(
                output=destination,
                selection_lock=disk_lock,
                outer_family_id=outer,
                panel_receipt=panel_receipt,
            )
    outer_input_map = {
        _identifier(row.get("outer_held_family_id"), label="report outer family"): row
        for row in tuple(outer_fragments)
    }
    if _v14._canonical_json_bytes(disk_outers) != _v14._canonical_json_bytes(
        outer_input_map
    ):
        raise ValueError("V20b report outer inputs differ from write-once fragments")
    pair_map = disk_pairs
    inner_map = disk_inners
    selection_lock = disk_lock
    # Loaders already perform full reconstruction.  Re-run core reduction here
    # so report classification cannot trust copied scalar decisions.
    rebuilt_selection = _build_selection_receipt(
        panel_receipt=panel_receipt,
        pair_fragments=pair_map,
        inner_fragments=inner_map,
    )
    locked_selection = _mapping(
        selection_lock.get("selection_receipt"), label="locked selection"
    )
    if _v14._canonical_json_bytes(rebuilt_selection) != _v14._canonical_json_bytes(
        locked_selection
    ):
        raise ValueError("V20b report selection-lock decision drifted")
    authorized = _selection_gate_passed(rebuilt_selection)
    outer_rows = tuple(outer_fragments)
    if authorized and len(outer_rows) != _EXPECTED_FAMILIES:
        raise ValueError("passing V20b inner gate requires all outer scores")
    if not authorized and outer_rows:
        raise ValueError("failed V20b inner gate cannot contain outer scores")
    validation = _core.build_nested_microstep_validation_receipt(
        panel_receipt=panel_receipt,
        shared_fit_receipts=tuple(
            _mapping(row.get("shared_fit_receipt"), label="report shared fit")
            for row in pairs
        ),
        selection_receipt=rebuilt_selection,
        outer_scores=tuple(
            _mapping(row.get("outer_score"), label="report outer score")
            for row in outer_rows
        ),
    )
    work = dict(
        _mapping(validation.get("work_accounting"), label="validation work accounting")
    )
    access_accounting = _authenticated_teacher_access_accounting(
        prerequisite=prerequisite,
        pair_fragments=pair_map,
        inner_fragments=inner_map,
        selection_lock=selection_lock,
        outer_fragments=disk_outers,
        work_accounting=work,
    )
    supplied_overhead = dict(resume_overhead)
    expected_overhead_keys = set(
        _resume_overhead(
            pair_endpoint_reconstruction_count=0,
            outer_endpoint_reconstruction_count=0,
        )
    )
    if set(supplied_overhead) != expected_overhead_keys:
        raise ValueError("V20b resume-overhead fields differ")
    recomputed_overhead = _resume_overhead(
        pair_endpoint_reconstruction_count=int(
            supplied_overhead["pair_endpoint_reconstruction_count"]
        ),
        outer_endpoint_reconstruction_count=int(
            supplied_overhead["outer_endpoint_reconstruction_count"]
        ),
    )
    if _v14._canonical_json_bytes(recomputed_overhead) != _v14._canonical_json_bytes(
        supplied_overhead
    ):
        raise ValueError("V20b resume-overhead arithmetic drifted")
    expected_integrity = {
        "v20a_authenticated_and_rebuilt_before_model_work": True,
        "all_28_pair_fragments_authenticated": True,
        "all_8_inner_selection_fragments_authenticated": True,
        "outer_family_absent_from_each_selector": True,
        "selection_lock_published_before_outer_capabilities": True,
        "outer_schedule_all_or_none": True,
        "outer_score_fragment_count": _EXPECTED_FAMILIES if authorized else 0,
        "all_teacher_capability_accesses_authenticated": True,
        "all_execution_hash_checks_recomputed_from_receipts": True,
        "guard_opened": False,
        "calibration_b_opened": False,
        "provider_sidecar_written": False,
    }
    if dict(integrity) != expected_integrity:
        raise ValueError("V20b integrity receipt differs")
    classification = str(validation.get("classification"))
    passed = validation.get("passed") is True
    report: dict[str, object] = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "experiment_stage": "v20b",
        "scientific_status": "true_nested_family_disjoint_finite_microstep_validation",
        "artifact": {
            "path": destination.as_posix(),
            "write_once": True,
            "file_mode": "0600",
            "scalar_and_hash_only": True,
            "provider_tensor_sidecar": False,
        },
        "prerequisite": dict(prerequisite),
        "fixed_protocol": {
            **_FIXED_PROTOCOL,
            "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        },
        "panel_receipt": dict(panel_receipt),
        "panel_binding_sha256": panel_binding_sha256,
        "bridge_binding_sha256": bridge,
        "pair_fragment_sha256s": {
            key: row["fragment_sha256"] for key, row in sorted(pair_map.items())
        },
        "inner_fragment_sha256s": {
            key: row["fragment_sha256"] for key, row in sorted(inner_map.items())
        },
        "selection_lock_sha256": selection_lock["selection_lock_sha256"],
        "selection_receipt": rebuilt_selection,
        "validation_receipt": validation,
        "work_accounting": work,
        "authenticated_teacher_access_accounting": access_accounting,
        "resume_overhead": supplied_overhead,
        "integrity": expected_integrity,
        "classification": classification,
        "passed": passed,
        "candidate": None,
        "provider_sidecar": None,
        "fresh_guard_authorized": passed,
        "calibration_b_authorized": False,
        "held_fidelity_claim": passed,
        "serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "end_to_end_parameter_or_flop_claim": False,
        "success_authorizes": (
            "fresh_family_disjoint_guard_validation_only"
            if passed
            else "no_fresh_guard_or_candidate"
        ),
    }
    _v14._scalar_report(report)
    return report


_REPORT_INPUT_KEYS = frozenset(
    {
        "prerequisite",
        "panel_receipt",
        "panel_binding_sha256",
        "bridge_binding_sha256",
        "pair_fragments",
        "inner_fragments",
        "selection_lock",
        "outer_fragments",
        "resume_overhead",
        "integrity",
    }
)
_REPORT_READY_KEYS = frozenset(
    {
        "schema",
        "format_version",
        "target_output",
        "runner_protocol_sha256",
        "core_protocol_sha256",
        "v20a_report_sha256",
        "v20a_file_sha256",
        "selection_lock_sha256",
        "candidate",
        "provider_sidecar",
        "report_inputs",
        "checkpoint_sha256",
    }
)


def _validate_report_inputs(value: object) -> dict[str, object]:
    inputs = _mapping(value, label="V20b report-ready inputs")
    if set(inputs) != _REPORT_INPUT_KEYS:
        raise ValueError("V20b report-ready input fields differ")
    for key in (
        "prerequisite",
        "panel_receipt",
        "selection_lock",
        "resume_overhead",
        "integrity",
    ):
        _mapping(inputs.get(key), label=f"V20b report-ready {key}")
    pairs = _sequence(inputs.get("pair_fragments"), label="report-ready pairs")
    inners = _sequence(inputs.get("inner_fragments"), label="report-ready inners")
    outers = _sequence(inputs.get("outer_fragments"), label="report-ready outers")
    if len(pairs) != _PAIR_COUNT or len(inners) != _EXPECTED_FAMILIES:
        raise ValueError("V20b report-ready fragment geometry differs")
    if len(outers) not in {0, _EXPECTED_FAMILIES}:
        raise ValueError("V20b report-ready outer schedule is partial")
    _sha(inputs.get("panel_binding_sha256"), label="report-ready panel binding")
    _sha(inputs.get("bridge_binding_sha256"), label="report-ready bridge")
    _v14._scalar_report(dict(inputs))
    return dict(inputs)


def _publish_report_ready_checkpoint(
    *,
    output: Path | str,
    report_inputs: Mapping[str, object],
) -> dict[str, object]:
    destination = _validate_output(output)
    inputs = _validate_report_inputs(report_inputs)
    lock = _mapping(inputs.get("selection_lock"), label="report-ready selection lock")
    payload = {
        "schema": _REPORT_READY_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "target_output": destination.as_posix(),
        "runner_protocol_sha256": _RUNNER_PROTOCOL_SHA256,
        "core_protocol_sha256": _core.NESTED_MICROSTEP_PROTOCOL_SHA256,
        "v20a_report_sha256": _V20A_LOGICAL_SHA256,
        "v20a_file_sha256": _V20A_FILE_SHA256,
        "selection_lock_sha256": _sha(
            lock.get("selection_lock_sha256"), label="report-ready selection lock"
        ),
        "candidate": None,
        "provider_sidecar": None,
        "report_inputs": inputs,
    }
    return _publish_scalar_fragment(
        payload,
        path=_report_ready_path(destination),
        domain=_REPORT_READY_DOMAIN,
        hash_key="checkpoint_sha256",
        label="V20b report-ready checkpoint",
    )


def _load_report_ready_checkpoint(
    *,
    output: Path | str,
) -> dict[str, object]:
    destination = _validate_output(output)
    value = _load_scalar_fragment(
        path=_report_ready_path(destination),
        domain=_REPORT_READY_DOMAIN,
        hash_key="checkpoint_sha256",
        label="V20b report-ready checkpoint",
    )
    if set(value) != _REPORT_READY_KEYS:
        raise ValueError("V20b report-ready checkpoint fields differ")
    if (
        value.get("schema") != _REPORT_READY_SCHEMA
        or value.get("format_version") != _FORMAT_VERSION
        or value.get("target_output") != destination.as_posix()
        or value.get("runner_protocol_sha256") != _RUNNER_PROTOCOL_SHA256
        or value.get("core_protocol_sha256") != _core.NESTED_MICROSTEP_PROTOCOL_SHA256
        or value.get("v20a_report_sha256") != _V20A_LOGICAL_SHA256
        or value.get("v20a_file_sha256") != _V20A_FILE_SHA256
        or value.get("candidate") is not None
        or value.get("provider_sidecar") is not None
    ):
        raise ValueError("V20b report-ready checkpoint authority differs")
    inputs = _validate_report_inputs(value.get("report_inputs"))
    lock = _mapping(inputs.get("selection_lock"), label="report-ready selection lock")
    if value.get("selection_lock_sha256") != lock.get("selection_lock_sha256"):
        raise ValueError("V20b report-ready selection-lock binding differs")
    return inputs


def _build_report_from_inputs(
    *,
    output: Path | str,
    report_inputs: Mapping[str, object],
) -> dict[str, object]:
    return build_finite_microstep_nested_validation_report(
        artifact_path=_validate_output(output),
        **_validate_report_inputs(report_inputs),  # type: ignore[arg-type]
    )


def _publish_report(
    report: Mapping[str, object],
    *,
    output: Path | str,
) -> dict[str, object]:
    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V20b report")
    value = dict(report)
    if value.get("candidate") is not None or value.get("provider_sidecar") is not None:
        raise ValueError("V20b cannot publish a candidate or provider sidecar")
    _v14._scalar_report(value)
    value["report_sha256"] = _v14._sha256(value, domain=_REPORT_DOMAIN)
    reservation = _v14._reserve_outputs((destination,))
    stage: Path | None = None
    try:
        stage = _v14._stage_json(value, destination)
        _secure_stat(stage, label="staged V20b report")
        reservation.publish((stage,))
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)
    published = _secure_stat(destination, label="published V20b report")
    return {
        **value,
        "artifact": {
            **dict(_mapping(value.get("artifact"), label="V20b artifact")),
            "file_sha256": _v14._file_sha256(destination),
            "file_bytes": published.st_size,
        },
    }


def _pair_fragment_map(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    families: Sequence[str],
    panel_receipt: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for excluded in combinations(tuple(sorted(families)), 2):
        key = _core.nested_microstep_fit_pair_key(*excluded)
        result[key] = _load_pair_fragment(
            output=output,
            panel_binding_sha256=panel_binding_sha256,
            excluded_family_ids=excluded,
            panel_receipt=panel_receipt,
        )
    if len(result) != _PAIR_COUNT:
        raise RuntimeError("V20b pair fragment map geometry differs")
    return result


def _inner_fragment_map(
    *,
    output: Path | str,
    panel_binding_sha256: str,
    families: Sequence[str],
    pair_fragments: Mapping[str, Mapping[str, object]],
    panel_receipt: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    result = {
        outer: _load_inner_fragment(
            output=output,
            panel_binding_sha256=panel_binding_sha256,
            outer_family_id=outer,
            pair_fragments=pair_fragments,
            panel_receipt=panel_receipt,
        )
        for outer in sorted(families)
    }
    if len(result) != _EXPECTED_FAMILIES:
        raise RuntimeError("V20b inner fragment map geometry differs")
    return result


def _peek_selection_lock(
    *,
    output: Path | str,
    prerequisite: Mapping[str, object],
) -> tuple[dict[str, object], str, str]:
    value = _load_scalar_fragment(
        path=_selection_lock_path(output),
        domain=_SELECTION_LOCK_DOMAIN,
        hash_key="selection_lock_sha256",
        label="V20b selection lock",
    )
    if set(value) != _SELECTION_LOCK_KEYS:
        raise ValueError("V20b selection-lock fields differ")
    bridge = _sha(value.get("bridge_binding_sha256"), label="selection-lock bridge")
    if bridge != prerequisite.get("authenticated_bridge_binding_sha256"):
        raise ValueError("V20b selection-lock bridge differs from V20a")
    panel_receipt = _mapping(
        prerequisite.get("nested_panel_receipt"), label="nested panel receipt"
    )
    binding = _panel_binding_sha256(
        panel_receipt=panel_receipt,
        bridge_binding_sha256=bridge,
    )
    if value.get("panel_binding_sha256") != binding:
        raise ValueError("V20b selection-lock panel binding differs")
    return value, binding, bridge


def _load_checkpoint_graph(
    *,
    output: Path | str,
    prerequisite: Mapping[str, object],
    authenticated_v20a_folds: Mapping[str, Mapping[str, object]],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, dict[str, object]],
    str,
    str,
]:
    peek, binding, bridge = _peek_selection_lock(
        output=output,
        prerequisite=prerequisite,
    )
    panel = _mapping(
        prerequisite.get("nested_panel_receipt"), label="nested panel receipt"
    )
    families = tuple(
        sorted(
            _mapping(
                panel.get("family_prompt_sha256s"), label="nested panel families"
            )
        )
    )
    pairs = _pair_fragment_map(
        output=output,
        panel_binding_sha256=binding,
        families=families,
        panel_receipt=panel,
    )
    inners = _inner_fragment_map(
        output=output,
        panel_binding_sha256=binding,
        families=families,
        pair_fragments=pairs,
        panel_receipt=panel,
    )
    lock = _load_selection_lock(
        output=output,
        panel_binding_sha256=binding,
        pair_fragments=pairs,
        inner_fragments=inners,
        panel_receipt=panel,
        authenticated_v20a_folds=authenticated_v20a_folds,
    )
    if lock.get("selection_lock_sha256") != peek.get("selection_lock_sha256"):
        raise RuntimeError("V20b selection-lock changed while loading")
    outers: dict[str, dict[str, object]] = {}
    if lock.get("outer_schedule_authorized") is True:
        for outer in families:
            path = _outer_fragment_path(output, outer)
            if path.exists():
                outers[outer] = _load_outer_fragment(
                    output=output,
                    selection_lock=lock,
                    outer_family_id=outer,
                    panel_receipt=panel,
                )
    return pairs, inners, lock, outers, binding, bridge


def _integrity_receipt(*, outer_scored: bool) -> dict[str, object]:
    return {
        "v20a_authenticated_and_rebuilt_before_model_work": True,
        "all_28_pair_fragments_authenticated": True,
        "all_8_inner_selection_fragments_authenticated": True,
        "outer_family_absent_from_each_selector": True,
        "selection_lock_published_before_outer_capabilities": True,
        "outer_schedule_all_or_none": True,
        "outer_score_fragment_count": _EXPECTED_FAMILIES if outer_scored else 0,
        "all_teacher_capability_accesses_authenticated": True,
        "all_execution_hash_checks_recomputed_from_receipts": True,
        "guard_opened": False,
        "calibration_b_opened": False,
        "provider_sidecar_written": False,
    }


def _report_inputs_from_graph(
    *,
    prerequisite: Mapping[str, object],
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    selection_lock: Mapping[str, object],
    outer_fragments: Mapping[str, Mapping[str, object]],
    panel_binding_sha256: str,
    bridge_binding_sha256: str,
) -> dict[str, object]:
    authorized = selection_lock.get("outer_schedule_authorized") is True
    if authorized and len(outer_fragments) != _EXPECTED_FAMILIES:
        raise ValueError("V20b report inputs require the complete outer schedule")
    if not authorized and outer_fragments:
        raise ValueError("V20b failed selection cannot have outer fragments")
    lock_overhead = _mapping(
        selection_lock.get("resume_overhead"), label="selection resume overhead"
    )
    outer_reconstructions = sum(
        int(fragment.get("resume_outer_endpoint_reconstructed") is True)
        for fragment in outer_fragments.values()
    )
    overhead = _resume_overhead(
        pair_endpoint_reconstruction_count=int(
            lock_overhead["pair_endpoint_reconstruction_count"]
        ),
        outer_endpoint_reconstruction_count=outer_reconstructions,
    )
    return {
        "prerequisite": dict(prerequisite),
        "panel_receipt": dict(
            _mapping(
                prerequisite.get("nested_panel_receipt"), label="nested panel receipt"
            )
        ),
        "panel_binding_sha256": panel_binding_sha256,
        "bridge_binding_sha256": bridge_binding_sha256,
        "pair_fragments": tuple(
            pair_fragments[key] for key in sorted(pair_fragments)
        ),
        "inner_fragments": tuple(
            inner_fragments[key] for key in sorted(inner_fragments)
        ),
        "selection_lock": dict(selection_lock),
        "outer_fragments": tuple(
            outer_fragments[key] for key in sorted(outer_fragments)
        ),
        "resume_overhead": overhead,
        "integrity": _integrity_receipt(outer_scored=authorized),
    }


def _finalize_checkpoint_graph(
    *,
    output: Path | str,
    prerequisite: Mapping[str, object],
    pair_fragments: Mapping[str, Mapping[str, object]],
    inner_fragments: Mapping[str, Mapping[str, object]],
    selection_lock: Mapping[str, object],
    outer_fragments: Mapping[str, Mapping[str, object]],
    panel_binding_sha256: str,
    bridge_binding_sha256: str,
) -> dict[str, object]:
    inputs = _report_inputs_from_graph(
        prerequisite=prerequisite,
        pair_fragments=pair_fragments,
        inner_fragments=inner_fragments,
        selection_lock=selection_lock,
        outer_fragments=outer_fragments,
        panel_binding_sha256=panel_binding_sha256,
        bridge_binding_sha256=bridge_binding_sha256,
    )
    checkpoint = _report_ready_path(output)
    if checkpoint.exists():
        serialized = _load_report_ready_checkpoint(output=output)
        if _v14._canonical_json_bytes(serialized) != _v14._canonical_json_bytes(inputs):
            raise ValueError("V20b existing report-ready checkpoint inputs differ")
    else:
        _publish_report_ready_checkpoint(output=output, report_inputs=inputs)
        serialized = _load_report_ready_checkpoint(output=output)
    report = _build_report_from_inputs(output=output, report_inputs=serialized)
    return _publish_report(report, output=output)


def _collect_live_fit_authority(
    context: object,
    *,
    prerequisite: Mapping[str, object],
) -> tuple[tuple[object, ...], object, tuple[str, ...]]:
    records, teacher_vault = _v19._collect_fit_records_and_teacher_vault(context)
    families = _validate_live_authority(
        prerequisite=prerequisite,
        context=context,
        records=records,
    )
    current_collection = {
        "prompt_count": len(records),
        "family_count": len(families),
        "record_receipt_sha256s": {
            record.sequence.example_id: record.receipt_sha256 for record in records
        },
        "teacher_vault_receipt": teacher_vault.receipt(),
        "held_teacher_rows_cached": True,
        "held_teacher_rows_scored": False,
        "full_native_logits_transiently_materialized_then_discarded": True,
        "only_supervised_source_rows_cached_in_native_dtype": True,
        "raw_fit_trace_or_teacher_tensor_serialization": False,
    }
    authenticated = _mapping(
        prerequisite.get("authenticated_fit_collection"),
        label="authenticated V20a fit collection",
    )
    if _v14._canonical_json_bytes(current_collection) != _v14._canonical_json_bytes(
        authenticated
    ):
        raise RuntimeError("live V20b fit collection differs from V20a")
    return tuple(records), teacher_vault, families


def _reconstruct_pair_workspace(
    context: object,
    records: Sequence[object],
    teacher_vault: object,
    *,
    fragment: Mapping[str, object],
    panel_receipt: Mapping[str, object],
) -> _EndpointWorkspace:
    workspace = _fit_endpoint_from_scratch(
        context,
        records,
        teacher_vault,
        excluded_family_ids=tuple(fragment["excluded_family_ids"]),
        panel_receipt=panel_receipt,
        outer_fit=False,
    )
    if _v14._canonical_json_bytes(workspace.fit_receipt) != _v14._canonical_json_bytes(
        fragment.get("shared_fit_receipt")
    ) or _v14._canonical_json_bytes(
        workspace.fit_training_evidence
    ) != _v14._canonical_json_bytes(fragment.get("fit_training_evidence")):
        raise RuntimeError("resumed V20b pair endpoint did not reconstruct")
    return workspace


def _panel_for_role(
    fragment: Mapping[str, object],
    *,
    outer_family_id: str,
    inner_family_id: str,
) -> Mapping[str, object]:
    for panel in _sequence(fragment.get("directed_panels"), label="pair panels"):
        if (
            isinstance(panel, Mapping)
            and panel.get("outer_held_family_id") == outer_family_id
            and panel.get("inner_held_family_id") == inner_family_id
        ):
            return panel
    raise ValueError("V20b pair fragment omitted a directed role")


def run_gemma3_l3_l4_complete_h4_finite_microstep_nested_validation(
    *,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run or safely resume the fixed V20b nested validation."""

    destination = _validate_output(output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite V20b report")

    # This authentication and exact rebuild intentionally precede every path
    # that can create the model context.
    prerequisite, _v20a_payload, v20a_folds = _load_authenticated_v20a_artifact()
    panel_receipt = dict(
        _mapping(
            prerequisite.get("nested_panel_receipt"), label="nested panel receipt"
        )
    )

    if _report_ready_path(destination).exists():
        inputs = _load_report_ready_checkpoint(output=destination)
        report = _build_report_from_inputs(
            output=destination,
            report_inputs=inputs,
        )
        return _publish_report(report, output=destination)

    # A complete checkpoint graph can finalize without Gemma.  A partial
    # outer schedule falls through and resumes only the missing fixed rows.
    if _selection_lock_path(destination).exists():
        pairs, inners, lock, outers, binding, bridge = _load_checkpoint_graph(
            output=destination,
            prerequisite=prerequisite,
            authenticated_v20a_folds=v20a_folds,
        )
        authorized = lock.get("outer_schedule_authorized") is True
        if (authorized and len(outers) == _EXPECTED_FAMILIES) or (
            not authorized and not outers
        ):
            return _finalize_checkpoint_graph(
                output=destination,
                prerequisite=prerequisite,
                pair_fragments=pairs,
                inner_fragments=inners,
                selection_lock=lock,
                outer_fragments=outers,
                panel_binding_sha256=binding,
                bridge_binding_sha256=bridge,
            )

    context = prepare_complete_h4_rank320_live_context(cache_dir=cache_dir)
    try:
        context.validate_immutable_inputs()
        records, teacher_vault, families = _collect_live_fit_authority(
            context,
            prerequisite=prerequisite,
        )
        bridge = _sha(
            context.bridge.bridge_binding_sha256, label="live V20b bridge"
        )
        binding = _panel_binding_sha256(
            panel_receipt=panel_receipt,
            bridge_binding_sha256=bridge,
        )

        pair_fragments: dict[str, dict[str, object]] = {}
        pair_workspaces: dict[str, _EndpointWorkspace] = {}
        pair_reconstruction_count = 0
        for excluded in combinations(families, 2):
            key = _core.nested_microstep_fit_pair_key(*excluded)
            path = _pair_fragment_path(destination, key)
            if path.exists():
                fragment = _load_pair_fragment(
                    output=destination,
                    panel_binding_sha256=binding,
                    excluded_family_ids=excluded,
                    panel_receipt=panel_receipt,
                )
            else:
                workspace = _fit_endpoint_from_scratch(
                    context,
                    records,
                    teacher_vault,
                    excluded_family_ids=excluded,
                    panel_receipt=panel_receipt,
                    outer_fit=False,
                )
                panels = _evaluate_pair_matrix(
                    context,
                    records,
                    teacher_vault,
                    workspace=workspace,
                )
                fragment = _publish_pair_fragment(
                    output=destination,
                    panel_binding_sha256=binding,
                    workspace=workspace,
                    directed_panels=panels,
                )
                fragment = _load_pair_fragment(
                    output=destination,
                    panel_binding_sha256=binding,
                    excluded_family_ids=excluded,
                    panel_receipt=panel_receipt,
                )
                pair_workspaces[key] = workspace
            pair_fragments[key] = fragment
        if len(pair_fragments) != _PAIR_COUNT:
            raise RuntimeError("V20b did not complete all 28 pair matrices")

        inner_fragments: dict[str, dict[str, object]] = {}
        for outer in families:
            inner_path = _inner_fragment_path(destination, outer)
            if inner_path.exists():
                inner_fragments[outer] = _load_inner_fragment(
                    output=destination,
                    panel_binding_sha256=binding,
                    outer_family_id=outer,
                    pair_fragments=pair_fragments,
                    panel_receipt=panel_receipt,
                )
                continue
            relevant = {
                key: fragment
                for key, fragment in pair_fragments.items()
                if outer in tuple(fragment["excluded_family_ids"])
            }
            panels = tuple(
                _panel_for_role(
                    relevant[_core.nested_microstep_fit_pair_key(outer, inner)],
                    outer_family_id=outer,
                    inner_family_id=inner,
                )
                for inner in families
                if inner != outer
            )
            shared_by_sha = {
                str(fragment["shared_fit_receipt"]["artifact_sha256"]): fragment[
                    "shared_fit_receipt"
                ]
                for fragment in pair_fragments.values()
            }
            preview = _select_outer_policy(
                outer_family_id=outer,
                panels=panels,
                panel_receipt=panel_receipt,
                shared_fit_receipts=shared_by_sha,
            )
            selected = preview.get("selected")
            roles: list[dict[str, object]] = []
            mirror_evidence: list[dict[str, object]] = []
            for panel in panels:
                inner = str(panel["inner_held_family_id"])
                key = _core.nested_microstep_fit_pair_key(outer, inner)
                fragment = relevant[key]
                fit = _mapping(
                    fragment.get("shared_fit_receipt"), label="inner shared fit"
                )
                mirror: dict[str, object] | None = None
                if isinstance(selected, Mapping):
                    workspace = pair_workspaces.get(key)
                    if workspace is None:
                        workspace = _reconstruct_pair_workspace(
                            context,
                            records,
                            teacher_vault,
                            fragment=fragment,
                            panel_receipt=panel_receipt,
                        )
                        pair_workspaces[key] = workspace
                        pair_reconstruction_count += 1
                    mirror_record = _score_inner_mirror(
                        context,
                        records,
                        teacher_vault,
                        workspace=workspace,
                        panel=panel,
                        path=str(selected["path"]),
                        alpha=float(selected["alpha"]),
                    )
                    mirror_evidence.append(mirror_record)
                    mirror = dict(
                        _mapping(
                            _mapping(
                                mirror_record.get("score_evidence"),
                                label="inner mirror evidence",
                            ).get("score"),
                            label="inner mirror score",
                        )
                    )
                roles.append(
                    _build_inner_role(
                        panel_receipt=panel_receipt,
                        shared_fit_receipt=fit,
                        panel=panel,
                        matched_negative=mirror,
                    )
                )
            _publish_inner_fragment(
                output=destination,
                panel_binding_sha256=binding,
                outer_family_id=outer,
                pair_fragments=tuple(relevant.values()),
                inner_roles=roles,
                matched_negative_evidence=mirror_evidence,
                selection_preview=preview,
            )
            inner_fragments[outer] = _load_inner_fragment(
                output=destination,
                panel_binding_sha256=binding,
                outer_family_id=outer,
                pair_fragments=pair_fragments,
                panel_receipt=panel_receipt,
            )

        selection_receipt = _build_selection_receipt(
            panel_receipt=panel_receipt,
            pair_fragments=pair_fragments,
            inner_fragments=inner_fragments,
        )
        selection_items = _selection_item_by_outer(selection_receipt)
        previews = _selection_preview_by_outer(inner_fragments)
        outer_runtime: dict[str, tuple[_EndpointWorkspace, object, object]] = {}

        if _selection_lock_path(destination).exists():
            lock = _load_selection_lock(
                output=destination,
                panel_binding_sha256=binding,
                pair_fragments=pair_fragments,
                inner_fragments=inner_fragments,
                panel_receipt=panel_receipt,
                authenticated_v20a_folds=v20a_folds,
            )
        else:
            outer_refits: list[dict[str, object]] = []
            if _selection_gate_passed(selection_receipt):
                for outer in families:
                    workspace, refit, positive, negative = _prepare_outer_refit(
                        context,
                        records,
                        teacher_vault,
                        outer_family_id=outer,
                        panel_receipt=panel_receipt,
                        selection_preview=previews[outer],
                        selection_item=selection_items[outer],
                        v20a_fold=v20a_folds[outer],
                    )
                    outer_refits.append(refit)
                    outer_runtime[outer] = (workspace, positive, negative)
            _publish_selection_lock(
                output=destination,
                panel_binding_sha256=binding,
                bridge_binding_sha256=bridge,
                pair_fragments=pair_fragments,
                inner_fragments=inner_fragments,
                selection_receipt=selection_receipt,
                outer_refits=outer_refits,
                pair_endpoint_reconstruction_count=pair_reconstruction_count,
            )
            lock = _load_selection_lock(
                output=destination,
                panel_binding_sha256=binding,
                pair_fragments=pair_fragments,
                inner_fragments=inner_fragments,
                panel_receipt=panel_receipt,
                authenticated_v20a_folds=v20a_folds,
            )

        outer_fragments: dict[str, dict[str, object]] = {}
        if lock.get("outer_schedule_authorized") is True:
            refits = _outer_refit_map(lock)
            for outer in families:
                fragment_path = _outer_fragment_path(destination, outer)
                if fragment_path.exists():
                    outer_fragments[outer] = _load_outer_fragment(
                        output=destination,
                        selection_lock=lock,
                        outer_family_id=outer,
                        panel_receipt=panel_receipt,
                    )
                    continue
                reconstructed = False
                runtime = outer_runtime.get(outer)
                if runtime is None:
                    workspace, rebuilt, positive, negative = _prepare_outer_refit(
                        context,
                        records,
                        teacher_vault,
                        outer_family_id=outer,
                        panel_receipt=panel_receipt,
                        selection_preview=previews[outer],
                        selection_item=selection_items[outer],
                        v20a_fold=v20a_folds[outer],
                    )
                    if _v14._canonical_json_bytes(rebuilt) != _v14._canonical_json_bytes(
                        refits[outer]
                    ):
                        raise RuntimeError("resumed V20b outer refit did not reconstruct")
                    runtime = (workspace, positive, negative)
                    reconstructed = True
                workspace, positive, negative = runtime
                outer_score, capability_receipt, score_evidence = _score_outer_family(
                    context,
                    records,
                    teacher_vault,
                    outer_family_id=outer,
                    workspace=workspace,
                    positive_provider=positive,
                    mirror_provider=negative,
                    refit_receipt=refits[outer],
                    selection_item=selection_items[outer],
                    panel_receipt=panel_receipt,
                    output=destination,
                    selection_lock=lock,
                    panel_binding_sha256=binding,
                    pair_fragments=pair_fragments,
                    inner_fragments=inner_fragments,
                    authenticated_v20a_folds=v20a_folds,
                )
                _publish_outer_fragment(
                    output=destination,
                    selection_lock=lock,
                    outer_family_id=outer,
                    refit_receipt=refits[outer],
                    outer_score=outer_score,
                    score_evidence=score_evidence,
                    capability_receipt=capability_receipt,
                    resume_outer_endpoint_reconstructed=reconstructed,
                )
                outer_fragments[outer] = _load_outer_fragment(
                    output=destination,
                    selection_lock=lock,
                    outer_family_id=outer,
                    panel_receipt=panel_receipt,
                )

        context.validate_immutable_inputs()
        teacher_vault.validate_integrity()
        return _finalize_checkpoint_graph(
            output=destination,
            prerequisite=prerequisite,
            pair_fragments=pair_fragments,
            inner_fragments=inner_fragments,
            selection_lock=lock,
            outer_fragments=outer_fragments,
            panel_binding_sha256=binding,
            bridge_binding_sha256=bridge,
        )
    finally:
        context.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_finite_microstep_nested_validation(
        output=arguments.output,
        cache_dir=arguments.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
