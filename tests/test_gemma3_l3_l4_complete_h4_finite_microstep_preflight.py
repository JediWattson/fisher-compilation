from __future__ import annotations

import copy
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import torch

from fisher_graph.complete_h4_fisher_finite_microstep import (
    FisherFiniteMicrostepReceipt,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_finite_microstep_preflight as preflight,
)


_SHA = "a" * 64
_FAMILIES = tuple(f"family-{index}" for index in range(8))


def _hashes(prefix: str, *, count: int = 2) -> dict[str, str]:
    return {
        f"example-{index}": f"{prefix}{index:063x}"[-64:]
        for index in range(count)
    }


def _core_receipt(
    *,
    base_provider_artifact_sha256: str,
    proposal_provider_artifact_sha256: str,
    selected_provider_artifact_sha256: str,
    path: str,
    alpha: float,
    microstep_evidence_sha256: str,
    parameter_sha256s: dict[str, str],
    rank: int = 256,
    conditional_rank: int = 16,
) -> dict[str, object]:
    return FisherFiniteMicrostepReceipt(
        base_provider_artifact_sha256=base_provider_artifact_sha256,
        proposal_provider_artifact_sha256=proposal_provider_artifact_sha256,
        selected_provider_artifact_sha256=selected_provider_artifact_sha256,
        parameter_artifact_sha256="f" * 64,
        microstep_path=path,
        alpha=alpha,
        microstep_protocol_sha256=preflight._MICROSTEP_PROTOCOL_SHA256,
        microstep_evidence_sha256=microstep_evidence_sha256,
        selected_tensor_sha256s=parameter_sha256s,
        prepared_float_scalar_count=1,
        logical_macs_per_token_upper_bound=1,
        rank=rank,
        conditional_rank=conditional_rank,
    ).metadata()


def _candidate(
    *,
    path: str = "joint",
    alpha: float = 1.0e-3,
    objective: float = 0.8,
    changed: bool = True,
    finite: bool = True,
    trust: bool = True,
    rank: bool = True,
    base_provider_sha256: str = "1" * 64,
    proposal_provider_sha256: str = "2" * 64,
) -> dict[str, object]:
    base_parameters = {
        "direction_left": "3" * 64,
        "direction_right": "4" * 64,
        "pedal_weight": "5" * 64,
        "pedal_bias": "6" * 64,
    }
    parameters = {
        **base_parameters,
        "direction_left": (
            "7" * 64 if changed else base_parameters["direction_left"]
        ),
    }
    base_h4 = _hashes("8")
    h4 = {**base_h4, "example-0": "9" * 64} if changed else base_h4
    base_logits = _hashes("a")
    logits = (
        {**base_logits, "example-1": "b" * 64}
        if changed
        else base_logits
    )
    execution_change = preflight.detect_execution_change(
        base_parameter_sha256s=base_parameters,
        candidate_parameter_sha256s=parameters,
        base_h4_sha256s=base_h4,
        candidate_h4_sha256s=h4,
        base_logits_sha256s=base_logits,
        candidate_logits_sha256s=logits,
    )
    provider_sha256 = "b" * 64
    evidence_sha256 = "d" * 64
    receipt = _core_receipt(
        base_provider_artifact_sha256=base_provider_sha256,
        proposal_provider_artifact_sha256=proposal_provider_sha256,
        selected_provider_artifact_sha256=provider_sha256,
        path=path,
        alpha=alpha,
        microstep_evidence_sha256=evidence_sha256,
        parameter_sha256s=parameters,
    )
    return {
        "path": path,
        "alpha": alpha,
        "objective": objective,
        "execution_changed": changed,
        "execution_change": execution_change,
        "finite": finite,
        "pointwise_trust_passed": trust,
        "rank_is_16": rank,
        "base_provider_artifact_sha256": base_provider_sha256,
        "proposal_provider_artifact_sha256": proposal_provider_sha256,
        "provider_artifact_sha256": provider_sha256,
        "microstep_artifact_sha256": receipt["artifact_sha256"],
        "microstep_receipt_sha256": receipt["artifact_sha256"],
        "microstep_evidence_sha256": evidence_sha256,
        "microstep_receipt": receipt,
        "parameter_sha256s": parameters,
        "post_cast_h4_sha256s": h4,
        "supervised_full_vocab_logits_sha256s": logits,
    }


def _positive_grid(
    *,
    paths: tuple[str, ...] = ("joint",),
    objective: float = 2.0,
    changed: bool = True,
) -> tuple[dict[str, object], ...]:
    return tuple(
        _candidate(
            path=path,
            alpha=alpha,
            objective=objective,
            changed=changed,
        )
        for path in paths
        for alpha in preflight.POSITIVE_ALPHAS
    )


def _matched_negative(
    selected: dict[str, object],
    *,
    objective: float = 0.9,
    finite: bool = True,
    trust: bool = True,
    rank: bool = True,
    changed: bool = True,
) -> dict[str, object]:
    return _candidate(
        path=str(selected["path"]),
        alpha=-float(selected["alpha"]),
        objective=objective,
        finite=finite,
        trust=trust,
        rank=rank,
        changed=changed,
        base_provider_sha256=str(selected["base_provider_artifact_sha256"]),
        proposal_provider_sha256=str(
            selected["proposal_provider_artifact_sha256"]
        ),
    )


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _fit_families(held_family_id: str) -> tuple[str, ...]:
    return tuple(family for family in _FAMILIES if family != held_family_id)


def _fit_examples(held_family_id: str) -> tuple[str, ...]:
    return tuple(
        f"{family}/example-{index}"
        for family in _fit_families(held_family_id)
        for index in range(2)
    )


def _baseline(
    held_family_id: str,
    *,
    objective: float = 1.0,
) -> dict[str, object]:
    held_index = _FAMILIES.index(held_family_id)
    examples = _fit_examples(held_family_id)
    return {
        "objective": objective,
        "family_objectives": {
            family: objective for family in _fit_families(held_family_id)
        },
        "provider_artifact_sha256": _digest(100 + held_index),
        "parameter_sha256s": {
            name: _digest(200 + held_index * 10 + index)
            for index, name in enumerate(
                ("direction_left", "direction_right", "pedal_weight", "pedal_bias")
            )
        },
        "post_cast_h4_sha256s": {
            example: _digest(500 + held_index * 20 + index)
            for index, example in enumerate(examples)
        },
        "supervised_full_vocab_logits_sha256s": {
            example: _digest(800 + held_index * 20 + index)
            for index, example in enumerate(examples)
        },
        "finite": True,
        "pointwise_trust_passed": True,
        "rank_is_16": True,
    }


def _state_hashes(seed: int) -> dict[str, str]:
    return {
        key: _digest(seed + index)
        for index, key in enumerate(
            (
                "direction_left_sha256",
                "direction_right_sha256",
                "pedal_weight_sha256",
                "pedal_bias_sha256",
            )
        )
    }


def _training_sequence_sha256s(held_family_id: str) -> tuple[str, ...]:
    held_index = _FAMILIES.index(held_family_id)
    return tuple(
        _digest(5_100 + held_index * 20 + index) for index in range(14)
    )


def _record_receipt_sha256s() -> dict[str, str]:
    return {
        f"{family}/example-{index}": _digest(6_000 + family_index * 2 + index)
        for family_index, family in enumerate(_FAMILIES)
        for index in range(2)
    }


def _optimization_receipt(
    held_family_id: str,
    baseline: dict[str, object],
) -> dict[str, object]:
    held_index = _FAMILIES.index(held_family_id)
    fit_examples = _fit_examples(held_family_id)
    family_zero = dict(baseline["family_objectives"])  # type: ignore[arg-type]
    family_one = {family: 0.8 for family in family_zero}
    return {
        "held_family_id": held_family_id,
        "coordinate_objective": "reverse_vjp_fisher",
        "fit_protocol_sha256": _digest(7_000),
        "start_provider_artifact_sha256": _digest(1_300 + held_index),
        "training_family_ids": _fit_families(held_family_id),
        "training_sequence_sha256s": _training_sequence_sha256s(
            held_family_id
        ),
        "training_record_receipt_sha256s": tuple(
            _record_receipt_sha256s()[example] for example in fit_examples
        ),
        "checkpoint_scores": (baseline["objective"], 0.8),
        "checkpoint_family_scores": (family_zero, family_one),
        "checkpoint_provider_artifact_sha256s": (
            baseline["provider_artifact_sha256"],
            _digest(1_400 + held_index),
        ),
        "checkpoint_state_receipts": (
            _state_hashes(1_000 + held_index * 10),
            _state_hashes(1_100 + held_index * 10),
        ),
        "selected_checkpoint": 0,
        "teacher_capability_artifact_sha256": _digest(4_000 + held_index),
        "capability_receipt": {
            "artifact_sha256": _digest(4_000 + held_index),
            "held_family_id": held_family_id,
            "authorized_example_count": 14,
            "authorized_family_count": 7,
            "per_example_access_counts": {
                example: 5 for example in fit_examples
            },
            "held_family_capability_excluded": True,
            "teacher_rows_consumed_only_through_capability": True,
        },
    }


def _v19_receipt_sha256(held_family_id: str) -> str:
    receipt = _optimization_receipt(held_family_id, _baseline(held_family_id))
    return preflight._v14._sha256(receipt, domain=preflight._EVIDENCE_DOMAIN)


def _endpoint_binding(
    held_family_id: str,
    baseline: dict[str, object],
) -> dict[str, object]:
    held_index = _FAMILIES.index(held_family_id)
    checkpoint_state = _state_hashes(1_000 + held_index * 10)
    adam_state = _state_hashes(1_100 + held_index * 10)
    family_objectives = dict(baseline["family_objectives"])  # type: ignore[arg-type]
    payload: dict[str, object] = {
        "held_family_id": held_family_id,
        "pinned_v19_optimization_receipt_sha256": _v19_receipt_sha256(
            held_family_id
        ),
        "parent_provider_artifact_sha256": _digest(1_200 + held_index),
        "pinned_parent_provider_artifact_sha256": _digest(1_200 + held_index),
        "start_provider_artifact_sha256": _digest(1_300 + held_index),
        "pinned_start_provider_artifact_sha256": _digest(1_300 + held_index),
        "base_provider_artifact_sha256": baseline["provider_artifact_sha256"],
        "pinned_base_provider_artifact_sha256": baseline[
            "provider_artifact_sha256"
        ],
        "proposal_provider_artifact_sha256": _digest(1_400 + held_index),
        "pinned_proposal_provider_artifact_sha256": _digest(1_400 + held_index),
        "checkpoint_zero_state_sha256s": checkpoint_state,
        "pinned_checkpoint_zero_state_sha256s": checkpoint_state,
        "first_adam_state_sha256s": adam_state,
        "pinned_first_adam_state_sha256s": adam_state,
        "checkpoint_zero_objective": baseline["objective"],
        "pinned_checkpoint_zero_objective": baseline["objective"],
        "checkpoint_zero_family_objectives": family_objectives,
        "pinned_checkpoint_zero_family_objectives": family_objectives,
        "training_family_ids": _fit_families(held_family_id),
        "training_sequence_sha256s": _training_sequence_sha256s(
            held_family_id
        ),
        "training_record_receipt_sha256s": _optimization_receipt(
            held_family_id, baseline
        )["training_record_receipt_sha256s"],
        "checks": {
            "parent_exact": True,
            "start_exact": True,
            "checkpoint_zero_exact": True,
            "first_adam_exact": True,
        },
    }
    payload["receipt_sha256"] = preflight._v14._sha256(
        payload, domain=preflight._ENDPOINT_DOMAIN
    )
    return payload


def _protocol_candidate(
    *,
    baseline: dict[str, object],
    endpoint: dict[str, object],
    path: str,
    alpha: float,
    objective: float,
    changed: bool = True,
    finite: bool = True,
    trust: bool = True,
    rank: bool = True,
) -> dict[str, object]:
    path_index = preflight.MICROSTEP_PATHS.index(path)
    alpha_index = preflight.POSITIVE_ALPHAS.index(abs(alpha))
    serial = path_index * 20 + alpha_index + (100 if alpha < 0.0 else 0)
    parameters = dict(baseline["parameter_sha256s"])  # type: ignore[arg-type]
    first_parameter = next(iter(parameters))
    parameters[first_parameter] = _digest(2_000 + serial)
    h4 = dict(baseline["post_cast_h4_sha256s"])  # type: ignore[arg-type]
    logits = dict(
        baseline["supervised_full_vocab_logits_sha256s"]  # type: ignore[arg-type]
    )
    if changed:
        h4[next(iter(h4))] = _digest(2_200 + serial)
        logits[next(reversed(logits))] = _digest(2_400 + serial)
    execution_change = preflight.detect_execution_change(
        base_parameter_sha256s=baseline["parameter_sha256s"],  # type: ignore[arg-type]
        candidate_parameter_sha256s=parameters,
        base_h4_sha256s=baseline["post_cast_h4_sha256s"],  # type: ignore[arg-type]
        candidate_h4_sha256s=h4,
        base_logits_sha256s=baseline[
            "supervised_full_vocab_logits_sha256s"
        ],  # type: ignore[arg-type]
        candidate_logits_sha256s=logits,
    )
    provider_sha256 = _digest(2_600 + serial)
    evidence_sha256 = _digest(3_200 + serial)
    receipt = _core_receipt(
        base_provider_artifact_sha256=str(
            endpoint["base_provider_artifact_sha256"]
        ),
        proposal_provider_artifact_sha256=str(
            endpoint["proposal_provider_artifact_sha256"]
        ),
        selected_provider_artifact_sha256=provider_sha256,
        path=path,
        alpha=alpha,
        microstep_evidence_sha256=evidence_sha256,
        parameter_sha256s=parameters,
    )
    return {
        "path": path,
        "alpha": alpha,
        "objective": objective,
        "family_objectives": {
            family: objective for family in baseline["family_objectives"]  # type: ignore[union-attr]
        },
        "execution_changed": execution_change["execution_changed"],
        "execution_change": execution_change,
        "finite": finite,
        "pointwise_trust_passed": trust,
        "rank_is_16": rank,
        "base_provider_artifact_sha256": endpoint[
            "base_provider_artifact_sha256"
        ],
        "proposal_provider_artifact_sha256": endpoint[
            "proposal_provider_artifact_sha256"
        ],
        "provider_artifact_sha256": provider_sha256,
        "microstep_artifact_sha256": receipt["artifact_sha256"],
        "microstep_receipt_sha256": receipt["artifact_sha256"],
        "microstep_evidence_sha256": evidence_sha256,
        "microstep_receipt": receipt,
        "parameter_sha256s": parameters,
        "post_cast_h4_sha256s": h4,
        "supervised_full_vocab_logits_sha256s": logits,
    }


def _protocol_grid(
    *,
    baseline: dict[str, object],
    endpoint: dict[str, object],
    paths: tuple[str, ...],
    winner_path: str = "joint",
    winner_alpha: float = 1.0e-3,
    winner_objective: float = 0.8,
    changed: bool = True,
) -> tuple[dict[str, object], ...]:
    return tuple(
        _protocol_candidate(
            baseline=baseline,
            endpoint=endpoint,
            path=path,
            alpha=alpha,
            objective=(
                winner_objective
                if path == winner_path and alpha == winner_alpha
                else 1.5
            ),
            changed=changed,
        )
        for path in paths
        for alpha in preflight.POSITIVE_ALPHAS
    )


def _capability_receipt(
    held_family_id: str,
    *,
    executions_per_prompt: int,
) -> dict[str, object]:
    examples = _fit_examples(held_family_id)
    return {
        "held_family_id": held_family_id,
        "authorized_example_count": 14,
        "authorized_family_count": 7,
        "held_family_capability_excluded": True,
        "teacher_rows_consumed_only_through_capability": True,
        "access_count": 14 * executions_per_prompt,
        "artifact_sha256": _digest(4_000 + _FAMILIES.index(held_family_id)),
        "per_example_access_counts": {
            example: executions_per_prompt for example in examples
        },
    }


def _ownership_receipt(
    held_family_id: str,
    *,
    provider_artifact_sha256: str,
) -> dict[str, object]:
    held_index = _FAMILIES.index(held_family_id)
    payload: dict[str, object] = {
        "held_family_id": held_family_id,
        "held_sequence_sha256s": (
            _digest(5_000 + held_index * 2),
            _digest(5_001 + held_index * 2),
        ),
        "provider_artifact_sha256": provider_artifact_sha256,
        "fit_family_ids": _fit_families(held_family_id),
        "fit_sequence_sha256s": tuple(
            _training_sequence_sha256s(held_family_id)
        ),
        "held_family_absent_from_fit_family_ids": True,
        "held_sequences_disjoint_from_fit_sequences": True,
    }
    payload["receipt_sha256"] = preflight._v14._sha256(
        payload, domain=preflight._v19._OWNERSHIP_DOMAIN
    )
    return payload


def _rehash_endpoint(endpoint: dict[str, object]) -> None:
    payload = {key: value for key, value in endpoint.items() if key != "receipt_sha256"}
    endpoint["receipt_sha256"] = preflight._v14._sha256(
        payload, domain=preflight._ENDPOINT_DOMAIN
    )


def _rehash_ownership(ownership: dict[str, object]) -> None:
    payload = {
        key: value for key, value in ownership.items() if key != "receipt_sha256"
    }
    ownership["receipt_sha256"] = preflight._v14._sha256(
        payload, domain=preflight._v19._OWNERSHIP_DOMAIN
    )


def test_ownership_receipt_hash_is_canonical_across_tuple_json_lists() -> None:
    tuple_receipt = _ownership_receipt(
        _FAMILIES[0],
        provider_artifact_sha256="1" * 64,
    )
    json_receipt = copy.deepcopy(tuple_receipt)
    for key in (
        "held_sequence_sha256s",
        "fit_family_ids",
        "fit_sequence_sha256s",
    ):
        json_receipt[key] = list(json_receipt[key])  # type: ignore[arg-type]

    tuple_sha256 = preflight._ownership_receipt_sha256(
        tuple_receipt,
        label="tuple ownership",
    )
    json_sha256 = preflight._ownership_receipt_sha256(
        json_receipt,
        label="JSON ownership",
    )

    assert tuple_sha256 == json_sha256 == tuple_receipt["receipt_sha256"]

    forged = copy.deepcopy(json_receipt)
    forged["fit_sequence_sha256s"][0] = "e" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="receipt hash drifted"):
        preflight._ownership_receipt_sha256(forged, label="forged ownership")


def test_parameter_hashes_use_the_core_microstep_receipt_domain() -> None:
    provider = SimpleNamespace(
        direction_left=torch.arange(12, dtype=torch.float64).reshape(6, 2),
        direction_right=torch.arange(8, dtype=torch.float64).reshape(2, 4),
        pedal_weight=torch.tensor((0.2, -0.4, 0.6), dtype=torch.float64),
        pedal_bias=torch.tensor((0.1,), dtype=torch.float64),
    )

    hashes = preflight._parameter_sha256s(provider)

    assert hashes == (
        preflight.fisher_finite_microstep_selected_tensor_sha256s(provider)
    )
    assert hashes != {
        name: preflight._v14._tensor_sha256(getattr(provider, name))
        for name in hashes
    }


def _replace_core_receipt_geometry(
    candidate: dict[str, object],
    *,
    rank: int,
    conditional_rank: int,
) -> None:
    current = candidate["microstep_receipt"]
    assert isinstance(current, dict)
    replacement = _core_receipt(
        base_provider_artifact_sha256=str(
            candidate["base_provider_artifact_sha256"]
        ),
        proposal_provider_artifact_sha256=str(
            candidate["proposal_provider_artifact_sha256"]
        ),
        selected_provider_artifact_sha256=str(
            candidate["provider_artifact_sha256"]
        ),
        path=str(candidate["path"]),
        alpha=float(candidate["alpha"]),
        microstep_evidence_sha256=str(candidate["microstep_evidence_sha256"]),
        parameter_sha256s=dict(candidate["parameter_sha256s"]),  # type: ignore[arg-type]
        rank=rank,
        conditional_rank=conditional_rank,
    )
    candidate["microstep_receipt"] = replacement
    candidate["microstep_receipt_sha256"] = replacement["artifact_sha256"]
    candidate["microstep_artifact_sha256"] = replacement["artifact_sha256"]


def _protocol_fold(
    held_family_id: str,
    *,
    expanded: bool,
    winner_path: str = "joint",
    winner_alpha: float = 1.0e-3,
    winner_objective: float = 0.8,
    mirror_objective: float = 0.9,
    changed: bool = True,
    extra_prior_mirror_count: int = 0,
) -> dict[str, object]:
    baseline = _baseline(held_family_id)
    endpoint = _endpoint_binding(held_family_id, baseline)
    candidates = _protocol_grid(
        baseline=baseline,
        endpoint=endpoint,
        paths=preflight.MICROSTEP_PATHS if expanded else ("joint",),
        winner_path=winner_path,
        winner_alpha=winner_alpha,
        winner_objective=winner_objective,
        changed=changed,
    )
    selected = preflight.select_best_positive_microstep(
        baseline_objective=float(baseline["objective"]),
        candidates=candidates,
    )
    negative = (
        _protocol_candidate(
            baseline=baseline,
            endpoint=endpoint,
            path=str(selected["path"]),
            alpha=-float(selected["alpha"]),
            objective=mirror_objective,
        )
        if isinstance(selected, dict)
        else None
    )
    qualification = (
        preflight.evaluate_fold_qualification(
            held_family_id=held_family_id,
            baseline_objective=float(baseline["objective"]),
            selected_positive=selected,
            matched_negative=negative,
        )
        if expanded
        else preflight.evaluate_sentinel_decision(
            baseline_objective=float(baseline["objective"]),
            selected_positive=selected,
            matched_negative=negative,
        )
    )
    mirror_count = int(negative is not None) + extra_prior_mirror_count
    return {
        "held_family_id": held_family_id,
        "baseline": baseline,
        "positive_candidates": candidates,
        "matched_negative": negative,
        "qualification": qualification,
        "capability_receipt": _capability_receipt(
            held_family_id,
            executions_per_prompt=1 + len(candidates) + mirror_count,
        ),
        "ownership_receipt": _ownership_receipt(
            held_family_id,
            provider_artifact_sha256=str(baseline["provider_artifact_sha256"]),
        ),
        "endpoint_binding": endpoint,
        "held_scoring_performed": False,
    }


def _expanded_case(
    *,
    winner_objective: float = 0.8,
    mirror_objectives: tuple[float, ...] = (0.9,) * 8,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    sentinel = _protocol_fold(
        _FAMILIES[0],
        expanded=False,
        winner_objective=winner_objective,
        mirror_objective=mirror_objectives[0],
    )
    sentinel["expanded_winner_reused_sentinel_mirror"] = True
    folds = tuple(
        _protocol_fold(
            family,
            expanded=True,
            winner_objective=winner_objective,
            mirror_objective=mirror,
        )
        for family, mirror in zip(_FAMILIES, mirror_objectives, strict=True)
    )
    return sentinel, folds


def _authenticated_v19_artifact(
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    receipt_hashes: dict[str, str] = {}
    bundles: dict[str, dict[str, object]] = {}
    for family in _FAMILIES:
        baseline = _baseline(family)
        receipt = _optimization_receipt(family, baseline)
        receipt_sha256 = preflight._v14._sha256(
            receipt, domain=preflight._EVIDENCE_DOMAIN
        )
        receipt_hashes[family] = receipt_sha256
        family_index = _FAMILIES.index(family)
        bundles[family] = {
            "optimization_receipt": receipt,
            "optimization_receipt_sha256": receipt_sha256,
            "parent_provider_artifact_sha256": _digest(1_200 + family_index),
            "start_provider_artifact_sha256": _digest(1_300 + family_index),
            "capability_receipt": receipt["capability_receipt"],
            "ownership_receipt": _ownership_receipt(
                family,
                provider_artifact_sha256=str(
                    baseline["provider_artifact_sha256"]
                ),
            ),
        }
    prerequisite: dict[str, object] = {
        "path": preflight._V19_OUTPUT.as_posix(),
        "format_version": 19,
        "report_sha256": preflight._V19_LOGICAL_SHA256,
        "file_sha256": preflight._V19_FILE_SHA256,
        "classification": preflight._V19_CLASSIFICATION,
        "passed": False,
        "candidate": None,
        "full_refit_qualification": None,
        "provider_sidecar_absent": True,
        "fisher_optimization_receipt_sha256s": receipt_hashes,
        "authenticated_panel": {"prompt_count": 16, "family_count": 8},
        "authenticated_bridge_binding_sha256": _SHA,
        "authenticated_fit_collection": {
            "prompt_count": 16,
            "family_count": 8,
            "trace_receipt_sha256s": tuple(
                _record_receipt_sha256s().values()
            ),
            "teacher_vault": {
                "artifact_sha256": _digest(6_100),
                "example_count": 16,
                "family_count": 8,
            },
            "held_rows_cached_but_capability_excluded_and_not_consumed_by_fold_fit": True,
        },
    }
    return prerequisite, bundles


@pytest.fixture(autouse=True)
def _authenticated_v19_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preflight,
        "_load_authenticated_v19_artifact",
        lambda: copy.deepcopy(_authenticated_v19_artifact()),
    )


def _report_kwargs(
    *,
    sentinel: dict[str, object],
    folds: tuple[dict[str, object], ...],
    work: dict[str, object],
    artifact_path: Path | str = Path(".local-runs/test-v20a.json"),
) -> dict[str, object]:
    return {
        "artifact_path": artifact_path,
        "panel": {"prompt_count": 16, "family_count": 8},
        "bridge_binding_sha256": _SHA,
        "prerequisite": _authenticated_v19_artifact()[0],
        "fit_collection": {
            "prompt_count": 16,
            "family_count": 8,
            "held_teacher_rows_cached": True,
            "held_teacher_rows_scored": False,
            "record_receipt_sha256s": {
                **_record_receipt_sha256s()
            },
            "teacher_vault_receipt": {
                "artifact_sha256": _digest(6_100),
                "example_count": 16,
                "family_count": 8,
            },
        },
        "sentinel": sentinel,
        "folds": folds,
        "work": work,
        "integrity": {
            "immutable_inputs_validated": True,
            "v19_prerequisite_exact": True,
            "all_tested_fold_endpoint_bindings_exact": True,
            "held_teacher_rows_capability_excluded": True,
            "held_score_row_count": 0,
            "held_scoring_performed": False,
            "guard_opened": False,
            "calibration_b_opened": False,
            "provider_sidecar_written": False,
        },
    }


def test_protocol_constants_are_exact_and_v19_is_pinned() -> None:
    assert preflight.ALPHA_LADDER == (
        0.0,
        1.0e-6,
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
    )
    assert preflight.MICROSTEP_PATHS == (
        "direction_only",
        "pedal_only",
        "joint",
    )
    assert preflight._V19_LOGICAL_SHA256 == (
        "4f0439858b7e636ae648aa12d3cdb6837350510f10b520ab1c09e69074417d46"
    )
    assert preflight._V19_FILE_SHA256 == (
        "b29e45590c3085c18ba9ad516a3bf508d34a83c57a622f8069035d3e457a9a1e"
    )
    assert preflight._V19_CLASSIFICATION == (
        "finite_joint_pedal_outer_fidelity_insufficient"
    )


def test_internal_positive_grid_accepts_exact_incremental_path_subset_only() -> None:
    paths = ("direction_only", "pedal_only")
    rows = tuple(
        _candidate(path=path, alpha=alpha)
        for path in paths
        for alpha in preflight.POSITIVE_ALPHAS
    )

    assert preflight._validate_exact_positive_grid(
        rows,
        expected_paths=paths,
    ) == rows
    with pytest.raises(ValueError, match="path geometry"):
        preflight._validate_exact_positive_grid(rows)
    with pytest.raises(ValueError, match="not exhaustive"):
        preflight._validate_exact_positive_grid(
            rows[:-1],
            expected_paths=paths,
        )
    with pytest.raises(ValueError, match="path geometry"):
        preflight._validate_exact_positive_grid(
            rows,
            expected_paths=("direction_only", "direction_only"),
        )


def test_execution_change_requires_post_cast_h4_and_supervised_logits() -> None:
    base_parameters = {"left": "1" * 64, "right": "2" * 64}
    changed_parameters = {"left": "3" * 64, "right": "2" * 64}
    base_h4 = _hashes("4")
    changed_h4 = {**base_h4, "example-0": "5" * 64}
    base_logits = _hashes("6")
    changed_logits = {**base_logits, "example-1": "7" * 64}

    changed = preflight.detect_execution_change(
        base_parameter_sha256s=base_parameters,
        candidate_parameter_sha256s=changed_parameters,
        base_h4_sha256s=base_h4,
        candidate_h4_sha256s=changed_h4,
        base_logits_sha256s=base_logits,
        candidate_logits_sha256s=changed_logits,
    )
    assert changed["parameter_changed"] is True
    assert changed["h4_changed_prompt_count"] == 1
    assert changed["logits_changed_prompt_count"] == 1
    assert changed["execution_changed"] is True

    for candidate_h4, candidate_logits in (
        (base_h4, base_logits),
        (changed_h4, base_logits),
        (base_h4, changed_logits),
    ):
        no_op = preflight.detect_execution_change(
            base_parameter_sha256s=base_parameters,
            candidate_parameter_sha256s=changed_parameters,
            base_h4_sha256s=base_h4,
            candidate_h4_sha256s=candidate_h4,
            base_logits_sha256s=base_logits,
            candidate_logits_sha256s=candidate_logits,
        )
        assert no_op["execution_changed"] is False


def test_execution_change_rejects_prompt_or_parameter_geometry_drift() -> None:
    with pytest.raises(ValueError, match="parameter hash geometry"):
        preflight.detect_execution_change(
            base_parameter_sha256s={"left": "1" * 64},
            candidate_parameter_sha256s={"right": "2" * 64},
            base_h4_sha256s={"a": "3" * 64},
            candidate_h4_sha256s={"a": "3" * 64},
            base_logits_sha256s={"a": "4" * 64},
            candidate_logits_sha256s={"a": "4" * 64},
        )
    with pytest.raises(ValueError, match="prompt hash geometry|ownership"):
        preflight.detect_execution_change(
            base_parameter_sha256s={"left": "1" * 64},
            candidate_parameter_sha256s={"left": "2" * 64},
            base_h4_sha256s={"a": "3" * 64},
            candidate_h4_sha256s={"a": "3" * 64},
            base_logits_sha256s={"b": "4" * 64},
            candidate_logits_sha256s={"b": "4" * 64},
        )


def test_live_causal_check_compares_native_h4_to_canonical_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "_EXPECTED_VOCABULARY", 3)
    native_base = torch.tensor(
        [[1.25, -2.5], [3.0, 4.0], [-5.5, 6.25]],
        dtype=torch.float32,
    )
    candidate = native_base.clone()
    candidate[1] += 0.5
    sequence = SimpleNamespace(
        support_mask=torch.tensor([False, True, False]),
        base_h4=native_base.to(dtype=torch.float64),
    )
    record = SimpleNamespace(sequence=sequence)
    execution = SimpleNamespace(
        candidate_h4=candidate.unsqueeze(0),
        logits=torch.tensor(
            [[[0.0, 0.1, 0.2], [0.2, 0.1, 0.0], [0.1, 0.0, 0.2]]],
            dtype=torch.float32,
        ),
        h4_head_sha256=_SHA,
    )
    score, _h4_hash, _logits_hash = preflight._execution_hashes_and_score(
        execution=execution,
        record=record,
        teacher=torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32),
        supervised_indices=torch.tensor([1], dtype=torch.int64),
        provider_artifact_sha256=_SHA,
    )
    assert score >= 0.0

    escaped = copy.deepcopy(execution)
    escaped.candidate_h4[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="escaped complete-H4 causal support"):
        preflight._execution_hashes_and_score(
            execution=escaped,
            record=record,
            teacher=torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32),
            supervised_indices=torch.tensor([1], dtype=torch.int64),
            provider_artifact_sha256=_SHA,
        )


def test_positive_selector_uses_exact_objective_alpha_path_tie_order() -> None:
    rows = list(_positive_grid(paths=preflight.MICROSTEP_PATHS))
    for row in rows:
        if row["path"] in preflight.MICROSTEP_PATHS and row["alpha"] == 1.0e-3:
            row["objective"] = 0.8
        if row["path"] == "joint" and row["alpha"] == 1.0e-4:
            row["objective"] = 0.8
    selected = preflight.select_best_positive_microstep(
        baseline_objective=1.0, candidates=rows
    )
    assert selected["path"] == "joint"
    assert selected["alpha"] == 1.0e-4  # same objective, then smaller alpha

    for row in rows:
        row["objective"] = 0.8 if row["alpha"] == 1.0e-3 else 2.0
    selected = preflight.select_best_positive_microstep(
        baseline_objective=1.0, candidates=rows
    )
    assert selected["path"] == "direction_only"
    assert selected["alpha"] == 1.0e-3  # direction, pedal, joint path order


def test_positive_selector_ignores_execution_noops_but_still_returns_a_loser() -> None:
    rows = list(_positive_grid(changed=False))
    rows[0]["objective"] = 0.1
    rows[1].update(objective=1.2, execution_changed=True)
    selected = preflight.select_best_positive_microstep(
        baseline_objective=1.0, candidates=rows
    )
    assert selected is rows[1]
    assert (
        preflight.select_best_positive_microstep(
            baseline_objective=1.0,
            candidates=_positive_grid(changed=False),
        )
        is None
    )


@pytest.mark.parametrize("alpha", (0.0, -1.0e-3, 0.5))
def test_positive_selector_rejects_nonpositive_or_unregistered_alpha(
    alpha: float,
) -> None:
    rows = list(_positive_grid())
    rows[0]["alpha"] = alpha
    with pytest.raises(ValueError, match="positive candidate alpha"):
        preflight.select_best_positive_microstep(
            baseline_objective=1.0,
            candidates=rows,
        )


def test_positive_selector_rejects_incomplete_or_mixed_path_grid() -> None:
    with pytest.raises(ValueError, match="exhaustive"):
        preflight.select_best_positive_microstep(
            baseline_objective=1.0,
            candidates=_positive_grid()[:-1],
        )
    mixed = list(_positive_grid())
    mixed[-1]["path"] = "pedal_only"
    with pytest.raises(ValueError, match="path geometry"):
        preflight.select_best_positive_microstep(
            baseline_objective=1.0,
            candidates=mixed,
        )


def test_sentinel_requires_joint_positive_and_matched_negative_direction() -> None:
    selected = _candidate(path="joint", alpha=1.0e-3, objective=0.8)
    passed = preflight.evaluate_sentinel_decision(
        baseline_objective=1.0,
        selected_positive=selected,
        matched_negative=_matched_negative(selected, objective=0.9),
    )
    assert passed["positive_beats_baseline_beyond_floor"] is True
    assert passed["positive_beats_mirror_beyond_floor"] is True
    assert passed["passed"] is True
    assert passed["selected_positive"] == {
        "path": "joint",
        "alpha": 1.0e-3,
        "objective": 0.8,
        "provider_artifact_sha256": "b" * 64,
        "microstep_artifact_sha256": selected["microstep_artifact_sha256"],
        "execution_change": selected["execution_change"],
    }

    ambiguous = preflight.evaluate_sentinel_decision(
        baseline_objective=1.0,
        selected_positive=selected,
        matched_negative=_matched_negative(selected, objective=0.7),
    )
    assert ambiguous["positive_beats_baseline_beyond_floor"] is True
    assert ambiguous["positive_beats_mirror_beyond_floor"] is False
    assert ambiguous["passed"] is False
    with pytest.raises(ValueError, match="joint path"):
        preflight.evaluate_sentinel_decision(
            baseline_objective=1.0,
            selected_positive=_candidate(path="pedal_only"),
            matched_negative=_matched_negative(
                _candidate(path="pedal_only"), objective=0.9
            ),
        )


def test_sentinel_without_execution_changing_positive_stops_without_mirror() -> None:
    result = preflight.evaluate_sentinel_decision(
        baseline_objective=1.0,
        selected_positive=None,
        matched_negative=None,
    )
    assert result["selected_positive"] is None
    assert result["positive_execution_changed"] is False
    assert result["passed"] is False
    with pytest.raises(ValueError, match="mirror without a positive"):
        preflight.evaluate_sentinel_decision(
            baseline_objective=1.0,
            selected_positive=None,
            matched_negative=_candidate(alpha=-1.0e-3),
        )


def test_fold_qualification_binds_held_family_and_fails_trust_rank_or_finite() -> None:
    for field in ("finite", "trust", "rank"):
        options = {field: False}
        result = preflight.evaluate_fold_qualification(
            held_family_id="held-family",
            baseline_objective=1.0,
            selected_positive=(selected := _candidate(**options)),
            matched_negative=_matched_negative(selected, objective=0.9),
        )
        assert result["held_family_id"] == "held-family"
        assert result["finite_trust_rank_passed"] is False
        assert result["passed"] is False


def test_matched_negative_cast_or_logits_noop_cannot_pass_signed_direction() -> None:
    selected = _candidate(objective=0.8, changed=True)
    no_op_mirror = _matched_negative(
        selected,
        objective=0.9,
        changed=False,
    )
    result = preflight.evaluate_sentinel_decision(
        baseline_objective=1.0,
        selected_positive=selected,
        matched_negative=no_op_mirror,
    )
    assert result["positive_execution_changed"] is True
    assert result["matched_negative_execution_changed"] is False
    assert result["positive_beats_baseline_beyond_floor"] is True
    assert result["positive_beats_mirror_beyond_floor"] is True
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    (
        ("alpha", 1.0e-3, "path/alpha"),
        ("path", "pedal_only", "path/alpha"),
        ("base_provider_artifact_sha256", "e" * 64, "endpoint"),
        ("proposal_provider_artifact_sha256", "e" * 64, "endpoint"),
        ("provider_artifact_sha256", "not-a-sha", "SHA-256"),
        ("microstep_evidence_sha256", "e" * 64, "core receipt"),
    ),
)
def test_matched_negative_must_be_exactly_bound_and_authenticated(
    field: str,
    value: object,
    pattern: str,
) -> None:
    selected = _candidate(objective=0.8)
    mirror = _matched_negative(selected, objective=0.9)
    mirror[field] = value
    with pytest.raises(ValueError, match=pattern):
        preflight.evaluate_sentinel_decision(
            baseline_objective=1.0,
            selected_positive=selected,
            matched_negative=mirror,
        )


def test_candidate_core_receipt_accepts_rank256_conditional16_and_rejects_swap() -> None:
    fold = _protocol_fold(_FAMILIES[0], expanded=False)
    baseline = fold["baseline"]
    candidate = copy.deepcopy(fold["positive_candidates"][0])  # type: ignore[index]
    assert isinstance(baseline, dict)
    assert isinstance(candidate, dict)
    validated = preflight._validate_candidate_authentication(
        candidate,
        baseline=baseline,
        signed=False,
    )
    assert validated["path"] == "joint"

    _replace_core_receipt_geometry(
        candidate,
        rank=16,
        conditional_rank=256,
    )
    with pytest.raises(ValueError, match="core microstep receipt binding"):
        preflight._validate_candidate_authentication(
            candidate,
            baseline=baseline,
            signed=False,
        )


def test_work_accounting_matches_actual_sentinel_and_expanded_stages() -> None:
    sentinel_stop = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=0,
    )
    assert sentinel_stop["full_model_forward_count"] == 144
    assert sentinel_stop["full_suffix_backward_traversal_count"] == 30
    assert sentinel_stop["local_head_autograd_contraction_count"] == 14
    assert sentinel_stop["total_autograd_grad_call_count"] == 44
    assert sentinel_stop["teacher_capability_access_count"] == 112

    sentinel_mirror = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    assert sentinel_mirror["full_model_forward_count"] == 158
    assert sentinel_mirror["teacher_capability_access_count"] == 126

    expanded_same_winner = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=8,
    )
    expanded_changed_winner = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=9,
    )
    assert expanded_same_winner["full_model_forward_count"] == 2_608
    assert expanded_changed_winner["full_model_forward_count"] == 2_622
    for work in (expanded_same_winner, expanded_changed_winner):
        assert work["full_suffix_backward_traversal_count"] == 128
        assert work["local_head_autograd_contraction_count"] == 112
        assert work["total_autograd_grad_call_count"] == 240
        breakdown = work["breakdown"]
        assert work["full_model_forward_count"] == sum(
            value for key, value in breakdown.items() if "forward" in key
        )


def test_work_accounting_rejects_nonprotocol_geometry() -> None:
    for values in ((1, 6, 0), (2, 42, 2), (8, 167, 8), (8, 168, 10)):
        with pytest.raises(ValueError, match="work geometry"):
            preflight._work_accounting(
                tested_fold_count=values[0],
                positive_candidate_count=values[1],
                mirror_candidate_count=values[2],
            )


def test_failed_sentinel_report_short_circuits_and_never_produces_candidate() -> None:
    sentinel = _protocol_fold(
        _FAMILIES[0],
        expanded=False,
        winner_objective=1.1,
        mirror_objective=1.0,
    )
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    report = preflight.build_finite_microstep_preflight_report(
        **_report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)
    )
    assert report["classification"] == "finite_microstep_no_descent_interval_sentinel"
    assert report["passed"] is False
    assert report["nested_v20b_authorized"] is False
    assert report["candidate"] is None
    assert report["provider_sidecar"] is None
    assert report["artifact"]["scalar_and_hash_only"] is True
    assert report["artifact"]["file_mode"] == "0600"
    assert report["artifact"]["provider_tensor_sidecar"] is False

    with pytest.raises(ValueError, match="short-circuit|failed sentinel"):
        preflight.build_finite_microstep_preflight_report(
            **_report_kwargs(
                sentinel=sentinel,
                folds=(sentinel, sentinel),
                work=work,
            )
        )


def test_passing_sentinel_requires_full_unique_expansion_and_recomputes_macro() -> None:
    sentinel, folds = _expanded_case()
    work = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=8,
    )
    report = preflight.build_finite_microstep_preflight_report(
        **_report_kwargs(sentinel=sentinel, folds=folds, work=work)
    )
    assert report["fit_macro"]["checkpoint_zero_teacher_kl"] == 1.0
    assert report["fit_macro"]["selected_positive_teacher_kl"] == 0.8
    assert report["fit_macro"]["relative_improvement"] == pytest.approx(0.2)
    assert report["classification"] == (
        "finite_microstep_preflight_passed_for_nested_validation"
    )
    assert report["passed"] is True
    assert report["nested_v20b_authorized"] is True
    assert report["candidate"] is None
    assert report["provider_sidecar"] is None

    duplicate = (folds[0],) * 8
    with pytest.raises(ValueError, match="all eight|expanded fold"):
        preflight.build_finite_microstep_preflight_report(
            **_report_kwargs(sentinel=sentinel, folds=duplicate, work=work)
        )


def test_report_classifies_direction_ambiguity_and_below_materiality() -> None:
    work = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=8,
    )
    sentinel, ambiguous = _expanded_case(
        mirror_objectives=(0.9, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9)
    )
    report = preflight.build_finite_microstep_preflight_report(
        **_report_kwargs(sentinel=sentinel, folds=ambiguous, work=work)
    )
    assert report["classification"] == "finite_microstep_direction_ambiguous"

    tiny_sentinel, tiny = _expanded_case(
        winner_objective=0.995,
        mirror_objectives=(0.999,) * 8,
    )
    report = preflight.build_finite_microstep_preflight_report(
        **_report_kwargs(sentinel=tiny_sentinel, folds=tiny, work=work)
    )
    assert report["classification"] == "finite_microstep_descent_below_materiality"
    assert report["passed"] is False


def test_report_recomputes_decision_floor_and_work_instead_of_trusting_copies() -> None:
    sentinel = _protocol_fold(
        _FAMILIES[0],
        expanded=False,
        winner_objective=1.1,
        mirror_objective=1.0,
    )
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    base = _report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)

    forged_decision = copy.deepcopy(base)
    forged_decision["sentinel"]["qualification"]["passed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="decision arithmetic"):
        preflight.build_finite_microstep_preflight_report(**forged_decision)

    forged_floor = copy.deepcopy(base)
    qualification = forged_floor["sentinel"]["qualification"]  # type: ignore[index]
    qualification["objective_numerical_improvement_floor"] *= 2  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="numerical floor|decision arithmetic"):
        preflight.build_finite_microstep_preflight_report(**forged_floor)

    forged_work = copy.deepcopy(base)
    forged_work["work"]["full_model_forward_count"] += 1  # type: ignore[index,operator]
    with pytest.raises(ValueError, match="work ledger"):
        preflight.build_finite_microstep_preflight_report(**forged_work)


def test_report_enforces_exact_grid_and_authenticates_execution_receipts() -> None:
    sentinel = _protocol_fold(_FAMILIES[0], expanded=False)
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    base = _report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)

    incomplete = copy.deepcopy(base)
    rows = incomplete["sentinel"]["positive_candidates"]  # type: ignore[index]
    incomplete["sentinel"]["positive_candidates"] = rows[:-1]  # type: ignore[index]
    with pytest.raises(ValueError, match="exhaustive"):
        preflight.build_finite_microstep_preflight_report(**incomplete)

    forged_execution = copy.deepcopy(base)
    mirror_change = forged_execution["sentinel"]["matched_negative"][  # type: ignore[index]
        "execution_change"
    ]
    mirror_change["h4_changed_prompt_count"] = 0  # type: ignore[index]
    with pytest.raises(ValueError, match="execution-change receipt drifted"):
        preflight.build_finite_microstep_preflight_report(**forged_execution)


def test_report_rejects_held_scoring_and_capability_authority_drift() -> None:
    sentinel = _protocol_fold(_FAMILIES[0], expanded=False)
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    base = _report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)

    held_scored = copy.deepcopy(base)
    held_scored["sentinel"]["held_scoring_performed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="consumed held score"):
        preflight.build_finite_microstep_preflight_report(**held_scored)

    not_excluded = copy.deepcopy(base)
    not_excluded["sentinel"]["capability_receipt"][  # type: ignore[index]
        "held_family_capability_excluded"
    ] = False
    with pytest.raises(ValueError, match="capability authority"):
        preflight.build_finite_microstep_preflight_report(**not_excluded)

    wrong_artifact = copy.deepcopy(base)
    wrong_artifact["sentinel"]["capability_receipt"][  # type: ignore[index]
        "artifact_sha256"
    ] = "e" * 64
    with pytest.raises(ValueError, match="authenticated V19 authority"):
        preflight.build_finite_microstep_preflight_report(**wrong_artifact)

    wrong_authorized_key = copy.deepcopy(base)
    counts = wrong_authorized_key["sentinel"]["capability_receipt"][  # type: ignore[index]
        "per_example_access_counts"
    ]
    count = counts.pop(next(iter(counts)))  # type: ignore[union-attr]
    counts["held-family/example-forbidden"] = count  # type: ignore[index]
    with pytest.raises(ValueError, match="authenticated V19 authority|fit authority"):
        preflight.build_finite_microstep_preflight_report(**wrong_authorized_key)


def test_report_rejects_ownership_and_authenticated_endpoint_drift() -> None:
    sentinel = _protocol_fold(_FAMILIES[0], expanded=False)
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    base = _report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)

    forged_ownership = copy.deepcopy(base)
    ownership = forged_ownership["sentinel"]["ownership_receipt"]  # type: ignore[index]
    ownership["held_sequence_sha256s"] = (_digest(9_500), _digest(9_501))  # type: ignore[index]
    _rehash_ownership(ownership)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="authenticated V19"):
        preflight.build_finite_microstep_preflight_report(**forged_ownership)

    forged_pin = copy.deepcopy(base)
    endpoint = forged_pin["sentinel"]["endpoint_binding"]  # type: ignore[index]
    endpoint["pinned_v19_optimization_receipt_sha256"] = "e" * 64  # type: ignore[index]
    _rehash_endpoint(endpoint)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="optimization receipt pin"):
        preflight.build_finite_microstep_preflight_report(**forged_pin)

    self_consistent_lineage = copy.deepcopy(base)
    endpoint = self_consistent_lineage["sentinel"]["endpoint_binding"]  # type: ignore[index]
    endpoint["parent_provider_artifact_sha256"] = "e" * 64  # type: ignore[index]
    endpoint["pinned_parent_provider_artifact_sha256"] = "e" * 64  # type: ignore[index]
    _rehash_endpoint(endpoint)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lineage.*authenticated V19"):
        preflight.build_finite_microstep_preflight_report(
            **self_consistent_lineage
        )


def test_report_rejects_forged_prerequisite_panel_bridge_and_collection() -> None:
    sentinel = _protocol_fold(_FAMILIES[0], expanded=False)
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    base = _report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)

    forged_prerequisite = copy.deepcopy(base)
    forged_prerequisite["prerequisite"][  # type: ignore[index]
        "fisher_optimization_receipt_sha256s"
    ][_FAMILIES[0]] = "e" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="prerequisite differs"):
        preflight.build_finite_microstep_preflight_report(
            **forged_prerequisite
        )

    wrong_panel = copy.deepcopy(base)
    wrong_panel["panel"]["family_count"] = 7  # type: ignore[index]
    with pytest.raises(ValueError, match="panel differs"):
        preflight.build_finite_microstep_preflight_report(**wrong_panel)

    wrong_bridge = copy.deepcopy(base)
    wrong_bridge["bridge_binding_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="bridge differs"):
        preflight.build_finite_microstep_preflight_report(**wrong_bridge)

    wrong_record = copy.deepcopy(base)
    records = wrong_record["fit_collection"]["record_receipt_sha256s"]  # type: ignore[index]
    records[next(iter(records))] = "e" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="fit collection differs"):
        preflight.build_finite_microstep_preflight_report(**wrong_record)

    wrong_vault = copy.deepcopy(base)
    wrong_vault["fit_collection"]["teacher_vault_receipt"][  # type: ignore[index]
        "artifact_sha256"
    ] = "e" * 64
    with pytest.raises(ValueError, match="fit collection differs"):
        preflight.build_finite_microstep_preflight_report(**wrong_vault)


def test_report_panel_authentication_normalizes_only_json_tuple_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prerequisite, bundles = _authenticated_v19_artifact()
    authenticated_panel = {
        "prompt_count": 16,
        "family_count": 8,
        "prompt_sha256s": [_digest(9_700), _digest(9_701)],
        "source_prompt_indices": [0, 3],
    }
    prerequisite["authenticated_panel"] = authenticated_panel
    monkeypatch.setattr(
        preflight,
        "_load_authenticated_v19_artifact",
        lambda: copy.deepcopy((prerequisite, bundles)),
    )
    sentinel = _protocol_fold(
        _FAMILIES[0],
        expanded=False,
        winner_objective=1.1,
        mirror_objective=1.0,
    )
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    base = _report_kwargs(sentinel=sentinel, folds=(sentinel,), work=work)
    base["prerequisite"] = copy.deepcopy(prerequisite)
    base["panel"] = {
        **authenticated_panel,
        "prompt_sha256s": tuple(authenticated_panel["prompt_sha256s"]),
        "source_prompt_indices": tuple(
            authenticated_panel["source_prompt_indices"]
        ),
    }

    report = preflight.build_finite_microstep_preflight_report(**base)

    assert report["panel"]["prompt_sha256s"] == tuple(  # type: ignore[index]
        authenticated_panel["prompt_sha256s"]
    )
    forged = copy.deepcopy(base)
    forged["panel"]["source_prompt_indices"] = (0, 4)  # type: ignore[index]
    with pytest.raises(ValueError, match="panel differs"):
        preflight.build_finite_microstep_preflight_report(**forged)


def test_sentinel_is_cross_bound_to_first_expanded_fold_joint_grid() -> None:
    sentinel, folds = _expanded_case()
    first = copy.deepcopy(folds[0])
    rows = list(first["positive_candidates"])  # type: ignore[arg-type]
    joint_indices = [
        index for index, row in enumerate(rows) if row["path"] == "joint"
    ]
    reversed_joint = [rows[index] for index in reversed(joint_indices)]
    for index, row in zip(joint_indices, reversed_joint, strict=True):
        rows[index] = row
    first["positive_candidates"] = tuple(rows)
    drifted = (first, *folds[1:])
    work = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=8,
    )
    with pytest.raises(ValueError, match="sentinel does not bind"):
        preflight.build_finite_microstep_preflight_report(
            **_report_kwargs(sentinel=sentinel, folds=drifted, work=work)
        )


def test_expansion_counts_extra_mirror_when_all_path_winner_changes() -> None:
    sentinel, folds = _expanded_case()
    sentinel["expanded_winner_reused_sentinel_mirror"] = False
    first = copy.deepcopy(folds[0])
    rows = list(first["positive_candidates"])  # type: ignore[arg-type]
    direction = next(
        row
        for row in rows
        if row["path"] == "direction_only" and row["alpha"] == 1.0e-3
    )
    direction["objective"] = 0.7
    direction["family_objectives"] = {
        family: 0.7 for family in direction["family_objectives"]
    }
    baseline = first["baseline"]
    endpoint = first["endpoint_binding"]
    mirror = _protocol_candidate(
        baseline=baseline,  # type: ignore[arg-type]
        endpoint=endpoint,  # type: ignore[arg-type]
        path="direction_only",
        alpha=-1.0e-3,
        objective=0.9,
    )
    first["positive_candidates"] = tuple(rows)
    first["matched_negative"] = mirror
    first["qualification"] = preflight.evaluate_fold_qualification(
        held_family_id=_FAMILIES[0],
        baseline_objective=1.0,
        selected_positive=direction,
        matched_negative=mirror,
    )
    first["capability_receipt"] = _capability_receipt(
        _FAMILIES[0], executions_per_prompt=24
    )
    changed_folds = (first, *folds[1:])
    work = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=9,
    )
    report = preflight.build_finite_microstep_preflight_report(
        **_report_kwargs(sentinel=sentinel, folds=changed_folds, work=work)
    )
    assert report["work_accounting"]["full_model_forward_count"] == 2_622
    assert report["work_accounting"]["mirror_candidate_execution_count"] == 9


def _run_mocked_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    sentinel_objective: float,
    expanded_winner_path: str,
) -> tuple[dict[str, object], dict[str, object], list[tuple[object, ...]]]:
    events: list[tuple[object, ...]] = []
    captured: dict[str, object] = {}

    class _Context:
        panel_receipt = {"prompt_count": 16, "family_count": 8}
        bridge = SimpleNamespace(bridge_binding_sha256=_SHA)

        def validate_immutable_inputs(self) -> None:
            events.append(("validate",))

        def close(self) -> None:
            events.append(("close",))

    class _Vault:
        def validate_integrity(self) -> None:
            events.append(("vault_validate",))

        def receipt(self) -> dict[str, object]:
            return {
                "artifact_sha256": _digest(6_100),
                "example_count": 16,
                "family_count": 8,
            }

    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(family_id=family, example_id=f"{family}/example-{index}"),
            receipt_sha256=_record_receipt_sha256s()[f"{family}/example-{index}"],
        )
        for family in _FAMILIES
        for index in range(2)
    )

    def load_v19() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
        events.append(("load_v19",))
        return copy.deepcopy(_authenticated_v19_artifact())

    def collect(_context: object) -> tuple[tuple[object, ...], object]:
        events.append(("collect",))
        return records, _Vault()

    def prepare_workspace(
        _context: object,
        _records: object,
        _vault: object,
        *,
        held_family_id: str,
        pinned_bundle: object,
    ) -> SimpleNamespace:
        assert isinstance(pinned_bundle, dict)
        events.append(("prepare_fold", held_family_id))
        return SimpleNamespace(
            held_family_id=held_family_id,
            baseline={"objective": 1.0},
        )

    def evaluate_grid(
        _context: object,
        workspace: object,
        *,
        paths: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        held = workspace.held_family_id
        events.append(("grid", held, tuple(paths)))
        rows = list(_positive_grid(paths=tuple(paths), objective=2.0))
        for row in rows:
            if row["path"] == "joint" and row["alpha"] == 1.0e-3:
                row["objective"] = sentinel_objective if held == _FAMILIES[0] else 0.8
            if (
                expanded_winner_path == "direction_only"
                and row["path"] == "direction_only"
                and row["alpha"] == 1.0e-3
            ):
                row["objective"] = 0.7
        return tuple(rows)

    def evaluate_negative(
        _context: object,
        workspace: object,
        selected: object,
    ) -> dict[str, object] | None:
        events.append(("mirror", workspace.held_family_id))
        if not isinstance(selected, dict):
            return None
        return _matched_negative(
            selected,
            objective=1.2 if sentinel_objective > 1.0 else 0.9,
        )

    def fold_payload(
        workspace: object,
        *,
        positives: object,
        matched_negative: object,
        qualification: object,
        expanded_winner_reused_sentinel_mirror: bool | None = None,
    ) -> dict[str, object]:
        rows = tuple(positives)
        events.append(("fold_payload", workspace.held_family_id, len(rows)))
        payload: dict[str, object] = {
            "held_family_id": workspace.held_family_id,
            "positive_candidates": rows,
            "matched_negative": matched_negative,
            "qualification": qualification,
            "held_scoring_performed": False,
        }
        if expanded_winner_reused_sentinel_mirror is not None:
            payload["expanded_winner_reused_sentinel_mirror"] = (
                expanded_winner_reused_sentinel_mirror
            )
        return payload

    def build_report(**kwargs: object) -> dict[str, object]:
        events.append(("report",))
        captured.update(kwargs)
        return {
            "candidate": None,
            "provider_sidecar": None,
            "work_accounting": kwargs["work"],
            "integrity": kwargs["integrity"],
            "sentinel": kwargs["sentinel"],
            "folds": kwargs["folds"],
        }

    def publish(report: dict[str, object], *, output: Path) -> dict[str, object]:
        events.append(("publish", output.name))
        assert report["candidate"] is None
        assert report["provider_sidecar"] is None
        return {**report, "report_sha256": "f" * 64}

    monkeypatch.setattr(preflight, "_is_under_local_runs", lambda _path: True)
    monkeypatch.setattr(preflight, "_load_authenticated_v19_artifact", load_v19)
    monkeypatch.setattr(
        preflight,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: (events.append(("prepare_context",)) or _Context()),
    )
    monkeypatch.setattr(
        preflight._v19,
        "_collect_fit_records_and_teacher_vault",
        collect,
    )
    monkeypatch.setattr(preflight, "_prepare_fold_workspace", prepare_workspace)
    monkeypatch.setattr(preflight, "_evaluate_positive_grid", evaluate_grid)
    monkeypatch.setattr(preflight, "_evaluate_matched_negative", evaluate_negative)
    monkeypatch.setattr(preflight, "_fold_payload", fold_payload)
    monkeypatch.setattr(
        preflight,
        "build_finite_microstep_preflight_report",
        build_report,
    )
    monkeypatch.setattr(preflight, "_publish", publish)

    result = preflight.run_gemma3_l3_l4_complete_h4_finite_microstep_preflight(
        output=tmp_path / "orchestration.json"
    )
    return result, captured, events


def test_live_orchestration_short_circuits_after_failed_joint_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, captured, events = _run_mocked_orchestration(
        monkeypatch,
        tmp_path,
        sentinel_objective=1.1,
        expanded_winner_path="joint",
    )
    assert [event[1] for event in events if event[0] == "prepare_fold"] == [
        _FAMILIES[0]
    ]
    assert [event for event in events if event[0] == "grid"] == [
        ("grid", _FAMILIES[0], ("joint",))
    ]
    assert len(captured["folds"]) == 1  # type: ignore[arg-type]
    work = captured["work"]
    assert work["full_model_forward_count"] == 158  # type: ignore[index]
    assert work["teacher_capability_access_count"] == 126  # type: ignore[index]
    assert captured["integrity"]["held_score_row_count"] == 0  # type: ignore[index]
    assert result["candidate"] is None
    assert result["provider_sidecar"] is None
    assert events.index(("prepare_fold", _FAMILIES[0])) < events.index(
        ("grid", _FAMILIES[0], ("joint",))
    )
    assert events[-2][0] == "publish"
    assert events[-1] == ("close",)


@pytest.mark.parametrize(
    ("expanded_winner_path", "expected_mirrors", "expected_forwards", "expected_accesses", "reused"),
    (
        ("joint", 8, 2_608, 2_576, True),
        ("direction_only", 9, 2_622, 2_590, False),
    ),
)
def test_live_orchestration_expands_all_folds_and_accounts_mirror_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    expanded_winner_path: str,
    expected_mirrors: int,
    expected_forwards: int,
    expected_accesses: int,
    reused: bool,
) -> None:
    result, captured, events = _run_mocked_orchestration(
        monkeypatch,
        tmp_path,
        sentinel_objective=0.8,
        expanded_winner_path=expanded_winner_path,
    )
    assert [event[1] for event in events if event[0] == "prepare_fold"] == list(
        _FAMILIES
    )
    grid_calls = [event for event in events if event[0] == "grid"]
    assert len(grid_calls) == 9
    assert grid_calls[0] == ("grid", _FAMILIES[0], ("joint",))
    assert grid_calls[1] == (
        "grid",
        _FAMILIES[0],
        ("direction_only", "pedal_only"),
    )
    assert len(captured["folds"]) == 8  # type: ignore[arg-type]
    assert captured["sentinel"][  # type: ignore[index]
        "expanded_winner_reused_sentinel_mirror"
    ] is reused
    work = captured["work"]
    assert work["positive_nonzero_candidate_count"] == 168  # type: ignore[index]
    assert work["mirror_candidate_execution_count"] == expected_mirrors  # type: ignore[index]
    assert work["full_model_forward_count"] == expected_forwards  # type: ignore[index]
    assert work["teacher_capability_access_count"] == expected_accesses  # type: ignore[index]
    assert len([event for event in events if event[0] == "mirror"]) == expected_mirrors
    assert captured["integrity"]["held_scoring_performed"] is False  # type: ignore[index]
    assert result["candidate"] is None
    assert result["provider_sidecar"] is None


def _failed_report_ready_inputs(output: Path) -> dict[str, object]:
    sentinel = _protocol_fold(
        _FAMILIES[0],
        expanded=False,
        winner_objective=1.1,
        mirror_objective=1.0,
    )
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    values = _report_kwargs(
        sentinel=sentinel,
        folds=(sentinel,),
        work=work,
        artifact_path=output,
    )
    values.pop("artifact_path")
    return values


def test_report_ready_checkpoint_is_authenticated_write_once_and_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "checkpointed-v20a.json"
    inputs = _failed_report_ready_inputs(output)

    checkpoint = preflight._publish_report_ready_checkpoint(
        output=output,
        report_inputs=inputs,
    )

    assert checkpoint == preflight._report_ready_checkpoint_path(output)
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600
    assert checkpoint.stat().st_nlink == 1
    loaded = preflight._load_report_ready_checkpoint(output=output)
    report = preflight._build_report_from_inputs(
        output=output,
        report_inputs=loaded,
    )
    assert report["classification"] == "finite_microstep_no_descent_interval_sentinel"
    with pytest.raises(FileExistsError, match="overwrite"):
        preflight._publish_report_ready_checkpoint(
            output=output,
            report_inputs=inputs,
        )

    forged = json.loads(checkpoint.read_text(encoding="utf-8"))
    forged["v19_file_sha256"] = "e" * 64
    checkpoint.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint hash drifted"):
        preflight._load_report_ready_checkpoint(output=output)


def test_runner_resumes_report_ready_checkpoint_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "resumed-v20a.json"
    preflight._publish_report_ready_checkpoint(
        output=output,
        report_inputs=_failed_report_ready_inputs(output),
    )

    def forbidden_context(**_kwargs: object) -> object:
        raise AssertionError("report-ready resume must not load Gemma")

    monkeypatch.setattr(
        preflight,
        "prepare_complete_h4_rank320_live_context",
        forbidden_context,
    )

    result = preflight.run_gemma3_l3_l4_complete_h4_finite_microstep_preflight(
        output=output,
    )

    assert output.exists()
    assert result["classification"] == "finite_microstep_no_descent_interval_sentinel"
    assert result["candidate"] is None
    assert result["provider_sidecar"] is None


def test_expanded_report_ready_checkpoint_round_trips_full_fold_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "expanded-checkpointed-v20a.json"
    sentinel, folds = _expanded_case()
    work = preflight._work_accounting(
        tested_fold_count=8,
        positive_candidate_count=168,
        mirror_candidate_count=8,
    )
    inputs = _report_kwargs(
        sentinel=sentinel,
        folds=folds,
        work=work,
        artifact_path=output,
    )
    inputs.pop("artifact_path")
    preflight._publish_report_ready_checkpoint(
        output=output,
        report_inputs=inputs,
    )

    loaded = preflight._load_report_ready_checkpoint(output=output)
    report = preflight._build_report_from_inputs(
        output=output,
        report_inputs=loaded,
    )

    assert len(report["folds"]) == 8
    assert report["work_accounting"]["full_model_forward_count"] == 2_608
    assert report["classification"] == (
        "finite_microstep_preflight_passed_for_nested_validation"
    )


def test_report_is_scalar_only_and_publish_is_write_once_0600(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = _protocol_fold(
        _FAMILIES[0],
        expanded=False,
        winner_objective=1.1,
        mirror_objective=1.0,
    )
    work = preflight._work_accounting(
        tested_fold_count=1,
        positive_candidate_count=7,
        mirror_candidate_count=1,
    )
    output = tmp_path / "v20a.json"
    monkeypatch.setattr(preflight, "_is_under_local_runs", lambda path: True)
    report = preflight.build_finite_microstep_preflight_report(
        **_report_kwargs(
            sentinel=sentinel,
            folds=(sentinel,),
            work=work,
            artifact_path=output,
        )
    )
    published = preflight._publish(report, output=output)
    assert output.exists()
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert len(published["report_sha256"]) == 64
    assert published["artifact"]["file_bytes"] == output.stat().st_size
    with pytest.raises(FileExistsError, match="overwrite"):
        preflight._publish(copy.deepcopy(report), output=output)

    leaked = copy.deepcopy(report)
    leaked["integrity"]["raw_logits"] = torch.zeros(1)
    with pytest.raises(TypeError, match="non-scalar data Tensor"):
        preflight._publish(leaked, output=tmp_path / "leaked.json")


def test_publication_rejects_candidate_sidecar_and_v19_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight, "_is_under_local_runs", lambda path: True)
    report = {
        "candidate": {"arm": "forbidden"},
        "provider_sidecar": None,
    }
    with pytest.raises(ValueError, match="cannot publish"):
        preflight._publish(report, output=tmp_path / "candidate.json")
    report = {"candidate": None, "provider_sidecar": {"path": "forbidden.pt"}}
    with pytest.raises(ValueError, match="cannot publish"):
        preflight._publish(report, output=tmp_path / "sidecar.json")
    with pytest.raises(ValueError, match="write-once V19"):
        preflight._validate_output(
            preflight._V19_OUTPUT.parent / "nested" / ".." / preflight._V19_OUTPUT.name
        )
