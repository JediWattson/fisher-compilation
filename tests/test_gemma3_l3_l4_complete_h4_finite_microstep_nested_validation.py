from __future__ import annotations

import copy
from itertools import combinations
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat

import pytest

from fisher_graph import complete_h4_fisher_nested_microstep as core
from fisher_graph import (
    gemma3_l3_l4_complete_h4_finite_microstep_nested_validation as nested,
)


_SHA = "a" * 64
_FAMILIES = tuple(f"family-{index}" for index in range(8))


def _digest(index: int) -> str:
    return f"{index:064x}"


def _panel() -> dict[str, object]:
    return core.build_nested_microstep_panel_receipt(
        {
            family: (_digest(100 + 2 * index), _digest(101 + 2 * index))
            for index, family in enumerate(_FAMILIES)
        }
    )


def _shared_fit(
    panel: dict[str, object],
    left: str,
    right: str,
    *,
    seed: int,
) -> dict[str, object]:
    return core.build_nested_microstep_shared_fit_receipt(
        panel_receipt=panel,
        excluded_family_ids=(left, right),
        base_provider_artifact_sha256=_digest(seed),
        proposal_provider_artifact_sha256=_digest(seed + 1),
        fit_protocol_sha256=_digest(seed + 2),
        fit_evidence_sha256=_digest(seed + 3),
        rank=256,
        conditional_rank=16,
        finite=True,
        pointwise_trust_passed=True,
    )


def _baseline(
    fit: dict[str, object],
    *,
    objective: float = 1.0,
    seed: int = 1_000,
) -> dict[str, object]:
    return core.build_nested_microstep_baseline_score(
        objective=objective,
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        provider_artifact_sha256=str(fit["base_provider_artifact_sha256"]),
        execution_receipt_sha256=_digest(seed),
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
    )


def _candidate(
    fit: dict[str, object],
    *,
    path: str,
    alpha: float,
    objective: float,
    seed: int,
) -> dict[str, object]:
    return core.build_nested_microstep_candidate_score(
        path=path,
        alpha=alpha,
        objective=objective,
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        provider_artifact_sha256=_digest(seed),
        microstep_receipt_sha256=_digest(seed + 1),
        execution_change_receipt_sha256=_digest(seed + 2),
        execution_changed=True,
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
    )


def _positive_grid(
    fit: dict[str, object],
    *,
    winner: tuple[str, float] = ("direction_only", 0.1),
    seed: int = 2_000,
    all_objective: float | None = None,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for path_index, path in enumerate(core.NESTED_MICROSTEP_PATHS):
        for alpha_index, alpha in enumerate(core.NESTED_MICROSTEP_POSITIVE_ALPHAS):
            objective = (
                all_objective
                if all_objective is not None
                else (0.8 if (path, alpha) == winner else 0.9)
            )
            rows.append(
                _candidate(
                    fit,
                    path=path,
                    alpha=alpha,
                    objective=objective,
                    seed=seed + 100 * path_index + 3 * alpha_index,
                )
            )
    return tuple(rows)


def _capability(
    outer: str,
    *,
    accesses: int = 22,
    example_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    ids = example_ids or (f"{outer}/inner-0", f"{outer}/inner-1")
    return {
        "artifact_sha256": _digest(8_000 + _FAMILIES.index(outer)),
        "held_family_id": outer,
        "authorized_example_count": 2,
        "authorized_family_count": 1,
        "access_count": 2 * accesses,
        "per_example_access_counts": {item: accesses for item in ids},
        "held_family_capability_excluded": True,
        "teacher_rows_consumed_only_through_capability": True,
    }


def _score_evidence(
    fit: dict[str, object],
    *,
    objective: float = 1.0,
    seed: int = 9_000,
    family: str = _FAMILIES[0],
) -> nested._ScoreEvidence:
    ids = (f"example-{seed}-0", f"example-{seed}-1")
    parameters = {
        name: _digest(seed + 10 + index)
        for index, name in enumerate(
            ("direction_left", "direction_right", "pedal_weight", "pedal_bias")
        )
    }
    h4 = {item: _digest(seed + 20 + index) for index, item in enumerate(ids)}
    logits = {item: _digest(seed + 30 + index) for index, item in enumerate(ids)}
    summary = core.build_nested_microstep_baseline_score(
        objective=objective,
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        provider_artifact_sha256=str(fit["base_provider_artifact_sha256"]),
        execution_receipt_sha256=nested._score_execution_receipt(
            provider_artifact_sha256=str(fit["base_provider_artifact_sha256"]),
            fit_receipt_sha256=str(fit["artifact_sha256"]),
            scored_family_id=family,
            h4_sha256s=h4,
            logits_sha256s=logits,
        ),
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
    )
    return nested._ScoreEvidence(
        summary=summary,
        parameter_sha256s=parameters,
        h4_sha256s=h4,
        logits_sha256s=logits,
        execution_receipt_sha256=str(summary["execution_receipt_sha256"]),
    )


def _candidate_score_evidence(
    fit: dict[str, object],
    baseline: nested._ScoreEvidence,
    *,
    path: str,
    alpha: float,
    objective: float,
    seed: int,
    family: str,
    shared_receipt: dict[str, object] | None = None,
) -> nested._ScoreEvidence:
    ids = tuple(baseline.h4_sha256s)
    if shared_receipt is None:
        parameter_sha256s = {
            name: _digest(seed + 10 + index)
            for index, name in enumerate(
                ("direction_left", "direction_right", "pedal_weight", "pedal_bias")
            )
        }
        receipt = nested.FisherFiniteMicrostepReceipt(
            base_provider_artifact_sha256=str(
                fit["base_provider_artifact_sha256"]
            ),
            proposal_provider_artifact_sha256=str(
                fit["proposal_provider_artifact_sha256"]
            ),
            selected_provider_artifact_sha256=_digest(seed),
            parameter_artifact_sha256=_digest(seed + 1),
            microstep_path=path,
            alpha=alpha,
            microstep_protocol_sha256=core.NESTED_MICROSTEP_PROTOCOL_SHA256,
            microstep_evidence_sha256=nested._pair_candidate_evidence_from_fit(
                fit,
                path=path,
                alpha=alpha,
            ),
            selected_tensor_sha256s=parameter_sha256s,
            prepared_float_scalar_count=1,
            logical_macs_per_token_upper_bound=1,
            rank=256,
            conditional_rank=16,
        ).metadata()
    else:
        receipt = dict(shared_receipt)
        parameter_sha256s = dict(receipt["selected_tensor_sha256s"])
    h4 = {item: _digest(seed + 20 + index) for index, item in enumerate(ids)}
    logits = {item: _digest(seed + 30 + index) for index, item in enumerate(ids)}
    change = nested._v20a.detect_execution_change(
        base_parameter_sha256s=baseline.parameter_sha256s,
        candidate_parameter_sha256s=parameter_sha256s,
        base_h4_sha256s=baseline.h4_sha256s,
        candidate_h4_sha256s=h4,
        base_logits_sha256s=baseline.logits_sha256s,
        candidate_logits_sha256s=logits,
    )
    summary = core.build_nested_microstep_candidate_score(
        path=path,
        alpha=alpha,
        objective=objective,
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        provider_artifact_sha256=str(receipt["selected_provider_artifact_sha256"]),
        microstep_receipt_sha256=str(receipt["artifact_sha256"]),
        execution_change_receipt_sha256=str(change["receipt_sha256"]),
        execution_changed=bool(change["execution_changed"]),
        finite=True,
        pointwise_trust_passed=True,
        rank_is_16=True,
    )
    execution_receipt = nested._score_execution_receipt(
        provider_artifact_sha256=str(receipt["selected_provider_artifact_sha256"]),
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        scored_family_id=family,
        h4_sha256s=h4,
        logits_sha256s=logits,
    )
    return nested._ScoreEvidence(
        summary=summary,
        parameter_sha256s=parameter_sha256s,
        h4_sha256s=h4,
        logits_sha256s=logits,
        execution_receipt_sha256=execution_receipt,
        execution_change=change,
        microstep_receipt=receipt,
    )


def _positive_evidence_grid(
    fit: dict[str, object],
    baseline: nested._ScoreEvidence,
    *,
    family: str,
    winner: tuple[str, float] = ("direction_only", 0.1),
    seed: int = 2_000,
    all_objective: float | None = None,
    shared_receipts: dict[tuple[str, float], dict[str, object]] | None = None,
) -> tuple[nested._ScoreEvidence, ...]:
    rows: list[nested._ScoreEvidence] = []
    for path_index, path in enumerate(core.NESTED_MICROSTEP_PATHS):
        for alpha_index, alpha in enumerate(core.NESTED_MICROSTEP_POSITIVE_ALPHAS):
            objective = (
                all_objective
                if all_objective is not None
                else (0.8 if (path, alpha) == winner else 0.9)
            )
            rows.append(
                _candidate_score_evidence(
                    fit,
                    baseline,
                    path=path,
                    alpha=alpha,
                    objective=objective,
                    seed=seed + 100 * path_index + 3 * alpha_index,
                    family=family,
                    shared_receipt=(
                        None
                        if shared_receipts is None
                        else shared_receipts[(path, alpha)]
                    ),
                )
            )
    return tuple(rows)


def _fit_training_evidence(
    fit: dict[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    families = tuple(fit["training_family_ids"])
    example_families = {
        f"{family}/fit-{index}": family
        for family in families
        for index in range(2)
    }
    ids = tuple(example_families)
    capability = {
        "artifact_sha256": _digest(seed),
        "held_family_id": None,
        "authorized_example_count": len(ids),
        "authorized_family_count": len(families),
        "access_count": len(ids),
        "per_example_access_counts": {item: 1 for item in ids},
        "held_family_capability_excluded": False,
        "teacher_rows_consumed_only_through_capability": True,
    }
    payload = {
        "fit_receipt_sha256": fit["artifact_sha256"],
        "provider_artifact_sha256": fit["base_provider_artifact_sha256"],
        "parameter_sha256s": {
            f"parameter-{index}": _digest(seed + 10 + index) for index in range(4)
        },
        "example_family_ids": example_families,
        "post_cast_h4_sha256s": {
            item: _digest(seed + 100 + index) for index, item in enumerate(ids)
        },
        "supervised_full_vocab_logits_sha256s": {
            item: _digest(seed + 200 + index) for index, item in enumerate(ids)
        },
        "capability_receipt": capability,
        "raw_tensors_or_logits_serialized": False,
    }
    return {
        **payload,
        "execution_receipt_sha256": nested._v14._sha256(
            payload,
            domain=nested._FIT_EXECUTION_DOMAIN,
        ),
    }


def _barrier_lock(panel: dict[str, object]) -> dict[str, object]:
    refits = tuple(
        {
            "outer_held_family_id": family,
            "artifact_sha256": _digest(9_500 + index),
        }
        for index, family in enumerate(_FAMILIES)
    )
    return {
        "selection_lock_sha256": _digest(9_400),
        "outer_schedule_authorized": True,
        "outer_refits": refits,
        "selection_receipt": {
            "panel_artifact_sha256": panel["artifact_sha256"],
        },
    }


def _authenticated_v20a_folds() -> dict[str, dict[str, object]]:
    return {
        family: {
            "held_family_id": family,
            "endpoint_binding": {
                "held_family_id": family,
                "base_provider_artifact_sha256": _digest(9_800 + 2 * index),
                "proposal_provider_artifact_sha256": _digest(9_801 + 2 * index),
            },
        }
        for index, family in enumerate(_FAMILIES)
    }


def _publish_adversarial_complete_lock(
    *,
    output: Path,
    panel: dict[str, object],
    attack: str,
) -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
    str,
]:
    authenticated_folds = _authenticated_v20a_folds()
    selection_rows: list[dict[str, object]] = []
    refits: list[dict[str, object]] = []
    for index, family in enumerate(_FAMILIES):
        selected = {
            "key": core.nested_microstep_candidate_key("direction_only", 0.1),
            "path": "direction_only",
            "alpha": 0.1,
        }
        if attack == "wrong_key" and index == 0:
            selected["key"] = core.nested_microstep_candidate_key("pedal_only", 0.1)
        selection_item = {
            "outer_held_family_id": family,
            "selected": selected,
            "passed": True,
            "artifact_sha256": _digest(10_000 + index),
        }
        selection_rows.append(selection_item)
        endpoint = authenticated_folds[family]["endpoint_binding"]
        base = str(endpoint["base_provider_artifact_sha256"])
        proposal = str(endpoint["proposal_provider_artifact_sha256"])
        path = "direction_only"
        alpha = 0.1
        if attack == "wrong_valid_policy" and index == 0:
            path = "pedal_only"
        if attack == "replaced_v20a_endpoints" and index == 0:
            base = _digest(10_100)
            proposal = _digest(10_101)
        refit: dict[str, object] = {
            "outer_held_family_id": family,
            "selection_artifact_sha256": selection_item["artifact_sha256"],
            "selected_path": path,
            "selected_alpha": alpha,
            "outer_fit_receipt": {},
            "fit_training_evidence": {},
            "base_provider_artifact_sha256": base,
            "proposal_provider_artifact_sha256": proposal,
            "positive_provider_artifact_sha256": _digest(10_200 + 4 * index),
            "positive_microstep_receipt_sha256": _digest(10_201 + 4 * index),
            "positive_microstep_receipt": {},
            "mirror_provider_artifact_sha256": _digest(10_202 + 4 * index),
            "mirror_microstep_receipt_sha256": _digest(10_203 + 4 * index),
            "mirror_microstep_receipt": {},
            "runtime_flags": {},
            "v20a_base_endpoint_sha256": base,
            "v20a_proposal_endpoint_sha256": proposal,
            "outer_scoring_capability_created": False,
        }
        refit["artifact_sha256"] = nested._v14._sha256(
            refit,
            domain=nested._FULL_FIT_DOMAIN,
        )
        refits.append(refit)
    selection_receipt = {
        "passed": True,
        "outer_validation_authorized": True,
        "outer_selections": tuple(selection_rows),
    }
    pair_fragments = {
        f"pair-{index}": {"fragment_sha256": _digest(10_500 + index)}
        for index in range(28)
    }
    inner_fragments = {
        family: {"fragment_sha256": _digest(10_600 + index)}
        for index, family in enumerate(_FAMILIES)
    }
    binding = _digest(10_700)
    lock = nested._publish_scalar_fragment(
        {
            "schema": nested._SELECTION_LOCK_SCHEMA,
            "format_version": nested._FORMAT_VERSION,
            "target_output": nested._validate_output(output).as_posix(),
            "runner_protocol_sha256": nested._RUNNER_PROTOCOL_SHA256,
            "core_protocol_sha256": core.NESTED_MICROSTEP_PROTOCOL_SHA256,
            "v20a_report_sha256": nested._V20A_LOGICAL_SHA256,
            "v20a_file_sha256": nested._V20A_FILE_SHA256,
            "panel_binding_sha256": binding,
            "bridge_binding_sha256": _digest(10_701),
            "pair_fragment_sha256s": {
                key: fragment["fragment_sha256"]
                for key, fragment in sorted(pair_fragments.items())
            },
            "inner_fragment_sha256s": {
                key: fragment["fragment_sha256"]
                for key, fragment in sorted(inner_fragments.items())
            },
            "selection_receipt": selection_receipt,
            "outer_refits": tuple(refits),
            "outer_schedule_authorized": True,
            "resume_overhead": nested._resume_overhead(
                pair_endpoint_reconstruction_count=0,
                outer_endpoint_reconstruction_count=0,
            ),
            "candidate": None,
            "provider_sidecar": None,
        },
        path=nested._selection_lock_path(output),
        domain=nested._SELECTION_LOCK_DOMAIN,
        hash_key="selection_lock_sha256",
        label="adversarial complete V20b selection lock",
    )
    return lock, pair_fragments, inner_fragments, authenticated_folds, binding


def test_scalar_fragment_is_write_once_0600_and_rejects_hash_and_link_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fragment.json"
    payload = {"schema": "test", "value": 3, "candidate": None}
    published = nested._publish_scalar_fragment(
        payload,
        path=path,
        domain=b"test-domain\0",
        hash_key="sha256",
        label="test fragment",
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_nlink == 1
    assert nested._load_scalar_fragment(
        path=path,
        domain=b"test-domain\0",
        hash_key="sha256",
        label="test fragment",
    ) == published
    with pytest.raises(FileExistsError, match="overwrite"):
        nested._publish_scalar_fragment(
            payload,
            path=path,
            domain=b"test-domain\0",
            hash_key="sha256",
            label="test fragment",
        )

    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["value"] = 4
    path.write_text(json.dumps(forged), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="hash drifted"):
        nested._load_scalar_fragment(
            path=path,
            domain=b"test-domain\0",
            hash_key="sha256",
            label="test fragment",
        )

    other = tmp_path / "linked.json"
    os.link(path, other)
    with pytest.raises(RuntimeError, match="unsafe"):
        nested._secure_stat(path, label="linked fragment")


def test_outer_capability_reauthenticates_complete_lock_immediately_before_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    panel = _panel()
    lock = _barrier_lock(panel)
    binding = _digest(9_600)
    pair_fragments = {"pair": {"fragment_sha256": _digest(9_601)}}
    inner_fragments = {"inner": {"fragment_sha256": _digest(9_602)}}
    output = tmp_path / "v20b.json"

    class _Vault:
        def capability(self, ids: object, *, held_family_id: object) -> object:
            events.append(("capability", tuple(ids), held_family_id))
            return object()

    def load_lock(**kwargs: object) -> dict[str, object]:
        events.append(
            (
                "load",
                kwargs["output"],
                kwargs["panel_binding_sha256"],
                kwargs["pair_fragments"],
                kwargs["inner_fragments"],
                kwargs["panel_receipt"],
            )
        )
        return copy.deepcopy(lock)

    monkeypatch.setattr(nested, "_load_selection_lock", load_lock)
    nested._outer_capability_after_selection_lock(
        _Vault(),
        ("a", "b"),
        output=output,
        expected_selection_lock=lock,
        expected_outer_family_id=_FAMILIES[0],
        expected_refit_receipt=lock["outer_refits"][0],
        panel_binding_sha256=binding,
        pair_fragments=pair_fragments,
        inner_fragments=inner_fragments,
        panel_receipt=panel,
        authenticated_v20a_folds=_authenticated_v20a_folds(),
    )
    assert events == [
        (
            "load",
            output,
            binding,
            pair_fragments,
            inner_fragments,
            panel,
        ),
        ("capability", ("a", "b"), None),
    ]


def test_outer_capability_rejects_empty_selection_lock_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    panel = _panel()
    expected = _barrier_lock(panel)
    output = tmp_path / "v20b.json"
    lock_path = nested._selection_lock_path(output)
    lock_path.write_text("{}", encoding="utf-8")
    lock_path.chmod(0o600)

    class _Vault:
        def capability(self, _ids: object, *, held_family_id: object) -> object:
            raise AssertionError(f"capability issued with held={held_family_id}")

    with pytest.raises(ValueError, match="fields differ"):
        nested._outer_capability_after_selection_lock(
            _Vault(),
            ("a", "b"),
            output=output,
            expected_selection_lock=expected,
            expected_outer_family_id=_FAMILIES[0],
            expected_refit_receipt=expected["outer_refits"][0],
            panel_binding_sha256=_digest(9_610),
            pair_fragments={},
            inner_fragments={},
            panel_receipt=panel,
            authenticated_v20a_folds=_authenticated_v20a_folds(),
        )


def test_outer_capability_rejects_self_hashed_lock_with_wrong_authority_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    panel = _panel()
    expected = _barrier_lock(panel)
    output = tmp_path / "v20b.json"
    binding = _digest(9_620)
    nested._publish_scalar_fragment(
        {
            "schema": nested._SELECTION_LOCK_SCHEMA,
            "format_version": nested._FORMAT_VERSION,
            "target_output": nested._validate_output(output).as_posix(),
            "runner_protocol_sha256": _digest(9_621),
            "core_protocol_sha256": core.NESTED_MICROSTEP_PROTOCOL_SHA256,
            "v20a_report_sha256": nested._V20A_LOGICAL_SHA256,
            "v20a_file_sha256": nested._V20A_FILE_SHA256,
            "panel_binding_sha256": binding,
            "bridge_binding_sha256": _digest(9_622),
            "pair_fragment_sha256s": {},
            "inner_fragment_sha256s": {},
            "selection_receipt": {},
            "outer_refits": (),
            "outer_schedule_authorized": False,
            "resume_overhead": nested._resume_overhead(
                pair_endpoint_reconstruction_count=0,
                outer_endpoint_reconstruction_count=0,
            ),
            "candidate": None,
            "provider_sidecar": None,
        },
        path=nested._selection_lock_path(output),
        domain=nested._SELECTION_LOCK_DOMAIN,
        hash_key="selection_lock_sha256",
        label="adversarial V20b selection lock",
    )

    class _Vault:
        def capability(self, _ids: object, *, held_family_id: object) -> object:
            raise AssertionError(f"capability issued with held={held_family_id}")

    with pytest.raises(ValueError, match="authority differs"):
        nested._outer_capability_after_selection_lock(
            _Vault(),
            ("a", "b"),
            output=output,
            expected_selection_lock=expected,
            expected_outer_family_id=_FAMILIES[0],
            expected_refit_receipt=expected["outer_refits"][0],
            panel_binding_sha256=binding,
            pair_fragments={},
            inner_fragments={},
            panel_receipt=panel,
            authenticated_v20a_folds=_authenticated_v20a_folds(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("replacement", "was replaced"),
        ("truncated_refits", "all eight locked refits"),
        ("wrong_refit", "refit differs"),
    ),
)
def test_outer_capability_rejects_lock_replacement_truncation_and_wrong_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    panel = _panel()
    expected = _barrier_lock(panel)
    disk = copy.deepcopy(expected)
    if mutation == "replacement":
        disk["selection_lock_sha256"] = _digest(9_700)
    elif mutation == "truncated_refits":
        disk["outer_refits"] = disk["outer_refits"][:-1]
    else:
        disk["outer_refits"][0]["artifact_sha256"] = _digest(9_701)

    monkeypatch.setattr(
        nested,
        "_load_selection_lock",
        lambda **_kwargs: copy.deepcopy(disk),
    )

    class _Vault:
        def capability(self, _ids: object, *, held_family_id: object) -> object:
            raise AssertionError(f"capability issued with held={held_family_id}")

    with pytest.raises(ValueError, match=message):
        nested._outer_capability_after_selection_lock(
            _Vault(),
            ("a", "b"),
            output=tmp_path / "v20b.json",
            expected_selection_lock=expected,
            expected_outer_family_id=_FAMILIES[0],
            expected_refit_receipt=expected["outer_refits"][0],
            panel_binding_sha256=_digest(9_710),
            pair_fragments={},
            inner_fragments={},
            panel_receipt=panel,
            authenticated_v20a_folds=_authenticated_v20a_folds(),
        )


@pytest.mark.parametrize(
    ("attack", "message"),
    (
        ("wrong_valid_policy", "policy differs from frozen selection"),
        ("wrong_key", "key differs from frozen selection"),
        ("replaced_v20a_endpoints", "authenticated V20a endpoints"),
    ),
)
def test_complete_self_hashed_lock_rejects_policy_and_v20a_endpoint_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    message: str,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    panel = _panel()
    output = tmp_path / f"{attack}.json"
    lock, pairs, inners, authenticated_folds, binding = (
        _publish_adversarial_complete_lock(
            output=output,
            panel=panel,
            attack=attack,
        )
    )
    monkeypatch.setattr(
        nested,
        "_build_selection_receipt",
        lambda **_kwargs: copy.deepcopy(lock["selection_receipt"]),
    )

    class _Vault:
        def capability(self, _ids: object, *, held_family_id: object) -> object:
            raise AssertionError(f"capability issued with held={held_family_id}")

    with pytest.raises(ValueError, match=message):
        nested._outer_capability_after_selection_lock(
            _Vault(),
            ("a", "b"),
            output=output,
            expected_selection_lock=lock,
            expected_outer_family_id=_FAMILIES[0],
            expected_refit_receipt=lock["outer_refits"][0],
            panel_binding_sha256=binding,
            pair_fragments=pairs,
            inner_fragments=inners,
            panel_receipt=panel,
            authenticated_v20a_folds=authenticated_folds,
        )


def test_pair_fragment_rejects_outer_leakage_even_after_self_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "v20b.json"
    panel = _panel()
    fit = _shared_fit(panel, _FAMILIES[0], _FAMILIES[1], seed=10)
    workspace = SimpleNamespace(
        excluded_family_ids=(_FAMILIES[0], _FAMILIES[1]),
        fit_receipt=fit,
        fit_training_evidence=_fit_training_evidence(fit, seed=9_900),
    )
    base0 = _score_evidence(fit, seed=10_000, family=_FAMILIES[1])
    base1 = _score_evidence(fit, seed=11_000, family=_FAMILIES[0])
    grid0 = _positive_evidence_grid(
        fit, base0, family=_FAMILIES[1], seed=12_000
    )
    # Reciprocal roles must share provider/microstep artifacts at each key;
    # only objectives and execution receipts are role-specific.
    shared_receipts = {
        (str(row.summary["path"]), float(row.summary["alpha"])): dict(
            row.microstep_receipt or {}
        )
        for row in grid0
    }
    grid1 = _positive_evidence_grid(
        fit,
        base1,
        family=_FAMILIES[0],
        seed=20_000,
        shared_receipts=shared_receipts,
    )
    role0 = nested._role_panel(
        outer_family_id=_FAMILIES[0],
        inner_family_id=_FAMILIES[1],
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        baseline=base0,
        positive_candidates=grid0,
        capability_receipt=_capability(
            _FAMILIES[0], example_ids=tuple(base0.h4_sha256s)
        ),
    )
    role1 = nested._role_panel(
        outer_family_id=_FAMILIES[1],
        inner_family_id=_FAMILIES[0],
        fit_receipt_sha256=str(fit["artifact_sha256"]),
        baseline=base1,
        positive_candidates=grid1,
        capability_receipt=_capability(
            _FAMILIES[1], example_ids=tuple(base1.h4_sha256s)
        ),
    )
    binding = _digest(30_000)
    nested._publish_pair_fragment(
        output=output,
        panel_binding_sha256=binding,
        workspace=workspace,
        directed_panels=(role0, role1),
    )
    loaded = nested._load_pair_fragment(
        output=output,
        panel_binding_sha256=binding,
        excluded_family_ids=(_FAMILIES[0], _FAMILIES[1]),
        panel_receipt=panel,
    )
    assert len(loaded["directed_panels"]) == 2

    path = nested._pair_fragment_path(output, str(fit["fit_key"]))
    original = json.loads(path.read_text(encoding="utf-8"))
    forged = copy.deepcopy(original)
    candidate_h4 = forged["directed_panels"][0][
        "positive_candidate_evidence"
    ][0]["post_cast_h4_sha256s"]
    candidate_h4[next(iter(candidate_h4))] = _digest(99_000)
    forged.pop("fragment_sha256")
    forged["fragment_sha256"] = nested._v14._sha256(
        forged,
        domain=nested._PAIR_FRAGMENT_DOMAIN,
    )
    path.write_text(json.dumps(forged), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="score execution receipt drifted"):
        nested._load_pair_fragment(
            output=output,
            panel_binding_sha256=binding,
            excluded_family_ids=(_FAMILIES[0], _FAMILIES[1]),
            panel_receipt=panel,
        )

    forged = copy.deepcopy(original)
    forged["directed_panels"][0]["outer_rows_consumed"] = True
    forged.pop("fragment_sha256")
    forged["fragment_sha256"] = nested._v14._sha256(
        forged,
        domain=nested._PAIR_FRAGMENT_DOMAIN,
    )
    path.write_text(json.dumps(forged), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="capability or phase boundary"):
        nested._load_pair_fragment(
            output=output,
            panel_binding_sha256=binding,
            excluded_family_ids=(_FAMILIES[0], _FAMILIES[1]),
            panel_receipt=panel,
        )


def test_pre_mirror_selector_is_outer_disjoint_and_baseline_wins_exact_tie() -> None:
    panel = _panel()
    outer = _FAMILIES[0]
    shared_by_sha: dict[str, dict[str, object]] = {}
    shared_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for pair_index, (left, right) in enumerate(combinations(_FAMILIES, 2)):
        fit = _shared_fit(panel, left, right, seed=100 + 10 * pair_index)
        shared_by_sha[str(fit["artifact_sha256"])] = fit
        shared_by_pair[(left, right)] = fit
    panels: list[dict[str, object]] = []
    for index, inner in enumerate(_FAMILIES[1:]):
        fit = shared_by_pair[tuple(sorted((outer, inner)))]
        base = _score_evidence(
            fit,
            objective=0.8,
            seed=30_000 + 100 * index,
            family=inner,
        )
        rows = _positive_evidence_grid(
            fit,
            base,
            family=inner,
            seed=40_000 + 1_000 * index,
            all_objective=0.8,
        )
        panels.append(
            nested._role_panel(
                outer_family_id=outer,
                inner_family_id=inner,
                fit_receipt_sha256=str(fit["artifact_sha256"]),
                baseline=base,
                positive_candidates=rows,
                capability_receipt=_capability(
                    outer, example_ids=tuple(base.h4_sha256s)
                ),
            )
        )

    selected = nested._select_outer_policy(
        outer_family_id=outer,
        panels=panels,
        panel_receipt=panel,
        shared_fit_receipts=shared_by_sha,
    )
    assert selected["selected"] is None

    leaked = copy.deepcopy(panels)
    leaked[0]["outer_held_family_id"] = _FAMILIES[2]
    with pytest.raises(ValueError, match="role ownership"):
        nested._select_outer_policy(
            outer_family_id=outer,
            panels=leaked,
            panel_receipt=panel,
            shared_fit_receipts=shared_by_sha,
        )


def test_report_ready_resume_does_not_create_model_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "resumed-v20b.json"
    prerequisite = {"nested_panel_receipt": {"artifact_sha256": _SHA}}
    inputs = {
        "prerequisite": prerequisite,
        "panel_receipt": {"artifact_sha256": _SHA},
        "panel_binding_sha256": _digest(50_000),
        "bridge_binding_sha256": _digest(50_001),
        "pair_fragments": tuple({"pair": index} for index in range(28)),
        "inner_fragments": tuple({"outer": index} for index in range(8)),
        "selection_lock": {"selection_lock_sha256": _digest(50_002)},
        "outer_fragments": (),
        "resume_overhead": nested._resume_overhead(
            pair_endpoint_reconstruction_count=0,
            outer_endpoint_reconstruction_count=0,
        ),
        "integrity": {},
    }
    nested._publish_report_ready_checkpoint(output=output, report_inputs=inputs)
    events: list[str] = []

    def authenticate() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        events.append("authenticate")
        return prerequisite, {}, {}

    def forbidden_context(**_kwargs: object) -> object:
        raise AssertionError("report-ready resume must not load Gemma")

    monkeypatch.setattr(nested, "_load_authenticated_v20a_artifact", authenticate)
    monkeypatch.setattr(
        nested,
        "prepare_complete_h4_rank320_live_context",
        forbidden_context,
    )
    monkeypatch.setattr(
        nested,
        "_build_report_from_inputs",
        lambda **_kwargs: {
            "artifact": {"path": output.as_posix()},
            "candidate": None,
            "provider_sidecar": None,
            "classification": "synthetic",
        },
    )
    monkeypatch.setattr(
        nested,
        "_publish_report",
        lambda report, **_kwargs: {**report, "report_sha256": _SHA},
    )

    result = nested.run_gemma3_l3_l4_complete_h4_finite_microstep_nested_validation(
        output=output
    )
    assert events == ["authenticate"]
    assert result["classification"] == "synthetic"


def test_complete_graph_reload_uses_fresh_authenticated_v20a_folds_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    output = tmp_path / "complete-v20b.json"
    lock_path = nested._selection_lock_path(output)
    lock_path.write_text("{}", encoding="utf-8")
    lock_path.chmod(0o600)
    panel = _panel()
    prerequisite = {
        "nested_panel_receipt": panel,
        "authenticated_bridge_binding_sha256": _digest(55_000),
    }
    authenticated_folds = _authenticated_v20a_folds()
    events: list[str] = []

    def authenticate() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        events.append("fresh_v20a")
        return prerequisite, {}, authenticated_folds

    def load_graph(**kwargs: object) -> tuple[object, ...]:
        assert kwargs["authenticated_v20a_folds"] is authenticated_folds
        events.append("complete_graph")
        return (
            {},
            {},
            {"outer_schedule_authorized": False},
            {},
            _digest(55_001),
            _digest(55_000),
        )

    monkeypatch.setattr(nested, "_load_authenticated_v20a_artifact", authenticate)
    monkeypatch.setattr(nested, "_load_checkpoint_graph", load_graph)
    monkeypatch.setattr(
        nested,
        "_finalize_checkpoint_graph",
        lambda **_kwargs: {"classification": "fresh-auth-no-model"},
    )
    monkeypatch.setattr(
        nested,
        "prepare_complete_h4_rank320_live_context",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("complete graph must not create a model")
        ),
    )

    result = nested.run_gemma3_l3_l4_complete_h4_finite_microstep_nested_validation(
        output=output
    )
    assert events == ["fresh_v20a", "complete_graph"]
    assert result["classification"] == "fresh-auth-no-model"


def test_real_v20a_authentication_accepts_a_passing_prerequisite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "v20a.json"
    folds = tuple(
        {
            "held_family_id": family,
            "ownership_receipt": {
                "held_sequence_sha256s": (
                    _digest(70_000 + 2 * index),
                    _digest(70_001 + 2 * index),
                )
            },
        }
        for index, family in enumerate(_FAMILIES)
    )
    logical_payload: dict[str, object] = {
        "schema": nested._v20a._SCHEMA,
        "format_version": 20,
        "classification": nested._V20A_CLASSIFICATION,
        "passed": True,
        "nested_v20b_authorized": True,
        "candidate": None,
        "provider_sidecar": None,
        "panel": {"artifact_sha256": _digest(71_000)},
        "bridge_binding_sha256": _digest(71_001),
        "prerequisite": {"artifact_sha256": _digest(71_002)},
        "fit_collection": {"artifact_sha256": _digest(71_003)},
        "sentinel": {"artifact_sha256": _digest(71_004)},
        "folds": folds,
        "work_accounting": {"full_model_forward_count": 1},
        "integrity": {"held_scoring_performed": False},
    }
    logical_sha = nested._v14._sha256(
        logical_payload,
        domain=nested._v20a._REPORT_DOMAIN,
    )
    document = {**logical_payload, "report_sha256": logical_sha}
    report_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    report_path.chmod(0o600)
    monkeypatch.setattr(nested, "_V20A_OUTPUT", report_path)
    monkeypatch.setattr(nested, "_V20A_LOGICAL_SHA256", logical_sha)
    monkeypatch.setattr(
        nested,
        "_V20A_FILE_SHA256",
        nested._v14._file_sha256(report_path),
    )
    monkeypatch.setattr(
        nested._v20a,
        "build_finite_microstep_preflight_report",
        lambda **_kwargs: dict(logical_payload),
    )
    monkeypatch.setattr(
        nested._v18,
        "_validate_prerequisite_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the V16 false-result validator must not handle V20a")
        ),
    )

    prerequisite, authenticated, by_family = (
        nested._load_authenticated_v20a_artifact()
    )

    assert prerequisite["passed"] is True
    assert prerequisite["report_rebuilt_before_model_work"] is True
    assert authenticated["report_sha256"] == logical_sha
    assert set(by_family) == set(_FAMILIES)


def test_live_path_authenticates_and_rebuilds_v20a_before_model_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    events: list[str] = []
    prerequisite = {"nested_panel_receipt": {"artifact_sha256": _SHA}}

    def authenticate() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        events.append("v20a_rebuilt")
        return prerequisite, {}, {}

    def context(**_kwargs: object) -> object:
        events.append("model_created")
        raise RuntimeError("stop after phase-order proof")

    monkeypatch.setattr(nested, "_load_authenticated_v20a_artifact", authenticate)
    monkeypatch.setattr(nested, "prepare_complete_h4_rank320_live_context", context)
    with pytest.raises(RuntimeError, match="phase-order proof"):
        nested.run_gemma3_l3_l4_complete_h4_finite_microstep_nested_validation(
            output=tmp_path / "phase-v20b.json"
        )
    assert events == ["v20a_rebuilt", "model_created"]


def test_work_and_resume_overhead_do_not_double_count_retries() -> None:
    work = core.nested_microstep_work_accounting(outer_scored=True)
    assert work["physical_shared_pair_fit_count"] == 28


def _synthetic_authenticated_access_graph() -> tuple[object, ...]:
    example_families = {
        f"{family}/example-{index}": family
        for family in _FAMILIES
        for index in range(2)
    }
    teacher_hashes = {
        example: _digest(110_000 + index)
        for index, example in enumerate(sorted(example_families))
    }

    def capability(
        example_ids: tuple[str, ...],
        *,
        held: str | None,
        accesses: int,
    ) -> dict[str, object]:
        families = {example_families[example] for example in example_ids}
        artifact = nested._v14._sha256(
            {
                "authorized_example_ids": tuple(sorted(example_ids)),
                "authorized_family_ids": tuple(sorted(families)),
                "held_family_id": held,
                "teacher_row_sha256s": {
                    example: teacher_hashes[example]
                    for example in sorted(example_ids)
                },
            },
            domain=nested._v19._CAPABILITY_DOMAIN,
        )
        return {
            "artifact_sha256": artifact,
            "held_family_id": held,
            "authorized_example_count": len(example_ids),
            "authorized_family_count": len(families),
            "access_count": len(example_ids) * accesses,
            "per_example_access_counts": {
                example: accesses for example in sorted(example_ids)
            },
            "held_family_capability_excluded": held is not None,
            "teacher_rows_consumed_only_through_capability": True,
        }

    def hashes(example_ids: tuple[str, ...], seed: int) -> dict[str, str]:
        return {
            example: _digest(seed + index)
            for index, example in enumerate(sorted(example_ids))
        }

    def fit_evidence(
        example_ids: tuple[str, ...],
        *,
        held: str | None,
        seed: int,
    ) -> dict[str, object]:
        return {
            "example_family_ids": {
                example: example_families[example] for example in example_ids
            },
            "post_cast_h4_sha256s": hashes(example_ids, seed),
            "supervised_full_vocab_logits_sha256s": hashes(
                example_ids, seed + 100
            ),
            "capability_receipt": capability(
                example_ids,
                held=held,
                accesses=1,
            ),
        }

    def score_evidence(example_ids: tuple[str, ...], seed: int) -> dict[str, object]:
        return {
            "post_cast_h4_sha256s": hashes(example_ids, seed),
            "supervised_full_vocab_logits_sha256s": hashes(
                example_ids, seed + 100
            ),
        }

    pair_fragments: dict[str, dict[str, object]] = {}
    seed = 120_000
    for left, right in combinations(_FAMILIES, 2):
        training_ids = tuple(
            example
            for example, family in example_families.items()
            if family not in {left, right}
        )
        directed = []
        for outer, inner in ((left, right), (right, left)):
            ids = tuple(
                example
                for example, family in example_families.items()
                if family == inner
            )
            directed.append(
                {
                    "inner_held_family_id": inner,
                    "baseline_h4_sha256s": hashes(ids, seed),
                    "baseline_logits_sha256s": hashes(ids, seed + 10),
                    "positive_candidate_evidence": tuple(
                        score_evidence(ids, seed + 20 + 10 * index)
                        for index in range(21)
                    ),
                    "capability_receipt": capability(
                        ids,
                        held=outer,
                        accesses=22,
                    ),
                }
            )
            seed += 300
        pair_fragments[f"{left}:{right}"] = {
            "fit_training_evidence": fit_evidence(
                training_ids,
                held=None,
                seed=seed,
            ),
            "directed_panels": tuple(directed),
        }
        seed += 300

    inner_fragments: dict[str, dict[str, object]] = {}
    for outer in _FAMILIES:
        mirrors = []
        for inner in _FAMILIES:
            if inner == outer:
                continue
            ids = tuple(
                example
                for example, family in example_families.items()
                if family == inner
            )
            mirrors.append(
                {
                    "inner_held_family_id": inner,
                    "score_evidence": score_evidence(ids, seed),
                    "capability_receipt": capability(
                        ids,
                        held=outer,
                        accesses=1,
                    ),
                }
            )
            seed += 20
        inner_fragments[outer] = {"matched_negative_evidence": tuple(mirrors)}

    outer_refits = []
    outer_fragments: dict[str, dict[str, object]] = {}
    for outer in _FAMILIES:
        training_ids = tuple(
            example
            for example, family in example_families.items()
            if family != outer
        )
        outer_refits.append(
            {
                "fit_training_evidence": fit_evidence(
                    training_ids,
                    held=outer,
                    seed=seed,
                )
            }
        )
        seed += 100
        ids = tuple(
            example
            for example, family in example_families.items()
            if family == outer
        )
        outer_fragments[outer] = {
            "outer_held_family_id": outer,
            "score_evidence": {
                name: score_evidence(ids, seed + 20 * index)
                for index, name in enumerate(
                    ("baseline", "selected_positive", "matched_negative")
                )
            },
            "capability_receipt": capability(ids, held=None, accesses=3),
        }
        seed += 100

    vault_payload = {
        "example_family_ids": dict(sorted(example_families.items())),
        "teacher_row_sha256s": dict(sorted(teacher_hashes.items())),
    }
    prerequisite = {
        "authenticated_fit_collection": {
            "teacher_vault_receipt": {
                "artifact_sha256": nested._v14._sha256(
                    vault_payload,
                    domain=nested._v19._TEACHER_VAULT_DOMAIN,
                ),
                "example_count": 16,
                "family_count": 8,
                "teacher_row_sha256s": teacher_hashes,
                "source_rows_cached_in_native_dtype_on_cpu": True,
                "float64_teacher_log_probabilities_or_probabilities_cached": False,
            }
        }
    }
    return (
        prerequisite,
        pair_fragments,
        inner_fragments,
        {"outer_refits": tuple(outer_refits)},
        outer_fragments,
        core.nested_microstep_work_accounting(outer_scored=True),
    )


def test_teacher_access_accounting_rebuilds_all_3072_accesses() -> None:
    prerequisite, pairs, inners, lock, outers, work = (
        _synthetic_authenticated_access_graph()
    )
    receipt = nested._authenticated_teacher_access_accounting(
        prerequisite=prerequisite,
        pair_fragments=pairs,
        inner_fragments=inners,
        selection_lock=lock,
        outer_fragments=outers,
        work_accounting=work,
    )
    assert receipt["teacher_capability_access_count"] == 3_072
    assert receipt["post_cast_h4_hash_check_count"] == 3_072
    assert receipt["supervised_full_vocab_logits_hash_check_count"] == 3_072
    assert receipt["phase_access_counts"] == {
        "pair_training": 336,
        "inner_positive_scoring": 2_464,
        "inner_mirror_scoring": 112,
        "outer_refit_training": 112,
        "outer_scoring": 48,
    }
    assert receipt["capability_receipt_count"] == 156


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("access_count", "execution evidence count differs"),
        ("artifact", "artifact drifted"),
    ),
)
def test_teacher_access_accounting_rejects_forged_receipts(
    mutation: str,
    message: str,
) -> None:
    prerequisite, pairs, inners, lock, outers, work = copy.deepcopy(
        _synthetic_authenticated_access_graph()
    )
    capability = next(iter(pairs.values()))["directed_panels"][0][
        "capability_receipt"
    ]
    if mutation == "access_count":
        capability["access_count"] -= 1
    else:
        capability["artifact_sha256"] = _digest(199_999)
    with pytest.raises(ValueError, match=message):
        nested._authenticated_teacher_access_accounting(
            prerequisite=prerequisite,
            pair_fragments=pairs,
            inner_fragments=inners,
            selection_lock=lock,
            outer_fragments=outers,
            work_accounting=work,
        )
    assert work["ordered_inner_role_count"] == 56
    assert work["full_model_forward_count"] == 3_104
    assert work["teacher_capability_access_count"] == 3_072

    overhead = nested._resume_overhead(
        pair_endpoint_reconstruction_count=3,
        outer_endpoint_reconstruction_count=2,
    )
    assert overhead["excluded_from_canonical_scientific_work"] is True
    assert overhead["extra_full_model_forward_count"] == 3 * 12 + 2 * 14
    assert overhead["extra_teacher_capability_access_count"] == 3 * 12 + 2 * 14
    assert overhead["extra_post_cast_h4_hash_check_count"] == 3 * 12 + 2 * 14
    assert (
        overhead["extra_supervised_full_vocab_logits_hash_check_count"]
        == 3 * 12 + 2 * 14
    )
    assert work["physical_shared_pair_fit_count"] == 28


def test_report_ready_rejects_partial_outer_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nested, "_is_under_local_runs", lambda _path: True)
    inputs = {
        "prerequisite": {},
        "panel_receipt": {},
        "panel_binding_sha256": _digest(60_000),
        "bridge_binding_sha256": _digest(60_001),
        "pair_fragments": tuple({"pair": index} for index in range(28)),
        "inner_fragments": tuple({"outer": index} for index in range(8)),
        "selection_lock": {"selection_lock_sha256": _digest(60_002)},
        "outer_fragments": ({"outer": 0},),
        "resume_overhead": nested._resume_overhead(
            pair_endpoint_reconstruction_count=0,
            outer_endpoint_reconstruction_count=0,
        ),
        "integrity": {},
    }
    with pytest.raises(ValueError, match="outer schedule is partial"):
        nested._publish_report_ready_checkpoint(
            output=tmp_path / "partial-v20b.json",
            report_inputs=inputs,
        )
