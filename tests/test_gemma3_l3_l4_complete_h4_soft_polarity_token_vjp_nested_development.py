from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_soft_polarity_token_vjp_nested_development
    as runner,
)


def _sha(label: str) -> str:
    return runner._v14._sha256({"label": label}, domain=b"v20q-runner-test\0")


def _parent_authority() -> SimpleNamespace:
    families = tuple(sorted(runner._V20P_FOLD_SHA256S))
    return SimpleNamespace(
        prerequisite={
            "nested_panel_receipt": {"artifact_sha256": _sha("panel")},
            "authenticated_bridge_binding_sha256": _sha("bridge"),
        },
        source={"artifact_sha256": _sha("parent-source")},
        authenticated_v20g_folds={family: {"family": family} for family in families},
        authenticated_v20i_folds={family: {"family": family} for family in families},
        authenticated_v20l_folds={family: {"family": family} for family in families},
        authenticated_v20m_folds={family: {"family": family} for family in families},
        authenticated_v20o_folds={family: {"family": family} for family in families},
    )


def _completed_v20p_report() -> dict[str, object]:
    return {
        "report_sha256": runner._V20P_LOGICAL_SHA256,
        "source_receipt": {"artifact_sha256": runner._V20P_SOURCE_SHA256},
        "fold_fragment_sha256s_by_family": dict(runner._V20P_FOLD_SHA256S),
        "all_eight_outer_folds_completed": True,
        "decision": {"integrity_passed": True},
        "classification": "failed",
        "development_oof_passed": False,
        "rollback_to_base": True,
        "calibration_b_opened": False,
        "final_refit": None,
    }


def _install_prerequisite_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    report: dict[str, object] | None = None,
    file_sha256: str | None = None,
) -> tuple[SimpleNamespace, list[str]]:
    parent = _parent_authority()
    completed = _completed_v20p_report() if report is None else report
    loaded_families: list[str] = []

    monkeypatch.setattr(runner._v20p, "_load_prerequisites", lambda: parent)
    monkeypatch.setattr(
        runner._v14,
        "_file_sha256",
        lambda _path: runner._V20P_FILE_SHA256
        if file_sha256 is None
        else file_sha256,
    )
    monkeypatch.setattr(
        runner._v20p,
        "_load_existing_report",
        lambda *args, **kwargs: completed,
    )

    def load_fold(*, outer_family_id: str, **kwargs: object) -> dict[str, object]:
        assert kwargs["source"] is parent.source
        assert kwargs["authenticated_v20g_fold"] is parent.authenticated_v20g_folds[
            outer_family_id
        ]
        assert kwargs["authenticated_v20i_fold"] is parent.authenticated_v20i_folds[
            outer_family_id
        ]
        assert kwargs["authenticated_v20l_fold"] is parent.authenticated_v20l_folds[
            outer_family_id
        ]
        assert kwargs["authenticated_v20m_fold"] is parent.authenticated_v20m_folds[
            outer_family_id
        ]
        assert kwargs["authenticated_v20o_fold"] is parent.authenticated_v20o_folds[
            outer_family_id
        ]
        loaded_families.append(outer_family_id)
        return {
            "outer_held_family_id": outer_family_id,
            "fragment_sha256": runner._V20P_FOLD_SHA256S[outer_family_id],
        }

    monkeypatch.setattr(runner._v20p, "_load_fold_fragment", load_fold)
    return parent, loaded_families


def test_protocol_geometry_and_output_protection_are_frozen() -> None:
    assert runner.DEFAULT_OUTPUT.name.endswith("v20q.json")
    assert runner.DEFAULT_OUTPUT != runner._V20P_OUTPUT
    assert runner._FEATURES == ("c1", "c2", "c1_times_c2", "source_z")
    assert runner._SEED_SIGNS == (-1, 1)
    assert runner._SEED_ABS_B == 0.5
    assert runner._FIXED_PROTOCOL["runtime_provider"] == (
        "exact_unchanged_v20p_local_signed_field"
    )
    assert runner._FIXED_PROTOCOL["failure_policy"] == (
        "rollback_to_base_no_claims_no_B"
    )
    assert runner._FIXED_PROTOCOL["fresh_validation_claim"] is False
    assert runner._FIXED_PROTOCOL["calibration_b_eligible"] is False
    assert runner._FIXED_PROTOCOL["compression_claim_authorized"] is False

    with pytest.raises(ValueError, match="preserve immutable V20p authority"):
        runner._validate_output(runner._V20P_OUTPUT)
    for family in sorted(runner._V20P_FOLD_SHA256S):
        with pytest.raises(ValueError, match="preserve immutable V20p authority"):
            runner._validate_output(runner._v20p._fold_path(runner._V20P_OUTPUT, family))
    with pytest.raises(ValueError):
        runner._validate_output(Path("outside-v20q.json"))


def _teacher_capability_records() -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id="family-a",
                example_id="example-a",
            )
        ),
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id="family-b",
                example_id="example-b",
            )
        ),
    )


def _teacher_vault() -> object:
    return runner._v19._TeacherRowVault(
        {
            "example-a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "example-b": torch.arange(20, dtype=torch.float32).reshape(5, 4),
        },
        {"example-a": "family-a", "example-b": "family-b"},
    )


def test_v20q_cached_teacher_capability_preserves_receipt_without_per_read_rehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _teacher_capability_records()
    accelerated_vault = _teacher_vault()
    legacy_vault = _teacher_vault()
    original_hash = runner._v14._tensor_sha256
    hash_calls = 0

    def counted_hash(value: torch.Tensor) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return original_hash(value)

    monkeypatch.setattr(runner._v14, "_tensor_sha256", counted_hash)
    capability = runner._issue_v20q_cached_teacher_capability(
        accelerated_vault,
        records,
        held_family_id="outer-family",
        phase="focused_test",
    )
    issuance_hash_calls = hash_calls
    assert issuance_hash_calls > 0

    for _ in range(11):
        for record in records:
            sequence = record.sequence
            capability.get(
                sequence.example_id,
                family_id=sequence.family_id,
            )
    assert hash_calls == issuance_hash_calls

    receipt = runner._finalize_v20q_cached_teacher_capability(
        capability,
        expected_example_ids=tuple(
            record.sequence.example_id for record in records
        ),
        expected_family_count=2,
        expected_held_family_id="outer-family",
        expected_accesses_per_example=11,
        label="V20q cached teacher focused test",
    )
    assert hash_calls == issuance_hash_calls + len(records)
    assert receipt["access_count"] == 22
    assert receipt["per_example_access_counts"] == {
        "example-a": 11,
        "example-b": 11,
    }
    accounting = capability.phase_access_accounting()
    assert accounting == {
        "phase": "focused_test",
        "authorized_example_count": 2,
        "physical_legacy_teacher_row_fetch_count": 2,
        "logical_teacher_row_access_count": 22,
        "per_example_logical_access_counts": {
            "example-a": 11,
            "example-b": 11,
        },
        "issuance_integrity_check_count": 1,
        "completion_integrity_check_count": 1,
        "per_logical_read_full_row_rehash_count": 0,
        "completion_integrity_check_passed": True,
        "persisted_capability_receipt_schema_unchanged": True,
    }

    legacy = legacy_vault.capability(
        ("example-a", "example-b"), held_family_id="outer-family"
    )
    for _ in range(11):
        legacy.get("example-a", family_id="family-a")
        legacy.get("example-b", family_id="family-b")
    assert receipt == legacy.receipt()
    with pytest.raises(RuntimeError, match="already finalized"):
        capability.get("example-a", family_id="family-a")


def test_v20q_cached_teacher_capability_final_integrity_check_detects_mutation() -> None:
    records = _teacher_capability_records()
    capability = runner._issue_v20q_cached_teacher_capability(
        _teacher_vault(),
        records,
        held_family_id="outer-family",
        phase="mutation_test",
    )
    row = capability.get("example-a", family_id="family-a")
    row[0, 0] += 1.0

    with pytest.raises(RuntimeError, match="payload drifted"):
        capability.receipt()
    accounting = capability.phase_access_accounting()
    assert accounting["completion_integrity_check_count"] == 1
    assert accounting["completion_integrity_check_passed"] is False


def _exact_execution_receipt(
    *,
    phase: str,
    outer_family_id: str,
    inner_family_id: str | None,
    candidate_id: str,
    provider_sha256: str,
    runtime_sha256: str,
    evidence_sha256: str,
) -> dict[str, object]:
    objectives = {"example-a": 0.2, "example-b": 0.4}
    return runner._hashed(
        {
            "phase": phase,
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "logical_candidate_id": candidate_id,
            "provider_artifact_sha256": provider_sha256,
            "runtime_provider_artifact_sha256": runtime_sha256,
            "evidence_sha256": evidence_sha256,
            "objective": math.fsum(objectives.values()) / len(objectives),
            "objectives_by_example": objectives,
            "post_cast_h4_sha256s": {
                example_id: _sha(f"h4:{candidate_id}:{example_id}")
                for example_id in objectives
            },
            "supervised_full_vocab_logits_sha256s": {
                example_id: _sha(f"logits:{candidate_id}:{example_id}")
                for example_id in objectives
            },
            "execution_sha256s": {
                example_id: _sha(f"execution:{candidate_id}:{example_id}")
                for example_id in objectives
            },
            "exact_float64_full_vocabulary_teacher_KL": True,
            "raw_teacher_logit_h4_or_token_tensors_serialized": False,
        },
        domain=runner._EXECUTION_DOMAIN,
    )


def test_rehashed_inner_objective_map_must_match_execution_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_id = "focused_candidate"
    outer_family_id = "outer-family"
    inner_family_id = "inner-family"
    provider_sha256 = _sha("focused-provider")
    runtime_sha256 = _sha("focused-runtime")
    candidate_sha256 = _sha("focused-candidate")
    monkeypatch.setattr(
        runner._token_protocol,
        "SOFT_POLARITY_TOKEN_VJP_CANDIDATE_IDS",
        (candidate_id,),
    )
    manifest = runner._hashed(
        {
            "candidate_order": [candidate_id],
            "candidate_count": 1,
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "candidate_provider_artifact_sha256s": {
                candidate_id: provider_sha256
            },
            "candidate_runtime_provider_artifact_sha256s": {
                candidate_id: runtime_sha256
            },
            "candidate_receipt_sha256s": {candidate_id: candidate_sha256},
            "all_174_logical_candidates_and_traces_frozen_before_inner_capability": True,
            "inner_capability_count_at_freeze": 0,
            "outer_family_used_for_fit_or_selection": False,
        },
        domain=runner._PROVIDER_DOMAIN,
    )
    evidence_sha256 = runner._execution_seed(
        provider_manifest_sha256=str(manifest["artifact_sha256"]),
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        logical_candidate_id=candidate_id,
        provider_artifact_sha256=provider_sha256,
        phase="inner_held_exact_KL",
    )
    execution = _exact_execution_receipt(
        phase="inner_held_exact_KL",
        outer_family_id=outer_family_id,
        inner_family_id=inner_family_id,
        candidate_id=candidate_id,
        provider_sha256=provider_sha256,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )
    candidate = {
        "candidate_id": candidate_id,
        "candidate_provider_sha256": provider_sha256,
        "artifact_sha256": candidate_sha256,
    }
    inner = runner._hashed(
        {
            "outer_held_family_id": outer_family_id,
            "inner_held_family_id": inner_family_id,
            "provider_manifest": manifest,
            "candidate_receipts": {candidate_id: candidate},
            "execution_receipts_by_candidate": {candidate_id: execution},
            "exact_objective_by_candidate": {
                candidate_id: execution["objective"]
            },
            "selection_not_yet_performed": True,
            "all_candidates_frozen_before_inner_held_exact_KL": True,
            "outer_family_used_for_fit_score_or_selection": False,
        },
        domain=runner._EXECUTION_DOMAIN,
    )
    receipt = {"inner_family_receipts": {inner_family_id: inner}}
    runner._selection_inputs(receipt)

    tampered_inner = copy.deepcopy(inner)
    tampered_inner["exact_objective_by_candidate"][candidate_id] = 0.25
    tampered_inner.pop("artifact_sha256")
    tampered_inner["artifact_sha256"] = runner._v14._sha256(
        tampered_inner, domain=runner._EXECUTION_DOMAIN
    )
    with pytest.raises(ValueError, match="semantic binding differs"):
        runner._selection_inputs(
            {"inner_family_receipts": {inner_family_id: tampered_inner}}
        )


def _held_semantic_fixture() -> tuple[
    dict[str, object], dict[str, object], dict[str, object], str, str
]:
    outer_family_id = "outer-family"
    candidate_id = runner._token_protocol.SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID
    provider_sha256 = _sha("outer-provider")
    runtime_sha256 = _sha("outer-runtime")
    manifest = runner._hashed(
        {
            "provider_artifact_sha256": provider_sha256,
            "runtime_provider_artifact_sha256": runtime_sha256,
        },
        domain=runner._PROVIDER_DOMAIN,
    )
    evidence_sha256 = runner._execution_seed(
        provider_manifest_sha256=str(manifest["artifact_sha256"]),
        outer_family_id=outer_family_id,
        inner_family_id=None,
        logical_candidate_id=candidate_id,
        provider_artifact_sha256=provider_sha256,
        phase="outer_held_exact_KL",
    )
    execution = _exact_execution_receipt(
        phase="outer_held_exact_KL",
        outer_family_id=outer_family_id,
        inner_family_id=None,
        candidate_id=candidate_id,
        provider_sha256=provider_sha256,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )
    inherited_scores = {
        arm: 0.5 + index / 10.0 for index, arm in enumerate(runner._v20p._ARMS)
    }
    authenticated_v20p_fold = {
        "fold_receipt": {"held_objective_by_arm": inherited_scores},
        "held_evidence": {
            "arm_evidence": {
                "local_signed_field_reflected": {
                    "post_cast_h4_sha256s": {
                        example_id: _sha(f"incumbent-h4:{example_id}")
                        for example_id in ("example-a", "example-b")
                    },
                    "supervised_full_vocab_logits_sha256s": {
                        example_id: _sha(f"incumbent-logits:{example_id}")
                        for example_id in ("example-a", "example-b")
                    },
                }
            }
        },
    }
    held = runner._expected_held_evidence(
        outer_family_id=outer_family_id,
        selected_candidate_id=candidate_id,
        manifest=manifest,
        held_evidence={
            "candidate_execution": execution,
            "outer_capability_receipt": {"access_count": 2},
        },
        authenticated_v20p_fold=authenticated_v20p_fold,
    )
    return held, manifest, authenticated_v20p_fold, outer_family_id, candidate_id


def test_rehashed_held_summary_must_replay_from_execution_and_v20p() -> None:
    held, manifest, authenticated_v20p_fold, outer_family_id, candidate_id = (
        _held_semantic_fixture()
    )
    runner._validate_held_evidence_semantics(
        held,
        outer_family_id=outer_family_id,
        selected_candidate_id=candidate_id,
        manifest=manifest,
        authenticated_v20p_fold=authenticated_v20p_fold,
    )

    tampered = copy.deepcopy(held)
    tampered["candidate_objective"] = 0.25
    tampered["candidate_strictly_beats_v20p_incumbent"] = True
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = runner._v14._sha256(
        tampered, domain=runner._EXECUTION_DOMAIN
    )
    with pytest.raises(ValueError, match="held evidence does not replay"):
        runner._validate_held_evidence_semantics(
            tampered,
            outer_family_id=outer_family_id,
            selected_candidate_id=candidate_id,
            manifest=manifest,
            authenticated_v20p_fold=authenticated_v20p_fold,
        )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    (
        ("candidate_objective", 0.75),
        ("candidate_strictly_beats_v20p_incumbent", True),
        ("selected_nonzero_continuous_candidate", True),
        ("selected_inner_oof_mean_beats_incumbent", True),
        ("candidate_exact_output_differs_from_v20p_incumbent", False),
        ("candidate_field_nonconstant", False),
        ("candidate_field_has_negative", False),
        ("candidate_field_has_positive", False),
        ("all_secant_stability_gates_passed", False),
        ("deployed_direction_has_negative_predicted_derivative", False),
        ("candidate_pointwise_trust_passed", False),
        ("provider_frozen_before_outer_score", False),
        ("outer_family_used_for_fit_or_selection", True),
        ("exact_execution", False),
    ),
)
def test_rehashed_fold_gate_summary_must_replay_from_nested_evidence(
    field: str, tampered_value: object
) -> None:
    candidate_id = runner._token_protocol.SOFT_POLARITY_TOKEN_VJP_INCUMBENT_CANDIDATE_ID
    aggregate = {"family_equal_exact_kl": 0.5}
    selection = {
        "selected_candidate_id": candidate_id,
        "aggregate_by_candidate": {candidate_id: aggregate},
    }
    final_refit = {
        "selected_candidate_id": candidate_id,
        "feature_id": "c1",
        "b": 0.5,
        "a": -1.0,
        "provider_frozen_before_outer_held_objective": True,
        "outer_held_objective_consumed": False,
    }
    chart_collection = {"chart_receipts_by_id": {}}
    manifest = {
        "provider_and_trace_frozen_before_outer_capability": True,
        "outer_capability_count_at_freeze": 0,
    }
    held = {
        "candidate_objective": 0.5,
        "inherited_v20p_held_objective_by_arm": {
            "local_signed_field_reflected": 0.5
        },
        "candidate_exact_output_differs_from_v20p_incumbent": True,
        "provider_frozen_before_outer_held_objective": True,
        "outer_held_objective_used_for_adaptation": False,
    }
    trace = {
        "local_signed_scalar_nonconstant": True,
        "local_signed_scalar_has_negative": True,
        "local_signed_scalar_has_positive": True,
        "pointwise_trust_passed": True,
    }
    receipt = runner._expected_fold_receipt(
        outer_family_id="outer-family",
        selection_receipt=selection,
        final_refit_receipt=final_refit,
        chart_collection_receipt=chart_collection,
        manifest=manifest,
        held_evidence=held,
        trace=trace,
    )
    runner._validate_fold_receipt_semantics(
        receipt,
        outer_family_id="outer-family",
        selection_receipt=selection,
        final_refit_receipt=final_refit,
        chart_collection_receipt=chart_collection,
        manifest=manifest,
        held_evidence=held,
        trace=trace,
    )

    tampered = copy.deepcopy(receipt)
    tampered[field] = tampered_value
    tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = runner._v14._sha256(
        tampered, domain=runner._DECISION_DOMAIN
    )
    with pytest.raises(ValueError, match="fold receipt does not replay"):
        runner._validate_fold_receipt_semantics(
            tampered,
            outer_family_id="outer-family",
            selection_receipt=selection,
            final_refit_receipt=final_refit,
            chart_collection_receipt=chart_collection,
            manifest=manifest,
            held_evidence=held,
            trace=trace,
        )


def test_prerequisites_authenticate_report_and_all_eight_folds_model_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, loaded_families = _install_prerequisite_stubs(monkeypatch)
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("prerequisite authentication constructed Gemma"),
    )

    authorities = runner._load_prerequisites()

    assert authorities.parent is parent
    assert authorities.v20p_report == _completed_v20p_report()
    assert tuple(loaded_families) == tuple(sorted(runner._V20P_FOLD_SHA256S))
    assert {
        family: fold["fragment_sha256"]
        for family, fold in authorities.authenticated_v20p_folds.items()
    } == runner._V20P_FOLD_SHA256S
    assert authorities.source["v20p_report_sha256"] == runner._V20P_LOGICAL_SHA256
    assert authorities.source["v20p_file_sha256"] == runner._V20P_FILE_SHA256
    assert authorities.source["v20p_source_receipt_sha256"] == (
        runner._V20P_SOURCE_SHA256
    )
    assert authorities.source["authenticated_before_model_construction"] is True
    assert authorities.source["historically_reused_A16_only"] is True
    assert authorities.source["fresh_validation_claim"] is False
    assert authorities.source["calibration_b_manifest_read"] is False
    assert authorities.source["calibration_b_tokenized"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "file_hash",
        "report_hash",
        "source_hash",
        "fold_hashes",
        "fold_count",
        "integrity",
        "development_result",
        "rollback",
        "calibration_b",
        "final_refit",
    ),
)
def test_prerequisite_authentication_fails_closed_before_fold_replay(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    report = copy.deepcopy(_completed_v20p_report())
    file_hash = runner._V20P_FILE_SHA256
    if mutation == "file_hash":
        file_hash = _sha("drifted-file")
    elif mutation == "report_hash":
        report["report_sha256"] = _sha("drifted-report")
    elif mutation == "source_hash":
        report["source_receipt"] = {"artifact_sha256": _sha("drifted-source")}
    elif mutation == "fold_hashes":
        family = sorted(runner._V20P_FOLD_SHA256S)[0]
        report["fold_fragment_sha256s_by_family"][family] = _sha("drifted-fold")
    elif mutation == "fold_count":
        report["all_eight_outer_folds_completed"] = False
    elif mutation == "integrity":
        report["decision"]["integrity_passed"] = False
    elif mutation == "development_result":
        report["development_oof_passed"] = True
    elif mutation == "rollback":
        report["rollback_to_base"] = False
    elif mutation == "calibration_b":
        report["calibration_b_opened"] = True
    elif mutation == "final_refit":
        report["final_refit"] = {"artifact_sha256": _sha("forbidden-refit")}

    _, loaded_families = _install_prerequisite_stubs(
        monkeypatch,
        report=report,
        file_sha256=file_hash,
    )
    with pytest.raises(RuntimeError, match="pinned .*V20p .*drifted|authority differs"):
        runner._load_prerequisites()
    assert loaded_families == []


def test_execution_seeds_are_deterministic_and_bind_candidate_and_phase() -> None:
    common = {
        "provider_manifest_sha256": _sha("manifest"),
        "outer_family_id": "outer-family",
        "inner_family_id": "inner-family",
        "logical_candidate_id": next(iter(runner._CANDIDATE_SPEC_BY_ID)),
        "provider_artifact_sha256": _sha("provider"),
        "phase": "inner_held_exact_KL",
    }
    seed = runner._execution_seed(**common)
    assert seed == runner._execution_seed(**common)
    changed_candidate = dict(common)
    changed_candidate["logical_candidate_id"] = tuple(
        runner._CANDIDATE_SPEC_BY_ID
    )[1]
    assert seed != runner._execution_seed(**changed_candidate)
    changed_phase = dict(common)
    changed_phase["phase"] = "outer_held_exact_KL"
    assert seed != runner._execution_seed(**changed_phase)


def _decision_fragments(
    families: tuple[str, ...] | None = None,
) -> dict[str, dict[str, object]]:
    fragments: dict[str, dict[str, object]] = {}
    family_ids = (
        tuple(f"family-{index}" for index in range(8))
        if families is None
        else families
    )
    for index, family in enumerate(family_ids):
        inherited = {arm: 1.0 for arm in runner._v20p._ARMS}
        # The historical fixed-minus result remains a diagnostic rather than a
        # gate, so allow it to beat this candidate without changing the pass.
        inherited["fixed_minus"] = 0.25
        fragments[family] = {
            "fragment_sha256": _sha(f"fragment:{family}"),
            "chart_collection_receipt": {"chart_receipts_by_id": {}},
            "fold_receipt": {
                "candidate_objective": 0.5,
                "inherited_v20p_held_objective_by_arm": inherited,
                "selected_nonzero_continuous_candidate": index < 6,
                "selected_inner_oof_mean_beats_incumbent": index < 6,
                "candidate_exact_output_differs_from_v20p_incumbent": index < 6,
                "all_secant_stability_gates_passed": True,
                "deployed_direction_has_negative_predicted_derivative": True,
                "candidate_pointwise_trust_passed": True,
                "provider_frozen_before_outer_score": True,
                "outer_family_used_for_fit_or_selection": False,
                "exact_execution": True,
            },
        }
    return fragments


def test_decision_positive_case_uses_every_predeclared_gate() -> None:
    decision = runner._aggregate_decision(_decision_fragments())

    assert decision["macro_objective_by_system"]["token_vjp_candidate"] == 0.5
    assert decision["candidate_strict_win_count_by_v20p_arm"][
        "local_signed_field_reflected"
    ] == 8
    assert decision["candidate_strict_win_count_by_v20p_arm"]["fixed_minus"] == 0
    assert decision["primary_candidate_vs_v20p_incumbent_gate_passed"] is True
    assert decision["selected_nonzero_continuous_count"] == 6
    assert decision["selected_continuous_and_inner_better_count"] == 6
    assert decision["continuous_fit_gate_passed"] is True
    assert decision["candidate_exact_output_differs_count"] == 6
    assert decision["exact_output_difference_gate_passed"] is True
    assert decision[
        "all_deployed_fit_directions_descending_and_secants_stable"
    ] is True
    assert decision["all_candidate_pointwise_trust_passed"] is True
    assert decision["runtime_health_gate_passed"] is True
    assert decision["integrity_passed"] is True
    assert decision["development_oof_passed"] is True


@pytest.mark.parametrize(
    "mutation,failed_gate",
    (
        ("primary", "primary_candidate_vs_v20p_incumbent_gate_passed"),
        ("continuous", "continuous_fit_gate_passed"),
        ("inner", "continuous_fit_gate_passed"),
        ("difference", "exact_output_difference_gate_passed"),
        (
            "secant",
            "all_deployed_fit_directions_descending_and_secants_stable",
        ),
        (
            "derivative",
            "all_deployed_fit_directions_descending_and_secants_stable",
        ),
        ("pointwise", "all_candidate_pointwise_trust_passed"),
        ("runtime", "runtime_health_gate_passed"),
    ),
)
def test_decision_fails_closed_at_each_exact_threshold(
    mutation: str, failed_gate: str
) -> None:
    fragments = _decision_fragments(tuple(sorted(runner._V20P_FOLD_SHA256S)))
    families = sorted(fragments)
    if mutation == "primary":
        # Keep the macro below the incumbent while reducing strict wins to four.
        for family in families[4:]:
            fragments[family]["fold_receipt"]["candidate_objective"] = 1.1
    elif mutation == "continuous":
        fragments[families[5]]["fold_receipt"][
            "selected_nonzero_continuous_candidate"
        ] = False
    elif mutation == "inner":
        fragments[families[5]]["fold_receipt"][
            "selected_inner_oof_mean_beats_incumbent"
        ] = False
    elif mutation == "difference":
        fragments[families[5]]["fold_receipt"][
            "candidate_exact_output_differs_from_v20p_incumbent"
        ] = False
    elif mutation == "secant":
        fragments[families[0]]["fold_receipt"][
            "all_secant_stability_gates_passed"
        ] = False
    elif mutation == "derivative":
        fragments[families[0]]["fold_receipt"][
            "deployed_direction_has_negative_predicted_derivative"
        ] = False
    elif mutation == "pointwise":
        fragments[families[0]]["fold_receipt"][
            "candidate_pointwise_trust_passed"
        ] = False
    elif mutation == "runtime":
        fragments[families[0]]["fold_receipt"][
            "provider_frozen_before_outer_score"
        ] = False

    decision = runner._aggregate_decision(fragments)
    assert decision[failed_gate] is False
    assert decision["development_oof_passed"] is False


def test_runner_work_accounting_is_frozen() -> None:
    work = runner._runner_work_accounting()

    assert work["canonical_model_forward_count"] == 20544
    assert work["total_model_forward_count"] == 20544
    assert work["canonical_teacher_access_count"] == 20512
    assert work["total_teacher_access_count"] == 20512
    assert work["token_vjp_chart_model_forward_count"] == 896
    assert work["inner_exact_candidate_model_forward_count"] == 19488
    assert work["outer_held_candidate_model_forward_count"] == 16
    assert work["inner_logical_candidate_count"] == 9744
    assert work["logical_candidates_per_inner_fold"] == 174
    assert work["inner_folds_per_outer_fold"] == 7
    assert work["token_vjp_chart_count"] == 64
    assert work["token_vjp_prompt_record_count"] == 896
    assert work["runtime_provider_parameter_count_delta_vs_v20p"] == 0
    assert work["runtime_provider_logical_macs_delta_vs_v20p"] == 0
    assert work["compiler_only_fisher_vjp_work_excluded_from_inference"] is True
    assert work["calibration_b_forward_or_tokenization_count"] == 0


def test_parser_default_entry_and_cli_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = runner.build_parser().parse_args([])
    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.cache_dir is None
    assert runner.DEFAULT_OUTPUT.name.endswith("v20q.json")

    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"classification": "model-free-cli-test", "passed": False}

    monkeypatch.setattr(
        runner,
        "run_gemma3_l3_l4_complete_h4_soft_polarity_token_vjp_nested_development",
        fake_run,
    )
    assert runner.main([]) == 0
    assert observed == {"output": runner.DEFAULT_OUTPUT, "cache_dir": None}
    assert capsys.readouterr().out.strip() == (
        runner._v14._canonical_json_bytes(
            {"classification": "model-free-cli-test", "passed": False}
        ).decode("ascii")
    )
    pyproject = Path(__file__).parents[1].joinpath("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-soft-polarity-v20q-"
        "token-vjp-nested"
    ) in pyproject


def _report_authorities() -> runner._Authorities:
    folds = {
        family: {"fragment_sha256": runner._V20P_FOLD_SHA256S[family]}
        for family in sorted(runner._V20P_FOLD_SHA256S)
    }
    return runner._Authorities(
        parent=SimpleNamespace(),
        v20p_report={"classification": "failed"},
        authenticated_v20p_folds=folds,
        source={"artifact_sha256": _sha("v20q-source")},
    )


def test_report_rebuild_preserves_development_only_claim_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "v20q.json"
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    fragments = _decision_fragments()
    authorities = _report_authorities()
    panel = {"artifact_sha256": _sha("panel")}
    report = runner._build_report(
        output=output,
        authorities=authorities,
        panel_receipt=panel,
        bridge_binding_sha256=_sha("bridge"),
        fold_fragments=fragments,
    )

    assert report["passed"] is True
    assert report["development_oof_passed"] is True
    assert report["rollback_to_base"] is False
    assert report["classification"].endswith("nested_oof_passed")
    assert report["historically_reused_A16_only"] is True
    assert report["fresh_family_disjoint_scoring_performed"] is False
    assert report["fresh_validation_claim_authorized"] is False
    assert report["final_refit"] is None
    assert report["full_refit_performed"] is False
    assert report["calibration_b_eligible"] is False
    assert report["calibration_b_opened"] is False
    assert report["compression_claim_authorized"] is False
    assert report["fidelity_claim_authorized"] is False
    assert report["speed_claim_authorized"] is False
    assert report["serving_claim_authorized"] is False

    failed_fragments = copy.deepcopy(fragments)
    failed_fragments[sorted(failed_fragments)[5]]["fold_receipt"][
        "candidate_exact_output_differs_from_v20p_incumbent"
    ] = False
    failed = runner._build_report(
        output=output,
        authorities=authorities,
        panel_receipt=panel,
        bridge_binding_sha256=_sha("bridge"),
        fold_fragments=failed_fragments,
    )
    assert failed["passed"] is False
    assert failed["rollback_to_base"] is True
    assert failed["next_rung"] == "stop_token_vjp_local_field_refit_class"


def test_existing_report_is_rebuilt_and_authenticated_without_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "v20q.json"
    authorities = _report_authorities()
    fragments = _decision_fragments(tuple(sorted(runner._V20P_FOLD_SHA256S)))
    panel = {"artifact_sha256": _sha("panel")}
    bridge = _sha("bridge")
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    rebuilt = runner._build_report(
        output=output,
        authorities=authorities,
        panel_receipt=panel,
        bridge_binding_sha256=bridge,
        fold_fragments=fragments,
    )
    signed = dict(rebuilt)
    signed["report_sha256"] = runner._v14._sha256(
        rebuilt, domain=runner._REPORT_DOMAIN
    )
    monkeypatch.setattr(
        runner._v20b,
        "_load_scalar_fragment",
        lambda **kwargs: signed,
    )
    monkeypatch.setattr(
        runner,
        "_load_fold_fragment",
        lambda *, outer_family_id, **kwargs: fragments[outer_family_id],
    )
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("report replay constructed Gemma"),
    )

    assert runner._load_existing_report(
        output,
        authorities=authorities,
        panel_receipt=panel,
        bridge_binding_sha256=bridge,
    ) == signed

    tampered = copy.deepcopy(signed)
    tampered["classification"] = "tampered-but-rehashed"
    unsigned = dict(tampered)
    unsigned.pop("report_sha256")
    tampered["report_sha256"] = runner._v14._sha256(
        unsigned, domain=runner._REPORT_DOMAIN
    )
    monkeypatch.setattr(
        runner._v20b,
        "_load_scalar_fragment",
        lambda **kwargs: tampered,
    )
    with pytest.raises(ValueError, match="report reconstruction differs"):
        runner._load_existing_report(
            output,
            authorities=authorities,
            panel_receipt=panel,
            bridge_binding_sha256=bridge,
        )


def test_chart_collector_wires_primary_and_audit_provider_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAfterSecants(RuntimeError):
        pass

    artifact_by_role: dict[str, str] = {}
    calls: list[dict[str, object]] = []

    def fake_materialize(*args: object, **kwargs: object) -> tuple[object, str]:
        role = str(kwargs["logical_candidate_id"]).rsplit("__", 1)[-1]
        artifact = _sha(f"provider:{role}")
        artifact_by_role[role] = artifact
        runtime = SimpleNamespace(artifact_sha256=_sha(f"runtime:{role}"))
        return SimpleNamespace(artifact_sha256=artifact, runtime_provider=runtime), _sha(
            f"seed:{role}"
        )

    def fake_secants(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, object]:
        calls.append(dict(kwargs))
        center = torch.zeros((1, 2, 2), dtype=torch.float32)
        tangents = torch.ones((2, 1, 2, 2), dtype=torch.float64)
        return center, tangents, SimpleNamespace(artifact_sha256=_sha("secant"))

    def stop_after_both_secants(*args: object, **kwargs: object) -> object:
        assert len(calls) == 2
        raise StopAfterSecants

    sequence = SimpleNamespace(
        example_id="example",
        family_id="training-family",
        base_h4=torch.zeros((2, 2), dtype=torch.float32),
        support_mask=torch.tensor([True, False]),
    )
    record = SimpleNamespace(sequence=sequence)
    teacher_rows = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
    supervised_grid = torch.tensor([[0, 0]], dtype=torch.int64)
    teacher_grid, _ = runner.build_selected_teacher_grid(
        teacher_rows, supervised_grid, batch_size=1, sequence_length=2
    )
    fake_vjp = SimpleNamespace(
        teacher_logits_sha256=runner._shadow_runtime._runtime_tensor_sha256(
            teacher_grid
        ),
        h4_head_sha256=_sha("runtime:center"),
        vjp_chunk_size=runner._VJP_CHUNK_SIZE,
        objective_dtype=str(torch.float64),
        backward_call_count=1,
        token_count=1,
        supervised_indices=supervised_grid,
        artifact_sha256=_sha("vjp"),
        execution=SimpleNamespace(artifact_sha256=_sha("execution")),
        teacher_logits_shape=tuple(teacher_grid.shape),
        model_forward_count=1,
        validate_integrity=lambda: None,
    )
    context = SimpleNamespace(
        adapter=object(),
        bridge=SimpleNamespace(
            execute_h4_token_teacher_kl_vjps=lambda *args, **kwargs: fake_vjp
        ),
    )
    capability = SimpleNamespace(get=lambda *args, **kwargs: teacher_rows)

    monkeypatch.setattr(runner._v20p, "_selected_direction", lambda value: (1.0,))
    monkeypatch.setattr(runner, "_materialize_provider", fake_materialize)
    monkeypatch.setattr(runner._v20b, "_ordered_records", lambda value: (record,))
    monkeypatch.setattr(
        runner._v20a,
        "_verified_model_inputs",
        lambda *args, **kwargs: (
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.int64)},
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_training_field_correction",
        lambda provider, value: torch.zeros((2, 2), dtype=torch.float64),
    )
    monkeypatch.setattr(
        runner, "build_soft_polarity_post_cast_h4_secants", fake_secants
    )
    monkeypatch.setattr(
        runner, "soft_polarity_post_cast_h4_secant_stability", stop_after_both_secants
    )

    with pytest.raises(StopAfterSecants):
        runner._collect_chart_records(
            context,
            SimpleNamespace(),
            (record,),
            capability,
            outer_family_id="outer-family",
            reflection_fit={
                "artifact_sha256": _sha("reflection"),
                "selected_variant_artifact_sha256": _sha("direction"),
            },
            response=(0.25, 0.125, 0.0),
            feature_id="c1",
            seed_sign=-1,
        )

    assert len(calls) == 2
    expected_roles = (
        (
            "primary_bias_minus",
            "primary_bias_plus",
            "primary_slope_minus",
            "primary_slope_plus",
        ),
        (
            "audit_bias_minus",
            "audit_bias_plus",
            "audit_slope_minus",
            "audit_slope_plus",
        ),
    )
    for call, roles in zip(calls, expected_roles, strict=True):
        assert call["reference_provider_sha256"] == artifact_by_role["center"]
        assert call["bias_minus_provider_sha256"] == artifact_by_role[roles[0]]
        assert call["bias_plus_provider_sha256"] == artifact_by_role[roles[1]]
        assert call["slope_minus_provider_sha256"] == artifact_by_role[roles[2]]
        assert call["slope_plus_provider_sha256"] == artifact_by_role[roles[3]]
