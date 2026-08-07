from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_downstream_retention_experiment as experiment
from fisher_graph.gemma3_downstream_retention_experiment import (
    DEFAULT_DOWNSTREAM_PANEL,
    ForcedChoiceConditionResult,
    ForcedChoicePanel,
    _claim_payload,
    _validate_guard_assessment,
    _write_or_resume_claim,
    evaluate_forced_choice_retention,
    load_forced_choice_panel,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions(
    panel: ForcedChoicePanel,
    *,
    wrong_indices: set[int],
) -> tuple[int, ...]:
    return tuple(
        (
            (example.correct_choice + 1) % len(example.choices)
            if index in wrong_indices
            else example.correct_choice
        )
        for index, example in enumerate(panel.examples)
    )


def _condition(
    panel: ForcedChoicePanel,
    name: str,
    *,
    wrong_indices: set[int],
    nll: float,
) -> ForcedChoiceConditionResult:
    return ForcedChoiceConditionResult(
        name=name,  # type: ignore[arg-type]
        predictions=_predictions(panel, wrong_indices=wrong_indices),
        restricted_choice_nll=(nll,) * len(panel.examples),
        gold_margins=(1.0,) * len(panel.examples),
    )


def test_frozen_downstream_panel_has_balanced_unique_contract() -> None:
    panel = load_forced_choice_panel(DEFAULT_DOWNSTREAM_PANEL)

    assert len(panel.examples) == 60
    assert len(panel.family_ids) == 6
    assert all(
        sum(example.family_id == family for example in panel.examples) == 10
        for family in panel.family_ids
    )
    assert len({example.prompt_sha256 for example in panel.examples}) == 60
    assert len(panel.semantic_sha256) == 64


def test_panel_loader_rejects_any_byte_drift(tmp_path: Path) -> None:
    source = DEFAULT_DOWNSTREAM_PANEL.read_bytes()
    changed = tmp_path / "panel.json"
    changed.write_bytes(source + b"\n")

    with pytest.raises(ValueError, match="bytes differ"):
        load_forced_choice_panel(changed)


def test_retention_pilot_passes_strict_paired_count_gates() -> None:
    panel = load_forced_choice_panel(DEFAULT_DOWNSTREAM_PANEL)
    native = _condition(
        panel,
        "native",
        wrong_indices=set(),
        nll=0.50,
    )
    edgeless = _condition(
        panel,
        "edgeless",
        wrong_indices={0, 10, 20, 30, 40, 50},
        nll=0.60,
    )
    candidate = _condition(
        panel,
        "candidate",
        wrong_indices={1, 21, 41},
        nll=0.53,
    )

    result = evaluate_forced_choice_retention(
        panel,
        native=native,
        edgeless=edgeless,
        candidate=candidate,
    )

    assert result["status"] == "downstream_retention_pilot_pass"
    assert result["passed"] is True
    paired = result["paired_candidate_vs_native"]
    assert paired["accuracy_retained_fraction"] == pytest.approx(0.95)
    assert paired["native_win_preservation"] == pytest.approx(0.95)
    assert paired["native_only_correct_count"] == 3
    assert all(result["gates"].values())
    value = result["conditional_edge_value_added"]
    assert value["candidate_choice_nll_below_edgeless"] is True
    assert value["candidate_accuracy_not_below_edgeless"] is True


def test_retention_pilot_fails_below_ninety_percent() -> None:
    panel = load_forced_choice_panel(DEFAULT_DOWNSTREAM_PANEL)
    native = _condition(
        panel,
        "native",
        wrong_indices=set(),
        nll=0.50,
    )
    edgeless = _condition(
        panel,
        "edgeless",
        wrong_indices=set(range(12)),
        nll=0.70,
    )
    candidate = _condition(
        panel,
        "candidate",
        wrong_indices=set(range(9)),
        nll=0.55,
    )

    result = evaluate_forced_choice_retention(
        panel,
        native=native,
        edgeless=edgeless,
        candidate=candidate,
    )

    assert result["status"] == "downstream_retention_pilot_fail"
    assert result["passed"] is False
    assert (
        result["gates"]["global_accuracy_retention_at_least_0_90"]
        is False
    )
    assert result["paired_candidate_vs_native"][
        "accuracy_retained_fraction"
    ] == pytest.approx(0.85)


def test_retention_is_inconclusive_when_native_denominator_is_weak() -> None:
    panel = load_forced_choice_panel(DEFAULT_DOWNSTREAM_PANEL)
    native_wrong = set(range(len(panel.examples)))
    for offset in range(0, len(panel.examples), 10):
        native_wrong.remove(offset)
    native = _condition(
        panel,
        "native",
        wrong_indices=native_wrong,
        nll=1.0,
    )
    edgeless = _condition(
        panel,
        "edgeless",
        wrong_indices=native_wrong,
        nll=1.0,
    )
    candidate = _condition(
        panel,
        "candidate",
        wrong_indices=native_wrong,
        nll=1.0,
    )

    result = evaluate_forced_choice_retention(
        panel,
        native=native,
        edgeless=edgeless,
        candidate=candidate,
    )

    assert result["status"] == "inconclusive_native_denominator"
    assert result["passed"] is False
    assert result["gates"]["suite_adequate"] is False


def test_prequalified_family_cannot_be_dropped_post_hoc() -> None:
    panel = load_forced_choice_panel(DEFAULT_DOWNSTREAM_PANEL)
    native = _condition(
        panel,
        "native",
        wrong_indices=set(range(3, 10)),
        nll=0.50,
    )
    edgeless = _condition(
        panel,
        "edgeless",
        wrong_indices=set(range(10)),
        nll=0.60,
    )
    candidate = _condition(
        panel,
        "candidate",
        wrong_indices=set(range(10)),
        nll=0.55,
    )

    result = evaluate_forced_choice_retention(
        panel,
        native=native,
        edgeless=edgeless,
        candidate=candidate,
        prequalified_family_ids=panel.family_ids,
    )

    assert result["passed"] is False
    assert result["adequacy"]["observed_qualified_families"] == 6
    assert result["adequacy"]["family_eligibility_source"] == (
        "separate_native_only_qualification_split"
    )
    assert result["gates"][
        "no_qualified_family_loses_more_than_two"
    ] is False
    first_family = panel.examples[0].family_id
    assert result["family_comparisons"][first_family][
        "native_only_correct_count"
    ] == 3


def test_claim_can_resume_only_with_identical_protocol(tmp_path: Path) -> None:
    candidate = {
        "scientific_payload_sha256": "a" * 64,
        "dynamic_graph_sha256": "b" * 64,
        "compiler_pipeline_sha256": "c" * 64,
        "interaction_promotion_sha256": "d" * 64,
    }
    payload = _claim_payload(
        candidate=candidate,
        guard_assessment_sha256="1" * 64,
        panel_file_sha256="e" * 64,
        evaluator_file_sha256="f" * 64,
    )
    path = tmp_path / "claim.json"

    first = _write_or_resume_claim(path, payload)
    second = _write_or_resume_claim(path, payload)

    assert first == second
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["claim_sha256"] == first
    changed = dict(payload)
    changed["panel_id"] = "different-panel"
    with pytest.raises(FileExistsError, match="different bytes"):
        _write_or_resume_claim(path, changed)


def test_guard_assessment_rejects_fabricated_positive_three_field_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.assessment.json"
    path.write_text(
        json.dumps(
            {
                "candidate": {"scientific_payload_sha256": "a" * 64},
                "scientific_status": {
                    "role": "family_disjoint_calibration_a_guard"
                },
                "guard_nll_improvement_over_edgeless": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one JSON object"):
        _validate_guard_assessment(
            path,
            candidate={
                "scientific_payload_sha256": "a" * 64,
                "dynamic_graph_sha256": "b" * 64,
                "compiler_pipeline_sha256": "c" * 64,
                "interaction_promotion_sha256": "d" * 64,
            },
            expected_tensor_file="candidate.pt",
            expected_guard_manifest_sha256="e" * 64,
            expected_guard_assessment_file_sha256=_file_sha256(path),
        )


def _valid_guard_assessment() -> dict[str, object]:
    condition = lambda nll: {  # noqa: E731
        "delta_nll_per_token": nll - 1.0,
        "native_to_candidate_kl_per_token": 0.1,
        "nll_per_token": nll,
        "top1_agreement_to_native": 0.9,
    }
    return {
        "schema": "fisher_graph.gemma3_state_conditioned_shape_flow_assessment",
        "format_version": 1,
        "candidate": {
            "scientific_payload_sha256": "a" * 64,
            "dynamic_graph_sha256": "b" * 64,
            "compiler_pipeline_sha256": "c" * 64,
            "interaction_promotion_sha256": "d" * 64,
            "tensor_file": "candidate.pt",
        },
        "scientific_status": {
            "fresh_validation": False,
            "guard_claimed_before_materialization": True,
            "heldout_confirmation": False,
            "open_development": True,
            "role": "family_disjoint_calibration_a_guard",
            "test_data_used": False,
        },
        "guard": {
            "claim_sha256": "e" * 64,
            "example_count": 4,
            "family_ids": ["family-a", "family-b", "family-c", "family-d"],
            "role_manifest_sha256": "f" * 64,
            "tokenized_split_sha256": "1" * 64,
        },
        "behavior": {
            "assessment_role": "open_development_assessment",
            "conditions": {
                "edgeless_graph": condition(1.2),
                "interacting_graph": condition(1.1),
                "matched_deletion": condition(1.5),
            },
            "execution_path": "graph_executor",
            "graph_comparison": {
                "deletion_equivalence_atol": 0.0,
                "deletion_equivalence_rtol": 0.0,
                "deletion_equivalence_scope": "supervised_logits",
                "deletion_max_abs_logit_difference": 0.0,
                "deletion_paths_agree": True,
                "edgeless_edge_count": 0,
                "interacting_edge_count": 3,
                "interaction_parameter_delta": 10,
                "matched_deletion_resource_scope": "runtime_branch",
                "node_artifacts_identical": True,
                "node_count": 4,
                "nodewise_dense_agrees_with_edgeless": None,
                "nodewise_dense_equivalence_atol": None,
                "nodewise_dense_equivalence_rtol": None,
                "nodewise_dense_equivalence_scope": None,
                "nodewise_dense_max_abs_logit_difference": None,
                "nodewise_dense_supplied": False,
            },
            "heldout_confirmation": False,
            "latency_or_kernel_speed_claim": False,
            "logical_valid_tokens": 4,
            "native": {"nll_per_token": 1.0},
            "resource_accounting": {},
            "supervised_tokens": 4,
        },
        "flow": {
            "assessment_read_only": True,
            "coefficients_fitted": False,
            "evaluation_kind": (
                "fisher_graph.state_conditioned_modal_flow_evaluation"
            ),
            "families": [{}, {}, {}, {}],
            "interaction_artifact_sha256s": ["2" * 64] * 3,
            "observations": 4,
            "residual_width": 8,
            "routed_graph_uses_source_state_only": True,
            "source_free": True,
            "teacher_used_for_scoring_only": True,
        },
        "guard_nll_improvement_over_edgeless": 0.1,
        "resource_summary": {
            "candidate_whole_model_learned_parameters": 920,
            "logical_executed_modal_graph_macs": 300,
            "logical_modal_graph_macs": 400,
            "logical_valid_tokens": 4,
            "modal_graph_learned_parameters": 20,
            "native_removed_learned_parameters": 100,
            "source_whole_model_learned_parameters": 1000,
        },
        "routing_execution": {
            "exactly_one_selected_edge_per_valid_token": True,
            "selected_edge_rows": 4,
            "selected_edge_rows_by_interaction": {
                "edge-a": 2,
                "edge-b": 1,
                "edge-c": 1,
            },
            "valid_tokens": 4,
        },
        "source_model_unchanged": True,
    }


def test_guard_assessment_authenticates_persisted_claim_and_full_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_guard_assessment()
    path = tmp_path / "candidate.assessment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "load_gemma3_l3_l4_progressive_guard_claim",
        lambda **_kwargs: SimpleNamespace(claim_sha256="e" * 64),
    )
    candidate = {
        "scientific_payload_sha256": "a" * 64,
        "dynamic_graph_sha256": "b" * 64,
        "compiler_pipeline_sha256": "c" * 64,
        "interaction_promotion_sha256": "d" * 64,
    }

    loaded, file_sha256 = _validate_guard_assessment(
        path,
        candidate=candidate,
        expected_tensor_file="candidate.pt",
        expected_guard_manifest_sha256="f" * 64,
        expected_guard_assessment_file_sha256=_file_sha256(path),
    )

    assert loaded == payload
    assert len(file_sha256) == 64
    payload["flow"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="flow receipt"):
        _validate_guard_assessment(
            path,
            candidate=candidate,
            expected_tensor_file="candidate.pt",
            expected_guard_manifest_sha256="f" * 64,
            expected_guard_assessment_file_sha256=_file_sha256(path),
        )


def test_guard_assessment_rejects_unpersisted_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "candidate.assessment.json"
    path.write_text(json.dumps(_valid_guard_assessment()), encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "load_gemma3_l3_l4_progressive_guard_claim",
        lambda **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(ValueError, match="authenticated persisted claim"):
        _validate_guard_assessment(
            path,
            candidate={
                "scientific_payload_sha256": "a" * 64,
                "dynamic_graph_sha256": "b" * 64,
                "compiler_pipeline_sha256": "c" * 64,
                "interaction_promotion_sha256": "d" * 64,
            },
            expected_tensor_file="candidate.pt",
            expected_guard_manifest_sha256="f" * 64,
            expected_guard_assessment_file_sha256=_file_sha256(path),
        )


def test_guard_assessment_default_is_byte_locked_to_frozen_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate.assessment.json"
    path.write_text(json.dumps(_valid_guard_assessment()), encoding="utf-8")

    with pytest.raises(ValueError, match="bytes differ"):
        _validate_guard_assessment(
            path,
            candidate={
                "scientific_payload_sha256": "a" * 64,
                "dynamic_graph_sha256": "b" * 64,
                "compiler_pipeline_sha256": "c" * 64,
                "interaction_promotion_sha256": "d" * 64,
            },
            expected_tensor_file="candidate.pt",
            expected_guard_manifest_sha256="f" * 64,
        )


@pytest.mark.parametrize("promotion_passed", [None, False])
def test_candidate_metadata_requires_explicit_promotion(
    promotion_passed: bool | None,
) -> None:
    selection = {
        "dynamic_graph_sha256": "b" * 64,
        "compiler_pipeline_sha256": "c" * 64,
        "interaction_promotion_sha256": "d" * 64,
    }
    if promotion_passed is not None:
        selection["promotion_passed"] = promotion_passed
    raw = {
        "experiment": {},
        "selection": selection,
        "compiler_pipeline": {},
        "scientific_payload_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="did not pass"):
        experiment._candidate_metadata(raw)


def test_cli_returns_nonzero_for_failed_or_inconclusive_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "assess_gemma3_downstream_retention",
        lambda **_kwargs: {
            "evaluation": {
                "status": "inconclusive_native_denominator",
                "passed": False,
            }
        },
    )

    assert experiment.main([]) == 2
