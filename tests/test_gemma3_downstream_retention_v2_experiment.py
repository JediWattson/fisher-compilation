from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import fisher_graph.gemma3_downstream_retention_v2_experiment as experiment
from fisher_graph.gemma3_downstream_retention_v2_experiment import (
    DEFAULT_QUALIFIED_DOWNSTREAM_BANK,
    load_native_qualification_bank,
    select_native_qualified_families,
)


def _gold_predictions() -> tuple[int, ...]:
    bank = load_native_qualification_bank(DEFAULT_QUALIFIED_DOWNSTREAM_BANK)
    return tuple(
        row.correct_choice for row in bank.qualification_panel.examples
    )


def test_native_qualification_bank_is_balanced_and_role_disjoint() -> None:
    bank = load_native_qualification_bank(DEFAULT_QUALIFIED_DOWNSTREAM_BANK)

    assert len(bank.families) == 8
    assert len(bank.qualification_panel.examples) == 40
    for family in bank.families:
        assert len(family.qualification) == 5
        assert len(family.evaluation) == 10
        assert not (
            {row.prompt_sha256 for row in family.qualification}
            & {row.prompt_sha256 for row in family.evaluation}
        )


def test_native_qualification_selects_first_six_declared_eligible() -> None:
    bank = load_native_qualification_bank(DEFAULT_QUALIFIED_DOWNSTREAM_BANK)
    selected, receipt = select_native_qualified_families(
        bank,
        _gold_predictions(),
    )

    assert selected == tuple(family.family_id for family in bank.families[:6])
    assert receipt["sufficient_eligible_families"] is True
    assert receipt["selected_family_ids"] == selected
    panel = bank.evaluation_panel(selected)
    assert len(panel.examples) == 60
    assert panel.family_ids == tuple(sorted(selected))


def test_native_qualification_skips_ineligible_family_without_candidate_data() -> None:
    bank = load_native_qualification_bank(DEFAULT_QUALIFIED_DOWNSTREAM_BANK)
    predictions = list(_gold_predictions())
    for index in range(5):
        predictions[index] = (predictions[index] + 1) % 4

    selected, receipt = select_native_qualified_families(bank, predictions)

    assert bank.families[0].family_id not in selected
    assert selected == tuple(family.family_id for family in bank.families[1:7])
    assert receipt["family_native_correct_counts"][
        bank.families[0].family_id
    ] == 0


def test_checked_in_downstream_summary_is_source_safe_and_scoped() -> None:
    path = Path(
        "artifacts/research/state_conditioned_downstream_retention_v2.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["format_version"] == 2
    assert raw["status"] == (
        "declared_label_retention_pilot_pass_with_label_ambiguity"
    )
    results = raw["declared_label_results"]
    assert results["native_correct"] == 56
    assert results["candidate_correct"] == 56
    assert results["native_win_preservation"] == 55 / 56
    assert results["candidate_prediction_disagreement_count"] == 3
    boundary = raw["scientific_boundary"]
    assert boundary["externally_standardized_benchmark"] is False
    assert boundary["fresh_validation"] is False
    assert boundary["whole_model_compression_proven"] is False
    assert boundary["audit_replay_after_evaluator_hardening"] is True
    audit = raw["label_validity_audit"]
    assert audit["label_audit_passed"] is False
    assert audit["ambiguous_selected_family_ids"] == [
        "animal_movement",
        "object_material",
        "country_continent",
    ]
    integrity = raw["integrity"]
    assert integrity["source_safe"] is True
    assert integrity["source_model_unchanged"] is True
    for name in (
        "assessment_file_sha256",
        "downstream_claim_sha256",
        "evaluation_stream_sha256",
        "evaluator_file_sha256",
        "guard_assessment_file_sha256",
        "qualification_stream_sha256",
        "shared_evaluator_file_sha256",
    ):
        assert len(integrity[name]) == 64

    payload = dict(raw)
    observed_summary_sha256 = payload.pop("summary_payload_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert observed_summary_sha256 == hashlib.sha256(
        b"fisher-graph:downstream-retention-summary:v2\0" + encoded
    ).hexdigest()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                nested
                for item in value.values()
                for nested in keys(item)
            }
        if isinstance(value, list):
            return {nested for item in value for nested in keys(item)}
        return set()

    assert not (
        {
            "activations",
            "choices",
            "gradients",
            "input_ids",
            "logits",
            "prompt",
            "weights",
        }
        & keys(raw)
    )


def test_v2_cli_returns_nonzero_when_native_qualification_is_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "assess_gemma3_native_qualified_downstream_retention",
        lambda **_kwargs: {
            "scientific_status": {
                "status": "inconclusive_native_qualification"
            }
        },
    )

    assert experiment.main([]) == 2
