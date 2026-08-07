from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import fisher_graph.downstream_retention_summary as projection


FAMILIES = (
    "lexical_synonym",
    "noun_plural",
    "animal_movement",
    "object_material",
    "country_continent",
    "country_language",
)
NATIVE_CORRECT = {
    "lexical_synonym": 10,
    "noun_plural": 10,
    "animal_movement": 7,
    "object_material": 9,
    "country_continent": 10,
    "country_language": 10,
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(
    *,
    nll: float,
    stream_digit: str,
) -> dict[str, object]:
    families = {
        family: {
            "accuracy": correct / 10,
            "correct_count": correct,
            "example_count": 10,
            "mean_gold_margin": 1.0,
            "restricted_choice_nll": nll,
        }
        for family, correct in NATIVE_CORRECT.items()
    }
    return {
        "accuracy": 56 / 60,
        "correct_count": 56,
        "example_count": 60,
        "families": families,
        "mean_gold_margin": 1.0,
        "prediction_stream_sha256": stream_digit * 64,
        "restricted_choice_nll": nll,
    }


def _family_comparisons() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for family, native_correct in NATIVE_CORRECT.items():
        both_correct = native_correct - int(family == "object_material")
        result[family] = {
            "accuracy_retained_fraction": 1.0,
            "both_correct_count": both_correct,
            "candidate_correct_count": native_correct,
            "eligibility_source": "separate_native_only_qualification_split",
            "example_count": 10,
            "native_correct_count": native_correct,
            "native_only_correct_count": native_correct - both_correct,
            "native_win_preservation": both_correct / native_correct,
            "primary_eligible": True,
        }
    return result


def _resources() -> dict[str, object]:
    candidate = {
        "candidate_whole_model_learned_parameters": 267_789_763,
        "logical_executed_modal_graph_macs_per_token": 126_816.0,
        "logical_modal_graph_macs_per_token": 130_400.0,
        "logical_native_removed_macs_per_token": 441_600.0,
        "modal_graph_learned_parameters": 133_187,
        "native_removed_learned_parameters": 441_600,
        "net_stored_parameter_savings": 308_413,
        "replacement_scope": "partial_native_mlp_mode_replacement",
        "source_whole_model_learned_parameters": 268_098_176,
    }
    return {"candidate": candidate, "edgeless": dict(candidate)}


def _valid_assessment() -> dict[str, object]:
    package = Path(projection.__file__).resolve().parent
    repository = package.parents[1]
    native_nll = 0.30
    edgeless_nll = 0.29
    candidate_nll = 0.28
    both_correct = 55
    native_correct = 56
    qualification_counts = {
        "lexical_synonym": 4,
        "noun_plural": 5,
        "animal_movement": 3,
        "object_material": 5,
        "country_continent": 5,
        "country_language": 5,
        "color_association": 5,
        "calendar_successor": 2,
    }
    eligible = [
        family for family, count in qualification_counts.items() if count >= 3
    ]
    return {
        "schema": (
            "fisher_graph."
            "gemma3_native_qualified_downstream_retention_assessment"
        ),
        "format_version": 1,
        "model": {
            "model_id": "google/gemma-3-270m",
            "revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "adapter_model_fingerprint": "1" * 64,
        },
        "scientific_status": {
            "status": "downstream_retention_pilot_pass",
            "role": "native_qualified_downstream_retention_diagnostic",
            "candidate_executed_during_family_qualification": False,
            "candidate_refit_or_search": False,
            "task_suite_used_for_candidate_selection": False,
            "prior_candidate_diagnostic_informed_panel_redesign": True,
            "externally_standardized_benchmark": False,
            "fresh_validation": False,
            "test_data_used": False,
        },
        "candidate": dict(projection._EXPECTED_CANDIDATE),
        "prior_guard": {
            "assessment_file": (
                "state-conditioned-shape-flow-gain-dev-v2.assessment.json"
            ),
            "assessment_file_sha256": "2" * 64,
            "guard_nll_improvement_over_edgeless": 0.006,
        },
        "bank": {
            "bank_id": "gemma3-native-qualified-forced-choice-v2",
            "bank_file_sha256": _file_sha256(
                repository / "examples/gemma3_downstream_qualification_v2.json"
            ),
            "qualification_stream_sha256": "3" * 64,
            "evaluation_stream_sha256": "4" * 64,
            "claim_sha256": "5" * 64,
            "evaluator_file_sha256": _file_sha256(
                package / "gemma3_downstream_retention_v2_experiment.py"
            ),
            "shared_evaluator_file_sha256": _file_sha256(
                package / "gemma3_downstream_retention_experiment.py"
            ),
            "candidate_development_prompt_overlap_count": 0,
            "contains_prompt_or_choice_text": False,
            "contains_token_ids_or_logits": False,
        },
        "qualification": {
            "eligible_family_ids": eligible,
            "family_native_correct_counts": qualification_counts,
            "minimum_native_correct_per_family": 3,
            "selected_family_ids": list(FAMILIES),
            "sufficient_eligible_families": True,
        },
        "evaluation": {
            "adequacy": {
                "family_eligibility_source": (
                    "separate_native_only_qualification_split"
                ),
                "minimum_native_correct": 30,
                "minimum_qualified_families": 5,
                "observed_native_correct": native_correct,
                "observed_qualified_families": len(FAMILIES),
                "qualified_family_ids": list(FAMILIES),
            },
            "conditional_edge_value_added": {
                "accuracy_delta_over_edgeless": 0.0,
                "candidate_accuracy_not_below_edgeless": True,
                "candidate_choice_nll_below_edgeless": True,
                "edgeless_correct_candidate_wrong_count": 0,
                "edgeless_wrong_candidate_correct_count": 0,
                "restricted_choice_nll_improvement_over_edgeless": (
                    edgeless_nll - candidate_nll
                ),
            },
            "conditions": {
                "native": _condition(
                    nll=native_nll,
                    stream_digit="6",
                ),
                "edgeless": _condition(
                    nll=edgeless_nll,
                    stream_digit="7",
                ),
                "candidate": _condition(
                    nll=candidate_nll,
                    stream_digit="8",
                ),
            },
            "family_comparisons": _family_comparisons(),
            "paired_candidate_vs_native": {
                "accuracy_delta": 0.0,
                "accuracy_ratio_to_native": 1.0,
                "accuracy_retained_fraction": 1.0,
                "both_correct_count": both_correct,
                "both_wrong_count": 3,
                "candidate_correct_count": native_correct,
                "candidate_only_correct_count": 1,
                "native_correct_count": native_correct,
                "native_only_correct_count": 1,
                "native_win_preservation": both_correct / native_correct,
                "native_win_preservation_one_sided_90pct_wilson_lower": (
                    projection._wilson_lower(both_correct, native_correct)
                ),
                "prediction_agreement": 0.95,
                "restricted_choice_nll_delta": candidate_nll - native_nll,
                "restricted_choice_perplexity_multiplier": math.exp(
                    candidate_nll - native_nll
                ),
            },
            "gates": {name: True for name in projection._EXPECTED_GATE_NAMES},
            "passed": True,
            "status": "downstream_retention_pilot_pass",
        },
        "resources": _resources(),
        "source_model_unchanged": True,
    }


def _build(assessment: dict[str, object]) -> dict[str, object]:
    return projection.build_source_safe_downstream_retention_summary(
        assessment,
        assessment_file_sha256="9" * 64,
        as_of="2026-08-06",
    )


def test_valid_assessment_projects_source_safely_with_self_hash() -> None:
    result = _build(_valid_assessment())

    assert result["schema"] == (
        "fisher_graph.state_conditioned_downstream_retention_summary"
    )
    assert result["format_version"] == 2
    assert result["declared_label_results"]["native_correct"] == 56
    assert result["declared_label_results"]["candidate_correct"] == 56
    assert result["integrity"]["source_safe"] is True
    assert result["integrity"]["assessment_file_sha256"] == "9" * 64

    payload = dict(result)
    observed = payload.pop("summary_payload_sha256")
    expected = hashlib.sha256(
        projection._SUMMARY_HASH_DOMAIN
        + projection._canonical_json_bytes(payload)
    ).hexdigest()
    assert observed == expected

    serialized = json.dumps(result, sort_keys=True)
    assert "private prompt sentinel" not in serialized
    assert not ({"prompt", "choices", "input_ids", "logits"} & _all_keys(result))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            nested for item in value.values() for nested in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _all_keys(item)}
    return set()


@pytest.mark.parametrize("replacement", ({}, None))
def test_projection_rejects_empty_or_false_gate_receipts(
    replacement: dict[str, bool] | None,
) -> None:
    assessment = _valid_assessment()
    gates = assessment["evaluation"]["gates"]
    assert isinstance(gates, dict)
    if replacement is None:
        gates["suite_adequate"] = False
    else:
        assessment["evaluation"]["gates"] = replacement

    with pytest.raises(ValueError, match="gates"):
        _build(assessment)


def test_projection_rejects_paired_count_drift() -> None:
    assessment = _valid_assessment()
    assessment["evaluation"]["paired_candidate_vs_native"][
        "native_only_correct_count"
    ] = 2

    with pytest.raises(ValueError, match="paired correctness counts"):
        _build(assessment)


def test_projection_rejects_nll_drift() -> None:
    assessment = _valid_assessment()
    assessment["evaluation"]["paired_candidate_vs_native"][
        "restricted_choice_nll_delta"
    ] = -0.01

    with pytest.raises(ValueError, match="NLL delta"):
        _build(assessment)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("claim_sha256", "not-a-sha", "lowercase SHA-256"),
        ("evaluator_file_sha256", "0" * 64, "current source bytes"),
    ),
)
def test_projection_rejects_hash_drift(
    field: str,
    value: str,
    message: str,
) -> None:
    assessment = _valid_assessment()
    assessment["bank"][field] = value

    with pytest.raises(ValueError, match=message):
        _build(assessment)


@pytest.mark.parametrize(
    ("section", "field"),
    (("model", "prompt"), ("candidate", "weights")),
)
def test_projection_rejects_extra_source_sensitive_fields(
    section: str,
    field: str,
) -> None:
    assessment = _valid_assessment()
    assessment[section][field] = "private prompt sentinel"

    with pytest.raises(ValueError, match=f"{section} fields"):
        _build(assessment)


def test_projection_rejects_family_accounting_drift() -> None:
    assessment = _valid_assessment()
    assessment["evaluation"]["family_comparisons"]["object_material"][
        "candidate_correct_count"
    ] = 8

    with pytest.raises(ValueError, match="paired family"):
        _build(assessment)


def test_projection_rejects_resource_accounting_drift() -> None:
    assessment = _valid_assessment()
    assessment["resources"]["candidate"]["net_stored_parameter_savings"] += 1

    with pytest.raises(ValueError, match="resource accounting"):
        _build(assessment)
