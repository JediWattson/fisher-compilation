from __future__ import annotations

import copy
from pathlib import Path

import pytest

import fisher_graph.gemma3_l10_l17_open_a_progressive_evaluation as progressive


def _metric(native: float, delta: float, kl: float, top1: float) -> dict[str, float]:
    return {
        "nll_per_token": native + delta,
        "delta_nll_per_token": delta,
        "native_to_candidate_kl_per_token": kl,
        "top1_agreement_to_native": top1,
    }


def _assessment() -> dict[str, object]:
    native = 2.0
    conditions = {
        "layer10_dynamic": _metric(native, 0.01, 0.02, 0.94),
        "layer17_adaptive_edgeless": _metric(native, 0.02, 0.02, 0.93),
        "composed_edgeless": _metric(native, 0.031, 0.03, 0.91),
        "composed_primary": _metric(native, 0.03, 0.03, 0.92),
        "matched_double_deletion": _metric(native, 0.20, 0.20, 0.60),
    }
    families = {
        f"family_{index:02d}": {
            "supervised_tokens": 32,
            "native": {"nll_per_token": native},
            "conditions": copy.deepcopy(conditions),
        }
        for index in range(4)
    }
    resources = {
        condition: {
            **values,
            "executed_peak_live_modal_width": (
                0 if condition == "matched_double_deletion" else 48
            ),
        }
        for condition, values in progressive._EXPECTED_CONDITION_RESOURCES.items()
    }
    return {
        "execution_path": "heterogeneous_layer10_dynamic_layer17_adaptive_graph",
        "assessment_role": "adaptive_open_development_composition",
        "heldout_confirmation": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "example_count": 128,
        "family_count": 4,
        "supervised_tokens": 128,
        "logical_valid_tokens": 256,
        "native": {"nll_per_token": native},
        "conditions": conditions,
        "equal_family_macro": progressive._equal_family_macro(
            families, conditions=progressive._CONDITIONS
        ),
        "families": families,
        "graph_comparison": {
            "node_count": 8,
            "primary_interaction_count": 3,
            "edgeless_interaction_count": 0,
            "layer17_interaction_count": 0,
            "primary_edges_are_layer10_only": True,
            "node_artifacts_identical_between_composed_arms": True,
            "double_deletion_paths_agree": True,
            "deletion_equivalence_atol": 0.0,
            "deletion_equivalence_rtol": 0.0,
            "deletion_max_abs_logit_difference": 0.0,
        },
        "resource_accounting": resources,
        "observed_resources": dict(progressive._EXPECTED_RESOURCES),
        "latency_or_kernel_speed_claim": False,
    }


def _corpus() -> dict[str, object]:
    return {
        "corpus_artifact_file": "corpus.json",
        "corpus_artifact_file_sha256": "1" * 64,
        "corpus_artifact_sha256": "2" * 64,
        "receipt_file": "receipt.json",
        "receipt_file_sha256": "3" * 64,
        "receipt_sha256": "4" * 64,
        "selection_role_file": "selection.json",
        "selection_role_file_sha256": "5" * 64,
        "selection_manifest_sha256": "6" * 64,
        "ordered_membership_sha256": "7" * 64,
        "tokenizer_contract_sha256": "8" * 64,
        "example_count": 128,
        "family_count": 4,
        "assessment_role": "already_open_calibration_a_selection",
    }


def _authorization(corpus: dict[str, object]) -> dict[str, object]:
    expected = progressive._EXPECTED_AUTHORITIES
    bundle = {
        "bundle_file": "bundle.pt",
        "bundle_file_sha256": expected["composition_bundle_file_sha256"],
        "composition_payload_sha256": expected["composition_payload_sha256"],
        "combined_edgeless_graph_sha256": expected[
            "combined_edgeless_graph_sha256"
        ],
        "combined_primary_graph_sha256": expected[
            "combined_primary_graph_sha256"
        ],
        "model_fingerprint": "d" * 64,
        "parameter_cluster_plan_sha256": "e" * 64,
        "layer10_candidate_tensor_file_sha256": expected[
            "layer10_candidate_tensor_file_sha256"
        ],
        "layer10_candidate_scientific_payload_sha256": expected[
            "layer10_candidate_scientific_payload_sha256"
        ],
        "layer10_guard_evidence_file_sha256": expected[
            "layer10_guard_evidence_file_sha256"
        ],
        "layer10_guard_evidence_logical_sha256": expected[
            "layer10_guard_evidence_logical_sha256"
        ],
        "layer17_candidate_tensor_file_sha256": expected[
            "layer17_candidate_tensor_file_sha256"
        ],
        "layer17_candidate_scientific_payload_sha256": expected[
            "layer17_candidate_scientific_payload_sha256"
        ],
        "layer17_edgeless_graph_sha256": expected[
            "layer17_edgeless_graph_sha256"
        ],
        "layer17_adaptive_evidence_file_sha256": expected[
            "layer17_adaptive_result_file_sha256"
        ],
        "layer17_adaptive_evidence_logical_sha256": expected[
            "layer17_adaptive_result_sha256"
        ],
        "resources": dict(progressive._EXPECTED_RESOURCES),
    }
    without_digest = {
        "authorization_kind": (
            "frozen_layer10_guard_plus_passing_layer17_lofo_adaptive_selection"
        ),
        "authorization_completed_before_selection_open": True,
        "selection_access_authorized": True,
        "bundle": bundle,
        "layer17_lofo_report_file": "lofo.json",
        "layer17_lofo_report_file_sha256": expected[
            "layer17_lofo_report_file_sha256"
        ],
        "layer17_lofo_report_sha256": expected["layer17_lofo_report_sha256"],
        "layer17_adaptive_result_file": "adaptive.json",
        "layer17_adaptive_result_file_sha256": expected[
            "layer17_adaptive_result_file_sha256"
        ],
        "layer17_adaptive_result_sha256": expected[
            "layer17_adaptive_result_sha256"
        ],
        "prior_selection_binding": copy.deepcopy(corpus),
        "claim_role": "already_open_adaptive_development_selection",
        "fit_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "heldout_confirmation": False,
        "serving_authorized": False,
        "full_model_compiled": False,
        "source_safe": True,
    }
    return {
        **without_digest,
        "authority_sha256": progressive._domain_sha256(
            progressive._AUTHORITY_DOMAIN, without_digest
        ),
    }


def _result() -> dict[str, object]:
    assessment = _assessment()
    corpus = _corpus()
    payload: dict[str, object] = {
        "schema": progressive._SCHEMA,
        "format_version": 1,
        "scientific_role": "adaptive_open_development_composition",
        "heldout_confirmation": False,
        "authorization": _authorization(corpus),
        "corpus": corpus,
        "runtime": {
            "model_id": "google/gemma-3-270m",
            "requested_revision": "revision",
            "model_fingerprint": "d" * 64,
            "device": "cpu",
            "dtype": "float32",
            "tokenization_batch_size": 4,
            "max_length": 256,
            "vocabulary_chunk_size": 16384,
            "local_files_only": True,
        },
        "tokenization": {
            "family_stream_count": 4,
            "family_stream_catalog_sha256": "f" * 64,
            "example_count": 128,
            "logical_valid_tokens": 256,
            "supervised_tokens": 128,
            "max_length": 256,
            "tokenization_batch_size": 4,
            "contains_prompt_text": False,
            "contains_prompt_identities": False,
            "contains_token_ids": False,
        },
        "assessment": assessment,
        "decision": progressive.progressive_composition_decision(assessment),
        "bundle_changed": False,
        "evidence_changed": False,
        "selection_opened": True,
        "fit_opened": False,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "full_model_logits_scored": True,
        "full_model_compiled": False,
        "serving_authorized": False,
        "latency_or_kernel_speed_claim": False,
        "safety": dict(progressive._SAFETY),
    }
    return {
        **payload,
        "result_sha256": progressive._domain_sha256(
            progressive._RESULT_DOMAIN, payload
        ),
    }


def _rehash(result: dict[str, object]) -> None:
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    result["result_sha256"] = progressive._domain_sha256(
        progressive._RESULT_DOMAIN, payload
    )


def test_progressive_decision_passes_all_predeclared_gates() -> None:
    decision = progressive.progressive_composition_decision(_assessment())

    assert decision["all_required_gates_pass"] is True
    assert all(row["passed"] for row in decision["gate_table"])
    assert decision["derived_metrics"]["macro_deletion_recovery_fraction"] == pytest.approx(
        0.85
    )
    assert decision["derived_metrics"][
        "incremental_parameter_savings_vs_layer17_only"
    ] == 509_245


def test_progressive_decision_fails_interaction_excess_without_rewriting_policy() -> None:
    assessment = _assessment()
    replacement = _metric(2.0, 0.05, 0.03, 0.92)
    assessment["conditions"]["composed_primary"] = replacement
    for family in assessment["families"].values():
        family["conditions"]["composed_primary"] = copy.deepcopy(replacement)
    assessment["equal_family_macro"] = progressive._equal_family_macro(
        assessment["families"], conditions=progressive._CONDITIONS
    )

    decision = progressive.progressive_composition_decision(assessment)
    gates = {row["gate_id"]: row for row in decision["gate_table"]}
    assert gates["macro_interaction_excess_nll"]["passed"] is False
    assert decision["all_required_gates_pass"] is False
    assert decision["policy"]["maximum_macro_interaction_excess_nll"] == 0.01


def test_progressive_decision_fails_closed_on_nonpositive_deletion_denominator() -> None:
    assessment = _assessment()
    replacement = _metric(2.0, 0.0, 0.0, 1.0)
    assessment["conditions"]["matched_double_deletion"] = replacement
    for family in assessment["families"].values():
        family["conditions"]["matched_double_deletion"] = copy.deepcopy(replacement)
    assessment["equal_family_macro"] = progressive._equal_family_macro(
        assessment["families"], conditions=progressive._CONDITIONS
    )

    decision = progressive.progressive_composition_decision(assessment)
    assert decision["derived_metrics"][
        "macro_deletion_recovery_denominator_valid"
    ] is False
    assert decision["derived_metrics"][
        "family_deletion_recovery_invalid_denominator_count"
    ] == 4
    assert decision["all_required_gates_pass"] is False


def test_result_validator_replays_decision_authority_and_source_safety() -> None:
    result = _result()
    validated = progressive.validate_gemma3_l10_l17_open_a_progressive_result(
        result
    )
    assert validated["decision"]["all_required_gates_pass"] is True

    tampered = copy.deepcopy(result)
    tampered["decision"]["all_required_gates_pass"] = False
    _rehash(tampered)
    with pytest.raises(ValueError, match="decision"):
        progressive.validate_gemma3_l10_l17_open_a_progressive_result(tampered)

    leaked = copy.deepcopy(result)
    leaked["assessment"]["prompt"] = "secret prompt"
    _rehash(leaked)
    with pytest.raises(ValueError, match="assessment fields|forbidden"):
        progressive.validate_gemma3_l10_l17_open_a_progressive_result(leaked)


def test_result_validator_rejects_authority_and_resource_tampering() -> None:
    result = _result()
    result["authorization"]["bundle"][
        "layer17_candidate_tensor_file_sha256"
    ] = "0" * 64
    authority = result["authorization"]
    authority_payload = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    authority["authority_sha256"] = progressive._domain_sha256(
        progressive._AUTHORITY_DOMAIN, authority_payload
    )
    _rehash(result)
    with pytest.raises(ValueError, match="authority|authorization"):
        progressive.validate_gemma3_l10_l17_open_a_progressive_result(result)

    result = _result()
    result["assessment"]["observed_resources"][
        "executed_graph_macs_per_token"
    ] += 1
    result["decision"] = progressive.progressive_composition_decision(
        result["assessment"]
    )
    _rehash(result)
    validated = progressive.validate_gemma3_l10_l17_open_a_progressive_result(
        result
    )
    assert validated["decision"]["all_required_gates_pass"] is False


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    progressive._write_exclusive_atomic(path, {"value": 1})
    assert path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="overwrite"):
        progressive._write_exclusive_atomic(path, {"value": 2})


def test_prevalidation_checkpoint_survives_late_validator_failure_and_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    checkpoint = progressive._prevalidation_checkpoint_path(output)
    real_validator = progressive.validate_gemma3_l10_l17_open_a_progressive_result
    with monkeypatch.context() as scoped:
        scoped.setattr(
            progressive,
            "validate_gemma3_l10_l17_open_a_progressive_result",
            lambda value: (_ for _ in ()).throw(ValueError("late ULP defect")),
        )
        with pytest.raises(ValueError, match="late ULP defect"):
            progressive._publish_with_prevalidation_checkpoint(
                _result(), output=output
            )

    assert checkpoint.is_file()
    assert not output.exists()
    loaded = progressive.load_gemma3_l10_l17_open_a_prevalidation_checkpoint(
        checkpoint
    )
    assert loaded["status"] == "unvalidated"
    monkeypatch.setattr(
        progressive,
        "validate_gemma3_l10_l17_open_a_progressive_result",
        real_validator,
    )
    recovered = (
        progressive.finalize_gemma3_l10_l17_open_a_prevalidation_checkpoint(
            checkpoint
        )
    )
    assert recovered["result_sha256"] == _result()["result_sha256"]
    assert output.is_file()
    assert not checkpoint.exists()


def test_checkpoint_rejects_tampering_and_alternate_recovery_path(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    checkpoint_path = progressive._prevalidation_checkpoint_path(output)
    checkpoint = progressive._build_prevalidation_checkpoint(
        _result(), final_output=output
    )
    progressive._write_exclusive_atomic(checkpoint_path, checkpoint)

    with pytest.raises(ValueError, match="differs"):
        progressive.finalize_gemma3_l10_l17_open_a_prevalidation_checkpoint(
            checkpoint_path, output=tmp_path / "elsewhere.json"
        )

    tampered = copy.deepcopy(checkpoint)
    tampered["unvalidated_result"]["assessment"]["families"]["family_00"][
        "conditions"
    ]["composed_primary"]["delta_nll_per_token"] += 0.01
    progressive._write_exclusive_atomic(tmp_path / "tampered.json", tampered)
    with pytest.raises(ValueError, match="checkpoint path|hash"):
        progressive.load_gemma3_l10_l17_open_a_prevalidation_checkpoint(
            tmp_path / "tampered.json"
        )


def test_scale_aware_metric_identity_accepts_one_ulp_macro_difference() -> None:
    assessment = _assessment()
    macro = assessment["equal_family_macro"]["conditions"]["composed_primary"]
    macro["delta_nll_per_token"] = __import__("math").nextafter(
        macro["delta_nll_per_token"], float("inf")
    )

    decision = progressive.progressive_composition_decision(assessment)
    assert decision["all_required_gates_pass"] is True


def test_evaluator_authorizes_before_selection_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def forbidden_selection_loader(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("selection must not open")

    monkeypatch.setattr(
        progressive, "_load_open_selection_authority", forbidden_selection_loader
    )
    monkeypatch.setattr(
        progressive,
        "_authorize_before_selection",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("authority rejected")),
    )
    with pytest.raises(ValueError, match="authority rejected"):
        progressive.evaluate_gemma3_l10_l17_open_a_progressive(
            corpus_artifact_path=tmp_path / "corpus.json",
            selection_path=tmp_path / "selection.json",
            receipt_path=tmp_path / "receipt.json",
            output=tmp_path / "output.json",
        )
    assert called is False
