from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development
    as runner,
)


FAMILIES = tuple(f"development_family_{index}" for index in range(8))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fold(
    family: str,
    *,
    candidate: float = 0.5,
    base: float = 1.0,
    fixed_plus: float = 1.1,
    fixed_minus: float = 0.1,
    unreflected: float = 0.8,
    mirror: float = 0.9,
    matched_linear: float = 0.8,
    matched_boundary: float = 0.85,
    matched_v20m: float = 0.8,
    response: tuple[float, float, float] = (0.125, 0.125, -0.0625),
    lambda_: float = 0.5,
    changed: bool = True,
    changed_from_boundary: bool = True,
    changed_from_v20m: bool = True,
    changed_from_linear: bool = True,
    interior_distinct: bool = True,
    healthy: bool = True,
    anchors: bool = True,
    boundary_anchor: bool = True,
    v20m_anchor: bool = True,
) -> dict[str, object]:
    return {
        "outer_held_family_id": family,
        "selected_response": response,
        "selected_lambda": lambda_,
        "selected_variant_id": "reflect_coordinate_1",
        "selected_variant_artifact_sha256": _sha(f"variant:{family}"),
        "arm_order": runner._ARMS,
        "held_objective_by_arm": {
            "base": base,
            "fixed_plus": fixed_plus,
            "fixed_minus": fixed_minus,
            "matched_linear_reflected": matched_linear,
            "matched_v20l_boundary_reflected": matched_boundary,
            "same_simplex_response_unreflected": unreflected,
            "simplex_shrinkage_reflected": candidate,
            "simplex_response_reflected_exact_mirror": mirror,
            "matched_v20m_simplex_reflected": matched_v20m,
        },
        "candidate_provider_distinct_from_base": changed,
        "candidate_exact_execution_changed_from_base": changed,
        "candidate_exact_execution_changed_from_matched_v20l_boundary": (
            changed_from_boundary
        ),
        "candidate_exact_execution_changed_from_matched_v20m": changed_from_v20m,
        "candidate_exact_execution_changed_from_matched_linear": changed_from_linear,
        "interior_candidate_exact_distinct_from_both_endpoints": interior_distinct,
        "all_runtime_health_passed": healthy,
        "all_v20g_control_output_anchors_passed": anchors,
        "matched_v20l_boundary_exact_output_anchor_passed": boundary_anchor,
        "matched_v20m_exact_output_anchor_passed": v20m_anchor,
    }


def _fragments(
    folds: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {
        family: {
            "fragment_sha256": _sha(f"fragment:{family}"),
            "fold_receipt": fold,
        }
        for family, fold in folds.items()
    }


def _source() -> dict[str, object]:
    result: dict[str, object] = {
        "artifact_sha256": _sha("source"),
        "v20j_report_sha256": _sha("v20j-report"),
        "v20j_file_sha256": _sha("v20j-file"),
        "v20j_source_receipt_sha256": _sha("v20j-source"),
        "v20k_report_sha256": _sha("v20k-report"),
        "v20k_file_sha256": _sha("v20k-file"),
        "v20k_source_receipt_sha256": _sha("v20k-source"),
        "v20l_report_sha256": _sha("v20l-report"),
        "v20l_file_sha256": _sha("v20l-file"),
        "v20l_source_receipt_sha256": _sha("v20l-source"),
        "v20l_classification": "completed-development",
        "v20l_passed": False,
        "v20l_rollback_to_base": True,
    }
    for rung in ("v20g", "v20i", "v20j", "v20k", "v20l", "v20m"):
        result[f"{rung}_fold_fragment_sha256s_by_family"] = {
            family: _sha(f"{rung}:{family}") for family in FAMILIES
        }
    return result


def _install_prerequisite_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tamper: str | None = None,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    report_sha = _sha("v20m-report-pin")
    file_sha = _sha("v20m-file-pin")
    source_sha = _sha("v20m-source-pin")
    fold_pins = {family: _sha(f"v20m-pin:{family}") for family in FAMILIES}
    monkeypatch.setattr(runner, "_V20M_OUTPUT", tmp_path / "v20m.json")
    monkeypatch.setattr(runner, "_V20M_LOGICAL_SHA256", report_sha)
    monkeypatch.setattr(runner, "_V20M_FILE_SHA256", file_sha)
    monkeypatch.setattr(runner, "_V20M_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(runner, "_V20M_FOLD_SHA256S", fold_pins)

    prerequisite = {
        "nested_panel_receipt": {
            "artifact_sha256": _sha("panel"),
            "family_prompt_sha256s": {family: _sha(f"panel:{family}") for family in FAMILIES},
        },
        "authenticated_bridge_binding_sha256": _sha("bridge"),
    }
    v20a_folds = {family: {"fragment_sha256": _sha(f"v20a:{family}")} for family in FAMILIES}
    v20g_folds = {family: {"fragment_sha256": _sha(f"v20g:{family}")} for family in FAMILIES}
    v20i_folds = {family: {"fragment_sha256": _sha(f"v20i:{family}")} for family in FAMILIES}
    v20l_folds = {family: {"fragment_sha256": _sha(f"v20l:{family}")} for family in FAMILIES}
    v20m_source = {
        "artifact_sha256": source_sha,
        "v20g_fold_fragment_sha256s_by_family": {
            family: v20g_folds[family]["fragment_sha256"] for family in FAMILIES
        },
        "v20i_fold_fragment_sha256s_by_family": {
            family: v20i_folds[family]["fragment_sha256"] for family in FAMILIES
        },
        "v20l_fold_fragment_sha256s_by_family": {
            family: v20l_folds[family]["fragment_sha256"] for family in FAMILIES
        },
    }
    report: dict[str, object] = {
        "report_sha256": (
            _sha("tampered-report") if tamper == "report" else report_sha
        ),
        "all_eight_outer_folds_completed": True,
        "decision": {"integrity_passed": True},
        "final_refit": None,
        "calibration_b_opened": False,
        "source_receipt": {"artifact_sha256": source_sha},
        "fold_fragment_sha256s_by_family": dict(fold_pins),
        "classification": "completed-development",
        "passed": True,
        "rollback_to_base": False,
    }
    monkeypatch.setattr(
        runner._v20m,
        "_load_prerequisites",
        lambda: (
            prerequisite,
            v20a_folds,
            {"report_sha256": _sha("v20g-report")},
            v20g_folds,
            {"report_sha256": _sha("v20i-report")},
            v20i_folds,
            {"report_sha256": _sha("v20l-report")},
            v20l_folds,
            v20m_source,
        ),
    )
    monkeypatch.setattr(
        runner._v14,
        "_file_sha256",
        lambda path: _sha("tampered-file") if tamper == "file" else file_sha,
    )
    monkeypatch.setattr(
        runner._v20m,
        "_load_existing_report",
        lambda *args, **kwargs: report,
    )

    def load_fold(*args, outer_family_id: str, **kwargs):
        observed = fold_pins[outer_family_id]
        if tamper == "fold" and outer_family_id == FAMILIES[0]:
            observed = _sha("tampered-fold")
        return {"fragment_sha256": observed}

    monkeypatch.setattr(runner._v20m, "_load_fold_fragment", load_fold)
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda *args, **kwargs: pytest.fail(
            "prerequisite authentication constructed the model"
        ),
    )
    return report, {
        family: {"fragment_sha256": fold_pins[family]} for family in FAMILIES
    }


def test_parser_protocol_nine_arms_output_protection_and_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = runner.build_parser().parse_args([])

    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.cache_dir is None
    assert runner.DEFAULT_OUTPUT.name.endswith("v20n.json")
    assert len(runner._RESPONSES) == 19
    assert runner._SHRINKAGE_ANCHOR_LAMBDAS == (0.0, 0.5, 1.0)
    assert runner._ARMS == (
        "base",
        "fixed_plus",
        "fixed_minus",
        "matched_linear_reflected",
        "matched_v20l_boundary_reflected",
        "same_simplex_response_unreflected",
        "simplex_shrinkage_reflected",
        "simplex_response_reflected_exact_mirror",
        "matched_v20m_simplex_reflected",
    )
    assert runner._PRIMARY_ARM == "simplex_shrinkage_reflected"
    for protected in (
        runner._v20g.DEFAULT_OUTPUT,
        runner._v20i.DEFAULT_OUTPUT,
        runner._v20j.DEFAULT_OUTPUT,
        runner._v20k.DEFAULT_OUTPUT,
        runner._v20l.DEFAULT_OUTPUT,
        runner._v20m.DEFAULT_OUTPUT,
    ):
        with pytest.raises(ValueError, match="preserve immutable prerequisite"):
            runner._validate_output(protected)
    with pytest.raises(ValueError, match="directly under .local-runs"):
        runner._validate_output(Path("outside.json"))

    pyproject = Path(__file__).parents[1].joinpath("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-soft-polarity-v20n-"
        "simplex-shrinkage-nested"
    ) in pyproject
    assert runner._FIXED_PROTOCOL["outer_freeze_barrier"] == (
        "all_nine_providers_and_traces_before_outer_capability"
    )
    assert "NLL" not in str(runner._FIXED_PROTOCOL)

    observed: dict[str, object] = {}

    def fake_run(*, output: Path, cache_dir: Path | None) -> dict[str, object]:
        observed.update(output=output, cache_dir=cache_dir)
        return {"schema": "v20n-test"}

    monkeypatch.setattr(
        runner,
        "run_gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development",
        fake_run,
    )
    output = tmp_path / "report.json"
    cache = tmp_path / "cache"
    assert runner.main(["--output", str(output), "--cache-dir", str(cache)]) == 0
    assert observed == {"output": output, "cache_dir": cache}
    assert '"schema": "v20n-test"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "field",
    (
        "_V20M_LOGICAL_SHA256",
        "_V20M_FILE_SHA256",
        "_V20M_SOURCE_SHA256",
        "_V20M_FOLD_SHA256S",
    ),
)
def test_prerequisites_fail_closed_when_any_immediate_v20m_pin_is_unset(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    assert runner._V20M_LOGICAL_SHA256 is not None
    assert runner._V20M_FILE_SHA256 is not None
    assert runner._V20M_SOURCE_SHA256 is not None
    assert runner._V20M_FOLD_SHA256S is not None
    monkeypatch.setattr(runner, field, None)

    with pytest.raises(RuntimeError, match="fail-closed.*V20m"):
        runner._load_prerequisites()


def test_prerequisites_authenticate_v20m_report_file_and_all_folds_model_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected_report, expected_folds = _install_prerequisite_mocks(
        monkeypatch, tmp_path
    )

    loaded = runner._load_prerequisites()

    assert len(loaded) == 11
    assert loaded[8] == expected_report
    assert loaded[9] == expected_folds
    source = loaded[10]
    assert source["v20m_report_sha256"] == runner._V20M_LOGICAL_SHA256
    assert source["v20m_file_sha256"] == runner._V20M_FILE_SHA256
    assert source["v20m_source_receipt_sha256"] == runner._V20M_SOURCE_SHA256
    assert source["v20m_fold_fragment_sha256s_by_family"] == (
        runner._V20M_FOLD_SHA256S
    )
    assert source["authenticated_before_model_construction"] is True


@pytest.mark.parametrize(
    ("tamper", "match"),
    (
        ("file", "file hash drifted"),
        ("report", "development authority differs"),
        ("fold", "fold authority differs"),
    ),
)
def test_prerequisites_reject_tampered_v20m_authority_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    _install_prerequisite_mocks(monkeypatch, tmp_path, tamper=tamper)

    with pytest.raises(RuntimeError, match=match):
        runner._load_prerequisites()


def test_decision_positive_case_and_exact_v20m_five_win_threshold() -> None:
    passing = {family: _fold(family) for family in FAMILIES}
    decision = runner._aggregate_decision(passing)

    assert decision["candidate_win_count_by_reference_arm"]["base"] == 8
    assert decision["candidate_win_count_by_reference_arm"]["fixed_plus"] == 8
    assert decision["candidate_win_count_by_reference_arm"][
        "matched_v20m_simplex_reflected"
    ] == 8
    assert decision["selected_interior_lambda_count"] == 8
    assert decision["continuous_shrinkage_evidence_gate_passed"] is True
    assert decision[
        "all_interior_candidates_exact_distinct_from_both_endpoints"
    ] is True
    assert decision["integrity_passed"] is True
    assert decision["primary_development_gate_passed"] is True
    assert decision["mechanism_gate_passed"] is True
    assert decision["development_oof_passed"] is True

    exactly_five = {
        family: _fold(
            family,
            matched_v20m=0.8 if index < 5 else 0.1,
        )
        for index, family in enumerate(FAMILIES)
    }
    threshold = runner._aggregate_decision(exactly_five)
    assert threshold["macro_objective_by_arm"]["simplex_shrinkage_reflected"] < (
        threshold["macro_objective_by_arm"]["matched_v20m_simplex_reflected"]
    )
    assert threshold["candidate_win_count_by_reference_arm"][
        "matched_v20m_simplex_reflected"
    ] == 5
    assert threshold["mechanism_gate_passed"] is True


@pytest.mark.parametrize(
    ("case", "expected_integrity", "expected_primary", "expected_mechanism"),
    (
        ("base_wins", True, False, True),
        ("mirror_wins", True, True, False),
        ("v20m_wins", True, True, False),
        ("four_interior_lambdas", True, True, False),
        ("four_nontrivial_responses", True, True, False),
        ("not_changed", True, False, False),
        ("not_exact_distinct", False, False, False),
        ("v20m_anchor_failed", False, False, False),
    ),
)
def test_decision_negative_gates(
    case: str,
    expected_integrity: bool,
    expected_primary: bool,
    expected_mechanism: bool,
) -> None:
    folds: dict[str, dict[str, object]] = {}
    for index, family in enumerate(FAMILIES):
        options: dict[str, object] = {}
        if case == "base_wins":
            options["base"] = 0.9 if index < 5 else 0.4
        elif case == "mirror_wins":
            options["mirror"] = 0.9 if index < 5 else 0.4
        elif case == "v20m_wins":
            options["matched_v20m"] = 0.8 if index < 4 else 0.4
        elif case == "four_interior_lambdas":
            options["lambda_"] = 0.5 if index < 4 else 1.0
        elif case == "four_nontrivial_responses":
            options["response"] = (
                (0.125, 0.125, -0.0625)
                if index < 4
                else (0.125, 0.0, 0.0)
            )
        elif case == "not_changed" and index == 0:
            options["changed"] = False
        elif case == "not_exact_distinct" and index == 0:
            options["interior_distinct"] = False
        elif case == "v20m_anchor_failed" and index == 0:
            options["v20m_anchor"] = False
        folds[family] = _fold(family, **options)

    decision = runner._aggregate_decision(folds)
    assert decision["integrity_passed"] is expected_integrity
    assert decision["primary_development_gate_passed"] is expected_primary
    assert decision["mechanism_gate_passed"] is expected_mechanism
    assert decision["development_oof_passed"] is (
        expected_primary and expected_mechanism
    )
    if case == "four_interior_lambdas":
        assert decision["selected_interior_lambda_count"] == 4
        assert decision["continuous_shrinkage_evidence_gate_passed"] is False
    if case == "four_nontrivial_responses":
        assert decision["selected_nontrivial_simplex_response_count"] == 4
        assert decision["simplex_response_evidence_gate_passed"] is False


def test_exact_canonical_work_accounting_and_stage_split() -> None:
    work = runner._runner_work_accounting()

    assert work["live_authority_collection_model_forward_count"] == 32
    assert work["endpoint_reconstruction_model_forward_count"] == 112
    assert work["inner_original_response_model_forward_count"] == 2128
    assert work[
        "inner_simplex_shrinkage_lambda_half_model_forward_count"
    ] == 112
    assert work["inner_simplex_shrinkage_vertex_model_forward_count"] == 112
    assert work[
        "inner_conditional_leave_one_family_out_model_forward_count"
    ] == 2352
    assert work["outer_held_model_forward_count"] == 144
    assert work["canonical_model_forward_count"] == 2640
    assert work["canonical_teacher_access_count"] == 2608
    assert work["canonical_suffix_backward_count"] == 128
    assert work["canonical_local_autograd_contraction_count"] == 112
    assert work["simplex_response_candidate_count"] == 1064
    assert work["simplex_shrinkage_lambda_half_candidate_count"] == 56
    assert work["simplex_shrinkage_vertex_candidate_count"] == 56
    assert work["inner_provider_candidate_count"] == 1176
    assert work["inner_providers_and_traces_staged_per_outer_fold"] == 147
    assert work["inner_providers_and_traces_staged_global_count"] == 1176
    assert work["outer_arm_provider_count"] == 72
    assert work["inner_response_trace_example_count"] == 2128
    assert work[
        "inner_simplex_shrinkage_lambda_half_trace_example_count"
    ] == 112
    assert work["inner_simplex_shrinkage_vertex_trace_example_count"] == 112
    assert work["all_eight_final_refit_model_forward_count"] == 0
    assert work["calibration_b_forward_or_tokenization_count"] == 0


def test_shrinkage_receipt_key_set_and_runtime_artifact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _sha("bridge")
    base = _sha("base")
    proposal = _sha("proposal")
    provider = _sha("provider")
    runtime = _sha("runtime")
    transfer = _sha("transfer")
    response = (0.25, 0.25, -0.125)
    lambda_ = 0.5
    direction = (1.0, 0.0, 0.0, 0.0)
    endpoint = {
        "base_provider_artifact_sha256": base,
        "proposal_provider_artifact_sha256": proposal,
    }
    v20i = {
        "provider_manifest": {
            "provider_receipts": {
                "fixed_plus": {
                    "rank": 320,
                    "conditional_rank": 16,
                    "prepared_float_scalar_count": 560_003,
                    "logical_macs_per_token_upper_bound": 371_849,
                }
            }
        }
    }
    metadata = {
        "artifact_sha256": provider,
        "bridge_binding_sha256": bridge,
        "rank": 320,
        "conditional_rank": 16,
        "prepared_float_scalar_count": 560_007,
        "logical_macs_per_token_upper_bound": 371_850,
    }
    validated_payload = {
        "protocol_sha256": (
            runner.FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256
        ),
        "compiled_simplex_response_provider_artifact_sha256": runtime,
        "bridge_binding_sha256": bridge,
        "base_provider_artifact_sha256": base,
        "proposal_provider_artifact_sha256": proposal,
        "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
        "transfer_evidence_sha256": transfer,
        "direction_sha256": (
            runner.fisher_soft_polarity_simplex_shrinkage_direction_sha256(
                runner._v20g._eta_tensor(direction)
            )
        ),
        "source_radius": response[0],
        "source_shrink_mass": response[1],
        "source_polarity_bias": response[2],
        "shrinkage_lambda": lambda_,
    }
    monkeypatch.setattr(
        runner,
        "validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence",
        lambda payload, observed_metadata: SimpleNamespace(
            artifact_sha256=provider,
            payload=validated_payload,
            metadata={"box_certificate": {}},
        ),
    )
    receipt = runner._hashed(
        {
            "role": "inner_simplex_shrinkage_vertex",
            "source_response": response,
            "source_response_key": runner._parameters_key(response),
            "lambda": lambda_,
            "lambda_hex": lambda_.hex(),
            "effective_response": (0.25, 0.125, -0.0625),
            "direction": direction,
            "direction_box_corner_scores": runner._box_corner_scores(direction),
            "box_certificate": {},
            "provider_artifact_sha256": provider,
            "runtime_provider_artifact_sha256": runtime,
            "lineage_wrapper_not_inference_executor": True,
            "provider_metadata": metadata,
            "provider_metadata_sha256": runner._v14._sha256(
                metadata, domain=runner._PROVIDER_DOMAIN
            ),
            "provider_payload": {"synthetic": True},
            "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
            "transfer_evidence_sha256": transfer,
            "rank": 320,
            "conditional_rank": 16,
            "prepared_float_scalar_count": 560_007,
            "logical_macs_per_token_upper_bound": 371_850,
            "analysis_only": True,
            "raw_provider_tensors_serialized": False,
        },
        domain=runner._PROVIDER_DOMAIN,
    )
    assert set(receipt) == runner._SHRINKAGE_PROVIDER_RECEIPT_KEYS

    kwargs = {
        "expected_role": "inner_simplex_shrinkage_vertex",
        "expected_provider_artifact_sha256": provider,
        "expected_endpoint_receipt": endpoint,
        "expected_bridge_binding_sha256": bridge,
        "authenticated_v20i_fold": v20i,
        "expected_response": response,
        "expected_lambda": lambda_,
        "expected_direction": direction,
        "expected_transfer_evidence_sha256": transfer,
    }
    runner._validate_shrinkage_provider_receipt_evidence(receipt, **kwargs)
    serialized_receipt = json.loads(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    )
    runner._validate_shrinkage_provider_receipt_evidence(
        serialized_receipt, **kwargs
    )

    forged_runtime = copy.deepcopy(receipt)
    forged_runtime["runtime_provider_artifact_sha256"] = _sha("forged-runtime")
    with pytest.raises(ValueError, match="runtime provider artifact"):
        runner._validate_shrinkage_provider_receipt_evidence(
            forged_runtime, **kwargs
        )

    missing_key = copy.deepcopy(receipt)
    missing_key.pop("runtime_provider_artifact_sha256")
    with pytest.raises(ValueError, match="key set"):
        runner._validate_shrinkage_provider_receipt_evidence(missing_key, **kwargs)


def test_exact_scoring_executes_and_hashes_the_compiled_runtime_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_artifact = _sha("lineage-wrapper")
    runtime_artifact = _sha("compiled-runtime")

    class Wrapper:
        def __init__(self) -> None:
            self.artifact_sha256 = wrapper_artifact
            self.runtime_provider = SimpleNamespace(
                artifact_sha256=runtime_artifact
            )

    monkeypatch.setattr(
        runner,
        "AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider",
        Wrapper,
    )
    provider = Wrapper()
    record = SimpleNamespace(
        sequence=SimpleNamespace(
            example_id="example-0",
            family_id="family-0",
        )
    )
    observed: dict[str, object] = {}

    class Bridge:
        def execute(self, adapter, model_inputs, *, h4_head):
            observed["executed_provider"] = h4_head
            return "execution"

    class Capability:
        def get(self, example_id: str, *, family_id: str):
            return "teacher"

    context = SimpleNamespace(adapter="adapter", bridge=Bridge())
    monkeypatch.setattr(
        runner._v20b, "_ordered_records", lambda records: tuple(records)
    )
    monkeypatch.setattr(
        runner._v20a,
        "_verified_model_inputs",
        lambda context, record: ("inputs", (0,), None),
    )

    def score(**kwargs):
        observed["scored_provider_artifact"] = kwargs[
            "provider_artifact_sha256"
        ]
        return 0.25, _sha("h4"), _sha("logits")

    monkeypatch.setattr(runner._v20a, "_execution_hashes_and_score", score)
    evidence = _sha("evidence")
    domain = b"v20n-runtime-test\0"
    objectives, h4, logits, executions = runner._score_exact_provider(
        context,
        (record,),
        Capability(),
        provider=provider,
        phase="runtime_provider_replay",
        outer_family_id="outer-family",
        inner_family_id="family-0",
        role="simplex_shrinkage_reflected",
        evidence_sha256=evidence,
        domain=domain,
    )

    assert observed["executed_provider"] is provider.runtime_provider
    assert observed["scored_provider_artifact"] == runtime_artifact
    assert objectives == {"example-0": 0.25}
    assert h4 == {"example-0": _sha("h4")}
    assert logits == {"example-0": _sha("logits")}
    assert executions == {
        "example-0": runner._execution_sha256(
            phase="runtime_provider_replay",
            outer_family_id="outer-family",
            inner_family_id="family-0",
            role="simplex_shrinkage_reflected",
            provider_artifact_sha256=runtime_artifact,
            example_id="example-0",
            family_id="family-0",
            objective=0.25,
            h4_sha256=_sha("h4"),
            logits_sha256=_sha("logits"),
            evidence_sha256=evidence,
            domain=domain,
        )
    }


def test_all_half_then_all_vertex_providers_freeze_before_stage_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = FAMILIES[-1]
    inner_families = FAMILIES[:-1]
    selected_response = (0.125, 0.125, -0.0625)
    linear_response = (0.125, 0.0, 0.0)
    selected_key = runner._response_key(selected_response)
    linear_key = runner._response_key(linear_response)
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=family,
                example_id=f"{family}:{index}",
            )
        )
        for family in inner_families
        for index in range(2)
    )
    endpoint = SimpleNamespace(training_records=records)
    inner_evidence: dict[str, dict[str, object]] = {}
    for family in inner_families:
        example_ids = tuple(
            record.sequence.example_id
            for record in records
            if record.sequence.family_id == family
        )
        response_evidence = {
            linear_key: {
                "objectives_by_example": {
                    example: 1.0625 for example in example_ids
                },
                "post_cast_h4_sha256s": {
                    example: _sha(f"linear-h4:{example}") for example in example_ids
                },
                "supervised_full_vocab_logits_sha256s": {
                    example: _sha(f"linear-logits:{example}")
                    for example in example_ids
                },
            },
            selected_key: {
                "objectives_by_example": {
                    example: 1.5625 for example in example_ids
                },
                "post_cast_h4_sha256s": {
                    example: _sha(f"source-h4:{example}") for example in example_ids
                },
                "supervised_full_vocab_logits_sha256s": {
                    example: _sha(f"source-logits:{example}")
                    for example in example_ids
                },
            },
        }
        inner_evidence[family] = {
            "reflection_fit_receipt": {
                "artifact_sha256": _sha(f"fit:{family}"),
                "selected_variant_artifact_sha256": _sha(f"variant:{family}"),
                "selected_variant_available": True,
                "selected_normalized_direction": (1.0, 0.0, 0.0, 0.0),
            },
            "objective_by_response": {
                linear_key: 1.0625,
                selected_key: 1.5625,
            },
            "response_evidence": response_evidence,
        }
    response_selection = {
        "artifact_sha256": _sha("response-selection"),
        "objectives_by_inner_family_and_response": {},
        "family_equal_objective_by_response": {},
        "simplex_response_selection_receipt": {},
        "selected_response": selected_response,
    }
    v20m_inner = {"inner_evidence_by_family": inner_evidence}
    authenticated_v20m = {
        "response_selection_receipt": copy.deepcopy(response_selection),
        "inner_receipt": {
            "inner_evidence_by_family": copy.deepcopy(inner_evidence)
        },
    }
    monkeypatch.setattr(
        runner,
        "_fit_inner_response",
        lambda *args, **kwargs: (v20m_inner, response_selection),
    )

    events: list[str] = []

    def materialize(*args, **kwargs):
        role = str(kwargs["role"])
        inner = str(kwargs["inner_family_id"])
        events.append(f"build:{role}:{inner}")
        return (
            SimpleNamespace(artifact_sha256=_sha(f"{role}:{inner}")),
            _sha(f"seed:{role}:{inner}"),
        )

    monkeypatch.setattr(runner, "_materialize_shrinkage_provider", materialize)
    monkeypatch.setattr(
        runner,
        "_provider_trace",
        lambda provider, records, *, role: {
            "artifact_sha256": _sha(f"trace:{role}:{provider.artifact_sha256}")
        },
    )
    monkeypatch.setattr(
        runner,
        "_shrinkage_provider_receipt",
        lambda provider, **kwargs: {
            "artifact_sha256": _sha(f"receipt:{provider.artifact_sha256}")
        },
    )

    class FirstVertexCapability(RuntimeError):
        pass

    class Capability:
        def receipt(self) -> dict[str, object]:
            return {}

    class Vault:
        calls = 0

        def capability(self, *args, **kwargs):
            self.calls += 1
            events.append(f"capability:{self.calls}")
            if self.calls == 1:
                assert len(
                    [event for event in events if event.startswith("build:inner_simplex_shrinkage_lambda_half")]
                ) == 7
            if self.calls == 8:
                assert len(
                    [event for event in events if event.startswith("build:inner_simplex_shrinkage_vertex")]
                ) == 7
                raise FirstVertexCapability
            return Capability()

    def score(context, records, capability, *, role: str, **kwargs):
        events.append(f"score:{role}")
        objectives = {
            record.sequence.example_id: 1.0625 for record in records
        }
        return (
            objectives,
            {example: _sha(f"h4:{example}") for example in objectives},
            {example: _sha(f"logits:{example}") for example in objectives},
            {example: _sha(f"execution:{example}") for example in objectives},
        )

    monkeypatch.setattr(runner, "_score_exact_provider", score)
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )

    with pytest.raises(FirstVertexCapability):
        runner._fit_inner_shrinkage(
            object(),
            endpoint,
            {},
            Vault(),
            outer_family_id=outer,
            authenticated_v20g_fold={},
            authenticated_v20m_fold=authenticated_v20m,
        )

    assert events.count("score:inner_simplex_shrinkage_lambda_half") == 7
    first_capability = events.index("capability:1")
    last_half_build = max(
        index
        for index, event in enumerate(events)
        if event.startswith("build:inner_simplex_shrinkage_lambda_half")
    )
    assert last_half_build < first_capability
    first_vertex_capability = events.index("capability:8")
    last_vertex_build = max(
        index
        for index, event in enumerate(events)
        if event.startswith("build:inner_simplex_shrinkage_vertex")
    )
    assert last_vertex_build < first_vertex_capability


@pytest.mark.parametrize("tampered_field", ("objective", "h4", "logits"))
def test_outer_scoring_rejects_each_exact_v20m_anchor_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tampered_field: str,
) -> None:
    outer = FAMILIES[0]
    records = tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(
                family_id=outer,
                example_id=f"{outer}:{index}",
            )
        )
        for index in range(2)
    )
    providers = {
        arm: SimpleNamespace(artifact_sha256=_sha(f"provider:{arm}"))
        for arm in runner._ARMS
    }
    traces = {
        arm: {
            "artifact_sha256": _sha(f"trace:{arm}"),
            "finite": True,
            "pointwise_trust_passed": True,
            "endpoint_conditional_ranks_are_16": True,
        }
        for arm in runner._ARMS
    }
    manifest = {
        "artifact_sha256": _sha("outer-manifest"),
        "provider_artifact_sha256s": {
            arm: providers[arm].artifact_sha256 for arm in runner._ARMS
        },
        "all_nine_providers_frozen_before_outer_capability": True,
        "all_nine_traces_frozen_before_outer_capability": True,
        "outer_capability_count_at_freeze": 0,
        "outer_objectives_or_teacher_rows_used_at_freeze": False,
    }
    monkeypatch.setattr(
        runner,
        "_freeze_outer_providers",
        lambda *args, **kwargs: (providers, manifest, traces),
    )
    results: dict[
        str,
        tuple[dict[str, float], dict[str, str], dict[str, str], dict[str, str]],
    ] = {}
    for arm_index, arm in enumerate(runner._ARMS):
        objectives = {
            record.sequence.example_id: 1.0 + arm_index * 0.1 + index * 0.01
            for index, record in enumerate(records)
        }
        results[arm] = (
            objectives,
            {
                example: _sha(f"h4:{arm}:{example}") for example in objectives
            },
            {
                example: _sha(f"logits:{arm}:{example}")
                for example in objectives
            },
            {
                example: _sha(f"execution:{arm}:{example}")
                for example in objectives
            },
        )
    monkeypatch.setattr(
        runner,
        "_score_exact_provider",
        lambda *args, role, **kwargs: results[role],
    )
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )

    class Capability:
        def receipt(self) -> dict[str, object]:
            return {}

    class Vault:
        def capability(self, *args, **kwargs):
            return Capability()

    def inherited(arm: str) -> dict[str, object]:
        objectives, h4, logits, _executions = results[arm]
        return {
            "objective": sum(objectives.values()) / len(objectives),
            "objectives_by_example": copy.deepcopy(objectives),
            "post_cast_h4_sha256s": copy.deepcopy(h4),
            "supervised_full_vocab_logits_sha256s": copy.deepcopy(logits),
        }

    v20g = {
        "held_evidence": {
            "arm_evidence": {
                arm: inherited(arm)
                for arm in ("base", "fixed_plus", "fixed_minus")
            }
        }
    }
    v20l = {
        "held_evidence": {
            "arm_evidence": {
                "signed_stack_reflected": inherited(
                    "matched_v20l_boundary_reflected"
                )
            }
        }
    }
    v20m_arm = inherited("matched_v20m_simplex_reflected")
    first_example = records[0].sequence.example_id
    if tampered_field == "objective":
        v20m_arm["objective"] = float(v20m_arm["objective"]) + 1.0
    elif tampered_field == "h4":
        v20m_arm["post_cast_h4_sha256s"][first_example] = _sha("forged-h4")
    else:
        v20m_arm["supervised_full_vocab_logits_sha256s"][first_example] = _sha(
            "forged-logits"
        )
    v20m = {
        "held_evidence": {
            "arm_evidence": {"simplex_response_reflected": v20m_arm}
        }
    }

    with pytest.raises(RuntimeError, match="V20m.*exact objective/output"):
        runner._score_outer_arms(
            object(),
            object(),
            records,
            Vault(),
            {},
            {
                "selected_variant_id": "reflect_coordinate_1",
                "selected_variant_artifact_sha256": _sha("variant"),
            },
            selected_response=(0.125, 0.125, -0.0625),
            selected_lambda=0.5,
            outer_family_id=outer,
            authenticated_v20g_fold=v20g,
            authenticated_v20l_fold=v20l,
            authenticated_v20m_fold=v20m,
        )


def test_report_makes_no_serving_compression_speed_or_current_refit_claims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    folds = {family: _fold(family) for family in FAMILIES}
    report = runner._build_report(
        output=tmp_path / "v20n.json",
        source=_source(),
        v20g_report={
            "report_sha256": _sha("v20g-report"),
            "classification": "failed",
            "passed": False,
            "rollback_to_base": True,
        },
        v20i_report={
            "report_sha256": _sha("v20i-report"),
            "classification": "diagnostic-complete",
            "passed": False,
            "rollback_to_base": True,
        },
        v20m_report={
            "report_sha256": _sha("v20m-report"),
            "classification": "completed-development",
            "development_oof_passed": True,
            "primary_development_gate_passed": True,
            "mechanism_gate_passed": True,
            "passed": True,
            "rollback_to_base": False,
        },
        panel_receipt={"artifact_sha256": _sha("panel")},
        bridge_binding_sha256=_sha("bridge"),
        fold_fragments=_fragments(folds),
    )

    assert report["passed"] is True
    assert report["final_refit_authorized_for_next_fresh_shadow"] is True
    assert report["fresh_family_disjoint_shadow_eligible"] is True
    assert report["fresh_family_disjoint_scoring_performed"] is False
    assert report["all_eight_final_refit_completed"] is False
    assert report["full_refit_performed"] is False
    assert report["final_refit"] is None
    assert report["final_provider_frozen"] is False
    assert report["calibration_b_opened"] is False
    assert report["serving_claim_authorized"] is False
    assert report["compression_claim_authorized"] is False
    assert report["speed_claim_authorized"] is False
    assert report["candidate"] is None
    assert report["provider_sidecar"] is None


def test_completed_fold_fast_path_is_model_free_and_threads_v20m_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "v20n.json"
    prerequisite = {
        "nested_panel_receipt": {
            "artifact_sha256": _sha("panel"),
            "family_prompt_sha256s": {
                family: _sha(f"panel:{family}") for family in FAMILIES
            },
        },
        "authenticated_bridge_binding_sha256": _sha("bridge"),
    }
    maps = tuple(
        {
            family: {"fragment_sha256": _sha(f"authority-{index}:{family}")}
            for family in FAMILIES
        }
        for index in range(5)
    )
    v20a, v20g, v20i, v20l, v20m = maps
    v20g_report = {"report_sha256": _sha("v20g-report")}
    v20i_report = {"report_sha256": _sha("v20i-report")}
    v20l_report = {"report_sha256": _sha("v20l-report")}
    v20m_report = {"report_sha256": _sha("v20m-report")}
    source = {"artifact_sha256": _sha("source")}
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_load_prerequisites",
        lambda: (
            prerequisite,
            v20a,
            v20g_report,
            v20g,
            v20i_report,
            v20i,
            v20l_report,
            v20l,
            v20m_report,
            v20m,
            source,
        ),
    )
    fold_paths = {
        family: tmp_path / f"{family}.fold.json" for family in FAMILIES
    }
    for path in fold_paths.values():
        path.touch()
    monkeypatch.setattr(
        runner,
        "_fold_path",
        lambda output, family: fold_paths[family],
    )
    fragments = {
        family: {
            "fragment_sha256": _sha(f"fragment:{family}"),
            "fold_receipt": _fold(family),
        }
        for family in FAMILIES
    }
    monkeypatch.setattr(
        runner,
        "_load_fold_fragment",
        lambda *args, outer_family_id, **kwargs: fragments[outer_family_id],
    )
    observed: dict[str, object] = {}

    def build_report(**kwargs):
        observed["build_v20m_report"] = kwargs["v20m_report"]
        observed["build_folds"] = kwargs["fold_fragments"]
        return {"schema": "v20n-completed-fold-report"}

    def load_report(output, **kwargs):
        observed["load_v20m_report"] = kwargs["v20m_report"]
        return {"loaded": True}

    monkeypatch.setattr(runner, "_build_report", build_report)
    monkeypatch.setattr(runner, "_load_existing_report", load_report)
    monkeypatch.setattr(
        runner._v20b, "_publish_scalar_fragment", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda *args, **kwargs: pytest.fail(
            "completed-fold fast path constructed the model"
        ),
    )

    result = runner.run_gemma3_l3_l4_complete_h4_soft_polarity_simplex_shrinkage_nested_development(
        output=destination
    )

    assert result == {"loaded": True}
    assert observed["build_v20m_report"] is v20m_report
    assert observed["load_v20m_report"] is v20m_report
    assert observed["build_folds"] == fragments
