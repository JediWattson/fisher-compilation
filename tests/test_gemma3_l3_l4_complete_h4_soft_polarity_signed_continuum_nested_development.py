from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development
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
    signed_scalar: float | None = None,
    changed: bool = True,
    changed_from_boundary: bool = True,
    changed_from_v20m: bool = True,
    changed_from_linear: bool = True,
    interior_distinct: bool = True,
    endpoint_anchor: bool = True,
    healthy: bool = True,
    anchors: bool = True,
    boundary_anchor: bool = True,
    v20m_anchor: bool = True,
    mirror_anchor: bool = True,
) -> dict[str, object]:
    if signed_scalar is None:
        signed_scalar = -0.5 if FAMILIES.index(family) < 2 else 0.5
    return {
        "outer_held_family_id": family,
        "selected_response": response,
        "selected_signed_scalar": signed_scalar,
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
            "signed_continuum_reflected": candidate,
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
        "interior_candidate_exact_distinct_from_all_three_anchors": (
            interior_distinct
        ),
        "selected_endpoint_exact_anchor_passed": endpoint_anchor,
        "all_runtime_health_passed": healthy,
        "all_v20g_control_output_anchors_passed": anchors,
        "matched_v20l_boundary_exact_output_anchor_passed": boundary_anchor,
        "matched_v20m_exact_output_anchor_passed": v20m_anchor,
        "exact_mirror_v20m_exact_output_anchor_passed": mirror_anchor,
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
    for rung in ("v20g", "v20i", "v20j", "v20k", "v20l", "v20m", "v20n"):
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
    report_sha = _sha("v20n-report-pin")
    file_sha = _sha("v20n-file-pin")
    source_sha = _sha("v20n-source-pin")
    fold_pins = {family: _sha(f"v20n-pin:{family}") for family in FAMILIES}
    monkeypatch.setattr(runner, "_V20N_OUTPUT", tmp_path / "v20n.json")
    monkeypatch.setattr(runner, "_V20N_LOGICAL_SHA256", report_sha)
    monkeypatch.setattr(runner, "_V20N_FILE_SHA256", file_sha)
    monkeypatch.setattr(runner, "_V20N_SOURCE_SHA256", source_sha)
    monkeypatch.setattr(runner, "_V20N_FOLD_SHA256S", fold_pins)

    prerequisite = {
        "nested_panel_receipt": {
            "artifact_sha256": _sha("panel"),
            "family_prompt_sha256s": {
                family: _sha(f"panel:{family}") for family in FAMILIES
            },
        },
        "authenticated_bridge_binding_sha256": _sha("bridge"),
    }
    authority_folds = tuple(
        {
            family: {"fragment_sha256": _sha(f"authority-{index}:{family}")}
            for family in FAMILIES
        }
        for index in range(5)
    )
    v20a, v20g, v20i, v20l, v20m = authority_folds
    v20n_source = {
        "artifact_sha256": source_sha,
        "v20g_fold_fragment_sha256s_by_family": {
            family: v20g[family]["fragment_sha256"] for family in FAMILIES
        },
        "v20i_fold_fragment_sha256s_by_family": {
            family: v20i[family]["fragment_sha256"] for family in FAMILIES
        },
        "v20l_fold_fragment_sha256s_by_family": {
            family: v20l[family]["fragment_sha256"] for family in FAMILIES
        },
        "v20m_fold_fragment_sha256s_by_family": {
            family: v20m[family]["fragment_sha256"] for family in FAMILIES
        },
    }
    report: dict[str, object] = {
        "report_sha256": _sha("tampered-report") if tamper == "report" else report_sha,
        "all_eight_outer_folds_completed": True,
        "decision": {"integrity_passed": True},
        "final_refit": None,
        "calibration_b_opened": False,
        "source_receipt": {
            "artifact_sha256": (
                _sha("tampered-source") if tamper == "source" else source_sha
            )
        },
        "fold_fragment_sha256s_by_family": dict(fold_pins),
        "classification": "completed-development",
        "passed": True,
        "rollback_to_base": False,
    }
    monkeypatch.setattr(
        runner._v20n,
        "_load_prerequisites",
        lambda: (
            prerequisite,
            v20a,
            {"report_sha256": _sha("v20g-report")},
            v20g,
            {"report_sha256": _sha("v20i-report")},
            v20i,
            {"report_sha256": _sha("v20l-report")},
            v20l,
            {"report_sha256": _sha("v20m-report")},
            v20m,
            v20n_source,
        ),
    )
    monkeypatch.setattr(
        runner._v14,
        "_file_sha256",
        lambda path: _sha("tampered-file") if tamper == "file" else file_sha,
    )
    monkeypatch.setattr(
        runner._v20n,
        "_load_existing_report",
        lambda *args, **kwargs: report,
    )

    def load_fold(*args, outer_family_id: str, **kwargs):
        observed = fold_pins[outer_family_id]
        if tamper == "fold" and outer_family_id == FAMILIES[0]:
            observed = _sha("tampered-fold")
        return {"fragment_sha256": observed}

    monkeypatch.setattr(runner._v20n, "_load_fold_fragment", load_fold)
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


def test_parser_protocol_anchors_arms_and_output_protection() -> None:
    arguments = runner.build_parser().parse_args([])

    assert arguments.output == runner.DEFAULT_OUTPUT
    assert arguments.cache_dir is None
    assert runner.DEFAULT_OUTPUT.name.endswith("v20o.json")
    assert len(runner._RESPONSES) == 19
    assert runner._SIGNED_CONTINUUM_ANCHOR_VALUES == (-1.0, 0.0, 1.0)
    assert runner._PRIMARY_ARM == "signed_continuum_reflected"
    assert len(runner._ARMS) == 9
    assert runner._ARMS[-2:] == (
        "simplex_response_reflected_exact_mirror",
        "matched_v20m_simplex_reflected",
    )
    for protected in (
        runner._v20g.DEFAULT_OUTPUT,
        runner._v20i.DEFAULT_OUTPUT,
        runner._v20l.DEFAULT_OUTPUT,
        runner._v20m.DEFAULT_OUTPUT,
        runner._v20n.DEFAULT_OUTPUT,
    ):
        with pytest.raises(ValueError, match="preserve immutable prerequisite"):
            runner._validate_output(protected)
    with pytest.raises(ValueError, match="directly under .local-runs"):
        runner._validate_output(Path("outside.json"))
    assert runner._FIXED_PROTOCOL["outer_freeze_barrier"] == (
        "all_nine_providers_and_traces_before_outer_capability"
    )
    assert "NLL" not in str(runner._FIXED_PROTOCOL)


@pytest.mark.parametrize(
    "field",
    (
        "_V20N_LOGICAL_SHA256",
        "_V20N_FILE_SHA256",
        "_V20N_SOURCE_SHA256",
        "_V20N_FOLD_SHA256S",
    ),
)
def test_prerequisites_fail_closed_when_any_immediate_v20n_pin_is_unset(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setattr(runner, field, None)
    with pytest.raises(RuntimeError, match="fail-closed.*V20n"):
        runner._load_prerequisites()


def test_prerequisites_authenticate_v20n_report_file_source_and_folds_model_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_report, expected_folds = _install_prerequisite_mocks(
        monkeypatch, tmp_path
    )
    loaded = runner._load_prerequisites()

    assert len(loaded) == 13
    assert loaded[10] == expected_report
    assert loaded[11] == expected_folds
    source = loaded[12]
    assert source["v20n_report_sha256"] == runner._V20N_LOGICAL_SHA256
    assert source["v20n_file_sha256"] == runner._V20N_FILE_SHA256
    assert source["v20n_source_receipt_sha256"] == runner._V20N_SOURCE_SHA256
    assert source["v20n_fold_fragment_sha256s_by_family"] == (
        runner._V20N_FOLD_SHA256S
    )
    assert source["authenticated_before_model_construction"] is True


@pytest.mark.parametrize(
    ("tamper", "match"),
    (
        ("file", "file hash drifted"),
        ("report", "development authority differs"),
        ("source", "development authority differs"),
        ("fold", "fold authority differs"),
    ),
)
def test_prerequisites_reject_tampered_v20n_authority_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    _install_prerequisite_mocks(monkeypatch, tmp_path, tamper=tamper)
    with pytest.raises(RuntimeError, match=match):
        runner._load_prerequisites()


def test_decision_positive_case_has_both_strict_interior_signs() -> None:
    decision = runner._aggregate_decision(
        {family: _fold(family) for family in FAMILIES}
    )

    assert decision["selected_interior_signed_scalar_count"] == 8
    assert decision["selected_negative_interior_signed_scalar_count"] == 2
    assert decision["selected_positive_interior_signed_scalar_count"] == 6
    assert decision["continuous_signed_continuum_evidence_gate_passed"] is True
    assert decision[
        "all_interior_candidates_exact_distinct_from_all_three_anchors"
    ] is True
    assert decision["integrity_passed"] is True
    assert decision["primary_development_gate_passed"] is True
    assert decision["mechanism_gate_passed"] is True
    assert decision["development_oof_passed"] is True
    assert "interior_candidate_exact_distinct_from_both_endpoints_by_family" not in decision


@pytest.mark.parametrize(
    ("reference", "threshold", "gate"),
    (
        ("base", 6, "primary_development_gate_passed"),
        ("fixed_plus", 6, "primary_development_gate_passed"),
        ("same_simplex_response_unreflected", 5, "mechanism_gate_passed"),
        ("simplex_response_reflected_exact_mirror", 6, "mechanism_gate_passed"),
        ("matched_linear_reflected", 5, "mechanism_gate_passed"),
        ("matched_v20l_boundary_reflected", 5, "mechanism_gate_passed"),
        ("matched_v20m_simplex_reflected", 5, "mechanism_gate_passed"),
    ),
)
def test_decision_uses_exact_predeclared_win_thresholds(
    reference: str, threshold: int, gate: str
) -> None:
    keyword = {
        "base": "base",
        "fixed_plus": "fixed_plus",
        "same_simplex_response_unreflected": "unreflected",
        "simplex_response_reflected_exact_mirror": "mirror",
        "matched_linear_reflected": "matched_linear",
        "matched_v20l_boundary_reflected": "matched_boundary",
        "matched_v20m_simplex_reflected": "matched_v20m",
    }[reference]
    exact = {
        family: _fold(
            family,
            **{keyword: 0.8 if index < threshold else 0.4},
        )
        for index, family in enumerate(FAMILIES)
    }
    exact_decision = runner._aggregate_decision(exact)
    assert exact_decision["candidate_win_count_by_reference_arm"][reference] == threshold
    assert exact_decision[gate] is True

    below = {
        family: _fold(
            family,
            **{keyword: 0.8 if index < threshold - 1 else 0.4},
        )
        for index, family in enumerate(FAMILIES)
    }
    below_decision = runner._aggregate_decision(below)
    assert below_decision["candidate_win_count_by_reference_arm"][reference] == threshold - 1
    assert below_decision[gate] is False


def test_endpoint_signs_do_not_satisfy_strict_interior_sign_diversity() -> None:
    values = (0.2, 0.3, 0.4, 0.5, 0.6, -1.0, 0.0, 1.0)
    decision = runner._aggregate_decision(
        {
            family: _fold(family, signed_scalar=values[index])
            for index, family in enumerate(FAMILIES)
        }
    )

    assert decision["selected_interior_signed_scalar_count"] == 5
    assert decision["selected_negative_interior_signed_scalar_count"] == 0
    assert decision["selected_positive_interior_signed_scalar_count"] == 5
    assert decision["continuous_signed_continuum_evidence_gate_passed"] is False
    assert decision["mechanism_gate_passed"] is False


@pytest.mark.parametrize("failure", ("three_anchor_distinct", "endpoint_anchor", "mirror_anchor"))
def test_decision_integrity_rejects_anchor_failures(failure: str) -> None:
    folds = {family: _fold(family) for family in FAMILIES}
    target = folds[FAMILIES[0]]
    if failure == "three_anchor_distinct":
        target["interior_candidate_exact_distinct_from_all_three_anchors"] = False
    elif failure == "endpoint_anchor":
        target["selected_endpoint_exact_anchor_passed"] = False
    else:
        target["exact_mirror_v20m_exact_output_anchor_passed"] = False

    decision = runner._aggregate_decision(folds)
    assert decision["integrity_passed"] is False
    assert decision["development_oof_passed"] is False


def test_exact_canonical_work_accounting_and_stage_split() -> None:
    work = runner._runner_work_accounting()

    assert work["live_authority_collection_model_forward_count"] == 32
    assert work["endpoint_reconstruction_model_forward_count"] == 112
    assert work["inner_original_response_model_forward_count"] == 2128
    assert work["inner_signed_continuum_missing_anchor_model_forward_count"] == 224
    assert work["inner_signed_continuum_vertex_model_forward_count"] == 112
    assert work["inner_conditional_leave_one_family_out_model_forward_count"] == 2464
    assert work["outer_held_model_forward_count"] == 144
    assert work["canonical_model_forward_count"] == 2752
    assert work["canonical_teacher_access_count"] == 2720
    assert work["canonical_suffix_backward_count"] == 128
    assert work["canonical_local_autograd_contraction_count"] == 112
    assert work["simplex_response_candidate_count"] == 1064
    assert work["signed_continuum_missing_anchor_candidate_count"] == 112
    assert work["signed_continuum_vertex_candidate_count"] == 56
    assert work["inner_provider_candidate_count"] == 1232
    assert work["inner_providers_and_traces_staged_per_outer_fold"] == 154
    assert work["inner_providers_and_traces_staged_global_count"] == 1232
    assert work["outer_arm_provider_count"] == 72
    assert work["all_eight_final_refit_model_forward_count"] == 0
    assert work["calibration_b_forward_or_tokenization_count"] == 0


def test_exact_scoring_executes_compiled_runtime_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper_artifact = _sha("lineage-wrapper")
    runtime_artifact = _sha("compiled-runtime")

    class Wrapper:
        def __init__(self) -> None:
            self.artifact_sha256 = wrapper_artifact
            self.runtime_provider = SimpleNamespace(artifact_sha256=runtime_artifact)

    monkeypatch.setattr(
        runner,
        "AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider",
        Wrapper,
    )
    provider = Wrapper()
    record = SimpleNamespace(
        sequence=SimpleNamespace(example_id="example-0", family_id="family-0")
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
    monkeypatch.setattr(runner._v20b, "_ordered_records", lambda records: tuple(records))
    monkeypatch.setattr(
        runner._v20a,
        "_verified_model_inputs",
        lambda context, record: ("inputs", (0,), None),
    )

    def score(**kwargs):
        observed["scored_provider_artifact"] = kwargs["provider_artifact_sha256"]
        return 0.25, _sha("h4"), _sha("logits")

    monkeypatch.setattr(runner._v20a, "_execution_hashes_and_score", score)
    evidence = _sha("evidence")
    domain = b"v20o-runtime-test\0"
    objectives, _h4, _logits, executions = runner._score_exact_provider(
        context,
        (record,),
        Capability(),
        provider=provider,
        phase="runtime_provider_replay",
        outer_family_id="outer-family",
        inner_family_id="family-0",
        role="signed_continuum_reflected",
        evidence_sha256=evidence,
        domain=domain,
    )

    assert observed["executed_provider"] is provider.runtime_provider
    assert observed["scored_provider_artifact"] == runtime_artifact
    assert objectives == {"example-0": 0.25}
    assert executions["example-0"] == runner._execution_sha256(
        phase="runtime_provider_replay",
        outer_family_id="outer-family",
        inner_family_id="family-0",
        role="signed_continuum_reflected",
        provider_artifact_sha256=runtime_artifact,
        example_id="example-0",
        family_id="family-0",
        objective=0.25,
        h4_sha256=_sha("h4"),
        logits_sha256=_sha("logits"),
        evidence_sha256=evidence,
        domain=domain,
    )


def test_signed_continuum_receipt_replays_runtime_artifact_and_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _sha("bridge")
    base = _sha("base")
    proposal = _sha("proposal")
    provider = _sha("provider")
    runtime = _sha("runtime")
    transfer = _sha("transfer")
    response = (0.25, 0.25, -0.125)
    signed_scalar = -0.25
    direction = (1.0, 0.0, 0.0, 0.0)
    compiled_direction = (-1.0, -0.0, -0.0, -0.0)
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
                    "prepared_float_scalar_count": 100,
                    "logical_macs_per_token_upper_bound": 200,
                }
            }
        }
    }
    metadata = {
        "artifact_sha256": provider,
        "bridge_binding_sha256": bridge,
        "rank": 320,
        "conditional_rank": 16,
        "prepared_float_scalar_count": 105,
        "logical_macs_per_token_upper_bound": 201,
    }
    source_direction_sha = (
        runner.fisher_soft_polarity_signed_continuum_direction_sha256(
            runner._v20g._eta_tensor(direction)
        )
    )
    compiled_direction_sha = (
        runner.fisher_soft_polarity_signed_continuum_direction_sha256(
            runner._v20g._eta_tensor(compiled_direction)
        )
    )
    validated_payload = {
        "protocol_sha256": runner.FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
        "compiled_runtime_provider_artifact_sha256": runtime,
        "bridge_binding_sha256": bridge,
        "base_provider_artifact_sha256": base,
        "proposal_provider_artifact_sha256": proposal,
        "transfer_protocol_sha256": runner._TRANSFER_PROTOCOL_SHA256,
        "transfer_evidence_sha256": transfer,
        "source_direction_sha256": source_direction_sha,
        "compiled_direction_sha256": compiled_direction_sha,
        "compiled_direction_sign": -1,
        "source_radius": response[0],
        "source_shrink_mass": response[1],
        "source_polarity_bias": response[2],
        "signed_scalar": signed_scalar,
        "compiled_mix": 0.25,
    }
    monkeypatch.setattr(
        runner,
        "validate_fisher_soft_polarity_signed_continuum_provider_evidence",
        lambda payload, observed_metadata: SimpleNamespace(
            artifact_sha256=provider,
            payload=validated_payload,
            metadata={"box_certificate": {}},
        ),
    )
    receipt = runner._hashed(
        {
            "role": "inner_signed_continuum_vertex",
            "source_response": response,
            "source_response_key": runner._parameters_key(response),
            "signed_scalar": signed_scalar,
            "signed_scalar_hex": signed_scalar.hex(),
            "compiled_direction_sign": -1,
            "compiled_mix": 0.25,
            "compiled_mix_hex": (0.25).hex(),
            "source_direction": direction,
            "compiled_direction": compiled_direction,
            "source_direction_box_corner_scores": runner._box_corner_scores(direction),
            "compiled_direction_box_corner_scores": runner._box_corner_scores(
                compiled_direction
            ),
            "source_direction_sha256": source_direction_sha,
            "compiled_direction_sha256": compiled_direction_sha,
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
            "prepared_float_scalar_count": 105,
            "logical_macs_per_token_upper_bound": 201,
            "analysis_only": True,
            "raw_provider_tensors_serialized": False,
        },
        domain=runner._PROVIDER_DOMAIN,
    )
    assert set(receipt) == runner._SIGNED_CONTINUUM_PROVIDER_RECEIPT_KEYS
    kwargs = {
        "expected_role": "inner_signed_continuum_vertex",
        "expected_provider_artifact_sha256": provider,
        "expected_endpoint_receipt": endpoint,
        "expected_bridge_binding_sha256": bridge,
        "authenticated_v20i_fold": v20i,
        "expected_response": response,
        "expected_signed_scalar": signed_scalar,
        "expected_direction": direction,
        "expected_transfer_evidence_sha256": transfer,
    }
    runner._validate_signed_continuum_provider_receipt_evidence(receipt, **kwargs)

    forged = copy.deepcopy(receipt)
    forged["runtime_provider_artifact_sha256"] = _sha("forged-runtime")
    with pytest.raises(ValueError, match="runtime provider artifact"):
        runner._validate_signed_continuum_provider_receipt_evidence(
            forged, **kwargs
        )

    missing = copy.deepcopy(receipt)
    missing.pop("compiled_direction_sha256")
    with pytest.raises(ValueError, match="key set"):
        runner._validate_signed_continuum_provider_receipt_evidence(
            missing, **kwargs
        )


@pytest.mark.parametrize(
    ("signed_scalar", "anchor_arm", "anchor_id"),
    (
        (-1.0, "simplex_response_reflected_exact_mirror", "signed_minus_one"),
        (0.0, "fixed_plus", "signed_zero"),
        (1.0, "matched_v20m_simplex_reflected", "signed_plus_one"),
    ),
)
def test_outer_selected_endpoints_require_exact_anchor_outputs(
    monkeypatch: pytest.MonkeyPatch,
    signed_scalar: float,
    anchor_arm: str,
    anchor_id: str,
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

    def bundle(arm: str, value: float):
        objectives = {
            record.sequence.example_id: value + index * 0.01
            for index, record in enumerate(records)
        }
        return (
            objectives,
            {example: _sha(f"h4:{arm}:{example}") for example in objectives},
            {example: _sha(f"logits:{arm}:{example}") for example in objectives},
            {example: _sha(f"execution:{arm}:{example}") for example in objectives},
        )

    results = {
        arm: bundle(arm, 1.0 + index * 0.1)
        for index, arm in enumerate(runner._ARMS)
    }
    endpoint_bundle = results[anchor_arm]
    results[runner._PRIMARY_ARM] = tuple(
        copy.deepcopy(value) for value in endpoint_bundle
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
                arm: inherited(arm) for arm in ("base", "fixed_plus", "fixed_minus")
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
    v20m = {
        "held_evidence": {
            "arm_evidence": {
                "simplex_response_reflected": inherited(
                    "matched_v20m_simplex_reflected"
                ),
                "simplex_response_reflected_exact_mirror": inherited(
                    "simplex_response_reflected_exact_mirror"
                ),
            }
        }
    }
    reflection = {
        "selected_variant_id": "reflect_coordinate_1",
        "selected_variant_artifact_sha256": _sha("variant"),
    }
    _manifest, held, fold = runner._score_outer_arms(
        object(),
        object(),
        records,
        Vault(),
        {},
        reflection,
        selected_response=(0.125, 0.125, -0.0625),
        selected_signed_scalar=signed_scalar,
        outer_family_id=outer,
        authenticated_v20g_fold=v20g,
        authenticated_v20l_fold=v20l,
        authenticated_v20m_fold=v20m,
    )
    assert held["selected_endpoint_exact_anchor_applicable"] is True
    assert held["selected_endpoint_exact_anchor_id"] == anchor_id
    assert held["selected_endpoint_exact_anchor_passed"] is True
    assert fold["selected_endpoint_exact_anchor_passed"] is True

    forged = list(results[runner._PRIMARY_ARM])
    forged_logits = copy.deepcopy(forged[2])
    forged_logits[records[0].sequence.example_id] = _sha("forged-candidate-logits")
    forged[2] = forged_logits
    results[runner._PRIMARY_ARM] = tuple(forged)
    with pytest.raises(RuntimeError, match="selected signed endpoint"):
        runner._score_outer_arms(
            object(),
            object(),
            records,
            Vault(),
            {},
            reflection,
            selected_response=(0.125, 0.125, -0.0625),
            selected_signed_scalar=signed_scalar,
            outer_family_id=outer,
            authenticated_v20g_fold=v20g,
            authenticated_v20l_fold=v20l,
            authenticated_v20m_fold=v20m,
        )


def test_all_fourteen_missing_then_all_seven_vertex_providers_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = FAMILIES[-1]
    inner_families = FAMILIES[:-1]
    response = (0.125, 0.125, -0.0625)
    response_key = runner._response_key(response)
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
        examples = tuple(
            record.sequence.example_id
            for record in records
            if record.sequence.family_id == family
        )
        inner_evidence[family] = {
            "reflection_fit_receipt": {
                "artifact_sha256": _sha(f"fit:{family}"),
                "selected_variant_artifact_sha256": _sha(f"variant:{family}"),
                "selected_variant_available": True,
                "selected_normalized_direction": (1.0, 0.0, 0.0, 0.0),
            },
            "response_evidence": {
                response_key: {
                    "objective": 1.0,
                    "objectives_by_example": {example: 1.0 for example in examples},
                    "post_cast_h4_sha256s": {
                        example: _sha(f"plus-h4:{example}") for example in examples
                    },
                    "supervised_full_vocab_logits_sha256s": {
                        example: _sha(f"plus-logits:{example}") for example in examples
                    },
                }
            },
        }
    response_selection = {
        "artifact_sha256": _sha("response-selection"),
        "objectives_by_inner_family_and_response": {},
        "family_equal_objective_by_response": {},
        "simplex_response_selection_receipt": {},
        "selected_response": response,
    }
    v20m_inner = {"inner_evidence_by_family": inner_evidence}
    authenticated_v20m = {
        "response_selection_receipt": copy.deepcopy(response_selection),
        "inner_receipt": {"inner_evidence_by_family": copy.deepcopy(inner_evidence)},
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
        return SimpleNamespace(artifact_sha256=_sha(f"{role}:{inner}")), _sha(
            f"seed:{role}:{inner}"
        )

    monkeypatch.setattr(runner, "_materialize_signed_continuum_provider", materialize)
    monkeypatch.setattr(
        runner,
        "_provider_trace",
        lambda provider, records, *, role: {
            "artifact_sha256": _sha(f"trace:{role}:{provider.artifact_sha256}")
        },
    )
    monkeypatch.setattr(
        runner,
        "_signed_continuum_provider_receipt",
        lambda provider, **kwargs: {
            "artifact_sha256": _sha(f"receipt:{provider.artifact_sha256}")
        },
    )
    monkeypatch.setattr(
        runner._signed_continuum_fit,
        "build_soft_polarity_signed_continuum_anchor_receipt",
        lambda **kwargs: {"artifact_sha256": _sha("anchors")},
    )
    monkeypatch.setattr(
        runner._signed_continuum_fit,
        "build_soft_polarity_signed_continuum_quadratic_proposal_receipt",
        lambda **kwargs: {
            "artifact_sha256": _sha("proposal"),
            "proposed_signed_scalar": 0.25,
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
                assert len([event for event in events if event.startswith("build:inner_signed_continuum_")]) == 14
            if self.calls == 8:
                assert len([event for event in events if event.startswith("build:inner_signed_continuum_vertex")]) == 7
                raise FirstVertexCapability
            return Capability()

    def score(context, held, capability, *, role: str, **kwargs):
        events.append(f"score:{role}")
        objectives = {record.sequence.example_id: 1.0 for record in held}
        return (
            objectives,
            {example: _sha(f"h4:{role}:{example}") for example in objectives},
            {example: _sha(f"logits:{role}:{example}") for example in objectives},
            {example: _sha(f"execution:{role}:{example}") for example in objectives},
        )

    monkeypatch.setattr(runner, "_score_exact_provider", score)
    monkeypatch.setattr(
        runner._v20b, "_validate_capability_receipt", lambda *args, **kwargs: None
    )

    with pytest.raises(FirstVertexCapability):
        runner._fit_inner_signed_continuum(
            object(),
            endpoint,
            {},
            Vault(),
            outer_family_id=outer,
            authenticated_v20g_fold={},
            authenticated_v20m_fold=authenticated_v20m,
        )

    assert events.count("score:inner_signed_continuum_mirror_anchor") == 7
    assert events.count("score:inner_signed_continuum_fixed_plus_anchor") == 7
    first_anchor_capability = events.index("capability:1")
    last_anchor_build = max(
        index
        for index, event in enumerate(events)
        if event.startswith("build:inner_signed_continuum_")
        and not event.startswith("build:inner_signed_continuum_vertex")
    )
    assert last_anchor_build < first_anchor_capability
    first_vertex_capability = events.index("capability:8")
    last_vertex_build = max(
        index
        for index, event in enumerate(events)
        if event.startswith("build:inner_signed_continuum_vertex")
    )
    assert last_vertex_build < first_vertex_capability


def test_report_has_v20n_authority_and_no_serving_or_compression_claims(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    report = runner._build_report(
        output=tmp_path / "v20o.json",
        source=_source(),
        v20g_report={"report_sha256": _sha("v20g-report")},
        v20i_report={"report_sha256": _sha("v20i-report")},
        v20m_report={
            "report_sha256": _sha("v20m-report"),
            "classification": "completed-development",
            "passed": True,
            "rollback_to_base": False,
        },
        v20n_report={
            "report_sha256": _sha("v20n-report"),
            "classification": "completed-development",
            "decision": {"integrity_passed": True},
            "passed": True,
            "rollback_to_base": False,
        },
        panel_receipt={"artifact_sha256": _sha("panel")},
        bridge_binding_sha256=_sha("bridge"),
        fold_fragments=_fragments(
            {family: _fold(family) for family in FAMILIES}
        ),
    )

    assert report["passed"] is True
    assert report["v20n_authority"]["integrity_passed"] is True
    assert report["all_eight_final_refit_completed"] is False
    assert report["final_refit"] is None
    assert report["calibration_b_opened"] is False
    assert report["serving_claim_authorized"] is False
    assert report["compression_claim_authorized"] is False
    assert report["speed_claim_authorized"] is False
    assert report["candidate"] is None
    assert report["provider_sidecar"] is None


def _install_run_mocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, list[dict[str, dict[str, object]]], dict[str, object]]:
    destination = tmp_path / "v20o.json"
    prerequisite = {
        "nested_panel_receipt": {
            "artifact_sha256": _sha("panel"),
            "family_prompt_sha256s": {
                family: _sha(f"panel:{family}") for family in FAMILIES
            },
        },
        "authenticated_bridge_binding_sha256": _sha("bridge"),
    }
    maps = [
        {
            family: {"fragment_sha256": _sha(f"authority-{index}:{family}")}
            for family in FAMILIES
        }
        for index in range(6)
    ]
    v20a, v20g, v20i, v20l, v20m, v20n = maps
    v20n_report = {
        "report_sha256": _sha("v20n-report"),
        "decision": {"integrity_passed": True},
    }
    monkeypatch.setattr(runner, "_validate_output", lambda value: Path(value))
    monkeypatch.setattr(
        runner,
        "_load_prerequisites",
        lambda: (
            prerequisite,
            v20a,
            {"report_sha256": _sha("v20g-report")},
            v20g,
            {"report_sha256": _sha("v20i-report")},
            v20i,
            {"report_sha256": _sha("v20l-report")},
            v20l,
            {"report_sha256": _sha("v20m-report")},
            v20m,
            v20n_report,
            v20n,
            {"artifact_sha256": _sha("source")},
        ),
    )
    return destination, maps, v20n_report


def test_completed_fold_fast_path_is_model_free_and_threads_v20n(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination, _maps, v20n_report = _install_run_mocks(
        monkeypatch, tmp_path
    )
    fold_paths = {family: tmp_path / f"{family}.fold.json" for family in FAMILIES}
    for path in fold_paths.values():
        path.touch()
    monkeypatch.setattr(runner, "_fold_path", lambda output, family: fold_paths[family])
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
        observed["build_v20n_report"] = kwargs["v20n_report"]
        observed["build_folds"] = kwargs["fold_fragments"]
        return {"schema": "v20o-completed-fold-report"}

    def load_report(output, **kwargs):
        observed["load_v20n_report"] = kwargs["v20n_report"]
        observed["load_v20n_folds"] = kwargs["authenticated_v20n_folds"]
        return {"loaded": True}

    monkeypatch.setattr(runner, "_build_report", build_report)
    monkeypatch.setattr(runner, "_load_existing_report", load_report)
    monkeypatch.setattr(runner._v20b, "_publish_scalar_fragment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda *args, **kwargs: pytest.fail(
            "completed-fold fast path constructed the model"
        ),
    )

    result = runner.run_gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development(
        output=destination
    )

    assert result == {"loaded": True}
    assert observed["build_v20n_report"] is v20n_report
    assert observed["load_v20n_report"] is v20n_report
    assert observed["build_folds"] == fragments
    assert set(observed["load_v20n_folds"]) == set(FAMILIES)


def test_incomplete_fold_path_constructs_exactly_one_live_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination, _maps, _v20n_report = _install_run_mocks(monkeypatch, tmp_path)
    fold_paths = {family: tmp_path / f"{family}.missing.json" for family in FAMILIES}
    monkeypatch.setattr(runner, "_fold_path", lambda output, family: fold_paths[family])
    calls = 0

    class Context:
        bridge = SimpleNamespace(bridge_binding_sha256=_sha("bridge"))

        def validate_immutable_inputs(self) -> None:
            pass

        def close(self) -> None:
            pass

    def prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Context()

    class StopLive(RuntimeError):
        pass

    monkeypatch.setattr(runner, "prepare_complete_h4_rank320_live_context", prepare)
    monkeypatch.setattr(
        runner._v20b,
        "_collect_live_fit_authority",
        lambda *args, **kwargs: (_ for _ in ()).throw(StopLive()),
    )

    with pytest.raises(StopLive):
        runner.run_gemma3_l3_l4_complete_h4_soft_polarity_signed_continuum_nested_development(
            output=destination
        )
    assert calls == 1
