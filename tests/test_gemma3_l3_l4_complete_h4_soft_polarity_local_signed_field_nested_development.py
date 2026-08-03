from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_complete_h4_soft_polarity_local_signed_field_nested_development
    as runner,
)


def _sha(label: str) -> str:
    return runner._v14._sha256({"label": label}, domain=b"v20p-test\0")


def _decision_fragments() -> dict[str, dict[str, object]]:
    fragments: dict[str, dict[str, object]] = {}
    for index in range(8):
        family = f"family-{index}"
        scores = {arm: 1.0 for arm in runner._ARMS}
        scores[runner._PRIMARY_ARM] = 0.5
        scores["fixed_minus"] = 0.25
        fragments[family] = {
            "fold_receipt": {
                "held_objective_by_arm": scores,
                "selected_adaptive": index < 6,
                "candidate_field_nonconstant": index < 6,
                "candidate_field_has_negative": index == 0,
                "candidate_field_has_positive": index == 0,
                "candidate_exact_output_distinct_by_anchor": {
                    "signed_minus_one": index < 6,
                    "signed_zero": index < 6,
                    "signed_plus_one": index < 6,
                },
                "all_inner_endpoint_exact_output_anchors_passed": True,
                "all_inherited_control_exact_output_anchors_passed": True,
                "all_runtime_health_passed": True,
                "selection_frozen_before_outer_score": True,
                "outer_family_used_for_fit_or_selection": False,
                "exact_execution": True,
            }
        }
    return fragments


def test_protocol_library_parser_and_output_are_frozen() -> None:
    assert (
        runner._RUNNER_PROTOCOL_SHA256
        == "0f633c70ab0411146002d799c9ddbd17aa9dfbac56c7ebcfcbfe01d7b0903889"
    )
    assert (
        runner._TRANSFER_PROTOCOL_SHA256
        == "84a6dab103e2a9d1f5b40ed43ae2afc6c5e4739a0115eef350b42a07331468c6"
    )
    assert len(runner._FIELD_LIBRARY) == 27
    assert len(runner._FIELD_CANDIDATE_IDS) == 27
    assert runner._FIELD_LIBRARY[-3:] == (
        ("source_z", -1.0, 0.0),
        ("source_z", 0.0, 0.0),
        ("source_z", 1.0, 0.0),
    )
    assert runner._ARMS == (
        "base",
        "fixed_plus",
        "fixed_minus",
        "matched_linear_reflected",
        "matched_v20l_boundary_reflected",
        "same_simplex_response_unreflected",
        "local_signed_field_reflected",
        "simplex_response_reflected_exact_mirror",
        "matched_v20m_simplex_reflected",
    )
    assert runner._FIXED_PROTOCOL["fresh_validation_claim"] is False
    assert runner._FIXED_PROTOCOL["calibration_b_eligible"] is False
    assert runner._FIXED_PROTOCOL["failure_policy"] == "rollback_to_base_no_claims_no_B"
    args = runner.build_parser().parse_args([])
    assert args.output == runner.DEFAULT_OUTPUT
    assert args.cache_dir is None
    with pytest.raises(ValueError):
        runner._validate_output(Path("outside-v20p.json"))
    with pytest.raises(ValueError):
        runner._validate_output(runner._V20O_OUTPUT)


def test_exact_canonical_work_accounting() -> None:
    work = runner._runner_work_accounting()
    assert work["total_model_forward_count"] == 5440
    assert work["total_teacher_access_count"] == 5408
    assert work["total_suffix_backward_count"] == 128
    assert work["total_local_autograd_contraction_count"] == 112
    assert work["inner_provider_candidate_count"] == 2576
    assert work["inner_providers_and_traces_staged_per_outer_fold"] == 322
    assert work["outer_arm_provider_count"] == 72
    assert (
        work["live_authority_collection_model_forward_count"]
        + work["endpoint_reconstruction_model_forward_count"]
        + work["inner_original_response_model_forward_count"]
        + work["inner_local_signed_field_model_forward_count"]
        + work["outer_held_model_forward_count"]
        == 5440
    )


def test_decision_positive_case_uses_all_predeclared_gates() -> None:
    decision = runner._aggregate_decision(_decision_fragments())
    assert decision["primary_development_gate_passed"] is True
    assert decision["mechanism_gate_passed"] is True
    assert decision["selected_adaptive_count"] == 6
    assert decision["held_field_nonconstant_count"] == 6
    assert decision["held_field_has_negative_count"] == 1
    assert decision["held_field_has_positive_count"] == 1
    assert decision["held_field_crosses_zero_count"] == 1
    assert decision["candidate_exact_output_distinct_count_by_anchor"] == {
        "signed_minus_one": 6,
        "signed_zero": 6,
        "signed_plus_one": 6,
    }
    assert decision["local_signed_field_evidence_gate_passed"] is True
    assert decision["integrity_passed"] is True
    assert decision["development_oof_passed"] is True


def test_decision_accepts_canonical_json_arm_key_order() -> None:
    fragments = json.loads(
        runner._v14._canonical_json_bytes(_decision_fragments()).decode("ascii")
    )
    assert tuple(
        fragments["family-0"]["fold_receipt"]["held_objective_by_arm"]
    ) != runner._ARMS
    decision = runner._aggregate_decision(fragments)
    assert decision["development_oof_passed"] is True


@pytest.mark.parametrize(
    "mutation,gate",
    (
        ("base_wins", "primary_development_gate_passed"),
        ("v20m_wins", "mechanism_gate_passed"),
        ("adaptive", "local_signed_field_evidence_gate_passed"),
        ("nonconstant", "local_signed_field_evidence_gate_passed"),
        ("negative", "local_signed_field_evidence_gate_passed"),
        ("positive", "local_signed_field_evidence_gate_passed"),
        ("distinct", "local_signed_field_evidence_gate_passed"),
        ("endpoint", "integrity_passed"),
    ),
)
def test_decision_fails_closed_at_exact_thresholds(mutation: str, gate: str) -> None:
    fragments = _decision_fragments()
    families = sorted(fragments)
    if mutation == "base_wins":
        for family in families[:3]:
            fragments[family]["fold_receipt"]["held_objective_by_arm"]["base"] = 0.4
    elif mutation == "v20m_wins":
        for family in families[:4]:
            fragments[family]["fold_receipt"]["held_objective_by_arm"][
                "matched_v20m_simplex_reflected"
            ] = 0.4
    elif mutation == "adaptive":
        fragments[families[5]]["fold_receipt"]["selected_adaptive"] = False
    elif mutation == "nonconstant":
        fragments[families[5]]["fold_receipt"]["candidate_field_nonconstant"] = False
    elif mutation == "negative":
        fragments[families[0]]["fold_receipt"]["candidate_field_has_negative"] = False
    elif mutation == "positive":
        fragments[families[0]]["fold_receipt"]["candidate_field_has_positive"] = False
    elif mutation == "distinct":
        fragments[families[5]]["fold_receipt"]["candidate_exact_output_distinct_by_anchor"][
            "signed_zero"
        ] = False
    elif mutation == "endpoint":
        fragments[families[0]]["fold_receipt"][
            "all_inner_endpoint_exact_output_anchors_passed"
        ] = False
    decision = runner._aggregate_decision(fragments)
    assert decision[gate] is False
    assert decision["development_oof_passed"] is False


def _provider_receipt(index: int = 0) -> dict[str, object]:
    feature_id, bias, slope = runner._field_spec(index)
    response = (0.25, 0.125, -0.0625)
    provider_artifact = _sha("provider")
    runtime_artifact = _sha("runtime")
    runtime_payload = {
        "feature_name": feature_id,
        "feature_id": runner._provider_feature_id(feature_id),
        "field_bias": bias,
        "field_bias_hex": bias.hex(),
        "field_slope": slope,
        "field_slope_hex": slope.hex(),
        "radius": response[0],
        "shrink_mass": response[1],
        "polarity_bias": response[2],
    }
    payload = {
        "compiled_runtime_provider_artifact_sha256": runtime_artifact,
        "compiled_runtime_provider_payload": runtime_payload,
    }
    metadata = {
        "rank": 320,
        "conditional_rank": 16,
        "prepared_float_scalar_count": 9,
        "logical_macs_per_token_upper_bound": 100,
    }
    return runner._hashed(
        {
            "role": "inner_local_signed_field_candidate",
            "candidate_id": runner._FIELD_CANDIDATE_IDS[index],
            "candidate_index": index,
            "feature_id": feature_id,
            "field_bias": bias,
            "field_bias_hex": bias.hex(),
            "field_slope": slope,
            "field_slope_hex": slope.hex(),
            "source_response": response,
            "provider_artifact_sha256": provider_artifact,
            "runtime_provider_artifact_sha256": runtime_artifact,
            "provider_payload": payload,
            "provider_metadata": metadata,
            "provider_metadata_sha256": runner._v14._sha256(
                metadata, domain=runner._PROVIDER_DOMAIN
            ),
            "rank": 320,
            "conditional_rank": 16,
            "prepared_float_scalar_count": 9,
            "logical_macs_per_token_upper_bound": 100,
            "lineage_wrapper_not_inference_executor": True,
            "raw_provider_tensors_serialized": False,
            "analysis_only": True,
        },
        domain=runner._PROVIDER_DOMAIN,
    )


def _rehash_provider(receipt: dict[str, object]) -> dict[str, object]:
    value = dict(receipt)
    value.pop("artifact_sha256", None)
    return runner._hashed(value, domain=runner._PROVIDER_DOMAIN)


def test_provider_receipt_rejects_rehashed_semantic_relabeling(monkeypatch) -> None:
    receipt = _provider_receipt()
    monkeypatch.setattr(
        runner,
        "validate_fisher_soft_polarity_local_signed_field_provider_evidence",
        lambda payload, metadata: SimpleNamespace(
            artifact_sha256=receipt["provider_artifact_sha256"]
        ),
    )
    runner._validate_field_provider_receipt(
        receipt, expected_role="inner_local_signed_field_candidate"
    )
    for key, replacement in (
        ("role", "outer_local_signed_field_reflected"),
        ("candidate_id", runner._FIELD_CANDIDATE_IDS[1]),
        ("candidate_index", 1),
        ("feature_id", "source_z"),
        ("field_bias", 0.0),
        ("field_bias_hex", 0.0.hex()),
        ("field_slope", 1.0),
        ("source_response", (0.125, 0.0, 0.0)),
        ("analysis_only", False),
    ):
        tampered = dict(receipt)
        tampered[key] = replacement
        tampered = _rehash_provider(tampered)
        with pytest.raises(ValueError):
            runner._validate_field_provider_receipt(
                tampered, expected_role="inner_local_signed_field_candidate"
            )


def test_provider_receipt_rejects_integer_aliases_for_float_candidate_fields(
    monkeypatch,
) -> None:
    receipt = _provider_receipt(24)
    monkeypatch.setattr(
        runner,
        "validate_fisher_soft_polarity_local_signed_field_provider_evidence",
        lambda payload, metadata: SimpleNamespace(
            artifact_sha256=receipt["provider_artifact_sha256"]
        ),
    )
    runner._validate_field_provider_receipt(
        receipt, expected_role="inner_local_signed_field_candidate"
    )
    for key in ("field_bias", "field_slope"):
        tampered = dict(receipt)
        tampered[key] = int(tampered[key])
        tampered = _rehash_provider(tampered)
        with pytest.raises(ValueError):
            runner._validate_field_provider_receipt(
                tampered, expected_role="inner_local_signed_field_candidate"
            )


def test_field_trace_rejects_rehashed_gate_flag_or_stat_tampering() -> None:
    trace = runner._hashed(
        {
            "local_signed_scalar_sha256s": {"example": _sha("scalar")},
            "local_signed_scalar_min": -0.25,
            "local_signed_scalar_max": 0.5,
            "local_signed_scalar_distinct_count": 4,
            "local_signed_scalar_nonconstant": True,
            "local_signed_scalar_has_negative": True,
            "local_signed_scalar_has_positive": True,
            "local_signed_scalar_inside_closed_unit_interval": True,
            "raw_local_scalar_tensors_serialized": False,
        },
        domain=runner._TRACE_DOMAIN,
    )
    runner._validate_hashed(
        trace, domain=runner._TRACE_DOMAIN, label="test field trace"
    )
    runner._validate_field_trace_semantics(trace)
    for key, replacement in (
        ("local_signed_scalar_min", -1.5),
        ("local_signed_scalar_max", -0.5),
        ("local_signed_scalar_distinct_count", 1),
        ("local_signed_scalar_nonconstant", False),
        ("local_signed_scalar_has_negative", False),
        ("local_signed_scalar_has_positive", False),
        ("local_signed_scalar_inside_closed_unit_interval", False),
    ):
        tampered = dict(trace)
        tampered.pop("artifact_sha256")
        tampered[key] = replacement
        tampered = runner._hashed(tampered, domain=runner._TRACE_DOMAIN)
        runner._validate_hashed(
            tampered, domain=runner._TRACE_DOMAIN, label="rehashed field trace"
        )
        with pytest.raises(ValueError):
            runner._validate_field_trace_semantics(tampered)


def test_field_trace_rejects_integer_aliases_for_float_extrema() -> None:
    trace = runner._hashed(
        {
            "local_signed_scalar_sha256s": {"example": _sha("constant-scalar")},
            "local_signed_scalar_min": 1.0,
            "local_signed_scalar_max": 1.0,
            "local_signed_scalar_distinct_count": 1,
            "local_signed_scalar_nonconstant": False,
            "local_signed_scalar_has_negative": False,
            "local_signed_scalar_has_positive": True,
            "local_signed_scalar_inside_closed_unit_interval": True,
            "raw_local_scalar_tensors_serialized": False,
        },
        domain=runner._TRACE_DOMAIN,
    )
    runner._validate_field_trace_semantics(trace)
    for key in ("local_signed_scalar_min", "local_signed_scalar_max"):
        tampered = dict(trace)
        tampered.pop("artifact_sha256")
        tampered[key] = 1
        tampered = runner._hashed(tampered, domain=runner._TRACE_DOMAIN)
        with pytest.raises(ValueError):
            runner._validate_field_trace_semantics(tampered)


def test_existing_report_fast_path_is_model_free(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "v20p.json"
    destination.write_text("{}", encoding="utf-8")
    authorities = SimpleNamespace(prerequisite={"nested_panel_receipt": {"artifact_sha256": _sha("panel")}, "authenticated_bridge_binding_sha256": _sha("bridge")})
    monkeypatch.setattr(runner, "_validate_output", lambda path: destination)
    monkeypatch.setattr(runner, "_load_prerequisites", lambda: authorities)
    expected = {"classification": "replayed"}
    monkeypatch.setattr(runner, "_load_existing_report", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        runner,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: pytest.fail("complete report replay must not construct Gemma"),
    )
    assert (
        runner.run_gemma3_l3_l4_complete_h4_soft_polarity_local_signed_field_nested_development(
            output=destination
        )
        == expected
    )


def test_scalar_fragment_publication_is_mode_0600_and_atomic(tmp_path: Path) -> None:
    output = tmp_path / "v20p.json"
    family = "family-one"
    monkey_output = output
    payload = {"value": 1}
    original = runner._fold_path
    try:
        runner._fold_path = lambda _output, _family: tmp_path / "fold.json"
        runner._publish_fold_fragment(payload, output=monkey_output, outer_family_id=family)
        path = tmp_path / "fold.json"
        assert path.stat().st_mode & 0o777 == 0o600
        with pytest.raises(FileExistsError):
            runner._publish_fold_fragment(
                payload, output=monkey_output, outer_family_id=family
            )
    finally:
        runner._fold_path = original


def test_outer_teacher_capability_opens_only_after_freeze_without_forbidding_held_rows() -> None:
    source = inspect.getsource(runner._score_outer_arms)
    replay_source = inspect.getsource(runner._validate_fold_fragment)
    assert "held_family_id=None" in source
    assert "expected_held_family_id=None" in source
    assert "held_family_id=outer" not in source
    assert "expected_held_family_id=None" in replay_source
    assert "expected_held_family_id=outer_family_id" not in replay_source


def test_fold_replay_uses_the_live_v20o_response_wrapper_and_v20m_contents() -> None:
    validate_source = inspect.getsource(runner._validate_fold_fragment)
    fit_source = inspect.getsource(runner._fit_inner_local_field)
    assert (
        'authenticated_v20o_fold.get("response_selection_receipt")'
        in validate_source
    )
    assert (
        'authenticated_v20m_fold.get("response_selection_receipt")'
        not in validate_source
    )
    for committed_field in (
        "objectives_by_inner_family_and_response",
        "family_equal_objective_by_response",
        "simplex_response_selection_receipt",
        "selected_response",
    ):
        assert f'"{committed_field}"' in fit_source


def test_canonical_json_arm_maps_use_exact_keys_not_deserialized_order() -> None:
    canonical_order = {
        arm: index for index, arm in enumerate(sorted(runner._ARMS))
    }
    assert tuple(canonical_order) != runner._ARMS
    assert runner._has_exact_arm_keys(
        canonical_order,
        dict(canonical_order),
        dict(canonical_order),
    )
    missing = dict(canonical_order)
    missing.pop(next(iter(missing)))
    assert not runner._has_exact_arm_keys(canonical_order, missing)
    canonical_controls = {
        arm: True for arm in sorted(set(runner._ARMS) - {runner._PRIMARY_ARM})
    }
    assert set(canonical_controls) == set(runner._ARMS) - {runner._PRIMARY_ARM}


def test_outer_execution_seeds_are_deterministic_and_arm_specific() -> None:
    manifest = _sha("outer-manifest")
    family = "held-family"
    seeds = {
        arm: runner._outer_execution_seed(
            manifest_sha256=manifest,
            outer_family_id=family,
            arm=arm,
            provider_artifact_sha256=_sha(f"provider-{arm}"),
        )
        for arm in runner._ARMS
    }
    assert len(set(seeds.values())) == len(runner._ARMS)
    for arm in runner._ARMS:
        assert seeds[arm] == runner._outer_execution_seed(
            manifest_sha256=manifest,
            outer_family_id=family,
            arm=arm,
            provider_artifact_sha256=_sha(f"provider-{arm}"),
        )
    with pytest.raises(ValueError):
        runner._outer_execution_seed(
            manifest_sha256=manifest,
            outer_family_id=family,
            arm="not-an-arm",
            provider_artifact_sha256=_sha("provider"),
        )


def test_transfer_and_inner_execution_seeds_are_rederived_during_replay() -> None:
    selection_source = inspect.getsource(runner._validate_field_selection)
    fold_source = inspect.getsource(runner._validate_fold_fragment)
    assert "expected_transfer_seed = _field_transfer_seed(" in selection_source
    assert "expected_execution_seed = _inner_execution_seed(" in selection_source
    assert "candidate_seed != expected_candidate_seed" in fold_source
    for committed_field in (
        "outer_held_family_id",
        "endpoint_receipt_sha256",
        "source_direction_receipt_sha256",
        "outer_reflection_fit_receipt_sha256",
        "matched_v20l_source_fold_sha256",
        "matched_v20m_source_fold_sha256",
        "all_nine_traces_frozen_before_outer_capability",
        "outer_objectives_or_teacher_rows_used_at_freeze",
    ):
        assert f'manifest.get("{committed_field}")' in fold_source

    seed = runner._inner_execution_seed(
        manifest_sha256=_sha("inner-manifest"),
        outer_family_id="outer-family",
        inner_family_id="inner-family",
        candidate_id=runner._FIELD_CANDIDATE_IDS[0],
        provider_artifact_sha256=_sha("inner-provider"),
    )
    assert seed == runner._inner_execution_seed(
        manifest_sha256=_sha("inner-manifest"),
        outer_family_id="outer-family",
        inner_family_id="inner-family",
        candidate_id=runner._FIELD_CANDIDATE_IDS[0],
        provider_artifact_sha256=_sha("inner-provider"),
    )
    assert seed != runner._inner_execution_seed(
        manifest_sha256=_sha("inner-manifest"),
        outer_family_id="outer-family",
        inner_family_id="different-family",
        candidate_id=runner._FIELD_CANDIDATE_IDS[0],
        provider_artifact_sha256=_sha("inner-provider"),
    )
