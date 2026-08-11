from __future__ import annotations

from dataclasses import replace
import json

import pytest

import fisher_graph.gemma3_parts_first_supermode_experiment as experiment
from fisher_graph.gemma3_parts_first_supermode_experiment import (
    AtomResponseAxisEntry,
    CaptureRole,
    CHART_COUNT_SEARCH_POLICY,
    DEMO_CHART_SURFACE_ID,
    DEMO_EXACT_CONTROL_EVIDENCE_SHA256,
    DEMO_NATIVE_GATE_RECEIPT_SHA256,
    DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256,
    DEMO_SOURCE_MODEL_FINGERPRINT_SHA256,
    ExactLiftCaptureReceipt,
    FiniteInterventionEvidence,
    MODEL_CHART_SURFACE_ID,
    PartsFirstCaptureBundle,
    PartsFirstPilotReport,
    PartsFirstPilotConfig,
    MODEL_SCORE_OBJECTIVE_ID,
    PINNED_MODEL_FAMILY_IDS,
    PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256,
    build_demo_capture_bundle,
    load_parts_first_capture_bundle,
    main,
    preflight_parts_first_capture_bundle,
    run_gemma3_parts_first_supermode_experiment,
    run_parts_first_supermode_from_captures,
    validate_parts_first_supermode_report,
    write_parts_first_capture_bundle,
    write_parts_first_supermode_report,
)
from fisher_graph.gemma3_parts_first_supermode_protocol import (
    SOURCE_MODEL_FINGERPRINT_SHA256,
)


def _demo_evidence(artifact, role, atlas, config):
    chart_ids = tuple(atlas.chart_id(row.chart_coordinates) for row in role.rows)
    baseline = tuple(row.target_score for row in role.rows)
    deltas = tuple(
        (config.gate_alpha - 1.0) * 0.4
        if chart_id == artifact.candidate.chart_id
        else 0.0
        for chart_id in chart_ids
    )
    return FiniteInterventionEvidence(
        candidate_sha256=artifact.candidate_sha256,
        c_row_ids_sha256=role.row_ids_sha256,
        gate_alpha=config.gate_alpha,
        baseline_scores=baseline,
        intervened_scores=tuple(
            value + delta for value, delta in zip(baseline, deltas)
        ),
        first_order_predicted_score_deltas=deltas,
        baseline_chart_ids=chart_ids,
        online_chart_ids_after=chart_ids,
        native_noop_max_abs_error=0.0,
        finite_intervention_candidate_count=1,
        finite_intervention_forward_count=len(role.rows),
        executor_kind="deterministic_demo",
        native_gate_execution_receipt_sha256=DEMO_NATIVE_GATE_RECEIPT_SHA256,
        source_model_fingerprint_before=DEMO_SOURCE_MODEL_FINGERPRINT_SHA256,
        source_model_fingerprint_after=DEMO_SOURCE_MODEL_FINGERPRINT_SHA256,
        source_execution_fingerprint_before=(
            DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
        source_execution_fingerprint_after=(
            DEMO_SOURCE_EXECUTION_FINGERPRINT_SHA256
        ),
    )


def test_demo_runs_A_freezes_B_and_executes_exactly_one_C_gate() -> None:
    result = run_gemma3_parts_first_supermode_experiment(demo=True)
    assert isinstance(result, PartsFirstPilotReport)
    report = result.to_dict()

    assert report["diagnostic_outcome"] == "parts_first_supermode_pilot_completed"
    assert report["capture"]["chart_surface_id"] == DEMO_CHART_SURFACE_ID
    assert report["preflight"]["chart_surface_id"] == DEMO_CHART_SURFACE_ID
    assert report["A_discovery"]["candidate_count"] == 2
    assert report["A_discovery"]["chart_selection"] == {
        "configured_maximum_chart_count": 2,
        "selected_chart_count": 2,
        "minimum_rows_per_chart": 4,
        "minimum_families_per_chart": 3,
        "chart_count_search_policy": CHART_COUNT_SEARCH_POLICY,
        "attempted_chart_counts": [2],
    }
    assert report["A_discovery"]["roles_read"] == ["A_discovery"]
    assert report["A_discovery"]["B_rows_read"] == 0
    assert report["A_discovery"]["C_rows_read"] == 0
    assert report["A_discovery"]["frozen_before_B"] is True
    assert report["B_confirmation"]["fit_or_refit_performed"] is False
    assert report["B_confirmation"]["B_reranking_performed"] is False

    selected = report["B_confirmation"]["selected_candidate_sha256"]
    assert selected is not None
    passing_in_A_order = [
        item["candidate_sha256"]
        for item in report["B_confirmation"]["evaluations"]
        if item["passed"]
    ]
    assert selected == passing_in_A_order[0]
    assert report["C_intervention"]["candidate_sha256"] == selected
    assert report["C_intervention"]["finite_intervention_candidate_count"] == 1
    assert report["C_intervention"]["finite_intervention_forward_count"] == 48
    assert report["C_intervention"]["selected_chart_row_count"] == 24
    assert report["C_intervention"]["passed"] is True
    assert report["execution_firewall"]["C_intervention_callback_count"] == 1
    assert report["execution_firewall"]["C_finite_intervention_candidate_count"] == 1
    assert report["execution_firewall"]["C_finite_intervention_forward_count"] == 48


def test_demo_candidate_is_parts_first_ordered_and_cross_layer() -> None:
    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    candidate = report["A_discovery"]["candidates"][0]
    stages = candidate["stages"]

    assert [item["family_id"] for item in stages] == [
        "tied_lexical_hidden_column",
        "attention_qo_head",
        "mlp_channel",
        "tied_lexical_hidden_column",
    ]
    assert [item["execution_ordinal"] for item in stages] == [0, 1, 4, 38]
    assert all(item["within_boundary_operator"] == "additive_coalition" for item in stages)
    assert candidate["cross_boundary_operator"] == "ordered_conjunction_not_vector_sum"
    assert candidate["core_path_validated"] is True
    assert candidate["group_storage_credit_scalar_count"] == 0
    assert report["A_discovery"]["layers_used_as_generator_ids"] is False
    assert report["A_discovery"]["cross_boundary_vector_sum_used"] is False


def test_demo_is_deterministic_and_makes_no_model_or_compression_claim() -> None:
    first = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    second = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()

    assert first == second
    assert first["scientific_status"]["deterministic_demo_only"] is True
    assert first["scientific_status"]["model_facing_capture_consumed"] is False
    assert (
        first["scientific_status"]["model_facing_finite_intervention_executed"]
        is False
    )
    assert first["scientific_status"]["full_preregistered_protocol_completed"] is False
    assert all(value is False for value in first["claims"].values())
    validate_parts_first_supermode_report(first)


def test_dry_run_validates_but_does_not_discover_or_call_C() -> None:
    calls = 0

    def forbidden(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("dry-run crossed the execution boundary")

    result = run_gemma3_parts_first_supermode_experiment(
        captures=build_demo_capture_bundle(),
        intervention_runner=forbidden,
        dry_run=True,
    )

    assert isinstance(result, dict)
    assert result["passed"] is True
    assert result["demo_only"] is True
    assert result["scientific_evidence_eligible"] is False
    assert calls == 0


def test_selected_path_without_typed_C_evidence_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="no C finite-intervention evidence"):
        run_parts_first_supermode_from_captures(build_demo_capture_bundle())


def test_intervention_callback_is_called_once_and_cannot_return_untyped_data() -> None:
    calls = 0

    def invalid(*_args):
        nonlocal calls
        calls += 1
        return {"not": "typed evidence"}

    with pytest.raises(TypeError, match="invalid receipt"):
        run_parts_first_supermode_from_captures(
            build_demo_capture_bundle(),
            intervention_runner=invalid,
        )
    assert calls == 1


def test_embedded_evidence_roundtrips_and_cli_runs_frozen_capture(
    tmp_path,
) -> None:
    captured: list[FiniteInterventionEvidence] = []

    def record(artifact, role, atlas, config):
        evidence = _demo_evidence(artifact, role, atlas, config)
        captured.append(evidence)
        return evidence

    bundle = build_demo_capture_bundle()
    assert (
        bundle.exact_lift_receipt.exact_control_evidence_sha256
        == DEMO_EXACT_CONTROL_EVIDENCE_SHA256
    )
    run_parts_first_supermode_from_captures(bundle, intervention_runner=record)
    assert len(captured) == 1
    embedded = replace(
        bundle,
        intervention_evidence=captured[0],
        capture_sha256="",
    )
    capture_path = tmp_path / "capture.json"
    report_path = tmp_path / "report.json"
    write_parts_first_capture_bundle(embedded, capture_path)

    assert load_parts_first_capture_bundle(capture_path) == embedded
    assert main(["--captures", str(capture_path), "--output", str(report_path)]) == 0
    report = validate_parts_first_supermode_report(
        json.loads(report_path.read_text(encoding="utf-8"))
    )
    assert report["C_intervention"]["finite_intervention_candidate_count"] == 1
    assert report["C_intervention"]["finite_intervention_forward_count"] == 48
    assert report["execution_firewall"]["C_intervention_callback_count"] == 0


@pytest.mark.parametrize("output_kind", ("capture", "report"))
def test_output_writers_rollback_partial_temp_files(
    monkeypatch,
    tmp_path,
    output_kind: str,
) -> None:
    bundle = build_demo_capture_bundle()
    report = run_parts_first_supermode_from_captures(
        bundle,
        intervention_runner=_demo_evidence,
    )
    target = tmp_path / f"{output_kind}.json"

    def partial_dump(_value, handle, **_kwargs):
        handle.write("{")
        raise OSError("injected serialization failure")

    monkeypatch.setattr(experiment.json, "dump", partial_dump)
    with pytest.raises(OSError, match="injected serialization failure"):
        if output_kind == "capture":
            write_parts_first_capture_bundle(bundle, target)
        else:
            write_parts_first_supermode_report(report, target)

    assert not target.exists()
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize("output_kind", ("capture", "report"))
def test_output_writers_atomically_refuse_to_clobber_existing_files(
    tmp_path,
    output_kind: str,
) -> None:
    bundle = build_demo_capture_bundle()
    report = run_parts_first_supermode_from_captures(
        bundle,
        intervention_runner=_demo_evidence,
    )
    target = tmp_path / f"{output_kind}.json"
    target.write_text("existing sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        if output_kind == "capture":
            write_parts_first_capture_bundle(bundle, target)
        else:
            write_parts_first_supermode_report(report, target)

    assert target.read_text(encoding="utf-8") == "existing sentinel\n"
    assert tuple(tmp_path.iterdir()) == (target,)


def test_demo_exact_control_evidence_uses_an_explicit_sentinel() -> None:
    bundle = build_demo_capture_bundle()
    receipt = replace(
        bundle.exact_lift_receipt,
        exact_control_evidence_sha256="1" * 64,
    )

    with pytest.raises(ValueError, match="exact-control evidence sentinel"):
        replace(
            bundle,
            exact_lift_receipt=receipt,
            capture_sha256="",
        )


def test_demo_chart_surface_uses_an_explicit_sentinel() -> None:
    bundle = build_demo_capture_bundle()

    with pytest.raises(ValueError, match="chart surface sentinel"):
        replace(
            bundle,
            chart_surface_id="another_demo_chart_surface",
            capture_sha256="",
        )


def test_capture_hash_and_role_hash_fail_closed_on_self_inconsistent_edit() -> None:
    state = build_demo_capture_bundle().to_dict()
    state["roles"][0]["rows"][0]["target_score"] += 1.0

    with pytest.raises(ValueError, match="capture role hash"):
        PartsFirstCaptureBundle.from_dict(state)


def test_A_artifact_is_unchanged_when_only_B_is_destroyed() -> None:
    bundle = build_demo_capture_bundle()
    baseline = run_parts_first_supermode_from_captures(
        bundle,
        intervention_runner=_demo_evidence,
    ).to_dict()
    roles = list(bundle.roles)
    original_B = roles[1]
    zero_rows = tuple(
        replace(row, atom_responses=(0.0,) * len(row.atom_responses))
        for row in original_B.rows
    )
    roles[1] = CaptureRole(
        role="B_confirmation",
        rows=zero_rows,
        exact_observer_passed=False,
    )
    changed = replace(bundle, roles=tuple(roles), capture_sha256="")
    rejected = run_parts_first_supermode_from_captures(changed).to_dict()

    assert (
        rejected["A_discovery"]["candidate_catalog_sha256"]
        == baseline["A_discovery"]["candidate_catalog_sha256"]
    )
    assert rejected["B_confirmation"]["selected_candidate_sha256"] is None
    assert rejected["C_intervention"] is None
    assert rejected["execution_firewall"]["C_finite_intervention_forward_count"] == 0


def test_capture_roles_must_be_disjoint() -> None:
    bundle = build_demo_capture_bundle()
    roles = list(bundle.roles)
    first_B = replace(roles[1].rows[0], row_id=roles[0].rows[0].row_id)
    roles[1] = CaptureRole(
        role="B_confirmation",
        rows=(first_B, *roles[1].rows[1:]),
        exact_observer_passed=False,
    )
    with pytest.raises(ValueError, match="row identities must be disjoint"):
        replace(bundle, roles=tuple(roles), capture_sha256="")


def test_axis_rejects_atom_at_wrong_boundary_and_shared_terminal() -> None:
    with pytest.raises(ValueError, match="no unique occurrence"):
        AtomResponseAxisEntry(640, 4)
    with pytest.raises(ValueError, match="shared substrate"):
        AtomResponseAxisEntry(0, 37)


def test_capture_source_selection_and_loader_fail_closed() -> None:
    bundle = build_demo_capture_bundle()
    with pytest.raises(ValueError, match="exactly one"):
        run_gemma3_parts_first_supermode_experiment()
    with pytest.raises(ValueError, match="exactly one"):
        run_gemma3_parts_first_supermode_experiment(captures=bundle, demo=True)
    with pytest.raises(ValueError, match="non-model bundle"):
        run_gemma3_parts_first_supermode_experiment(
            model_capture_loader=lambda: bundle,
            dry_run=True,
        )


def test_model_capture_cannot_impersonate_exact_lift_with_demo_receipt() -> None:
    bundle = build_demo_capture_bundle()
    with pytest.raises(ValueError, match="exact lift gate"):
        replace(bundle, source_kind="model_capture", capture_sha256="")


def test_C_evidence_must_bind_selected_candidate_and_frozen_alpha() -> None:
    bundle = build_demo_capture_bundle()

    def wrong_candidate(artifact, role, atlas, config):
        evidence = _demo_evidence(artifact, role, atlas, config)
        return replace(evidence, candidate_sha256="0" * 64, evidence_sha256="")

    with pytest.raises(ValueError, match="B-selected candidate"):
        run_parts_first_supermode_from_captures(
            bundle,
            intervention_runner=wrong_candidate,
        )

    def wrong_alpha(artifact, role, atlas, config):
        evidence = _demo_evidence(artifact, role, atlas, config)
        return replace(evidence, gate_alpha=0.25, evidence_sha256="")

    with pytest.raises(ValueError, match="gate alpha"):
        run_parts_first_supermode_from_captures(
            bundle,
            intervention_runner=wrong_alpha,
        )


def test_finite_evidence_requires_one_forward_per_fresh_C_row() -> None:
    bundle = build_demo_capture_bundle()

    def duplicate_forward(artifact, role, atlas, config):
        evidence = _demo_evidence(artifact, role, atlas, config)
        return replace(
            evidence,
            finite_intervention_forward_count=2,
            evidence_sha256="",
        )

    with pytest.raises(ValueError, match="row count"):
        run_parts_first_supermode_from_captures(
            bundle,
            intervention_runner=duplicate_forward,
        )

    def duplicate_candidate(artifact, role, atlas, config):
        evidence = _demo_evidence(artifact, role, atlas, config)
        return replace(
            evidence,
            finite_intervention_candidate_count=2,
            evidence_sha256="",
        )

    with pytest.raises(ValueError, match="exactly one finite intervention candidate"):
        run_parts_first_supermode_from_captures(
            bundle,
            intervention_runner=duplicate_candidate,
        )


def test_C_router_stability_excludes_ungated_rows() -> None:
    bundle = build_demo_capture_bundle()
    config = PartsFirstPilotConfig()
    discovery = experiment._discover_on_A(
        role=bundle.role_by_id["A_discovery"],
        axis=bundle.response_axis,
        catalog=experiment._canonical_catalog(),
        config=config,
    )
    artifact = discovery.candidates[0]
    atlas = discovery.chart_atlas
    candidate_chart = artifact.candidate.chart_id
    C_rows = bundle.role_by_id["C_intervention"].rows
    selected_rows = tuple(
        row
        for row in C_rows
        if atlas.chart_id(row.chart_coordinates) == candidate_chart
    )[:3]
    unselected_rows = tuple(
        row
        for row in C_rows
        if atlas.chart_id(row.chart_coordinates) != candidate_chart
    )[:9]
    role = CaptureRole(
        role="C_intervention",
        rows=selected_rows + unselected_rows,
        exact_observer_passed=False,
    )
    evidence = _demo_evidence(artifact, role, atlas, config)
    other_chart = next(
        chart_id
        for chart_id in ("chart.0", "chart.1")
        if chart_id != candidate_chart
    )
    switched = tuple(
        other_chart if chart_id == candidate_chart else chart_id
        for chart_id in evidence.baseline_chart_ids
    )
    evidence = replace(
        evidence,
        online_chart_ids_after=switched,
        evidence_sha256="",
    )

    receipt = experiment._evaluate_C_intervention(
        role=role,
        artifact=artifact,
        atlas=atlas,
        evidence=evidence,
        config=config,
        expected_source_model_fingerprint=(
            bundle.source_model_fingerprint_sha256
        ),
        expected_source_execution_fingerprint=(
            bundle.source_execution_fingerprint_sha256
        ),
    )

    assert len(selected_rows) == 3
    assert len(unselected_rows) == 9
    assert receipt.router_stability_fraction == 0.0
    assert receipt.passed is False
    assert "online_router_stability_failed" in receipt.failure_reasons


def test_C_requires_more_than_one_selected_chart_row() -> None:
    bundle = build_demo_capture_bundle()
    config = PartsFirstPilotConfig()
    discovery = experiment._discover_on_A(
        role=bundle.role_by_id["A_discovery"],
        axis=bundle.response_axis,
        catalog=experiment._canonical_catalog(),
        config=config,
    )
    artifact = discovery.candidates[0]
    atlas = discovery.chart_atlas
    candidate_chart = artifact.candidate.chart_id
    C_rows = bundle.role_by_id["C_intervention"].rows
    selected_rows = tuple(
        row
        for row in C_rows
        if atlas.chart_id(row.chart_coordinates) == candidate_chart
    )[:1]
    unselected_rows = tuple(
        row
        for row in C_rows
        if atlas.chart_id(row.chart_coordinates) != candidate_chart
    )[:11]
    role = CaptureRole(
        role="C_intervention",
        rows=selected_rows + unselected_rows,
        exact_observer_passed=False,
    )
    evidence = _demo_evidence(artifact, role, atlas, config)

    receipt = experiment._evaluate_C_intervention(
        role=role,
        artifact=artifact,
        atlas=atlas,
        evidence=evidence,
        config=config,
        expected_source_model_fingerprint=(
            bundle.source_model_fingerprint_sha256
        ),
        expected_source_execution_fingerprint=(
            bundle.source_execution_fingerprint_sha256
        ),
    )

    assert receipt.selected_chart_row_count == 1
    assert receipt.passed is False
    assert receipt.failure_reasons == (
        "insufficient_selected_chart_support",
    )


def test_report_hash_and_claim_boundary_are_authenticated() -> None:
    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    report["claims"]["compression_success"] = True
    with pytest.raises(ValueError, match="hash"):
        validate_parts_first_supermode_report(report)

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    report["claims"]["compression_success"] = True
    payload = dict(report)
    payload.pop("report_sha256")
    # A coordinated rehash is still rejected by semantic validation.
    from fisher_graph import gemma3_parts_first_supermode_experiment as module

    report["report_sha256"] = module._json_sha256(
        payload,
        domain=module._REPORT_HASH_DOMAIN,
    )
    with pytest.raises(ValueError, match="unsupported claim"):
        validate_parts_first_supermode_report(report)

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    report["C_intervention"]["selected_chart_row_count"] = 1
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._REPORT_HASH_DOMAIN,
    )
    with pytest.raises(ValueError, match="selected-chart support"):
        validate_parts_first_supermode_report(report)

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    report["capture"]["chart_surface_id"] = "forged_chart_surface"
    report["preflight"]["chart_surface_id"] = "forged_chart_surface"
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._REPORT_HASH_DOMAIN,
    )
    with pytest.raises(ValueError, match="chart surface binding"):
        validate_parts_first_supermode_report(report)

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    atlas = report["A_discovery"]["chart_atlas"]
    atlas["A_role_sha256"] = "1" * 64
    atlas_payload = dict(atlas)
    atlas_payload.pop("atlas_sha256")
    atlas["atlas_sha256"] = experiment._json_sha256(
        atlas_payload,
        domain=experiment._CHART_HASH_DOMAIN,
    )
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._REPORT_HASH_DOMAIN,
    )
    with pytest.raises(ValueError, match="atlas A-role binding"):
        validate_parts_first_supermode_report(report)

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    report["configuration"]["minimum_rows_per_chart"] = 1
    report["A_discovery"]["chart_selection"][
        "minimum_rows_per_chart"
    ] = 1
    atlas = report["A_discovery"]["chart_atlas"]
    atlas["minimum_rows_per_chart"] = 1
    atlas_payload = dict(atlas)
    atlas_payload.pop("atlas_sha256")
    atlas["atlas_sha256"] = experiment._json_sha256(
        atlas_payload,
        domain=experiment._CHART_HASH_DOMAIN,
    )
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._REPORT_HASH_DOMAIN,
    )
    with pytest.raises(ValueError, match="typed configuration or atlas"):
        validate_parts_first_supermode_report(report)

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    atlas = report["A_discovery"]["chart_atlas"]
    atlas["centers"] = [["bad", None] for _ in atlas["centers"]]
    atlas_payload = dict(atlas)
    atlas_payload.pop("atlas_sha256")
    atlas["atlas_sha256"] = experiment._json_sha256(
        atlas_payload,
        domain=experiment._CHART_HASH_DOMAIN,
    )
    payload = dict(report)
    payload.pop("report_sha256")
    report["report_sha256"] = experiment._json_sha256(
        payload,
        domain=experiment._REPORT_HASH_DOMAIN,
    )
    with pytest.raises(ValueError, match="typed configuration or atlas"):
        validate_parts_first_supermode_report(report)


def test_dry_run_refuses_output_and_cli_requires_explicit_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not write"):
        run_gemma3_parts_first_supermode_experiment(
            demo=True,
            dry_run=True,
            output=tmp_path / "report.json",
        )
    with pytest.raises(SystemExit):
        main([])


def test_preflight_binds_all_three_part_families() -> None:
    bundle = build_demo_capture_bundle()
    preflight = preflight_parts_first_capture_bundle(bundle)
    assert preflight["response_axis_family_counts"] == {
        "tied_lexical_hidden_column": 8,
        "attention_qo_head": 4,
        "mlp_channel": 4,
    }
    truncated = replace(
        bundle,
        response_axis=bundle.response_axis[:8],
        roles=tuple(
            CaptureRole(
                role=role.role,
                rows=tuple(
                    replace(row, atom_responses=row.atom_responses[:8])
                    for row in role.rows
                ),
                exact_observer_passed=False,
            )
            for role in bundle.roles
        ),
        capture_sha256="",
    )
    with pytest.raises(ValueError, match="lexical, attention, and MLP"):
        preflight_parts_first_capture_bundle(truncated)


def test_semantic_family_lineage_is_exact_and_controls_shuffle_within_family() -> None:
    bundle = build_demo_capture_bundle()
    state = bundle.to_dict()
    assert state["roles"][0]["rows"][0]["family_id"] == "family.0"
    assert state["roles"][0]["family_row_counts"] == {
        f"family.{index}": 8 for index in range(6)
    }

    report = run_gemma3_parts_first_supermode_experiment(demo=True).to_dict()
    assert report["A_discovery"]["A_family_row_counts_by_chart"] == {
        "chart.0": {f"family.{index}": 4 for index in range(6)},
        "chart.1": {f"family.{index}": 4 for index in range(6)},
    }
    for evaluation in report["B_confirmation"]["evaluations"]:
        assert evaluation["shuffled_row_control"] == (
            "deterministic_permutation_within_semantic_family"
        )
        assert evaluation["shuffled_row_cross_chart_source_count"] >= 0


def test_A_chart_falls_back_to_largest_K_with_frozen_family_support() -> None:
    bundle = build_demo_capture_bundle()
    A = bundle.role_by_id["A_discovery"]
    rows = tuple(
        replace(
            row,
            chart_coordinates=(
                0.0 if int(row.family_id.rsplit(".", 1)[1]) < 3 else 10.0,
                0.0 if int(row.family_id.rsplit(".", 1)[1]) < 3 else 10.0,
            ),
        )
        for row in A.rows
    )
    role = CaptureRole(
        role="A_discovery",
        rows=rows,
        exact_observer_passed=False,
    )
    config = PartsFirstPilotConfig(minimum_families_per_chart=4)

    atlas = experiment._fit_chart_atlas(role, config)
    repeated = experiment._fit_chart_atlas(role, config)
    discovery = experiment._discover_on_A(
        role=role,
        axis=bundle.response_axis,
        catalog=experiment._canonical_catalog(),
        config=config,
    )

    assert atlas == repeated
    assert atlas.configured_maximum_chart_count == 2
    assert atlas.selected_chart_count == 1
    assert atlas.attempted_chart_counts == (2, 1)
    assert atlas.chart_count_search_policy == CHART_COUNT_SEARCH_POLICY
    assert atlas.minimum_rows_per_chart == 4
    assert atlas.minimum_families_per_chart == 4
    assert len(atlas.centers) == 1
    assert discovery.to_dict()["chart_selection"] == atlas.selection_dict()


def test_A_chart_fit_fails_when_no_K_satisfies_frozen_support() -> None:
    role = build_demo_capture_bundle().role_by_id["A_discovery"]

    with pytest.raises(ValueError, match="no deterministic A chart count"):
        experiment._fit_chart_atlas(
            role,
            PartsFirstPilotConfig(minimum_rows_per_chart=len(role.rows) + 1),
        )


def _model_role_from_demo(role: CaptureRole, per_family: int) -> CaptureRole:
    demo_to_model_family = {
        f"family.{index}": family_id
        for index, family_id in enumerate(PINNED_MODEL_FAMILY_IDS)
    }
    counts = {family_id: 0 for family_id in PINNED_MODEL_FAMILY_IDS}
    selected = []
    for row in role.rows:
        family_id = demo_to_model_family[row.family_id]
        if counts[family_id] < per_family:
            selected.append(replace(row, family_id=family_id))
            counts[family_id] += 1
    assert set(counts.values()) == {per_family}
    return CaptureRole(
        role=role.role,
        rows=tuple(selected),
        exact_observer_passed=True,
    )


def test_model_capture_exact_gate_rejects_partial_occurrence_axis() -> None:
    bundle = build_demo_capture_bundle()
    roles = (
        _model_role_from_demo(bundle.roles[0], 6),
        _model_role_from_demo(bundle.roles[1], 2),
        _model_role_from_demo(bundle.roles[2], 2),
    )
    exact = ExactLiftCaptureReceipt(
        model_facing=True,
        exact_39_event_order_passed=True,
        all_37576_physical_atoms_closed=True,
        all_shared_substrate_slices_closed=True,
        tied_lexical_storage_identity_passed=True,
        source_state_unchanged=True,
        exact_control_evidence_sha256="1" * 64,
    )

    with pytest.raises(ValueError, match="live exact-control evidence"):
        replace(
            bundle,
            source_kind="model_capture",
            source_model_fingerprint_sha256=SOURCE_MODEL_FINGERPRINT_SHA256,
            source_execution_fingerprint_sha256=(
                PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            score_objective_id=MODEL_SCORE_OBJECTIVE_ID,
            chart_surface_id=MODEL_CHART_SURFACE_ID,
            roles=roles,
            exact_lift_receipt=replace(
                exact,
                exact_control_evidence_sha256=(
                    DEMO_EXACT_CONTROL_EVIDENCE_SHA256
                ),
            ),
            capture_sha256="",
        )

    with pytest.raises(ValueError, match="complete 38216-occurrence axis"):
        replace(
            bundle,
            source_kind="model_capture",
            source_model_fingerprint_sha256=SOURCE_MODEL_FINGERPRINT_SHA256,
            source_execution_fingerprint_sha256=(
                PINNED_SOURCE_EXECUTION_FINGERPRINT_SHA256
            ),
            score_objective_id=MODEL_SCORE_OBJECTIVE_ID,
            chart_surface_id=MODEL_CHART_SURFACE_ID,
            roles=roles,
            exact_lift_receipt=exact,
            capture_sha256="",
        )


def test_model_capture_requires_pinned_execution_and_score_objective() -> None:
    bundle = build_demo_capture_bundle()
    roles = (
        _model_role_from_demo(bundle.roles[0], 6),
        _model_role_from_demo(bundle.roles[1], 2),
        _model_role_from_demo(bundle.roles[2], 2),
    )
    exact = ExactLiftCaptureReceipt(
        True,
        True,
        True,
        True,
        True,
        True,
        "1" * 64,
    )
    with pytest.raises(ValueError, match="pinned model"):
        replace(
            bundle,
            source_kind="model_capture",
            source_model_fingerprint_sha256=SOURCE_MODEL_FINGERPRINT_SHA256,
            source_execution_fingerprint_sha256="0" * 64,
            score_objective_id=MODEL_SCORE_OBJECTIVE_ID,
            chart_surface_id=MODEL_CHART_SURFACE_ID,
            roles=roles,
            exact_lift_receipt=exact,
            capture_sha256="",
        )
    with pytest.raises(ValueError, match="score objective sentinel"):
        replace(
            bundle,
            score_objective_id="another_demo_score",
            capture_sha256="",
        )


def test_live_gate_evidence_requires_unchanged_source_fingerprints() -> None:
    bundle = build_demo_capture_bundle()
    captured: list[FiniteInterventionEvidence] = []

    def record(artifact, role, atlas, config):
        evidence = _demo_evidence(artifact, role, atlas, config)
        captured.append(evidence)
        return evidence

    run_parts_first_supermode_from_captures(bundle, intervention_runner=record)
    with pytest.raises(ValueError, match="changed source model or execution"):
        replace(
            captured[0],
            executor_kind="live_model_bridge",
            native_gate_execution_receipt_sha256="1" * 64,
            source_model_fingerprint_after="2" * 64,
            evidence_sha256="",
        )
