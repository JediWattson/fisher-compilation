from __future__ import annotations

import json
from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_conformal_route_diagnostic as diagnostic,
)
from fisher_graph.gemma3_l3_l4_iterative_conformal_route import (
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE,
)


def _live_parent_lineage() -> dict[str, str]:
    return {
        key: character * 64
        for key, character in zip(
            sorted(diagnostic._LIVE_PARENT_LINEAGE_KEYS),
            "123456789",
            strict=True,
        )
    }


def _prior_payload() -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph."
            "gemma3_l3_l4_iterative_state_experts_analysis.sign_v1"
        ),
        "semantics": {
            "iteration": 3,
        },
        "report_sha256": "a" * 64,
        "collection_sha256": "b" * 64,
        "lineage": {
            **_live_parent_lineage(),
            "prior_iteration_report_sha256": "c" * 64,
            "prior_iteration_report_file_sha256": "d" * 64,
            "prior_iteration_collection_sha256": "e" * 64,
        },
        "decision": {
            "retained": False,
            "ready_for_new_selection": False,
            "deployment_authorized": False,
        },
        "retained_full_fit": None,
    }


def _required_hashes() -> dict[str, str]:
    return {
        "expected_materialization_report_sha256": "1" * 64,
        "expected_materialization_report_file_sha256": "2" * 64,
        "expected_factorial_report_sha256": "3" * 64,
        "expected_factorial_report_file_sha256": "4" * 64,
        "expected_prior_iteration_report_sha256": "a" * 64,
        "expected_prior_iteration_report_file_sha256": "5" * 64,
        "expected_prior_iteration_collection_sha256": "b" * 64,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("retained", True),
        ("ready_for_new_selection", True),
        ("deployment_authorized", True),
        ("retained_full_fit", {"provider": "forbidden"}),
    ),
)
def test_load_prior_requires_rejection_without_selection_or_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    payload = _prior_payload()
    if field == "retained_full_fit":
        payload[field] = value
    else:
        decision = payload["decision"]
        assert isinstance(decision, dict)
        decision[field] = value
    source = tmp_path / "experts.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match="rejected state-experts"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_load_prior_requires_explicit_null_retained_full_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _prior_payload()
    del payload["retained_full_fit"]
    source = tmp_path / "experts.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match="rejected state-experts"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_load_prior_requires_all_three_exact_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "experts.json"
    source.write_text(
        json.dumps(_prior_payload()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        lambda _report: None,
    )

    loaded = diagnostic._load_rejected_prior_iteration(
        source,
        expected_report_sha256="a" * 64,
        expected_report_file_sha256="5" * 64,
        expected_collection_sha256="b" * 64,
    )
    assert loaded["retained_full_fit"] is None

    with pytest.raises(ValueError, match="file hash mismatch"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="6" * 64,
            expected_collection_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="logical hash mismatch"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="f" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="collection hash mismatch"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="f" * 64,
        )


def test_load_prior_rejects_byte_tamper_before_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "experts.json"
    payload = _prior_payload()
    source.write_text(json.dumps(payload), encoding="utf-8")
    expected_file_sha256 = diagnostic._file_sha256(source)
    source.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    validated = False

    def fake_validate(_report: object) -> None:
        nonlocal validated
        validated = True

    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        fake_validate,
    )

    with pytest.raises(ValueError, match="file hash mismatch"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256=expected_file_sha256,
            expected_collection_sha256="b" * 64,
        )

    assert validated is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "fisher_graph.some_other_report"),
        ("iteration", 2),
    ),
)
def test_load_prior_requires_iteration_three_state_experts_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    payload = _prior_payload()
    if field == "iteration":
        semantics = payload["semantics"]
        assert isinstance(semantics, dict)
        semantics["iteration"] = value
    else:
        payload[field] = value
    source = tmp_path / "experts.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match="iteration-three state experts"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_load_prior_requires_exact_state_experts_lineage_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _prior_payload()
    lineage = payload["lineage"]
    assert isinstance(lineage, dict)
    del lineage["prior_iteration_collection_sha256"]
    source = tmp_path / "experts.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match="lineage fields differ"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_wrapper_binds_frozen_parent_immediate_prior_and_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "experts.json"
    output = tmp_path / "conformal.json"
    monkeypatch.setattr(
        diagnostic,
        "_load_rejected_prior_iteration",
        lambda *_args, **_kwargs: _prior_payload(),
    )
    validated: list[dict[str, object]] = []
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_conformal_route_report",
        lambda report: validated.append(dict(report)),
    )
    captured: dict[str, object] = {}
    report = {
        "execution": {
            "total_model_forward_count": 64,
            "fit_example_count": 16,
            "family_count": 8,
            "examples_per_family": 2,
        },
        "raw_prompts_retained": False,
        "raw_tensors_retained": False,
    }

    def fake_base(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        Path(kwargs["output"]).write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_residual_diagnostic",
        fake_base,
    )

    result = diagnostic.run_gemma_iterative_conformal_route_diagnostic(
        prior_iteration_report_path=prior_path,
        output=output,
        **_required_hashes(),
    )

    assert result["execution"]["total_model_forward_count"] == 64
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert validated == [report, report]
    recipe = captured["_diagnostic_recipe"]
    assert recipe.campaign_recipe is (
        GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE
    )
    assert recipe.campaign_recipe.recipe_id == (
        "causal_top2_lag_b_modal_balance_affine_conformal_route"
    )
    assert recipe.campaign_recipe.fit_record_jacobian_field == (
        "jacobian_by_conformal_coefficient"
    )
    assert recipe.campaign_recipe.fold_coefficient_field == (
        "coefficients_by_conformal_coefficient"
    )
    assert recipe.campaign_recipe.coefficient_count == 4
    assert recipe.campaign_recipe.expected_learned_parameter_count == 4
    assert (
        recipe.campaign_recipe.expected_logical_macs_per_token_upper_bound
        == 8
    )
    recipe_audit = dict(recipe.campaign_recipe.audit_recipe_fields)
    assert recipe_audit["execution_mode"] == (
        "fit_only_two_phase_family_blocked_iterative_conformal_route"
    )
    assert recipe_audit["conformal_matrix_shape"] == (2, 2)
    assert recipe_audit["conformal_coefficient_order"] == (
        "shared_real",
        "shared_imag",
        "contrast_real",
        "contrast_imag",
    )
    assert recipe_audit["endpoint_operator_norm_bound"] == 0.25
    assert recipe.make_fit_record is diagnostic._conformal_make_fit_record
    assert recipe.fit_fold is diagnostic._conformal_fit_fold
    assert recipe.fit_full is diagnostic._conformal_fit_full
    assert recipe.build_report is (
        diagnostic.build_gemma_iterative_conformal_route_report
    )
    assert recipe.validate_report is (
        diagnostic.validate_gemma_iterative_conformal_route_report
    )
    assert recipe.expected_parent_lineage == _live_parent_lineage()
    assert recipe.extra_lineage == {
        "prior_iteration_report_sha256": "a" * 64,
        "prior_iteration_report_file_sha256": "5" * 64,
        "prior_iteration_collection_sha256": "b" * 64,
    }
    assert recipe.extra_immutable_inputs == (
        ("prior_iteration_report", prior_path, "5" * 64),
    )
    assert (
        "gemma3_l3_l4_iterative_conformal_route_diagnostic.py"
        in recipe.source_code_files
    )
    assert (
        "gemma3_l3_l4_iterative_conformal_route.py"
        in recipe.source_code_files
    )
    assert (
        "gemma3_l3_l4_iterative_conformal_route_analysis.py"
        in recipe.source_code_files
    )
    assert (
        "gemma3_l3_l4_iterative_state_router_analysis.py"
        in recipe.source_code_files
    )
    assert captured["output"] == output


def test_prerequisite_failure_never_enters_campaign_or_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "experts.json"
    prior_path.write_text(
        json.dumps(_prior_payload()),
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.json"
    entered = False

    def fake_base(**_kwargs: object) -> dict[str, object]:
        nonlocal entered
        entered = True
        return {}

    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_experts_report",
        lambda _report: None,
    )
    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_residual_diagnostic",
        fake_base,
    )

    with pytest.raises(ValueError, match="logical hash mismatch"):
        diagnostic.run_gemma_iterative_conformal_route_diagnostic(
            prior_iteration_report_path=prior_path,
            output=output,
            **{
                **_required_hashes(),
                "expected_prior_iteration_report_sha256": "f" * 64,
            },
        )

    assert entered is False
    assert not output.exists()


def test_conformal_callbacks_bind_parent_and_drop_generic_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_h4 = object()
    lag_b_correction = object()
    captured: list[tuple[str, dict[str, object]]] = []

    def fake_record(**kwargs: object) -> object:
        captured.append(("record", kwargs))
        return "record"

    def fake_fold(**kwargs: object) -> object:
        captured.append(("fold", kwargs))
        return "fold"

    def fake_full(**kwargs: object) -> object:
        captured.append(("full", kwargs))
        return "full"

    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_conformal_route_fit_record",
        fake_record,
    )
    monkeypatch.setattr(
        diagnostic,
        "fit_gemma_iterative_conformal_route_fold_provider",
        fake_fold,
    )
    monkeypatch.setattr(
        diagnostic,
        "fit_gemma_iterative_conformal_route_full_provider",
        fake_full,
    )

    assert (
        diagnostic._conformal_make_fit_record(
            parent_h4=parent_h4,
            lag_b_correction=lag_b_correction,
            example="example",
        )
        == "record"
    )
    assert (
        diagnostic._conformal_fit_fold(
            parent_artifact_sha256="a" * 64,
            records=("record",),
            held_family="family",
        )
        == "fold"
    )
    assert (
        diagnostic._conformal_fit_full(
            parent_artifact_sha256="a" * 64,
            records=("record",),
            parent_h4=parent_h4,
        )
        == "full"
    )

    assert captured == [
        (
            "record",
            {
                "parent_h4": parent_h4,
                "example": "example",
            },
        ),
        (
            "fold",
            {
                "records": ("record",),
                "held_family": "family",
                "parent_artifact_sha256": "a" * 64,
            },
        ),
        (
            "full",
            {
                "records": ("record",),
                "parent_h4": parent_h4,
                "parent_artifact_sha256": "a" * 64,
            },
        ),
    ]


def test_published_report_is_self_validated_and_must_exactly_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "experts.json"
    output = tmp_path / "conformal.json"
    monkeypatch.setattr(
        diagnostic,
        "_load_rejected_prior_iteration",
        lambda *_args, **_kwargs: _prior_payload(),
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_conformal_route_report",
        lambda _report: None,
    )

    def fake_base(**kwargs: object) -> dict[str, object]:
        Path(kwargs["output"]).write_text(
            json.dumps({"report_sha256": "published"}),
            encoding="utf-8",
        )
        return {"report_sha256": "returned"}

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_residual_diagnostic",
        fake_base,
    )

    with pytest.raises(RuntimeError, match="differs from validated"):
        diagnostic.run_gemma_iterative_conformal_route_diagnostic(
            prior_iteration_report_path=prior_path,
            output=output,
            **_required_hashes(),
        )


def test_published_report_exact_replay_uses_json_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "conformal.json"
    output.write_text(
        json.dumps({"coefficient_order": ["a", "b"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_conformal_route_report",
        lambda _report: None,
    )

    diagnostic._validate_published_report(
        output,
        expected_report={"coefficient_order": ("a", "b")},
    )


def test_parser_and_cli_name_are_frozen_to_fit_only() -> None:
    destinations = {
        action.dest
        for action in diagnostic.build_parser()._actions  # noqa: SLF001
    }
    assert destinations == {
        "help",
        "corpus_artifact",
        "fit_input",
        "materialization_report",
        "materialization_report_sha256",
        "materialization_report_file_sha256",
        "factorial_report",
        "factorial_report_sha256",
        "factorial_report_file_sha256",
        "prior_iteration_report",
        "prior_iteration_report_sha256",
        "prior_iteration_report_file_sha256",
        "prior_iteration_collection_sha256",
        "graph_candidate",
        "basis_package",
        "base_artifact",
        "refit_artifact",
        "output",
        "cache_dir",
    }
    assert diagnostic.CLI_NAME == (
        "fisher-graph-gemma-l3-l4-iterative-conformal-route-dev"
    )
    assert diagnostic.DEFAULT_OUTPUT == Path(
        ".local-runs/google--gemma-3-270m/"
        "progressive-a-iterative-conformal-route-v1.report.json"
    )


def test_main_forwards_every_prerequisite_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_conformal_route_diagnostic",
        fake_run,
    )
    output = tmp_path / "conformal.json"
    assert (
        diagnostic.main(
            [
                "--materialization-report-sha256",
                "1" * 64,
                "--materialization-report-file-sha256",
                "2" * 64,
                "--factorial-report-sha256",
                "3" * 64,
                "--factorial-report-file-sha256",
                "4" * 64,
                "--prior-iteration-report-sha256",
                "a" * 64,
                "--prior-iteration-report-file-sha256",
                "5" * 64,
                "--prior-iteration-collection-sha256",
                "b" * 64,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert captured[
        "expected_materialization_report_sha256"
    ] == "1" * 64
    assert captured[
        "expected_materialization_report_file_sha256"
    ] == "2" * 64
    assert captured["expected_factorial_report_sha256"] == "3" * 64
    assert captured[
        "expected_factorial_report_file_sha256"
    ] == "4" * 64
    assert captured[
        "expected_prior_iteration_report_sha256"
    ] == "a" * 64
    assert captured[
        "expected_prior_iteration_report_file_sha256"
    ] == "5" * 64
    assert captured[
        "expected_prior_iteration_collection_sha256"
    ] == "b" * 64
    assert captured["output"] == output
    assert json.loads(capsys.readouterr().out) == {"ok": True}
