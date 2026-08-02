from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_state_experts_diagnostic as diagnostic,
)
from fisher_graph.gemma3_l3_l4_iterative_state_experts import (
    GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE,
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


def _prior_payload(
    *,
    retained: bool = False,
    deployment_authorized: bool = False,
    retained_full_fit: object = None,
) -> dict[str, object]:
    return {
        "schema": (
            "fisher_graph.gemma3_l3_l4_iterative_state_router_analysis"
        ),
        "semantics": {
            "iteration": 2,
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
            "retained": retained,
            "deployment_authorized": deployment_authorized,
        },
        "retained_full_fit": retained_full_fit,
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
    ("changes", "message"),
    (
        ({"retained": True}, "rejected state-router parent"),
        (
            {"deployment_authorized": True},
            "rejected state-router parent",
        ),
        (
            {"retained_full_fit": {"provider": "forbidden"}},
            "rejected state-router parent",
        ),
    ),
)
def test_load_prior_requires_rejection_without_provider_or_deployment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    message: str,
) -> None:
    source = tmp_path / "router.json"
    source.write_text(
        json.dumps(_prior_payload(**changes)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_router_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match=message):
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
    source = tmp_path / "router.json"
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
        "validate_gemma_iterative_state_router_report",
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "fisher_graph.some_other_report"),
        ("iteration", 1),
    ),
)
def test_load_prior_requires_iteration_two_state_router_schema(
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
    source = tmp_path / "router.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_router_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match="iteration-two state router"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_load_prior_requires_exact_router_lineage_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _prior_payload()
    lineage = payload["lineage"]
    assert isinstance(lineage, dict)
    del lineage["prior_iteration_collection_sha256"]
    source = tmp_path / "router.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "5" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_state_router_report",
        lambda _report: None,
    )

    with pytest.raises(ValueError, match="lineage fields differ"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="5" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_wrapper_binds_only_live_parent_and_immediate_prior_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "router.json"
    output = tmp_path / "experts.json"
    monkeypatch.setattr(
        diagnostic,
        "_load_rejected_prior_iteration",
        lambda *_args, **_kwargs: _prior_payload(),
    )
    captured: dict[str, object] = {}
    report = {
        "execution": {
            "total_model_forward_count": 64,
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

    result = diagnostic.run_gemma_iterative_state_experts_diagnostic(
        prior_iteration_report_path=prior_path,
        output=output,
        **_required_hashes(),
    )

    assert result["execution"]["total_model_forward_count"] == 64
    assert json.loads(output.read_text(encoding="utf-8")) == report
    recipe = captured["_diagnostic_recipe"]
    assert recipe.campaign_recipe is (
        GEMMA_ITERATIVE_STATE_EXPERTS_CAMPAIGN_RECIPE
    )
    assert recipe.campaign_recipe.fit_record_jacobian_field == (
        "jacobian_by_expert_route_edge"
    )
    assert recipe.campaign_recipe.fold_coefficient_field == (
        "coefficients_by_expert_route_edge"
    )
    assert recipe.campaign_recipe.coefficient_count == 8
    assert recipe.campaign_recipe.expected_learned_parameter_count == 8
    assert (
        recipe.campaign_recipe.expected_logical_macs_per_token_upper_bound
        == 6
    )
    assert dict(recipe.campaign_recipe.audit_recipe_fields)[
        "execution_mode"
    ] == "fit_only_two_phase_family_blocked_iterative_state_experts"
    assert recipe.make_fit_record is diagnostic._experts_make_fit_record
    assert recipe.fit_fold is diagnostic._experts_fit_fold
    assert recipe.fit_full is diagnostic._experts_fit_full
    assert recipe.build_report is (
        diagnostic.build_gemma_iterative_state_experts_report
    )
    assert recipe.validate_report is (
        diagnostic.validate_gemma_iterative_state_experts_report
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
    assert captured["output"] == output


def test_prerequisite_failure_never_enters_campaign_or_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_path = tmp_path / "router.json"
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
        "validate_gemma_iterative_state_router_report",
        lambda _report: None,
    )
    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_residual_diagnostic",
        fake_base,
    )

    with pytest.raises(ValueError, match="logical hash mismatch"):
        diagnostic.run_gemma_iterative_state_experts_diagnostic(
            prior_iteration_report_path=prior_path,
            output=output,
            **{
                **_required_hashes(),
                "expected_prior_iteration_report_sha256": "f" * 64,
            },
        )

    assert entered is False
    assert not output.exists()


def test_experts_callbacks_bind_parent_and_drop_generic_correction(
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
        "build_gemma_iterative_state_experts_fit_record",
        fake_record,
    )
    monkeypatch.setattr(
        diagnostic,
        "fit_gemma_iterative_state_experts_fold_provider",
        fake_fold,
    )
    monkeypatch.setattr(
        diagnostic,
        "fit_gemma_iterative_state_experts_full_provider",
        fake_full,
    )

    assert (
        diagnostic._experts_make_fit_record(
            parent_h4=parent_h4,
            lag_b_correction=lag_b_correction,
            example="example",
        )
        == "record"
    )
    assert (
        diagnostic._experts_fit_fold(
            parent_artifact_sha256="a" * 64,
            records=("record",),
            held_family="family",
        )
        == "fold"
    )
    assert (
        diagnostic._experts_fit_full(
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


def test_parser_and_cli_entry_are_frozen_to_fit_only() -> None:
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
    assert diagnostic.DEFAULT_OUTPUT == Path(
        ".local-runs/google--gemma-3-270m/"
        "progressive-a-iterative-state-experts-sign-v1.report.json"
    )
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert project["project"]["scripts"][
        "fisher-graph-gemma-l3-l4-iterative-state-experts-dev"
    ] == (
        "fisher_graph."
        "gemma3_l3_l4_iterative_state_experts_diagnostic:main"
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
        "run_gemma_iterative_state_experts_diagnostic",
        fake_run,
    )
    output = tmp_path / "experts.json"
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
