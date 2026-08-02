from __future__ import annotations

import json
from pathlib import Path

import pytest

import fisher_graph.gemma3_l3_l4_iterative_state_router_diagnostic as diagnostic
from fisher_graph.gemma3_l3_l4_iterative_state_router import (
    GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE,
)


def _prior_payload(*, retained: bool = False) -> dict[str, object]:
    return {
        "report_sha256": "a" * 64,
        "collection_sha256": "b" * 64,
        "lineage": {
            "parent_artifact_sha256": "c" * 64,
        },
        "decision": {
            "retained": retained,
            "deployment_authorized": False,
        },
        "retained_full_fit": None,
    }


def test_load_prior_requires_the_rejected_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "prior.json"
    source.write_text(
        json.dumps(_prior_payload()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diagnostic,
        "_file_sha256",
        lambda _path: "d" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_residual_report",
        lambda _report: None,
    )
    loaded = diagnostic._load_rejected_prior_iteration(
        source,
        expected_report_sha256="a" * 64,
        expected_report_file_sha256="d" * 64,
        expected_collection_sha256="b" * 64,
    )
    assert loaded["decision"]["retained"] is False

    source.write_text(
        json.dumps(_prior_payload(retained=True)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rejected position parent"):
        diagnostic._load_rejected_prior_iteration(
            source,
            expected_report_sha256="a" * 64,
            expected_report_file_sha256="d" * 64,
            expected_collection_sha256="b" * 64,
        )


def test_wrapper_binds_prior_lineage_and_router_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "_load_rejected_prior_iteration",
        lambda *_args, **_kwargs: _prior_payload(),
    )
    captured: dict[str, object] = {}

    def fake_base(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_residual_diagnostic",
        fake_base,
    )
    result = diagnostic.run_gemma_iterative_state_router_diagnostic(
        expected_materialization_report_sha256="1" * 64,
        expected_materialization_report_file_sha256="2" * 64,
        expected_factorial_report_sha256="3" * 64,
        expected_factorial_report_file_sha256="4" * 64,
        expected_prior_iteration_report_sha256="a" * 64,
        expected_prior_iteration_report_file_sha256="5" * 64,
        expected_prior_iteration_collection_sha256="b" * 64,
    )
    assert result == {"ok": True}
    recipe = captured["_diagnostic_recipe"]
    assert recipe.campaign_recipe is (
        GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE
    )
    assert recipe.expected_parent_lineage == {
        "parent_artifact_sha256": "c" * 64,
    }
    assert recipe.extra_lineage == {
        "prior_iteration_report_sha256": "a" * 64,
        "prior_iteration_report_file_sha256": "5" * 64,
        "prior_iteration_collection_sha256": "b" * 64,
    }


def test_parser_exposes_prerequisites_but_no_search_knobs() -> None:
    destinations = {
        action.dest
        for action in diagnostic.build_parser()._actions  # noqa: SLF001
    }
    assert {
        "prior_iteration_report",
        "prior_iteration_report_sha256",
        "prior_iteration_report_file_sha256",
        "prior_iteration_collection_sha256",
    } <= destinations
    assert not destinations & {
        "selection",
        "guard",
        "assessment",
        "calibration_b",
        "rank",
        "ridge",
        "operator_norm_bound",
        "route_mode_count",
    }
