from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_occupancy_residualized_diagnostic as diagnostic,
)


def _run_kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "corpus_artifact_path": tmp_path / "corpus.json",
        "fit_input_path": tmp_path / "fit.json",
        "materialization_report_path": tmp_path / "materialization.json",
        "expected_materialization_report_sha256": "1" * 64,
        "expected_materialization_report_file_sha256": "2" * 64,
        "factorial_report_path": tmp_path / "factorial.json",
        "expected_factorial_report_sha256": "3" * 64,
        "expected_factorial_report_file_sha256": "4" * 64,
        "prior_iteration_report_path": tmp_path / "prior.json",
        "expected_prior_iteration_report_sha256": "5" * 64,
        "expected_prior_iteration_report_file_sha256": "6" * 64,
        "expected_prior_iteration_collection_sha256": "7" * 64,
        "selection_panel_path": tmp_path / "selection-panel.json",
        "expected_selection_panel_artifact_sha256": "8" * 64,
        "expected_selection_panel_file_sha256": "9" * 64,
        "selection_input_path": tmp_path / "selection-private.json",
        "selection_claim_path": tmp_path / "selection.claim.json",
        "graph_candidate_path": tmp_path / "graph.json",
        "basis_package_path": tmp_path / "basis.json",
        "base_artifact_path": tmp_path / "base.json",
        "refit_artifact_path": tmp_path / "refit.json",
        "output": tmp_path / "residualized-report.json",
        "cache_dir": tmp_path / "cache",
    }


def _required_parser_args() -> list[str]:
    return [
        "--materialization-report-sha256",
        "1" * 64,
        "--materialization-report-file-sha256",
        "2" * 64,
        "--factorial-report-sha256",
        "3" * 64,
        "--factorial-report-file-sha256",
        "4" * 64,
        "--prior-iteration-report-sha256",
        "5" * 64,
        "--prior-iteration-report-file-sha256",
        "6" * 64,
        "--prior-iteration-collection-sha256",
        "7" * 64,
        "--selection-panel-sha256",
        "8" * 64,
        "--selection-panel-file-sha256",
        "9" * 64,
    ]


def test_callback_is_a_mandatory_stop_and_publishes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _run_kwargs(tmp_path)
    destination = Path(kwargs["output"])
    claim = Path(kwargs["selection_claim_path"])
    fit_records = ({"fit": "one"}, {"fit": "two"})
    direct_folds = {
        "centered_cumulative_occupancy_route": ({"fold": "cumulative"},),
        "centered_ew_occupancy_route": ({"fold": "ew"},),
    }
    report = {
        "schema": "test.residualized_occupancy_development",
        "format_version": 1,
        "selection_opened": False,
    }
    observed: dict[str, Any] = {
        "continued_after_callback": False,
        "validations": [],
    }

    def fake_build(
        *,
        fit_records: object,
        direct_fold_receipts_by_arm: object,
    ) -> dict[str, object]:
        observed["fit_records"] = fit_records
        observed["direct_folds"] = direct_fold_receipts_by_arm
        return report

    def fake_validate(value: object) -> None:
        observed["validations"].append(value)

    def fake_selection_diagnostic(**selection_kwargs: object) -> object:
        observed["selection_kwargs"] = selection_kwargs
        callback = selection_kwargs["_build_development_selection"]
        assert callable(callback)
        callback(
            fit_records=fit_records,
            fold_receipts_by_arm=direct_folds,
        )
        observed["continued_after_callback"] = True
        claim.write_text("forbidden", encoding="utf-8")
        return {"forbidden": True}

    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_residualized_occupancy_development_report",
        fake_build,
    )
    monkeypatch.setattr(
        diagnostic,
        "validate_gemma_iterative_residualized_occupancy_development_report",
        fake_validate,
    )
    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_occupancy_selection_diagnostic",
        fake_selection_diagnostic,
    )

    result = (
        diagnostic
        .run_gemma_iterative_residualized_occupancy_development_diagnostic(
            **kwargs
        )
    )

    assert result == report
    assert observed["fit_records"] is fit_records
    assert observed["direct_folds"] is direct_folds
    assert observed["continued_after_callback"] is False
    assert not claim.exists()
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert observed["validations"] == [report, report]
    selection_kwargs = observed["selection_kwargs"]
    assert selection_kwargs["output"] == destination
    assert selection_kwargs["selection_claim_path"] == claim


def test_underlying_driver_return_is_rejected_without_publication_or_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _run_kwargs(tmp_path)
    destination = Path(kwargs["output"])
    claim = Path(kwargs["selection_claim_path"])
    observed = {"continued": False}

    monkeypatch.setattr(
        diagnostic,
        "build_gemma_iterative_residualized_occupancy_development_report",
        lambda **_kwargs: {"report": "captured"},
    )

    def fake_selection_diagnostic(**selection_kwargs: object) -> object:
        callback = selection_kwargs["_build_development_selection"]
        assert callable(callback)
        try:
            callback(
                fit_records=({"fit": "one"},),
                fold_receipts_by_arm={"arm": ({"fold": "one"},)},
            )
        except RuntimeError:
            pass
        observed["continued"] = True
        return {"unexpected": "selection diagnostic returned"}

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_occupancy_selection_diagnostic",
        fake_selection_diagnostic,
    )

    with pytest.raises(
        RuntimeError,
        match="crossed its mandatory stop boundary",
    ):
        (
            diagnostic
            .run_gemma_iterative_residualized_occupancy_development_diagnostic(
                **kwargs
            )
        )

    assert observed["continued"] is True
    assert not destination.exists()
    assert not claim.exists()


@pytest.mark.parametrize(
    ("existing_key", "message"),
    (
        ("output", "refusing to overwrite residualized occupancy report"),
        (
            "selection_claim_path",
            "requires an unclaimed fresh boundary",
        ),
    ),
)
def test_existing_output_or_claim_fails_before_the_base_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_key: str,
    message: str,
) -> None:
    kwargs = _run_kwargs(tmp_path)
    existing = Path(kwargs[existing_key])
    existing.write_text("existing", encoding="utf-8")

    def forbidden_base_driver(**_kwargs: object) -> object:
        raise AssertionError("base diagnostic must not run")

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_occupancy_selection_diagnostic",
        forbidden_base_driver,
    )

    with pytest.raises(FileExistsError, match=message):
        (
            diagnostic
            .run_gemma_iterative_residualized_occupancy_development_diagnostic(
                **kwargs
            )
        )

    assert existing.read_text(encoding="utf-8") == "existing"
    if existing_key == "selection_claim_path":
        assert not Path(kwargs["output"]).exists()


def test_parser_defaults_to_the_development_only_report_path() -> None:
    args = diagnostic.build_parser().parse_args(_required_parser_args())

    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert "stop before any fresh-panel claim" in (
        diagnostic.build_parser().description or ""
    )
