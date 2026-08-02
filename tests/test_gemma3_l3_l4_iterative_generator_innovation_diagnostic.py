from __future__ import annotations

from pathlib import Path

import pytest

from fisher_graph import (
    gemma3_l3_l4_iterative_generator_innovation_diagnostic as diagnostic,
)


def test_diagnostic_cli_wires_authenticated_inputs_without_model_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"synthetic": True}

    monkeypatch.setattr(
        diagnostic,
        "run_gemma_iterative_generator_innovation_development_diagnostic",
        fake_run,
    )
    plan = tmp_path / "plan.json"
    panel = tmp_path / "panel.json"
    private = tmp_path / "private.json"
    prior = tmp_path / "prior.json"
    output = tmp_path / "report.json"
    repeated_hash = "a" * 64

    assert diagnostic.main(
        [
            "--generator-plan",
            str(plan),
            "--generator-plan-sha256",
            "1" * 64,
            "--generator-plan-file-sha256",
            "2" * 64,
            "--generator-panel-receipt",
            str(panel),
            "--generator-panel-receipt-sha256",
            "3" * 64,
            "--generator-panel-receipt-file-sha256",
            "4" * 64,
            "--generator-private-role-input",
            str(private),
            "--generator-private-role-input-file-sha256",
            "5" * 64,
            "--prior-occupancy-panel",
            str(prior),
            "--materialization-report-sha256",
            repeated_hash,
            "--materialization-report-file-sha256",
            repeated_hash,
            "--factorial-report-sha256",
            repeated_hash,
            "--factorial-report-file-sha256",
            repeated_hash,
            "--output",
            str(output),
        ]
    ) == 0
    assert captured["plan_path"] == plan
    assert captured["expected_plan_sha256"] == "1" * 64
    assert captured["expected_plan_file_sha256"] == "2" * 64
    assert captured["panel_receipt_path"] == panel
    assert captured["expected_panel_receipt_sha256"] == "3" * 64
    assert captured["expected_panel_receipt_file_sha256"] == "4" * 64
    assert captured["private_role_input_path"] == private
    assert captured["expected_private_role_input_file_sha256"] == "5" * 64
    assert captured["prior_occupancy_panel_path"] == prior
    assert captured["output"] == output
    assert '"synthetic": true' in capsys.readouterr().out


def test_diagnostic_parser_requires_all_live_identity_receipts() -> None:
    parser = diagnostic.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--materialization-report-sha256",
                "a" * 64,
                "--materialization-report-file-sha256",
                "b" * 64,
                "--factorial-report-sha256",
                "c" * 64,
                "--factorial-report-file-sha256",
                "d" * 64,
            ]
        )
