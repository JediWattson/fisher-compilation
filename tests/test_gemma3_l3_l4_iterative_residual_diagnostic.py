from __future__ import annotations

import pytest

from fisher_graph.gemma3_l3_l4_iterative_residual_diagnostic import (
    build_parser,
)


def test_parser_exposes_only_fit_artifacts_and_frozen_hashes() -> None:
    parser = build_parser()
    destinations = {
        action.dest for action in parser._actions  # noqa: SLF001
    }
    assert {
        "corpus_artifact",
        "fit_input",
        "materialization_report",
        "materialization_report_sha256",
        "materialization_report_file_sha256",
        "factorial_report",
        "factorial_report_sha256",
        "factorial_report_file_sha256",
        "graph_candidate",
        "basis_package",
        "base_artifact",
        "refit_artifact",
        "output",
        "cache_dir",
        "help",
    } == destinations
    assert not destinations & {
        "selection",
        "guard",
        "assessment",
        "calibration_b",
        "rank",
        "ridge",
        "alpha",
        "lag_count",
        "candidate_count",
        "gate_threshold",
    }


@pytest.mark.parametrize(
    "flag",
    (
        "--selection",
        "--guard",
        "--assessment",
        "--calibration-b",
        "--rank",
        "--ridge",
        "--alpha",
        "--lag-count",
    ),
)
def test_parser_rejects_protected_roles_and_search_knobs(flag: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--materialization-report-sha256",
                "a" * 64,
                "--materialization-report-file-sha256",
                "b" * 64,
                "--factorial-report-sha256",
                "c" * 64,
                "--factorial-report-file-sha256",
                "d" * 64,
                flag,
                "value",
            ]
        )
