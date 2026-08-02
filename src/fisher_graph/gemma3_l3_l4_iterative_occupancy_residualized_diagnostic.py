"""Run the residualized occupancy rung on reusable development data only.

This command deliberately interrupts the Iteration-5 driver at its
development-selector callback.  It receives the already collected,
authenticated scalar fit records and direct LOFO receipts, builds the
residualized replay report, and exits before full providers, a durable claim,
or private selection-panel materialization can occur.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from .gemma3_l3_l4_iterative_occupancy_residualized_development import (
    build_gemma_iterative_residualized_occupancy_development_report,
    validate_gemma_iterative_residualized_occupancy_development_report,
)
from .gemma3_l3_l4_iterative_occupancy_selection_diagnostic import (
    _publish_report,
    build_parser as build_selection_parser,
    run_gemma_iterative_occupancy_selection_diagnostic,
)


__all__ = [
    "CLI_NAME",
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma_iterative_residualized_occupancy_development_diagnostic",
]


CLI_NAME = (
    "fisher-graph-gemma-l3-l4-iterative-occupancy-residualized-dev"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "progressive-a-iterative-occupancy-residualized-dev-v1.report.json"
)


class _DevelopmentOnlyComplete(RuntimeError):
    def __init__(self, report: Mapping[str, object]) -> None:
        super().__init__("residualized occupancy development is complete")
        self.report = dict(report)


def run_gemma_iterative_residualized_occupancy_development_diagnostic(
    *,
    corpus_artifact_path: Path | str,
    fit_input_path: Path | str,
    materialization_report_path: Path | str,
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    factorial_report_path: Path | str,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    prior_iteration_report_path: Path | str,
    expected_prior_iteration_report_sha256: str,
    expected_prior_iteration_report_file_sha256: str,
    expected_prior_iteration_collection_sha256: str,
    selection_panel_path: Path | str,
    expected_selection_panel_artifact_sha256: str,
    expected_selection_panel_file_sha256: str,
    selection_input_path: Path | str,
    selection_claim_path: Path | str,
    graph_candidate_path: Path | str,
    basis_package_path: Path | str,
    base_artifact_path: Path | str,
    refit_artifact_path: Path | str,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Collect 32 reusable forwards and stop before every fresh boundary."""

    destination = Path(output)
    claim = Path(selection_claim_path)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite residualized occupancy report"
        )
    if claim.exists():
        raise FileExistsError(
            "residualized development requires an unclaimed fresh boundary"
        )

    def _capture(
        *,
        fit_records: Sequence[object],
        fold_receipts_by_arm: Mapping[str, Sequence[object]],
    ) -> Mapping[str, object]:
        report = (
            build_gemma_iterative_residualized_occupancy_development_report(
                fit_records=fit_records,
                direct_fold_receipts_by_arm=fold_receipts_by_arm,
            )
        )
        raise _DevelopmentOnlyComplete(report)

    try:
        run_gemma_iterative_occupancy_selection_diagnostic(
            corpus_artifact_path=corpus_artifact_path,
            fit_input_path=fit_input_path,
            materialization_report_path=materialization_report_path,
            expected_materialization_report_sha256=(
                expected_materialization_report_sha256
            ),
            expected_materialization_report_file_sha256=(
                expected_materialization_report_file_sha256
            ),
            factorial_report_path=factorial_report_path,
            expected_factorial_report_sha256=(
                expected_factorial_report_sha256
            ),
            expected_factorial_report_file_sha256=(
                expected_factorial_report_file_sha256
            ),
            prior_iteration_report_path=prior_iteration_report_path,
            expected_prior_iteration_report_sha256=(
                expected_prior_iteration_report_sha256
            ),
            expected_prior_iteration_report_file_sha256=(
                expected_prior_iteration_report_file_sha256
            ),
            expected_prior_iteration_collection_sha256=(
                expected_prior_iteration_collection_sha256
            ),
            selection_panel_path=selection_panel_path,
            expected_selection_panel_artifact_sha256=(
                expected_selection_panel_artifact_sha256
            ),
            expected_selection_panel_file_sha256=(
                expected_selection_panel_file_sha256
            ),
            selection_input_path=selection_input_path,
            selection_claim_path=selection_claim_path,
            graph_candidate_path=graph_candidate_path,
            basis_package_path=basis_package_path,
            base_artifact_path=base_artifact_path,
            refit_artifact_path=refit_artifact_path,
            output=destination,
            cache_dir=cache_dir,
            _build_development_selection=_capture,
        )
    except _DevelopmentOnlyComplete as completed:
        report = completed.report
    else:
        raise RuntimeError(
            "residualized development crossed its mandatory stop boundary"
        )

    if claim.exists():
        raise RuntimeError(
            "residualized development unexpectedly created a fresh claim"
        )
    validate_gemma_iterative_residualized_occupancy_development_report(
        report
    )
    _publish_report(destination, report)
    with destination.open("r", encoding="utf-8") as handle:
        replay = json.load(handle)
    validate_gemma_iterative_residualized_occupancy_development_report(
        replay
    )
    if json.dumps(replay, sort_keys=True) != json.dumps(
        report, sort_keys=True
    ):
        raise RuntimeError("published residualized development report differs")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = build_selection_parser()
    parser.description = (
        "Run fold-local occupancy residualization on the reusable A-fit "
        "panel and stop before any fresh-panel claim or opening."
    )
    parser.set_defaults(output=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = (
        run_gemma_iterative_residualized_occupancy_development_diagnostic(
            corpus_artifact_path=args.corpus_artifact,
            fit_input_path=args.fit_input,
            materialization_report_path=args.materialization_report,
            expected_materialization_report_sha256=(
                args.materialization_report_sha256
            ),
            expected_materialization_report_file_sha256=(
                args.materialization_report_file_sha256
            ),
            factorial_report_path=args.factorial_report,
            expected_factorial_report_sha256=(
                args.factorial_report_sha256
            ),
            expected_factorial_report_file_sha256=(
                args.factorial_report_file_sha256
            ),
            prior_iteration_report_path=args.prior_iteration_report,
            expected_prior_iteration_report_sha256=(
                args.prior_iteration_report_sha256
            ),
            expected_prior_iteration_report_file_sha256=(
                args.prior_iteration_report_file_sha256
            ),
            expected_prior_iteration_collection_sha256=(
                args.prior_iteration_collection_sha256
            ),
            selection_panel_path=args.selection_panel,
            expected_selection_panel_artifact_sha256=(
                args.selection_panel_sha256
            ),
            expected_selection_panel_file_sha256=(
                args.selection_panel_file_sha256
            ),
            selection_input_path=args.selection_input,
            selection_claim_path=args.selection_claim,
            graph_candidate_path=args.graph_candidate,
            basis_package_path=args.basis_package,
            base_artifact_path=args.base_artifact,
            refit_artifact_path=args.refit_artifact,
            output=args.output,
            cache_dir=args.cache_dir,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
