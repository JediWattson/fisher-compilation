"""Execute iteration two: a causal top-two modal state router for Gemma."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_l3_l4_basis_package import DEFAULT_BASIS_PACKAGE
from .gemma3_l3_l4_graph_organized_svd_experiment import (
    DEFAULT_OUTPUT as DEFAULT_GRAPH_CANDIDATE,
)
from .gemma3_l3_l4_iterative_residual_analysis import (
    validate_gemma_iterative_residual_report,
)
from .gemma3_l3_l4_iterative_residual_diagnostic import (
    _GemmaIterativeDiagnosticRecipe,
    run_gemma_iterative_residual_diagnostic,
)
from .gemma3_l3_l4_iterative_state_router import (
    GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE,
    build_gemma_iterative_state_router_fit_record,
    fit_gemma_iterative_state_router_fold_provider,
    fit_gemma_iterative_state_router_full_provider,
)
from .gemma3_l3_l4_iterative_state_router_analysis import (
    build_gemma_iterative_state_router_report,
    validate_gemma_iterative_state_router_report,
)
from .gemma3_l3_l4_progressive_a_campaign import _file_sha256


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma_iterative_state_router_diagnostic",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
_DEFAULT_EXPANDED_CORPUS = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
)
_DEFAULT_EXPANDED_FIT_INPUT = (
    _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
)
_DEFAULT_MATERIALIZATION_REPORT = (
    _LOCAL_ROOT / "progressive-a-h4-damping-materialization-v1.report.json"
)
_DEFAULT_FACTORIAL_REPORT = (
    _LOCAL_ROOT / "progressive-a-x4-h4-factorial-fit-v1.report.json"
)
_DEFAULT_PRIOR_ITERATION_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-residual-position-v1.report.json"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-state-router-top2-v1.report.json"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _load_rejected_prior_iteration(
    path: Path | str,
    *,
    expected_report_sha256: str,
    expected_report_file_sha256: str,
    expected_collection_sha256: str,
) -> dict[str, object]:
    """Load and prove that iteration one is the rejected direct parent."""

    source = Path(path)
    if _file_sha256(source) != expected_report_file_sha256:
        raise ValueError("prior iteration report file hash mismatch")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("prior iteration report must be a JSON object")
    validate_gemma_iterative_residual_report(value)
    if value.get("report_sha256") != expected_report_sha256:
        raise ValueError("prior iteration report logical hash mismatch")
    if value.get("collection_sha256") != expected_collection_sha256:
        raise ValueError("prior iteration collection hash mismatch")
    decision = _mapping(
        value.get("decision"),
        label="prior iteration decision",
    )
    if (
        decision.get("retained") is not False
        or decision.get("deployment_authorized") is not False
        or value.get("retained_full_fit") is not None
    ):
        raise ValueError(
            "state-router iteration requires the rejected position parent"
        )
    return value


def _router_make_fit_record(
    *,
    parent_h4: object,
    lag_b_correction: object,
    **kwargs: object,
) -> object:
    # The fused recipe consumes the authenticated parent modal boundary
    # directly; the generic campaign still computes the decoded parent
    # correction to preserve the iteration-one execution envelope.
    del lag_b_correction
    return build_gemma_iterative_state_router_fit_record(
        parent_h4=parent_h4,  # type: ignore[arg-type]
        **kwargs,
    )


def _router_fit_fold(
    *,
    parent_artifact_sha256: str,
    **kwargs: object,
) -> object:
    return fit_gemma_iterative_state_router_fold_provider(
        **kwargs,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def _router_fit_full(
    *,
    parent_artifact_sha256: str,
    **kwargs: object,
) -> object:
    return fit_gemma_iterative_state_router_full_provider(
        **kwargs,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def run_gemma_iterative_state_router_diagnostic(
    *,
    corpus_artifact_path: Path | str = _DEFAULT_EXPANDED_CORPUS,
    fit_input_path: Path | str = _DEFAULT_EXPANDED_FIT_INPUT,
    materialization_report_path: Path | str = (
        _DEFAULT_MATERIALIZATION_REPORT
    ),
    expected_materialization_report_sha256: str,
    expected_materialization_report_file_sha256: str,
    factorial_report_path: Path | str = _DEFAULT_FACTORIAL_REPORT,
    expected_factorial_report_sha256: str,
    expected_factorial_report_file_sha256: str,
    prior_iteration_report_path: Path | str = (
        _DEFAULT_PRIOR_ITERATION_REPORT
    ),
    expected_prior_iteration_report_sha256: str,
    expected_prior_iteration_report_file_sha256: str,
    expected_prior_iteration_collection_sha256: str,
    graph_candidate_path: Path | str = DEFAULT_GRAPH_CANDIDATE,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
) -> dict[str, object]:
    """Run the frozen state-router recipe on expanded A-fit only."""

    prior = _load_rejected_prior_iteration(
        prior_iteration_report_path,
        expected_report_sha256=(
            expected_prior_iteration_report_sha256
        ),
        expected_report_file_sha256=(
            expected_prior_iteration_report_file_sha256
        ),
        expected_collection_sha256=(
            expected_prior_iteration_collection_sha256
        ),
    )
    prior_lineage = _mapping(
        prior.get("lineage"),
        label="prior iteration lineage",
    )
    diagnostic_recipe = _GemmaIterativeDiagnosticRecipe(
        campaign_recipe=GEMMA_ITERATIVE_STATE_ROUTER_CAMPAIGN_RECIPE,
        make_fit_record=_router_make_fit_record,
        fit_fold=_router_fit_fold,
        build_report=build_gemma_iterative_state_router_report,
        fit_full=_router_fit_full,
        validate_report=validate_gemma_iterative_state_router_report,
        report_label="iterative state-router",
        expected_parent_lineage={
            str(key): str(value)
            for key, value in prior_lineage.items()
        },
        extra_lineage={
            "prior_iteration_report_sha256": (
                expected_prior_iteration_report_sha256
            ),
            "prior_iteration_report_file_sha256": (
                expected_prior_iteration_report_file_sha256
            ),
            "prior_iteration_collection_sha256": (
                expected_prior_iteration_collection_sha256
            ),
        },
        extra_immutable_inputs=(
            (
                "prior_iteration_report",
                Path(prior_iteration_report_path),
                expected_prior_iteration_report_file_sha256,
            ),
        ),
        source_code_files=(
            "gemma3_l3_l4_iterative_state_router_diagnostic.py",
            "gemma3_l3_l4_iterative_state_router.py",
            "gemma3_l3_l4_iterative_state_router_analysis.py",
            "gemma3_l3_l4_two_head_lowerer.py",
        ),
    )
    return run_gemma_iterative_residual_diagnostic(
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
        graph_candidate_path=graph_candidate_path,
        basis_package_path=basis_package_path,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        output=output,
        cache_dir=cache_dir,
        _diagnostic_recipe=diagnostic_recipe,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen causal top-two state-router iteration on "
            "reusable A-fit."
        )
    )
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=_DEFAULT_EXPANDED_CORPUS,
    )
    parser.add_argument(
        "--fit-input",
        type=Path,
        default=_DEFAULT_EXPANDED_FIT_INPUT,
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=_DEFAULT_MATERIALIZATION_REPORT,
    )
    parser.add_argument("--materialization-report-sha256", required=True)
    parser.add_argument(
        "--materialization-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--factorial-report",
        type=Path,
        default=_DEFAULT_FACTORIAL_REPORT,
    )
    parser.add_argument("--factorial-report-sha256", required=True)
    parser.add_argument(
        "--factorial-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-iteration-report",
        type=Path,
        default=_DEFAULT_PRIOR_ITERATION_REPORT,
    )
    parser.add_argument("--prior-iteration-report-sha256", required=True)
    parser.add_argument(
        "--prior-iteration-report-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--prior-iteration-collection-sha256",
        required=True,
    )
    parser.add_argument(
        "--graph-candidate",
        type=Path,
        default=DEFAULT_GRAPH_CANDIDATE,
    )
    parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gemma_iterative_state_router_diagnostic(
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
        expected_factorial_report_sha256=args.factorial_report_sha256,
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
        graph_candidate_path=args.graph_candidate,
        basis_package_path=args.basis_package,
        base_artifact_path=args.base_artifact,
        refit_artifact_path=args.refit_artifact,
        output=args.output,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
