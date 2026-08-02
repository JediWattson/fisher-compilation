"""Execute iteration four: a pooled conformal route over Gemma modal state."""

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
from .gemma3_l3_l4_iterative_conformal_route import (
    GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE,
    build_gemma_iterative_conformal_route_fit_record,
    fit_gemma_iterative_conformal_route_fold_provider,
    fit_gemma_iterative_conformal_route_full_provider,
)
from .gemma3_l3_l4_iterative_conformal_route_analysis import (
    build_gemma_iterative_conformal_route_report,
    validate_gemma_iterative_conformal_route_report,
)
from .gemma3_l3_l4_iterative_residual_diagnostic import (
    _GemmaIterativeDiagnosticRecipe,
    run_gemma_iterative_residual_diagnostic,
)
from .gemma3_l3_l4_iterative_state_experts_analysis import (
    validate_gemma_iterative_state_experts_report,
)
from .gemma3_l3_l4_progressive_a_campaign import _file_sha256


__all__ = [
    "CLI_NAME",
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma_iterative_conformal_route_diagnostic",
]


CLI_NAME = "fisher-graph-gemma-l3-l4-iterative-conformal-route-dev"
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
    / "progressive-a-iterative-state-experts-sign-v1.report.json"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-conformal-route-v1.report.json"
)
_STATE_EXPERTS_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_iterative_state_experts_analysis.sign_v1"
)
_LIVE_PARENT_LINEAGE_KEYS = frozenset(
    {
        "parent_artifact_sha256",
        "parent_h4_head_sha256",
        "accepted_x4_head_sha256",
        "bridge_binding_sha256",
        "model_sha256",
        "adapter_execution_sha256",
        "fit_manifest_sha256",
        "factorial_report_sha256",
        "factorial_report_file_sha256",
    }
)
_PRIOR_ITERATION_LINEAGE_KEYS = frozenset(
    {
        "prior_iteration_report_sha256",
        "prior_iteration_report_file_sha256",
        "prior_iteration_collection_sha256",
    }
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
    """Load and prove that rejected iteration three is the direct parent."""

    source = Path(path)
    if _file_sha256(source) != expected_report_file_sha256:
        raise ValueError("prior state-experts report file hash mismatch")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError("prior state-experts report must be a JSON object")
    validate_gemma_iterative_state_experts_report(value)
    if value.get("schema") != _STATE_EXPERTS_SCHEMA:
        raise ValueError("prior report is not iteration-three state experts")
    semantics = _mapping(
        value.get("semantics"),
        label="prior state-experts semantics",
    )
    if semantics.get("iteration") != 3:
        raise ValueError("prior report is not iteration-three state experts")
    if value.get("report_sha256") != expected_report_sha256:
        raise ValueError("prior state-experts report logical hash mismatch")
    if value.get("collection_sha256") != expected_collection_sha256:
        raise ValueError("prior state-experts collection hash mismatch")
    decision = _mapping(
        value.get("decision"),
        label="prior state-experts decision",
    )
    if (
        decision.get("retained") is not False
        or decision.get("ready_for_new_selection") is not False
        or decision.get("deployment_authorized") is not False
        or "retained_full_fit" not in value
        or value.get("retained_full_fit") is not None
    ):
        raise ValueError(
            "conformal-route iteration requires rejected state-experts "
            "with no selection, deployment, or retained full fit"
        )
    lineage = _mapping(
        value.get("lineage"),
        label="prior state-experts lineage",
    )
    if set(lineage) != (
        _LIVE_PARENT_LINEAGE_KEYS | _PRIOR_ITERATION_LINEAGE_KEYS
    ):
        raise ValueError("prior state-experts lineage fields differ")
    return value


def _conformal_make_fit_record(
    *,
    parent_h4: object,
    lag_b_correction: object,
    **kwargs: object,
) -> object:
    # The route is fit against the authenticated accepted-X4 + lag-B parent.
    # The rejected iteration-three provider is provenance only and is never
    # stacked into the candidate execution path.
    del lag_b_correction
    return build_gemma_iterative_conformal_route_fit_record(
        parent_h4=parent_h4,  # type: ignore[arg-type]
        **kwargs,
    )


def _conformal_fit_fold(
    *,
    parent_artifact_sha256: str,
    **kwargs: object,
) -> object:
    return fit_gemma_iterative_conformal_route_fold_provider(
        **kwargs,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def _conformal_fit_full(
    *,
    parent_artifact_sha256: str,
    **kwargs: object,
) -> object:
    return fit_gemma_iterative_conformal_route_full_provider(
        **kwargs,
        parent_artifact_sha256=parent_artifact_sha256,
    )


def _validate_published_report(
    path: Path | str,
    *,
    expected_report: Mapping[str, object],
) -> None:
    """Self-validate the returned and exactly replayed published report."""

    validate_gemma_iterative_conformal_route_report(expected_report)
    with Path(path).open("r", encoding="utf-8") as handle:
        published = json.load(handle)
    if not isinstance(published, dict):
        raise TypeError(
            "published iterative conformal-route report must be a JSON object"
        )
    validate_gemma_iterative_conformal_route_report(published)
    expected_canonical = json.dumps(
        expected_report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    published_canonical = json.dumps(
        published,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if published_canonical != expected_canonical:
        raise RuntimeError(
            "published iterative conformal-route report differs from "
            "validated in-memory report"
        )


def run_gemma_iterative_conformal_route_diagnostic(
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
    """Run the frozen pooled conformal-route recipe on expanded A-fit only."""

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
        label="prior state-experts lineage",
    )
    diagnostic_recipe = _GemmaIterativeDiagnosticRecipe(
        campaign_recipe=GEMMA_ITERATIVE_CONFORMAL_ROUTE_CAMPAIGN_RECIPE,
        make_fit_record=_conformal_make_fit_record,
        fit_fold=_conformal_fit_fold,
        build_report=build_gemma_iterative_conformal_route_report,
        fit_full=_conformal_fit_full,
        validate_report=validate_gemma_iterative_conformal_route_report,
        report_label="iterative conformal-route",
        expected_parent_lineage={
            key: str(prior_lineage[key])
            for key in sorted(_LIVE_PARENT_LINEAGE_KEYS)
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
            "gemma3_l3_l4_iterative_conformal_route_diagnostic.py",
            "gemma3_l3_l4_iterative_conformal_route.py",
            "gemma3_l3_l4_iterative_conformal_route_analysis.py",
            "gemma3_l3_l4_iterative_state_experts.py",
            "gemma3_l3_l4_iterative_state_experts_analysis.py",
            "gemma3_l3_l4_iterative_state_router.py",
            "gemma3_l3_l4_iterative_state_router_analysis.py",
            "gemma3_l3_l4_two_head_lowerer.py",
        ),
    )
    report = run_gemma_iterative_residual_diagnostic(
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
    _validate_published_report(output, expected_report=report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen pooled affine conformal route on reusable A-fit."
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
    report = run_gemma_iterative_conformal_route_diagnostic(
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
