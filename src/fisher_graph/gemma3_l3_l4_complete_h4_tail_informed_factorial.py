"""Locked A16 single-projector tail-informed complete-H4 factorial.

The eight arms compare the unweighted U320 prefixes at total ranks
192/224/256/320 against one nested global projector ordered as U192, the full
numerical SVD span of the U192 causal-tail residual, and deterministically
orthogonalized remaining U320 directions.  This is a same-A capacity screen;
it does not authorize serving, compression, or learned prediction.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from . import gemma3_l3_l4_complete_h4_projection_basis_rank_ladder as ladder
from . import gemma3_l3_l4_complete_h4_projection_experiment as frozen


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_PARENT_LADDER",
    "build_parser",
    "main",
    "run_gemma3_l3_l4_complete_h4_tail_informed_factorial",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_PARENT_LADDER = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-"
    "projection-basis-rank-ladder-a-fit16-dev-v1.json"
)
DEFAULT_OUTPUT = _LOCAL_ROOT / (
    "modal-generator-l3-l4-graph-wavelet-signed-g8-complete-h4-"
    "tail-informed-single-projector-factorial-a-fit16-dev-v1.json"
)

_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_complete_h4_tail_informed_single_projector_"
    "factorial_development"
)
_FORMAT_VERSION = 1
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-complete-h4-tail-informed-single-projector-"
    b"factorial:v1\0"
)
_ROLE = (
    "reused_calibration_a_truth_leaking_complete_h4_tail_informed_"
    "single_global_projector_factorial"
)
_PARENT_FILE_SHA256 = (
    "eb25c0fef53e6dd0a7a5be9726222278fd95848a5368b0f7225ecfe109cc26b9"
)
_PARENT_REPORT_SHA256 = (
    "647c4ad889199dad4f50851218e6df3e4de9254fba247bd98dd15ad5d0cb1cee"
)


def run_gemma3_l3_l4_complete_h4_tail_informed_factorial(
    *,
    fit_source_artifact_path: Path | str = frozen.DEFAULT_INTERIOR_ARTIFACT,
    parent_artifact_path: Path | str = frozen.DEFAULT_PARENT_ARTIFACT,
    candidate_artifact_path: Path | str = frozen.DEFAULT_CANDIDATE_ARTIFACT,
    basis_package_path: Path | str = frozen.DEFAULT_BASIS_PACKAGE,
    base_artifact_path: Path | str = frozen.DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = (
        frozen.DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT
    ),
    panel_path: Path | str = frozen.DEFAULT_PANEL,
    rank64_x4_baseline_path: Path | str = frozen.DEFAULT_RANK64_X4_BASELINE,
    complete_h4_identity_path: Path | str = frozen.DEFAULT_COMPLETE_H4_IDENTITY,
    rank64_projection_baseline_path: Path | str = (
        ladder.DEFAULT_RANK64_PROJECTION_BASELINE
    ),
    parent_ladder_path: Path | str = DEFAULT_PARENT_LADDER,
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    max_length: int = frozen.DEFAULT_MAX_LENGTH,
) -> dict[str, object]:
    config = ladder._TailInformedFactorialConfig(
        schema=_SCHEMA,
        format_version=_FORMAT_VERSION,
        report_domain=_REPORT_DOMAIN,
        role=_ROLE,
        parent_ladder_path=parent_ladder_path,
        parent_ladder_file_sha256=_PARENT_FILE_SHA256,
        parent_ladder_report_sha256=_PARENT_REPORT_SHA256,
    )
    return ladder.run_gemma3_l3_l4_complete_h4_projection_basis_rank_ladder(
        fit_source_artifact_path=fit_source_artifact_path,
        parent_artifact_path=parent_artifact_path,
        candidate_artifact_path=candidate_artifact_path,
        basis_package_path=basis_package_path,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        panel_path=panel_path,
        rank64_x4_baseline_path=rank64_x4_baseline_path,
        complete_h4_identity_path=complete_h4_identity_path,
        rank64_projection_baseline_path=rank64_projection_baseline_path,
        output=output,
        cache_dir=cache_dir,
        max_length=max_length,
        _tail_informed_factorial=config,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = ladder.build_parser()
    parser.description = (
        "Run the locked A16 complete-H4 tail-informed single-projector factorial"
    )
    parser.set_defaults(output=DEFAULT_OUTPUT)
    parser.add_argument(
        "--parent-basis-rank-ladder",
        default=DEFAULT_PARENT_LADDER,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_complete_h4_tail_informed_factorial(
        fit_source_artifact_path=arguments.fit_source_artifact,
        parent_artifact_path=arguments.parent_artifact,
        candidate_artifact_path=arguments.candidate_artifact,
        basis_package_path=arguments.basis_package,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        panel_path=arguments.panel,
        rank64_x4_baseline_path=arguments.rank64_x4_baseline,
        complete_h4_identity_path=arguments.complete_h4_identity,
        rank64_projection_baseline_path=arguments.rank64_projection_baseline,
        parent_ladder_path=arguments.parent_basis_rank_ladder,
        output=arguments.output,
        cache_dir=arguments.cache_dir,
        max_length=arguments.max_length,
    )
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "artifact": report["artifact"],
                "selection": report["selection"],
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
