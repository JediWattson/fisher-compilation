"""Run the public synthetic compact-message completion qualification campaign.

This command is deliberately separate from the current-corpus P0 runner.  It
uses only the deterministic public synthetic fixture and cannot create a
scientific result or a guard claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any


COMMAND_NAME = (
    "fisher-graph-compact-message-completion-synthetic-qualification"
)
EVIDENCE_SCOPE = "synthetic_qualification_only"
INPUT_SCOPE = "deterministic_public_synthetic_fixture_only"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=COMMAND_NAME,
        description=(
            "Run the deterministic public/synthetic compact-message completion "
            "qualification. This does not run the current-corpus P0 command and "
            "cannot establish a scientific result or guard claim."
        ),
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help=(
            "fresh output directory for the isolated qualification; it must "
            "not already exist and must not be under .local-runs or a corpus "
            "directory"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "exact"),
        default="quick",
        help=(
            "optimizer schedule: quick for a short mechanical check (default), "
            "or exact for the frozen full schedules; both remain synthetic-only"
        ),
    )
    return parser


def _campaign_runner() -> Callable[..., Any]:
    """Import the tensor-backed campaign only after argument parsing."""

    from fisher_graph.fisher_compact_message_completion_campaign import (
        run_synthetic_campaign,
    )

    return run_synthetic_campaign


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value).__name__} as JSON")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = _campaign_runner()(args.output_root, profile=args.profile)
    if result.scientific_claim_eligible is not False:
        raise RuntimeError(
            "synthetic qualification unexpectedly became scientific-claim eligible"
        )
    if result.protected_panel_accessed is not False:
        raise RuntimeError(
            "synthetic qualification unexpectedly accessed a protected panel"
        )
    print(
        json.dumps(
            {
                "campaign_result": asdict(result),
                "evidence_scope": EVIDENCE_SCOPE,
                "guard_claim_eligible": False,
                "input_scope": INPUT_SCOPE,
                "protected_panel_accessed": False,
                "scientific_claim_eligible": False,
                "target_model_execution_performed": False,
                "statement": (
                    "Public synthetic qualification only; no scientific result "
                    "and no guard claim."
                ),
            },
            default=_json_default,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMMAND_NAME",
    "EVIDENCE_SCOPE",
    "INPUT_SCOPE",
    "build_parser",
    "main",
]
