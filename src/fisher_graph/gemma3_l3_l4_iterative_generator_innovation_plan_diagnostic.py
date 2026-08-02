"""Publish the fixed Fisher-generator innovation plan without model calls."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .gemma3_l3_l4_iterative_generator_innovation_plan import (
    build_gemma_iterative_generator_innovation_plan,
    replay_gemma_iterative_generator_innovation_plan,
    validate_gemma_iterative_generator_innovation_plan,
)


__all__ = [
    "CLI_NAME",
    "DEFAULT_CORRECTIVE_REPORT",
    "DEFAULT_OUTPUT",
    "DEFAULT_TOKEN_FISHER_REPORT",
    "build_parser",
    "main",
    "publish_generator_innovation_plan_once",
    "run_gemma_iterative_generator_innovation_plan_diagnostic",
]


CLI_NAME = "fisher-graph-gemma-l3-l4-generator-innovation-plan"
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_TOKEN_FISHER_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-token-loss-fisher-dev-v1.report.json"
)
DEFAULT_CORRECTIVE_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-fisher-corrective-dev-v1.report.json"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-generator-innovation-plan-v1.report.json"
)


def _load_json(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return dict(value), hashlib.sha256(raw).hexdigest()


def publish_generator_innovation_plan_once(
    destination: Path,
    plan: Mapping[str, object],
) -> None:
    """Validate and atomically install a plan without permitting overwrite."""

    validate_gemma_iterative_generator_innovation_plan(plan)
    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite generator innovation plan"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        replay = json.loads(temporary.read_text(encoding="utf-8"))
        validate_gemma_iterative_generator_innovation_plan(replay)
        if json.dumps(replay, sort_keys=True) != json.dumps(
            plan, sort_keys=True
        ):
            raise RuntimeError("temporary generator innovation plan differs")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite generator innovation plan"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_gemma_iterative_generator_innovation_plan_diagnostic(
    *,
    token_fisher_report_path: Path | str = DEFAULT_TOKEN_FISHER_REPORT,
    expected_token_fisher_report_sha256: str,
    expected_token_fisher_report_file_sha256: str,
    corrective_report_path: Path | str = DEFAULT_CORRECTIVE_REPORT,
    expected_corrective_report_sha256: str,
    expected_corrective_report_file_sha256: str,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Authenticate both sources, build, replay, and publish the fixed plan."""

    token_report, token_file_sha256 = _load_json(
        Path(token_fisher_report_path),
        label="token Fisher report",
    )
    corrective_report, corrective_file_sha256 = _load_json(
        Path(corrective_report_path),
        label="corrective report",
    )
    if (
        token_report.get("report_sha256")
        != expected_token_fisher_report_sha256
        or token_file_sha256 != expected_token_fisher_report_file_sha256
    ):
        raise ValueError("token Fisher report identity differs")
    if (
        corrective_report.get("report_sha256")
        != expected_corrective_report_sha256
        or corrective_file_sha256 != expected_corrective_report_file_sha256
    ):
        raise ValueError("corrective report identity differs")
    plan = build_gemma_iterative_generator_innovation_plan(
        token_fisher_report=token_report,
        token_fisher_report_file_sha256=token_file_sha256,
        corrective_report=corrective_report,
        corrective_report_file_sha256=corrective_file_sha256,
    )
    validate_gemma_iterative_generator_innovation_plan(plan)
    replay_gemma_iterative_generator_innovation_plan(
        token_fisher_report=token_report,
        token_fisher_report_file_sha256=token_file_sha256,
        corrective_report=corrective_report,
        corrective_report_file_sha256=corrective_file_sha256,
        plan=plan,
    )
    destination = Path(output)
    publish_generator_innovation_plan_once(destination, plan)
    published, _published_file_sha256 = _load_json(
        destination,
        label="published generator innovation plan",
    )
    replay_gemma_iterative_generator_innovation_plan(
        token_fisher_report=token_report,
        token_fisher_report_file_sha256=token_file_sha256,
        corrective_report=corrective_report,
        corrective_report_file_sha256=corrective_file_sha256,
        plan=published,
    )
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Freeze and hash the raw-Fisher two-generator basis, causal "
            "innovation feature, controls, and staged validation gates. "
            "This command performs no model execution."
        ),
    )
    parser.add_argument(
        "--token-fisher-report",
        type=Path,
        default=DEFAULT_TOKEN_FISHER_REPORT,
    )
    parser.add_argument("--token-fisher-report-sha256", required=True)
    parser.add_argument("--token-fisher-report-file-sha256", required=True)
    parser.add_argument(
        "--corrective-report",
        type=Path,
        default=DEFAULT_CORRECTIVE_REPORT,
    )
    parser.add_argument("--corrective-report-sha256", required=True)
    parser.add_argument("--corrective-report-file-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = run_gemma_iterative_generator_innovation_plan_diagnostic(
        token_fisher_report_path=args.token_fisher_report,
        expected_token_fisher_report_sha256=(
            args.token_fisher_report_sha256
        ),
        expected_token_fisher_report_file_sha256=(
            args.token_fisher_report_file_sha256
        ),
        corrective_report_path=args.corrective_report,
        expected_corrective_report_sha256=args.corrective_report_sha256,
        expected_corrective_report_file_sha256=(
            args.corrective_report_file_sha256
        ),
        output=args.output,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
