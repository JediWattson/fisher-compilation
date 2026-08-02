"""Replay the partially pooled corrective screen without model execution."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .gemma3_l3_l4_iterative_fisher_corrective_development import (
    build_gemma_iterative_fisher_corrective_development_report,
    replay_gemma_iterative_fisher_corrective_development_report,
    validate_gemma_iterative_fisher_corrective_development_report,
)
from .gemma3_l3_l4_iterative_token_fisher_development import (
    validate_gemma_iterative_token_fisher_development_report,
)


__all__ = [
    "CLI_NAME",
    "DEFAULT_OUTPUT",
    "DEFAULT_TOKEN_FISHER_REPORT",
    "build_parser",
    "main",
    "run_gemma_iterative_fisher_corrective_development_diagnostic",
]


CLI_NAME = (
    "fisher-graph-gemma-l3-l4-iterative-fisher-corrective-dev"
)
_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_TOKEN_FISHER_REPORT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-token-loss-fisher-dev-v1.report.json"
)
DEFAULT_OUTPUT = (
    _LOCAL_ROOT
    / "progressive-a-iterative-fisher-corrective-dev-v1.report.json"
)


def _load_json(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must contain a JSON object")
    return dict(value), file_sha256


def _publish_json_once(
    destination: Path,
    report: Mapping[str, object],
) -> None:
    """Validate temporary bytes, then atomically install without overwrite."""

    if destination.exists():
        raise FileExistsError(
            "refusing to overwrite Fisher corrective development report"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
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
        if json.dumps(replay, sort_keys=True) != json.dumps(
            report, sort_keys=True
        ):
            raise RuntimeError("temporary corrective report differs")
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                "refusing to overwrite Fisher corrective development report"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_gemma_iterative_fisher_corrective_development_diagnostic(
    *,
    token_fisher_report_path: Path | str = DEFAULT_TOKEN_FISHER_REPORT,
    expected_token_fisher_report_sha256: str,
    expected_token_fisher_report_file_sha256: str,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Run the zero-forward adaptive screen and publish one strict report."""

    source = Path(token_fisher_report_path)
    destination = Path(output)
    upstream, actual_file_sha256 = _load_json(
        source, label="token Fisher report"
    )
    validate_gemma_iterative_token_fisher_development_report(upstream)
    if upstream.get("report_sha256") != expected_token_fisher_report_sha256:
        raise ValueError("token Fisher logical hash differs")
    if actual_file_sha256 != expected_token_fisher_report_file_sha256:
        raise ValueError("token Fisher file hash differs")
    report = (
        build_gemma_iterative_fisher_corrective_development_report(
            token_fisher_report=upstream,
            token_fisher_report_file_sha256=actual_file_sha256,
        )
    )
    validate_gemma_iterative_fisher_corrective_development_report(report)
    replay_gemma_iterative_fisher_corrective_development_report(
        token_fisher_report=upstream,
        token_fisher_report_file_sha256=actual_file_sha256,
        report=report,
    )
    _publish_json_once(destination, report)
    replay, replay_file_sha256 = _load_json(
        destination, label="published corrective report"
    )
    if replay_file_sha256 != hashlib.sha256(
        destination.read_bytes()
    ).hexdigest():
        raise RuntimeError("published corrective file hash replay differs")
    validate_gemma_iterative_fisher_corrective_development_report(replay)
    replay_gemma_iterative_fisher_corrective_development_report(
        token_fisher_report=upstream,
        token_fisher_report_file_sha256=actual_file_sha256,
        report=replay,
    )
    if json.dumps(replay, sort_keys=True) != json.dumps(
        report, sort_keys=True
    ):
        raise RuntimeError("published corrective report differs")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "Run nested family-held-out partial pooling over the frozen "
            "exact-token Fisher sufficient statistics. This performs zero "
            "model forwards and cannot compile a provider."
        ),
    )
    parser.add_argument(
        "--token-fisher-report",
        type=Path,
        default=DEFAULT_TOKEN_FISHER_REPORT,
    )
    parser.add_argument(
        "--token-fisher-report-sha256",
        required=True,
    )
    parser.add_argument(
        "--token-fisher-report-file-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = (
        run_gemma_iterative_fisher_corrective_development_diagnostic(
            token_fisher_report_path=args.token_fisher_report,
            expected_token_fisher_report_sha256=(
                args.token_fisher_report_sha256
            ),
            expected_token_fisher_report_file_sha256=(
                args.token_fisher_report_file_sha256
            ),
            output=args.output,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
