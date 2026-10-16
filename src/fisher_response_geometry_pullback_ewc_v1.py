"""Standalone synthetic CLI for the Pullback-Fisher input chart + EWC consolidation ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "fisher_response_geometry_pullback_ewc_v1"
P0_STATUS = "development_complete_independent_data_unavailable"
CONFIG_PATH = "examples/fisher_response_geometry_pullback_ewc_v1.json"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_output_directory(raw_output: str) -> Path:
    candidate = Path(raw_output)
    if candidate.is_symlink() or any(parent.is_symlink() for parent in candidate.parents):
        raise ValueError("output path must not traverse a symlink")
    resolved = candidate.resolve(strict=False)
    forbidden = {".local-runs", "data", "datasets", "corpus", "corpora", "calibration", "calibration_a", "calibration_b", ".git"}
    parts = {part.lower() for part in resolved.parts}
    if parts & forbidden or any(part.startswith(("corpus_", "corpus-", "calibration_", "calibration-")) for part in parts):
        raise ValueError("output path cannot be inside protected data or runtime paths")
    if resolved.exists():
        raise ValueError("output must be a fresh, nonexistent directory")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("inspect", help="authenticate configuration without tensor imports")
    synthetic = commands.add_parser("synthetic", help="execute the Pullback-Fisher + EWC synthetic chart ablation")
    synthetic.add_argument("--output", required=True, help="fresh output directory")
    commands.add_parser("model-facing", help="show the unchanged P0 execution block")
    args = parser.parse_args(argv)

    if args.mode == "model-facing":
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "status": P0_STATUS,
                    "model_facing_execution": "blocked",
                },
                sort_keys=True,
            )
        )
        return 2

    config_file = _ROOT / CONFIG_PATH
    config = json.loads(config_file.read_bytes())
    if args.mode == "inspect":
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "status": "configuration_authenticated",
                    "frozen_config": config,
                    "p0_status": P0_STATUS,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        output = _validate_output_directory(args.output)
        if str(_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(_ROOT / "src"))
        from fisher_graph.fisher_response_geometry_pullback_ewc import (
            evaluate_pullback_ewc_ablation,
            render_markdown,
        )

        report = evaluate_pullback_ewc_ablation(_ROOT)
        markdown = render_markdown(report)
        output.mkdir(parents=True, exist_ok=False)
        (output / "frozen_config.json").write_bytes(_canonical_bytes(config) + b"\n")
        (output / "evaluation.json").write_bytes(_canonical_bytes(report) + b"\n")
        (output / "evaluation.md").write_bytes(markdown.encode("utf-8"))
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "status": "synthetic_pullback_ewc_ablation_complete",
            "p0_status": P0_STATUS,
            "artifacts": [
                {
                    "path": name,
                    "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                    "bytes": (output / name).stat().st_size,
                }
                for name in ("frozen_config.json", "evaluation.json", "evaluation.md")
            ],
        }
        (output / "artifact_manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        print(
            json.dumps(
                {
                    "protocol_id": PROTOCOL_ID,
                    "status": manifest["status"],
                    "report": str(output / "evaluation.json"),
                    "markdown": str(output / "evaluation.md"),
                    "p0_status": P0_STATUS,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"pullback-ewc ablation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
