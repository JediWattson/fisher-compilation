"""Gated synthetic normalization-response CLI; no model-facing runner exists.

Inspect and model-facing rejection use the standard library only.  Direct
source loading deliberately avoids the eager ``fisher_graph`` package import.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence


_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_OUTPUT_COMPONENTS = {
    ".local-runs", ".git", ".codex", ".agents", "data", "datasets", "corpus",
    "corpora", "calibration", "guard",
}


def _load_source(name: str, relative: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, _ROOT / relative)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load synthetic implementation: {relative}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _protocol() -> ModuleType:
    return _load_source(
        "_fisher_response_geometry_protocol",
        "src/fisher_graph/fisher_response_geometry_protocol.py",
    )


def _new_output_directory(raw: str) -> Path:
    path = Path(raw).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("output must be a fresh, nonexistent directory")
    resolved = path.resolve(strict=False)
    components = {part.lower() for part in (*path.parts, *resolved.parts)}
    if components & _FORBIDDEN_OUTPUT_COMPONENTS or any(
        part.startswith(("corpus_", "corpus-", "calibration_", "calibration-"))
        for part in components
    ):
        raise ValueError("output cannot be placed in protected data or runtime paths")
    if not path.parent.is_dir():
        raise ValueError("output parent must already exist")
    # mkdir is exclusive: an existing directory or symlink is never reused.
    path.mkdir(mode=0o700)
    return path


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("inspect", help="inspect frozen metadata with stdlib only")
    synthetic = commands.add_parser("synthetic", help="run bounded generated fixtures")
    synthetic.add_argument("--output", required=True, help="fresh output directory")
    commands.add_parser("model-facing", help="show the preserved fail-closed P0 gate")
    args = parser.parse_args(argv)
    protocol = _protocol()
    if args.mode == "model-facing":
        # No source hash read, configuration read, tensor import, corpus lookup,
        # retirement call, model adapter, or loader precedes this rejection.
        print(json.dumps({
            "protocol_id": protocol.PROTOCOL_ID,
            "status": protocol.CURRENT_P0_STATUS,
            "model_facing_execution": "blocked",
            "reason": "This runner implements synthetic qualification only; P0 remains closed.",
        }, sort_keys=True))
        return 2
    try:
        config = protocol.load_frozen_config(_ROOT)
        sources = protocol.source_manifest(_ROOT)
        if args.mode == "inspect":
            print(json.dumps({
                "protocol_id": protocol.PROTOCOL_ID,
                "evidence_boundary": protocol.EVIDENCE_BOUNDARY,
                "config": config,
                "config_sha256": protocol.config_digest(config),
                "source_manifest": sources,
            }, indent=2, sort_keys=True))
            return 0
        output = _new_output_directory(args.output)
        kernel = _load_source(
            "_fisher_response_geometry_synthetic_kernel",
            "src/fisher_graph/fisher_response_geometry.py",
        )
        sections = kernel.run_synthetic_qualification(config)
        if sources != protocol.source_manifest(_ROOT):
            raise ValueError("a pinned source changed during synthetic execution")
        report = protocol.build_report(
            config=config, source_hashes=sources, sections=sections,
        )
        config_file = output / "frozen_config.json"
        report_file = output / "report.json"
        _write_exclusive(config_file, protocol.canonical_json_bytes(config) + b"\n")
        _write_exclusive(report_file, protocol.canonical_json_bytes(report) + b"\n")
        manifest = {
            "schema": "fisher_graph.fisher_response_geometry_probe.artifacts.v1",
            "protocol_id": protocol.PROTOCOL_ID,
            "evidence_class": "synthetic_qualification_only",
            "config_sha256": protocol.config_digest(config),
            "artifacts": [
                {"path": path.name, "sha256": protocol.sha256_file(path),
                 "bytes": path.stat().st_size}
                for path in (config_file, report_file)
            ],
            "source_manifest": sources,
        }
        _write_exclusive(
            output / "artifact_manifest.json",
            protocol.canonical_json_bytes(manifest) + b"\n",
        )
        failed_sections = [
            name for name, section in sections.items() if section["status"] == "failed"
        ]
        print(json.dumps({
            "status": report["status"],
            "evidence_class": "synthetic_qualification_only",
            "p0_status": protocol.CURRENT_P0_STATUS,
            "report": str(report_file),
            "numerical_qualification": "failed" if failed_sections else "passed",
            "failed_sections": failed_sections,
        }, sort_keys=True))
        return 1 if failed_sections else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"synthetic normalization-response probe: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
