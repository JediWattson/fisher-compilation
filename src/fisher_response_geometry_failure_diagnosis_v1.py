"""Separate, authenticated synthetic failure diagnosis with a closed model gate.

The original experiment is an immutable input.  Inspect and model-facing
rejection use the standard library only; no eager package import is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "fisher_response_geometry_failure_diagnosis_v1"
P0_STATUS = "development_complete_independent_data_unavailable"
REPLAY_SOURCE = "src/fisher_graph/fisher_response_geometry_replay.py"
DIAGNOSIS_SOURCE = "src/fisher_graph/fisher_response_geometry_failure_diagnosis.py"
ORIGINAL_SHIM = "src/fisher_response_geometry_probe_v1.py"
SOURCE_PATHS = (
    "src/fisher_response_geometry_failure_diagnosis_v1.py",
    DIAGNOSIS_SOURCE,
    REPLAY_SOURCE,
    "docs/fisher-response-geometry-failure-diagnosis-v1.md",
    "tests/test_fisher_response_geometry_failure_diagnosis.py",
    "tests/test_fisher_response_geometry_replay.py",
    "tests/test_fisher_response_geometry_failure_diagnosis_cli.py",
)
EVIDENCE_BOUNDARY = {
    "scope": "additional_diagnostics_on_already_examined_synthetic_fixture",
    "independent_confirmation": False,
    "target_model_loaded": False,
    "protected_data_opened": False,
    "tokenizer_access": False,
    "live_attention_executed": False,
    "charts_refitted": False,
    "router_retrained": False,
    "proposed_fix_implemented": False,
    "model_facing_scientific_claim": False,
    "deployment_or_compression_claim": False,
    "p0_status": P0_STATUS,
}
_INPUT_TENSORS = (
    "train_h", "train_m", "eval_h", "eval_m",
    "probe_directions", "state_probe_directions",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _source_bytes(relative: str) -> bytes:
    path = _ROOT / relative
    if path.is_symlink() or path.resolve(strict=False) != path or not path.is_file():
        raise ValueError(f"required source is missing or a symlink: {relative}")
    return path.read_bytes()


def _source_manifest() -> list[dict[str, Any]]:
    result = []
    for relative in SOURCE_PATHS:
        payload = _source_bytes(relative)
        result.append({
            "path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    return result


def _load_source(name: str, relative: str, expected_sha256: str) -> ModuleType:
    """Execute the exact hashed source bytes, without package or pycache imports."""
    source = _source_bytes(relative)
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError(f"source changed before import: {relative}")
    path = _ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise RuntimeError(f"cannot load diagnostic source: {relative}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _verify_unchanged(
    replay_module: ModuleType, original_identity: dict[str, Any],
    sources: list[dict[str, Any]],
) -> None:
    if _canonical(replay_module._authenticate(_ROOT)[2]) != _canonical(original_identity):
        raise ValueError("original baseline identity changed during diagnosis")
    if _source_manifest() != sources:
        raise ValueError("a diagnostic source changed during execution")


def _validate_report(report: Any, replay: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise ValueError("diagnosis must return an object")
    if (report.get("schema") != "fisher_graph.fisher_response_geometry_failure_diagnosis.v1"
        or report.get("status") != "synthetic_diagnostic_replay_complete"):
        raise ValueError("unknown diagnosis schema or completion status")
    if _canonical(report.get("evidence_boundary")) != _canonical(EVIDENCE_BOUNDARY):
        raise ValueError("diagnosis violates the unchanged synthetic evidence boundary")
    if _canonical(report.get("original_identity")) != _canonical(replay["original_identity"]):
        raise ValueError("diagnosis original identity differs from authenticated replay")
    if (replay["reproduction"].get("status") != "reproduced"
        or _canonical(report.get("reproduction")) != _canonical(replay["reproduction"])):
        raise ValueError("diagnosis does not preserve the successful reproduction evidence")
    if not isinstance(report.get("target_panels"), list) or not report["target_panels"]:
        raise ValueError("diagnosis must contain target comparison panels")
    if not isinstance(report.get("offset_and_group_diagnosis"), dict):
        raise ValueError("diagnosis must contain the offset and group measurements")
    # Refuse nonfinite measurements before publishing any successful artifact.
    _canonical(report)


def _replay_input_artifact(replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "fisher_graph.fisher_response_geometry_replayed_inputs.v1",
        "protocol_id": PROTOCOL_ID,
        "scope": "regenerated_original_synthetic_observations_only",
        "independent_confirmation": False,
        "new_training_or_evaluation_examples": False,
        "original_config_sha256": replay["original_identity"]["config_sha256"],
        "dtype": "float64",
        "tensors": {name: replay[name].detach().cpu().tolist() for name in _INPUT_TENSORS},
        "tensor_shapes": {name: list(replay[name].shape) for name in _INPUT_TENSORS},
        "training_family_ids": list(replay["train_ids"]),
        "evaluation_family_ids": list(replay["eval_ids"]),
        "reconstruction_policy": replay["reproduction"]["regenerated_items"],
    }


def _manifest(
    output: Path, filenames: Sequence[str], *, sources: list[dict[str, Any]],
    original_identity: dict[str, Any], command: list[str], status: str,
    torch_version: str | None,
) -> dict[str, Any]:
    artifacts = []
    for name in filenames:
        payload = (output / name).read_bytes()
        artifacts.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    return {
        "schema": "fisher_graph.fisher_response_geometry_failure_diagnosis.artifacts.v1",
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "original_identity": original_identity,
        "original_identity_verified_before_and_after_execution": True,
        "source_manifest": sources,
        "artifacts": artifacts,
        "command_argv": command,
        "working_directory": str(Path.cwd()),
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "torch_version": torch_version,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("inspect", help="authenticate frozen metadata without tensor imports")
    synthetic = commands.add_parser("synthetic", help="diagnose the fixed existing synthetic fixture")
    synthetic.add_argument("--output", required=True, help="fresh output directory")
    commands.add_parser("model-facing", help="show the preserved fail-closed P0 gate")
    args = parser.parse_args(argv)
    if args.mode == "model-facing":
        print(json.dumps({
            "protocol_id": PROTOCOL_ID, "status": P0_STATUS,
            "model_facing_execution": "blocked",
            "reason": "Only diagnosis of the existing synthetic fixture is implemented; P0 remains closed.",
        }, sort_keys=True))
        return 2
    command = list(sys.argv) if argv is None else [str(Path(__file__)), *argv]
    try:
        sources = _source_manifest()
        hashes = {entry["path"]: entry["sha256"] for entry in sources}
        replay_module = _load_source("_failure_diagnosis_replay", REPLAY_SOURCE, hashes[REPLAY_SOURCE])
        _, config, original_identity, _ = replay_module._authenticate(_ROOT)
        if args.mode == "inspect":
            print(json.dumps({
                "protocol_id": PROTOCOL_ID, "status": "original_metadata_authenticated",
                "evidence_boundary": EVIDENCE_BOUNDARY,
                "original_identity": original_identity,
                "frozen_config": config, "source_manifest": sources,
            }, indent=2, sort_keys=True))
            return 0
        original_hashes = {entry["path"]: entry["sha256"] for entry in original_identity["source_manifest"]}
        shim = _load_source("_failure_diagnosis_original_io", ORIGINAL_SHIM, original_hashes[ORIGINAL_SHIM])
        output = shim._new_output_directory(args.output)
        try:
            replay = replay_module.load_replay(_ROOT)
        except replay_module.ReplayReproductionError as error:
            _verify_unchanged(replay_module, original_identity, sources)
            failure = {
                "schema": "fisher_graph.fisher_response_geometry_failure_diagnosis.reproduction_failure.v1",
                "protocol_id": PROTOCOL_ID, "status": "baseline_reproduction_failed",
                "evidence_boundary": EVIDENCE_BOUNDARY,
                "original_identity": original_identity,
                "reproduction": error.reproduction,
                "comparisons_executed": False,
            }
            shim._write_exclusive(output / "failure.json", _canonical(failure) + b"\n")
            torch_module = sys.modules.get("torch")
            manifest = _manifest(
                output, ("failure.json",), sources=sources,
                original_identity=original_identity, command=command,
                status=failure["status"],
                torch_version=str(torch_module.__version__) if torch_module is not None else None,
            )
            shim._write_exclusive(output / "artifact_manifest.json", _canonical(manifest) + b"\n")
            print(json.dumps({"status": failure["status"], "failure": str(output / "failure.json"), "p0_status": P0_STATUS}, sort_keys=True))
            return 1
        if _canonical(replay["original_identity"]) != _canonical(original_identity):
            raise ValueError("replay original identity differs from initial authentication")
        diagnosis = _load_source("_failure_diagnosis_numeric", DIAGNOSIS_SOURCE, hashes[DIAGNOSIS_SOURCE])
        report = diagnosis.run_diagnosis(replay)
        _validate_report(report, replay)
        markdown = diagnosis.render_markdown(report)
        inputs = _replay_input_artifact(replay)
        _verify_unchanged(replay_module, original_identity, sources)
        payloads = {
            "diagnosis.json": _canonical(report) + b"\n",
            "diagnosis.md": markdown.encode("utf-8"),
            "replay_inputs.json": _canonical(inputs) + b"\n",
        }
        for name, payload in payloads.items():
            shim._write_exclusive(output / name, payload)
        manifest = _manifest(
            output, tuple(payloads), sources=sources,
            original_identity=original_identity, command=command,
            status=report["status"], torch_version=str(replay["kernel"].torch.__version__),
        )
        shim._write_exclusive(output / "artifact_manifest.json", _canonical(manifest) + b"\n")
        print(json.dumps({
            "protocol_id": PROTOCOL_ID, "status": report["status"],
            "report": str(output / "diagnosis.json"),
            "markdown": str(output / "diagnosis.md"),
            "p0_status": P0_STATUS,
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        print(f"synthetic failure diagnosis: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
