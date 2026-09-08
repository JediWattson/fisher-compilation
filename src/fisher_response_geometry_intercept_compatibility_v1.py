"""One frozen synthetic intercept-compatibility ablation; model execution is closed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
from typing import Any, Sequence


_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "fisher_response_geometry_intercept_compatibility_v1"
P0_STATUS = "development_complete_independent_data_unavailable"
PROTOCOL_SOURCE = "src/fisher_graph/fisher_response_geometry_intercept_compatibility_protocol.py"
SOURCE_PATHS = (
    "src/fisher_response_geometry_intercept_compatibility_v1.py",
    PROTOCOL_SOURCE,
    "src/fisher_graph/fisher_response_geometry_intercept_compatibility.py",
    "src/fisher_graph/fisher_response_geometry_intercept_compatibility_evaluation.py",
    "examples/fisher_response_geometry_intercept_compatibility_v1.json",
    "docs/fisher-response-geometry-intercept-compatibility-v1.md",
    "tests/test_fisher_response_geometry_intercept_compatibility.py",
    "tests/test_fisher_response_geometry_intercept_compatibility_evaluation.py",
    "tests/test_fisher_response_geometry_intercept_compatibility_cli.py",
)
FREEZE_FILES = (
    "frozen_config.json", "lambda_zero_control.json", "candidate.json", "block_scales.json",
)


class AblationFailure(ValueError):
    def __init__(self, reason: str, details: object = None) -> None:
        super().__init__(reason)
        self.details = details


def _protocol() -> ModuleType:
    path = _ROOT / PROTOCOL_SOURCE
    if path.is_symlink() or path.resolve(strict=False) != path:
        raise ValueError("protocol source must not be a symlink")
    source = path.read_bytes()
    specification = importlib.util.spec_from_file_location("_intercept_compatibility_protocol", path)
    if specification is None:
        raise RuntimeError("cannot load ablation protocol")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    module._loaded_source_sha256 = hashlib.sha256(source).hexdigest()
    return module


def _artifact_rows(protocol: ModuleType, output: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        payload = protocol.fixed_bytes(output, name)
        rows.append({"path": name, "sha256": protocol.digest(payload), "bytes": len(payload)})
    return rows


def _training_view(replay: dict[str, Any]) -> MappingProxyType:
    """The fitter receives no evaluation rows, probes, labels or cached targets."""
    return MappingProxyType({
        name: replay[name]
        for name in ("config", "kernel", "tensors", "saved_artifact", "train_h", "train_m")
    })


def _replay_snapshot(protocol: ModuleType, replay: dict[str, Any]) -> str:
    tensors = {name: value.detach().cpu().tolist() for name, value in replay["tensors"].items()}
    inputs = {
        name: replay[name].detach().cpu().tolist()
        for name in ("train_h", "train_m", "eval_h", "eval_m", "probe_directions", "state_probe_directions")
    }
    banks = {
        name: [{
            **{field: getattr(chart, field).detach().cpu().tolist() for field in (
                "state_anchor", "message_anchor", "beta", "B", "A", "U", "Q", "G",
            )},
            "state_radius": chart.state_radius, "message_radius": chart.message_radius,
        } for chart in charts] for name, charts in replay["banks"].items()
    }
    return protocol.digest(protocol.canonical_bytes({
        "config": replay["config"], "saved_artifact": replay["saved_artifact"],
        "tensors": tensors, "inputs": inputs, "banks": banks,
        "train_ids": replay["train_ids"], "eval_ids": replay["eval_ids"],
        "router_centers": {name: value.tolist() for name, value in replay["router_centers"].items()},
    }))


def _verify_saved_observations(protocol: ModuleType, replay: dict[str, Any], saved: dict[str, Any]) -> None:
    for name, values in saved["tensors"].items():
        if protocol.canonical_bytes(replay[name].detach().cpu().tolist()) != protocol.canonical_bytes(values):
            raise AblationFailure("replayed observations differ from the sealed diagnosis inputs", {"tensor": name})
    if (list(replay["train_ids"]) != saved["training_family_ids"]
        or list(replay["eval_ids"]) != saved["evaluation_family_ids"]):
        raise AblationFailure("replayed family identities differ from the sealed diagnosis inputs")


def _check_unchanged(
    protocol: ModuleType, baseline_identity: dict[str, Any], sources: list[dict[str, Any]],
    replay: dict[str, Any] | None = None, replay_snapshot: str | None = None,
) -> None:
    current = protocol.authenticate_baselines(_ROOT)["identity"]
    if protocol.canonical_bytes(current) != protocol.canonical_bytes(baseline_identity):
        raise AblationFailure("sealed baseline identities changed during the ablation")
    if protocol.source_manifest(_ROOT, SOURCE_PATHS) != sources:
        raise AblationFailure("a new ablation source changed during execution")
    if replay is not None and _replay_snapshot(protocol, replay) != replay_snapshot:
        raise AblationFailure("the fixed replay inputs, geometry or control banks were mutated")


def _reload_frozen_candidate(
    protocol: ModuleType, output: Path, expected_freeze_sha256: str,
    fit_module: ModuleType, replay: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate and load the saved candidate; this runs before evaluator import."""
    freeze_bytes = protocol.fixed_bytes(output, "freeze_manifest.json")
    if protocol.digest(freeze_bytes) != expected_freeze_sha256:
        raise AblationFailure("freeze manifest changed before evaluation")
    freeze = json.loads(freeze_bytes)
    if tuple(entry["path"] for entry in freeze["artifacts"]) != FREEZE_FILES:
        raise AblationFailure("freeze inventory differs from the declared four artifacts")
    payloads = {}
    for entry in freeze["artifacts"]:
        payload = protocol.fixed_bytes(output, entry["path"])
        if protocol.digest(payload) != entry["sha256"] or len(payload) != entry["bytes"]:
            raise AblationFailure("a frozen artifact changed before evaluation", {"path": entry["path"]})
        payloads[entry["path"]] = json.loads(payload)
    config = protocol.validate_config(payloads["frozen_config.json"])
    control_result = payloads["lambda_zero_control.json"]["result"]
    if control_result.get("status") != "reproduced":
        raise AblationFailure("lambda-zero control did not pass before candidate evaluation", control_result)
    candidate_artifact = payloads["candidate.json"]
    if candidate_artifact["compatibility_weight"] != config["candidate_lambda"]:
        raise AblationFailure("frozen candidate is not the sole lambda-one arm")
    scales = payloads["block_scales.json"]
    if protocol.canonical_bytes(scales["blocks"]) != protocol.canonical_bytes(candidate_artifact["blocks"]):
        raise AblationFailure("frozen block scales differ from the serialized candidate")
    if scales["candidate_artifact_sha256"] != candidate_artifact["artifact_sha256"]:
        raise AblationFailure("block scales are bound to a different candidate")
    return fit_module.load_arm(replay, candidate_artifact), config, control_result, freeze


def _validate_evaluation(
    protocol: ModuleType, report: Any, replay: dict[str, Any], control_result: dict[str, Any],
) -> None:
    if not isinstance(report, dict):
        raise AblationFailure("evaluation did not return an object")
    if (report.get("schema") != "fisher_graph.fisher_response_geometry_intercept_compatibility.v1"
        or report.get("status") != "synthetic_development_ablation_complete"):
        raise AblationFailure("unknown ablation evaluation schema/status")
    if protocol.canonical_bytes(report.get("evidence_boundary")) != protocol.canonical_bytes(protocol.EVIDENCE_BOUNDARY):
        raise AblationFailure("evaluation violates the single synthetic ablation boundary")
    if protocol.canonical_bytes(report.get("lambda_zero_control")) != protocol.canonical_bytes(control_result):
        raise AblationFailure("evaluation does not preserve the rebuilt-control evidence")
    if protocol.canonical_bytes(report.get("original_reproduction")) != protocol.canonical_bytes(replay["reproduction"]):
        raise AblationFailure("evaluation does not preserve original reproduction evidence")
    protocol.canonical_bytes(report)


def _write_manifest(
    protocol: ModuleType, io: ModuleType, output: Path, *, status: str,
    sources: list[dict[str, Any]], baselines: dict[str, Any], command: list[str],
    evaluation_started: bool, verified_unchanged: bool,
) -> None:
    names = tuple(name for name in (
        *FREEZE_FILES, "freeze_manifest.json", "evaluation.json", "evaluation.md", "failure.json",
    ) if (output / name).is_file())
    torch_module = sys.modules.get("torch")
    manifest = {
        "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.artifacts.v1",
        "protocol_id": PROTOCOL_ID, "status": status,
        "authenticated_baselines": baselines, "source_manifest": sources,
        "sealed_inputs_verified_unchanged": verified_unchanged,
        "candidate_evaluation_started": evaluation_started,
        "artifacts": _artifact_rows(protocol, output, names),
        "command_argv": command, "working_directory": str(Path.cwd()),
        "environment": {
            "python_version": sys.version, "python_executable": sys.executable,
            "torch_version": str(torch_module.__version__) if torch_module is not None else None,
        },
    }
    io._write_exclusive(output / "artifact_manifest.json", protocol.canonical_bytes(manifest) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("inspect", help="authenticate frozen metadata without tensor imports")
    synthetic = commands.add_parser("synthetic", help="execute the single frozen development ablation")
    synthetic.add_argument("--output", required=True, help="fresh output directory")
    commands.add_parser("model-facing", help="show the unchanged P0 execution block")
    args = parser.parse_args(argv)
    if args.mode == "model-facing":
        print(json.dumps({"protocol_id": PROTOCOL_ID, "status": P0_STATUS,
                          "model_facing_execution": "blocked"}, sort_keys=True))
        return 2
    command = list(sys.argv) if argv is None else [str(Path(__file__)), *argv]
    protocol = io = output = replay = None
    sources: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}
    stage, evaluation_started = "preflight", False
    try:
        protocol = _protocol()
        config = protocol.load_config(_ROOT)
        sources = protocol.source_manifest(_ROOT, SOURCE_PATHS)
        hashes = {entry["path"]: entry["sha256"] for entry in sources}
        if hashes[PROTOCOL_SOURCE] != protocol._loaded_source_sha256:
            raise AblationFailure("protocol source changed during bootstrap")
        authenticated = protocol.authenticate_baselines(_ROOT)
        baselines = authenticated["identity"]
        if args.mode == "inspect":
            print(json.dumps({
                "protocol_id": PROTOCOL_ID, "status": "both_sealed_baselines_authenticated",
                "frozen_config": config, "evidence_boundary": protocol.EVIDENCE_BOUNDARY,
                "authenticated_baselines": baselines, "source_manifest": sources,
            }, indent=2, sort_keys=True))
            return 0
        old_hashes = {entry["path"]: entry["sha256"] for entry in baselines["original_probe"]["source_manifest"]}
        io = protocol.load_source(_ROOT, "_intercept_compatibility_original_io", protocol.ORIGINAL_IO_SOURCE, old_hashes[protocol.ORIGINAL_IO_SOURCE])
        output = io._new_output_directory(args.output)
        io._write_exclusive(output / "frozen_config.json", protocol.canonical_bytes(config) + b"\n")
        stage = "original_replay"
        replay = authenticated["replay_module"].load_replay(_ROOT)
        _verify_saved_observations(protocol, replay, authenticated["saved_replay_inputs"])
        snapshot = _replay_snapshot(protocol, replay)
        diagnosis_module = protocol.load_source(
            _ROOT, "_intercept_compatibility_old_diagnosis", protocol.OLD_DIAGNOSIS_SOURCE,
            authenticated["diagnosis_source_hashes"][protocol.OLD_DIAGNOSIS_SOURCE],
        )
        fit_module = protocol.load_source(_ROOT, "_intercept_compatibility_fit", protocol.FIT_SOURCE, hashes[protocol.FIT_SOURCE])
        stage = "lambda_zero_rebuild"
        control = fit_module.fit_arm(_training_view(replay), compatibility_weight=int(config["control_lambda"]))
        control_result = fit_module.verify_lambda_zero(replay, control, diagnosis_module)
        control_artifact = {
            "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.control.v1",
            "result": control_result, "rebuilt_arm": fit_module.serialize_arm(control),
        }
        io._write_exclusive(output / "lambda_zero_control.json", protocol.canonical_bytes(control_artifact) + b"\n")
        if control_result.get("status") != "reproduced":
            raise AblationFailure("lambda-zero rebuilt control failed baseline equivalence", control_result)
        _check_unchanged(protocol, baselines, sources, replay, snapshot)
        stage = "candidate_training"
        candidate = fit_module.fit_arm(_training_view(replay), compatibility_weight=int(config["candidate_lambda"]))
        candidate_artifact = fit_module.serialize_arm(candidate)
        io._write_exclusive(output / "candidate.json", protocol.canonical_bytes(candidate_artifact) + b"\n")
        block_scales = {
            "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.block_scales.v1",
            "candidate_artifact_sha256": candidate_artifact["artifact_sha256"],
            "blocks": candidate_artifact["blocks"],
        }
        io._write_exclusive(output / "block_scales.json", protocol.canonical_bytes(block_scales) + b"\n")
        _check_unchanged(protocol, baselines, sources, replay, snapshot)
        stage = "candidate_freeze"
        freeze = {
            "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.freeze.v1",
            "protocol_id": PROTOCOL_ID,
            "lambda_zero_status": control_result["status"],
            "candidate_evaluation_started": False,
            "candidate_lambda": config["candidate_lambda"],
            "authenticated_baselines": baselines, "source_manifest": sources,
            "fixed_replay_snapshot_sha256": snapshot,
            "artifacts": _artifact_rows(protocol, output, FREEZE_FILES),
        }
        freeze_bytes = protocol.canonical_bytes(freeze) + b"\n"
        io._write_exclusive(output / "freeze_manifest.json", freeze_bytes)
        freeze_sha256 = protocol.digest(freeze_bytes)
        stage = "frozen_candidate_reload"
        loaded_candidate, loaded_config, loaded_control, loaded_freeze = _reload_frozen_candidate(
            protocol, output, freeze_sha256, fit_module, replay,
        )
        evaluation_module = protocol.load_source(
            _ROOT, "_intercept_compatibility_evaluation", protocol.EVALUATION_SOURCE,
            hashes[protocol.EVALUATION_SOURCE],
        )
        stage, evaluation_started = "candidate_evaluation", True
        report = evaluation_module.run_evaluation(
            replay=replay, candidate=loaded_candidate,
            old_diagnosis=authenticated["old_diagnosis"], diagnosis_module=diagnosis_module,
            fit_module=fit_module, config=loaded_config, control_result=loaded_control,
        )
        _validate_evaluation(protocol, report, replay, loaded_control)
        report["authenticated_baselines"] = baselines
        report["freeze_identity"] = {
            "path": "freeze_manifest.json", "sha256": freeze_sha256,
            "artifacts": loaded_freeze["artifacts"],
        }
        report["new_source_manifest"] = sources
        markdown = evaluation_module.render_markdown(report)
        _check_unchanged(protocol, baselines, sources, replay, snapshot)
        _reload_frozen_candidate(protocol, output, freeze_sha256, fit_module, replay)
        stage = "publication"
        io._write_exclusive(output / "evaluation.json", protocol.canonical_bytes(report) + b"\n")
        io._write_exclusive(output / "evaluation.md", markdown.encode("utf-8"))
        _write_manifest(protocol, io, output, status=report["status"], sources=sources,
                        baselines=baselines, command=command, evaluation_started=True,
                        verified_unchanged=True)
        print(json.dumps({"protocol_id": PROTOCOL_ID, "status": report["status"],
                          "report": str(output / "evaluation.json"),
                          "markdown": str(output / "evaluation.md"), "p0_status": P0_STATUS}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        if protocol is not None and io is not None and output is not None:
            failure = {
                "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.failure.v1",
                "protocol_id": PROTOCOL_ID, "status": "ablation_execution_failed",
                "stage": stage, "reason": str(error),
                "details": getattr(error, "details", getattr(error, "reproduction", None)),
                "candidate_evaluation_started": evaluation_started,
                "authenticated_baselines": baselines,
                "p0_status": P0_STATUS,
                "performance_conclusion": "unavailable_due_to_execution_failure",
            }
            try:
                io._write_exclusive(output / "failure.json", protocol.canonical_bytes(failure) + b"\n")
                verified = protocol.canonical_bytes(protocol.authenticate_baselines(_ROOT)["identity"]) == protocol.canonical_bytes(baselines)
                if not (output / "artifact_manifest.json").exists():
                    _write_manifest(protocol, io, output, status=failure["status"], sources=sources,
                                    baselines=baselines, command=command,
                                    evaluation_started=evaluation_started, verified_unchanged=verified)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as publication_error:
                print(f"failure artifact publication: {publication_error}", file=sys.stderr)
        print(f"synthetic intercept-compatibility ablation ({stage}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
