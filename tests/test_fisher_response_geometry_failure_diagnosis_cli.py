"""Isolated CLI guards for additional diagnosis of the sealed synthetic fixture."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src/fisher_response_geometry_failure_diagnosis_v1.py"
ORIGINAL_DIRECTORY = ROOT / "artifacts/research/fisher-response-geometry-probe-v1-synthetic"
ORIGINAL_FILES = tuple(ORIGINAL_DIRECTORY / name for name in (
    "report.json", "frozen_config.json", "artifact_manifest.json",
))


def _guarded_run(*arguments: str, tensors_allowed: bool = False) -> subprocess.CompletedProcess[str]:
    guard = textwrap.dedent(
        """
        import importlib.abc
        import os
        from pathlib import Path
        import runpy
        import sys

        tensors_allowed = sys.argv[1] == "yes"
        runner = sys.argv[2]
        arguments = sys.argv[3:]
        root = Path(runner).resolve().parents[1]
        sys.dont_write_bytecode = True

        class ForbiddenImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = {"fisher_graph", "transformers", "adapters", "mlx",
                             "safetensors", "sentencepiece", "tokenizers"}
                if not tensors_allowed:
                    forbidden.update({"torch", "numpy"})
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("forbidden import: " + fullname)

        def audit(event, arguments):
            if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(arguments[0]))
            parts = {part.lower() for part in path.parts}
            if parts & {".local-runs", "corpus", "corpora", "calibration"}:
                raise AssertionError("protected data access: " + str(path))
            if any(part.startswith(("corpus_", "corpus-", "calibration_", "calibration-"))
                   for part in parts):
                raise AssertionError("protected corpus access: " + str(path))
            if path.suffix.lower() in {".pt", ".safetensors", ".bin"}:
                raise AssertionError("model/tensor file access: " + str(path))
            installed_module_source = (
                ".venv" in parts and "site-packages" in parts
                and path.suffix.lower() in {".py", ".pyc", ".so", ".pyd"}
            )
            if path.is_relative_to(root) and not installed_module_source and parts & {"data", "datasets"}:
                raise AssertionError("repository data access: " + str(path))
            if mode == "model-facing" and (
                "fisher-response-geometry-probe-v1-synthetic" in parts
                or path.name == "fisher_response_geometry_replay.py"
            ):
                raise AssertionError("model-facing gate read replay inputs: " + str(path))

        mode = arguments[0] if arguments else None
        sys.meta_path.insert(0, ForbiddenImport())
        sys.addaudithook(audit)
        sys.argv = [runner, *arguments]
        runpy.run_path(runner, run_name="__main__")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", guard, "yes" if tensors_allowed else "no",
         str(RUNNER), *arguments],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=60,
    )


def _runner() -> dict[str, object]:
    return runpy.run_path(str(RUNNER))


def test_inspect_authenticates_original_metadata_without_tensor_or_package_imports() -> None:
    before = {path: path.read_bytes() for path in ORIGINAL_FILES}
    result = _guarded_run("inspect")
    assert result.returncode == 0, result.stderr
    notice = json.loads(result.stdout)
    assert notice["protocol_id"] == "fisher_response_geometry_failure_diagnosis_v1"
    assert notice["status"] == "original_metadata_authenticated"
    assert notice["evidence_boundary"]["p0_status"] == "development_complete_independent_data_unavailable"
    assert notice["evidence_boundary"]["independent_confirmation"] is False
    assert len(notice["source_manifest"]) == 7
    for entry in notice["original_identity"]["artifacts"]:
        payload = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_model_facing_is_blocked_before_replay_reads_or_tensor_imports() -> None:
    result = _guarded_run("model-facing")
    assert result.returncode == 2, result.stderr
    notice = json.loads(result.stdout)
    assert notice["protocol_id"] == "fisher_response_geometry_failure_diagnosis_v1"
    assert notice["status"] == "development_complete_independent_data_unavailable"
    assert notice["model_facing_execution"] == "blocked"


@pytest.mark.parametrize("arguments", [
    ("synthetic", "--output", "unused", "--model", "anything"),
    ("synthetic", "--output", "unused", "--data", "anything"),
    ("synthetic", "--output", "unused", "--config", "anything"),
    ("synthetic", "--output", "unused", "--input", "anything"),
    ("synthetic", "--output", "unused", "--refit"),
    ("model-facing", "--authorize"),
])
def test_no_model_data_override_or_refit_flags_exist(arguments: tuple[str, ...]) -> None:
    result = _guarded_run(*arguments)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_existing_output_is_rejected_before_tensor_import_and_not_modified(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve this", encoding="utf-8")
    result = _guarded_run("synthetic", "--output", str(existing))
    assert result.returncode == 1, result.stderr
    assert "fresh, nonexistent directory" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve this"
    assert list(existing.iterdir()) == [sentinel]


def test_output_symlink_is_rejected_before_tensor_import(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(target, target_is_directory=True)
    result = _guarded_run("synthetic", "--output", str(link))
    assert result.returncode == 1, result.stderr
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("component", [".local-runs", "data", "corpus", "calibration_a", ".git"])
def test_protected_output_path_is_rejected_without_tensor_import(tmp_path: Path, component: str) -> None:
    parent = tmp_path / component
    parent.mkdir()
    output = parent / "fresh"
    result = _guarded_run("synthetic", "--output", str(output))
    assert result.returncode == 1, result.stderr
    assert "protected data or runtime paths" in result.stderr
    assert not output.exists()


def test_full_diagnosis_under_guards_emits_bound_artifacts_and_preserves_originals(tmp_path: Path) -> None:
    before = {path: path.read_bytes() for path in ORIGINAL_FILES}
    output = tmp_path / "failure-diagnosis"
    result = _guarded_run("synthetic", "--output", str(output), tensors_allowed=True)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {
        "diagnosis.json", "diagnosis.md", "replay_inputs.json", "artifact_manifest.json",
    }
    report = json.loads((output / "diagnosis.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    inputs = json.loads((output / "replay_inputs.json").read_text(encoding="utf-8"))
    assert report["status"] == "synthetic_diagnostic_replay_complete"
    assert report["reproduction"]["status"] == "reproduced"
    assert report["evidence_boundary"]["charts_refitted"] is False
    assert report["evidence_boundary"]["independent_confirmation"] is False
    assert len(report["evaluation_row_ids"]) == 48
    assert len(report["target_panels"]) == 5
    assert manifest["original_identity"] == report["original_identity"]
    assert manifest["original_identity_verified_before_and_after_execution"] is True
    assert manifest["command_argv"] == [str(RUNNER), "synthetic", "--output", str(output)]
    assert manifest["environment"]["torch_version"]
    for entry in manifest["artifacts"]:
        payload = (output / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["bytes"]
    for entry in manifest["source_manifest"]:
        payload = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["bytes"]
    assert inputs["new_training_or_evaluation_examples"] is False
    assert inputs["tensor_shapes"]["train_h"] == [96, 6]
    assert inputs["tensor_shapes"]["eval_m"] == [48, 8]
    assert inputs["tensor_shapes"]["probe_directions"] == [48, 8]
    assert set(inputs["tensors"]) == {
        "train_h", "train_m", "eval_h", "eval_m", "probe_directions", "state_probe_directions",
    }
    assert set(inputs["training_family_ids"]).isdisjoint(inputs["evaluation_family_ids"])
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_reproduction_discrepancy_publishes_failure_evidence_without_running_diagnosis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _runner()
    namespace = runner["main"].__globals__
    original_loader = namespace["_load_source"]
    discrepancy = {"status": "discrepancy", "metric_comparisons": [{
        "saved": 1.0, "replayed": 2.0, "within_tolerance": False,
    }]}
    imported = []

    def injected_loader(name: str, relative: str, digest: str):
        imported.append(relative)
        module = original_loader(name, relative, digest)
        if relative == namespace["REPLAY_SOURCE"]:
            def fail(_root: Path):
                raise module.ReplayReproductionError(copy.deepcopy(discrepancy))
            module.load_replay = fail
        return module

    monkeypatch.setitem(namespace, "_load_source", injected_loader)
    output = tmp_path / "reproduction-failed"
    assert runner["main"](["synthetic", "--output", str(output)]) == 1
    notice = json.loads(capsys.readouterr().out)
    assert notice["status"] == "baseline_reproduction_failed"
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["reproduction"] == discrepancy
    assert failure["comparisons_executed"] is False
    assert namespace["DIAGNOSIS_SOURCE"] not in imported
    assert {path.name for path in output.iterdir()} == {"failure.json", "artifact_manifest.json"}
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["path"] == "failure.json"
    assert manifest["original_identity_verified_before_and_after_execution"] is True


def test_report_validation_rejects_claim_changes_identity_drift_and_nonfinite_errors() -> None:
    runner = _runner()
    replay = {"original_identity": {"sha256": "saved"}, "reproduction": {"status": "reproduced"}}
    report = {
        "schema": "fisher_graph.fisher_response_geometry_failure_diagnosis.v1",
        "status": "synthetic_diagnostic_replay_complete",
        "evidence_boundary": copy.deepcopy(runner["EVIDENCE_BOUNDARY"]),
        **copy.deepcopy(replay), "target_panels": [{"synthetic_error": 1.0}],
        "offset_and_group_diagnosis": {},
    }
    validate = runner["_validate_report"]
    validate(report, replay)
    for field in ("charts_refitted", "independent_confirmation", "target_model_loaded", "proposed_fix_implemented"):
        changed = copy.deepcopy(report)
        changed["evidence_boundary"][field] = True
        with pytest.raises(ValueError, match="evidence boundary"):
            validate(changed, replay)
    changed = copy.deepcopy(report)
    changed["original_identity"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="original identity"):
        validate(changed, replay)
    changed = copy.deepcopy(report)
    changed["target_panels"][0]["synthetic_error"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float"):
        validate(changed, replay)
