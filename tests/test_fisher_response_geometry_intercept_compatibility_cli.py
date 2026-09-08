"""Boundary, immutable-input and freeze-order checks for the one-arm ablation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src/fisher_response_geometry_intercept_compatibility_v1.py"


def _runner() -> dict[str, object]:
    return runpy.run_path(str(RUNNER))


def _protocol():
    return _runner()["_protocol"]()


def _guarded_run(*arguments: str, tensors_allowed: bool = False) -> subprocess.CompletedProcess[str]:
    guard = textwrap.dedent(
        """
        import hashlib
        import importlib.abc
        import json
        import os
        from pathlib import Path
        import runpy
        import sys

        tensors_allowed = sys.argv[1] == "yes"
        runner, arguments = sys.argv[2], sys.argv[3:]
        root = Path(runner).resolve().parents[1]
        mode = arguments[0] if arguments else None
        output = Path(arguments[arguments.index("--output") + 1]) if "--output" in arguments else None
        sys.dont_write_bytecode = True
        candidate_reloaded = False

        class ForbiddenImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = {"fisher_graph", "transformers", "adapters", "mlx",
                             "safetensors", "sentencepiece", "tokenizers"}
                if not tensors_allowed:
                    forbidden.update({"torch", "numpy"})
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("forbidden import: " + fullname)

        def audit(event, values):
            if event == "exec" and Path(values[0].co_filename).name == "fisher_response_geometry_intercept_compatibility_evaluation.py":
                if not candidate_reloaded:
                    raise AssertionError("candidate evaluator imported before authenticated reload")
                freeze = json.loads((output / "freeze_manifest.json").read_bytes())
                if freeze["candidate_evaluation_started"] is not False:
                    raise AssertionError("candidate scored before freeze")
                for entry in freeze["artifacts"]:
                    payload = (output / entry["path"]).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != entry["sha256"] or len(payload) != entry["bytes"]:
                        raise AssertionError("frozen bytes mismatch before evaluator import")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0]))
            parts = {part.lower() for part in path.parts}
            if parts & {".local-runs", "corpus", "corpora", "calibration"}:
                raise AssertionError("protected data access: " + str(path))
            if any(part.startswith(("corpus_", "corpus-", "calibration_", "calibration-")) for part in parts):
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
                or "fisher-response-geometry-failure-diagnosis-v1" in parts
                or path.name in {"fisher_response_geometry_replay.py", "fisher_response_geometry_intercept_compatibility_protocol.py"}
            ):
                raise AssertionError("model-facing gate opened research inputs: " + str(path))

        sys.meta_path.insert(0, ForbiddenImport())
        sys.addaudithook(audit)
        sys.argv = [runner, *arguments]
        loaded = runpy.run_path(runner)
        namespace = loaded["main"].__globals__
        original_reload = namespace["_reload_frozen_candidate"]
        def reload_then_allow_evaluator(*values, **kwargs):
            global candidate_reloaded
            result = original_reload(*values, **kwargs)
            candidate_reloaded = True
            return result
        namespace["_reload_frozen_candidate"] = reload_then_allow_evaluator
        raise SystemExit(loaded["main"](arguments))
        """
    )
    return subprocess.run(
        [sys.executable, "-c", guard, "yes" if tensors_allowed else "no", str(RUNNER), *arguments],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=90,
    )


def _sealed_files(protocol) -> tuple[Path, ...]:
    identity = protocol.authenticate_baselines(ROOT)["identity"]
    paths = {entry["path"] for group in ("original_probe", "diagnosis")
             for inventory in ("artifacts", "source_manifest")
             for entry in identity[group][inventory]}
    return tuple(ROOT / relative for relative in sorted(paths))


def test_inspect_authenticates_both_sealed_inputs_without_tensor_or_package_imports() -> None:
    result = _guarded_run("inspect")
    assert result.returncode == 0, result.stderr
    notice = json.loads(result.stdout)
    assert notice["status"] == "both_sealed_baselines_authenticated"
    assert notice["frozen_config"]["candidate_lambda"] == 1.0
    assert notice["frozen_config"]["control_lambda"] == 0.0
    assert len(notice["source_manifest"]) == 9
    assert notice["evidence_boundary"] == _protocol().EVIDENCE_BOUNDARY
    for group in ("original_probe", "diagnosis"):
        for inventory in ("artifacts", "source_manifest"):
            for entry in notice["authenticated_baselines"][group][inventory]:
                payload = (ROOT / entry["path"]).read_bytes()
                assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_model_facing_gate_blocks_before_research_reads_or_tensor_imports() -> None:
    result = _guarded_run("model-facing")
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout) == {
        "protocol_id": "fisher_response_geometry_intercept_compatibility_v1",
        "status": "development_complete_independent_data_unavailable",
        "model_facing_execution": "blocked",
    }


@pytest.mark.parametrize("flag", ["--model", "--data", "--config", "--input", "--lambda", "--seed", "--charts", "--rank"])
def test_no_override_or_model_data_flags_exist(flag: str) -> None:
    result = _guarded_run("synthetic", "--output", "unused", flag, "anything")
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_model_execution_cannot_be_authorized_through_a_cli_flag() -> None:
    result = _guarded_run("model-facing", "--authorize")
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_existing_output_is_preserved_and_rejected_before_tensor_import(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve this", encoding="utf-8")
    result = _guarded_run("synthetic", "--output", str(output))
    assert result.returncode == 1, result.stderr
    assert "fresh, nonexistent directory" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve this"
    assert list(output.iterdir()) == [sentinel]


def test_output_symlink_is_rejected_before_tensor_import(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(target, target_is_directory=True)
    result = _guarded_run("synthetic", "--output", str(link))
    assert result.returncode == 1, result.stderr
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("component", [".local-runs", "data", "corpus", "calibration_a", ".git"])
def test_protected_output_is_rejected_before_tensor_import(tmp_path: Path, component: str) -> None:
    parent = tmp_path / component
    parent.mkdir()
    output = parent / "fresh"
    result = _guarded_run("synthetic", "--output", str(output))
    assert result.returncode == 1, result.stderr
    assert "protected data or runtime paths" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("relative", [
    "artifacts/research/fisher-response-geometry-failure-diagnosis-v1/diagnosis.json",
    "artifacts/research/fisher-response-geometry-probe-v1-synthetic/frozen_config.json",
    "src/fisher_graph/fisher_response_geometry_failure_diagnosis.py",
    "src/fisher_graph/fisher_response_geometry.py",
])
def test_sealed_artifact_or_source_corruption_is_rejected(tmp_path: Path, relative: str) -> None:
    protocol = _protocol()
    for source in _sealed_files(protocol):
        destination = tmp_path / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="mismatch|differ"):
        protocol.authenticate_baselines(tmp_path)


@pytest.mark.parametrize("key,value", [
    ("candidate_lambda", 0.5), ("control_lambda", 1.0), ("seed", 1730),
    ("chart_count", 4), ("relative_constant_tolerance", 1e-8), ("finite_radii", [0.01]),
])
def test_frozen_config_has_no_tuning_surface(key: str, value: object) -> None:
    protocol = _protocol()
    changed = copy.deepcopy(protocol.FROZEN_CONFIG)
    changed[key] = value
    with pytest.raises(ValueError, match="single frozen ablation"):
        protocol.validate_config(changed)


def test_training_view_exposes_only_the_six_training_inputs_and_is_read_only() -> None:
    names = ("config", "kernel", "tensors", "saved_artifact", "train_h", "train_m")
    replay = {name: object() for name in (*names, "eval_h", "eval_m", "eval_ids", "probe_directions")}
    view = _runner()["_training_view"](replay)
    assert tuple(view) == names
    assert all(view[name] is replay[name] for name in names)
    with pytest.raises(TypeError):
        view["eval_h"] = replay["eval_h"]


def _freeze_fixture(output: Path, runner: dict[str, object]):
    protocol = runner["_protocol"]()
    output.mkdir()
    candidate = {"compatibility_weight": 1, "blocks": {"training_only": True}, "artifact_sha256": "example"}
    values = {
        "frozen_config.json": protocol.FROZEN_CONFIG,
        "lambda_zero_control.json": {"result": {"status": "reproduced"}},
        "candidate.json": candidate,
        "block_scales.json": {"blocks": candidate["blocks"], "candidate_artifact_sha256": "example"},
    }
    for name, value in values.items():
        (output / name).write_bytes(protocol.canonical_bytes(value) + b"\n")
    freeze = {"artifacts": runner["_artifact_rows"](protocol, output, runner["FREEZE_FILES"])}
    payload = protocol.canonical_bytes(freeze) + b"\n"
    (output / "freeze_manifest.json").write_bytes(payload)
    return protocol, protocol.digest(payload), candidate


@pytest.mark.parametrize("name", ["freeze_manifest.json", "candidate.json", "frozen_config.json", "lambda_zero_control.json", "block_scales.json"])
def test_frozen_byte_corruption_stops_before_candidate_load(tmp_path: Path, name: str) -> None:
    runner = _runner()
    output = tmp_path / "frozen"
    protocol, freeze_hash, _ = _freeze_fixture(output, runner)
    target = output / name
    target.write_bytes(target.read_bytes() + b" ")
    loads = []
    fit = SimpleNamespace(load_arm=lambda *args: loads.append(args))
    with pytest.raises(ValueError, match="changed before evaluation"):
        runner["_reload_frozen_candidate"](protocol, output, freeze_hash, fit, {})
    assert loads == []


def test_frozen_candidate_load_uses_saved_bytes_and_preserves_control(tmp_path: Path) -> None:
    runner = _runner()
    output = tmp_path / "frozen"
    protocol, freeze_hash, expected_candidate = _freeze_fixture(output, runner)
    saved_calls, reloaded = [], object()
    def load_arm(replay, artifact):
        saved_calls.append((replay, artifact))
        return reloaded
    replay = {"training_identity": "fixed"}
    loaded, config, control, freeze = runner["_reload_frozen_candidate"](
        protocol, output, freeze_hash, SimpleNamespace(load_arm=load_arm), replay,
    )
    assert loaded is reloaded
    assert saved_calls == [(replay, expected_candidate)]
    assert config == protocol.FROZEN_CONFIG
    assert control == {"status": "reproduced"}
    assert tuple(entry["path"] for entry in freeze["artifacts"]) == runner["FREEZE_FILES"]


def test_control_failure_publishes_evidence_and_stops_before_lambda_one_or_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    namespace = runner["main"].__globals__
    protocol = runner["_protocol"]()
    original_loader = protocol.load_source
    fits, imports = [], []
    discrepancy = {"status": "discrepancy", "reason": "injected partition mismatch"}
    def load_source(root, name, relative, expected_sha256):
        imports.append(relative)
        module = original_loader(root, name, relative, expected_sha256)
        if relative == protocol.FIT_SOURCE:
            original_fit = module.fit_arm
            def fit_arm(replay, *, compatibility_weight):
                fits.append(compatibility_weight)
                assert type(compatibility_weight) is int and compatibility_weight == 0
                assert set(replay) == {"config", "kernel", "tensors", "saved_artifact", "train_h", "train_m"}
                return original_fit(replay, compatibility_weight=compatibility_weight)
            module.fit_arm = fit_arm
            module.verify_lambda_zero = lambda *args: copy.deepcopy(discrepancy)
        return module
    protocol.load_source = load_source
    monkeypatch.setitem(namespace, "_protocol", lambda: protocol)
    output = tmp_path / "failed-control"
    assert runner["main"](["synthetic", "--output", str(output)]) == 1
    assert fits == [0]
    assert protocol.EVALUATION_SOURCE not in imports
    assert {path.name for path in output.iterdir()} == {
        "frozen_config.json", "lambda_zero_control.json", "failure.json", "artifact_manifest.json",
    }
    failure = json.loads((output / "failure.json").read_bytes())
    assert failure["stage"] == "lambda_zero_rebuild"
    assert failure["details"] == discrepancy
    assert failure["candidate_evaluation_started"] is False
    assert failure["performance_conclusion"] == "unavailable_due_to_execution_failure"
    manifest = json.loads((output / "artifact_manifest.json").read_bytes())
    assert manifest["sealed_inputs_verified_unchanged"] is True
    assert manifest["candidate_evaluation_started"] is False


def test_full_ablation_freezes_before_evaluator_import_and_preserves_all_sealed_files(tmp_path: Path) -> None:
    protocol = _protocol()
    before = {path: path.read_bytes() for path in _sealed_files(protocol)}
    output = tmp_path / "intercept-compatibility"
    result = _guarded_run("synthetic", "--output", str(output), tensors_allowed=True)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {
        "frozen_config.json", "lambda_zero_control.json", "candidate.json", "block_scales.json",
        "freeze_manifest.json", "evaluation.json", "evaluation.md", "artifact_manifest.json",
    }
    report = json.loads((output / "evaluation.json").read_bytes())
    freeze = json.loads((output / "freeze_manifest.json").read_bytes())
    manifest = json.loads((output / "artifact_manifest.json").read_bytes())
    assert report["status"] == "synthetic_development_ablation_complete"
    assert report["original_reproduction"]["status"] == "reproduced"
    assert report["lambda_zero_control"]["status"] == "reproduced"
    assert report["evidence_boundary"] == protocol.EVIDENCE_BOUNDARY
    assert freeze["candidate_evaluation_started"] is False
    assert freeze["candidate_lambda"] == 1.0
    assert manifest["candidate_evaluation_started"] is True
    assert manifest["sealed_inputs_verified_unchanged"] is True
    assert manifest["authenticated_baselines"] == report["authenticated_baselines"] == freeze["authenticated_baselines"]
    assert manifest["source_manifest"] == report["new_source_manifest"] == freeze["source_manifest"]
    assert manifest["command_argv"] == [str(RUNNER), "synthetic", "--output", str(output)]
    assert manifest["environment"]["torch_version"]
    for entry in (*manifest["artifacts"], *freeze["artifacts"]):
        payload = (output / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["bytes"]
    for entry in manifest["source_manifest"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert all(path.read_bytes() == payload for path, payload in before.items())


def test_evaluation_claim_boundary_and_reproduction_evidence_cannot_change() -> None:
    runner, protocol = _runner(), _protocol()
    replay = {"reproduction": {"status": "reproduced", "metric_comparisons": []}}
    control = {"status": "reproduced", "consistent_label_permutation": [0, 1, 2]}
    report = {
        "schema": "fisher_graph.fisher_response_geometry_intercept_compatibility.v1",
        "status": "synthetic_development_ablation_complete",
        "evidence_boundary": copy.deepcopy(protocol.EVIDENCE_BOUNDARY),
        "lambda_zero_control": copy.deepcopy(control),
        "original_reproduction": copy.deepcopy(replay["reproduction"]),
    }
    validate = runner["_validate_evaluation"]
    validate(protocol, report, replay, control)
    for field in ("independent_confirmation", "target_model_loaded", "second_modification_implemented", "original_artifacts_modified"):
        changed = copy.deepcopy(report)
        changed["evidence_boundary"][field] = True
        with pytest.raises(ValueError, match="boundary"):
            validate(protocol, changed, replay, control)
    for key in ("lambda_zero_control", "original_reproduction"):
        changed = copy.deepcopy(report)
        changed[key]["status"] = "discrepancy"
        with pytest.raises(ValueError, match="evidence"):
            validate(protocol, changed, replay, control)
    changed = copy.deepcopy(report)
    changed["error"] = float("nan")
    with pytest.raises(ValueError, match="Out of range float"):
        validate(protocol, changed, replay, control)
