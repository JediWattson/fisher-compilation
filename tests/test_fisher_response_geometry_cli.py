"""Subprocess boundary checks for the synthetic-only normalization probe."""

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
RUNNER = ROOT / "src/fisher_response_geometry_probe_v1.py"
PROTOCOL = runpy.run_path(
    str(ROOT / "src/fisher_graph/fisher_response_geometry_protocol.py")
)


def _guarded_run(*arguments: str, tensors_allowed: bool = False) -> subprocess.CompletedProcess[str]:
    """Reject protected reads and forbidden imports before invoking the CLI."""

    guard = textwrap.dedent(
        """
        import importlib.abc
        import os
        from pathlib import Path
        import runpy
        import sys

        tensors_allowed = sys.argv[1] == "yes"
        runner = sys.argv[2]
        args = sys.argv[3:]
        sys.dont_write_bytecode = True

        class ForbiddenImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = ("transformers", "adapters", "mlx", "safetensors",
                             "sentencepiece", "fisher_graph")
                if not tensors_allowed:
                    forbidden += ("torch", "numpy")
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("forbidden import: " + fullname)

        def audit(event, args):
            if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(args[0]))
            parts = {part.lower() for part in path.parts}
            if ".local-runs" in parts or "corpus" in parts or "corpora" in parts:
                raise AssertionError("protected data access: " + str(path))
            if path.suffix.lower() in {".safetensors", ".pt", ".bin"}:
                raise AssertionError("model/tensor file access: " + str(path))
            if any(part.startswith(("calibration_", "calibration-", "corpus_", "corpus-"))
                   for part in parts):
                raise AssertionError("protected corpus access: " + str(path))

        sys.meta_path.insert(0, ForbiddenImport())
        sys.addaudithook(audit)
        sys.argv = [runner, *args]
        runpy.run_path(runner, run_name="__main__")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", guard, "yes" if tensors_allowed else "no", str(RUNNER),
         *arguments],
        cwd=ROOT, text=True, capture_output=True, check=False, timeout=60,
    )


def _synthetic_report_fixture() -> dict[str, object]:
    config = copy.deepcopy(PROTOCOL["FROZEN_CONFIG"])
    scaled = {name: 0.0 for name in (
        "analytic_matches_autodiff", "dense_analytic_matches_autodiff",
        "radial_identity", "radial_autodiff_identity",
    )}
    scaled["frozen_scale_adversary_rejected"] = 2.0
    checks = dict.fromkeys(scaled, True)
    checks["finite_difference_sweep"] = True
    fd = [{"step": step, "absolute_error": 0.0, "relative_error": 0.0}
          for step in config["fd_steps"]]
    dtype_rows = []
    for name, prefix in (("float32", "fp32"), ("float16", "fp16"), ("bfloat16", "bfloat16")):
        dtype_rows.append({
            "input_projection_output_dtype": name, "normalizer_compute_dtype": "float32",
            "max_absolute_jvp_error": 0.0, "relative_jvp_error": 0.0,
            "absolute_tolerance": config["numerical_tolerances"][f"{prefix}_jvp_atol"],
            "relative_tolerance": config["numerical_tolerances"][f"{prefix}_jvp_rtol"],
            "max_scaled_tolerance_error": 0.0, "passed": True,
            "finite_difference_sweep": copy.deepcopy(fd),
            "finite_difference_interpretation": "reported_only_cast_roundoff_can_dominate_small_steps",
        })
        checks[f"{name}_cast_semantics"] = True
    derivative = {
        "status": "passed", "checks": checks, "scaled_tolerance_errors": scaled,
        "float64_analytic_autodiff_max_absolute_error": 0.0,
        "float64_dense_jacobian_max_absolute_error": 0.0,
        "radial_identity_max_absolute_error": 0.0, "radial_expected_norm": 0.0,
        "radial_frozen_scale_control_norm": 1.0, "radial_frozen_scale_error_norm": 1.0,
        "float64_finite_difference_sweep": fd, "dtype_semantics": dtype_rows,
    }
    sections = {name: {"status": "measured", "synthetic_test_measurement": 1.0}
                for name in PROTOCOL["SECTION_NAMES"]}
    sections["derivative_correctness"] = derivative
    return PROTOCOL["build_report"](
        config=config,
        source_hashes=[{"path": path, "sha256": "0" * 64}
                       for path in PROTOCOL["SOURCE_PATHS"]],
        sections=sections,
    )


def test_inspect_is_stdlib_only_and_config_is_frozen() -> None:
    result = _guarded_run("inspect")
    assert result.returncode == 0, result.stderr
    inspected = json.loads(result.stdout)
    assert inspected["config"] == PROTOCOL["FROZEN_CONFIG"]
    assert inspected["config_sha256"] == PROTOCOL["config_digest"](inspected["config"])
    assert inspected["evidence_boundary"] == PROTOCOL["EVIDENCE_BOUNDARY"]
    assert [entry["path"] for entry in inspected["source_manifest"]] == list(
        PROTOCOL["SOURCE_PATHS"]
    )


def test_model_facing_fails_closed_before_tensor_import_or_data_open() -> None:
    result = _guarded_run("model-facing")
    assert result.returncode == 2, result.stderr
    result_json = json.loads(result.stdout)
    assert result_json["status"] == PROTOCOL["CURRENT_P0_STATUS"]
    assert result_json["model_facing_execution"] == "blocked"


@pytest.mark.parametrize("arguments", [
    ("synthetic", "--output", "unused", "--config", "evil.json"),
    ("synthetic", "--output", "unused", "--model", "anything"),
    ("synthetic", "--output", "unused", "--dataset", "anything"),
    ("synthetic", "--output", "unused", "--live-attention"),
    ("model-facing", "--corpus", "anything"),
])
def test_cli_has_no_data_model_or_future_experiment_overrides(arguments: tuple[str, ...]) -> None:
    result = _guarded_run(*arguments)
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_synthetic_runner_requires_a_fresh_output_before_tensor_import(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    result = _guarded_run("synthetic", "--output", str(existing))
    assert result.returncode == 1
    assert "fresh, nonexistent directory" in result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_synthetic_runner_rejects_symlink_output_before_tensor_import(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "linked"
    symlink.symlink_to(target, target_is_directory=True)
    result = _guarded_run("synthetic", "--output", str(symlink))
    assert result.returncode == 1
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("name", [".local-runs", "corpus", "calibration_a", ".git"])
def test_synthetic_runner_cannot_write_into_protected_paths(tmp_path: Path, name: str) -> None:
    parent = tmp_path / name
    parent.mkdir()
    output = parent / "fresh"
    result = _guarded_run("synthetic", "--output", str(output))
    assert result.returncode == 1
    assert "protected data or runtime paths" in result.stderr
    assert not output.exists()


def test_config_rejects_overrides_and_boolean_integer_substitution() -> None:
    config = copy.deepcopy(PROTOCOL["FROZEN_CONFIG"])
    config["seed"] = 99
    with pytest.raises(ValueError, match="frozen synthetic"):
        PROTOCOL["validate_frozen_config"](config)
    config = copy.deepcopy(PROTOCOL["FROZEN_CONFIG"])
    config["epsilon"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        PROTOCOL["validate_frozen_config"](config)
    config = copy.deepcopy(PROTOCOL["FROZEN_CONFIG"])
    config["metric_rtol"] = True
    with pytest.raises(ValueError, match="frozen synthetic"):
        PROTOCOL["validate_frozen_config"](config)


def test_report_schema_rejects_claim_escalation_nonfinite_and_missing_measurements() -> None:
    report = _synthetic_report_fixture()
    PROTOCOL["validate_report"](report)
    for key in ("model_loaded", "protected_data_opened", "guard_consumed",
                "live_attention_executed", "model_facing_scientific_evidence",
                "compression_or_speedup_claim", "reusable_attention_routing_geometry_claim",
                "accuracy_better_than_exact_reference_claim"):
        changed = copy.deepcopy(report)
        changed["evidence_boundary"][key] = True
        with pytest.raises(ValueError, match="evidence boundary"):
            PROTOCOL["validate_report"](changed)
    changed = copy.deepcopy(report)
    changed["evidence_boundary"]["model_loaded"] = 0
    with pytest.raises(ValueError, match="evidence boundary"):
        PROTOCOL["validate_report"](changed)
    changed = copy.deepcopy(report)
    changed["sections"]["derivative_correctness"] = {"status": "passed"}
    with pytest.raises(ValueError, match="status alone"):
        PROTOCOL["validate_report"](changed)
    changed = copy.deepcopy(report)
    changed["sections"]["derivative_correctness"] = {
        "status": "passed", "analytic_correct": True, "radial_correct": True,
    }
    with pytest.raises(ValueError, match="numerical evidence, not only flags"):
        PROTOCOL["validate_report"](changed)
    changed = copy.deepcopy(report)
    changed["sections"]["derivative_correctness"]["error"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        PROTOCOL["validate_report"](changed)
    changed = copy.deepcopy(report)
    del changed["sections"]["chart_stability"]
    with pytest.raises(ValueError, match="six declared sections"):
        PROTOCOL["validate_report"](changed)


@pytest.mark.parametrize("name", [
    "analytic_matches_autodiff", "dense_analytic_matches_autodiff",
    "radial_identity", "radial_autodiff_identity", "frozen_scale_adversary_rejected",
])
def test_derivative_gates_are_recomputed_from_numerical_evidence(name: str) -> None:
    report = _synthetic_report_fixture()
    derivative = report["sections"]["derivative_correctness"]
    derivative["scaled_tolerance_errors"][name] = (
        0.0 if name == "frozen_scale_adversary_rejected" else 2.0
    )
    with pytest.raises(ValueError, match="contradicts numerical evidence"):
        PROTOCOL["validate_report"](report)
    derivative["checks"][name] = False
    with pytest.raises(ValueError, match="status contradicts recomputed"):
        PROTOCOL["validate_report"](report)
    derivative["status"] = "failed"
    PROTOCOL["validate_report"](report)


def test_finite_difference_gate_cannot_claim_pass_after_failed_or_shortened_sweep() -> None:
    report = _synthetic_report_fixture()
    derivative = report["sections"]["derivative_correctness"]
    for row in derivative["float64_finite_difference_sweep"]:
        row["relative_error"] = 1.0
    with pytest.raises(ValueError, match="finite_difference_sweep"):
        PROTOCOL["validate_report"](report)
    derivative["float64_finite_difference_sweep"].pop()
    with pytest.raises(ValueError, match="entire frozen step sweep"):
        PROTOCOL["validate_report"](report)


def test_dtype_tolerances_pass_flags_and_semantics_are_enforced() -> None:
    report = _synthetic_report_fixture()
    dtype = report["sections"]["derivative_correctness"]["dtype_semantics"][0]
    dtype["max_scaled_tolerance_error"] = 2.0
    with pytest.raises(ValueError, match="dtype pass flag contradicts"):
        PROTOCOL["validate_report"](report)
    dtype["max_scaled_tolerance_error"] = 0.0
    dtype["absolute_tolerance"] = 100.0
    with pytest.raises(ValueError, match="tolerance differs from the frozen"):
        PROTOCOL["validate_report"](report)
    dtype["absolute_tolerance"] = PROTOCOL["FROZEN_CONFIG"]["numerical_tolerances"]["fp32_jvp_atol"]
    dtype["normalizer_compute_dtype"] = "float64"
    with pytest.raises(ValueError, match="repository float32 semantics"):
        PROTOCOL["validate_report"](report)


def test_synthetic_cli_emits_hashed_artifacts_without_model_or_data_access(tmp_path: Path) -> None:
    output = tmp_path / "normalization-synthetic"
    result = _guarded_run("synthetic", "--output", str(output), tensors_allowed=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    PROTOCOL["validate_report"](report)
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_class"] == "synthetic_qualification_only"
    assert {entry["path"] for entry in manifest["artifacts"]} == {
        "frozen_config.json", "report.json"
    }
    for artifact in manifest["artifacts"]:
        payload = (output / artifact["path"]).read_bytes()
        assert artifact["sha256"] == hashlib.sha256(payload).hexdigest()
        assert artifact["bytes"] == len(payload)
    assert report["config_sha256"] == manifest["config_sha256"]
    assert report["source_manifest"] == manifest["source_manifest"]
    assert report["evidence_boundary"]["p0_status"] == PROTOCOL["CURRENT_P0_STATUS"]


def test_failed_numerical_gate_exits_nonzero_but_preserves_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    runner = runpy.run_path(str(RUNNER))
    namespace = runner["main"].__globals__
    sections = _synthetic_report_fixture()["sections"]
    derivative = sections["derivative_correctness"]
    derivative["scaled_tolerance_errors"]["radial_identity"] = 2.0
    derivative["checks"]["radial_identity"] = False
    derivative["status"] = "failed"
    monkeypatch.setitem(namespace, "_protocol", lambda: SimpleNamespace(**PROTOCOL))
    monkeypatch.setitem(namespace, "_load_source", lambda *_: SimpleNamespace(
        run_synthetic_qualification=lambda _: sections,
    ))
    output = tmp_path / "failed-qualification"
    assert runner["main"](["synthetic", "--output", str(output)]) == 1
    notice = json.loads(capsys.readouterr().out)
    assert notice["numerical_qualification"] == "failed"
    assert notice["failed_sections"] == ["derivative_correctness"]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    PROTOCOL["validate_report"](report)
    assert report["sections"]["derivative_correctness"]["status"] == "failed"
    assert (output / "artifact_manifest.json").is_file()


def test_malformed_numerical_result_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    runner = runpy.run_path(str(RUNNER))
    namespace = runner["main"].__globals__
    sections = _synthetic_report_fixture()["sections"]
    del sections["derivative_correctness"]["scaled_tolerance_errors"]
    monkeypatch.setitem(namespace, "_protocol", lambda: SimpleNamespace(**PROTOCOL))
    monkeypatch.setitem(namespace, "_load_source", lambda *_: SimpleNamespace(
        run_synthetic_qualification=lambda _: sections,
    ))
    output = tmp_path / "malformed-qualification"
    assert runner["main"](["synthetic", "--output", str(output)]) == 1
    notice = capsys.readouterr()
    assert not notice.out
    assert "all five scaled tolerance errors" in notice.err
    assert list(output.iterdir()) == []
