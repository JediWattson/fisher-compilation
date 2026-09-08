"""The saved synthetic fixture is replayed without refitting or mutable inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPLAY_PATH = ROOT / "src/fisher_graph/fisher_response_geometry_replay.py"
_SPEC = importlib.util.spec_from_file_location("_test_fisher_response_geometry_replay", REPLAY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
replay_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = replay_module
_SPEC.loader.exec_module(replay_module)


def _original_paths() -> list[str]:
    return [
        *replay_module.ORIGINAL_SOURCE_PATHS,
        *(f"{replay_module.ORIGINAL_DIRECTORY}/{name}" for name in (
            "report.json", "frozen_config.json", "artifact_manifest.json",
        )),
    ]


@pytest.fixture
def isolated_original(tmp_path: Path) -> Path:
    """Copy the fixed allowlist only; never traverse research-data directories."""

    root = tmp_path / "pinned-repository"
    for relative in _original_paths():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return root


def test_replay_reproduces_absolute_derivative_finite_and_projection_baselines() -> None:
    replay = replay_module.load_replay()
    evidence = replay["reproduction"]
    assert evidence["status"] == "reproduced"
    assert evidence["training_row_count"] == 96
    assert evidence["evaluation_row_count"] == 48
    assert len(evidence["metric_comparisons"]) >= 170
    assert all(evidence["exact_checks"].values())
    assert all(row["within_tolerance"] for row in evidence["metric_comparisons"])
    baseline = evidence["baseline_absolute_residual_update"]
    assert baseline["response_charts"]["runtime_router"]["fisher_relative_l2"] == pytest.approx(0.8926561273956386, rel=0, abs=1e-13)
    assert baseline["response_charts"]["direct_discovery_assignment"]["fisher_relative_l2"] == pytest.approx(0.884313051910265, rel=0, abs=1e-13)
    for name in ("runtime_router", "direct_discovery_assignment"):
        assert baseline["activation_charts"][name]["fisher_relative_l2"] == pytest.approx(0.1362931496408758, rel=0, abs=1e-13)
    assert baseline["response_charts"]["runtime_supported_row_count"] == 38
    assert baseline["activation_charts"]["runtime_supported_row_count"] == 48
    labels = {row["measurement"] for row in evidence["metric_comparisons"]}
    assert "response_charts.projected_derivative_relative_l2" in labels
    assert "radius_0.1.response_charts.fisher_relative_l2" in labels
    assert "exact_analytic_projected_normalization.fisher_relative_l2" in labels


def test_replay_is_deterministic_and_uses_saved_coordinate_map() -> None:
    first, second = replay_module.load_replay(), replay_module.load_replay()
    torch = first["kernel"].torch
    for key in ("train_h", "train_m", "eval_h", "eval_m", "probe_directions",
                "state_probe_directions", "train_signatures", "evaluation_signatures"):
        assert torch.equal(first[key], second[key])
    assert first["train_ids"] == second["train_ids"]
    assert first["eval_ids"] == second["eval_ids"]
    assert first["reproduction"] == second["reproduction"]
    assert first["projection"].coordinate_map is first["tensors"]["D_G"]
    assert not set(first["train_ids"]) & set(first["eval_ids"])
    for assignment in ("runtime_assignments", "direct_assignments", "train_direct_assignments"):
        for family in first[assignment]:
            assert torch.equal(first[assignment][family], second[assignment][family])


def test_replay_never_refits_or_invokes_the_original_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    original_load = replay_module._load_kernel

    def forbidden(*_args, **_kwargs):
        raise AssertionError("replay attempted a forbidden fit or qualification run")

    def guarded_load(root, source):
        kernel = original_load(root, source)
        for name in ("run_synthetic_qualification", "_orthogonal_rows", "_existing_chart_primitives", "FisherProjection"):
            monkeypatch.setattr(kernel, name, forbidden)
        for name in ("svd", "eigh", "lstsq", "solve"):
            monkeypatch.setattr(kernel.torch.linalg, name, forbidden)
        return kernel

    monkeypatch.setattr(replay_module, "_load_kernel", guarded_load)
    assert replay_module.load_replay()["reproduction"]["status"] == "reproduced"


@pytest.mark.parametrize("relative", [
    f"{replay_module.ORIGINAL_DIRECTORY}/report.json",
    f"{replay_module.ORIGINAL_DIRECTORY}/frozen_config.json",
    f"{replay_module.ORIGINAL_DIRECTORY}/artifact_manifest.json",
    "src/fisher_graph/fisher_response_geometry.py",
    "src/fisher_graph/gemma3_attention_generator.py",
    "examples/fisher_response_geometry_probe_v1.json",
])
def test_corruption_fails_before_numerical_import(
    isolated_original: Path, relative: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = isolated_original / relative
    target.write_bytes(target.read_bytes() + b" ")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("kernel import occurred before authentication succeeded")

    monkeypatch.setattr(replay_module, "_load_kernel", forbidden)
    with pytest.raises(replay_module.ReplayIntegrityError, match="SHA-256 mismatch"):
        replay_module.load_replay(isolated_original)


def test_missing_or_symlinked_original_fails_closed(isolated_original: Path) -> None:
    relative = "src/fisher_graph/fisher_response_geometry.py"
    target = isolated_original / relative
    target.unlink()
    with pytest.raises(replay_module.ReplayIntegrityError, match="missing or a symlink"):
        replay_module.load_replay(isolated_original)
    target.symlink_to(ROOT / relative)
    with pytest.raises(replay_module.ReplayIntegrityError, match="missing or a symlink"):
        replay_module.load_replay(isolated_original)


def test_reproduction_discrepancy_is_reported_without_changing_saved_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = replay_module._replay_inputs

    def changed_inputs(torch, config):
        result = original(torch, config)
        result["eval_m"][0, 0] += 0.25
        return result

    monkeypatch.setattr(replay_module, "_replay_inputs", changed_inputs)
    with pytest.raises(replay_module.ReplayReproductionError) as captured:
        replay_module.load_replay()
    evidence = captured.value.reproduction
    assert evidence["status"] == "discrepancy"
    failed = [row for row in evidence["metric_comparisons"] if not row["within_tolerance"]]
    assert any(row["measurement"] == "response_charts.runtime_all_rows.fisher_relative_l2" for row in failed)
    assert all("saved" in row and "replayed" in row and "absolute_difference" in row for row in failed)


def test_original_sources_artifacts_and_modification_times_are_preserved() -> None:
    def snapshot():
        return {relative: (
            hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
            (ROOT / relative).stat().st_mtime_ns,
        ) for relative in _original_paths()}

    before = snapshot()
    replay_module.load_replay()
    assert snapshot() == before


def test_replay_avoids_models_corpora_and_eager_package_imports() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import json
        import os
        from pathlib import Path
        import sys
        sys.dont_write_bytecode = True

        class Imports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'fisher_graph', 'transformers', 'adapters', 'mlx', 'safetensors', 'sentencepiece'}:
                    raise AssertionError('forbidden import: ' + fullname)

        def audit(event, args):
            if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(args[0]))
            parts = {part.lower() for part in path.parts}
            if parts & {'.local-runs', 'corpus', 'corpora', 'datasets'}:
                raise AssertionError('protected data access: ' + str(path))
            if any(part.startswith(('corpus_', 'corpus-', 'calibration_', 'calibration-')) for part in parts):
                raise AssertionError('protected data access: ' + str(path))
            if path.suffix.lower() in {'.safetensors', '.pt', '.bin'}:
                raise AssertionError('model file access: ' + str(path))

        sys.meta_path.insert(0, Imports())
        sys.addaudithook(audit)
        spec = importlib.util.spec_from_file_location('_guarded_replay', sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert 'torch' not in sys.modules
        result = module.load_replay()
        print(json.dumps({'status': result['reproduction']['status']}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(REPLAY_PATH)], cwd=ROOT,
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "reproduced"}
