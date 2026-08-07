from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat

import pytest
import torch

from fisher_graph import gemma3_l10_l17_a5d_prepublication_bundle as bundle


def _inputs() -> dict[str, object]:
    return {
        "source_bindings": {"source": "a" * 64},
        "runtime": {"device": "cpu"},
        "configuration": {"outer_fold_index": 0},
        "capture": {"capture_sha256": "b" * 64},
        "source_anchored_residual": {
            "receipt_sha256": "c" * 64,
            "contains_tensor_payloads": False,
        },
        "residual_cv": {
            "receipt_sha256": "d" * 64,
            "contains_tensor_payloads": False,
        },
        "evidence_receipts": {
            "source_anchored_residual": {"receipt_sha256": "c" * 64},
            "residual_cv": {"receipt_sha256": "d" * 64},
        },
        "selected_executable": {
            "selection_freeze_sha256": "e" * 64,
            "additive_residual": None,
        },
        "chronology": {"executable_frozen_event": 2},
        "outer_evaluation": {"outer_evaluation_sha256": "f" * 64},
        "comparison_to_a5c": {"same_outer_fold": True},
    }


def test_bundle_roundtrips_privately_and_refuses_overwrite(tmp_path: Path) -> None:
    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)
    inputs = _inputs()

    saved = bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=inputs
    )
    loaded = bundle.load_a5d_prepublication_bundle(path, final_output=final)

    assert loaded == saved
    assert loaded["report_inputs"] == inputs
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert loaded["target_report_schema"].endswith(
        "a5d_source_anchored_residual_generator"
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        bundle.save_a5d_prepublication_bundle(
            path, final_output=final, report_inputs=inputs
        )


def test_bundle_rejects_wrong_fields_tensors_and_sensitive_keys(
    tmp_path: Path,
) -> None:
    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)

    missing = _inputs()
    missing.pop("residual_cv")
    with pytest.raises(ValueError, match="fields"):
        bundle.save_a5d_prepublication_bundle(
            path, final_output=final, report_inputs=missing
        )

    tensor = _inputs()
    tensor["capture"] = {"value": torch.zeros(1)}
    with pytest.raises(TypeError, match="tensor"):
        bundle.save_a5d_prepublication_bundle(
            path, final_output=final, report_inputs=tensor
        )

    sensitive = _inputs()
    sensitive["capture"] = {"prompt_ids": ["outer-secret"]}
    with pytest.raises(ValueError, match="forbidden sensitive field"):
        bundle.save_a5d_prepublication_bundle(
            path, final_output=final, report_inputs=sensitive
        )


def test_bundle_hash_and_private_file_boundary_fail_closed(tmp_path: Path) -> None:
    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)
    bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=_inputs()
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["report_inputs"]["configuration"]["outer_fold_index"] = 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="report-input hash"):
        bundle.load_a5d_prepublication_bundle(path, final_output=final)

    path.unlink()
    bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=_inputs()
    )
    path.chmod(0o644)
    with pytest.raises(ValueError, match="file boundary"):
        bundle.load_a5d_prepublication_bundle(path, final_output=final)


def test_bundle_rejects_symlink_and_hardlink_boundaries(tmp_path: Path) -> None:
    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o600)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="unavailable"):
        bundle.load_a5d_prepublication_bundle(path, final_output=final)

    path.unlink()
    bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=_inputs()
    )
    alias = tmp_path / "hardlink.json"
    os.link(path, alias)
    with pytest.raises(ValueError, match="file boundary"):
        bundle.load_a5d_prepublication_bundle(path, final_output=final)


def test_finalize_publishes_then_removes_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)
    inputs = _inputs()
    bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=inputs
    )
    built = {"built": True}

    def fake_build(**received: object) -> dict[str, object]:
        assert received == inputs
        return built

    def fake_save(
        destination: Path | str, value: object
    ) -> dict[str, object]:
        assert Path(destination) == final
        assert value == built
        final.write_text("published\n", encoding="utf-8")
        return built

    monkeypatch.setattr(bundle, "build_gemma3_l10_l17_a5d_report", fake_build)
    monkeypatch.setattr(bundle, "save_gemma3_l10_l17_a5d_report", fake_save)

    assert bundle.finalize_a5d_prepublication_bundle(path) == built
    assert final.read_text(encoding="utf-8") == "published\n"
    assert not path.exists()


@pytest.mark.parametrize("failure_stage", ["build", "save"])
def test_finalize_preserves_bundle_on_every_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)
    saved = bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=_inputs()
    )

    def fake_build(**_: object) -> dict[str, object]:
        if failure_stage == "build":
            raise RuntimeError("builder failed")
        return {"built": True}

    def fake_save(*_: object, **__: object) -> dict[str, object]:
        raise FileExistsError("publication failed")

    monkeypatch.setattr(bundle, "build_gemma3_l10_l17_a5d_report", fake_build)
    monkeypatch.setattr(bundle, "save_gemma3_l10_l17_a5d_report", fake_save)

    with pytest.raises((RuntimeError, FileExistsError)):
        bundle.finalize_a5d_prepublication_bundle(path, output=final)
    assert path.exists()
    assert bundle.load_a5d_prepublication_bundle(
        path, final_output=final
    ) == saved


def test_default_path_and_strict_json_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON basename"):
        bundle.default_a5d_prepublication_bundle_path(tmp_path / "a5d.txt")

    final = tmp_path / "a5d.json"
    path = bundle.default_a5d_prepublication_bundle_path(final)
    wrong = tmp_path / "wrong.prepublication-bundle.json"
    with pytest.raises(ValueError, match="differs"):
        bundle.save_a5d_prepublication_bundle(
            wrong, final_output=final, report_inputs=_inputs()
        )

    bundle.save_a5d_prepublication_bundle(
        path, final_output=final, report_inputs=_inputs()
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["report_inputs"]["configuration"]["value"] = float("nan")
    path.write_text(json.dumps(raw), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="non-finite"):
        bundle.load_a5d_prepublication_bundle(path, final_output=final)
