from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import fisher_graph.gemma3_l10_l17_adaptive_composition_bundle as freeze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(index: int) -> str:
    return f"{index:064x}"


def _guard() -> dict[str, object]:
    return {
        "evidence_file_sha256": _digest(10),
        "logical_sha256": _digest(11),
        "status": "passed",
        "assessment_role": "claimed_closed_guard_assessment",
        "heldout_confirmation": True,
        "fresh_validation": False,
    }


def _authorities(tmp_path: Path) -> dict[str, object]:
    legacy = tmp_path / "legacy.pt"
    layer10 = tmp_path / "layer10.pt"
    layer17 = tmp_path / "layer17.pt"
    adaptive = tmp_path / "adaptive.json"
    layer10.write_bytes(b"exact-layer10-candidate")
    layer17.write_bytes(b"exact-layer17-candidate")
    adaptive.write_text('{"source_safe":true}\n', encoding="utf-8")

    layer10_scientific = _digest(20)
    layer17_scientific = _digest(21)
    layer10_candidate = {
        "schema": freeze.GEMMA3_STATE_CONDITIONED_MODAL_GRAPH_SCHEMA,
        "scientific_payload_sha256": layer10_scientific,
    }
    layer17_candidate = {
        "schema": freeze.GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA,
        "scientific_payload_sha256": layer17_scientific,
    }
    legacy_bundle = {
        "parents": (
            {
                "role": "layer10",
                "candidate_tensor_file": layer10.name,
                "candidate_tensor_file_sha256": _sha256(layer10),
                "candidate_scientific_payload_sha256": layer10_scientific,
                "guard_evidence": _guard(),
                "candidate": copy.deepcopy(layer10_candidate),
            },
            {"role": "layer17"},
        )
    }
    challenger_label = "adaptive_a_fit"
    adaptive_result = {
        "result_sha256": _digest(30),
        "scientific_role": (
            "already_open_adaptive_development_fixed_capacity_refit"
        ),
        "heldout_confirmation": False,
        "candidate_changed": False,
        "fit_opened": False,
        "selection_opened": True,
        "guard_opened": False,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "safety": {
            "source_safe": True,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_logits": False,
            "contains_model_or_candidate_weights": False,
        },
        "candidate_pair": {
            "baseline_label": "frozen_v9",
            "challenger_label": challenger_label,
            "comparison_kind": "fixed_capacity_refit",
        },
        "candidates": {
            "frozen_v9": {},
            challenger_label: {
                "candidate_artifact_schema": (
                    freeze.GEMMA3_LAYER17_V8_ALL_FAMILY_REFIT_SCHEMA
                ),
                "tensor_file": layer17.name,
                "tensor_file_sha256": _sha256(layer17),
                "scientific_payload_sha256": layer17_scientific,
            },
        },
        "candidate_tensor_file_sha256s_after": {
            "frozen_v9": _digest(31),
            challenger_label: _sha256(layer17),
        },
        "authorization": {
            "challenger_tensor_file_sha256": _sha256(layer17),
            "challenger_scientific_payload_sha256": layer17_scientific,
            "selection_access_authorized": True,
            "heldout_confirmation": False,
            "serving_authorized": False,
        },
        "adaptive_selection": {
            "all_required_gates_pass": True,
            "adaptive_candidate_selected": True,
            "next_action": (
                "retain_adaptive_candidate_for_open_development_only"
            ),
        },
    }
    return {
        "legacy_path": legacy,
        "layer10_path": layer10,
        "layer17_path": layer17,
        "adaptive_path": adaptive,
        "legacy_bundle": legacy_bundle,
        "layer10_candidate": layer10_candidate,
        "layer17_candidate": layer17_candidate,
        "adaptive_result": adaptive_result,
    }


def _install_loaders(
    monkeypatch: pytest.MonkeyPatch,
    authorities: dict[str, object],
) -> None:
    monkeypatch.setattr(
        freeze,
        "load_gemma3_layer10_layer17_composition_bundle",
        lambda path: authorities["legacy_bundle"],
    )
    monkeypatch.setattr(
        freeze,
        "load_gemma3_state_conditioned_modal_graph_candidate",
        lambda path: authorities["layer10_candidate"],
    )
    monkeypatch.setattr(
        freeze,
        "load_gemma3_layer17_v8_all_family_refit_candidate",
        lambda path: authorities["layer17_candidate"],
    )
    monkeypatch.setattr(
        freeze,
        "load_gemma3_layer17_open_a_capacity_result",
        lambda path: authorities["adaptive_result"],
    )


def _freeze_arguments(
    authorities: dict[str, object],
    output: Path,
) -> dict[str, object]:
    return {
        "legacy_bundle_path": authorities["legacy_path"],
        "layer10_candidate_path": authorities["layer10_path"],
        "layer17_candidate_path": authorities["layer17_path"],
        "adaptive_result_path": authorities["adaptive_path"],
        "output": output,
    }


def test_freeze_binds_both_authorities_and_calls_generalized_composer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities = _authorities(tmp_path)
    _install_loaders(monkeypatch, authorities)
    captured: dict[str, object] = {}

    def save(output: Path, **arguments: object) -> dict[str, object]:
        captured["output"] = output
        captured.update(arguments)
        return {
            "artifact": {"composition_payload_sha256": _digest(40)},
            "report_sha256": _digest(41),
            "safety": {"source_safe": True},
        }

    monkeypatch.setattr(
        freeze,
        "save_gemma3_layer10_layer17_composition_bundle",
        save,
    )
    output = tmp_path / "adaptive-composition.pt"
    report = freeze.freeze_gemma3_l10_l17_adaptive_composition_bundle(
        **_freeze_arguments(authorities, output)
    )

    assert report["report_sha256"] == _digest(41)
    assert captured["output"] == output
    assert captured["layer10_candidate_path"] == authorities["layer10_path"]
    assert captured["layer17_candidate_path"] == authorities["layer17_path"]
    layer10_evidence = captured["layer10_guard_evidence"]
    layer17_evidence = captured["layer17_guard_evidence"]
    assert isinstance(layer10_evidence, freeze.SourceSafeGuardEvidenceRecord)
    assert layer10_evidence.state_dict() == _guard()
    assert isinstance(layer17_evidence, freeze.SourceSafeGuardEvidenceRecord)
    assert layer17_evidence.evidence_file_sha256 == _sha256(
        authorities["adaptive_path"]  # type: ignore[arg-type]
    )
    assert layer17_evidence.logical_sha256 == _digest(30)
    assert layer17_evidence.status == "passed"
    assert layer17_evidence.assessment_role == "open_development_assessment"
    assert layer17_evidence.heldout_confirmation is False
    assert layer17_evidence.fresh_validation is False


@pytest.mark.parametrize("existing", ["tensor", "report"])
def test_freeze_refuses_overwrite_before_reading_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: str,
) -> None:
    output = tmp_path / "adaptive-composition.pt"
    occupied = output if existing == "tensor" else output.with_suffix(".json")
    occupied.write_bytes(b"occupied")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("authority loader must not run")

    monkeypatch.setattr(
        freeze,
        "load_gemma3_layer10_layer17_composition_bundle",
        forbidden,
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        freeze.freeze_gemma3_l10_l17_adaptive_composition_bundle(output=output)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("candidate_tensor_file_sha256", _digest(50)),
        ("candidate_scientific_payload_sha256", _digest(51)),
    ),
)
def test_freeze_rejects_layer10_that_differs_from_legacy_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    authorities = _authorities(tmp_path)
    parent = authorities["legacy_bundle"]["parents"][0]  # type: ignore[index]
    parent[field] = replacement
    _install_loaders(monkeypatch, authorities)

    with pytest.raises(ValueError, match="layer10 candidate differs"):
        freeze.freeze_gemma3_l10_l17_adaptive_composition_bundle(
            **_freeze_arguments(
                authorities,
                tmp_path / "adaptive-composition.pt",
            )
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("adaptive_selection", "all_required_gates_pass"), False),
        (("adaptive_selection", "adaptive_candidate_selected"), False),
        (("guard_opened",), True),
        (("safety", "contains_prompt_text"), True),
        (
            ("candidate_tensor_file_sha256s_after", "adaptive_a_fit"),
            _digest(60),
        ),
        (
            ("authorization", "challenger_tensor_file_sha256"),
            _digest(61),
        ),
        (
            (
                "candidates",
                "adaptive_a_fit",
                "scientific_payload_sha256",
            ),
            _digest(62),
        ),
    ),
)
def test_freeze_rejects_failed_or_unbound_adaptive_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    replacement: object,
) -> None:
    authorities = _authorities(tmp_path)
    cursor = authorities["adaptive_result"]
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    _install_loaders(monkeypatch, authorities)
    monkeypatch.setattr(
        freeze,
        "save_gemma3_layer10_layer17_composition_bundle",
        lambda *args, **kwargs: pytest.fail("composer must not run"),
    )

    with pytest.raises(ValueError, match="does not authorize"):
        freeze.freeze_gemma3_l10_l17_adaptive_composition_bundle(
            **_freeze_arguments(
                authorities,
                tmp_path / "adaptive-composition.pt",
            )
        )


def test_public_freeze_surface_cannot_open_data_or_models() -> None:
    parameters = set(
        inspect.signature(
            freeze.freeze_gemma3_l10_l17_adaptive_composition_bundle
        ).parameters
    )
    assert parameters == {
        "legacy_bundle_path",
        "layer10_candidate_path",
        "layer17_candidate_path",
        "adaptive_result_path",
        "output",
    }
    parser = freeze.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "legacy_bundle",
        "layer10_candidate",
        "layer17_candidate",
        "adaptive_result",
        "output",
    }
    forbidden = {
        "prompt",
        "corpus",
        "selection",
        "guard",
        "calibration_b",
        "validation",
        "test",
        "model",
        "device",
        "cache",
    }
    assert not destinations & forbidden


def test_cli_defaults_and_source_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict[str, object] = {}

    def frozen(**arguments: object) -> dict[str, object]:
        observed.update(arguments)
        return {
            "artifact": {"composition_payload_sha256": _digest(70)},
            "report_sha256": _digest(71),
            "safety": {"source_safe": True},
        }

    monkeypatch.setattr(
        freeze,
        "freeze_gemma3_l10_l17_adaptive_composition_bundle",
        frozen,
    )
    assert freeze.main([]) == 0
    summary = json.loads(capsys.readouterr().out)

    assert observed == {
        "legacy_bundle_path": freeze.DEFAULT_LEGACY_COMPOSITION_BUNDLE,
        "layer10_candidate_path": freeze.DEFAULT_LAYER10_CANDIDATE,
        "layer17_candidate_path": freeze.DEFAULT_ADAPTIVE_A_FIT_CANDIDATE,
        "adaptive_result_path": freeze.DEFAULT_ADAPTIVE_OPEN_A_OUTPUT,
        "output": freeze.DEFAULT_ADAPTIVE_COMPOSITION_OUTPUT,
    }
    assert summary == {
        "composition_payload_sha256": _digest(70),
        "heldout_confirmation": False,
        "output": str(freeze.DEFAULT_ADAPTIVE_COMPOSITION_OUTPUT),
        "report_sha256": _digest(71),
        "serving_authorized": False,
        "source_safe": True,
    }
