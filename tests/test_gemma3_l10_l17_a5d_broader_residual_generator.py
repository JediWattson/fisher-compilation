from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l10_l17_a5d_broader_residual_generator as runner,
)


def _digest(seed: int) -> str:
    return f"{seed:064x}"


def _canonical_a5c() -> dict[str, object]:
    return {
        "report_sha256": runner._EXPECTED_A5C_REPORT_SHA256,
        "outer_evaluation": {
            "outer_fold_index": 0,
            "logical_valid_tokens": 36,
            "supervised_tokens": 35,
            "conditions": {
                "frozen_uncorrected_composition": {
                    **runner._EXPECTED_A5C_FROZEN_COMPOSITION,
                    "graph_sha256": _digest(90),
                }
            }
        },
    }


class _ReceiptObject:
    def __init__(self, receipt: dict[str, object]) -> None:
        self._receipt = receipt

    def receipt(self) -> dict[str, object]:
        return dict(self._receipt)


class _Executable:
    selection_freeze_sha256 = _digest(70)

    def report_section(self) -> dict[str, object]:
        return {
            "kind": "frozen_source_fallback",
            "selection_freeze_sha256": self.selection_freeze_sha256,
        }


def _workspace() -> SimpleNamespace:
    layer10_names = tuple(
        runner_module_name
        for runner_module_name in (
            "layer10.node.0",
            "layer10.node.1",
            "layer10.node.2",
            "layer10.node.3",
        )
    )
    layer17_names = (
        "layer17.node.0",
        "layer17.node.1",
        "layer17.node.2",
        "layer17.node.3",
    )
    layer10_graph = SimpleNamespace(
        traversal_order=layer10_names, artifact_sha256=_digest(20)
    )
    layer17_graph = SimpleNamespace(
        traversal_order=layer17_names, artifact_sha256=_digest(21)
    )
    composition = SimpleNamespace(artifact_sha256=_digest(22))
    all_rows = object()
    bridge = object()
    return SimpleNamespace(
        adapter=object(),
        breadth=SimpleNamespace(all_rows=all_rows),
        bridge=bridge,
        blocks={"held": ("sealed-held",)},
        held_family_alias="held",
        frozen_compiled_block_states=torch.zeros(1, 2),
        compiled_correction_base_states=torch.zeros(1, 2),
        native_block_states=torch.zeros(1, 2),
        compiled_inputs=torch.zeros(1_798, 2),
        bases_by_node={name: object() for name in layer17_names},
        fragment_id_by_node={
            name: f"fragment.{index}"
            for index, name in enumerate(layer17_names)
        },
        layer10_graph=layer10_graph,
        layer17_graph=layer17_graph,
        layer10_lowerings_by_node={
            name: SimpleNamespace(artifact_sha256=_digest(30 + index))
            for index, name in enumerate(layer10_names)
        },
        layer17_lowerings_by_node={name: object() for name in layer17_names},
        source_composition_graph=composition,
        source_composition_lowerings=tuple(object() for _ in range(8)),
        target_token_locality_lineage_sha256=_digest(40),
        target_report_receipt={"receipt_sha256": _digest(41)},
        bridge_receipt={"receipt_sha256": _digest(42)},
        breadth_receipt={"receipt_sha256": _digest(43)},
        capture_sha256=_digest(44),
        capture_audit_sha256=_digest(45),
        capture=SimpleNamespace(
            native_rows=SimpleNamespace(row_key_sha256=_digest(48))
        ),
        row_catalog_sha256=_digest(46),
        training_family_aliases=tuple(f"family.{index}" for index in range(7)),
        training_example_ids=tuple(f"example.{index}" for index in range(28)),
        runtime_metadata={
            "model_id": runner.DEFAULT_MODEL_ID,
            "requested_revision": runner._EXPECTED_MODEL_REVISION,
            "model_fingerprint": _digest(47),
            "device": "cpu",
            "dtype": "float32",
            "local_files_only": True,
        },
    )


def test_prepublication_recovery_bypasses_a5c_auth_and_model_orchestration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "a5d.json"
    checkpoint = tmp_path / "a5d.prepublication-bundle.json"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    recovered = {"recovered": True}

    monkeypatch.setattr(
        runner,
        "default_a5d_prepublication_bundle_path",
        lambda _: checkpoint,
    )
    monkeypatch.setattr(
        runner,
        "finalize_a5d_prepublication_bundle",
        lambda path, *, output: (
            recovered
            if Path(path) == checkpoint
            else pytest.fail("wrong checkpoint")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_authenticate_canonical_a5c_file",
        lambda _: pytest.fail("A5c auth ran during recovery"),
    )
    monkeypatch.setattr(
        runner,
        "run_gemma3_l10_l17_a5c_broader_selected_generator",
        lambda **_: pytest.fail("model orchestration ran during recovery"),
    )

    assert runner.run_gemma3_l10_l17_a5d_broader_residual_generator(
        revision=runner._EXPECTED_MODEL_REVISION,
        output=output,
    ) == recovered


def test_continuation_uses_breadth_bridge_and_freezes_before_held(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = _workspace()
    canonical = _canonical_a5c()
    output = tmp_path / "a5d.json"
    a5c_path = tmp_path / "a5c.json"
    events: list[str] = []
    captured: dict[str, object] = {}
    target_receipt = {"receipt_sha256": _digest(50), "kind": "target"}
    cv_receipt = {"receipt_sha256": _digest(51), "kind": "cv"}
    residual_targets = _ReceiptObject(target_receipt)
    selection = _ReceiptObject(cv_receipt)
    executable = _Executable()
    executors = (object(), object(), object(), object())
    held_batches = (object(),)
    evaluation = {
        "outer_fold_index": 0,
        "logical_valid_tokens": 36,
        "supervised_tokens": 35,
        "conditions": {
            "frozen_uncorrected_composition": {
                **runner._EXPECTED_A5C_FROZEN_COMPOSITION,
                "owning_graph_sha256": _digest(90),
                "additive_graph_sha256": None,
            }
        },
    }

    file_hash_calls = 0

    def fake_file_sha(path: Path) -> str:
        nonlocal file_hash_calls
        assert Path(path) == a5c_path
        file_hash_calls += 1
        events.append(
            "file-auth" if file_hash_calls == 1 else "file-reauth"
        )
        return runner._EXPECTED_A5C_FILE_SHA256

    def fake_load(path: Path) -> dict[str, object]:
        assert Path(path) == a5c_path
        assert "score" in events
        events.append("report-parse")
        return canonical

    monkeypatch.setattr(runner, "_file_sha256", fake_file_sha)
    monkeypatch.setattr(
        runner, "load_gemma3_l10_l17_a5c_report", fake_load
    )

    def fake_a5c(**kwargs: object) -> dict[str, object]:
        assert kwargs["revision"] == runner._EXPECTED_MODEL_REVISION
        assert kwargs["output"] == output
        assert kwargs["device_name"] == "cpu"
        assert kwargs["dtype"] == "float32"
        continuation = kwargs["_continuation"]
        assert callable(continuation)
        return continuation(workspace)

    monkeypatch.setattr(
        runner,
        "run_gemma3_l10_l17_a5c_broader_selected_generator",
        fake_a5c,
    )

    def fake_targets(**kwargs: object) -> _ReceiptObject:
        events.append("target")
        assert kwargs["oracle_rows"] is workspace.breadth.all_rows
        assert kwargs["frozen_compiled_block_states"] is (
            workspace.frozen_compiled_block_states
        )
        return residual_targets

    monkeypatch.setattr(
        runner, "build_a5d_source_anchored_residual_targets", fake_targets
    )

    def fake_select(**kwargs: object) -> _ReceiptObject:
        events.append("select")
        assert kwargs["bridge"] is workspace.bridge
        assert kwargs["targets"] is residual_targets
        assert kwargs["ridge_grid"] == runner.A5D_RIDGE_GRID
        assert kwargs["alpha_grid"] == runner.A5D_ALPHA_GRID
        return selection

    monkeypatch.setattr(
        runner, "select_a5d_family_disjoint_residual", fake_select
    )

    def fake_freeze(**kwargs: object) -> _Executable:
        events.append("freeze")
        assert kwargs["selection"] is selection
        lineage = kwargs["lineage"]
        assert lineage["source_anchored_residual_receipt_sha256"] == _digest(50)
        assert lineage["residual_cv_receipt_sha256"] == _digest(51)
        assert lineage["coordinate_row_bank_receipt_sha256"] == _digest(42)
        captured["lineage"] = lineage
        return executable

    monkeypatch.setattr(runner, "freeze_a5d_executable", fake_freeze)

    def fake_executors(*args: object) -> tuple[object, ...]:
        events.append("executors")
        assert args[-1] is executable
        return executors

    monkeypatch.setattr(runner, "build_a5d_scoring_executors", fake_executors)

    def fake_held(**kwargs: object) -> tuple[object, ...]:
        events.append("held")
        assert "freeze" in events
        assert events.index("freeze") < events.index("held")
        assert kwargs["blocks"] is workspace.blocks
        assert kwargs["executable"] is executable
        return held_batches

    monkeypatch.setattr(
        runner, "select_a5d_held_scoring_batch_after_freeze", fake_held
    )

    def fake_score(**kwargs: object) -> dict[str, object]:
        events.append("score")
        assert events.index("held") < events.index("score")
        assert kwargs["batches"] is held_batches
        assert tuple(
            kwargs[name]
            for name in (
                "layer10_executor",
                "selected_layer17_executor",
                "frozen_composition_executor",
                "selected_composition_executor",
            )
        ) == executors
        return evaluation

    monkeypatch.setattr(
        runner, "score_a5d_source_anchored_residual_fold", fake_score
    )
    monkeypatch.setattr(
        runner,
        "compact_a5d_source_anchored_residual_receipt",
        lambda receipt: {
            "receipt_sha256": receipt["receipt_sha256"],
            "compact": "target",
        },
    )
    monkeypatch.setattr(
        runner,
        "compact_a5d_family_residual_cv_receipt",
        lambda receipt: {
            "receipt_sha256": receipt["receipt_sha256"],
            "compact": "cv",
        },
    )
    monkeypatch.setattr(
        runner,
        "a5d_outer_evaluation_sha256",
        lambda value: _digest(60) if value is evaluation else pytest.fail(),
    )

    def fake_publish(
        *, output: Path, report_inputs: dict[str, object]
    ) -> dict[str, object]:
        events.append("publish")
        assert output == Path(tmp_path / "a5d.json")
        captured["report_inputs"] = report_inputs
        return {"published": True}

    monkeypatch.setattr(
        runner, "publish_a5d_report_with_prepublication_bundle", fake_publish
    )
    monkeypatch.setattr(runner.gc, "collect", lambda: 0)

    result = runner.run_gemma3_l10_l17_a5d_broader_residual_generator(
        revision=runner._EXPECTED_MODEL_REVISION,
        output=output,
        a5c_path=a5c_path,
    )

    assert result == {"published": True}
    assert events == [
        "file-auth",
        "target",
        "select",
        "freeze",
        "executors",
        "held",
        "score",
        "file-reauth",
        "report-parse",
        "publish",
    ]
    report_inputs = captured["report_inputs"]
    assert isinstance(report_inputs, dict)
    assert set(report_inputs) == {
        "source_bindings",
        "runtime",
        "configuration",
        "capture",
        "source_anchored_residual",
        "residual_cv",
        "evidence_receipts",
        "selected_executable",
        "chronology",
        "outer_evaluation",
        "comparison_to_a5c",
    }
    assert report_inputs["evidence_receipts"] == {
        "source_anchored_residual": target_receipt,
        "residual_cv": cv_receipt,
    }
    assert report_inputs["selected_executable"] == executable.report_section()
    assert report_inputs["capture"]["source_row_catalog_sha256"] == _digest(48)
    assert report_inputs["comparison_to_a5c"] == {
        "a5c_file_sha256": runner._EXPECTED_A5C_FILE_SHA256,
        "a5c_report_sha256": runner._EXPECTED_A5C_REPORT_SHA256,
        "same_outer_fold": True,
        "same_held_example_policy": True,
        "a5c_frozen_composition": runner._EXPECTED_A5C_FROZEN_COMPOSITION,
    }
    assert report_inputs["chronology"] == {
        "residual_cv_completed_event": 1,
        "executable_frozen_event": 2,
        "outer_held_batch_selected_event": 3,
        "outer_held_model_evaluated_event": 4,
        "outer_held_batch_selected_or_scored_before_freeze": False,
        "executable_frozen_before_outer_held_batch_selection": True,
        "executable_frozen_before_outer_held_model_evaluation": True,
        "residual_cv_receipt_sha256": _digest(51),
        "selection_freeze_sha256": _digest(70),
        "outer_evaluation_sha256": _digest(60),
    }


@pytest.mark.parametrize(
    ("revision", "device", "dtype"),
    [
        ("0" * 40, "cpu", "float32"),
        (runner._EXPECTED_MODEL_REVISION, "mps", "float32"),
        (runner._EXPECTED_MODEL_REVISION, "cpu", "float16"),
    ],
)
def test_noncanonical_runtime_is_rejected_before_a5c_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    revision: str,
    device: str,
    dtype: str,
) -> None:
    monkeypatch.setattr(
        runner,
        "_authenticate_canonical_a5c_file",
        lambda _: pytest.fail("A5c was accessed"),
    )
    with pytest.raises(ValueError, match="canonical pinned"):
        runner.run_gemma3_l10_l17_a5d_broader_residual_generator(
            revision=revision,
            output=tmp_path / "a5d.json",
            device_name=device,
            dtype=dtype,
        )


def test_preflight_authenticates_only_file_bytes_without_report_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a5c.json"
    path.write_text("source", encoding="utf-8")
    monkeypatch.setattr(
        runner, "_file_sha256", lambda _: runner._EXPECTED_A5C_FILE_SHA256
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l10_l17_a5c_report",
        lambda _: pytest.fail("preflight parsed held A5c results"),
    )
    assert runner._authenticate_canonical_a5c_file(path) == path


def test_post_score_auth_requires_exact_frozen_metric_and_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "a5c.json"
    path.write_text("source", encoding="utf-8")
    value = _canonical_a5c()
    fresh = {
        "outer_fold_index": 0,
        "logical_valid_tokens": 36,
        "supervised_tokens": 35,
        "conditions": {
            "frozen_uncorrected_composition": {
                **runner._EXPECTED_A5C_FROZEN_COMPOSITION,
                "owning_graph_sha256": _digest(90),
                "additive_graph_sha256": None,
            }
        },
    }
    monkeypatch.setattr(
        runner, "_file_sha256", lambda _: runner._EXPECTED_A5C_FILE_SHA256
    )
    monkeypatch.setattr(
        runner, "load_gemma3_l10_l17_a5c_report", lambda _: value
    )
    assert runner._authenticate_canonical_a5c_after_outer_scoring(
        path, outer_evaluation=fresh
    ) is value

    broken = {
        **fresh,
        "conditions": {
            name: dict(condition)
            for name, condition in fresh["conditions"].items()
        },
    }
    broken["conditions"][
        "frozen_uncorrected_composition"
    ]["native_to_candidate_kl_per_token"] = 9.0
    with pytest.raises(ValueError, match="does not match canonical A5c"):
        runner._authenticate_canonical_a5c_after_outer_scoring(
            path, outer_evaluation=broken
        )

    wrong_policy = {**fresh, "supervised_tokens": 34}
    with pytest.raises(ValueError, match="does not match canonical A5c"):
        runner._authenticate_canonical_a5c_after_outer_scoring(
            path, outer_evaluation=wrong_policy
        )
