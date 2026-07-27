from __future__ import annotations

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_structured_mlp_dense_supermode_dev_experiment as experiment,
)
from fisher_graph.gemma3_structured_mlp_dense_supermode_dev_experiment import (
    GEMMA_DENSE_SUPERMODE_POOL_WIDTH,
    GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH,
    GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH,
    GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH,
    _BuiltCandidates,
    _DIAGONAL_DELETION,
    _DENSE,
    _NATIVE_PIVOT,
    _diagnostic_gate_report,
    _execute_fit_then_guard,
)
from fisher_graph.gemma3_structured_mlp_pseudo_unit_a_experiment import (
    _standard_thresholds,
)

from test_structured_mlp_compression import _executor


def test_frozen_dense_rung_shapes_are_consistent() -> None:
    assert GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH == 2048
    assert GEMMA_DENSE_SUPERMODE_POOL_WIDTH == 512
    assert GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH == 384
    assert GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH == 1920
    assert (
        GEMMA_DENSE_SUPERMODE_SOURCE_WIDTH
        - GEMMA_DENSE_SUPERMODE_POOL_WIDTH
        + GEMMA_DENSE_SUPERMODE_RETAINED_POOL_WIDTH
        == GEMMA_DENSE_SUPERMODE_RUNTIME_WIDTH
    )


def test_module_entrypoint_exposes_cli_help() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            (
                "fisher_graph."
                "gemma3_structured_mlp_dense_supermode_dev_experiment"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--parent-artifact" in completed.stdout
    assert "reused, nonconfirmatory v9 A-guard" in completed.stdout


def _built() -> _BuiltCandidates:
    dense = _executor(4, projection_bias=False, seed=12_001)
    pivot = _executor(4, projection_bias=False, seed=12_002)
    deletion = _executor(4, projection_bias=False, seed=12_003)
    return _BuiltCandidates(
        executors={
            _DENSE: dense,
            _NATIVE_PIVOT: pivot,
            _DIAGONAL_DELETION: deletion,
        },
        dense=None,  # type: ignore[arg-type]
        native_pivot_report={},
        deletion_report={},
        score_report={},
        plan_report={},
        structured_training_batches=1,
    )


def test_fit_freezes_extensible_candidate_names_before_guard() -> None:
    events: list[str] = []
    candidates = _built()

    fit, built, guard, result, frozen = _execute_fit_then_guard(
        materialize_fit=lambda: events.append("fit") or "fit",
        build_from_fit=lambda value: (
            events.append(f"build:{value}") or candidates
        ),
        materialize_guard=lambda: events.append("guard") or "guard",
        evaluate_guard=lambda value, split: (
            events.append(f"evaluate:{split}") or value
        ),
    )

    assert fit == "fit"
    assert built is candidates
    assert guard == "guard"
    assert result is candidates
    assert set(frozen) == {_DENSE, _NATIVE_PIVOT, _DIAGONAL_DELETION}
    assert events == ["fit", "build:fit", "guard", "evaluate:guard"]


def test_fit_guard_rejects_candidate_mutation() -> None:
    candidates = _built()

    def mutate(value: _BuiltCandidates, _guard: object) -> None:
        value.executors[_DENSE].feed_forward.gate_proj.weight.data.zero_()

    with pytest.raises(RuntimeError, match="mutated"):
        _execute_fit_then_guard(
            materialize_fit=lambda: "fit",
            build_from_fit=lambda _fit: candidates,
            materialize_guard=lambda: "guard",
            evaluate_guard=mutate,
        )


def _passing_evaluation(
    *,
    dense_block_nrmse: float,
    control_block_nrmse: float,
    pivot_block_nrmse: float = 0.013,
) -> dict[str, object]:
    names = (_DENSE, _NATIVE_PIVOT, _DIAGONAL_DELETION)
    behavior = {
        name: {
            "delta_nll_per_token": 0.0,
            "top1_agreement_to_baseline": 1.0,
            "teacher_kl_per_token": 0.0,
            "per_example_delta_nll_per_token": {"p90_absolute": 0.0},
            "per_example_top1_agreement": {"p10": 1.0},
        }
        for name in names
    }
    direct = {
        _DENSE: {
            "block_delta_nrmse": dense_block_nrmse,
            "block_delta_cosine": 1.0,
        },
        _NATIVE_PIVOT: {
            "block_delta_nrmse": pivot_block_nrmse,
            "block_delta_cosine": 1.0,
        },
        _DIAGONAL_DELETION: {
            "block_delta_nrmse": control_block_nrmse,
            "block_delta_cosine": 1.0,
        },
    }
    branch = {
        "attention_delta": {
            "block_delta_nrmse": 0.0,
            "block_delta_cosine": 1.0,
        },
        "feed_forward_delta": {
            "block_delta_nrmse": 0.0,
            "block_delta_cosine": 1.0,
        },
    }
    return {
        "behavior": behavior,
        "direct": direct,
        "branches": {name: branch for name in names},
        "execution_audits": {name: {"passed": True} for name in names},
        "ordinary_vs_segmented_native": {"passed": True},
        "native_boundary_replay": {"passed": True},
    }


def test_reused_guard_burn_filter_never_authorizes_heldout() -> None:
    gates = _diagnostic_gate_report(
        _passing_evaluation(
            dense_block_nrmse=0.010,
            control_block_nrmse=0.012,
        ),
        thresholds=_standard_thresholds(),
    )

    assert gates["dense_diagnostic_passed"] is True
    assert gates["any_candidate_authorizes_calibration_b"] is False
    for row in gates["candidates"].values():
        assert row["authorizes_calibration_b"] is False
        assert row["nonconfirmatory_reused_guard"] is True


@pytest.mark.parametrize(
    ("dense_nrmse", "control_nrmse"),
    (
        (0.016, 0.020),
        (0.010, 0.009),
        (0.010, 0.010),
        (0.010, 0.000),
    ),
)
def test_burn_filter_requires_margin_and_strict_control_win(
    dense_nrmse: float,
    control_nrmse: float,
) -> None:
    gates = _diagnostic_gate_report(
        _passing_evaluation(
            dense_block_nrmse=dense_nrmse,
            pivot_block_nrmse=0.020,
            control_block_nrmse=control_nrmse,
        ),
        thresholds=_standard_thresholds(),
    )

    assert gates["dense_diagnostic_passed"] is False


def test_ordinary_gate_status_is_distinct_from_control_burn_filter() -> None:
    gates = _diagnostic_gate_report(
        _passing_evaluation(
            dense_block_nrmse=0.010,
            pivot_block_nrmse=0.009,
            control_block_nrmse=0.008,
        ),
        thresholds=_standard_thresholds(),
    )

    assert gates["candidates"][_DENSE]["standard_passed"] is True
    assert gates["dense_diagnostic_passed"] is False
    assert experiment._dense_standard_gates_passed(gates) is True


def test_strict_loader_rejects_digest_valid_empty_semantics(
    tmp_path,
    monkeypatch,
) -> None:
    fingerprint = "1" * 64

    class FakeCandidate:
        def __init__(self, *, executor, artifact_state, report) -> None:
            self.executor = executor

    fake_executor = SimpleNamespace(
        execution_fingerprint=lambda: fingerprint,
    )
    monkeypatch.setattr(
        experiment.StructuredTransformerLayerExecutor,
        "from_artifact_state_dict",
        staticmethod(lambda *_args, **_kwargs: fake_executor),
    )
    monkeypatch.setattr(
        experiment,
        "StructuredMLPDenseSupermodeCandidate",
        FakeCandidate,
    )
    payload = {
        "schema": experiment.GEMMA_DENSE_SUPERMODE_DEV_SCHEMA,
        "format_version": (
            experiment.GEMMA_DENSE_SUPERMODE_DEV_FORMAT_VERSION
        ),
        "scientific_status": {},
        "model": {},
        "protocol": {},
        "parent": {},
        "calibration_a_fit": {},
        "reused_calibration_a_guard": {},
        "pipeline": {},
        "native_pivot_baseline": {},
        "deletion_baseline": {},
        "resource_report": {},
        "contains_source_model_weights": False,
        "contains_compressed_executor_weights": True,
        "contains_prompt_text": False,
        "contains_tokenizer_state": False,
        "contains_teacher_targets": False,
        "contains_fisher_taylor_scores": False,
        "executor": {},
    }
    scientific_digest = experiment._payload_sha256(payload)
    report = experiment._build_json_report(
        payload,
        tensor_file="bogus.pt",
        scientific_payload_sha256=scientific_digest,
        main_artifact_written=True,
    )
    report_digest = experiment._report_sha256(report)
    artifact_path = tmp_path / "bogus.pt"
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": scientific_digest,
            "report_sha256": report_digest,
        },
        artifact_path,
    )
    artifact_path.with_suffix(".json").write_text(
        json.dumps({**report, "report_sha256": report_digest}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tokenized stream"):
        experiment.load_gemma3_dense_supermode_dev_artifact(artifact_path)


def test_json_sibling_must_be_derived_from_scientific_payload(
    tmp_path,
) -> None:
    source = tmp_path / "candidate.pt"
    payload = {
        "schema": experiment.GEMMA_DENSE_SUPERMODE_DEV_SCHEMA,
        "scientific_status": {"outcome": "real"},
        "executor": {},
    }
    scientific_digest = "2" * 64
    expected = experiment._build_json_report(
        payload,
        tensor_file=source.name,
        scientific_payload_sha256=scientific_digest,
        main_artifact_written=True,
    )
    expected_digest = experiment._report_sha256(expected)
    source.with_suffix(".json").write_text(
        json.dumps({**expected, "report_sha256": expected_digest}),
        encoding="utf-8",
    )
    loaded = experiment._load_and_validate_json_sibling(
        source,
        payload,
        scientific_payload_sha256=scientific_digest,
        report_sha256=expected_digest,
    )
    assert loaded["scientific_status"] == {"outcome": "real"}

    forged = copy.deepcopy(expected)
    forged["scientific_status"] = {"outcome": "forged"}
    forged_digest = experiment._report_sha256(forged)
    source.with_suffix(".json").write_text(
        json.dumps({**forged, "report_sha256": forged_digest}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sibling binding"):
        experiment._load_and_validate_json_sibling(
            source,
            payload,
            scientific_payload_sha256=scientific_digest,
            report_sha256=forged_digest,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("layer_index", 3),
        ("max_length", 128),
        ("tokenization_batch_size", 2),
        ("device_name", "mps"),
        ("dtype", "bfloat16"),
    ),
)
def test_runner_rejects_protocol_drift_before_external_access(
    tmp_path,
    name: str,
    value: object,
) -> None:
    kwargs = {
        "parent_artifact_path": tmp_path / "parent.pt",
        "prompt_splits_path": tmp_path / "prompts.json",
        "family_manifest_path": tmp_path / "families.json",
        "corpus_audit_path": tmp_path / "audit.json",
        "revision": "a" * 40,
        "output": tmp_path / "candidate.pt",
        name: value,
    }

    with pytest.raises(ValueError, match="requires"):
        experiment.run_gemma3_structured_mlp_dense_supermode_dev_experiment(
            **kwargs,
        )
