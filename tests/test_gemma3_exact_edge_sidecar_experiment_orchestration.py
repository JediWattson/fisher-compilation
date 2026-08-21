from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from fisher_graph import gemma3_exact_edge_sidecar_experiment as experiment


class _Guard:
    def __init__(self, _: object) -> None:
        pass

    def assert_unchanged(self) -> None:
        pass


class _Adapter:
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.layers = tuple(range(18))

    def model_fingerprint(self) -> str:
        return "a" * 64

    def execution_fingerprint(self) -> str:
        return "b" * 64


class _Split:
    def examples_for(self, role: str) -> tuple[SimpleNamespace, ...]:
        return (SimpleNamespace(example_id=role, family_id="family", prompt="x"),)


class _Catalog:
    artifact_sha256 = "c" * 64

    def assert_integrity(self) -> None:
        pass


class _CausalReceipt:
    protocol_sha256 = experiment.EXPECTED_PROTOCOL_SHA256
    parent_edge_catalog_sha256 = "c" * 64
    parent_predictive_receipt_sha256 = "d" * 64
    C_row_identity_sha256 = "e" * 64
    source_fingerprint_before = "a" * 64


def _patch_public_run(monkeypatch, *, predictive_passed: bool):
    model = nn.Linear(1, 1, bias=False)
    compilation = SimpleNamespace(executor=object())
    catalog = _Catalog()
    predictive = SimpleNamespace(
        passed=predictive_passed,
        receipt_sha256="d" * 64,
    )
    captures = {
        role: SimpleNamespace(
            role=role,
            capture=SimpleNamespace(row_identity_sha256=(
                "e" * 64 if role == "C_intervention" else "f" * 64
            )),
        )
        for role in experiment.ROLE_ORDER
    }
    assembled_causal: list[object | None] = []
    monkeypatch.setattr(experiment, "PINNED_SOURCE_PARAMETER_COUNT", 1)
    monkeypatch.setattr(
        experiment,
        "_protocol_receipt",
        lambda _: ({}, {"protocol_sha256": experiment.EXPECTED_PROTOCOL_SHA256}),
    )
    monkeypatch.setattr(experiment, "load_pinned_edge_sidecar_split", lambda _: _Split())
    monkeypatch.setattr(experiment, "resolve_torch_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(
        experiment,
        "resolve_gemma3_huggingface_paths",
        lambda _: {"hub_cache": "/tmp"},
    )
    monkeypatch.setattr(experiment, "load_gemma3", lambda **_: (object(), model))
    monkeypatch.setattr(experiment, "Gemma3CausalLMAdapter", _Adapter)
    monkeypatch.setattr(experiment, "_FrozenModelTensorGuard", _Guard)
    monkeypatch.setattr(
        experiment,
        "compile_gemma3_joint_exact_control",
        lambda *_, **__: compilation,
    )
    monkeypatch.setattr(experiment, "_prepare_role_batches", lambda **_: (object(),))
    monkeypatch.setattr(
        experiment,
        "capture_exact_edge_sidecar_role",
        lambda **kwargs: captures[kwargs["role"]],
    )
    monkeypatch.setattr(experiment, "fit_exact_edge_sidecar_catalog", lambda _: catalog)
    monkeypatch.setattr(
        experiment,
        "evaluate_exact_edge_sidecar_predictive",
        lambda **_: predictive,
    )
    monkeypatch.setattr(
        experiment,
        "EdgeSidecarCausalRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(experiment, "EdgeSidecarCausalReceipt", _CausalReceipt)
    monkeypatch.setattr(experiment, "_model_provenance", lambda *_, **__: {})
    monkeypatch.setattr(experiment, "_peak_resident_bytes", lambda: 1)

    def assemble(**kwargs):
        assembled_causal.append(kwargs["causal"])
        return {"report_sha256": "1" * 64, "qualification": {}}

    monkeypatch.setattr(experiment, "_assemble_report", assemble)
    monkeypatch.setattr(
        experiment,
        "write_exact_edge_sidecar_experiment_report",
        lambda report, destination: report,
    )
    return assembled_causal


def test_public_run_invokes_injected_C_evaluator_after_B_pass(monkeypatch, tmp_path) -> None:
    assembled_causal = _patch_public_run(monkeypatch, predictive_passed=True)
    callback = Mock(return_value=_CausalReceipt())

    report = experiment.run_gemma3_exact_edge_sidecar_experiment(
        output=tmp_path / "report.json",
        causal_evaluator=callback,
    )

    assert report["report_sha256"] == "1" * 64
    callback.assert_called_once()
    assert isinstance(assembled_causal[0], _CausalReceipt)


def test_public_run_never_invokes_C_evaluator_when_B_fails(monkeypatch, tmp_path) -> None:
    assembled_causal = _patch_public_run(monkeypatch, predictive_passed=False)
    callback = Mock(side_effect=AssertionError("C callback must remain untouched"))

    experiment.run_gemma3_exact_edge_sidecar_experiment(
        output=tmp_path / "report.json",
        causal_evaluator=callback,
    )

    callback.assert_not_called()
    assert assembled_causal == [None]
