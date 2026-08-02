from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fisher_graph import (
    gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development as runner,
)


def _panel_payload() -> dict[str, object]:
    prompts = [f"synthetic calibration A prompt {index}" for index in range(16)]
    return {
        "schema": "fisher_graph.local_v9_a_fit_development_export",
        "format_version": 1,
        "source_corpus_id": "structured-strong-v9",
        "source_role": "calibration_a_fit_only",
        "scientific_status": "development_only",
        "selection_rule": "first_16_authenticated_fit_partition_positions",
        "calibration_b_exported": False,
        "guard_exported": False,
        "validation_exported": False,
        "test_exported": False,
        "model_or_tokenizer_accessed": False,
        "prompts": prompts,
        "family_ids": [f"family_{index // 2}" for index in range(16)],
        "fit_positions": list(range(16)),
        "source_prompt_indices": list(range(100, 116)),
        "prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in prompts
        ],
        "source_fit_prompt_index_sha256": "a" * 64,
    }


def _write_panel(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    payload = _panel_payload()
    if mutate is not None:
        mutate(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_PANEL_FILE_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return payload


def test_load_panel_accepts_only_balanced_calibration_a_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic-a-fit.json"
    payload = _write_panel(source, monkeypatch)

    examples, receipt = runner._load_panel(source)

    assert len(examples) == 16
    assert tuple(example.example_id for example in examples) == tuple(
        f"a_fit_{index:03d}" for index in range(16)
    )
    assert tuple(example.family_id for example in examples) == tuple(
        payload["family_ids"]
    )
    assert tuple(example.prompt for example in examples) == tuple(
        payload["prompts"]
    )
    assert receipt == {
        "file_sha256": runner._EXPECTED_PANEL_FILE_SHA256,
        "schema": "fisher_graph.local_v9_a_fit_development_export",
        "source_corpus_id": "structured-strong-v9",
        "source_role": "calibration_a_fit_only",
        "source_fit_prompt_index_sha256": "a" * 64,
        "example_count": 16,
        "family_count": 8,
        "examples_per_family": 2,
        "prompt_sha256s": tuple(payload["prompt_sha256"]),
        "source_prompt_indices": tuple(range(100, 116)),
        "contains_prompt_text": False,
        "calibration_b_exported": False,
        "validation_exported": False,
        "test_exported": False,
    }
    assert not any(
        prompt in json.dumps(receipt)
        for prompt in payload["prompts"]
    )


def test_load_panel_authenticates_file_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic-a-fit.json"
    source.write_text("not JSON\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_EXPECTED_PANEL_FILE_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="panel file differs"):
        runner._load_panel(source)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("source_role", "calibration_b"),
        lambda value: value.__setitem__("scientific_status", "qualification"),
        lambda value: value.__setitem__("calibration_b_exported", True),
        lambda value: value.__setitem__("guard_exported", True),
        lambda value: value.__setitem__("validation_exported", True),
        lambda value: value.__setitem__("test_exported", True),
        lambda value: value.__setitem__("model_or_tokenizer_accessed", True),
        lambda value: value.__setitem__(
            "family_ids",
            ["family_0"] * 16,
        ),
        lambda value: value["family_ids"].__setitem__(0, ""),
        lambda value: value["fit_positions"].__setitem__(0, "0"),
        lambda value: value["fit_positions"].__setitem__(1, 0),
        lambda value: value["source_prompt_indices"].__setitem__(0, True),
        lambda value: value["prompts"].__setitem__(1, value["prompts"][0]),
        lambda value: value["prompt_sha256"].__setitem__(0, "0" * 64),
    ),
    ids=(
        "wrong-source-role",
        "wrong-scientific-status",
        "calibration-b-exported",
        "guard-exported",
        "validation-exported",
        "test-exported",
        "model-or-tokenizer-accessed",
        "unbalanced-families",
        "empty-family",
        "noninteger-fit-position",
        "duplicate-fit-position",
        "boolean-source-index",
        "duplicate-prompt",
        "prompt-hash-drift",
    ),
)
def test_load_panel_rejects_protocol_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    source = tmp_path / "synthetic-a-fit.json"
    _write_panel(source, monkeypatch, mutate=mutate)

    with pytest.raises(ValueError, match="panel protocol differs"):
        runner._load_panel(source)


def test_output_must_be_json_below_local_runs(tmp_path: Path) -> None:
    valid = tmp_path / ".local-runs" / "shadow.json"

    assert runner._validate_output(valid) == valid
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output(tmp_path / "shadow.json")
    with pytest.raises(ValueError, match="JSON under .local-runs"):
        runner._validate_output(tmp_path / ".local-runs" / "shadow.pt")


def test_frozen_tokenizer_guard_accepts_only_initial_then_post_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = {"bytes": 10, "sha256": "a" * 64}
    post = {"bytes": 11, "sha256": "b" * 64}
    monkeypatch.setattr(runner, "_EXPECTED_A_FIT_TOKENIZER_POST_BYTES", 11)
    monkeypatch.setattr(
        runner,
        "_EXPECTED_A_FIT_TOKENIZER_POST_SHA256",
        "b" * 64,
    )
    states = iter((initial, post, post, post))
    monkeypatch.setattr(
        runner,
        "_tokenizer_backend_identity",
        lambda _tokenizer: next(states),
    )
    guard = runner._frozen_tokenizer_integrity_check(
        object(),
        {
            "backend_serialized_bytes": initial["bytes"],
            "backend_serialized_sha256": initial["sha256"],
            "post_tokenization_backend_serialized_bytes": post["bytes"],
            "post_tokenization_backend_serialized_sha256": post["sha256"],
        },
    )

    guard("before")
    guard("after")
    guard("before")
    guard("after")


def test_frozen_tokenizer_guard_rejects_backend_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_EXPECTED_A_FIT_TOKENIZER_POST_BYTES", 11)
    monkeypatch.setattr(
        runner,
        "_EXPECTED_A_FIT_TOKENIZER_POST_SHA256",
        "b" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_tokenizer_backend_identity",
        lambda _tokenizer: {"bytes": 99, "sha256": "c" * 64},
    )
    guard = runner._frozen_tokenizer_integrity_check(
        object(),
        {
            "backend_serialized_bytes": 10,
            "backend_serialized_sha256": "a" * 64,
            "post_tokenization_backend_serialized_bytes": 11,
            "post_tokenization_backend_serialized_sha256": "b" * 64,
        },
    )

    with pytest.raises(ValueError, match="drifted before"):
        guard("before")


@pytest.mark.parametrize("max_length", (True, 1, 9, 257, 128.0))
def test_invalid_max_length_is_rejected_before_panel_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_length: object,
) -> None:
    monkeypatch.setattr(
        runner,
        "_load_panel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid arguments must not open the panel")
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"max_length must lie in \[10, 256\]",
    ):
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development(
            output=tmp_path / ".local-runs" / "shadow.json",
            max_length=max_length,  # type: ignore[arg-type]
        )


def test_shadow_orchestration_is_source_authoritative_and_publishes_metrics_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = tmp_path / "synthetic-a-fit.json"
    panel_payload = _write_panel(panel, monkeypatch)
    output = tmp_path / ".local-runs" / "shadow.json"
    events: list[object] = []
    candidate = SimpleNamespace(
        artifact_sha256="b" * 64,
        plan=SimpleNamespace(artifact_sha256="c" * 64),
        method="signed_graph_wavelet_local_svd_g8",
        model={
            "model_id": "synthetic/gemma",
            "resolved_commit": "d" * 40,
            "source_model_sha256": runner._EXPECTED_RAW_MODEL_SHA256,
        },
    )
    basis = object()
    tokenizer = object()
    runtime = object()
    evaluation = {
        "behavioral": {"source_nll": 1.0, "candidate_nll": 1.1},
        "affected_behavioral": {"top1_agreement": 0.9},
        "target_modal": {"relative_l2_error": 0.1},
        "full_width_boundary": {"relative_l2_error": 0.2},
        "coverage": {"example_count": 16, "model_forward_count": 48},
    }

    class FakeAdapter:
        factorized = False

        def model_fingerprint(self) -> str:
            if self.factorized:
                return runner._EXPECTED_FACTORIZED_MODEL_SHA256
            return runner._EXPECTED_RAW_MODEL_SHA256

        def execution_fingerprint(self) -> str:
            assert self.factorized
            return runner._EXPECTED_FACTORIZED_EXECUTION_SHA256

    adapter = FakeAdapter()

    class FakeSwitcher:
        def __init__(
            self,
            received_adapter: object,
            scopes: object,
        ) -> None:
            assert received_adapter is adapter
            assert scopes == {runner._FACTORIZED_SCOPE: ("replacement",)}
            events.append("switcher-created")

        def switch(self, scope: str) -> None:
            assert scope == runner._FACTORIZED_SCOPE
            adapter.factorized = True
            events.append("factorized")

        def close(self) -> None:
            adapter.factorized = False
            events.append("restored")

    class FakeRuntimeFactory:
        @classmethod
        def from_signed_g8_candidate(
            cls,
            received_candidate: object,
            received_basis: object,
            **kwargs: object,
        ) -> object:
            assert received_candidate is candidate
            assert received_basis is basis
            assert kwargs == {
                "expected_basis_payload_sha256": (
                    runner.DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
                ),
                "expected_live_model_sha256": (
                    runner._EXPECTED_FACTORIZED_MODEL_SHA256
                ),
                "expected_adapter_execution_sha256": (
                    runner._EXPECTED_FACTORIZED_EXECUTION_SHA256
                ),
                "analysis_device": "cpu",
            }
            events.append("runtime-built")
            return runtime

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        assert adapter.factorized
        assert kwargs["runtime"] is runtime
        assert kwargs["adapter"] is adapter
        assert kwargs["tokenizer"] is tokenizer
        assert kwargs["max_length"] == 96
        assert kwargs["model_input_device"] == "cpu"
        assert callable(kwargs["tokenizer_integrity_check"])
        examples = kwargs["examples"]
        assert len(examples) == 16
        assert tuple(example.prompt for example in examples) == tuple(
            panel_payload["prompts"]
        )
        events.append("source-authoritative-evaluation")
        return evaluation

    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_graph_wavelet_signed_g8_candidate",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        runner,
        "load_gemma3_l3_l4_basis_package",
        lambda *_args, **_kwargs: basis,
    )
    monkeypatch.setattr(
        runner,
        "default_gemma3_l3_l4_graph_organized_svd_shadow_protocol",
        lambda: "frozen-tokenizer-protocol",
    )
    monkeypatch.setattr(
        runner,
        "_load_and_validate_frozen_local_tokenizer",
        lambda **kwargs: (
            tokenizer,
            {
                "tokenizer_class": "SyntheticTokenizer",
                "configuration_sha256": "e" * 64,
                "backend_serialized_bytes": 10,
                "backend_serialized_sha256": "f" * 64,
                "post_tokenization_backend_serialized_bytes": 11,
                "post_tokenization_backend_serialized_sha256": "0" * 64,
            },
        )
        if kwargs == {"protocol": "frozen-tokenizer-protocol"}
        else (_ for _ in ()).throw(AssertionError("tokenizer protocol drift")),
    )
    monkeypatch.setattr(
        runner,
        "resolve_gemma3_huggingface_paths",
        lambda cache_dir: {"hub_cache": cache_dir},
    )
    monkeypatch.setattr(runner, "resolve_torch_device", lambda value: value)
    monkeypatch.setattr(
        runner,
        "_load_local_gemma3_model_only",
        lambda **kwargs: events.append(("model-loaded", kwargs)) or object(),
    )
    monkeypatch.setattr(runner, "Gemma3CausalLMAdapter", lambda _model: adapter)
    monkeypatch.setattr(
        runner,
        "restore_gemma3_full_mlp_stack_refit_runtime",
        lambda *_args: SimpleNamespace(replacements=("replacement",)),
    )
    monkeypatch.setattr(
        runner,
        "PreparedGemma3FullMLPStackSwitcher",
        FakeSwitcher,
    )
    monkeypatch.setattr(
        runner,
        "Gemma3L3L4ConditionalSpectralShadowRuntime",
        FakeRuntimeFactory,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_gemma3_l3_l4_conditional_spectral_development_shadow",
        fake_evaluate,
    )

    result = (
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development(
            candidate_artifact_path="candidate.pt",
            basis_package_path="basis.pt",
            base_artifact_path="base.pt",
            refit_artifact_path="refit.pt",
            panel_path=panel,
            output=output,
            cache_dir=tmp_path / "cache",
            max_length=96,
        )
    )

    assert events[-4:] == [
        "factorized",
        "runtime-built",
        "source-authoritative-evaluation",
        "restored",
    ]
    assert adapter.factorized is False
    assert result["evaluation"] == evaluation
    assert result["scientific_status"] == {
        "development_smoke_complete": True,
        "source_outputs_authoritative": True,
        "candidate_outputs_used_for_metrics_only": True,
        "calibration_b_opened": False,
        "validation_opened": False,
        "test_opened": False,
        "formal_qualification": False,
        "candidate_serving_authorized": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
    }
    assert result["safety"]["contains_prompt_text"] is False
    assert result["artifact"]["file_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    published = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(published, sort_keys=True)
    assert published["evaluation"] == evaluation
    assert published["panel"]["prompt_sha256s"] == panel_payload[
        "prompt_sha256"
    ]
    assert not any(prompt in serialized for prompt in panel_payload["prompts"])
    assert '"logits":' not in serialized
    assert '"token_ids":' not in serialized

    failure_output = tmp_path / ".local-runs" / "shadow-failure.json"
    monkeypatch.setattr(
        runner,
        "evaluate_gemma3_l3_l4_conditional_spectral_development_shadow",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic shadow failure")
        ),
    )
    with pytest.raises(RuntimeError, match="synthetic shadow failure"):
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development(
            candidate_artifact_path="candidate.pt",
            basis_package_path="basis.pt",
            base_artifact_path="base.pt",
            refit_artifact_path="refit.pt",
            panel_path=panel,
            output=failure_output,
            cache_dir=tmp_path / "cache",
            max_length=96,
        )
    assert adapter.factorized is False
    assert events[-2:] == ["runtime-built", "restored"]
    assert not failure_output.exists()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_gemma3_l3_l4_graph_wavelet_signed_g8_shadow_development(
            panel_path=panel,
            output=output,
        )
