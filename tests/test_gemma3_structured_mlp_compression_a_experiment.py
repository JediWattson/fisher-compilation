from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from fisher_graph import (
    gemma3_structured_mlp_compression_a_experiment as experiment,
)
from fisher_graph.structured_mlp_compression_pipeline import (
    STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION,
    STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA,
    StructuredMLPFirstRungCandidate,
)

from test_structured_mlp_compression import _executor


_REVISION = "a" * 40


class _AOnlyPrompts:
    calibration_a = tuple(
        f"calibration A only {index}" for index in range(256)
    )

    @property
    def calibration_b(self) -> object:
        raise AssertionError("calibration B must not be inspected")

    @property
    def validation(self) -> object:
        raise AssertionError("validation must not be inspected")

    @property
    def test(self) -> object:
        raise AssertionError("test must not be inspected")


def _resource_construction(executor: object) -> dict[str, object]:
    residual_width = executor.width
    source_width = 2_048
    retained_width = 1_536
    removed_width = source_width - retained_width
    removed_parameters = removed_width * 3 * residual_width
    source_parameters = (
        executor.learned_parameter_count + removed_parameters
    )
    source_macs = 3 * residual_width * source_width
    compressed_macs = 3 * residual_width * retained_width
    return {
        "rung": {
            "source_intermediate_width": source_width,
            "retained_intermediate_width": retained_width,
            "removed_intermediate_width": removed_width,
        },
        "parameters": {
            "source_full_layer": source_parameters,
            "compressed_full_layer": executor.learned_parameter_count,
            "removed_full_layer": removed_parameters,
            "expected_removed_from_mlp_slices": removed_parameters,
            "retained_ratio": (
                executor.learned_parameter_count / source_parameters
            ),
        },
        "compute_per_valid_token": {
            "scope": (
                "gate_up_down_linear_weight_matmuls_only; "
                "nonlinear_activation_flops_excluded_as "
                "implementation_dependent; attention_and_norm_unchanged"
            ),
            "macs": {
                "source": source_macs,
                "compressed": compressed_macs,
                "removed": source_macs - compressed_macs,
            },
            "flops_two_per_mac": {
                "source": 2 * source_macs,
                "compressed": 2 * compressed_macs,
                "removed": 2 * (source_macs - compressed_macs),
            },
            "bias_additions": {
                "source": 0,
                "compressed": 0,
                "removed": 0,
            },
            "gate_up_multiplications": {
                "source": source_width,
                "compressed": retained_width,
                "removed": removed_width,
            },
            "nonlinear_activation_elements": {
                "source": source_width,
                "compressed": retained_width,
                "removed": removed_width,
            },
        },
    }


def _fidelity(
    executor: object,
    *,
    passed: bool,
) -> dict[str, object]:
    direct = {
        "sequences": 256,
        "block_delta_nrmse": 0.0 if passed else 1.0,
        "block_delta_cosine": 1.0,
    }
    branch = {
        "block_delta_nrmse": 0.0,
        "block_delta_cosine": 1.0,
    }
    gates = {
        "direct": {
            "block_delta_nrmse": passed,
            "block_delta_cosine": True,
        },
        "branches": {
            "attention_delta_nrmse": True,
            "attention_delta_cosine": True,
            "feed_forward_delta_nrmse": True,
            "feed_forward_delta_cosine": True,
        },
        "passed": passed,
    }
    return {
        "split": "calibration_a",
        "evaluation_scope": (
            "aggregate_direct_fidelity_no_suffix_behavior"
        ),
        "source_layer_calls": 0,
        "executor_fingerprints": {
            "structured_source_visibility": (
                executor.execution_fingerprint()
            )
        },
        "direct": {"structured_source_visibility": direct},
        "branches": {
            "structured_source_visibility": {
                "attention_delta": branch,
                "feed_forward_delta": branch,
            }
        },
        "gates": {"structured_source_visibility": gates},
        "primary_passed": passed,
    }


def _run_mocked(
    tmp_path: Path,
    *,
    passed: bool,
) -> tuple[Path, list[str]]:
    parent = _executor(2_048, projection_bias=False, seed=501).eval()
    compressed = _executor(1_536, projection_bias=False, seed=502).eval()
    parent_fingerprint = parent.execution_fingerprint()
    construction = _resource_construction(compressed)
    candidate_report = {
        "schema": STRUCTURED_MLP_FIRST_RUNG_PIPELINE_SCHEMA,
        "format_version": (
            STRUCTURED_MLP_FIRST_RUNG_PIPELINE_FORMAT_VERSION
        ),
        "report_sha256": "d" * 64,
        "parent_authentication": {
            "execution_fingerprint": parent_fingerprint,
        },
        "selection": {
            "algorithm": "test-fisher-taylor",
            "source_width": 2_048,
            "retained_width": 1_536,
            "valid_rows": 10_000,
            "input_batches_sha256": "1" * 64,
            "unit_scores_sha256": "2" * 64,
            "retained_score_fraction": 0.96,
            "selection_sha256": "3" * 64,
        },
        "terminal_projection_refit": {
            "algorithm": "activation_only_mlp_down_projection_ridge_v1",
        },
        "final_candidate": {
            "execution_fingerprint": (
                compressed.execution_fingerprint()
            ),
        },
        "construction": construction,
    }
    candidate = StructuredMLPFirstRungCandidate(
        executor=compressed,
        artifact_state=compressed.artifact_state_dict(),
        report=candidate_report,
    )
    roles = ("calibration_a", "calibration_b", "validation", "test")
    role_counts = {
        "calibration_a": 256,
        "calibration_b": 64,
        "validation": 64,
        "test": 64,
    }
    prompt_hashes = {
        role: [
            hashlib.sha256(f"prompt:{role}:{index}".encode()).hexdigest()
            for index in range(role_counts[role])
        ]
        for role in roles
    }
    family_hashes = {
        role: [
            hashlib.sha256(f"family:{role}".encode()).hexdigest()
            for _ in range(role_counts[role])
        ]
        for role in roles
    }
    source_corpus = {
        "corpus_id": "structured-strong-v6",
        "prompt_status": experiment.PROMPT_STATUS,
        "family_status": experiment.FAMILY_STATUS,
        "counts": role_counts,
        "prompt_sha256_by_role": prompt_hashes,
        "family_sha256_by_role": family_hashes,
        "ordered_prompt_sha256_by_role": {
            role: experiment._ordered_sha256(prompt_hashes[role])
            for role in roles
        },
        "ordered_family_sha256_by_role": {
            role: experiment._ordered_sha256(family_hashes[role])
            for role in roles
        },
        "corpus_audit_payload_sha256": "5" * 64,
        "prompt_fixture_file_sha256": "6" * 64,
        "family_manifest_file_sha256": "7" * 64,
    }
    stream = {
        "schema": "fisher_graph.tokenized_calibration_stream",
        "format_version": 2,
        "split": "calibration_a",
        "batches": 1,
        "sequences": 256,
        "serialized_sha256": "c" * 64,
        "source_prompt_sha256": prompt_hashes["calibration_a"],
        "examples": [],
    }
    corpus = experiment._StaticCorpus(
        prompts=_AOnlyPrompts(),
        prompt_metadata={"counts": role_counts},
        family_metadata={"counts": role_counts},
        audit_binding={"audit_payload_sha256": "5" * 64},
        source_corpus=source_corpus,
    )
    parent_training = {
        "bootstrap": {
            "layer_id": "layer.0",
            "calibration_split_sha256": "c" * 64,
        },
    }
    parent_protocol = {"tokenized_splits": {"calibration_a": stream}}
    parent_loaded = {
        "metadata": {
            "tensor_file_sha256": "8" * 64,
            "scientific_payload_sha256": "9" * 64,
            "report_sha256": "a" * 64,
        },
        "model": {
            "resolved_commit": _REVISION,
            "config_sha256": "b" * 64,
        },
    }
    score_report = {
        "schema": "fisher_graph.structured_mlp_fisher_taylor_collection",
        "format_version": 1,
        "objective": {"name": "test"},
        "provenance": {"calibration_split_sha256": "c" * 64},
        "accounting": {
            "valid_rows": 10_000,
            "batch_count": 1,
        },
        "source_audit": {"source_model_executed": True},
        "heldout_opened": False,
        "collection_sha256": "e" * 64,
    }
    model = nn.Linear(1, 1).eval().requires_grad_(False)
    adapter = SimpleNamespace(
        plan_layer_block=lambda _start, _end: SimpleNamespace(
            layer_ids=("layer.0",)
        )
    )
    materialized_splits: list[str] = []

    def materialize(
        _tokenizer: object,
        prompt_values: object,
        *,
        split_name: str,
        **_kwargs: object,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        assert prompt_values == _AOnlyPrompts.calibration_a
        materialized_splits.append(split_name)
        return (object(),), stream

    output = tmp_path / "candidate.pt"
    with (
        patch.object(
            experiment,
            "_load_structured_v6_corpus_preflight",
            return_value=corpus,
        ),
        patch.object(
            experiment,
            "load_gemma3_structured_single_layer_artifact",
            return_value=parent_loaded,
        ),
        patch.object(
            experiment,
            "_authenticate_parent",
            return_value=(parent, parent_training, parent_protocol),
        ),
        patch.object(
            experiment,
            "load_gemma3",
            return_value=(SimpleNamespace(), model),
        ),
        patch.object(
            experiment,
            "_model_provenance",
            return_value={
                "resolved_commit": _REVISION,
                "config_sha256": "b" * 64,
            },
        ),
        patch.object(
            experiment,
            "Gemma3CausalLMAdapter",
            return_value=adapter,
        ),
        patch.object(experiment, "_materialize_split", side_effect=materialize),
        patch.object(
            experiment,
            "_tokenized_stream_contract",
            return_value={"valid_tokens": 10_000},
        ),
        patch.object(
            experiment,
            "collect_structured_training_batches",
            return_value=(
                SimpleNamespace(targets=object(), batch=object()),
            ),
        ),
        patch.object(
            experiment,
            "_require_complete_middle_layer_demand",
        ),
        patch.object(
            experiment,
            "collect_gemma_mlp_fisher_taylor_batches",
            return_value=((object(),), score_report),
        ),
        patch.object(
            experiment,
            "build_gemma_mlp_first_rung_candidate",
            return_value=candidate,
        ),
        patch.object(
            experiment,
            "evaluate_calibration_a_fidelity",
            return_value=_fidelity(compressed, passed=passed),
        ),
    ):
        if passed:
            experiment.run_gemma3_structured_mlp_compression_a_experiment(
                parent_artifact_path=tmp_path / "parent.pt",
                prompt_splits_path=tmp_path / "prompts.json",
                family_manifest_path=tmp_path / "families.json",
                corpus_audit_path=tmp_path / "audit.json",
                revision=_REVISION,
                model_id="test-model",
                layer_index=0,
                output=output,
            )
        else:
            with pytest.raises(
                RuntimeError,
                match="failed calibration-A fidelity",
            ):
                experiment.run_gemma3_structured_mlp_compression_a_experiment(
                    parent_artifact_path=tmp_path / "parent.pt",
                    prompt_splits_path=tmp_path / "prompts.json",
                    family_manifest_path=tmp_path / "families.json",
                    corpus_audit_path=tmp_path / "audit.json",
                    revision=_REVISION,
                    model_id="test-model",
                    layer_index=0,
                    output=output,
                )
    return output, materialized_splits


def test_a_only_candidate_persists_and_strict_roundtrips(
    tmp_path: Path,
) -> None:
    output, splits = _run_mocked(tmp_path, passed=True)

    assert splits == ["calibration_a"]
    loaded = (
        experiment.load_gemma3_structured_mlp_compression_a_artifact(
            output
        )
    )
    assert loaded["scientific_status"][
        "scientific_compression_success"
    ] is False
    assert loaded["protocol"]["heldout_tokenized"] is False
    assert loaded["protocol"]["heldout_ledger_created"] is False
    assert (
        loaded["executor"]
        .config.transformer.feed_forward.intermediate_width
        == 1_536
    )
    assert loaded["resource_report"]["parameters"][
        "removed_full_layer"
    ] > 0
    raw = torch.load(output, map_location="cpu", weights_only=True)
    assert raw["contains_prompt_text"] is False
    assert raw["contains_teacher_targets"] is False
    assert raw["contains_fisher_taylor_scores"] is False

    tampered_path = tmp_path / "candidate-tampered.pt"
    raw["resource_report"]["parameters"]["removed_full_layer"] += 1
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    raw["scientific_payload_sha256"] = experiment._payload_sha256(
        payload
    )
    tampered_report = experiment._build_report(
        payload,
        tensor_file=tampered_path.name,
        scientific_payload_sha256=raw["scientific_payload_sha256"],
    )
    raw["report_sha256"] = experiment._report_sha256(tampered_report)
    torch.save(raw, tampered_path)
    tampered_path.with_suffix(".json").write_text(
        json.dumps(tampered_report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resource report"):
        experiment.load_gemma3_structured_mlp_compression_a_artifact(
            tampered_path
        )


def test_a_failure_writes_preflight_without_artifact_or_heldout(
    tmp_path: Path,
) -> None:
    output, splits = _run_mocked(tmp_path, passed=False)

    assert splits == ["calibration_a"]
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    preflight = output.with_suffix(".calibration-a.json")
    assert preflight.is_file()
    report = json.loads(preflight.read_text(encoding="utf-8"))
    assert report["scientific_status"]["heldout_opened"] is False
    assert report["main_artifact_written"] is False
