from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import fisher_graph.gemma3_structured_mlp_compression_heldout_experiment as heldout
from fisher_graph.gemma3_full_width_single_layer_experiment import (
    DEFAULT_BLOCK_DELTA_COSINE_MIN,
    DEFAULT_BLOCK_DELTA_NRMSE_MAX,
    DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS,
    DEFAULT_MINIMUM_HELDOUT_PROMPTS,
    DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS,
    DEFAULT_MINIMUM_LENGTH_BUCKETS,
    DEFAULT_NLL_ATOL,
    DEFAULT_PER_PROMPT_P10_TOP1_MIN,
    DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
    DEFAULT_TEACHER_KL_MAX,
    DEFAULT_TOP1_MIN,
)
from fisher_graph.gemma3_structured_single_layer_experiment import (
    DEFAULT_BRANCH_DELTA_COSINE_MIN,
    DEFAULT_BRANCH_DELTA_NRMSE_MAX,
    DEFAULT_NATIVE_PARITY_TOLERANCE,
)

from test_gemma3_structured_single_layer_experiment import (
    _families_for_prompts,
    _lexically_broad_prompt_payload,
    _write_corpus_audit,
)
from test_structured_mlp_compression import _executor
from test_structured_operator_bootstrap import _FakeCausalLM


def _thresholds() -> dict[str, float]:
    return {
        "nll_atol": DEFAULT_NLL_ATOL,
        "top1_min": DEFAULT_TOP1_MIN,
        "teacher_kl_max": DEFAULT_TEACHER_KL_MAX,
        "p90_abs_nll_max": DEFAULT_PER_PROMPT_P90_ABS_NLL_MAX,
        "p10_top1_min": DEFAULT_PER_PROMPT_P10_TOP1_MIN,
        "block_delta_nrmse_max": DEFAULT_BLOCK_DELTA_NRMSE_MAX,
        "block_delta_cosine_min": DEFAULT_BLOCK_DELTA_COSINE_MIN,
        "branch_delta_nrmse_max": DEFAULT_BRANCH_DELTA_NRMSE_MAX,
        "branch_delta_cosine_min": DEFAULT_BRANCH_DELTA_COSINE_MIN,
        "native_parity_tolerance": DEFAULT_NATIVE_PARITY_TOLERANCE,
    }


def _candidate(
    path: Path,
    *,
    source_prompt_file_sha256: str,
    source_family_file_sha256: str,
) -> dict[str, object]:
    path.write_bytes(b"strict candidate fixture")
    executor = _executor(1_536, projection_bias=False)
    source_parameters = executor.learned_parameter_count + 1_000
    forbidden = {
        role: [
            hashlib.sha256(
                f"source-prompt-{role}-{index}".encode()
            ).hexdigest()
            for index in range(2)
        ]
        for role in heldout._ROLES
    }
    forbidden_families = {
        role: [
            hashlib.sha256(
                f"source-family-{role}-{index}".encode()
            ).hexdigest()
            for index in range(2)
        ]
        for role in heldout._ROLES
    }
    normalized_tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token="</s>",
        padding_side="right",
    )
    return {
        "model": {
            "model_id": "tiny-heldout-gemma",
            "resolved_commit": "a" * 40,
        },
        "protocol": {
            "source_corpus": {
                "prompt_sha256_by_role": forbidden,
                "family_sha256_by_role": forbidden_families,
                "prompt_fixture_file_sha256": (
                    source_prompt_file_sha256
                ),
                "family_manifest_file_sha256": (
                    source_family_file_sha256
                ),
            },
            "thresholds": _thresholds(),
            "layer_index": 0,
            "layer_id": "layer.0",
            "maximum_tokenized_length": 256,
            "tokenization_batch_size": 8,
            "tokenizer": heldout._tokenizer_provenance(
                normalized_tokenizer
            ),
        },
        "parent": {},
        "executor": executor,
        "calibration_a": {},
        "pipeline": {
            "pipeline_report_sha256": "b" * 64,
            "selection_sha256": "c" * 64,
            "score_collection_sha256": "d" * 64,
            "final_candidate": {
                "execution_fingerprint": (
                    executor.execution_fingerprint()
                ),
                "artifact_state_sha256": "e" * 64,
            },
        },
        "resource_report": {
            "rung": {
                "source_intermediate_width": 2_048,
                "retained_intermediate_width": 1_536,
                "removed_intermediate_width": 512,
            },
            "parameters": {
                "source_full_layer": source_parameters,
                "compressed_full_layer": executor.learned_parameter_count,
                "removed_full_layer": 1_000,
                "retained_ratio": (
                    executor.learned_parameter_count / source_parameters
                ),
            },
            "compute_per_valid_token": {
                "macs": {
                    "source": 100,
                    "compressed": 95,
                    "removed": 5,
                },
                "flops_two_per_mac": {
                    "source": 200,
                    "compressed": 190,
                    "removed": 10,
                },
            },
        },
        "scientific_status": {
            "calibration_a_passed": True,
            "heldout_opened": False,
            "scientific_compression_success": False,
        },
        "metadata": {
            "scientific_payload_sha256": "1" * 64,
            "report_sha256": "2" * 64,
            "tensor_file_sha256": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        },
        "report": {},
    }


def _v7_fixture(
    tmp_path: Path,
    *,
    source_prompt_file_sha256: str,
    source_family_file_sha256: str,
) -> tuple[Path, Path, Path]:
    prompts = _lexically_broad_prompt_payload()
    for role, count in (
        ("calibration_a", DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS),
        ("calibration_b", DEFAULT_MINIMUM_HELDOUT_PROMPTS),
        ("validation", DEFAULT_MINIMUM_HELDOUT_PROMPTS),
        ("test", DEFAULT_MINIMUM_HELDOUT_PROMPTS),
    ):
        base = list(prompts[role])
        prompts[role] = [
            f"{base[index % len(base)]} replica{index:04d}"
            for index in range(count)
        ]
    prompt_path = tmp_path / "structured-strong-v7-prompts.json"
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "structured-strong-v7-families.json"
    family_path.write_text(
        json.dumps(_families_for_prompts(prompts)),
        encoding="utf-8",
    )
    audit_path = _write_corpus_audit(
        tmp_path,
        prompts,
        format_version=2,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "format_version": 3,
            "corpus_id": "structured-strong-v7",
            "heldout_splits_evaluated": False,
            "heldout_splits_tokenized": False,
            "heldout_splits_unevaluated": True,
            "heldout_splits_untokenized": True,
            "calibration_b_model_evaluated": False,
            "validation_model_evaluated": False,
            "prior_raw_prompt_overlap_count": 0,
            "prior_normalized_prompt_overlap_count": 0,
            "prior_domain_slug_overlap_count": 0,
            "prior_template_marker_overlap_count": 0,
            "prior_template_signature_overlap_count": 0,
            "prior_5_6_7_8_word_ngram_overlap_count": 0,
            "prior_generator_imported_or_reused": False,
            "prior_prompt_files": [
                {"file_sha256": source_prompt_file_sha256}
            ],
            "prior_family_files": [
                {"file_sha256": source_family_file_sha256}
            ],
        }
    )
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return prompt_path, family_path, audit_path


def _stream(
    split_name: str,
    source_prompt_sha256: list[str],
) -> dict[str, object]:
    lengths = (30, 60, 120, 200)
    return {
        "schema": "fisher_graph.tokenized_calibration_stream",
        "format_version": 2,
        "split": split_name,
        "batches": 1,
        "sequences": len(source_prompt_sha256),
        "serialized_sha256": hashlib.sha256(
            f"stream-{split_name}".encode()
        ).hexdigest(),
        "source_prompt_sha256": list(source_prompt_sha256),
        "examples": [
            {
                "example_id": f"prompt.{index:06d}",
                "serialized_sha256": hashlib.sha256(
                    f"serialized-{split_name}-{index}".encode()
                ).hexdigest(),
                "content_sha256": hashlib.sha256(
                    f"content-{split_name}-{index}".encode()
                ).hexdigest(),
                "valid_tokens": lengths[index % len(lengths)],
                "supervised_positions": (
                    lengths[index % len(lengths)] - 1
                ),
            }
            for index in range(len(source_prompt_sha256))
        ],
    }


def _evaluation(
    *,
    executor_parameters: int,
    source_parameters: int,
    valid_tokens: int,
    source_macs: int,
    passed: bool,
) -> dict[str, object]:
    name = heldout.STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE
    behavior = {
        "delta_nll_per_token": 0.0 if passed else 1e9,
        "top1_agreement_to_baseline": 1.0,
        "teacher_kl_per_token": 0.0,
        "per_example_delta_nll_per_token": {
            "p90_absolute": 0.0 if passed else 1e9,
        },
        "per_example_top1_agreement": {"p10": 1.0},
    }
    direct = {
        "block_delta_nrmse": 0.0,
        "block_delta_cosine": 1.0,
    }
    branches = {
        "attention_delta": direct,
        "feed_forward_delta": direct,
    }
    return {
        "behavior": {name: behavior},
        "direct": {name: direct},
        "branches": {name: branches},
        "execution_audits": {name: {"passed": True}},
        "ordinary_vs_segmented_native": {"passed": True},
        "native_boundary_replay": {"passed": True},
        "logical_accounting": {
            name: {
                "valid_tokens": valid_tokens,
                "logical_causal_key_pairs": valid_tokens,
                "attention_projection_macs": 0,
                "attention_score_macs": 0,
                "attention_value_macs": 0,
                "feed_forward_macs": (
                    source_macs - valid_tokens * 5
                ),
                "logical_total_macs": (
                    source_macs - valid_tokens * 5
                ),
            }
        },
        "boundaries": (object(),),
        "_test_resource": {
            "executor_parameters": executor_parameters,
            "source_parameters": source_parameters,
        },
    }


def test_claim_identity_is_order_insensitive_and_exclusive(
    tmp_path: Path,
) -> None:
    hashes = ["a" * 64, "b" * 64]
    first = heldout._claim_path(tmp_path, hashes)
    repeated = heldout._claim_path(tmp_path, tuple(reversed(hashes)))
    assert first == repeated
    kwargs = {
        "prompt_hashes": hashes,
        "family_hashes": ["c" * 64, "d" * 64],
        "candidate_binding": {"execution_fingerprint": "e" * 64},
        "v7_binding": {"corpus_id": "structured-strong-v7"},
        "model_resolved_commit": "f" * 40,
        "layer_id": "layer.0",
        "thresholds": _thresholds(),
        "token_contract": {"runtime_device": "cpu"},
    }
    heldout._exclusive_heldout_claim(first, **kwargs)
    with pytest.raises(FileExistsError):
        heldout._exclusive_heldout_claim(repeated, **kwargs)


def test_static_preflight_rejects_source_prompt_reuse(
    tmp_path: Path,
) -> None:
    source_prompt_file_sha256 = "3" * 64
    source_family_file_sha256 = "4" * 64
    candidate_path = tmp_path / "candidate.pt"
    loaded = _candidate(
        candidate_path,
        source_prompt_file_sha256=source_prompt_file_sha256,
        source_family_file_sha256=source_family_file_sha256,
    )
    candidate = heldout._candidate_preflight(
        loaded,
        candidate_artifact_path=candidate_path,
    )
    prompt_path, family_path, audit_path = _v7_fixture(
        tmp_path,
        source_prompt_file_sha256=source_prompt_file_sha256,
        source_family_file_sha256=source_family_file_sha256,
    )
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    from fisher_graph.gemma3_stability_experiment import (
        load_gemma3_prompt_splits,
    )

    v7 = load_gemma3_prompt_splits(prompt_path)
    reused = v7.metadata()["per_prompt_sha256"]["calibration_b"][0]
    candidate["forbidden_prompt_sha256"] = frozenset({reused})
    with pytest.raises(ValueError, match="reuses"):
        heldout._v7_corpus_preflight(
            prompt_splits_path=prompt_path,
            family_manifest_path=family_path,
            corpus_audit_path=audit_path,
            candidate=candidate,
            minimum_calibration_a_prompts=1,
            minimum_heldout_prompts=1,
        )
    assert prompts["test"]


def test_output_collision_and_weakened_minima_fail_before_candidate_load(
    tmp_path: Path,
) -> None:
    output = tmp_path / "heldout.pt"
    output.write_bytes(b"existing")
    common = {
        "candidate_artifact_path": tmp_path / "candidate.pt",
        "prompt_splits_path": tmp_path / "prompts.json",
        "family_manifest_path": tmp_path / "families.json",
        "corpus_audit_path": tmp_path / "audit.json",
        "output": output,
        "calibration_b_ledger_dir": tmp_path / "ledger",
        "device_name": "cpu",
        "dtype": "float32",
    }
    with patch.object(
        heldout,
        "_load_compression_a_candidate",
        side_effect=AssertionError("candidate load must not be reached"),
    ) as loader:
        with pytest.raises(FileExistsError, match="output already exists"):
            heldout.run_gemma3_structured_mlp_compression_heldout_experiment(
                **common,
            )
        output.unlink()
        with pytest.raises(ValueError, match="frozen standard data minima"):
            heldout.run_gemma3_structured_mlp_compression_heldout_experiment(
                **common,
                minimum_heldout_prompts=1,
            )
    loader.assert_not_called()


def _run_fixture(
    tmp_path: Path,
    *,
    calibration_b_passed: bool,
    validation_passed: bool = True,
) -> tuple[Path, list[str]]:
    source_prompt_file_sha256 = "3" * 64
    source_family_file_sha256 = "4" * 64
    candidate_path = tmp_path / "candidate.pt"
    loaded = _candidate(
        candidate_path,
        source_prompt_file_sha256=source_prompt_file_sha256,
        source_family_file_sha256=source_family_file_sha256,
    )
    prompt_path, family_path, audit_path = _v7_fixture(
        tmp_path,
        source_prompt_file_sha256=source_prompt_file_sha256,
        source_family_file_sha256=source_family_file_sha256,
    )
    prompt_metadata = heldout.load_gemma3_prompt_splits(
        prompt_path
    ).metadata()
    output = tmp_path / "heldout.pt"
    calls: list[str] = []
    executor = loaded["executor"]
    assert hasattr(executor, "learned_parameter_count")
    source_parameters = executor.learned_parameter_count + 1_000
    source_macs = 100_000
    calibration_b_stream = _stream(
        "calibration_b",
        prompt_metadata["per_prompt_sha256"]["calibration_b"],
    )
    valid_tokens = sum(
        example["valid_tokens"]
        for example in calibration_b_stream["examples"]
    )
    calibration_b_evaluation = _evaluation(
        executor_parameters=executor.learned_parameter_count,
        source_parameters=source_parameters,
        valid_tokens=valid_tokens,
        source_macs=source_macs,
        passed=calibration_b_passed,
    )
    calibration_b_evaluation.pop("_test_resource")
    validation_evaluation = _evaluation(
        executor_parameters=executor.learned_parameter_count,
        source_parameters=source_parameters,
        valid_tokens=valid_tokens,
        source_macs=source_macs,
        passed=validation_passed,
    )
    validation_evaluation.pop("_test_resource")

    model = _FakeCausalLM().eval().requires_grad_(False)
    model.config._commit_hash = "a" * 40
    model.model.config._commit_hash = "a" * 40
    loaded["model"] = heldout._model_provenance(
        model,
        model_id="tiny-heldout-gemma",
        requested_revision="a" * 40,
    )

    def materialize(
        _tokenizer: object,
        _prompts: object,
        *,
        split_name: str,
        **_: object,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        calls.append(split_name)
        assert any(
            (
                tmp_path
                / "ledger"
                / heldout.STRUCTURED_MLP_COMPRESSION_HELDOUT_LEDGER_NAMESPACE
            ).glob("*.json")
        )
        return (
            (object(),),
            _stream(
                split_name,
                prompt_metadata["per_prompt_sha256"][split_name],
            ),
        )

    with (
        patch.object(
            heldout,
            "_load_compression_a_candidate",
            return_value=loaded,
        ),
        patch.object(
            heldout,
            "resolve_gemma3_huggingface_paths",
            return_value={"hub_cache": tmp_path / "cache"},
        ),
        patch.object(
            heldout,
            "load_gemma3",
            return_value=(
                SimpleNamespace(
                    pad_token_id=0,
                    eos_token="</s>",
                    padding_side="left",
                ),
                model,
            ),
        ),
        patch.object(
            heldout,
            "_source_block_static",
            return_value={
                "parameter_count": source_parameters,
                "parameter_bytes": source_parameters * 4,
            },
        ),
        patch.object(
            heldout,
            "_source_block_macs",
            return_value={"total_macs": source_macs},
        ),
        patch.object(
            heldout,
            "_materialize_split",
            side_effect=materialize,
        ),
        patch.object(
            heldout,
            "evaluate_structured_candidates",
            side_effect=(
                (
                    calibration_b_evaluation,
                    validation_evaluation,
                )
                if calibration_b_passed
                else (calibration_b_evaluation,)
            ),
        ),
    ):
        report = (
            heldout.run_gemma3_structured_mlp_compression_heldout_experiment(
                candidate_artifact_path=candidate_path,
                prompt_splits_path=prompt_path,
                family_manifest_path=family_path,
                corpus_audit_path=audit_path,
                output=output,
                calibration_b_ledger_dir=tmp_path / "ledger",
                cache_dir=tmp_path / "cache",
                device_name="cpu",
                dtype="float32",
                minimum_calibration_a_prompts=(
                    DEFAULT_MINIMUM_CALIBRATION_A_PROMPTS
                ),
                minimum_heldout_prompts=(
                    DEFAULT_MINIMUM_HELDOUT_PROMPTS
                ),
                minimum_heldout_supervised_tokens=(
                    DEFAULT_MINIMUM_HELDOUT_SUPERVISED_TOKENS
                ),
                minimum_length_buckets=DEFAULT_MINIMUM_LENGTH_BUCKETS,
            )
        )
    return output, calls


def test_b_failure_consumes_claim_and_never_tokenizes_validation_or_test(
    tmp_path: Path,
) -> None:
    output, calls = _run_fixture(
        tmp_path,
        calibration_b_passed=False,
    )
    assert calls == ["calibration_b"]
    loaded = (
        heldout.load_gemma3_structured_mlp_compression_heldout_artifact(
            output
        )
    )
    assert loaded["scientific_status"]["outcome"] == (
        "rejected_on_calibration_b"
    )
    assert loaded["validation"] == heldout._unevaluated_validation()
    assert not loaded["scientific_status"]["test_tokenized"]


@pytest.mark.parametrize(
    ("validation_passed", "expected_outcome"),
    (
        (
            True,
            "single_layer_structured_mlp_compression_passed",
        ),
        (False, "rejected_on_validation"),
    ),
)
def test_b_pass_opens_validation_but_never_test(
    tmp_path: Path,
    validation_passed: bool,
    expected_outcome: str,
) -> None:
    output, calls = _run_fixture(
        tmp_path,
        calibration_b_passed=True,
        validation_passed=validation_passed,
    )
    assert calls == ["calibration_b", "validation"]
    loaded = (
        heldout.load_gemma3_structured_mlp_compression_heldout_artifact(
            output
        )
    )
    status = loaded["scientific_status"]
    assert status["outcome"] == expected_outcome
    assert status["validation_evaluated"]
    assert not status["test_tokenized"]
    assert not status["test_evaluated"]
    assert not status["model_level_promotion_authorized"]
    assert status["parameter_reduction_supported"] is validation_passed


def test_rehashed_metric_tamper_fails_gate_recomputation(
    tmp_path: Path,
) -> None:
    output, _ = _run_fixture(
        tmp_path,
        calibration_b_passed=False,
    )
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    name = heldout.STRUCTURED_MLP_COMPRESSION_HELDOUT_CANDIDATE
    artifact["calibration_b"]["behavior"][name][
        "delta_nll_per_token"
    ] = 0.0
    payload = {
        key: value
        for key, value in artifact.items()
        if key
        not in {"scientific_payload_sha256", "report_sha256"}
    }
    artifact["scientific_payload_sha256"] = heldout._sha256(
        payload,
        domain=heldout._PAYLOAD_DOMAIN,
    )
    forged_report = heldout._report_from_payload(
        payload,
        output=output,
        scientific_payload_sha256=artifact[
            "scientific_payload_sha256"
        ],
    )
    artifact["report_sha256"] = heldout._sha256(
        forged_report,
        domain=heldout._REPORT_DOMAIN,
    )
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(forged_report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evaluation binding"):
        heldout.load_gemma3_structured_mlp_compression_heldout_artifact(
            output
        )
