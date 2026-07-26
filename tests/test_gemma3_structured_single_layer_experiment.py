import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fisher_graph import (
    gemma3_structured_single_layer_experiment as structured_experiment,
)

transformers = pytest.importorskip("transformers")
try:
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
except ImportError:
    pytest.skip(
        "installed transformers does not provide Gemma 3",
        allow_module_level=True,
    )

from fisher_graph.gemma3_full_width_single_layer_experiment import (
    FAMILY_STATUS,
    PROMPT_STATUS,
)
from fisher_graph.gemma3_structured_single_layer_experiment import (
    _build_report,
    _report_sha256,
    _scientific_payload_sha256,
    _tensor_sha256,
    load_gemma3_structured_single_layer_artifact,
    run_gemma3_structured_single_layer_experiment,
)


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"
    name_or_path = "tiny-local-gemma3-test-tokenizer"
    vocab_size = 31
    model_max_length = 8
    bos_token_id = 2
    eos_token_id = 1
    init_kwargs: dict[str, object] = {}

    def __init__(self) -> None:
        self.prompt_calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        prompts: list[str],
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        self.prompt_calls.append(tuple(prompts))
        unpadded = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            row = [
                self.bos_token_id,
                3 + digest[0] % 28,
                3 + digest[1] % 28,
            ]
            if digest[2] % 2:
                row.append(3 + digest[3] % 28)
            row.append(self.eos_token_id)
            unpadded.append(row)
        width = max(len(row) for row in unpadded)
        input_ids = torch.tensor(
            [
                row + [self.pad_token_id] * (width - len(row))
                for row in unpadded
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [True] * len(row) + [False] * (width - len(row))
                for row in unpadded
            ],
            dtype=torch.bool,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def _tiny_gemma3() -> Gemma3ForCausalLM:
    torch.manual_seed(91_104)
    config = Gemma3TextConfig(
        vocab_size=31,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        query_pre_attn_scalar=4,
        sliding_window=8,
        layer_types=["full_attention"],
        use_cache=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    config._commit_hash = "a" * 40
    model = Gemma3ForCausalLM(config).eval()
    model.requires_grad_(False)
    return model


def _prompt_payload() -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": PROMPT_STATUS,
        "calibration_a": [
            f"structured train prompt {index}" for index in range(4)
        ],
        "calibration_b": [
            f"structured selection prompt {index}" for index in range(2)
        ],
        "validation": [
            f"structured validation prompt {index}" for index in range(2)
        ],
        "test": [
            f"structured reserved test prompt {index}" for index in range(2)
        ],
    }


def _family_payload() -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_family_manifest",
        "format_version": 1,
        "scientific_status": FAMILY_STATUS,
        "calibration_a": ["train-a", "train-a", "train-b", "train-b"],
        "calibration_b": ["selection-a", "selection-b"],
        "validation": ["validation-a", "validation-b"],
        "test": ["test-a", "test-b"],
    }


def _write_corpus_audit(
    tmp_path: Path,
    prompts: dict[str, object],
    *,
    format_version: int = 1,
) -> Path:
    generator = tmp_path / "generate_test_corpus.py"
    generator.write_text("# frozen test corpus generator\n", encoding="utf-8")
    generator_sha256 = hashlib.sha256(generator.read_bytes()).hexdigest()
    roles = ("calibration_a", "calibration_b", "validation", "test")
    role_prompts = {
        role: prompts[role]
        for role in roles
    }
    audit = {
        "schema": "fisher_graph.structured_test_corpus_audit",
        "format_version": format_version,
        "generator": generator.name,
        "generator_sha256": generator_sha256,
        "counts": {
            role: len(role_prompts[role])
            for role in roles
        },
        "prompt_sha256_by_role": {
            role: [
                hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                for prompt in role_prompts[role]
            ]
            for role in roles
        },
        "unique_prompt_count": sum(
            len(role_prompts[role])
            for role in roles
        ),
        "cross_role_family_overlap_count": 0,
        "tokenizer_or_model_accessed": False,
        "test_model_evaluated": False,
        "corpus_frozen_before_model_load": True,
        "prior_local_exact_prompt_overlap_count": 0,
    }
    if format_version >= 2:
        audit["approximate_word_counts_by_role"] = {
            role: [
                len(prompt.split())
                for prompt in role_prompts[role]
            ]
            for role in roles
        }
    path = tmp_path / "corpus-audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def _lexically_broad_prompt_payload() -> dict[str, object]:
    roles = ("calibration_a", "calibration_b", "validation", "test")
    target_lengths = (8, 9, 10, 11, 16, 20, 24, 32, 33, 48, 72, 96)
    target_lengths += (97, 112, 128, 144)
    payload: dict[str, object] = {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": PROMPT_STATUS,
    }
    for role in roles:
        compact_role = role.replace("_", "")
        payload[role] = [
            " ".join(
                (
                    f"{compact_role}{index}",
                    *(
                        f"word{word_index}"
                        for word_index in range(length - 1)
                    ),
                )
            )
            for index, length in enumerate(target_lengths)
        ]
    return payload


def _families_for_prompts(
    prompts: dict[str, object],
) -> dict[str, object]:
    roles = ("calibration_a", "calibration_b", "validation", "test")
    return {
        "schema": "fisher_graph.gemma3_prompt_family_manifest",
        "format_version": 1,
        "scientific_status": FAMILY_STATUS,
        **{
            role: [
                f"{role}-family-{index}"
                for index, _ in enumerate(prompts[role])
            ]
            for role in roles
        },
    }


def _rehash_payload(artifact: dict[str, object]) -> None:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    artifact["scientific_payload_sha256"] = (
        _scientific_payload_sha256(payload)
    )


def _tiny_run_kwargs(
    tmp_path: Path,
    *,
    prompt_path: Path,
    family_path: Path,
    output: Path,
    ledger_dir: Path,
    **overrides: object,
) -> dict[str, object]:
    values: dict[str, object] = {
        "prompt_splits_path": prompt_path,
        "family_manifest_path": family_path,
        "model_id": "tiny-local-gemma3",
        "revision": "a" * 40,
        "cache_dir": tmp_path / "cache",
        "layer_index": 0,
        "max_length": 8,
        "tokenization_batch_size": 2,
        "local_warmup_steps": 0,
        "train_steps": 1,
        "train_positions_per_sequence": 1,
        "learning_rate": 1e-3,
        "selection_nll_atol": 1e6,
        "selection_top1_min": 0.0,
        "selection_teacher_kl_max": 1e6,
        "selection_p90_abs_nll_max": 1e6,
        "selection_p10_top1_min": 0.0,
        "block_delta_nrmse_max": 1e6,
        "block_delta_cosine_min": -1.0,
        "branch_delta_nrmse_max": 1e6,
        "branch_delta_cosine_min": -1.0,
        "minimum_calibration_a_prompts": 1,
        "minimum_heldout_prompts": 1,
        "minimum_fisher_rows": 1,
        "minimum_train_supervised_tokens": 1,
        "minimum_heldout_supervised_tokens": 1,
        "minimum_length_buckets": 1,
        "device_name": "cpu",
        "dtype": "float32",
        "output": output,
        "calibration_b_ledger_dir": ledger_dir,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize("mutation", ("missing", "malformed"))
def test_format2_corpus_word_counts_fail_before_model_or_ledger(
    tmp_path: Path,
    mutation: str,
) -> None:
    case = tmp_path / mutation
    case.mkdir()
    prompts = _lexically_broad_prompt_payload()
    prompt_path = case / "prompts.json"
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = case / "families.json"
    family_path.write_text(
        json.dumps(_families_for_prompts(prompts)),
        encoding="utf-8",
    )
    audit_path = _write_corpus_audit(
        case,
        prompts,
        format_version=2,
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        audit.pop("approximate_word_counts_by_role")
    else:
        audit["approximate_word_counts_by_role"]["calibration_a"][0] += 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    ledger_dir = case / "ledger"

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            side_effect=AssertionError("model load must not be reached"),
        ) as load_model,
        pytest.raises(ValueError, match="approximate word counts"),
    ):
        run_gemma3_structured_single_layer_experiment(
            **_tiny_run_kwargs(
                case,
                prompt_path=prompt_path,
                family_path=family_path,
                output=case / "result.pt",
                ledger_dir=ledger_dir,
                corpus_audit_path=audit_path,
            )
        )

    load_model.assert_not_called()
    assert not ledger_dir.exists()


def test_format2_corpus_missing_length_band_fails_before_model_or_ledger(
    tmp_path: Path,
) -> None:
    prompts = _lexically_broad_prompt_payload()
    prompts["test"][:4] = [
        " ".join(
            (
                f"testnomicro{index}",
                *(f"word{word_index}" for word_index in range(15)),
            )
        )
        for index in range(4)
    ]
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_families_for_prompts(prompts)),
        encoding="utf-8",
    )
    audit_path = _write_corpus_audit(
        tmp_path,
        prompts,
        format_version=2,
    )
    ledger_dir = tmp_path / "ledger"

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            side_effect=AssertionError("model load must not be reached"),
        ) as load_model,
        pytest.raises(ValueError, match="lexical length breadth"),
    ):
        run_gemma3_structured_single_layer_experiment(
            **_tiny_run_kwargs(
                tmp_path,
                prompt_path=prompt_path,
                family_path=family_path,
                output=tmp_path / "result.pt",
                ledger_dir=ledger_dir,
                corpus_audit_path=audit_path,
            )
        )

    load_model.assert_not_called()
    assert not ledger_dir.exists()


def test_valid_format2_corpus_breadth_is_bound_before_model_load(
    tmp_path: Path,
) -> None:
    prompts_payload = _lexically_broad_prompt_payload()
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(prompts_payload),
        encoding="utf-8",
    )
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_families_for_prompts(prompts_payload)),
        encoding="utf-8",
    )
    audit_path = _write_corpus_audit(
        tmp_path,
        prompts_payload,
        format_version=2,
    )
    prompts = structured_experiment.load_gemma3_prompt_splits(prompt_path)
    binding = structured_experiment._corpus_audit_binding(
        audit_path,
        prompts=prompts,
        prompt_path=prompt_path,
        family_path=family_path,
    )
    assert binding is not None
    lexical = binding["lexical_length_audit"]
    assert lexical["all_roles_cover_all_bands"]
    assert all(
        count == 4
        for role_counts in lexical["counts_by_role"].values()
        for count in role_counts.values()
    )
    protocol = {
        "prompt_splits": {
            "counts": prompts.metadata()["counts"],
        },
        "prompt_fixture_file_sha256": binding[
            "prompt_fixture_file_sha256"
        ],
        "family_manifest_file_sha256": binding[
            "family_manifest_file_sha256"
        ],
    }
    assert structured_experiment._validate_corpus_audit_binding(
        binding,
        protocol=protocol,
    )
    ledger_dir = tmp_path / "ledger"

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            side_effect=RuntimeError("model-load-reached"),
        ) as load_model,
        pytest.raises(RuntimeError, match="model-load-reached"),
    ):
        run_gemma3_structured_single_layer_experiment(
            **_tiny_run_kwargs(
                tmp_path,
                prompt_path=prompt_path,
                family_path=family_path,
                output=tmp_path / "result.pt",
                ledger_dir=ledger_dir,
                corpus_audit_path=audit_path,
            )
        )

    load_model.assert_called_once()
    assert not ledger_dir.exists()


def test_tiny_real_gemma_fit_validation_and_strict_artifact(
    tmp_path: Path,
) -> None:
    model = _tiny_gemma3()
    parameter_versions = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
    }
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    corpus_audit_path = _write_corpus_audit(tmp_path, prompts)
    output = tmp_path / "structured.pt"
    tokenizer = RecordingTokenizer()

    with patch(
        "fisher_graph.gemma3_structured_single_layer_experiment."
        "load_gemma3",
        return_value=(tokenizer, model),
    ):
        report = run_gemma3_structured_single_layer_experiment(
            prompt_splits_path=prompt_path,
            family_manifest_path=family_path,
            corpus_audit_path=corpus_audit_path,
            model_id="tiny-local-gemma3",
            revision="a" * 40,
            cache_dir=tmp_path / "cache",
            layer_index=0,
            max_length=8,
            tokenization_batch_size=2,
            local_warmup_steps=0,
            train_steps=1,
            train_positions_per_sequence=1,
            learning_rate=1e-3,
            selection_nll_atol=1e6,
            selection_top1_min=0.0,
            selection_teacher_kl_max=1e6,
            selection_p90_abs_nll_max=1e6,
            selection_p10_top1_min=0.0,
            block_delta_nrmse_max=1e6,
            block_delta_cosine_min=-1.0,
            branch_delta_nrmse_max=1e6,
            branch_delta_cosine_min=-1.0,
            minimum_calibration_a_prompts=1,
            minimum_heldout_prompts=1,
            minimum_fisher_rows=1,
            minimum_train_supervised_tokens=1,
            minimum_heldout_supervised_tokens=1,
            minimum_length_buckets=1,
            device_name="cpu",
            dtype="float32",
            output=output,
            calibration_b_ledger_dir=tmp_path / "heldout-ledger",
        )

    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert output.with_suffix(".calibration-a.json").is_file()
    preflight = json.loads(
        output.with_suffix(".calibration-a.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(preflight["training"]) == {
        "structured_source_visibility",
        "attention_output_disabled_control",
    }
    for name, training_report in preflight["training"].items():
        assert training_report["snapshots"]
        assert (
            training_report["final_execution_fingerprint"]
            == preflight["executor_fingerprints"][name]
            == preflight["fidelity"]["executor_fingerprints"][name]
        )
    assert report["format_version"] == 4
    recipe = report["protocol"]["training_recipe"]
    assert recipe["coordinate_loss_weight"] == 1.0
    assert recipe["energy_loss_weight"] == 1.0
    assert (
        recipe["rmsnorm_initialization"]
        == "calibration_a_activation_pair_coordinate_least_squares_v1"
    )
    status = report["scientific_status"]
    assert status["calibration_b_passed"]
    assert status["validation_evaluated"]
    assert status["validation_passed"]
    assert status["source_layer_calls_in_student_path"] == 0
    assert status["source_layer_removed_from_student_path"]
    assert not status["compression_attempted"]
    assert not status["parameter_reduction_supported"]
    assert not status["analytic_mac_reduction_supported"]
    assert not status["resource_values_used_as_fidelity_gates"]
    assert status["corpus_audit_bound"]
    assert not status["preregistered_fixed_schedule_enforced"]
    assert all(
        int(parameter._version) == parameter_versions[name]
        for name, parameter in model.named_parameters()
    )

    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert set(prompts["validation"]).issubset(tokenized)
    assert not set(prompts["test"]) & tokenized

    loaded = load_gemma3_structured_single_layer_artifact(output)
    primary = loaded["executors"]["structured_source_visibility"]
    control = loaded["executors"][
        "attention_output_disabled_control"
    ]
    assert primary.config.causal_edges_enabled
    assert not control.config.causal_edges_enabled
    assert primary.learned_parameter_count == (
        loaded["selection"]["accounting"][
            "structured_source_visibility"
        ]["source_layer_parameter_count"]
    )
    assert loaded["selection"]["execution_audits"][
        "structured_source_visibility"
    ]["source_block_calls_total"] == 0
    assert loaded["validation"]["execution_audits"][
        "structured_source_visibility"
    ]["source_block_calls_total"] == 0
    assert loaded["selection"]["ordinary_vs_segmented_native"]["passed"]
    assert loaded["validation"]["ordinary_vs_segmented_native"]["passed"]
    assert loaded["selection"]["native_boundary_replay"]["passed"]
    assert loaded["validation"]["native_boundary_replay"]["passed"]
    assert loaded["training"]["calibration_a_fidelity"][
        "primary_passed"
    ]
    claim = loaded["protocol"]["calibration_b_claim"]
    assert claim["state"] == "claimed_before_tokenization"
    assert (
        claim["executor_fingerprints"]
        == loaded["selection"]["executor_fingerprints"]
        == loaded["validation"]["executor_fingerprints"]
    )
    initialization = loaded["training"][
        "structured_source_visibility"
    ]["rmsnorm_initialization"]
    assert initialization["source_module_or_parameter_read"] is False
    assert initialization["direct_source_tensor_copy"] is False

    legacy_output = tmp_path / "synthetic-format3.pt"
    legacy = torch.load(output, map_location="cpu", weights_only=True)
    legacy["format_version"] = 3
    legacy["protocol"].pop("calibration_b_claim")
    for field in (
        "coordinate_loss_weight",
        "energy_loss_weight",
        "rmsnorm_initialization",
    ):
        legacy["protocol"]["training_recipe"].pop(field)
    legacy["training"].pop("calibration_a_fidelity")
    for name in (
        "structured_source_visibility",
        "attention_output_disabled_control",
    ):
        legacy["training"][name].pop("final_execution_fingerprint")
        legacy["training"]["structural_probes"][name].pop(
            "executor_execution_fingerprint"
        )
    legacy["selection"].pop("executor_fingerprints")
    legacy["validation"].pop("executor_fingerprints")
    legacy_payload = {
        key: value
        for key, value in legacy.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    legacy_digest = _scientific_payload_sha256(legacy_payload)
    legacy_report = _build_report(
        legacy_payload,
        tensor_file=str(legacy_output),
        scientific_digest=legacy_digest,
    )
    legacy["scientific_payload_sha256"] = legacy_digest
    legacy["report_sha256"] = _report_sha256(legacy_report)
    torch.save(legacy, legacy_output)
    legacy_output.with_suffix(".json").write_text(
        json.dumps(
            legacy_report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_loaded = load_gemma3_structured_single_layer_artifact(
        legacy_output
    )
    assert legacy_loaded["scientific_status"] == loaded[
        "scientific_status"
    ]

    fisher = loaded["training"]["activation_fisher"]
    assert fisher["matrix"].shape == (8, 8)
    assert fisher["selected_target_rows"] == 4
    assert fisher["valid_boundary_rows"] >= 4
    assert fisher["effective_nonzero_gradient_rows"] >= 1
    assert (
        fisher["estimator"]
        == "ground_truth_CE_activation_gradient_second_moment"
    )
    assert not fisher["expected_model_fisher_claim"]
    assert not fisher["cross_position_blocks_included"]
    assert torch.isfinite(fisher["matrix"]).all()

    tampered = tmp_path / "tampered-executor.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    state = changed["executors"]["structured_source_visibility"][
        "model_state_dict"
    ]
    tensor_name = next(
        name
        for name, value in state.items()
        if value.is_floating_point() and value.numel() > 0
    )
    state[tensor_name].view(-1)[0] += 1.0
    _rehash_payload(changed)
    torch.save(changed, tampered)
    shutil.copyfile(
        output.with_suffix(".json"),
        tampered.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_gemma3_structured_single_layer_artifact(tampered)

    forged_record_fingerprint = tmp_path / "forged-record-fingerprint.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["structured_source_visibility"][
        "final_execution_fingerprint"
    ] = "b" * 64
    _rehash_payload(changed)
    torch.save(changed, forged_record_fingerprint)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_record_fingerprint.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="training executor fingerprint"):
        load_gemma3_structured_single_layer_artifact(
            forged_record_fingerprint
        )

    forged_prompt_binding = tmp_path / "forged-prompt-binding.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["tokenized_splits"]["calibration_b"][
        "source_prompt_sha256"
    ][0] = "b" * 64
    changed["selection"]["tokenized_stream"][
        "source_prompt_sha256"
    ][0] = "b" * 64
    _rehash_payload(changed)
    torch.save(changed, forged_prompt_binding)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_prompt_binding.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="prompt binding"):
        load_gemma3_structured_single_layer_artifact(
            forged_prompt_binding
        )

    forged_family_binding = tmp_path / "forged-family-binding.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["prompt_families"][
        "per_prompt_family_sha256"
    ]["calibration_b"][0] = "b" * 64
    _rehash_payload(changed)
    torch.save(changed, forged_family_binding)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_family_binding.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="family binding"):
        load_gemma3_structured_single_layer_artifact(
            forged_family_binding
        )

    forged_claim_binding = tmp_path / "forged-claim-binding.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["calibration_b_claim"][
        "role_prompt_count"
    ] += 1
    _rehash_payload(changed)
    torch.save(changed, forged_claim_binding)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_claim_binding.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="claim binding"):
        load_gemma3_structured_single_layer_artifact(
            forged_claim_binding
        )

    forged_status = tmp_path / "forged-status.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["scientific_status"]["compression_attempted"] = True
    _rehash_payload(changed)
    torch.save(changed, forged_status)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_status.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="status"):
        load_gemma3_structured_single_layer_artifact(forged_status)

    forged_fisher = tmp_path / "forged-fisher.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["activation_fisher"]["matrix"][0, 0] += 1.0
    _rehash_payload(changed)
    torch.save(changed, forged_fisher)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_fisher.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="Fisher matrix"):
        load_gemma3_structured_single_layer_artifact(forged_fisher)

    forged_scale = tmp_path / "forged-scale-provenance.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["structured_scales"][
        "calibration_split_sha256"
    ] = "b" * 64
    _rehash_payload(changed)
    torch.save(changed, forged_scale)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_scale.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="scales provenance"):
        load_gemma3_structured_single_layer_artifact(forged_scale)

    forged_scale_policy = tmp_path / "forged-scale-policy.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["relative_median_scale_floor"] = "1.0"
    _rehash_payload(changed)
    torch.save(changed, forged_scale_policy)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_scale_policy.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="relative median scale floor"):
        load_gemma3_structured_single_layer_artifact(
            forged_scale_policy
        )

    forged_scale_rows = tmp_path / "forged-scale-rows.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["structured_scales"]["valid_rows"] += 1
    _rehash_payload(changed)
    torch.save(changed, forged_scale_rows)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_scale_rows.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="scale rows"):
        load_gemma3_structured_single_layer_artifact(forged_scale_rows)

    forged_scale_floor = tmp_path / "forged-scale-floor.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    scale_name = "normalized_attention_input"
    scale = changed["training"]["structured_scales"]["values"][scale_name]
    scale[0] = changed["training"]["structured_scales"]["floor"]
    changed["training"]["structured_scales"]["sha256"][scale_name] = (
        _tensor_sha256(
            scale,
            domain=(
                b"fisher_graph.structured_layer.scale.v1\0"
                + scale_name.encode("utf-8")
                + b"\0"
            ),
        )
    )
    _rehash_payload(changed)
    torch.save(changed, forged_scale_floor)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_scale_floor.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="recorded floor"):
        load_gemma3_structured_single_layer_artifact(forged_scale_floor)

    forged_corpus_audit = tmp_path / "forged-corpus-audit.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["corpus_audit"]["payload"][
        "prior_local_exact_prompt_overlap_count"
    ] = 1
    _rehash_payload(changed)
    torch.save(changed, forged_corpus_audit)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_corpus_audit.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="corpus audit binding"):
        load_gemma3_structured_single_layer_artifact(
            forged_corpus_audit
        )

    forged_training_positions = tmp_path / "forged-positions.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["training_recipe"][
        "train_positions_per_sequence"
    ] += 1
    _rehash_payload(changed)
    torch.save(changed, forged_training_positions)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_training_positions.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="Fisher matrix"):
        load_gemma3_structured_single_layer_artifact(
            forged_training_positions
        )

    forged_energy_weight = tmp_path / "forged-energy-weight.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["training_recipe"]["energy_loss_weight"] = 2.0
    _rehash_payload(changed)
    torch.save(changed, forged_energy_weight)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_energy_weight.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="training report"):
        load_gemma3_structured_single_layer_artifact(
            forged_energy_weight
        )

    forged_initialization = tmp_path / "forged-initialization.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["training_recipe"][
        "rmsnorm_initialization"
    ] = "forged"
    _rehash_payload(changed)
    torch.save(changed, forged_initialization)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_initialization.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="RMSNorm initialization"):
        load_gemma3_structured_single_layer_artifact(
            forged_initialization
        )

    forged_a_preflight = tmp_path / "forged-a-preflight.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["calibration_a_fidelity"][
        "primary_passed"
    ] = False
    _rehash_payload(changed)
    torch.save(changed, forged_a_preflight)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_a_preflight.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="calibration-A preflight"):
        load_gemma3_structured_single_layer_artifact(
            forged_a_preflight
        )

    forged_accounting = tmp_path / "forged-accounting.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["selection"]["accounting"]["structured_source_visibility"][
        "logical_analytic_mac_count"
    ] += 1
    _rehash_payload(changed)
    torch.save(changed, forged_accounting)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_accounting.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="accounting does not recompute"):
        load_gemma3_structured_single_layer_artifact(forged_accounting)

    forged_test_stream = tmp_path / "forged-test-stream.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["protocol"]["tokenized_splits"]["test"] = changed["protocol"][
        "tokenized_splits"
    ]["calibration_b"]
    _rehash_payload(changed)
    torch.save(changed, forged_test_stream)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_test_stream.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="reserved-test"):
        load_gemma3_structured_single_layer_artifact(forged_test_stream)


def test_failed_calibration_b_never_tokenizes_validation_or_test(
    tmp_path: Path,
) -> None:
    model = _tiny_gemma3()
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    output = tmp_path / "rejected.pt"
    tokenizer = RecordingTokenizer()

    with patch(
        "fisher_graph.gemma3_structured_single_layer_experiment."
        "load_gemma3",
        return_value=(tokenizer, model),
    ):
        report = run_gemma3_structured_single_layer_experiment(
            prompt_splits_path=prompt_path,
            family_manifest_path=family_path,
            model_id="tiny-local-gemma3",
            revision="a" * 40,
            cache_dir=tmp_path / "cache",
            layer_index=0,
            max_length=8,
            tokenization_batch_size=2,
            local_warmup_steps=0,
            train_steps=1,
            train_positions_per_sequence=1,
            learning_rate=1e-3,
            selection_nll_atol=0.0,
            selection_top1_min=1.0,
            selection_teacher_kl_max=0.0,
            selection_p90_abs_nll_max=0.0,
            selection_p10_top1_min=1.0,
            block_delta_nrmse_max=1e6,
            block_delta_cosine_min=-1.0,
            branch_delta_nrmse_max=1e6,
            branch_delta_cosine_min=-1.0,
            minimum_calibration_a_prompts=1,
            minimum_heldout_prompts=1,
            minimum_fisher_rows=1,
            minimum_train_supervised_tokens=1,
            minimum_heldout_supervised_tokens=1,
            minimum_length_buckets=1,
            device_name="cpu",
            dtype="float32",
            output=output,
            calibration_b_ledger_dir=tmp_path / "heldout-ledger",
        )

    status = report["scientific_status"]
    assert status["outcome"] == "rejected_on_calibration_b"
    assert not status["calibration_b_passed"]
    assert not status["validation_evaluated"]
    assert not status["validation_passed"]
    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert not set(prompts["validation"]) & tokenized
    assert not set(prompts["test"]) & tokenized

    loaded = load_gemma3_structured_single_layer_artifact(output)
    assert not loaded["selection"]["passed"]
    assert loaded["validation"] == {
        "evaluated": False,
        "reason": "calibration_b_failed_validation_not_tokenized",
        "behavior": None,
        "direct": None,
        "branches": None,
        "execution_audits": None,
        "ordinary_vs_segmented_native": None,
        "native_boundary_replay": None,
        "logical_accounting": None,
        "gates": None,
        "accounting": None,
        "resource_gates_applied": False,
        "resource_diagnostics_only": True,
        "passed": False,
        "locked_candidate": None,
        "tokenized_stream": None,
        "tokenized_stream_contract": None,
        "executor_fingerprints": None,
    }


def test_operator_bootstrap_format5_is_compact_zero_update_and_bound(
    tmp_path: Path,
) -> None:
    model = _tiny_gemma3()
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompts["calibration_a"] = [
        f"structured bootstrap train prompt {index}"
        for index in range(12)
    ]
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    families = _family_payload()
    families["calibration_a"] = [
        "train-bootstrap-a"
        if index < 6
        else "train-bootstrap-b"
        for index in range(12)
    ]
    family_path.write_text(
        json.dumps(families),
        encoding="utf-8",
    )
    output = tmp_path / "bootstrap.pt"
    ledger_dir = tmp_path / "heldout-ledger"
    tokenizer = RecordingTokenizer()
    original_bootstrap = (
        structured_experiment.bootstrap_structured_operator_executor_
    )
    compact_capture_ids: list[int] = []

    def audited_bootstrap(
        destination: object,
        batches: object,
        **kwargs: object,
    ) -> object:
        assert isinstance(batches, tuple)
        assert len(batches) == 1
        compact = batches[0]
        compact_capture_ids.append(id(compact))
        assert compact.valid_positions.shape[1] == 1
        assert bool(compact.valid_positions.all())
        assert len(compact.activations) == 18
        assert all(
            value.shape[:2] == compact.valid_positions.shape
            for value in compact.activations.values()
        )
        return original_bootstrap(destination, batches, **kwargs)

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            return_value=(tokenizer, model),
        ),
        patch.object(
            structured_experiment,
            "bootstrap_structured_operator_executor_",
            side_effect=audited_bootstrap,
        ),
        patch.object(
            structured_experiment.torch.optim,
            "AdamW",
            side_effect=AssertionError(
                "operator bootstrap must not construct AdamW"
            ),
        ),
    ):
        report = run_gemma3_structured_single_layer_experiment(
            **_tiny_run_kwargs(
                tmp_path,
                prompt_path=prompt_path,
                family_path=family_path,
                output=output,
                ledger_dir=ledger_dir,
                operator_bootstrap=True,
                local_warmup_steps=0,
                train_steps=0,
                operator_bootstrap_rows=32,
                operator_bootstrap_rank_rtol=1e-15,
                operator_bootstrap_max_condition=1e16,
            )
        )

    assert report["format_version"] == 5
    assert len(compact_capture_ids) == 2
    assert compact_capture_ids[0] == compact_capture_ids[1]
    recipe = report["protocol"]["training_recipe"]
    assert recipe["optimizer"] == "none"
    assert recipe["optimizer_steps"] == 0
    assert recipe["suffix_training_steps"] == 0
    for name in (
        "structured_source_visibility",
        "attention_output_disabled_control",
    ):
        training = report["training"][name]
        assert training["optimizer"] == "none"
        assert training["optimizer_steps"] == 0
        assert training["suffix_training_steps"] == 0
        bootstrap = training["bootstrap"]
        assert bootstrap["row_selection"][
            "selection_applied_before_activation_capture"
        ]
        assert bootstrap["row_selection"][
            "capture_contains_only_selected_rows"
        ]
        assert not bootstrap["activation_targets_serialized"]
        assert not bootstrap["sufficient_statistics_serialized"]
        assert training["capture_audit"][
            "source_model_executed_for_activation_capture"
        ]
        assert not training["capture_audit"][
            "compiler_source_parameter_tensor_read"
        ]
    loaded = load_gemma3_structured_single_layer_artifact(output)
    assert loaded["scientific_status"]["calibration_b_evaluated"]

    tampered = tmp_path / "bootstrap-tampered.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["structured_source_visibility"]["bootstrap"][
        "coefficient_sha256"
    ] = "b" * 64
    _rehash_payload(changed)
    torch.save(changed, tampered)
    shutil.copyfile(
        output.with_suffix(".json"),
        tampered.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="bootstrap report binding"):
        load_gemma3_structured_single_layer_artifact(tampered)

    policy_tampered = tmp_path / "bootstrap-policy-tampered.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["structured_source_visibility"]["bootstrap"][
        "solver"
    ]["maximum_nullity"] = 0
    _rehash_payload(changed)
    torch.save(changed, policy_tampered)
    shutil.copyfile(
        output.with_suffix(".json"),
        policy_tampered.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="bootstrap report binding"):
        load_gemma3_structured_single_layer_artifact(policy_tampered)


def test_operator_bootstrap_stop_after_a_ignores_b_ledger_and_stops(
    tmp_path: Path,
) -> None:
    model = _tiny_gemma3()
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompts["calibration_a"] = [
        f"structured bootstrap stop prompt {index}"
        for index in range(12)
    ]
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    families = _family_payload()
    families["calibration_a"] = [
        f"train-stop-{index // 6}" for index in range(12)
    ]
    family_path.write_text(json.dumps(families), encoding="utf-8")
    ledger_dir = tmp_path / "heldout-ledger"
    metadata = structured_experiment.load_gemma3_prompt_splits(
        prompt_path
    ).metadata()
    claim_path = structured_experiment._calibration_b_claim_path(
        ledger_dir,
        metadata["per_prompt_sha256"]["calibration_b"],
    )
    claim_path.parent.mkdir(parents=True)
    claim_path.write_text("{already-consumed", encoding="utf-8")
    output = tmp_path / "stopped.pt"
    tokenizer = RecordingTokenizer()
    bootstrap_calls = 0
    original_bootstrap = (
        structured_experiment.bootstrap_structured_operator_executor_
    )

    def counted_bootstrap(*args: object, **kwargs: object) -> object:
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            return_value=(tokenizer, model),
        ),
        patch.object(
            structured_experiment,
            "bootstrap_structured_operator_executor_",
            side_effect=counted_bootstrap,
        ),
        patch.object(
            structured_experiment.torch.optim,
            "AdamW",
            side_effect=AssertionError("A-only bootstrap cannot use AdamW"),
        ),
    ):
        report = run_gemma3_structured_single_layer_experiment(
            **_tiny_run_kwargs(
                tmp_path,
                prompt_path=prompt_path,
                family_path=family_path,
                output=output,
                ledger_dir=ledger_dir,
                operator_bootstrap=True,
                local_warmup_steps=0,
                train_steps=0,
                operator_bootstrap_rows=32,
                operator_bootstrap_rank_rtol=1e-15,
                operator_bootstrap_max_condition=1e16,
                stop_after_calibration_a=True,
            )
        )

    assert bootstrap_calls == 1
    assert report["scientific_status"] == {
        "outcome": "calibration_a_only_passed",
        "calibration_a_passed": True,
        "stopped_after_calibration_a": True,
        "calibration_b_opened": False,
        "diagnostic_only": True,
        "scientific_parent": False,
    }
    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert set(prompts["calibration_a"]).issubset(tokenized)
    assert not set(prompts["calibration_b"]) & tokenized
    assert claim_path.read_text(encoding="utf-8") == "{already-consumed"
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert output.with_suffix(".calibration-a.json").is_file()


def test_operator_bootstrap_a_failure_never_opens_control_or_b(
    tmp_path: Path,
) -> None:
    model = _tiny_gemma3()
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompts["calibration_a"] = [
        f"structured bootstrap reject prompt {index}"
        for index in range(12)
    ]
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    families = _family_payload()
    families["calibration_a"] = [
        f"train-reject-{index // 6}" for index in range(12)
    ]
    family_path.write_text(json.dumps(families), encoding="utf-8")
    ledger_dir = tmp_path / "heldout-ledger"
    output = tmp_path / "bootstrap-rejected.pt"
    tokenizer = RecordingTokenizer()
    bootstrap_calls = 0
    original_bootstrap = (
        structured_experiment.bootstrap_structured_operator_executor_
    )

    def counted_bootstrap(*args: object, **kwargs: object) -> object:
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        return original_bootstrap(*args, **kwargs)

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            return_value=(tokenizer, model),
        ),
        patch.object(
            structured_experiment,
            "bootstrap_structured_operator_executor_",
            side_effect=counted_bootstrap,
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="calibration-A direct fidelity failed",
        ):
            run_gemma3_structured_single_layer_experiment(
                **_tiny_run_kwargs(
                    tmp_path,
                    prompt_path=prompt_path,
                    family_path=family_path,
                    output=output,
                    ledger_dir=ledger_dir,
                    operator_bootstrap=True,
                    local_warmup_steps=0,
                    train_steps=0,
                    operator_bootstrap_rows=32,
                    operator_bootstrap_rank_rtol=1e-15,
                    operator_bootstrap_max_condition=1e16,
                    block_delta_nrmse_max=0.0,
                )
            )

    assert bootstrap_calls == 1
    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert not set(prompts["calibration_b"]) & tokenized
    assert not ledger_dir.exists()
    assert not output.exists()
    preflight = json.loads(
        output.with_suffix(".calibration-a.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight["scientific_status"]["outcome"] == (
        "rejected_on_calibration_a"
    )
    assert set(preflight["training"]) == {
        "structured_source_visibility"
    }


def test_failed_calibration_a_preflight_never_tokenizes_calibration_b(
    tmp_path: Path,
) -> None:
    model = _tiny_gemma3()
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    output = tmp_path / "nested" / "outputs" / "a-rejected.pt"
    tokenizer = RecordingTokenizer()

    with patch(
        "fisher_graph.gemma3_structured_single_layer_experiment."
        "load_gemma3",
        return_value=(tokenizer, model),
    ):
        with pytest.raises(
            RuntimeError,
            match="calibration-A direct fidelity failed",
        ):
            run_gemma3_structured_single_layer_experiment(
                prompt_splits_path=prompt_path,
                family_manifest_path=family_path,
                model_id="tiny-local-gemma3",
                revision="a" * 40,
                cache_dir=tmp_path / "cache",
                layer_index=0,
                max_length=8,
                tokenization_batch_size=2,
                local_warmup_steps=0,
                train_steps=1,
                train_positions_per_sequence=1,
                learning_rate=1e-3,
                selection_nll_atol=1e6,
                selection_top1_min=0.0,
                selection_teacher_kl_max=1e6,
                selection_p90_abs_nll_max=1e6,
                selection_p10_top1_min=0.0,
                block_delta_nrmse_max=0.0,
                block_delta_cosine_min=-1.0,
                branch_delta_nrmse_max=1e6,
                branch_delta_cosine_min=-1.0,
                minimum_calibration_a_prompts=1,
                minimum_heldout_prompts=1,
                minimum_fisher_rows=1,
                minimum_train_supervised_tokens=1,
                minimum_heldout_supervised_tokens=1,
                minimum_length_buckets=1,
                device_name="cpu",
                dtype="float32",
                output=output,
                calibration_b_ledger_dir=tmp_path / "heldout-ledger",
            )

    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert set(prompts["calibration_a"]).issubset(tokenized)
    assert not set(prompts["calibration_b"]) & tokenized
    assert not set(prompts["validation"]) & tokenized
    assert not set(prompts["test"]) & tokenized
    assert output.with_suffix(".calibration-a.json").is_file()
    preflight = json.loads(
        output.with_suffix(".calibration-a.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(preflight["training"]) == {
        "structured_source_visibility"
    }
    primary_training = preflight["training"][
        "structured_source_visibility"
    ]
    assert primary_training["snapshots"]
    assert (
        primary_training["final_execution_fingerprint"]
        == preflight["executor_fingerprints"][
            "structured_source_visibility"
        ]
        == preflight["fidelity"]["executor_fingerprints"][
            "structured_source_visibility"
        ]
    )
    assert not output.exists()
    assert not output.with_suffix(".json").exists()


def test_calibration_b_claim_precedes_tokenization_and_blocks_reuse(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(_prompt_payload()),
        encoding="utf-8",
    )
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    ledger_dir = tmp_path / "heldout-ledger"
    output = tmp_path / "first.pt"
    model = _tiny_gemma3()
    tokenizer = RecordingTokenizer()
    original_materialize = structured_experiment._materialize_split
    observed_claim = False

    def guarded_materialize(*args: object, **kwargs: object) -> object:
        nonlocal observed_claim
        if kwargs.get("split_name") == "calibration_b":
            markers = list(ledger_dir.glob("*.json"))
            assert len(markers) == 1
            claim = json.loads(markers[0].read_text(encoding="utf-8"))
            assert claim["state"] == "claimed_before_tokenization"
            observed_claim = True
        return original_materialize(*args, **kwargs)

    with (
        patch(
            "fisher_graph.gemma3_structured_single_layer_experiment."
            "load_gemma3",
            return_value=(tokenizer, model),
        ),
        patch.object(
            structured_experiment,
            "_materialize_split",
            side_effect=guarded_materialize,
        ),
    ):
        run_gemma3_structured_single_layer_experiment(
            **_tiny_run_kwargs(
                tmp_path,
                prompt_path=prompt_path,
                family_path=family_path,
                output=output,
                ledger_dir=ledger_dir,
            )
        )

    assert observed_claim
    reordered_prompts = _prompt_payload()
    reordered_prompts["calibration_b"] = list(
        reversed(reordered_prompts["calibration_b"])
    )
    reordered_prompt_path = tmp_path / "reordered-prompts.json"
    reordered_prompt_path.write_text(
        json.dumps(reordered_prompts),
        encoding="utf-8",
    )
    reordered_families = _family_payload()
    reordered_families["calibration_b"] = list(
        reversed(reordered_families["calibration_b"])
    )
    reordered_family_path = tmp_path / "reordered-families.json"
    reordered_family_path.write_text(
        json.dumps(reordered_families),
        encoding="utf-8",
    )
    with patch.object(
        structured_experiment,
        "load_gemma3",
        side_effect=AssertionError("model must not load on heldout reuse"),
    ):
        with pytest.raises(FileExistsError, match="already claimed"):
            run_gemma3_structured_single_layer_experiment(
                **_tiny_run_kwargs(
                    tmp_path,
                    prompt_path=reordered_prompt_path,
                    family_path=reordered_family_path,
                    output=tmp_path / "different-output.pt",
                    ledger_dir=ledger_dir,
                )
            )


def test_calibration_b_exception_consumes_claim_and_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload()
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    ledger_dir = tmp_path / "heldout-ledger"
    model = _tiny_gemma3()
    tokenizer = RecordingTokenizer()
    original_materialize = structured_experiment._materialize_split

    def failing_materialize(*args: object, **kwargs: object) -> object:
        if kwargs.get("split_name") == "calibration_b":
            assert list(ledger_dir.glob("*.json"))
            raise RuntimeError("synthetic calibration-B tokenization failure")
        return original_materialize(*args, **kwargs)

    with (
        patch.object(
            structured_experiment,
            "load_gemma3",
            return_value=(tokenizer, model),
        ),
        patch.object(
            structured_experiment,
            "_materialize_split",
            side_effect=failing_materialize,
        ),
    ):
        with pytest.raises(RuntimeError, match="synthetic"):
            run_gemma3_structured_single_layer_experiment(
                **_tiny_run_kwargs(
                    tmp_path,
                    prompt_path=prompt_path,
                    family_path=family_path,
                    output=tmp_path / "tokenization-failed.pt",
                    ledger_dir=ledger_dir,
                )
            )

    markers = list(ledger_dir.glob("*.json"))
    assert len(markers) == 1
    with patch.object(
        structured_experiment,
        "load_gemma3",
        side_effect=AssertionError("consumed claim must fail before load"),
    ):
        with pytest.raises(FileExistsError, match="already claimed"):
            run_gemma3_structured_single_layer_experiment(
                **_tiny_run_kwargs(
                    tmp_path,
                    prompt_path=prompt_path,
                    family_path=family_path,
                    output=tmp_path / "retry.pt",
                    ledger_dir=ledger_dir,
                )
            )

    markers[0].write_text("{corrupt", encoding="utf-8")
    with patch.object(
        structured_experiment,
        "load_gemma3",
        side_effect=AssertionError("corrupt claim must fail before load"),
    ):
        with pytest.raises(FileExistsError, match="already claimed"):
            run_gemma3_structured_single_layer_experiment(
                **_tiny_run_kwargs(
                    tmp_path,
                    prompt_path=prompt_path,
                    family_path=family_path,
                    output=tmp_path / "corrupt-retry.pt",
                    ledger_dir=ledger_dir,
                )
            )
