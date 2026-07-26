import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_full_width_single_layer_experiment import (
    FAMILY_STATUS,
    PROMPT_STATUS,
    _report_sha256,
    _scientific_payload_sha256,
    _tracked_prompt_exclusion_audit,
    load_gemma3_full_width_single_layer_artifact,
    load_prompt_family_manifest,
    run_gemma3_full_width_single_layer_experiment,
)
from fisher_graph.model import ToyTransformer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"
    name_or_path = "full-width-layer-test-tokenizer"
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
                2,
                3 + digest[0] % 28,
                3 + digest[1] % 28,
            ]
            if digest[2] % 2:
                row.append(3 + digest[3] % 28)
            row.append(1)
            unpadded.append(row)
        width = max(len(row) for row in unpadded)
        rows = [
            row + [self.pad_token_id] * (width - len(row))
            for row in unpadded
        ]
        masks = [
            [True] * len(row) + [False] * (width - len(row))
            for row in unpadded
        ]
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(masks, dtype=torch.bool),
        }


def _prompt_payload(prefix: str) -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": PROMPT_STATUS,
        "calibration_a": [
            f"{prefix} train prompt {index}" for index in range(4)
        ],
        "calibration_b": [
            f"{prefix} selection prompt {index}" for index in range(2)
        ],
        "validation": [
            f"{prefix} validation prompt {index}" for index in range(2)
        ],
        "test": [
            f"{prefix} reserved test prompt {index}" for index in range(2)
        ],
    }


def _family_payload() -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_family_manifest",
        "format_version": 1,
        "scientific_status": FAMILY_STATUS,
        "calibration_a": ["train-a", "train-a", "train-b", "train-b"],
        "calibration_b": ["select-a", "select-b"],
        "validation": ["validation-a", "validation-b"],
        "test": ["test-a", "test-b"],
    }


def _all_behavior_gates(
    _behavior: object,
    **_kwargs: object,
) -> dict[str, bool]:
    return {
        "absolute_delta_nll_per_token": True,
        "aggregate_top1_agreement": True,
        "teacher_kl_per_token": True,
        "per_prompt_p90_absolute_delta_nll": True,
        "per_prompt_p10_top1_agreement": True,
    }


def _model() -> ToyTransformer:
    torch.manual_seed(91104)
    model = ToyTransformer(
        TransformerConfig(
            vocab_size=31,
            max_sequence_length=8,
            d_model=8,
            n_heads=2,
            n_layers=3,
            d_ff=12,
            dropout=0.0,
        )
    ).eval()
    model.requires_grad_(False)
    return model


def test_family_manifest_requires_cross_role_disjointness(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(_prompt_payload("family")),
        encoding="utf-8",
    )
    from fisher_graph.gemma3_stability_experiment import (
        load_gemma3_prompt_splits,
    )

    prompts = load_gemma3_prompt_splits(prompt_path)
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    manifest = load_prompt_family_manifest(
        family_path,
        prompts=prompts,
    )
    assert manifest.metadata()["cross_role_overlap_count"] == 0

    changed = _family_payload()
    changed["validation"][0] = "select-a"
    family_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        load_prompt_family_manifest(family_path, prompts=prompts)


def test_prompt_exclusion_includes_legacy_text_fixture(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    payload = _prompt_payload("legacy")
    payload["calibration_a"][0] = (
        "The quiet library opened before sunrise."
    )
    prompt_path.write_text(json.dumps(payload), encoding="utf-8")
    from fisher_graph.gemma3_stability_experiment import (
        load_gemma3_prompt_splits,
    )

    prompts = load_gemma3_prompt_splits(prompt_path)
    with pytest.raises(ValueError, match="tracked prior"):
        _tracked_prompt_exclusion_audit(
            prompts,
            prompt_path=prompt_path,
        )


def test_full_width_layer_fit_validation_and_strict_artifact(
    tmp_path: Path,
) -> None:
    model = _model()
    versions = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
    }
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload("fresh")
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    output = tmp_path / "full-width.pt"
    tokenizer = RecordingTokenizer()
    with patch(
        "fisher_graph.gemma3_full_width_single_layer_experiment."
        "load_gemma3",
        return_value=(tokenizer, model),
    ), patch(
        "fisher_graph.gemma3_full_width_single_layer_experiment."
        "Gemma3CausalLMAdapter",
        side_effect=ToyTransformerAdapter,
    ), patch(
        "fisher_graph.gemma3_full_width_single_layer_experiment."
        "_behavior_gates",
        side_effect=_all_behavior_gates,
    ):
        report = run_gemma3_full_width_single_layer_experiment(
            prompt_splits_path=prompt_path,
            family_manifest_path=family_path,
            cache_dir=tmp_path / "cache",
            layer_index=0,
            max_length=8,
            tokenization_batch_size=2,
            hidden_width=8,
            executor_layers=1,
            head_count=2,
            feed_forward_width=12,
            local_warmup_steps=0,
            train_steps=1,
            train_positions_per_sequence=1,
            selection_nll_atol=1e6,
            selection_top1_min=0.0,
            selection_teacher_kl_max=1e6,
            selection_p90_abs_nll_max=1e6,
            selection_p10_top1_min=0.0,
            block_delta_nrmse_max=1e6,
            block_delta_cosine_min=-1.0,
            max_stored_coefficient_ratio=100.0,
            max_analytic_mac_ratio=100.0,
            minimum_calibration_a_prompts=1,
            minimum_heldout_prompts=1,
            minimum_fisher_rows=1,
            minimum_train_supervised_tokens=1,
            minimum_heldout_supervised_tokens=1,
            minimum_length_buckets=1,
            device_name="cpu",
            dtype="float32",
            output=output,
        )

    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert report["scientific_status"]["retained_rank"] == 8
    assert not report["scientific_status"]["rank_reduction_attempted"]
    assert report["scientific_status"][
        "source_layer_removed_from_student_path"
    ]
    assert report["scientific_status"]["validation_evaluated"]
    assert report["scientific_status"][
        "candidate_passed_recorded_single_seed_protocol"
    ]
    assert not report["scientific_status"][
        "fidelity_viable_replacement"
    ]
    assert not report["scientific_status"][
        "parameter_reduction_observed_under_recorded_protocol"
    ]
    assert not report["scientific_status"][
        "analytic_mac_reduction_observed_under_recorded_protocol"
    ]
    assert report["training"]["activation_fisher"]["width"] == 8
    assert all(
        probe["synthetic_invalid_padding_slots"] == 2
        and probe["synthetic_padding_passed"] is True
        for probe in report["training"]["structural_probes"].values()
    )
    assert all(
        int(parameter._version) == versions[name]
        for name, parameter in model.named_parameters()
    )
    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert set(prompts["validation"]).issubset(tokenized)
    assert not set(prompts["test"]) & tokenized

    loaded = load_gemma3_full_width_single_layer_artifact(output)
    assert loaded["executors"]["full_causal"].retained_rank == 8
    assert loaded["executors"][
        "full_causal"
    ].config.causal_edges_enabled
    assert not loaded["executors"][
        "same_position_control"
    ].config.causal_edges_enabled
    assert loaded["selection"]["execution_audits"]["full_causal"][
        "source_block_calls_total"
    ] == 0
    fisher = loaded["training"]["activation_fisher"]
    assert fisher["matrix"].shape == (8, 8)
    assert fisher["normalized_diagonal"].shape == (8,)
    assert fisher["training_metric"].shape == (8, 8)
    assert torch.isfinite(fisher["matrix"]).all()

    tampered = tmp_path / "tampered.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["executors"]["full_causal"]["executor"][
        "model_state_dict"
    ]["output_head.bias"][0] += 1.0
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    changed["scientific_payload_sha256"] = (
        _scientific_payload_sha256(payload)
    )
    torch.save(changed, tampered)
    shutil.copyfile(
        output.with_suffix(".json"),
        tampered.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_gemma3_full_width_single_layer_artifact(tampered)

    forged_gate = tmp_path / "forged-gate.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["selection"]["behavior_gates"]["full_causal"][
        "aggregate_top1_agreement"
    ] = False
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    changed["scientific_payload_sha256"] = (
        _scientific_payload_sha256(payload)
    )
    torch.save(changed, forged_gate)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_gate.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="gates do not recompute"):
        load_gemma3_full_width_single_layer_artifact(forged_gate)

    forged_fisher = tmp_path / "forged-fisher.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["training"]["activation_fisher"][
        "rank_for_99_percent_trace"
    ] = 1
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    changed["scientific_payload_sha256"] = (
        _scientific_payload_sha256(payload)
    )
    torch.save(changed, forged_fisher)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_fisher.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="Fisher statistics"):
        load_gemma3_full_width_single_layer_artifact(forged_fisher)

    forged_logical_macs = tmp_path / "forged-logical-macs.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    accounting = changed["selection"]["accounting"]["full_causal"]
    accounting["logical_analytic_mac_count"] += 1
    accounting["analytic_mac_ratio_to_source"] = (
        accounting["logical_analytic_mac_count"]
        / accounting["source_layer_analytic_mac_count"]
    )
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    changed["scientific_payload_sha256"] = (
        _scientific_payload_sha256(payload)
    )
    torch.save(changed, forged_logical_macs)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_logical_macs.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="accounting"):
        load_gemma3_full_width_single_layer_artifact(
            forged_logical_macs
        )

    forged_source_macs = tmp_path / "forged-source-macs.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    accounting = changed["selection"]["accounting"]["full_causal"]
    accounting["source_layer_analytic_mac_count"] += 1
    accounting["analytic_mac_ratio_to_source"] = (
        accounting["logical_analytic_mac_count"]
        / accounting["source_layer_analytic_mac_count"]
    )
    accounting["reference_dense_mac_ratio_to_source"] = (
        accounting["reference_dense_prefix_mac_count"]
        / accounting["source_layer_analytic_mac_count"]
    )
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    changed["scientific_payload_sha256"] = (
        _scientific_payload_sha256(payload)
    )
    torch.save(changed, forged_source_macs)
    shutil.copyfile(
        output.with_suffix(".json"),
        forged_source_macs.with_suffix(".json"),
    )
    with pytest.raises(ValueError, match="accounting"):
        load_gemma3_full_width_single_layer_artifact(forged_source_macs)


def test_failed_calibration_b_does_not_tokenize_validation(
    tmp_path: Path,
) -> None:
    model = _model()
    prompt_path = tmp_path / "prompts.json"
    prompts = _prompt_payload("fail")
    prompt_path.write_text(json.dumps(prompts), encoding="utf-8")
    family_path = tmp_path / "families.json"
    family_path.write_text(
        json.dumps(_family_payload()),
        encoding="utf-8",
    )
    output = tmp_path / "negative.pt"
    tokenizer = RecordingTokenizer()
    with patch(
        "fisher_graph.gemma3_full_width_single_layer_experiment."
        "load_gemma3",
        return_value=(tokenizer, model),
    ), patch(
        "fisher_graph.gemma3_full_width_single_layer_experiment."
        "Gemma3CausalLMAdapter",
        side_effect=ToyTransformerAdapter,
    ):
        report = run_gemma3_full_width_single_layer_experiment(
            prompt_splits_path=prompt_path,
            family_manifest_path=family_path,
            cache_dir=tmp_path / "cache",
            layer_index=0,
            max_length=8,
            tokenization_batch_size=2,
            hidden_width=8,
            executor_layers=1,
            head_count=2,
            feed_forward_width=12,
            local_warmup_steps=0,
            train_steps=1,
            train_positions_per_sequence=1,
            max_stored_coefficient_ratio=0.0,
            max_analytic_mac_ratio=100.0,
            minimum_calibration_a_prompts=1,
            minimum_heldout_prompts=1,
            minimum_fisher_rows=1,
            minimum_train_supervised_tokens=1,
            minimum_heldout_supervised_tokens=1,
            minimum_length_buckets=1,
            device_name="cpu",
            dtype="float32",
            output=output,
        )
    assert not report["scientific_status"]["calibration_b_passed"]
    assert not report["scientific_status"]["validation_evaluated"]
    tokenized = {
        prompt
        for call in tokenizer.prompt_calls
        for prompt in call
    }
    assert not set(prompts["validation"]) & tokenized
    assert not set(prompts["test"]) & tokenized

    loaded = load_gemma3_full_width_single_layer_artifact(output)
    assert not loaded["validation"]["evaluated"]

    forged = tmp_path / "forged.pt"
    changed = torch.load(output, map_location="cpu", weights_only=True)
    changed["scientific_status"]["calibration_b_passed"] = True
    payload = {
        key: value
        for key, value in changed.items()
        if key not in {
            "scientific_payload_sha256",
            "report_sha256",
        }
    }
    digest = _scientific_payload_sha256(payload)
    changed["scientific_payload_sha256"] = digest
    forged_report = copy.deepcopy(report)
    forged_report["scientific_status"] = copy.deepcopy(
        changed["scientific_status"]
    )
    changed["report_sha256"] = _report_sha256(forged_report)
    torch.save(changed, forged)
    forged.with_suffix(".json").write_text(
        json.dumps(forged_report),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status"):
        load_gemma3_full_width_single_layer_artifact(forged)
