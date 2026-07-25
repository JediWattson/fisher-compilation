import copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from fisher_graph.adapters import ToyTransformerAdapter
from fisher_graph.codimension_projection import (
    CodimensionOneDeltaProjector,
    canonical_unit_direction,
)
from fisher_graph.config import TransformerConfig
from fisher_graph.gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
)
from fisher_graph.gemma3_rotated_span_executor_experiment import (
    _PROMPT_STATUS,
    _build_report,
    _report_sha256,
    _scientific_payload_sha256,
    load_gemma3_rotated_span_executor_artifact,
    run_gemma3_rotated_span_executor,
)
from fisher_graph.gemma3_stability_experiment import (
    load_gemma3_prompt_splits,
)
from fisher_graph.model import ToyTransformer


class RecordingTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"
    name_or_path = "rotated-span-executor-test-tokenizer"
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
        rows = [
            [2, 4 + index, 9 + index, 1]
            for index in range(len(prompts))
        ]
        input_ids = torch.tensor(rows)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(
                input_ids,
                dtype=torch.bool,
            ),
        }


def _prompt_payload(prefix: str) -> dict[str, object]:
    return {
        "schema": "fisher_graph.gemma3_prompt_splits",
        "format_version": 1,
        "scientific_status": _PROMPT_STATUS,
        "calibration_a": [
            f"{prefix} train prompt {index}" for index in range(64)
        ],
        "calibration_b": [
            f"{prefix} selection prompt {index}" for index in range(16)
        ],
        "validation": [
            f"{prefix} validation prompt {index}" for index in range(16)
        ],
        "test": [
            f"{prefix} reserved test prompt {index}" for index in range(16)
        ],
    }


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fake_rotation(
    *,
    model: ToyTransformer,
    prompt_path: Path,
) -> dict[str, object]:
    adapter = ToyTransformerAdapter(model)
    plan = adapter.plan_layer_block(0, 1)
    metadata = load_gemma3_prompt_splits(prompt_path).metadata()
    normal = canonical_unit_direction(
        torch.tensor(
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8],
            dtype=torch.float64,
        )
    )
    model_metadata = _model_provenance(
        model,
        model_id=DEFAULT_MODEL_ID,
        requested_revision=None,
    )
    status = {
        "basis_ordering_supported": True,
        "rank_639_fidelity_viable": True,
        "selection_failed": False,
        "test_evaluated": False,
    }
    return {
        "model": model_metadata,
        "rotated_projector": CodimensionOneDeltaProjector(normal),
        "locked_candidate": {
            "candidate_id": "rank_7.calibration_a_balanced_tail_rotation",
            "normal_source": "calibration_a_balanced_tail_rotation",
            "retained_rank": 7,
            "residual_width": 8,
            "retained_fraction": 0.875,
            "removed_dimensions": 1,
            "projection": (
                "target_informed_shared_euclidean_codimension_one_"
                "block_delta_projection"
            ),
        },
        "metadata": {
            "scientific_payload_sha256": _sha("rotation-payload"),
            "report_sha256": _sha("rotation-report"),
            "protocol": {
                "residual_width": 8,
                "start_layer": 0,
                "end_layer_inclusive": 1,
                "layer_ids": plan.layer_ids,
                "canonical_boundaries": plan.activation_sites,
                "prompt_splits": metadata,
            },
            "source_projection": {
                "prompt_disjointness": {
                    "projection_prompt_sha256": (
                        _sha("projection-prompt"),
                    ),
                    "weighted_prompt_sha256": (
                        _sha("weighted-prompt"),
                    ),
                    "gated_prompt_sha256": (_sha("gated-prompt"),),
                }
            },
        },
        "report": {
            "schema": "fisher_graph.gemma3_codimension_rotation",
            "format_version": 1,
            "scientific_status": status,
        },
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


def test_true_replacement_fit_validation_and_strict_artifact() -> None:
    torch.manual_seed(8801)
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
    versions = {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prompts_path = root / "prompts.json"
        prompts = _prompt_payload("fresh")
        prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
        rotation_prompt_path = root / "rotation-prompts.json"
        rotation_prompts = _prompt_payload("predecessor")
        rotation_prompt_path.write_text(
            json.dumps(rotation_prompts),
            encoding="utf-8",
        )
        source_path = root / "rotation.pt"
        source_path.write_bytes(b"synthetic rotation binding")
        rotation = _fake_rotation(
            model=model,
            prompt_path=rotation_prompt_path,
        )
        output = root / "executor.pt"
        tokenizer = RecordingTokenizer()
        with patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "load_gemma3_codimension_rotation_artifact",
            return_value=rotation,
        ), patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "load_gemma3",
            return_value=(tokenizer, model),
        ), patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ), patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "_behavior_gates",
            side_effect=_all_behavior_gates,
        ):
            report = run_gemma3_rotated_span_executor(
                rotation_artifact_path=source_path,
                cache_dir=root / "cache",
                prompt_splits_path=prompts_path,
                max_length=8,
                tokenization_batch_size=2,
                expert_count=1,
                expert_rank=2,
                router_width=2,
                modal_warmup_steps=0,
                train_steps=1,
                train_positions_per_sequence=1,
                learning_rate=1e-3,
                selection_nll_atol=1e6,
                selection_top1_min=0.0,
                selection_teacher_kl_max=1e6,
                selection_p90_abs_nll_max=1e6,
                selection_p10_top1_min=0.0,
                max_stored_coefficient_ratio=100.0,
                max_analytic_mac_ratio=100.0,
                device_name="cpu",
                dtype="float32",
                output=output,
            )

        assert output.is_file()
        assert output.with_suffix(".json").is_file()
        assert report["scientific_status"][
            "source_block_removed_from_student_path"
        ]
        assert report["scientific_status"][
            "fidelity_viable_replacement"
        ]
        assert report["analysis"]["training"]["structural_probes"][
            "batching_probe_scope"
        ] == "batched_padded_vs_trimmed_single_right_padding"
        assert all(
            int(parameter._version) == versions[name]
            for name, parameter in model.named_parameters()
        )
        flattened_calls = [
            prompt
            for call in tokenizer.prompt_calls
            for prompt in call
        ]
        assert not set(prompts["test"]) & set(flattened_calls)
        assert set(prompts["validation"]).issubset(flattened_calls)

        loaded = load_gemma3_rotated_span_executor_artifact(output)
        assert loaded["executor"].width == 8
        assert loaded["executor"].retained_rank == 7
        assert loaded["scientific_status"][
            "fidelity_viable_replacement"
        ]
        expected_parameter_reduction = all(
            loaded[split]["accounting"][
                "stored_coefficient_ratio_to_source"
            ]
            < 1.0
            for split in ("selection", "validation")
        )
        expected_mac_reduction = all(
            loaded[split]["accounting"][
                "analytic_mac_ratio_to_source"
            ]
            < 1.0
            for split in ("selection", "validation")
        )
        assert loaded["scientific_status"][
            "parameter_reduction_supported"
        ] is expected_parameter_reduction
        assert loaded["scientific_status"][
            "analytic_mac_reduction_supported"
        ] is expected_mac_reduction
        assert loaded["selection"]["execution_audit"][
            "source_block_calls_total"
        ] == 0

        tampered = root / "tampered.pt"
        changed = torch.load(output, map_location="cpu", weights_only=True)
        changed["executor"]["graph"]["model_state_dict"][
            "same_position_bias"
        ][0] += 1.0
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
        try:
            load_gemma3_rotated_span_executor_artifact(tampered)
        except ValueError as error:
            assert "fingerprint" in str(error)
        else:
            raise AssertionError("tampered executor state was accepted")


def test_calibration_b_failure_does_not_tokenize_validation() -> None:
    torch.manual_seed(8802)
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
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prompts_path = root / "prompts.json"
        prompts = _prompt_payload("fail")
        prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
        rotation_prompt_path = root / "rotation-prompts.json"
        rotation_prompt_path.write_text(
            json.dumps(_prompt_payload("old")),
            encoding="utf-8",
        )
        source_path = root / "rotation.pt"
        source_path.write_bytes(b"synthetic rotation binding")
        rotation = _fake_rotation(
            model=model,
            prompt_path=rotation_prompt_path,
        )
        output = root / "negative.pt"
        tokenizer = RecordingTokenizer()
        with patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "load_gemma3_codimension_rotation_artifact",
            return_value=rotation,
        ), patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "load_gemma3",
            return_value=(tokenizer, model),
        ), patch(
            "fisher_graph.gemma3_rotated_span_executor_experiment."
            "Gemma3CausalLMAdapter",
            side_effect=ToyTransformerAdapter,
        ):
            report = run_gemma3_rotated_span_executor(
                rotation_artifact_path=source_path,
                cache_dir=root / "cache",
                prompt_splits_path=prompts_path,
                max_length=8,
                tokenization_batch_size=2,
                expert_count=1,
                expert_rank=2,
                router_width=2,
                modal_warmup_steps=0,
                train_steps=1,
                train_positions_per_sequence=1,
                max_stored_coefficient_ratio=0.0,
                max_analytic_mac_ratio=100.0,
                device_name="cpu",
                dtype="float32",
                output=output,
            )
        assert not report["scientific_status"]["calibration_b_passed"]
        assert not report["scientific_status"]["validation_evaluated"]
        flattened_calls = {
            prompt
            for call in tokenizer.prompt_calls
            for prompt in call
        }
        assert not set(prompts["validation"]) & flattened_calls
        assert not set(prompts["test"]) & flattened_calls
        loaded = load_gemma3_rotated_span_executor_artifact(output)
        assert not loaded["validation"]["evaluated"]

        forged = root / "forged-pass.pt"
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
        forged_report = _build_report(
            payload,
            output=forged,
            scientific_digest=digest,
        )
        changed["report_sha256"] = _report_sha256(forged_report)
        torch.save(changed, forged)
        forged.with_suffix(".json").write_text(
            json.dumps(forged_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        try:
            load_gemma3_rotated_span_executor_artifact(forged)
        except ValueError as error:
            assert "scientific status" in str(error)
        else:
            raise AssertionError("forged scientific pass was accepted")
