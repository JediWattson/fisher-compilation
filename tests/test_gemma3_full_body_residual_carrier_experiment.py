from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph import gemma3_full_body_residual_carrier_experiment as experiment
from fisher_graph.complete_block_residual_forms import ResidualForm


def _tiny_model() -> torch.nn.Module:
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma3TextConfig(
        vocab_size=47,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=3,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config._commit_hash = experiment.DEFAULT_REVISION
    torch.manual_seed(74_003)
    return transformers.Gemma3ForCausalLM(config).eval().requires_grad_(False)


def _left_padded_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [0, 0, 2, 7, 3, 4],
                [2, 8, 9, 4, 3, 5],
            ],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                [False, False, True, True, True, True],
                [True, True, True, True, True, True],
            ]
        ),
    }


class _Tokenizer:
    padding_side = "left"
    pad_token_id = 0

    def __call__(
        self,
        prompts: list[str],
        *,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert return_tensors == "pt"
        rows: list[list[int]] = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            length = 4 + digest[0] % 2
            rows.append(
                [2]
                + [3 + digest[index] % 40 for index in range(length - 2)]
                + [1]
            )
        width = max(len(row) for row in rows)
        ids = []
        masks = []
        for row in rows:
            pad = width - len(row)
            ids.append([0] * pad + row)
            masks.append([False] * pad + [True] * len(row))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.bool),
        }


def _source_body_parameters(adapter: Gemma3CausalLMAdapter) -> int:
    return sum(
        parameter.numel()
        for layer in adapter.layers
        for parameter in adapter.source_module(layer.id).parameters()
    )


def test_compile_clones_one_direct_block_per_layer_without_rng_or_aliases() -> None:
    model = _tiny_model()
    adapter = Gemma3CausalLMAdapter(model)
    torch.manual_seed(91_117)
    before = torch.random.get_rng_state().clone()

    stack = experiment._compile_complete_body(
        adapter,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.equal(torch.random.get_rng_state(), before)
    assert stack.layer_ids == tuple(layer.id for layer in adapter.layers)
    assert len(stack.blocks) == len(adapter.layers) == 3
    assert all(block.form is ResidualForm.DIRECT_OUTPUT for block in stack.blocks)
    assert stack.learned_parameter_count == _source_body_parameters(adapter)
    assert stack.owns_source_model_weights is True
    assert stack.executor_local_source_free is False
    independence = experiment._assert_source_independence(
        model,
        {"compiled_full_body": stack},
    )
    assert independence["passed"] is True


def test_tiny_mixed_attention_full_body_matches_native_on_valid_queries() -> None:
    model = _tiny_model()
    adapter = Gemma3CausalLMAdapter(model)
    stack = experiment._compile_complete_body(
        adapter,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    metrics, stages, audit, accounting = experiment._evaluate_complete_body(
        adapter,
        (_left_padded_batch(),),
        stack,
    )
    candidate = metrics["compiled_full_body"]
    assert candidate["compared_query_rows"] == 10
    assert candidate["supervised_tokens"] == 8
    assert candidate["maximum_absolute_logit_error"] == 0.0
    assert candidate["delta_nll_per_token"] == 0.0
    assert candidate["native_to_candidate_kl_per_query"] == 0.0
    assert candidate["top1_agreement_to_native"] == 1.0
    assert all(value["maximum_absolute_error"] == 0.0 for value in stages.values())
    assert all(value["byte_identical"] is True for value in stages.values())
    assert audit["native_source_block_calls"] == {
        "layer.0": 1,
        "layer.1": 1,
        "layer.2": 1,
    }
    assert audit["compiled_source_block_calls"] == {
        "layer.0": 0,
        "layer.1": 0,
        "layer.2": 0,
    }
    assert audit["carrier_versions"] == [6]
    assert audit["mutation_receipt_counts"] == [6]
    assert len(audit["compiled_required_leaf_calls"]) == 3 * 13
    assert all(
        count == 1 for count in audit["compiled_required_leaf_calls"].values()
    )
    assert accounting["removed_parameter_count"] == 0
    assert accounting["removed_logical_macs"] == 0
    assert accounting["logical_total_macs"] > 0

    gates = experiment._exactness_gates(
        metrics,
        stages,
        stage_atol=experiment.DEFAULT_STAGE_ATOL,
        logit_atol=experiment.DEFAULT_LOGIT_ATOL,
        nll_atol=experiment.DEFAULT_NLL_ATOL,
        kl_max=experiment.DEFAULT_KL_MAX,
    )
    assert gates["passed"] is True


def test_resource_ledger_keeps_native_boundary_and_refuses_compression() -> None:
    model = _tiny_model()
    adapter = Gemma3CausalLMAdapter(model)
    stack = experiment._compile_complete_body(
        adapter,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    metrics, _stages, _audit, accounting = experiment._evaluate_complete_body(
        adapter,
        (_left_padded_batch(),),
        stack,
    )
    source_parameters = sum(value.numel() for value in model.parameters())
    source_stored = source_parameters + sum(
        value.numel() for value in model.buffers()
    )
    resources = experiment._resource_report(
        source_model_parameters=source_parameters,
        source_model_stored_scalars=source_stored,
        source_body_parameters=_source_body_parameters(adapter),
        stack=stack,
        logical_accounting=accounting,
        evaluated_valid_tokens=metrics["compiled_full_body"][
            "compared_query_rows"
        ],
        batch_count=1,
    )

    assert resources["candidate_deployment_parameter_count"] == source_parameters
    assert resources["native_external_boundary_parameter_count"] > 0
    assert resources["target_body_parameter_reduction_fraction"] == 0.0
    assert resources["whole_model_parameter_reduction_fraction"] == 0.0
    assert resources["target_body_logical_mac_reduction_fraction"] == 0.0
    assert resources["native_embedding_final_norm_and_head_retained"] is True
    assert resources["contains_cloned_source_checkpoint_tensors"] is True
    assert resources["compression_attempted"] is False
    assert resources["source_free_artifact_established"] is False
    assert resources["latency_measured"] is False
    assert resources["kernel_speedup_claimed"] is False
    assert resources["carrier_mutations"] == 6
    assert resources["logical_residual_mutation_scalar_additions"] == (
        2 * 3 * 10 * 16
    )


def test_runner_fails_closed_before_loading_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned Gemma"):
        experiment.run_gemma3_full_body_residual_carrier_experiment(
            revision="wrong",
            output=tmp_path / "unused.json",
        )
    with pytest.raises(ValueError, match="CPU float32"):
        experiment.run_gemma3_full_body_residual_carrier_experiment(
            device_name="mps",
            output=tmp_path / "unused.json",
        )
    with pytest.raises(ValueError, match="thresholds are pinned"):
        experiment.run_gemma3_full_body_residual_carrier_experiment(
            stage_atol=1.0,
            output=tmp_path / "unused.json",
        )
    existing = tmp_path / "existing.json"
    existing.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run_gemma3_full_body_residual_carrier_experiment(
            output=existing,
        )
    assert existing.read_text(encoding="utf-8") == "preserve me"


def test_mocked_runner_writes_strict_weightless_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model()
    tokenizer = _Tokenizer()
    split = {
        "panel_file": "synthetic.json",
        "panel_sha256": "a" * 64,
        "family_count": 2,
        "fit_example_count": 1,
        "evaluation_example_count": 3,
        "fit_example_ids_sha256": "b" * 64,
        "evaluation_example_ids_sha256": "c" * 64,
        "prompt_disjoint": True,
        "family_disjoint": False,
    }
    evaluation_prompts = ("held secret one", "held secret two", "held secret three")
    monkeypatch.setattr(experiment, "_EXPECTED_PINNED_LAYER_COUNT", 3)
    monkeypatch.setattr(
        experiment,
        "_load_prompt_split",
        lambda _path: (("fit secret",), evaluation_prompts, split),
    )
    monkeypatch.setattr(
        experiment,
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(
        experiment,
        "load_gemma3",
        lambda **_kwargs: (tokenizer, model),
    )
    output = tmp_path / "report.json"

    report = experiment.run_gemma3_full_body_residual_carrier_experiment(
        output=output,
        batch_size=2,
    )

    assert report == experiment.load_gemma3_full_body_residual_carrier_report(
        output
    )
    assert report["report_sha256"] == experiment._report_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert report["exactness_gates"]["passed"] is True
    assert report["execution_audit"]["carrier_versions"] == [6, 6]
    assert report["execution_audit"]["compiled_source_block_calls"] == {
        "layer.0": 0,
        "layer.1": 0,
        "layer.2": 0,
    }
    assert all(
        count == 2
        for count in report["execution_audit"][
            "compiled_required_leaf_calls"
        ].values()
    )
    assert report["compiled_body"]["execution_fingerprint_before"] == report[
        "compiled_body"
    ]["execution_fingerprint_after"]
    assert report["resources"]["compression_attempted"] is False
    assert report["scientific_status"][
        "native_embedding_final_norm_and_tied_head_retained"
    ] is True
    raw = output.read_text(encoding="utf-8")
    assert "fit secret" not in raw
    assert all(prompt not in raw for prompt in evaluation_prompts)
    assert "model_state_dict" not in raw

    def publish_tamper(
        name: str,
        mutate: object,
    ) -> Path:
        tampered = json.loads(raw)
        mutate(tampered)
        unhashed = {
            key: value
            for key, value in tampered.items()
            if key != "report_sha256"
        }
        tampered["report_sha256"] = experiment._report_sha256(unhashed)
        path = tmp_path / f"tampered-{name}.json"
        path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
        return path

    metric_tamper = publish_tamper(
        "metric",
        lambda value: value["metrics"]["compiled_full_body"].__setitem__(
            "maximum_absolute_logit_error",
            100.0,
        ),
    )
    with pytest.raises(ValueError, match="exactness gates"):
        experiment.load_gemma3_full_body_residual_carrier_report(metric_tamper)

    token_tamper = publish_tamper(
        "tokens",
        lambda value: value["protocol"]["tokenization"].__setitem__(
            "valid_token_count",
            value["protocol"]["tokenization"]["valid_token_count"] + 1,
        ),
    )
    with pytest.raises(ValueError, match="metric token scope"):
        experiment.load_gemma3_full_body_residual_carrier_report(token_tamper)

    resource_tamper = publish_tamper(
        "resources",
        lambda value: value["resources"].__setitem__(
            "candidate_deployment_parameter_count",
            value["resources"]["candidate_deployment_parameter_count"] + 1,
        ),
    )
    with pytest.raises(ValueError, match="resource closure"):
        experiment.load_gemma3_full_body_residual_carrier_report(
            resource_tamper
        )

    threshold_tamper = publish_tamper(
        "threshold",
        lambda value: value["exactness_gates"]["thresholds"].__setitem__(
            "stage_atol",
            1.0,
        ),
    )
    with pytest.raises(ValueError, match="thresholds"):
        experiment.load_gemma3_full_body_residual_carrier_report(
            threshold_tamper
        )

    token_hash_tamper = publish_tamper(
        "token-hash",
        lambda value: value["protocol"]["tokenization"].__setitem__(
            "combined_sha256",
            "d" * 64,
        ),
    )
    with pytest.raises(ValueError, match="combined token hash"):
        experiment.load_gemma3_full_body_residual_carrier_report(
            token_hash_tamper
        )

    claim_tamper = publish_tamper(
        "claim",
        lambda value: value["claims"].__setitem__(
            "parameter_compression",
            True,
        ),
    )
    with pytest.raises(ValueError, match="resource closure"):
        experiment.load_gemma3_full_body_residual_carrier_report(claim_tamper)
