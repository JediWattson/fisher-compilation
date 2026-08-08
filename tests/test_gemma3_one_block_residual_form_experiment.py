from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fisher_graph.complete_block_residual_forms import (
    CompleteBlockResidualForm,
    ResidualForm,
)
from fisher_graph import gemma3_one_block_residual_form_experiment as experiment
from test_structured_transformer_layer_executor import (
    _executor,
    _layer_spec,
    _sequence,
)


class _Config:
    model_type = "gemma3_text"
    architectures = ["SyntheticGemma3ForCausalLM"]
    hidden_size = 8
    num_hidden_layers = 1
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 2
    max_position_embeddings = 32
    sliding_window = 3
    layer_types = ["sliding_attention"]
    _commit_hash = experiment.DEFAULT_REVISION

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "model_type",
                "architectures",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "max_position_embeddings",
                "sliding_window",
                "layer_types",
            )
        }


class _NativeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.executor = _executor()
        self.execution: object | None = None

    def forward(self, hidden: torch.Tensor, sequence: object) -> torch.Tensor:
        self.execution = self.executor.forward_components(hidden, sequence)
        return self.execution.output


class _SyntheticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(120_401)
        self.config = _Config()
        self.embedding = nn.Embedding(19, 8)
        self.source = _NativeBlock()
        self.head = nn.Linear(8, 19, bias=False)


class _SyntheticAdapter:
    def __init__(self, model: _SyntheticModel) -> None:
        self.module = model
        self.layers = (_layer_spec(),)
        self.segments = (SimpleNamespace(ordinal=0),)

    def prepare_sequence(self, batch: dict[str, torch.Tensor]) -> object:
        return _sequence(batch["attention_mask"].bool())

    def embed(
        self,
        batch: dict[str, torch.Tensor],
        sequence: object,
    ) -> object:
        del sequence
        return SimpleNamespace(hidden_states=self.module.embedding(batch["input_ids"]))

    def run_segment(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("the only source segment must be skipped")

    def project_logits(
        self,
        hidden: torch.Tensor,
        sequence: object,
    ) -> torch.Tensor:
        del sequence
        return self.module.head(hidden)

    def source_module(self, layer_id: str) -> nn.Module:
        assert layer_id == "layer.0"
        return self.module.source

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        capture_sites: tuple[str, ...],
    ) -> object:
        sequence = self.prepare_sequence(batch)
        hidden = self.embed(batch, sequence).hidden_states
        output = self.module.source(hidden, sequence)
        execution = self.module.source.execution
        assert execution is not None
        sites = experiment._native_stage_sites(self, 0)
        activations = {
            site: getattr(execution, name)
            for name, site in sites.items()
        }
        assert set(activations) == set(capture_sites)
        return SimpleNamespace(
            logits=self.project_logits(output, sequence),
            activations=activations,
            sequence=sequence,
        )

    def model_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, value in self.module.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().numpy().tobytes())
        return digest.hexdigest()

    def execution_fingerprint(self) -> str:
        return "e" * 64


class _Tokenizer:
    def __call__(
        self,
        prompts: list[str],
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        rows = []
        for prompt in prompts:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            rows.append([1, 2 + digest[0] % 16, 2 + digest[1] % 16, 2])
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones(len(rows), 4, dtype=torch.bool),
        }


def _forms(model: _SyntheticModel) -> dict[str, CompleteBlockResidualForm]:
    state = model.source.executor.state_dict()
    result: dict[str, CompleteBlockResidualForm] = {}
    for arm, residual_form in experiment._resolve_residual_forms().items():
        executor = _executor()
        executor.load_state_dict(state, strict=True)
        result[arm] = CompleteBlockResidualForm(executor, residual_form).eval()
    return result


def _batch(
    offset: int = 0,
    *,
    padded: bool = False,
) -> dict[str, torch.Tensor]:
    mask = torch.ones(1, 4, dtype=torch.bool)
    if padded:
        mask[:, -1] = False
    return {
        "input_ids": torch.tensor(
            [[1, 2 + offset, 3 + offset, 4 + offset]],
            dtype=torch.long,
        ),
        "attention_mask": mask,
    }


def test_residual_form_enum_resolves_complete_canonical_panel() -> None:
    resolved = experiment._resolve_residual_forms()

    assert tuple(resolved) == experiment.ARM_ORDER
    assert resolved["explicit_residual"] is ResidualForm.EXPLICIT
    assert resolved["direct_state_update"] is ResidualForm.DIRECT_OUTPUT
    assert (
        resolved["drop_block_identity_control"]
        is ResidualForm.DROP_BLOCK_IDENTITY
    )


def test_synthetic_whole_model_path_audits_source_and_graph_calls() -> None:
    model = _SyntheticModel().eval().requires_grad_(False)
    adapter = _SyntheticAdapter(model)
    forms = _forms(model)

    metrics, stages, audit, accounting = experiment._evaluate_forms(
        adapter,
        (_batch(0), _batch(1, padded=True)),
        layer_index=0,
        forms=forms,
    )

    for arm in experiment.EXACT_ARMS:
        assert metrics[arm]["maximum_absolute_logit_error"] == 0.0
        assert metrics[arm]["top1_agreement_to_native"] == 1.0
        assert all(
            value["maximum_absolute_error"] == 0.0
            for value in stages[arm].values()
        )
    assert metrics["explicit_vs_direct"]["maximum_absolute_error"] == 0.0
    assert all(
        value["maximum_absolute_error"] == 0.0
        for value in stages["explicit_vs_direct"].values()
    )
    assert metrics["drop_block_identity_control"]["logit_nrmse"] > 0.0
    assert audit["native"]["source_block_calls"] == 2
    assert audit["passed"] is True
    for arm in experiment.ARM_ORDER:
        assert audit["arms"][arm]["source_block_calls"] == 0
        assert audit["arms"][arm]["execution_api_calls"] == 2
        assert audit["arms"][arm]["source_layer_calls"] == audit[
            "arms"
        ][arm]["expected_source_layer_calls"]
        assert all(
            count == 2
            for count in audit["arms"][arm][
                "required_leaf_module_calls"
            ].values()
        )
    assert audit["arms"]["explicit_residual"][
        "declared_graph_topology"
    ]["public_standalone_residual_add_nodes"] == 2
    assert audit["arms"]["direct_state_update"][
        "declared_graph_topology"
    ]["public_standalone_residual_add_nodes"] == 0
    assert audit["arms"]["explicit_residual"][
        "observed_state_update_module_calls"
    ] == {
        "module.attention_residual_add": 2,
        "module.feed_forward_residual_add": 2,
    }
    assert audit["arms"]["direct_state_update"][
        "observed_state_update_module_calls"
    ] == {
        "module.attention_state_generator": 2,
        "module.complete_state_generator": 2,
    }
    assert accounting["logical_total_macs"] > 0


def test_optional_real_gemma_middle_block_path_matches_native_with_left_padding(
) -> None:
    try:
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    except ImportError:
        pytest.skip("optional Transformers Gemma 3 dependency is not installed")

    torch.manual_seed(401_903)
    config = Gemma3TextConfig(
        vocab_size=32,
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
    model = Gemma3ForCausalLM(config).eval().requires_grad_(False)
    adapter = experiment.Gemma3CausalLMAdapter(model)
    forms = experiment._compile_forms(
        adapter,
        layer_index=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    batch = {
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

    metrics, stages, audit, accounting = experiment._evaluate_forms(
        adapter,
        (batch,),
        layer_index=1,
        forms=forms,
    )
    gates = experiment._exactness_gates(
        metrics,
        stages,
        stage_atol=experiment.DEFAULT_STAGE_ATOL,
        logit_atol=experiment.DEFAULT_LOGIT_ATOL,
        nll_atol=experiment.DEFAULT_NLL_ATOL,
        kl_max=experiment.DEFAULT_KL_MAX,
    )

    assert gates["passed"] is True
    assert audit["passed"] is True
    assert audit["native"]["source_block_calls"] == 1
    expected_source_calls = {
        "layer.0": 1,
        "layer.1": 0,
        "layer.2": 1,
    }
    for arm in experiment.ARM_ORDER:
        arm_audit = audit["arms"][arm]
        assert arm_audit["source_block_calls"] == 0
        assert arm_audit["source_layer_calls"] == expected_source_calls
        assert arm_audit["expected_source_layer_calls"] == expected_source_calls
        assert all(
            count == 1
            for count in arm_audit["required_leaf_module_calls"].values()
        )
    for arm in experiment.EXACT_ARMS:
        assert metrics[arm]["supervised_tokens"] == 8
        assert metrics[arm]["compared_query_rows"] == 10
        assert metrics[arm]["maximum_absolute_logit_error"] <= (
            experiment.DEFAULT_LOGIT_ATOL
        )
        assert all(
            value["maximum_absolute_error"] <= experiment.DEFAULT_STAGE_ATOL
            for value in stages[arm].values()
        )
    assert metrics["native"]["supervised_tokens"] == 8
    assert metrics["explicit_vs_direct"]["byte_identical"] is True
    assert all(
        value["byte_identical"] is True
        for value in stages["explicit_vs_direct"].values()
    )
    assert accounting["valid_tokens"] == 10


def test_stage_and_logit_metrics_are_valid_row_scoped() -> None:
    accumulator = experiment._TensorMetricAccumulator()
    reference = torch.zeros(1, 3, 2)
    candidate = reference.clone()
    candidate[:, 2] = 100.0
    accumulator.update(
        reference,
        candidate,
        torch.tensor([[True, True, False]]),
    )
    assert accumulator.result()["maximum_absolute_error"] == 0.0

    logits = experiment._LogitMetricAccumulator()
    native = torch.zeros(1, 3, 5)
    changed = native.clone()
    changed[:, 2] = 100.0
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[True, True, False]]),
    }
    logits.update(native, changed, batch)
    assert logits.result(native_nll_per_token=math_log(5.0))[
        "maximum_absolute_logit_error"
    ] == 0.0

    left_padded_logits = experiment._LogitMetricAccumulator()
    native = torch.zeros(1, 3, 5)
    changed = native.clone()
    changed[:, 0] = 100.0
    left_padded_batch = {
        "input_ids": torch.tensor([[0, 2, 3]]),
        "attention_mask": torch.tensor([[False, True, True]]),
    }
    left_padded_logits.update(native, changed, left_padded_batch)
    assert left_padded_logits.result(native_nll_per_token=math_log(5.0))[
        "maximum_absolute_logit_error"
    ] == 0.0

    final_query_logits = experiment._LogitMetricAccumulator()
    native = torch.zeros(1, 3, 5)
    changed = native.clone()
    changed[:, -1, 1] = 100.0
    all_valid_batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[True, True, True]]),
    }
    final_query_logits.update(native, changed, all_valid_batch)
    final_result = final_query_logits.result(
        native_nll_per_token=math_log(5.0)
    )
    assert final_result["compared_query_rows"] == 3
    assert final_result["maximum_absolute_logit_error"] == 100.0
    assert final_result["delta_nll_per_token"] == pytest.approx(0.0)


def math_log(value: float) -> float:
    return float(torch.tensor(value).log().item())


def test_resource_report_refuses_compression_and_speed_claims() -> None:
    model = _SyntheticModel().eval().requires_grad_(False)
    resources = experiment._resource_report(
        source_model_parameters=sum(value.numel() for value in model.parameters()),
        source_model_stored_scalars=sum(
            value.numel()
            for value in (*model.parameters(), *model.buffers())
        ),
        source_layer_parameters=sum(
            value.numel() for value in model.source.parameters()
        ),
        forms=_forms(model),
        logical_accounting={
            "valid_tokens": 4,
            "logical_causal_key_pairs": 10,
            "attention_projection_macs": 100,
            "attention_score_macs": 20,
            "attention_value_macs": 20,
            "feed_forward_macs": 200,
            "logical_total_macs": 340,
        },
    )

    assert resources["compression_attempted"] is False
    assert resources["compression_supported_by_this_experiment"] is False
    assert resources["logical_deployment_parameter_reduction_fraction"] == 0.0
    assert resources["logical_deployment_mac_reduction_fraction"] == 0.0
    assert resources["latency_or_speedup_claim"] is False
    assert resources["identity_arithmetic_removed"] is False
    assert resources[
        "public_residual_add_module_names_absent_in_direct_form"
    ] is True
    assert resources["direct_to_explicit_parameter_ratio"] == 1.0
    assert resources["physical_resident_experiment_scalars"] > resources[
        "source_whole_model_parameter_count"
    ]
    assert all(
        arm["contains_cloned_source_checkpoint_tensors"] is True
        for arm in resources["arms"].values()
    )


def test_runner_rejects_wrong_runtime_and_existing_json_before_model_load(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="pinned Gemma"):
        experiment.run_gemma3_one_block_residual_form_experiment(
            revision="wrong",
            output=tmp_path / "unused.json",
        )
    with pytest.raises(ValueError, match="CPU float32"):
        experiment.run_gemma3_one_block_residual_form_experiment(
            device_name="mps",
            output=tmp_path / "unused.json",
        )
    with pytest.raises(ValueError, match="tolerances are pinned"):
        experiment.run_gemma3_one_block_residual_form_experiment(
            stage_atol=1.0,
            output=tmp_path / "unused.json",
        )
    existing = tmp_path / "existing.json"
    existing.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run_gemma3_one_block_residual_form_experiment(output=existing)
    assert existing.read_text(encoding="utf-8") == "do not replace"


def test_mocked_runner_writes_hashed_json_without_weights_or_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SyntheticModel().eval().requires_grad_(False)
    adapter = _SyntheticAdapter(model)
    forms = _forms(model)
    split = {
        "panel_file": "synthetic.json",
        "panel_sha256": "a" * 64,
        "family_count": 1,
        "fit_example_count": 1,
        "evaluation_example_count": 2,
        "fit_example_ids_sha256": "b" * 64,
        "evaluation_example_ids_sha256": "c" * 64,
        "prompt_disjoint": True,
        "family_disjoint": False,
    }
    monkeypatch.setattr(
        experiment,
        "_load_prompt_split",
        lambda _path: (("fit secret",), ("held one", "held two"), split),
    )
    monkeypatch.setattr(
        experiment,
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(
        experiment,
        "load_gemma3",
        lambda **_kwargs: (_Tokenizer(), model),
    )
    monkeypatch.setattr(
        experiment,
        "Gemma3CausalLMAdapter",
        lambda loaded: adapter if loaded is model else None,
    )
    monkeypatch.setattr(
        experiment,
        "_compile_forms",
        lambda *_args, **_kwargs: forms,
    )
    destination = tmp_path / "report.json"

    report = experiment.run_gemma3_one_block_residual_form_experiment(
        output=destination,
        panel_path=tmp_path / "synthetic-panel.json",
        batch_size=2,
        layer_index=0,
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == report
    report_hash = loaded.pop("report_sha256")
    assert report_hash == experiment._report_sha256(loaded)
    serialized = destination.read_text(encoding="utf-8")
    assert "fit secret" not in serialized
    assert "held one" not in serialized
    assert report["contains_model_weights"] is False
    assert report["contains_executor_weights"] is False
    tokenization = report["protocol"]["tokenization"]
    assert tokenization["example_count"] == 2
    assert tokenization["batch_count"] == 1
    assert tokenization["padding_row_counts"] == {
        "none": 2,
        "left": 0,
        "right": 0,
    }
    assert tokenization["contains_prompt_text"] is False
    assert len(tokenization["combined_sha256"]) == 64
    assert report["claims"]["parameter_compression"] is False
    assert report["claims"][
        "exact_valid_token_complete_block_representation_tested"
    ] is True
    assert report["scientific_status"]["identity_information_removed"] is False
    assert report["scientific_status"]["runtime_graph_fusion_established"] is False
    assert report["exactness_gates"]["explicit_vs_direct"]["passed"] is True
    assert report["control_separation"][
        "incoming_block_identity_control_nonvacuous"
    ] is True
    assert report["source_restoration"]["passed"] is True
    assert (
        experiment.load_gemma3_one_block_residual_form_report(destination)
        == report
    )

    tampered = json.loads(destination.read_text(encoding="utf-8"))
    tampered["resources"]["compression_attempted"] = True
    tampered.pop("report_sha256")
    tampered["report_sha256"] = experiment._report_sha256(tampered)
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="resource claims"):
        experiment.load_gemma3_one_block_residual_form_report(tampered_path)

    metric_tampered = json.loads(destination.read_text(encoding="utf-8"))
    metric_tampered["metrics"]["explicit_residual"][
        "maximum_absolute_logit_error"
    ] = 123456.0
    metric_tampered.pop("report_sha256")
    metric_tampered["report_sha256"] = experiment._report_sha256(
        metric_tampered
    )
    metric_tampered_path = tmp_path / "metric-tampered.json"
    metric_tampered_path.write_text(
        json.dumps(metric_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactness was not recomputed"):
        experiment.load_gemma3_one_block_residual_form_report(
            metric_tampered_path
        )

    token_tampered = json.loads(destination.read_text(encoding="utf-8"))
    token_tampered["protocol"]["tokenization"]["combined_sha256"] = "0" * 64
    token_tampered.pop("report_sha256")
    token_tampered["report_sha256"] = experiment._report_sha256(
        token_tampered
    )
    token_tampered_path = tmp_path / "token-tampered.json"
    token_tampered_path.write_text(
        json.dumps(token_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenization digest"):
        experiment.load_gemma3_one_block_residual_form_report(
            token_tampered_path
        )

    call_tampered = json.loads(destination.read_text(encoding="utf-8"))
    call_tampered["execution_audit"]["arms"]["explicit_residual"][
        "execution_api_calls"
    ] = 0
    call_tampered.pop("report_sha256")
    call_tampered["report_sha256"] = experiment._report_sha256(call_tampered)
    call_tampered_path = tmp_path / "call-tampered.json"
    call_tampered_path.write_text(
        json.dumps(call_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source-call audit"):
        experiment.load_gemma3_one_block_residual_form_report(
            call_tampered_path
        )

    map_tampered = json.loads(destination.read_text(encoding="utf-8"))
    direct_audit = map_tampered["execution_audit"]["arms"][
        "direct_state_update"
    ]
    direct_audit["source_layer_calls"] = {}
    direct_audit["expected_source_layer_calls"] = {}
    direct_audit["selected_source_block_skipped"] = False
    map_tampered.pop("report_sha256")
    map_tampered["report_sha256"] = experiment._report_sha256(map_tampered)
    map_tampered_path = tmp_path / "map-tampered.json"
    map_tampered_path.write_text(
        json.dumps(map_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source-call audit"):
        experiment.load_gemma3_one_block_residual_form_report(
            map_tampered_path
        )

    resource_tampered = json.loads(destination.read_text(encoding="utf-8"))
    resource_tampered["resources"]["arms"]["direct_state_update"][
        "runtime_stored_coefficient_count"
    ] += 1
    resource_tampered.pop("report_sha256")
    resource_tampered["report_sha256"] = experiment._report_sha256(
        resource_tampered
    )
    resource_tampered_path = tmp_path / "resource-count-tampered.json"
    resource_tampered_path.write_text(
        json.dumps(resource_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="per-arm resources"):
        experiment.load_gemma3_one_block_residual_form_report(
            resource_tampered_path
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        experiment.run_gemma3_one_block_residual_form_experiment(
            output=destination,
            panel_path=tmp_path / "synthetic-panel.json",
            layer_index=0,
        )
