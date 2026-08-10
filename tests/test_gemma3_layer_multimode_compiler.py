from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch
from torch import nn

import fisher_graph.gemma3_layer_multimode_capture as capture_module
import fisher_graph.gemma3_layer_multimode_compiler as compiler_module
from fisher_graph.gemma3_layer_multimode_compiler import (
    CompiledGemma3LayerMultimodeModel,
    Float32LayerMultimodeMLP,
    FrozenLayerMultimodeCoefficients,
    Gemma3LayerMultimodeCompilationReceipt,
    LayerMultimodeCompilationError,
    SelectedLayerMultimodeCoefficientInventory,
    build_selected_layer_multimode_coefficient_inventory,
    compile_gemma3_layer_multimode_model,
    load_compiled_layer_multimode_bundle,
    save_compiled_layer_multimode_bundle,
)
from fisher_graph.gemma3_layer_multimode_flow import (
    AffineLocalLayerMultimodeFlow,
    FixedLayerChartAtlas,
    LayerMultimodeFlowConfig,
)


HIDDEN = 4
INTERMEDIATE = 8
VOCAB = 11
SELECTION_SHA256 = "b" * 64
SOURCE_MODEL_SHA256 = "a" * 64
CAPTURE_SHA256 = "c" * 64


class ToyGemmaConfig:
    model_type = "gemma3_text"

    def __init__(self, hidden_width: int, layer_count: int) -> None:
        self.hidden_size = hidden_width
        self.intermediate_size = INTERMEDIATE
        self.vocab_size = VOCAB
        self.num_hidden_layers = layer_count
        self.num_attention_heads = 1
        self.num_key_value_heads = 1
        self.head_dim = hidden_width
        self.query_pre_attn_scalar = hidden_width
        self.max_position_embeddings = 32
        self.sliding_window = 4
        self.layer_types = ["sliding_attention"] * layer_count
        self.rms_norm_eps = 1.0e-6
        self.attention_dropout = 0.0
        self.attention_bias = False
        self.hidden_activation = "gelu_pytorch_tanh"
        self.attn_logit_softcapping = None
        self.use_bidirectional_attention = False

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in vars(self).items()
        }


class ToyRMSNorm(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rms = value.square().mean(dim=-1, keepdim=True).add(1.0e-6).rsqrt()
        return value * rms * self.weight


class ToyNativeMLP(nn.Module):
    call_count = 0

    def __init__(self, hidden_width: int, intermediate_width: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_width, intermediate_width, bias=False)
        self.up_proj = nn.Linear(hidden_width, intermediate_width, bias=False)
        self.down_proj = nn.Linear(intermediate_width, hidden_width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        type(self).call_count += 1
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(value)) * self.up_proj(value))


class ToyAttention(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_width, hidden_width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class BehaviorallyDifferentToyAttention(ToyAttention):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return -self.proj(value)


class ToyLayer(nn.Module):
    def __init__(self, hidden_width: int, intermediate_width: int) -> None:
        super().__init__()
        self.self_attn = ToyAttention(hidden_width)
        self.pre_feedforward_layernorm = ToyRMSNorm(hidden_width)
        self.mlp = ToyNativeMLP(hidden_width, intermediate_width)
        self.post_feedforward_layernorm = ToyRMSNorm(hidden_width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(hidden)
        residual = hidden
        x = self.pre_feedforward_layernorm(residual)
        return residual + self.post_feedforward_layernorm(self.mlp(x))


class ToyBackbone(nn.Module):
    def __init__(self, config: ToyGemmaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, config.hidden_size)
        self.layers = nn.ModuleList(
            ToyLayer(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = ToyRMSNorm(config.hidden_size)
        self.rotary_emb = nn.Identity()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class ToyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        layer_count: int = 18,
        hidden_width: int = HIDDEN,
        tied: bool = True,
    ) -> None:
        super().__init__()
        self.config = ToyGemmaConfig(hidden_width, layer_count)
        self.model = ToyBackbone(self.config)
        self.lm_head = nn.Linear(hidden_width, VOCAB, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids))


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _a_partition_receipt() -> capture_module.ADevelopmentPartitionReceipt:
    axis = tuple(
        (ordinal, f"mode.mutation.{2 + 2 * ordinal}")
        for ordinal in range(18)
    )
    return capture_module.ADevelopmentPartitionReceipt(
        source_full_A_bridge_receipt_sha256=_sha("full-A-bridge"),
        source_role_capture_receipt_sha256=_sha("A-role-capture"),
        source_runtime_session_binding_sha256=_sha("A-runtime-session"),
        source_role_panel_binding_sha256=_sha("A-role-panel"),
        source_batch_axis_catalog_sha256=_sha("A-batch-axis"),
        source_shared_spool_row_catalog_sha256=_sha("A-spool-rows"),
        source_ordered_objective_row_catalog_sha256=_sha("A-objective-rows"),
        source_score_objective_receipt_sha256=_sha("A-score-objective"),
        canonical_layer_mode_axis=axis,
        canonical_layer_mode_axis_sha256=(
            capture_module._A_partition_layer_axis_sha256(axis)
        ),
        fit_target_token_counts=capture_module.A_FIT_TARGET_TOKEN_COUNTS,
        validation_target_token_counts=(
            capture_module.A_VALIDATION_TARGET_TOKEN_COUNTS
        ),
        mask_hash_abi=capture_module._A_MASK_HASH_ABI,
        fit_mask_sha256=_sha("A-fit-mask"),
        validation_mask_sha256=_sha("A-validation-mask"),
        fit_example_catalog_sha256=_sha("A-fit-examples"),
        validation_example_catalog_sha256=_sha("A-validation-examples"),
        fit_row_count=10_548,
        validation_row_count=14_184,
        row_count=24_732,
        layer_count=18,
    )


def _basis(chart_count: int, hidden_width: int, rank: int) -> torch.Tensor:
    rows = []
    for chart in range(chart_count):
        generator = torch.Generator().manual_seed(100 + chart)
        matrix = torch.randn(hidden_width, rank, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(matrix, mode="reduced")
        rows.append(q.to(torch.float32))
    return torch.stack(rows)


def _coefficient(
    ordinal: int,
    *,
    hidden_width: int = HIDDEN,
    chart_count: int = 1,
    modes: int = 1,
    rank: int = 2,
) -> FrozenLayerMultimodeCoefficients:
    generator = torch.Generator().manual_seed(1000 + ordinal)
    log_temperature = 0.0 if chart_count == 1 else 0.2
    return FrozenLayerMultimodeCoefficients(
        layer_ordinal=ordinal,
        centers=torch.randn(
            chart_count, hidden_width, generator=generator, dtype=torch.float32
        ),
        bases=_basis(chart_count, hidden_width, rank),
        router_weight=torch.randn(
            chart_count, modes, rank, generator=generator, dtype=torch.float32
        ),
        router_bias=torch.randn(
            chart_count, modes, generator=generator, dtype=torch.float32
        ),
        generator_weight=torch.randn(
            chart_count,
            modes,
            rank,
            rank,
            generator=generator,
            dtype=torch.float32,
        ),
        generator_bias=torch.randn(
            chart_count, modes, rank, generator=generator, dtype=torch.float32
        ),
        log_chart_temperature=torch.tensor(log_temperature, dtype=torch.float32),
        source_flow_model_parameter_sha256=_sha(f"model-{ordinal}"),
        source_fit_receipt_sha256=_sha(f"fit-{ordinal}"),
        selection_receipt_sha256=SELECTION_SHA256,
    )


def _coefficients(**kwargs: int) -> tuple[FrozenLayerMultimodeCoefficients, ...]:
    return tuple(_coefficient(ordinal, **kwargs) for ordinal in range(18))


def _compile(source: ToyCausalLM | None = None) -> CompiledGemma3LayerMultimodeModel:
    return compiler_module._compile_gemma3_layer_multimode_model_for_test(
        ToyCausalLM().eval() if source is None else source,
        _coefficients(),
        source_model_sha256=SOURCE_MODEL_SHA256,
        selection_receipt_sha256=SELECTION_SHA256,
    )


def _eval_toy_model() -> ToyCausalLM:
    return ToyCausalLM().eval()


def _state_clones(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_float64_flow_and_float32_deploy_have_close_direct_delta() -> None:
    generator = torch.Generator().manual_seed(7)
    centers = torch.randn(2, 6, generator=generator, dtype=torch.float64)
    basis_rows = []
    for _ in range(2):
        q, _ = torch.linalg.qr(
            torch.randn(6, 3, generator=generator, dtype=torch.float64),
            mode="reduced",
        )
        basis_rows.append(q)
    atlas = FixedLayerChartAtlas(
        centers=centers,
        bases=torch.stack(basis_rows),
        source_fit_sha256="d" * 64,
    )
    flow = AffineLocalLayerMultimodeFlow(
        atlas=atlas,
        config=LayerMultimodeFlowConfig(
            layer_ordinal=3,
            modes_per_chart=2,
            chart_temperature=1.3,
            mode_temperature=1.0,
            source_mlp_intermediate_width=8,
            fit_chart_temperature=True,
            initialization_seed=19,
        ),
    )
    coefficients = FrozenLayerMultimodeCoefficients.from_float64_flow(
        flow,
        source_fit_receipt_sha256="e" * 64,
        selection_receipt_sha256=SELECTION_SHA256,
    )
    deploy = Float32LayerMultimodeMLP(coefficients)
    x = torch.randn(6, 6, generator=generator, dtype=torch.float64) * 0.2
    h = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    core = flow(
        chart_inputs=x,
        residual_states=h,
        source_capture_sha256=CAPTURE_SHA256,
    )
    actual = deploy(x.to(torch.float32).reshape(2, 3, 6)).reshape(6, 6)
    torch.testing.assert_close(
        actual,
        core.predicted_velocity.detach().to(torch.float32),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert not hasattr(deploy, "gate_proj")
    assert not hasattr(deploy, "up_proj")
    assert not hasattr(deploy, "down_proj")
    assert deploy.native_mlp_fallback_present is False
    assert deploy.returns_post_output_rmsnorm_teacher_delta is True


def test_physical_compiler_replaces_exactly_18_and_preserves_exact_paths() -> None:
    torch.manual_seed(4)
    source = ToyCausalLM()
    source.eval()
    before = _state_clones(source)
    source_storage = {
        parameter.untyped_storage().data_ptr() for parameter in source.parameters()
    }
    source_mlp_parameter_ids = {
        id(parameter)
        for layer in source.model.layers
        for parameter in layer.mlp.parameters()
    }
    compiled = _compile(source)

    assert len(compiled.model.model.layers) == 18
    assert all(
        type(layer.mlp) is Float32LayerMultimodeMLP
        for layer in compiled.model.model.layers
    )
    assert all(
        type(layer.post_feedforward_layernorm) is nn.Identity
        for layer in compiled.model.model.layers
    )
    assert all(
        type(layer.pre_feedforward_layernorm) is ToyRMSNorm
        for layer in compiled.model.model.layers
    )
    assert (
        compiled.model.get_input_embeddings().weight
        is compiled.model.get_output_embeddings().weight
    )
    assert compiled.model.get_input_embeddings().weight is not source.get_input_embeddings().weight
    assert source.get_input_embeddings().weight is source.get_output_embeddings().weight
    assert all(
        id(parameter) not in source_mlp_parameter_ids
        for parameter in compiled.model.parameters()
    )
    assert not source_storage.intersection(
        parameter.untyped_storage().data_ptr() for parameter in compiled.model.parameters()
    )

    for name, value in before.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0.0, atol=0.0)
    compiled_names = tuple(compiled.model.state_dict())
    assert not any(".mlp.gate_proj." in name for name in compiled_names)
    assert not any(".mlp.up_proj." in name for name in compiled_names)
    assert not any(".mlp.down_proj." in name for name in compiled_names)
    assert not any(".post_feedforward_layernorm." in name for name in compiled_names)
    for ordinal in range(18):
        torch.testing.assert_close(
            compiled.model.model.layers[ordinal].self_attn.proj.weight,
            source.model.layers[ordinal].self_attn.proj.weight,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            compiled.model.model.layers[ordinal].pre_feedforward_layernorm.weight,
            source.model.layers[ordinal].pre_feedforward_layernorm.weight,
            rtol=0.0,
            atol=0.0,
        )
    compiled.assert_integrity()
    assert (
        compiled.receipt.protected_architecture_sha256
        == compiler_module._protected_architecture_sha256(compiled.model)
    )


def test_exact_parameter_and_storage_accounting_closes() -> None:
    compiled = _compile()
    receipt = compiled.receipt
    expected_owned = 1 * HIDDEN + 1 * HIDDEN * 2 + 1 + 1 * 1 * (2 + 1) + 1 * 1 * (2 * 2 + 2)
    for row in receipt.layer_accounting:
        assert row.removed_native_mlp_parameters == 3 * HIDDEN * INTERMEDIATE
        assert row.removed_output_rmsnorm_parameters == HIDDEN
        assert row.retained_input_rmsnorm_parameters == HIDDEN
        assert row.replacement_owned_parameters == expected_owned
        assert row.native_boundary_parameters == 2 * HIDDEN + 3 * HIDDEN * INTERMEDIATE
        assert row.compiled_boundary_parameters == HIDDEN + expected_owned
        assert row.replacement_float32_bytes == 4 * expected_owned
    assert receipt.compiled_parameter_count == receipt.source_parameter_count + sum(
        row.compiled_boundary_parameters - row.native_boundary_parameters
        for row in receipt.layer_accounting
    )
    assert receipt.compiled_parameter_storage_bytes == 4 * receipt.compiled_parameter_count
    assert receipt.tied_embedding_parameter_identity is True
    assert receipt.production_provenance_authorized is False
    assert receipt.source_execution_fingerprint_sha256 is None
    assert receipt.recomputed_source_execution_fingerprint_sha256 is None
    assert receipt.compiled_execution_fingerprint_sha256 == (
        compiler_module.Gemma3CausalLMAdapter(
            compiled.model
        ).execution_fingerprint()
    )
    assert all(not module.training for module in compiled.model.modules())
    assert receipt.native_mlp_fallback_present is False
    assert receipt.source_parameter_alias_present is False

    roundtrip = json.loads(json.dumps(receipt.state_dict()))
    restored = Gemma3LayerMultimodeCompilationReceipt.from_state_dict(roundtrip)
    assert restored == receipt
    attacked = json.loads(json.dumps(receipt.state_dict()))
    attacked["replacement_count"] = True
    with pytest.raises(LayerMultimodeCompilationError):
        Gemma3LayerMultimodeCompilationReceipt.from_state_dict(attacked)
    attacked = json.loads(json.dumps(receipt.state_dict()))
    attacked["layer_accounting"] = tuple(attacked["layer_accounting"])
    with pytest.raises(LayerMultimodeCompilationError):
        Gemma3LayerMultimodeCompilationReceipt.from_state_dict(attacked)
    attacked = json.loads(json.dumps(receipt.state_dict()))
    attacked["compiled_execution_fingerprint_sha256"] = "0" * 64
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="compilation receipt hash mismatch",
    ):
        Gemma3LayerMultimodeCompilationReceipt.from_state_dict(attacked)
    compiled.model.model.layers[0].self_attn.scaling = 0.75
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="compiled model execution fingerprint drifted",
    ):
        compiled.assert_integrity()


def test_compiler_fails_closed_without_mutating_source() -> None:
    source = ToyCausalLM().eval()
    before = _state_clones(source)
    with pytest.raises(TypeError, match="typed selected coefficient inventory"):
        compile_gemma3_layer_multimode_model(
            source,
            _coefficients(),
        )
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="entirely in eval mode",
    ):
        compiler_module._compile_gemma3_layer_multimode_model_for_test(
            ToyCausalLM(),
            _coefficients(),
            source_model_sha256=SOURCE_MODEL_SHA256,
            selection_receipt_sha256=SELECTION_SHA256,
        )
    with pytest.raises(LayerMultimodeCompilationError, match="exactly 18"):
        compiler_module._compile_gemma3_layer_multimode_model_for_test(
            ToyCausalLM(layer_count=17).eval(),
            _coefficients(),
            source_model_sha256=SOURCE_MODEL_SHA256,
            selection_receipt_sha256=SELECTION_SHA256,
        )
    with pytest.raises(LayerMultimodeCompilationError, match="already be tied"):
        compiler_module._compile_gemma3_layer_multimode_model_for_test(
            ToyCausalLM(tied=False).eval(),
            _coefficients(),
            source_model_sha256=SOURCE_MODEL_SHA256,
            selection_receipt_sha256=SELECTION_SHA256,
        )
    wrong_selection = list(_coefficients())
    wrong_selection[0] = _coefficient(0)
    object.__setattr__(wrong_selection[0], "selection_receipt_sha256", "f" * 64)
    with pytest.raises(LayerMultimodeCompilationError, match="bind"):
        compiler_module._compile_gemma3_layer_multimode_model_for_test(
            source,
            wrong_selection,
            source_model_sha256=SOURCE_MODEL_SHA256,
            selection_receipt_sha256=SELECTION_SHA256,
        )
    tampered = list(_coefficients())
    tampered[0].centers.add_(1.0)
    with pytest.raises(LayerMultimodeCompilationError, match="changed"):
        compiler_module._compile_gemma3_layer_multimode_model_for_test(
            source,
            tampered,
            source_model_sha256=SOURCE_MODEL_SHA256,
            selection_receipt_sha256=SELECTION_SHA256,
        )
    for name, value in before.items():
        torch.testing.assert_close(source.state_dict()[name], value, rtol=0.0, atol=0.0)
    assert all(type(layer.mlp) is ToyNativeMLP for layer in source.model.layers)
    assert all(
        type(layer.post_feedforward_layernorm) is ToyRMSNorm
        for layer in source.model.layers
    )


def test_exact_source_fingerprints_reject_weight_config_execution_and_mode_drift() -> None:
    source = ToyCausalLM().eval()
    before = _state_clones(source)
    adapter = compiler_module.Gemma3CausalLMAdapter(source)
    expected = adapter.model_fingerprint()
    expected_execution = adapter.execution_fingerprint()
    assert (
        compiler_module._require_source_model_fingerprint(
            source,
            expected_sha256=expected,
        )
        == expected
    )
    assert compiler_module._require_source_fingerprints(
        source,
        expected_model_sha256=expected,
        expected_execution_sha256=expected_execution,
    ) == (expected, expected_execution)

    wrong_weight = copy.deepcopy(source)
    with torch.no_grad():
        wrong_weight.model.layers[0].self_attn.proj.weight[0, 0].add_(1.0)
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="actual source-model fingerprint differs",
    ):
        compiler_module._require_source_fingerprints(
            wrong_weight,
            expected_model_sha256=expected,
            expected_execution_sha256=expected_execution,
        )

    wrong_config = copy.deepcopy(source)
    wrong_config.config.rms_norm_eps = 2.0e-6
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="actual source-model fingerprint differs",
    ):
        compiler_module._require_source_fingerprints(
            wrong_config,
            expected_model_sha256=expected,
            expected_execution_sha256=expected_execution,
        )

    wrong_execution = copy.deepcopy(source)
    wrong_execution.model.layers[0].self_attn.scaling = 0.75
    wrong_execution_adapter = compiler_module.Gemma3CausalLMAdapter(
        wrong_execution
    )
    assert wrong_execution_adapter.model_fingerprint() == expected
    assert wrong_execution_adapter.execution_fingerprint() != expected_execution
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="source execution fingerprint differs",
    ):
        compiler_module._require_source_fingerprints(
            wrong_execution,
            expected_model_sha256=expected,
            expected_execution_sha256=expected_execution,
        )

    train_mode = copy.deepcopy(source).train()
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="entirely in eval mode",
    ):
        compiler_module._require_source_fingerprints(
            train_mode,
            expected_model_sha256=expected,
            expected_execution_sha256=expected_execution,
        )
    for name, value in before.items():
        torch.testing.assert_close(
            source.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )


def test_typed_inventory_rejects_free_sha_and_r2_spoof() -> None:
    from test_gemma3_layer_multimode_selection import _grid
    from fisher_graph.gemma3_layer_multimode_selection import (
        select_layer_multimode_candidates,
    )

    partition = _a_partition_receipt()
    selection = select_layer_multimode_candidates(
        tuple(
            replace(
                candidate,
                fit_a_sha256=partition.fit_partition_sha256,
                validation_a_sha256=partition.validation_partition_sha256,
                artifact_sha256="",
            )
            for candidate in _grid()
        ),
        allow_test_only_nonproduction=True,
        allow_test_only_nonlive_authority=True,
    )
    coefficients = tuple(
        replace(
            value,
            selection_receipt_sha256=selection.artifact_sha256,
            coefficient_sha256="",
        )
        for value in _coefficients()
    )
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="requires a live typed layer-materialization inventory",
    ):
        build_selected_layer_multimode_coefficient_inventory(
            selection,
            (),
            a_development_partition_receipt=partition,
            layer_materialization_inventory=None,
            split_refit_evidence=(),
        )
    with pytest.raises(LayerMultimodeCompilationError, match="exact bool"):
        build_selected_layer_multimode_coefficient_inventory(
            selection,
            (),
            a_development_partition_receipt=partition,
            layer_materialization_inventory=None,
            split_refit_evidence=(),
            allow_test_unbound_materialization=1,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exact typed materialization"):
        build_selected_layer_multimode_coefficient_inventory(
            selection,
            (),
            a_development_partition_receipt=partition,
            layer_materialization_inventory=object(),  # type: ignore[arg-type]
            split_refit_evidence=(),
        )
    with pytest.raises(TypeError, match="typed builder"):
        SelectedLayerMultimodeCoefficientInventory(
            a_development_partition_receipt=partition,
            selection_receipt=selection,
            coefficients=coefficients,
            selected_candidate_artifact_sha256s=tuple(
                layer.selected_candidate_sha256 for layer in selection.layers
            ),
            selected_fit_receipt_sha256s=tuple(
                value.source_fit_receipt_sha256 for value in coefficients
            ),
            selected_source_rows_receipt_sha256s=tuple(
                _sha(f"source-rows-{ordinal}") for ordinal in range(18)
            ),
            selected_fit_source_sha256s=tuple(
                _sha(f"fit-source-{ordinal}") for ordinal in range(18)
            ),
            selected_validation_source_sha256s=tuple(
                _sha(f"validation-source-{ordinal}")
                for ordinal in range(18)
            ),
            selected_atlas_sha256s=tuple(
                _sha(f"atlas-{ordinal}") for ordinal in range(18)
            ),
            selected_model_parameter_sha256s=tuple(
                value.source_flow_model_parameter_sha256
                for value in coefficients
            ),
            split_refit_evidence_sha256s=(None,) * 18,
        )


def test_typed_inventory_replays_real_selected_fits_and_shared_A_receipt() -> None:
    from test_gemma3_layer_multimode_fit import _policy, _rows
    from test_gemma3_layer_multimode_selection import _digest, _grid
    from fisher_graph.gemma3_layer_multimode_fit import (
        fit_layer_multimode_candidate,
        layer_multimode_candidate_evaluation_from_fit,
    )
    from fisher_graph.gemma3_layer_multimode_selection import (
        select_layer_multimode_candidates,
    )

    partition = _a_partition_receipt()
    base = _rows(family_count=6)
    generator = torch.Generator().manual_seed(123)
    states = torch.zeros_like(base.chart_states)
    states[:, :16] = torch.randn(
        len(states),
        16,
        generator=generator,
    )
    velocities = torch.zeros_like(states)
    velocities[:, :16] = 0.4 * states[:, :16]
    family_ordinals = {
        family: ordinal
        for ordinal, family in enumerate(sorted(set(base.family_ids)))
    }
    family_ids = tuple(
        f"family-{family_ordinals[family]}" for family in base.family_ids
    )
    row_keys = tuple(
        (
            f"family-{family_ordinals[family]}",
            example,
            query,
            target,
        )
        for family, example, query, target in base.row_keys
    )
    fits = []
    selected_evaluations = []
    for ordinal in range(18):
        rows = replace(
            base,
            layer_ordinal=ordinal,
            mode_id=f"mode.mutation.{2 + 2 * ordinal}",
            family_ids=family_ids,
            row_keys=row_keys,
            chart_states=states,
            teacher_velocities=velocities,
            chart_state_sha256=capture_module._tensor_sha256(states),
            teacher_velocity_sha256=(
                capture_module._tensor_sha256(velocities)
            ),
            receipt_sha256="",
        )
        fitted = fit_layer_multimode_candidate(
            rows=rows,
            chart_count=1,
            modes_per_chart=1,
            tangent_rank=16,
            policy=_policy(steps=1),
            allow_test_policy=True,
        )
        selected = layer_multimode_candidate_evaluation_from_fit(
            fitted,
            source_model_sha256=_digest("model"),
            fit_a_partition_sha256=partition.fit_partition_sha256,
            validation_a_partition_sha256=(
                partition.validation_partition_sha256
            ),
            evaluation_protocol_sha256=_digest("evaluation-protocol"),
        )
        assert selected.validation_velocity_nrmse < 0.01
        fits.append(fitted)
        selected_evaluations.append(selected)

    assert all(fit.live_fitter_authorized is True for fit in fits)
    compiler_module._require_live_selected_fit_authority(
        tuple(fits),
        (None,) * 18,
    )
    copied_fits = (replace(fits[0]), *fits[1:])
    assert copied_fits[0].live_fitter_authorized is False
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="original live fitter result",
    ):
        compiler_module._require_live_selected_fit_authority(
            copied_fits,
            (None,) * 18,
        )
    foreign_rows = _sha("foreign-fit-rows")
    relabelled_atlas_receipt = replace(
        fits[0].atlas_fit.receipt,
        source_rows_receipt_sha256=foreign_rows,
        receipt_sha256="",
    )
    relabelled_atlas_fit = replace(
        fits[0].atlas_fit,
        receipt=relabelled_atlas_receipt,
    )
    relabelled_fit_receipt = replace(
        fits[0].receipt,
        source_rows_receipt_sha256=foreign_rows,
        atlas_fit=relabelled_atlas_receipt,
        receipt_sha256="",
    )
    relabelled_fit = replace(
        fits[0],
        atlas_fit=relabelled_atlas_fit,
        receipt=relabelled_fit_receipt,
    )
    relabelled_fit.assert_integrity()
    assert relabelled_fit.live_fitter_authorized is False
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="original live fitter result",
    ):
        compiler_module._require_live_selected_fit_authority(
            (relabelled_fit, *fits[1:]),
            (None,) * 18,
        )

    candidates = []
    for candidate in _grid():
        selected = selected_evaluations[candidate.layer_ordinal]
        if (
            candidate.chart_count,
            candidate.dynamic_mode_count,
            candidate.tangent_rank,
        ) == (1, 1, 16):
            candidates.append(selected)
            continue
        families = tuple(
            replace(
                old,
                family_id=target.family_id,
                scalar_count=target.scalar_count,
                prediction_sse=0.04 * target.teacher_energy,
                teacher_energy=target.teacher_energy,
                velocity_nrmse=0.2,
                prediction_energy=target.teacher_energy,
                prediction_teacher_dot=0.97 * target.teacher_energy,
                velocity_cosine=0.97,
                artifact_sha256="",
            )
            for old, target in zip(
                candidate.family_fidelity,
                selected.family_fidelity,
                strict=True,
            )
        )
        scalar_count = sum(value.scalar_count for value in families)
        teacher_energy = sum(value.teacher_energy for value in families)
        prediction_sse = sum(value.prediction_sse for value in families)
        prediction_energy = sum(value.prediction_energy for value in families)
        prediction_teacher_dot = sum(
            value.prediction_teacher_dot for value in families
        )
        teacher_rms = math.sqrt(teacher_energy / scalar_count)
        candidates.append(
            replace(
                candidate,
                source_rows_receipt_sha256=(
                    selected.source_rows_receipt_sha256
                ),
                fit_policy_sha256=selected.fit_policy_sha256,
                fit_policy_test_only=selected.fit_policy_test_only,
                source_capture_scope=selected.source_capture_scope,
                production_fit_authorized=(
                    selected.production_fit_authorized
                ),
                fit_source_sha256=selected.fit_source_sha256,
                validation_source_sha256=selected.validation_source_sha256,
                atlas_sha256=(
                    selected.atlas_sha256
                    if candidate.chart_count == 1
                    and candidate.tangent_rank == 16
                    else candidate.atlas_sha256
                ),
                fit_a_sha256=partition.fit_partition_sha256,
                validation_a_sha256=partition.validation_partition_sha256,
                validation_scalar_count=scalar_count,
                validation_prediction_sse=float(prediction_sse),
                validation_teacher_energy=float(teacher_energy),
                validation_velocity_nrmse=float(
                    math.sqrt(prediction_sse / teacher_energy)
                ),
                validation_prediction_energy=float(prediction_energy),
                validation_prediction_teacher_dot=float(
                    prediction_teacher_dot
                ),
                aggregate_velocity_cosine=float(
                    prediction_teacher_dot
                    / math.sqrt(prediction_energy * teacher_energy)
                ),
                family_fidelity=families,
                per_chart_minimum_pairwise_velocity_separation_rms=tuple(
                    float(ratio * teacher_rms)
                    for ratio in (
                        candidate.per_chart_minimum_pairwise_velocity_separation_ratio
                    )
                ),
                artifact_sha256="",
            )
        )
    with pytest.raises(ValueError, match="324 original live adapter"):
        select_layer_multimode_candidates(
            tuple(candidates),
            allow_test_only_nonproduction=True,
        )
    selection = select_layer_multimode_candidates(
        tuple(candidates),
        allow_test_only_nonproduction=True,
        allow_test_only_nonlive_authority=True,
    )
    assert all(
        layer.selected_candidate_sha256
        == selected_evaluations[ordinal].artifact_sha256
        for ordinal, layer in enumerate(selection.layers)
    )
    assert selection.live_selector_authorized is True
    assert selection.live_full_grid_selection_authorized is False
    parsed_selection = type(selection).from_state_dict(
        json.loads(json.dumps(selection.state_dict()))
    )
    forged_full_grid = replace(selection, artifact_sha256="")
    for nonlive_selection in (
        selection,
        copy.copy(selection),
        copy.deepcopy(selection),
        parsed_selection,
        forged_full_grid,
    ):
        assert nonlive_selection.live_full_grid_selection_authorized is False
        with pytest.raises(
            LayerMultimodeCompilationError,
            match="original live-authorized full-grid selection",
        ):
            compiler_module._require_live_full_grid_selection_authority(
                nonlive_selection
            )

    inventory = build_selected_layer_multimode_coefficient_inventory(
        selection,
        tuple(fits),
        a_development_partition_receipt=partition,
        layer_materialization_inventory=None,
        split_refit_evidence=(None,) * 18,
        allow_test_unbound_materialization=True,
    )
    inventory.assert_integrity()
    assert (
        inventory.metadata()["a_development_partition_receipt_sha256"]
        == partition.receipt_sha256
    )
    assert inventory.metadata()[
        "capture_time_layer_materialization_receipt_required"
    ] is True
    assert inventory.metadata()[
        "capture_time_layer_materialization_receipt_bound"
    ] is False
    assert inventory.metadata()["partition_to_flow_rows_proof_claimed"] is False
    assert inventory.metadata()["production_provenance_authorized"] is False
    for clone in (copy.copy(inventory), copy.deepcopy(inventory)):
        with pytest.raises(
            LayerMultimodeCompilationError,
            match="exact live typed-builder result",
        ):
            clone.assert_integrity()
    with pytest.raises(TypeError, match="init=False"):
        replace(
            inventory,
            production_provenance_authorized=True,
        )
    with pytest.raises(TypeError, match="typed builder"):
        replace(inventory, inventory_sha256="")
    assert len(inventory.coefficients) == 18
    for ordinal, (coefficient, fitted) in enumerate(
        zip(inventory.coefficients, fits, strict=True)
    ):
        assert coefficient.layer_ordinal == ordinal
        assert coefficient.tangent_rank == 16
        assert coefficient.source_fit_receipt_sha256 == fitted.receipt.receipt_sha256
        assert (
            inventory.selected_source_rows_receipt_sha256s[ordinal]
            == fitted.receipt.source_rows_receipt_sha256
        )
        assert (
            coefficient.source_flow_model_parameter_sha256
            == fitted.model.model_parameter_sha256
        )
        assert coefficient.centers.data_ptr() != fitted.model.centers.data_ptr()

    unbound_source = ToyCausalLM().eval()
    unbound_source_state = _state_clones(unbound_source)
    with pytest.raises(LayerMultimodeCompilationError, match="unauthorized"):
        compile_gemma3_layer_multimode_model(unbound_source, inventory)
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="requires H=640 and I=2048",
    ):
        compile_gemma3_layer_multimode_model(
            unbound_source,
            inventory,
            allow_test_unbound_materialization=True,
        )
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="exact bool",
    ):
        compile_gemma3_layer_multimode_model(
            unbound_source,
            inventory,
            allow_test_unbound_materialization=1,  # type: ignore[arg-type]
        )
    for name, value in unbound_source_state.items():
        torch.testing.assert_close(
            unbound_source.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    wrong_partition = replace(
        partition,
        source_full_A_bridge_receipt_sha256=_sha("other-full-A-bridge"),
        fit_partition_sha256="",
        validation_partition_sha256="",
        receipt_sha256="",
    )
    with pytest.raises(LayerMultimodeCompilationError, match="partitions differ"):
        build_selected_layer_multimode_coefficient_inventory(
            selection,
            tuple(fits),
            a_development_partition_receipt=wrong_partition,
            layer_materialization_inventory=None,
            split_refit_evidence=(None,) * 18,
            allow_test_unbound_materialization=True,
        )


def test_production_freezing_rejects_copied_split_evaluator_evidence() -> None:
    from test_gemma3_layer_multimode_fit import _policy, _rows
    from fisher_graph.gemma3_layer_multimode_fit import (
        evaluate_layer_multimode_split_refit,
        fit_layer_multimode_candidate,
    )

    rows = _rows(family_count=6, high_rank=True)
    policy = _policy(steps=1)
    fitted = fit_layer_multimode_candidate(
        rows=rows,
        chart_count=1,
        modes_per_chart=2,
        tangent_rank=16,
        policy=policy,
        allow_test_policy=True,
    )
    split = evaluate_layer_multimode_split_refit(
        rows=rows,
        chart_count=1,
        modes_per_chart=2,
        tangent_rank=16,
        policy=policy,
        allow_test_policy=True,
    )
    compiler_module._require_live_selected_fit_authority(
        (fitted,) * 18,
        (split,) * 18,
    )
    copied = replace(split)
    assert copied.live_split_evaluator_authorized is False
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="original live split-evaluator evidence",
    ):
        compiler_module._require_live_selected_fit_authority(
            (fitted,) * 18,
            (copied, *(split,) * 17),
        )


def test_source_free_bundle_roundtrip_never_calls_native_mlp(tmp_path: Path) -> None:
    torch.manual_seed(22)
    compiled = _compile()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected = compiled.model(input_ids)
    path = tmp_path / "compiled.pt"
    with pytest.raises(LayerMultimodeCompilationError, match="test-only opt-in"):
        save_compiled_layer_multimode_bundle(compiled, path)
    saved = save_compiled_layer_multimode_bundle(
        compiled,
        path,
        allow_test_unbound_materialization=True,
    )
    assert saved.bundle_sha256 == _file_sha(path)
    assert saved.byte_count == path.stat().st_size
    with pytest.raises(FileExistsError, match="overwrite"):
        save_compiled_layer_multimode_bundle(
            compiled,
            path,
            allow_test_unbound_materialization=True,
        )
    assert _file_sha(path) == saved.bundle_sha256

    ToyNativeMLP.call_count = 0
    with pytest.raises(LayerMultimodeCompilationError, match="test-only opt-in"):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
        )
    with pytest.raises(LayerMultimodeCompilationError, match="production-shape policy"):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_unbound_materialization=True,
        )
    with pytest.raises(
        LayerMultimodeCompilationError,
        match="entirely in eval mode",
    ):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=ToyCausalLM,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )
    loaded = load_compiled_layer_multimode_bundle(
        path,
        expected_bundle_sha256=saved.bundle_sha256,
        model_factory=_eval_toy_model,
        allow_test_nonproduction_shape=True,
        allow_test_unbound_materialization=True,
    )
    actual = loaded.model(input_ids)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert ToyNativeMLP.call_count == 0
    assert loaded.receipt == compiled.receipt
    assert all(not module.training for module in loaded.model.modules())
    assert loaded.receipt.compiled_execution_fingerprint_sha256 == (
        compiler_module.Gemma3CausalLMAdapter(
            loaded.model
        ).execution_fingerprint()
    )
    assert (
        loaded.receipt.protected_architecture_sha256
        == compiler_module._protected_architecture_sha256(loaded.model)
    )
    assert loaded.model.get_input_embeddings().weight is loaded.model.get_output_embeddings().weight
    assert all(type(layer.mlp) is Float32LayerMultimodeMLP for layer in loaded.model.model.layers)
    assert all(
        type(layer.post_feedforward_layernorm) is nn.Identity
        for layer in loaded.model.model.layers
    )

    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"manifest_json", "state_dict"}
    assert not any(".mlp.gate_proj." in name for name in raw["state_dict"])
    assert not any(".post_feedforward_layernorm." in name for name in raw["state_dict"])


def test_bundle_rejects_external_and_internal_tampering(tmp_path: Path) -> None:
    compiled = _compile()
    path = tmp_path / "compiled.pt"
    saved = save_compiled_layer_multimode_bundle(
        compiled,
        path,
        allow_test_unbound_materialization=True,
    )
    with pytest.raises(LayerMultimodeCompilationError, match="external bundle hash"):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256="0" * 64,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )

    raw = torch.load(path, map_location="cpu", weights_only=True)
    tensor_name = "model.layers.0.mlp.centers"
    raw["state_dict"][tensor_name][0, 0] += 0.5
    tampered = tmp_path / "tampered.pt"
    torch.save(raw, tampered)
    tampered.chmod(0o600)
    with pytest.raises(LayerMultimodeCompilationError, match="failed integrity"):
        load_compiled_layer_multimode_bundle(
            tampered,
            expected_bundle_sha256=_file_sha(tampered),
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )

    manifest_attack = torch.load(path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_attack["manifest_json"])
    manifest["compilation_receipt"]["replacement_count"] = True
    manifest_attack["manifest_json"] = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    )
    attacked = tmp_path / "attacked.pt"
    torch.save(manifest_attack, attacked)
    attacked.chmod(0o600)
    with pytest.raises(LayerMultimodeCompilationError):
        load_compiled_layer_multimode_bundle(
            attacked,
            expected_bundle_sha256=_file_sha(attacked),
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )

    integer_float_attack = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    integer_manifest = json.loads(integer_float_attack["manifest_json"])
    integer_manifest["layer_coefficients"][0]["mode_temperature"] = 1
    manifest_without_hash = {
        key: value
        for key, value in integer_manifest.items()
        if key != "manifest_sha256"
    }
    integer_manifest["manifest_sha256"] = compiler_module._digest(
        compiler_module._MANIFEST_DOMAIN,
        manifest_without_hash,
    )
    integer_float_attack["manifest_json"] = compiler_module._canonical_json(
        integer_manifest
    )
    integer_path = tmp_path / "integer-float-attack.pt"
    torch.save(integer_float_attack, integer_path)
    integer_path.chmod(0o600)
    with pytest.raises(LayerMultimodeCompilationError, match="JSON float"):
        load_compiled_layer_multimode_bundle(
            integer_path,
            expected_bundle_sha256=_file_sha(integer_path),
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )


def test_atomic_save_failure_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _compile()
    destination = tmp_path / "failed.pt"

    def fail_save(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(compiler_module.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="injected"):
        save_compiled_layer_multimode_bundle(
            compiled,
            destination,
            allow_test_unbound_materialization=True,
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_post_publish_verification_failure_rolls_back_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _compile()
    destination = tmp_path / "rolled-back.pt"
    strict_read = compiler_module._strict_read_bundle_bytes
    fsync_directory = compiler_module._fsync_directory
    calls = 0
    fsync_calls = 0

    def fail_second_read(path: Path) -> tuple[bytes, str, os.stat_result]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected post-link read failure")
        return strict_read(path)

    def fail_cleanup_fsync(path: Path) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("injected cleanup fsync failure")
        fsync_directory(path)

    monkeypatch.setattr(
        compiler_module,
        "_strict_read_bundle_bytes",
        fail_second_read,
    )
    monkeypatch.setattr(
        compiler_module,
        "_fsync_directory",
        fail_cleanup_fsync,
    )
    with pytest.raises(OSError, match="post-link"):
        save_compiled_layer_multimode_bundle(
            compiled,
            destination,
            allow_test_unbound_materialization=True,
        )
    assert calls == 2
    assert fsync_calls == 2
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_save_rollback_does_not_unlink_path_swapped_by_another_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _compile()
    destination = tmp_path / "swapped.pt"
    strict_read = compiler_module._strict_read_bundle_bytes
    calls = 0
    replacement = b"unrelated replacement"

    def swap_before_post_publish_read(
        path: Path,
    ) -> tuple[bytes, str, os.stat_result]:
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.unlink()
            destination.write_bytes(replacement)
            destination.chmod(0o600)
            raise OSError("injected swapped-path failure")
        return strict_read(path)

    monkeypatch.setattr(
        compiler_module,
        "_strict_read_bundle_bytes",
        swap_before_post_publish_read,
    )
    with pytest.raises(OSError, match="swapped-path"):
        save_compiled_layer_multimode_bundle(
            compiled,
            destination,
            allow_test_unbound_materialization=True,
        )
    assert destination.read_bytes() == replacement


def test_bundle_rejects_symlink_hardlink_permissions_and_size_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _compile()
    path = tmp_path / "compiled.pt"
    saved = save_compiled_layer_multimode_bundle(
        compiled,
        path,
        allow_test_unbound_materialization=True,
    )

    symlink = tmp_path / "symlink.pt"
    symlink.symlink_to(path)
    with pytest.raises(LayerMultimodeCompilationError, match="non-symlink"):
        load_compiled_layer_multimode_bundle(
            symlink,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )
    symlink.unlink()

    hardlink = tmp_path / "hardlink.pt"
    os.link(path, hardlink)
    with pytest.raises(LayerMultimodeCompilationError, match="one filesystem link"):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )
    hardlink.unlink()

    path.chmod(0o640)
    with pytest.raises(LayerMultimodeCompilationError, match="0600"):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )
    path.chmod(0o600)

    monkeypatch.setattr(
        compiler_module,
        "MAX_LAYER_MULTIMODE_BUNDLE_BYTES",
        saved.byte_count - 1,
    )

    def forbidden_read(descriptor: int, count: int) -> bytes:
        raise AssertionError("oversized bundle must be rejected before read")

    monkeypatch.setattr(compiler_module.os, "read", forbidden_read)
    with pytest.raises(LayerMultimodeCompilationError, match="size"):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )


def test_descriptor_pinned_load_rejects_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = _compile()
    path = tmp_path / "compiled.pt"
    saved = save_compiled_layer_multimode_bundle(
        compiled,
        path,
        allow_test_unbound_materialization=True,
    )
    replacement_bytes = path.read_bytes()
    real_read = os.read
    replaced = False

    def replace_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, count)
        if not replaced:
            replaced = True
            path.unlink()
            path.write_bytes(replacement_bytes)
            path.chmod(0o600)
        return chunk

    monkeypatch.setattr(compiler_module.os, "read", replace_after_first_read)
    with pytest.raises(LayerMultimodeCompilationError):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=_eval_toy_model,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )
    assert replaced is True


def test_bundle_load_rejects_wrong_architecture_shell(tmp_path: Path) -> None:
    compiled = _compile()
    path = tmp_path / "compiled.pt"
    saved = save_compiled_layer_multimode_bundle(
        compiled,
        path,
        allow_test_unbound_materialization=True,
    )
    with pytest.raises(LayerMultimodeCompilationError):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=lambda: ToyCausalLM(hidden_width=5).eval(),
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )

    def same_shape_different_attention() -> ToyCausalLM:
        shell = ToyCausalLM()
        for layer in shell.model.layers:
            replacement = BehaviorallyDifferentToyAttention(HIDDEN)
            replacement.load_state_dict(layer.self_attn.state_dict())
            layer.self_attn = replacement
        return shell.eval()

    with pytest.raises(
        LayerMultimodeCompilationError,
        match="receipt does not reproduce",
    ):
        load_compiled_layer_multimode_bundle(
            path,
            expected_bundle_sha256=saved.bundle_sha256,
            model_factory=same_shape_different_attention,
            allow_test_nonproduction_shape=True,
            allow_test_unbound_materialization=True,
        )
