from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fisher_graph.adapters import (
    Gemma3CausalLMAdapter,
    module_state_fingerprint,
)
from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorExecutor,
    Gemma3ModalGeneratorMLP,
    Gemma3ModalGeneratorReplacement,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorConfig,
    ModalGeneratorFactors,
    ModalGeneratorPlan,
)


def _plan(
    *,
    source_model_sha256: str = "a" * 64,
    generator_id: str = "layer.0.cluster.0",
    input_site: str = "layer.0.mlp.normalized_input",
    output_site: str = "layer.0.mlp.operator_output",
    input_factor: Tensor | None = None,
    output_factor: Tensor | None = None,
    bias: Tensor | None = None,
    fit_intercept: bool = True,
) -> ModalGeneratorPlan:
    if input_factor is None:
        input_factor = torch.tensor(
            (
                (0.5, -0.2),
                (0.1, 0.4),
                (-0.3, 0.25),
                (0.2, 0.15),
            ),
            dtype=torch.float64,
        )
    if output_factor is None:
        output_factor = torch.tensor(
            (
                (0.4, -0.1, 0.3, 0.2),
                (-0.2, 0.5, 0.1, -0.4),
            ),
            dtype=torch.float64,
        )
    if bias is None:
        bias = torch.tensor(
            (0.05, -0.1, 0.2, -0.15),
            dtype=torch.float64,
        )
        if not fit_intercept:
            bias = torch.zeros_like(bias)
    rank = input_factor.shape[1]
    binding = ModalGeneratorBinding.create(
        generator_id=generator_id,
        input_kind="native_layer_input",
        input_site=input_site,
        output_site=output_site,
        source_model_sha256=source_model_sha256,
        input_catalog_sha256="b" * 64,
        output_catalog_sha256="c" * 64,
        cluster_plan_sha256="d" * 64,
        fit_split_sha256="e" * 64,
        eval_split_sha256="f" * 64,
    )
    config = ModalGeneratorConfig(
        ranks=(rank,),
        fit_intercept=fit_intercept,
    )
    factors = ModalGeneratorFactors(
        rank=rank,
        input_factor=input_factor,
        output_factor=output_factor,
        bias=bias,
    )
    matrix_cost = factors.input_width * rank + rank * factors.output_width
    bias_cost = factors.output_width if fit_intercept else 0
    return ModalGeneratorPlan(
        binding=binding,
        config=config,
        factors=factors,
        parameter_count=matrix_cost + bias_cost,
        macs_per_token=matrix_cost + bias_cost,
    )


class _GemmaMLP(nn.Module):
    def __init__(
        self,
        width: int = 4,
        intermediate: int = 6,
        *,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            width,
            intermediate,
            bias=bias,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            width,
            intermediate,
            bias=bias,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            intermediate,
            width,
            bias=bias,
            dtype=dtype,
        )

    def features(self, values: Tensor) -> Tensor:
        return F.gelu(
            self.gate_proj(values),
            approximate="tanh",
        ) * self.up_proj(values)

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(self.features(values))


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _storage_pointers(module: nn.Module) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }


def test_compiled_mlp_matches_retained_native_plus_dense_generator() -> None:
    torch.manual_seed(8801)
    source = _freeze(_GemmaMLP())
    assert isinstance(source, _GemmaMLP)
    source_fingerprint = module_state_fingerprint(source)
    plan = _plan()
    compiled = Gemma3ModalGeneratorMLP(
        source,
        removed_mode_indices=(1, 4),
        generator_plan=plan,
        activation="gelu_pytorch_tanh",
    )
    values = torch.randn(2, 3, 4)
    retained_indices = torch.tensor((0, 2, 3, 5))
    native_features = source.features(values).index_select(
        -1,
        retained_indices,
    )
    retained = F.linear(
        native_features,
        source.down_proj.weight.index_select(1, retained_indices),
    )
    generated = (
        values.double()
        @ plan.factors.input_factor
        @ plan.factors.output_factor
        + plan.factors.bias
    ).to(values.dtype)
    expected = retained + generated

    torch.testing.assert_close(
        compiled(values, condition="generated"),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        compiled(values, condition="matched_deletion"),
        retained,
        rtol=2e-6,
        atol=2e-6,
    )
    assert not torch.equal(
        compiled(values, condition="generated"),
        compiled(values, condition="matched_deletion"),
    )

    # The removed z coordinates have no physical gate row, up row, or down
    # column in the runtime module.
    assert tuple(compiled.retained_source_indices.tolist()) == (0, 2, 3, 5)
    assert compiled.gate_proj.weight.shape == (4, 4)
    assert compiled.up_proj.weight.shape == (4, 4)
    assert compiled.down_proj.weight.shape == (4, 4)
    assert compiled.generator_input_proj.weight.shape == (2, 4)
    assert compiled.generator_output_proj.weight.shape == (4, 2)
    assert compiled.generator_output_proj.bias is not None
    assert compiled.down_proj.weight.is_contiguous()
    assert module_state_fingerprint(source) == source_fingerprint
    assert not (_storage_pointers(source) & _storage_pointers(compiled))


def test_compiled_mlp_exact_parameter_and_mac_accounting() -> None:
    torch.manual_seed(8802)
    source = _freeze(_GemmaMLP())
    assert isinstance(source, _GemmaMLP)
    plan = _plan()
    compiled = Gemma3ModalGeneratorMLP(
        source,
        removed_mode_indices=(0, 2, 5),
        generator_plan=plan,
        activation="gelu_pytorch_tanh",
    )

    assert compiled.source_native_parameter_count == 3 * 4 * 6 == 72
    assert compiled.native_removed_parameter_count == 3 * 4 * 3 == 36
    assert compiled.generator_parameter_count == plan.parameter_count == 20
    assert compiled.generator_macs_per_token == plan.macs_per_token == 20
    assert compiled.candidate_parameter_count == 72 - 36 + 20 == 56
    assert sum(parameter.numel() for parameter in compiled.parameters()) == 56
    assert compiled.net_parameter_savings == 16
    assert compiled.native_removed_macs_per_token == 36
    assert compiled.net_macs_saved_per_token == 16


def test_bias_free_generator_has_no_runtime_bias_parameter() -> None:
    source = _freeze(_GemmaMLP())
    plan = _plan(fit_intercept=False)
    compiled = Gemma3ModalGeneratorMLP(
        source,
        removed_mode_indices=(1,),
        generator_plan=plan,
        activation="gelu_pytorch_tanh",
    )

    assert compiled.generator_output_proj.bias is None
    assert plan.parameter_count == 16
    assert plan.macs_per_token == 16
    assert compiled.generator_parameter_count == 16


def test_full_native_mlp_can_be_replaced_by_one_modal_generator() -> None:
    source = _freeze(_GemmaMLP())
    plan = _plan()
    compiled = Gemma3ModalGeneratorMLP(
        source,
        removed_mode_indices=tuple(range(6)),
        generator_plan=plan,
        activation="gelu_pytorch_tanh",
    )
    values = torch.randn(2, 3, 4)
    expected = (
        values.double()
        @ plan.factors.input_factor
        @ plan.factors.output_factor
        + plan.factors.bias
    ).to(values.dtype)

    torch.testing.assert_close(compiled(values), expected)
    assert torch.count_nonzero(
        compiled(values, condition="matched_deletion")
    ) == 0
    assert compiled.retained_width == 0
    assert compiled.is_full_native_replacement is True
    assert compiled.gate_proj.weight.shape == (0, 4)
    assert compiled.up_proj.weight.shape == (0, 4)
    assert compiled.down_proj.weight.shape == (4, 0)
    assert compiled.native_removed_parameter_count == 72
    assert compiled.candidate_parameter_count == plan.parameter_count
    assert compiled.net_parameter_savings == 72 - plan.parameter_count


def test_compilation_rejects_nonfinite_runtime_cast_and_unsafe_source() -> None:
    half_source = _freeze(_GemmaMLP(dtype=torch.float16))
    huge_plan = _plan(
        input_factor=torch.full((4, 1), 1e100, dtype=torch.float64),
        output_factor=torch.ones((1, 4), dtype=torch.float64),
        bias=torch.zeros(4, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="runtime model dtype"):
        Gemma3ModalGeneratorMLP(
            half_source,
            removed_mode_indices=(1,),
            generator_plan=huge_plan,
            activation="gelu_pytorch_tanh",
        )

    with pytest.raises(ValueError, match="frozen eval"):
        Gemma3ModalGeneratorMLP(
            _GemmaMLP(),
            removed_mode_indices=(1,),
            generator_plan=_plan(),
            activation="gelu_pytorch_tanh",
        )
    with pytest.raises(ValueError, match="bias-free"):
        Gemma3ModalGeneratorMLP(
            _freeze(_GemmaMLP(bias=True)),
            removed_mode_indices=(1,),
            generator_plan=_plan(),
            activation="gelu_pytorch_tanh",
        )


def test_compilation_rejects_bad_indices_width_and_poisoned_plan() -> None:
    source = _freeze(_GemmaMLP())
    for indices in ((), (1, 1), (2, 1), (-1,), (6,)):
        with pytest.raises(ValueError, match="removed_mode_indices"):
            Gemma3ModalGeneratorMLP(
                source,
                removed_mode_indices=indices,
                generator_plan=_plan(),
                activation="gelu_pytorch_tanh",
            )

    narrow = _plan(
        input_factor=torch.ones((3, 1), dtype=torch.float64),
        output_factor=torch.ones((1, 4), dtype=torch.float64),
        bias=torch.zeros(4, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="residual width"):
        Gemma3ModalGeneratorMLP(
            source,
            removed_mode_indices=(1,),
            generator_plan=narrow,
            activation="gelu_pytorch_tanh",
        )

    poisoned = _plan()
    poisoned.factors.input_factor[0, 0] += 1.0
    with pytest.raises(ValueError, match="hash|does not match"):
        Gemma3ModalGeneratorMLP(
            source,
            removed_mode_indices=(1,),
            generator_plan=poisoned,
            activation="gelu_pytorch_tanh",
        )


class _Config:
    model_type = "gemma3_text"
    hidden_size = 4
    intermediate_size = 6
    vocab_size = 17
    num_hidden_layers = 3
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    query_pre_attn_scalar = 4
    max_position_embeddings = 32
    sliding_window = 4
    layer_types = [
        "full_attention",
        "full_attention",
        "full_attention",
    ]
    rope_parameters = {
        "full_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }
    }
    rms_norm_eps = 1e-6
    attention_dropout = 0.0
    attention_bias = False
    hidden_activation = "gelu_pytorch_tanh"
    final_logit_softcapping = None
    attn_logit_softcapping = None
    use_bidirectional_attention = False
    _attn_implementation = "eager"


class _Norm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, values: Tensor) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return (normalized * (1.0 + self.weight.float())).to(values.dtype)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.q_norm = _Norm(2)
        self.k_norm = _Norm(2)

    def forward(self, hidden_states: Tensor, **_: object) -> tuple[Tensor, None]:
        return torch.zeros_like(hidden_states), None


class _MLP(_GemmaMLP):
    pass


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = _Norm(4)
        self.self_attn = _Attention()
        self.post_attention_layernorm = _Norm(4)
        self.pre_feedforward_layernorm = _Norm(4)
        self.mlp = _MLP()
        self.post_feedforward_layernorm = _Norm(4)

    def forward(self, hidden_states: Tensor, **kwargs: object) -> Tensor:
        residual = hidden_states
        attention, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        feed_forward = self.mlp(
            self.pre_feedforward_layernorm(hidden_states)
        )
        return residual + self.post_feedforward_layernorm(feed_forward)


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.embed_tokens = nn.Embedding(17, 4)
        self.layers = nn.ModuleList((_Layer(), _Layer(), _Layer()))
        self.norm = _Norm(4)
        self.rotary_emb = nn.Identity()

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.model = _TextModel()
        self.lm_head = nn.Linear(4, 17, bias=False)
        with torch.no_grad():
            for layer in self.model.layers:
                for parameter in layer.self_attn.parameters():
                    parameter.zero_()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(self, **kwargs: object) -> SimpleNamespace:
        result = self.model(**kwargs)
        return SimpleNamespace(logits=self.lm_head(result.last_hidden_state))


def _adapter(seed: int = 8803) -> Gemma3CausalLMAdapter:
    torch.manual_seed(seed)
    model = _CausalLM().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return Gemma3CausalLMAdapter(model)


def _bound_plan(
    adapter: Gemma3CausalLMAdapter,
    ordinal: int,
    *,
    generator_id: str | None = None,
    source_model_sha256: str | None = None,
    input_site: str | None = None,
    output_site: str | None = None,
) -> ModalGeneratorPlan:
    transformer = adapter.layers[ordinal].transformer
    assert transformer is not None
    stage = next(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    return _plan(
        source_model_sha256=(
            adapter.model_fingerprint()
            if source_model_sha256 is None
            else source_model_sha256
        ),
        generator_id=generator_id or f"layer.{ordinal}.cluster.0",
        input_site=input_site or stage.normalized_input_site,
        output_site=output_site or stage.operator_output_site,
    )


def _batch() -> CalibrationBatch:
    input_ids = torch.tensor(
        (
            (1, 2, 3, 4),
            (5, 6, 0, 0),
        )
    )
    valid = torch.tensor(
        (
            (True, True, True, True),
            (True, True, False, False),
        )
    )
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": valid,
        },
        targets=torch.where(
            valid,
            input_ids.roll(-1, dims=1),
            torch.full_like(input_ids, -100),
        ),
        valid_positions=valid,
        example_ids=("modal-generator-a", "modal-generator-b"),
    )


def _executor_fixture() -> tuple[
    Gemma3CausalLMAdapter,
    Gemma3ModalGeneratorExecutor,
]:
    adapter = _adapter()
    replacements = (
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=0,
            removed_mode_indices=(1, 4),
            generator_plan=_bound_plan(adapter, 0),
        ),
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=2,
            removed_mode_indices=(3,),
            generator_plan=_bound_plan(adapter, 2),
        ),
    )
    return adapter, Gemma3ModalGeneratorExecutor(adapter, replacements)


def test_multilayer_executor_overlay_accounting_and_source_restoration() -> None:
    adapter, executor = _executor_fixture()
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()
    generated = executor.run(batch.model_inputs, condition="generated")
    deletion = executor.run(
        batch.model_inputs,
        condition="matched_deletion",
    )

    assert not torch.equal(
        generated.model_output.logits[batch.valid_positions],
        deletion.model_output.logits[batch.valid_positions],
    )
    assert (
        generated.replacement_scope
        == "partial_native_mlp_mode_replacement"
    )
    assert generated.replaced_layer_count == 2
    assert generated.removed_mode_count == 3
    assert generated.native_removed_learned_parameters == 3 * 4 * 3 == 36
    assert generated.modal_generator_learned_parameters == 2 * 20 == 40
    assert generated.net_stored_parameter_savings == -4
    assert generated.valid_tokens == 6
    assert generated.logical_linear_macs_native_removed == 6 * 36 == 216
    assert generated.logical_modal_generator_macs == 6 * 40 == 240
    assert generated.logical_executed_modal_generator_macs == 240
    assert generated.net_logical_macs_saved == -24
    assert deletion.logical_modal_generator_macs == 240
    assert deletion.logical_executed_modal_generator_macs == 0
    assert deletion.net_logical_macs_saved == 216
    assert (
        generated.candidate_whole_model_learned_parameters
        == generated.source_whole_model_learned_parameters + 4
    )
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_fingerprint
    assert tuple(executor.compiled_mlps) == ("0", "2")
    assert executor.compiled_mlps["0"].gate_proj.out_features == 4
    assert executor.compiled_mlps["2"].gate_proj.out_features == 5


def test_multilayer_executor_restores_source_after_model_error() -> None:
    adapter, executor = _executor_fixture()
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()

    def fail(
        _module: nn.Module,
        _args: tuple[object, ...],
    ) -> None:
        raise RuntimeError("sentinel modal generator failure")

    handle = executor.compiled_mlps[
        "2"
    ].generator_input_proj.register_forward_pre_hook(fail)
    try:
        with pytest.raises(
            RuntimeError,
            match="sentinel modal generator failure",
        ):
            executor.run(batch.model_inputs)
    finally:
        handle.remove()
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == source_mlps
    assert adapter.model_fingerprint() == source_fingerprint

    # A failed overlay cannot leave the executor stuck active.
    result = executor.run(batch.model_inputs)
    assert result.replaced_layer_count == 2


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"source_model_sha256": "9" * 64}, "live Gemma model"),
        ({"input_site": "layer.0.wrong.input"}, "input site"),
        ({"output_site": "layer.0.wrong.output"}, "output site"),
    ),
)
def test_executor_rejects_bad_model_and_site_bindings(
    override: dict[str, str],
    message: str,
) -> None:
    adapter = _adapter()
    replacement = Gemma3ModalGeneratorReplacement(
        layer_ordinal=0,
        removed_mode_indices=(1,),
        generator_plan=_bound_plan(adapter, 0, **override),
    )
    with pytest.raises(ValueError, match=message):
        Gemma3ModalGeneratorExecutor(adapter, (replacement,))


def test_executor_rejects_duplicate_layers_ids_order_and_foreign_plan() -> None:
    adapter = _adapter()
    first = Gemma3ModalGeneratorReplacement(
        layer_ordinal=0,
        removed_mode_indices=(1,),
        generator_plan=_bound_plan(adapter, 0),
    )
    duplicate = Gemma3ModalGeneratorReplacement(
        layer_ordinal=0,
        removed_mode_indices=(2,),
        generator_plan=_bound_plan(
            adapter,
            0,
            generator_id="layer.0.cluster.1",
        ),
    )
    with pytest.raises(ValueError, match="only one replacement"):
        Gemma3ModalGeneratorExecutor(adapter, (first, duplicate))

    second = Gemma3ModalGeneratorReplacement(
        layer_ordinal=1,
        removed_mode_indices=(1,),
        generator_plan=_bound_plan(
            adapter,
            1,
            generator_id="layer.0.cluster.0",
        ),
    )
    with pytest.raises(ValueError, match="unique"):
        Gemma3ModalGeneratorExecutor(adapter, (first, second))
    with pytest.raises(ValueError, match="layer order"):
        Gemma3ModalGeneratorExecutor(adapter, (second, first))

    foreign_adapter = _adapter(seed=8804)
    foreign = Gemma3ModalGeneratorReplacement(
        layer_ordinal=0,
        removed_mode_indices=(1,),
        generator_plan=_bound_plan(foreign_adapter, 0),
    )
    with pytest.raises(ValueError, match="live Gemma model"):
        Gemma3ModalGeneratorExecutor(adapter, (foreign,))


def test_executor_rejects_live_source_drift_and_plan_header_poisoning() -> None:
    adapter, executor = _executor_fixture()
    batch = _batch()
    with torch.no_grad():
        adapter.module.model.layers[0].mlp.gate_proj.weight[0, 0] += 0.25
    with pytest.raises(ValueError, match="fingerprint drifted"):
        executor.run(batch.model_inputs)

    compiled_adapter, compiled_executor = _executor_fixture()
    compiled_source_fingerprint = compiled_adapter.model_fingerprint()
    with torch.no_grad():
        compiled_executor.compiled_mlps[
            "0"
        ].generator_input_proj.weight[0, 0] += 0.25
    with pytest.raises(ValueError, match="plan binding drifted"):
        compiled_executor.run(batch.model_inputs)
    assert compiled_adapter.model_fingerprint() == compiled_source_fingerprint

    fresh = _adapter()
    plan = _bound_plan(fresh, 0)
    object.__setattr__(plan, "artifact_sha256", "0" * 64)
    replacement = Gemma3ModalGeneratorReplacement(
        layer_ordinal=0,
        removed_mode_indices=(1,),
        generator_plan=plan,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        Gemma3ModalGeneratorExecutor(fresh, (replacement,))


def test_fragment_bound_dense_replacement_is_derived_from_lowering() -> None:
    from test_gemma3_modal_generator_graph_executor import _fixture

    fixture = _fixture()
    lowering = fixture.lowerings[0]
    replacement = Gemma3ModalGeneratorReplacement.from_lowering(lowering)
    fragment = next(
        value
        for value in lowering.fragment_plan.fragments
        if value.artifact_sha256 == lowering.selected_fragment_sha256
    )

    assert replacement.layer_ordinal == fragment.layer_ordinal
    assert replacement.removed_mode_indices == fragment.removed_mode_indices
    assert replacement.generator_plan.artifact_sha256 == (
        lowering.fused_residual_plan.artifact_sha256
    )
    executor = Gemma3ModalGeneratorExecutor(
        fixture.adapter,  # type: ignore[arg-type]
        (replacement,),
    )
    deletion = executor.run(_batch().model_inputs, condition="matched_deletion")
    assert deletion.logical_executed_modal_generator_macs == 0

    with pytest.raises(ValueError, match="requires its authenticated"):
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=fragment.layer_ordinal,
            removed_mode_indices=fragment.removed_mode_indices,
            generator_plan=lowering.fused_residual_plan,
        )
    with pytest.raises(ValueError, match="binding disagree"):
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=fragment.layer_ordinal,
            removed_mode_indices=(5,),
            generator_plan=lowering.fused_residual_plan,
            lowering=lowering,
        )
    drifted = Gemma3ModalGeneratorReplacement.from_lowering(lowering)
    object.__setattr__(drifted, "removed_mode_indices", (5,))
    with pytest.raises(ValueError, match="drifted after creation"):
        Gemma3ModalGeneratorExecutor(
            fixture.adapter,  # type: ignore[arg-type]
            (drifted,),
        )
