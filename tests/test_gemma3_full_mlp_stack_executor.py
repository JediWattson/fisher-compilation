from __future__ import annotations

import pytest
import torch
from torch import nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.gemma3_full_mlp_stack_executor import (
    Gemma3FullMLPStackExecutor,
)
from fisher_graph.gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)

from test_gemma3_modal_generator_executor import (
    _adapter,
    _batch,
    _bound_plan,
    _plan,
)


def _full_replacements(
    adapter: object,
) -> tuple[Gemma3ModalGeneratorReplacement, ...]:
    return tuple(
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=ordinal,
            removed_mode_indices=tuple(range(6)),
            generator_plan=_bound_plan(adapter, ordinal),
        )
        for ordinal in range(3)
    )


def _fixture() -> tuple[object, Gemma3FullMLPStackExecutor]:
    adapter = _adapter()
    return adapter, Gemma3FullMLPStackExecutor(
        adapter,
        _full_replacements(adapter),
    )


def test_full_stack_executes_every_layer_and_reports_honest_scopes() -> None:
    adapter, executor = _fixture()
    batch = _batch()
    source_model = adapter.module
    source_parameters = sum(
        parameter.numel() for parameter in source_model.parameters()
    )
    source_mlps = tuple(layer.mlp for layer in source_model.model.layers)
    source_attention = tuple(
        layer.self_attn for layer in source_model.model.layers
    )
    source_embeddings = source_model.model.embed_tokens
    source_final_norm = source_model.model.norm
    source_head = source_model.lm_head
    source_fingerprint = adapter.model_fingerprint()

    generated = executor.run(batch.model_inputs, condition="generated")
    deletion = executor.run(
        batch.model_inputs,
        condition="matched_deletion",
    )

    native_mlp_parameters = 3 * 3 * 4 * 6
    generator_parameters = 3 * 20
    logical_candidate = (
        source_parameters - native_mlp_parameters + generator_parameters
    )
    assert generated.replacement_scope == (
        "full_native_mlp_stack_replacement"
    )
    assert generated.native_components_retained == (
        "embeddings",
        "attention",
        "normalization",
        "language_model_head",
    )
    assert generated.logical_candidate_excludes_native_mlp_stack is True
    assert generated.experimental_resident_source_state_retained is True
    assert generated.replaced_layer_count == 3
    assert generated.removed_mode_count == 18
    assert generated.source_whole_model_learned_parameters == source_parameters
    assert (
        generated.logical_native_mlp_stack_learned_parameters
        == native_mlp_parameters
    )
    assert (
        generated.logical_retained_native_non_mlp_learned_parameters
        == source_parameters - native_mlp_parameters
    )
    assert (
        generated.logical_generator_stack_learned_parameters
        == generator_parameters
    )
    assert generated.logical_candidate_learned_parameters == logical_candidate
    assert (
        generated.candidate_whole_model_learned_parameters
        == logical_candidate
    )
    assert (
        generated.logical_net_stored_parameter_savings
        == native_mlp_parameters - generator_parameters
    )
    assert (
        generated.experimental_resident_source_learned_parameters
        == source_parameters
    )
    assert (
        generated.experimental_resident_compiled_learned_parameters
        == generator_parameters
    )
    assert (
        generated.experimental_resident_total_learned_parameters
        == source_parameters + generator_parameters
    )
    assert (
        generated.experimental_resident_overhead_vs_logical_candidate
        == native_mlp_parameters
    )
    assert (
        executor.logical_candidate_learned_parameters == logical_candidate
    )
    assert (
        executor.experimental_resident_learned_parameters
        == source_parameters + generator_parameters
    )
    assert (
        sum(parameter.numel() for parameter in executor.parameters())
        == generator_parameters
    )

    assert generated.valid_tokens == 6
    assert (
        generated.logical_linear_macs_native_mlp_stack
        == 6 * native_mlp_parameters
    )
    assert generated.logical_generator_macs == 6 * generator_parameters
    assert (
        generated.logical_executed_generator_macs
        == 6 * generator_parameters
    )
    assert generated.logical_generator_bias_additions == 6 * 3 * 4
    assert (
        generated.logical_executed_generator_bias_additions == 6 * 3 * 4
    )
    assert (
        generated.net_logical_macs_saved
        == 6 * (native_mlp_parameters - generator_parameters)
    )
    assert deletion.logical_executed_generator_macs == 0
    assert deletion.logical_executed_generator_bias_additions == 0
    assert (
        deletion.net_logical_macs_saved == 6 * native_mlp_parameters
    )
    assert not torch.equal(
        generated.model_output.logits,
        deletion.model_output.logits,
    )

    assert tuple(layer.mlp for layer in source_model.model.layers) == source_mlps
    assert tuple(
        layer.self_attn for layer in source_model.model.layers
    ) == source_attention
    assert source_model.model.embed_tokens is source_embeddings
    assert source_model.model.norm is source_final_norm
    assert source_model.lm_head is source_head
    assert adapter.model_fingerprint() == source_fingerprint
    assert tuple(executor.compiled_mlps) == ("0", "1", "2")
    assert all(
        candidate.is_full_native_replacement
        for candidate in executor.compiled_mlps.values()
    )
    assert all(
        candidate.retained_width == 0
        for candidate in executor.compiled_mlps.values()
    )


def test_full_stack_restores_every_native_mlp_after_model_error() -> None:
    adapter, executor = _fixture()
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()

    def fail(
        _module: nn.Module,
        _arguments: tuple[object, ...],
    ) -> None:
        raise RuntimeError("sentinel full-stack failure")

    handle = executor.compiled_mlps[
        "2"
    ].generator_input_proj.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="sentinel full-stack failure"):
            executor.run(batch.model_inputs)
    finally:
        handle.remove()

    assert tuple(
        layer.mlp for layer in adapter.module.model.layers
    ) == source_mlps
    assert adapter.model_fingerprint() == source_fingerprint
    recovered = executor.run(batch.model_inputs)
    assert recovered.replaced_layer_count == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "one replacement per Gemma layer"),
        ("reordered", "ordinals must exactly cover"),
        ("partial", "remove exactly"),
        ("wrong_width", "residual width"),
    ),
)
def test_full_stack_rejects_incomplete_or_nonfull_replacements(
    mutation: str,
    message: str,
) -> None:
    adapter = _adapter()
    replacements = list(_full_replacements(adapter))
    if mutation == "missing":
        replacements.pop()
    elif mutation == "reordered":
        replacements[0], replacements[1] = replacements[1], replacements[0]
    elif mutation == "partial":
        replacements[1] = Gemma3ModalGeneratorReplacement(
            layer_ordinal=1,
            removed_mode_indices=(0, 1, 2, 3, 4),
            generator_plan=_bound_plan(adapter, 1),
        )
    elif mutation == "wrong_width":
        replacements[2] = Gemma3ModalGeneratorReplacement(
            layer_ordinal=2,
            removed_mode_indices=tuple(range(6)),
            generator_plan=_plan(
                source_model_sha256=adapter.model_fingerprint(),
                generator_id="layer.2.full",
                input_site="layer.2.mlp.normalized_input",
                output_site="layer.2.mlp.operator_output",
                input_factor=torch.ones((3, 1), dtype=torch.float64),
                output_factor=torch.ones((1, 4), dtype=torch.float64),
                bias=torch.zeros(4, dtype=torch.float64),
            ),
        )
    else:
        raise AssertionError("unknown test mutation")

    with pytest.raises((TypeError, ValueError), match=message):
        Gemma3FullMLPStackExecutor(adapter, replacements)


def test_full_stack_rejects_layer_count_drift_and_preserves_source() -> None:
    adapter = _adapter()
    source_fingerprint = adapter.model_fingerprint()
    adapter.module.config.num_hidden_layers = 4
    with pytest.raises(ValueError, match="layer counts differ"):
        Gemma3FullMLPStackExecutor(
            adapter,
            _full_replacements(adapter),
        )
    assert adapter.model_fingerprint() == source_fingerprint


def test_forward_uses_generated_condition() -> None:
    adapter, executor = _fixture()
    result = executor(_batch().model_inputs)
    assert result.condition == "generated"
    assert result.replacement_scope == "full_native_mlp_stack_replacement"
