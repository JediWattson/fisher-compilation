"""Qwen execution and Fisher-capture regressions using real, tiny HF models."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

from fisher_graph.adapters import Qwen3CausalLMAdapter
from fisher_graph.gemma3_incremental_decode import JointGraphDecodeSession
from fisher_graph.qwen3_joint_compiler import compile_qwen3_joint_exact_control
from fisher_graph.qwen3_joint_executor import (
    Qwen3JointGraphAttentionOperatorOutputSnapshot,
    Qwen3JointGraphScoreGradientBundle,
    build_qwen3_event_schedule,
)


@pytest.fixture(scope="module", params=[2, 28], ids=["two_layers", "28_layers"])
def compiled_qwen(request):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            config = transformers.Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=request.param,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=64,
                tie_word_embeddings=True,
                attention_dropout=0.0,
            )
            config._attn_implementation = "eager"
            model = transformers.Qwen3ForCausalLM(config).eval().requires_grad_(False)
        adapter = Qwen3CausalLMAdapter(model)
        executor = compile_qwen3_joint_exact_control(adapter)
        yield model, adapter, executor, request.param
    finally:
        torch.set_num_threads(previous_threads)


def _batch():
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _loss(logits):
    labels = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 6]])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def test_exact_forward_owns_weights_and_reports_qwen_topology(compiled_qwen):
    model, adapter, executor, layers = compiled_qwen
    batch = _batch()
    sequence = adapter.prepare_sequence(batch)
    with torch.inference_mode():
        native = model(**batch).logits
        execution = executor.execute(batch["input_ids"], sequence)
    torch.testing.assert_close(execution.logits, native, rtol=0, atol=0)
    assert execution.layer_count == layers
    assert execution.carrier_version == 2 * layers
    assert execution.event_ids == tuple(
        event.event_id for event in build_qwen3_event_schedule(layers)
    )
    source_storages = {
        value.untyped_storage().data_ptr()
        for value in (*model.parameters(), *model.buffers())
    }
    compiled_storages = {
        value.untyped_storage().data_ptr()
        for value in (*executor.parameters(), *executor.buffers())
    }
    assert not source_storages.intersection(compiled_storages)
    manifest = executor.architecture_manifest()
    assert manifest["kind"] == "qwen3_joint_cross_block_chart_executor"
    assert manifest["exact_control_arm"] is True
    assert manifest["candidate_arm"] is False
    assert manifest["layer_count"] == layers
    assert manifest["event_count"] == 2 * layers + 3
    assert manifest["global_attention_block_ordinals"] == tuple(range(layers))
    assert manifest["resource_inventory"]["inventory_closes"] is True

    with pytest.raises(ValueError, match="event inventory"):
        replace(execution, event_ids=tuple(reversed(execution.event_ids)))
    with pytest.raises(ValueError, match="event inventory"):
        replace(execution, layer_count=layers + 1)


def test_auxiliary_gradients_match_native_input_and_terminal_pullback(
    compiled_qwen, monkeypatch
):
    model, adapter, executor, layers = compiled_qwen
    batch = _batch()
    sequence = adapter.prepare_sequence(batch)
    native_input = model.model.embed_tokens(batch["input_ids"]).detach().requires_grad_()
    native_loss = _loss(
        model(inputs_embeds=native_input, attention_mask=batch["attention_mask"]).logits
    )
    native_input_gradient = torch.autograd.grad(native_loss, native_input)[0]

    full = executor.forward_components_with_observer(
        batch["input_ids"], sequence,
        observer=lambda **kwargs: None,
        retain_gradient_tape=True,
    )
    objective = _loss(full.execution.logits)
    terminal_gradient = torch.autograd.grad(
        objective, full.execution.normalized_output, retain_graph=True
    )[0]
    bundle = full.gradient_tape.score_gradients_with_auxiliary(objective)
    assert isinstance(bundle, Qwen3JointGraphScoreGradientBundle)
    assert bundle.layer_count == layers
    assert len(bundle.site_gradients) == 4 * layers + 1
    assert len(bundle.attention_operator_output_gradients) == layers
    torch.testing.assert_close(
        bundle.site_gradients["lexical.ingress.output"], native_input_gradient,
        rtol=1e-5, atol=1e-7,
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        full.gradient_tape.score_gradients_with_auxiliary(objective)

    def forbidden_egress(*args, **kwargs):
        raise AssertionError("logits-free execution called lexical egress")

    observed = {}
    with monkeypatch.context() as patch:
        patch.setattr(executor.lexical, "egress", forbidden_egress)
        backbone = executor.forward_backbone_with_observer(
            batch["input_ids"], sequence,
            observer=lambda event_id, role, values: observed.update(
                {f"{event_id}.{role}": values}
            ),
        )
    torch.testing.assert_close(
        backbone.execution.normalized_output, full.execution.normalized_output,
        rtol=0, atol=0,
    )
    assert backbone.execution.layer_count == layers
    snapshots = backbone.execution.attention_operator_output_snapshots
    assert len(snapshots) == layers
    assert snapshots[-1].block_ordinal == layers - 1
    for snapshot in snapshots:
        assert isinstance(snapshot, Qwen3JointGraphAttentionOperatorOutputSnapshot)
        assert snapshot.layer_count == layers
        assert not snapshot.values.requires_grad
        torch.testing.assert_close(
            snapshot.values, observed[f"{snapshot.event_id}.output"], rtol=0, atol=0
        )
    pulled = backbone.gradient_tape.score_gradients_from_terminal_normalized_output(
        terminal_gradient
    )
    for name, gradient in bundle.site_gradients.items():
        torch.testing.assert_close(pulled.site_gradients[name], gradient, rtol=0, atol=0)
    for name, gradient in bundle.auxiliary_gradients.items():
        torch.testing.assert_close(
            pulled.auxiliary_gradients[name], gradient, rtol=0, atol=0
        )
    with pytest.raises(RuntimeError, match="already consumed"):
        backbone.gradient_tape.score_gradients_from_terminal_normalized_output(
            terminal_gradient
        )
    with pytest.raises(ValueError, match="canonical score-gradient inventory"):
        replace(bundle, site_gradients={})
    with pytest.raises(ValueError, match="block_ordinal"):
        replace(snapshots[-1], block_ordinal=layers)


def test_ragged_incremental_decode_matches_native_teacher_forcing(compiled_qwen):
    model, adapter, executor, _ = compiled_qwen
    ids = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    sequence = adapter.prepare_sequence(
        {"input_ids": ids, "attention_mask": mask, "position_ids": positions}
    )
    with torch.inference_mode():
        session, logits = JointGraphDecodeSession.prefill(executor, ids, sequence)
        for _ in range(4):
            native = model(
                input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
            ).logits
            torch.testing.assert_close(logits[:, -1], native[:, -1], rtol=1e-5, atol=1e-7)
            next_ids = logits[:, -1].argmax(-1, keepdim=True)
            assert torch.equal(next_ids[:, 0], native[:, -1].argmax(-1))
            ids = torch.cat((ids, next_ids), dim=-1)
            mask = torch.cat((mask, torch.ones_like(next_ids)), dim=-1)
            logits = session.step(next_ids)

