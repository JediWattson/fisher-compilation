from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_parts_first_atom_capture import (
    PartsFirstShape,
    bind_parts_first_fixture_surfaces,
    bind_parts_first_gemma_adapter,
    build_parts_first_atom_response_rows,
    iter_parts_first_atom_response_batches,
    parts_first_capture_site_ids,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows


LAYERS = 18
HIDDEN = 4
VOCABULARY = 7
HEADS = 2
HEAD_DIMENSION = 2
MLP_WIDTH = 3
OBSERVATIONS = 3


def _fixture():
    shape = PartsFirstShape(
        layer_count=LAYERS,
        hidden_width=HIDDEN,
        vocabulary_size=VOCABULARY,
        query_head_count=HEADS,
        head_dimension=HEAD_DIMENSION,
        mlp_width=MLP_WIDTH,
    )
    generator = torch.Generator().manual_seed(20_260_811)
    tied = torch.randn(VOCABULARY, HIDDEN, generator=generator)
    attention_query = tuple(
        torch.randn(HEADS * HEAD_DIMENSION, HIDDEN, generator=generator)
        for _ in range(LAYERS)
    )
    attention = tuple(
        torch.randn(HIDDEN, HEADS * HEAD_DIMENSION, generator=generator)
        for _ in range(LAYERS)
    )
    mlp = tuple(
        torch.randn(HIDDEN, MLP_WIDTH, generator=generator)
        for _ in range(LAYERS)
    )
    mlp_gate = tuple(
        torch.randn(MLP_WIDTH, HIDDEN, generator=generator)
        for _ in range(LAYERS)
    )
    mlp_up = tuple(
        torch.randn(MLP_WIDTH, HIDDEN, generator=generator)
        for _ in range(LAYERS)
    )
    binding = bind_parts_first_fixture_surfaces(
        shape=shape,
        tied_weight=tied,
        embed_scale=2.0,
        attention_query_weights=attention_query,
        attention_output_weights=attention,
        mlp_gate_weights=mlp_gate,
        mlp_up_weights=mlp_up,
        mlp_down_weights=mlp,
        source_model_fingerprint="2" * 64,
        execution_fingerprint="3" * 64,
    )
    sites = parts_first_capture_site_ids(binding)
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def random(rows: int, width: int) -> torch.Tensor:
        return torch.randn(
            rows,
            width,
            dtype=torch.float64,
            generator=generator,
        )

    activations[sites[0]] = random(OBSERVATIONS, HIDDEN)
    gradients[sites[0]] = random(OBSERVATIONS, HIDDEN)
    cursor = 1
    for layer in range(LAYERS):
        normalized_attention = sites[cursor]
        attention_context = sites[cursor + 1]
        attention_output = sites[cursor + 2]
        normalized_mlp = sites[cursor + 3]
        mlp_down_input = sites[cursor + 4]
        mlp_output = sites[cursor + 5]
        cursor += 6
        activations[normalized_attention] = random(OBSERVATIONS, HIDDEN)
        gradients[normalized_attention] = random(OBSERVATIONS, HIDDEN)
        context = random(OBSERVATIONS, HEADS * HEAD_DIMENSION)
        activations[attention_context] = context
        gradients[attention_context] = random(
            OBSERVATIONS,
            HEADS * HEAD_DIMENSION,
        )
        activations[attention_output] = context @ attention[layer].double().T
        gradients[attention_output] = random(OBSERVATIONS, HIDDEN)
        activations[normalized_mlp] = random(OBSERVATIONS, HIDDEN)
        gradients[normalized_mlp] = random(OBSERVATIONS, HIDDEN)
        down_input = random(OBSERVATIONS, MLP_WIDTH)
        activations[mlp_down_input] = down_input
        gradients[mlp_down_input] = random(OBSERVATIONS, MLP_WIDTH)
        activations[mlp_output] = down_input @ mlp[layer].double().T
        gradients[mlp_output] = random(OBSERVATIONS, HIDDEN)
    assert sites[cursor] == "final_norm"
    activations["final_norm"] = random(OBSERVATIONS, HIDDEN)
    gradients["final_norm"] = random(OBSERVATIONS, HIDDEN)
    rows = ActivationScoreGradientRows(
        activations=activations,
        score_gradients=gradients,
        logical_positions=torch.tensor([1, 2, 5], dtype=torch.int64),
        loss=4.25,
        example_id="example.fixture.1",
    )
    return binding, rows


def test_model_free_rows_emit_all_part_families_in_execution_order() -> None:
    binding, rows = _fixture()
    batch = build_parts_first_atom_response_rows(binding, rows)

    assert len(parts_first_capture_site_ids(binding)) == 110
    assert len(batch.blocks) == 38
    assert tuple(block.event_index for block in batch.blocks) == (
        0,
        *range(1, 37),
        38,
    )
    assert batch.blocks[0].family_id == "tied_lexical_hidden_column"
    assert batch.blocks[-1].family_id == "tied_lexical_hidden_column"
    assert tuple(block.family_id for block in batch.blocks[1:-1:2]) == (
        "attention_qo_head",
    ) * LAYERS
    assert tuple(block.family_id for block in batch.blocks[2:-1:2]) == (
        "mlp_channel",
    ) * LAYERS
    assert batch.blocks[0].part_responses.shape == (OBSERVATIONS, HIDDEN)
    assert batch.blocks[1].part_responses.shape == (OBSERVATIONS, HEADS)
    assert batch.blocks[2].part_responses.shape == (OBSERVATIONS, MLP_WIDTH)
    assert batch.blocks[-1].part_responses.shape == (OBSERVATIONS, HIDDEN)
    assert all(block.receipt.operator_closure_passed for block in batch.blocks)
    assert all(block.receipt.response_closure_passed for block in batch.blocks)
    assert [row.logical_position for row in batch.row_keys] == [1, 2, 5]
    assert all(row.example_id == rows.example_id for row in batch.row_keys)


def test_attention_and_mlp_responses_equal_explicit_native_formulas() -> None:
    binding, rows = _fixture()
    batch = build_parts_first_atom_response_rows(binding, rows)
    attention = batch.blocks[1]
    mlp = batch.blocks[2]
    sites = parts_first_capture_site_ids(binding)

    attention_context = rows.activations[sites[2]].reshape(
        OBSERVATIONS,
        HEADS,
        HEAD_DIMENSION,
    )
    output_slices = binding.attention_output_weights[0].double().reshape(
        HIDDEN,
        HEADS,
        HEAD_DIMENSION,
    ).permute(1, 2, 0)
    head_outputs = torch.einsum(
        "rhd,hdo->rho",
        attention_context,
        output_slices,
    )
    expected_attention = (
        head_outputs * rows.score_gradients[sites[3]].unsqueeze(1)
    ).sum(dim=-1)
    torch.testing.assert_close(attention.part_responses, expected_attention)

    down_input = rows.activations[sites[5]]
    down_gradient = rows.score_gradients[sites[6]]
    expected_mlp = down_input * (
        down_gradient @ binding.mlp_down_weights[0].double()
    )
    torch.testing.assert_close(mlp.part_responses, expected_mlp)


def test_float32_head_layout_closes_when_response_contraction_cancels() -> None:
    binding, rows = _fixture()
    sites = parts_first_capture_site_ids(binding)
    generator = torch.Generator().manual_seed(9_876)
    context32 = torch.randn(
        OBSERVATIONS,
        HEADS * HEAD_DIMENSION,
        dtype=torch.float32,
        generator=generator,
    )
    weight32 = binding.attention_output_weights[0]
    native_output32 = torch.nn.functional.linear(context32, weight32)

    # Streaming promotes the captured float32 values to float64.  Choose an
    # otherwise ordinary score gradient orthogonal to the native output, so
    # the reference contraction cancels while the float64 reconstruction
    # retains the checkpoint's expected sub-micro numerical discrepancy.
    formula64 = context32.double() @ weight32.double().T
    reconstruction_error = formula64 - native_output32.double()
    cancellation_gradients = []
    for native_row, error_row in zip(
        native_output32.double(),
        reconstruction_error,
        strict=True,
    ):
        orthogonal = error_row - (
            (error_row @ native_row) / (native_row @ native_row)
        ) * native_row
        cancellation_gradients.append(orthogonal / orthogonal.norm())
    attention_gradient = torch.stack(cancellation_gradients)

    activations = dict(rows.activations)
    gradients = dict(rows.score_gradients)
    activations[sites[2]] = context32.double()
    activations[sites[3]] = native_output32.double()
    gradients[sites[3]] = attention_gradient
    cancellation_rows = ActivationScoreGradientRows(
        activations=activations,
        score_gradients=gradients,
        logical_positions=rows.logical_positions,
        loss=rows.loss,
        example_id="example.float32-attention-cancellation.1",
    )

    batch = build_parts_first_atom_response_rows(binding, cancellation_rows)
    receipt = batch.blocks[1].receipt
    assert receipt.operator_max_abs_error <= receipt.operator_max_abs_limit
    assert receipt.response_max_abs_error <= receipt.response_max_abs_limit
    assert receipt.response_nrmse > 1.0e-5
    assert receipt.operator_closure_passed
    assert receipt.response_closure_passed


def test_tied_columns_recur_while_body_ownership_is_disjoint() -> None:
    binding, rows = _fixture()
    batch = build_parts_first_atom_response_rows(binding, rows)
    ingress = batch.blocks[0]
    egress = batch.blocks[-1]
    assert ingress.physical_atom_ordinals == egress.physical_atom_ordinals == (
        0,
        1,
        2,
        3,
    )
    body_sets = [set(block.physical_atom_ordinals) for block in batch.blocks[1:-1]]
    assert not any(set(ingress.physical_atom_ordinals) & values for values in body_sets)
    assert len(set().union(*body_sets)) == LAYERS * HEADS + LAYERS * MLP_WIDTH
    assert max(set().union(*body_sets)) == binding.shape.physical_atom_count - 1


def test_response_mutation_and_site_reordering_are_rejected() -> None:
    binding, rows = _fixture()
    batch = build_parts_first_atom_response_rows(binding, rows)
    block = batch.blocks[2]
    changed = block.part_responses.clone()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="closure receipt"):
        replace(block, part_responses=changed)

    reversed_rows = ActivationScoreGradientRows(
        activations=dict(reversed(tuple(rows.activations.items()))),
        score_gradients=dict(reversed(tuple(rows.score_gradients.items()))),
        logical_positions=rows.logical_positions,
        loss=rows.loss,
        example_id=rows.example_id,
    )
    with pytest.raises(ValueError, match="inventory/order"):
        build_parts_first_atom_response_rows(binding, reversed_rows)


def test_fixture_binding_cannot_enter_the_live_model_iterator() -> None:
    binding, _rows = _fixture()
    with pytest.raises(ValueError, match="live capture"):
        iter_parts_first_atom_response_batches(
            binding,
            (),
            score_objective=lambda _run, _batch: torch.tensor(0.0),
        )


def test_production_factory_requires_the_real_adapter_type() -> None:
    with pytest.raises(TypeError, match="Gemma3CausalLMAdapter"):
        bind_parts_first_gemma_adapter(object())  # type: ignore[arg-type]


def test_fixture_binding_fails_closed_on_projection_geometry() -> None:
    binding, _rows = _fixture()
    with pytest.raises(ValueError, match="attention output weight geometry"):
        bind_parts_first_fixture_surfaces(
            shape=binding.shape,
            tied_weight=binding.tied_weight,
            embed_scale=binding.embed_scale,
            attention_query_weights=binding.attention_query_weights,
            attention_output_weights=(
                torch.ones(HIDDEN, 1),
                *binding.attention_output_weights[1:],
            ),
            mlp_gate_weights=binding.mlp_gate_weights,
            mlp_up_weights=binding.mlp_up_weights,
            mlp_down_weights=binding.mlp_down_weights,
        )
