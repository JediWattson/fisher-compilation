from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_full_mlp_stack_rows import (
    FullMLPStackLayerRows,
    collect_full_mlp_stack_layer_rows,
    collect_full_mlp_stack_rows,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from test_gemma3_modal_generator_executor import _adapter
from test_gemma3_modal_generator_graph_executor import (
    _fragment_plan,
    _layer_sites,
)


DTYPE = torch.float64


def _fixture():
    adapter = _adapter(seed=73_001)
    plan = _fragment_plan(adapter, full_layer=True)
    fragments = plan.fragments
    source_mlp = adapter.module.model.layers[0].mlp
    return fragments, source_mlp.down_proj.weight.detach().clone()


def _multilayer_fixture():
    adapter = _adapter(seed=73_003)
    layer_zero = _fragment_plan(adapter, full_layer=True).fragments
    input_site, output_site = _layer_sites(adapter, 1)
    transformer = adapter.layers[1].transformer
    assert transformer is not None
    assert transformer.operator_sites is not None
    layer_one = tuple(
        replace(
            fragment,
            layer_ordinal=1,
            layer_id=adapter.layers[1].id,
            activation_site=(
                transformer.operator_sites.feed_forward_down_input
            ),
            input_site=input_site,
            output_site=output_site,
            group_indices=tuple(
                value + 6 for value in fragment.group_indices
            ),
            fisher_ranks=tuple(
                value + 6 for value in fragment.fisher_ranks
            ),
            artifact_sha256="",
        )
        for fragment in layer_zero
    )
    fragments = {1: tuple(reversed(layer_one)), 0: layer_zero}
    down = {
        ordinal: (
            adapter.module.model.layers[ordinal]
            .mlp.down_proj.weight.detach()
            .clone()
        )
        for ordinal in fragments
    }
    return fragments, down


def _multilayer_row(
    fragments_by_layer,
    *,
    example_id: str,
    positions: tuple[int, ...],
):
    observations = len(positions)
    activations = {}
    gradients = {}
    expected = {}
    for ordinal, fragments in fragments_by_layer.items():
        first = fragments[0]
        x = (
            torch.arange(
                observations * first.input_width,
                dtype=DTYPE,
            ).reshape(observations, first.input_width)
            + 10.0 * ordinal
        )
        z = (
            torch.arange(
                observations * 6,
                dtype=DTYPE,
            ).reshape(observations, 6)
            / float(ordinal + 2)
            + 20.0 * ordinal
        )
        gradient = torch.linspace(
            0.1 + ordinal,
            1.0 + ordinal,
            observations * 6,
            dtype=DTYPE,
        ).reshape(observations, 6)
        activations[first.input_site] = x
        activations[first.activation_site] = z
        gradients[first.input_site] = torch.zeros_like(x)
        gradients[first.activation_site] = gradient
        expected[ordinal] = (x, z, gradient)
    return (
        ActivationScoreGradientRows(
            activations=activations,
            score_gradients=gradients,
            logical_positions=torch.tensor(positions, dtype=torch.int64),
            loss=1.0,
            example_id=example_id,
        ),
        expected,
    )


def _row(
    fragments,
    *,
    example_id: str = "example.a",
    positions: tuple[int, ...] = (2, 5),
    offset: float = 0.0,
) -> tuple[ActivationScoreGradientRows, torch.Tensor, torch.Tensor]:
    first = fragments[0]
    observations = len(positions)
    x = (
        torch.arange(
            observations * first.input_width,
            dtype=DTYPE,
        ).reshape(observations, first.input_width)
        + offset
    )
    z = (
        torch.arange(
            observations * 6,
            dtype=DTYPE,
        ).reshape(observations, 6)
        / 7.0
        + offset
    )
    z_gradient = torch.linspace(
        0.2,
        1.1,
        observations * 6,
        dtype=DTYPE,
    ).reshape(observations, 6)
    value = ActivationScoreGradientRows(
        activations={
            first.input_site: x,
            first.activation_site: z,
        },
        score_gradients={
            first.input_site: torch.zeros_like(x),
            first.activation_site: z_gradient,
        },
        logical_positions=torch.tensor(positions, dtype=torch.int64),
        loss=1.0,
        example_id=example_id,
    )
    return value, z, z_gradient


def test_collects_one_exact_full_layer_table_with_stable_keys() -> None:
    fragments, down = _fixture()
    first, first_z, first_gradient = _row(fragments)
    second, second_z, second_gradient = _row(
        fragments,
        example_id="example.b",
        positions=(1,),
        offset=3.0,
    )

    collected = collect_full_mlp_stack_layer_rows(
        (first, second),
        fragments=tuple(reversed(fragments)),
        down_projection_weight=down,
    )

    assert isinstance(collected, FullMLPStackLayerRows)
    assert collected.layer_ordinal == 0
    assert collected.intermediate_width == 6
    assert collected.observations == 3
    assert collected.sequences == 2
    assert collected.row_keys == (
        ("example.a", 2),
        ("example.a", 5),
        ("example.b", 1),
    )
    assert len(collected.row_key_sha256) == 64
    assert collected.fragment_ids == tuple(
        fragment.fragment_id for fragment in fragments
    )
    expected_z = torch.cat((first_z, second_z), dim=0)
    expected_gradient = torch.cat(
        (first_gradient, second_gradient),
        dim=0,
    )
    torch.testing.assert_close(
        collected.inputs,
        torch.cat(
            (
                first.activations[fragments[0].input_site],
                second.activations[fragments[0].input_site],
            ),
            dim=0,
        ),
    )
    torch.testing.assert_close(
        collected.contributions,
        expected_z @ down.to(dtype=DTYPE).T,
    )
    torch.testing.assert_close(
        collected.fisher_weights,
        (expected_z * expected_gradient).square().sum(dim=1),
    )
    assert not hasattr(collected, "rows_by_fragment")

    repeated = collect_full_mlp_stack_layer_rows(
        (first, second),
        fragments=fragments,
        down_projection_weight=down,
    )
    assert repeated.row_key_sha256 == collected.row_key_sha256
    assert repeated.fragment_sha256s == collected.fragment_sha256s


def test_rejects_missing_overlapping_and_mixed_layer_fragments() -> None:
    fragments, down = _fixture()
    row, _, _ = _row(fragments)

    with pytest.raises(ValueError, match="do not exhaust"):
        collect_full_mlp_stack_layer_rows(
            (row,),
            fragments=(fragments[0],),
            down_projection_weight=down,
        )

    overlapping = replace(
        fragments[1],
        channel_indices=(2, 4, 5),
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="channels overlap"):
        collect_full_mlp_stack_layer_rows(
            (row,),
            fragments=(fragments[0], overlapping),
            down_projection_weight=down,
        )

    adapter = _adapter(seed=73_002)
    cross_layer = _fragment_plan(adapter, full_layer=False).fragments
    with pytest.raises(ValueError, match="one common MLP layer"):
        collect_full_mlp_stack_layer_rows(
            (row,),
            fragments=(cross_layer[0], cross_layer[2]),
            down_projection_weight=down,
        )


def test_rejects_unstable_keys_wrong_sites_and_shape_drift() -> None:
    fragments, down = _fixture()
    row, _, _ = _row(fragments)
    without_id = replace(row, example_id=None)
    with pytest.raises(ValueError, match="stable example ids"):
        collect_full_mlp_stack_layer_rows(
            (without_id,),
            fragments=fragments,
            down_projection_weight=down,
        )

    with pytest.raises(ValueError, match="keys are duplicated"):
        collect_full_mlp_stack_layer_rows(
            (row, row),
            fragments=fragments,
            down_projection_weight=down,
        )

    extra_site = ActivationScoreGradientRows(
        activations={
            **row.activations,
            "unexpected.site": torch.ones((2, 1), dtype=DTYPE),
        },
        score_gradients={
            **row.score_gradients,
            "unexpected.site": torch.ones((2, 1), dtype=DTYPE),
        },
        logical_positions=row.logical_positions,
        loss=row.loss,
        example_id=row.example_id,
    )
    with pytest.raises(ValueError, match="exactly the input"):
        collect_full_mlp_stack_layer_rows(
            (extra_site,),
            fragments=fragments,
            down_projection_weight=down,
        )

    with pytest.raises(ValueError, match="output width"):
        collect_full_mlp_stack_layer_rows(
            (row,),
            fragments=fragments,
            down_projection_weight=torch.ones((5, 6), dtype=DTYPE),
        )


def test_closes_row_stream_after_success_and_failure() -> None:
    fragments, down = _fixture()
    row, _, _ = _row(fragments)
    closed: list[str] = []

    def successful_stream():
        try:
            yield row
        finally:
            closed.append("success")

    collect_full_mlp_stack_layer_rows(
        successful_stream(),
        fragments=fragments,
        down_projection_weight=down,
    )
    assert closed == ["success"]

    def failing_stream():
        try:
            yield replace(row, example_id=None)
        finally:
            closed.append("failure")

    with pytest.raises(ValueError, match="stable example ids"):
        collect_full_mlp_stack_layer_rows(
            failing_stream(),
            fragments=fragments,
            down_projection_weight=down,
        )
    assert closed == ["success", "failure"]


def test_one_pass_multilayer_collection_orders_layers_and_shares_keys() -> None:
    fragments, down = _multilayer_fixture()
    row, expected = _multilayer_row(
        fragments,
        example_id="stack.a",
        positions=(0, 3, 8),
    )
    iterations: list[int] = []

    def one_pass_stream():
        iterations.append(1)
        yield row

    collected = collect_full_mlp_stack_rows(
        one_pass_stream(),
        fragments_by_layer=fragments,
        down_projection_weights=down,
    )

    assert iterations == [1]
    assert tuple(value.layer_ordinal for value in collected) == (0, 1)
    assert collected[0].row_keys is collected[1].row_keys
    assert collected[0].row_key_sha256 == collected[1].row_key_sha256
    assert collected[0].row_keys == (
        ("stack.a", 0),
        ("stack.a", 3),
        ("stack.a", 8),
    )
    for layer in collected:
        x, z, gradient = expected[layer.layer_ordinal]
        torch.testing.assert_close(layer.inputs, x)
        torch.testing.assert_close(
            layer.contributions,
            z @ down[layer.layer_ordinal].to(dtype=DTYPE).T,
        )
        torch.testing.assert_close(
            layer.fisher_weights,
            (z * gradient).square().sum(dim=1),
        )
        assert not hasattr(layer, "rows_by_fragment")


def test_multilayer_collection_requires_exact_stack_bindings_and_sites() -> None:
    fragments, down = _multilayer_fixture()
    row, _ = _multilayer_row(
        fragments,
        example_id="stack.a",
        positions=(0, 1),
    )

    with pytest.raises(ValueError, match="exactly cover declared layers"):
        collect_full_mlp_stack_rows(
            (row,),
            fragments_by_layer=fragments,
            down_projection_weights={0: down[0]},
        )

    wrong_key = {0: fragments[0], 2: fragments[1]}
    with pytest.raises(ValueError, match="key differs"):
        collect_full_mlp_stack_rows(
            (row,),
            fragments_by_layer=wrong_key,
            down_projection_weights={0: down[0], 2: down[1]},
        )

    first_layer_one = fragments[1][0]
    duplicate_group = replace(
        first_layer_one,
        group_indices=fragments[0][0].group_indices,
        artifact_sha256="",
    )
    group_overlap = {
        0: fragments[0],
        1: (duplicate_group, *fragments[1][1:]),
    }
    with pytest.raises(ValueError, match="groups overlap across layers"):
        collect_full_mlp_stack_rows(
            (row,),
            fragments_by_layer=group_overlap,
            down_projection_weights=down,
        )

    missing_site = row.activations.copy()
    missing_gradient = row.score_gradients.copy()
    removed_site = fragments[1][0].activation_site
    del missing_site[removed_site]
    del missing_gradient[removed_site]
    incomplete = ActivationScoreGradientRows(
        activations=missing_site,
        score_gradients=missing_gradient,
        logical_positions=row.logical_positions,
        loss=row.loss,
        example_id=row.example_id,
    )
    with pytest.raises(ValueError, match="exactly the input"):
        collect_full_mlp_stack_rows(
            (incomplete,),
            fragments_by_layer=fragments,
            down_projection_weights=down,
        )
