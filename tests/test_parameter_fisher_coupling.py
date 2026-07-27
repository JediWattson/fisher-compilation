from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib

import pytest
import torch

from fisher_graph.parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPLayerParameterSpec,
    NaturalMLPParameterGroupCatalog,
    build_grouped_virtual_gate_fisher,
    build_natural_mlp_parameter_group_catalog,
)
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _layer(
    ordinal: int,
    *,
    width: int,
    input_width: int,
    output_width: int,
) -> NaturalMLPLayerParameterSpec:
    activation = CrossBlockLayerSpec(
        layer_id=f"model.layers.{ordinal}",
        layer_ordinal=ordinal,
        activation_site=f"model.layers.{ordinal}.mlp.gated",
        width=width,
    )
    return NaturalMLPLayerParameterSpec.from_cross_block_layer_spec(
        activation,
        input_width=input_width,
        output_width=output_width,
        parameter_prefix=f"model.layers.{ordinal}.mlp",
    )


def _catalog() -> NaturalMLPParameterGroupCatalog:
    # The builder canonicalizes caller order.
    return build_natural_mlp_parameter_group_catalog(
        model_fingerprint=_digest("model"),
        layer_specs=(
            _layer(1, width=2, input_width=3, output_width=3),
            _layer(0, width=3, input_width=4, output_width=5),
        ),
    )


def _fisher(
    scores: torch.Tensor,
    *,
    normalization: str = "mean_over_prompts",
) -> GroupedVirtualGateFisher:
    return build_grouped_virtual_gate_fisher(
        scores,
        catalog=_catalog(),
        calibration_split_sha256=_digest("calibration"),
        objective_sha256=_digest("causal-nll"),
        normalization=normalization,
    )


def test_natural_group_catalog_maps_exact_rows_columns_and_counts() -> None:
    catalog = _catalog()

    assert [value.layer_ordinal for value in catalog.layer_specs] == [0, 1]
    assert catalog.group_count == 5
    assert catalog.total_parameter_count == 3 * (4 + 4 + 5) + 2 * (
        3 + 3 + 3
    )
    assert not catalog.contains_model_weights

    first = catalog.groups[0]
    assert first.group_index == 0
    assert first.key.layer_ordinal == 0
    assert first.key.channel_index == 0
    assert first.gate_proj.axis == 0
    assert first.gate_proj.matrix_shape == (3, 4)
    assert first.gate_proj.parameter_count == 4
    assert first.up_proj.axis == 0
    assert first.up_proj.parameter_count == 4
    assert first.down_proj.axis == 1
    assert first.down_proj.matrix_shape == (5, 3)
    assert first.down_proj.parameter_count == 5
    assert first.parameter_count == 13

    third = catalog.groups[2]
    assert third.gate_proj.index == 2
    assert third.up_proj.index == 2
    assert third.down_proj.index == 2
    mode = third.key.as_mode_key(fisher_rank=7)
    assert mode.mode_index == 2
    assert mode.fisher_rank == 7
    assert catalog.layer_specs[0].cross_block_layer_spec.width == 3

    second_layer = catalog.groups[3]
    assert second_layer.group_index == 3
    assert second_layer.key.layer_ordinal == 1
    assert second_layer.parameter_count == 9


@pytest.mark.parametrize(
    ("normalization", "divisor"),
    (("sum_over_prompts", 1.0), ("mean_over_prompts", 4.0)),
)
def test_implicit_couplings_equal_explicit_score_gram(
    normalization: str,
    divisor: float,
) -> None:
    scores = torch.tensor(
        [
            [1.0, -2.0, 0.0, 0.5, 2.0],
            [0.0, 1.0, 3.0, -1.0, 0.5],
            [2.0, 0.0, -1.0, 1.0, -0.5],
            [-1.0, 1.5, 0.25, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fisher = _fisher(scores, normalization=normalization)
    explicit = scores.T @ scores / divisor

    torch.testing.assert_close(
        fisher.fisher_mass,
        torch.diag(explicit),
        rtol=0,
        atol=0,
    )
    assert fisher.normalization_divisor == divisor
    assert fisher.rank_upper_bound == 4
    assert not fisher.contains_dense_group_fisher
    assert not fisher.contains_model_weights
    assert not fisher.contains_raw_prompts

    all_groups = tuple(range(scores.shape[1]))
    for chunk_size in (1, 2, 7):
        actual = fisher.coupling_block(
            all_groups,
            all_groups,
            chunk_size=chunk_size,
        )
        torch.testing.assert_close(actual, explicit, rtol=1e-14, atol=1e-14)

    pairs = fisher.coupling_pairs(
        (0, 0, 2, 4),
        (1, 4, 3, 4),
        chunk_size=1,
    )
    expected = explicit[(0, 0, 2, 4), (1, 4, 3, 4)]
    torch.testing.assert_close(pairs, expected, rtol=1e-14, atol=1e-14)


def test_fisher_ranked_mode_catalog_preserves_score_column_order() -> None:
    scores = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, 3.0],
            [0.0, 0.5, 0.0, 4.0, 1.0],
        ],
        dtype=torch.float64,
    )
    fisher = _fisher(scores, normalization="sum_over_prompts")

    modes = fisher.fisher_ranked_mode_catalog()

    assert tuple(mode.layer_ordinal for mode in modes) == (0, 0, 0, 1, 1)
    assert tuple(mode.mode_index for mode in modes) == (0, 1, 2, 0, 1)
    assert tuple(mode.fisher_rank for mode in modes[:3]) == (1, 2, 0)
    assert tuple(mode.fisher_rank for mode in modes[3:]) == (0, 1)


def test_signed_and_absolute_queries_keep_coupling_sign_distinct() -> None:
    scores = torch.tensor(
        [
            [1.0, -2.0, 0.5, 0.0, 0.0],
            [2.0, -4.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, 0.25, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fisher = _fisher(scores, normalization="sum_over_prompts")

    assert fisher.coupling(0, 1) == -12.0
    assert fisher.coupling(0, 1, absolute=True) == 12.0
    signed = fisher.coupling_block((0, 1), (0, 1), chunk_size=1)
    absolute = fisher.coupling_block(
        (0, 1),
        (0, 1),
        absolute=True,
        chunk_size=2,
    )
    assert signed[0, 1].item() == -12.0
    torch.testing.assert_close(absolute, signed.abs(), rtol=0, atol=0)


def test_top_k_is_deterministic_chunk_invariant_and_stably_tied() -> None:
    scores = torch.tensor(
        [
            [3.0, 2.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    fisher = _fisher(scores, normalization="sum_over_prompts")

    one = fisher.top_k_edges(4, block_size=1)
    two = fisher.top_k_edges(4, block_size=2)
    all_at_once = fisher.top_k_edges(4, block_size=20)

    assert one == two == all_at_once
    assert [
        (edge.first_group_index, edge.second_group_index)
        for edge in one
    ] == [(0, 1), (0, 3), (1, 3), (0, 2)]
    assert [
        (edge.ranking_strength, edge.first_group_index, edge.second_group_index)
        for edge in one
    ] == [
        (6.0, 0, 1),
        (6.0, 0, 3),
        (4.0, 1, 3),
        (3.0, 0, 2),
    ]


def test_layer_pair_filters_and_zero_mass_groups() -> None:
    scores = torch.tensor(
        [
            [3.0, 2.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    fisher = _fisher(scores, normalization="sum_over_prompts")

    assert fisher.fisher_mass[-1].item() == 0.0
    assert fisher.coupling(0, 4) == 0.0
    assert all(
        edge.first_group_index != 4 and edge.second_group_index != 4
        for edge in fisher.top_k_edges(20)
    )

    same = fisher.top_k_edges(20, layer_policy="same_layer")
    cross = fisher.top_k_edges(20, layer_policy="cross_layer")
    assert all(
        edge.first.layer_ordinal == edge.second.layer_ordinal
        for edge in same
    )
    assert all(
        edge.first.layer_ordinal != edge.second.layer_ordinal
        for edge in cross
    )
    assert {
        (edge.first_group_index, edge.second_group_index)
        for edge in cross
    } == {(0, 3), (1, 3), (2, 3)}


def test_block_iterator_matches_explicit_upper_blocks() -> None:
    scores = torch.arange(1, 16, dtype=torch.float64).reshape(3, 5)
    fisher = _fisher(scores, normalization="mean_over_prompts")
    explicit = scores.T @ scores / 3.0

    yielded = list(
        fisher.iter_coupling_blocks(
            block_size=2,
            upper_triangle_only=True,
        )
    )
    assert len(yielded) == 6
    for rows, columns, block in yielded:
        torch.testing.assert_close(
            block,
            explicit[list(rows)][:, list(columns)],
            rtol=1e-14,
            atol=1e-14,
        )


def test_artifacts_round_trip_and_reject_hash_poisoning() -> None:
    scores = torch.arange(1, 16, dtype=torch.float64).reshape(3, 5)
    fisher = _fisher(scores)

    catalog_state = fisher.catalog.state_dict()
    restored_catalog = NaturalMLPParameterGroupCatalog.from_state_dict(
        catalog_state
    )
    assert restored_catalog.artifact_sha256 == fisher.catalog.artifact_sha256

    state = fisher.state_dict()
    restored = GroupedVirtualGateFisher.from_state_dict(state)
    assert restored.artifact_sha256 == fisher.artifact_sha256
    torch.testing.assert_close(restored.score_factor, scores)

    poisoned_factor = copy.deepcopy(state)
    poisoned_factor["score_factor"][0, 0] += 1.0
    with pytest.raises(ValueError, match="score-factor hash mismatch"):
        GroupedVirtualGateFisher.from_state_dict(poisoned_factor)

    poisoned_mass = copy.deepcopy(state)
    poisoned_mass["fisher_mass"][0] += 1.0
    with pytest.raises(ValueError, match="mass hash mismatch"):
        GroupedVirtualGateFisher.from_state_dict(poisoned_mass)

    poisoned_catalog = copy.deepcopy(catalog_state)
    poisoned_catalog["layer_specs"][0]["input_width"] += 1
    with pytest.raises(ValueError):
        NaturalMLPParameterGroupCatalog.from_state_dict(poisoned_catalog)


def test_serialized_boundary_contains_no_weights_prompts_or_dense_fisher() -> None:
    scores = torch.arange(1, 16, dtype=torch.float64).reshape(3, 5)
    fisher = _fisher(scores)
    catalog_state = fisher.catalog.state_dict()
    fisher_state = fisher.state_dict()

    def tensors(value: object) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [value]
        if isinstance(value, Mapping):
            return [
                item
                for child in value.values()
                for item in tensors(child)
            ]
        if isinstance(value, (tuple, list)):
            return [item for child in value for item in tensors(child)]
        return []

    assert tensors(catalog_state) == []
    assert {
        tuple(value.shape) for value in tensors(fisher_state)
    } == {(3, 5), (5,)}
    assert (5, 5) not in {
        tuple(value.shape) for value in tensors(fisher_state)
    }

    metadata = fisher.metadata()
    assert metadata["contains_dense_group_fisher"] is False
    assert metadata["contains_model_weights"] is False
    assert metadata["contains_raw_prompts"] is False
    serialized_keys = repr(tuple(fisher_state.keys())).lower()
    assert "prompt_text" not in serialized_keys
    assert "token_id" not in serialized_keys
