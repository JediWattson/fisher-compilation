from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_parts_first_supermode_surfaces import (
    decompose_attention_head_groups,
    decompose_mlp_channel_groups,
    decompose_tied_lexical_egress_column_groups,
    decompose_tied_lexical_ingress_column_groups,
)


def test_mlp_channels_close_at_the_down_projection_boundary() -> None:
    down_input = torch.tensor(
        [[[1.0, -2.0, 3.0, 0.5], [0.0, 4.0, -1.0, 2.0]]],
        dtype=torch.float64,
    )
    weight = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 7.0
    native = down_input @ weight.T
    surface = decompose_mlp_channel_groups(
        down_input=down_input,
        down_projection_weight=weight,
        groups=((0, 2), (1, 3)),
        boundary_id="layer.7.mlp.operator_output",
        reference_output=native,
    )
    assert surface.additive_closure_passed
    torch.testing.assert_close(surface.reconstructed_output, native)
    gradient = torch.ones_like(native)
    coordinates = surface.score_coordinates(gradient)
    assert coordinates.shape == (1, 2, 2)
    torch.testing.assert_close(
        coordinates.sum(dim=-1), native.sum(dim=-1)
    )


def test_attention_query_heads_close_after_o_with_shared_values() -> None:
    probabilities = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.25, 0.75]],
                [[1.0, 0.0], [0.5, 0.5]],
            ]
        ],
        dtype=torch.float64,
    )
    # Both query heads may repeat the same KV substrate. Ownership remains in
    # their Q/O slices, not in this repeated value tensor.
    repeated_values = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ]
        ],
        dtype=torch.float64,
    )
    output_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 3.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    surface = decompose_attention_head_groups(
        probabilities=probabilities,
        repeated_values=repeated_values,
        output_projection_weight=output_weight,
        groups=((0,), (1,)),
        boundary_id="layer.4.attention.operator_output",
    )
    assert surface.additive_closure_passed
    expected = sum(surface.group_contributions)
    torch.testing.assert_close(surface.reference_output, expected)


def test_same_tied_columns_close_ingress_and_raw_egress() -> None:
    weight = torch.arange(30, dtype=torch.float64).reshape(6, 5) / 10.0
    groups = ((0, 2, 4), (1, 3))
    token_ids = torch.tensor([[0, 5], [2, 1]])
    ingress = decompose_tied_lexical_ingress_column_groups(
        input_ids=token_ids,
        tied_weight=weight,
        embed_scale=2.0,
        groups=groups,
        reference_output=weight[token_ids] * 2.0,
    )
    hidden = torch.tensor(
        [[[1.0, 0.0, -1.0, 2.0, 0.5]]], dtype=torch.float64
    )
    egress = decompose_tied_lexical_egress_column_groups(
        hidden_states=hidden,
        tied_weight=weight,
        groups=groups,
        reference_output=hidden @ weight.T,
    )
    assert ingress.groups == egress.groups
    assert ingress.membership_sha256 == egress.membership_sha256
    assert ingress.additive_closure_passed
    assert egress.additive_closure_passed
    torch.testing.assert_close(ingress.reference_output, weight[token_ids] * 2)
    torch.testing.assert_close(egress.reference_output, hidden @ weight.T)


@pytest.mark.parametrize(
    "groups",
    [((0, 1), (1, 2)), ((0,), (2,)), ((0, 0), (1, 2))],
)
def test_physical_partition_rejects_overlap_gaps_and_duplicates(groups) -> None:
    with pytest.raises(ValueError):
        decompose_mlp_channel_groups(
            down_input=torch.ones((1, 3), dtype=torch.float64),
            down_projection_weight=torch.ones((2, 3), dtype=torch.float64),
            groups=groups,
            boundary_id="layer.0.mlp.operator_output",
        )


def test_cross_boundary_gradient_shape_cannot_be_silently_reused() -> None:
    surface = decompose_mlp_channel_groups(
        down_input=torch.ones((1, 3), dtype=torch.float64),
        down_projection_weight=torch.ones((2, 3), dtype=torch.float64),
        groups=((0,), (1, 2)),
        boundary_id="layer.0.mlp.operator_output",
    )
    with pytest.raises(ValueError, match="gradient shape"):
        surface.score_coordinates(torch.ones((1, 4), dtype=torch.float64))


def test_surface_clones_inputs_and_recomputes_every_integrity_claim() -> None:
    down_input = torch.ones((1, 3), dtype=torch.float64)
    weight = torch.ones((2, 3), dtype=torch.float64)
    surface = decompose_mlp_channel_groups(
        down_input=down_input,
        down_projection_weight=weight,
        groups=((0,), (1, 2)),
        boundary_id="layer.0.mlp.operator_output",
    )
    down_input.zero_()
    weight.zero_()
    surface.assert_integrity()
    with pytest.raises(ValueError, match="reconstructed output hash"):
        replace(
            surface,
            reconstructed_output=torch.zeros_like(surface.reconstructed_output),
            reconstructed_output_sha256=("0" * 64),
        )


def test_mutating_a_stored_tensor_is_detected_before_reuse() -> None:
    surface = decompose_mlp_channel_groups(
        down_input=torch.ones((1, 3), dtype=torch.float64),
        down_projection_weight=torch.ones((2, 3), dtype=torch.float64),
        groups=((0,), (1, 2)),
        boundary_id="layer.0.mlp.operator_output",
    )
    surface.group_contributions[0].zero_()
    with pytest.raises(ValueError, match="contribution hash"):
        surface.assert_integrity()


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("boundary_id", "layer.17.mlp.operator_output"),
        ("operator_kind", "attention_query_heads_after_O"),
        ("reference_source", "provided_operator_output"),
    ),
)
def test_surface_metadata_cannot_be_relabelled(
    field_name: str,
    field_value: str,
) -> None:
    surface = decompose_mlp_channel_groups(
        down_input=torch.ones((1, 3), dtype=torch.float64),
        down_projection_weight=torch.ones((2, 3), dtype=torch.float64),
        groups=((0,), (1, 2)),
        boundary_id="layer.0.mlp.operator_output",
    )
    with pytest.raises(ValueError, match="surface semantic hash"):
        replace(surface, **{field_name: field_value})


def test_cancelling_part_mutations_are_detected() -> None:
    surface = decompose_mlp_channel_groups(
        down_input=torch.ones((1, 3), dtype=torch.float64),
        down_projection_weight=torch.ones((2, 3), dtype=torch.float64),
        groups=((0,), (1, 2)),
        boundary_id="layer.0.mlp.operator_output",
    )
    surface.group_contributions[0].add_(1.0)
    surface.group_contributions[1].sub_(1.0)
    with pytest.raises(ValueError, match="contribution hash"):
        surface.assert_integrity()


def test_float32_checkpoint_width_closure_policy_is_realistic() -> None:
    generator = torch.Generator().manual_seed(20260811)
    down_input = torch.randn((2, 2_048), generator=generator)
    down_weight = torch.randn((640, 2_048), generator=generator)
    mlp = decompose_mlp_channel_groups(
        down_input=down_input,
        down_projection_weight=down_weight,
        groups=tuple(
            tuple(range(start, start + 256)) for start in range(0, 2_048, 256)
        ),
        boundary_id="layer.0.mlp.operator_output",
        reference_output=down_input @ down_weight.T,
    )
    assert mlp.additive_closure_passed

    probability = torch.softmax(
        torch.randn((1, 4, 16, 16), generator=generator), dim=-1
    )
    values = torch.randn((1, 4, 16, 256), generator=generator)
    output_weight = torch.randn((640, 1_024), generator=generator)
    slices = output_weight.reshape(640, 4, 256).permute(1, 2, 0)
    native_attention = sum(
        torch.einsum(
            "bqk,bkd,do->bqo",
            probability[:, head],
            values[:, head],
            slices[head],
        )
        for head in range(4)
    )
    attention = decompose_attention_head_groups(
        probabilities=probability,
        repeated_values=values,
        output_projection_weight=output_weight,
        groups=((0,), (1,), (2,), (3,)),
        boundary_id="layer.0.attention.operator_output",
        reference_output=native_attention,
    )
    assert attention.additive_closure_passed

    tied_weight = torch.randn((4_096, 640), generator=generator)
    terminal_hidden = torch.randn((2, 640), generator=generator)
    lexical = decompose_tied_lexical_egress_column_groups(
        hidden_states=terminal_hidden,
        tied_weight=tied_weight,
        groups=tuple(
            tuple(range(start, start + 80)) for start in range(0, 640, 80)
        ),
        reference_output=terminal_hidden @ tied_weight.T,
    )
    assert lexical.additive_closure_passed
    assert lexical.reference_source == "provided_operator_output"


def test_low_precision_audit_tensors_are_rejected() -> None:
    with pytest.raises(ValueError, match="audit precision"):
        decompose_mlp_channel_groups(
            down_input=torch.ones((1, 3), dtype=torch.bfloat16),
            down_projection_weight=torch.ones((2, 3), dtype=torch.bfloat16),
            groups=((0,), (1, 2)),
            boundary_id="layer.0.mlp.operator_output",
        )
