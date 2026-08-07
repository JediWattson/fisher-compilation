from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_l10_l17_trajectory_correction_fitting import (
    build_a3_raw_mlp_target,
    build_projected_correction_rows,
    fit_frozen_basis_coordinate_generators,
    project_joint_target_to_frozen_bases,
    replace_layer_nodes_in_composed_graph,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    LayerFragmentRows,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)
from fisher_graph.modal_generator_graph import ModalGeneratorGraphPlan
from test_gemma3_modal_generator_graph_executor import _fixture


def _slice_rows(
    rows: AlignedFragmentRows,
    start: int,
    stop: int,
) -> AlignedFragmentRows:
    return AlignedFragmentRows(
        rows_by_fragment={
            name: LayerFragmentRows(
                inputs=value.inputs[start:stop],
                contributions=value.contributions[start:stop],
                fisher_weights=value.fisher_weights[start:stop],
                sequences=stop - start,
            )
            for name, value in rows.rows_by_fragment.items()
        },
        row_keys=rows.row_keys[start:stop],
    )


def test_joint_projection_uses_affine_sum_for_a_partial_overlapping_span() -> None:
    fixture = _fixture(full_layer=True)
    basis = fixture.lowerings[0].computational_mode_basis
    generator = torch.Generator().manual_seed(81_903)
    target = torch.randn(11, basis.residual_width, generator=generator)

    projection = project_joint_target_to_frozen_bases(
        target,
        {"first": basis, "second": basis},
        node_order=("first", "second"),
    )

    mean_sum = 2.0 * basis.mean
    centered = target.to(dtype=torch.float64) - mean_sum
    expected = mean_sum + centered @ basis.encoder_basis @ basis.decoder_basis
    torch.testing.assert_close(projection.prediction, expected)
    assert projection.combined_basis_rank == basis.rank
    assert projection.metadata()["affine_offset_sha256"] != (
        projection.metadata()["target_sha256"]
    )


def test_a3_joint_projection_and_frozen_basis_refit_preserve_resources() -> None:
    fixture = _fixture(full_layer=True)
    source_graph = ModalGeneratorGraphPlan(
        model_fingerprint=fixture.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            fixture.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=fixture.graph_plan.nodes,
        interactions=(),
    )
    node_order = source_graph.traversal_order
    lowerings = {
        node.name: lowering
        for node, lowering in zip(
            source_graph.nodes,
            fixture.lowerings,
            strict=True,
        )
    }
    bases = {
        name: lowerings[name].computational_mode_basis for name in node_order
    }
    fragment_ids = {
        name: lowerings[name].mode_set_id for name in node_order
    }
    generator = torch.Generator().manual_seed(78_211)
    inputs = torch.randn(40, 4, generator=generator, dtype=torch.float64)
    node_coordinates = {
        name: inputs
        @ torch.randn(4, basis.rank, generator=generator, dtype=torch.float64)
        + torch.randn(basis.rank, generator=generator, dtype=torch.float64)
        for name, basis in bases.items()
    }
    expected_target = torch.stack(
        tuple(
            bases[name].decode(node_coordinates[name]) for name in node_order
        )
    ).sum(dim=0)
    compiled_selected = {
        name: torch.randn(
            40,
            4,
            generator=generator,
            dtype=torch.float64,
        )
        for name in node_order
    }
    retained = torch.randn(40, 4, generator=generator, dtype=torch.float64)
    compiled_full = retained + torch.stack(
        tuple(compiled_selected[name] for name in node_order)
    ).sum(dim=0)
    native_full = retained + expected_target
    algebraic_retained_audit = compiled_full - torch.stack(
        tuple(compiled_selected[name] for name in node_order)
    ).sum(dim=0)
    torch.testing.assert_close(
        algebraic_retained_audit,
        retained,
        atol=1e-12,
        rtol=1e-12,
    )
    target = build_a3_raw_mlp_target(
        native_full,
        retained,
    )
    torch.testing.assert_close(target, expected_target, atol=1e-12, rtol=1e-12)

    rows, projection = build_projected_correction_rows(
        inputs=inputs,
        target=target,
        fisher_weights_by_node={
            name: torch.linspace(1.0, 2.0, 40, dtype=torch.float64)
            for name in node_order
        },
        fragment_id_by_node=fragment_ids,
        bases_by_node=bases,
        node_order=node_order,
        row_keys=tuple((f"example-{index}", 0) for index in range(40)),
        sequences=40,
    )
    torch.testing.assert_close(
        projection.prediction,
        target,
        atol=1e-10,
        rtol=1e-10,
    )
    assert projection.metadata()["runtime_parameter_count"] == 0
    assert projection.metadata()["runtime_macs_per_token"] == 0
    assert projection.metadata()["projection_method"] == (
        "float64_affine_sum_svd_pseudoinverse_minimum_norm"
    )
    assert len(projection.metadata()["affine_offset_sha256"]) == 64
    assert projection.metadata()["mean_bias_sha256_by_node"] == {
        name: bases[name].mean_bias_sha256 for name in node_order
    }
    assert projection.metadata()["decoder_basis_sha256_by_node"] == {
        name: bases[name].decoder_basis_sha256 for name in node_order
    }

    overlapping = _slice_rows(rows, 0, 10)
    with pytest.raises(ValueError, match="row keys overlap"):
        fit_frozen_basis_coordinate_generators(
            overlapping,
            overlapping,
            source_graph=source_graph,
            source_lowerings_by_node=lowerings,
            fit_split_sha256="a" * 64,
            eval_split_sha256="b" * 64,
            generator_rank=2,
        )

    fitted = fit_frozen_basis_coordinate_generators(
        _slice_rows(rows, 0, 30),
        _slice_rows(rows, 30, 40),
        source_graph=source_graph,
        source_lowerings_by_node=lowerings,
        fit_split_sha256="a" * 64,
        eval_split_sha256="b" * 64,
        generator_rank=2,
    )
    assert fitted.graph_plan.parameter_count == source_graph.parameter_count
    assert fitted.graph_plan.macs_per_token == source_graph.macs_per_token
    assert fitted.graph_plan.interactions == ()
    assert fitted.metadata()["source_mean_bias_sha256_by_node"] == {
        name: bases[name].mean_bias_sha256 for name in node_order
    }
    assert fitted.metadata()["source_decoder_basis_sha256_by_node"] == {
        name: bases[name].decoder_basis_sha256 for name in node_order
    }
    for name in node_order:
        assert (
            fitted.lowerings_by_node[
                name
            ].computational_mode_basis.decoder_basis_sha256
            == bases[name].decoder_basis_sha256
        )
        assert (
            fitted.lowerings_by_node[
                name
            ].coordinate_generator_plan.binding.source_generator_plan_sha256
            == lowerings[name].coordinate_generator_plan.artifact_sha256
        )

    relocated = fit_frozen_basis_coordinate_generators(
        _slice_rows(rows, 0, 30),
        _slice_rows(rows, 30, 40),
        source_graph=source_graph,
        source_lowerings_by_node=lowerings,
        fit_split_sha256="a" * 64,
        eval_split_sha256="b" * 64,
        generator_rank=2,
        output_boundary="layer.0.mlp.delta",
    )
    assert relocated.graph_plan.parameter_count == source_graph.parameter_count
    assert relocated.graph_plan.macs_per_token == source_graph.macs_per_token
    assert all(
        node.output_boundary == "layer.0.mlp.delta"
        for node in relocated.graph_plan.nodes
    )
    for name in node_order:
        source = lowerings[name]
        rebound = relocated.lowerings_by_node[name]
        assert (
            rebound.coordinate_generator_plan.binding.target_kind
            == "relocated_computational_mode_coordinates"
        )
        assert (
            rebound.coordinate_generator_plan.binding.output_site
            == "layer.0.mlp.delta"
        )
        assert (
            rebound.computational_mode_basis.binding.source_kind
            == "relocated_layer_fragment"
        )
        assert (
            rebound.computational_mode_basis.binding.output_site
            == "layer.0.mlp.delta"
        )
        assert (
            rebound.computational_mode_basis.mean_bias_sha256
            == source.computational_mode_basis.mean_bias_sha256
        )
        assert (
            rebound.computational_mode_basis.encoder_basis_sha256
            == source.computational_mode_basis.encoder_basis_sha256
        )
        assert (
            rebound.computational_mode_basis.decoder_basis_sha256
            == source.computational_mode_basis.decoder_basis_sha256
        )

    replaced = replace_layer_nodes_in_composed_graph(
        source_graph,
        fitted.graph_plan,
        layer_ordinal=0,
    )
    assert replaced.parameter_count == source_graph.parameter_count
    assert replaced.macs_per_token == source_graph.macs_per_token
    assert tuple(node.name for node in replaced.nodes) == node_order

    relocated_replacement = replace_layer_nodes_in_composed_graph(
        source_graph,
        relocated.graph_plan,
        layer_ordinal=0,
    )
    assert relocated_replacement.parameter_count == (
        source_graph.parameter_count
    )
    assert relocated_replacement.macs_per_token == source_graph.macs_per_token
    assert all(
        node.output_boundary == "layer.0.mlp.delta"
        for node in relocated_replacement.nodes
    )
