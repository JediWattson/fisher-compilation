from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.full_mlp_stack_generators import (
    FullMLPStackGeneratorFit,
    fit_full_mlp_stack_generators,
)
from fisher_graph.gemma3_full_mlp_stack_rows import FullMLPStackLayerRows
from fisher_graph.gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)
from fisher_graph.parameter_layer_superfragments import (
    ParameterLayerSuperfragment,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sources() -> tuple[
    FullMLPStackLayerRows,
    FullMLPStackLayerRows,
    ParameterLayerSuperfragment,
]:
    fragment_sha256s = tuple(
        sorted((_digest("fragment-0"), _digest("fragment-1")))
    )
    superfragment = ParameterLayerSuperfragment(
        layer_ordinal=0,
        layer_id="model.layers.0",
        activation_site="model.layers.0.mlp.gated",
        input_site="model.layers.0.pre_feedforward_layernorm",
        output_site="model.layers.0.mlp.output",
        input_catalog_sha256=_digest("input-catalog"),
        input_width=4,
        output_width=4,
        member_fragment_sha256s=fragment_sha256s,
        group_indices=(0, 1, 2, 3),
        channel_indices=(0, 1, 2, 3),
        native_parameter_count=4 * (4 + 4 + 4),
        fisher_mass=8.0,
        source_fragment_plan_sha256=_digest("fragment-plan"),
        source_cluster_plan_sha256=_digest("cluster-plan"),
        source_fisher_coupling_sha256=_digest("fisher"),
        parameter_catalog_sha256=_digest("parameter-catalog"),
        source_model_sha256=_digest("model"),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(119)
    fit_inputs = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    eval_inputs = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    coefficient = torch.tensor(
        (
            (0.8, -0.1, 0.2, 0.4),
            (0.3, 0.7, -0.5, 0.1),
            (-0.2, 0.6, 0.9, -0.3),
            (0.5, 0.2, 0.4, 0.8),
        ),
        dtype=torch.float64,
    )
    bias = torch.tensor((0.1, -0.2, 0.05, 0.3), dtype=torch.float64)
    fit_contributions = fit_inputs @ coefficient + bias
    eval_contributions = eval_inputs @ coefficient + bias

    common = {
        "layer_ordinal": 0,
        "layer_id": "model.layers.0",
        "input_site": "model.layers.0.pre_feedforward_layernorm",
        "activation_site": "model.layers.0.mlp.gated",
        "output_site": "model.layers.0.mlp.output",
        "intermediate_width": 4,
        "fragment_ids": (
            "cluster.0/layer.0",
            "cluster.1/layer.0",
        ),
        "fragment_sha256s": fragment_sha256s,
    }
    fit = FullMLPStackLayerRows(
        **common,
        inputs=fit_inputs,
        contributions=fit_contributions,
        fisher_weights=torch.linspace(0.5, 1.5, 8, dtype=torch.float64),
        row_keys=tuple((f"fit-{index}", 0) for index in range(8)),
        sequences=8,
    )
    evaluation = FullMLPStackLayerRows(
        **common,
        inputs=eval_inputs,
        contributions=eval_contributions,
        fisher_weights=torch.linspace(0.6, 1.4, 5, dtype=torch.float64),
        row_keys=tuple((f"eval-{index}", 0) for index in range(5)),
        sequences=5,
    )
    return fit, evaluation, superfragment


def _fit() -> FullMLPStackGeneratorFit:
    fit, evaluation, superfragment = _sources()
    return fit_full_mlp_stack_generators(
        fit,
        evaluation,
        superfragment=superfragment,
        source_model_sha256=superfragment.source_model_sha256,
        parameter_catalog_sha256=(
            superfragment.parameter_catalog_sha256
        ),
        fisher_coupling_sha256=(
            superfragment.source_fisher_coupling_sha256
        ),
        superfragment_plan_sha256=_digest("superfragment-plan"),
        fit_split_sha256=_digest("fit-split"),
        eval_split_sha256=_digest("eval-split"),
        mode_ranks=(1, 2, 4),
        selected_mode_rank=4,
        generator_ranks=(1, 2, 4),
        selected_generator_rank=4,
    )


def test_full_layer_fit_builds_both_ladders_and_direct_dense_plan() -> None:
    fit_rows, eval_rows, superfragment = _sources()
    fitted = _fit()

    assert tuple(
        point.rank for point in fitted.computational_modes.points
    ) == (1, 2, 4)
    assert tuple(
        point.rank for point in fitted.coordinate_generators.points
    ) == (1, 2, 4)
    assert fitted.selected_mode_rank == 4
    assert fitted.selected_generator_rank == 4
    assert fitted.selected_basis.residual_width == 4
    assert fitted.selected_coordinate_plan.output_width == 4
    assert fitted.executable_plan.output_width == 4
    assert (
        fitted.executable_plan.binding
        .parameter_cluster_fragment_sha256
        is None
    )
    assert (
        fitted.executable_plan.binding.source_generator_plan_sha256
        == fitted.selected_coordinate_plan.artifact_sha256
    )
    assert (
        fitted.executable_plan.binding
        .computational_mode_basis_sha256
        == fitted.selected_basis.artifact_sha256
    )
    assert (
        fitted.executable_plan.binding.fisher_coupling_sha256
        == superfragment.source_fisher_coupling_sha256
    )

    coordinate_prediction = fitted.selected_basis.decode(
        fitted.selected_coordinate_plan.apply(eval_rows.inputs)
    )
    dense_prediction = fitted.executable_plan.apply(eval_rows.inputs)
    assert torch.allclose(
        coordinate_prediction,
        dense_prediction,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(
        dense_prediction,
        eval_rows.contributions,
        rtol=1e-9,
        atol=1e-9,
    )
    assert fitted.fit_row_key_sha256 == fit_rows.row_key_sha256
    assert fitted.eval_row_key_sha256 == eval_rows.row_key_sha256

    replacement = Gemma3ModalGeneratorReplacement(
        layer_ordinal=0,
        removed_mode_indices=(0, 1, 2, 3),
        generator_plan=fitted.executable_plan,
    )
    assert replacement.lowering is None


def test_result_reports_separate_native_coordinate_and_dense_resources() -> None:
    fitted = _fit()
    structural = fitted.structural_metadata
    resources = fitted.resource_metadata

    assert structural["aggregation_order"] == (
        "full_native_layer_before_modes"
    )
    assert structural["replacement_scope"] == "complete_native_mlp"
    assert structural["source_fragment_count"] == 2
    assert structural["singular_fragment_lowering_required"] is False
    assert resources["native_mlp_parameter_count"] == 48
    assert resources["native_mlp_linear_macs_per_token"] == 48
    assert resources["selected_basis_stored_scalar_count"] == 20
    assert resources["coordinate_generator_parameter_count"] == 36
    assert resources["dense_fused_parameter_count"] == 36
    assert resources["dense_fused_macs_per_token"] == 32
    assert resources["net_stored_parameter_savings"] == 12
    assert resources["net_linear_macs_saved_per_token"] == 16
    assert resources["dense_parameter_reduction_fraction"] == pytest.approx(
        0.25
    )
    assert resources["dense_execution_stores_basis_separately"] is False


def test_full_layer_fit_roundtrip_is_strict_and_authenticates_fusion() -> None:
    fitted = _fit()
    restored = FullMLPStackGeneratorFit.from_state_dict(
        fitted.state_dict()
    )

    assert restored.artifact_sha256 == fitted.artifact_sha256
    assert (
        restored.dense_fused_residual_plan.artifact_sha256
        == fitted.dense_fused_residual_plan.artifact_sha256
    )
    assert torch.equal(
        restored.dense_fused_residual_plan.factors.output_factor,
        fitted.dense_fused_residual_plan.factors.output_factor,
    )
    metadata = restored.metadata()
    assert metadata["contains_source_model_weights"] is False
    assert metadata["contains_raw_fit_rows"] is False
    assert metadata["contains_raw_eval_rows"] is False
    assert metadata["evaluation_used_for_rank_selection"] is False
    assert (
        metadata["dense_plan_executable_without_fragment_lowering"]
        is True
    )

    tensor_tamper = copy.deepcopy(fitted.state_dict())
    tensor_tamper["dense_fused_residual_plan"]["factors"][
        "output_factor"
    ][0, 0] += 0.25
    with pytest.raises(ValueError, match="sha256|hash"):
        FullMLPStackGeneratorFit.from_state_dict(tensor_tamper)

    summary_tamper = copy.deepcopy(fitted.state_dict())
    summary_tamper["resource_metadata"][
        "dense_fused_parameter_count"
    ] += 1
    with pytest.raises(ValueError, match="summaries"):
        FullMLPStackGeneratorFit.from_state_dict(summary_tamper)


def test_fit_enforces_rank_caps_provenance_and_row_disjointness() -> None:
    fit_rows, eval_rows, superfragment = _sources()
    arguments = {
        "superfragment": superfragment,
        "source_model_sha256": superfragment.source_model_sha256,
        "parameter_catalog_sha256": (
            superfragment.parameter_catalog_sha256
        ),
        "fisher_coupling_sha256": (
            superfragment.source_fisher_coupling_sha256
        ),
        "superfragment_plan_sha256": _digest("superfragment-plan"),
        "fit_split_sha256": _digest("fit-split"),
        "eval_split_sha256": _digest("eval-split"),
        "mode_ranks": (1, 2, 4),
        "selected_mode_rank": 4,
        "generator_ranks": (1, 2, 4),
        "selected_generator_rank": 4,
    }

    with pytest.raises(ValueError, match="fit-row/residual thin rank"):
        fit_full_mlp_stack_generators(
            fit_rows,
            eval_rows,
            **{**arguments, "mode_ranks": (1, 4, 5)},
        )
    with pytest.raises(ValueError, match="selected mode rank"):
        fit_full_mlp_stack_generators(
            fit_rows,
            eval_rows,
            **{
                **arguments,
                "mode_ranks": (1, 2),
                "selected_mode_rank": 2,
                "generator_ranks": (1, 3),
                "selected_generator_rank": 3,
            },
        )
    with pytest.raises(ValueError, match="provenance"):
        fit_full_mlp_stack_generators(
            fit_rows,
            eval_rows,
            **{
                **arguments,
                "source_model_sha256": _digest("foreign-model"),
            },
        )

    overlapping_eval = replace(
        eval_rows,
        row_keys=(
            fit_rows.row_keys[0],
            *eval_rows.row_keys[1:],
        ),
        row_key_sha256="",
    )
    with pytest.raises(ValueError, match="row identities"):
        fit_full_mlp_stack_generators(
            fit_rows,
            overlapping_eval,
            **arguments,
        )

    wrong_members = replace(
        eval_rows,
        fragment_sha256s=(
            _digest("foreign-fragment"),
            eval_rows.fragment_sha256s[1],
        ),
    )
    with pytest.raises(ValueError, match="different layers"):
        fit_full_mlp_stack_generators(
            fit_rows,
            wrong_members,
            **arguments,
        )
