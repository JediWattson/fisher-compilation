from __future__ import annotations

import pytest
import torch
from torch import nn

from fisher_graph import gemma3_all_mode_generator_graph_executor as executor
from fisher_graph import (
    gemma3_full_model_all_mode_generator_graph_experiment as full_model,
)


def _exact_graph() -> executor.Gemma3AllModeGeneratorGraphMLP:
    source = nn.Module()
    source.gate_proj = nn.Linear(2, 3, bias=False)
    source.up_proj = nn.Linear(2, 3, bias=False)
    source.down_proj = nn.Linear(3, 2, bias=False)
    source.act_fn = nn.SiLU()
    return executor.compile_exact_all_mode_generator_graph(
        source,
        model_fingerprint="1" * 64,
        layer_ordinal=0,
    )


def _affine_graph() -> executor.Gemma3AllModeGeneratorGraphMLP:
    return executor.compile_fitted_affine_all_mode_generator_graph(
        affine_weight=torch.zeros(3, 2),
        affine_bias=torch.zeros(3),
        decoder_weight=torch.zeros(2, 3),
        model_fingerprint="1" * 64,
        layer_ordinal=0,
        fit_split_sha256="2" * 64,
    )


def _chart_graph() -> executor.Gemma3AllModeGeneratorGraphMLP:
    return executor.compile_chart_conditioned_all_mode_generator_graph(
        affine_weight=torch.zeros(3, 2),
        affine_bias=torch.zeros(3),
        decoder_weight=torch.zeros(2, 3),
        chart_centers=torch.zeros(2, 2),
        chart_bases=torch.ones(2, 2, 1),
        chart_distance_scales=torch.ones(2),
        chart_edge_weight=torch.zeros(2, 2, 3),
        chart_temperature=1.0,
        model_fingerprint="1" * 64,
        layer_ordinal=0,
        fit_split_sha256="2" * 64,
    )


def test_full_model_scope_is_every_consecutive_layer() -> None:
    assert full_model._full_model_layer_ordinals(4) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="layer count"):
        full_model._full_model_layer_ordinals(0)


def test_resource_ledger_distinguishes_exact_affine_and_chart_storage() -> None:
    exact = full_model._graph_resource_ledger(
        {0: _exact_graph()},
        source_whole_model_parameters=1_000,
    )
    affine = full_model._graph_resource_ledger(
        {0: _affine_graph()},
        source_whole_model_parameters=1_000,
    )
    chart = full_model._graph_resource_ledger(
        {0: _chart_graph()},
        source_whole_model_parameters=1_000,
    )

    assert exact["native_target_mlp_parameter_scalars"] == 18
    assert exact["compiled_stored_coefficient_scalars"] == 18
    assert exact["whole_model_stored_scalar_reduction_fraction"] == 0.0
    assert affine["compiled_stored_coefficient_scalars"] == 15
    assert affine["target_mlp_stored_scalar_reduction_fraction"] == pytest.approx(
        1.0 / 6.0
    )
    assert affine["target_mlp_estimated_mac_reduction_fraction"] == pytest.approx(
        1.0 / 3.0
    )
    assert chart["compiled_stored_coefficient_scalars"] == 37
    assert chart["compiled_estimated_macs_per_token"] == 38
    assert chart["target_mlp_stored_scalar_reduction_fraction"] < 0.0


def test_runtime_module_contract_tracks_each_compiled_operation() -> None:
    assert set(full_model._runtime_modules(_exact_graph())) == {
        "graph_forward",
        "mode_decoder",
        "gate_generators",
        "up_generators",
    }
    assert set(full_model._runtime_modules(_affine_graph())) == {
        "graph_forward",
        "mode_decoder",
        "affine_generators",
    }
    assert set(full_model._runtime_modules(_chart_graph())) == {
        "graph_forward",
        "mode_decoder",
        "affine_generators",
    }


def test_full_model_runner_rejects_noncanonical_runtime_before_loading() -> None:
    with pytest.raises(ValueError, match="pinned Gemma"):
        full_model.run_gemma3_full_model_all_mode_generator_graph(
            revision="not-the-pinned-revision",
        )
    with pytest.raises(ValueError, match="CPU float32"):
        full_model.run_gemma3_full_model_all_mode_generator_graph(
            revision=full_model.DEFAULT_REVISION,
            device_name="mps",
        )
