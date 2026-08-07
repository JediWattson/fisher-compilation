from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

import fisher_graph.gemma3_l10_l17_a5c_broader_selected_generator as a5c_runner
from fisher_graph.computational_modes import (
    ComputationalModeBinding,
    fit_computational_mode_rate_curve,
)
from fisher_graph.modal_generator_graph import (
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
)
from fisher_graph.modal_generator_lowering import (
    ModalGeneratorLowering,
    lower_coordinate_modal_generator,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    fit_modal_generator_rate_curve,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)


def _fragment_plan() -> ParameterClusterLayerFragmentPlan:
    fragment = ParameterClusterLayerFragment(
        cluster_id=2,
        layer_ordinal=4,
        layer_id="model.layers.4",
        activation_site="model.layers.4.mlp.gated",
        input_site="model.layers.4.mlp.input",
        output_site="model.layers.4.mlp.residual_delta",
        input_catalog_sha256="1" * 64,
        input_width=3,
        output_width=5,
        group_indices=(7, 8),
        channel_indices=(3, 5),
        fisher_ranks=(1, 0),
        axial_orientations=(1, -1),
        native_parameter_count=22,
        fisher_mass=7.5,
        source_cluster_plan_sha256="9" * 64,
        source_fisher_coupling_sha256="c" * 64,
        parameter_catalog_sha256="b" * 64,
        source_model_sha256="a" * 64,
    )
    return ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256="9" * 64,
        source_fisher_coupling_sha256="c" * 64,
        parameter_catalog_sha256="b" * 64,
        source_model_sha256="a" * 64,
        cluster_count=3,
        source_group_count=10,
        assigned_group_count=2,
        assigned_native_parameter_count=22,
        fragments=(fragment,),
    )


def _basis(fragment_plan: ParameterClusterLayerFragmentPlan):
    fragment = fragment_plan.fragments[0]
    generator = torch.Generator().manual_seed(91)
    raw = torch.randn(5, 2, generator=generator, dtype=torch.float64)
    decoder_columns, _ = torch.linalg.qr(raw, mode="reduced")
    fit_modes = torch.randn(
        30,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    eval_modes = torch.randn(
        12,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    mean = torch.tensor(
        [0.25, -0.5, 1.0, 0.0, 0.75],
        dtype=torch.float64,
    )
    fit = fit_modes @ decoder_columns.T + mean
    evaluation = eval_modes @ decoder_columns.T + mean
    binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind="layer_fragment",
        output_site="model.layers.4.mlp.residual_delta",
        source_model_sha256=fragment.source_model_sha256,
        parameter_catalog_sha256=fragment.parameter_catalog_sha256,
        fisher_coupling_sha256=(
            fragment.source_fisher_coupling_sha256
        ),
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256="f" * 64,
        eval_split_sha256="e" * 64,
    )
    return fit_computational_mode_rate_curve(
        fit,
        torch.ones(30, dtype=torch.float64),
        evaluation,
        torch.ones(12, dtype=torch.float64),
        (2,),
        binding=binding,
        selection_rule="fixed_rank",
        selected_rank=2,
    ).selected_basis


def _coordinate_plan(
    basis,
    fragment_plan,
    *,
    fit_intercept: bool = True,
):
    fragment = fragment_plan.fragments[0]
    generator = torch.Generator().manual_seed(113)
    X_fit = torch.randn(40, 3, generator=generator, dtype=torch.float64)
    X_eval = torch.randn(14, 3, generator=generator, dtype=torch.float64)
    coefficient = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0], [-1.5, 0.75]],
        dtype=torch.float64,
    )
    bias = torch.tensor([0.4, -0.2], dtype=torch.float64)
    Y_fit = X_fit @ coefficient + (bias if fit_intercept else 0.0)
    Y_eval = X_eval @ coefficient + (bias if fit_intercept else 0.0)
    binding = ModalGeneratorBinding.create(
        generator_id="layer.4.cluster.2.modes",
        input_kind="native_layer_input",
        input_site="model.layers.4.mlp.input",
        output_site=basis.binding.output_site,
        source_model_sha256=basis.binding.source_model_sha256,
        input_catalog_sha256="1" * 64,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=fragment_plan.artifact_sha256,
        fit_split_sha256=basis.binding.fit_split_sha256,
        eval_split_sha256=basis.binding.eval_split_sha256,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=basis.binding.fisher_coupling_sha256,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    curve = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        torch.ones(40, dtype=torch.float64),
        X_eval,
        Y_eval,
        (2,),
        binding=binding,
        fisher_weights_eval=torch.ones(14, dtype=torch.float64),
        fit_intercept=fit_intercept,
        selection_rule="fixed_rank",
        selected_rank=2,
    )
    return curve.selected_plan, X_eval


def _lowering(*, fit_intercept: bool = True):
    fragments = _fragment_plan()
    basis = _basis(fragments)
    plan, inputs = _coordinate_plan(
        basis,
        fragments,
        fit_intercept=fit_intercept,
    )
    return (
        lower_coordinate_modal_generator(plan, basis, fragments),
        inputs,
    )


def test_a5c_descriptor_reauthenticates_real_lowerings_and_graph_pairing() -> None:
    fitted = tuple(
        _lowering(fit_intercept=index % 2 == 0)[0] for index in range(4)
    )
    names = tuple(f"fragment.4.{index}" for index in range(4))
    nodes = tuple(
        lowering.to_graph_node(name=name, causal_order=index)
        for index, (name, lowering) in enumerate(zip(names, fitted, strict=True))
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=(
            fitted[0].coordinate_generator_plan.binding.source_model_sha256
        ),
        parameter_cluster_plan_sha256=(
            fitted[0].coordinate_generator_plan.binding.cluster_plan_sha256
        ),
        nodes=nodes,
        interactions=(),
    )
    by_name = dict(zip(names, fitted, strict=True))

    descriptor = a5c_runner._graph_descriptor(
        layer17_graph=graph,
        layer17_lowerings_by_node=by_name,
        composition_graph=graph,
        layer17_post_feedforward_delta_layer_ordinals=(),
        composition_post_feedforward_delta_layer_ordinals=(),
    )

    assert descriptor["layer17_lowering_sha256_by_node"] == {
        name: lowering.artifact_sha256 for name, lowering in by_name.items()
    }
    mismatched = dict(by_name)
    mismatched[names[0]] = fitted[1]
    with pytest.raises(ValueError, match="node and lowering weights are unpaired"):
        a5c_runner._graph_descriptor(
            layer17_graph=graph,
            layer17_lowerings_by_node=mismatched,
            composition_graph=graph,
            layer17_post_feedforward_delta_layer_ordinals=(),
            composition_post_feedforward_delta_layer_ordinals=(),
        )


def test_graph_state_is_exact_generated_coordinates_and_decoder_is_basis() -> None:
    lowering, inputs = _lowering()
    source = lowering.coordinate_generator_plan
    basis = lowering.computational_mode_basis
    node = lowering.to_graph_node(
        name="fragment.4.2",
        causal_order=0,
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=source.binding.source_model_sha256,
        parameter_cluster_plan_sha256=(
            source.binding.cluster_plan_sha256
        ),
        nodes=(node,),
        interactions=(),
    )
    execution = ModalGeneratorGraphExecutor(graph)(
        {source.binding.input_site: inputs},
        capture_modal_states=True,
    )
    coordinates = source.apply(inputs)
    expected = basis.decode(coordinates)

    assert torch.equal(execution.modal_states[node.name], coordinates)
    torch.testing.assert_close(
        execution.outputs[source.binding.output_site],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        node.weights.output_factor,
        basis.decoder_basis,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        node.weights.output_bias,
        basis.mean_bias,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        node.weights.latent_bias,
        source.factors.bias,
        rtol=0,
        atol=0,
    )
    assert node.weights.state_kind == "computational_mode_coordinates"


def test_dense_fused_plan_is_algebraically_identical_and_bound_to_basis() -> None:
    lowering, inputs = _lowering()
    source = lowering.coordinate_generator_plan
    basis = lowering.computational_mode_basis
    fused = lowering.fused_residual_plan

    torch.testing.assert_close(
        fused.apply(inputs),
        basis.decode(source.apply(inputs)),
        rtol=1e-12,
        atol=1e-12,
    )
    assert fused.binding.target_kind == "cluster_residual_contribution"
    assert fused.binding.computational_mode_basis_sha256 == (
        basis.artifact_sha256
    )
    assert fused.binding.source_generator_plan_sha256 == (
        source.artifact_sha256
    )
    assert fused.binding.parameter_cluster_fragment_sha256 == (
        basis.binding.parameter_cluster_sha256
    )


def test_bias_free_coordinate_plan_has_no_stored_latent_bias() -> None:
    lowering, inputs = _lowering(fit_intercept=False)
    weights = lowering.graph_weights
    source = lowering.coordinate_generator_plan
    basis = lowering.computational_mode_basis

    assert source.factors.bias is None
    assert weights.latent_bias is None
    assert weights.latent_bias_sha256 is None
    # Graph storage: fused input->K, K->residual decoder, and decoder mean.
    expected = (
        weights.input_width * weights.private_width
        + weights.private_width * weights.latent_width
        + weights.latent_width * weights.output_width
        + weights.output_width
    )
    assert weights.parameter_count == expected
    assert weights.bias_additions_per_token == weights.output_width
    torch.testing.assert_close(
        lowering.fused_residual_plan.apply(inputs),
        basis.decode(source.apply(inputs)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fanin_fanout_interactions_operate_on_true_mode_coordinates() -> None:
    lowering, _ = _lowering(fit_intercept=False)
    base = lowering.graph_weights
    root = lowering.to_graph_node(
        name="root",
        causal_order=0,
        input_boundary="x.root",
        output_boundary="y.root",
    )
    left = lowering.to_graph_node(
        name="left",
        causal_order=1,
        input_boundary="x.left",
        output_boundary="y.branch",
    )
    right = lowering.to_graph_node(
        name="right",
        causal_order=2,
        input_boundary="x.right",
        output_boundary="y.branch",
    )
    sink = lowering.to_graph_node(
        name="sink",
        causal_order=3,
        input_boundary="x.sink",
        output_boundary="y.sink",
    )
    identity = torch.eye(base.latent_width, dtype=torch.float64)
    zero = torch.zeros(base.latent_width, dtype=torch.float64)
    edges = (
        ModalGeneratorInteraction(
            "root",
            "left",
            identity,
            zero,
        ),
        ModalGeneratorInteraction(
            "root",
            "right",
            -identity,
            zero,
        ),
        ModalGeneratorInteraction(
            "left",
            "sink",
            identity,
            zero,
        ),
        ModalGeneratorInteraction(
            "right",
            "sink",
            identity,
            zero,
        ),
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=base.source_model_sha256,
        parameter_cluster_plan_sha256=(
            base.parameter_cluster_plan_sha256
        ),
        nodes=(root, left, right, sink),
        interactions=tuple(
            sorted(edges, key=lambda edge: edge.key)
        ),
    )
    values = {
        name: torch.randn(3, base.input_width, dtype=torch.float64)
        for name in ("x.root", "x.left", "x.right", "x.sink")
    }
    result = ModalGeneratorGraphExecutor(graph)(
        values,
        capture_modal_states=True,
    )
    own = {
        name: lowering.coordinate_generator_plan.apply(values[f"x.{name}"])
        for name in ("root", "left", "right", "sink")
    }
    torch.testing.assert_close(result.modal_states["root"], own["root"])
    torch.testing.assert_close(
        result.modal_states["left"],
        own["left"] + own["root"],
    )
    torch.testing.assert_close(
        result.modal_states["right"],
        own["right"] - own["root"],
    )
    torch.testing.assert_close(
        result.modal_states["sink"],
        own["sink"] + own["left"] + own["right"],
    )


def test_lowering_roundtrip_tamper_and_basis_swap_are_rejected() -> None:
    lowering, _ = _lowering()
    restored = ModalGeneratorLowering.from_state_dict(
        lowering.state_dict()
    )
    assert restored.artifact_sha256 == lowering.artifact_sha256
    assert restored.graph_weights.artifact_sha256 == (
        lowering.graph_weights.artifact_sha256
    )

    poisoned = copy.deepcopy(lowering.state_dict())
    poisoned["graph_weights"]["input_factor"][0, 0] += 1.0
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        ModalGeneratorLowering.from_state_dict(poisoned)

    fragments = _fragment_plan()
    basis = _basis(fragments)
    plan, _ = _coordinate_plan(basis, fragments)
    other_basis = replace(
        basis,
        mean_bias=basis.mean_bias + 0.5,
        mean_bias_sha256="",
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="computational-mode basis mismatch"):
        lower_coordinate_modal_generator(plan, other_basis, fragments)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("source_model_sha256", "source model mismatch"),
        ("fit_split_sha256", "fit split mismatch"),
        ("fisher_coupling_sha256", "Fisher coupling mismatch"),
        (
            "parameter_cluster_fragment_sha256",
            "parameter-cluster fragment mismatch",
        ),
    ),
)
def test_lowering_rejects_provenance_mismatch(field: str, message: str) -> None:
    fragments = _fragment_plan()
    basis = _basis(fragments)
    plan, _ = _coordinate_plan(basis, fragments)
    state = plan.binding.state_dict()
    state[field] = "0" * 64
    state["artifact_sha256"] = ""
    changed_binding = type(plan.binding)(**state)
    changed = replace(
        plan,
        binding=changed_binding,
        artifact_sha256="",
    )

    with pytest.raises(ValueError, match=message):
        lower_coordinate_modal_generator(changed, basis, fragments)
