from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    fit_computational_mode_rate_curve,
)
from fisher_graph.gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from fisher_graph.modal_generator_graph import (
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    StateConditionedModalGeneratorInteraction,
)
from fisher_graph.modal_generator_lowering import (
    ModalGeneratorLowering,
    lower_coordinate_modal_generator,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorPlan,
    fit_modal_generator_rate_curve,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from test_gemma3_modal_generator_executor import _adapter, _batch


_CLUSTER_SHA = "9" * 64
_FISHER_SHA = "c" * 64
_CATALOG_SHA = "b" * 64
_FIT_SHA = "f" * 64
_EVAL_SHA = "e" * 64
_INPUT_CATALOG_SHA = "1" * 64


@dataclass(frozen=True)
class _Fixture:
    adapter: object
    fragment_plan: ParameterClusterLayerFragmentPlan
    lowerings: tuple[ModalGeneratorLowering, ...]
    graph_plan: ModalGeneratorGraphPlan
    executor: Gemma3ModalGeneratorGraphExecutor


@dataclass(frozen=True)
class _AdditiveFixture:
    source: _Fixture
    lowerings: tuple[ModalGeneratorLowering, ...]
    graph_plan: ModalGeneratorGraphPlan
    executor: Gemma3ModalGeneratorGraphExecutor


def _fragment_plan(
    adapter,
    *,
    full_layer: bool,
) -> ParameterClusterLayerFragmentPlan:
    model_sha256 = adapter.model_fingerprint()
    if full_layer:
        specifications = (
            (0, 0, (0, 1, 2)),
            (1, 0, (3, 4, 5)),
        )
    else:
        specifications = (
            (0, 0, (0, 1)),
            (1, 0, (2,)),
            (2, 2, (0, 1)),
            (3, 2, (2, 3, 4)),
        )
    fragments = []
    next_group = 0
    for cluster_id, ordinal, channels in specifications:
        layer = adapter.layers[ordinal]
        transformer = layer.transformer
        assert transformer is not None
        assert transformer.operator_sites is not None
        input_site, output_site = _layer_sites(adapter, ordinal)
        groups = tuple(range(next_group, next_group + len(channels)))
        next_group += len(channels)
        fragments.append(
            ParameterClusterLayerFragment(
                cluster_id=cluster_id,
                layer_ordinal=ordinal,
                layer_id=layer.id,
                activation_site=(
                    transformer.operator_sites.feed_forward_down_input
                ),
                input_site=input_site,
                output_site=output_site,
                input_catalog_sha256=_INPUT_CATALOG_SHA,
                input_width=4,
                output_width=4,
                group_indices=groups,
                channel_indices=channels,
                fisher_ranks=groups,
                axial_orientations=tuple(1 for _ in channels),
                native_parameter_count=(
                    len(channels) * 3 * layer.residual_width
                ),
                fisher_mass=float(cluster_id + 1),
                source_cluster_plan_sha256=_CLUSTER_SHA,
                source_fisher_coupling_sha256=_FISHER_SHA,
                parameter_catalog_sha256=_CATALOG_SHA,
                source_model_sha256=model_sha256,
            )
        )
    return ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=_CLUSTER_SHA,
        source_fisher_coupling_sha256=_FISHER_SHA,
        parameter_catalog_sha256=_CATALOG_SHA,
        source_model_sha256=model_sha256,
        cluster_count=len(fragments),
        source_group_count=18,
        assigned_group_count=next_group,
        assigned_native_parameter_count=sum(
            fragment.native_parameter_count for fragment in fragments
        ),
        fragments=tuple(fragments),
    )


def _layer_sites(adapter, ordinal: int) -> tuple[str, str]:
    transformer = adapter.layers[ordinal].transformer
    assert transformer is not None
    stage = next(
        stage for stage in transformer.stages if stage.kind == "feed_forward"
    )
    return stage.normalized_input_site, stage.operator_output_site


def _lowering(
    adapter,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    fragment: ParameterClusterLayerFragment,
) -> ModalGeneratorLowering:
    input_site, output_site = _layer_sites(
        adapter,
        fragment.layer_ordinal,
    )
    generator = torch.Generator().manual_seed(10_000 + fragment.cluster_id)
    raw_decoder = torch.randn(
        4,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    decoder_columns, _ = torch.linalg.qr(raw_decoder, mode="reduced")
    fit_coordinates = torch.randn(
        32,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    eval_coordinates = torch.randn(
        12,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    mean = torch.randn(4, generator=generator, dtype=torch.float64)
    basis_binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind="layer_fragment",
        output_site=output_site,
        source_model_sha256=adapter.model_fingerprint(),
        parameter_catalog_sha256=_CATALOG_SHA,
        fisher_coupling_sha256=_FISHER_SHA,
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256=_FIT_SHA,
        eval_split_sha256=_EVAL_SHA,
    )
    basis = fit_computational_mode_rate_curve(
        fit_coordinates @ decoder_columns.T + mean,
        torch.ones(32, dtype=torch.float64),
        eval_coordinates @ decoder_columns.T + mean,
        torch.ones(12, dtype=torch.float64),
        (2,),
        binding=basis_binding,
        selection_rule="fixed_rank",
        selected_rank=2,
    ).selected_basis

    fit_inputs = torch.randn(
        40,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    eval_inputs = torch.randn(
        14,
        4,
        generator=generator,
        dtype=torch.float64,
    )
    coefficient = torch.randn(
        4,
        2,
        generator=generator,
        dtype=torch.float64,
    )
    bias = torch.randn(2, generator=generator, dtype=torch.float64)
    fit_targets = fit_inputs @ coefficient + bias
    eval_targets = eval_inputs @ coefficient + bias
    generator_binding = ModalGeneratorBinding.create(
        generator_id=(
            f"layer.{fragment.layer_ordinal}.cluster."
            f"{fragment.cluster_id}.coordinates"
        ),
        input_kind="native_layer_input",
        input_site=input_site,
        output_site=output_site,
        source_model_sha256=adapter.model_fingerprint(),
        input_catalog_sha256=_INPUT_CATALOG_SHA,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=fragment_plan.artifact_sha256,
        fit_split_sha256=_FIT_SHA,
        eval_split_sha256=_EVAL_SHA,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=_FISHER_SHA,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    coordinate_plan = fit_modal_generator_rate_curve(
        fit_inputs,
        fit_targets,
        torch.ones(40, dtype=torch.float64),
        eval_inputs,
        eval_targets,
        (2,),
        binding=generator_binding,
        fisher_weights_eval=torch.ones(14, dtype=torch.float64),
        fit_intercept=True,
        selection_rule="fixed_rank",
        selected_rank=2,
    ).selected_plan
    return lower_coordinate_modal_generator(
        coordinate_plan,
        basis,
        fragment_plan,
    )


def _relocate_lowering_output(
    lowering: ModalGeneratorLowering,
    output_site: str,
) -> ModalGeneratorLowering:
    """Rebind one frozen lowering without changing its executable tensors."""

    source_basis = lowering.computational_mode_basis
    source_mode_binding = source_basis.binding
    relocated_mode_binding = ComputationalModeBinding.create(
        mode_set_id=source_mode_binding.mode_set_id,
        source_kind="relocated_layer_fragment",
        output_site=output_site,
        source_model_sha256=source_mode_binding.source_model_sha256,
        parameter_catalog_sha256=(
            source_mode_binding.parameter_catalog_sha256
        ),
        fisher_coupling_sha256=source_mode_binding.fisher_coupling_sha256,
        parameter_cluster_sha256=(
            source_mode_binding.parameter_cluster_sha256
        ),
        fit_split_sha256=source_mode_binding.fit_split_sha256,
        eval_split_sha256=source_mode_binding.eval_split_sha256,
    )
    relocated_basis = ComputationalModeBasis(
        binding=relocated_mode_binding,
        config=source_basis.config,
        rank=source_basis.rank,
        mean_bias=source_basis.mean_bias,
        encoder_basis=source_basis.encoder_basis,
    )
    source_plan = lowering.coordinate_generator_plan
    source_binding = source_plan.binding
    relocated_binding = ModalGeneratorBinding.create(
        generator_id=source_binding.generator_id,
        input_kind=source_binding.input_kind,
        input_site=source_binding.input_site,
        output_site=output_site,
        source_model_sha256=source_binding.source_model_sha256,
        input_catalog_sha256=source_binding.input_catalog_sha256,
        output_catalog_sha256=relocated_basis.artifact_sha256,
        cluster_plan_sha256=source_binding.cluster_plan_sha256,
        fit_split_sha256=source_binding.fit_split_sha256,
        eval_split_sha256=source_binding.eval_split_sha256,
        target_kind="relocated_computational_mode_coordinates",
        fisher_coupling_sha256=source_binding.fisher_coupling_sha256,
        computational_mode_basis_sha256=relocated_basis.artifact_sha256,
        parameter_cluster_fragment_sha256=(
            source_binding.parameter_cluster_fragment_sha256
        ),
        source_generator_plan_sha256=source_plan.artifact_sha256,
    )
    relocated_plan = ModalGeneratorPlan(
        binding=relocated_binding,
        config=source_plan.config,
        factors=source_plan.factors,
        parameter_count=source_plan.parameter_count,
        macs_per_token=source_plan.macs_per_token,
    )
    result = lower_coordinate_modal_generator(
        relocated_plan,
        relocated_basis,
        lowering.fragment_plan,
    )
    assert result.computational_mode_basis.mean_bias_sha256 == (
        source_basis.mean_bias_sha256
    )
    assert result.computational_mode_basis.decoder_basis_sha256 == (
        source_basis.decoder_basis_sha256
    )
    return result


def _relocate_lowering_as_zero_mean_residual(
    lowering: ModalGeneratorLowering,
    output_site: str,
) -> ModalGeneratorLowering:
    """Relocate one lowering while removing only its decoded basis mean."""

    source_basis = lowering.computational_mode_basis
    source_mode_binding = source_basis.binding
    relocated_mode_binding = ComputationalModeBinding.create(
        mode_set_id=source_mode_binding.mode_set_id,
        source_kind="relocated_layer_fragment",
        output_site=output_site,
        source_model_sha256=source_mode_binding.source_model_sha256,
        parameter_catalog_sha256=(
            source_mode_binding.parameter_catalog_sha256
        ),
        fisher_coupling_sha256=source_mode_binding.fisher_coupling_sha256,
        parameter_cluster_sha256=(
            source_mode_binding.parameter_cluster_sha256
        ),
        fit_split_sha256=source_mode_binding.fit_split_sha256,
        eval_split_sha256=source_mode_binding.eval_split_sha256,
    )
    relocated_basis = ComputationalModeBasis(
        binding=relocated_mode_binding,
        config=source_basis.config,
        rank=source_basis.rank,
        mean_bias=torch.zeros_like(source_basis.mean_bias),
        encoder_basis=source_basis.encoder_basis,
    )
    source_plan = lowering.coordinate_generator_plan
    source_binding = source_plan.binding
    relocated_binding = ModalGeneratorBinding.create(
        generator_id=source_binding.generator_id,
        input_kind=source_binding.input_kind,
        input_site=source_binding.input_site,
        output_site=output_site,
        source_model_sha256=source_binding.source_model_sha256,
        input_catalog_sha256=source_binding.input_catalog_sha256,
        output_catalog_sha256=relocated_basis.artifact_sha256,
        cluster_plan_sha256=source_binding.cluster_plan_sha256,
        fit_split_sha256=source_binding.fit_split_sha256,
        eval_split_sha256=source_binding.eval_split_sha256,
        target_kind="relocated_computational_mode_coordinates",
        fisher_coupling_sha256=source_binding.fisher_coupling_sha256,
        computational_mode_basis_sha256=relocated_basis.artifact_sha256,
        parameter_cluster_fragment_sha256=(
            source_binding.parameter_cluster_fragment_sha256
        ),
        source_generator_plan_sha256=source_plan.artifact_sha256,
    )
    relocated_plan = ModalGeneratorPlan(
        binding=relocated_binding,
        config=source_plan.config,
        factors=source_plan.factors,
        parameter_count=source_plan.parameter_count,
        macs_per_token=source_plan.macs_per_token,
    )
    result = lower_coordinate_modal_generator(
        relocated_plan,
        relocated_basis,
        lowering.fragment_plan,
    )
    assert not bool(
        torch.count_nonzero(
            result.computational_mode_basis.mean_bias
        )
    )
    assert result.graph_weights.output_bias is None
    # The coordinate-generator intercept is a learned residual offset, not a
    # forbidden decoded basis mean, and must survive this relocation.
    assert result.graph_weights.latent_bias is not None
    return result


def _relocated_graph(
    source: _Fixture,
    lowerings: tuple[ModalGeneratorLowering, ...],
) -> ModalGeneratorGraphPlan:
    source_nodes_by_fragment = {
        lowering.selected_fragment_sha256: node
        for node, lowering in zip(
            source.graph_plan.nodes,
            source.lowerings,
            strict=True,
        )
    }
    nodes = tuple(
        lowering.to_graph_node(
            name=(
                source_nodes_by_fragment[
                    lowering.selected_fragment_sha256
                ].name
            ),
            causal_order=(
                source_nodes_by_fragment[
                    lowering.selected_fragment_sha256
                ].causal_order
            ),
        )
        for lowering in lowerings
    )
    names = {node.name for node in nodes}
    interactions = tuple(
        edge
        for edge in source.graph_plan.interactions
        if edge.source_node in names and edge.target_node in names
    )
    return ModalGeneratorGraphPlan(
        model_fingerprint=source.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            source.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=nodes,
        interactions=interactions,
    )


def _additive_fixture(*, scale: float = 1.0) -> _AdditiveFixture:
    source = _fixture(full_layer=True)
    transformer = source.adapter.layers[0].transformer
    assert transformer is not None
    stage = next(
        value for value in transformer.stages if value.kind == "feed_forward"
    )
    lowerings = tuple(
        _relocate_lowering_as_zero_mean_residual(
            lowering,
            stage.delta_site,
        )
        for lowering in source.lowerings
    )
    graph_plan = _relocated_graph(source, lowerings)
    executor = Gemma3ModalGeneratorGraphExecutor(
        source.adapter,
        source.graph_plan,
        source.lowerings,
        additive_post_feedforward_graph_plan=graph_plan,
        additive_post_feedforward_lowerings=lowerings,
        additive_post_feedforward_scale=scale,
    )
    return _AdditiveFixture(
        source=source,
        lowerings=lowerings,
        graph_plan=graph_plan,
        executor=executor,
    )


def _post_delta_fixture(
    *,
    full_layer: bool = False,
    model_dtype: torch.dtype = torch.float32,
) -> _Fixture:
    source = _fixture(full_layer=full_layer, model_dtype=model_dtype)
    target_ordinal = 0 if full_layer else 2
    transformer = source.adapter.layers[target_ordinal].transformer
    assert transformer is not None
    stage = next(
        value for value in transformer.stages if value.kind == "feed_forward"
    )
    relocated_lowerings = tuple(
        (
            _relocate_lowering_output(lowering, stage.delta_site)
            if next(
                fragment
                for fragment in lowering.fragment_plan.fragments
                if fragment.artifact_sha256
                == lowering.selected_fragment_sha256
            ).layer_ordinal
            == target_ordinal
            else lowering
        )
        for lowering in source.lowerings
    )
    relocated_by_weight = {
        lowering.coordinate_generator_plan.binding.generator_id: lowering
        for lowering in relocated_lowerings
    }
    nodes = tuple(
        relocated_by_weight[
            source.lowerings[index].coordinate_generator_plan.binding.generator_id
        ].to_graph_node(
            name=node.name,
            causal_order=node.causal_order,
        )
        for index, node in enumerate(source.graph_plan.nodes)
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=source.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            source.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=nodes,
        interactions=source.graph_plan.interactions,
    )
    executor = Gemma3ModalGeneratorGraphExecutor(
        source.adapter,
        graph,
        relocated_lowerings,
        post_feedforward_delta_layer_ordinals=(target_ordinal,),
    )
    return _Fixture(
        adapter=source.adapter,
        fragment_plan=source.fragment_plan,
        lowerings=relocated_lowerings,
        graph_plan=graph,
        executor=executor,
    )


def _fixture(
    *,
    full_layer: bool = False,
    model_dtype: torch.dtype = torch.float32,
) -> _Fixture:
    adapter = _adapter(seed=9_001 if full_layer else 9_000)
    adapter.module.to(dtype=model_dtype)
    fragment_plan = _fragment_plan(adapter, full_layer=full_layer)
    lowerings = tuple(
        _lowering(adapter, fragment_plan, fragment)
        for fragment in fragment_plan.fragments
    )
    if full_layer:
        names = ("l0.a", "l0.b")
    else:
        names = ("l0.a", "l0.b", "l2.c", "l2.d")
    nodes = tuple(
        lowering.to_graph_node(name=name, causal_order=index)
        for index, (name, lowering) in enumerate(zip(names, lowerings))
    )
    if full_layer:
        edge_pairs = (("l0.a", "l0.b"),)
    else:
        edge_pairs = (
            ("l0.a", "l2.c"),
            ("l0.a", "l2.d"),
            ("l0.b", "l2.c"),
        )
    edges = tuple(
        ModalGeneratorInteraction(
            source_node=source,
            target_node=target,
            message_matrix=torch.tensor(
                ((0.2, -0.1), (0.3, 0.4)),
                dtype=torch.float64,
            ),
            message_bias=torch.tensor(
                (0.05, -0.025),
                dtype=torch.float64,
            ),
        )
        for source, target in edge_pairs
    )
    graph_plan = ModalGeneratorGraphPlan(
        model_fingerprint=adapter.model_fingerprint(),
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        nodes=nodes,
        interactions=edges,
    )
    executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        graph_plan,
        lowerings,
    )
    return _Fixture(
        adapter=adapter,
        fragment_plan=fragment_plan,
        lowerings=lowerings,
        graph_plan=graph_plan,
        executor=executor,
    )


def _module_storage(module: nn.Module) -> set[int]:
    return {
        value.untyped_storage().data_ptr()
        for value in module.state_dict().values()
        if value.numel()
    }


def _plan_storage(plan: ModalGeneratorGraphPlan) -> set[int]:
    tensors: list[Tensor] = []
    for node in plan.nodes:
        weights = node.weights
        tensors.extend(
            tensor
            for tensor in (
                weights.input_factor,
                weights.state_factor,
                weights.latent_bias,
                weights.output_factor,
                weights.output_bias,
            )
            if tensor is not None
        )
    for edge in plan.interactions:
        tensors.extend(
            tensor
            for tensor in (
                edge.message_matrix,
                edge.message_bias,
                getattr(edge, "gate_weight", None),
                getattr(edge, "gate_bias", None),
                getattr(edge, "quadratic_left", None),
                getattr(edge, "quadratic_right", None),
                getattr(edge, "quadratic_output", None),
            )
            if tensor is not None
        )
    return {
        tensor.untyped_storage().data_ptr()
        for tensor in tensors
        if tensor.numel()
    }


def test_incremental_runtime_matches_generic_coordinate_graph() -> None:
    fixture = _fixture()
    inputs = {
        boundary: torch.randn(2, 3, width)
        for boundary, width in fixture.graph_plan.input_boundary_widths.items()
    }
    generic = ModalGeneratorGraphExecutor(fixture.graph_plan).execute(
        inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
    )
    incremental = fixture.executor.execute_graph_inputs(
        inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
    )

    assert incremental.traversal_order == fixture.graph_plan.traversal_order
    assert incremental.modal_states is not None
    assert incremental.edge_messages is not None
    assert set(incremental.outputs) == set(generic.outputs)
    assert set(incremental.modal_states) == set(generic.modal_states or ())
    assert set(incremental.edge_messages) == set(generic.edge_messages or ())
    for name, expected in generic.outputs.items():
        torch.testing.assert_close(incremental.outputs[name], expected)
    for name, expected in (generic.modal_states or {}).items():
        torch.testing.assert_close(incremental.modal_states[name], expected)
    for name, expected in (generic.edge_messages or {}).items():
        torch.testing.assert_close(incremental.edge_messages[name], expected)

    # Root graph states are exactly the generated computational coordinates,
    # not a private reduced-regression latent.
    for node_name, lowering in zip(("l0.a", "l0.b"), fixture.lowerings[:2]):
        boundary = lowering.coordinate_generator_plan.binding.input_site
        expected = lowering.coordinate_generator_plan.apply(inputs[boundary])
        torch.testing.assert_close(
            incremental.modal_states[node_name],
            expected.to(dtype=inputs[boundary].dtype),
        )
    assert fixture.executor.peak_live_modal_width == 6


def test_incremental_runtime_matches_state_conditioned_polynomial_graph() -> None:
    fixture = _fixture()
    plan = ModalGeneratorGraphPlan(
        model_fingerprint=fixture.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            fixture.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=fixture.graph_plan.nodes,
        interactions=(
            StateConditionedModalGeneratorInteraction(
                source_node="l0.a",
                target_node="l2.c",
                routing_group="l0.a.flow",
                message_matrix=torch.tensor(
                    ((0.2, -0.1), (0.3, 0.4)),
                    dtype=torch.float64,
                ),
                message_bias=torch.tensor((0.05, -0.025), dtype=torch.float64),
                gate_weight=torch.tensor((1.0, -0.5), dtype=torch.float64),
                gate_bias=torch.tensor((0.1,), dtype=torch.float64),
                quadratic_left=torch.tensor(((0.2,), (0.3,)), dtype=torch.float64),
                quadratic_right=torch.tensor(((-0.4,), (0.1,)), dtype=torch.float64),
                quadratic_output=torch.tensor(((0.5, -0.25),), dtype=torch.float64),
                top_k=1,
            ),
            StateConditionedModalGeneratorInteraction(
                source_node="l0.a",
                target_node="l2.d",
                routing_group="l0.a.flow",
                message_matrix=torch.tensor(
                    ((-0.15, 0.2), (0.1, -0.3)),
                    dtype=torch.float64,
                ),
                message_bias=torch.tensor((-0.02, 0.04), dtype=torch.float64),
                gate_weight=torch.tensor((-1.0, 0.5), dtype=torch.float64),
                gate_bias=torch.tensor((-0.1,), dtype=torch.float64),
                quadratic_left=torch.tensor(((0.1,), (-0.2,)), dtype=torch.float64),
                quadratic_right=torch.tensor(((0.3,), (0.25,)), dtype=torch.float64),
                quadratic_output=torch.tensor(((-0.4, 0.2),), dtype=torch.float64),
                top_k=1,
            ),
        ),
    )
    executor = Gemma3ModalGeneratorGraphExecutor(
        fixture.adapter,
        plan,
        fixture.lowerings,
    )
    generator = torch.Generator().manual_seed(91_337)
    inputs = {
        boundary: torch.randn(2, 3, width, generator=generator)
        for boundary, width in plan.input_boundary_widths.items()
    }

    generic = ModalGeneratorGraphExecutor(plan).execute(
        inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )
    incremental = executor.execute_graph_inputs(
        inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
        capture_routing=True,
    )

    assert generic.routing_weights is not None
    assert generic.evaluated_edge_rows is not None
    assert sum(generic.evaluated_edge_rows.values()) == 6
    for actual, expected in (
        (incremental.outputs, generic.outputs),
        (incremental.modal_states or {}, generic.modal_states or {}),
        (incremental.edge_messages or {}, generic.edge_messages or {}),
        (incremental.routing_weights or {}, generic.routing_weights or {}),
    ):
        assert set(actual) == set(expected)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value)
    assert executor.graph_runtime_parameter_count == plan.parameter_count
    assert executor.peak_live_modal_width == 2
    assert incremental.evaluated_edge_rows == generic.evaluated_edge_rows
    assert not (_plan_storage(plan) & _module_storage(executor.graph_runtime))

    batch = _batch()
    model_execution = executor.run(
        batch.model_inputs,
        capture_routing=True,
    )
    assert model_execution.graph_execution.routing_weights is not None
    expected_executed_macs = model_execution.valid_tokens * (
        plan.accounting.node_macs_per_token
        + plan.conditional_routing_macs_per_token
    )
    for edge in plan.interactions:
        assert isinstance(edge, StateConditionedModalGeneratorInteraction)
        selected_valid_rows = int(
            (
                model_execution.graph_execution.routing_weights[edge.key][
                    batch.valid_positions
                ]
                > 0
            ).sum().item()
        )
        expected_executed_macs += (
            selected_valid_rows * edge.message_macs_per_selected_token
        )
    assert model_execution.logical_modal_graph_macs == (
        model_execution.valid_tokens * plan.macs_per_token
    )
    assert (
        model_execution.logical_executed_modal_graph_macs
        == expected_executed_macs
        < model_execution.logical_modal_graph_macs
    )
    assert model_execution.net_logical_macs_saved == (
        model_execution.logical_linear_macs_native_removed
        - expected_executed_macs
    )
    uncaptured_model_execution = executor.run(batch.model_inputs)
    assert uncaptured_model_execution.graph_execution.routing_weights is None
    assert (
        uncaptured_model_execution.logical_executed_modal_graph_macs
        == expected_executed_macs
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        generic_bfloat = ModalGeneratorGraphExecutor(plan).execute(
            inputs,
            capture_edge_messages=True,
            capture_routing=True,
        )
        incremental_bfloat = executor.execute_graph_inputs(
            inputs,
            capture_edge_messages=True,
            capture_routing=True,
        )
    for actual, expected in (
        (incremental_bfloat.outputs, generic_bfloat.outputs),
        (
            incremental_bfloat.edge_messages or {},
            generic_bfloat.edge_messages or {},
        ),
        (
            incremental_bfloat.routing_weights or {},
            generic_bfloat.routing_weights or {},
        ),
    ):
        assert set(actual) == set(expected)
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value, atol=0, rtol=0)


def test_bfloat16_device_routing_preserves_float32_gate_boundaries() -> None:
    fixture = _fixture(model_dtype=torch.bfloat16)
    plan = ModalGeneratorGraphPlan(
        model_fingerprint=fixture.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            fixture.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=fixture.graph_plan.nodes,
        interactions=(
            StateConditionedModalGeneratorInteraction(
                source_node="l0.a",
                target_node="l2.c",
                routing_group="precision",
                message_matrix=torch.zeros(2, 2, dtype=torch.float64),
                message_bias=torch.full((2,), 10.0, dtype=torch.float64),
                gate_weight=torch.zeros(2, dtype=torch.float64),
                gate_bias=torch.tensor((1.0,), dtype=torch.float64),
            ),
            StateConditionedModalGeneratorInteraction(
                source_node="l0.a",
                target_node="l2.d",
                routing_group="precision",
                message_matrix=torch.zeros(2, 2, dtype=torch.float64),
                message_bias=torch.full((2,), 20.0, dtype=torch.float64),
                gate_weight=torch.zeros(2, dtype=torch.float64),
                gate_bias=torch.tensor((1.001,), dtype=torch.float64),
            ),
        ),
    )
    executor = Gemma3ModalGeneratorGraphExecutor(
        fixture.adapter,
        plan,
        fixture.lowerings,
    )
    inputs = {
        boundary: torch.randn(2, 3, width, dtype=torch.bfloat16)
        for boundary, width in plan.input_boundary_widths.items()
    }
    generic = ModalGeneratorGraphExecutor(plan).execute(
        inputs,
        capture_routing=True,
    )
    device = executor.execute_graph_inputs(inputs, capture_routing=True)
    assert generic.routing_weights is not None
    assert device.routing_weights is not None
    assert all(
        runtime.gate_weight.dtype == torch.float32
        for runtime in executor.graph_runtime.edge_runtimes
    )
    for key, expected in generic.routing_weights.items():
        torch.testing.assert_close(device.routing_weights[key], expected)
    assert torch.count_nonzero(generic.routing_weights["l0.a->l2.c"]) == 0
    assert torch.count_nonzero(generic.routing_weights["l0.a->l2.d"]) == 6

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_generic = ModalGeneratorGraphExecutor(plan).execute(
            inputs,
            capture_routing=True,
        )
        autocast_device = executor.execute_graph_inputs(
            inputs,
            capture_routing=True,
        )
    assert autocast_generic.routing_weights is not None
    assert autocast_device.routing_weights is not None
    for key, expected in generic.routing_weights.items():
        torch.testing.assert_close(
            autocast_generic.routing_weights[key],
            expected,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            autocast_device.routing_weights[key],
            expected,
            atol=0,
            rtol=0,
        )


def test_device_conditional_routing_stabilizes_tiny_temperature() -> None:
    fixture = _fixture()
    plan = ModalGeneratorGraphPlan(
        model_fingerprint=fixture.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            fixture.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=fixture.graph_plan.nodes,
        interactions=tuple(
            StateConditionedModalGeneratorInteraction(
                source_node="l0.a",
                target_node=target,
                routing_group="overflow",
                message_matrix=torch.zeros(2, 2, dtype=torch.float64),
                message_bias=torch.zeros(2, dtype=torch.float64),
                gate_weight=torch.zeros(2, dtype=torch.float64),
                gate_bias=torch.tensor((bias,), dtype=torch.float64),
                temperature=1e-45,
            )
            for target, bias in (("l2.c", 1.0), ("l2.d", 0.0))
        ),
    )
    executor = Gemma3ModalGeneratorGraphExecutor(
        fixture.adapter,
        plan,
        fixture.lowerings,
    )
    inputs = {
        boundary: torch.randn(1, 2, width)
        for boundary, width in plan.input_boundary_widths.items()
    }
    generic = ModalGeneratorGraphExecutor(plan).execute(
        inputs,
        capture_routing=True,
    )
    device = executor.execute_graph_inputs(inputs, capture_routing=True)
    assert generic.routing_weights is not None
    assert device.routing_weights is not None
    for key, expected in generic.routing_weights.items():
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(device.routing_weights[key], expected)
    assert torch.count_nonzero(generic.routing_weights["l0.a->l2.c"]) == 2
    assert torch.count_nonzero(generic.routing_weights["l0.a->l2.d"]) == 0


def test_incremental_runtime_matches_generic_graph_under_cpu_autocast() -> None:
    fixture = _fixture()
    generator = torch.Generator().manual_seed(12_345)
    inputs = {
        boundary: torch.randn(2, 3, width, generator=generator)
        for boundary, width in fixture.graph_plan.input_boundary_widths.items()
    }

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        generic = ModalGeneratorGraphExecutor(fixture.graph_plan).execute(
            inputs,
            capture_modal_states=True,
            capture_edge_messages=True,
        )
        incremental = fixture.executor.execute_graph_inputs(
            inputs,
            capture_modal_states=True,
            capture_edge_messages=True,
        )

    assert generic.modal_states is not None
    assert incremental.modal_states is not None
    assert generic.edge_messages is not None
    assert incremental.edge_messages is not None
    for actual, expected in (
        (incremental.outputs, generic.outputs),
        (incremental.modal_states, generic.modal_states),
        (incremental.edge_messages, generic.edge_messages),
    ):
        assert set(actual) == set(expected)
        for name, value in expected.items():
            assert actual[name].dtype == value.dtype == torch.bfloat16
            torch.testing.assert_close(actual[name], value, atol=0, rtol=0)


def test_model_walk_conditions_accounting_aliasing_and_restoration() -> None:
    fixture = _fixture()
    adapter = fixture.adapter
    executor = fixture.executor
    batch = _batch()
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()
    source_storage = _module_storage(adapter.module)
    compiled_storage = _module_storage(executor.compiled_mlps)
    runtime_storage = _module_storage(executor.graph_runtime)

    generated = executor.run(
        batch.model_inputs,
        condition="generated",
        capture_modal_states=True,
        capture_edge_messages=True,
    )
    deletion = executor.run(
        batch.model_inputs,
        condition="deletion",
        capture_modal_states=True,
        capture_edge_messages=True,
    )

    assert not torch.equal(
        generated.model_output.logits[batch.valid_positions],
        deletion.model_output.logits[batch.valid_positions],
    )
    assert generated.graph_execution.traversal_order == (
        fixture.graph_plan.traversal_order
    )
    assert deletion.graph_execution.traversal_order == ()
    assert deletion.graph_execution.modal_states == {}
    assert deletion.graph_execution.edge_messages == {}
    assert all(
        not bool(torch.count_nonzero(values))
        for values in deletion.graph_execution.outputs.values()
    )
    assert generated.replaced_layer_count == 2
    assert generated.graph_node_count == 4
    assert generated.fragment_count == 4
    assert generated.removed_mode_count == 8
    assert generated.native_removed_learned_parameters == 8 * 3 * 4 == 96
    assert generated.modal_graph_learned_parameters == (
        fixture.graph_plan.parameter_count
    )
    assert generated.net_stored_parameter_savings == (
        96 - fixture.graph_plan.parameter_count
    )
    assert generated.candidate_whole_model_learned_parameters == (
        generated.source_whole_model_learned_parameters
        - 96
        + fixture.graph_plan.parameter_count
    )
    assert generated.valid_tokens == 6
    assert generated.logical_linear_macs_native_removed == 6 * 96
    assert generated.logical_modal_graph_macs == (
        6 * fixture.graph_plan.macs_per_token
    )
    assert generated.logical_executed_modal_graph_macs == (
        generated.logical_modal_graph_macs
    )
    assert deletion.logical_executed_modal_graph_macs == 0
    expected_graph_additions = (
        generated.valid_tokens
        * fixture.graph_plan.accounting.elementwise_additions_per_token
    )
    assert (
        fixture.graph_plan.accounting.output_accumulation_additions_per_token
        == 8
    )
    assert generated.logical_modal_graph_additions == expected_graph_additions
    assert generated.logical_executed_modal_graph_additions == (
        expected_graph_additions
    )
    assert deletion.logical_modal_graph_additions == expected_graph_additions
    assert deletion.logical_executed_modal_graph_additions == 0
    assert generated.net_logical_macs_saved == (
        generated.logical_linear_macs_native_removed
        - generated.logical_modal_graph_macs
    )
    assert deletion.net_logical_macs_saved == (
        deletion.logical_linear_macs_native_removed
    )
    assert generated.peak_live_modal_width == 6
    assert deletion.peak_live_modal_width == 0
    assert (
        generated.replacement_scope
        == "partial_native_mlp_mode_replacement"
    )
    assert executor.compiled_mlps["0"].gate_proj.out_features == 3
    assert executor.compiled_mlps["2"].gate_proj.out_features == 1
    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        source_mlps
    )
    assert adapter.model_fingerprint() == source_fingerprint
    assert not (source_storage & compiled_storage)
    assert not (source_storage & runtime_storage)
    assert not (compiled_storage & runtime_storage)
    assert not (_plan_storage(fixture.graph_plan) & runtime_storage)


def test_union_can_fully_remove_native_mlp_and_same_layer_edge_executes() -> None:
    fixture = _fixture(full_layer=True)
    executor = fixture.executor
    batch = _batch()
    compact = executor.compiled_mlps["0"]

    assert compact.is_full_native_replacement
    assert compact.gate_proj.weight.shape == (0, 4)
    assert compact.up_proj.weight.shape == (0, 4)
    assert compact.down_proj.weight.shape == (4, 0)
    assert sum(parameter.numel() for parameter in compact.parameters()) == 0
    generated = executor.run(
        batch.model_inputs,
        condition="generated",
        capture_edge_messages=True,
    )
    deletion = executor.run(batch.model_inputs, condition="deletion")
    assert generated.replacement_scope == (
        "full_native_mlp_replacement_at_selected_layers"
    )
    assert generated.removed_mode_count == 6
    assert generated.graph_execution.edge_messages is not None
    assert set(generated.graph_execution.edge_messages) == {"l0.a->l0.b"}
    assert not torch.equal(
        generated.model_output.logits[batch.valid_positions],
        deletion.model_output.logits[batch.valid_positions],
    )


def test_model_error_restores_original_mlps_and_executor_remains_usable() -> None:
    fixture = _fixture()
    adapter = fixture.adapter
    executor = fixture.executor
    source_mlps = tuple(layer.mlp for layer in adapter.module.model.layers)
    source_fingerprint = adapter.model_fingerprint()

    def fail(_module: nn.Module, _args: tuple[object, ...]) -> None:
        raise RuntimeError("sentinel incremental graph failure")

    handle = executor.compiled_mlps["2"].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="sentinel incremental"):
            executor.run(_batch().model_inputs)
    finally:
        handle.remove()

    assert tuple(layer.mlp for layer in adapter.module.model.layers) == (
        source_mlps
    )
    assert adapter.model_fingerprint() == source_fingerprint
    assert executor.run(_batch().model_inputs).graph_execution.traversal_order


def test_constructor_rejects_non_bijective_lowerings_and_runtime_drift() -> None:
    fixture = _fixture()
    bad_lowerings = (
        fixture.lowerings[0],
        fixture.lowerings[0],
        fixture.lowerings[2],
        fixture.lowerings[3],
    )
    with pytest.raises(ValueError, match="one lowering|multiple graph"):
        Gemma3ModalGeneratorGraphExecutor(
            fixture.adapter,
            fixture.graph_plan,
            bad_lowerings,
        )

    parameter = next(fixture.executor.graph_runtime.parameters())
    with torch.no_grad():
        parameter[0, 0] += 0.25
    with pytest.raises(ValueError, match="runtime coefficients drifted"):
        fixture.executor.run(_batch().model_inputs)


def test_source_model_drift_is_rejected_before_overlay() -> None:
    fixture = _fixture()
    layer_zero = fixture.adapter.module.model.layers[0]
    with torch.no_grad():
        layer_zero.mlp.gate_proj.weight[0, 0] += 0.25
    with pytest.raises(ValueError, match="model fingerprint drifted"):
        fixture.executor.run(_batch().model_inputs)


def test_validated_transaction_amortizes_checks_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    executor = fixture.executor
    original_validate = executor._validate_live_state
    validation_calls = 0

    def counted_validate() -> None:
        nonlocal validation_calls
        validation_calls += 1
        original_validate()

    monkeypatch.setattr(executor, "_validate_live_state", counted_validate)
    generator = torch.Generator().manual_seed(44_901)
    graph_inputs = {
        boundary: torch.randn(2, 3, width, generator=generator)
        for boundary, width in fixture.graph_plan.input_boundary_widths.items()
    }
    batch = _batch()

    with executor.validated_transaction():
        executor.execute_graph_inputs(graph_inputs)
        executor.run(batch.model_inputs, condition="generated")
        executor.run(batch.model_inputs, condition="deletion")

    assert validation_calls == 2

    validation_calls = 0
    executor.execute_graph_inputs(graph_inputs)
    assert validation_calls == 1
    executor.run(batch.model_inputs)
    assert validation_calls == 3


def test_validated_transaction_is_nesting_safe_and_validates_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    executor = fixture.executor
    original_validate = executor._validate_live_state
    validation_calls = 0

    def counted_validate() -> None:
        nonlocal validation_calls
        validation_calls += 1
        original_validate()

    monkeypatch.setattr(executor, "_validate_live_state", counted_validate)

    with pytest.raises(RuntimeError, match="sentinel transaction failure"):
        with executor.validated_transaction():
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with executor.validated_transaction():
                    raise AssertionError("nested transaction became active")
            raise RuntimeError("sentinel transaction failure")

    assert validation_calls == 2
    assert executor._validated_transaction_active is False
    with executor.validated_transaction():
        pass
    assert validation_calls == 4

    executor._active = True
    try:
        with pytest.raises(RuntimeError, match="during graph execution"):
            with executor.validated_transaction():
                raise AssertionError("reentrant transaction became active")
    finally:
        executor._active = False
    assert validation_calls == 4


def test_validated_transaction_detects_exit_drift_and_clears_state() -> None:
    fixture = _fixture()
    executor = fixture.executor
    parameter = next(executor.graph_runtime.parameters())
    original = parameter.detach().clone()

    with pytest.raises(ValueError, match="runtime coefficients drifted"):
        with executor.validated_transaction():
            with torch.no_grad():
                parameter[0, 0] += 0.25

    assert executor._validated_transaction_active is False
    with torch.no_grad():
        parameter.copy_(original)
    with executor.validated_transaction():
        executor.run(_batch().model_inputs)


def test_generated_callback_overlay_repeats_full_cross_layer_sessions() -> None:
    fixture = _fixture()
    executor = fixture.executor
    batch = _batch()
    source_mlps = tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    )
    source_fingerprint = fixture.adapter.model_fingerprint()
    expected = executor.run(batch.model_inputs).model_output.logits.detach()
    assert executor.affected_layer_ordinals == (0, 2)
    assert executor.lowering_artifact_sha256s == tuple(
        lowering.artifact_sha256 for lowering in fixture.lowerings
    )

    def consume() -> tuple[Tensor, Tensor]:
        outputs = []
        for _ in range(2):
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            outputs.append(
                fixture.adapter.module(**call_inputs).logits.detach().clone()
            )
        return outputs[0], outputs[1]

    first, second = executor.run_with_generated_overlay(
        consume,
        expected_forward_calls=2,
    )

    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    ) == source_mlps
    assert fixture.adapter.model_fingerprint() == source_fingerprint


def test_compact_mlp_row_replay_is_chunked_authenticated_and_graph_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    executor = fixture.executor
    compact = executor.compiled_mlps["0"]
    source_fingerprint = fixture.adapter.model_fingerprint()
    compact_fingerprint = module_state_fingerprint(compact)
    generator = torch.Generator().manual_seed(99041)
    inputs = torch.randn(4101, 4, generator=generator, dtype=torch.float64)
    observed_chunk_rows: list[int] = []

    def observe_chunk(
        _module: nn.Module,
        arguments: tuple[Tensor, ...],
        _output: Tensor,
    ) -> None:
        observed_chunk_rows.append(int(arguments[0].shape[0]))

    def forbid_graph_session(*_args: object, **_kwargs: object) -> object:
        pytest.fail("compact replay must not start a graph session")

    monkeypatch.setattr(type(executor.graph_runtime), "start", forbid_graph_session)
    handle = compact.register_forward_hook(observe_chunk)
    try:
        output = executor.execute_compact_mlp_rows(0, inputs)
    finally:
        handle.remove()

    runtime_weight = compact.gate_proj.weight
    with torch.no_grad():
        expected = compact(
            inputs.to(
                device=runtime_weight.device,
                dtype=runtime_weight.dtype,
            )
        ).to(device="cpu", dtype=torch.float64)
    assert observed_chunk_rows == [2048, 2048, 5]
    assert output.device.type == "cpu"
    assert output.dtype == torch.float64
    torch.testing.assert_close(output, expected)
    assert fixture.adapter.model_fingerprint() == source_fingerprint
    assert module_state_fingerprint(compact) == compact_fingerprint


def test_compact_mlp_row_replay_rejects_invalid_layer_and_rows() -> None:
    fixture = _fixture()
    executor = fixture.executor
    good = torch.ones((2, 4), dtype=torch.float64)

    with pytest.raises(ValueError, match="not part of the compact"):
        executor.execute_compact_mlp_rows(1, good)
    with pytest.raises(ValueError, match="row input width drifted"):
        executor.execute_compact_mlp_rows(
            0,
            torch.ones((2, 3), dtype=torch.float64),
        )
    invalid = good.clone()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite nonempty"):
        executor.execute_compact_mlp_rows(0, invalid)


def test_generated_callback_overlay_requires_synchronous_consumption_and_restores() -> None:
    fixture = _fixture()
    executor = fixture.executor
    source_mlps = tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    )

    with pytest.raises(RuntimeError, match="executed exactly 1 times"):
        executor.run_with_generated_overlay(lambda: iter((_batch(),)))

    batch = _batch()

    def deferred_after_forward():
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        fixture.adapter.module(**call_inputs)
        return iter((_batch(),))

    with pytest.raises(TypeError, match="fully materialized"):
        executor.run_with_generated_overlay(deferred_after_forward)

    assert tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    ) == source_mlps
    assert executor.run(_batch().model_inputs).graph_execution.traversal_order


def test_generated_callback_overlay_restores_after_callback_error() -> None:
    fixture = _fixture()
    executor = fixture.executor
    source_mlps = tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    )

    def fail() -> None:
        raise RuntimeError("sentinel callback failure")

    with pytest.raises(RuntimeError, match="sentinel callback failure"):
        executor.run_with_generated_overlay(fail)

    assert tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    ) == source_mlps
    assert executor.run(_batch().model_inputs).graph_execution.traversal_order


def test_post_feedforward_delta_requires_relocated_layer_bindings() -> None:
    legacy = _fixture()
    relocated = _post_delta_fixture()
    layer_two_nodes = tuple(
        node
        for node, lowering in zip(
            relocated.graph_plan.nodes,
            relocated.lowerings,
            strict=True,
        )
        if next(
            fragment
            for fragment in lowering.fragment_plan.fragments
            if fragment.artifact_sha256 == lowering.selected_fragment_sha256
        ).layer_ordinal
        == 2
    )
    assert layer_two_nodes
    assert all(
        node.output_boundary == "layer.2.mlp.delta"
        for node in layer_two_nodes
    )
    assert all(
        node.output_boundary == "layer.0.mlp.operator_output"
        for node in relocated.graph_plan.nodes
        if node not in layer_two_nodes
    )
    assert relocated.executor.post_feedforward_delta_layer_ordinals == (2,)

    with pytest.raises(ValueError, match="fragment sites"):
        Gemma3ModalGeneratorGraphExecutor(
            relocated.adapter,
            relocated.graph_plan,
            relocated.lowerings,
        )
    with pytest.raises(ValueError, match="fragment sites"):
        Gemma3ModalGeneratorGraphExecutor(
            legacy.adapter,
            legacy.graph_plan,
            legacy.lowerings,
            post_feedforward_delta_layer_ordinals=(2,),
        )
    for invalid, message in (
        ((2, 2), "unique and in causal order"),
        ((True,), "unique and in causal order"),
        ((1,), "belong to the graph"),
    ):
        with pytest.raises(ValueError, match=message):
            Gemma3ModalGeneratorGraphExecutor(
                legacy.adapter,
                legacy.graph_plan,
                legacy.lowerings,
                post_feedforward_delta_layer_ordinals=invalid,
            )


def test_explicit_empty_post_delta_policy_preserves_legacy_execution() -> None:
    fixture = _fixture()
    explicit = Gemma3ModalGeneratorGraphExecutor(
        fixture.adapter,
        fixture.graph_plan,
        fixture.lowerings,
        post_feedforward_delta_layer_ordinals=(),
    )
    batch = _batch()
    implicit_result = fixture.executor.run(batch.model_inputs)
    explicit_result = explicit.run(batch.model_inputs)

    assert explicit.post_feedforward_delta_layer_ordinals == ()
    assert explicit.graph_plan.artifact_sha256 == (
        fixture.executor.graph_plan.artifact_sha256
    )
    assert explicit.lowering_artifact_sha256s == (
        fixture.executor.lowering_artifact_sha256s
    )
    torch.testing.assert_close(
        explicit_result.model_output.logits,
        implicit_result.model_output.logits,
        rtol=0,
        atol=0,
    )
    assert explicit_result.candidate_whole_model_learned_parameters == (
        implicit_result.candidate_whole_model_learned_parameters
    )
    assert explicit_result.logical_modal_graph_macs == (
        implicit_result.logical_modal_graph_macs
    )


def test_post_feedforward_delta_applies_only_on_declared_layer() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    batch = _batch()
    source_mlps = tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    )
    source_norms = tuple(
        layer.post_feedforward_layernorm
        for layer in fixture.adapter.module.model.layers
    )
    source_fingerprint = fixture.adapter.model_fingerprint()
    stages = {
        ordinal: next(
            stage
            for stage in fixture.adapter.layers[ordinal].transformer.stages
            if stage.kind == "feed_forward"
        )
        for ordinal in (0, 2)
    }
    capture_sites = tuple(
        site
        for ordinal in (0, 2)
        for site in (
            stages[ordinal].normalized_input_site,
            stages[ordinal].operator_output_site,
            stages[ordinal].delta_site,
        )
    )

    def capture():
        return fixture.adapter.forward(
            batch.model_inputs,
            capture_sites=capture_sites,
        )

    traced = executor.run_with_generated_overlay(capture)
    graph = executor.execute_graph_inputs(
        {
            stages[ordinal].normalized_input_site: traced.activations[
                stages[ordinal].normalized_input_site
            ]
            for ordinal in (0, 2)
        }
    )
    with torch.no_grad():
        layer_zero_input = traced.activations[stages[0].normalized_input_site]
        layer_zero_retained = executor.compiled_mlps["0"](layer_zero_input)
        layer_zero_raw = (
            layer_zero_retained
            + graph.outputs[stages[0].operator_output_site]
        )
        layer_zero_delta = source_norms[0](layer_zero_raw)
        layer_two_input = traced.activations[stages[2].normalized_input_site]
        layer_two_retained = executor.compiled_mlps["2"](layer_two_input)
        layer_two_delta = (
            source_norms[2](layer_two_retained)
            + graph.outputs[stages[2].delta_site]
        )
        wrong_pre_norm_delta = source_norms[2](
            layer_two_retained + graph.outputs[stages[2].delta_site]
        )

    torch.testing.assert_close(
        traced.activations[stages[0].operator_output_site],
        layer_zero_raw,
    )
    torch.testing.assert_close(
        traced.activations[stages[0].delta_site],
        layer_zero_delta,
    )
    torch.testing.assert_close(
        traced.activations[stages[2].operator_output_site],
        layer_two_retained,
    )
    torch.testing.assert_close(
        traced.activations[stages[2].delta_site],
        layer_two_delta,
    )
    assert not torch.allclose(layer_two_delta, wrong_pre_norm_delta)
    ordinary = executor.run(batch.model_inputs)
    torch.testing.assert_close(traced.logits, ordinary.model_output.logits)
    assert ordinary.modal_graph_learned_parameters == (
        fixture.graph_plan.parameter_count
    )
    assert tuple(layer.mlp for layer in fixture.adapter.module.model.layers) == (
        source_mlps
    )
    assert tuple(
        layer.post_feedforward_layernorm
        for layer in fixture.adapter.module.model.layers
    ) == source_norms
    assert fixture.adapter.model_fingerprint() == source_fingerprint


def test_post_feedforward_delta_callback_repeats_without_pending_state() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    batch = _batch()
    expected = executor.run(batch.model_inputs).model_output.logits.detach()
    source_mlps = tuple(
        layer.mlp for layer in fixture.adapter.module.model.layers
    )
    source_norms = tuple(
        layer.post_feedforward_layernorm
        for layer in fixture.adapter.module.model.layers
    )

    def consume() -> tuple[Tensor, Tensor]:
        outputs = []
        for _ in range(2):
            call_inputs: dict[str, object] = dict(batch.model_inputs)
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            outputs.append(
                fixture.adapter.module(**call_inputs).logits.detach().clone()
            )
        return outputs[0], outputs[1]

    first, second = executor.run_with_generated_overlay(
        consume,
        expected_forward_calls=2,
    )
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)
    assert tuple(layer.mlp for layer in fixture.adapter.module.model.layers) == (
        source_mlps
    )
    assert tuple(
        layer.post_feedforward_layernorm
        for layer in fixture.adapter.module.model.layers
    ) == source_norms


def test_post_feedforward_delta_diagnostic_attenuation_is_continuous() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    batch = _batch()
    stage = next(
        value
        for value in fixture.adapter.layers[2].transformer.stages
        if value.kind == "feed_forward"
    )
    capture_sites = (
        stage.normalized_input_site,
        stage.delta_site,
    )

    def capture():
        return fixture.adapter.forward(
            batch.model_inputs,
            capture_sites=capture_sites,
        )

    graph_sha256 = executor.graph_plan.artifact_sha256
    lowering_sha256s = executor.lowering_artifact_sha256s
    ordinary = executor.run_with_generated_overlay(capture)
    alpha_one = (
        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
            capture,
            layer_ordinal=2,
        )
    )
    alpha_zero = (
        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
            capture,
            layer_ordinal=2,
            alpha=0.0,
        )
    )
    alpha_quarter = (
        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
            capture,
            layer_ordinal=2,
            alpha=0.25,
        )
    )

    torch.testing.assert_close(
        alpha_one.logits,
        ordinary.logits,
        rtol=0,
        atol=0,
    )
    for site in capture_sites:
        torch.testing.assert_close(
            alpha_one.activations[site],
            ordinary.activations[site],
            rtol=0,
            atol=0,
        )
    for traced in (alpha_zero, alpha_quarter):
        torch.testing.assert_close(
            traced.activations[stage.normalized_input_site],
            ordinary.activations[stage.normalized_input_site],
            rtol=0,
            atol=0,
        )

    compact = executor.compiled_mlps["2"]
    post_norm = fixture.adapter.module.model.layers[
        2
    ].post_feedforward_layernorm
    with torch.no_grad():
        retained = post_norm(
            compact(
                alpha_zero.activations[stage.normalized_input_site]
            )
        )
    torch.testing.assert_close(
        alpha_zero.activations[stage.delta_site],
        retained,
    )
    expected_quarter = retained + 0.25 * (
        ordinary.activations[stage.delta_site] - retained
    )
    torch.testing.assert_close(
        alpha_quarter.activations[stage.delta_site],
        expected_quarter,
    )
    assert executor.graph_plan.artifact_sha256 == graph_sha256
    assert executor.lowering_artifact_sha256s == lowering_sha256s


def test_post_feedforward_delta_diagnostic_attenuation_is_strictly_scoped() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    callback = lambda: None

    with pytest.raises(TypeError, match="layer_ordinal must be an integer"):
        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
            callback,
            layer_ordinal=True,
        )
    with pytest.raises(ValueError, match="declared post-feed-forward"):
        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
            callback,
            layer_ordinal=0,
        )
    with pytest.raises(TypeError, match="alpha must be a real scalar"):
        executor.run_with_diagnostic_post_feedforward_delta_attenuation(
            callback,
            layer_ordinal=2,
            alpha=True,
        )
    for invalid in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
            executor.run_with_diagnostic_post_feedforward_delta_attenuation(
                callback,
                layer_ordinal=2,
                alpha=invalid,
            )


def test_post_feedforward_delta_diagnostic_override_uses_full_runtime_shape() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    batch = _batch()
    stage = next(
        value
        for value in fixture.adapter.layers[2].transformer.stages
        if value.kind == "feed_forward"
    )
    capture_sites = (stage.normalized_input_site, stage.delta_site)

    def capture():
        return fixture.adapter.forward(
            batch.model_inputs,
            capture_sites=capture_sites,
        )

    graph_sha256 = executor.graph_plan.artifact_sha256
    lowering_sha256s = executor.lowering_artifact_sha256s
    source_fingerprint = fixture.adapter.model_fingerprint()
    runtime_fingerprint = module_state_fingerprint(executor.graph_runtime)
    ordinary = executor.run_with_generated_overlay(capture)
    generated_seen: list[Tensor] = []
    replacement_seen: list[Tensor] = []

    def provide(generated: Tensor) -> Tensor:
        generated_seen.append(generated.detach().clone())
        replacement = (
            torch.arange(
                generated.numel(),
                device=generated.device,
                dtype=generated.dtype,
            ).reshape(generated.shape)
            + 1
        ) * 0.001
        replacement_seen.append(replacement.detach().clone())
        return replacement

    overridden = executor.run_with_diagnostic_post_feedforward_delta_override(
        capture,
        layer_ordinal=2,
        correction_provider=provide,
    )

    compact = executor.compiled_mlps["2"]
    post_norm = fixture.adapter.module.model.layers[
        2
    ].post_feedforward_layernorm
    with torch.no_grad():
        retained = post_norm(
            compact(ordinary.activations[stage.normalized_input_site])
        )
    assert len(generated_seen) == len(replacement_seen) == 1
    assert generated_seen[0].shape == (2, 4, 4)
    torch.testing.assert_close(
        generated_seen[0],
        ordinary.activations[stage.delta_site] - retained,
    )
    torch.testing.assert_close(
        overridden.activations[stage.normalized_input_site],
        ordinary.activations[stage.normalized_input_site],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        overridden.activations[stage.delta_site],
        retained + replacement_seen[0],
    )
    padding_mask = ~batch.model_inputs["attention_mask"]
    assert bool((replacement_seen[0][padding_mask] != 0).all())
    torch.testing.assert_close(
        overridden.activations[stage.delta_site][padding_mask],
        (retained + replacement_seen[0])[padding_mask],
    )
    assert executor.graph_plan.artifact_sha256 == graph_sha256
    assert executor.lowering_artifact_sha256s == lowering_sha256s
    assert fixture.adapter.model_fingerprint() == source_fingerprint
    assert module_state_fingerprint(executor.graph_runtime) == (
        runtime_fingerprint
    )


def test_post_feedforward_delta_diagnostic_override_validates_and_restores() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    batch = _batch()
    layers = fixture.adapter.module.model.layers
    source_mlps = tuple(layer.mlp for layer in layers)
    source_norms = tuple(layer.post_feedforward_layernorm for layer in layers)

    def forward() -> Tensor:
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        return fixture.adapter.module(**call_inputs).logits

    with pytest.raises(TypeError, match="layer_ordinal must be an integer"):
        executor.run_with_diagnostic_post_feedforward_delta_override(
            forward,
            layer_ordinal=True,
            correction_provider=lambda generated: generated,
        )
    with pytest.raises(ValueError, match="declared post-feed-forward"):
        executor.run_with_diagnostic_post_feedforward_delta_override(
            forward,
            layer_ordinal=0,
            correction_provider=lambda generated: generated,
        )
    with pytest.raises(TypeError, match="correction_provider must be callable"):
        executor.run_with_diagnostic_post_feedforward_delta_override(
            forward,
            layer_ordinal=2,
            correction_provider=None,  # type: ignore[arg-type]
        )

    invalid_providers = (
        (
            lambda _generated: "not a tensor",
            TypeError,
            "must return a Tensor",
        ),
        (
            lambda generated: generated[..., :-1],
            ValueError,
            "replacement shape",
        ),
        (
            lambda generated: generated.to(device="meta"),
            ValueError,
            "replacement device",
        ),
        (
            lambda generated: generated.to(dtype=torch.float64),
            ValueError,
            "replacement dtype",
        ),
        (
            lambda generated: torch.full_like(generated, float("nan")),
            ValueError,
            "replacement must be finite",
        ),
    )
    for provider, error_type, message in invalid_providers:
        with pytest.raises(error_type, match=message):
            executor.run_with_diagnostic_post_feedforward_delta_override(
                forward,
                layer_ordinal=2,
                correction_provider=provider,
            )
        assert tuple(layer.mlp for layer in layers) == source_mlps
        assert tuple(
            layer.post_feedforward_layernorm for layer in layers
        ) == source_norms

    expected = executor.run(batch.model_inputs).model_output.logits
    torch.testing.assert_close(
        executor.run_with_diagnostic_post_feedforward_delta_override(
            forward,
            layer_ordinal=2,
            correction_provider=lambda generated: generated,
        ),
        expected,
        rtol=0,
        atol=0,
    )


def test_post_feedforward_delta_restores_both_modules_after_errors() -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    batch = _batch()
    layers = fixture.adapter.module.model.layers
    source_mlps = tuple(layer.mlp for layer in layers)
    source_norms = tuple(layer.post_feedforward_layernorm for layer in layers)
    source_fingerprint = fixture.adapter.model_fingerprint()

    def fail_after_forward() -> None:
        call_inputs: dict[str, object] = dict(batch.model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["return_dict"] = True
        fixture.adapter.module(**call_inputs)
        raise RuntimeError("sentinel post-delta callback failure")

    with pytest.raises(RuntimeError, match="sentinel post-delta callback failure"):
        executor.run_with_generated_overlay(fail_after_forward)
    assert tuple(layer.mlp for layer in layers) == source_mlps
    assert tuple(layer.post_feedforward_layernorm for layer in layers) == (
        source_norms
    )

    def fail_norm(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
    ) -> None:
        raise RuntimeError("sentinel post-delta norm failure")

    handle = source_norms[2].register_forward_pre_hook(fail_norm)
    try:
        with pytest.raises(RuntimeError, match="sentinel post-delta norm failure"):
            executor.run(batch.model_inputs)
    finally:
        handle.remove()
    assert tuple(layer.mlp for layer in layers) == source_mlps
    assert tuple(layer.post_feedforward_layernorm for layer in layers) == (
        source_norms
    )
    assert fixture.adapter.model_fingerprint() == source_fingerprint
    assert executor.run(batch.model_inputs).graph_execution.traversal_order


def test_compact_post_feedforward_delta_row_replay_is_exact_and_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _post_delta_fixture()
    executor = fixture.executor
    compact = executor.compiled_mlps["2"]
    post_norm = fixture.adapter.module.model.layers[
        2
    ].post_feedforward_layernorm
    generator = torch.Generator().manual_seed(77_013)
    inputs = torch.randn(4101, 4, generator=generator, dtype=torch.float64)
    compact_chunks: list[int] = []
    norm_chunks: list[int] = []

    compact_handle = compact.register_forward_hook(
        lambda _module, args, _output: compact_chunks.append(args[0].shape[0])
    )
    norm_handle = post_norm.register_forward_hook(
        lambda _module, args, _output: norm_chunks.append(args[0].shape[0])
    )
    try:
        actual = executor.execute_compact_post_feedforward_delta_rows(
            2,
            inputs,
        )
    finally:
        compact_handle.remove()
        norm_handle.remove()

    runtime_weight = compact.gate_proj.weight
    with torch.no_grad():
        expected = post_norm(
            compact(
                inputs.to(
                    device=runtime_weight.device,
                    dtype=runtime_weight.dtype,
                )
            )
        ).to(device="cpu", dtype=torch.float64)
    assert compact_chunks == [2048, 2048, 5]
    assert norm_chunks == [2048, 2048, 5]
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)


def test_post_delta_full_removal_does_not_normalize_generator() -> None:
    fixture = _post_delta_fixture(full_layer=True)
    executor = fixture.executor
    batch = _batch()
    stage = next(
        value
        for value in fixture.adapter.layers[0].transformer.stages
        if value.kind == "feed_forward"
    )

    def capture():
        return fixture.adapter.forward(
            batch.model_inputs,
            capture_sites=(
                stage.normalized_input_site,
                stage.operator_output_site,
                stage.delta_site,
            ),
        )

    traced = executor.run_with_generated_overlay(capture)
    generated = executor.execute_graph_inputs(
        {
            stage.normalized_input_site: traced.activations[
                stage.normalized_input_site
            ]
        }
    ).outputs[stage.delta_site]
    assert torch.count_nonzero(
        traced.activations[stage.operator_output_site]
    ) == 0
    torch.testing.assert_close(
        traced.activations[stage.delta_site],
        generated,
    )
    normalized_generated = fixture.adapter.module.model.layers[
        0
    ].post_feedforward_layernorm(generated)
    assert not torch.allclose(generated, normalized_generated)


def test_additive_post_feedforward_defaults_are_bit_identical() -> None:
    fixture = _fixture(full_layer=True)
    explicit = Gemma3ModalGeneratorGraphExecutor(
        fixture.adapter,
        fixture.graph_plan,
        fixture.lowerings,
        additive_post_feedforward_graph_plan=None,
        additive_post_feedforward_lowerings=(),
        additive_post_feedforward_scale=1.0,
    )
    batch = _batch()
    implicit_result = fixture.executor.run(batch.model_inputs)
    explicit_result = explicit.run(batch.model_inputs)

    assert explicit.additive_post_feedforward_layer_ordinals == ()
    assert explicit.additive_post_feedforward_graph_plan is None
    assert explicit.additive_post_feedforward_scale == 1.0
    assert explicit.additive_post_feedforward_lowering_artifact_sha256s == ()
    assert explicit.additive_post_feedforward_graph_runtime_parameter_count == 0
    assert explicit.total_graph_runtime_parameter_count == (
        explicit.graph_runtime_parameter_count
    )
    assert explicit_result.additive_graph_execution is None
    torch.testing.assert_close(
        explicit_result.model_output.logits,
        implicit_result.model_output.logits,
        rtol=0,
        atol=0,
    )
    assert explicit_result.graph_execution.traversal_order == (
        implicit_result.graph_execution.traversal_order
    )
    for boundary, expected in implicit_result.graph_execution.outputs.items():
        torch.testing.assert_close(
            explicit_result.graph_execution.outputs[boundary],
            expected,
            rtol=0,
            atol=0,
        )
    assert (
        explicit_result.candidate_whole_model_learned_parameters,
        explicit_result.modal_graph_learned_parameters,
        explicit_result.logical_modal_graph_macs,
        explicit_result.logical_executed_modal_graph_macs,
        explicit_result.logical_modal_graph_additions,
        explicit_result.logical_executed_modal_graph_additions,
        explicit_result.peak_live_modal_width,
        explicit_result.replacement_scope,
    ) == (
        implicit_result.candidate_whole_model_learned_parameters,
        implicit_result.modal_graph_learned_parameters,
        implicit_result.logical_modal_graph_macs,
        implicit_result.logical_executed_modal_graph_macs,
        implicit_result.logical_modal_graph_additions,
        implicit_result.logical_executed_modal_graph_additions,
        implicit_result.peak_live_modal_width,
        implicit_result.replacement_scope,
    )


def test_additive_post_feedforward_is_applied_after_rmsnorm() -> None:
    fixture = _additive_fixture(scale=0.5)
    executor = fixture.executor
    adapter = fixture.source.adapter
    transformer = adapter.layers[0].transformer
    assert transformer is not None
    stage = next(
        value for value in transformer.stages if value.kind == "feed_forward"
    )
    batch = _batch()

    def capture():
        return adapter.forward(
            batch.model_inputs,
            capture_sites=(
                stage.normalized_input_site,
                stage.operator_output_site,
                stage.delta_site,
            ),
        )

    traced = executor.run_with_generated_overlay(capture)
    normalized_input = traced.activations[stage.normalized_input_site]
    source_graph = executor.execute_graph_inputs(
        {stage.normalized_input_site: normalized_input}
    )
    additive_graph = executor.execute_additive_post_feedforward_graph_inputs(
        normalized_input
    )
    source_raw = source_graph.outputs[stage.operator_output_site]
    additive_delta = additive_graph.outputs[stage.delta_site]
    post_norm = adapter.module.model.layers[0].post_feedforward_layernorm
    with torch.no_grad():
        expected = post_norm(source_raw) + 0.5 * additive_delta
        wrong_pre_norm_order = post_norm(
            source_raw + 0.5 * additive_delta
        )

    torch.testing.assert_close(
        traced.activations[stage.operator_output_site],
        source_raw,
    )
    torch.testing.assert_close(
        traced.activations[stage.delta_site],
        expected,
    )
    assert not torch.allclose(
        traced.activations[stage.delta_site],
        wrong_pre_norm_order,
    )


def test_additive_post_feedforward_scale_is_continuous() -> None:
    fixture = _additive_fixture(scale=1.0)
    source = fixture.source
    quarter = Gemma3ModalGeneratorGraphExecutor(
        source.adapter,
        source.graph_plan,
        source.lowerings,
        additive_post_feedforward_graph_plan=fixture.graph_plan,
        additive_post_feedforward_lowerings=fixture.lowerings,
        additive_post_feedforward_scale=0.25,
    )
    half = Gemma3ModalGeneratorGraphExecutor(
        source.adapter,
        source.graph_plan,
        source.lowerings,
        additive_post_feedforward_graph_plan=fixture.graph_plan,
        additive_post_feedforward_lowerings=fixture.lowerings,
        additive_post_feedforward_scale=0.5,
    )
    transformer = source.adapter.layers[0].transformer
    assert transformer is not None
    stage = next(
        value for value in transformer.stages if value.kind == "feed_forward"
    )
    batch = _batch()

    def capture(executor: Gemma3ModalGeneratorGraphExecutor):
        return executor.run_with_generated_overlay(
            lambda: source.adapter.forward(
                batch.model_inputs,
                capture_sites=(
                    stage.normalized_input_site,
                    stage.delta_site,
                ),
            )
        )

    baseline = capture(source.executor)
    alpha_quarter = capture(quarter)
    alpha_half = capture(half)
    alpha_one = capture(fixture.executor)
    for traced in (alpha_quarter, alpha_half, alpha_one):
        torch.testing.assert_close(
            traced.activations[stage.normalized_input_site],
            baseline.activations[stage.normalized_input_site],
            rtol=0,
            atol=0,
        )

    baseline_delta = baseline.activations[stage.delta_site]
    full_correction = (
        alpha_one.activations[stage.delta_site] - baseline_delta
    )
    assert bool(torch.count_nonzero(full_correction))
    torch.testing.assert_close(
        alpha_quarter.activations[stage.delta_site],
        baseline_delta + 0.25 * full_correction,
    )
    torch.testing.assert_close(
        alpha_half.activations[stage.delta_site],
        baseline_delta + 0.5 * full_correction,
    )


def test_additive_post_feedforward_deletion_has_no_residual() -> None:
    fixture = _additive_fixture(scale=0.5)
    batch = _batch()
    baseline = fixture.source.executor.run(
        batch.model_inputs,
        condition="deletion",
    )
    deletion = fixture.executor.run(
        batch.model_inputs,
        condition="deletion",
        capture_modal_states=True,
        capture_edge_messages=True,
    )

    torch.testing.assert_close(
        deletion.model_output.logits,
        baseline.model_output.logits,
        rtol=0,
        atol=0,
    )
    assert deletion.additive_graph_execution is not None
    assert deletion.additive_graph_execution.traversal_order == ()
    assert deletion.additive_graph_execution.modal_states == {}
    assert deletion.additive_graph_execution.edge_messages == {}
    assert all(
        not bool(torch.count_nonzero(values))
        for values in deletion.additive_graph_execution.outputs.values()
    )
    assert deletion.logical_executed_modal_graph_macs == 0
    assert deletion.logical_executed_modal_graph_additions == 0


def test_additive_post_feedforward_resource_accounting_is_exact() -> None:
    fixture = _additive_fixture(scale=0.5)
    executor = fixture.executor
    source_plan = fixture.source.graph_plan
    additive_plan = fixture.graph_plan
    result = executor.run(
        _batch().model_inputs,
        capture_modal_states=True,
        capture_edge_messages=True,
    )
    additive_runtime = executor.additive_post_feedforward_graph_runtime
    assert additive_runtime is not None

    total_graph_parameters = (
        source_plan.parameter_count + additive_plan.parameter_count
    )
    assert executor.graph_runtime_parameter_count == source_plan.parameter_count
    assert executor.additive_post_feedforward_graph_runtime_parameter_count == (
        additive_plan.parameter_count
    )
    assert executor.total_graph_runtime_parameter_count == (
        total_graph_parameters
    )
    assert executor.additive_post_feedforward_layer_ordinals == (0,)
    assert executor.additive_post_feedforward_graph_plan is not None
    assert executor.additive_post_feedforward_graph_plan.artifact_sha256 == (
        additive_plan.artifact_sha256
    )
    assert executor.additive_post_feedforward_scale == 0.5
    assert executor.additive_post_feedforward_lowering_artifact_sha256s == tuple(
        lowering.artifact_sha256 for lowering in fixture.lowerings
    )
    assert result.replaced_layer_count == 1
    assert result.graph_node_count == (
        len(source_plan.nodes) + len(additive_plan.nodes)
    )
    assert result.fragment_count == (
        len(source_plan.nodes) + len(additive_plan.nodes)
    )
    assert result.removed_mode_count == 6
    assert result.native_removed_learned_parameters == 6 * 3 * 4
    assert result.modal_graph_learned_parameters == total_graph_parameters
    assert result.candidate_whole_model_learned_parameters == (
        result.source_whole_model_learned_parameters
        - result.native_removed_learned_parameters
        + total_graph_parameters
    )
    assert result.net_stored_parameter_savings == (
        result.native_removed_learned_parameters - total_graph_parameters
    )
    assert result.logical_modal_graph_macs == result.valid_tokens * (
        source_plan.macs_per_token + additive_plan.macs_per_token
    )
    assert result.logical_executed_modal_graph_macs == (
        result.logical_modal_graph_macs
    )
    assert result.logical_modal_graph_additions == result.valid_tokens * (
        source_plan.accounting.elementwise_additions_per_token
        + additive_plan.accounting.elementwise_additions_per_token
    )
    assert result.logical_executed_modal_graph_additions == (
        result.logical_modal_graph_additions
    )
    assert result.net_logical_macs_saved == (
        result.logical_linear_macs_native_removed
        - result.logical_executed_modal_graph_macs
    )
    assert result.peak_live_modal_width == (
        executor.graph_runtime.peak_live_modal_width
        + additive_runtime.peak_live_modal_width
    )
    assert result.replacement_scope == (
        "full_native_mlp_replacement_at_selected_layers_"
        "with_nonowning_post_feedforward_additive_graph"
    )
    assert result.additive_graph_execution is not None
    assert result.additive_graph_execution.traversal_order == (
        additive_plan.traversal_order
    )
    assert not (
        _module_storage(fixture.source.adapter.module)
        & _module_storage(additive_runtime)
    )
    assert not (
        _module_storage(executor.graph_runtime)
        & _module_storage(additive_runtime)
    )
    assert not (_plan_storage(additive_plan) & _module_storage(additive_runtime))


def test_additive_post_feedforward_restores_modules_after_error() -> None:
    fixture = _additive_fixture()
    executor = fixture.executor
    adapter = fixture.source.adapter
    layers = adapter.module.model.layers
    source_mlps = tuple(layer.mlp for layer in layers)
    source_norms = tuple(layer.post_feedforward_layernorm for layer in layers)
    source_fingerprint = adapter.model_fingerprint()

    def fail_norm(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
    ) -> None:
        raise RuntimeError("sentinel additive post-norm failure")

    handle = source_norms[0].register_forward_pre_hook(fail_norm)
    try:
        with pytest.raises(
            RuntimeError,
            match="sentinel additive post-norm failure",
        ):
            executor.run(_batch().model_inputs)
    finally:
        handle.remove()

    assert tuple(layer.mlp for layer in layers) == source_mlps
    assert tuple(layer.post_feedforward_layernorm for layer in layers) == (
        source_norms
    )
    assert adapter.model_fingerprint() == source_fingerprint
    assert executor.run(
        _batch().model_inputs
    ).additive_graph_execution is not None


def test_additive_error_path_revalidates_restored_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _additive_fixture()
    executor = fixture.executor
    source_validate = executor._validate_live_state
    validation_calls = 0

    def count_validation() -> None:
        nonlocal validation_calls
        validation_calls += 1
        source_validate()

    monkeypatch.setattr(executor, "_validate_live_state", count_validation)

    def fail_norm(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
    ) -> None:
        raise RuntimeError("sentinel error-path validation probe")

    norm = fixture.source.adapter.module.model.layers[0].post_feedforward_layernorm
    handle = norm.register_forward_pre_hook(fail_norm)
    try:
        with pytest.raises(RuntimeError, match="sentinel error-path validation"):
            executor.run(_batch().model_inputs)
    finally:
        handle.remove()

    # One validation before overlays are installed and one after restoration.
    assert validation_calls == 2


def test_additive_post_feedforward_rejects_invalid_layer_contracts() -> None:
    source = _fixture(full_layer=True)
    transformer = source.adapter.layers[0].transformer
    assert transformer is not None
    stage = next(
        value for value in transformer.stages if value.kind == "feed_forward"
    )
    nonzero_lowerings = tuple(
        _relocate_lowering_output(lowering, stage.delta_site)
        for lowering in source.lowerings
    )
    nonzero_graph = _relocated_graph(source, nonzero_lowerings)
    with pytest.raises(ValueError, match="decode with zero mean"):
        Gemma3ModalGeneratorGraphExecutor(
            source.adapter,
            source.graph_plan,
            source.lowerings,
            additive_post_feedforward_graph_plan=nonzero_graph,
            additive_post_feedforward_lowerings=nonzero_lowerings,
        )

    multi_layer = _fixture()
    owning_lowerings = multi_layer.lowerings[:2]
    owning_graph = ModalGeneratorGraphPlan(
        model_fingerprint=multi_layer.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            multi_layer.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=multi_layer.graph_plan.nodes[:2],
        interactions=(),
    )
    layer_two_transformer = multi_layer.adapter.layers[2].transformer
    assert layer_two_transformer is not None
    layer_two_stage = next(
        value
        for value in layer_two_transformer.stages
        if value.kind == "feed_forward"
    )
    wrong_layer_lowerings = tuple(
        _relocate_lowering_as_zero_mean_residual(
            lowering,
            layer_two_stage.delta_site,
        )
        for lowering in multi_layer.lowerings[2:]
    )
    wrong_layer_graph = _relocated_graph(
        multi_layer,
        wrong_layer_lowerings,
    )
    with pytest.raises(ValueError, match="graph binding is invalid"):
        Gemma3ModalGeneratorGraphExecutor(
            multi_layer.adapter,
            owning_graph,
            owning_lowerings,
            additive_post_feedforward_graph_plan=wrong_layer_graph,
            additive_post_feedforward_lowerings=wrong_layer_lowerings,
        )

    post_delta_source = _post_delta_fixture(full_layer=True)
    post_delta_transformer = post_delta_source.adapter.layers[0].transformer
    assert post_delta_transformer is not None
    post_delta_stage = next(
        value
        for value in post_delta_transformer.stages
        if value.kind == "feed_forward"
    )
    overlap_lowerings = tuple(
        _relocate_lowering_as_zero_mean_residual(
            lowering,
            post_delta_stage.delta_site,
        )
        for lowering in post_delta_source.lowerings
    )
    overlap_graph = _relocated_graph(
        post_delta_source,
        overlap_lowerings,
    )
    with pytest.raises(ValueError, match="pre-normalization source graph"):
        Gemma3ModalGeneratorGraphExecutor(
            post_delta_source.adapter,
            post_delta_source.graph_plan,
            post_delta_source.lowerings,
            post_feedforward_delta_layer_ordinals=(0,),
            additive_post_feedforward_graph_plan=overlap_graph,
            additive_post_feedforward_lowerings=overlap_lowerings,
        )
