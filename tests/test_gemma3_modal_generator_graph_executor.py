from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.computational_modes import (
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


def _fixture(*, full_layer: bool = False) -> _Fixture:
    adapter = _adapter(seed=9_001 if full_layer else 9_000)
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
        tensors.extend((edge.message_matrix, edge.message_bias))
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
