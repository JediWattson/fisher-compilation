from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    collect_aligned_fragment_rows,
)
from fisher_graph.gemma3_same_layer_shape_flow import (
    SameLayerFragmentSelection,
    SameLayerShapeFlowRows,
    build_edgeless_same_layer_graph,
    collect_aligned_same_layer_fragment_rows,
    collect_edgeless_same_layer_shape_flow_rows,
    select_top_fisher_same_layer_fragments,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from test_gemma3_modal_generator_executor import _adapter, _batch
from test_gemma3_modal_generator_graph_executor import (
    _fragment_plan,
    _lowering,
)


DTYPE = torch.float64


def _same_layer_fixture():
    adapter = _adapter(seed=66_101)
    fragment_plan = _fragment_plan(adapter, full_layer=True)
    selection = select_top_fisher_same_layer_fragments(
        fragment_plan,
        count=2,
        minimum_fragment_modes=1,
        layer_ordinal=0,
    )
    lowerings = {
        fragment.fragment_id: _lowering(
            adapter,
            fragment_plan,
            fragment,
        )
        for fragment in selection.execution_order
    }
    edgeless = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=fragment_plan,
        lowerings_by_fragment=lowerings,
    )
    return adapter, fragment_plan, selection, edgeless


def test_same_layer_selection_is_authenticated_and_fisher_oriented() -> None:
    _, fragment_plan, selection, _ = _same_layer_fixture()

    assert tuple(
        fragment.cluster_id for fragment in selection.fisher_order
    ) == (1, 0)
    assert tuple(
        fragment.cluster_id for fragment in selection.execution_order
    ) == (1, 0)
    assert selection.source_fragment == selection.fisher_order[0]
    assert selection.target_fragments == selection.execution_order[1:]
    assert selection.removed_mode_indices == (0, 1, 2, 3, 4, 5)
    selection.validate_against(fragment_plan)

    restored = SameLayerFragmentSelection.from_state_dict(
        selection.state_dict()
    )
    assert restored.artifact_sha256 == selection.artifact_sha256
    assert restored.fragment_ids == selection.fragment_ids


def test_same_layer_selection_rejects_overlapping_modes() -> None:
    _, fragment_plan, selection, _ = _same_layer_fixture()
    source, target = selection.fisher_order
    overlapping_target = replace(
        target,
        channel_indices=(2, 4, 5),
        artifact_sha256="",
    )

    with pytest.raises(ValueError, match="mode-disjoint"):
        SameLayerFragmentSelection(
            source_fragment_plan_sha256=fragment_plan.artifact_sha256,
            source_model_sha256=fragment_plan.source_model_sha256,
            layer_ordinal=0,
            layer_id=source.layer_id,
            fisher_order=(source, overlapping_target),
            execution_order=(source, overlapping_target),
            minimum_fragment_modes=1,
            layer_selection_policy=selection.layer_selection_policy,
        )


def test_edgeless_same_layer_graph_executes_strict_source_first_order() -> None:
    adapter, _, selection, edgeless = _same_layer_fixture()
    nodes = edgeless.graph_plan.nodes

    assert not edgeless.graph_plan.interactions
    assert tuple(node.causal_order for node in nodes) == (0, 1)
    assert tuple(
        edgeless.fragment_id_by_node[node.name] for node in nodes
    ) == selection.fragment_ids

    executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless.graph_plan,
        edgeless.lowerings,
    )
    execution = executor.run(
        _batch().model_inputs,
        condition="generated",
        capture_modal_states=True,
    )

    assert execution.graph_execution.traversal_order == edgeless.node_names
    assert set(execution.graph_execution.modal_states or ()) == set(
        edgeless.node_names
    )
    assert tuple(executor.compiled_mlps) == ("0",)
    assert executor.compiled_mlps["0"].removed_mode_indices == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


def test_same_layer_native_rows_preserve_one_aligned_axis() -> None:
    adapter, _, selection, _ = _same_layer_fixture()
    first = selection.execution_order[0]
    sites = {
        first.input_site: torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            dtype=DTYPE,
        ),
        first.activation_site: torch.arange(
            12,
            dtype=DTYPE,
        ).reshape(2, 6),
    }
    row = ActivationScoreGradientRows(
        activations=sites,
        score_gradients={
            name: torch.ones_like(value) for name, value in sites.items()
        },
        logical_positions=torch.tensor([4, 7]),
        loss=1.0,
        example_id="same-layer-example.sha",
    )
    down = {
        fragment.fragment_id: torch.arange(
            24,
            dtype=DTYPE,
        ).reshape(4, 6)
        for fragment in selection.execution_order
    }

    with pytest.raises(ValueError, match="distinct layers"):
        collect_aligned_fragment_rows(
            (row,),
            fragments=selection.execution_order,
            down_projection_weights=down,
        )
    collected = collect_aligned_same_layer_fragment_rows(
        (row,),
        selection=selection,
        down_projection_weights=down,
    )

    assert collected.row_keys == (
        ("same-layer-example.sha", 4),
        ("same-layer-example.sha", 7),
    )
    assert collected.observations == 2
    assert collected.sequences == 1
    shared_inputs = tuple(
        collected.rows_by_fragment[fragment.fragment_id].inputs
        for fragment in selection.execution_order
    )
    assert all(
        value.untyped_storage().data_ptr()
        == shared_inputs[0].untyped_storage().data_ptr()
        for value in shared_inputs[1:]
    )
    for fragment in selection.execution_order:
        index = torch.tensor(fragment.removed_mode_indices)
        expected = (
            sites[fragment.activation_site].index_select(1, index)
            @ down[fragment.fragment_id].index_select(1, index).T
        )
        torch.testing.assert_close(
            collected.rows_by_fragment[
                fragment.fragment_id
            ].contributions,
            expected,
        )


def test_physical_same_layer_capture_has_shape_and_native_flow_rows() -> None:
    adapter, _, _, edgeless = _same_layer_fixture()
    executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless.graph_plan,
        edgeless.lowerings,
    )
    batch = _batch()
    expected = (
        ("modal-generator-a", 0),
        ("modal-generator-a", 1),
        ("modal-generator-a", 2),
        ("modal-generator-a", 3),
        ("modal-generator-b", 0),
        ("modal-generator-b", 1),
    )

    captured = collect_edgeless_same_layer_shape_flow_rows(
        adapter,
        executor,
        (batch,),
        plan=edgeless,
        expected_row_keys=expected,
    )

    assert captured.row_keys == expected
    assert captured.observations == 6
    assert set(captured.node_states) == set(edgeless.node_names)
    assert set(captured.teacher_flows) == set(edgeless.node_names)
    for name in edgeless.node_names:
        assert captured.node_states[name].shape == (6, 2)
        assert captured.teacher_coordinates[name].shape == (6, 2)
        torch.testing.assert_close(
            captured.teacher_flows[name],
            captured.teacher_coordinates[name] - captured.node_states[name],
        )
    roundtrip = SameLayerShapeFlowRows(
        node_states=captured.node_states,
        teacher_coordinates=captured.teacher_coordinates,
        row_keys=captured.row_keys,
        row_key_sha256=captured.row_key_sha256,
    )
    assert roundtrip.row_key_sha256 == captured.row_key_sha256
    # Capture hooks are gone and the graph executor remains reusable.
    executor.run(batch.model_inputs, condition="generated")
