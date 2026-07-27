from __future__ import annotations

import torch

from fisher_graph.gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    DistinctLayerFragmentSelection,
    EdgelessTerminalFanInRows,
    build_edgeless_terminal_fanin_plan,
    collect_aligned_fragment_rows,
    collect_edgeless_terminal_fanin_rows,
    fit_terminal_fanin_compilation,
    select_top_distinct_layer_fragments,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from test_gemma3_modal_generator_executor import _batch
from test_gemma3_modal_generator_graph_executor import (
    _fragment_plan,
    _lowering,
)
from test_modal_compiler_pipeline import (
    EVAL_HASH,
    FIT_HASH,
    _lowerings,
)


DTYPE = torch.float64


def _two_layer_fixture():
    from test_gemma3_modal_generator_executor import _adapter

    adapter = _adapter(seed=55_101)
    fragment_plan = _fragment_plan(adapter, full_layer=False)
    selection = select_top_distinct_layer_fragments(
        fragment_plan,
        count=2,
        minimum_fragment_modes=1,
    )
    lowerings = {
        fragment.fragment_id: _lowering(
            adapter,
            fragment_plan,
            fragment,
        )
        for fragment in selection.causal_order
    }
    edgeless = build_edgeless_terminal_fanin_plan(
        selection,
        fragment_plan=fragment_plan,
        lowerings_by_fragment=lowerings,
    )
    return adapter, fragment_plan, selection, edgeless


def test_top_distinct_selection_is_fisher_ranked_then_causally_sorted() -> None:
    _, _, selection, _ = _two_layer_fixture()

    assert tuple(
        (fragment.layer_ordinal, fragment.cluster_id)
        for fragment in selection.fisher_order
    ) == ((2, 3), (0, 1))
    assert tuple(
        (fragment.layer_ordinal, fragment.cluster_id)
        for fragment in selection.causal_order
    ) == ((0, 1), (2, 3))
    assert selection.terminal_fragment.layer_ordinal == 2


def test_one_pass_fragment_collection_preserves_shared_row_keys() -> None:
    _, _, selection, _ = _two_layer_fixture()
    first, second = selection.causal_order
    sites = {
        first.input_site: torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
            dtype=DTYPE,
        ),
        first.activation_site: torch.arange(
            12,
            dtype=DTYPE,
        ).reshape(2, 6),
        second.input_site: torch.tensor(
            [[-1.0, -2.0, -3.0, -4.0], [1.0, 2.0, 3.0, 4.0]],
            dtype=DTYPE,
        ),
        second.activation_site: torch.arange(
            12,
            24,
            dtype=DTYPE,
        ).reshape(2, 6),
    }
    gradients = {
        name: torch.ones_like(value) for name, value in sites.items()
    }
    row = ActivationScoreGradientRows(
        activations=sites,
        score_gradients=gradients,
        logical_positions=torch.tensor([4, 7]),
        loss=1.0,
        example_id="example.sha",
    )
    down = {
        fragment.fragment_id: torch.arange(
            24,
            dtype=DTYPE,
        ).reshape(4, 6)
        for fragment in selection.causal_order
    }

    collected = collect_aligned_fragment_rows(
        (row,),
        fragments=selection.causal_order,
        down_projection_weights=down,
    )

    assert collected.row_keys == (("example.sha", 4), ("example.sha", 7))
    assert collected.observations == 2
    assert collected.sequences == 1
    for fragment in selection.causal_order:
        rows = collected.rows_by_fragment[fragment.fragment_id]
        index = torch.tensor(fragment.removed_mode_indices)
        expected = (
            sites[fragment.activation_site].index_select(1, index)
            @ down[fragment.fragment_id].index_select(1, index).T
        )
        torch.testing.assert_close(rows.contributions, expected)


def test_physical_edgeless_capture_uses_exact_batch_row_axis() -> None:
    adapter, _, _, edgeless = _two_layer_fixture()
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

    captured = collect_edgeless_terminal_fanin_rows(
        adapter,
        executor,
        (batch,),
        plan=edgeless,
        expected_row_keys=expected,
    )

    assert captured.row_keys == expected
    assert captured.observations == 6
    assert set(captured.node_states) == {
        node.name for node in edgeless.graph_plan.nodes
    }
    for name in captured.node_states:
        assert captured.node_states[name].shape == (
            6,
            edgeless.graph_plan.nodes[0].latent_width,
        )
        assert captured.teacher_coordinates[name].shape == (
            6,
            edgeless.graph_plan.nodes[0].latent_width,
        )
    # Hook handles are removed and the executor remains reusable.
    executor.run(batch.model_inputs, condition="generated")


def test_terminal_only_fit_builds_authenticated_pipeline() -> None:
    trace, catalog, fisher, clusters, fragments, lowerings_by_node = (
        _lowerings()
    )
    selection = DistinctLayerFragmentSelection(
        fisher_order=fragments.fragments,
        causal_order=tuple(
            sorted(
                fragments.fragments,
                key=lambda fragment: fragment.layer_ordinal,
            )
        ),
        minimum_fragment_modes=1,
    )
    by_fragment = {
        lowering.mode_set_id: lowering
        for lowering in lowerings_by_node.values()
    }
    edgeless = build_edgeless_terminal_fanin_plan(
        selection,
        fragment_plan=fragments,
        lowerings_by_fragment=by_fragment,
    )
    source_name, target_name = (
        node.name for node in edgeless.graph_plan.nodes
    )
    fit_source = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        dtype=DTYPE,
    )
    eval_source = torch.tensor(
        [[-4.0], [-0.5], [0.5], [4.0]],
        dtype=DTYPE,
    )
    fit_zero = torch.zeros_like(fit_source)
    eval_zero = torch.zeros_like(eval_source)
    fit_runtime = EdgelessTerminalFanInRows(
        node_states={
            source_name: fit_source,
            target_name: fit_zero,
        },
        teacher_coordinates={
            source_name: fit_source,
            target_name: 2.0 * fit_source,
        },
        row_keys=tuple(("fit", index) for index in range(6)),
    )
    eval_runtime = EdgelessTerminalFanInRows(
        node_states={
            source_name: eval_source,
            target_name: eval_zero,
        },
        teacher_coordinates={
            source_name: eval_source,
            target_name: 2.0 * eval_source,
        },
        row_keys=tuple(("eval", index) for index in range(4)),
    )

    compiled = fit_terminal_fanin_compilation(
        edgeless=edgeless,
        fit_rows=fit_runtime,
        eval_rows=eval_runtime,
        target_fisher_weights_fit=torch.ones(6, dtype=DTYPE),
        target_fisher_weights_eval=torch.ones(4, dtype=DTYPE),
        fit_prompt_trace=trace,
        parameter_catalog=catalog,
        fisher_coupling=fisher,
        parameter_clusters=clusters,
        fragment_plan=fragments,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
        minimum_heldout_improvement=1e-6,
    )

    assert len(compiled.interaction_selection.interactions) == 1
    edge = compiled.graph_plan.interactions[0]
    assert (edge.source_node, edge.target_node) == (
        source_name,
        target_name,
    )
    assert compiled.compiler_pipeline.graph_plan.artifact_sha256 == (
        compiled.graph_plan.artifact_sha256
    )
    assert set(
        compiled.source_replacement_accounting.fragment_ids
    ) == {fragment.fragment_id for fragment in fragments.fragments}
