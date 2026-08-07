from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_l10_l17_a5d_evaluation as evaluation
from fisher_graph.modal_generator_graph import ModalGeneratorGraphPlan
from test_gemma3_modal_generator_graph_executor import (
    _additive_fixture,
    _batch,
    _fixture,
)


def _resource_inputs(execution: object) -> tuple[dict[str, object], dict[str, int]]:
    static = {
        field: getattr(execution, field) for field in evaluation._GRAPH_STATIC_FIELDS
    }
    totals = {
        field: getattr(execution, field) for field in evaluation._GRAPH_LOGICAL_FIELDS
    }
    return static, totals


def _compact(
    executor: object,
    execution: object,
) -> dict[str, object]:
    static, totals = _resource_inputs(execution)
    return evaluation._compact_resources(
        executor=executor,
        static=static,
        totals=totals,
        logical_valid_tokens=execution.valid_tokens,
        peak_live_modal_width=execution.peak_live_modal_width,
    )


def _panel_executor(
    plan: ModalGeneratorGraphPlan,
    *,
    affected_layer_ordinals: tuple[int, ...],
) -> SimpleNamespace:
    return SimpleNamespace(
        graph_plan=plan,
        affected_layer_ordinals=affected_layer_ordinals,
        post_feedforward_delta_layer_ordinals=(),
        additive_post_feedforward_graph_plan=None,
        additive_post_feedforward_graph_runtime=None,
        additive_post_feedforward_layer_ordinals=(),
        additive_post_feedforward_lowering_artifact_sha256s=(),
        additive_post_feedforward_scale=1.0,
    )


def test_dual_execution_validates_generated_and_deletion_traversals() -> None:
    fixture = _additive_fixture(scale=0.5)
    executor = fixture.executor
    generated = executor.run(_batch().model_inputs, condition="generated")
    deletion = executor.run(_batch().model_inputs, condition="deletion")

    evaluation._validate_dual_execution(
        generated,
        executor,
        condition="generated",
        label="selected",
    )
    evaluation._validate_dual_execution(
        deletion,
        executor,
        condition="deletion",
        label="matched_double_deletion",
    )

    assert generated.graph_execution.traversal_order == (
        executor.graph_plan.traversal_order
    )
    assert generated.additive_graph_execution is not None
    assert generated.additive_graph_execution.traversal_order == (
        fixture.graph_plan.traversal_order
    )
    assert deletion.graph_execution.traversal_order == ()
    assert deletion.additive_graph_execution is not None
    assert deletion.additive_graph_execution.traversal_order == ()


def test_additive_resource_accounting_separates_both_graphs() -> None:
    fixture = _additive_fixture(scale=0.5)
    executor = fixture.executor
    execution = executor.run(_batch().model_inputs, condition="generated")

    resources = _compact(executor, execution)
    owning = executor.graph_plan
    additive = fixture.graph_plan

    assert resources["owning_graph_node_count"] == len(owning.nodes)
    assert resources["additive_graph_node_count"] == len(additive.nodes)
    assert resources["total_graph_node_count"] == len(owning.nodes) + len(
        additive.nodes
    )
    assert resources["owning_graph_parameters"] == owning.parameter_count
    assert resources["additive_graph_parameters"] == additive.parameter_count
    assert resources["total_graph_parameters"] == (
        owning.parameter_count + additive.parameter_count
    )
    assert resources["total_dense_graph_macs_per_token"] == (
        owning.macs_per_token + additive.macs_per_token
    )
    assert resources["total_dense_graph_additions_per_token"] == (
        owning.accounting.elementwise_additions_per_token
        + additive.accounting.elementwise_additions_per_token
    )
    assert resources["executed_graph_macs_per_token"] == (
        resources["total_dense_graph_macs_per_token"]
    )


def test_no_additive_fallback_preserves_source_only_accounting_and_panel() -> None:
    fallback = _fixture(full_layer=True)
    generated = fallback.executor.run(
        _batch().model_inputs,
        condition="generated",
    )
    deletion = fallback.executor.run(
        _batch().model_inputs,
        condition="deletion",
    )

    evaluation._validate_dual_execution(
        generated,
        fallback.executor,
        condition="generated",
        label="fallback",
    )
    evaluation._validate_dual_execution(
        deletion,
        fallback.executor,
        condition="deletion",
        label="fallback_deletion",
    )
    resources = _compact(fallback.executor, generated)
    assert generated.additive_graph_execution is None
    assert deletion.additive_graph_execution is None
    assert resources["additive_graph_node_count"] == 0
    assert resources["additive_graph_parameters"] == 0
    assert resources["additive_dense_graph_macs_per_token"] == 0
    assert resources["total_graph_parameters"] == resources[
        "owning_graph_parameters"
    ]
    assert resources["total_dense_graph_macs_per_token"] == resources[
        "owning_dense_graph_macs_per_token"
    ]

    composition = _fixture().graph_plan
    edgeless = ModalGeneratorGraphPlan(
        model_fingerprint=composition.model_fingerprint,
        parameter_cluster_plan_sha256=composition.parameter_cluster_plan_sha256,
        nodes=composition.nodes,
        interactions=(),
    )
    assert evaluation._validate_executor_panel(
        layer10=_panel_executor(
            composition,
            affected_layer_ordinals=(10,),
        ),
        layer17=_panel_executor(
            edgeless,
            affected_layer_ordinals=(17,),
        ),
        frozen_composition=_panel_executor(
            composition,
            affected_layer_ordinals=(10, 17),
        ),
        selected_composition=_panel_executor(
            composition,
            affected_layer_ordinals=(10, 17),
        ),
    ) is False


def test_tampered_owning_additive_and_deletion_traversals_fail_closed() -> None:
    fixture = _additive_fixture()
    executor = fixture.executor
    generated = executor.run(_batch().model_inputs, condition="generated")
    deletion = executor.run(_batch().model_inputs, condition="deletion")

    bad_owning = replace(
        generated,
        graph_execution=replace(
            generated.graph_execution,
            traversal_order=(),
        ),
    )
    with pytest.raises(RuntimeError, match="owning graph traversal drifted"):
        evaluation._validate_dual_execution(
            bad_owning,
            executor,
            condition="generated",
            label="tampered",
        )

    assert generated.additive_graph_execution is not None
    bad_additive = replace(
        generated,
        additive_graph_execution=replace(
            generated.additive_graph_execution,
            traversal_order=(),
        ),
    )
    with pytest.raises(RuntimeError, match="additive graph traversal drifted"):
        evaluation._validate_dual_execution(
            bad_additive,
            executor,
            condition="generated",
            label="tampered",
        )

    assert deletion.additive_graph_execution is not None
    bad_deletion = replace(
        deletion,
        additive_graph_execution=replace(
            deletion.additive_graph_execution,
            traversal_order=fixture.graph_plan.traversal_order,
        ),
    )
    with pytest.raises(RuntimeError, match="additive graph traversal drifted"):
        evaluation._validate_dual_execution(
            bad_deletion,
            executor,
            condition="deletion",
            label="tampered_deletion",
        )


def test_tampered_resource_totals_and_static_counts_fail_closed() -> None:
    fixture = _additive_fixture(scale=0.5)
    execution = fixture.executor.run(
        _batch().model_inputs,
        condition="generated",
    )
    static, totals = _resource_inputs(execution)

    bad_totals = dict(totals)
    bad_totals["logical_modal_graph_macs"] += execution.valid_tokens
    with pytest.raises(RuntimeError, match="exact resource identities drifted"):
        evaluation._compact_resources(
            executor=fixture.executor,
            static=static,
            totals=bad_totals,
            logical_valid_tokens=execution.valid_tokens,
            peak_live_modal_width=execution.peak_live_modal_width,
        )

    bad_static = dict(static)
    bad_static["modal_graph_learned_parameters"] += 1
    with pytest.raises(RuntimeError, match="exact resource identities drifted"):
        evaluation._compact_resources(
            executor=fixture.executor,
            static=bad_static,
            totals=totals,
            logical_valid_tokens=execution.valid_tokens,
            peak_live_modal_width=execution.peak_live_modal_width,
        )
