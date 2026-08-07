from __future__ import annotations

import copy
from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.computational_modes import (
    ComputationalModeBinding,
    fit_computational_mode_rate_curve,
)
from fisher_graph.fisher_prompt_clustering import (
    FisherPromptClusterConfig,
    build_fisher_prompt_clusters,
)
from fisher_graph.modal_compiler_pipeline import (
    ModalCompilerPipeline,
    ModalRefitFisherAuthority,
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from fisher_graph.modal_generator_graph import (
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    StateConditionedModalGeneratorInteraction,
)
from fisher_graph.modal_generator_lowering import (
    lower_coordinate_modal_generator,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    fit_modal_generator_rate_curve,
)
from fisher_graph.modal_interaction_fitting import (
    select_modal_interactions_greedily,
)
from fisher_graph.modal_interaction_promotion import (
    ModalInteractionGraphPromotion,
    build_modal_interaction_graph_promotion,
)
from fisher_graph.parameter_cluster_fragments import (
    build_parameter_cluster_layer_fragments,
)
from fisher_graph.parameter_fisher_coupling import (
    NaturalMLPLayerParameterSpec,
    build_grouped_virtual_gate_fisher_from_trace,
    build_natural_mlp_parameter_group_catalog,
)
from fisher_graph.prompt_mode_tracing import (
    PromptModeTraceProvenance,
    collect_prompt_mode_trace,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows


DTYPE = torch.float64
MODEL_HASH = "a" * 64
FIT_HASH = "b" * 64
EVAL_HASH = "c" * 64
OBJECTIVE_HASH = "d" * 64
REFIT_HASH = "e" * 64
DIAGNOSTIC_HASH = "f" * 64


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _analysis_chain(*, layer_count: int = 2):
    if layer_count not in (2, 3):
        raise ValueError("test analysis chain supports two or three layers")
    specs = tuple(
        NaturalMLPLayerParameterSpec(
            layer_id=f"model.layers.{index}",
            layer_ordinal=index,
            activation_site=f"model.layers.{index}.mlp.gated",
            input_site=f"model.layers.{index}.mlp.input",
            output_site=f"model.layers.{index}.mlp.residual_delta",
            intermediate_width=1,
            input_width=16,
            output_width=4,
            gate_proj_path=f"model.layers.{index}.mlp.gate.weight",
            up_proj_path=f"model.layers.{index}.mlp.up.weight",
            down_proj_path=f"model.layers.{index}.mlp.down.weight",
        )
        for index in range(layer_count)
    )
    catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint=MODEL_HASH,
        layer_specs=specs,
    )
    # Orthogonal prompt-effect profiles produce two unambiguous clusters.
    effects = (
        torch.tensor(
            [
                [3.0, 0.0],
                [2.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
            ],
            dtype=DTYPE,
        )
        if layer_count == 2
        else torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=DTYPE,
        )
    )
    trace_rows = tuple(
        ActivationScoreGradientRows(
            activations={
                spec.activation_site: effects[row_index, column_index]
                .reshape(1, 1)
                .clone()
                for column_index, spec in enumerate(specs)
            },
            score_gradients={
                spec.activation_site: torch.ones(1, 1, dtype=DTYPE)
                for spec in specs
            },
            logical_positions=torch.tensor([0], dtype=torch.int64),
            loss=float(row_index + 1),
            example_id=f"pipeline-fit-{row_index}",
        )
        for row_index in range(effects.shape[0])
    )
    trace = collect_prompt_mode_trace(
        trace_rows,
        layer_specs=tuple(value.cross_block_layer_spec for value in specs),
        provenance=PromptModeTraceProvenance(
            source_model_fingerprint=MODEL_HASH,
            calibration_split_sha256=FIT_HASH,
            objective_sha256=OBJECTIVE_HASH,
        ),
    )
    fisher = build_grouped_virtual_gate_fisher_from_trace(
        trace,
        catalog=catalog,
    )
    cluster_config = FisherPromptClusterConfig(
        model_fingerprint=MODEL_HASH,
        calibration_split_sha256=FIT_HASH,
        objective_sha256=OBJECTIVE_HASH,
        source_fisher_coupling_sha256=fisher.artifact_sha256,
        layer_specs=tuple(
            value.cross_block_layer_spec for value in specs
        ),
        mode_catalog=fisher.fisher_ranked_mode_catalog(),
        cluster_count=layer_count,
        max_iterations=20,
    )
    clusters = build_fisher_prompt_clusters(effects, cluster_config)
    fragments = build_parameter_cluster_layer_fragments(clusters, fisher)
    assert len(fragments.fragments) == layer_count
    return trace, catalog, fisher, clusters, fragments


def _basis(
    fragment,
    fisher_sha: str,
    *,
    axis: int,
    fit_split_sha256: str = FIT_HASH,
    eval_split_sha256: str = EVAL_HASH,
):
    fit_modes = torch.tensor(
        [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
        dtype=DTYPE,
    )
    eval_modes = torch.tensor(
        [[-4.0], [-0.5], [0.5], [4.0]],
        dtype=DTYPE,
    )
    decoder = torch.zeros(1, 4, dtype=DTYPE)
    decoder[0, axis] = 1.0
    fit = fit_modes @ decoder
    evaluation = eval_modes @ decoder
    binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind="layer_fragment",
        output_site=(
            f"model.layers.{fragment.layer_ordinal}.mlp.residual_delta"
        ),
        source_model_sha256=MODEL_HASH,
        parameter_catalog_sha256=fragment.parameter_catalog_sha256,
        fisher_coupling_sha256=fisher_sha,
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
    )
    curve = fit_computational_mode_rate_curve(
        fit,
        torch.ones(fit.shape[0], dtype=DTYPE),
        evaluation,
        torch.ones(evaluation.shape[0], dtype=DTYPE),
        (1,),
        binding=binding,
        selection_rule="fixed_rank",
        selected_rank=1,
    )
    return curve.selected_basis


def _generator(
    basis,
    fragment,
    fragment_plan,
    *,
    index: int,
    fit_split_sha256: str = FIT_HASH,
    eval_split_sha256: str = EVAL_HASH,
):
    X_fit_seed = torch.tensor(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -2.0, 1.0],
            [1.0, 0.0, -1.0],
            [2.0, 1.0, 0.0],
            [0.0, 2.0, -1.0],
        ],
        dtype=DTYPE,
    )
    X_eval_seed = torch.tensor(
        [
            [-3.0, 1.0, 0.0],
            [0.0, -1.0, 2.0],
            [1.5, 0.5, -1.0],
            [2.0, -2.0, 1.0],
        ],
        dtype=DTYPE,
    )
    X_fit = torch.cat(
        (X_fit_seed, torch.zeros(X_fit_seed.shape[0], 13, dtype=DTYPE)),
        dim=1,
    )
    X_eval = torch.cat(
        (X_eval_seed, torch.zeros(X_eval_seed.shape[0], 13, dtype=DTYPE)),
        dim=1,
    )
    coefficient_seed = torch.tensor(
        [[1.0 + index], [-0.5], [0.25]],
        dtype=DTYPE,
    )
    coefficient = torch.cat(
        (coefficient_seed, torch.zeros(13, 1, dtype=DTYPE)),
        dim=0,
    )
    Y_fit = X_fit @ coefficient
    Y_eval = X_eval @ coefficient
    binding = ModalGeneratorBinding.create(
        generator_id=f"generator.{index}",
        input_kind="native_layer_input",
        input_site=f"model.layers.{index}.mlp.input",
        output_site=basis.binding.output_site,
        source_model_sha256=MODEL_HASH,
        input_catalog_sha256=fragment.input_catalog_sha256,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=fragment_plan.artifact_sha256,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=eval_split_sha256,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=basis.binding.fisher_coupling_sha256,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    curve = fit_modal_generator_rate_curve(
        X_fit,
        Y_fit,
        torch.ones(X_fit.shape[0], dtype=DTYPE),
        X_eval,
        Y_eval,
        (1,),
        binding=binding,
        fisher_weights_eval=torch.ones(X_eval.shape[0], dtype=DTYPE),
        fit_intercept=False,
        selection_rule="fixed_rank",
        selected_rank=1,
    )
    return curve.selected_plan


def _lowerings(
    *,
    layer_count: int = 2,
    fit_split_sha256: str = FIT_HASH,
    eval_split_sha256: str = EVAL_HASH,
):
    trace, catalog, fisher, clusters, fragments = _analysis_chain(
        layer_count=layer_count
    )
    values = {}
    for index, fragment in enumerate(fragments.fragments):
        basis = _basis(
            fragment,
            fisher.artifact_sha256,
            axis=index,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
        )
        generator = _generator(
            basis,
            fragment,
            fragments,
            index=fragment.layer_ordinal,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=eval_split_sha256,
        )
        values[f"node.{fragment.layer_ordinal}"] = (
            lower_coordinate_modal_generator(
                generator,
                basis,
                fragments,
            )
        )
    return trace, catalog, fisher, clusters, fragments, values


def _compiled(
    *,
    with_interaction: bool = True,
    with_source_accounting: bool = True,
):
    trace, catalog, fisher, clusters, fragments, lowerings = _lowerings()
    graph_nodes = tuple(
        lowering.to_graph_node(
            name=name,
            causal_order=next(
                fragment.layer_ordinal
                for fragment in fragments.fragments
                if fragment.artifact_sha256
                == lowering.selected_fragment_sha256
            ),
        )
        for name, lowering in sorted(lowerings.items())
    )
    selection = None
    interactions = ()
    if with_interaction:
        source_fit = torch.tensor(
            [[-2.0], [-1.0], [1.0], [2.0]],
            dtype=DTYPE,
        )
        source_eval = torch.tensor(
            [[-3.0], [-0.5], [0.5], [3.0]],
            dtype=DTYPE,
        )
        selection = select_modal_interactions_greedily(
            {
                "node.0": source_fit,
                "node.1": torch.zeros_like(source_fit),
            },
            {
                "node.0": source_eval,
                "node.1": torch.zeros_like(source_eval),
            },
            {"node.1": 1.5 * source_fit},
            {"node.1": 1.5 * source_eval},
            node_causal_orders={"node.0": 0, "node.1": 1},
            generator_artifact_sha256s={
                name: lowering.coordinate_generator_plan.artifact_sha256
                for name, lowering in lowerings.items()
            },
            source_model_sha256=MODEL_HASH,
            parameter_cluster_plan_sha256=fragments.artifact_sha256,
            fit_split_sha256=FIT_HASH,
            eval_split_sha256=EVAL_HASH,
            candidate_edges=(("node.0", "node.1"),),
            fit_intercept=False,
            minimum_heldout_improvement=1e-6,
        )
        interactions = selection.interactions
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_HASH,
        parameter_cluster_plan_sha256=fragments.artifact_sha256,
        nodes=graph_nodes,
        interactions=interactions,
    )
    accounting = None
    if with_source_accounting:
        accounting = build_modal_source_replacement_accounting(
            catalog,
            fragments,
            tuple(value.fragment_id for value in fragments.fragments),
        )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=trace,
        parameter_catalog=catalog,
        grouped_fisher=fisher,
        fisher_clusters=clusters,
        parameter_cluster_fragments=fragments,
        lowerings_by_node=lowerings,
        graph_plan=graph,
        interaction_selection=selection,
        source_replacement_accounting=accounting,
    )
    return pipeline, (
        trace,
        catalog,
        fisher,
        clusters,
        fragments,
        lowerings,
        graph,
    )


def _refit_compiled():
    trace, catalog, fisher, clusters, fragments, lowerings = _lowerings(
        fit_split_sha256=REFIT_HASH,
        eval_split_sha256=DIAGNOSTIC_HASH,
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_HASH,
        parameter_cluster_plan_sha256=fragments.artifact_sha256,
        nodes=tuple(
            lowering.to_graph_node(
                name=name,
                causal_order=next(
                    fragment.layer_ordinal
                    for fragment in fragments.fragments
                    if fragment.artifact_sha256
                    == lowering.selected_fragment_sha256
                ),
            )
            for name, lowering in sorted(lowerings.items())
        ),
        interactions=(),
    )
    authority = ModalRefitFisherAuthority(
        fit_split_sha256=REFIT_HASH,
        eval_split_sha256=DIAGNOSTIC_HASH,
        eval_role="balanced_within_fit_fixed_rank_diagnostic",
        fisher_normalization="equal_family_total_mass_per_fragment",
        source_model_sha256=MODEL_HASH,
        parameter_catalog_sha256=catalog.artifact_sha256,
        topology_grouped_fisher_sha256=fisher.artifact_sha256,
        topology_fisher_calibration_split_sha256=FIT_HASH,
        topology_fisher_cluster_plan_sha256=clusters.artifact_sha256,
        topology_fragment_plan_sha256=fragments.artifact_sha256,
        authorizing_report_sha256="1" * 64,
        refit_protocol_sha256="2" * 64,
        fit_authority_sha256="3" * 64,
        fit_receipt_sha256="4" * 64,
        fit_corpus_artifact_sha256="5" * 64,
        fit_manifest_sha256="6" * 64,
        fit_materialization_sha256="7" * 64,
        fit_example_count=256,
        fit_family_count=8,
        equal_family_weighting=True,
        eval_subset_within_fit=True,
        eval_used_for_selection=False,
    )
    accounting = build_modal_source_replacement_accounting(
        catalog,
        fragments,
        tuple(value.fragment_id for value in fragments.fragments),
    )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=trace,
        parameter_catalog=catalog,
        grouped_fisher=fisher,
        fisher_clusters=clusters,
        parameter_cluster_fragments=fragments,
        lowerings_by_node=lowerings,
        graph_plan=graph,
        modal_refit_fisher_authority=authority,
        source_replacement_accounting=accounting,
    )
    return pipeline, authority, (
        trace,
        catalog,
        fisher,
        clusters,
        fragments,
        lowerings,
        graph,
    )


def _conditional_compiled(*, mixed: bool = False):
    trace, catalog, fisher, clusters, fragments, lowerings = _lowerings(
        layer_count=3
    )
    graph_nodes = tuple(
        lowering.to_graph_node(
            name=name,
            causal_order=next(
                fragment.layer_ordinal
                for fragment in fragments.fragments
                if fragment.artifact_sha256
                == lowering.selected_fragment_sha256
            ),
        )
        for name, lowering in sorted(lowerings.items())
    )
    conditional = tuple(
        StateConditionedModalGeneratorInteraction(
            source_node="node.0",
            target_node=target,
            routing_group="next_mode",
            message_matrix=torch.tensor([[scale]], dtype=DTYPE),
            message_bias=torch.tensor([0.0], dtype=DTYPE),
            gate_weight=torch.tensor([[1.0, -1.0][index]], dtype=DTYPE),
            gate_bias=torch.tensor([0.0], dtype=DTYPE),
            temperature=0.75,
            top_k=1,
        )
        for index, (target, scale) in enumerate(
            (("node.1", 0.5), ("node.2", -0.25))
        )
    )
    static = (
        (
            ModalGeneratorInteraction(
                source_node="node.1",
                target_node="node.2",
                message_matrix=torch.tensor([[0.125]], dtype=DTYPE),
                message_bias=torch.tensor([0.0], dtype=DTYPE),
            ),
        )
        if mixed
        else ()
    )
    interactions = tuple(
        sorted(
            (*conditional, *static),
            key=lambda edge: (edge.source_node, edge.target_node),
        )
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_HASH,
        parameter_cluster_plan_sha256=fragments.artifact_sha256,
        nodes=graph_nodes,
        interactions=interactions,
    )
    promotion = build_modal_interaction_graph_promotion(
        graph,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
        selection_metric="weighted_nrmse",
        baseline_metric_value=1.0,
        candidate_metric_value=0.5,
        minimum_heldout_improvement=0.1,
        heldout_observations=8,
    )
    accounting = build_modal_source_replacement_accounting(
        catalog,
        fragments,
        tuple(value.fragment_id for value in fragments.fragments),
    )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=trace,
        parameter_catalog=catalog,
        grouped_fisher=fisher,
        fisher_clusters=clusters,
        parameter_cluster_fragments=fragments,
        lowerings_by_node=lowerings,
        graph_plan=graph,
        interaction_selection=promotion,
        source_replacement_accounting=accounting,
    )
    return pipeline, promotion


def test_full_chain_is_machine_checkable_and_executable() -> None:
    pipeline, sources = _compiled()
    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources

    assert pipeline.model_fingerprint == MODEL_HASH
    assert fisher.source_prompt_trace_sha256 == trace.artifact_sha256
    assert pipeline.parameter_catalog.artifact_sha256 == (
        catalog.artifact_sha256
    )
    assert pipeline.grouped_fisher.referenced_artifact_sha256 == (
        fisher.artifact_sha256
    )
    assert pipeline.fisher_clusters.referenced_artifact_sha256 == (
        clusters.artifact_sha256
    )
    assert pipeline.parameter_cluster_fragments.artifact_sha256 == (
        fragments.artifact_sha256
    )
    assert pipeline.graph_plan.artifact_sha256 == graph.artifact_sha256
    assert pipeline.interaction_selection is not None
    assert len(pipeline.graph_plan.interactions) == 1
    assert {value.node_name for value in pipeline.nodes} == set(lowerings)
    assert {
        value.mode_set_id for value in pipeline.nodes
    } == {value.fragment_id for value in fragments.fragments}
    assert all(
        value.coordinate_generator.binding.target_kind
        == "computational_mode_coordinates"
        for value in pipeline.nodes
    )


def test_legacy_selection_graph_and_pipeline_hashes_are_frozen() -> None:
    pipeline, _ = _compiled()

    assert pipeline.interaction_selection is not None
    assert pipeline.interaction_selection.artifact_sha256 == (
        "90ded7e6a9c159ab261df5b3a1bec210796419ff93fc47c3df74060af8e3172a"
    )
    assert pipeline.graph_plan.artifact_sha256 == (
        "ccee5d2f6aba2d5f8c9029878060d85ff94c16e033b94d448dd0f01ddf35a8ae"
    )
    assert pipeline.artifact_sha256 == (
        "9bf8e5449725fe98f17abe955d527122d306af01ccc4df309ae4063307132b12"
    )
    assert pipeline.modal_refit_fisher_authority is None
    assert "modal_refit_fisher_authority" not in pipeline.state_dict()
    assert "modal_refit_fisher_authority_sha256" not in pipeline.state_dict()


def test_refit_fisher_authority_is_explicit_cross_bound_and_roundtrips() -> None:
    pipeline, authority, sources = _refit_compiled()
    restored = ModalCompilerPipeline.from_state_dict(pipeline.state_dict())

    assert restored.modal_refit_fisher_authority is not None
    assert restored.modal_refit_fisher_authority.artifact_sha256 == (
        authority.artifact_sha256
    )
    assert restored.fit_split_sha256 == REFIT_HASH
    assert restored.eval_split_sha256 == DIAGNOSTIC_HASH
    assert restored.grouped_fisher.metadata["calibration_split_sha256"] == (
        FIT_HASH
    )
    assert restored.artifact_sha256 == pipeline.artifact_sha256
    assert restored.metadata()["modal_refit_fisher_authority_sha256"] == (
        authority.artifact_sha256
    )

    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources
    with pytest.raises(ValueError, match="modal fit split"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=graph,
        )


def test_refit_fisher_authority_tampering_and_split_aliases_fail_closed() -> None:
    pipeline, authority, sources = _refit_compiled()

    poisoned = copy.deepcopy(pipeline.state_dict())
    poisoned["modal_refit_fisher_authority"][
        "fit_materialization_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="authority hash mismatch"):
        ModalCompilerPipeline.from_state_dict(poisoned)

    partial = copy.deepcopy(pipeline.state_dict())
    partial.pop("modal_refit_fisher_authority")
    with pytest.raises(ValueError, match="pipeline fields"):
        ModalCompilerPipeline.from_state_dict(partial)

    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources
    wrong_topology = replace(
        authority,
        topology_fragment_plan_sha256="0" * 64,
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="differs from pipeline topology"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=graph,
            modal_refit_fisher_authority=wrong_topology,
        )

    with pytest.raises(ValueError, match="splits must remain distinct"):
        replace(
            authority,
            eval_split_sha256=FIT_HASH,
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="safety/role flags"):
        replace(authority, guard_opened=True, artifact_sha256="")
    with pytest.raises(ValueError, match="protected role"):
        replace(
            authority,
            eval_role="sealed_validation_diagnostic",
            artifact_sha256="",
        )


@pytest.mark.parametrize("mixed", (False, True))
def test_conditional_graph_promotion_roundtrips_through_pipeline(
    mixed: bool,
) -> None:
    pipeline, promotion = _conditional_compiled(mixed=mixed)
    restored = ModalCompilerPipeline.from_state_dict(pipeline.state_dict())

    assert isinstance(
        restored.interaction_selection,
        ModalInteractionGraphPromotion,
    )
    assert restored.interaction_selection.artifact_sha256 == (
        promotion.artifact_sha256
    )
    assert restored.interaction_selection.graph_plan_sha256 == (
        restored.graph_plan.artifact_sha256
    )
    assert restored.interaction_selection.conditional_interaction_count == 2
    assert restored.interaction_selection.interaction_count == (3 if mixed else 2)
    assert restored.artifact_sha256 == pipeline.artifact_sha256


def test_conditional_promotion_fails_closed_on_binding_count_and_kind() -> None:
    pipeline, promotion = _conditional_compiled()

    wrong_graph = replace(
        promotion,
        graph_plan_sha256="0" * 64,
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="authorize the exact graph"):
        replace(
            pipeline,
            interaction_selection=wrong_graph,
            artifact_sha256="",
        )

    wrong_count = replace(
        promotion,
        conditional_interaction_count=1,
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="authorize the exact graph"):
        replace(
            pipeline,
            interaction_selection=wrong_count,
            artifact_sha256="",
        )

    poisoned_metric = copy.deepcopy(pipeline.state_dict())
    poisoned_metric["interaction_selection"]["candidate_metric_value"] = 0.25
    with pytest.raises(ValueError, match="promotion hash mismatch"):
        ModalCompilerPipeline.from_state_dict(poisoned_metric)

    unknown_kind = copy.deepcopy(pipeline.state_dict())
    unknown_kind["interaction_selection"]["artifact_kind"] = (
        "fisher_graph.unrecognized_interaction_authorization"
    )
    with pytest.raises(
        ValueError,
        match="unsupported interaction authorization kind",
    ):
        ModalCompilerPipeline.from_state_dict(unknown_kind)


def test_static_only_graph_cannot_bypass_legacy_selection() -> None:
    pipeline, _ = _compiled()

    with pytest.raises(
        ValueError,
        match="at least one state-conditioned interaction",
    ):
        build_modal_interaction_graph_promotion(
            pipeline.graph_plan,
            fit_split_sha256=FIT_HASH,
            eval_split_sha256=EVAL_HASH,
            selection_metric="weighted_nrmse",
            baseline_metric_value=1.0,
            candidate_metric_value=0.5,
            heldout_observations=4,
        )


def test_exact_source_group_and_net_savings_accounting() -> None:
    pipeline, sources = _compiled()
    catalog = sources[1]

    # Two natural groups, each owning 2*16 + 4 matrix parameters/MACs.
    assert pipeline.replaced_parameter_group_count == 2
    assert pipeline.replaced_parameter_group_indices == (0, 1)
    assert pipeline.replaced_fragment_ids == tuple(
        sorted(value.fragment_id for value in sources[4].fragments)
    )
    assert pipeline.source_parameter_count == 72
    assert pipeline.source_macs_per_token == 72
    assert pipeline.source_parameter_count == sum(
        value.parameter_count for value in catalog.groups
    )
    assert pipeline.net_parameter_savings == (
        72 - pipeline.graph_parameter_count
    )
    assert pipeline.net_macs_saved_per_token == (
        72 - pipeline.graph_macs_per_token
    )
    assert pipeline.net_parameter_savings > 0
    assert pipeline.net_macs_saved_per_token > 0


def test_no_source_accounting_means_no_savings_claim() -> None:
    pipeline, _ = _compiled(with_source_accounting=False)
    metadata = pipeline.metadata()

    assert metadata["has_exact_source_accounting"] is False
    assert pipeline.replaced_parameter_group_count is None
    assert pipeline.replaced_parameter_group_indices is None
    assert pipeline.replaced_fragment_ids is None
    assert pipeline.source_parameter_count is None
    assert pipeline.source_macs_per_token is None
    assert pipeline.net_parameter_savings is None
    assert pipeline.net_macs_saved_per_token is None


def test_pipeline_roundtrip_is_deterministic_and_isolates_source_mutation() -> None:
    pipeline, sources = _compiled()
    restored = ModalCompilerPipeline.from_state_dict(pipeline.state_dict())

    assert restored.artifact_sha256 == pipeline.artifact_sha256
    assert restored.metadata() == pipeline.metadata()
    assert restored.graph_parameter_count == pipeline.graph_parameter_count

    # The manifest owns authenticated copies of executable tensors.
    sources[5]["node.0"].graph_weights.input_factor[0, 0] += 100.0
    assert not torch.equal(
        sources[5]["node.0"].graph_weights.input_factor,
        pipeline.nodes[0].lowering.graph_weights.input_factor,
    )


def test_nested_tensor_reference_and_metric_tampering_are_rejected() -> None:
    pipeline, _ = _compiled()

    poisoned_basis = copy.deepcopy(pipeline.state_dict())
    poisoned_basis["nodes"][0]["lowering"]["computational_mode_basis"][
        "encoder_basis"
    ][0, 0] += 1.0
    with pytest.raises(
        ValueError,
        match="encoder_basis|orthonormal",
    ):
        ModalCompilerPipeline.from_state_dict(poisoned_basis)

    poisoned_reference = copy.deepcopy(pipeline.state_dict())
    metadata = json_loads(
        poisoned_reference["grouped_fisher"]["metadata_json"]
    )
    metadata["objective_sha256"] = "0" * 64
    poisoned_reference["grouped_fisher"]["metadata_json"] = json_dumps(
        metadata
    )
    with pytest.raises(ValueError, match="metadata hash mismatch"):
        ModalCompilerPipeline.from_state_dict(poisoned_reference)

    poisoned_graph = copy.deepcopy(pipeline.state_dict())
    poisoned_graph["graph_plan"]["nodes"][0]["weights"]["input_factor"][
        0, 0
    ] += 1.0
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        ModalCompilerPipeline.from_state_dict(poisoned_graph)


def json_loads(value: str):
    import json

    return json.loads(value)


def json_dumps(value: object) -> str:
    import json

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_catalog_fisher_cluster_fragment_mismatch_is_rejected() -> None:
    _, sources = _compiled(with_interaction=False)
    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources
    other_catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint=MODEL_HASH,
        layer_specs=(
            NaturalMLPLayerParameterSpec(
                layer_id="model.layers.9",
                layer_ordinal=9,
                activation_site="model.layers.9.mlp.gated",
                input_site="model.layers.9.mlp.input",
                output_site="model.layers.9.mlp.residual_delta",
                intermediate_width=1,
                input_width=16,
                output_width=4,
                gate_proj_path="x.gate",
                up_proj_path="x.up",
                down_proj_path="x.down",
            ),
        ),
    )

    with pytest.raises(ValueError, match="provenance differ"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=other_catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=graph,
        )


def test_grouped_fisher_must_match_the_exact_supplied_prompt_trace() -> None:
    _, sources = _compiled(with_interaction=False)
    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources
    wrong_trace = collect_prompt_mode_trace(
        (
            ActivationScoreGradientRows(
                activations={
                    spec.activation_site: torch.ones(
                        1,
                        spec.width,
                        dtype=DTYPE,
                    )
                    for spec in trace.layer_specs
                },
                score_gradients={
                    spec.activation_site: torch.ones(
                        1,
                        spec.width,
                        dtype=DTYPE,
                    )
                    for spec in trace.layer_specs
                },
                logical_positions=torch.tensor([0], dtype=torch.int64),
                loss=0.0,
                example_id="different-prompt-trace",
            ),
        ),
        layer_specs=trace.layer_specs,
        provenance=trace.provenance,
    )
    assert wrong_trace.artifact_sha256 != trace.artifact_sha256

    with pytest.raises(
        ValueError,
        match="not bound to the supplied authenticated prompt trace",
    ):
        build_modal_compiler_pipeline(
            source_prompt_trace=wrong_trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=graph,
        )


def test_one_to_one_lowering_mapping_and_graph_node_hash_are_enforced() -> None:
    _, sources = _compiled(with_interaction=False)
    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources

    with pytest.raises(ValueError, match="one-to-one"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node={
                "node.0": lowerings["node.0"],
                "node.1": lowerings["node.0"],
            },
            graph_plan=graph,
        )

    swapped = {
        "node.0": lowerings["node.1"],
        "node.1": lowerings["node.0"],
    }
    with pytest.raises(ValueError, match="graph (node|weights) mismatch"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=swapped,
            graph_plan=graph,
        )


def test_graph_interactions_require_exact_authenticated_selection() -> None:
    pipeline, sources = _compiled(with_interaction=True)
    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources
    assert graph.interactions

    with pytest.raises(ValueError, match="require authenticated selection"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=graph,
            interaction_selection=None,
        )

    no_edge_graph = replace(
        graph,
        interactions=(),
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="do not equal selected"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=no_edge_graph,
            interaction_selection=pipeline.interaction_selection,
        )


def test_source_accounting_must_cover_every_compiled_fragment() -> None:
    _, sources = _compiled(with_interaction=False)
    trace, catalog, fisher, clusters, fragments, lowerings, graph = sources
    partial = build_modal_source_replacement_accounting(
        catalog,
        fragments,
        (fragments.fragments[0].fragment_id,),
    )

    with pytest.raises(ValueError, match="cover every compiled node"):
        build_modal_compiler_pipeline(
            source_prompt_trace=trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragments,
            lowerings_by_node=lowerings,
            graph_plan=graph,
            source_replacement_accounting=partial,
        )


def test_manifest_state_is_private_and_contains_only_executable_weights() -> None:
    pipeline, _ = _compiled()
    metadata = pipeline.metadata()
    for field in (
        "contains_prompt_text",
        "contains_token_ids",
        "contains_raw_prompt_rows",
        "contains_raw_activation_rows",
        "contains_raw_gradient_rows",
        "contains_grouped_fisher_score_rows",
        "contains_cluster_centroids",
        "contains_source_model_weights",
        "contains_source_parameter_values",
    ):
        assert metadata[field] is False
    assert metadata["contains_executable_modal_bases"] is True
    assert metadata["contains_executable_generator_weights"] is True
    assert metadata["contains_executable_graph"] is True
    assert metadata["executable"] is True

    forbidden = {
        "prompt_text",
        "prompts",
        "token_ids",
        "activation_rows",
        "gradient_rows",
        "score_factor",
        "assignments",
        "orientations",
        "similarities",
        "centroids",
        "source_model_weights",
        "source_parameter_values",
        "parameter_values",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(pipeline.state_dict())
