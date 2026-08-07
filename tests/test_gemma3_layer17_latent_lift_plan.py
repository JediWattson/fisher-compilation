from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from fisher_graph.gemma3_layer17_latent_lift_plan import (
    Gemma3Layer17LatentLiftMetricTriple,
    build_gemma3_layer17_latent_lift_edgeless_graph,
    build_gemma3_layer17_latent_lift_pareto_rejection,
    build_gemma3_layer17_latent_lift_plan,
)
from fisher_graph.gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_NATIVE_MODE_COUNTS,
)
from fisher_graph.modal_generator_graph import (
    LinearModalGeneratorNodeWeights,
    ModalGeneratorGraphPlan,
    ModalGeneratorInteraction,
    ModalGeneratorNode,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
)


MODEL_SHA = "a" * 64
PLAN_SHA = "b" * 64
CATALOG_SHA = "c" * 64
FISHER_SHA = "d" * 64
CLUSTER_SHA = "e" * 64
INPUT_CATALOG_SHA = "f" * 64
DTYPE = torch.float64


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fragment(
    *,
    layer: int,
    cluster: int,
    mode_count: int,
    offset: int,
) -> ParameterClusterLayerFragment:
    indices = tuple(range(offset, offset + mode_count))
    return ParameterClusterLayerFragment(
        cluster_id=cluster,
        layer_ordinal=layer,
        layer_id=f"model.layers.{layer}",
        activation_site=f"model.layers.{layer}.mlp.down_input",
        input_site=f"model.layers.{layer}.mlp.input",
        output_site=f"model.layers.{layer}.mlp.residual_delta",
        input_catalog_sha256=INPUT_CATALOG_SHA,
        input_width=640,
        output_width=640,
        group_indices=indices,
        channel_indices=indices,
        fisher_ranks=indices,
        axial_orientations=(1,) * mode_count,
        native_parameter_count=mode_count * 3 * 640,
        fisher_mass=float(mode_count),
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        source_model_sha256=MODEL_SHA,
    )


def _node(
    fragment: ParameterClusterLayerFragment,
    *,
    index: int,
    generator_rank: int,
) -> ModalGeneratorNode:
    mode_rank = 32
    weights = LinearModalGeneratorNodeWeights(
        generator_artifact_sha256=_sha(f"generator-{index}"),
        source_model_sha256=MODEL_SHA,
        parameter_cluster_plan_sha256=PLAN_SHA,
        input_factor=torch.zeros(640, generator_rank, dtype=DTYPE),
        state_factor=torch.zeros(generator_rank, mode_rank, dtype=DTYPE),
        output_factor=torch.zeros(mode_rank, 640, dtype=DTYPE),
        latent_bias=torch.zeros(mode_rank, dtype=DTYPE),
        output_bias=torch.zeros(640, dtype=DTYPE),
        state_kind="computational_mode_coordinates",
        computational_mode_basis_sha256=_sha(f"basis-{index}"),
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    return ModalGeneratorNode(
        name=f"l{fragment.layer_ordinal}.cluster-{fragment.cluster_id}",
        causal_order=fragment.layer_ordinal * 1_000_000 + index,
        input_boundary=fragment.input_site,
        output_boundary=fragment.output_site,
        weights=weights,
    )


def _source() -> tuple[
    ModalGeneratorGraphPlan,
    dict[str, ParameterClusterLayerFragment],
]:
    layer10_counts = (100, 80, 80, 74)
    layer10 = tuple(
        _fragment(
            layer=10,
            cluster=cluster,
            mode_count=count,
            offset=offset,
        )
        for cluster, count, offset in zip(
            (1, 2, 3, 4),
            layer10_counts,
            (0, 100, 180, 260),
            strict=True,
        )
    )
    layer17_clusters = tuple(
        int(fragment_id.split("/", 1)[0].split(".")[1])
        for fragment_id in LAYER17_FRAGMENT_IDS
    )
    layer17 = tuple(
        _fragment(
            layer=17,
            cluster=cluster,
            mode_count=count,
            offset=offset,
        )
        for cluster, count, offset in zip(
            layer17_clusters,
            LAYER17_NATIVE_MODE_COUNTS,
            (0, 54, 92, 177),
            strict=True,
        )
    )
    fragments = (*layer10, *layer17)
    nodes = tuple(
        _node(fragment, index=index, generator_rank=16)
        for index, fragment in enumerate(fragments)
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA,
        parameter_cluster_plan_sha256=PLAN_SHA,
        nodes=nodes,
        interactions=(),
    )
    return graph, {
        node.name: fragment
        for node, fragment in zip(nodes, fragments, strict=True)
    }


def _subgraph(
    graph: ModalGeneratorGraphPlan,
    *,
    layer: int,
    generator_rank: int | None = None,
) -> ModalGeneratorGraphPlan:
    nodes = tuple(
        node
        for node in graph.nodes
        if node.name.startswith(f"l{layer}.")
    )
    if generator_rank is not None:
        nodes = tuple(
            _node(
                replace(
                    next(
                        fragment
                        for fragment in _source()[1].values()
                        if fragment.artifact_sha256
                        == node.weights.parameter_cluster_fragment_sha256
                    )
                ),
                index=index + 4,
                generator_rank=generator_rank,
            )
            for index, node in enumerate(nodes)
        )
    return ModalGeneratorGraphPlan(
        model_fingerprint=graph.model_fingerprint,
        parameter_cluster_plan_sha256=graph.parameter_cluster_plan_sha256,
        nodes=nodes,
        interactions=(),
    )


def test_plans_exact_layer17_and_combined_resources() -> None:
    graph, fragments = _source()
    plan = build_gemma3_layer17_latent_lift_plan(
        graph,
        fragments_by_node=fragments,
    )
    resources = plan.resources

    assert plan.edge_policy == "edgeless"
    assert plan.interaction_count == 0
    assert plan.scientific_status == (
        "rejected_open_development_diagnostic_arm"
    )
    assert plan.mode_rank == 32
    assert plan.source_generator_rank == 16
    assert plan.target_generator_rank == 32
    assert set(plan.layer17_fragment_ids) == set(LAYER17_FRAGMENT_IDS)
    assert resources.layer17_source_parameters == 127_616
    assert resources.layer17_target_parameters == 170_624
    assert resources.layer17_source_macs_per_token == 124_928
    assert resources.layer17_target_macs_per_token == 167_936
    assert resources.rank_lift_added_parameters == 43_008
    assert resources.rank_lift_added_macs_per_token == 43_008
    assert resources.combined_native_removed_parameters == 1_082_880
    assert resources.combined_target_parameters == 298_240
    assert resources.combined_target_macs_per_token == 292_864
    assert resources.combined_target_net_parameter_savings == 784_640
    assert resources.combined_target_net_macs_saved_per_token == 790_016
    assert resources.combined_target_parameter_fraction == pytest.approx(
        0.2754137116
    )
    plan.validate_integrity()
    json.dumps(plan.state_dict(), allow_nan=False)


def test_records_source_safe_three_metric_pareto_rejection() -> None:
    graph, fragments = _source()
    plan = build_gemma3_layer17_latent_lift_plan(
        graph,
        fragments_by_node=fragments,
    )
    rank16 = Gemma3Layer17LatentLiftMetricTriple(
        delta_nll_per_token=0.095999,
        native_to_candidate_kl_per_token=0.072458,
        top1_agreement_to_native=0.84929,
    )
    rank32 = Gemma3Layer17LatentLiftMetricTriple(
        delta_nll_per_token=0.108179,
        native_to_candidate_kl_per_token=0.074074,
        top1_agreement_to_native=0.84772,
    )
    decision = build_gemma3_layer17_latent_lift_pareto_rejection(
        plan,
        evidence_example_count=128,
        rank16=rank16,
        rank32=rank32,
    )

    assert decision.pareto_rejected is True
    assert decision.scientific_status == (
        "rejected_open_development_diagnostic_arm"
    )
    assert decision.delta_nll_rank32_minus_rank16 == pytest.approx(0.01218)
    assert decision.delta_kl_rank32_minus_rank16 == pytest.approx(0.001616)
    assert decision.delta_top1_rank32_minus_rank16 == pytest.approx(-0.00157)
    state = decision.state_dict()
    assert state["safety"] == {
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_data_rows": False,
        "contains_activations": False,
        "contains_gradients": False,
        "aggregate_metrics_only": True,
    }
    decision.validate_integrity()
    json.dumps(state, allow_nan=False)

    not_dominated = Gemma3Layer17LatentLiftMetricTriple(
        delta_nll_per_token=0.09,
        native_to_candidate_kl_per_token=0.074074,
        top1_agreement_to_native=0.84772,
    )
    with pytest.raises(ValueError, match="not Pareto-worse"):
        build_gemma3_layer17_latent_lift_pareto_rejection(
            plan,
            evidence_example_count=128,
            rank16=rank16,
            rank32=not_dominated,
        )


def test_rejects_edges_and_layer17_source_rank_drift() -> None:
    graph, fragments = _source()
    edge = ModalGeneratorInteraction(
        source_node=graph.nodes[0].name,
        target_node=graph.nodes[1].name,
        message_matrix=torch.zeros(32, 32, dtype=DTYPE),
        message_bias=torch.zeros(32, dtype=DTYPE),
    )
    edged = ModalGeneratorGraphPlan(
        model_fingerprint=graph.model_fingerprint,
        parameter_cluster_plan_sha256=graph.parameter_cluster_plan_sha256,
        nodes=graph.nodes,
        interactions=(edge,),
    )
    with pytest.raises(ValueError, match="rejects every dynamic or static edge"):
        build_gemma3_layer17_latent_lift_plan(
            edged,
            fragments_by_node=fragments,
        )

    lifted_layer17 = _subgraph(graph, layer=17, generator_rank=32)
    drifted = ModalGeneratorGraphPlan(
        model_fingerprint=graph.model_fingerprint,
        parameter_cluster_plan_sha256=graph.parameter_cluster_plan_sha256,
        nodes=(*graph.nodes[:4], *lifted_layer17.nodes),
        interactions=(),
    )
    with pytest.raises(ValueError, match="rank-16 into mode rank 32"):
        build_gemma3_layer17_latent_lift_plan(
            drifted,
            fragments_by_node=fragments,
        )


def test_builds_only_the_planned_edgeless_graph() -> None:
    source, fragments = _source()
    plan = build_gemma3_layer17_latent_lift_plan(
        source,
        fragments_by_node=fragments,
    )
    layer10 = _subgraph(source, layer=10)
    layer17 = _subgraph(source, layer=17, generator_rank=32)
    candidate = build_gemma3_layer17_latent_lift_edgeless_graph(
        plan,
        layer10_parent_graph=layer10,
        fitted_layer17_graph=layer17,
    )

    assert not candidate.interactions
    assert candidate.nodes[:4] == layer10.nodes
    assert candidate.parameter_count == 298_240
    assert candidate.macs_per_token == 292_864

    edge = ModalGeneratorInteraction(
        source_node=layer17.nodes[0].name,
        target_node=layer17.nodes[1].name,
        message_matrix=torch.zeros(32, 32, dtype=DTYPE),
        message_bias=torch.zeros(32, dtype=DTYPE),
    )
    edged_layer17 = ModalGeneratorGraphPlan(
        model_fingerprint=layer17.model_fingerprint,
        parameter_cluster_plan_sha256=layer17.parameter_cluster_plan_sha256,
        nodes=layer17.nodes,
        interactions=(edge,),
    )
    with pytest.raises(ValueError, match="rejects edges"):
        build_gemma3_layer17_latent_lift_edgeless_graph(
            plan,
            layer10_parent_graph=layer10,
            fitted_layer17_graph=edged_layer17,
        )
