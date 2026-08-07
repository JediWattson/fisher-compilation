from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fisher_graph.gemma3_layer17_capped_node_fit as capped
from fisher_graph.computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    ComputationalModeConfig,
)
from fisher_graph.gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_NATIVE_MODE_COUNTS,
    build_layer17_node_rank_resource_row,
    resolve_layer17_node_ranks,
)
from fisher_graph.gemma3_same_layer_shape_flow import (
    build_edgeless_same_layer_graph,
    select_top_fisher_same_layer_fragments,
)
from fisher_graph.modal_generator_lowering import (
    ModalGeneratorLowering,
    lower_coordinate_modal_generator,
)
from fisher_graph.modal_generators import (
    ModalGeneratorBinding,
    ModalGeneratorConfig,
    ModalGeneratorFactors,
    ModalGeneratorPlan,
)
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)


MODEL_SHA = "a" * 64
CLUSTER_SHA = "b" * 64
FISHER_SHA = "c" * 64
CATALOG_SHA = "d" * 64
INPUT_CATALOG_SHA = "e" * 64
FIT_SHA = "f" * 64
SELECTION_SHA = "1" * 64


def _fragment_plan() -> ParameterClusterLayerFragmentPlan:
    clusters = (0, 28, 34, 54)
    offsets = (0, 54, 92, 177)
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=cluster,
            layer_ordinal=17,
            layer_id="model.layers.17",
            activation_site="model.layers.17.mlp.down_input",
            input_site="model.layers.17.mlp.input",
            output_site="model.layers.17.mlp.residual_delta",
            input_catalog_sha256=INPUT_CATALOG_SHA,
            input_width=640,
            output_width=640,
            group_indices=tuple(range(offset, offset + count)),
            channel_indices=tuple(range(offset, offset + count)),
            fisher_ranks=tuple(range(offset, offset + count)),
            axial_orientations=(1,) * count,
            native_parameter_count=count * 3 * 640,
            fisher_mass=float(4 - index),
            source_cluster_plan_sha256=CLUSTER_SHA,
            source_fisher_coupling_sha256=FISHER_SHA,
            parameter_catalog_sha256=CATALOG_SHA,
            source_model_sha256=MODEL_SHA,
        )
        for index, (cluster, count, offset) in enumerate(
            zip(clusters, LAYER17_NATIVE_MODE_COUNTS, offsets, strict=True)
        )
    )
    return ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        source_model_sha256=MODEL_SHA,
        cluster_count=55,
        source_group_count=sum(LAYER17_NATIVE_MODE_COUNTS),
        assigned_group_count=sum(LAYER17_NATIVE_MODE_COUNTS),
        assigned_native_parameter_count=sum(
            fragment.native_parameter_count for fragment in fragments
        ),
        fragments=fragments,
    )


def _lowering(
    plan: ParameterClusterLayerFragmentPlan,
    fragment: ParameterClusterLayerFragment,
    *,
    mode_rank: int,
    generator_rank: int,
) -> ModalGeneratorLowering:
    mode_binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind="layer_fragment",
        output_site=fragment.output_site,
        source_model_sha256=MODEL_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        fisher_coupling_sha256=FISHER_SHA,
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256=FIT_SHA,
        eval_split_sha256=SELECTION_SHA,
    )
    basis = ComputationalModeBasis(
        binding=mode_binding,
        config=ComputationalModeConfig(
            ranks=(mode_rank,),
            selection_rule="fixed_rank",
            selected_rank=mode_rank,
        ),
        rank=mode_rank,
        mean_bias=torch.ones(640, dtype=torch.float64),
        encoder_basis=torch.eye(640, dtype=torch.float64)[:, :mode_rank],
    )
    generator_binding = ModalGeneratorBinding.create(
        generator_id=f"layer.17.cluster.{fragment.cluster_id}.coordinates",
        input_kind="native_layer_input",
        input_site=fragment.input_site,
        output_site=fragment.output_site,
        source_model_sha256=MODEL_SHA,
        input_catalog_sha256=INPUT_CATALOG_SHA,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=plan.artifact_sha256,
        fit_split_sha256=FIT_SHA,
        eval_split_sha256=SELECTION_SHA,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=FISHER_SHA,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    factors = ModalGeneratorFactors(
        rank=generator_rank,
        input_factor=torch.zeros(
            640,
            generator_rank,
            dtype=torch.float64,
        ),
        output_factor=torch.zeros(
            generator_rank,
            mode_rank,
            dtype=torch.float64,
        ),
        bias=torch.ones(mode_rank, dtype=torch.float64),
    )
    parameter_count = 640 * generator_rank + generator_rank * mode_rank + mode_rank
    coordinate_plan = ModalGeneratorPlan(
        binding=generator_binding,
        config=ModalGeneratorConfig(
            ranks=(generator_rank,),
            fit_intercept=True,
            selection_rule="fixed_rank",
            selected_rank=generator_rank,
        ),
        factors=factors,
        parameter_count=parameter_count,
        macs_per_token=parameter_count,
    )
    return lower_coordinate_modal_generator(coordinate_plan, basis, plan)


def _candidate_parts(mode_rank_cap: int = 32, generator_rank: int = 16):
    plan = _fragment_plan()
    selection = select_top_fisher_same_layer_fragments(
        plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    ranks = resolve_layer17_node_ranks(mode_rank_cap)
    lowerings = {
        fragment.fragment_id: _lowering(
            plan,
            fragment,
            mode_rank=rank,
            generator_rank=generator_rank,
        )
        for fragment, rank in zip(selection.execution_order, ranks, strict=True)
    }
    graph = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=plan,
        lowerings_by_fragment=lowerings,
    )
    return plan, selection, graph


def _splits() -> dict[str, object]:
    fit_export_sha = "2" * 64
    selection_export_sha = "3" * 64

    def tokenized(split: str, digest: str, content: str) -> dict[str, object]:
        return {
            "schema": "fisher_graph.tokenized_calibration_stream",
            "format_version": 1,
            "split": split,
            "batches": 1,
            "sequences": 1,
            "serialized_sha256": digest,
            "source_prompt_sha256": ("4" * 64,),
            "content_sha256": (content,),
            "valid_tokens": {"minimum": 5, "maximum": 5, "total": 5},
            "supervised_positions": {
                "minimum": 4,
                "maximum": 4,
                "total": 4,
            },
            "contains_prompt_text": False,
            "contains_token_ids": False,
        }

    return {
        "policy": {
            "fit_export_sha256": fit_export_sha,
            "eval_export_sha256": selection_export_sha,
            "prompt_disjoint": True,
            "source_prompt_index_disjoint": True,
            "heldout_guard_used": False,
            "calibration_b_used": False,
            "validation_used": False,
            "test_used": False,
        },
        "fit_export": {"artifact_sha256": fit_export_sha},
        "selection_export": {"artifact_sha256": selection_export_sha},
        "fit_tokenized": tokenized("fit", FIT_SHA, "5" * 64),
        "selection_tokenized": tokenized(
            "selection",
            SELECTION_SHA,
            "6" * 64,
        ),
    }


def _evaluation(graph, *, mode_rank_cap: int, generator_rank: int):
    row = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=mode_rank_cap,
        generator_rank=generator_rank,
        edge_policy="edgeless",
    )
    valid = 5
    source_whole = 1_000_000
    removed = row.source_macs_per_token * valid
    graph_macs = row.graph_dense_macs_per_token * valid
    additions = graph.graph_plan.accounting.elementwise_additions_per_token * valid
    metrics = {
        "nll_per_token": 1.1,
        "delta_nll_per_token": 0.1,
        "native_to_candidate_kl_per_token": 0.01,
        "top1_agreement_to_native": 0.9,
    }
    return {
        "execution_path": "incremental_modal_generator_graph_traversal",
        "supervised_tokens": 4,
        "logical_valid_tokens": valid,
        "native": {"nll_per_token": 1.0},
        "conditions": {
            "generated": metrics,
            "deletion": {**metrics, "delta_nll_per_token": 0.2},
        },
        "graph": {
            "node_count": 4,
            "interaction_count": 0,
            "traversal_order": graph.graph_plan.traversal_order,
        },
        "resource_accounting": {
            "replacement_scope": "partial_native_mlp_mode_replacement",
            "replaced_layer_count": 1,
            "graph_node_count": 4,
            "fragment_count": 4,
            "removed_mode_count": sum(LAYER17_NATIVE_MODE_COUNTS),
            "source_whole_model_learned_parameters": source_whole,
            "candidate_whole_model_learned_parameters": (
                source_whole
                - row.source_parameter_count
                + row.graph_parameter_count
            ),
            "native_removed_learned_parameters": row.source_parameter_count,
            "modal_graph_learned_parameters": row.graph_parameter_count,
            "net_stored_parameter_savings": row.net_parameter_savings,
            "graph_runtime_storage": (
                "registered_copied_device_local_graph_parameters"
            ),
            "planned_peak_live_modal_width": max(row.node_ranks),
            "generated": {
                "logical_linear_macs_native_removed": removed,
                "logical_modal_graph_macs": graph_macs,
                "logical_executed_modal_graph_macs": graph_macs,
                "logical_modal_graph_additions": additions,
                "logical_executed_modal_graph_additions": additions,
                "executed_peak_live_modal_width": max(row.node_ranks),
                "net_logical_macs_saved": removed - graph_macs,
            },
            "deletion": {
                "logical_linear_macs_native_removed": removed,
                "logical_modal_graph_macs": graph_macs,
                "logical_executed_modal_graph_macs": 0,
                "logical_modal_graph_additions": additions,
                "logical_executed_modal_graph_additions": 0,
                "executed_peak_live_modal_width": 0,
                "net_logical_macs_saved": removed,
            },
            "parameter_savings_positive": row.net_parameter_savings > 0,
            "logical_mac_savings_positive_generated": (
                row.net_dense_macs_saved_per_token > 0
            ),
            "latency_or_kernel_speed_claim": False,
        },
    }


def _build_arguments(mode_rank_cap: int = 32, generator_rank: int = 16):
    _, selection, graph = _candidate_parts(mode_rank_cap, generator_rank)
    return {
        "experiment": {
            "experiment_kind": "synthetic_layer17_capped_node",
            "heldout_confirmation": False,
        },
        "splits": _splits(),
        "evaluation": _evaluation(
            graph,
            mode_rank_cap=mode_rank_cap,
            generator_rank=generator_rank,
        ),
        "mode_rank_cap": mode_rank_cap,
        "generator_rank": generator_rank,
        "selection": selection,
        "lowerings_by_node": graph.lowerings_by_node,
        "edgeless_graph": graph.graph_plan,
        "compiler_pipeline": None,
    }


def test_cap64_resolves_per_frozen_fragment_and_names_output() -> None:
    assert resolve_layer17_node_ranks(64) == (54, 38, 64, 53)
    assert capped.default_gemma3_layer17_capped_node_output(64, 16) == Path(
        ".local-runs/google--gemma-3-270m/"
        "layer17-capped-node-c64-r16-edgeless-dev-v1.pt"
    )


def test_executable_candidate_roundtrips_with_exact_resources(tmp_path) -> None:
    arguments = _build_arguments()
    output = tmp_path / "candidate.pt"
    report = capped.save_gemma3_layer17_capped_node_candidate(
        output,
        **arguments,
    )
    loaded = capped.load_gemma3_layer17_capped_node_candidate(output)
    graph, lowerings, pipeline = capped.restore_gemma3_layer17_capped_node_runtime(
        loaded
    )
    row = build_layer17_node_rank_resource_row(
        label="candidate",
        mode_rank_cap=32,
        generator_rank=16,
        edge_policy="edgeless",
    )

    assert graph.interactions == ()
    assert graph.parameter_count == row.graph_parameter_count
    assert graph.macs_per_token == row.graph_dense_macs_per_token
    assert len(lowerings) == 4
    assert pipeline is None
    assert report["graph"]["interaction_count"] == 0
    assert report["safety"]["contains_tensors"] is False
    assert output.with_suffix(".json").is_file()


def test_loader_rejects_evaluation_and_split_lineage_tamper() -> None:
    payload = capped.build_gemma3_layer17_capped_node_candidate(
        **_build_arguments()
    )
    empty_evaluation = copy.deepcopy(payload)
    empty_evaluation["evaluation"] = {}
    with pytest.raises(ValueError, match="evaluation fields"):
        capped.restore_gemma3_layer17_capped_node_runtime(empty_evaluation)

    wrong_split = copy.deepcopy(payload)
    wrong_split["splits"]["fit_tokenized"]["serialized_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="fit_tokenized lineage"):
        capped.restore_gemma3_layer17_capped_node_runtime(wrong_split)


def test_per_fragment_fitter_receives_cap64_ranks(monkeypatch) -> None:
    plan = _fragment_plan()
    selection = select_top_fisher_same_layer_fragments(
        plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(capped, "fit_layer_cluster_modal_generator", fake_fit)
    rows = SimpleNamespace(
        rows_by_fragment={fragment_id: object() for fragment_id in LAYER17_FRAGMENT_IDS}
    )
    result = capped.fit_layer17_capped_node_pilots(
        rows,
        rows,
        selection=selection,
        source_model_sha256=MODEL_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        fisher_coupling_sha256=FISHER_SHA,
        fragment_plan=plan,
        fit_split_sha256=FIT_SHA,
        selection_split_sha256=SELECTION_SHA,
        mode_rank_cap=64,
        generator_rank=16,
    )

    assert tuple(result) == LAYER17_FRAGMENT_IDS
    assert tuple(call["selected_mode_rank"] for call in calls) == (
        54,
        38,
        64,
        53,
    )
    assert all(call["generator_ranks"] == (16,) for call in calls)


def test_live_entrypoint_resolves_upstream_loader_name(monkeypatch, tmp_path) -> None:
    export = SimpleNamespace(metadata=lambda: {})
    monkeypatch.setattr(capped, "load_development_prompt_export", lambda path: export)
    monkeypatch.setattr(
        capped,
        "validate_development_split_pair",
        lambda fit, selection: {},
    )

    def stop_after_loader(path):
        raise RuntimeError("upstream-loader-called")

    monkeypatch.setattr(
        capped,
        "load_gemma3_modal_generator_dev_artifact",
        stop_after_loader,
    )
    with pytest.raises(RuntimeError, match="upstream-loader-called"):
        capped.fit_gemma3_layer17_capped_node_candidate(
            revision="a" * 40,
            output=tmp_path / "candidate.pt",
            mode_rank_cap=64,
            generator_rank=16,
        )


def test_cli_exposes_only_open_fit_inputs() -> None:
    parser = capped.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--fit-export" in options
    assert "--selection-export" in options
    assert not any(
        forbidden in option
        for option in options
        for forbidden in ("guard", "calibration-b", "validation", "test")
    )
