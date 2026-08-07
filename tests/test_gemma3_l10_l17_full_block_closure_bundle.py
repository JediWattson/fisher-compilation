from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from fisher_graph.computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    ComputationalModeConfig,
)
from fisher_graph.gemma3_l10_l17_full_block_closure_bundle import (
    build_gemma3_l10_l17_full_block_closure_fold_bundle,
    load_gemma3_l10_l17_full_block_closure_fold_bundle,
    restore_gemma3_l10_l17_full_block_closure_fold,
    save_gemma3_l10_l17_full_block_closure_fold_bundle,
    validate_gemma3_l10_l17_full_block_closure_fold_bundle,
)
from fisher_graph.modal_generator_graph import (
    ModalGeneratorGraphExecutor,
    ModalGeneratorGraphPlan,
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
EVAL_SHA = "1" * 64
RUNTIME_CATALOG_SHA = "2" * 64
SOURCE_COMPOSITION_SHA = "3" * 64
NORMALIZED_INPUT = "layer.17.mlp.normalized_input"
RAW_OUTPUT = "layer.17.mlp.operator_output"
POST_FEEDFORWARD_DELTA = "layer.17.mlp.delta"
MODE_RANKS = (48, 38, 48, 48)


def _fragment_plan() -> ParameterClusterLayerFragmentPlan:
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=cluster_id,
            layer_ordinal=17,
            layer_id="model.layers.17",
            activation_site="model.layers.17.mlp.down_input",
            input_site=NORMALIZED_INPUT,
            output_site=RAW_OUTPUT,
            input_catalog_sha256=INPUT_CATALOG_SHA,
            input_width=640,
            output_width=640,
            group_indices=(cluster_id,),
            channel_indices=(cluster_id,),
            fisher_ranks=(cluster_id,),
            axial_orientations=(1,),
            native_parameter_count=1_920,
            fisher_mass=float(cluster_id + 1),
            source_cluster_plan_sha256=CLUSTER_SHA,
            source_fisher_coupling_sha256=FISHER_SHA,
            parameter_catalog_sha256=CATALOG_SHA,
            source_model_sha256=MODEL_SHA,
        )
        for cluster_id in range(4)
    )
    return ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        source_model_sha256=MODEL_SHA,
        cluster_count=4,
        source_group_count=4,
        assigned_group_count=4,
        assigned_native_parameter_count=7_680,
        fragments=fragments,
    )


def _lowering(
    plan: ParameterClusterLayerFragmentPlan,
    fragment: ParameterClusterLayerFragment,
    *,
    mode_rank: int,
    output_boundary: str,
) -> ModalGeneratorLowering:
    relocated = output_boundary != fragment.output_site
    mode_binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind=(
            "relocated_layer_fragment" if relocated else "layer_fragment"
        ),
        output_site=output_boundary,
        source_model_sha256=MODEL_SHA,
        parameter_catalog_sha256=CATALOG_SHA,
        fisher_coupling_sha256=FISHER_SHA,
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256=FIT_SHA,
        eval_split_sha256=EVAL_SHA,
    )
    # Identity columns are orthonormal and already obey the basis sign rule.
    basis = ComputationalModeBasis(
        binding=mode_binding,
        config=ComputationalModeConfig(
            ranks=(mode_rank,),
            selection_rule="fixed_rank",
            selected_rank=mode_rank,
        ),
        rank=mode_rank,
        mean_bias=torch.full(
            (640,),
            0.001 * (fragment.cluster_id + 1),
            dtype=torch.float64,
        ),
        encoder_basis=torch.eye(640, dtype=torch.float64)[:, :mode_rank],
    )
    generator_binding = ModalGeneratorBinding.create(
        generator_id=f"layer.17.cluster.{fragment.cluster_id}.coordinates",
        input_kind="native_layer_input",
        input_site=NORMALIZED_INPUT,
        output_site=output_boundary,
        source_model_sha256=MODEL_SHA,
        input_catalog_sha256=INPUT_CATALOG_SHA,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=plan.artifact_sha256,
        fit_split_sha256=FIT_SHA,
        eval_split_sha256=EVAL_SHA,
        target_kind=(
            "relocated_computational_mode_coordinates"
            if relocated
            else "computational_mode_coordinates"
        ),
        fisher_coupling_sha256=FISHER_SHA,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
        source_generator_plan_sha256=("9" * 64 if relocated else None),
    )
    factors = ModalGeneratorFactors(
        rank=16,
        input_factor=torch.full(
            (640, 16),
            0.0001 * (fragment.cluster_id + 1),
            dtype=torch.float64,
        ),
        output_factor=torch.full(
            (16, mode_rank),
            0.0002 * (fragment.cluster_id + 1),
            dtype=torch.float64,
        ),
        bias=torch.full(
            (mode_rank,),
            0.0003 * (fragment.cluster_id + 1),
            dtype=torch.float64,
        ),
    )
    generator = ModalGeneratorPlan(
        binding=generator_binding,
        config=ModalGeneratorConfig(
            ranks=(16,),
            fit_intercept=True,
            selection_rule="fixed_rank",
            selected_rank=16,
        ),
        factors=factors,
        parameter_count=640 * 16 + 16 * mode_rank + mode_rank,
        macs_per_token=640 * 16 + 16 * mode_rank + mode_rank,
    )
    return lower_coordinate_modal_generator(generator, basis, plan)


def _runtime(
    *,
    output_boundary: str = POST_FEEDFORWARD_DELTA,
) -> tuple[ModalGeneratorGraphPlan, dict[str, ModalGeneratorLowering]]:
    fragment_plan = _fragment_plan()
    names = tuple(
        f"gemma3.layer-17.cluster-{index}.modal-generator.graph-node"
        for index in range(4)
    )
    lowerings = tuple(
        _lowering(
            fragment_plan,
            fragment,
            mode_rank=mode_rank,
            output_boundary=output_boundary,
        )
        for fragment, mode_rank in zip(
            fragment_plan.fragments,
            MODE_RANKS,
            strict=True,
        )
    )
    nodes = tuple(
        lowering.to_graph_node(
            name=name,
            causal_order=17_000_000 + index,
            input_boundary=NORMALIZED_INPUT,
            output_boundary=output_boundary,
        )
        for index, (name, lowering) in enumerate(
            zip(names, lowerings, strict=True)
        )
    )
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_SHA,
        parameter_cluster_plan_sha256=fragment_plan.artifact_sha256,
        nodes=nodes,
        interactions=(),
    )
    return graph, dict(zip(names, lowerings, strict=True))


def _folds(
    *,
    output_boundary: str = POST_FEEDFORWARD_DELTA,
) -> list[dict[str, object]]:
    graph, lowerings = _runtime(output_boundary=output_boundary)
    return [
        {
            "fold_id": f"fold-{index}",
            "held_family_alias": f"family-{index}",
            "protocol_fold_sha256": f"{index + 10:064x}",
            "fit_split_sha256": f"{index + 20:064x}",
            "held_split_sha256": f"{index + 30:064x}",
            "graph_plan": graph,
            "lowerings_by_node": lowerings,
        }
        for index in range(8)
    ]


def _bundle() -> dict[str, object]:
    return build_gemma3_l10_l17_full_block_closure_fold_bundle(
        model_fingerprint=MODEL_SHA,
        source_runtime_catalog_sha256=RUNTIME_CATALOG_SHA,
        source_composition_graph_sha256=SOURCE_COMPOSITION_SHA,
        folds=_folds(),
    )


def test_build_validate_restore_and_execute_exact_postnorm_fold() -> None:
    bundle = _bundle()
    validated = validate_gemma3_l10_l17_full_block_closure_fold_bundle(
        bundle
    )

    assert validated["scientific_payload_sha256"] == (
        bundle["scientific_payload_sha256"]
    )
    assert validated["safety"]["contains_prompt_text"] is False
    assert validated["safety"]["contains_activation_or_gradient_tensors"] is (
        False
    )
    assert len(validated["folds"]) == 8
    assert {
        record["held_family_alias"] for record in validated["folds"]
    } == {f"family-{index}" for index in range(8)}

    graph, lowerings = restore_gemma3_l10_l17_full_block_closure_fold(
        validated,
        3,
    )
    assert graph.parameter_count == 163_094
    assert graph.macs_per_token == 160_352
    assert graph.interactions == ()
    assert set(lowerings) == set(graph.traversal_order)
    assert all(
        node.input_boundary == NORMALIZED_INPUT
        and node.output_boundary == POST_FEEDFORWARD_DELTA
        for node in graph.nodes
    )
    assert all(
        lowering.coordinate_generator_plan.binding.output_site
        == POST_FEEDFORWARD_DELTA
        and lowering.computational_mode_basis.binding.output_site
        == POST_FEEDFORWARD_DELTA
        for lowering in lowerings.values()
    )

    execution = ModalGeneratorGraphExecutor(graph).execute(
        {NORMALIZED_INPUT: torch.zeros(2, 640, dtype=torch.float64)}
    )
    assert execution.outputs[POST_FEEDFORWARD_DELTA].shape == (2, 640)


def test_builder_rejects_raw_mlp_boundary_and_runtime_mismatch() -> None:
    with pytest.raises(ValueError, match="fixed A4 Layer17 graph"):
        build_gemma3_l10_l17_full_block_closure_fold_bundle(
            model_fingerprint=MODEL_SHA,
            source_runtime_catalog_sha256=RUNTIME_CATALOG_SHA,
            source_composition_graph_sha256=SOURCE_COMPOSITION_SHA,
            folds=_folds(output_boundary=RAW_OUTPUT),
        )

    folds = _folds()
    folds[0] = dict(folds[0])
    lowerings = dict(folds[0]["lowerings_by_node"])
    lowerings.pop(next(iter(lowerings)))
    folds[0]["lowerings_by_node"] = lowerings
    with pytest.raises(ValueError, match="catalog differs from graph"):
        build_gemma3_l10_l17_full_block_closure_fold_bundle(
            model_fingerprint=MODEL_SHA,
            source_runtime_catalog_sha256=RUNTIME_CATALOG_SHA,
            source_composition_graph_sha256=SOURCE_COMPOSITION_SHA,
            folds=folds,
        )


def test_validator_rejects_forbidden_fields_safety_and_tensor_tampering() -> None:
    bundle = _bundle()

    forbidden = copy.deepcopy(bundle)
    forbidden["prompt_text"] = "must never be serialized"
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_gemma3_l10_l17_full_block_closure_fold_bundle(forbidden)

    safety = copy.deepcopy(bundle)
    safety["safety"]["contains_prompt_text"] = True
    with pytest.raises(ValueError, match="identity/safety boundary"):
        validate_gemma3_l10_l17_full_block_closure_fold_bundle(safety)

    tensor = copy.deepcopy(bundle)
    tensor["folds"][0]["lowering_records"][0]["lowering"][
        "graph_weights"
    ]["input_factor"][0, 0] += 1.0
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        validate_gemma3_l10_l17_full_block_closure_fold_bundle(tensor)

    metadata = copy.deepcopy(bundle)
    metadata["folds"][0]["application_boundary"] = RAW_OUTPUT
    with pytest.raises(ValueError, match="metadata drifted"):
        validate_gemma3_l10_l17_full_block_closure_fold_bundle(metadata)

    aliases = copy.deepcopy(bundle)
    aliases["folds"][1]["held_family_alias"] = aliases["folds"][0][
        "held_family_alias"
    ]
    with pytest.raises(ValueError, match="aliases are not unique"):
        validate_gemma3_l10_l17_full_block_closure_fold_bundle(aliases)


def test_restore_rejects_noncanonical_fold_indices() -> None:
    bundle = _bundle()
    for index in (-1, 8, True, 1.0):
        with pytest.raises(IndexError, match="out of range"):
            restore_gemma3_l10_l17_full_block_closure_fold(bundle, index)


def test_save_load_companion_report_and_refuse_every_collision(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    output = tmp_path / "a4-folds.pt"
    report = save_gemma3_l10_l17_full_block_closure_fold_bundle(
        output,
        bundle,
    )
    loaded = load_gemma3_l10_l17_full_block_closure_fold_bundle(output)
    with output.with_suffix(".json").open(encoding="utf-8") as handle:
        on_disk_report = json.load(handle)

    assert report == on_disk_report
    assert report["fold_count"] == 8
    assert report["contains_executable_generator_weights"] is True
    assert report["contains_prompt_or_activation_rows"] is False
    assert loaded["scientific_payload_sha256"] == (
        bundle["scientific_payload_sha256"]
    )
    assert len(report["tensor_file_sha256"]) == 64

    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_l10_l17_full_block_closure_fold_bundle(output, bundle)

    report_collision = tmp_path / "report-collision.json"
    report_collision.write_text("preserve-me\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_l10_l17_full_block_closure_fold_bundle(
            report_collision.with_suffix(".pt"),
            bundle,
        )
    assert report_collision.read_text(encoding="utf-8") == "preserve-me\n"
    assert not report_collision.with_suffix(".pt").exists()

    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_l10_l17_full_block_closure_fold_bundle(
            tmp_path / "wrong-extension.bin",
            bundle,
        )
