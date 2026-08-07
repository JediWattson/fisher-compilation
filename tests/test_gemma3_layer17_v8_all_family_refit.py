from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_layer17_v8_all_family_refit as refit
import fisher_graph.gemma3_layer17_family_lofo_authority as lofo_authority
from fisher_graph.computational_modes import (
    ComputationalModeBasis,
    ComputationalModeBinding,
    ComputationalModeConfig,
)
from fisher_graph.gemma3_layer17_capped_node_fit import (
    fit_layer17_capped_node_pilots,
)
from fisher_graph.gemma3_layer17_family_lofo_protocol import (
    FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
    V8_FAMILY_LOFO_FAMILY_ALIASES,
    build_default_v8_layer17_family_lofo_protocol,
)
from fisher_graph.gemma3_layer17_node_rank_ladder import (
    LAYER17_FRAGMENT_IDS,
    LAYER17_NATIVE_MODE_COUNTS,
)
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    LayerFragmentRows,
)
from fisher_graph.gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
)
from fisher_graph.gemma3_same_layer_shape_flow import (
    build_edgeless_same_layer_graph,
    select_top_fisher_same_layer_fragments,
)
from fisher_graph.modal_compiler_pipeline import (
    AuthenticatedArtifactReference,
    ModalCompilerNodeArtifact,
    ModalCompilerPipeline,
    ModalRefitFisherAuthority,
    build_modal_source_replacement_accounting,
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
from fisher_graph.parameter_fisher_coupling import (
    NaturalMLPLayerParameterSpec,
    NaturalMLPParameterGroupCatalog,
    build_natural_mlp_parameter_group_catalog,
    natural_mlp_input_catalog_sha256,
)


MODEL_SHA = "a" * 64
CLUSTER_SHA = "b" * 64
FISHER_SHA = "c" * 64
CATALOG_SHA = "d" * 64
AUTHORITY_SHA = "e" * 64
MATERIALIZATION_SHA = "f" * 64
LOFO_SHA = "1" * 64
RUNTIME_FIT_SHA = "2" * 64
RUNTIME_DIAGNOSTIC_SHA = "3" * 64
TOPOLOGY_FIT_SHA = "4" * 64
OBJECTIVE_SHA = "5" * 64


def _fragment_plan() -> ParameterClusterLayerFragmentPlan:
    input_width = 20
    output_width = 64
    input_site = "model.layers.17.mlp.input"
    input_catalog_sha256 = natural_mlp_input_catalog_sha256(
        source_model_sha256=MODEL_SHA,
        input_site=input_site,
        input_width=input_width,
    )
    clusters = (0, 28, 34, 54)
    offsets = (0, 54, 92, 177)
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=cluster,
            layer_ordinal=17,
            layer_id="model.layers.17",
            activation_site="model.layers.17.mlp.down_input",
            input_site=input_site,
            output_site="model.layers.17.mlp.residual_delta",
            input_catalog_sha256=input_catalog_sha256,
            input_width=input_width,
            output_width=output_width,
            group_indices=tuple(range(offset, offset + count)),
            channel_indices=tuple(range(offset, offset + count)),
            fisher_ranks=tuple(range(offset, offset + count)),
            axial_orientations=(1,) * count,
            native_parameter_count=count * (2 * input_width + output_width),
            fisher_mass=float(4 - index),
            source_cluster_plan_sha256=CLUSTER_SHA,
            source_fisher_coupling_sha256=FISHER_SHA,
            parameter_catalog_sha256=CATALOG_SHA,
            source_model_sha256=MODEL_SHA,
        )
        for index, (cluster, count, offset) in enumerate(
            zip(
                clusters,
                LAYER17_NATIVE_MODE_COUNTS,
                offsets,
                strict=True,
            )
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


def _family_rows() -> dict[str, AlignedFragmentRows]:
    generator = torch.Generator().manual_seed(20260806)
    teachers = {
        fragment_id: torch.randn(
            20,
            64,
            generator=generator,
            dtype=torch.float64,
        )
        for fragment_id in LAYER17_FRAGMENT_IDS
    }
    result: dict[str, AlignedFragmentRows] = {}
    for family_index, alias in enumerate(V8_FAMILY_LOFO_FAMILY_ALIASES):
        row_keys = tuple(
            (f"opaque-{alias}-{row_index:02d}", 0)
            for row_index in range(32)
        )
        inputs = torch.randn(
            32,
            20,
            generator=generator,
            dtype=torch.float64,
        ) + family_index * 0.05
        rows = {
            fragment_id: LayerFragmentRows(
                inputs=inputs + fragment_index * 0.01,
                contributions=(
                    (inputs + fragment_index * 0.01)
                    @ teachers[fragment_id]
                    + 0.001
                    * torch.randn(
                        32,
                        64,
                        generator=generator,
                        dtype=torch.float64,
                    )
                ),
                fisher_weights=(
                    torch.linspace(0.25, 2.0, 32, dtype=torch.float64)
                    * (family_index + 1)
                    * (fragment_index + 1)
                ),
                sequences=32,
            )
            for fragment_index, fragment_id in enumerate(
                LAYER17_FRAGMENT_IDS
            )
        }
        result[alias] = AlignedFragmentRows(
            rows_by_fragment=rows,
            row_keys=row_keys,
        )
    return result


def _numeric_fit_tensors(pilot: object) -> tuple[torch.Tensor, ...]:
    computational_modes = getattr(pilot, "computational_modes")
    modal_generators = getattr(pilot, "modal_generators")
    basis = computational_modes.selected_basis
    generator = modal_generators.selected_plan
    assert basis is not None
    assert generator is not None
    assert generator.factors.bias is not None
    return (
        basis.mean_bias,
        basis.encoder_basis,
        generator.factors.input_factor,
        generator.factors.output_factor,
        generator.factors.bias,
    )


def _runtime_catalog_and_fragment_plan() -> tuple[
    NaturalMLPParameterGroupCatalog,
    ParameterClusterLayerFragmentPlan,
]:
    input_width = 640
    output_width = 640
    input_site = "model.layers.17.mlp.input"
    catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint=MODEL_SHA,
        layer_specs=(
            NaturalMLPLayerParameterSpec(
                layer_id="model.layers.17",
                layer_ordinal=17,
                activation_site="model.layers.17.mlp.down_input",
                input_site=input_site,
                output_site="model.layers.17.mlp.residual_delta",
                intermediate_width=sum(LAYER17_NATIVE_MODE_COUNTS),
                input_width=input_width,
                output_width=output_width,
                gate_proj_path="model.layers.17.mlp.gate.weight",
                up_proj_path="model.layers.17.mlp.up.weight",
                down_proj_path="model.layers.17.mlp.down.weight",
            ),
        ),
    )
    input_catalog_sha256 = natural_mlp_input_catalog_sha256(
        source_model_sha256=MODEL_SHA,
        input_site=input_site,
        input_width=input_width,
    )
    clusters = (0, 28, 34, 54)
    offsets = (0, 54, 92, 177)
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=cluster,
            layer_ordinal=17,
            layer_id="model.layers.17",
            activation_site="model.layers.17.mlp.down_input",
            input_site=input_site,
            output_site="model.layers.17.mlp.residual_delta",
            input_catalog_sha256=input_catalog_sha256,
            input_width=input_width,
            output_width=output_width,
            group_indices=tuple(range(offset, offset + count)),
            channel_indices=tuple(range(offset, offset + count)),
            fisher_ranks=tuple(range(offset, offset + count)),
            axial_orientations=(1,) * count,
            native_parameter_count=count * (2 * input_width + output_width),
            fisher_mass=float(4 - index),
            source_cluster_plan_sha256=CLUSTER_SHA,
            source_fisher_coupling_sha256=FISHER_SHA,
            parameter_catalog_sha256=catalog.artifact_sha256,
            source_model_sha256=MODEL_SHA,
        )
        for index, (cluster, count, offset) in enumerate(
            zip(
                clusters,
                LAYER17_NATIVE_MODE_COUNTS,
                offsets,
                strict=True,
            )
        )
    )
    return catalog, ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=CLUSTER_SHA,
        source_fisher_coupling_sha256=FISHER_SHA,
        parameter_catalog_sha256=catalog.artifact_sha256,
        source_model_sha256=MODEL_SHA,
        cluster_count=55,
        source_group_count=catalog.group_count,
        assigned_group_count=catalog.group_count,
        assigned_native_parameter_count=sum(
            fragment.native_parameter_count for fragment in fragments
        ),
        fragments=fragments,
    )


def _runtime_lowering(
    plan: ParameterClusterLayerFragmentPlan,
    fragment: ParameterClusterLayerFragment,
    *,
    mode_rank: int,
) -> ModalGeneratorLowering:
    basis = ComputationalModeBasis(
        binding=ComputationalModeBinding.create(
            mode_set_id=fragment.fragment_id,
            source_kind="layer_fragment",
            output_site=fragment.output_site,
            source_model_sha256=MODEL_SHA,
            parameter_catalog_sha256=fragment.parameter_catalog_sha256,
            fisher_coupling_sha256=FISHER_SHA,
            parameter_cluster_sha256=fragment.artifact_sha256,
            fit_split_sha256=RUNTIME_FIT_SHA,
            eval_split_sha256=RUNTIME_DIAGNOSTIC_SHA,
        ),
        config=ComputationalModeConfig(
            ranks=(mode_rank,),
            selection_rule="fixed_rank",
            selected_rank=mode_rank,
        ),
        rank=mode_rank,
        mean_bias=torch.linspace(
            -0.5,
            0.5,
            fragment.output_width,
            dtype=torch.float64,
        ),
        encoder_basis=torch.cat(
            (
                torch.eye(mode_rank, dtype=torch.float64),
                torch.zeros(
                    fragment.output_width - mode_rank,
                    mode_rank,
                    dtype=torch.float64,
                ),
            ),
            dim=0,
        ),
    )
    generator_rank = 16
    input_factor = torch.zeros(
        fragment.input_width,
        generator_rank,
        dtype=torch.float64,
    )
    input_factor[:generator_rank] = torch.eye(
        generator_rank,
        dtype=torch.float64,
    )
    output_factor = torch.zeros(
        generator_rank,
        mode_rank,
        dtype=torch.float64,
    )
    output_factor[:, :generator_rank] = torch.eye(
        generator_rank,
        dtype=torch.float64,
    )
    factors = ModalGeneratorFactors(
        rank=generator_rank,
        input_factor=input_factor,
        output_factor=output_factor,
        bias=torch.linspace(-0.25, 0.25, mode_rank, dtype=torch.float64),
    )
    parameter_count = (
        fragment.input_width * generator_rank
        + generator_rank * mode_rank
        + mode_rank
    )
    coordinate_plan = ModalGeneratorPlan(
        binding=ModalGeneratorBinding.create(
            generator_id=(
                f"layer.17.cluster.{fragment.cluster_id}.coordinates"
            ),
            input_kind="native_layer_input",
            input_site=fragment.input_site,
            output_site=fragment.output_site,
            source_model_sha256=MODEL_SHA,
            input_catalog_sha256=fragment.input_catalog_sha256,
            output_catalog_sha256=basis.artifact_sha256,
            cluster_plan_sha256=plan.artifact_sha256,
            fit_split_sha256=RUNTIME_FIT_SHA,
            eval_split_sha256=RUNTIME_DIAGNOSTIC_SHA,
            target_kind="computational_mode_coordinates",
            fisher_coupling_sha256=FISHER_SHA,
            computational_mode_basis_sha256=basis.artifact_sha256,
            parameter_cluster_fragment_sha256=fragment.artifact_sha256,
        ),
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


def _artifact_reference(
    *,
    kind: str,
    digest: str,
    metadata: dict[str, object],
) -> AuthenticatedArtifactReference:
    return AuthenticatedArtifactReference(
        referenced_artifact_kind=kind,
        referenced_artifact_sha256=digest,
        metadata_json=json.dumps(
            {"artifact_kind": kind, "artifact_sha256": digest, **metadata},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
    )


def _runtime_authority_metadata() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    protocol = build_default_v8_layer17_family_lofo_protocol()
    public_binding = lofo_authority._public_protocol_binding(protocol)
    public_protocol = {
        key: public_binding[key]
        for key in (
            "protocol_artifact_sha256",
            "fit_membership_sha256",
            "family_alias_mapping_sha256",
            "fold_count",
            "folds",
        )
    }
    corpus_authority = protocol["corpus_authority"]
    fit_role = protocol["role_bindings"]["fit"]
    authority_payload: dict[str, object] = {
        "schema": lofo_authority.GEMMA3_LAYER17_FAMILY_LOFO_AUTHORITY_SCHEMA,
        "format_version": 1,
        "scientific_role": "open_development_calibration_a_fit_family_lofo",
        "heldout_confirmation": False,
        "receipt": {
            "receipt_sha256": "6" * 64,
            "receipt_file_sha256": "7" * 64,
        },
        "protocol": public_protocol,
        "corpus": {
            "corpus_artifact_sha256": corpus_authority[
                "artifact_sha256"
            ],
            "corpus_artifact_file_sha256": "8" * 64,
            "tokenizer_contract_sha256": corpus_authority[
                "tokenizer_contract_sha256"
            ],
            "fit_manifest_sha256": fit_role["manifest_sha256"],
            "fit_role_file_sha256": fit_role["source_file_sha256"],
            "example_count": 256,
            "block_count": 8,
            "examples_per_block": 32,
            "block_labels": tuple(V8_FAMILY_LOFO_FAMILY_ALIASES),
        },
        "access": dict(lofo_authority._AUTHORITY_ACCESS),
        "safety": dict(lofo_authority._PUBLIC_SAFETY),
    }
    authority = {
        **authority_payload,
        "authority_sha256": lofo_authority._domain_sha256(
            lofo_authority._AUTHORITY_DOMAIN,
            authority_payload,
        ),
    }
    blocks = {
        alias: {
            "example_count": 32,
            "batch_count": 1,
            "logical_valid_tokens": 128,
            "supervised_tokens": 96,
        }
        for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
    }
    materialization_payload: dict[str, object] = {
        "schema": (
            lofo_authority.GEMMA3_LAYER17_FAMILY_LOFO_MATERIALIZATION_SCHEMA
        ),
        "format_version": 1,
        "scientific_role": "open_development_calibration_a_fit_family_lofo",
        "heldout_confirmation": False,
        "authority_sha256": authority["authority_sha256"],
        "tokenization": {
            "block_count": 8,
            "block_labels": tuple(V8_FAMILY_LOFO_FAMILY_ALIASES),
            "example_count": 256,
            "examples_per_block": 32,
            "batch_count": 8,
            "logical_valid_tokens": 1024,
            "supervised_tokens": 768,
            "max_length": 8,
            "tokenization_batch_size": 32,
            "device": "cpu",
            "stream_catalog_sha256": "9" * 64,
            "blocks": blocks,
        },
        "access": dict(lofo_authority._MATERIALIZATION_ACCESS),
        "safety": dict(lofo_authority._PUBLIC_SAFETY),
    }
    materialization = {
        **materialization_payload,
        "materialization_sha256": lofo_authority._domain_sha256(
            lofo_authority._MATERIALIZATION_DOMAIN,
            materialization_payload,
        ),
    }
    lofo_authority.validate_gemma3_layer17_family_lofo_authority_metadata(
        authority
    )
    lofo_authority.validate_gemma3_layer17_family_lofo_materialization_metadata(
        materialization
    )
    return authority, materialization, protocol


def _runtime_fit_receipt() -> dict[str, object]:
    payload: dict[str, object] = {
        "fit_split_sha256": RUNTIME_FIT_SHA,
        "diagnostic_split_sha256": RUNTIME_DIAGNOSTIC_SHA,
        "family_aliases": tuple(V8_FAMILY_LOFO_FAMILY_ALIASES),
        "family_count": 8,
        "fit_example_count": 256,
        "fit_observations": 1024,
        "fit_sequences": 256,
        "fisher_normalization": refit._FISHER_NORMALIZATION,
        "fit_fisher_total_mass": 1.0,
        "fit_fisher_total_mass_per_family": 0.125,
        "all_coefficients_fit_on_all_normalized_rows": True,
        "diagnostic_subset_within_fit": True,
        "diagnostic_subset_balanced_by_family": True,
        "diagnostic_observations_per_family": 128,
        "diagnostic_observations": 1024,
        "diagnostic_used_for_fixed_rank_curve_construction": True,
        "diagnostic_used_for_selection": False,
        "diagnostic_used_for_early_stopping": False,
        "diagnostic_supports_assessment_claim": False,
    }
    return {
        **payload,
        "fit_receipt_sha256": refit._sha256(
            payload,
            domain=refit._FIT_RECEIPT_DOMAIN,
        ),
    }


def _synthetic_full_refit_build_arguments(
    tmp_path: Path,
) -> dict[str, object]:
    catalog, plan = _runtime_catalog_and_fragment_plan()
    selection = select_top_fisher_same_layer_fragments(
        plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    lowerings_by_fragment = {
        fragment.fragment_id: _runtime_lowering(
            plan,
            fragment,
            mode_rank=mode_rank,
        )
        for fragment, mode_rank in zip(
            selection.execution_order,
            refit.resolve_layer17_node_ranks(48),
            strict=True,
        )
    }
    graph_bundle = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=plan,
        lowerings_by_fragment=lowerings_by_fragment,
    )
    graph = graph_bundle.graph_plan
    lowerings = graph_bundle.lowerings_by_node
    authority, materialization, protocol = _runtime_authority_metadata()
    fit_receipt = _runtime_fit_receipt()
    fit_collection = {
        "materialization": materialization,
        "capture_count": 1,
        "captured_examples": 256,
        "captured_observations": 1024,
        "captured_sequences": 256,
        "captured_row_key_sha256": "a" * 64,
        "family_observations": {
            alias: 128 for alias in V8_FAMILY_LOFO_FAMILY_ALIASES
        },
        "model_rows_recollected": False,
    }
    fisher_reference = _artifact_reference(
        kind="fisher_graph.grouped_virtual_gate_empirical_fisher",
        digest=FISHER_SHA,
        metadata={
            "source_prompt_trace_sha256": "b" * 64,
            "source_trace_authenticated": True,
            "catalog_artifact_sha256": catalog.artifact_sha256,
            "calibration_split_sha256": TOPOLOGY_FIT_SHA,
            "objective_sha256": OBJECTIVE_SHA,
        },
    )
    cluster_reference = _artifact_reference(
        kind="fisher_graph.fisher_prompt_cluster_plan",
        digest=CLUSTER_SHA,
        metadata={
            "config": {
                "model_fingerprint": MODEL_SHA,
                "source_fisher_coupling_sha256": FISHER_SHA,
                "calibration_split_sha256": TOPOLOGY_FIT_SHA,
                "objective_sha256": OBJECTIVE_SHA,
            }
        },
    )
    authority_corpus = authority["corpus"]
    assert isinstance(authority_corpus, dict)
    refit_authority = ModalRefitFisherAuthority(
        fit_split_sha256=RUNTIME_FIT_SHA,
        eval_split_sha256=RUNTIME_DIAGNOSTIC_SHA,
        eval_role="balanced_within_fit_fixed_rank_diagnostic",
        fisher_normalization=refit._FISHER_NORMALIZATION,
        source_model_sha256=MODEL_SHA,
        parameter_catalog_sha256=catalog.artifact_sha256,
        topology_grouped_fisher_sha256=FISHER_SHA,
        topology_fisher_calibration_split_sha256=TOPOLOGY_FIT_SHA,
        topology_fisher_cluster_plan_sha256=CLUSTER_SHA,
        topology_fragment_plan_sha256=plan.artifact_sha256,
        authorizing_report_sha256=LOFO_SHA,
        refit_protocol_sha256=FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256,
        fit_authority_sha256=str(authority["authority_sha256"]),
        fit_receipt_sha256=str(fit_receipt["fit_receipt_sha256"]),
        fit_corpus_artifact_sha256=str(
            authority_corpus["corpus_artifact_sha256"]
        ),
        fit_manifest_sha256=str(authority_corpus["fit_manifest_sha256"]),
        fit_materialization_sha256=str(
            materialization["materialization_sha256"]
        ),
        fit_example_count=256,
        fit_family_count=8,
        equal_family_weighting=True,
        eval_subset_within_fit=True,
        eval_used_for_selection=False,
    )
    nodes = tuple(
        ModalCompilerNodeArtifact(
            node_name=node.name,
            lowering=lowerings[node.name],
            graph_node_artifact_sha256=node.artifact_sha256,
        )
        for node in graph.nodes
    )
    pipeline = ModalCompilerPipeline(
        parameter_catalog=catalog,
        grouped_fisher=fisher_reference,
        fisher_clusters=cluster_reference,
        parameter_cluster_fragments=plan,
        nodes=nodes,
        graph_plan=graph,
        modal_refit_fisher_authority=refit_authority,
        source_replacement_accounting=(
            build_modal_source_replacement_accounting(
                catalog,
                plan,
                LAYER17_FRAGMENT_IDS,
            )
        ),
    )
    base_sha256 = "c" * 64
    lofo_source = tmp_path / "passing-lofo.json"
    lofo_source.write_text("{}\n", encoding="utf-8")
    lofo_report = {
        "decision": {
            "all_required_gates_pass": True,
            "next_action": (
                "freeze_full_eight_family_refit_then_replay_eligible_open_"
                "development_assessment"
            ),
        },
        "experiment": {
            "adapter_model_fingerprint": MODEL_SHA,
            "requested_revision": "d" * 40,
        },
        "protocol": {
            "artifact_sha256": (
                FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            )
        },
        "authority": authority,
        "fit_collection": {
            "materialization_sha256": materialization[
                "materialization_sha256"
            ],
            "captured_row_key_sha256": fit_collection[
                "captured_row_key_sha256"
            ],
        },
        "lineage": {
            "base_artifact_file_sha256": base_sha256,
            "fragment_plan_sha256": plan.artifact_sha256,
            "fragment_selection_sha256": selection.artifact_sha256,
            "frozen_v9_candidate_file": "baseline.pt",
            "frozen_v9_candidate_file_sha256": "e" * 64,
            "frozen_v9_candidate_scientific_sha256": "f" * 64,
        },
        "heldout_confirmation": False,
        "serving_authorized": False,
        "compression_claim": False,
        "report_sha256": LOFO_SHA,
    }
    lofo_lineage = refit._passing_lofo_lineage(
        lofo_report,
        report_path=lofo_source,
    )
    return {
        "experiment": {
            "experiment_kind": (
                "gemma3_layer17_v8_fit_all_family_refit_v1"
            ),
            "scientific_role": (
                "calibration_a_fit_all_family_refit_candidate"
            ),
            "model_id": refit.DEFAULT_MODEL_ID,
            "requested_revision": "d" * 40,
            "adapter_model_fingerprint": MODEL_SHA,
            "source_model_unchanged": True,
            "heldout_confirmation": False,
            "assessment_metrics_present": False,
            "serving_authorized": False,
            "full_eight_family_refit_completed": True,
            "fit_family_count": 8,
            "fit_example_count": 256,
            "selection_opened": False,
            "lofo_report_sha256": LOFO_SHA,
            "lofo_protocol_sha256": (
                FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            ),
            "lofo_authority_sha256": authority["authority_sha256"],
            "frozen_v9_candidate_file_sha256": "e" * 64,
            "frozen_v9_candidate_scientific_sha256": "f" * 64,
        },
        "authority": authority,
        "protocol": protocol,
        "fit_collection": fit_collection,
        "fit_receipt": fit_receipt,
        "lofo_lineage": lofo_lineage,
        "selection": selection,
        "lowerings_by_node": lowerings,
        "edgeless_graph": graph,
        "compiler_pipeline": pipeline,
        "base_artifact_file": "base.pt",
        "base_artifact_file_sha256": base_sha256,
    }


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def test_all_family_rows_are_exhaustive_equal_mass_and_diagnostic_only():
    family_rows = _family_rows()
    fit, diagnostic, receipt = refit.build_layer17_all_family_refit_rows(
        family_rows,
        authority_sha256=AUTHORITY_SHA,
        materialization_sha256=MATERIALIZATION_SHA,
        lofo_report_sha256=LOFO_SHA,
        diagnostic_observations_per_family=3,
    )

    assert fit.observations == 256
    assert fit.sequences == 256
    assert diagnostic.observations == 24
    assert diagnostic.sequences == 24
    assert set(diagnostic.row_keys).issubset(fit.row_keys)
    assert receipt["all_coefficients_fit_on_all_normalized_rows"] is True
    assert receipt["diagnostic_subset_within_fit"] is True
    assert receipt["diagnostic_used_for_selection"] is False
    assert receipt["diagnostic_used_for_early_stopping"] is False
    assert receipt["diagnostic_supports_assessment_claim"] is False
    assert receipt["fit_split_sha256"] != receipt["diagnostic_split_sha256"]

    for fragment in fit.rows_by_fragment.values():
        assert fragment.fisher_weights.sum().item() == pytest.approx(1.0)
        offset = 0
        for alias in V8_FAMILY_LOFO_FAMILY_ALIASES:
            count = family_rows[alias].observations
            family_mass = fragment.fisher_weights[
                offset : offset + count
            ].sum()
            assert family_mass.item() == pytest.approx(1.0 / 8.0)
            offset += count


def test_changing_only_diagnostic_rows_cannot_change_fixed_rank_fit_tensors():
    family_rows = _family_rows()
    fit_a, diagnostic_a, receipt_a = (
        refit.build_layer17_all_family_refit_rows(
            family_rows,
            authority_sha256=AUTHORITY_SHA,
            materialization_sha256=MATERIALIZATION_SHA,
            lofo_report_sha256=LOFO_SHA,
            diagnostic_observations_per_family=3,
        )
    )
    fit_b, diagnostic_b, receipt_b = (
        refit.build_layer17_all_family_refit_rows(
            family_rows,
            authority_sha256=AUTHORITY_SHA,
            materialization_sha256=MATERIALIZATION_SHA,
            lofo_report_sha256=LOFO_SHA,
            diagnostic_observations_per_family=4,
        )
    )
    assert receipt_a["fit_split_sha256"] == receipt_b["fit_split_sha256"]
    assert receipt_a["diagnostic_split_sha256"] != receipt_b[
        "diagnostic_split_sha256"
    ]
    for fragment_id in LAYER17_FRAGMENT_IDS:
        torch.testing.assert_close(
            fit_a.rows_by_fragment[fragment_id].inputs,
            fit_b.rows_by_fragment[fragment_id].inputs,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            fit_a.rows_by_fragment[fragment_id].fisher_weights,
            fit_b.rows_by_fragment[fragment_id].fisher_weights,
            rtol=0.0,
            atol=0.0,
        )

    plan = _fragment_plan()
    selection = select_top_fisher_same_layer_fragments(
        plan,
        count=4,
        minimum_fragment_modes=32,
        layer_ordinal=17,
    )
    common = {
        "selection": selection,
        "source_model_sha256": MODEL_SHA,
        "parameter_catalog_sha256": CATALOG_SHA,
        "fisher_coupling_sha256": FISHER_SHA,
        "fragment_plan": plan,
        "mode_rank_cap": 48,
        "generator_rank": 16,
        "ridge": 0.0,
    }
    pilots_a = fit_layer17_capped_node_pilots(
        fit_a,
        diagnostic_a,
        fit_split_sha256=str(receipt_a["fit_split_sha256"]),
        selection_split_sha256=str(
            receipt_a["diagnostic_split_sha256"]
        ),
        **common,
    )
    pilots_b = fit_layer17_capped_node_pilots(
        fit_b,
        diagnostic_b,
        fit_split_sha256=str(receipt_b["fit_split_sha256"]),
        selection_split_sha256=str(
            receipt_b["diagnostic_split_sha256"]
        ),
        **common,
    )
    for fragment_id in LAYER17_FRAGMENT_IDS:
        left = pilots_a[fragment_id]
        right = pilots_b[fragment_id]
        assert (
            left.computational_modes.evaluation_used_for_basis_fit is False
        )
        assert (
            left.computational_modes.evaluation_used_for_rank_selection
            is False
        )
        assert left.modal_generators.config.selection_rule == "fixed_rank"
        assert left.modal_generators.config.ranks == (16,)
        for left_tensor, right_tensor in zip(
            _numeric_fit_tensors(left),
            _numeric_fit_tensors(right),
            strict=True,
        ):
            torch.testing.assert_close(
                left_tensor,
                right_tensor,
                rtol=0.0,
                atol=0.0,
            )


def _minimal_passing_lofo_report() -> dict[str, object]:
    return {
        "decision": {
            "all_required_gates_pass": True,
            "next_action": (
                "freeze_full_eight_family_refit_then_replay_eligible_open_"
                "development_assessment"
            ),
        },
        "experiment": {
            "adapter_model_fingerprint": MODEL_SHA,
            "requested_revision": "2" * 40,
        },
        "protocol": {
            "artifact_sha256": (
                FROZEN_V8_LAYER17_FAMILY_LOFO_PROTOCOL_SHA256
            )
        },
        "authority": {
            "authority_sha256": AUTHORITY_SHA,
            "receipt": {"receipt_sha256": "3" * 64},
            "corpus": {
                "corpus_artifact_sha256": "4" * 64,
                "fit_manifest_sha256": "5" * 64,
            },
        },
        "fit_collection": {
            "materialization_sha256": MATERIALIZATION_SHA,
            "captured_row_key_sha256": "6" * 64,
        },
        "lineage": {
            "base_artifact_file_sha256": "7" * 64,
            "fragment_plan_sha256": "8" * 64,
            "fragment_selection_sha256": "9" * 64,
            "frozen_v9_candidate_file": "baseline.pt",
            "frozen_v9_candidate_file_sha256": "a" * 64,
            "frozen_v9_candidate_scientific_sha256": "b" * 64,
        },
        "heldout_confirmation": False,
        "serving_authorized": False,
        "compression_claim": False,
        "report_sha256": LOFO_SHA,
    }


def test_passing_lofo_binding_fails_closed_before_refit(tmp_path: Path):
    source = tmp_path / "passing-lofo.json"
    source.write_bytes(b"source-safe-placeholder")
    report = _minimal_passing_lofo_report()
    lineage = refit._passing_lofo_lineage(report, report_path=source)
    assert lineage["lofo_all_required_gates_pass"] is True
    assert lineage["lofo_report_sha256"] == LOFO_SHA
    assert lineage["frozen_v9_candidate_scientific_sha256"] == "b" * 64

    failed = copy.deepcopy(report)
    failed["decision"]["all_required_gates_pass"] = False
    with pytest.raises(ValueError, match="does not authorize"):
        refit._passing_lofo_lineage(failed, report_path=source)


def test_tensor_bearing_full_candidate_save_load_and_report_roundtrip(
    tmp_path: Path,
) -> None:
    output = tmp_path / ".local-runs" / "synthetic-full-refit.pt"
    report = refit.save_gemma3_layer17_v8_all_family_refit_candidate(
        output,
        **_synthetic_full_refit_build_arguments(tmp_path),
    )
    candidate = refit.load_gemma3_layer17_v8_all_family_refit_candidate(
        output
    )
    graph, lowerings, pipeline = (
        refit.restore_gemma3_layer17_v8_all_family_refit_runtime(candidate)
    )

    assert _contains_tensor(candidate) is True
    assert _contains_tensor(refit._scientific_projection(candidate)) is False
    assert any(
        isinstance(
            lowering.coordinate_generator_plan.factors.input_factor,
            torch.Tensor,
        )
        for lowering in lowerings.values()
    )
    assert graph.artifact_sha256 == pipeline.graph_plan.artifact_sha256

    report_path = output.with_suffix(".json")
    decoded = json.loads(report_path.read_text(encoding="utf-8"))
    assert decoded == report
    assert _contains_tensor(decoded) is False
    assert all(
        field not in decoded
        for field in (
            "lowering_records",
            "edgeless_graph",
            "compiler_pipeline",
        )
    )
    assert decoded["safety"] == {
        **refit._SAFETY,
        "contains_executable_generator_weights": False,
        "contains_tensors": False,
    }
    assert decoded["artifact"] == {
        "tensor_file": output.name,
        "scientific_payload_sha256": candidate[
            "scientific_payload_sha256"
        ],
    }
    assert refit._json_clone(report) == report


def test_all_family_refit_authority_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    arguments = _synthetic_full_refit_build_arguments(tmp_path)
    fit_receipt = copy.deepcopy(arguments["fit_receipt"])
    fit_receipt["fit_receipt_sha256"] = "0" * 64

    with pytest.raises(
        ValueError,
        match="refit-Fisher authority provenance drifted",
    ):
        refit._validate_full_refit_pipeline_authority(
            arguments["compiler_pipeline"],
            authority=arguments["authority"],
            fit_collection=arguments["fit_collection"],
            fit_receipt=fit_receipt,
            lofo_lineage=arguments["lofo_lineage"],
        )


def test_fixed_numeric_metadata_rejects_bool_and_string_coercions():
    with pytest.raises(ValueError, match="finite JSON number"):
        refit._finite_number(False, label="ridge")
    with pytest.raises(ValueError, match="finite JSON number"):
        refit._finite_number("0.0", label="ridge")


def test_cli_and_runner_have_only_fit_authority_and_lofo_inputs():
    assert refit.DEFAULT_LAYER17_V8_ALL_FAMILY_REFIT_OUTPUT.name == (
        "layer17-capped-node-c48-r16-edgeless-a-fit-v8-full-refit-dev-v1.pt"
    )
    parser = refit.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--lofo-report",
        "--corpus-receipt",
        "--corpus-artifact",
        "--fit-input",
    }.issubset(options)
    forbidden = ("selection", "guard", "calibration-b", "validation", "test")
    assert all(
        token not in option
        for option in options
        for token in forbidden
    )
    parameters = inspect.signature(
        refit.run_gemma3_layer17_v8_all_family_refit
    ).parameters
    assert all(
        token not in name.replace("_", "-")
        for name in parameters
        for token in forbidden
    )
