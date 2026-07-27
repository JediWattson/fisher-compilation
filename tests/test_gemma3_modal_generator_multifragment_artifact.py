from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_modal_generator_multifragment_artifact as artifact
from fisher_graph.computational_modes import (
    ComputationalModeBinding,
    fit_computational_mode_rate_curve,
)
from fisher_graph.gemma3_modal_generator_multifragment_artifact import (
    GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION,
    GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA,
    Gemma3ModalGeneratorMultifragmentNodeRecord,
    build_gemma3_modal_generator_multifragment_evaluation,
    build_gemma3_modal_generator_multifragment_evaluation_from_rung,
    build_gemma3_modal_generator_multifragment_model_metadata,
    build_gemma3_modal_generator_multifragment_payload,
    build_gemma3_modal_generator_multifragment_protocol,
    build_gemma3_modal_generator_multifragment_scientific_status,
    build_gemma3_modal_generator_multifragment_splits,
    build_gemma3_modal_generator_multifragment_upstream_metadata,
    load_gemma3_modal_generator_multifragment_artifact,
    save_gemma3_modal_generator_multifragment_artifact,
)
from fisher_graph.modal_compiler_pipeline import (
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from fisher_graph.modal_generator_graph import ModalGeneratorGraphPlan
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
from test_modal_compiler_pipeline import (
    DTYPE,
    EVAL_HASH,
    FIT_HASH,
    MODEL_HASH,
    _analysis_chain,
)


ASSESSMENT_HASH = "e" * 64
UPSTREAM_EVAL_HASH = "f" * 64


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


SOURCE_EVALUATION_EXPORT_HASH = _sha("source-evaluation-export")
RAW_PARTITION_PLAN_HASH = _sha("raw-partition-plan")
SELECTION_PARTITION_HASH = _sha("selection-partition")
ASSESSMENT_PARTITION_HASH = _sha("assessment-partition")


def _curve_and_lowering(
    fragment,
    fragment_plan,
    fisher_sha256: str,
    *,
    node_index: int,
):
    generator = torch.Generator().manual_seed(80_000 + node_index)
    fit_coordinate = torch.randn(12, 1, generator=generator, dtype=DTYPE)
    eval_coordinate = torch.randn(8, 1, generator=generator, dtype=DTYPE)
    decoder = torch.zeros(1, fragment.output_width, dtype=DTYPE)
    decoder[0, node_index] = 1.0
    mode_binding = ComputationalModeBinding.create(
        mode_set_id=fragment.fragment_id,
        source_kind="layer_fragment",
        output_site=fragment.output_site,
        source_model_sha256=MODEL_HASH,
        parameter_catalog_sha256=fragment.parameter_catalog_sha256,
        fisher_coupling_sha256=fisher_sha256,
        parameter_cluster_sha256=fragment.artifact_sha256,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
    )
    modes = fit_computational_mode_rate_curve(
        fit_coordinate @ decoder,
        torch.ones(12, dtype=DTYPE),
        eval_coordinate @ decoder,
        torch.ones(8, dtype=DTYPE),
        (1,),
        binding=mode_binding,
        selection_rule="fixed_rank",
        selected_rank=1,
    )
    basis = modes.selected_basis
    assert basis is not None

    fit_inputs = torch.randn(
        12,
        fragment.input_width,
        generator=generator,
        dtype=DTYPE,
    )
    eval_inputs = torch.randn(
        8,
        fragment.input_width,
        generator=generator,
        dtype=DTYPE,
    )
    coefficient = torch.zeros(fragment.input_width, 1, dtype=DTYPE)
    coefficient[node_index, 0] = 1.25 + node_index
    generator_binding = ModalGeneratorBinding.create(
        generator_id=f"artifact.generator.{node_index}",
        input_kind="native_layer_input",
        input_site=fragment.input_site,
        output_site=fragment.output_site,
        source_model_sha256=MODEL_HASH,
        input_catalog_sha256=fragment.input_catalog_sha256,
        output_catalog_sha256=basis.artifact_sha256,
        cluster_plan_sha256=fragment_plan.artifact_sha256,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
        target_kind="computational_mode_coordinates",
        fisher_coupling_sha256=fisher_sha256,
        computational_mode_basis_sha256=basis.artifact_sha256,
        parameter_cluster_fragment_sha256=fragment.artifact_sha256,
    )
    generators = fit_modal_generator_rate_curve(
        fit_inputs,
        fit_inputs @ coefficient,
        torch.ones(12, dtype=DTYPE),
        eval_inputs,
        eval_inputs @ coefficient,
        (1,),
        binding=generator_binding,
        fisher_weights_eval=torch.ones(8, dtype=DTYPE),
        fit_intercept=False,
        selection_rule="fixed_rank",
        selected_rank=1,
    )
    selected_generator = generators.selected_plan
    assert selected_generator is not None
    lowering = lower_coordinate_modal_generator(
        selected_generator,
        basis,
        fragment_plan,
    )
    return modes, generators, lowering


def _arguments() -> dict[str, object]:
    trace, catalog, fisher, clusters, fragments = _analysis_chain()
    curves = tuple(
        _curve_and_lowering(
            fragment,
            fragments,
            fisher.artifact_sha256,
            node_index=index,
        )
        for index, fragment in enumerate(fragments.fragments)
    )
    names = tuple(f"node.{index}" for index in range(len(curves)))
    records = tuple(
        Gemma3ModalGeneratorMultifragmentNodeRecord(
            node_name=name,
            computational_modes=modes,
            modal_generators=generators,
            lowering=lowering,
        )
        for name, (modes, generators, lowering) in zip(names, curves)
    )
    nodes = tuple(
        record.lowering.to_graph_node(
            name=record.node_name,
            causal_order=index,
        )
        for index, record in enumerate(records)
    )
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
            names[0]: source_fit,
            names[1]: torch.zeros_like(source_fit),
        },
        {
            names[0]: source_eval,
            names[1]: torch.zeros_like(source_eval),
        },
        {names[1]: 1.5 * source_fit},
        {names[1]: 1.5 * source_eval},
        node_causal_orders={names[0]: 0, names[1]: 1},
        generator_artifact_sha256s={
            record.node_name: (
                record.lowering.coordinate_generator_plan.artifact_sha256
            )
            for record in records
        },
        source_model_sha256=MODEL_HASH,
        parameter_cluster_plan_sha256=fragments.artifact_sha256,
        fit_split_sha256=FIT_HASH,
        eval_split_sha256=EVAL_HASH,
        candidate_edges=((names[0], names[1]),),
        fit_intercept=False,
        minimum_heldout_improvement=1e-6,
    )
    assert len(selection.interactions) == 1
    graph = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_HASH,
        parameter_cluster_plan_sha256=fragments.artifact_sha256,
        nodes=nodes,
        interactions=selection.interactions,
    )
    edgeless = ModalGeneratorGraphPlan(
        model_fingerprint=MODEL_HASH,
        parameter_cluster_plan_sha256=fragments.artifact_sha256,
        nodes=nodes,
        interactions=(),
    )
    accounting = build_modal_source_replacement_accounting(
        catalog,
        fragments,
        tuple(fragment.fragment_id for fragment in fragments.fragments),
    )
    pipeline = build_modal_compiler_pipeline(
        source_prompt_trace=trace,
        parameter_catalog=catalog,
        grouped_fisher=fisher,
        fisher_clusters=clusters,
        parameter_cluster_fragments=fragments,
        lowerings_by_node={
            record.node_name: record.lowering for record in records
        },
        graph_plan=graph,
        interaction_selection=selection,
        source_replacement_accounting=accounting,
    )
    model = build_gemma3_modal_generator_multifragment_model_metadata(
        model_id="google/gemma-3-270m",
        requested_revision="1" * 40,
        resolved_commit="1" * 40,
        adapter_model_fingerprint=MODEL_HASH,
        source_whole_model_learned_parameters=10_000,
    )
    splits = build_gemma3_modal_generator_multifragment_splits(
        fit_split_sha256=FIT_HASH,
        upstream_evaluation_split_sha256=UPSTREAM_EVAL_HASH,
        selection_split_sha256=EVAL_HASH,
        assessment_split_sha256=ASSESSMENT_HASH,
        source_evaluation_export_sha256=SOURCE_EVALUATION_EXPORT_HASH,
        raw_partition_plan_sha256=RAW_PARTITION_PLAN_HASH,
        selection_partition_sha256=SELECTION_PARTITION_HASH,
        assessment_partition_sha256=ASSESSMENT_PARTITION_HASH,
        fit_content_sha256s=(_sha("fit.0"), _sha("fit.1")),
        upstream_evaluation_content_sha256s=(
            _sha("selection.0"),
            _sha("selection.1"),
            _sha("assessment.0"),
            _sha("assessment.1"),
        ),
        selection_content_sha256s=(
            _sha("selection.0"),
            _sha("selection.1"),
        ),
        assessment_content_sha256s=(
            _sha("assessment.0"),
            _sha("assessment.1"),
        ),
    )
    conditions = {
        "native": {"nll_per_token": 2.0},
        "interaction_graph": {
            "nll_per_token": 1.9,
            "delta_nll_per_token": -0.1,
            "native_to_candidate_kl_per_token": 0.01,
            "top1_agreement_to_native": 0.9,
        },
        "edgeless_graph": {
            "nll_per_token": 1.95,
            "delta_nll_per_token": -0.05,
            "native_to_candidate_kl_per_token": 0.02,
            "top1_agreement_to_native": 0.88,
        },
        "matched_deletion": {
            "nll_per_token": 2.2,
            "delta_nll_per_token": 0.2,
            "native_to_candidate_kl_per_token": 0.2,
            "top1_agreement_to_native": 0.7,
        },
        "dense_fused_edgeless": {
            "nll_per_token": 1.95,
            "delta_nll_per_token": -0.05,
            "native_to_candidate_kl_per_token": 0.02,
            "top1_agreement_to_native": 0.88,
        },
    }
    return {
        "scientific_status": (
            build_gemma3_modal_generator_multifragment_scientific_status()
        ),
        "model": model,
        "protocol": build_gemma3_modal_generator_multifragment_protocol(
            compiler_pipeline=pipeline,
            fragment_selection_rule=(
                "highest_fisher_mass_same_cluster_causal_pair"
            ),
            interaction_weighting=(
                "native_reference_target_fragment_fisher"
            ),
        ),
        "splits": splits,
        "upstream_metadata": (
            build_gemma3_modal_generator_multifragment_upstream_metadata(
                source_scientific_payload_sha256=_sha("upstream-v3"),
                source_evaluation_export_sha256=(
                    SOURCE_EVALUATION_EXPORT_HASH
                ),
                fit_prompt_trace=trace,
                parameter_catalog=catalog,
                fisher_coupling=fisher,
                parameter_clusters=clusters,
                parameter_cluster_fragments=fragments,
            )
        ),
        "fit_prompt_trace": trace,
        "parameter_catalog": catalog,
        "fisher_coupling": fisher,
        "parameter_clusters": clusters,
        "parameter_cluster_fragments": fragments,
        "node_records": records,
        "interaction_selection": selection,
        "edgeless_graph": edgeless,
        "compiler_pipeline": pipeline,
        "evaluation": (
            build_gemma3_modal_generator_multifragment_evaluation(
                assessment_split_sha256=ASSESSMENT_HASH,
                supervised_tokens=12,
                logical_valid_tokens=16,
                conditions=conditions,
                edgeless_dense_max_abs_logit_difference=1e-12,
                edgeless_dense_absolute_tolerance=1e-10,
            )
        ),
    }


def _rehash(payload: dict[str, object]) -> None:
    without_digest = {
        key: value
        for key, value in payload.items()
        if key != "scientific_payload_sha256"
    }
    payload["scientific_payload_sha256"] = artifact._payload_sha256(
        without_digest
    )


def _save_raw(path: Path, payload: dict[str, object]) -> None:
    torch.save(payload, path)


def test_strict_roundtrip_report_and_exact_resource_accounting(
    tmp_path: Path,
) -> None:
    arguments = _arguments()
    output = tmp_path / "multifragment-v1.pt"
    report = save_gemma3_modal_generator_multifragment_artifact(
        output,
        **arguments,
    )
    restored = load_gemma3_modal_generator_multifragment_artifact(output)

    assert restored["schema"] == GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_SCHEMA
    assert restored["format_version"] == (
        GEMMA3_MODAL_GENERATOR_MULTIFRAGMENT_FORMAT_VERSION
    )
    assert len(restored["node_records"]) == 2
    assert len(
        restored["compiler_pipeline"]["graph_plan"]["interactions"]
    ) == 1
    assert restored["edgeless_graph"]["interactions"] == []
    assert restored["splits"]["partition"][
        "source_evaluation_export_sha256"
    ] == SOURCE_EVALUATION_EXPORT_HASH
    assert restored["splits"]["partition"][
        "raw_partition_plan_sha256"
    ] == RAW_PARTITION_PLAN_HASH
    assert restored["splits"]["partition"][
        "selection_partition_sha256"
    ] == SELECTION_PARTITION_HASH
    assert restored["splits"]["partition"][
        "assessment_partition_sha256"
    ] == ASSESSMENT_PARTITION_HASH
    assert restored["upstream_metadata"][
        "source_evaluation_export_sha256"
    ] == SOURCE_EVALUATION_EXPORT_HASH
    assert restored["resource_accounting"]["interaction_graph"][
        "replacement_learned_parameters"
    ] == arguments["compiler_pipeline"].graph_parameter_count
    assert restored["resource_accounting"]["matched_deletion"][
        "matrix_macs_per_token"
    ] == 0
    assert report["artifact"]["tensor_file"] == output.name
    assert report["evaluation"]["assessment_split_sha256"] == ASSESSMENT_HASH
    assert output.with_suffix(".json").exists()

    canonical = arguments["evaluation"]
    from_rung = build_gemma3_modal_generator_multifragment_evaluation_from_rung(
        assessment_split_sha256=ASSESSMENT_HASH,
        rung_evaluation={
            "execution_path": "unified_modal_generator_graph_rung",
            "assessment_role": "open_development_assessment",
            "heldout_confirmation": False,
            "supervised_tokens": canonical["supervised_tokens"],
            "logical_valid_tokens": canonical["logical_valid_tokens"],
            "native": canonical["conditions"]["native"],
            "conditions": {
                "interacting_graph": canonical["conditions"][
                    "interaction_graph"
                ],
                "edgeless_graph": canonical["conditions"]["edgeless_graph"],
                "matched_deletion": canonical["conditions"][
                    "matched_deletion"
                ],
                "nodewise_dense_fused": canonical["conditions"][
                    "dense_fused_edgeless"
                ],
            },
            "graph_comparison": {
                "nodewise_dense_supplied": True,
                "nodewise_dense_agrees_with_edgeless": True,
                "nodewise_dense_equivalence_scope": "supervised_logits",
                "nodewise_dense_max_abs_logit_difference": 1e-12,
                "nodewise_dense_equivalence_atol": 1e-10,
                "nodewise_dense_equivalence_rtol": 0.0,
            },
            "resource_accounting": {},
            "latency_or_kernel_speed_claim": False,
        },
    )
    assert from_rung == canonical

    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_modal_generator_multifragment_artifact(
            output,
            **arguments,
        )


def test_loader_rejects_rehashed_free_text_and_shadow_metadata(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_modal_generator_multifragment_payload(
        **_arguments()
    )
    free_text = copy.deepcopy(payload)
    free_text["scientific_status"]["audit_note"] = (
        "Recover the original secret prompt here."
    )
    _rehash(free_text)
    free_text_path = tmp_path / "free-text.pt"
    _save_raw(free_text_path, free_text)
    with pytest.raises(ValueError, match="non-machine string"):
        load_gemma3_modal_generator_multifragment_artifact(free_text_path)

    shadow = copy.deepcopy(payload)
    shadow["upstream_metadata"]["fit_prompt_trace_sha256"] = _sha(
        "shadow-trace"
    )
    _rehash(shadow)
    shadow_path = tmp_path / "shadow.pt"
    _save_raw(shadow_path, shadow)
    with pytest.raises(ValueError, match="upstream metadata"):
        load_gemma3_modal_generator_multifragment_artifact(shadow_path)

    node_shadow = copy.deepcopy(payload)
    node_shadow["node_records"][0]["selected_basis_sha256"] = _sha(
        "shadow-basis"
    )
    _rehash(node_shadow)
    node_path = tmp_path / "node-shadow.pt"
    _save_raw(node_path, node_shadow)
    with pytest.raises(ValueError, match="selected_basis_sha256"):
        load_gemma3_modal_generator_multifragment_artifact(node_path)


def test_loader_rejects_rehashed_edge_accounting_and_split_tamper(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_modal_generator_multifragment_payload(
        **_arguments()
    )
    edge = copy.deepcopy(payload)
    edge["interaction_selection"]["steps"][0]["candidate"]["factors"][
        "message_matrix"
    ][0, 0] += 1.0
    _rehash(edge)
    edge_path = tmp_path / "edge.pt"
    _save_raw(edge_path, edge)
    with pytest.raises(ValueError, match="message_matrix hash mismatch"):
        load_gemma3_modal_generator_multifragment_artifact(edge_path)

    accounting = copy.deepcopy(payload)
    accounting["resource_accounting"]["interaction_graph"][
        "net_stored_parameter_savings"
    ] += 1
    _rehash(accounting)
    accounting_path = tmp_path / "accounting.pt"
    _save_raw(accounting_path, accounting)
    with pytest.raises(ValueError, match="resource accounting"):
        load_gemma3_modal_generator_multifragment_artifact(accounting_path)

    overlap = copy.deepcopy(payload)
    overlap["splits"]["assessment"]["content_sha256"] = overlap["splits"][
        "selection"
    ]["content_sha256"]
    overlap["splits"]["assessment"]["content_count"] = overlap["splits"][
        "selection"
    ]["content_count"]
    _rehash(overlap)
    overlap_path = tmp_path / "overlap.pt"
    _save_raw(overlap_path, overlap)
    with pytest.raises(ValueError, match="selection and assessment"):
        load_gemma3_modal_generator_multifragment_artifact(overlap_path)

    raw_plan = copy.deepcopy(payload)
    raw_plan["splits"]["partition"]["raw_partition_plan_sha256"] = _sha(
        "tampered-raw-partition-plan"
    )
    _rehash(raw_plan)
    raw_plan_path = tmp_path / "raw-partition-plan.pt"
    _save_raw(raw_plan_path, raw_plan)
    with pytest.raises(ValueError, match="split metadata"):
        load_gemma3_modal_generator_multifragment_artifact(raw_plan_path)

    raw_export = copy.deepcopy(payload)
    raw_export["splits"]["partition"][
        "source_evaluation_export_sha256"
    ] = _sha("tampered-source-evaluation-export")
    raw_partition_lineage = {
        key: raw_export["splits"]["partition"][key]
        for key in (
            "source_evaluation_export_sha256",
            "raw_partition_plan_sha256",
            "selection_partition_sha256",
            "assessment_partition_sha256",
        )
    }
    raw_export["splits"]["partition"][
        "raw_partition_binding_sha256"
    ] = artifact._json_sha256(
        raw_partition_lineage,
        domain=artifact._RAW_PARTITION_BINDING_DOMAIN,
    )
    _rehash(raw_export)
    raw_export_path = tmp_path / "raw-evaluation-export.pt"
    _save_raw(raw_export_path, raw_export)
    with pytest.raises(
        ValueError,
        match="raw evaluation export and upstream metadata",
    ):
        load_gemma3_modal_generator_multifragment_artifact(raw_export_path)


def test_loader_rejects_rehashed_overclaim_and_forbidden_source_fields(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_modal_generator_multifragment_payload(
        **_arguments()
    )
    overclaim = copy.deepcopy(payload)
    overclaim["scientific_status"]["compression_claim"] = True
    _rehash(overclaim)
    overclaim_path = tmp_path / "overclaim.pt"
    _save_raw(overclaim_path, overclaim)
    with pytest.raises(ValueError, match="overclaims"):
        load_gemma3_modal_generator_multifragment_artifact(overclaim_path)

    forbidden = copy.deepcopy(payload)
    forbidden["model"]["input_ids"] = (1, 2, 3)
    _rehash(forbidden)
    forbidden_path = tmp_path / "forbidden.pt"
    _save_raw(forbidden_path, forbidden)
    with pytest.raises(ValueError, match="forbidden field"):
        load_gemma3_modal_generator_multifragment_artifact(forbidden_path)
