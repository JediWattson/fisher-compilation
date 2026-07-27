from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_full_mlp_stack_artifact as artifact
from fisher_graph.full_mlp_stack_generators import (
    fit_full_mlp_stack_generators,
)
from fisher_graph.gemma3_full_mlp_stack_artifact import (
    GEMMA3_FULL_MLP_STACK_FORMAT_VERSION,
    GEMMA3_FULL_MLP_STACK_SCHEMA,
    build_gemma3_full_mlp_stack_payload,
    build_gemma3_full_mlp_stack_report,
    load_gemma3_full_mlp_stack_artifact,
    save_gemma3_full_mlp_stack_artifact,
)
from fisher_graph.gemma3_full_mlp_stack_rows import FullMLPStackLayerRows
from fisher_graph.parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from fisher_graph.parameter_layer_superfragments import (
    build_parameter_layer_superfragments,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _analysis() -> tuple[object, tuple[object, ...]]:
    source_model = _sha("model")
    catalog = _sha("catalog")
    fisher = _sha("fisher")
    clusters = _sha("clusters")
    fragments = tuple(
        ParameterClusterLayerFragment(
            cluster_id=0,
            layer_ordinal=ordinal,
            layer_id=f"model.layers.{ordinal}",
            activation_site=f"model.layers.{ordinal}.mlp.gated",
            input_site=f"model.layers.{ordinal}.pre_ffw_norm",
            output_site=f"model.layers.{ordinal}.mlp.output",
            input_catalog_sha256=_sha(f"input-catalog-{ordinal}"),
            input_width=2,
            output_width=2,
            group_indices=(ordinal,),
            channel_indices=(0,),
            fisher_ranks=(ordinal,),
            axial_orientations=(1,),
            native_parameter_count=6,
            fisher_mass=float(ordinal + 1),
            source_cluster_plan_sha256=clusters,
            source_fisher_coupling_sha256=fisher,
            parameter_catalog_sha256=catalog,
            source_model_sha256=source_model,
        )
        for ordinal in range(18)
    )
    fragment_plan = ParameterClusterLayerFragmentPlan(
        source_cluster_plan_sha256=clusters,
        source_fisher_coupling_sha256=fisher,
        parameter_catalog_sha256=catalog,
        source_model_sha256=source_model,
        cluster_count=1,
        source_group_count=18,
        assigned_group_count=18,
        assigned_native_parameter_count=18 * 6,
        fragments=fragments,
    )
    plan = build_parameter_layer_superfragments(fragment_plan)
    fit_split = _sha("fit-split")
    selection_split = _sha("selection-split")
    fits = []
    fit_inputs = torch.tensor(
        ((-1.0, 0.5), (0.2, 1.5), (1.1, -0.3)),
        dtype=torch.float64,
    )
    eval_inputs = torch.tensor(
        ((-0.7, 0.9), (1.4, 0.1)),
        dtype=torch.float64,
    )
    coefficient = torch.tensor(
        ((0.8, -0.2), (0.3, 0.9)),
        dtype=torch.float64,
    )
    for superfragment, fragment in zip(
        plan.superfragments,
        fragments,
        strict=True,
    ):
        common = {
            "layer_ordinal": superfragment.layer_ordinal,
            "layer_id": superfragment.layer_id,
            "input_site": superfragment.input_site,
            "activation_site": superfragment.activation_site,
            "output_site": superfragment.output_site,
            "intermediate_width": 1,
            "fragment_ids": (fragment.fragment_id,),
            "fragment_sha256s": (fragment.artifact_sha256,),
        }
        bias = torch.tensor(
            (0.01 * (superfragment.layer_ordinal + 1), -0.1),
            dtype=torch.float64,
        )
        fit_rows = FullMLPStackLayerRows(
            **common,
            inputs=fit_inputs,
            contributions=fit_inputs @ coefficient + bias,
            fisher_weights=torch.tensor((1.0, 0.8, 1.2)),
            row_keys=(
                ("fit.a", 0),
                ("fit.b", 0),
                ("fit.c", 0),
            ),
            sequences=3,
        )
        eval_rows = FullMLPStackLayerRows(
            **common,
            inputs=eval_inputs,
            contributions=eval_inputs @ coefficient + bias,
            fisher_weights=torch.tensor((0.9, 1.1)),
            row_keys=(("selection.a", 0), ("selection.b", 0)),
            sequences=2,
        )
        fits.append(
            fit_full_mlp_stack_generators(
                fit_rows,
                eval_rows,
                superfragment=superfragment,
                source_model_sha256=source_model,
                parameter_catalog_sha256=catalog,
                fisher_coupling_sha256=fisher,
                superfragment_plan_sha256=plan.artifact_sha256,
                fit_split_sha256=fit_split,
                eval_split_sha256=selection_split,
                mode_ranks=(1,),
                selected_mode_rank=1,
                generator_ranks=(1,),
                selected_generator_rank=1,
                ridge=1e-6,
            )
        )
    return plan, tuple(fits)


def _arguments() -> dict[str, object]:
    plan, fits = _analysis()
    source_parameters = 1_000
    native_parameters = sum(
        fit.superfragment.native_parameter_count for fit in fits
    )
    generator_parameters = sum(
        fit.executable_plan.parameter_count for fit in fits
    )
    retained_parameters = source_parameters - native_parameters
    candidate_parameters = retained_parameters + generator_parameters
    native_macs = native_parameters
    generator_macs = sum(
        fit.executable_plan.macs_per_token for fit in fits
    )
    valid_tokens = 10
    fit_content = (_sha("fit.0"), _sha("fit.1"))
    selection_content = (_sha("selection.0"), _sha("selection.1"))
    assessment_content = (_sha("assessment.0"), _sha("assessment.1"))
    model = {
        "model_id": "google/gemma-3-270m",
        "requested_revision": "1" * 40,
        "resolved_commit": "1" * 40,
        "adapter_model_fingerprint": plan.source_model_sha256,
        "source_whole_model_learned_parameters": source_parameters,
        "local_files_only": True,
    }
    protocol = {
        "scope": "full_native_mlp_stack_replacement",
        "transformer_layer_count": 18,
        "source_fragment_count": plan.source_fragment_count,
        "removed_mode_count": plan.assigned_group_count,
        "mode_ranks": (1,),
        "selected_mode_rank": 1,
        "generator_ranks": (1,),
        "selected_generator_rank": 1,
        "generator_ridge": 1e-6,
        "fit_rule": "fisher_weighted_full_layer_before_modes",
        "execution_path": "edgeless_dense_fused_residual_generators",
        "native_components_retained": (
            "embeddings",
            "attention",
            "normalization",
            "language_model_head",
        ),
        "local_files_only": True,
    }
    splits = {
        "fit_export": {"artifact_sha256": _sha("fit-export")},
        "eval_export": {"artifact_sha256": _sha("eval-export")},
        "fit": {
            "role": "generator_fit",
            "serialized_sha256": _sha("fit-split"),
            "content_sha256": fit_content,
        },
        "upstream_evaluation": {
            "role": "development_partition_source",
            "serialized_sha256": _sha("upstream-eval-split"),
            "content_sha256": (*selection_content, *assessment_content),
        },
        "selection": {
            "role": "generator_selection",
            "serialized_sha256": _sha("selection-split"),
            "content_sha256": selection_content,
        },
        "assessment": {
            "role": "open_development_assessment",
            "serialized_sha256": _sha("assessment-split"),
            "content_sha256": assessment_content,
        },
        "partition": {"artifact_sha256": _sha("partition")},
        "provenance": {
            "assurance": "caller_declared_self_attested",
            "externally_authenticated": False,
            "selection_assessment_disjoint": True,
            "heldout_confirmation": False,
        },
    }
    upstream = {
        "source_schema": "fisher_graph.upstream",
        "source_format_version": 3,
        "source_scientific_payload_sha256": _sha("upstream-payload"),
        "fit_prompt_trace_sha256": _sha("fit-trace"),
        "parameter_catalog_sha256": plan.parameter_catalog_sha256,
        "fisher_coupling_sha256": (
            plan.source_fisher_coupling_sha256
        ),
        "parameter_clusters_sha256": plan.source_cluster_plan_sha256,
        "parameter_cluster_fragments_sha256": (
            plan.source_fragment_plan_sha256
        ),
    }
    static = {
        "replacement_scope": "full_native_mlp_stack_replacement",
        "replaced_layer_count": 18,
        "removed_mode_count": plan.assigned_group_count,
        "source_whole_model_learned_parameters": source_parameters,
        "logical_native_mlp_stack_learned_parameters": native_parameters,
        "logical_retained_native_non_mlp_learned_parameters": (
            retained_parameters
        ),
        "logical_generator_stack_learned_parameters": generator_parameters,
        "logical_candidate_learned_parameters": candidate_parameters,
        "logical_net_stored_parameter_savings": (
            source_parameters - candidate_parameters
        ),
        "logical_linear_macs_native_mlp_stack": (
            valid_tokens * native_macs
        ),
        "logical_generator_macs": valid_tokens * generator_macs,
        "logical_generator_bias_additions": valid_tokens * 2 * 18,
    }
    evaluation = {
        "execution_path": "edgeless_full_mlp_stack_rung",
        "assessment_role": "open_development_assessment",
        "heldout_confirmation": False,
        "assessment_membership_exact": True,
        "assessment_used_for_fitting": False,
        "supervised_tokens": 8,
        "logical_valid_tokens": valid_tokens,
        "declared_scope": {
            "replacement_scope": "full_native_mlp_stack_replacement",
            "layer_count": 18,
            "removed_mode_count": plan.assigned_group_count,
            "mode_counts_by_layer": (1,) * 18,
            "all_declared_layers_and_modes_replaced": True,
        },
        "conditions": {
            "native": {
                "nll_per_token": 2.0,
                "delta_nll_per_token": 0.0,
                "native_to_candidate_kl_per_token": 0.0,
                "top1_agreement_to_native": 1.0,
            },
            "generated_full_stack": {
                "nll_per_token": 2.1,
                "delta_nll_per_token": 0.1,
                "native_to_candidate_kl_per_token": 0.01,
                "top1_agreement_to_native": 0.9,
            },
            "matched_deletion": {
                "nll_per_token": 3.0,
                "delta_nll_per_token": 1.0,
                "native_to_candidate_kl_per_token": 0.2,
                "top1_agreement_to_native": 0.6,
            },
        },
        "control_validation": {
            "physical_scope_identical": True,
            "generated_compute_executed": True,
            "matched_deletion_compute_zero": True,
        },
        "resource_accounting": {
            "generated_full_stack": {
                **static,
                "logical_executed_generator_macs": (
                    valid_tokens * generator_macs
                ),
                "logical_executed_generator_bias_additions": (
                    valid_tokens * 2 * 18
                ),
            },
            "matched_deletion": {
                **static,
                "logical_executed_generator_macs": 0,
                "logical_executed_generator_bias_additions": 0,
            },
        },
        "latency_or_kernel_speed_claim": False,
        "assessment_split_sha256": _sha("assessment-split"),
    }
    return {
        "model": model,
        "protocol": protocol,
        "splits": splits,
        "upstream_metadata": upstream,
        "superfragment_plan": plan,
        "generator_fits": fits,
        "evaluation": evaluation,
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


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def test_strict_roundtrip_compact_report_and_refuse_overwrite(
    tmp_path: Path,
) -> None:
    arguments = _arguments()
    output = tmp_path / "full-stack.pt"
    report = save_gemma3_full_mlp_stack_artifact(output, **arguments)
    restored = load_gemma3_full_mlp_stack_artifact(output)

    assert restored["schema"] == GEMMA3_FULL_MLP_STACK_SCHEMA
    assert (
        restored["format_version"]
        == GEMMA3_FULL_MLP_STACK_FORMAT_VERSION
    )
    assert len(restored["generator_fits"]) == 18
    assert tuple(
        value["superfragment"]["layer_ordinal"]
        for value in restored["generator_fits"]
    ) == tuple(range(18))
    assert restored["scientific_status"]["compression_claim"] is False
    assert restored["scientific_status"]["heldout_confirmation"] is False
    assert (
        restored["scientific_status"]["whole_transformer_replaced"]
        is False
    )
    assert report["artifact"]["tensor_file"] == output.name
    assert len(report["layers"]) == 18
    assert _contains_tensor(report) is False
    assert json.loads(
        output.with_suffix(".json").read_text(encoding="utf-8")
    ) == json.loads(
        json.dumps(report, sort_keys=True, allow_nan=False)
    )
    assert not tuple(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError, match="overwrite"):
        save_gemma3_full_mlp_stack_artifact(output, **arguments)


def test_loader_rejects_rehashed_tensor_order_and_resource_tamper(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_full_mlp_stack_payload(**_arguments())

    tensor = copy.deepcopy(payload)
    tensor["generator_fits"][0]["dense_fused_residual_plan"]["factors"][
        "output_factor"
    ][0, 0] += 0.2
    _rehash(tensor)
    tensor_path = tmp_path / "tensor.pt"
    torch.save(tensor, tensor_path)
    with pytest.raises(ValueError, match="sha256|hash"):
        load_gemma3_full_mlp_stack_artifact(tensor_path)

    reordered = copy.deepcopy(payload)
    values = list(reordered["generator_fits"])
    values[0], values[1] = values[1], values[0]
    reordered["generator_fits"] = tuple(values)
    _rehash(reordered)
    reordered_path = tmp_path / "reordered.pt"
    torch.save(reordered, reordered_path)
    with pytest.raises(ValueError, match="ordered fits"):
        load_gemma3_full_mlp_stack_artifact(reordered_path)

    resources = copy.deepcopy(payload)
    resources["resource_accounting"]["net_stored_parameter_savings"] += 1
    _rehash(resources)
    resources_path = tmp_path / "resources.pt"
    torch.save(resources, resources_path)
    with pytest.raises(ValueError, match="resource accounting"):
        load_gemma3_full_mlp_stack_artifact(resources_path)

    bias_work = copy.deepcopy(payload)
    bias_work["evaluation"]["resource_accounting"][
        "generated_full_stack"
    ]["logical_executed_generator_bias_additions"] -= 1
    _rehash(bias_work)
    bias_path = tmp_path / "bias-work.pt"
    torch.save(bias_work, bias_path)
    with pytest.raises(ValueError, match="work controls"):
        load_gemma3_full_mlp_stack_artifact(bias_path)


def test_loader_rejects_overclaim_forbidden_source_and_wrong_scope(
    tmp_path: Path,
) -> None:
    payload = build_gemma3_full_mlp_stack_payload(**_arguments())

    overclaim = copy.deepcopy(payload)
    overclaim["scientific_status"]["compression_claim"] = True
    _rehash(overclaim)
    overclaim_path = tmp_path / "overclaim.pt"
    torch.save(overclaim, overclaim_path)
    with pytest.raises(ValueError, match="header|safety"):
        load_gemma3_full_mlp_stack_artifact(overclaim_path)

    forbidden = copy.deepcopy(payload)
    forbidden["model"]["input_ids"] = (1, 2, 3)
    _rehash(forbidden)
    forbidden_path = tmp_path / "forbidden.pt"
    torch.save(forbidden, forbidden_path)
    with pytest.raises(ValueError, match="forbidden field"):
        load_gemma3_full_mlp_stack_artifact(forbidden_path)

    wrong_scope = copy.deepcopy(payload)
    wrong_scope["evaluation"]["declared_scope"][
        "replacement_scope"
    ] = "whole_transformer_replacement"
    _rehash(wrong_scope)
    wrong_scope_path = tmp_path / "wrong-scope.pt"
    torch.save(wrong_scope, wrong_scope_path)
    with pytest.raises(ValueError, match="complete native MLP stack"):
        load_gemma3_full_mlp_stack_artifact(wrong_scope_path)

    metric = copy.deepcopy(payload)
    metric["evaluation"]["conditions"]["generated_full_stack"][
        "top1_agreement_to_native"
    ] = 1.1
    _rehash(metric)
    metric_path = tmp_path / "metric.pt"
    torch.save(metric, metric_path)
    with pytest.raises(ValueError, match="condition metrics"):
        load_gemma3_full_mlp_stack_artifact(metric_path)


def test_atomic_publish_removes_partial_pair_on_sidecar_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments()
    output = tmp_path / "atomic.pt"
    actual_link = artifact.os.link
    calls = 0

    def failing_second_link(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated sidecar publish failure")
        actual_link(source, destination)

    monkeypatch.setattr(artifact.os, "link", failing_second_link)
    with pytest.raises(OSError, match="sidecar"):
        save_gemma3_full_mlp_stack_artifact(output, **arguments)

    assert output.exists() is False
    assert output.with_suffix(".json").exists() is False
    assert not tuple(tmp_path.glob("*.tmp"))


def test_report_builder_is_deterministic_and_tensor_free() -> None:
    payload = build_gemma3_full_mlp_stack_payload(**_arguments())
    first = build_gemma3_full_mlp_stack_report(
        payload,
        tensor_file="full-stack.pt",
    )
    second = build_gemma3_full_mlp_stack_report(
        payload,
        tensor_file="full-stack.pt",
    )

    assert first == second
    assert first["report_sha256"] == second["report_sha256"]
    assert _contains_tensor(first) is False
