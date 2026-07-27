from __future__ import annotations

import copy
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_full_mlp_stack_refit_artifact as artifact
from fisher_graph.full_mlp_stack_generators import (
    FullMLPStackGeneratorFit,
    fit_full_mlp_stack_generators,
)
from fisher_graph.gemma3_full_mlp_stack_refit_artifact import (
    GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION,
    GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA,
    build_gemma3_full_mlp_stack_refit_payload,
    compiled_prefix_catalog_sha256,
    frozen_baseline_conditions_sha256,
    load_gemma3_full_mlp_stack_refit_artifact,
    save_gemma3_full_mlp_stack_refit_artifact,
    trajectory_breakpoint_row_sha256,
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


def _metrics(nll: float, *, native_nll: float = 2.0) -> dict[str, float]:
    return {
        "nll_per_token": nll,
        "delta_nll_per_token": nll - native_nll,
        "native_to_candidate_kl_per_token": max(nll - native_nll, 0.0)
        / 2.0,
        "top1_agreement_to_native": max(0.0, 1.0 - (nll - native_nll)),
    }


def _fit(
    superfragment: object,
    fragment: ParameterClusterLayerFragment,
    *,
    plan_sha256: str,
    fit_split: str,
    selection_split: str,
    source_model: str,
    catalog: str,
    fisher: str,
    refit: bool,
) -> FullMLPStackGeneratorFit:
    ordinal = fragment.layer_ordinal
    fit_inputs = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 40, dtype=torch.float64),
            torch.linspace(0.5, 2.0, 40, dtype=torch.float64),
        ),
        dim=1,
    )
    selection_inputs = torch.stack(
        (
            torch.linspace(-0.8, 1.2, 20, dtype=torch.float64),
            torch.linspace(0.1, 1.4, 20, dtype=torch.float64),
        ),
        dim=1,
    )
    coefficient = torch.tensor(
        (
            (0.82 if refit else 0.8, -0.2),
            (0.3, 0.91 if refit else 0.9),
        ),
        dtype=torch.float64,
    )
    bias = torch.tensor(
        (0.01 * (ordinal + 1), -0.1),
        dtype=torch.float64,
    )
    common = {
        "layer_ordinal": superfragment.layer_ordinal,
        "layer_id": superfragment.layer_id,
        "input_site": superfragment.input_site,
        "activation_site": superfragment.activation_site,
        "output_site": superfragment.output_site,
        "intermediate_width": superfragment.mode_count,
        "fragment_ids": (fragment.fragment_id,),
        "fragment_sha256s": (fragment.artifact_sha256,),
    }
    marker = "refit" if refit else "source"
    fit_rows = FullMLPStackLayerRows(
        **common,
        inputs=fit_inputs,
        contributions=fit_inputs @ coefficient + bias,
        fisher_weights=torch.linspace(
            0.8,
            1.2,
            40,
            dtype=torch.float64,
        ),
        row_keys=tuple((f"{marker}.fit.{index}", 0) for index in range(40)),
        sequences=40,
    )
    selection_rows = FullMLPStackLayerRows(
        **common,
        inputs=selection_inputs,
        contributions=selection_inputs @ coefficient + bias,
        fisher_weights=torch.linspace(
            0.9,
            1.1,
            20,
            dtype=torch.float64,
        ),
        row_keys=tuple(
            (f"{marker}.selection.{index}", 0) for index in range(20)
        ),
        sequences=20,
    )
    return fit_full_mlp_stack_generators(
        fit_rows,
        selection_rows,
        superfragment=superfragment,
        source_model_sha256=source_model,
        parameter_catalog_sha256=catalog,
        fisher_coupling_sha256=fisher,
        superfragment_plan_sha256=plan_sha256,
        fit_split_sha256=fit_split,
        eval_split_sha256=selection_split,
        mode_ranks=(1,),
        selected_mode_rank=1,
        generator_ranks=(1,),
        selected_generator_rank=1,
        ridge=1e-6,
    )


def _analysis() -> tuple[
    tuple[dict[str, object], ...],
    tuple[FullMLPStackGeneratorFit, ...],
    tuple[dict[str, object], ...],
]:
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
            group_indices=tuple(range(ordinal * 3, ordinal * 3 + 3)),
            channel_indices=(0, 1, 2),
            fisher_ranks=tuple(range(ordinal * 3, ordinal * 3 + 3)),
            axial_orientations=(1, 1, 1),
            native_parameter_count=18,
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
        source_group_count=18 * 3,
        assigned_group_count=18 * 3,
        assigned_native_parameter_count=18 * 18,
        fragments=fragments,
    )
    plan = build_parameter_layer_superfragments(fragment_plan)
    fit_split = _sha("fit-split")
    selection_split = _sha("selection-split")
    source_fits = tuple(
        _fit(
            superfragment,
            fragment,
            plan_sha256=plan.artifact_sha256,
            fit_split=fit_split,
            selection_split=selection_split,
            source_model=source_model,
            catalog=catalog,
            fisher=fisher,
            refit=False,
        )
        for superfragment, fragment in zip(
            plan.superfragments,
            fragments,
            strict=True,
        )
    )
    refit_fits = tuple(
        _fit(
            plan.superfragments[ordinal],
            fragments[ordinal],
            plan_sha256=plan.artifact_sha256,
            fit_split=fit_split,
            selection_split=selection_split,
            source_model=source_model,
            catalog=catalog,
            fisher=fisher,
            refit=True,
        )
        for ordinal in range(10, 18)
    )
    source_summaries = tuple(
        {
            "layer_ordinal": fit.superfragment.layer_ordinal,
            "layer_id": fit.superfragment.layer_id,
            "input_site": fit.superfragment.input_site,
            "output_site": fit.superfragment.output_site,
            "input_width": fit.superfragment.input_width,
            "intermediate_width": fit.superfragment.mode_count,
            "residual_width": fit.superfragment.output_width,
            "source_fit_sha256": fit.artifact_sha256,
            "superfragment_sha256": fit.superfragment.artifact_sha256,
            "superfragment_plan_sha256": fit.superfragment_plan_sha256,
            "source_model_sha256": fit.superfragment.source_model_sha256,
            "parameter_catalog_sha256": (
                fit.superfragment.parameter_catalog_sha256
            ),
            "source_fisher_coupling_sha256": (
                fit.superfragment.source_fisher_coupling_sha256
            ),
            "source_fragment_plan_sha256": (
                fit.superfragment.source_fragment_plan_sha256
            ),
            "source_cluster_plan_sha256": (
                fit.superfragment.source_cluster_plan_sha256
            ),
            "dense_plan_sha256": fit.executable_plan.artifact_sha256,
            "selected_mode_rank": fit.selected_mode_rank,
            "selected_generator_rank": fit.selected_generator_rank,
            "native_mlp_parameter_count": (
                fit.resource_metadata["native_mlp_parameter_count"]
            ),
            "dense_fused_parameter_count": (
                fit.resource_metadata["dense_fused_parameter_count"]
            ),
            "dense_fused_macs_per_token": (
                fit.resource_metadata["dense_fused_macs_per_token"]
            ),
        }
        for fit in source_fits
    )
    deployed_hashes = [
        row["dense_plan_sha256"] for row in source_summaries[:10]
    ]
    layer_refits = []
    for source_fit, refit_fit in zip(
        source_fits[10:],
        refit_fits,
        strict=True,
    ):
        ordinal = refit_fit.superfragment.layer_ordinal
        old_point = source_fit.coordinate_generators.point_for_rank(
            source_fit.selected_generator_rank
        )
        new_point = refit_fit.coordinate_generators.point_for_rank(
            refit_fit.selected_generator_rank
        )
        assert old_point is not None and new_point is not None
        prefix_ordinals = tuple(range(ordinal))
        layer_refits.append(
            {
                "layer_ordinal": ordinal,
                "generated_prefix_ordinals": prefix_ordinals,
                "generated_prefix_plan_sha256s": tuple(deployed_hashes),
                "generated_prefix_catalog_sha256": (
                    compiled_prefix_catalog_sha256(
                        prefix_ordinals,
                        tuple(deployed_hashes),
                    )
                ),
                "source_fit_sha256": source_fit.artifact_sha256,
                "refit_fit_sha256": refit_fit.artifact_sha256,
                "fit_row_key_sha256": refit_fit.fit_row_key_sha256,
                "selection_row_key_sha256": refit_fit.eval_row_key_sha256,
                "fit_observations": 40,
                "fit_sequences": 40,
                "selection_observations": 20,
                "selection_sequences": 20,
                "old_selected_mode_rank": source_fit.selected_mode_rank,
                "old_selected_generator_rank": (
                    source_fit.selected_generator_rank
                ),
                "refit_selected_mode_rank": refit_fit.selected_mode_rank,
                "refit_selected_generator_rank": (
                    refit_fit.selected_generator_rank
                ),
                "old_plan_fit_metrics": old_point.fit_metrics.metadata(),
                "old_plan_selection_metrics": (
                    old_point.eval_metrics.metadata()
                ),
                "refit_plan_fit_metrics": new_point.fit_metrics.metadata(),
                "refit_plan_selection_metrics": (
                    new_point.eval_metrics.metadata()
                ),
                "refit_resource_metadata": refit_fit.resource_metadata,
            }
        )
        deployed_hashes.append(refit_fit.executable_plan.artifact_sha256)
    return source_summaries, refit_fits, tuple(layer_refits)


@pytest.fixture(scope="module")
def base_arguments() -> dict[str, object]:
    source_summaries, refit_fits, layer_refits = _analysis()
    fit_content = tuple(_sha(f"fit.{index}") for index in range(40))
    selection_content = tuple(
        _sha(f"selection.{index}") for index in range(20)
    )
    assessment_content = tuple(
        _sha(f"assessment.{index}") for index in range(20)
    )
    baseline = {
        "native": _metrics(2.0),
        "generated_full_stack": _metrics(2.3),
        "matched_deletion": _metrics(8.0),
    }
    breakpoint = {
        "depth": 10,
        "metrics": _metrics(2.1),
        "resources_sha256": _sha("breakpoint-resources"),
    }
    return {
        "model": {
            "model_id": "google/gemma-3-270m",
            "requested_revision": "1" * 40,
            "resolved_commit": "1" * 40,
            "adapter_model_fingerprint": _sha("model"),
            "source_whole_model_learned_parameters": 1_000,
            "local_files_only": True,
        },
        "frozen_sources": {
            "full_stack": {
                "schema": (
                    "fisher_graph.gemma3_full_native_mlp_stack_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("full-stack-file"),
                "scientific_payload_sha256": _sha("full-stack-payload"),
                "baseline_conditions_sha256": (
                    frozen_baseline_conditions_sha256(baseline)
                ),
            },
            "trajectory": {
                "schema": (
                    "fisher_graph."
                    "gemma3_frozen_full_mlp_stack_trajectory_development"
                ),
                "format_version": 1,
                "artifact_file_sha256": _sha("trajectory-file"),
                "scientific_payload_sha256": _sha("trajectory-payload"),
                "breakpoint_direction": "prefix",
                "breakpoint_depth": 10,
                "breakpoint_row_sha256": (
                    trajectory_breakpoint_row_sha256(breakpoint)
                ),
            },
        },
        "splits": {
            "fit": {
                "role": "generator_fit",
                "serialized_sha256": _sha("fit-split"),
                "content_sha256": fit_content,
                "example_count": 40,
                "logical_valid_tokens": 4_000,
                "supervised_tokens": 3_960,
            },
            "selection": {
                "role": "generator_selection",
                "serialized_sha256": _sha("selection-split"),
                "content_sha256": selection_content,
                "example_count": 20,
                "logical_valid_tokens": 2_000,
                "supervised_tokens": 1_980,
            },
            "assessment": {
                "role": "open_development_assessment",
                "serialized_sha256": _sha("assessment-split"),
                "content_sha256": assessment_content,
                "example_count": 20,
                "logical_valid_tokens": 2_000,
                "supervised_tokens": 1_980,
            },
            "provenance": {
                "assurance": "caller_declared_self_attested",
                "externally_authenticated": False,
                "fit_used_for_generator_refit": True,
                "selection_used_for_generator_refit": True,
                "assessment_used_for_generator_refit": False,
                "assessment_used_for_generator_rank_selection": False,
                "fit_selection_assessment_disjoint": True,
                "assessment_evaluated_only_after_refit_freeze": True,
                "heldout_confirmation": False,
            },
        },
        "protocol": {
            "scope": (
                "sequential_compiled_trajectory_full_mlp_stack_refit"
            ),
            "transformer_layer_count": 18,
            "refit_start_layer": 10,
            "unchanged_layer_ordinals": tuple(range(10)),
            "refit_layer_order": tuple(range(10, 18)),
            "refit_rule": (
                "sequential_teacher_on_actual_compiled_prefix"
            ),
            "fisher_weighting": "current_compiled_prefix_rows",
            "jacobian_policy": (
                "no_explicit_jacobian_correction_in_direct_refit_rung"
            ),
            "rank_policy": "fixed_from_frozen_full_stack",
            "resource_budget_policy": (
                "exact_per_layer_equality_to_frozen_full_stack"
            ),
            "execution_path": (
                "frozen_prefix_then_sequential_refit_dense_generators"
            ),
            "source_model_weights_mutated": False,
            "assessment_role": "open_development_assessment",
            "assessment_after_refit_freeze": True,
            "generator_rank_selection_performed": False,
            "heldout_confirmation": False,
            "compression_claim": False,
            "latency_or_kernel_speed_claim": False,
            "local_files_only": True,
        },
        "source_layer_summaries": source_summaries,
        "refit_generator_fits": refit_fits,
        "layer_refits": layer_refits,
        "evaluation": {
            "execution_path": "sequential_refit_full_mlp_stack_rung",
            "assessment_role": "open_development_assessment",
            "heldout_confirmation": False,
            "assessment_membership_exact": True,
            "refit_frozen_before_assessment": True,
            "fit_and_selection_used_for_refit": True,
            "assessment_used_for_refit": False,
            "generator_rank_selection_performed": False,
            "latency_or_kernel_speed_claim": False,
            "supervised_tokens": 1_980,
            "logical_valid_tokens": 2_000,
            "assessment_split_sha256": _sha("assessment-split"),
            "frozen_baseline_conditions_sha256": (
                frozen_baseline_conditions_sha256(baseline)
            ),
            "conditions": {
                "native": baseline["native"],
                "frozen_generated_full_stack": (
                    baseline["generated_full_stack"]
                ),
                "sequential_refit_full_stack": _metrics(2.15),
                "matched_deletion": baseline["matched_deletion"],
            },
            "control_validation": {
                "native_matches_frozen_full_stack_artifact": True,
                "frozen_generated_matches_frozen_full_stack_artifact": True,
                "matched_deletion_matches_frozen_full_stack_artifact": True,
                "frozen_generated_matches_trajectory_full_stack_endpoint": (
                    True
                ),
                "physical_scope_identical": True,
                "refit_generator_compute_executed": True,
                "matched_deletion_compute_zero": True,
            },
        },
    }


def _arguments(base: dict[str, object]) -> dict[str, object]:
    return {
        key: (
            value
            if key == "refit_generator_fits"
            else copy.deepcopy(value)
        )
        for key, value in base.items()
    }


def _build(base: dict[str, object]) -> dict[str, object]:
    return build_gemma3_full_mlp_stack_refit_payload(
        **_arguments(base),  # type: ignore[arg-type]
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(child) for child in value)
    return False


def test_builds_strict_overlay_with_exact_scope(
    base_arguments: dict[str, object],
) -> None:
    payload = _build(base_arguments)

    assert payload["schema"] == GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA
    assert (
        payload["format_version"]
        == GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION
    )
    assert len(payload["source_layer_summaries"]) == 18
    assert len(payload["refit_generator_fits"]) == 8
    assert len(payload["unchanged_prefix_fit_sha256s"]) == 10
    assert tuple(
        row["superfragment"]["layer_ordinal"]
        for row in payload["refit_generator_fits"]
    ) == tuple(range(10, 18))
    assert payload["scientific_status"]["compression_claim"] is False
    assert payload["scientific_status"]["heldout_confirmation"] is False


def test_roundtrip_saves_tensor_free_report(
    tmp_path: Path,
    base_arguments: dict[str, object],
) -> None:
    output = tmp_path / "refit.pt"
    report = save_gemma3_full_mlp_stack_refit_artifact(
        output,
        **_arguments(base_arguments),  # type: ignore[arg-type]
    )
    loaded = load_gemma3_full_mlp_stack_refit_artifact(output)

    assert loaded["scientific_payload_sha256"] == (
        report["artifact"]["scientific_payload_sha256"]
    )
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert _contains_tensor(loaded)
    assert not _contains_tensor(report)
    json.dumps(report, allow_nan=False)
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_digest_is_deterministic_across_mapping_order(
    base_arguments: dict[str, object],
) -> None:
    first = _build(base_arguments)
    arguments = _arguments(base_arguments)
    arguments["model"] = dict(
        reversed(list(arguments["model"].items()))  # type: ignore[union-attr]
    )
    second = build_gemma3_full_mlp_stack_refit_payload(
        **arguments,  # type: ignore[arg-type]
    )

    assert first["scientific_payload_sha256"] == (
        second["scientific_payload_sha256"]
    )


def test_refuses_overwrite_and_wrong_suffix(
    tmp_path: Path,
    base_arguments: dict[str, object],
) -> None:
    output = tmp_path / "refit.pt"
    output.write_bytes(b"owned")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_gemma3_full_mlp_stack_refit_artifact(
            output,
            **_arguments(base_arguments),  # type: ignore[arg-type]
        )
    assert output.read_bytes() == b"owned"

    with pytest.raises(ValueError, match=r"must use \.pt"):
        save_gemma3_full_mlp_stack_refit_artifact(
            tmp_path / "refit.json",
            **_arguments(base_arguments),  # type: ignore[arg-type]
        )


def test_requires_exact_sequential_layer_order(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["layer_refits"][0]["layer_ordinal"] = 11  # type: ignore[index]

    with pytest.raises(ValueError, match="ordered 10 through 17"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_requires_exact_compiled_prefix_lineage(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["layer_refits"][1][  # type: ignore[index]
        "generated_prefix_plan_sha256s"
    ] = tuple([_sha("wrong")] * 11)

    with pytest.raises(ValueError, match="compiled-prefix lineage"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_requires_unchanged_rank_and_resource_budget(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["layer_refits"][0][  # type: ignore[index]
        "old_selected_generator_rank"
    ] = 2

    with pytest.raises(ValueError, match="changed a fixed rank"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_requires_same_row_observations_for_old_and_refit_metrics(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["layer_refits"][0]["old_plan_fit_metrics"][  # type: ignore[index]
        "observations"
    ] = 39

    with pytest.raises(ValueError, match="observations differ"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_rejects_assessment_leakage_and_overlap(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["splits"]["provenance"][  # type: ignore[index]
        "assessment_used_for_generator_refit"
    ] = True
    with pytest.raises(ValueError, match="overclaims or permits leakage"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )

    arguments = _arguments(base_arguments)
    assessment_hashes = list(
        arguments["splits"]["assessment"]["content_sha256"]  # type: ignore[index]
    )
    assessment_hashes[0] = arguments["splits"]["fit"][  # type: ignore[index]
        "content_sha256"
    ][0]
    arguments["splits"]["assessment"]["content_sha256"] = (  # type: ignore[index]
        tuple(assessment_hashes)
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_rejects_frozen_baseline_drift(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["evaluation"]["conditions"][  # type: ignore[index]
        "frozen_generated_full_stack"
    ]["top1_agreement_to_native"] = 0.1

    with pytest.raises(ValueError, match="differs from its frozen baseline"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_rejects_forbidden_prompt_rows(
    base_arguments: dict[str, object],
) -> None:
    arguments = _arguments(base_arguments)
    arguments["evaluation"]["prompt_text"] = "secret"  # type: ignore[index]

    with pytest.raises(ValueError, match="forbidden field"):
        build_gemma3_full_mlp_stack_refit_payload(
            **arguments,  # type: ignore[arg-type]
        )


def test_load_detects_tensor_payload_tampering(
    tmp_path: Path,
    base_arguments: dict[str, object],
) -> None:
    payload = _build(base_arguments)
    payload["evaluation"]["supervised_tokens"] = 1_981  # type: ignore[index]
    output = tmp_path / "tampered.pt"
    torch.save(payload, output)

    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_gemma3_full_mlp_stack_refit_artifact(output)


def test_post_save_failure_removes_only_owned_outputs(
    tmp_path: Path,
    base_arguments: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "refit.pt"

    def fail_load(path: Path | str) -> dict[str, object]:
        raise ValueError("injected post-save failure")

    monkeypatch.setattr(
        artifact,
        "load_gemma3_full_mlp_stack_refit_artifact",
        fail_load,
    )
    with pytest.raises(ValueError, match="injected post-save failure"):
        save_gemma3_full_mlp_stack_refit_artifact(
            output,
            **_arguments(base_arguments),  # type: ignore[arg-type]
        )

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_domain_helpers_bind_order_and_row_content() -> None:
    hashes = (_sha("a"), _sha("b"))
    first = compiled_prefix_catalog_sha256((0, 1), hashes)
    assert first == compiled_prefix_catalog_sha256((0, 1), hashes)
    assert first != compiled_prefix_catalog_sha256(
        (0, 1),
        tuple(reversed(hashes)),
    )

    row = {"depth": 10, "metric_sha256": _sha("metric")}
    assert trajectory_breakpoint_row_sha256(row) != (
        trajectory_breakpoint_row_sha256(
            {"depth": 10, "metric_sha256": _sha("other")}
        )
    )
