from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_FIT_EXPORT,
    DEFAULT_OUTPUT,
    GEMMA3_MODAL_GENERATOR_DEV_FORMAT_VERSION,
    GEMMA3_MODAL_GENERATOR_DEV_SCHEMA,
    LayerFragmentRows,
    _assert_source_safe_artifact,
    _payload_sha256,
    build_fit_fisher_cluster_pilot,
    build_parser,
    build_single_node_modal_compiler_pipeline,
    collect_layer_fragment_rows,
    evaluate_modal_generator_conditions,
    evaluate_modal_generator_graph_conditions,
    fit_layer_cluster_modal_generator,
    load_development_prompt_export,
    load_gemma3_modal_generator_dev_artifact,
    run_gemma3_modal_generator_dev_experiment,
    validate_development_split_pair,
)
from fisher_graph.modal_generator_graph import ModalGeneratorGraphExecutor
from fisher_graph.parameter_fisher_coupling import (
    NaturalMLPLayerParameterSpec,
    build_natural_mlp_parameter_group_catalog,
)
from fisher_graph.prompt_mode_tracing import (
    PromptModeTraceProvenance,
    collect_prompt_mode_trace,
)
from fisher_graph.streaming_analysis import ActivationScoreGradientRows
from fisher_graph.structured_mlp_cross_block_bundling import (
    CrossBlockLayerSpec,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_export(
    path: Path,
    *,
    prompts: tuple[str, ...],
    positions: tuple[int, ...],
    source_indices: tuple[int, ...] | None = None,
) -> None:
    if source_indices is None:
        source_indices = positions
    payload = {
        "schema": "fisher_graph.local_v9_a_fit_development_export",
        "format_version": 1,
        "scientific_status": "development_only",
        "source_corpus_id": "fake-v9",
        "source_role": "calibration_a_fit_only",
        "selection_rule": "unit_test_fixed_positions",
        "fit_positions": list(positions),
        "source_prompt_indices": list(source_indices),
        "source_fit_prompt_index_sha256": _sha("source-fit"),
        "prompts": list(prompts),
        "prompt_sha256": [_sha(prompt) for prompt in prompts],
        "family_ids": [f"family.{index % 2}" for index in positions],
        "guard_exported": False,
        "calibration_b_exported": False,
        "validation_exported": False,
        "test_exported": False,
        "model_or_tokenizer_accessed": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_development_exports_are_self_attested_overlap_checked_and_source_safe(
    tmp_path: Path,
) -> None:
    fit_path = tmp_path / "fit.json"
    eval_path = tmp_path / "eval.json"
    _write_export(
        fit_path,
        prompts=("fit alpha", "fit beta"),
        positions=(0, 1),
    )
    _write_export(
        eval_path,
        prompts=("eval alpha", "eval beta"),
        positions=(2, 3),
    )
    fit = load_development_prompt_export(fit_path)
    evaluation = load_development_prompt_export(eval_path)
    policy = validate_development_split_pair(fit, evaluation)

    assert policy["prompt_disjoint"] is True
    assert policy["source_prompt_index_disjoint"] is True
    assert policy["heldout_guard_used"] is False
    assert policy["export_provenance_assurance"] == (
        "declared_self_attested"
    )
    assert policy["export_provenance_externally_authenticated"] is False
    assert policy["split_membership_provenance"] == (
        "caller_declared_self_attested"
    )
    assert policy["split_membership_externally_authenticated"] is False
    assert policy["declared_membership_overlap_checked"] is True
    assert fit.metadata()["provenance_assurance"] == (
        "declared_self_attested"
    )
    assert fit.metadata()["externally_authenticated"] is False
    assert fit.prompts == ("fit alpha", "fit beta")
    assert "prompts" not in fit.metadata()
    assert "fit alpha" not in json.dumps(fit.metadata())

    _write_export(
        eval_path,
        prompts=("fit alpha", "other"),
        positions=(4, 5),
    )
    overlapping = load_development_prompt_export(eval_path)
    with pytest.raises(ValueError, match="declared memberships must not overlap"):
        validate_development_split_pair(fit, overlapping)


def test_runner_fails_closed_on_overlap_before_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_path = tmp_path / "fit.json"
    eval_path = tmp_path / "eval.json"
    _write_export(fit_path, prompts=("same prompt",), positions=(0,))
    _write_export(
        eval_path,
        prompts=("same prompt",),
        positions=(1,),
        source_indices=(1,),
    )

    def forbidden_model_load(**_: object) -> object:
        raise AssertionError("model loading must not happen")

    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_dev_experiment.load_gemma3",
        forbidden_model_load,
    )
    with pytest.raises(ValueError, match="declared memberships must not overlap"):
        run_gemma3_modal_generator_dev_experiment(
            fit_export_path=fit_path,
            eval_export_path=eval_path,
            revision="a" * 40,
            output=tmp_path / "result.pt",
        )


def test_runner_rejects_tokenized_content_overlap_before_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_gemma3_modal_generator_executor import _adapter, _batch

    fit_path = tmp_path / "fit.json"
    eval_path = tmp_path / "eval.json"
    _write_export(fit_path, prompts=("fit prompt",), positions=(0,))
    _write_export(eval_path, prompts=("eval prompt",), positions=(1,))
    model = _adapter().module
    shared_content = _sha("same token tensors after tokenization")

    def materialize(
        _tokenizer: object,
        _prompts: tuple[str, ...],
        *,
        split_name: str,
        **_: object,
    ):
        return (
            (_batch(),),
            {
                "serialized_sha256": _sha(split_name),
                "examples": [
                    {
                        "content_sha256": shared_content,
                        "valid_tokens": 6,
                        "supervised_positions": 6,
                    }
                ],
            },
        )

    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_dev_experiment."
        "resolve_gemma3_huggingface_paths",
        lambda _cache: {"hub_cache": tmp_path},
    )
    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_dev_experiment.load_gemma3",
        lambda **_: (object(), model),
    )
    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_dev_experiment."
        "_model_provenance",
        lambda *_args, requested_revision, **_kwargs: {
            "resolved_commit": requested_revision,
        },
    )
    monkeypatch.setattr(
        "fisher_graph.gemma3_modal_generator_dev_experiment."
        "_materialize_split",
        materialize,
    )
    with pytest.raises(ValueError, match="tokenized content overlaps"):
        run_gemma3_modal_generator_dev_experiment(
            fit_export_path=fit_path,
            eval_export_path=eval_path,
            revision="a" * 40,
            output=tmp_path / "result.pt",
        )


def _layer_specs() -> tuple[CrossBlockLayerSpec, ...]:
    return (
        CrossBlockLayerSpec(
            layer_id="layer.0",
            layer_ordinal=0,
            activation_site="layer.0.mlp.down_input",
            width=4,
        ),
        CrossBlockLayerSpec(
            layer_id="layer.1",
            layer_ordinal=1,
            activation_site="layer.1.mlp.down_input",
            width=4,
        ),
    )


def _trace_effect_rows() -> tuple[ActivationScoreGradientRows, ...]:
    effects = torch.tensor(
        [
            [8.0, 7.0, 0.2, -0.2, 4.0, 3.0, 0.1, -0.1],
            [7.0, 6.0, -0.2, 0.2, 3.0, 2.0, -0.1, 0.1],
            [8.0, 7.0, 0.2, -0.2, 4.0, 3.0, 0.1, -0.1],
            [7.0, 6.0, -0.2, 0.2, 3.0, 2.0, -0.1, 0.1],
        ],
        dtype=torch.float64,
    )
    specs = _layer_specs()
    result = []
    for prompt_index, prompt_effects in enumerate(effects):
        first = prompt_effects[:4].unsqueeze(0)
        second = prompt_effects[4:].unsqueeze(0)
        result.append(
            ActivationScoreGradientRows(
                activations={
                    specs[0].activation_site: torch.ones_like(first),
                    specs[1].activation_site: torch.ones_like(second),
                },
                score_gradients={
                    specs[0].activation_site: first,
                    specs[1].activation_site: second,
                },
                logical_positions=torch.tensor([prompt_index]),
                loss=float(prompt_index + 1),
                example_id=f"fit.{prompt_index}",
            )
        )
    return tuple(result)


def _discover_fragment():
    model_sha = _sha("model")
    split_sha = _sha("fit-split")
    objective_sha = _sha("objective")
    specs = _layer_specs()
    trace = collect_prompt_mode_trace(
        _trace_effect_rows(),
        layer_specs=specs,
        provenance=PromptModeTraceProvenance(
            source_model_fingerprint=model_sha,
            calibration_split_sha256=split_sha,
            objective_sha256=objective_sha,
        ),
    )
    catalog = build_natural_mlp_parameter_group_catalog(
        model_fingerprint=model_sha,
        layer_specs=tuple(
            NaturalMLPLayerParameterSpec.from_cross_block_layer_spec(
                spec,
                input_width=3,
                output_width=3,
            )
            for spec in specs
        ),
    )
    fisher, clusters, fragment_plan, selected = (
        build_fit_fisher_cluster_pilot(
            trace,
            parameter_catalog=catalog,
            cluster_count=2,
            minimum_fragment_modes=2,
        )
    )
    return trace, catalog, fisher, clusters, fragment_plan, selected


def test_fit_fisher_cluster_pilot_uses_exact_fragment_bridge() -> None:
    trace, catalog, fisher, clusters, fragment_plan, selected = (
        _discover_fragment()
    )

    assert fisher.score_factor.equal(trace.prompt_effects)
    assert clusters.config.source_fisher_coupling_sha256 == (
        fisher.artifact_sha256
    )
    assert fragment_plan.source_cluster_plan_sha256 == clusters.artifact_sha256
    assert fragment_plan.parameter_catalog_sha256 == catalog.artifact_sha256
    assert selected == fragment_plan.top_by_fisher_mass(
        fragment_plan.fragment_count
    )[0]
    assert selected.mode_count >= 2
    assert selected.artifact_sha256
    assert selected.source_fisher_coupling_sha256 == fisher.artifact_sha256


def test_collect_fragment_rows_computes_native_residual_and_token_fisher() -> None:
    down = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    z = torch.tensor([[2.0, 9.0, 3.0, 8.0], [4.0, 7.0, 5.0, 6.0]])
    gradient = torch.tensor(
        [[0.5, 0.0, 2.0, 0.0], [1.0, 0.0, 3.0, 0.0]]
    )
    row = ActivationScoreGradientRows(
        activations={"input": x, "down": z},
        score_gradients={"input": torch.zeros_like(x), "down": gradient},
        logical_positions=torch.tensor([0, 1]),
        loss=1.0,
        example_id="fragment.0",
    )

    result = collect_layer_fragment_rows(
        (row,),
        input_site="input",
        down_input_site="down",
        mode_indices=(0, 2),
        down_projection_weight=down,
    )

    expected = z[:, (0, 2)] @ down[:, (0, 2)].T
    expected_weights = (
        z[:, (0, 2)] * gradient[:, (0, 2)]
    ).square().sum(dim=1)
    torch.testing.assert_close(result.inputs, x.double())
    torch.testing.assert_close(result.contributions, expected.double())
    torch.testing.assert_close(result.fisher_weights, expected_weights.double())
    assert result.sequences == 1


def test_computational_modes_feed_coordinate_generator_and_fold_decoder() -> None:
    _, catalog, fisher, _, fragment_plan, selected = _discover_fragment()
    generator = torch.Generator().manual_seed(9021)
    x_fit = torch.randn(32, 3, generator=generator, dtype=torch.float64)
    x_eval = torch.randn(16, 3, generator=generator, dtype=torch.float64)
    coefficient = torch.tensor(
        [[1.0, 0.5, -0.25], [0.0, 1.0, 0.5], [0.5, -0.25, 1.0]],
        dtype=torch.float64,
    )
    offset = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    y_fit = x_fit @ coefficient + offset
    y_eval = x_eval @ coefficient + offset
    fit = LayerFragmentRows(
        inputs=x_fit,
        contributions=y_fit,
        fisher_weights=torch.linspace(0.5, 2.0, 32),
        sequences=8,
    )
    evaluation = LayerFragmentRows(
        inputs=x_eval,
        contributions=y_eval,
        fisher_weights=torch.linspace(0.7, 1.7, 16),
        sequences=4,
    )

    fitted = fit_layer_cluster_modal_generator(
        fit,
        evaluation,
        selection=selected,
        source_model_sha256=catalog.model_fingerprint,
        parameter_catalog_sha256=catalog.artifact_sha256,
        fisher_coupling_sha256=fisher.artifact_sha256,
        fragment_plan=fragment_plan,
        fit_split_sha256=_sha("fit-tokenized"),
        eval_split_sha256=_sha("eval-tokenized"),
        input_site=selected.input_site,
        output_site=selected.output_site,
        mode_ranks=(1, 2),
        selected_mode_rank=2,
        generator_ranks=(1, 2),
        selected_generator_rank=2,
    )

    basis = fitted.computational_modes.selected_basis
    coordinate_plan = fitted.modal_generators.selected_plan
    assert basis is not None
    assert coordinate_plan is not None
    assert basis.binding.parameter_cluster_sha256 == selected.artifact_sha256
    assert coordinate_plan.binding.cluster_plan_sha256 == (
        fragment_plan.artifact_sha256
    )
    assert coordinate_plan.binding.output_catalog_sha256 == (
        basis.artifact_sha256
    )
    assert coordinate_plan.binding.target_kind == (
        "computational_mode_coordinates"
    )
    assert fitted.lowering.selected_fragment_sha256 == (
        selected.artifact_sha256
    )
    assert fitted.lowering.graph_weights.state_kind == (
        "computational_mode_coordinates"
    )
    assert fitted.lowering.fused_residual_plan.artifact_sha256 == (
        fitted.executable_plan.artifact_sha256
    )
    expected = basis.decode(coordinate_plan.apply(x_eval))
    actual = fitted.executable_plan.apply(x_eval)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    graph_execution = ModalGeneratorGraphExecutor(
        fitted.graph_plan
    ).execute(
        {selected.input_site: x_eval},
        capture_modal_states=True,
    )
    assert fitted.graph_plan.interactions == ()
    assert graph_execution.traversal_order == (
        fitted.graph_plan.traversal_order
    )
    assert graph_execution.modal_states is not None
    torch.testing.assert_close(
        graph_execution.modal_states[
            fitted.graph_plan.nodes[0].name
        ],
        coordinate_plan.apply(x_eval),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        graph_execution.outputs[selected.output_site],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert fitted.executable_plan.output_width == y_eval.shape[1]
    assert len(fitted.dense_generator_rate_curve) == 2


def _compiled_pilot():
    trace, catalog, fisher, clusters, fragment_plan, selected = (
        _discover_fragment()
    )
    generator = torch.Generator().manual_seed(9_033)
    x_fit = torch.randn(32, 3, generator=generator, dtype=torch.float64)
    x_eval = torch.randn(16, 3, generator=generator, dtype=torch.float64)
    coefficient = torch.tensor(
        [[1.0, 0.5, -0.25], [0.0, 1.0, 0.5], [0.5, -0.25, 1.0]],
        dtype=torch.float64,
    )
    offset = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    fitted = fit_layer_cluster_modal_generator(
        LayerFragmentRows(
            inputs=x_fit,
            contributions=x_fit @ coefficient + offset,
            fisher_weights=torch.linspace(0.5, 2.0, 32),
            sequences=8,
        ),
        LayerFragmentRows(
            inputs=x_eval,
            contributions=x_eval @ coefficient + offset,
            fisher_weights=torch.linspace(0.7, 1.7, 16),
            sequences=4,
        ),
        selection=selected,
        source_model_sha256=catalog.model_fingerprint,
        parameter_catalog_sha256=catalog.artifact_sha256,
        fisher_coupling_sha256=fisher.artifact_sha256,
        fragment_plan=fragment_plan,
        fit_split_sha256=trace.provenance.calibration_split_sha256,
        eval_split_sha256=_sha("eval-split"),
        input_site=selected.input_site,
        output_site=selected.output_site,
        mode_ranks=(1, 2),
        selected_mode_rank=2,
        generator_ranks=(1, 2),
        selected_generator_rank=2,
    )
    accounting, pipeline = build_single_node_modal_compiler_pipeline(
        fit_prompt_trace=trace,
        parameter_catalog=catalog,
        fisher_coupling=fisher,
        parameter_clusters=clusters,
        fragment_plan=fragment_plan,
        selection=selected,
        fitted=fitted,
    )
    eval_trace = collect_prompt_mode_trace(
        _trace_effect_rows(),
        layer_specs=_layer_specs(),
        provenance=PromptModeTraceProvenance(
            source_model_fingerprint=catalog.model_fingerprint,
            calibration_split_sha256=_sha("eval-split"),
            objective_sha256=_sha("objective"),
        ),
    )
    return (
        trace,
        eval_trace,
        catalog,
        fisher,
        clusters,
        fragment_plan,
        selected,
        fitted,
        accounting,
        pipeline,
    )


def test_single_node_pipeline_has_exact_source_and_graph_accounting() -> None:
    (
        _,
        _,
        _,
        _,
        _,
        _,
        selected,
        fitted,
        accounting,
        pipeline,
    ) = _compiled_pilot()

    assert pipeline.graph_plan.artifact_sha256 == (
        fitted.graph_plan.artifact_sha256
    )
    assert pipeline.graph_plan.traversal_order == (
        fitted.graph_plan.traversal_order
    )
    assert pipeline.graph_plan.interactions == ()
    assert pipeline.source_replacement_accounting is not None
    assert accounting.fragment_ids == (selected.fragment_id,)
    assert accounting.group_indices == selected.group_indices
    assert accounting.source_parameter_count == (
        selected.native_parameter_count
    )
    assert pipeline.graph_parameter_count == (
        fitted.graph_plan.parameter_count
    )
    assert pipeline.net_parameter_savings == (
        accounting.source_parameter_count
        - fitted.graph_plan.parameter_count
    )


class _FakeNativeModel(nn.Module):
    def forward(
        self,
        *,
        logits: Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> object:
        assert use_cache is False
        assert return_dict is True
        return SimpleNamespace(logits=logits)


class _FakeAdapter:
    def __init__(self) -> None:
        self.module = _FakeNativeModel()


class _FakeExecutor:
    def run(
        self,
        model_inputs: dict[str, Tensor],
        *,
        condition: str,
    ) -> object:
        native = model_inputs["logits"]
        generated = condition == "generated"
        if generated:
            candidate = native.clone()
        else:
            candidate = native.roll(shifts=1, dims=-1)
        executed_macs = 6 if generated else 0
        return SimpleNamespace(
            model_output=SimpleNamespace(logits=candidate),
            replacement_scope="partial_native_mlp_mode_replacement",
            replaced_layer_count=1,
            removed_mode_count=2,
            source_whole_model_learned_parameters=100,
            candidate_whole_model_learned_parameters=94,
            native_removed_learned_parameters=9,
            modal_generator_learned_parameters=3,
            net_stored_parameter_savings=6,
            valid_tokens=2,
            logical_linear_macs_native_removed=18,
            logical_modal_generator_macs=6,
            logical_executed_modal_generator_macs=executed_macs,
            logical_modal_generator_bias_additions=2,
            logical_executed_modal_generator_bias_additions=(
                2 if generated else 0
            ),
            net_logical_macs_saved=18 - executed_macs,
        )


class _FakeGraphExecutor:
    peak_live_modal_width = 2
    graph_plan = SimpleNamespace(
        nodes=(object(),),
        interactions=(),
        traversal_order=("node.0",),
    )

    def run(
        self,
        model_inputs: dict[str, Tensor],
        *,
        condition: str,
    ) -> object:
        native = model_inputs["logits"]
        generated = condition == "generated"
        candidate = native.clone() if generated else native.roll(
            shifts=1,
            dims=-1,
        )
        executed_macs = 6 if generated else 0
        return SimpleNamespace(
            model_output=SimpleNamespace(logits=candidate),
            graph_execution=SimpleNamespace(
                traversal_order=("node.0",) if generated else (),
            ),
            replacement_scope="partial_native_mlp_mode_replacement",
            replaced_layer_count=1,
            graph_node_count=1,
            fragment_count=1,
            removed_mode_count=2,
            source_whole_model_learned_parameters=100,
            candidate_whole_model_learned_parameters=94,
            native_removed_learned_parameters=9,
            modal_graph_learned_parameters=3,
            net_stored_parameter_savings=6,
            valid_tokens=2,
            logical_linear_macs_native_removed=18,
            logical_modal_graph_macs=6,
            logical_executed_modal_graph_macs=executed_macs,
            logical_modal_graph_additions=4,
            logical_executed_modal_graph_additions=(
                4 if generated else 0
            ),
            net_logical_macs_saved=18 - executed_macs,
            peak_live_modal_width=2 if generated else 0,
            graph_runtime_storage=(
                "registered_copied_device_local_graph_parameters"
            ),
        )


def test_fake_model_evaluation_reports_native_generated_and_deletion() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0, -1.0], [0.0, 4.0, -2.0]]],
        dtype=torch.float32,
    )
    batch = CalibrationBatch(
        model_inputs={"logits": logits},
        targets=torch.tensor([[0, 1]]),
        valid_positions=torch.ones((1, 2), dtype=torch.bool),
        example_ids=("eval.0",),
    )
    report = evaluate_modal_generator_conditions(
        _FakeAdapter(),  # type: ignore[arg-type]
        _FakeExecutor(),  # type: ignore[arg-type]
        (batch,),
    )

    generated = report["conditions"]["generated"]
    deletion = report["conditions"]["matched_deletion"]
    assert generated["delta_nll_per_token"] == pytest.approx(0.0)
    assert generated["native_to_candidate_kl_per_token"] == pytest.approx(0.0)
    assert generated["top1_agreement_to_native"] == 1.0
    assert deletion["top1_agreement_to_native"] == 0.0
    assert report["resource_accounting"]["net_stored_parameter_savings"] == 6
    assert report["resource_accounting"]["net_logical_macs_saved"] == 12
    assert report["resource_accounting"][
        "logical_executed_modal_generator_macs"
    ] == 6
    assert report["resource_accounting"]["matched_deletion"][
        "logical_modal_generator_macs"
    ] == 6
    assert report["resource_accounting"]["matched_deletion"][
        "logical_executed_modal_generator_macs"
    ] == 0
    assert report["resource_accounting"]["matched_deletion"][
        "net_logical_macs_saved"
    ] == 18


def test_fake_graph_evaluation_traverses_generated_and_suppresses_deletion() -> None:
    logits = torch.tensor(
        [[[5.0, 0.0, -1.0], [0.0, 4.0, -2.0]]],
        dtype=torch.float32,
    )
    batch = CalibrationBatch(
        model_inputs={"logits": logits},
        targets=torch.tensor([[0, 1]]),
        valid_positions=torch.ones((1, 2), dtype=torch.bool),
        example_ids=("eval.0",),
    )
    report = evaluate_modal_generator_graph_conditions(
        _FakeAdapter(),  # type: ignore[arg-type]
        _FakeGraphExecutor(),  # type: ignore[arg-type]
        (batch,),
    )

    assert report["execution_path"] == (
        "incremental_modal_generator_graph_traversal"
    )
    assert report["graph"]["traversal_order"] == ("node.0",)
    assert report["conditions"]["generated"][
        "top1_agreement_to_native"
    ] == 1.0
    assert report["conditions"]["deletion"][
        "top1_agreement_to_native"
    ] == 0.0
    resources = report["resource_accounting"]
    assert resources["modal_graph_learned_parameters"] == 3
    assert resources["generated"][
        "logical_executed_modal_graph_macs"
    ] == 6
    assert resources["deletion"][
        "logical_executed_modal_graph_macs"
    ] == 0
    assert resources["generated"]["net_logical_macs_saved"] == 12
    assert resources["deletion"]["net_logical_macs_saved"] == 18


def _strict_artifact_payload() -> dict[str, object]:
    (
        fit_trace,
        eval_trace,
        catalog,
        fisher,
        clusters,
        fragment_plan,
        selected,
        fitted,
        accounting,
        pipeline,
    ) = _compiled_pilot()
    graph_resources = {
        "native_removed_learned_parameters": (
            accounting.source_parameter_count
        ),
        "modal_graph_learned_parameters": pipeline.graph_parameter_count,
        "net_stored_parameter_savings": pipeline.net_parameter_savings,
    }
    without_digest: dict[str, object] = {
        "schema": GEMMA3_MODAL_GENERATOR_DEV_SCHEMA,
        "format_version": GEMMA3_MODAL_GENERATOR_DEV_FORMAT_VERSION,
        "scientific_status": {
            "development_export_provenance": "declared_self_attested",
            "external_export_authentication_claim": False,
            "numerical_extraction_provenance": (
                "caller_declared_self_attested"
            ),
            "split_membership_provenance": "caller_declared_self_attested",
            "numerical_extraction_externally_authenticated": False,
            "split_membership_externally_authenticated": False,
        },
        "model": {
            "adapter_model_fingerprint": catalog.model_fingerprint,
        },
        "protocol": {
            "recipe": (
                "weights",
                "fisher_coupling",
                "parameter_clusters",
                "computational_modes",
                "modal_generators",
                "graph_of_generator_interactions",
                "inference_by_graph_traversal",
            ),
            "primary_execution_path": (
                "incremental_modal_generator_graph_traversal"
            ),
            "graph_node_count": 1,
            "graph_interaction_count": 0,
            "graph_traversal_order": pipeline.graph_plan.traversal_order,
            "numerical_extraction_provenance": (
                "caller_declared_self_attested"
            ),
            "split_membership_provenance": "caller_declared_self_attested",
        },
        "splits": {
            "policy": {
                "export_provenance_assurance": "declared_self_attested",
                "export_provenance_externally_authenticated": False,
                "split_membership_provenance": (
                    "caller_declared_self_attested"
                ),
                "split_membership_externally_authenticated": False,
                "declared_membership_overlap_checked": True,
            },
            "fit_export": {
                "provenance_assurance": "declared_self_attested",
                "externally_authenticated": False,
            },
            "eval_export": {
                "provenance_assurance": "declared_self_attested",
                "externally_authenticated": False,
            },
            "fit_tokenized": {
                "content_sha256": (_sha("fit-token-content"),),
            },
            "eval_tokenized": {
                "content_sha256": (_sha("eval-token-content"),),
            },
            "tokenized_content_disjointness": {
                "fit_content_count": 1,
                "eval_content_count": 1,
                "overlap_count": 0,
                "passed": True,
            },
        },
        "fit_prompt_trace": fit_trace.state_dict(),
        "eval_prompt_trace": eval_trace.state_dict(),
        "parameter_catalog": catalog.state_dict(),
        "fisher_coupling": fisher.state_dict(),
        "parameter_clusters": clusters.state_dict(),
        "parameter_cluster_fragments": fragment_plan.state_dict(),
        "selected_layer_cluster": selected.metadata(),
        "computational_modes": fitted.computational_modes.state_dict(),
        "modal_generators": fitted.modal_generators.state_dict(),
        "modal_generator_lowering": fitted.lowering.state_dict(),
        "modal_generator_graph": fitted.graph_plan.state_dict(),
        "source_replacement_accounting": accounting.state_dict(),
        "modal_compiler_pipeline": pipeline.state_dict(),
        "dense_fused_executable_generator": (
            fitted.executable_plan.state_dict()
        ),
        "computational_mode_metadata": (
            fitted.computational_modes.metadata()
        ),
        "modal_generator_metadata": fitted.modal_generators.metadata(),
        "modal_generator_lowering_metadata": fitted.lowering.metadata(),
        "modal_generator_graph_metadata": {
            "artifact_sha256": fitted.graph_plan.artifact_sha256,
            "node_count": len(fitted.graph_plan.nodes),
            "interaction_count": len(fitted.graph_plan.interactions),
            "traversal_order": fitted.graph_plan.traversal_order,
            "parameter_count": fitted.graph_plan.parameter_count,
            "macs_per_token": fitted.graph_plan.macs_per_token,
        },
        "source_replacement_accounting_metadata": {
            "artifact_sha256": accounting.artifact_sha256,
            "fragment_ids": accounting.fragment_ids,
            "group_indices": accounting.group_indices,
            "source_parameter_count": accounting.source_parameter_count,
            "source_macs_per_token": accounting.source_macs_per_token,
        },
        "modal_compiler_pipeline_metadata": pipeline.metadata(),
        "dense_generator_rate_curve": fitted.dense_generator_rate_curve,
        "evaluation": {
            "primary_graph_traversal": {
                "execution_path": (
                    "incremental_modal_generator_graph_traversal"
                ),
                "resource_accounting": graph_resources,
            },
            "dense_fused_optimization_comparison": {
                "execution_path": (
                    "dense_fused_generator_optimization_comparison"
                ),
            },
            "resource_accounting_paths_are_separate": True,
        },
        "contains_source_model_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_token_rows": False,
        "contains_tokenizer_state": False,
        "contains_generator_weights": True,
    }
    return {
        **without_digest,
        "scientific_payload_sha256": _payload_sha256(without_digest),
    }


def test_strict_loader_validates_graph_and_compiler_pipeline(
    tmp_path: Path,
) -> None:
    payload = _strict_artifact_payload()
    path = tmp_path / "graph-pilot.pt"
    torch.save(payload, path)
    restored = load_gemma3_modal_generator_dev_artifact(path)

    assert restored["modal_generator_graph"]["artifact_sha256"] == (
        payload["modal_generator_graph"]["artifact_sha256"]
    )
    assert restored["modal_compiler_pipeline"]["artifact_sha256"] == (
        payload["modal_compiler_pipeline"]["artifact_sha256"]
    )

    poisoned = copy.deepcopy(payload)
    poisoned["modal_compiler_pipeline"]["graph_plan"]["nodes"][0][
        "weights"
    ]["input_factor"][0, 0] += 1.0
    without_digest = {
        key: value
        for key, value in poisoned.items()
        if key != "scientific_payload_sha256"
    }
    poisoned["scientific_payload_sha256"] = _payload_sha256(without_digest)
    poisoned_path = tmp_path / "poisoned-graph-pilot.pt"
    torch.save(poisoned, poisoned_path)
    with pytest.raises(ValueError, match="input_factor hash mismatch"):
        load_gemma3_modal_generator_dev_artifact(poisoned_path)

    unexpected = copy.deepcopy(payload)
    unexpected["unexpected"] = False
    without_digest = {
        key: value
        for key, value in unexpected.items()
        if key != "scientific_payload_sha256"
    }
    unexpected["scientific_payload_sha256"] = _payload_sha256(
        without_digest
    )
    unexpected_path = tmp_path / "unexpected-field.pt"
    torch.save(unexpected, unexpected_path)
    with pytest.raises(ValueError, match="top-level fields"):
        load_gemma3_modal_generator_dev_artifact(unexpected_path)

    unsafe = copy.deepcopy(payload)
    unsafe["model"]["input_ids"] = (1, 2, 3)
    without_digest = {
        key: value
        for key, value in unsafe.items()
        if key != "scientific_payload_sha256"
    }
    unsafe["scientific_payload_sha256"] = _payload_sha256(without_digest)
    unsafe_path = tmp_path / "unsafe.pt"
    torch.save(unsafe, unsafe_path)
    with pytest.raises(ValueError, match="forbidden field"):
        load_gemma3_modal_generator_dev_artifact(unsafe_path)

    overlapping = copy.deepcopy(payload)
    overlapping["splits"]["eval_tokenized"]["content_sha256"] = (
        overlapping["splits"]["fit_tokenized"]["content_sha256"]
    )
    without_digest = {
        key: value
        for key, value in overlapping.items()
        if key != "scientific_payload_sha256"
    }
    overlapping["scientific_payload_sha256"] = _payload_sha256(
        without_digest
    )
    overlapping_path = tmp_path / "overlapping-content.pt"
    torch.save(overlapping, overlapping_path)
    with pytest.raises(ValueError, match="tokenized content overlaps"):
        load_gemma3_modal_generator_dev_artifact(overlapping_path)


def test_strict_loader_rejects_nested_freeform_text_after_rehash(
    tmp_path: Path,
) -> None:
    payload = _strict_artifact_payload()
    payload["scientific_status"]["audit"] = {
        "note": "Trace the hidden modal generator for this prompt."
    }
    without_digest = {
        key: value
        for key, value in payload.items()
        if key != "scientific_payload_sha256"
    }
    payload["scientific_payload_sha256"] = _payload_sha256(without_digest)
    path = tmp_path / "nested-freeform-text.pt"
    torch.save(payload, path)

    with pytest.raises(
        ValueError,
        match=r"non-machine string at scientific_status\.audit\.note",
    ):
        load_gemma3_modal_generator_dev_artifact(path)

    shadowed = _strict_artifact_payload()
    shadowed["computational_mode_metadata"]["binding"][
        "mode_set_id"
    ] = "hiddenprompt"
    without_digest = {
        key: value
        for key, value in shadowed.items()
        if key != "scientific_payload_sha256"
    }
    shadowed["scientific_payload_sha256"] = _payload_sha256(without_digest)
    shadowed_path = tmp_path / "shadowed-metadata.pt"
    torch.save(shadowed, shadowed_path)
    with pytest.raises(
        ValueError,
        match="saved computational mode metadata is inconsistent",
    ):
        load_gemma3_modal_generator_dev_artifact(shadowed_path)


def test_artifact_privacy_guard_and_cli_defaults() -> None:
    _assert_source_safe_artifact(
        {
            "contains_prompt_text": False,
            "prompt_sha256s": (_sha("secret prompt"),),
            "generator": {"weights_sha256": _sha("weights")},
        },
        prompt_texts=frozenset({"secret prompt"}),
    )
    with pytest.raises(RuntimeError, match="raw prompt text"):
        _assert_source_safe_artifact(
            {"label": "secret prompt"},
            prompt_texts=frozenset({"secret prompt"}),
        )
    with pytest.raises(RuntimeError, match="forbidden field"):
        _assert_source_safe_artifact(
            {"input_ids": torch.tensor([1, 2])},
            prompt_texts=frozenset(),
        )

    arguments = build_parser().parse_args(["--revision", "a" * 40])
    assert arguments.fit_export == DEFAULT_FIT_EXPORT
    assert arguments.eval_export == DEFAULT_EVAL_EXPORT
    assert arguments.output == DEFAULT_OUTPUT
    assert arguments.device == "cpu"
