from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.adapters import (
    Gemma3CausalLMAdapter,
    StructuredOperatorSites,
    module_state_fingerprint,
)
from fisher_graph.compiler import CalibrationBatch
from fisher_graph.structured_mlp_compression import (
    GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH,
    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
    StructuredMLPFisherTaylorBatch,
    select_fisher_taylor_mlp_units,
)
from fisher_graph.structured_mlp_compression_pipeline import (
    build_gemma_mlp_first_rung_candidate,
    collect_gemma_mlp_fisher_taylor_batches,
    refit_structured_mlp_down_projection_from_targets_,
)
from fisher_graph.structured_operator_bootstrap import (
    STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
    STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION,
    STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA,
    structured_operator_coefficient_sha256,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
)

from test_structured_mlp_compression import (
    _executor,
    _provenance,
    _score_batch,
    _targets,
)
from test_structured_operator_bootstrap import _FakeCausalLM


def _calibration_batch() -> CalibrationBatch:
    valid = torch.tensor(
        [
            [False, False, True, True],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )
    targets = torch.tensor(
        [
            [-100, -100, 5, 6],
            [-100, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    return CalibrationBatch(
        model_inputs={
            "input_ids": torch.tensor(
                [
                    [0, 0, 2, 3],
                    [0, 4, 5, 6],
                ],
                dtype=torch.long,
            ),
            "attention_mask": valid.clone(),
        },
        targets=targets,
        valid_positions=valid,
        example_ids=("example-alpha", "example-beta"),
    )


def test_collector_captures_suffix_nll_gradient_and_excludes_padding() -> None:
    torch.manual_seed(90_101)
    model = _FakeCausalLM().eval().requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    batch = _calibration_batch()
    before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    before_fingerprint = module_state_fingerprint(model)

    score_batches, report = collect_gemma_mlp_fisher_taylor_batches(
        adapter,
        (batch,),
        layer_id="layer.0",
        calibration_split_sha256="a" * 64,
    )

    assert len(score_batches) == 1
    score_batch = score_batches[0]
    assert score_batch.provenance.layer_id == "layer.0"
    assert score_batch.batch_id.startswith("calibration_a:000000:")
    assert score_batch.valid_mask.device.type == "cpu"
    assert torch.equal(score_batch.valid_mask, batch.valid_positions)
    assert score_batch.valid_rows == 5
    assert score_batch.projection_input.shape == (2, 4, 6)
    assert score_batch.score_gradient.shape == (2, 4, 6)
    assert bool(
        torch.isfinite(
            score_batch.score_gradient[score_batch.valid_mask]
        ).all()
    )
    assert float(
        score_batch.score_gradient[score_batch.valid_mask]
        .abs()
        .sum()
        .item()
    ) > 0
    assert report["accounting"] == {
        "batch_count": 1,
        "valid_rows": 5,
        "padding_rows": 3,
        "supervised_tokens": 5,
        "padding_rows_excluded_from_scoring": True,
    }
    assert report["batches"][0]["example_ids"] == (
        "example-alpha",
        "example-beta",
    )
    assert report["batches"][0]["batch_id"] == score_batch.batch_id
    assert report["source_audit"]["source_parameter_gradients_observed"] is False
    assert report["source_audit"]["source_state_sha256_before"] == (
        before_fingerprint
    )
    assert report["source_audit"]["source_state_sha256_after"] == (
        before_fingerprint
    )
    assert report["heldout_opened"] is False
    assert all(parameter.grad is None for parameter in model.parameters())
    assert module_state_fingerprint(model) == before_fingerprint
    for name, expected in before.items():
        torch.testing.assert_close(model.state_dict()[name], expected)

    poisoned = replace(
        score_batch,
        projection_input=score_batch.projection_input.clone(),
        score_gradient=score_batch.score_gradient.clone(),
    )
    poisoned.projection_input[~poisoned.valid_mask] = torch.nan
    poisoned.score_gradient[~poisoned.valid_mask] = torch.nan
    unpoisoned_selection = select_fisher_taylor_mlp_units(
        score_batches,
        calibration_split_sha256="a" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="b" * 64,
        retained_width=3,
        expected_source_width=6,
    )
    poisoned_selection = select_fisher_taylor_mlp_units(
        (poisoned,),
        calibration_split_sha256="a" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="b" * 64,
        retained_width=3,
        expected_source_width=6,
    )
    assert (
        poisoned_selection.selection_sha256
        == unpoisoned_selection.selection_sha256
    )


def _format5_binding(
    executor: StructuredTransformerLayerExecutor,
) -> dict[str, object]:
    provenance = _provenance()
    return {
        "fitting_method": "activation_only_structured_operator_bootstrap",
        "optimizer": "none",
        "optimizer_steps": 0,
        "suffix_training_steps": 0,
        "final_execution_fingerprint": executor.execution_fingerprint(),
        "bootstrap": {
            "schema": STRUCTURED_OPERATOR_BOOTSTRAP_SCHEMA,
            "format_version": (
                STRUCTURED_OPERATOR_BOOTSTRAP_FORMAT_VERSION
            ),
            "algorithm": STRUCTURED_OPERATOR_BOOTSTRAP_ALGORITHM,
            "layer_id": provenance.layer_id,
            "calibration_split_sha256": "c" * 64,
            "source_segment_fingerprint": (
                provenance.source_segment_fingerprint
            ),
            "coefficient_sha256": (
                structured_operator_coefficient_sha256(executor)
            ),
            "source_module_or_parameter_read": False,
            "direct_source_tensor_copy": False,
            "destination_source_weight_contamination": False,
            "destination_executor_local_source_free": True,
        },
    }


def _gemma_schema_executor() -> StructuredTransformerLayerExecutor:
    source = _executor(
        GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
        projection_bias=False,
    )
    transformer = source.config.transformer
    layer_id = "layer.0"
    operator_sites = StructuredOperatorSites(
        attention_query_projection=(
            f"{layer_id}.attention.query_projection"
        ),
        attention_query_normalized=(
            f"{layer_id}.attention.query_normalized"
        ),
        attention_key_projection=(
            f"{layer_id}.attention.key_projection"
        ),
        attention_key_normalized=(
            f"{layer_id}.attention.key_normalized"
        ),
        attention_value_projection=(
            f"{layer_id}.attention.value_projection"
        ),
        attention_context=f"{layer_id}.attention.context",
        feed_forward_gate_projection=(
            f"{layer_id}.mlp.gate_projection"
        ),
        feed_forward_up_projection=f"{layer_id}.mlp.up_projection",
        feed_forward_down_input=f"{layer_id}.mlp.down_input",
    )
    config = replace(
        source.config,
        transformer=replace(
            transformer,
            operator_sites=operator_sites,
        ),
    )
    result = StructuredTransformerLayerExecutor(config)
    result.load_state_dict(source.state_dict(), strict=True)
    return result.eval()


def test_down_only_refit_preserves_attention_exactly() -> None:
    teacher = _executor(6, projection_bias=False, seed=401)
    candidate = _executor(6, projection_bias=False, seed=402)
    provenance = _provenance()
    targets = (
        _targets(
            teacher,
            torch.randn(3, 4, teacher.width),
            torch.tensor(
                [
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, False, False],
                ]
            ),
            provenance,
        ),
    )
    attention_before = {
        name: value.detach().clone()
        for name, value in candidate.state_dict().items()
        if name.startswith("attention.")
        or name.startswith("attention_input_norm.")
        or name.startswith("attention_output_norm.")
    }

    report = refit_structured_mlp_down_projection_from_targets_(
        candidate,
        targets,
        calibration_split_sha256="c" * 64,
        ridge=1e-5,
    )

    assert report["attention_refit"] is False
    assert report["attention_tensors_preserved"]
    assert report["only_feed_forward_down_projection_written"]
    assert (
        report["projection"]["post_refit_operator_nrmse"]
        < report["projection"]["pre_refit_operator_nrmse"]
    )
    for name, expected in attention_before.items():
        torch.testing.assert_close(
            candidate.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )


def test_pipeline_builds_strict_first_rung_candidate_without_opening_b() -> None:
    parent = _gemma_schema_executor()
    provenance = _provenance()
    hidden = torch.randn(1, 3, parent.width)
    mask = torch.tensor([[True, True, False]])
    targets = (_targets(parent, hidden, mask, provenance),)
    score_batches = (
        _score_batch(
            batch_id="calibration-a-score",
            activations=torch.ones(
                1,
                2,
                GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
            ),
            gradients=torch.arange(
                1,
                GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH + 1,
                dtype=torch.float32,
            ).view(1, 1, -1).expand(1, 2, -1),
            mask=torch.tensor([[True, False]]),
            provenance=provenance,
        ),
    )
    parent_fingerprint = parent.execution_fingerprint()
    candidate = build_gemma_mlp_first_rung_candidate(
        parent,
        targets,
        score_batches,
        calibration_split_sha256="c" * 64,
        parent_artifact_format_version=5,
        parent_training_binding=_format5_binding(parent),
        ridge=1e-5,
    )

    assert (
        candidate.executor.config.transformer.feed_forward.intermediate_width
        == GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH
    )
    assert not candidate.executor.owns_source_model_weights
    assert parent.execution_fingerprint() == parent_fingerprint
    report = candidate.report
    assert report["status"] == {
        "candidate_built": True,
        "ready_for_existing_heldout_evaluator": True,
        "heldout_opened": False,
        "scientific_success_claimed": False,
        "compression_candidate_only": True,
    }
    assert report["parent_authentication"][
        "strict_executor_state_roundtrip_verified"
    ]
    assert report["refit_targets"]["padding_rows_excluded_from_digest"]
    assert (
        report["terminal_projection_refit"]["algorithm"]
        == "activation_only_mlp_down_projection_ridge_v1"
    )
    assert report["terminal_projection_refit"]["projection"][
        "input_width"
    ] == GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH
    assert report["final_candidate"]["strict_state_roundtrip_verified"]
    assert report["final_candidate"]["parent_executor_unchanged"]
    assert report["final_candidate"]["attention_refit"] is False
    assert report["final_candidate"]["attention_tensors_preserved"]
    assert (
        report["final_candidate"]["attention_state_sha256_parent"]
        == report["final_candidate"]["attention_state_sha256_final"]
    )
    assert report["data_policy"][
        "must_reuse_parent_format5_calibration_a"
    ]
    assert report["data_policy"][
        "fresh_heldout_reserved_for_post_build_evaluation"
    ]
    restored = StructuredTransformerLayerExecutor.from_artifact_state_dict(
        candidate.artifact_state
    )
    assert (
        restored.execution_fingerprint()
        == candidate.executor.execution_fingerprint()
    )


def test_pipeline_rejects_unbound_or_non_format5_parent() -> None:
    parent = _gemma_schema_executor()
    provenance = _provenance()
    targets = (
        _targets(
            parent,
            torch.randn(1, 1, parent.width),
            torch.ones(1, 1, dtype=torch.bool),
            provenance,
        ),
    )
    score_batches: tuple[StructuredMLPFisherTaylorBatch, ...] = (
        _score_batch(
            batch_id="score",
            activations=torch.ones(
                1,
                1,
                GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
            ),
            gradients=torch.ones(
                1,
                1,
                GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
            ),
            mask=torch.ones(1, 1, dtype=torch.bool),
            provenance=provenance,
        ),
    )
    with pytest.raises(ValueError, match="format-5"):
        build_gemma_mlp_first_rung_candidate(
            parent,
            targets,
            score_batches,
            calibration_split_sha256="c" * 64,
            parent_artifact_format_version=4,
            parent_training_binding=_format5_binding(parent),
        )
    tampered = _format5_binding(parent)
    tampered["final_execution_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="not authenticated"):
        build_gemma_mlp_first_rung_candidate(
            parent,
            targets,
            score_batches,
            calibration_split_sha256="c" * 64,
            parent_artifact_format_version=5,
            parent_training_binding=tampered,
        )
