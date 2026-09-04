import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
)
from fisher_graph.gemma3_attention_transport_full_model_fit import (
    FullModelCrossLayerFitConfig,
    evaluate_full_model_cross_layer_fisher,
    fit_full_model_cross_layer_attention_transport,
)
from fisher_graph.gemma3_attention_transport_full_model_partition import (
    ENDPOINT_TRAIN_INVOCATION_IDS,
    MIDDLE_HOLDOUT_INVOCATION_IDS,
)


def _capture(invocation_id: str, seed: int) -> AttentionTransportRowBatch:
    generator = torch.Generator().manual_seed(seed)
    rows = 3
    heads = 2
    hidden = 8
    contexts = torch.randn(
        rows, heads, hidden, dtype=torch.float64, generator=generator
    )
    operators = torch.randn(
        heads, hidden, hidden, dtype=torch.float64, generator=generator
    )
    outputs = torch.einsum("aoh,nah->no", operators, contexts)
    return AttentionTransportRowBatch(
        invocation_id=invocation_id,
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.randn(
            rows, hidden, dtype=torch.float64, generator=generator
        ),
        chart_inputs=torch.randn(
            rows, hidden, dtype=torch.float64, generator=generator
        ),
        head_operators=operators,
        native_v_o_parameter_count=2 * hidden * hidden,
    )


def _endpoint_captures() -> tuple[AttentionTransportRowBatch, ...]:
    return tuple(
        _capture(invocation_id, 100 + index)
        for index, invocation_id in enumerate(ENDPOINT_TRAIN_INVOCATION_IDS)
    )


def test_endpoint_only_fit_builds_shared_artifact_and_evaluates() -> None:
    result = fit_full_model_cross_layer_attention_transport(
        _endpoint_captures(),
        config=FullModelCrossLayerFitConfig(
            dictionary_steps=0,
            structural_rounds=1,
            structural_epochs_per_round=1,
            router_steps=1,
        ),
    )

    assert result.fit_capture_count == 12
    assert result.fit_row_count == 36
    assert result.endpoint_invocation_ids == ENDPOINT_TRAIN_INVOCATION_IDS
    assert result.artifact.parameter_accounting().free_layer_id_parameters == 0
    assert result.endpoint_evaluation.row_count == 36
    assert (
        result.endpoint_evaluation.deployable_relative_fisher_loss
        >= result.endpoint_evaluation.oracle_relative_fisher_loss
    )


def test_held_evaluation_is_separate_from_fit() -> None:
    result = fit_full_model_cross_layer_attention_transport(
        _endpoint_captures(),
        config=FullModelCrossLayerFitConfig(
            dictionary_steps=0,
            structural_rounds=1,
            structural_epochs_per_round=1,
            router_steps=1,
        ),
    )
    held = tuple(
        _capture(invocation_id, 400 + index)
        for index, invocation_id in enumerate(MIDDLE_HOLDOUT_INVOCATION_IDS)
    )

    before = result.artifact.artifact_sha256
    evaluation = evaluate_full_model_cross_layer_fisher(result.artifact, held)

    assert evaluation.row_count == 18
    assert result.artifact.artifact_sha256 == before
    assert tuple(key for key, _ in evaluation.per_invocation_deployable_relative_fisher_loss) == MIDDLE_HOLDOUT_INVOCATION_IDS


def test_fit_rejects_held_middle_teacher_rows() -> None:
    contaminated = (*_endpoint_captures(), _capture("mutation.13", 999))
    try:
        fit_full_model_cross_layer_attention_transport(
            contaminated,
            config=FullModelCrossLayerFitConfig(
                dictionary_steps=0,
                structural_rounds=1,
                structural_epochs_per_round=1,
                router_steps=1,
            ),
        )
    except ValueError as error:
        assert "endpoint" in str(error)
    else:
        raise AssertionError("held middle teacher rows entered the fitter")
