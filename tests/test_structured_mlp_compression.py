from dataclasses import replace

import pytest
import torch
from torch import nn

from fisher_graph.structured_layer_distillation import (
    StructuredLayerProvenance,
    StructuredLayerTargets,
    refit_structured_terminal_projections_from_targets_,
)
from fisher_graph.structured_mlp_compression import (
    GEMMA_MLP_FIRST_RUNG_RETAINED_WIDTH,
    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
    StructuredMLPFisherTaylorBatch,
    StructuredMLPUnitSelection,
    build_width_compressed_structured_executor,
    prepare_width_compressed_mlp_refit_targets,
    select_fisher_taylor_mlp_units,
    select_gemma_mlp_first_rung_units,
)
from fisher_graph.structured_transformer_layer_executor import (
    StructuredTransformerLayerExecutor,
    StructuredTransformerLayerExecutorConfig,
)

from test_structured_transformer_layer_executor import (
    _layer_spec,
    _sequence,
)


def _executor(
    intermediate_width: int,
    *,
    projection_bias: bool,
    seed: int = 81_993,
) -> StructuredTransformerLayerExecutor:
    layer = _layer_spec(
        attention_kind="global_causal",
        window_size=None,
    )
    assert layer.transformer is not None
    transformer = replace(
        layer.transformer,
        feed_forward=replace(
            layer.transformer.feed_forward,
            intermediate_width=intermediate_width,
            projection_bias=projection_bias,
        ),
    )
    torch.manual_seed(seed)
    return StructuredTransformerLayerExecutor(
        StructuredTransformerLayerExecutorConfig.from_layer_spec(
            replace(layer, transformer=transformer)
        )
    ).eval()


def _provenance(digit: str = "a") -> StructuredLayerProvenance:
    return StructuredLayerProvenance(
        layer_id="layer.0",
        output_site="layer.0.output",
        source_segment_fingerprint=digit * 64,
    )


def _score_batch(
    *,
    batch_id: str,
    activations: torch.Tensor,
    gradients: torch.Tensor,
    mask: torch.Tensor,
    provenance: StructuredLayerProvenance | None = None,
) -> StructuredMLPFisherTaylorBatch:
    return StructuredMLPFisherTaylorBatch(
        provenance=_provenance() if provenance is None else provenance,
        batch_id=batch_id,
        projection_input=activations,
        score_gradient=gradients,
        valid_mask=mask,
    )


def _selection_for_executor(
    executor: StructuredTransformerLayerExecutor,
    *,
    retained_width: int,
    provenance: StructuredLayerProvenance | None = None,
) -> StructuredMLPUnitSelection:
    width = executor.config.transformer.feed_forward.intermediate_width
    activations = torch.ones(1, 1, width)
    gradients = torch.arange(
        1,
        width + 1,
        dtype=torch.float32,
    ).view(1, 1, width)
    return select_fisher_taylor_mlp_units(
        (
            _score_batch(
                batch_id="selection",
                activations=activations,
                gradients=gradients,
                mask=torch.ones(1, 1, dtype=torch.bool),
                provenance=provenance,
            ),
        ),
        calibration_split_sha256="b" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=executor.execution_fingerprint(),
        retained_width=retained_width,
        expected_source_width=width,
    )


def _targets(
    executor: StructuredTransformerLayerExecutor,
    hidden: torch.Tensor,
    mask: torch.Tensor,
    provenance: StructuredLayerProvenance,
) -> StructuredLayerTargets:
    sequence = _sequence(mask)
    with torch.no_grad():
        result = executor.forward_components(hidden, sequence)
    return StructuredLayerTargets(
        provenance=provenance,
        sequence=sequence,
        block_input=hidden.detach().clone(),
        normalized_attention_input=(
            result.normalized_attention_input.detach().clone()
        ),
        attention_operator_output=(
            result.attention_operator_output.detach().clone()
        ),
        attention_delta=result.attention_delta.detach().clone(),
        post_attention=result.post_attention.detach().clone(),
        normalized_feed_forward_input=(
            result.normalized_feed_forward_input.detach().clone()
        ),
        feed_forward_operator_output=(
            result.feed_forward_operator_output.detach().clone()
        ),
        feed_forward_delta=result.feed_forward_delta.detach().clone(),
        output=result.output.detach().clone(),
        attention_projection_input=(
            result.attention_projection_input.detach().clone()
        ),
        feed_forward_projection_input=(
            result.feed_forward_projection_input.detach().clone()
        ),
    )


def test_fisher_taylor_selection_is_stable_tied_and_padding_safe() -> None:
    provenance = _provenance()
    mask_a = torch.tensor([[True, True, False]])
    mask_b = torch.tensor([[True, True, True]])
    activation_a = torch.ones(1, 3, 6)
    gradient_a = (
        torch.tensor([3.0, 3.0, 2.0, 1.0, 0.5, 0.0])
        .view(1, 1, 6)
        .expand(1, 3, 6)
        .clone()
    )
    activation_a[:, 2] = torch.nan
    gradient_a[:, 2] = torch.nan
    activation_b = torch.ones(1, 3, 6)
    gradient_b = (
        torch.tensor([3.0, 3.0, 2.0, 1.0, 0.5, 0.0])
        .view(1, 1, 6)
        .expand(1, 3, 6)
        .clone()
    )
    batches = (
        _score_batch(
            batch_id="batch-b",
            activations=activation_b,
            gradients=gradient_b,
            mask=mask_b,
            provenance=provenance,
        ),
        _score_batch(
            batch_id="batch-a",
            activations=activation_a,
            gradients=gradient_a,
            mask=mask_a,
            provenance=provenance,
        ),
    )

    first = select_fisher_taylor_mlp_units(
        batches,
        calibration_split_sha256="c" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="d" * 64,
        retained_width=3,
        expected_source_width=6,
    )
    repeated = select_fisher_taylor_mlp_units(
        tuple(reversed(batches)),
        calibration_split_sha256="c" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="d" * 64,
        retained_width=3,
        expected_source_width=6,
    )

    assert first.ranked_indices == (0, 1, 2, 3, 4, 5)
    assert first.selected_indices == (0, 1, 2)
    assert first.valid_rows == 5
    assert first.selection_sha256 == repeated.selection_sha256
    assert first.metadata() == repeated.metadata()
    torch.testing.assert_close(
        first.unit_scores,
        torch.tensor(
            [9.0, 9.0, 4.0, 1.0, 0.25, 0.0],
            dtype=torch.float64,
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="score accumulation is invalid",
    ):
        select_fisher_taylor_mlp_units(
            (
                _score_batch(
                    batch_id="zero-signal",
                    activations=torch.ones(1, 2, 6),
                    gradients=torch.zeros(1, 2, 6),
                    mask=torch.ones(1, 2, dtype=torch.bool),
                    provenance=provenance,
                ),
            ),
            calibration_split_sha256="c" * 64,
            activation_site="layer.0.mlp.down_input",
            parent_executor_fingerprint="d" * 64,
            retained_width=3,
        )


def test_gemma_first_rung_slices_paired_units_and_accounts_exactly() -> None:
    source = _executor(
        GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
        projection_bias=True,
    )
    selection = select_gemma_mlp_first_rung_units(
        (
            _score_batch(
                batch_id="gemma-first-rung",
                activations=torch.ones(
                    1,
                    1,
                    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
                ),
                gradients=torch.arange(
                    1,
                    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH + 1,
                    dtype=torch.float32,
                ).view(
                    1,
                    1,
                    GEMMA_MLP_FIRST_RUNG_SOURCE_WIDTH,
                ),
                mask=torch.ones(1, 1, dtype=torch.bool),
            ),
        ),
        calibration_split_sha256="e" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=source.execution_fingerprint(),
    )
    rng_before = torch.random.get_rng_state().clone()
    compressed, report = build_width_compressed_structured_executor(
        source,
        selection,
    )
    rng_after = torch.random.get_rng_state()
    repeated, repeated_report = (
        build_width_compressed_structured_executor(
            source,
            selection,
        )
    )

    assert torch.equal(rng_after, rng_before)
    assert selection.source_width == 2_048
    assert selection.retained_width == 1_536
    assert selection.selected_indices == tuple(range(512, 2_048))
    assert (
        compressed.config.transformer.feed_forward.intermediate_width
        == 1_536
    )
    indices = torch.tensor(selection.selected_indices)
    torch.testing.assert_close(
        compressed.feed_forward.gate_proj.weight,
        source.feed_forward.gate_proj.weight.index_select(0, indices),
    )
    torch.testing.assert_close(
        compressed.feed_forward.up_proj.weight,
        source.feed_forward.up_proj.weight.index_select(0, indices),
    )
    torch.testing.assert_close(
        compressed.feed_forward.down_proj.weight,
        source.feed_forward.down_proj.weight.index_select(1, indices),
    )
    torch.testing.assert_close(
        compressed.feed_forward.gate_proj.bias,
        source.feed_forward.gate_proj.bias.index_select(0, indices),
    )
    torch.testing.assert_close(
        compressed.feed_forward.up_proj.bias,
        source.feed_forward.up_proj.bias.index_select(0, indices),
    )
    torch.testing.assert_close(
        compressed.feed_forward.down_proj.bias,
        source.feed_forward.down_proj.bias,
    )
    source_state = source.state_dict()
    compressed_state = compressed.state_dict()
    sliced = {
        "feed_forward.gate_proj.weight",
        "feed_forward.gate_proj.bias",
        "feed_forward.up_proj.weight",
        "feed_forward.up_proj.bias",
        "feed_forward.down_proj.weight",
    }
    for name in source_state:
        if name not in sliced:
            torch.testing.assert_close(
                compressed_state[name],
                source_state[name],
                rtol=0.0,
                atol=0.0,
            )
    expected_removed_parameters = 512 * (3 * source.width + 2)
    assert report["parameters"]["removed_full_layer"] == (
        expected_removed_parameters
    )
    assert report["parameters"][
        "expected_removed_from_mlp_slices"
    ] == expected_removed_parameters
    assert report["compute_per_valid_token"]["macs"] == {
        "source": 3 * source.width * 2_048,
        "compressed": 3 * source.width * 1_536,
        "removed": 3 * source.width * 512,
    }
    assert report["compute_per_valid_token"]["flops_two_per_mac"] == {
        "source": 2 * 3 * source.width * 2_048,
        "compressed": 2 * 3 * source.width * 1_536,
        "removed": 2 * 3 * source.width * 512,
    }
    assert (
        report["pairing"]["gate_rows"]
        == report["pairing"]["up_rows"]
    )
    assert (
        report["pairing"]["up_rows"]
        == report["pairing"]["down_columns"]
    )
    assert report["preservation"]["attention_preserved"] is True
    assert report["preservation"]["normalizations_preserved"] is True
    assert (
        report["preservation"]["projection_bias_schema_preserved"]
        is True
    )
    assert (
        report["preservation"]["native_source_parameter_read"]
        is False
    )
    assert report == repeated_report
    for name, value in compressed.state_dict().items():
        torch.testing.assert_close(
            repeated.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    assert not compressed.owns_source_model_weights


def test_compressed_refit_inputs_integrate_with_terminal_ridge() -> None:
    source = _executor(8, projection_bias=False)
    provenance = _provenance("f")
    generator = torch.Generator().manual_seed(19_119)
    targets = tuple(
        _targets(
            source,
            torch.randn(4, 6, source.width, generator=generator),
            torch.ones(4, 6, dtype=torch.bool),
            provenance,
        )
        for _ in range(3)
    )
    score_batches = tuple(
        StructuredMLPFisherTaylorBatch.from_structured_targets(
            target,
            torch.tensor(
                [8.0, 7.0, 6.0, 5.0, 4.0, 1.0, 0.5, 0.25]
            ).view(1, 1, 8).expand_as(
                target.feed_forward_projection_input
            ),
            batch_id=f"batch-{index}",
        )
        for index, target in enumerate(targets)
    )
    selection = select_fisher_taylor_mlp_units(
        score_batches,
        calibration_split_sha256="1" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=source.execution_fingerprint(),
        retained_width=5,
        expected_source_width=8,
    )
    compressed, _build_report = (
        build_width_compressed_structured_executor(
            source,
            selection,
        )
    )
    refit_targets, input_report = (
        prepare_width_compressed_mlp_refit_targets(
            targets,
            selection,
        )
    )

    assert all(
        target.feed_forward_projection_input.shape[-1] == 5
        for target in refit_targets
    )
    assert input_report["valid_rows"] == 72
    fit_report = refit_structured_terminal_projections_from_targets_(
        compressed,
        refit_targets,
        calibration_split_sha256="1" * 64,
        ridge=1e-8,
    )
    down = fit_report["projections"]["feed_forward.down_proj"]
    assert (
        down["post_refit_operator_nrmse"]
        < down["pre_refit_operator_nrmse"]
    )
    assert fit_report["valid_rows"] == input_report["valid_rows"]


def test_compression_fails_closed_on_parent_schema_and_provenance_drift() -> None:
    source = _executor(8, projection_bias=False)
    selection = _selection_for_executor(
        source,
        retained_width=5,
    )
    with torch.no_grad():
        source.feed_forward.gate_proj.weight[0, 0].add_(1.0)
    with pytest.raises(
        ValueError,
        match="parent executor fingerprint",
    ):
        build_width_compressed_structured_executor(
            source,
            selection,
        )

    drifted = _executor(8, projection_bias=False)
    drifted.feed_forward.gate_proj = nn.Linear(
        drifted.width,
        7,
        bias=False,
    )
    drifted_selection = _selection_for_executor(
        drifted,
        retained_width=5,
    )
    with pytest.raises(
        ValueError,
        match="parameter schema drifted",
    ):
        build_width_compressed_structured_executor(
            drifted,
            drifted_selection,
        )

    clean = _executor(8, projection_bias=False)
    clean_selection = _selection_for_executor(
        clean,
        retained_width=5,
    )
    target = _targets(
        clean,
        torch.randn(2, 4, clean.width),
        torch.ones(2, 4, dtype=torch.bool),
        _provenance("9"),
    )
    with pytest.raises(
        ValueError,
        match="selection provenance",
    ):
        prepare_width_compressed_mlp_refit_targets(
            (target,),
            clean_selection,
        )
