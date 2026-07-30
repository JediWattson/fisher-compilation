from __future__ import annotations

from dataclasses import replace
import io
import json

import pytest
import torch

from fisher_graph.causal_edge_jvp import apply_causal_lag_convolution
from fisher_graph.compiler.progressive import (
    ProgressiveCandidate,
    ProgressiveResourceFootprint,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    GemmaTwoHeadFitSequence,
)
from fisher_graph.gemma3_l3_l4_two_head_lowerer import (
    GemmaCausalResidualHead,
    GemmaL3L4TwoHeadArtifact,
    GemmaL3L4TwoHeadMutationLowerer,
    fit_gemma_causal_residual_head,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _fit_sequences(
    *,
    kernel: torch.Tensor,
    decoder: torch.Tensor,
) -> tuple[GemmaTwoHeadFitSequence, ...]:
    generator = torch.Generator().manual_seed(2901)
    result = []
    for index, family in enumerate(("family-a", "family-b", "family-b")):
        length = 9
        source = torch.randn(
            length,
            kernel.shape[1],
            generator=generator,
            dtype=torch.float64,
        )
        positions = torch.tensor(
            [0, 1, 2, 4, 5, 6, 8, 9, 10],
            dtype=torch.int64,
        )
        valid = torch.ones(length, dtype=torch.bool)
        if index == 1:
            valid[3] = False
            source[3] = 0
        modal = apply_causal_lag_convolution(
            source,
            kernel=kernel,
            logical_positions=positions,
            valid_mask=valid,
        )
        residual = modal @ decoder
        candidate = torch.zeros_like(residual)
        gradient = torch.randn(
            residual.shape,
            generator=generator,
            dtype=torch.float64,
        )
        result.append(
            GemmaTwoHeadFitSequence(
                example_id=f"example-{index}",
                family_id=family,
                model_inputs_sha256=_sha(10 + index),
                runtime_binding_sha256=_sha(20),
                source_modes=source,
                logical_positions=positions,
                valid_target_mask=valid,
                source_eligible_mask=valid.clone(),
                target_affected_mask=valid.clone(),
                native_x4=residual,
                candidate_x4=candidate,
                native_h4=residual * 0.5,
                candidate_h4=candidate,
                x4_loss_gradient=gradient,
                h4_loss_gradient=gradient * 0.25,
            )
        )
    return tuple(result)


def _head(site: str) -> GemmaCausalResidualHead:
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    kernel = torch.tensor(
        [
            [[0.7, -0.1], [0.2, 0.4]],
            [[-0.2, 0.3], [0.1, 0.5]],
        ],
        dtype=torch.float64,
    )
    return fit_gemma_causal_residual_head(
        site=site,
        sequences=_fit_sequences(kernel=kernel, decoder=decoder),
        directions=decoder,
        parent_runtime_binding_sha256=_sha(20),
        residual_map_sha256=_sha(21),
        analysis_artifact_sha256=_sha(22),
        fit_manifest_sha256=_sha(23),
        bridge_binding_sha256=_sha(24),
        lag_count=2,
        ridge=1.0e-8,
    )


def _independent_state_head() -> GemmaCausalResidualHead:
    return GemmaCausalResidualHead(
        site="layer.4.output",
        parent_runtime_binding_sha256=_sha(20),
        residual_map_sha256=_sha(21),
        analysis_artifact_sha256=_sha(22),
        fit_manifest_sha256=_sha(23),
        bridge_binding_sha256=_sha(24),
        decoder=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        lag_kernel=torch.tensor(
            [[[0.25, -0.5]]],
            dtype=torch.float64,
        ),
        state_kernel=torch.tensor(
            [[2.0, -1.0]],
            dtype=torch.float64,
        ),
        conditioning=(
            "l3_source_modes_plus_independent_realized_h4_modes_v1"
        ),
        ridge=1.0e-8,
        fit_row_count=3,
        family_ids=("family-a",),
        fit_sequence_sha256s=(_sha(10),),
        fit_objective="candidate_nll_vjp_metric_ridge_v1",
        weighted_residual_rmse=0.25,
        normalized_nll_direction_rmse=0.125,
        linearized_nll_residual_rmse=0.5,
        state_encoder=torch.tensor(
            [[0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
    )


def _one_pass_prefix() -> Gemma3L3L4OnePassPrefix:
    grid = (1, 3)
    valid = torch.ones(grid, dtype=torch.bool)
    return Gemma3L3L4OnePassPrefix(
        source_modes=torch.zeros((*grid, 1), dtype=torch.float64),
        clamped_y3=torch.zeros((*grid, 3), dtype=torch.float64),
        predicted_target_modal_delta=torch.zeros(
            (*grid, 2),
            dtype=torch.float64,
        ),
        decoded_base_x4_delta=torch.zeros(
            (*grid, 3),
            dtype=torch.float64,
        ),
        logical_positions=torch.arange(3, dtype=torch.int64).unsqueeze(0),
        valid_target_mask=valid,
        source_eligible_mask=valid.clone(),
        target_affected_mask=torch.tensor(
            [[True, False, True]],
            dtype=torch.bool,
        ),
        bridge_binding_sha256=_sha(24),
    )


def _artifact() -> GemmaL3L4TwoHeadArtifact:
    return GemmaL3L4TwoHeadArtifact(
        parent_artifact_sha256=_sha(30),
        parent_receipt_sha256=_sha(31),
        residual_map_sha256=_sha(32),
        analysis_artifact_sha256=_sha(33),
        bridge_binding_sha256=_sha(24),
        live_model_sha256=_sha(34),
        adapter_execution_sha256=_sha(35),
        heads=(
            _head("layer.4.mlp.normalized_input"),
            _head("layer.4.output"),
        ),
        recipe_sha256=_sha(36),
    )


def _loss_metric_sequences() -> tuple[GemmaTwoHeadFitSequence, ...]:
    source = torch.ones(2, 1, dtype=torch.float64)
    residual = torch.tensor(
        [[0.0, 0.0], [0.0, 10.0]],
        dtype=torch.float64,
    )
    gradient = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    candidate_gradient = torch.tensor(
        [[1.0, 0.0], [-1.0, 1.0]],
        dtype=torch.float64,
    )
    mask = torch.ones(2, dtype=torch.bool)
    zeros = torch.zeros_like(residual)
    return (
        GemmaTwoHeadFitSequence(
            example_id="loss-metric-example",
            family_id="loss-metric-family",
            model_inputs_sha256=_sha(90),
            runtime_binding_sha256=_sha(20),
            source_modes=source,
            logical_positions=torch.arange(2, dtype=torch.int64),
            valid_target_mask=mask,
            source_eligible_mask=mask,
            target_affected_mask=mask,
            native_x4=zeros,
            candidate_x4=zeros,
            native_h4=residual,
            candidate_h4=zeros,
            x4_loss_gradient=zeros,
            h4_loss_gradient=gradient,
            candidate_h4_loss_gradient=candidate_gradient,
        ),
    )


def _candidate(
    artifact: GemmaL3L4TwoHeadArtifact,
) -> ProgressiveCandidate:
    resources = ProgressiveResourceFootprint(
        candidate_execution_sha256=artifact.execution_sha256,
        accounting_artifact_sha256=_sha(40),
        parameter_scope="unit.parameters",
        compute_scope="unit.macs",
        runtime_id="unit.runtime",
        runtime_dtype="float64",
        sequence_scope_sha256=_sha(41),
        compiled_learned_parameters=20,
        retained_source_learned_parameters=100,
        support_learned_parameters=1,
        compiled_runtime_parameter_bytes=160,
        retained_source_runtime_parameter_bytes=800,
        support_runtime_parameter_bytes=8,
        compiled_logical_macs_per_token=20,
        retained_source_logical_macs_per_token=100,
        support_logical_macs_per_token=1,
        cost_complete=True,
    )
    return ProgressiveCandidate(
        candidate_id="unit-two-head-child",
        iteration=1,
        artifact_sha256=artifact.artifact_sha256,
        execution_sha256=artifact.execution_sha256,
        runtime_binding_sha256=artifact.runtime_binding_sha256,
        resources=resources,
        mutation_kind="add_residual_edge",
        parent_artifact_sha256=_sha(30),
        proposal_sha256=_sha(42),
    )


def test_finite_residual_head_recovers_causal_lag_kernel() -> None:
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    kernel = torch.tensor(
        [
            [[0.7, -0.1], [0.2, 0.4]],
            [[-0.2, 0.3], [0.1, 0.5]],
            [[0.05, 0.2], [-0.3, 0.1]],
        ],
        dtype=torch.float64,
    )
    sequences = _fit_sequences(kernel=kernel, decoder=decoder)

    head = fit_gemma_causal_residual_head(
        site="layer.4.mlp.normalized_input",
        sequences=sequences,
        directions=decoder,
        parent_runtime_binding_sha256=_sha(20),
        residual_map_sha256=_sha(21),
        analysis_artifact_sha256=_sha(22),
        fit_manifest_sha256=_sha(23),
        bridge_binding_sha256=_sha(24),
        lag_count=3,
        ridge=1.0e-10,
    )

    torch.testing.assert_close(
        head.lag_kernel,
        kernel,
        atol=2.0e-8,
        rtol=2.0e-8,
    )
    assert head.fit_row_count == sum(
        sequence.affected_rows for sequence in sequences
    )
    assert head.family_ids == ("family-a", "family-b")
    assert head.weighted_residual_rmse < 1.0e-8
    assert head.logical_macs_per_token_upper_bound == (
        3 * 2 * 2 + 2 * 3
    )


def test_source_nll_vjp_metric_trades_hidden_rmse_for_loss_direction() -> None:
    decoder = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    arguments = {
        "site": "layer.4.output",
        "sequences": _loss_metric_sequences(),
        "directions": decoder,
        "parent_runtime_binding_sha256": _sha(20),
        "residual_map_sha256": _sha(21),
        "analysis_artifact_sha256": _sha(22),
        "fit_manifest_sha256": _sha(23),
        "bridge_binding_sha256": _sha(24),
        "lag_count": 1,
        "ridge": 1.0e-8,
    }

    baseline = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="hidden_residual_ridge",
    )
    loss_metric = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="source_nll_vjp_metric_ridge_v1",
    )
    repeated = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="source_nll_vjp_metric_ridge_v1",
    )

    assert (
        loss_metric.linearized_nll_residual_rmse
        < baseline.linearized_nll_residual_rmse
    )
    assert (
        loss_metric.normalized_nll_direction_rmse
        < baseline.normalized_nll_direction_rmse
    )
    assert (
        loss_metric.weighted_residual_rmse
        > baseline.weighted_residual_rmse
    )
    assert (
        loss_metric.prepared_float_scalar_count
        == baseline.prepared_float_scalar_count
    )
    assert (
        loss_metric.logical_macs_per_token_upper_bound
        == baseline.logical_macs_per_token_upper_bound
    )
    assert loss_metric.artifact_sha256 != baseline.artifact_sha256
    assert repeated.artifact_sha256 == loss_metric.artifact_sha256
    restored = GemmaCausalResidualHead.from_state_dict(
        loss_metric.state_dict()
    )
    assert restored.artifact_sha256 == loss_metric.artifact_sha256
    assert (
        restored.fit_objective
        == "source_nll_vjp_metric_ridge_v1"
    )


def test_source_nll_vjp_metric_is_baseline_equivalent_without_signal() -> None:
    sequences = tuple(
        replace(
            sequence,
            x4_loss_gradient=torch.zeros_like(
                sequence.x4_loss_gradient
            ),
            h4_loss_gradient=torch.zeros_like(
                sequence.h4_loss_gradient
            ),
        )
        for sequence in _fit_sequences(
            kernel=torch.tensor(
                [[[0.7, -0.1], [0.2, 0.4]]],
                dtype=torch.float64,
            ),
            decoder=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
        )
    )
    arguments = {
        "site": "layer.4.output",
        "sequences": sequences,
        "directions": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
        "parent_runtime_binding_sha256": _sha(20),
        "residual_map_sha256": _sha(21),
        "analysis_artifact_sha256": _sha(22),
        "fit_manifest_sha256": _sha(23),
        "bridge_binding_sha256": _sha(24),
        "lag_count": 1,
        "ridge": 1.0e-8,
    }

    baseline = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="hidden_residual_ridge",
    )
    loss_metric = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="source_nll_vjp_metric_ridge_v1",
    )

    assert torch.equal(loss_metric.lag_kernel, baseline.lag_kernel)
    assert loss_metric.linearized_nll_residual_rmse == 0.0
    assert loss_metric.normalized_nll_direction_rmse == 0.0
    assert loss_metric.artifact_sha256 != baseline.artifact_sha256


def test_candidate_conditioned_vjp_uses_the_candidate_h4_tangent() -> None:
    decoder = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    sequences = _loss_metric_sequences()
    arguments = {
        "site": "layer.4.output",
        "sequences": sequences,
        "directions": decoder,
        "parent_runtime_binding_sha256": _sha(20),
        "residual_map_sha256": _sha(21),
        "analysis_artifact_sha256": _sha(22),
        "fit_manifest_sha256": _sha(23),
        "bridge_binding_sha256": _sha(24),
        "lag_count": 1,
        "ridge": 1.0e-8,
    }

    source_metric = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="source_nll_vjp_metric_ridge_v1",
    )
    candidate_metric = fit_gemma_causal_residual_head(
        **arguments,
        fit_objective="candidate_nll_vjp_metric_ridge_v1",
    )

    assert float(source_metric.lag_kernel[0, 0, 0]) > 0.0
    assert float(candidate_metric.lag_kernel[0, 0, 0]) < 0.0
    sequence = sequences[0]
    candidate_gradient = sequence.candidate_h4_loss_gradient
    assert candidate_gradient is not None

    def candidate_direction_error(head: GemmaCausalResidualHead) -> float:
        prediction = (
            torch.ones(2, 1, dtype=torch.float64)
            @ head.lag_kernel.reshape(1, 1)
            @ head.decoder
        )
        error = sequence.h4_residual_rows - prediction
        return float(
            torch.sqrt(
                (
                    (candidate_gradient * error).sum(dim=1).square()
                ).mean()
            )
        )

    assert (
        candidate_direction_error(candidate_metric)
        < candidate_direction_error(source_metric)
    )
    assert (
        candidate_metric.prepared_float_scalar_count
        == source_metric.prepared_float_scalar_count
    )
    assert (
        candidate_metric.logical_macs_per_token_upper_bound
        == source_metric.logical_macs_per_token_upper_bound
    )
    restored = GemmaCausalResidualHead.from_state_dict(
        candidate_metric.state_dict()
    )
    assert (
        restored.fit_objective
        == "candidate_nll_vjp_metric_ridge_v1"
    )

    missing = tuple(
        replace(sequence, candidate_h4_loss_gradient=None)
        for sequence in sequences
    )
    with pytest.raises(ValueError, match="lacks its VJP"):
        fit_gemma_causal_residual_head(
            **{**arguments, "sequences": missing},
            fit_objective="candidate_nll_vjp_metric_ridge_v1",
        )
    with pytest.raises(ValueError, match="H4-only"):
        fit_gemma_causal_residual_head(
            **{**arguments, "site": "layer.4.mlp.normalized_input"},
            fit_objective="candidate_nll_vjp_metric_ridge_v1",
        )


def test_realized_h4_conditioning_recovers_a_state_only_residual() -> None:
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    state_kernel = torch.tensor(
        [[0.7, -0.2], [0.3, 0.5]],
        dtype=torch.float64,
    )
    realized_modes = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
            [0.25, -0.75],
            [2.0, -1.0],
        ],
        dtype=torch.float64,
    )
    candidate_h4 = realized_modes @ decoder
    residual = realized_modes @ state_kernel @ decoder
    mask = torch.ones(6, dtype=torch.bool)
    zeros = torch.zeros_like(candidate_h4)
    sequence = GemmaTwoHeadFitSequence(
        example_id="state-only",
        family_id="state-family",
        model_inputs_sha256=_sha(91),
        runtime_binding_sha256=_sha(20),
        source_modes=torch.zeros(6, 1, dtype=torch.float64),
        logical_positions=torch.arange(6, dtype=torch.int64),
        valid_target_mask=mask,
        source_eligible_mask=mask,
        target_affected_mask=mask,
        native_x4=zeros,
        candidate_x4=zeros,
        native_h4=candidate_h4 + residual,
        candidate_h4=candidate_h4,
        x4_loss_gradient=zeros,
        h4_loss_gradient=torch.ones_like(candidate_h4),
        candidate_h4_loss_gradient=torch.ones_like(candidate_h4),
    )
    arguments = {
        "site": "layer.4.output",
        "sequences": (sequence,),
        "directions": decoder,
        "parent_runtime_binding_sha256": _sha(20),
        "residual_map_sha256": _sha(21),
        "analysis_artifact_sha256": _sha(22),
        "fit_manifest_sha256": _sha(23),
        "bridge_binding_sha256": _sha(24),
        "lag_count": 1,
        "ridge": 1.0e-10,
        "fit_objective": "candidate_nll_vjp_metric_ridge_v1",
    }

    source_only = fit_gemma_causal_residual_head(**arguments)
    conditioned = fit_gemma_causal_residual_head(
        **arguments,
        conditioning=(
            "l3_source_modes_plus_realized_h4_decoder_modes_v1"
        ),
    )

    assert source_only.weighted_residual_rmse > 0.1
    assert conditioned.weighted_residual_rmse < 1.0e-8
    torch.testing.assert_close(
        conditioned.state_kernel,
        state_kernel,
        atol=2.0e-8,
        rtol=2.0e-8,
    )
    assert conditioned.prepared_float_scalar_count == (
        source_only.prepared_float_scalar_count + decoder.shape[0] ** 2
    )
    assert conditioned.logical_macs_per_token_upper_bound == (
        source_only.logical_macs_per_token_upper_bound
        + decoder.shape[1] * decoder.shape[0]
        + decoder.shape[0] ** 2
    )
    assert conditioned.conditioning == (
        "l3_source_modes_plus_realized_h4_decoder_modes_v1"
    )
    restored = GemmaCausalResidualHead.from_state_dict(
        conditioned.state_dict()
    )
    assert restored.artifact_sha256 == conditioned.artifact_sha256
    torch.testing.assert_close(
        restored.state_kernel,
        conditioned.state_kernel,
    )
    tampered = {
        name: value.clone()
        for name, value in conditioned.state_dict().items()
    }
    tampered["state_kernel"][0, 0] += 0.125
    with pytest.raises(ValueError, match="tensor hash differs"):
        GemmaCausalResidualHead.from_state_dict(tampered)
    with pytest.raises(ValueError, match="H4-only"):
        fit_gemma_causal_residual_head(
            **{
                **arguments,
                "site": "layer.4.mlp.normalized_input",
            },
            conditioning=(
                "l3_source_modes_plus_realized_h4_decoder_modes_v1"
            ),
        )


def test_independent_state_encoder_applies_the_declared_modal_map() -> None:
    head = _independent_state_head()
    prefix = _one_pass_prefix()
    realized_state = torch.tensor(
        [
            [
                [3.0, -2.0, 2.0],
                [7.0, 11.0, 5.0],
                [-4.0, 8.0, -1.0],
            ]
        ],
        dtype=torch.float64,
    )

    correction = head.correction(prefix, realized_state)
    state_encoder = head.state_encoder
    assert state_encoder is not None
    expected = (
        realized_state
        @ state_encoder.T
        @ head.state_kernel
        @ head.decoder
    )
    expected[:, 1] = 0.0

    torch.testing.assert_close(correction, expected)
    assert head.state_rank == 1
    assert head.prepared_float_scalar_count == 13
    assert head.logical_macs_per_token_upper_bound == 13
    assert not torch.equal(state_encoder, head.decoder[:1])


def test_independent_state_encoder_geometry_is_strict() -> None:
    head = _independent_state_head()

    with pytest.raises(ValueError, match="state encoder geometry"):
        replace(
            head,
            state_encoder=None,
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="state encoder rows"):
        replace(
            head,
            state_encoder=torch.tensor(
                [[0.0, 0.0, 2.0]],
                dtype=torch.float64,
            ),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="state kernel differs"):
        replace(
            head,
            state_kernel=torch.eye(2, dtype=torch.float64),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="only for the H4 head"):
        replace(
            head,
            site="layer.4.mlp.normalized_input",
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="cannot have a state encoder"):
        replace(
            _head("layer.4.output"),
            state_encoder=head.state_encoder,
            artifact_sha256="",
        )


def test_causal_head_does_not_let_future_modes_change_earlier_rows() -> None:
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    kernel = torch.tensor(
        [
            [[0.7, -0.1], [0.2, 0.4]],
            [[-0.2, 0.3], [0.1, 0.5]],
        ],
        dtype=torch.float64,
    )
    source = torch.randn(
        6,
        2,
        generator=torch.Generator().manual_seed(41),
        dtype=torch.float64,
    )
    positions = torch.arange(6, dtype=torch.int64)
    valid = torch.ones(6, dtype=torch.bool)
    baseline = apply_causal_lag_convolution(
        source,
        kernel=kernel,
        logical_positions=positions,
        valid_mask=valid,
    )
    perturbed = source.clone()
    perturbed[5] += 10_000
    changed = apply_causal_lag_convolution(
        perturbed,
        kernel=kernel,
        logical_positions=positions,
        valid_mask=valid,
    )

    assert torch.equal(baseline[:5], changed[:5])
    assert not torch.equal(baseline[5], changed[5])


def test_head_tensor_mutation_fails_integrity() -> None:
    decoder = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    kernel = torch.tensor(
        [
            [[0.7, -0.1], [0.2, 0.4]],
            [[-0.2, 0.3], [0.1, 0.5]],
        ],
        dtype=torch.float64,
    )
    head = fit_gemma_causal_residual_head(
        site="layer.4.output",
        sequences=_fit_sequences(kernel=kernel, decoder=decoder),
        directions=decoder,
        parent_runtime_binding_sha256=_sha(20),
        residual_map_sha256=_sha(21),
        analysis_artifact_sha256=_sha(22),
        fit_manifest_sha256=_sha(23),
        bridge_binding_sha256=_sha(24),
        lag_count=2,
        ridge=1.0e-8,
    )

    head.lag_kernel[0, 0, 0] += 1

    with pytest.raises(RuntimeError, match="payload drifted"):
        head.validate_integrity()


def test_residual_head_tensor_only_state_round_trips_strictly() -> None:
    head = _head("layer.4.output")
    state = head.state_dict()
    metadata = json.loads(bytes(state["metadata_utf8"].tolist()))

    assert (
        head.artifact_sha256
        == "1d6d10bb2b54edf6b97b89f39119c4d14f5403950f454e51904abf8eae961756"
    )
    assert head.state_encoder is None
    empty_encoder_head = replace(
        head,
        state_encoder=torch.empty((0, 0), dtype=torch.float64),
        artifact_sha256="",
    )
    assert empty_encoder_head.state_encoder is None
    assert empty_encoder_head.artifact_sha256 == head.artifact_sha256
    assert metadata["schema"].endswith("_state.v3")
    assert metadata["format_version"] == 3
    assert set(state) == {
        "metadata_utf8",
        "decoder",
        "lag_kernel",
        "state_kernel",
    }
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = GemmaCausalResidualHead.from_state_dict(loaded)

    assert restored.artifact_sha256 == head.artifact_sha256
    assert restored.site == head.site
    torch.testing.assert_close(restored.decoder, head.decoder)
    torch.testing.assert_close(restored.lag_kernel, head.lag_kernel)
    torch.testing.assert_close(restored.state_kernel, head.state_kernel)
    assert restored.decoder.data_ptr() != head.decoder.data_ptr()
    assert restored.lag_kernel.data_ptr() != head.lag_kernel.data_ptr()
    assert restored.state_kernel is not head.state_kernel


def test_independent_state_encoder_uses_strict_v4_state() -> None:
    head = _independent_state_head()
    state = head.state_dict()
    metadata = json.loads(bytes(state["metadata_utf8"].tolist()))

    assert set(state) == {
        "metadata_utf8",
        "decoder",
        "lag_kernel",
        "state_kernel",
        "state_encoder",
    }
    assert metadata["schema"].endswith("_state.v4")
    assert metadata["format_version"] == 4
    assert metadata["conditioning"] == (
        "l3_source_modes_plus_independent_realized_h4_modes_v1"
    )
    restored = GemmaCausalResidualHead.from_state_dict(state)

    assert restored.artifact_sha256 == head.artifact_sha256
    assert restored.state_encoder is not None
    assert head.state_encoder is not None
    torch.testing.assert_close(
        restored.state_encoder,
        head.state_encoder,
    )
    assert restored.state_encoder.data_ptr() != head.state_encoder.data_ptr()

    tampered = {
        name: value.clone()
        for name, value in state.items()
    }
    tampered["state_encoder"][0, 2] = -1.0
    with pytest.raises(ValueError, match="tensor hash differs"):
        GemmaCausalResidualHead.from_state_dict(tampered)

    missing = dict(state)
    missing.pop("state_kernel")
    with pytest.raises(ValueError, match="keys differ"):
        GemmaCausalResidualHead.from_state_dict(missing)

    extra = {**state, "unexpected": torch.zeros(1)}
    with pytest.raises(ValueError, match="keys differ"):
        GemmaCausalResidualHead.from_state_dict(extra)

    head.state_encoder[0, 2] = -1.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        head.validate_integrity()


def test_residual_head_state_rejects_missing_extra_and_tampered_data() -> None:
    state = _head("layer.4.output").state_dict()

    missing = dict(state)
    missing.pop("lag_kernel")
    with pytest.raises(ValueError, match="keys differ"):
        GemmaCausalResidualHead.from_state_dict(missing)

    extra = {**state, "unexpected": torch.zeros(1)}
    with pytest.raises(ValueError, match="keys differ"):
        GemmaCausalResidualHead.from_state_dict(extra)

    non_tensor = {**state, "decoder": "not-a-tensor"}
    with pytest.raises(TypeError, match="all be Tensors"):
        GemmaCausalResidualHead.from_state_dict(non_tensor)

    malformed_metadata = {
        **state,
        "metadata_utf8": torch.tensor([0x7B], dtype=torch.uint8),
    }
    with pytest.raises(ValueError, match="canonical JSON"):
        GemmaCausalResidualHead.from_state_dict(malformed_metadata)

    tampered = {
        name: value.clone()
        for name, value in state.items()
    }
    tampered["lag_kernel"][0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="tensor hash differs"):
        GemmaCausalResidualHead.from_state_dict(tampered)


def test_two_head_artifact_tensor_only_state_round_trips_strictly() -> None:
    artifact = _artifact()
    state = artifact.state_dict()

    assert (
        artifact.artifact_sha256
        == "d0ac75cd1620cb0b5713d4ef28ef5e8ea97b213a772438324e69025e12112fca"
    )
    assert all(isinstance(value, torch.Tensor) for value in state.values())
    assert set(state) == {
        "metadata_utf8",
        "heads.0.metadata_utf8",
        "heads.0.decoder",
        "heads.0.lag_kernel",
        "heads.0.state_kernel",
        "heads.1.metadata_utf8",
        "heads.1.decoder",
        "heads.1.lag_kernel",
        "heads.1.state_kernel",
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = GemmaL3L4TwoHeadArtifact.from_state_dict(loaded)

    assert restored.artifact_sha256 == artifact.artifact_sha256
    assert restored.execution_sha256 == artifact.execution_sha256
    assert (
        restored.runtime_binding_sha256
        == artifact.runtime_binding_sha256
    )
    assert tuple(head.artifact_sha256 for head in restored.heads) == tuple(
        head.artifact_sha256 for head in artifact.heads
    )
    for original, reloaded in zip(
        artifact.heads,
        restored.heads,
        strict=True,
    ):
        torch.testing.assert_close(reloaded.decoder, original.decoder)
        torch.testing.assert_close(
            reloaded.lag_kernel,
            original.lag_kernel,
        )
        torch.testing.assert_close(
            reloaded.state_kernel,
            original.state_kernel,
        )
        assert reloaded.decoder.data_ptr() != original.decoder.data_ptr()


def test_mixed_v3_v4_heads_round_trip_in_a_v2_outer_artifact() -> None:
    legacy = _artifact()
    artifact = replace(
        legacy,
        heads=(legacy.heads[0], _independent_state_head()),
        artifact_sha256="",
    )
    state = artifact.state_dict()
    metadata = json.loads(bytes(state["metadata_utf8"].tolist()))

    assert metadata["schema"].endswith("_artifact_state.v2")
    assert metadata["format_version"] == 2
    assert "heads.0.state_encoder" not in state
    assert "heads.1.state_encoder" in state

    restored = GemmaL3L4TwoHeadArtifact.from_state_dict(state)

    assert restored.artifact_sha256 == artifact.artifact_sha256
    assert restored.execution_sha256 == artifact.execution_sha256
    assert (
        restored.runtime_binding_sha256
        == artifact.runtime_binding_sha256
    )
    assert tuple(
        head.artifact_sha256 for head in restored.heads
    ) == tuple(head.artifact_sha256 for head in artifact.heads)
    assert restored.heads[0].state_encoder is None
    assert restored.heads[1].state_encoder is not None
    assert (
        restored.prepared_float_scalar_count
        == artifact.prepared_float_scalar_count
    )
    assert (
        restored.logical_macs_per_token_upper_bound
        == artifact.logical_macs_per_token_upper_bound
    )


def test_single_head_artifact_state_uses_only_its_declared_slot() -> None:
    two_head = _artifact()
    artifact = replace(
        two_head,
        heads=(two_head.heads[0],),
        artifact_sha256="",
    )

    state = artifact.state_dict()
    restored = GemmaL3L4TwoHeadArtifact.from_state_dict(state)

    assert set(state) == {
        "metadata_utf8",
        "heads.0.metadata_utf8",
        "heads.0.decoder",
        "heads.0.lag_kernel",
        "heads.0.state_kernel",
    }
    assert restored.artifact_sha256 == artifact.artifact_sha256
    assert len(restored.heads) == 1
    assert restored.heads[0].site == "layer.4.mlp.normalized_input"


def test_two_head_artifact_state_rejects_nested_tamper_and_shape_drift() -> None:
    state = _artifact().state_dict()

    missing = dict(state)
    missing.pop("heads.1.decoder")
    with pytest.raises(ValueError, match="keys differ"):
        GemmaL3L4TwoHeadArtifact.from_state_dict(missing)

    extra = {**state, "heads.2.decoder": torch.zeros(1)}
    with pytest.raises(ValueError, match="keys differ"):
        GemmaL3L4TwoHeadArtifact.from_state_dict(extra)

    tampered = {
        name: value.clone()
        for name, value in state.items()
    }
    tampered["heads.0.lag_kernel"][0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="tensor hash differs"):
        GemmaL3L4TwoHeadArtifact.from_state_dict(tampered)

    wrong_shape = {
        name: value.clone()
        for name, value in state.items()
    }
    wrong_shape["heads.0.decoder"] = wrong_shape[
        "heads.0.decoder"
    ][:, :-1]
    with pytest.raises(ValueError, match="tensor hash differs"):
        GemmaL3L4TwoHeadArtifact.from_state_dict(wrong_shape)


def test_lowerer_artifact_export_authenticates_and_isolates_candidate() -> None:
    artifact = _artifact()
    candidate = _candidate(artifact)
    lowerer = object.__new__(GemmaL3L4TwoHeadMutationLowerer)
    lowerer._artifacts = {artifact.artifact_sha256: artifact}
    lowerer._candidate_receipts = {
        artifact.artifact_sha256: candidate.receipt_sha256
    }
    lowerer._authenticate = lambda: None

    exported = lowerer.artifact_for(candidate)

    assert exported.artifact_sha256 == artifact.artifact_sha256
    assert exported is not artifact
    assert exported.heads[0] is not artifact.heads[0]
    exported.heads[0].lag_kernel[0, 0, 0] += 1
    artifact.validate_integrity()
    lowerer.artifact_for(candidate).validate_integrity()

    forged = replace(
        candidate,
        resources=replace(
            candidate.resources,
            accounting_artifact_sha256=_sha(43),
        ),
    )
    with pytest.raises(ValueError, match="candidate receipt"):
        lowerer.artifact_for(forged)

    unknown = replace(
        candidate,
        artifact_sha256=_sha(44),
    )
    with pytest.raises(ValueError, match="not built by this lowerer"):
        lowerer.artifact_for(unknown)
