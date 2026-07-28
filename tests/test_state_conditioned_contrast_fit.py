from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.gated_executor import GatedCausalModalExecutorConfig
import fisher_graph.state_conditioned_contrast_fit as contrast_fit_module
from fisher_graph.state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastAwareReferenceProviderPlan,
    ContrastTrainingMetrics,
    IndexedReferenceBatch,
    ReferenceProviderContrastPair,
    fit_contrast_aware_reference_provider,
)
from fisher_graph.state_conditioned_reference_provider import (
    SyntheticReferenceBatch,
)


_WIDTH = 64
_BINDING = "b" * 64
_RMS_EPSILON = 1e-6
_ENDPOINT_IDS = (
    "sensitivity/negative",
    "sensitivity/positive",
    "null/negative",
    "null/positive",
)


def _indexed_batch(
    split: str = "fit",
    *,
    invalid_offset: float = 0.0,
) -> IndexedReferenceBatch:
    batch_size = len(_ENDPOINT_IDS)
    sequence_length = 4
    mode_index = torch.arange(_WIDTH, dtype=torch.float64)
    position_index = torch.arange(
        sequence_length,
        dtype=torch.float64,
    )
    common = (
        0.08
        * torch.sin(
            (position_index[:, None] + 1.0)
            * (mode_index[None, :] + 1.0)
            / 13.0
        )
    )
    modal = common.unsqueeze(0).repeat(batch_size, 1, 1)

    # A deliberately non-prefix sensitivity uses one low and one high mode.
    modal[0, :, 2] -= 0.35
    modal[0, :, 61] += 0.20
    modal[1, :, 2] += 0.35
    modal[1, :, 61] -= 0.20

    null = torch.zeros(
        batch_size,
        sequence_length,
        1,
        dtype=torch.float64,
    )
    null[2, :, 0] = -2.0
    null[3, :, 0] = 2.0
    # The two null endpoints have equal non-null RMS despite differing exact
    # null coordinates.
    null_square = 4.0
    nonnull_fraction = 1.0 - null_square / _WIDTH
    row_rms = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.float64,
    )
    row_rms[2:] = math.sqrt(
        (1.0 + null_square * _RMS_EPSILON / _WIDTH)
        / nonnull_fraction
    )

    target = (
        0.55 * modal
        + 0.20 * modal.roll(shifts=7, dims=-1)
        - 0.10 * modal.roll(shifts=-11, dims=-1)
    )
    # Exact-null coordinates are absent from the teacher mapping.
    target[3] = target[2]

    mask = torch.tensor(
        [
            [True, False, True, True],
            [True, False, True, True],
            [True, False, True, True],
            [True, False, True, True],
        ],
        dtype=torch.bool,
    )
    positions = torch.tensor(
        [
            [0, -9, 3, 7],
            [0, -9, 3, 7],
            [0, -9, 3, 7],
            [0, -9, 3, 7],
        ],
        dtype=torch.int64,
    )
    if invalid_offset:
        modal[:, 1] += invalid_offset
        null[:, 1] += invalid_offset
        row_rms[:, 1] += abs(invalid_offset)
        target[:, 1] -= invalid_offset

    return IndexedReferenceBatch(
        batch=SyntheticReferenceBatch(
            split=split,
            modal_coordinates=modal,
            null_coordinates=null,
            row_rms=row_rms,
            target_modes=target,
            logical_positions=positions,
            valid_mask=mask,
            synthetic_binding_sha256=_BINDING,
        ),
        endpoint_ids=_ENDPOINT_IDS,
    )


def _pairs() -> tuple[ReferenceProviderContrastPair, ...]:
    return (
        ReferenceProviderContrastPair(
            pair_id="pair/sensitivity",
            family="signed",
            role="expected_sensitivity",
            left_endpoint_id=_ENDPOINT_IDS[0],
            right_endpoint_id=_ENDPOINT_IDS[1],
            rank_stratum="nonadjacent-low-high",
        ),
        ReferenceProviderContrastPair(
            pair_id="pair/null",
            family="exact-null",
            role="intended_null",
            left_endpoint_id=_ENDPOINT_IDS[2],
            right_endpoint_id=_ENDPOINT_IDS[3],
            rank_stratum="exact-null",
        ),
    )


def _provider_chart(
    *,
    modal_primal: Tensor,
    modal_tangent: Tensor,
    null_primal: Tensor | None = None,
    row_rms_primal: Tensor | None = None,
    null_tangent: Tensor | None = None,
    row_rms_tangent: Tensor | None = None,
) -> dict[str, Tensor]:
    sequence_length = int(modal_primal.shape[0])
    return {
        "provider_chart_modal_primal": modal_primal,
        "provider_chart_null_primal": (
            torch.zeros(
                sequence_length,
                1,
                dtype=torch.float64,
            )
            if null_primal is None
            else null_primal
        ),
        "provider_chart_row_rms_primal": (
            torch.ones(sequence_length, dtype=torch.float64)
            if row_rms_primal is None
            else row_rms_primal
        ),
        "provider_chart_modal_tangent": modal_tangent,
        "provider_chart_null_tangent": (
            torch.zeros(
                sequence_length,
                1,
                dtype=torch.float64,
            )
            if null_tangent is None
            else null_tangent
        ),
        "provider_chart_row_rms_tangent": (
            torch.zeros(sequence_length, dtype=torch.float64)
            if row_rms_tangent is None
            else row_rms_tangent
        ),
    }


def _config(rank: int = 4) -> GatedCausalModalExecutorConfig:
    return GatedCausalModalExecutorConfig(
        input_modes=rank + 2,
        output_modes=rank,
        expert_count=1,
        expert_rank=rank,
        router_width=rank,
        same_position_skip=False,
        source_normalized_routing=False,
    )


def _objective() -> ContrastAwareObjective:
    return ContrastAwareObjective(
        pointwise_weight=1.0,
        sensitivity_relative_delta_weight=1.0,
        sensitivity_direction_weight=0.25,
        midpoint_jvp_weight=1.0,
        intended_null_weight=1.0,
    )


def _fit(
    *,
    batch: IndexedReferenceBatch | None = None,
    rank: int = 4,
    steps: int = 2,
) -> ContrastAwareReferenceProviderPlan:
    return fit_contrast_aware_reference_provider(
        modal_center=torch.zeros(_WIDTH, dtype=torch.float64),
        gain_log_center=0.0,
        gain_log_scale=1.0,
        residual_width=_WIDTH,
        rms_epsilon=_RMS_EPSILON,
        target_center=torch.zeros(_WIDTH, dtype=torch.float64),
        target_scale=torch.ones(_WIDTH, dtype=torch.float64),
        fit_batches=(_indexed_batch() if batch is None else batch,),
        contrast_pairs=_pairs(),
        executor_config=_config(rank),
        objective=_objective(),
        fisher_metric_weight=torch.linspace(
            0.5,
            1.5,
            _WIDTH,
            dtype=torch.float64,
        ),
        steps=steps,
        learning_rate=0.01,
        seed=713,
    )


def test_index_pair_and_objective_round_trip_and_reject_tampering() -> None:
    indexed = _indexed_batch()
    restored_indexed = IndexedReferenceBatch.from_state_dict(
        indexed.state_dict()
    )
    assert restored_indexed.artifact_sha256 == indexed.artifact_sha256
    assert restored_indexed.endpoint_ids == _ENDPOINT_IDS

    pair = _pairs()[0]
    restored_pair = ReferenceProviderContrastPair.from_state_dict(
        pair.state_dict()
    )
    assert restored_pair == pair

    objective = _objective()
    restored_objective = ContrastAwareObjective.from_state_dict(
        objective.state_dict()
    )
    assert restored_objective == objective
    assert (
        "supplied_hidden_midpoint_provider_chart_primal"
        in objective.state_dict()["jvp_semantics"]
    )
    assert (
        "endpoint_arithmetic_forbidden"
        in objective.state_dict()["jvp_semantics"]
    )

    indexed_tamper = copy.deepcopy(indexed.state_dict())
    indexed_tamper["endpoint_ids"] = (
        "changed",
        *indexed_tamper["endpoint_ids"][1:],
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        IndexedReferenceBatch.from_state_dict(indexed_tamper)

    pair_tamper = copy.deepcopy(pair.state_dict())
    pair_tamper["right_endpoint_id"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        ReferenceProviderContrastPair.from_state_dict(pair_tamper)

    objective_tamper = copy.deepcopy(objective.state_dict())
    objective_tamper["intended_null_weight"] = 2.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ContrastAwareObjective.from_state_dict(objective_tamper)
    objective_semantics_tamper = copy.deepcopy(objective.state_dict())
    objective_semantics_tamper["jvp_semantics"] = (
        "endpoint arithmetic is acceptable"
    )
    with pytest.raises(ValueError, match="semantics drifted"):
        ContrastAwareObjective.from_state_dict(objective_semantics_tamper)

    metrics = _fit(steps=1).final_metrics
    assert (
        ContrastTrainingMetrics.from_state_dict(metrics.state_dict())
        == metrics
    )
    metrics_tamper = copy.deepcopy(metrics.state_dict())
    metrics_tamper["weighted_total"] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ContrastTrainingMetrics.from_state_dict(metrics_tamper)


def test_teacher_jvp_provider_chart_schema_round_trips_and_fails_closed() -> None:
    batch = _indexed_batch().batch
    modal_primal = 0.35 * batch.modal_coordinates[0] + 0.65 * (
        batch.modal_coordinates[1]
    )
    modal_tangent = (
        batch.modal_coordinates[1] - batch.modal_coordinates[0]
    )
    chart = _provider_chart(
        modal_primal=modal_primal,
        modal_tangent=modal_tangent,
        null_primal=torch.full((4, 1), 0.125, dtype=torch.float64),
        row_rms_primal=torch.linspace(
            0.9,
            1.2,
            4,
            dtype=torch.float64,
        ),
        null_tangent=torch.full((4, 1), -0.025, dtype=torch.float64),
        row_rms_tangent=torch.linspace(
            -0.03,
            0.04,
            4,
            dtype=torch.float64,
        ),
    )
    pair = ReferenceProviderContrastPair(
        pair_id="pair/chart-round-trip",
        family="signed",
        role="expected_sensitivity",
        left_endpoint_id=_ENDPOINT_IDS[0],
        right_endpoint_id=_ENDPOINT_IDS[1],
        rank_stratum="nonadjacent-low-high",
        teacher_midpoint_jvp=modal_tangent,
        **chart,
    )
    restored = ReferenceProviderContrastPair.from_state_dict(
        pair.state_dict()
    )
    assert restored.artifact_sha256 == pair.artifact_sha256
    torch.testing.assert_close(
        restored.provider_chart_modal_primal,
        modal_primal,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        restored.provider_chart_row_rms_tangent,
        chart["provider_chart_row_rms_tangent"],
        rtol=0.0,
        atol=0.0,
    )

    base_arguments = {
        "pair_id": "pair/chart-firewall",
        "family": "signed",
        "role": "expected_sensitivity",
        "left_endpoint_id": _ENDPOINT_IDS[0],
        "right_endpoint_id": _ENDPOINT_IDS[1],
        "rank_stratum": "nonadjacent-low-high",
    }
    with pytest.raises(ValueError, match="complete provider-chart"):
        ReferenceProviderContrastPair(
            **base_arguments,
            teacher_midpoint_jvp=modal_tangent,
        )
    with pytest.raises(ValueError, match="require a teacher JVP"):
        ReferenceProviderContrastPair(
            **base_arguments,
            **chart,
        )
    partial = dict(chart)
    partial["provider_chart_null_tangent"] = None
    with pytest.raises(ValueError, match="complete provider-chart"):
        ReferenceProviderContrastPair(
            **base_arguments,
            teacher_midpoint_jvp=modal_tangent,
            **partial,
        )
    invalid_shape = dict(chart)
    invalid_shape["provider_chart_modal_tangent"] = modal_tangent[:, :-1]
    with pytest.raises(ValueError, match="must have shape"):
        ReferenceProviderContrastPair(
            **base_arguments,
            teacher_midpoint_jvp=modal_tangent,
            **invalid_shape,
        )
    nonfinite = dict(chart)
    nonfinite["provider_chart_null_primal"] = (
        chart["provider_chart_null_primal"].clone()
    )
    nonfinite["provider_chart_null_primal"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ReferenceProviderContrastPair(
            **base_arguments,
            teacher_midpoint_jvp=modal_tangent,
            **nonfinite,
        )
    nonpositive_rms = dict(chart)
    nonpositive_rms["provider_chart_row_rms_primal"] = torch.zeros(
        4,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="must be positive"):
        ReferenceProviderContrastPair(
            **base_arguments,
            teacher_midpoint_jvp=modal_tangent,
            **nonpositive_rms,
        )

    intended_null_arguments = dict(base_arguments)
    intended_null_arguments["role"] = "intended_null"
    with pytest.raises(ValueError, match="expected-sensitivity"):
        ReferenceProviderContrastPair(
            **intended_null_arguments,
            teacher_midpoint_jvp=modal_tangent,
            **chart,
        )

    missing_field = copy.deepcopy(pair.state_dict())
    del missing_field["provider_chart_modal_primal"]
    with pytest.raises(ValueError, match="fields do not match"):
        ReferenceProviderContrastPair.from_state_dict(missing_field)
    tampered = copy.deepcopy(pair.state_dict())
    tampered["provider_chart_modal_primal"][0, 0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ReferenceProviderContrastPair.from_state_dict(tampered)


def test_fixed_step_fit_is_deterministic_packed_and_round_trips() -> None:
    first = _fit()
    second = _fit()

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.rank == first.latent_rank == 4
    assert first.encoder_weight.shape == (_WIDTH, 4)
    assert first.decoder_weight.shape == (4, _WIDTH)
    torch.testing.assert_close(
        first.encoder_weight,
        second.encoder_weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.decoder_weight,
        second.decoder_weight,
        rtol=0.0,
        atol=0.0,
    )

    # Packed means every native coordinate remains available to the provider;
    # in particular, neither a low non-leading mode nor a far high mode is
    # sliced away before the rank bottleneck.
    assert torch.linalg.vector_norm(first.encoder_weight[2]).item() > 0.0
    assert torch.linalg.vector_norm(first.encoder_weight[61]).item() > 0.0
    assert torch.linalg.vector_norm(first.decoder_weight[:, 2]).item() > 0.0
    assert torch.linalg.vector_norm(first.decoder_weight[:, 61]).item() > 0.0
    assert first.active_encoder_source_modes == _WIDTH
    assert first.active_decoder_target_modes == _WIDTH

    restored = ContrastAwareReferenceProviderPlan.from_state_dict(
        first.state_dict()
    )
    assert restored.artifact_sha256 == first.artifact_sha256
    runtime = restored.prepare(dtype=torch.float64)
    batch = _indexed_batch().batch
    prediction = runtime(
        batch.modal_coordinates,
        batch.null_coordinates,
        batch.row_rms,
        valid_mask=batch.valid_mask,
        logical_positions=batch.logical_positions,
    )
    assert prediction.shape == batch.target_modes.shape
    assert torch.isfinite(prediction).all()

    tampered = copy.deepcopy(first.state_dict())
    tampered["encoder_weight"][2, 0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ContrastAwareReferenceProviderPlan.from_state_dict(tampered)


def test_fit_split_and_endpoint_firewalls_fail_closed() -> None:
    with pytest.raises(ValueError, match="'fit'|split"):
        _fit(batch=_indexed_batch("selection"), steps=1)

    missing_endpoint = (
        _pairs()[0],
        ReferenceProviderContrastPair(
            pair_id="pair/missing",
            family="signed",
            role="expected_sensitivity",
            left_endpoint_id=_ENDPOINT_IDS[0],
            right_endpoint_id="not-present",
            rank_stratum="missing",
        ),
    )
    with pytest.raises(ValueError, match="endpoint"):
        fit_contrast_aware_reference_provider(
            modal_center=torch.zeros(_WIDTH, dtype=torch.float64),
            gain_log_center=0.0,
            gain_log_scale=1.0,
            residual_width=_WIDTH,
            rms_epsilon=_RMS_EPSILON,
            target_center=torch.zeros(_WIDTH, dtype=torch.float64),
            target_scale=torch.ones(_WIDTH, dtype=torch.float64),
            fit_batches=(_indexed_batch(),),
            contrast_pairs=missing_endpoint,
            executor_config=_config(),
            objective=_objective(),
            steps=1,
            learning_rate=0.01,
            seed=1,
        )

    base = _indexed_batch()
    batch = base.batch
    mismatched_mask = batch.valid_mask.clone()
    mismatched_positions = batch.logical_positions.clone()
    mismatched_mask[1, 1] = True
    mismatched_positions[1, 1] = 1
    misaligned = IndexedReferenceBatch(
        batch=SyntheticReferenceBatch(
            split="fit",
            modal_coordinates=batch.modal_coordinates,
            null_coordinates=batch.null_coordinates,
            row_rms=batch.row_rms,
            target_modes=batch.target_modes,
            logical_positions=mismatched_positions,
            valid_mask=mismatched_mask,
            synthetic_binding_sha256=_BINDING,
        ),
        endpoint_ids=_ENDPOINT_IDS,
    )
    with pytest.raises(ValueError, match="masks, and positions"):
        _fit(batch=misaligned, steps=1)


def test_masked_rows_are_absent_from_features_outputs_and_fit_identity() -> None:
    plan = _fit(steps=1)
    runtime = plan.prepare(dtype=torch.float64)
    clean = _indexed_batch()
    changed = _indexed_batch(invalid_offset=10_000.0)

    clean_features = runtime.encode_features(
        clean.batch.modal_coordinates,
        clean.batch.null_coordinates,
        clean.batch.row_rms,
        clean.batch.valid_mask,
    )
    changed_features = runtime.encode_features(
        changed.batch.modal_coordinates,
        changed.batch.null_coordinates,
        changed.batch.row_rms,
        changed.batch.valid_mask,
    )
    torch.testing.assert_close(
        clean_features,
        changed_features,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        clean_features[~clean.batch.valid_mask],
        torch.zeros_like(clean_features[~clean.batch.valid_mask]),
    )

    clean_output = runtime(
        clean.batch.modal_coordinates,
        clean.batch.null_coordinates,
        clean.batch.row_rms,
        valid_mask=clean.batch.valid_mask,
        logical_positions=clean.batch.logical_positions,
    )
    changed_output = runtime(
        changed.batch.modal_coordinates,
        changed.batch.null_coordinates,
        changed.batch.row_rms,
        valid_mask=changed.batch.valid_mask,
        logical_positions=changed.batch.logical_positions,
    )
    torch.testing.assert_close(clean_output, changed_output, rtol=0.0, atol=0.0)
    assert torch.equal(
        clean_output[~clean.batch.valid_mask],
        torch.zeros_like(clean_output[~clean.batch.valid_mask]),
    )

    # Padding contents are excluded from the authenticated scientific content
    # presented to the optimizer, so they cannot change a fixed-seed fit.
    clean_plan = _fit(batch=clean, steps=1)
    changed_plan = _fit(batch=changed, steps=1)
    torch.testing.assert_close(
        clean_plan.encoder_weight,
        changed_plan.encoder_weight,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        clean_plan.decoder_weight,
        changed_plan.decoder_weight,
        rtol=0.0,
        atol=0.0,
    )


def test_exact_null_changes_leave_features_and_predictions_exactly_invariant() -> None:
    plan = _fit(steps=1)
    runtime = plan.prepare(dtype=torch.float64)
    batch = _indexed_batch().batch
    modal = batch.modal_coordinates[2:3].clone()
    mask = batch.valid_mask[2:3].clone()
    positions = batch.logical_positions[2:3].clone()

    zero_null = torch.zeros((1, 4, 1), dtype=torch.float64)
    changed_null = torch.full((1, 4, 1), 2.0, dtype=torch.float64)
    base_rms = torch.ones((1, 4), dtype=torch.float64)
    changed_rms = torch.full_like(
        base_rms,
        math.sqrt(
            (1.0 + 4.0 * _RMS_EPSILON / _WIDTH)
            / (1.0 - 4.0 / _WIDTH)
        ),
    )

    zero_features = runtime.encode_features(
        modal,
        zero_null,
        base_rms,
        mask,
    )
    changed_features = runtime.encode_features(
        modal,
        changed_null,
        changed_rms,
        mask,
    )
    torch.testing.assert_close(
        zero_features,
        changed_features,
        rtol=0.0,
        atol=1e-15,
    )
    assert zero_features.shape[-1] == plan.rank + 2

    zero_output = runtime(
        modal,
        zero_null,
        base_rms,
        valid_mask=mask,
        logical_positions=positions,
    )
    changed_output = runtime(
        modal,
        changed_null,
        changed_rms,
        valid_mask=mask,
        logical_positions=positions,
    )
    torch.testing.assert_close(
        zero_output,
        changed_output,
        rtol=0.0,
        atol=1e-15,
    )


class _FixedPredictionModel(nn.Module):
    def __init__(self, prediction: Tensor) -> None:
        super().__init__()
        self.encoder_weight = nn.Parameter(
            torch.zeros((1,), dtype=torch.float64)
        )
        self.register_buffer("prediction", prediction, persistent=False)

    def forward_standardized(
        self,
        modal: Tensor,
        null: Tensor,
        rms: Tensor,
        mask: Tensor,
        positions: Tensor,
    ) -> Tensor:
        del null, rms, positions
        # The zero-valued dependency keeps this helper valid under torch.func
        # transforms while preserving the declared fixed prediction.
        result = self.prediction + modal.sum() * 0.0
        return torch.where(
            mask.unsqueeze(-1),
            result,
            torch.zeros_like(result),
        )


class _ModalIdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder_weight = nn.Parameter(
            torch.zeros((1,), dtype=torch.float64)
        )

    def forward_standardized(
        self,
        modal: Tensor,
        null: Tensor,
        rms: Tensor,
        mask: Tensor,
        positions: Tensor,
    ) -> Tensor:
        del null, rms, positions
        return torch.where(
            mask.unsqueeze(-1),
            modal,
            torch.zeros_like(modal),
        )


def _prepared_data(
    *,
    batch: IndexedReferenceBatch | None = None,
    pairs: tuple[ReferenceProviderContrastPair, ...] | None = None,
):
    return contrast_fit_module._prepare_fit_data(
        fit_batches=(_indexed_batch() if batch is None else batch,),
        contrast_pairs=_pairs() if pairs is None else pairs,
        require_fit_split=True,
    )


def _components(
    prediction: Tensor,
    *,
    metric_weight: Tensor | None = None,
    objective: ContrastAwareObjective | None = None,
):
    return contrast_fit_module._loss_components(
        _FixedPredictionModel(prediction),
        data=_prepared_data(),
        target_center=torch.zeros(_WIDTH, dtype=torch.float64),
        target_scale=torch.ones(_WIDTH, dtype=torch.float64),
        metric_weight=(
            torch.ones(_WIDTH, dtype=torch.float64)
            if metric_weight is None
            else metric_weight
        ),
        objective=_objective() if objective is None else objective,
    )


def test_loss_components_cancel_common_offsets_and_score_direction_exactly() -> None:
    batch = _indexed_batch().batch
    target = batch.target_modes
    common_offset = 0.25
    common = _components(target + common_offset)

    assert common.pointwise_mse.item() == pytest.approx(
        common_offset**2
    )
    assert common.sensitivity_relative_delta_mse.item() == pytest.approx(
        0.0,
        abs=1e-14,
    )
    assert common.sensitivity_direction_loss.item() == pytest.approx(
        0.0,
        abs=1e-14,
    )
    assert common.intended_null_absolute_mse.item() == pytest.approx(0.0)

    reversed_prediction = target.clone()
    reversed_prediction[0] = target[1]
    reversed_prediction[1] = target[0]
    reversed_components = _components(reversed_prediction)
    assert (
        reversed_components.sensitivity_relative_delta_mse.item()
        == pytest.approx(4.0)
    )
    assert reversed_components.sensitivity_direction_loss.item() == (
        pytest.approx(2.0)
    )


def test_loss_components_score_orthogonal_and_null_effects_without_tiny_division(
) -> None:
    indexed = _indexed_batch()
    batch = indexed.batch
    simple_target = torch.zeros_like(batch.target_modes)
    simple_target[1, :, 0] = 1.0
    simple_target = torch.where(
        batch.valid_mask.unsqueeze(-1),
        simple_target,
        torch.zeros_like(simple_target),
    )
    simple_batch = IndexedReferenceBatch(
        batch=SyntheticReferenceBatch(
            split="fit",
            modal_coordinates=batch.modal_coordinates,
            null_coordinates=batch.null_coordinates,
            row_rms=batch.row_rms,
            target_modes=simple_target,
            logical_positions=batch.logical_positions,
            valid_mask=batch.valid_mask,
            synthetic_binding_sha256=_BINDING,
        ),
        endpoint_ids=_ENDPOINT_IDS,
    )
    data = _prepared_data(batch=simple_batch)

    orthogonal_prediction = simple_target.clone()
    orthogonal_prediction[1] = 0.0
    orthogonal_prediction[1, :, 1] = 1.0
    orthogonal = contrast_fit_module._loss_components(
        _FixedPredictionModel(orthogonal_prediction),
        data=data,
        target_center=torch.zeros(_WIDTH, dtype=torch.float64),
        target_scale=torch.ones(_WIDTH, dtype=torch.float64),
        metric_weight=torch.ones(_WIDTH, dtype=torch.float64),
        objective=_objective(),
    )
    assert orthogonal.sensitivity_relative_delta_mse.item() == pytest.approx(
        2.0
    )
    assert orthogonal.sensitivity_direction_loss.item() == pytest.approx(1.0)

    hallucinated = simple_target.clone()
    hallucinated[3] = 0.3
    null_components = contrast_fit_module._loss_components(
        _FixedPredictionModel(hallucinated),
        data=data,
        target_center=torch.zeros(_WIDTH, dtype=torch.float64),
        target_scale=torch.ones(_WIDTH, dtype=torch.float64),
        metric_weight=torch.ones(_WIDTH, dtype=torch.float64),
        objective=_objective(),
    )
    assert null_components.intended_null_absolute_mse.item() == pytest.approx(
        0.3**2
    )
    assert torch.isfinite(null_components.intended_null_absolute_mse)


def test_fisher_metric_weights_pointwise_errors_before_squaring() -> None:
    batch = _indexed_batch().batch
    prediction = batch.target_modes.clone()
    prediction[..., 0] += 1.0
    unweighted = _components(prediction)
    metric = torch.ones(_WIDTH, dtype=torch.float64)
    metric[0] = 2.0
    weighted = _components(prediction, metric_weight=metric)

    assert unweighted.pointwise_mse.item() == pytest.approx(1.0 / _WIDTH)
    assert weighted.pointwise_mse.item() == pytest.approx(4.0 / _WIDTH)
    assert weighted.pointwise_mse.item() == pytest.approx(
        4.0 * unweighted.pointwise_mse.item()
    )


def test_midpoint_jvp_component_matches_exact_tangent_and_relative_error() -> None:
    indexed = _indexed_batch()
    batch = indexed.batch
    endpoint_tangent = (
        batch.modal_coordinates[1] - batch.modal_coordinates[0]
    )
    # Deliberately disagree with endpoint arithmetic.  The exact zero below
    # proves that the supplied hidden-midpoint chart tangent is authoritative.
    tangent = 0.6 * endpoint_tangent.roll(shifts=9, dims=-1)
    chart = _provider_chart(
        modal_primal=(
            0.2 * batch.modal_coordinates[0]
            + 0.8 * batch.modal_coordinates[1]
        ),
        modal_tangent=tangent,
        null_primal=torch.full((4, 1), 0.1, dtype=torch.float64),
        row_rms_primal=torch.linspace(
            0.95,
            1.05,
            4,
            dtype=torch.float64,
        ),
    )
    exact_pair = ReferenceProviderContrastPair(
        pair_id="pair/sensitivity-jvp",
        family="signed",
        role="expected_sensitivity",
        left_endpoint_id=_ENDPOINT_IDS[0],
        right_endpoint_id=_ENDPOINT_IDS[1],
        rank_stratum="nonadjacent-low-high",
        teacher_midpoint_jvp=tangent,
        **chart,
    )
    null_pair = _pairs()[1]
    exact_data = _prepared_data(pairs=(exact_pair, null_pair))
    arguments = {
        "model": _ModalIdentityModel(),
        "data": exact_data,
        "target_center": torch.zeros(_WIDTH, dtype=torch.float64),
        "target_scale": torch.ones(_WIDTH, dtype=torch.float64),
        "metric_weight": torch.ones(_WIDTH, dtype=torch.float64),
        "objective": _objective(),
    }
    exact = contrast_fit_module._loss_components(**arguments)
    assert exact.midpoint_jvp_relative_mse.item() == pytest.approx(0.0)

    doubled_pair = ReferenceProviderContrastPair(
        pair_id="pair/sensitivity-jvp-double",
        family="signed",
        role="expected_sensitivity",
        left_endpoint_id=_ENDPOINT_IDS[0],
        right_endpoint_id=_ENDPOINT_IDS[1],
        rank_stratum="nonadjacent-low-high",
        teacher_midpoint_jvp=2.0 * tangent,
        **chart,
    )
    doubled_data = _prepared_data(pairs=(doubled_pair, null_pair))
    doubled = contrast_fit_module._loss_components(
        _ModalIdentityModel(),
        data=doubled_data,
        target_center=torch.zeros(_WIDTH, dtype=torch.float64),
        target_scale=torch.ones(_WIDTH, dtype=torch.float64),
        metric_weight=torch.ones(_WIDTH, dtype=torch.float64),
        objective=_objective(),
    )
    assert doubled.midpoint_jvp_relative_mse.item() == pytest.approx(0.25)


def test_same_length_batched_jvp_matches_scalar_outputs_and_parameter_gradients(
) -> None:
    indexed = _indexed_batch()
    source = indexed.batch
    masks = source.valid_mask.clone()
    positions = source.logical_positions.clone()
    masks[2:] = torch.tensor(
        [True, True, False, True],
        dtype=torch.bool,
    )
    positions[2:] = torch.tensor(
        [0, 2, -9, 7],
        dtype=torch.int64,
    )
    parity_batch = IndexedReferenceBatch(
        batch=SyntheticReferenceBatch(
            split="fit",
            modal_coordinates=source.modal_coordinates,
            null_coordinates=source.null_coordinates,
            row_rms=source.row_rms,
            target_modes=source.target_modes,
            logical_positions=positions,
            valid_mask=masks,
            synthetic_binding_sha256=_BINDING,
        ),
        endpoint_ids=_ENDPOINT_IDS,
    )
    base_tangent = (
        source.modal_coordinates[1] - source.modal_coordinates[0]
    )
    pair_endpoints = (
        (_ENDPOINT_IDS[0], _ENDPOINT_IDS[1]),
        (_ENDPOINT_IDS[2], _ENDPOINT_IDS[3]),
        (_ENDPOINT_IDS[0], _ENDPOINT_IDS[1]),
    )
    pairs: list[ReferenceProviderContrastPair] = []
    for index, (left_id, right_id) in enumerate(pair_endpoints):
        modal_tangent = (
            (0.35 + 0.2 * index)
            * base_tangent.roll(shifts=3 + 5 * index, dims=-1)
        )
        modal_primal = (
            0.45 * source.modal_coordinates[index]
            + 0.55 * source.modal_coordinates[index + 1]
            + 0.01 * (index + 1)
        )
        chart = _provider_chart(
            modal_primal=modal_primal,
            modal_tangent=modal_tangent,
            null_primal=torch.full(
                (4, 1),
                0.05 * (index + 1),
                dtype=torch.float64,
            ),
            row_rms_primal=torch.linspace(
                0.85 + 0.05 * index,
                1.10 + 0.05 * index,
                4,
                dtype=torch.float64,
            ),
            null_tangent=torch.full(
                (4, 1),
                -0.015 * (index + 1),
                dtype=torch.float64,
            ),
            row_rms_tangent=torch.linspace(
                -0.02,
                0.03,
                4,
                dtype=torch.float64,
            )
            * (index + 1),
        )
        teacher = (
            0.7 * modal_tangent
            + 0.03
            * torch.cos(
                torch.arange(_WIDTH, dtype=torch.float64)
            ).view(1, -1)
        )
        pairs.append(
            ReferenceProviderContrastPair(
                pair_id=f"pair/batched-jvp-{index}",
                family="signed",
                role="expected_sensitivity",
                left_endpoint_id=left_id,
                right_endpoint_id=right_id,
                rank_stratum=f"stratum-{index}",
                teacher_midpoint_jvp=teacher,
                **chart,
            )
        )
    data = _prepared_data(
        batch=parity_batch,
        pairs=tuple(pairs),
    )
    model_arguments = {
        "modal_center": torch.zeros(_WIDTH, dtype=torch.float64),
        "gain_log_center": 0.0,
        "gain_log_scale": 1.0,
        "residual_width": _WIDTH,
        "rms_epsilon": _RMS_EPSILON,
        "target_center": torch.zeros(_WIDTH, dtype=torch.float64),
        "target_scale": torch.ones(_WIDTH, dtype=torch.float64),
        "executor_config": _config(),
        "seed": 991,
    }
    scalar_model = contrast_fit_module._PackedTrainingModule(
        **model_arguments
    )
    batched_model = contrast_fit_module._PackedTrainingModule(
        **model_arguments
    )
    target_scale = torch.linspace(
        0.8,
        1.2,
        _WIDTH,
        dtype=torch.float64,
    )
    metric_weight = torch.linspace(
        0.6,
        1.4,
        _WIDTH,
        dtype=torch.float64,
    )
    common = {
        "data": data,
        "target_scale": target_scale,
        "metric_weight": metric_weight,
        "objective": _objective(),
    }
    scalar_losses = contrast_fit_module._midpoint_jvp_losses(
        scalar_model,
        batch_same_length=False,
        **common,
    )
    batched_losses = contrast_fit_module._midpoint_jvp_losses(
        batched_model,
        batch_same_length=True,
        **common,
    )
    assert len(scalar_losses) == len(batched_losses) == len(pairs)
    torch.testing.assert_close(
        torch.stack(batched_losses),
        torch.stack(scalar_losses),
        rtol=1e-11,
        atol=1e-12,
    )

    scalar_total = torch.stack(scalar_losses).mean()
    batched_total = torch.stack(batched_losses).mean()
    scalar_total.backward()
    batched_total.backward()
    scalar_parameters = dict(scalar_model.named_parameters())
    batched_parameters = dict(batched_model.named_parameters())
    assert scalar_parameters.keys() == batched_parameters.keys()
    for name in scalar_parameters:
        scalar_gradient = scalar_parameters[name].grad
        batched_gradient = batched_parameters[name].grad
        assert (scalar_gradient is None) == (batched_gradient is None), name
        if scalar_gradient is not None:
            torch.testing.assert_close(
                batched_gradient,
                scalar_gradient,
                rtol=1e-10,
                atol=1e-12,
                msg=lambda message, parameter=name: (
                    f"{parameter} gradient mismatch: {message}"
                ),
            )


def test_storage_and_execution_accounting_include_packer_and_unpacker() -> None:
    plan = _fit(steps=1)
    accounting = plan.accounting()
    assert accounting.modal_modes == accounting.target_modes == _WIDTH
    assert accounting.latent_rank == plan.rank
    assert accounting.encoder_parameter_count == _WIDTH * plan.rank
    assert accounting.decoder_parameter_count == plan.rank * _WIDTH
    assert accounting.total_stored_scalar_count == (
        accounting.feature_codec_scalar_count
        + accounting.target_standardization_scalar_count
        + accounting.encoder_parameter_count
        + accounting.executor_parameter_count
        + accounting.decoder_parameter_count
    )

    runtime = plan.prepare(dtype=torch.float64)
    batch = _indexed_batch().batch
    execution = runtime.execution_accounting(
        valid_mask=batch.valid_mask,
        logical_positions=batch.logical_positions,
    )
    rows = batch.valid_row_count
    assert execution.valid_rows == rows
    assert execution.encoder_mac_count == rows * _WIDTH * plan.rank
    assert execution.decoder_mac_count == rows * plan.rank * _WIDTH
    assert execution.target_destandardization_mac_count == rows * _WIDTH
    assert execution.total_mac_count == (
        execution.core.total_mac_count
        + execution.encoder_mac_count
        + execution.decoder_mac_count
        + execution.target_destandardization_mac_count
    )
