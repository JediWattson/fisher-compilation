from __future__ import annotations

import copy
from dataclasses import replace
import json

import pytest
import torch

from fisher_graph.gemma3_layer_multimode_capture import (
    A_DEVELOPMENT_ROLE,
    B3_DISCOVERY_ROLE,
    FreshRoleCaptureNotAuthorizedError,
    LayerMLPFlowRows,
    PREDECESSOR_A_ROLE,
)
from fisher_graph.gemma3_layer_multimode_capture import _tensor_sha256
from fisher_graph.gemma3_layer_multimode_fit import (
    AtlasSupportError,
    LayerMultimodeAtlasFit,
    LayerMultimodeCandidateFitReceipt,
    LayerMultimodeSplitRefitEvidence,
    LayerMultimodeValidationEvidence,
    evaluate_layer_multimode_split_refit,
    fit_layer_chart_atlas,
    fit_layer_multimode_candidate,
    layer_multimode_candidate_evaluation_from_fit,
    make_layer_multimode_test_policy,
)
from fisher_graph.gemma3_layer_multimode_fit import (
    _clip_multimode_parameter_gradients,
)
from fisher_graph.gemma3_layer_multimode_flow import (
    FixedLayerChartAtlas,
    layer_multimode_flow_loss,
)
from fisher_graph.gemma3_layer_multimode_selection import (
    LayerMultimodeCandidateEvaluationReceipt,
)


def _sha(value: int) -> str:
    return f"{value:064x}"


_FIT_A_PARTITION_SHA256 = _sha(698)
_VALIDATION_A_PARTITION_SHA256 = _sha(699)


def _policy(*, steps: int = 2):
    return make_layer_multimode_test_policy(
        fit_target_token_counts=(4, 6),
        validation_target_token_counts=(5, 7),
        optimizer_steps=steps,
        minibatch_size=32,
    )


def _rows(
    *,
    validation_shift: float = 0.0,
    collapsed: bool = False,
    family_count: int = 2,
    high_rank: bool = False,
) -> LayerMLPFlowRows:
    families: list[str] = []
    examples: list[str] = []
    token_counts: list[int] = []
    ranges: list[tuple[int, int]] = []
    row_keys: list[tuple[str, str, int, int]] = []
    sources: list[str] = []
    state_rows: list[torch.Tensor] = []
    velocity_rows: list[torch.Tensor] = []
    cursor = 0
    ordinal = 0
    family_catalog = (
        "family.alpha",
        "family.beta",
        "family.delta",
        "family.epsilon",
        "family.gamma",
        "family.zeta",
    )[:family_count]
    for family_ordinal, family in enumerate(family_catalog):
        for length in (4, 5, 6, 7):
            example = f"{family}.length-{length}"
            families.append(family)
            examples.append(example)
            token_counts.append(length)
            ranges.append((cursor, cursor + length - 1))
            sources.append(_sha(100 + ordinal))
            cluster = -6.0 if family_ordinal == 0 else 6.0
            middle = (length - 2) / 2.0
            for query in range(length - 1):
                row_keys.append((family, example, query, query + 1))
                state = torch.zeros(640, dtype=torch.float32)
                local = float(query) - middle
                state[0] = cluster
                if high_rank:
                    global_row = cursor + query + 1
                    for dimension in range(1, 21):
                        state[dimension] = (
                            torch.sin(
                                torch.tensor(
                                    global_row * (dimension + 1) * 0.173
                                )
                            )
                            + torch.cos(
                                torch.tensor(
                                    (ordinal + 1) * (dimension + 3) * 0.119
                                )
                            )
                        )
                elif collapsed:
                    state[1] = local
                    state[2] = 2.0 * local
                else:
                    state[1] = local
                    state[2] = local * local - (middle * middle / 2.0)
                    state[3] = float((query + family_ordinal) % 2) - 0.5
                velocity = torch.zeros(640, dtype=torch.float32)
                slope = 0.35 if local < 0.0 else 0.75
                if high_rank:
                    for dimension in range(1, 17):
                        velocity[dimension] = (
                            slope * state[dimension]
                            + 0.1 * state[dimension + 1]
                        )
                else:
                    velocity[1] = slope * state[1] + 0.2 * state[2]
                    velocity[2] = -0.15 * state[1] + (0.5 - slope) * state[2]
                    velocity[3] = 0.1 * state[3]
                if length in (5, 7):
                    velocity[1] += validation_shift
                state_rows.append(state)
                velocity_rows.append(velocity)
            cursor += length - 1
            ordinal += 1
    states = torch.stack(state_rows).contiguous()
    velocities = torch.stack(velocity_rows).contiguous()
    return LayerMLPFlowRows(
        role=A_DEVELOPMENT_ROLE,
        layer_ordinal=3,
        mode_id="mode.mutation.8",
        family_ids=tuple(families),
        example_ids=tuple(examples),
        token_counts=tuple(token_counts),
        row_keys=tuple(row_keys),
        example_row_ranges=tuple(ranges),
        chart_states=states,
        teacher_velocities=velocities,
        predecessor_source_role=PREDECESSOR_A_ROLE,
        source_batch_lexical_nll_receipt_sha256=_sha(90),
        source_batch_execution_receipt_sha256=_sha(91),
        source_capture_receipt_sha256s=tuple(sources),
        chart_state_sha256=_tensor_sha256(states),
        teacher_velocity_sha256=_tensor_sha256(velocities),
    )


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for child in value.values():
            _assert_tensor_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_tensor_free(child)


def test_test_policy_is_loudly_non_authorizing_and_A_only() -> None:
    rows = _rows()
    policy = _policy(steps=1)
    with pytest.raises(ValueError, match="explicit opt-in"):
        fit_layer_chart_atlas(
            rows=rows,
            chart_count=1,
            tangent_rank=2,
            policy=policy,
        )
    with pytest.raises(FreshRoleCaptureNotAuthorizedError):
        replace(
            rows,
            role=B3_DISCOVERY_ROLE,
            receipt_sha256="",
        )


def test_production_fit_rejects_single_batch_development_rows() -> None:
    with pytest.raises(ValueError, match="full-A spool bridge"):
        fit_layer_chart_atlas(
            rows=_rows(),
            chart_count=1,
            tangent_rank=16,
        )


def test_atlas_is_deterministic_full_rank_and_temperature_safe() -> None:
    rows = _rows()
    policy = _policy(steps=1)
    first = fit_layer_chart_atlas(
        rows=rows,
        chart_count=2,
        tangent_rank=2,
        policy=policy,
        allow_test_policy=True,
    )
    second = fit_layer_chart_atlas(
        rows=rows,
        chart_count=2,
        tangent_rank=2,
        policy=policy,
        allow_test_policy=True,
    )
    assert first.atlas.atlas_sha256 == second.atlas.atlas_sha256
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
    assert first.live_atlas_fitter_authorized is True
    assert replace(first).live_atlas_fitter_authorized is False
    assert copy.copy(first).live_atlas_fitter_authorized is False
    assert copy.deepcopy(first).live_atlas_fitter_authorized is False
    assert first.receipt.hard_chart_counts == (8, 8)
    assert min(first.receipt.covariance_numerical_ranks) >= 2
    assert (
        first.receipt.chart_temperature_lower_bound
        < first.receipt.initial_chart_temperature
    )
    assert (
        first.receipt.initial_chart_temperature
        < first.receipt.chart_temperature_upper_bound
    )
    reused = tuple(
        fit_layer_multimode_candidate(
            rows=rows,
            chart_count=2,
            modes_per_chart=mode_count,
            tangent_rank=2,
            policy=policy,
            allow_test_policy=True,
            atlas_fit=first,
        )
        for mode_count in (1, 2, 4)
    )
    assert all(value.live_fitter_authorized for value in reused)
    assert all(value.atlas_fit is first for value in reused)
    assert {
        value.receipt.atlas_fit.atlas_sha256 for value in reused
    } == {first.atlas.atlas_sha256}
    assert {
        value.receipt.fit_source_sha256 for value in reused
    } == {first.receipt.fit_source_sha256}

    single = fit_layer_chart_atlas(
        rows=rows,
        chart_count=1,
        tangent_rank=2,
        policy=policy,
        allow_test_policy=True,
    )
    assert (
        single.receipt.chart_temperature_lower_bound,
        single.receipt.initial_chart_temperature,
        single.receipt.chart_temperature_upper_bound,
    ) == (1.0, 1.0, 1.0)

    forged_centers = first.atlas.centers.clone()
    forged_centers[0, 0] += 0.125
    forged_atlas = FixedLayerChartAtlas(
        centers=forged_centers.contiguous(),
        bases=first.atlas.bases.clone().contiguous(),
        source_fit_sha256=first.atlas.source_fit_sha256,
    )
    forged_atlas_receipt = replace(
        first.receipt,
        atlas_sha256=forged_atlas.atlas_sha256,
        receipt_sha256="",
    )
    forged_atlas_fit = LayerMultimodeAtlasFit(
        atlas=forged_atlas,
        receipt=forged_atlas_receipt,
    )
    forged_atlas_fit.assert_integrity()
    assert forged_atlas_fit.live_atlas_fitter_authorized is False
    with pytest.raises(ValueError, match="original live deterministic"):
        fit_layer_multimode_candidate(
            rows=rows,
            chart_count=2,
            modes_per_chart=1,
            tangent_rank=2,
            policy=policy,
            allow_test_policy=True,
            atlas_fit=forged_atlas_fit,
        )


def test_fit_is_deterministic_M1_safe_and_validation_cannot_change_parameters() -> None:
    policy = _policy()
    first = fit_layer_multimode_candidate(
        rows=_rows(),
        chart_count=1,
        modes_per_chart=1,
        tangent_rank=2,
        policy=policy,
        allow_test_policy=True,
    )
    second = fit_layer_multimode_candidate(
        rows=_rows(),
        chart_count=1,
        modes_per_chart=1,
        tangent_rank=2,
        policy=policy,
        allow_test_policy=True,
    )
    shifted_validation = fit_layer_multimode_candidate(
        rows=_rows(validation_shift=3.0),
        chart_count=1,
        modes_per_chart=1,
        tangent_rank=2,
        policy=policy,
        allow_test_policy=True,
    )
    assert first.receipt.receipt_sha256 == second.receipt.receipt_sha256
    assert (
        first.receipt.model_parameter_sha256
        == second.receipt.model_parameter_sha256
    )
    assert (
        first.receipt.model_parameter_sha256
        == shifted_validation.receipt.model_parameter_sha256
    )
    assert (
        first.receipt.fit_source_sha256
        == shifted_validation.receipt.fit_source_sha256
    )
    assert (
        first.receipt.validation_source_sha256
        != shifted_validation.receipt.validation_source_sha256
    )
    assert first.receipt.retained_rmsnorm_vector_count == 1
    assert first.receipt.output_rmsnorm_folded_into_teacher_velocity is True
    assert first.receipt.final_chart_temperature == 1.0
    assert first.receipt.chart_temperature_fitted_on_A is False
    assert first.live_fitter_authorized is True
    assert replace(first).live_fitter_authorized is False
    assert copy.copy(first).live_fitter_authorized is False
    assert copy.deepcopy(first).live_fitter_authorized is False
    evidence = first.receipt.validation
    assert evidence.chart_soft_masses == (1.0,)
    assert evidence.conditional_mode_masses == ((1.0,),)
    assert evidence.per_chart_mean_row_router_entropy_nats == (0.0,)
    assert evidence.per_chart_state_mode_mutual_information_nats == (0.0,)
    assert evidence.per_chart_minimum_pairwise_velocity_separation_ratio == (0.0,)


def test_multimode_evidence_is_chart_conditional_not_pooled() -> None:
    rows = _rows()
    fitted = fit_layer_multimode_candidate(
        rows=rows,
        chart_count=2,
        modes_per_chart=2,
        tangent_rank=2,
        policy=_policy(),
        allow_test_policy=True,
    )
    evidence = fitted.receipt.validation
    assert len(evidence.chart_soft_masses) == 2
    assert sum(evidence.chart_soft_masses) == pytest.approx(1.0)
    assert len(evidence.conditional_mode_masses) == 2
    for masses in evidence.conditional_mode_masses:
        assert len(masses) == 2
        assert sum(masses) == pytest.approx(1.0)
        assert min(masses) > 0.0
    assert len(evidence.per_chart_state_mode_mutual_information_nats) == 2
    assert len(evidence.per_chart_minimum_pairwise_velocity_separation_rms) == 2
    assert fitted.receipt.chart_temperature_fitted_on_A is True
    lower = fitted.atlas_fit.receipt.chart_temperature_lower_bound
    upper = fitted.atlas_fit.receipt.chart_temperature_upper_bound
    assert lower <= fitted.receipt.final_chart_temperature <= upper
    x = rows.chart_states[:8].to(torch.float64).contiguous()
    teacher = rows.teacher_velocities[:8].to(torch.float64).contiguous()
    result = fitted.model(
        chart_inputs=x,
        residual_states=torch.zeros_like(x),
        source_capture_sha256=_sha(999),
    )
    loss = layer_multimode_flow_loss(
        result=result,
        teacher_velocity=teacher,
        load_balance_coefficient=1.0,
        sparsity_coefficient=1.0,
        distinctness_coefficient=1.0,
    )
    for term in (loss.load_balance, loss.sparsity, loss.distinctness):
        gradient = torch.autograd.grad(
            term,
            fitted.model.log_chart_temperature,
            retain_graph=True,
            allow_unused=True,
        )[0]
        assert gradient is None or torch.equal(gradient, torch.zeros_like(gradient))
    assert (
        fitted.receipt.validation.sufficient_statistics_weighting_id
        == "family_example_balanced_normalized_row_weights_times_validation_N.v1"
    )


def test_regularizer_gradients_cannot_rescale_one_step_tau_update() -> None:
    rows = _rows()
    fitted = fit_layer_multimode_candidate(
        rows=rows,
        chart_count=2,
        modes_per_chart=2,
        tangent_rank=2,
        policy=_policy(steps=1),
        allow_test_policy=True,
    )
    models = (copy.deepcopy(fitted.model), copy.deepcopy(fitted.model))
    optimizers = tuple(
        torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.003,
            betas=(0.9, 0.99),
            eps=1.0e-8,
        )
        for model in models
    )
    x = rows.chart_states[:8].to(torch.float64).contiguous()
    teacher = rows.teacher_velocities[:8].to(torch.float64).contiguous()
    raw_tau_gradients: list[torch.Tensor] = []
    clipped_tau_gradients: list[torch.Tensor] = []
    non_tau_norms: list[float] = []
    for ordinal, (model, optimizer, regularizer_scale) in enumerate(
        zip(models, optimizers, (0.0, 1.0e12), strict=True)
    ):
        result = model(
            chart_inputs=x,
            residual_states=torch.zeros_like(x),
            source_capture_sha256=_sha(800 + ordinal),
        )
        loss = layer_multimode_flow_loss(
            result=result,
            teacher_velocity=teacher,
            load_balance_coefficient=regularizer_scale,
            sparsity_coefficient=regularizer_scale,
            distinctness_coefficient=regularizer_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        assert model.log_chart_temperature.grad is not None
        raw_tau_gradients.append(model.log_chart_temperature.grad.detach().clone())
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        non_tau_norm, _ = _clip_multimode_parameter_gradients(
            model,
            trainable,
            max_norm=5.0,
        )
        non_tau_norms.append(float(non_tau_norm))
        clipped_tau_gradients.append(
            model.log_chart_temperature.grad.detach().clone()
        )

    assert torch.equal(raw_tau_gradients[0], raw_tau_gradients[1])
    assert non_tau_norms[1] > non_tau_norms[0]
    assert non_tau_norms[1] > 5.0
    assert torch.equal(clipped_tau_gradients[0], clipped_tau_gradients[1])
    for optimizer in optimizers:
        optimizer.step()
    assert torch.equal(
        models[0].log_chart_temperature,
        models[1].log_chart_temperature,
    )


def test_receipt_roundtrip_is_json_safe_strict_and_rehashed() -> None:
    fitted = fit_layer_multimode_candidate(
        rows=_rows(),
        chart_count=1,
        modes_per_chart=1,
        tangent_rank=2,
        policy=_policy(steps=1),
        allow_test_policy=True,
    )
    state = json.loads(json.dumps(fitted.receipt.state_dict()))
    _assert_tensor_free(state)
    restored = LayerMultimodeCandidateFitReceipt.from_state_dict(state)
    assert restored.receipt_sha256 == fitted.receipt.receipt_sha256
    assert restored.fit_policy_test_only is True
    assert restored.production_fit_authorized is False
    assert restored.source_capture_scope == (
        "development_single_CapturedBatch_only"
    )
    assert restored.gradient_clip_grouping_id == (
        "non_tau_global_l2_and_chart_temperature_scalar_independent.v1"
    )
    assert LayerMultimodeValidationEvidence.from_state_dict(
        state["validation"]
    ).evidence_sha256 == fitted.receipt.validation.evidence_sha256

    bad_bool = json.loads(json.dumps(state))
    bad_bool["layer_ordinal"] = True
    with pytest.raises(TypeError):
        LayerMultimodeCandidateFitReceipt.from_state_dict(bad_bool)
    bad_number_type = json.loads(json.dumps(state))
    bad_number_type["mode_temperature"] = 1
    with pytest.raises(TypeError):
        LayerMultimodeCandidateFitReceipt.from_state_dict(bad_number_type)
    bad_list = json.loads(json.dumps(state))
    bad_list["validation"]["chart_soft_masses"] = tuple(
        bad_list["validation"]["chart_soft_masses"]
    )
    with pytest.raises(TypeError, match="JSON list"):
        LayerMultimodeCandidateFitReceipt.from_state_dict(bad_list)
    bad_hash = json.loads(json.dumps(state))
    bad_hash["validation"]["velocity_sse"] += 1.0
    with pytest.raises(ValueError):
        LayerMultimodeCandidateFitReceipt.from_state_dict(bad_hash)
    unknown = json.loads(json.dumps(state))
    unknown["unexpected"] = False
    with pytest.raises(ValueError, match="fields"):
        LayerMultimodeCandidateFitReceipt.from_state_dict(unknown)
    forged_authority = json.loads(json.dumps(state))
    forged_authority["production_fit_authorized"] = True
    forged_authority["receipt_sha256"] = ""
    with pytest.raises(ValueError, match="does not recompute"):
        LayerMultimodeCandidateFitReceipt.from_state_dict(forged_authority)
    grouping_drift = json.loads(json.dumps(state))
    grouping_drift["gradient_clip_grouping_id"] = "shared_global_clip.v0"
    grouping_drift["receipt_sha256"] = ""
    with pytest.raises(ValueError, match="grouping algorithm drifted"):
        LayerMultimodeCandidateFitReceipt.from_state_dict(grouping_drift)


def test_rank_and_chart_collapse_are_rejected_before_optimizer() -> None:
    with pytest.raises(AtlasSupportError, match="covariance rank"):
        fit_layer_chart_atlas(
            rows=_rows(collapsed=True),
            chart_count=1,
            tangent_rank=3,
            policy=_policy(steps=1),
            allow_test_policy=True,
        )
    with pytest.raises(AtlasSupportError, match=r"R\+1"):
        fit_layer_chart_atlas(
            rows=_rows(),
            chart_count=2,
            tangent_rank=8,
            policy=_policy(steps=1),
            allow_test_policy=True,
        )


def test_split_refit_checkerboard_is_bound_and_label_invariant() -> None:
    evidence = evaluate_layer_multimode_split_refit(
        rows=_rows(),
        chart_count=2,
        modes_per_chart=2,
        tangent_rank=2,
        policy=_policy(steps=1),
        allow_test_policy=True,
    )
    assert -1.0 <= evidence.matched_prediction_cosine <= 1.0
    assert -1.0 <= evidence.hard_route_adjusted_rand_index <= 1.0
    assert evidence.partition_mask_sha256s[0] != evidence.partition_mask_sha256s[1]
    assert evidence.base_fit_row_count == 16
    assert evidence.partition_row_counts == (8, 8)
    assert evidence.base_fit_example_count == 4
    assert evidence.partition_example_counts == (2, 2)
    assert evidence.intersection_row_count == 0
    assert evidence.union_equals_base_fit is True
    assert evidence.partition_union_mask_sha256 == evidence.base_fit_mask_sha256
    assert evidence.live_split_evaluator_authorized is True
    assert replace(evidence).live_split_evaluator_authorized is False
    assert copy.copy(evidence).live_split_evaluator_authorized is False
    assert copy.deepcopy(evidence).live_split_evaluator_authorized is False
    state = json.loads(json.dumps(evidence.state_dict()))
    restored = LayerMultimodeSplitRefitEvidence.from_state_dict(state)
    assert restored.evidence_sha256 == evidence.evidence_sha256
    assert restored.live_split_evaluator_authorized is False
    tampered = json.loads(json.dumps(state))
    tampered["matched_prediction_cosine"] = 0.0
    with pytest.raises(ValueError, match="hash mismatch"):
        LayerMultimodeSplitRefitEvidence.from_state_dict(tampered)
    for field in (
        "partition_mask_sha256s",
        "partition_example_catalog_sha256s",
        "partition_fit_receipt_sha256s",
        "partition_model_parameter_sha256s",
    ):
        duplicate = json.loads(json.dumps(state))
        duplicate[field][1] = duplicate[field][0]
        duplicate["evidence_sha256"] = ""
        with pytest.raises(ValueError, match="split halves must be distinct"):
            LayerMultimodeSplitRefitEvidence.from_state_dict(duplicate)
    bad_counts = json.loads(json.dumps(state))
    bad_counts["partition_row_counts"][1] -= 1
    bad_counts["evidence_sha256"] = ""
    with pytest.raises(ValueError, match="row counts do not close"):
        LayerMultimodeSplitRefitEvidence.from_state_dict(bad_counts)
    bad_intersection = json.loads(json.dumps(state))
    bad_intersection["intersection_row_count"] = 1
    bad_intersection["evidence_sha256"] = ""
    with pytest.raises(ValueError, match="intersection must be empty"):
        LayerMultimodeSplitRefitEvidence.from_state_dict(bad_intersection)
    bad_union = json.loads(json.dumps(state))
    bad_union["union_equals_base_fit"] = False
    bad_union["evidence_sha256"] = ""
    with pytest.raises(ValueError, match="union must equal base fit"):
        LayerMultimodeSplitRefitEvidence.from_state_dict(bad_union)


def test_selector_adapter_M1_emits_exact_sentinels_and_real_parser_roundtrips() -> None:
    fitted = fit_layer_multimode_candidate(
        rows=_rows(family_count=6, high_rank=True),
        chart_count=1,
        modes_per_chart=1,
        tangent_rank=16,
        policy=_policy(steps=1),
        allow_test_policy=True,
    )
    candidate = layer_multimode_candidate_evaluation_from_fit(
        fitted,
        source_model_sha256=_sha(700),
        fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
        validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
        evaluation_protocol_sha256=_sha(701),
    )
    assert candidate.split_refit_matched_prediction_cosine == 1.0
    assert candidate.split_refit_hard_route_ari == 1.0
    assert candidate.fit_a_sha256 == _FIT_A_PARTITION_SHA256
    assert candidate.validation_a_sha256 == _VALIDATION_A_PARTITION_SHA256
    assert candidate.fit_a_sha256 != fitted.receipt.fit_source_sha256
    assert candidate.fit_policy_sha256 == fitted.receipt.policy_sha256
    assert candidate.fit_policy_test_only is True
    assert candidate.source_capture_scope == fitted.receipt.source_capture_scope
    assert candidate.production_fit_authorized is False
    assert candidate.live_candidate_evaluation_authorized is True
    assert replace(candidate).live_candidate_evaluation_authorized is False
    assert copy.copy(candidate).live_candidate_evaluation_authorized is False
    assert copy.deepcopy(candidate).live_candidate_evaluation_authorized is False
    assert len(candidate.family_fidelity) == 6
    assert candidate.serialized_parameter_count == (
        fitted.receipt.candidate_total_parameter_count
    )
    state = json.loads(json.dumps(candidate.state_dict()))
    restored = LayerMultimodeCandidateEvaluationReceipt.from_state_dict(state)
    assert restored.artifact_sha256 == candidate.artifact_sha256
    assert restored.live_candidate_evaluation_authorized is False
    with pytest.raises(ValueError, match="partitions must be distinct"):
        layer_multimode_candidate_evaluation_from_fit(
            fitted,
            source_model_sha256=_sha(700),
            fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
            validation_a_partition_sha256=_FIT_A_PARTITION_SHA256,
            evaluation_protocol_sha256=_sha(701),
        )


def test_selector_adapter_rejects_copied_or_coordinated_relabelled_fit() -> None:
    fitted = fit_layer_multimode_candidate(
        rows=_rows(family_count=6, high_rank=True),
        chart_count=1,
        modes_per_chart=1,
        tangent_rank=16,
        policy=_policy(steps=1),
        allow_test_policy=True,
    )
    copied = replace(fitted)
    assert copied.live_fitter_authorized is False
    arguments = {
        "source_model_sha256": _sha(720),
        "fit_a_partition_sha256": _FIT_A_PARTITION_SHA256,
        "validation_a_partition_sha256": _VALIDATION_A_PARTITION_SHA256,
        "evaluation_protocol_sha256": _sha(721),
    }
    with pytest.raises(ValueError, match="nonlive test fit"):
        layer_multimode_candidate_evaluation_from_fit(copied, **arguments)
    nonproduction = layer_multimode_candidate_evaluation_from_fit(
        copied,
        **arguments,
        allow_test_only_nonlive_authority=True,
    )
    assert nonproduction.production_fit_authorized is False

    foreign_rows = _sha(722)
    relabelled_atlas_receipt = replace(
        fitted.atlas_fit.receipt,
        source_rows_receipt_sha256=foreign_rows,
        receipt_sha256="",
    )
    relabelled_atlas_fit = replace(
        fitted.atlas_fit,
        receipt=relabelled_atlas_receipt,
    )
    relabelled_receipt = replace(
        fitted.receipt,
        source_rows_receipt_sha256=foreign_rows,
        atlas_fit=relabelled_atlas_receipt,
        receipt_sha256="",
    )
    relabelled = replace(
        fitted,
        atlas_fit=relabelled_atlas_fit,
        receipt=relabelled_receipt,
    )
    relabelled.assert_integrity()
    assert relabelled.live_fitter_authorized is False
    with pytest.raises(ValueError, match="nonlive test fit"):
        layer_multimode_candidate_evaluation_from_fit(relabelled, **arguments)


def test_selector_adapter_requires_cross_bound_M_gt_1_split_evidence() -> None:
    rows = _rows(family_count=6, high_rank=True)
    policy = _policy(steps=1)
    fitted = fit_layer_multimode_candidate(
        rows=rows,
        chart_count=1,
        modes_per_chart=2,
        tangent_rank=16,
        policy=policy,
        allow_test_policy=True,
    )
    with pytest.raises(ValueError, match="requires split-refit"):
        layer_multimode_candidate_evaluation_from_fit(
            fitted,
            source_model_sha256=_sha(710),
            fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
            validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
            evaluation_protocol_sha256=_sha(711),
        )
    split = evaluate_layer_multimode_split_refit(
        rows=rows,
        chart_count=1,
        modes_per_chart=2,
        tangent_rank=16,
        policy=policy,
        allow_test_policy=True,
    )
    candidate = layer_multimode_candidate_evaluation_from_fit(
        fitted,
        source_model_sha256=_sha(710),
        fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
        validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
        evaluation_protocol_sha256=_sha(711),
        split_refit=split,
    )
    copied_split = replace(split)
    assert copied_split.live_split_evaluator_authorized is False
    with pytest.raises(ValueError, match="nonlive test split evidence"):
        layer_multimode_candidate_evaluation_from_fit(
            fitted,
            source_model_sha256=_sha(710),
            fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
            validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
            evaluation_protocol_sha256=_sha(711),
            split_refit=copied_split,
        )
    copied_candidate = layer_multimode_candidate_evaluation_from_fit(
        fitted,
        source_model_sha256=_sha(710),
        fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
        validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
        evaluation_protocol_sha256=_sha(711),
        split_refit=copied_split,
        allow_test_only_nonlive_authority=True,
    )
    assert copied_candidate.production_fit_authorized is False
    restored = LayerMultimodeCandidateEvaluationReceipt.from_state_dict(
        json.loads(json.dumps(candidate.state_dict()))
    )
    assert restored.artifact_sha256 == candidate.artifact_sha256
    coordinated_rehash = replace(
        split,
        policy_sha256=_sha(712),
        evidence_sha256="",
    )
    with pytest.raises(ValueError, match="sources/policy"):
        layer_multimode_candidate_evaluation_from_fit(
            fitted,
            source_model_sha256=_sha(710),
            fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
            validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
            evaluation_protocol_sha256=_sha(711),
            split_refit=coordinated_rehash,
            allow_test_only_nonlive_authority=True,
        )
    wrong_validation_rehash = replace(
        split,
        validation_source_sha256=_sha(713),
        evidence_sha256="",
    )
    with pytest.raises(ValueError, match="sources/policy"):
        layer_multimode_candidate_evaluation_from_fit(
            fitted,
            source_model_sha256=_sha(710),
            fit_a_partition_sha256=_FIT_A_PARTITION_SHA256,
            validation_a_partition_sha256=_VALIDATION_A_PARTITION_SHA256,
            evaluation_protocol_sha256=_sha(711),
            split_refit=wrong_validation_rehash,
            allow_test_only_nonlive_authority=True,
        )
