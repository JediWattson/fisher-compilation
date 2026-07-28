import copy
from dataclasses import fields

import pytest
import torch

from fisher_graph.gated_executor import (
    GatedCausalModalExecutorConfig,
    ResidualGatedCausalModalExecutor,
)
import fisher_graph.state_conditioned_reference_provider as provider_module
from fisher_graph.state_conditioned_reference_provider import (
    ReferenceProviderFeatureCodec,
    StateConditionedReferenceProviderPlan,
    StateConditionedReferenceProviderSelectionGates,
    SyntheticReferenceBatch,
    compile_state_conditioned_reference_provider_ladder,
    evaluate_state_conditioned_reference_provider,
    fit_state_conditioned_reference_provider,
)


_SOURCE = "a" * 64
_SYNTHETIC = "b" * 64


def _codec() -> ReferenceProviderFeatureCodec:
    return ReferenceProviderFeatureCodec(
        modal_center=torch.tensor([1.0, -1.0]),
        modal_whitener=torch.tensor([[2.0, 0.0], [0.0, 0.5]]),
        null_center=torch.tensor([0.25]),
        null_scale=torch.tensor([0.5]),
        log_rms_center=0.0,
        log_rms_scale=2.0,
        source_binding_sha256=_SOURCE,
    )


def _batch(
    split: str,
    *,
    offset: float = 0.0,
    all_valid: bool = False,
) -> SyntheticReferenceBatch:
    modal = torch.tensor(
        [
            [
                [1.0 + offset, -1.0],
                [2.0 + offset, 0.0],
                [3.0 + offset, 1.0],
                [4.0 + offset, 2.0],
            ]
        ],
        dtype=torch.float64,
    )
    null = torch.tensor(
        [[[0.25], [0.75], [-0.25], [1.25]]],
        dtype=torch.float64,
    ) + 0.1 * offset
    row_rms = torch.tensor(
        [[1.0, 0.0 if not all_valid else 1.5, 2.0, 4.0]],
        dtype=torch.float64,
    )
    valid = (
        torch.ones(1, 4, dtype=torch.bool)
        if all_valid
        else torch.tensor([[True, False, True, True]])
    )
    positions = torch.tensor([[0, -9, 3, 7]])
    if all_valid:
        positions = torch.tensor([[0, 1, 3, 7]])
    safe_rms = torch.where(valid, row_rms, torch.ones_like(row_rms))
    target = torch.stack(
        (
            0.4 * modal[..., 0]
            + 1.2 * null[..., 0]
            + 0.2 * torch.log(safe_rms),
            -0.3 * modal[..., 1] + 0.7 * null[..., 0],
        ),
        dim=-1,
    )
    return SyntheticReferenceBatch(
        split=split,
        modal_coordinates=modal,
        null_coordinates=null,
        row_rms=row_rms,
        target_modes=target,
        logical_positions=positions,
        valid_mask=valid,
        synthetic_binding_sha256=_SYNTHETIC,
    )


def _config(
    *,
    experts: int = 1,
    rank: int = 1,
    router_width: int = 1,
    source_normalized_routing: bool = False,
) -> GatedCausalModalExecutorConfig:
    return GatedCausalModalExecutorConfig(
        input_modes=_codec().feature_modes,
        output_modes=2,
        expert_count=experts,
        expert_rank=rank,
        router_width=router_width,
        same_position_skip=False,
        source_normalized_routing=source_normalized_routing,
    )


def _fit(
    *,
    config: GatedCausalModalExecutorConfig | None = None,
    steps: int = 3,
) -> StateConditionedReferenceProviderPlan:
    return fit_state_conditioned_reference_provider(
        feature_codec=_codec(),
        target_center=torch.tensor([0.5, -0.25]),
        target_scale=torch.tensor([2.0, 0.5]),
        fit_batches=(_batch("fit"),),
        executor_config=_config() if config is None else config,
        steps=steps,
        learning_rate=0.01,
        seed=73,
    )


def _manual_null_provider() -> StateConditionedReferenceProviderPlan:
    codec = _codec()
    batch = _batch("fit")
    executor = ResidualGatedCausalModalExecutor(
        _config(),
        dtype=torch.float64,
    )
    with torch.no_grad():
        for parameter in executor.parameters():
            parameter.zero_()
        # Feature order: constant, two modal coordinates, null, log RMS.
        executor.same_position_weight[3, 0] = 1.0
    return StateConditionedReferenceProviderPlan(
        feature_codec=codec,
        target_center=torch.zeros(2),
        target_scale=torch.ones(2),
        executor_artifact=executor.artifact_state_dict(),
        synthetic_binding_sha256=_SYNTHETIC,
        fit_batch_sha256s=(batch.artifact_sha256,),
        fit_batch_content_sha256s=(batch.content_sha256,),
        training_steps=1,
        learning_rate=0.01,
        seed=0,
        initial_standardized_mse=1.0,
        final_standardized_mse=1.0,
    )


def test_feature_abi_normalizes_modal_null_and_log_rms_and_masks_padding() -> None:
    codec = _codec()
    batch = _batch("fit")
    encoded = codec.prepare(dtype=torch.float64)(
        batch.modal_coordinates,
        batch.null_coordinates,
        batch.row_rms,
        batch.valid_mask,
    )

    assert encoded.shape == (1, 4, 5)
    torch.testing.assert_close(
        encoded[0, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        encoded[0, 2],
        torch.tensor(
            [1.0, 4.0, 1.0, -1.0, torch.log(torch.tensor(2.0)) / 2],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        encoded[0, 1],
        torch.zeros(5, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert codec.modal_modes == 2
    assert codec.null_modes == 1
    assert codec.feature_modes == 5
    assert codec.stored_scalar_count == 10


def test_null_coordinate_breaks_modal_and_rms_collision_at_runtime() -> None:
    runtime = _manual_null_provider().prepare(dtype=torch.float64)
    modal = torch.zeros(1, 2, 2, dtype=torch.float64)
    row_rms = torch.ones(1, 2, dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    positions = torch.tensor([[0, 1]])
    first_null = torch.tensor([[[0.25], [0.25]]], dtype=torch.float64)
    second_null = torch.tensor([[[0.25], [0.75]]], dtype=torch.float64)

    first = runtime(
        modal,
        first_null,
        row_rms,
        valid_mask=mask,
        logical_positions=positions,
    )
    second = runtime(
        modal,
        second_null,
        row_rms,
        valid_mask=mask,
        logical_positions=positions,
    )

    torch.testing.assert_close(
        first[0, 1, 0],
        torch.tensor(0.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        second[0, 1, 0],
        torch.tensor(1.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        first[..., 1],
        torch.zeros(1, 2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        second[..., 1],
        torch.zeros(1, 2, dtype=torch.float64),
    )


def test_codec_and_batch_artifacts_are_strict_and_tamper_evident() -> None:
    codec = _codec()
    restored_codec = ReferenceProviderFeatureCodec.from_state_dict(
        codec.state_dict()
    )
    assert restored_codec.artifact_sha256 == codec.artifact_sha256

    codec_tamper = copy.deepcopy(codec.state_dict())
    codec_tamper["null_scale"][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ReferenceProviderFeatureCodec.from_state_dict(codec_tamper)
    wrong_dtype = copy.deepcopy(codec.state_dict())
    wrong_dtype["modal_center"] = wrong_dtype["modal_center"].float()
    with pytest.raises(ValueError, match="float64"):
        ReferenceProviderFeatureCodec.from_state_dict(wrong_dtype)
    noncontiguous = copy.deepcopy(codec.state_dict())
    noncontiguous["modal_whitener"] = (
        noncontiguous["modal_whitener"].transpose(0, 1)
    )
    assert not noncontiguous["modal_whitener"].is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        ReferenceProviderFeatureCodec.from_state_dict(noncontiguous)

    batch = _batch("fit")
    restored_batch = SyntheticReferenceBatch.from_state_dict(
        batch.state_dict()
    )
    assert restored_batch.content_sha256 == batch.content_sha256
    batch_tamper = copy.deepcopy(batch.state_dict())
    batch_tamper["null_coordinates"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="content hash mismatch"):
        SyntheticReferenceBatch.from_state_dict(batch_tamper)
    unknown = copy.deepcopy(batch.state_dict())
    unknown["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        SyntheticReferenceBatch.from_state_dict(unknown)

    assert "prompt" not in {field.name for field in fields(SyntheticReferenceBatch)}
    assert "token" not in {field.name for field in fields(SyntheticReferenceBatch)}


def test_fixed_step_fit_is_deterministic_and_has_no_selection_input() -> None:
    first = _fit()
    second = _fit()

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.training_steps == 3
    assert first.seed == 73
    first_state = first.executor_artifact["model_state_dict"]
    second_state = second.executor_artifact["model_state_dict"]
    assert set(first_state) == set(second_state)
    for name in first_state:
        torch.testing.assert_close(
            first_state[name],
            second_state[name],
            rtol=0.0,
            atol=0.0,
        )
    with pytest.raises(ValueError, match="'fit'"):
        fit_state_conditioned_reference_provider(
            feature_codec=_codec(),
            target_center=torch.zeros(2),
            target_scale=torch.ones(2),
            fit_batches=(_batch("selection", offset=1.0),),
            executor_config=_config(),
            steps=1,
            learning_rate=0.01,
            seed=0,
        )


def test_plan_roundtrip_prepared_runtime_and_integrity_checks() -> None:
    plan = _fit()
    restored = StateConditionedReferenceProviderPlan.from_state_dict(
        plan.state_dict()
    )
    assert restored.artifact_sha256 == plan.artifact_sha256
    runtime = restored.prepare(dtype=torch.float64)
    batch = _batch("selection", offset=0.5)
    features = runtime.feature_codec(
        batch.modal_coordinates,
        batch.null_coordinates,
        batch.row_rms,
        batch.valid_mask,
    )
    standardized = runtime.executor(
        features,
        query_valid_mask=batch.valid_mask,
        key_valid_mask=batch.valid_mask,
        logical_positions=batch.logical_positions,
        key_logical_positions=batch.logical_positions,
    )
    expected = standardized * runtime.target_scale + runtime.target_center
    expected = torch.where(
        batch.valid_mask.unsqueeze(-1),
        expected,
        torch.zeros_like(expected),
    )
    actual = runtime(
        batch.modal_coordinates,
        batch.null_coordinates,
        batch.row_rms,
        valid_mask=batch.valid_mask,
        logical_positions=batch.logical_positions,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(
        actual[~batch.valid_mask],
        torch.zeros_like(actual[~batch.valid_mask]),
    )

    tampered = copy.deepcopy(plan.state_dict())
    tensor_name = next(iter(tampered["executor_artifact"]["model_state_dict"]))
    tampered["executor_artifact"]["model_state_dict"][tensor_name].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="plan hash mismatch"):
        StateConditionedReferenceProviderPlan.from_state_dict(tampered)

    with torch.no_grad():
        runtime.executor.same_position_bias[0] += 1.0
    with pytest.raises(ValueError, match="runtime integrity mismatch"):
        runtime.validate_integrity()


def test_source_normalized_executor_survives_plan_prepare() -> None:
    plan = _fit(
        config=_config(
            experts=2,
            rank=2,
            router_width=3,
            source_normalized_routing=True,
        ),
        steps=1,
    )
    restored = StateConditionedReferenceProviderPlan.from_state_dict(
        plan.state_dict()
    )
    runtime = restored.prepare(dtype=torch.float64)
    batch = _batch("selection", offset=0.5)

    assert runtime.executor.config.source_normalized_routing
    assert runtime.executor.source_score_weight is not None
    assert not runtime.executor.source_score_weight.requires_grad
    actual = runtime(
        batch.modal_coordinates,
        batch.null_coordinates,
        batch.row_rms,
        valid_mask=batch.valid_mask,
        logical_positions=batch.logical_positions,
    )
    assert torch.isfinite(actual).all()
    assert torch.equal(
        actual[~batch.valid_mask],
        torch.zeros_like(actual[~batch.valid_mask]),
    )
    runtime.validate_integrity()


def test_evaluation_checks_causality_padding_repeat_and_disjointness() -> None:
    plan = _fit()
    selection = _batch("selection", offset=0.75)
    evaluation = evaluate_state_conditioned_reference_provider(
        plan,
        (selection,),
        required_split="selection",
    )

    assert evaluation.structural_checks_passed
    assert evaluation.repeat_exact
    assert evaluation.causal_prefix_exact
    assert evaluation.padding_exact
    assert evaluation.invalid_outputs_zero
    assert evaluation.integrity_verified
    assert evaluation.pooled_standardized_relative_error >= 0.0
    assert -1.0 <= evaluation.pooled_standardized_cosine <= 1.0

    fit = _batch("fit")
    same_content_selection = SyntheticReferenceBatch(
        split="selection",
        modal_coordinates=fit.modal_coordinates,
        null_coordinates=fit.null_coordinates,
        row_rms=fit.row_rms,
        target_modes=fit.target_modes,
        logical_positions=fit.logical_positions,
        valid_mask=fit.valid_mask,
        synthetic_binding_sha256=fit.synthetic_binding_sha256,
    )
    assert same_content_selection.content_sha256 == fit.content_sha256
    with pytest.raises(ValueError, match="overlaps provider fit"):
        evaluate_state_conditioned_reference_provider(
            plan,
            (same_content_selection,),
            required_split="selection",
        )


def test_parameter_and_execution_accounting_include_codec_and_null_costs() -> None:
    plan = _manual_null_provider()
    accounting = plan.accounting()
    assert accounting.feature_codec_scalar_count == 10
    assert accounting.target_standardization_scalar_count == 4
    assert accounting.total_stored_scalar_count == (
        10 + 4 + accounting.executor_parameter_count
    )

    runtime = plan.prepare(dtype=torch.float64)
    batch = _batch("selection", offset=0.5)
    execution = runtime.execution_accounting(
        valid_mask=batch.valid_mask,
        logical_positions=batch.logical_positions,
    )
    assert execution.valid_rows == 3
    assert execution.modal_whitening_mac_count == 3 * 2 * 2
    assert execution.null_standardization_mac_count == 3
    assert execution.target_destandardization_mac_count == 3 * 2
    assert execution.total_mac_count == (
        execution.core.total_mac_count + 12 + 3 + 6
    )


def test_ladder_fits_every_candidate_before_selection_and_picks_smallest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real_fit = provider_module.fit_state_conditioned_reference_provider

    def recording_fit(**kwargs: object) -> StateConditionedReferenceProviderPlan:
        events.append("fit")
        return real_fit(**kwargs)

    monkeypatch.setattr(
        provider_module,
        "fit_state_conditioned_reference_provider",
        recording_fit,
    )

    def selection_factory() -> tuple[SyntheticReferenceBatch, ...]:
        assert events == ["fit", "fit"]
        events.append("selection")
        return (_batch("selection", offset=1.0),)

    small = _config(experts=1, rank=1, router_width=1)
    large = _config(experts=2, rank=2, router_width=3)
    compilation = compile_state_conditioned_reference_provider_ladder(
        feature_codec=_codec(),
        target_center=torch.tensor([0.5, -0.25]),
        target_scale=torch.tensor([2.0, 0.5]),
        fit_batches=(_batch("fit"),),
        selection_batch_factory=selection_factory,
        executor_configs=(large, small),
        steps=1,
        learning_rate=0.01,
        base_seed=91,
        selection_gates=StateConditionedReferenceProviderSelectionGates(
            max_pooled_standardized_relative_error=100.0,
            min_pooled_standardized_cosine=-1.0,
        ),
    )

    assert events == ["fit", "fit", "selection"]
    assert compilation.selected_plan is not None
    assert compilation.selected_plan.executor_config == small
    assert len(compilation.rate_curve) == 2
    assert all(point.structural_checks_passed for point in compilation.rate_curve)
