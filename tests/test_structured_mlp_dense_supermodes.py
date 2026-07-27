from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from fisher_graph.structured_layer_distillation import (
    StructuredLayerTargets,
)
from fisher_graph.structured_mlp_compression import (
    STRUCTURED_MLP_EXPLICIT_PLAN_SELECTION_ALGORITHM,
    build_width_compressed_structured_executor,
    select_fisher_taylor_mlp_units,
)
from fisher_graph.structured_mlp_compression_pipeline import (
    refit_structured_mlp_down_projection_from_targets_,
)
from fisher_graph.structured_mlp_dense_supermode_pipeline import (
    DenseSupermodeFitWeights,
    build_structured_mlp_dense_supermode_candidate,
    build_structured_mlp_dense_supermode_native_pivot_control,
    evaluate_structured_mlp_dense_supermode_candidate,
)
from fisher_graph.structured_mlp_dense_supermodes import (
    DenseSupermodeObjectiveWeights,
    build_fisher_jacobian_dense_supermode_plan,
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


def _activation_fixture(
    *,
    activation: str = "gelu_pytorch_tanh",
    dtype: torch.dtype = torch.float32,
) -> tuple[
    StructuredTransformerLayerExecutor,
    tuple[StructuredLayerTargets, ...],
    tuple,
]:
    source = _executor(
        6,
        projection_bias=False,
        seed=75_019,
    )
    if (
        source.config.transformer.feed_forward.activation
        != activation
    ):
        config = replace(
            source.config,
            transformer=replace(
                source.config.transformer,
                feed_forward=replace(
                    source.config.transformer.feed_forward,
                    activation=activation,
                ),
            ),
        )
        converted = StructuredTransformerLayerExecutor(config)
        converted.load_state_dict(source.state_dict(), strict=True)
        source = converted.eval()

    # Two distinct activation families share one gate.  Any linear mixture
    # of the two families is therefore exactly representable by one gated
    # unit with the same gate and a mixed up row.  Within each family the
    # native rows are exact duplicates.
    with torch.no_grad():
        common_gate = source.feed_forward.gate_proj.weight[0].clone()
        first_up = source.feed_forward.up_proj.weight[0].clone()
        second_up = source.feed_forward.up_proj.weight[3].clone()
        for index in range(6):
            source.feed_forward.gate_proj.weight[index].copy_(common_gate)
            source.feed_forward.up_proj.weight[index].copy_(
                first_up if index < 3 else second_up
            )
    source.to(dtype=dtype)
    source.eval()
    provenance = _provenance("5")
    generator = torch.Generator().manual_seed(31_771)
    hidden_a = torch.randn(
        5,
        7,
        source.width,
        generator=generator,
        dtype=dtype,
    )
    hidden_b = torch.randn(
        4,
        6,
        source.width,
        generator=generator,
        dtype=dtype,
    )
    mask_a = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, False],
            [True, True, True, True, True, False, False],
            [True, True, True, True, False, False, False],
            [True, True, True, False, False, False, False],
        ]
    )
    mask_b = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
        ]
    )
    targets = (
        _targets(source, hidden_a, mask_a, provenance),
        _targets(source, hidden_b, mask_b, provenance),
    )
    scores = []
    gradient_levels = torch.tensor(
        [9.0, 8.0, 7.0, 0.25, 0.20, 0.15],
        dtype=dtype,
    )
    for batch_id, target in zip(
        ("dense-family-a", "dense-family-b"),
        targets,
        strict=True,
    ):
        features = target.feed_forward_projection_input
        assert features is not None
        gradients = gradient_levels.view(1, 1, 6).expand_as(
            features
        ).clone()
        scores.append(
            _score_batch(
                batch_id=batch_id,
                activations=features.detach().clone(),
                gradients=gradients,
                mask=target.sequence.query_valid_mask.clone(),
                provenance=provenance,
            )
        )
    return source, targets, tuple(scores)


def _executor_distortion(
    executor: StructuredTransformerLayerExecutor,
    targets: tuple[StructuredLayerTargets, ...],
) -> dict[str, float]:
    operator_actual = []
    operator_target = []
    block_actual = []
    block_target = []
    with torch.no_grad():
        for target in targets:
            valid = target.sequence.query_valid_mask
            features = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            )
            operator_actual.append(
                executor.feed_forward.down_proj(features)[valid]
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            operator_target.append(
                target.feed_forward_operator_output[valid]
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            block_actual.append(
                executor.forward_components(
                    target.block_input,
                    target.sequence,
                )
                .output[valid]
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )
            block_target.append(
                target.output[valid]
                .detach()
                .to(device="cpu", dtype=torch.float64)
            )

    def nrmse(actual: list[torch.Tensor], expected: list[torch.Tensor]):
        actual_tensor = torch.cat(actual, dim=0)
        expected_tensor = torch.cat(expected, dim=0)
        numerator = float(
            (actual_tensor - expected_tensor).square().sum().item()
        )
        denominator = float(expected_tensor.square().sum().item())
        return math.sqrt(numerator / denominator)

    return {
        "feed_forward_operator_nrmse": nrmse(
            operator_actual,
            operator_target,
        ),
        "block_output_nrmse": nrmse(
            block_actual,
            block_target,
        ),
    }


def _oracle_family_score_batches(
    scores: tuple,
) -> tuple:
    # The fixture's two known generator families are 0:3 and 3:6. This
    # deliberately gives an oracle pruning control one representative of each.
    levels = torch.tensor(
        [1_000.0, 0.0, 0.0, 900.0, 0.0, 0.0],
        dtype=scores[0].score_gradient.dtype,
    )
    return tuple(
        replace(
            score,
            score_gradient=levels.to(
                device=score.score_gradient.device
            )
            .view(1, 1, 6)
            .expand_as(score.score_gradient)
            .clone(),
        )
        for score in scores
    )


def _actual_feature_targets(
    executor: StructuredTransformerLayerExecutor,
    targets: tuple[StructuredLayerTargets, ...],
) -> tuple[StructuredLayerTargets, ...]:
    result = []
    for target in targets:
        with torch.no_grad():
            features = executor.feed_forward_projection_features(
                target.normalized_feed_forward_input
            )
        result.append(
            replace(
                target,
                feed_forward_projection_input=features.detach().clone(),
            )
        )
    return tuple(result)


def test_plan_is_padding_safe_batch_order_invariant_and_dual() -> None:
    parent, targets, scores = _activation_fixture()
    poisoned = []
    for score in scores:
        activation = score.projection_input.clone()
        gradient = score.score_gradient.clone()
        activation[~score.valid_mask] = torch.nan
        gradient[~score.valid_mask] = torch.nan
        poisoned.append(
            replace(
                score,
                projection_input=activation,
                score_gradient=gradient,
            )
        )
    kwargs = {
        "source_down_weight": parent.feed_forward.down_proj.weight,
        "calibration_split_sha256": "6" * 64,
        "activation_site": "layer.0.mlp.down_input",
        "parent_executor_fingerprint": parent.execution_fingerprint(),
        "retained_pool_width": 2,
        "pool_indices": tuple(range(6)),
        "expected_source_width": 6,
        "objective_weights": DenseSupermodeObjectiveWeights(
            activation=1.0,
            output=1.0,
            fisher=0.1,
        ),
    }
    first = build_fisher_jacobian_dense_supermode_plan(
        tuple(reversed(poisoned)),
        **kwargs,
    )
    repeated = build_fisher_jacobian_dense_supermode_plan(
        scores,
        **kwargs,
    )

    assert first.plan_sha256 == repeated.plan_sha256
    assert first.metadata() == repeated.metadata()
    assert first.pool_width == 6
    assert first.retained_pool_width == 2
    assert first.runtime_width == 2
    assert first.valid_rows == 43
    torch.testing.assert_close(first.encoder, repeated.encoder, rtol=0, atol=0)
    torch.testing.assert_close(first.decoder, repeated.decoder, rtol=0, atol=0)
    torch.testing.assert_close(
        first.encoder.mT @ first.decoder,
        torch.eye(2, dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )
    features = scores[0].projection_input[scores[0].valid_mask]
    ideal = first.ideal_coordinates(features)
    assert ideal.shape == (int(scores[0].valid_mask.sum().item()), 2)
    assert first.reconstruct_pool_features(ideal).shape[-1] == 6


def test_fisher_gradient_changes_the_dense_retained_direction() -> None:
    provenance = _provenance("4")
    activations = torch.eye(3).view(1, 3, 3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    down = torch.eye(3)

    def plan_for(levels: tuple[float, float, float]):
        gradients = torch.diag(torch.tensor(levels)).view(1, 3, 3)
        batch = _score_batch(
            batch_id="gradient-sensitive",
            activations=activations,
            gradients=gradients,
            mask=mask,
            provenance=provenance,
        )
        return build_fisher_jacobian_dense_supermode_plan(
            (batch,),
            source_down_weight=down,
            calibration_split_sha256="7" * 64,
            activation_site="layer.0.mlp.down_input",
            parent_executor_fingerprint="8" * 64,
            retained_pool_width=1,
            pool_indices=(0, 1, 2),
            objective_weights=DenseSupermodeObjectiveWeights(
                activation=0.0,
                output=0.0,
                fisher=1.0,
            ),
        )

    first = plan_for((10.0, 1.0, 0.5))
    last = plan_for((0.5, 1.0, 10.0))
    assert first.pivot_source_indices == (0,)
    assert last.pivot_source_indices == (2,)
    assert first.plan_sha256 != last.plan_sha256
    assert abs(float(first.decoder[0, 0])) > abs(
        float(first.decoder[2, 0])
    )
    assert abs(float(last.decoder[2, 0])) > abs(
        float(last.decoder[0, 0])
    )


def test_inactive_zero_energy_objectives_are_skipped() -> None:
    provenance = _provenance("3")
    activations = torch.eye(4).view(1, 4, 4)
    batch = _score_batch(
        batch_id="zero-inactive-objectives",
        activations=activations,
        gradients=torch.zeros_like(activations),
        mask=torch.ones(1, 4, dtype=torch.bool),
        provenance=provenance,
    )
    plan = build_fisher_jacobian_dense_supermode_plan(
        (batch,),
        source_down_weight=torch.zeros(4, 4),
        calibration_split_sha256="3" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="4" * 64,
        retained_pool_width=2,
        pool_indices=(0, 1, 2, 3),
        objective_weights=DenseSupermodeObjectiveWeights(
            activation=1.0,
            output=0.0,
            fisher=0.0,
        ),
    )

    assert plan.activation_energy > 0.0
    assert plan.output_energy == 0.0
    assert plan.fisher_energy == 0.0
    assert plan.metadata()["objective_weights"] == {
        "activation": 1.0,
        "output": 0.0,
        "fisher": 0.0,
    }


def test_automatic_pool_selects_lowest_diagonal_fisher_units() -> None:
    provenance = _provenance("6")
    activations = torch.eye(4).view(1, 4, 4)
    gradients = (
        torch.tensor([4.0, 3.0, 1.0, 2.0])
        .view(1, 1, 4)
        .expand_as(activations)
        .clone()
    )
    batch = _score_batch(
        batch_id="automatic-low-score-pool",
        activations=activations,
        gradients=gradients,
        mask=torch.ones(1, 4, dtype=torch.bool),
        provenance=provenance,
    )
    plan = build_fisher_jacobian_dense_supermode_plan(
        (batch,),
        source_down_weight=torch.eye(4),
        calibration_split_sha256="5" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint="6" * 64,
        retained_pool_width=1,
        pool_width=2,
    )

    assert plan.pool_indices == (2, 3)
    assert plan.singleton_indices == (0, 1)
    assert plan.pool_selection == (
        "lowest_diagonal_fisher_stable_source_index"
    )


@pytest.mark.parametrize(
    "activation",
    ("gelu_pytorch_tanh", "silu"),
)
def test_dense_candidate_compacts_and_beats_scalar_fisher_deletion(
    activation: str,
) -> None:
    parent, targets, scores = _activation_fixture(
        activation=activation,
    )
    parent_fingerprint = parent.execution_fingerprint()
    parent_state = {
        name: value.detach().clone()
        for name, value in parent.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()
    plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256="9" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent_fingerprint,
        retained_pool_width=2,
        pool_indices=tuple(range(6)),
        objective_weights=DenseSupermodeObjectiveWeights(
            activation=1.0,
            output=1.0,
            fisher=0.1,
        ),
    )
    candidate = build_structured_mlp_dense_supermode_candidate(
        parent,
        plan,
        targets,
        scores,
        calibration_split_sha256="9" * 64,
        fit_weights=DenseSupermodeFitWeights(
            latent=1.0,
            output=2.0,
            fisher=0.25,
        ),
        generator_steps=320,
        generator_learning_rate=8e-3,
        generator_minibatch_rows=52,
    )
    evaluation = _executor_distortion(candidate.executor, targets)
    authenticated_evaluation = (
        evaluate_structured_mlp_dense_supermode_candidate(
            candidate,
            targets,
            evaluation_id=f"synthetic-{activation}-guard-v1",
            evaluation_split_sha256="b" * 64,
            expected_target_provenance=plan.provenance,
        )
    )

    deletion_selection = select_fisher_taylor_mlp_units(
        scores,
        calibration_split_sha256="9" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent_fingerprint,
        retained_width=2,
        expected_source_width=6,
    )
    deletion, _ = build_width_compressed_structured_executor(
        parent,
        deletion_selection,
    )
    refit_structured_mlp_down_projection_from_targets_(
        deletion,
        _actual_feature_targets(deletion, targets),
        calibration_split_sha256="9" * 64,
        ridge=1e-7,
    )
    deletion_evaluation = _executor_distortion(deletion, targets)

    oracle_scores = _oracle_family_score_batches(scores)
    oracle_selection = select_fisher_taylor_mlp_units(
        oracle_scores,
        calibration_split_sha256="9" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent_fingerprint,
        retained_width=2,
        expected_source_width=6,
    )
    oracle, _ = build_width_compressed_structured_executor(
        parent,
        oracle_selection,
    )
    refit_structured_mlp_down_projection_from_targets_(
        oracle,
        _actual_feature_targets(oracle, targets),
        calibration_split_sha256="9" * 64,
        ridge=1e-7,
    )
    oracle_evaluation = _executor_distortion(oracle, targets)

    assert deletion_selection.selected_indices == (0, 1)
    assert oracle_selection.selected_indices == (0, 3)
    assert candidate.executor.feed_forward.gate_proj.weight.shape == (
        2,
        parent.width,
    )
    assert candidate.executor.feed_forward.up_proj.weight.shape == (
        2,
        parent.width,
    )
    assert candidate.executor.feed_forward.down_proj.weight.shape == (
        parent.width,
        2,
    )
    assert all(
        value.is_contiguous()
        for value in (
            candidate.executor.feed_forward.gate_proj.weight,
            candidate.executor.feed_forward.up_proj.weight,
            candidate.executor.feed_forward.down_proj.weight,
        )
    )
    assert evaluation["feed_forward_operator_nrmse"] < 2e-3
    assert authenticated_evaluation[
        "feed_forward_operator_nrmse"
    ] == pytest.approx(evaluation["feed_forward_operator_nrmse"])
    assert authenticated_evaluation[
        "candidate_report_sha256"
    ] == candidate.report["report_sha256"]
    assert (
        authenticated_evaluation["evaluation_targets_sha256"]
        != candidate.report["report_sha256"]
    )
    assert (
        evaluation["feed_forward_operator_nrmse"]
        < deletion_evaluation["feed_forward_operator_nrmse"] * 0.1
    )
    assert oracle_evaluation["feed_forward_operator_nrmse"] < 2e-5
    assert oracle_evaluation["block_output_nrmse"] < 2e-5
    if activation == "gelu_pytorch_tanh":
        assert evaluation["feed_forward_operator_nrmse"] == pytest.approx(
            8.0767e-7,
            rel=0.03,
        )
        assert evaluation["block_output_nrmse"] == pytest.approx(
            1.4103e-6,
            rel=0.03,
        )
        assert deletion_evaluation[
            "feed_forward_operator_nrmse"
        ] == pytest.approx(0.370075, rel=2e-5)
        assert oracle_evaluation[
            "feed_forward_operator_nrmse"
        ] == pytest.approx(1.6623e-7, rel=0.03)
    resources = candidate.report["resources"]
    assert resources["parameters"]["removed_full_layer"] == (
        3 * parent.width * 4
    )
    assert resources["compute_per_valid_token"]["macs"] == {
        "source": 3 * parent.width * 6,
        "candidate": 3 * parent.width * 2,
        "removed": 3 * parent.width * 4,
        "retained_ratio": 1 / 3,
    }
    assert resources["runtime_storage"][
        "source_width_basis_stored_in_executor"
    ] is False
    assert candidate.report["deployment"][
        "executes_native_pool_k_wide_features"
    ] is False
    checkpoint_selection = candidate.report["generator_fit"][
        "checkpoint_selection"
    ]
    assert checkpoint_selection[
        "full_fit_evaluated_after_every_optimizer_step"
    ] is False
    assert checkpoint_selection["full_fit_checkpoint_count"] == 10
    assert candidate.report["generator_fit"]["selected_metrics"][
        "objective"
    ] <= candidate.report["generator_fit"]["initial_metrics"]["objective"]
    assert parent.execution_fingerprint() == parent_fingerprint
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    for name, expected in parent_state.items():
        torch.testing.assert_close(
            parent.state_dict()[name],
            expected,
            rtol=0,
            atol=0,
        )
    restored = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            candidate.artifact_state
        )
    )
    assert (
        restored.execution_fingerprint()
        == candidate.executor.execution_fingerprint()
    )


def test_groupwise_plan_keeps_nonpool_units_exact_and_rejects_drift() -> None:
    parent, targets, scores = _activation_fixture()
    plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256="a" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_pool_width=2,
        pool_indices=(0, 2, 3, 5),
    )
    assert plan.singleton_indices == (1, 4)
    assert plan.runtime_width == 4
    candidate = build_structured_mlp_dense_supermode_candidate(
        parent,
        plan,
        targets,
        scores,
        calibration_split_sha256="a" * 64,
        generator_steps=4,
    )
    singleton = torch.tensor(plan.singleton_indices)
    torch.testing.assert_close(
        candidate.executor.feed_forward.gate_proj.weight[
            : plan.singleton_count
        ],
        parent.feed_forward.gate_proj.weight.index_select(0, singleton),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate.executor.feed_forward.up_proj.weight[
            : plan.singleton_count
        ],
        parent.feed_forward.up_proj.weight.index_select(0, singleton),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate.executor.feed_forward.down_proj.weight[
            :, : plan.singleton_count
        ],
        parent.feed_forward.down_proj.weight.index_select(1, singleton),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match="does not match"):
        plan.validate_source_down_weight(
            parent.feed_forward.down_proj.weight + 1e-4
        )
    with pytest.raises(ValueError, match="either pool_indices or pool_width"):
        build_fisher_jacobian_dense_supermode_plan(
            scores,
            source_down_weight=parent.feed_forward.down_proj.weight,
            calibration_split_sha256="a" * 64,
            activation_site="layer.0.mlp.down_input",
            parent_executor_fingerprint=parent.execution_fingerprint(),
            retained_pool_width=2,
            pool_width=4,
            pool_indices=(0, 1, 2, 3),
        )


def test_native_pivot_control_keeps_singletons_and_pivots_then_refits_down(
) -> None:
    parent, targets, scores = _activation_fixture()
    parent_fingerprint = parent.execution_fingerprint()
    parent_state = {
        name: value.detach().clone()
        for name, value in parent.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()
    split = "c" * 64
    plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256=split,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent_fingerprint,
        retained_pool_width=2,
        pool_indices=(0, 2, 3, 5),
    )

    control, report = (
        build_structured_mlp_dense_supermode_native_pivot_control(
            parent,
            plan,
            targets,
            scores,
            calibration_split_sha256=split,
            down_ridge=1e-5,
        )
    )

    expected = tuple(
        sorted((*plan.singleton_indices, *plan.pivot_source_indices))
    )
    indices = torch.tensor(expected, dtype=torch.long)
    assert report["selection"]["algorithm"] == (
        STRUCTURED_MLP_EXPLICIT_PLAN_SELECTION_ALGORITHM
    )
    assert report["selection"]["selection_basis_sha256"] == (
        plan.plan_sha256
    )
    assert tuple(report["selection"]["selected_indices"]) == expected
    assert report["selection_rule"][
        "reference_scores_are_not_a_topk_selection_rule"
    ] is True
    assert control.feed_forward.gate_proj.weight.shape[0] == (
        plan.runtime_width
    )
    torch.testing.assert_close(
        control.feed_forward.gate_proj.weight,
        parent.feed_forward.gate_proj.weight.index_select(0, indices),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        control.feed_forward.up_proj.weight,
        parent.feed_forward.up_proj.weight.index_select(0, indices),
        rtol=0,
        atol=0,
    )
    assert report["refit_targets"][
        "actual_runtime_features_used_for_down_refit"
    ] is True
    assert report["refit_targets"][
        "native_selected_projection_features_used_for_down_refit"
    ] is False
    refit = report["terminal_projection_refit"]
    assert refit["projection"]["post_refit_operator_nrmse"] <= (
        refit["projection"]["pre_refit_operator_nrmse"] + 1e-12
    )
    assert refit["executor_fingerprint_after"] == (
        control.execution_fingerprint()
    )
    assert report["resources"]["parameters"]["removed_full_layer"] == (
        parent.learned_parameter_count - control.learned_parameter_count
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert parent.execution_fingerprint() == parent_fingerprint
    for name, expected_value in parent_state.items():
        torch.testing.assert_close(
            parent.state_dict()[name],
            expected_value,
            rtol=0,
            atol=0,
        )


def test_mutated_plan_and_candidate_fail_closed() -> None:
    parent, targets, scores = _activation_fixture()
    split = "d" * 64
    plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256=split,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_pool_width=2,
        pool_indices=tuple(range(6)),
    )
    parent.train()
    with pytest.raises(ValueError, match="frozen in eval mode"):
        build_structured_mlp_dense_supermode_candidate(
            parent,
            plan,
            targets,
            scores,
            calibration_split_sha256=split,
            generator_steps=1,
        )
    parent.eval()
    with torch.no_grad():
        plan.encoder[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="dual|digest|integrity"):
        plan.metadata()
    with pytest.raises(ValueError, match="dual|digest|integrity"):
        build_structured_mlp_dense_supermode_candidate(
            parent,
            plan,
            targets,
            scores,
            calibration_split_sha256=split,
            generator_steps=1,
        )

    clean_plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256=split,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_pool_width=2,
        pool_indices=tuple(range(6)),
    )
    candidate = build_structured_mlp_dense_supermode_candidate(
        parent,
        clean_plan,
        targets,
        scores,
        calibration_split_sha256=split,
        generator_steps=1,
    )
    candidate.executor.train()
    with pytest.raises(ValueError, match="eval mode"):
        evaluate_structured_mlp_dense_supermode_candidate(
            candidate,
            targets,
            evaluation_id="train-mode-rejection",
            evaluation_split_sha256="e" * 64,
            expected_target_provenance=clean_plan.provenance,
        )
    candidate.executor.eval()
    wrong_targets = tuple(
        replace(target, provenance=_provenance("f"))
        for target in targets
    )
    with pytest.raises(ValueError, match="source provenance"):
        evaluate_structured_mlp_dense_supermode_candidate(
            candidate,
            wrong_targets,
            evaluation_id="provenance-rejection",
            evaluation_split_sha256="e" * 64,
            expected_target_provenance=clean_plan.provenance,
        )
    with torch.no_grad():
        candidate.executor.feed_forward.gate_proj.weight[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="fingerprints differ"):
        candidate.validate_integrity()


def test_bfloat16_candidate_builds_without_false_full_output_replay() -> None:
    parent, targets, scores = _activation_fixture(dtype=torch.bfloat16)
    split = "1" * 64
    plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256=split,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_pool_width=2,
        pool_indices=tuple(range(6)),
    )
    candidate = build_structured_mlp_dense_supermode_candidate(
        parent,
        plan,
        targets,
        scores,
        calibration_split_sha256=split,
        generator_steps=2,
    )

    candidate.validate_integrity()
    assert candidate.executor.dtype is torch.bfloat16
    assert candidate.executor.feed_forward.gate_proj.weight.dtype is (
        torch.bfloat16
    )
    evaluation = evaluate_structured_mlp_dense_supermode_candidate(
        candidate,
        targets,
        evaluation_id="bfloat16-construction",
        evaluation_split_sha256="2" * 64,
        expected_target_provenance=plan.provenance,
    )
    assert math.isfinite(evaluation["feed_forward_operator_nrmse"])
