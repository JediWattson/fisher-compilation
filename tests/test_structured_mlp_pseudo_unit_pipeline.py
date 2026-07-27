from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import fisher_graph.structured_mlp_pseudo_unit_pipeline as pipeline_module
from fisher_graph.structured_mlp_pseudo_unit_bundling import (
    StructuredMLPPseudoUnitBundlingPlan,
    build_fisher_pseudo_unit_bundling_plan,
)
from fisher_graph.structured_mlp_pseudo_unit_pipeline import (
    build_structured_mlp_pseudo_unit_candidate,
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


def _fixture(
    *,
    projection_bias: bool = False,
) -> tuple[
    StructuredTransformerLayerExecutor,
    StructuredMLPPseudoUnitBundlingPlan,
    tuple,
    tuple,
]:
    parent = _executor(
        4,
        projection_bias=projection_bias,
        seed=51_903,
    )
    # Units zero and one are a deliberately exact native pair.  The ideal
    # modal coordinate is representable by one gated unit, but the dense
    # two-row initialization is not already that solution.
    with torch.no_grad():
        parent.feed_forward.gate_proj.weight[1].copy_(
            parent.feed_forward.gate_proj.weight[0]
        )
        parent.feed_forward.up_proj.weight[1].copy_(
            parent.feed_forward.up_proj.weight[0]
        )
        parent.feed_forward.down_proj.weight[:, 1].copy_(
            parent.feed_forward.down_proj.weight[:, 0]
        )
        if projection_bias:
            assert parent.feed_forward.gate_proj.bias is not None
            assert parent.feed_forward.up_proj.bias is not None
            parent.feed_forward.gate_proj.bias[1].copy_(
                parent.feed_forward.gate_proj.bias[0]
            )
            parent.feed_forward.up_proj.bias[1].copy_(
                parent.feed_forward.up_proj.bias[0]
            )
    parent.eval()
    provenance = _provenance("7")
    generator = torch.Generator().manual_seed(82_117)
    hidden = torch.randn(4, 6, parent.width, generator=generator)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, True, False],
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
        ]
    )
    target = _targets(parent, hidden, mask, provenance)
    features = target.feed_forward_projection_input
    assert features is not None
    gradients = torch.empty_like(features)
    gradients[..., 0] = 0.04
    gradients[..., 1] = 0.04
    gradients[..., 2] = 1.0
    gradients[..., 3] = 2.0
    score = _score_batch(
        batch_id="fit-family-a",
        activations=features.detach().clone(),
        gradients=gradients,
        mask=mask.clone(),
        provenance=provenance,
    )
    plan = build_fisher_pseudo_unit_bundling_plan(
        (score,),
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256="8" * 64,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_width=3,
        expected_source_width=4,
    )
    assert plan.pairs[0].source_indices == (0, 1)
    return parent, plan, (target,), (score,)


def _artifact_tensors(
    state: Mapping[str, object],
) -> Mapping[str, torch.Tensor]:
    values = state["model_state_dict"]
    assert isinstance(values, Mapping)
    assert all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in values.items()
    )
    return values  # type: ignore[return-value]


def test_pipeline_builds_repeatable_direct_and_actual_feature_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, plan, targets, scores = _fixture()
    parent_fingerprint = parent.execution_fingerprint()
    rng_before = torch.random.get_rng_state().clone()
    real_refit = (
        pipeline_module.refit_structured_mlp_down_projection_from_targets_
    )
    actual_feature_binding_observed = False

    def checking_refit(
        executor: StructuredTransformerLayerExecutor,
        batches: tuple,
        *,
        calibration_split_sha256: str,
        ridge: float,
    ) -> dict[str, object]:
        nonlocal actual_feature_binding_observed
        for original, transformed, score in zip(
            targets,
            batches,
            scores,
            strict=True,
        ):
            supplied = transformed.feed_forward_projection_input
            assert supplied is not None
            with torch.no_grad():
                actual = executor.feed_forward_projection_features(
                    original.normalized_feed_forward_input
                )
            torch.testing.assert_close(supplied, actual, rtol=0, atol=0)
            ideal = plan.ideal_features(score.projection_input)
            valid = score.valid_mask
            assert not torch.equal(
                supplied[valid][:, plan.singleton_count :],
                ideal[valid][:, plan.singleton_count :],
            )
        actual_feature_binding_observed = True
        return real_refit(
            executor,
            batches,
            calibration_split_sha256=calibration_split_sha256,
            ridge=ridge,
        )

    monkeypatch.setattr(
        pipeline_module,
        "refit_structured_mlp_down_projection_from_targets_",
        checking_refit,
    )
    first = build_structured_mlp_pseudo_unit_candidate(
        parent,
        plan,
        targets,
        scores,
        calibration_split_sha256="8" * 64,
        generator_steps=160,
        generator_learning_rate=1e-2,
        generator_minibatch_rows=64,
        down_ridge=1e-5,
    )
    assert actual_feature_binding_observed
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert parent.execution_fingerprint() == parent_fingerprint

    second = build_structured_mlp_pseudo_unit_candidate(
        parent,
        plan,
        targets,
        scores,
        calibration_split_sha256="8" * 64,
        generator_steps=160,
        generator_learning_rate=1e-2,
        generator_minibatch_rows=64,
        down_ridge=1e-5,
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert first.report["report_sha256"] == second.report["report_sha256"]
    for name, expected in _artifact_tensors(
        first.direct_artifact_state
    ).items():
        torch.testing.assert_close(
            _artifact_tensors(second.direct_artifact_state)[name],
            expected,
            rtol=0,
            atol=0,
        )

    assert first.executor is first.direct_executor
    assert (
        first.direct_executor.config.transformer.feed_forward
        .intermediate_width
        == 3
    )
    assert not first.direct_executor.owns_source_model_weights
    assert not first.refit_executor.owns_source_model_weights
    direct_down = first.direct_executor.feed_forward.down_proj.weight
    torch.testing.assert_close(
        direct_down,
        plan.direct_down_weight(parent.feed_forward.down_proj.weight),
        rtol=0,
        atol=0,
    )
    singleton = torch.tensor(plan.singleton_indices)
    torch.testing.assert_close(
        first.direct_executor.feed_forward.gate_proj.weight[
            : plan.singleton_count
        ],
        parent.feed_forward.gate_proj.weight.index_select(0, singleton),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.direct_executor.feed_forward.up_proj.weight[
            : plan.singleton_count
        ],
        parent.feed_forward.up_proj.weight.index_select(0, singleton),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.refit_executor.feed_forward.gate_proj.weight,
        first.direct_executor.feed_forward.gate_proj.weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.refit_executor.feed_forward.up_proj.weight,
        first.direct_executor.feed_forward.up_proj.weight,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        first.refit_executor.feed_forward.down_proj.weight,
        first.direct_executor.feed_forward.down_proj.weight,
    )

    report = first.report
    assert report["status"]["direct_bundle_is_primary"]
    assert report["status"]["global_down_refit_is_ablation"]
    assert report["status"]["guard_opened"] is False
    assert report["status"]["heldout_opened"] is False
    assert report["actual_runtime_features"][
        "actual_runtime_features_used_for_down_refit"
    ]
    assert report["generator_fit"][
        "improved_three_term_normalized_objective"
    ]
    assert report["generator_fit"][
        "only_bundle_gate_up_rows_written"
    ]
    assert report["resources"]["parameters"]["removed_full_layer"] == 24
    assert report["resources"]["compute_per_valid_token"]["macs"] == {
        "source": 96,
        "candidate": 72,
        "removed": 24,
        "retained_ratio": 0.75,
    }
    assert report["preservation"][
        "attention_and_norm_tensors_preserved_exactly"
    ]
    assert report["variants"]["direct"]["down_globally_refit"] is False
    assert report["variants"]["global_down_refit"][
        "down_globally_refit"
    ]

    restored_direct = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            first.direct_artifact_state
        )
    )
    restored_refit = (
        StructuredTransformerLayerExecutor.from_artifact_state_dict(
            first.refit_artifact_state
        )
    )
    assert (
        restored_direct.execution_fingerprint()
        == first.direct_executor.execution_fingerprint()
    )
    assert (
        restored_refit.execution_fingerprint()
        == first.refit_executor.execution_fingerprint()
    )


def test_pipeline_rejects_bias_and_score_target_drift() -> None:
    biased, biased_plan, biased_targets, biased_scores = _fixture(
        projection_bias=True,
    )
    biased_fingerprint = biased.execution_fingerprint()
    with pytest.raises(ValueError, match="bias-free"):
        build_structured_mlp_pseudo_unit_candidate(
            biased,
            biased_plan,
            biased_targets,
            biased_scores,
            calibration_split_sha256="8" * 64,
            generator_steps=1,
        )
    assert biased.execution_fingerprint() == biased_fingerprint

    parent, plan, targets, scores = _fixture()
    score = scores[0]
    drifted = _score_batch(
        batch_id=score.batch_id,
        activations=score.projection_input.clone(),
        gradients=score.score_gradient.clone(),
        mask=score.valid_mask.clone(),
        provenance=score.provenance,
    )
    drifted.projection_input[score.valid_mask, 0] += 0.25
    parent_fingerprint = parent.execution_fingerprint()
    with pytest.raises(ValueError, match="bundling plan"):
        build_structured_mlp_pseudo_unit_candidate(
            parent,
            plan,
            targets,
            (drifted,),
            calibration_split_sha256="8" * 64,
            generator_steps=1,
        )
    assert parent.execution_fingerprint() == parent_fingerprint
