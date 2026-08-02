from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    CandidateConditionedK64GainGradientExampleV4,
    fit_candidate_conditioned_k64_mean_kl_gains,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_gain_field import (
    fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_gain_finite import (
    THREE_ARM_FINITE_NAMES,
    V8_EXPECTED_CANDIDATE_FORWARD_COUNT,
    CandidateConditionedK64ThreeArmFiniteExample,
    CandidateConditionedK64ThreeArmGainSupport,
    build_candidate_conditioned_k64_three_arm_gain_support,
    candidate_conditioned_k64_gain_correction_rows,
    candidate_conditioned_k64_gain_scaled_tail_rows,
    compare_candidate_conditioned_k64_three_arm_finite_examples,
)
from fisher_graph.complete_h4_tail_candidate_state_gain_field import (
    STATE_FEATURE_RANK,
    CandidateConditionedK64StateFeatureExample,
    CandidateConditionedK64StateGainGradientExample,
    encode_candidate_conditioned_k64_state_features,
    fit_candidate_conditioned_k64_state_feature_codec,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _real_bundle():
    directions = torch.eye(64, dtype=torch.float64)
    feature_examples = []
    for index in range(7):
        rows = torch.zeros(8, 64, dtype=torch.float64)
        rows[:, :STATE_FEATURE_RANK] = torch.vstack(
            (
                torch.eye(STATE_FEATURE_RANK, dtype=torch.float64),
                -torch.eye(STATE_FEATURE_RANK, dtype=torch.float64),
            )
        )
        feature_examples.append(
            CandidateConditionedK64StateFeatureExample(
                example_id=f"fit-e{index}",
                family_id=f"f{index}",
                base_h4_support_rows=rows,
            )
        )
    codec = fit_candidate_conditioned_k64_state_feature_codec(
        feature_examples,
        held_family_id="held",
        ordered_directions=directions,
    )
    gradients = []
    for value in feature_examples:
        features = encode_candidate_conditioned_k64_state_features(
            codec,
            base_h4_support_rows=value.base_h4_support_rows,
            ordered_directions=directions,
        )
        scores = torch.zeros(5, 8, dtype=torch.float64)
        scores[0] = 0.125
        for coordinate in range(STATE_FEATURE_RANK):
            scores[coordinate + 1, 2 * coordinate] = 0.25
            scores[coordinate + 1, 2 * coordinate + 1] = -0.25
        gradients.append(
            CandidateConditionedK64StateGainGradientExample(
                example_id=value.example_id,
                family_id=value.family_id,
                codec_artifact_sha256=codec.artifact_sha256,
                standardized_state_features=features,
                token_row_direction_scores=scores,
                unit_token_teacher_kl=torch.ones(5, dtype=torch.float64),
            )
        )
    refit = fit_candidate_conditioned_k64_mean_kl_gains(
        tuple(
            CandidateConditionedK64GainGradientExampleV4(
                example_id=value.example_id,
                family_id=value.family_id,
                token_gain_gradients=torch.eye(64, dtype=torch.float64),
                token_teacher_kl=torch.ones(64, dtype=torch.float64),
            )
            for value in feature_examples
        ),
        held_family_id="held",
        parent_fold_artifact_sha256=_hash("parent"),
        ordered_directions_sha256=_runtime_tensor_sha256(directions),
        ordered_token_fisher_relevance=torch.ones(64, dtype=torch.float64),
    )
    joint, scalar = (
        fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control(
            refit,
            codec,
            gradients,
            ordered_directions=directions,
        )
    )
    return directions, refit, codec, scalar, joint


def _synthetic_support(
    held: str, prompt: int, *, binding_suffix: str = ""
) -> CandidateConditionedK64ThreeArmGainSupport:
    universe = tuple(f"f{index}" for index in range(8))
    families = tuple(value for value in universe if value != held)
    training_ids = tuple(f"fit-{value}" for value in families)
    return CandidateConditionedK64ThreeArmGainSupport(
        phase="held",
        held_family_id=held,
        example_id=f"held-{held}-{prompt}",
        family_id=held,
        training_family_ids=families,
        training_example_ids=training_ids,
        parent_fold_artifact_sha256=_hash(
            f"parent-fold-{held}{binding_suffix}"
        ),
        refit_artifact_sha256=_hash(f"refit-{held}{binding_suffix}"),
        codec_artifact_sha256=_hash(f"codec-{held}{binding_suffix}"),
        scalar_fit_artifact_sha256=_hash(f"scalar-{held}{binding_suffix}"),
        joint_fit_artifact_sha256=_hash(f"joint-{held}{binding_suffix}"),
        ordered_directions_codec_sha256=_hash(
            f"codec-directions-{held}{binding_suffix}"
        ),
        ordered_directions_refit_sha256=_hash(
            f"runtime-directions-{held}{binding_suffix}"
        ),
        base_h4_support_rows_sha256=_hash(f"base-{held}-{prompt}"),
        standardized_state_features=torch.zeros(3, 4, dtype=torch.float64),
        static_plus_gains=torch.full((64,), 0.99, dtype=torch.float64),
        exact_scalar_gains=torch.full((64,), 0.98, dtype=torch.float64),
        joint_row_gains=torch.full((3, 64), 0.97, dtype=torch.float64),
    )


def _tokens(mean: float) -> torch.Tensor:
    return torch.tensor((0.5 * mean, 1.5 * mean), dtype=torch.float64)


def _finite_example(
    held: str,
    prompt: int,
    *,
    static: float,
    scalar: float,
    joint: float,
    support: CandidateConditionedK64ThreeArmGainSupport | None = None,
) -> CandidateConditionedK64ThreeArmFiniteExample:
    support = support or _synthetic_support(held, prompt)
    static_tokens = _tokens(static)
    stem = f"{held}-{prompt}"
    return CandidateConditionedK64ThreeArmFiniteExample(
        gain_support=support,
        model_inputs_sha256=_hash(f"inputs-{stem}"),
        bridge_binding_sha256=_hash(f"bridge-{stem}"),
        prefix_artifact_sha256=_hash(f"prefix-{stem}"),
        support_mask_sha256=_hash(f"mask-{stem}"),
        teacher_logits_sha256=_hash(f"teacher-{stem}"),
        endpoint_supervised_grid_sha256=_hash(f"grid-{stem}"),
        pinned_unit_receipt_sha256=_hash(f"unit-receipt-{stem}"),
        pinned_unit_token_teacher_kl_sha256=_hash(f"unit-kl-{stem}"),
        pinned_unit_mean_teacher_kl=1.1,
        arm_provider_artifact_sha256s=tuple(
            _hash(f"provider-{arm}-{stem}") for arm in THREE_ARM_FINITE_NAMES
        ),
        arm_execution_artifact_sha256s=tuple(
            _hash(f"execution-{arm}-{stem}") for arm in THREE_ARM_FINITE_NAMES
        ),
        arm_correction_rows_sha256s=tuple(
            _hash(f"rows-{arm}-{stem}") for arm in THREE_ARM_FINITE_NAMES
        ),
        arm_full_correction_sha256s=tuple(
            _hash(f"correction-{arm}-{stem}") for arm in THREE_ARM_FINITE_NAMES
        ),
        pinned_v5_static_plus_token_teacher_kl_sha256=(
            _runtime_tensor_sha256(static_tokens)
        ),
        static_plus_token_teacher_kl=static_tokens,
        exact_scalar_token_teacher_kl=_tokens(scalar),
        joint_token_teacher_kl=_tokens(joint),
    )


def _held_grid(
    *, static: float, scalar: float, joint: float
) -> tuple[CandidateConditionedK64ThreeArmFiniteExample, ...]:
    return tuple(
        _finite_example(
            f"f{family}",
            prompt,
            static=static,
            scalar=scalar,
            joint=joint,
        )
        for family in range(8)
        for prompt in range(2)
    )


def _held_grid_by_family(
    *,
    static: tuple[float, ...],
    scalar: tuple[float, ...],
    joint: tuple[float, ...],
) -> tuple[CandidateConditionedK64ThreeArmFiniteExample, ...]:
    assert len(static) == len(scalar) == len(joint) == 8
    return tuple(
        _finite_example(
            f"f{family}",
            prompt,
            static=static[family],
            scalar=scalar[family],
            joint=joint[family],
        )
        for family in range(8)
        for prompt in range(2)
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_row_correction_generalizes_v5_static_algebra_and_autograd() -> None:
    directions = torch.eye(64, dtype=torch.float64)
    tail = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)
    supported = torch.randn(3, 64, dtype=torch.float64, requires_grad=True)
    static = torch.linspace(0.75, 1.25, 64, dtype=torch.float64)
    actual_static = candidate_conditioned_k64_gain_correction_rows(
        supported_rows=supported,
        tail_rows=tail,
        ordered_directions=directions,
        gains=static,
    )
    assert torch.equal(actual_static, supported + tail * static)

    row_gains = torch.linspace(
        0.5, 1.5, 3 * 64, dtype=torch.float64
    ).reshape(3, 64)
    row_gains.requires_grad_()
    scaled = candidate_conditioned_k64_gain_scaled_tail_rows(
        tail_rows=tail,
        ordered_directions=directions,
        gains=row_gains,
    )
    assert torch.equal(scaled, tail * row_gains)
    scaled.sum().backward()
    assert torch.equal(row_gains.grad, tail.detach())
    assert torch.equal(tail.grad, row_gains.detach())


def test_static_contraction_is_bitwise_the_legacy_v5_expression() -> None:
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(80, 64, generator=generator, dtype=torch.float64)
    directions = torch.linalg.qr(raw, mode="reduced").Q.T.contiguous()
    tail = torch.randn(9, 80, generator=generator, dtype=torch.float64)
    gains = torch.linspace(0.8, 1.2, 64, dtype=torch.float64)
    expected = (((tail @ directions.T) * gains) @ directions).contiguous()
    actual = candidate_conditioned_k64_gain_scaled_tail_rows(
        tail_rows=tail,
        ordered_directions=directions,
        gains=gains,
    )
    assert torch.equal(actual, expected)


def test_gain_support_uses_frozen_pre_gate_codec_and_exact_controls() -> None:
    directions, refit, codec, scalar, joint = _real_bundle()
    rows = torch.randn(6, 64, dtype=torch.float64)
    support = build_candidate_conditioned_k64_three_arm_gain_support(
        refit,
        codec,
        scalar,
        joint,
        phase="held",
        example_id="held-example",
        family_id="held",
        base_h4_support_rows=rows,
        ordered_directions=directions,
    )
    expected_features = encode_candidate_conditioned_k64_state_features(
        codec,
        base_h4_support_rows=rows,
        ordered_directions=directions,
    )
    assert torch.equal(support.standardized_state_features, expected_features)
    assert torch.equal(support.static_plus_gains, joint.static_plus_gains_tensor())
    assert torch.equal(support.exact_scalar_gains, scalar.gains_tensor())
    assert torch.equal(
        support.joint_row_gains,
        joint.row_gains_tensor(expected_features),
    )

    delta = joint.mean_gain_delta
    zero_logit = 1.0 + (1.0 / 64.0) * delta
    assert torch.equal(zero_logit, support.static_plus_gains)
    scalar_only = (
        1.0
        + (1.0 / 64.0)
        * (
            1.0
            + torch.tanh(
                torch.tensor(scalar.applied_coefficient, dtype=torch.float64)
            )
        )
        * delta
    )
    assert torch.equal(scalar_only, support.exact_scalar_gains)
    assert support.metadata()["feature_source"] == (
        "pre_gate_bridge_base_H4_support_rows"
    )


def test_gain_support_rejects_tune_and_cross_bound_fits() -> None:
    directions, refit, codec, scalar, joint = _real_bundle()
    rows = torch.randn(4, 64, dtype=torch.float64)
    with pytest.raises(ValueError, match="geometry"):
        build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            codec,
            scalar,
            joint,
            phase="tune",
            example_id="tune-example",
            family_id="f0",
            base_h4_support_rows=rows,
            ordered_directions=directions,
        )
    stale_scalar = replace(scalar, refit_artifact_sha256=_hash("stale-refit"))
    with pytest.raises(ValueError, match="bindings"):
        build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            codec,
            stale_scalar,
            joint,
            phase="held",
            example_id="held-example",
            family_id="held",
            base_h4_support_rows=rows,
            ordered_directions=directions,
        )
    wrong_directions = directions.clone()
    wrong_directions[0, 0] = -1.0
    with pytest.raises(ValueError, match="directions"):
        build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            codec,
            scalar,
            joint,
            phase="held",
            example_id="held-example",
            family_id="held",
            base_h4_support_rows=rows,
            ordered_directions=wrong_directions,
        )


def test_gain_support_authenticates_both_direction_hash_domains() -> None:
    directions, refit, codec, scalar, joint = _real_bundle()
    rows = torch.randn(4, 64, dtype=torch.float64)

    bad_codec = replace(codec, ordered_directions_sha256=_hash("bad-codec-D"))
    bad_codec_joint = replace(
        joint,
        codec_artifact_sha256=bad_codec.artifact_sha256,
        ordered_directions_codec_sha256=bad_codec.ordered_directions_sha256,
    )
    with pytest.raises(ValueError, match="directions"):
        build_candidate_conditioned_k64_three_arm_gain_support(
            refit,
            bad_codec,
            scalar,
            bad_codec_joint,
            phase="held",
            example_id="held-example",
            family_id="held",
            base_h4_support_rows=rows,
            ordered_directions=directions,
        )

    bad_refit = replace(
        refit, ordered_directions_sha256=_hash("bad-runtime-D")
    )
    bad_refit_scalar = replace(
        scalar, refit_artifact_sha256=bad_refit.artifact_sha256
    )
    bad_refit_joint = replace(
        joint,
        refit_artifact_sha256=bad_refit.artifact_sha256,
        ordered_directions_refit_sha256=bad_refit.ordered_directions_sha256,
    )
    with pytest.raises(ValueError, match="authentication"):
        build_candidate_conditioned_k64_three_arm_gain_support(
            bad_refit,
            codec,
            bad_refit_scalar,
            bad_refit_joint,
            phase="held",
            example_id="held-example",
            family_id="held",
            base_h4_support_rows=rows,
            ordered_directions=directions,
        )


def test_gain_support_rejects_held_family_in_fit_evidence() -> None:
    support = _synthetic_support("f0", 0)
    with pytest.raises(ValueError, match="geometry"):
        replace(
            support,
            training_family_ids=("f0", "f2", "f3", "f4", "f5", "f6", "f7"),
        )


def test_finite_example_binds_static_replay_and_defends_tensors() -> None:
    value = _finite_example(
        "f0", 0, static=1.0, scalar=0.9, joint=0.8
    )
    metadata = value.metadata()
    assert metadata["static_plus_replayed_pinned_v5_exactly"] is True
    assert metadata["unit_reference_executed_in_v8"] is False
    assert metadata["three_candidate_forwards_executed"] is True
    assert not _contains_tensor(metadata)
    returned = value.token_teacher_kl_tensor("v7_joint")
    returned[0] = 99.0
    assert value.mean_teacher_kl("v7_joint") == pytest.approx(0.8)

    with pytest.raises(ValueError, match="evidence"):
        replace(
            value,
            pinned_v5_static_plus_token_teacher_kl_sha256=_hash("wrong-plus"),
        )
    value.joint_token_teacher_kl[0] += 1.0
    with pytest.raises(RuntimeError, match="drifted"):
        value.validate_integrity()


def test_held_comparison_applies_v5_gates_to_joint_vs_each_control() -> None:
    comparison = compare_candidate_conditioned_k64_three_arm_finite_examples(
        _held_grid(static=1.0, scalar=0.9, joint=0.7)
    )
    metadata = comparison.metadata()
    assert comparison.cell_count == 16
    assert metadata["candidate_execution_count"] == 48
    assert V8_EXPECTED_CANDIDATE_FORWARD_COUNT == 48
    assert metadata["total_forward_count"] == 176
    assert metadata["total_backward_call_count"] == 494
    assert comparison.family_equal_mean_teacher_kl("v5_static_plus") == 1.0
    assert comparison.family_equal_mean_teacher_kl("v6_exact_scalar") == 0.9
    assert comparison.family_equal_mean_teacher_kl("v7_joint") == 0.7
    assert metadata["joint_cleared_both_controls"] is True
    controls = metadata["joint_vs_each_control_gates"]
    assert controls["v5_static_plus"]["passed"] is True
    assert controls["v6_exact_scalar"]["passed"] is True
    assert comparison.joint_vs_static_plus_passed is True
    assert comparison.joint_vs_exact_scalar_passed is True
    assert comparison.joint_cleared_both_controls is True
    assert "selected_arm" not in metadata
    assert "oracle" not in repr(metadata).lower()
    assert not _contains_tensor(metadata)


def test_held_aggregation_is_two_prompt_then_eight_family_equal() -> None:
    values = tuple(
        _finite_example(
            f"f{family}",
            prompt,
            static=(0.5 if prompt == 0 else 1.5) + family * 0.1,
            scalar=(0.4 if prompt == 0 else 1.4) + family * 0.1,
            joint=(0.3 if prompt == 0 else 1.3) + family * 0.1,
        )
        for family in range(8)
        for prompt in range(2)
    )
    comparison = compare_candidate_conditioned_k64_three_arm_finite_examples(
        values
    )
    assert torch.equal(
        comparison.outer_family_mean_teacher_kl[:, 1],
        torch.tensor(
            tuple(1.0 + family * 0.1 for family in range(8)),
            dtype=torch.float64,
        ),
    )
    assert comparison.family_equal_mean_teacher_kl("v5_static_plus") == (
        pytest.approx(1.35)
    )


def test_joint_control_gates_are_independent() -> None:
    comparison = compare_candidate_conditioned_k64_three_arm_finite_examples(
        _held_grid(static=1.0, scalar=0.5, joint=0.7)
    )
    controls = comparison.metadata()["joint_vs_each_control_gates"]
    assert controls["v5_static_plus"]["passed"] is True
    assert controls["v6_exact_scalar"]["passed"] is False
    assert comparison.metadata()["joint_cleared_both_controls"] is False


def test_each_v5_threshold_boundary_is_inclusive_and_independent() -> None:
    control = (1.0,) * 8
    cap = 1.05 + 1.0e-8
    winning_value = (8.0 * 0.98 - 2.0 * cap) / 6.0
    boundary_joint = (winning_value,) * 6 + (cap,) * 2
    boundary = compare_candidate_conditioned_k64_three_arm_finite_examples(
        _held_grid_by_family(
            static=control,
            scalar=control,
            joint=boundary_joint,
        )
    ).metadata()["joint_vs_each_control_gates"]["v5_static_plus"]
    assert boundary["passed"] is True
    assert dict(boundary["gates"]) == {
        "every_family_within_1_05_times_control_plus_1e_minus_8": True,
        "family_macro_relative_improvement_at_least_2pct": True,
        "held_family_improvement_count_at_least_6_of_8": True,
    }

    macro_fail = (winning_value + 0.01,) * 6 + (cap,) * 2
    macro_gates = dict(
        compare_candidate_conditioned_k64_three_arm_finite_examples(
            _held_grid_by_family(
                static=control, scalar=control, joint=macro_fail
            )
        ).metadata()["joint_vs_each_control_gates"]["v5_static_plus"]["gates"]
    )
    assert macro_gates["family_macro_relative_improvement_at_least_2pct"] is False
    assert macro_gates["held_family_improvement_count_at_least_6_of_8"] is True

    win_fail = (0.9,) * 5 + (1.0,) + (cap,) * 2
    win_gates = dict(
        compare_candidate_conditioned_k64_three_arm_finite_examples(
            _held_grid_by_family(static=control, scalar=control, joint=win_fail)
        ).metadata()["joint_vs_each_control_gates"]["v5_static_plus"]["gates"]
    )
    assert win_gates["family_macro_relative_improvement_at_least_2pct"] is True
    assert win_gates["held_family_improvement_count_at_least_6_of_8"] is False

    cap_fail = (0.9,) * 6 + (cap + 1.0e-9,) + (cap,)
    cap_gates = dict(
        compare_candidate_conditioned_k64_three_arm_finite_examples(
            _held_grid_by_family(static=control, scalar=control, joint=cap_fail)
        ).metadata()["joint_vs_each_control_gates"]["v5_static_plus"]["gates"]
    )
    assert cap_gates[
        "every_family_within_1_05_times_control_plus_1e_minus_8"
    ] is False
    assert cap_gates["held_family_improvement_count_at_least_6_of_8"] is True


def test_held_comparison_rejects_incomplete_and_cross_bound_grids() -> None:
    values = list(_held_grid(static=1.0, scalar=0.9, joint=0.7))
    with pytest.raises(ValueError, match="cell count"):
        compare_candidate_conditioned_k64_three_arm_finite_examples(values[:-1])

    stale_support = _synthetic_support("f0", 1, binding_suffix="-stale")
    values[1] = _finite_example(
        "f0",
        1,
        static=1.0,
        scalar=0.9,
        joint=0.7,
        support=stale_support,
    )
    with pytest.raises(ValueError, match="binding varies"):
        compare_candidate_conditioned_k64_three_arm_finite_examples(values)
