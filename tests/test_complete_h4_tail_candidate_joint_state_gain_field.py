from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    CandidateConditionedK64GainGradientExampleV4,
    fit_candidate_conditioned_k64_mean_kl_gains,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_gain_field import (
    JOINT_STATE_GAIN_PARAMETER_COUNT,
    CandidateConditionedK64JointInnerFamilyAnalyticRecord,
    CandidateConditionedK64JointStateGainAnalyticScreen,
    CandidateConditionedK64JointStateGainFieldFit,
    build_candidate_conditioned_k64_joint_inner_family_analytic_record,
    build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record,
    candidate_conditioned_k64_joint_tangent_design,
    fit_candidate_conditioned_k64_joint_state_gain_field,
    fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control,
    screen_candidate_conditioned_k64_joint_state_gain_capacity,
)
from fisher_graph.complete_h4_tail_candidate_state_gain_field import (
    STATE_FEATURE_RANK,
    CandidateConditionedK64StateFeatureExample,
    CandidateConditionedK64StateGainGradientExample,
    CandidateConditionedK64StaticAmplitudeControlFit,
    encode_candidate_conditioned_k64_state_features,
    fit_candidate_conditioned_k64_state_feature_codec,
    fit_candidate_conditioned_k64_static_amplitude_control,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    _runtime_tensor_sha256,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _joint_fit(
    *,
    held: str = "held",
    families: tuple[str, ...] = tuple(f"f{i}" for i in range(7)),
    codec_hash: str | None = None,
    refit_hash: str | None = None,
    gradient: torch.Tensor | None = None,
    fisher: torch.Tensor | None = None,
    logit: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> CandidateConditionedK64JointStateGainFieldFit:
    ids = tuple(f"g-{family}" for family in families)
    evidence = tuple(_hash(f"gradient-{family}") for family in families)
    default_fisher = torch.diag(
        torch.tensor([1.0, 0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    )
    default_logit = torch.diag(
        torch.tensor([1.0, 0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    )
    return CandidateConditionedK64JointStateGainFieldFit(
        held_family_id=held,
        training_family_ids=families,
        training_example_ids=ids,
        training_example_artifact_sha256s=evidence,
        refit_artifact_sha256=refit_hash or _hash(f"refit-{held}"),
        codec_artifact_sha256=codec_hash or _hash(f"codec-{held}-{families}"),
        ordered_directions_codec_sha256=_hash("codec-directions"),
        ordered_directions_refit_sha256=_hash("refit-directions"),
        feature_scale=(
            torch.ones(STATE_FEATURE_RANK, dtype=torch.float64)
            if scale is None
            else scale
        ),
        mean_gain_delta=torch.full((64,), -0.25, dtype=torch.float64),
        mean_gradient=(
            torch.tensor(
                [-0.5, -0.25, -0.25, -0.25, -0.25],
                dtype=torch.float64,
            )
            if gradient is None
            else gradient
        ),
        joint_fisher_gram=default_fisher if fisher is None else fisher,
        joint_logit_second_moment=default_logit if logit is None else logit,
    )


def _scalar_from_joint(
    value: CandidateConditionedK64JointStateGainFieldFit,
) -> CandidateConditionedK64StaticAmplitudeControlFit:
    return CandidateConditionedK64StaticAmplitudeControlFit(
        held_family_id=value.held_family_id,
        training_family_ids=value.training_family_ids,
        training_example_ids=value.training_example_ids,
        training_example_artifact_sha256s=(
            value.training_example_artifact_sha256s
        ),
        refit_artifact_sha256=value.refit_artifact_sha256,
        mean_gain_delta=value.mean_gain_delta,
        mean_gradient=float(value.mean_gradient[0]),
        fisher_energy=float(value.joint_fisher_gram[0, 0]),
    )


def _held_example(
    *,
    family: str,
    codec_hash: str,
    sign: float = 1.0,
) -> CandidateConditionedK64StateGainGradientExample:
    features = torch.zeros(8, STATE_FEATURE_RANK, dtype=torch.float64)
    scores = torch.full(
        (STATE_FEATURE_RANK, 8),
        -0.125 * sign,
        dtype=torch.float64,
    )
    for coordinate in range(STATE_FEATURE_RANK):
        features[2 * coordinate, coordinate] = 1.0
        features[2 * coordinate + 1, coordinate] = -1.0
        scores[coordinate, 2 * coordinate] += -0.5 * sign
        scores[coordinate, 2 * coordinate + 1] += 0.5 * sign
    return CandidateConditionedK64StateGainGradientExample(
        example_id=f"g-{family}",
        family_id=family,
        codec_artifact_sha256=codec_hash,
        standardized_state_features=features,
        token_row_direction_scores=scores,
        unit_token_teacher_kl=torch.ones(STATE_FEATURE_RANK, dtype=torch.float64),
    )


def _live_bundle():
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
                example_id=f"e{index}",
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
    v4_examples = tuple(
        CandidateConditionedK64GainGradientExampleV4(
            example_id=f"v4-{index}",
            family_id=f"f{index}",
            token_gain_gradients=torch.eye(64, dtype=torch.float64),
            token_teacher_kl=torch.ones(64, dtype=torch.float64),
        )
        for index in range(7)
    )
    refit = fit_candidate_conditioned_k64_mean_kl_gains(
        v4_examples,
        held_family_id="held",
        parent_fold_artifact_sha256="a" * 64,
        ordered_directions_sha256=_runtime_tensor_sha256(directions),
        ordered_token_fisher_relevance=torch.ones(64, dtype=torch.float64),
    )
    return directions, codec, tuple(gradients), refit


def _screen(
    *,
    positive_derivatives: set[tuple[int, int]] | None = None,
    low_residual_outers: set[int] | None = None,
    noop_outers: set[int] | None = None,
    scalar_wins_outers: set[int] | None = None,
    low_cosine_outers: set[int] | None = None,
) -> CandidateConditionedK64JointStateGainAnalyticScreen:
    positive = positive_derivatives or set()
    low_residual = low_residual_outers or set()
    noop = noop_outers or set()
    scalar_wins = scalar_wins_outers or set()
    low_cosine = low_cosine_outers or set()
    universe = tuple(f"f{i}" for i in range(8))
    folds = []
    for outer_index, outer in enumerate(universe):
        full_families = tuple(value for value in universe if value != outer)
        refit_hash = _hash(f"refit-{outer}")
        full_kwargs: dict[str, object] = {}
        if outer_index in low_residual:
            vector = torch.ones(STATE_FEATURE_RANK, dtype=torch.float64)
            fisher = torch.zeros(5, 5, dtype=torch.float64)
            fisher[0, 0] = 1.0
            fisher[0, 1:] = vector
            fisher[1:, 0] = vector
            fisher[1:, 1:] = torch.outer(vector, vector) + 0.01 * torch.eye(4)
            full_kwargs["fisher"] = fisher
        if outer_index in noop:
            full_kwargs["gradient"] = torch.zeros(5, dtype=torch.float64)
        full = _joint_fit(
            held=outer,
            families=full_families,
            refit_hash=refit_hash,
            **full_kwargs,
        )
        records: list[CandidateConditionedK64JointInnerFamilyAnalyticRecord] = []
        for inner_index, inner in enumerate(full_families):
            inner_families = tuple(
                value for value in full_families if value != inner
            )
            gradient = None
            if outer_index in low_cosine:
                gradient = torch.tensor(
                    [-0.5, 0.25, -0.25, -0.25, -0.25],
                    dtype=torch.float64,
                )
            if outer_index in scalar_wins:
                gradient = torch.tensor(
                    [-10.0, 0.01, 0.01, 0.01, 0.01],
                    dtype=torch.float64,
                )
            codec_hash = _hash(f"codec-{outer}-{inner}")
            inner_fit = _joint_fit(
                held=outer,
                families=inner_families,
                codec_hash=codec_hash,
                refit_hash=refit_hash,
                gradient=gradient,
            )
            records.append(
                build_candidate_conditioned_k64_joint_inner_family_analytic_record(
                    full,
                    inner_fit,
                    _scalar_from_joint(inner_fit),
                    _held_example(
                        family=inner,
                        codec_hash=codec_hash,
                        sign=(
                            -1.0
                            if (outer_index, inner_index) in positive
                            else 1.0
                        ),
                    ),
                )
            )
        folds.append(
            build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record(
                full,
                _scalar_from_joint(full),
                records,
            )
        )
    return screen_candidate_conditioned_k64_joint_state_gain_capacity(folds)


def test_joint_tangent_design_matches_direct_autograd() -> None:
    torch.manual_seed(72)
    features = torch.randn(7, STATE_FEATURE_RANK, dtype=torch.float64)
    scores = torch.randn(3, 7, dtype=torch.float64)
    example = CandidateConditionedK64StateGainGradientExample(
        example_id="e",
        family_id="f",
        codec_artifact_sha256="a" * 64,
        standardized_state_features=features,
        token_row_direction_scores=scores,
        unit_token_teacher_kl=torch.ones(3, dtype=torch.float64),
    )
    parameter = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    row_design = torch.cat((torch.ones(7, 1), features), dim=1)
    amplitudes = 1.0 + torch.tanh(row_design @ parameter)
    losses = scores @ amplitudes
    expected = torch.stack(
        [
            torch.autograd.grad(losses[index], parameter, retain_graph=True)[0]
            for index in range(losses.numel())
        ]
    )
    assert torch.allclose(
        candidate_conditioned_k64_joint_tangent_design(example),
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_joint_damping_uses_exact_five_dimensional_median() -> None:
    fisher = torch.diag(torch.tensor([1.0, 2.0, 100.0, 200.0, 300.0]))
    fit = _joint_fit(fisher=fisher)
    assert fit.damping == pytest.approx(10.0)


def test_joint_uses_one_row_logit_rms_trust_region() -> None:
    fit = _joint_fit(
        gradient=torch.full((5,), -100.0, dtype=torch.float64),
        fisher=torch.eye(5, dtype=torch.float64),
        logit=torch.eye(5, dtype=torch.float64),
    )
    assert fit.raw_logit_rms > 1.0
    assert fit.applied_logit_rms == pytest.approx(1.0)
    assert float(torch.linalg.vector_norm(fit.applied_parameter)) == pytest.approx(1.0)
    assert abs(fit.applied_intercept) < 1.0


def test_zero_joint_parameter_exactly_reproduces_v5_plus() -> None:
    fit = _joint_fit(gradient=torch.zeros(5, dtype=torch.float64))
    features = torch.randn(11, STATE_FEATURE_RANK, dtype=torch.float64)
    gains = fit.row_gains_tensor(features)
    assert torch.equal(fit.applied_parameter, torch.zeros(5))
    assert torch.equal(
        gains,
        fit.static_plus_gains_tensor().expand_as(gains),
    )


def test_direct_joint_solve_uses_scalar_state_cross_block() -> None:
    fisher = torch.eye(5, dtype=torch.float64)
    fisher[0, 1] = fisher[1, 0] = 0.5
    gradient = torch.tensor(
        [-1.0, -1.0, 0.0, 0.0, 0.0], dtype=torch.float64
    )
    fit = _joint_fit(gradient=gradient, fisher=fisher, logit=0.01 * torch.eye(5))
    expected = torch.linalg.solve(
        fisher + 0.1 * torch.eye(5, dtype=torch.float64),
        -gradient,
    )
    diagonal_only = -gradient / (fisher.diagonal() + 0.1)
    assert torch.allclose(fit.raw_parameter, expected, rtol=0.0, atol=1.0e-12)
    assert not torch.equal(fit.raw_parameter, diagonal_only)


def test_pair_helper_reproduces_exact_v6_scalar_artifact() -> None:
    directions, codec, examples, refit = _live_bundle()
    joint, scalar = (
        fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control(
            refit,
            codec,
            examples,
            ordered_directions=directions,
        )
    )
    independent = fit_candidate_conditioned_k64_static_amplitude_control(
        refit, codec, examples
    )
    assert scalar.artifact_sha256 == independent.artifact_sha256
    assert joint.mean_gradient[0] == scalar.mean_gradient
    assert joint.joint_fisher_gram[0, 0] == scalar.fisher_energy


def test_joint_fit_dually_authenticates_ordered_directions() -> None:
    directions, codec, examples, refit = _live_bundle()
    wrong_codec_side = directions.clone()
    wrong_codec_side[0] = torch.roll(wrong_codec_side[0], shifts=1)
    with pytest.raises(ValueError, match="V6 feature codec"):
        fit_candidate_conditioned_k64_joint_state_gain_field(
            refit,
            codec,
            examples,
            ordered_directions=wrong_codec_side,
        )
    mismatched_refit = replace(
        refit,
        ordered_directions_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="V4 refit"):
        fit_candidate_conditioned_k64_joint_state_gain_field(
            mismatched_refit,
            codec,
            examples,
            ordered_directions=directions,
        )


def test_inner_contributions_sum_and_cosine_excludes_intercept() -> None:
    full = _joint_fit()
    inner = _joint_fit(
        families=full.training_family_ids[:-1],
        codec_hash=_hash("inner-codec"),
        refit_hash=full.refit_artifact_sha256,
        gradient=torch.tensor(
            [0.5, -0.25, -0.25, -0.25, -0.25], dtype=torch.float64
        ),
    )
    held = _held_example(
        family=full.training_family_ids[-1],
        codec_hash=inner.codec_artifact_sha256,
    )
    record = build_candidate_conditioned_k64_joint_inner_family_analytic_record(
        full,
        inner,
        _scalar_from_joint(inner),
        held,
    )
    assert record.held_joint_total_derivative == pytest.approx(
        record.held_scalar_contribution + record.held_state_contribution
    )
    assert record.joint_minus_scalar_margin == pytest.approx(
        record.held_joint_total_derivative
        - record.held_scalar_comparator_derivative
    )
    assert full.applied_intercept * inner.applied_intercept < 0.0
    assert record.state_raw_slope_cosine == pytest.approx(1.0)
    assert record.metadata()["cosine_excludes_intercept"] is True


def test_inner_record_rejects_unrelated_example_in_six_family_split() -> None:
    full = _joint_fit()
    inner = _joint_fit(
        families=full.training_family_ids[:-1],
        codec_hash=_hash("inner-codec"),
        refit_hash=full.refit_artifact_sha256,
    )
    unrelated = replace(
        inner,
        training_example_ids=(
            "g-f0",
            "g-f1",
            "g-f2",
            "g-f3",
            "g-f4",
            "g-unrelated",
        ),
    )
    held = _held_example(
        family=full.training_family_ids[-1],
        codec_hash=unrelated.codec_artifact_sha256,
    )
    with pytest.raises(ValueError, match="does not match"):
        build_candidate_conditioned_k64_joint_inner_family_analytic_record(
            full,
            unrelated,
            _scalar_from_joint(unrelated),
            held,
        )


def test_full_joint_capacity_screen_passes_and_cell_count_is_not_a_gate() -> None:
    screen = _screen()
    assert screen.capacity_screen_passed is True
    assert screen.negative_inner_derivative_count == 56
    assert screen.joint_beats_scalar_fold_count == 8
    assert screen.joint_beats_scalar_cell_count == 56
    assert screen.cosine_stability_fold_count == 8
    assert screen.outcome == "joint_capacity_supported_for_finite_validation"
    metadata = screen.metadata()
    assert metadata["joint_beats_scalar_cell_count_is_gate"] is False
    assert metadata["analytic_forward_count"] == 112
    assert metadata["analytic_backward_count"] == 494
    assert metadata["finite_joint_candidate_execution_performed"] is False


def test_global_and_local_joint_derivative_gates_are_independent() -> None:
    only_41_negative = {
        (outer, inner)
        for outer in range(8)
        for inner in range(7)
        if outer * 7 + inner >= 41
    }
    screen = _screen(positive_derivatives=only_41_negative)
    assert screen.negative_inner_derivative_count == 41
    assert screen.negative_inner_global_gate_passed is False
    exact_boundary = _screen(
        positive_derivatives={
            (outer, inner) for outer in range(2) for inner in range(7)
        }
    )
    assert exact_boundary.negative_inner_derivative_count == 42
    assert exact_boundary.negative_inner_global_gate_passed is True
    assert exact_boundary.negative_inner_local_fold_count == 6
    assert exact_boundary.negative_inner_local_gate_passed is True
    concentrated = _screen(
        positive_derivatives={
            (outer, inner) for outer in range(3) for inner in range(7)
        }
    )
    assert concentrated.negative_inner_local_fold_count == 5
    assert concentrated.negative_inner_local_gate_passed is False


@pytest.mark.parametrize(
    ("keyword", "count_attribute", "gate_attribute"),
    (
        (
            "low_residual_outers",
            "residual_energy_pass_count",
            "residual_energy_gate_passed",
        ),
        ("noop_outers", "non_noop_fold_count", "non_noop_gate_passed"),
        (
            "scalar_wins_outers",
            "joint_beats_scalar_fold_count",
            "joint_beats_scalar_gate_passed",
        ),
        (
            "low_cosine_outers",
            "cosine_stability_fold_count",
            "cosine_stability_gate_passed",
        ),
    ),
)
def test_each_six_of_eight_gate_fails_at_five_and_passes_at_six(
    keyword: str,
    count_attribute: str,
    gate_attribute: str,
) -> None:
    five = _screen(**{keyword: {0, 1, 2}})
    six = _screen(**{keyword: {0, 1}})
    assert getattr(five, count_attribute) == 5
    assert getattr(five, gate_attribute) is False
    assert getattr(six, count_attribute) == 6
    assert getattr(six, gate_attribute) is True


def test_joint_metadata_contains_no_tensor_objects_and_detects_mutation() -> None:
    screen = _screen()
    inner = screen.fold_records[0].inner_family_records[0]
    payload = {
        "fit": inner.inner_joint_fit.metadata(),
        "inner": inner.metadata(),
        "fold": screen.fold_records[0].metadata(),
        "screen": screen.metadata(),
    }

    def assert_no_tensor(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for nested in value.values():
                assert_no_tensor(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                assert_no_tensor(nested)

    assert_no_tensor(payload)
    assert '"raw_tensors_serialized": true' not in json.dumps(payload)
    inner.inner_joint_fit.applied_parameter[0] += 1.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        screen.validate_integrity()
