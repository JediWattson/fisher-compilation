from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    contract_candidate_teacher_kl_gain_scores,
)
from fisher_graph.complete_h4_tail_candidate_state_gain_field import (
    STATE_FEATURE_RANK,
    STATE_GAIN_BASE_STEP,
    CandidateConditionedK64InnerFamilyAnalyticRecord,
    CandidateConditionedK64StateFeatureExample,
    CandidateConditionedK64StateGainAnalyticScreen,
    CandidateConditionedK64StateGainFieldFit,
    CandidateConditionedK64StateGainGradientExample,
    CandidateConditionedK64StaticAmplitudeControlFit,
    build_candidate_conditioned_k64_inner_family_analytic_record,
    build_candidate_conditioned_k64_state_gain_fold_analytic_record,
    contract_candidate_conditioned_k64_row_direction_scores,
    encode_candidate_conditioned_k64_state_features,
    fit_candidate_conditioned_k64_state_feature_codec,
    reduce_candidate_conditioned_k64_row_mode_scores,
    screen_candidate_conditioned_k64_state_gain_capacity,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _fit(
    *,
    held: str = "held",
    families: tuple[str, ...] = tuple(f"f{i}" for i in range(7)),
    codec_hash: str | None = None,
    refit_hash: str | None = None,
    gradient: torch.Tensor | None = None,
    fisher: torch.Tensor | None = None,
    feature: torch.Tensor | None = None,
    cross: torch.Tensor | None = None,
    static_energy: float = 0.0,
    scale: torch.Tensor | None = None,
) -> CandidateConditionedK64StateGainFieldFit:
    ids = tuple(f"g-{family}" for family in families)
    evidence = tuple(_hash(f"gradient-{family}") for family in families)
    return CandidateConditionedK64StateGainFieldFit(
        held_family_id=held,
        training_family_ids=families,
        training_example_ids=ids,
        training_example_artifact_sha256s=evidence,
        refit_artifact_sha256=refit_hash or _hash(f"refit-{held}"),
        codec_artifact_sha256=codec_hash or _hash(f"codec-{held}-{families}"),
        feature_scale=(
            torch.ones(STATE_FEATURE_RANK, dtype=torch.float64)
            if scale is None
            else scale
        ),
        mean_gain_delta=torch.full((64,), -0.25, dtype=torch.float64),
        mean_gradient=(
            torch.full((STATE_FEATURE_RANK,), -0.25, dtype=torch.float64)
            if gradient is None
            else gradient
        ),
        state_fisher_gram=(
            0.25 * torch.eye(STATE_FEATURE_RANK, dtype=torch.float64)
            if fisher is None
            else fisher
        ),
        feature_second_moment=(
            0.25 * torch.eye(STATE_FEATURE_RANK, dtype=torch.float64)
            if feature is None
            else feature
        ),
        state_static_cross_moment=(
            torch.zeros(STATE_FEATURE_RANK, dtype=torch.float64)
            if cross is None
            else cross
        ),
        static_fisher_energy=static_energy,
    )


def _static_from_fit(
    value: CandidateConditionedK64StateGainFieldFit,
    *,
    mean_gradient: float = 0.0,
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
        mean_gradient=mean_gradient,
        fisher_energy=value.static_fisher_energy,
    )


def _state_example(
    *,
    family: str,
    codec_hash: str,
    sign: float = 1.0,
    static_offset: float = 0.0,
) -> CandidateConditionedK64StateGainGradientExample:
    features = torch.zeros(8, STATE_FEATURE_RANK, dtype=torch.float64)
    scores = torch.zeros(STATE_FEATURE_RANK, 8, dtype=torch.float64)
    for coordinate in range(STATE_FEATURE_RANK):
        features[2 * coordinate, coordinate] = 1.0
        features[2 * coordinate + 1, coordinate] = -1.0
        scores[coordinate, 2 * coordinate] = -0.5 * sign
        scores[coordinate, 2 * coordinate + 1] = 0.5 * sign
    scores += static_offset
    return CandidateConditionedK64StateGainGradientExample(
        example_id=f"g-{family}",
        family_id=family,
        codec_artifact_sha256=codec_hash,
        standardized_state_features=features,
        token_row_direction_scores=scores,
        unit_token_teacher_kl=torch.ones(STATE_FEATURE_RANK, dtype=torch.float64),
    )


def _screen(
    *,
    positive_derivatives: set[tuple[int, int]] | None = None,
    low_residual_outers: set[int] | None = None,
    noop_outers: set[int] | None = None,
    scalar_wins_outers: set[int] | None = None,
    low_cosine_outers: set[int] | None = None,
) -> CandidateConditionedK64StateGainAnalyticScreen:
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
            full_kwargs = {
                "fisher": torch.outer(vector, vector)
                + 0.01 * torch.eye(STATE_FEATURE_RANK, dtype=torch.float64),
                "cross": vector,
                "static_energy": 1.0,
            }
        if outer_index in noop:
            full_kwargs["gradient"] = torch.zeros(
                STATE_FEATURE_RANK, dtype=torch.float64
            )
        full = _fit(
            held=outer,
            families=full_families,
            refit_hash=refit_hash,
            **full_kwargs,
        )
        records: list[CandidateConditionedK64InnerFamilyAnalyticRecord] = []
        for inner_index, inner in enumerate(full_families):
            inner_families = tuple(
                value for value in full_families if value != inner
            )
            codec_hash = _hash(f"codec-{outer}-{inner}")
            inner_gradient = None
            if outer_index in low_cosine:
                inner_gradient = torch.tensor(
                    [0.25, -0.25, -0.25, -0.25], dtype=torch.float64
                )
            inner_fit = _fit(
                held=outer,
                families=inner_families,
                codec_hash=codec_hash,
                refit_hash=refit_hash,
                gradient=inner_gradient,
                static_energy=(1.0 if outer_index in scalar_wins else 0.0),
            )
            records.append(
                build_candidate_conditioned_k64_inner_family_analytic_record(
                    full,
                    inner_fit,
                    _static_from_fit(
                        inner_fit,
                        mean_gradient=(
                            -10.0 if outer_index in scalar_wins else 0.0
                        ),
                    ),
                    _state_example(
                        family=inner,
                        codec_hash=codec_hash,
                        sign=(
                            -1.0
                            if (outer_index, inner_index) in positive
                            else 1.0
                        ),
                        static_offset=(
                            -0.25 if outer_index in scalar_wins else 0.0
                        ),
                    ),
                )
            )
        folds.append(
            build_candidate_conditioned_k64_state_gain_fold_analytic_record(
                full,
                _static_from_fit(full),
                records,
            )
        )
    return screen_candidate_conditioned_k64_state_gain_capacity(folds)


def test_row_resolved_contraction_matches_autograd_and_v4_sum() -> None:
    torch.manual_seed(71)
    rows, tokens, width = 5, 3, 83
    directions = torch.linalg.qr(
        torch.randn(width, 64, dtype=torch.float64), mode="reduced"
    ).Q.T.contiguous()
    tail = torch.randn(rows, width, dtype=torch.float64)
    gradients = torch.randn(tokens, rows, width, dtype=torch.float64)
    delta = torch.randn(64, dtype=torch.float64)
    amplitudes = torch.ones(rows, dtype=torch.float64, requires_grad=True)
    coordinates = tail @ directions.T
    gains = 1.0 + STATE_GAIN_BASE_STEP * amplitudes[:, None] * delta
    candidate = (coordinates * gains) @ directions
    losses = torch.einsum("rw,trw->t", candidate, gradients)
    expected = torch.stack(
        [
            torch.autograd.grad(losses[index], amplitudes, retain_graph=True)[0]
            for index in range(tokens)
        ]
    )
    actual = contract_candidate_conditioned_k64_row_direction_scores(
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradients,
        mean_gain_delta=delta,
    )
    assert torch.allclose(actual, expected, rtol=0.0, atol=1.0e-12)
    amplitudes_by_mode = tail @ directions.T
    gradient_coordinates = torch.einsum("trw,kw->trk", gradients, directions)
    reduced = reduce_candidate_conditioned_k64_row_mode_scores(
        token_row_mode_scores=(
            amplitudes_by_mode.unsqueeze(0) * gradient_coordinates
        ),
        mean_gain_delta=delta,
    )
    assert torch.equal(actual, reduced)
    v4 = contract_candidate_teacher_kl_gain_scores(
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradients,
    )
    assert torch.allclose(
        actual.sum(dim=1),
        STATE_GAIN_BASE_STEP * (v4 @ delta),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_codec_is_fit_only_family_equal_and_uses_first_four_directions() -> None:
    directions = torch.eye(64, dtype=torch.float64)
    family_values = (100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    examples = []
    for index, value in enumerate(family_values):
        count = 101 if index == 0 else 1
        rows = torch.zeros(count, 64, dtype=torch.float64)
        rows[:, :STATE_FEATURE_RANK] = value
        examples.append(
            CandidateConditionedK64StateFeatureExample(
                example_id=f"e{index}",
                family_id=f"f{index}",
                base_h4_support_rows=rows,
            )
        )
    codec = fit_candidate_conditioned_k64_state_feature_codec(
        examples,
        held_family_id="held",
        ordered_directions=directions,
    )
    expected_center = sum(family_values) / len(family_values)
    assert torch.equal(
        codec.feature_center,
        torch.full((STATE_FEATURE_RANK,), expected_center, dtype=torch.float64),
    )
    runtime = torch.zeros(2, 64, dtype=torch.float64)
    runtime[:, :STATE_FEATURE_RANK] = 7.0
    encoded = encode_candidate_conditioned_k64_state_features(
        codec,
        base_h4_support_rows=runtime,
        ordered_directions=directions,
    )
    assert torch.equal(
        encoded,
        (runtime[:, :STATE_FEATURE_RANK] - codec.feature_center)
        / codec.feature_scale,
    )
    runtime[0, 0] = 999.0
    assert encoded[0, 0] != 999.0
    assert codec.metadata()["post_gate_state_used"] is False if (
        "post_gate_state_used" in codec.metadata()
    ) else codec.metadata()["feature_source"].startswith("pre_gate")


def test_codec_rejects_zero_or_near_zero_family_equal_variance() -> None:
    examples = tuple(
        CandidateConditionedK64StateFeatureExample(
            example_id=f"e{i}",
            family_id=f"f{i}",
            base_h4_support_rows=torch.zeros(2, 64, dtype=torch.float64),
        )
        for i in range(7)
    )
    with pytest.raises(ValueError, match="zero or near-zero"):
        fit_candidate_conditioned_k64_state_feature_codec(
            examples,
            held_family_id="held",
            ordered_directions=torch.eye(64, dtype=torch.float64),
        )


def test_zero_weight_is_exact_v5_static_plus_and_never_reverses() -> None:
    fit = _fit(gradient=torch.zeros(STATE_FEATURE_RANK, dtype=torch.float64))
    features = torch.randn(9, STATE_FEATURE_RANK, dtype=torch.float64)
    gains = fit.row_gains_tensor(features)
    expected = fit.static_plus_gains_tensor().expand_as(gains)
    assert torch.equal(fit.applied_weight, torch.zeros(STATE_FEATURE_RANK))
    assert torch.equal(gains, expected)
    active = _fit()
    amplitudes = active.row_amplitudes_tensor(1000.0 * features)
    assert bool((amplitudes >= 0.0).all())
    assert bool((amplitudes <= 2.0).all())
    assert active.metadata()["incremental_float_count_including_codec"] == 12


def test_design_condition_is_singular_value_not_squared_gram_condition() -> None:
    ratio = 2500.0
    rho = (ratio - 1.0) / (ratio + 1.0)
    fisher = torch.eye(STATE_FEATURE_RANK, dtype=torch.float64)
    fisher[0, 1] = fisher[1, 0] = rho
    fit = _fit(fisher=fisher)
    assert fit.standardized_design_rank == STATE_FEATURE_RANK
    assert fit.standardized_design_condition == pytest.approx(50.0)


def test_damping_uses_v4_lower_middle_tensor_median() -> None:
    fisher = torch.diag(torch.tensor([1.0, 2.0, 100.0, 200.0]))
    fit = _fit(fisher=fisher)
    assert fit.damping == pytest.approx(0.2)


def test_residual_conditional_fisher_uses_uncentered_second_moments() -> None:
    independent = _fit(
        fisher=torch.eye(STATE_FEATURE_RANK, dtype=torch.float64),
        cross=torch.zeros(STATE_FEATURE_RANK, dtype=torch.float64),
        static_energy=0.0,
    )
    assert independent.residual_conditional_fisher_fraction == 1.0
    vector = torch.ones(STATE_FEATURE_RANK, dtype=torch.float64)
    collinear = _fit(
        fisher=torch.outer(vector, vector),
        cross=vector,
        static_energy=1.0,
    )
    assert collinear.residual_conditional_fisher_fraction == pytest.approx(0.0)
    with pytest.raises(ValueError, match="materially non-PSD"):
        _fit(
            fisher=torch.eye(STATE_FEATURE_RANK, dtype=torch.float64),
            cross=2.0 * vector,
            static_energy=1.0,
        )


def test_nested_records_use_post_trust_held_derivatives_and_raw_slope_cosine() -> None:
    full = _fit(scale=torch.ones(STATE_FEATURE_RANK, dtype=torch.float64))
    inner_families = full.training_family_ids[:-1]
    inner = _fit(
        families=inner_families,
        codec_hash=_hash("inner-codec"),
        refit_hash=full.refit_artifact_sha256,
        scale=torch.ones(STATE_FEATURE_RANK, dtype=torch.float64),
    )
    held = _state_example(
        family=full.training_family_ids[-1],
        codec_hash=inner.codec_artifact_sha256,
    )
    record = build_candidate_conditioned_k64_inner_family_analytic_record(
        full,
        inner,
        _static_from_fit(inner),
        held,
    )
    expected = float(
        torch.dot(
            held.state_tangent_design_tensor().mean(dim=0),
            inner.applied_weight,
        )
    )
    assert record.state_predicted_incremental_derivative == expected
    assert record.scalar_predicted_incremental_derivative == 0.0
    assert record.state_increment_is_negative is True
    assert record.codec_invariant_weight_cosine == pytest.approx(1.0)


def test_nested_record_rejects_an_unrelated_inner_example_id() -> None:
    full = _fit()
    inner = _fit(
        families=full.training_family_ids[:-1],
        codec_hash=_hash("inner-codec"),
        refit_hash=full.refit_artifact_sha256,
    )
    unrelated_inner = replace(
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
    held = _state_example(
        family=full.training_family_ids[-1],
        codec_hash=unrelated_inner.codec_artifact_sha256,
    )
    with pytest.raises(ValueError, match="does not match"):
        build_candidate_conditioned_k64_inner_family_analytic_record(
            full,
            unrelated_inner,
            _static_from_fit(unrelated_inner),
            held,
        )


def test_full_eight_fold_capacity_screen_passes_and_is_analytic_only() -> None:
    screen = _screen()
    assert screen.capacity_screen_passed is True
    assert screen.negative_inner_derivative_count == 56
    assert screen.negative_inner_local_fold_count == 8
    assert screen.state_beats_scalar_fold_count == 8
    assert screen.cosine_stability_fold_count == 8
    assert screen.outcome == "capacity_supported_for_finite_validation"
    metadata = screen.metadata()
    assert metadata["analytic_forward_count"] == 112
    assert metadata["analytic_backward_count"] == 494
    assert metadata["extra_backward_count"] == 0
    assert metadata["finite_tune_or_held_execution_performed"] is False
    assert metadata["authorizes_final_selection"] is False


def test_global_and_fold_local_negative_derivative_gates_are_independent() -> None:
    forty_one_positive = {
        (outer, inner)
        for outer in range(8)
        for inner in range(7)
        if outer * 7 + inner >= 41
    }
    forty_one = _screen(positive_derivatives=forty_one_positive)
    assert forty_one.negative_inner_derivative_count == 41
    assert forty_one.negative_inner_global_gate_passed is False

    concentrated_failures = {
        (outer, inner)
        for outer in range(3)
        for inner in range(7)
    }
    concentrated = _screen(positive_derivatives=concentrated_failures)
    assert concentrated.negative_inner_derivative_count == 35
    assert concentrated.negative_inner_global_gate_passed is False
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
            "state_beats_scalar_fold_count",
            "state_beats_scalar_gate_passed",
        ),
        (
            "low_cosine_outers",
            "cosine_stability_fold_count",
            "cosine_stability_gate_passed",
        ),
    ),
)
def test_six_of_eight_aggregate_boundaries_are_inclusive(
    keyword: str,
    count_attribute: str,
    gate_attribute: str,
) -> None:
    five_pass = _screen(**{keyword: {0, 1, 2}})
    six_pass = _screen(**{keyword: {0, 1}})
    assert getattr(five_pass, count_attribute) == 5
    assert getattr(five_pass, gate_attribute) is False
    assert getattr(six_pass, count_attribute) == 6
    assert getattr(six_pass, gate_attribute) is True


def test_metadata_contains_hashes_and_scalars_but_no_tensor_objects() -> None:
    screen = _screen()
    record = screen.fold_records[0].inner_family_records[0]
    payload = {
        "field": record.inner_field_fit.metadata(),
        "inner": record.metadata(),
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
    serialized = json.dumps(payload, sort_keys=True)
    assert '"raw_tensors_serialized": true' not in serialized


def test_mutating_a_hidden_tensor_breaks_nested_integrity() -> None:
    screen = _screen()
    record = screen.fold_records[0].inner_family_records[0]
    record.inner_field_fit.applied_weight[0] += 1.0
    with pytest.raises(RuntimeError, match="payload drifted"):
        screen.validate_integrity()
