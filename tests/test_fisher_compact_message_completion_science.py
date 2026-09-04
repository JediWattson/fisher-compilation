from __future__ import annotations

import math

import pytest
import torch

from fisher_graph.fisher_compact_message_completion_science import (
    ChartHeadCompletionBank,
    CompletionBankTrainingRows,
    CompletionTrainingConfig,
    CompletionTrainingRows,
    CompletionTrainingResult,
    FidelityMetrics,
    FidelityPromptObservation,
    J0Model,
    J1Model,
    J2Model,
    J2NModel,
    J2PlusModel,
    J3Model,
    J4Model,
    KMeansSupportError,
    O2CodecEvaluation,
    OperatorObservation,
    OperatorPairSummary,
    QContributionCodec,
    ScientificContractError,
    WeightedKMeansConfig,
    VFlowCodec,
    aggregate_operator_pair_summaries,
    bank_coefficient_loss,
    bootstrap_paired_comparison,
    canonicalize_tied_eigenvectors,
    coefficient_loss,
    combined_qv_second_moment,
    completion_response_operator,
    copy_j4_common_parameters_to_j2n,
    evaluate_completion_bank_loss,
    evaluate_completion_loss,
    evaluate_five_fidelity_gates,
    fit_fisher_codec,
    fit_q_contribution_codec,
    fit_v_flow_codec,
    fit_weighted_kmeans,
    frozen_completion_training_config,
    hierarchical_row_weights,
    iter_stratified_bootstrap_indices,
    j2plus_parameter_count,
    j3_parameter_count,
    j4_parameter_count,
    literal_matched_widths,
    make_frozen_completion_bank,
    make_paired_j2n_j4_banks,
    nearest_rank,
    percentile_interval,
    positive_operator_reference,
    primary_downstream_metric,
    relative_improvement,
    select_o2q_codec,
    select_restart,
    stable_rank,
    stabilize_support_covariance,
    summarize_fidelity,
    summarize_operator_pairs,
    tensor_weighted_median,
    trace_normalized_second_moment,
    train_completion_model,
    train_completion_bank,
    validate_frozen_completion_training_config,
    weighted_linear_quantile,
    weighted_mean_and_covariance,
    weighted_median,
    weighted_relative_l2,
    weighted_uncentered_second_moment,
    HierarchicalBatchSampler,
    HierarchicalRowKey,
)


def _raw(value: int) -> bytes:
    return bytes([value]) * 32


def _key(unit: int, sequence: int, position: int) -> HierarchicalRowKey:
    return HierarchicalRowKey(_raw(unit), _raw(sequence), position)


def _passing_metrics(*, delta: float = 0.01, kl: float = 0.01) -> FidelityMetrics:
    return FidelityMetrics(
        delta_nll_per_token=delta,
        source_to_candidate_kl_per_token=kl,
        top1_agreement=0.98,
        prompt_p90_absolute_delta_nll_per_token=0.04,
        prompt_p10_top1_agreement=0.95,
        prompt_count=4,
        supervised_token_count=20,
    )


def _failing_metrics() -> FidelityMetrics:
    return FidelityMetrics(
        delta_nll_per_token=0.051,
        source_to_candidate_kl_per_token=0.01,
        top1_agreement=0.98,
        prompt_p90_absolute_delta_nll_per_token=0.04,
        prompt_p10_top1_agreement=0.95,
        prompt_count=4,
        supervised_token_count=20,
    )


def test_hierarchical_row_weights_equalize_units_sequences_and_rows() -> None:
    keys = (
        _key(1, 11, 0),
        _key(1, 11, 1),
        _key(1, 12, 0),
        _key(2, 21, 0),
    )
    weights = hierarchical_row_weights(keys)
    torch.testing.assert_close(
        weights,
        torch.tensor([0.125, 0.125, 0.25, 0.5], dtype=torch.float64),
    )
    assert float(weights[:3].sum()) == pytest.approx(0.5)
    assert float(weights[3:].sum()) == pytest.approx(0.5)


def test_tied_eigenspace_canonicalization_is_rotation_invariant() -> None:
    values = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64)
    angle = 0.37
    rotated = identity.clone()
    rotated[:, :2] = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    first = canonicalize_tied_eigenvectors(values, identity)
    second = canonicalize_tied_eigenvectors(values, rotated)
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-12)


def test_weighted_linear_quantile_and_median_follow_cumulative_interpolation() -> None:
    values = torch.tensor([10.0, 0.0], dtype=torch.float64)
    weights = torch.tensor([0.75, 0.25], dtype=torch.float64)
    assert weighted_linear_quantile(values, weights, 0.0) == 0.0
    assert weighted_median(values, weights) == pytest.approx(10.0 / 3.0)
    assert weighted_linear_quantile(values, weights, 1.0) == 10.0
    with pytest.raises(ScientificContractError):
        weighted_median(values, torch.tensor([1.0, 0.0]))


def test_tensor_weighted_median_splits_each_row_mass_over_components() -> None:
    keys = (_key(1, 1, 0), _key(2, 2, 0))
    values = torch.tensor([[-4.0, 2.0], [8.0, 10.0]])
    result = tensor_weighted_median(
        values,
        torch.tensor([0.75, 0.25]),
        keys,
        absolute=True,
    )
    assert result == pytest.approx(8.0 / 3.0)


def test_weighted_relative_l2_does_not_expand_high_row_count_units() -> None:
    reference = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float64)
    reconstruction = torch.tensor([[2.0], [2.0], [1.0]], dtype=torch.float64)
    weights = hierarchical_row_weights(
        (_key(1, 1, 0), _key(1, 1, 1), _key(2, 2, 0))
    )
    assert weighted_relative_l2(reference, reconstruction, weights) == pytest.approx(
        math.sqrt(0.5)
    )


def test_weighted_moments_are_population_centered_or_explicitly_uncentered() -> None:
    rows = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    mean, covariance = weighted_mean_and_covariance(rows, weights)
    torch.testing.assert_close(mean, torch.tensor([1.5], dtype=torch.float64))
    torch.testing.assert_close(covariance, torch.tensor([[0.75]], dtype=torch.float64))
    torch.testing.assert_close(
        weighted_uncentered_second_moment(rows, weights),
        torch.tensor([[3.0]], dtype=torch.float64),
    )


def test_combined_qv_moment_is_half_total_half_mean_component() -> None:
    total = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    components = torch.stack((total, total.flip(1)), dim=1)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    observed = combined_qv_second_moment(total, components, weights)
    total_part = trace_normalized_second_moment(
        weighted_uncentered_second_moment(total, weights)
    )
    component_part = torch.stack(
        [
            trace_normalized_second_moment(
                weighted_uncentered_second_moment(components[:, index], weights)
            )
            for index in range(2)
        ]
    ).mean(0)
    torch.testing.assert_close(observed, 0.5 * total_part + 0.5 * component_part)


def test_weighted_kmeans_is_deterministic_under_input_permutation() -> None:
    rows = torch.tensor(
        [[-2.0, 0.0], [-1.8, 0.1], [2.0, 0.0], [1.8, -0.1]],
        dtype=torch.float64,
    )
    keys = tuple(_key(index + 1, index + 11, 0) for index in range(4))
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    config = WeightedKMeansConfig(cluster_count=2)
    first = fit_weighted_kmeans(rows, keys, weights, config)
    permutation = torch.tensor([2, 0, 3, 1])
    second = fit_weighted_kmeans(
        rows[permutation],
        tuple(keys[index] for index in permutation.tolist()),
        weights[permutation],
        config,
    )
    torch.testing.assert_close(first.centers, second.centers)
    assert first.initialization_row_keys == second.initialization_row_keys
    first_by_key = {key: int(first.assignments[index]) for index, key in enumerate(keys)}
    second_by_key = {
        keys[int(permutation[index])]: int(second.assignments[index])
        for index in range(4)
    }
    assert first_by_key == second_by_key


def test_weighted_kmeans_repairs_empty_clusters_without_reusing_a_donor() -> None:
    rows = torch.zeros(3, 2, dtype=torch.float64)
    keys = tuple(_key(index + 1, index + 11, 0) for index in range(3))
    result = fit_weighted_kmeans(
        rows,
        keys,
        torch.ones(3),
        WeightedKMeansConfig(cluster_count=3),
    )
    assert torch.bincount(result.assignments, minlength=3).tolist() == [1, 1, 1]
    assert [repair.empty_cluster_id for repair in result.repairs] == [1, 2]
    assert len({repair.donor_row_key for repair in result.repairs}) == 2
    with pytest.raises(KMeansSupportError):
        fit_weighted_kmeans(
            rows,
            keys,
            torch.ones(3),
            WeightedKMeansConfig(cluster_count=4),
        )


def test_support_covariance_applies_rcond_before_flooring() -> None:
    covariance = torch.diag(torch.tensor([4.0, 1.0, 1e-12], dtype=torch.float64))
    result = stabilize_support_covariance(covariance)
    assert result.active.tolist() == [False, True, True]
    torch.testing.assert_close(
        result.inverse_eigenvalues,
        torch.tensor([0.0, 1.0, 0.25], dtype=torch.float64),
    )
    assert result.support_floor == pytest.approx(2.5e-6)


def test_support_covariance_clips_only_tolerated_negative_eigenvalues() -> None:
    clipped = stabilize_support_covariance(
        torch.diag(torch.tensor([-1e-12, 1.0], dtype=torch.float64))
    )
    assert clipped.eigenvalues[0] == 0.0
    with pytest.raises(ScientificContractError):
        stabilize_support_covariance(
            torch.diag(torch.tensor([-1e-5, 1.0], dtype=torch.float64))
        )


def test_fisher_codec_obeys_dual_geometry_and_projector_invariants() -> None:
    codec = fit_fisher_codec(
        torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64)),
        torch.eye(2, dtype=torch.float64),
        torch.tensor([2.0, 1.0], dtype=torch.float64),
        rank=1,
        regularization_ratio=1e-3,
    )
    assert codec.rank == 1
    assert codec.invariants.passed
    assert codec.invariants.err_orth <= 1e-6
    assert codec.invariants.err_idem <= 1e-6
    assert codec.invariants.err_encoder <= 1e-10
    torch.testing.assert_close(
        codec.decoder.T @ codec.supported_fisher @ codec.decoder,
        torch.eye(1, dtype=torch.float64),
    )
    torch.testing.assert_close(codec.projector @ codec.projector, codec.projector)
    value = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(codec.project(value), codec.decode(codec.encode(value)))


def test_q_contribution_and_v_flow_factories_preserve_math_but_not_role() -> None:
    arguments = (
        torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64)),
        torch.eye(2, dtype=torch.float64),
        torch.tensor([2.0, 1.0], dtype=torch.float64),
    )
    q_codec = fit_q_contribution_codec(
        *arguments,
        rank=1,
        regularization_ratio=1e-3,
    )
    v_codec = fit_v_flow_codec(
        *arguments,
        rank=1,
        regularization_ratio=1e-3,
    )
    assert isinstance(q_codec, QContributionCodec)
    assert q_codec.codec_role == "q_contribution"
    assert isinstance(v_codec, VFlowCodec)
    assert v_codec.codec_role == "v_flow"
    torch.testing.assert_close(q_codec.encoder, v_codec.encoder, rtol=0.0, atol=0.0)
    torch.testing.assert_close(q_codec.decoder, v_codec.decoder, rtol=0.0, atol=0.0)


def test_fisher_codec_canonicalizes_a_rotated_tied_eigenspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_eigh = torch.linalg.eigh

    def fit_with_rotation(angle: float):
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = torch.tensor(
            [[cosine, -sine], [sine, cosine]],
            dtype=torch.float64,
        )

        def rotated_eigh(matrix: torch.Tensor):
            values, vectors = original_eigh(matrix)
            return values, vectors @ rotation

        monkeypatch.setattr(torch.linalg, "eigh", rotated_eigh)
        return fit_fisher_codec(
            torch.eye(2, dtype=torch.float64),
            torch.eye(2, dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            rank=2,
            regularization_ratio=1e-3,
        )

    first = fit_with_rotation(0.37)
    second = fit_with_rotation(1.11)
    torch.testing.assert_close(first.decoder, second.decoder, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(first.encoder, second.encoder, atol=1e-12, rtol=0.0)


def test_fidelity_summary_uses_literal_nearest_rank_tails() -> None:
    rows = tuple(
        FidelityPromptObservation(
            prompt_id=str(index),
            supervised_tokens=10,
            source_summed_nll=10.0,
            candidate_summed_nll=10.0 + index,
            source_to_candidate_summed_kl=0.1,
            top1_matches=index,
        )
        for index in range(1, 11)
    )
    result = summarize_fidelity(rows)
    assert result.prompt_p90_absolute_delta_nll_per_token == pytest.approx(0.9)
    assert result.prompt_p10_top1_agreement == pytest.approx(0.1)
    assert nearest_rank(tuple(range(1, 11)), 0.1) == 1.0


def test_five_fidelity_gates_use_inclusive_literal_boundaries() -> None:
    metrics = FidelityMetrics(
        delta_nll_per_token=-0.05,
        source_to_candidate_kl_per_token=0.05,
        top1_agreement=0.95,
        prompt_p90_absolute_delta_nll_per_token=0.10,
        prompt_p10_top1_agreement=0.90,
        prompt_count=1,
        supervised_token_count=1,
    )
    assert evaluate_five_fidelity_gates(metrics).passed


def test_o2q_selection_uses_both_panels_and_frozen_tie_hierarchy() -> None:
    first = O2CodecEvaluation(
        "c0",
        64,
        64,
        1e-3,
        _passing_metrics(),
        _passing_metrics(delta=0.02),
        _passing_metrics(),
        _passing_metrics(),
    )
    second = O2CodecEvaluation(
        "c1",
        32,
        32,
        1e-2,
        _passing_metrics(),
        _passing_metrics(delta=-0.01),
        _passing_metrics(),
        _passing_metrics(),
    )
    decision = select_o2q_codec((first, second))
    assert decision.selected_codec_id == "c1"
    assert decision.ranked_passing_codec_ids == ("c1", "c0")
    assert decision.may_fit_j_models


@pytest.mark.parametrize(
    ("v_passes", "reason"),
    [
        (True, "stalled_q_geometry_mismatch"),
        (False, "stalled_attention_chart_capacity"),
    ],
)
def test_o2q_early_stop_distinguishes_q_geometry_from_capacity(
    v_passes: bool,
    reason: str,
) -> None:
    evaluation = O2CodecEvaluation(
        "c0",
        32,
        32,
        1e-3,
        _failing_metrics(),
        _passing_metrics(),
        _passing_metrics() if v_passes else _failing_metrics(),
        _passing_metrics(),
    )
    decision = select_o2q_codec((evaluation,))
    assert decision.terminal_state == reason
    assert decision.reason_code == reason
    assert not decision.may_fit_j_models


def test_primary_downstream_metric_sums_tokens_then_equalizes_units() -> None:
    keys = (
        _key(1, 11, 0),
        _key(1, 11, 1),
        _key(1, 12, 0),
        _key(2, 21, 0),
    )
    errors = torch.tensor([[1.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    displacement = torch.ones_like(errors)
    result = primary_downstream_metric(
        errors,
        displacement,
        torch.eye(1, dtype=torch.float64),
        keys,
        epsilon_down=1.0,
    )
    # Unit 1: mean(sequence losses 2/3 and 4/2) = 4/3; unit 2: 9/2.
    assert result.mean_loss == pytest.approx((4.0 / 3.0 + 4.5) / 2.0)


def test_rho_handles_regular_and_both_near_zero_cases_without_nan() -> None:
    regular = relative_improvement(0.9, 1.0)
    assert regular.rho == pytest.approx(0.1)
    assert regular.relative_screen_passed
    both = relative_improvement(1e-13, 1e-13)
    assert both.rho is None
    assert both.rho_status == "both_at_numerical_floor"
    worse = relative_improvement(1e-3, 1e-13)
    assert worse.rho is None
    assert worse.rho_status == "candidate_worse_than_near_zero_baseline"


def test_stratified_bootstrap_is_deterministic_and_preserves_stratum_sizes() -> None:
    strata = ("a", "a", "b", "b")
    first = list(iter_stratified_bootstrap_indices(strata, replicates=4, seed=9))
    second = list(iter_stratified_bootstrap_indices(strata, replicates=4, seed=9))
    assert first == second
    for replicate in first:
        assert len(replicate) == 4
        assert sum(index < 2 for index in replicate) == 2
        assert sum(index >= 2 for index in replicate) == 2
    assert percentile_interval([0.0, 1.0], confidence=0.5) == pytest.approx(
        (0.25, 0.75)
    )


def test_bootstrap_relative_interval_requires_95_percent_valid_replicates() -> None:
    invalid = bootstrap_paired_comparison(
        [0.0, 0.0],
        [0.0, 0.0],
        ["all", "all"],
        replicates=20,
        seed=1,
    )
    assert invalid.rho_interval is None
    assert invalid.valid_rho_replicate_fraction == 0.0
    valid = bootstrap_paired_comparison(
        [0.8, 0.9],
        [1.0, 1.0],
        ["all", "all"],
        replicates=20,
        seed=1,
    )
    assert valid.rho_interval is not None
    assert valid.valid_rho_replicate_fraction == 1.0


def test_two_stage_operator_summary_excludes_under_supported_pairs() -> None:
    observations = (
        OperatorObservation(_key(1, 11, 0), 0, 0, 0.0, 1.0, 1.0),
        OperatorObservation(_key(2, 21, 0), 0, 0, 0.0, 3.0, 2.0),
        OperatorObservation(_key(1, 12, 0), 0, 1, 2.0, 2.0, 1.5),
        OperatorObservation(_key(2, 22, 0), 0, 1, 4.0, 4.0, 2.5),
        OperatorObservation(_key(3, 31, 0), 1, 0, 7.0, 7.0, 3.0),
    )
    pairs = summarize_operator_pairs(
        observations,
        expected_pairs=((0, 0), (0, 1), (1, 0)),
    )
    assert pairs[2].status == "insufficient_pair_support"
    aggregate = aggregate_operator_pair_summaries(pairs)
    assert aggregate.supported_chart_head_pair_count == 2
    assert aggregate.excluded_chart_head_pair_count == 1
    assert aggregate.zero_ambient_pair_count == 1


def test_positive_operator_reference_filters_zero_but_does_not_relabel_support() -> None:
    pairs = (
        OperatorPairSummary(0, 0, 2, "supported", 0.0, 1.0, 1.0),
        OperatorPairSummary(0, 1, 2, "supported", 2.0, 1.0, 1.0),
        OperatorPairSummary(1, 0, 2, "supported", 4.0, 1.0, 1.0),
    )
    result = positive_operator_reference(
        pairs,
        maximum_frozen_o1_relative_tolerance=1e-5,
    )
    assert result.status == "valid"
    assert result.operator_reference_norm == pytest.approx(2.0)
    assert result.zero_pair_count == 1
    invalid = positive_operator_reference(
        pairs[:1],
        maximum_frozen_o1_relative_tolerance=1e-5,
    )
    assert invalid.status == "invalid_no_positive_operator_reference"
    assert invalid.operator_reference_norm is None
    assert invalid.tau_r is None


def test_stable_rank_and_completion_response_operator() -> None:
    operator = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    assert stable_rank(operator, tolerance=0.0) == pytest.approx(1.25)
    result = completion_response_operator(
        torch.eye(2),
        torch.eye(2),
        torch.ones(2, 1),
        torch.tensor([3.0]),
        torch.ones(1, 2),
    )
    torch.testing.assert_close(
        result,
        torch.tensor([[5.0, 3.0], [3.0, 5.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("coefficient_dimension", "j4_count", "j2_widths", "j2_count", "j3_count"),
    [
        (32, 2848, (21, 22), 2859, 2845),
        (64, 5184, (26, 26), 5172, 5224),
    ],
)
def test_literal_parameter_contract(
    coefficient_dimension: int,
    j4_count: int,
    j2_widths: tuple[int, int],
    j2_count: int,
    j3_count: int,
) -> None:
    z_width, r_width, j3_width = literal_matched_widths(coefficient_dimension)
    assert (z_width, r_width) == j2_widths
    assert j4_parameter_count(coefficient_dimension) == j4_count
    assert j2plus_parameter_count(
        coefficient_dimension,
        z_width,
        r_width,
    ) == j2_count
    assert j3_parameter_count(coefficient_dimension, j3_width) == j3_count


def test_j_models_have_frozen_equations_and_shapes() -> None:
    c_base = torch.tensor([[5.0]])
    z = torch.tensor([[2.0, 0.0]])
    r = torch.tensor([[3.0, 0.0]])
    j4 = J4Model(2, 2, 1, interaction_rank=1)
    with torch.no_grad():
        j4.output_bias.fill_(1.0)
        j4.activation_weight.copy_(torch.tensor([[2.0, 0.0]]))
        j4.message_weight.copy_(torch.tensor([[3.0, 0.0]]))
        j4.interaction_output.fill_(4.0)
        j4.activation_interaction.copy_(torch.tensor([[1.0, 0.0]]))
        j4.message_interaction.copy_(torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(j4(c_base, z, r), torch.tensor([[43.0]]))
    torch.testing.assert_close(
        j4.forward_interaction_mean(c_base, z, r, torch.tensor([1.0])),
        torch.tensor([[31.0]]),
    )
    j2n = J2NModel(2, 2, 1, interaction_rank=1)
    copy_j4_common_parameters_to_j2n(j4, j2n)
    torch.testing.assert_close(j2n(c_base, z, r), torch.tensor([[19.0]]))
    torch.testing.assert_close(J1Model(2, 2, 1)(c_base, z, r), c_base)
    assert J0Model(2, 1)(c_base, z, torch.zeros(1, 32)).shape == (1, 1)
    assert J2Model(2, 2, 1)(c_base, z, r).shape == (1, 1)
    assert J2PlusModel(2, 2, 1, z_width=2, r_width=3)(c_base, z, r).shape == (
        1,
        1,
    )
    assert J3Model(2, 2, 1, width=3)(c_base, z, r).shape == (1, 1)


def test_frozen_model_parameter_counts_match_actual_torch_modules() -> None:
    for coefficient_dimension in (32, 64):
        z_width, r_width, j3_width = literal_matched_widths(coefficient_dimension)
        j4 = J4Model(32, 32, coefficient_dimension)
        j2plus = J2PlusModel(
            32,
            32,
            coefficient_dimension,
            z_width=z_width,
            r_width=r_width,
        )
        j3 = J3Model(32, 32, coefficient_dimension, width=j3_width)
        assert sum(parameter.numel() for parameter in j4.parameters()) == j4_parameter_count(
            coefficient_dimension
        )
        assert sum(
            parameter.numel() for parameter in j2plus.parameters()
        ) == j2plus_parameter_count(coefficient_dimension, z_width, r_width)
        assert sum(parameter.numel() for parameter in j3.parameters()) == j3_parameter_count(
            coefficient_dimension,
            j3_width,
        )


def test_j4_initialization_is_nested_and_uses_zero_output_bias_and_u() -> None:
    torch.manual_seed(123)
    j4 = J4Model(32, 32, 32)
    assert torch.count_nonzero(j4.output_bias) == 0
    assert torch.count_nonzero(j4.interaction_output) == 0
    assert torch.count_nonzero(j4.activation_weight) > 0
    assert torch.count_nonzero(j4.message_weight) > 0
    assert torch.count_nonzero(j4.activation_interaction) > 0
    assert torch.count_nonzero(j4.message_interaction) > 0


def test_all_trainable_j_families_use_xavier_weights_and_zero_biases() -> None:
    models = (
        J0Model(32, 32),
        J2Model(32, 32, 32),
        J2NModel(32, 32, 32),
        J2PlusModel(32, 32, 32, z_width=16, r_width=16),
        J3Model(32, 32, 32, width=16),
        J4Model(32, 32, 32),
    )
    for model in models:
        for name, parameter in model.named_parameters():
            if name.endswith("bias") or name == "output_bias":
                assert torch.count_nonzero(parameter) == 0
            elif name == "interaction_output":
                assert torch.count_nonzero(parameter) == 0
            else:
                assert torch.count_nonzero(parameter) > 0


def test_frozen_training_config_distinguishes_fit_and_final_step_refit() -> None:
    initial = frozen_completion_training_config("initial")
    refit = frozen_completion_training_config("refit")
    validate_frozen_completion_training_config(initial, "initial")
    validate_frozen_completion_training_config(refit, "refit")
    assert initial.maximum_steps == 2_000
    assert initial.early_stopping
    assert refit.maximum_steps == refit.minimum_steps == 1_000
    assert not refit.early_stopping
    with pytest.raises(ScientificContractError):
        validate_frozen_completion_training_config(
            CompletionTrainingConfig(maximum_steps=999, minimum_steps=999),
            "refit",
        )


def test_restart_tolerance_class_is_anchored_to_the_global_minimum() -> None:
    def result(seed: int, loss: float) -> CompletionTrainingResult:
        return CompletionTrainingResult(
            seed=seed,
            best_step=1,
            best_monitor_loss=loss,
            final_step=1,
            stopped_early=False,
            monitor_history=((1, loss),),
            state_dict={},
        )

    # Pairwise folding incorrectly walks from 1701 -> 2718 -> 3141.  The
    # global-minimum tolerance class contains 2718 and 3141, so seed 2718 wins.
    selected = select_restart(
        (result(1701, 1.8e-6), result(2718, 0.9e-6), result(3141, 0.0))
    )
    assert selected.seed == 2718


def test_chart_head_bank_routes_by_chart_and_preserves_all_heads() -> None:
    cells = [J2NModel(1, 1, 1) for _ in range(4)]
    for index, cell in enumerate(cells, start=1):
        with torch.no_grad():
            cell.output_bias.fill_(float(index))
            cell.activation_weight.zero_()
            cell.message_weight.zero_()
    bank = ChartHeadCompletionBank((2, 5), 2, cells)
    observed = bank(
        torch.zeros(3, 2, 1),
        torch.zeros(3, 1),
        torch.zeros(3, 2, 1),
        torch.tensor([2, 5, 2]),
    )
    torch.testing.assert_close(
        observed.squeeze(-1),
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]),
    )
    with pytest.raises(ScientificContractError):
        bank(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 2, 1),
            torch.tensor([4]),
        )


def test_frozen_bank_factories_have_exact_grid_parameter_counts() -> None:
    chart_ids = (0, 3)
    head_count = 4
    cell_count = len(chart_ids) * head_count
    for coefficient_dimension in (32, 64):
        z_width, r_width, j3_width = literal_matched_widths(coefficient_dimension)
        j4 = make_frozen_completion_bank(
            "J4",
            chart_ids,
            head_count,
            coefficient_dimension,
            initialization_seed=7,
        )
        j2plus = make_frozen_completion_bank(
            "J2+",
            chart_ids,
            head_count,
            coefficient_dimension,
            initialization_seed=7,
        )
        j3 = make_frozen_completion_bank(
            "J3",
            chart_ids,
            head_count,
            coefficient_dimension,
            initialization_seed=7,
        )
        assert j4.trainable_parameter_count == cell_count * j4_parameter_count(
            coefficient_dimension
        )
        assert j2plus.trainable_parameter_count == cell_count * j2plus_parameter_count(
            coefficient_dimension,
            z_width,
            r_width,
        )
        assert j3.trainable_parameter_count == cell_count * j3_parameter_count(
            coefficient_dimension,
            j3_width,
        )


def test_paired_j2n_j4_banks_have_byte_identical_common_initialization() -> None:
    j2n, j4 = make_paired_j2n_j4_banks(
        (0, 1),
        4,
        32,
        initialization_seed=1701,
    )
    for chart_id in j4.chart_ids:
        for head_id in range(j4.head_count):
            nested = j2n.cell(chart_id, head_id)
            interaction = j4.cell(chart_id, head_id)
            assert isinstance(nested, J2NModel)
            assert isinstance(interaction, J4Model)
            assert torch.equal(nested.output_bias, interaction.output_bias)
            assert torch.equal(
                nested.activation_weight,
                interaction.activation_weight,
            )
            assert torch.equal(nested.message_weight, interaction.message_weight)
            assert torch.count_nonzero(interaction.interaction_output) == 0
    c_base = torch.randn(3, 4, 32)
    z = torch.randn(3, 32)
    r = torch.randn(3, 4, 32)
    chart_ids = torch.tensor([0, 1, 0])
    assert torch.equal(
        j2n(c_base, z, r, chart_ids),
        j4(c_base, z, r, chart_ids),
    )


def test_bank_coefficient_loss_preserves_head_mean_before_row_weighting() -> None:
    prediction = torch.tensor([[[2.0], [0.0]], [[1.0], [1.0]]])
    target = torch.ones_like(prediction)
    loss = bank_coefficient_loss(
        prediction,
        target,
        torch.ones(2, 2),
        row_keys=(_key(1, 1, 0), _key(2, 2, 0)),
    )
    assert float(loss) == pytest.approx(0.25)


def test_coefficient_loss_uses_head_mean_and_hierarchical_rows() -> None:
    prediction = torch.tensor([[[2.0], [0.0]], [[1.0], [1.0]]])
    target = torch.ones_like(prediction)
    keys = (_key(1, 1, 0), _key(2, 2, 0))
    loss = coefficient_loss(
        prediction,
        target,
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        row_keys=keys,
    )
    assert float(loss) == pytest.approx(0.25)


def test_hierarchical_batch_sampler_is_reproducible() -> None:
    keys = (
        _key(1, 11, 0),
        _key(1, 11, 1),
        _key(1, 12, 0),
        _key(2, 21, 0),
    )
    first = HierarchicalBatchSampler(keys, seed=1701).sample(20)
    second = HierarchicalBatchSampler(keys, seed=1701).sample(20)
    assert torch.equal(first, second)


def test_completion_training_reduces_monitor_loss_on_synthetic_affine_rows() -> None:
    torch.manual_seed(7)
    z = torch.linspace(0.5, 4.0, 8).unsqueeze(1)
    target = 1.0 + 2.0 * z
    rows = CompletionTrainingRows(
        c_base=torch.zeros(8, 1),
        z=z,
        r=torch.zeros(8, 32),
        target=target,
        epsilon=torch.full((8,), 1e-3),
        row_keys=tuple(_key(index + 1, index + 21, 0) for index in range(8)),
    )
    model = J0Model(1, 1)
    initial = evaluate_completion_loss(model, rows, target_kind="full_coefficients")
    result = train_completion_model(
        model,
        rows,
        rows,
        seed=1701,
        config=CompletionTrainingConfig(
            learning_rate=0.05,
            warmup_steps=1,
            batch_size=32,
            maximum_steps=100,
            minimum_steps=10,
            evaluation_interval=10,
            early_stopping_patience=100,
            target_kind="full_coefficients",
        ),
    )
    assert result.best_monitor_loss < initial
    assert evaluate_completion_loss(
        model,
        rows,
        target_kind="full_coefficients",
    ) == pytest.approx(result.best_monitor_loss)


def test_one_bank_optimizer_trains_all_heads_without_row_expansion() -> None:
    row_count = 6
    bank = make_frozen_completion_bank(
        "J2N",
        (0,),
        2,
        32,
        initialization_seed=11,
    )
    rows = CompletionBankTrainingRows(
        c_base=torch.zeros(row_count, 2, 32),
        z=torch.zeros(row_count, 32),
        r=torch.zeros(row_count, 2, 32),
        chart_ids=torch.zeros(row_count, dtype=torch.int64),
        target=torch.ones(row_count, 2, 32),
        epsilon=torch.ones(row_count, 2),
        row_keys=tuple(_key(index + 1, index + 31, 0) for index in range(row_count)),
    )
    initial = evaluate_completion_bank_loss(
        bank,
        rows,
        target_kind="completion_delta",
    )
    result = train_completion_bank(
        bank,
        rows,
        rows,
        seed=1701,
        config=CompletionTrainingConfig(
            learning_rate=0.05,
            warmup_steps=1,
            batch_size=12,
            maximum_steps=40,
            minimum_steps=5,
            evaluation_interval=5,
            early_stopping_patience=40,
            target_kind="completion_delta",
        ),
    )
    assert result.best_monitor_loss < initial
    assert evaluate_completion_bank_loss(
        bank,
        rows,
        target_kind="completion_delta",
    ) == pytest.approx(result.best_monitor_loss)


def test_bank_refit_without_early_stopping_returns_final_step_not_best() -> None:
    def rows(target_value: float) -> CompletionBankTrainingRows:
        return CompletionBankTrainingRows(
            c_base=torch.zeros(1, 1, 1),
            z=torch.zeros(1, 1),
            r=torch.zeros(1, 1, 1),
            chart_ids=torch.zeros(1, dtype=torch.int64),
            target=torch.full((1, 1, 1), target_value),
            epsilon=torch.ones(1, 1),
            row_keys=(_key(1, 1, 0),),
        )

    bank = ChartHeadCompletionBank((0,), 1, (J2NModel(1, 1, 1),))
    result = train_completion_bank(
        bank,
        rows(1.0),
        rows(0.1),
        seed=1701,
        config=CompletionTrainingConfig(
            learning_rate=0.1,
            weight_decay=0.0,
            warmup_steps=1,
            batch_size=1,
            maximum_steps=3,
            minimum_steps=3,
            evaluation_interval=1,
            early_stopping_patience=3,
            early_stopping=False,
        ),
    )
    cell = bank.cell(0, 0)
    assert isinstance(cell, J2NModel)
    assert result.best_step == 1
    assert result.final_step == 3
    assert float(cell.output_bias.detach().item()) > 0.2
    assert torch.equal(result.state_dict["cells.0.output_bias"], cell.output_bias)
