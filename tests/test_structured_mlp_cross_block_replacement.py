from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.structured_mlp_cross_block_bundling import ModeKey
from fisher_graph.structured_mlp_cross_block_replacement import (
    REPLACEMENT_CONDITIONS,
    CrossBlockReplacementConditionMetric,
    CrossBlockReplacementFitRows,
    CrossBlockReplacementOracleProvenance,
    CrossBlockReplacementOracleResult,
    CrossBlockReplacementProvenance,
    CrossBlockScalarReplacementEvidence,
    aggregate_cross_block_replacement_conditions,
    fit_cross_block_scalar_replacement,
)


def _sha(character: str) -> str:
    return character * 64


def _endpoints() -> tuple[ModeKey, ModeKey]:
    return (
        ModeKey(
            layer_ordinal=2,
            layer_id="layer.2",
            activation_site="layer.2.feed_forward_down_input",
            mode_index=7,
            fisher_rank=3,
        ),
        ModeKey(
            layer_ordinal=9,
            layer_id="layer.9",
            activation_site="layer.9.feed_forward_down_input",
            mode_index=11,
            fisher_rank=5,
        ),
    )


def _fit_provenance() -> CrossBlockReplacementProvenance:
    return CrossBlockReplacementProvenance(
        model_fingerprint=_sha("1"),
        fit_split_sha256=_sha("2"),
        objective_sha256=_sha("3"),
        proposal_artifact_sha256=_sha("4"),
    )


def _fit_rows() -> tuple[CrossBlockReplacementFitRows, ...]:
    # The last observation in each family is deliberately noisy.  Its zero
    # score gradient makes the Fisher candidate recover the exact scale 2,
    # while the ordinary candidate remains detectably different.
    return (
        CrossBlockReplacementFitRows(
            example_id="a-0",
            family_id="family-a",
            logical_positions=torch.tensor([1, 2, 3]),
            anchor_values=torch.tensor(
                [1.0, 2.0, 10.0],
                dtype=torch.float64,
            ),
            consumer_values=torch.tensor(
                [2.0, 4.0, -30.0],
                dtype=torch.float64,
            ),
            consumer_score_gradients=torch.tensor(
                [1.0, 2.0, 0.0],
                dtype=torch.float64,
            ),
        ),
        CrossBlockReplacementFitRows(
            example_id="b-0",
            family_id="family-b",
            logical_positions=torch.tensor([1, 2, 3]),
            anchor_values=torch.tensor(
                [3.0, 4.0, 12.0],
                dtype=torch.float64,
            ),
            consumer_values=torch.tensor(
                [6.0, 8.0, 40.0],
                dtype=torch.float64,
            ),
            consumer_score_gradients=torch.tensor(
                [2.0, 1.0, 0.0],
                dtype=torch.float64,
            ),
        ),
        CrossBlockReplacementFitRows(
            example_id="c-0",
            family_id="family-c",
            logical_positions=torch.tensor([1, 2, 3]),
            anchor_values=torch.tensor(
                [5.0, 6.0, 14.0],
                dtype=torch.float64,
            ),
            consumer_values=torch.tensor(
                [10.0, 12.0, -50.0],
                dtype=torch.float64,
            ),
            consumer_score_gradients=torch.tensor(
                [1.0, 3.0, 0.0],
                dtype=torch.float64,
            ),
        ),
        CrossBlockReplacementFitRows(
            example_id="d-0",
            family_id="family-d",
            logical_positions=torch.tensor([1, 2, 3]),
            anchor_values=torch.tensor(
                [7.0, 8.0, 16.0],
                dtype=torch.float64,
            ),
            consumer_values=torch.tensor(
                [14.0, 16.0, 60.0],
                dtype=torch.float64,
            ),
            consumer_score_gradients=torch.tensor(
                [4.0, 1.0, 0.0],
                dtype=torch.float64,
            ),
        ),
    )


def _fit_evidence() -> CrossBlockScalarReplacementEvidence:
    anchor, consumer = _endpoints()
    return fit_cross_block_scalar_replacement(
        _fit_rows(),
        provenance=_fit_provenance(),
        anchor=anchor,
        consumer=consumer,
        family_fold_assignment={
            "family-a": 0,
            "family-b": 1,
            "family-c": 0,
            "family-d": 1,
        },
        fold_count=2,
    )


def test_fits_closed_form_unweighted_and_fisher_candidates() -> None:
    result = _fit_evidence()

    assert result.sequences == 4
    assert result.observations == 12
    assert result.families == 4
    assert result.fold_family_counts == (2, 2)
    assert result.fold_sequence_counts == (2, 2)
    assert result.unweighted.scale != pytest.approx(2.0)
    assert result.fisher_weighted.scale == pytest.approx(2.0)
    assert result.fisher_weighted.residual_square_sum == pytest.approx(0.0)
    assert result.fisher_weighted.residual_nrmse == pytest.approx(0.0)
    assert result.fisher_weighted.fold_sign_stable
    assert tuple(
        fit.scale for fit in result.fisher_weighted.leave_one_fold_out
    ) == pytest.approx((2.0, 2.0))
    assert tuple(
        fit.holdout_residual_nrmse
        for fit in result.fisher_weighted.leave_one_fold_out
    ) == pytest.approx((0.0, 0.0))
    assert result.proposed_scale_kind == (
        "consumer_fisher_weighted_no_intercept"
    )
    assert result.proposed_scale == pytest.approx(2.0)


def test_fit_artifact_is_fit_only_and_authorizes_nothing() -> None:
    result = _fit_evidence()

    assert not result.contains_corpus_rows
    assert result.fit_only
    assert result.model_parameter_updates == 0
    assert not result.authorizes_intervention
    assert not result.authorizes_compilation
    assert not result.authorizes_execution
    assert not result.authorizes_guard
    assert not result.authorizes_b
    assert result.metadata()["provenance"]["data_scope"] == "fit_only"
    assert (
        result.metadata()["provenance"]["model_parameter_updates"] == 0
    )


def test_fit_is_order_invariant_but_row_sensitive() -> None:
    rows = _fit_rows()
    anchor, consumer = _endpoints()
    arguments = {
        "provenance": _fit_provenance(),
        "anchor": anchor,
        "consumer": consumer,
        "family_fold_assignment": {
            "family-a": 0,
            "family-b": 1,
            "family-c": 0,
            "family-d": 1,
        },
        "fold_count": 2,
    }
    ordered = fit_cross_block_scalar_replacement(rows, **arguments)
    reversed_result = fit_cross_block_scalar_replacement(
        tuple(reversed(rows)),
        **arguments,
    )
    changed_rows = list(rows)
    changed_rows[0] = replace(
        rows[0],
        consumer_values=rows[0].consumer_values
        + torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
    )
    changed = fit_cross_block_scalar_replacement(
        changed_rows,
        **arguments,
    )

    assert reversed_result.artifact_sha256 == ordered.artifact_sha256
    assert changed.fit_row_stream_sha256 != ordered.fit_row_stream_sha256
    assert changed.artifact_sha256 != ordered.artifact_sha256


def test_fit_state_round_trip_and_hash_tamper_rejection() -> None:
    result = _fit_evidence()
    state = result.state_dict()

    restored = CrossBlockScalarReplacementEvidence.from_state_dict(state)

    assert restored.metadata() == result.metadata()
    tampered = dict(state)
    tampered["proposed_scale"] = result.proposed_scale + 0.25
    with pytest.raises(ValueError, match="proposed scale"):
        CrossBlockScalarReplacementEvidence.from_state_dict(tampered)
    tampered = dict(state)
    tampered["fit_row_stream_sha256"] = _sha("f")
    with pytest.raises(ValueError, match="artifact hash"):
        CrossBlockScalarReplacementEvidence.from_state_dict(tampered)


def test_fit_requires_exact_family_disjoint_fold_coverage() -> None:
    rows = _fit_rows()
    anchor, consumer = _endpoints()
    arguments = {
        "rows": rows,
        "provenance": _fit_provenance(),
        "anchor": anchor,
        "consumer": consumer,
        "fold_count": 2,
    }

    with pytest.raises(ValueError, match="exactly the fit families"):
        fit_cross_block_scalar_replacement(
            **arguments,
            family_fold_assignment={
                "family-a": 0,
                "family-b": 1,
                "family-c": 0,
            },
        )
    with pytest.raises(ValueError, match="every family fold"):
        fit_cross_block_scalar_replacement(
            **arguments,
            family_fold_assignment={
                "family-a": 0,
                "family-b": 0,
                "family-c": 0,
                "family-d": 0,
            },
        )


def test_fit_rejects_zero_fisher_support_and_backward_endpoint() -> None:
    rows = tuple(
        replace(
            row,
            consumer_score_gradients=torch.zeros_like(
                row.consumer_score_gradients
            ),
        )
        for row in _fit_rows()
    )
    anchor, consumer = _endpoints()
    arguments = {
        "rows": rows,
        "provenance": _fit_provenance(),
        "family_fold_assignment": {
            "family-a": 0,
            "family-b": 1,
            "family-c": 0,
            "family-d": 1,
        },
        "fold_count": 2,
    }

    with pytest.raises(ValueError, match="denominator must be positive"):
        fit_cross_block_scalar_replacement(
            **arguments,
            anchor=anchor,
            consumer=consumer,
        )
    with pytest.raises(ValueError, match="strictly forward"):
        fit_cross_block_scalar_replacement(
            **arguments,
            anchor=consumer,
            consumer=anchor,
        )


def test_fit_rows_clone_inputs_and_require_token_alignment() -> None:
    anchor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    row = CrossBlockReplacementFitRows(
        example_id="example",
        family_id="family",
        logical_positions=torch.tensor([3, 5]),
        anchor_values=anchor,
        consumer_values=2.0 * anchor,
        consumer_score_gradients=torch.ones(2, dtype=torch.float64),
    )
    anchor[0] = 100.0

    assert row.anchor_values.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(row, logical_positions=torch.tensor([3, 3]))
    with pytest.raises(ValueError, match="share a positive length"):
        replace(
            row,
            consumer_values=torch.ones(3, dtype=torch.float64),
        )


def _oracle_provenance() -> CrossBlockReplacementOracleProvenance:
    evidence = _fit_evidence()
    return CrossBlockReplacementOracleProvenance(
        model_fingerprint=evidence.provenance.model_fingerprint,
        evaluation_fit_split_sha256=_sha("9"),
        objective_sha256=evidence.provenance.objective_sha256,
        replacement_evidence_sha256=evidence.artifact_sha256,
        shuffle_plan_sha256=_sha("8"),
        shuffle_policy="family_derangement_same_logical_position",
    )


def _oracle_metrics() -> tuple[CrossBlockReplacementConditionMetric, ...]:
    # Each family has one example with two supervised tokens.  The replacement
    # recovers 75% of ablation KL and 75% of absolute NLL distortion, and it
    # beats the shuffled control.
    rows: list[CrossBlockReplacementConditionMetric] = []
    specifications = (
        ("native", 10.0, 0.0, 2),
        ("ablation", 14.0, 4.0, 0),
        ("replacement", 11.0, 1.0, 2),
        ("shuffled", 16.0, 6.0, 0),
    )
    for example_id, family_id in (
        ("a-0", "family-a"),
        ("b-0", "family-b"),
    ):
        for condition, nll, kl, matches in specifications:
            rows.append(
                CrossBlockReplacementConditionMetric(
                    example_id=example_id,
                    family_id=family_id,
                    condition=condition,
                    supervised_tokens=2,
                    summed_nll=nll,
                    teacher_kl_sum_to_native=kl,
                    top1_matches_to_native=matches,
                )
            )
    return tuple(rows)


def test_aggregates_paired_intervention_quartet_and_recovery() -> None:
    result = aggregate_cross_block_replacement_conditions(
        _oracle_metrics(),
        provenance=_oracle_provenance(),
    )
    conditions = {
        value.condition: value for value in result.conditions
    }

    assert tuple(conditions) == REPLACEMENT_CONDITIONS
    assert conditions["native"].nll_per_token == pytest.approx(5.0)
    assert conditions["ablation"].delta_nll_per_token_to_native == (
        pytest.approx(2.0)
    )
    assert conditions["replacement"].delta_nll_per_token_to_native == (
        pytest.approx(0.5)
    )
    assert conditions[
        "replacement"
    ].absolute_delta_nll_per_token_to_native == pytest.approx(0.5)
    assert conditions["replacement"].teacher_kl_per_token_to_native == (
        pytest.approx(0.5)
    )
    assert conditions["replacement"].top1_agreement_to_native == (
        pytest.approx(1.0)
    )
    assert result.replacement_kl_recovery_vs_ablation == pytest.approx(
        0.75
    )
    assert (
        result.replacement_absolute_nll_recovery_vs_ablation
        == pytest.approx(0.75)
    )
    assert result.replacement_kl_advantage_vs_shuffled == pytest.approx(
        2.5
    )
    assert result.replacement_top1_advantage_vs_shuffled == pytest.approx(
        1.0
    )
    assert tuple(
        value.family_id for value in result.family_aggregates
    ) == ("family-a", "family-b")


def test_oracle_is_order_invariant_authenticated_and_fail_closed() -> None:
    metrics = _oracle_metrics()
    result = aggregate_cross_block_replacement_conditions(
        metrics,
        provenance=_oracle_provenance(),
    )
    reversed_result = aggregate_cross_block_replacement_conditions(
        reversed(metrics),
        provenance=_oracle_provenance(),
    )

    assert result.artifact_sha256 == reversed_result.artifact_sha256
    assert not result.contains_corpus_rows
    assert result.fit_only
    assert result.model_parameter_updates == 0
    assert not result.authorizes_intervention
    assert not result.authorizes_compilation
    assert not result.authorizes_execution
    assert not result.authorizes_guard
    assert not result.authorizes_b
    assert (
        result.metadata()["provenance"]["evaluation_scope"]
        == "fit_only_native_intervention_oracle"
    )

    restored = CrossBlockReplacementOracleResult.from_state_dict(
        result.state_dict()
    )
    assert restored.metadata() == result.metadata()
    tampered = result.state_dict()
    tampered["metric_stream_sha256"] = _sha("e")
    with pytest.raises(ValueError, match="artifact hash"):
        CrossBlockReplacementOracleResult.from_state_dict(tampered)


def test_absolute_nll_recovery_does_not_cancel_opposite_example_deltas() -> None:
    metrics: list[CrossBlockReplacementConditionMetric] = []
    by_example = {
        "a": {
            "native": 10.0,
            "ablation": 12.0,
            "replacement": 11.0,
            "shuffled": 13.0,
        },
        "b": {
            "native": 10.0,
            "ablation": 8.0,
            "replacement": 9.0,
            "shuffled": 7.0,
        },
    }
    for example_id, values in by_example.items():
        for condition in REPLACEMENT_CONDITIONS:
            metrics.append(
                CrossBlockReplacementConditionMetric(
                    example_id=example_id,
                    family_id=f"family-{example_id}",
                    condition=condition,
                    supervised_tokens=2,
                    summed_nll=values[condition],
                    teacher_kl_sum_to_native=(
                        0.0 if condition == "native" else 1.0
                    ),
                    top1_matches_to_native=2,
                )
            )

    result = aggregate_cross_block_replacement_conditions(
        metrics,
        provenance=_oracle_provenance(),
    )
    conditions = {
        value.condition: value for value in result.conditions
    }

    assert conditions["ablation"].delta_nll_per_token_to_native == 0.0
    assert conditions[
        "ablation"
    ].absolute_delta_nll_per_token_to_native == pytest.approx(1.0)
    assert conditions[
        "replacement"
    ].absolute_delta_nll_per_token_to_native == pytest.approx(0.5)
    assert result.replacement_absolute_nll_recovery_vs_ablation == (
        pytest.approx(0.5)
    )


def test_oracle_requires_complete_and_strictly_paired_conditions() -> None:
    metrics = list(_oracle_metrics())
    with pytest.raises(ValueError, match="missing an oracle condition"):
        aggregate_cross_block_replacement_conditions(
            metrics[:-1],
            provenance=_oracle_provenance(),
        )

    duplicate = metrics + [metrics[0]]
    with pytest.raises(ValueError, match="only once"):
        aggregate_cross_block_replacement_conditions(
            duplicate,
            provenance=_oracle_provenance(),
        )

    mismatched = list(metrics)
    mismatched[1] = replace(mismatched[1], supervised_tokens=3)
    with pytest.raises(ValueError, match="share family and token count"):
        aggregate_cross_block_replacement_conditions(
            mismatched,
            provenance=_oracle_provenance(),
        )


def test_native_condition_must_be_exact_self_control() -> None:
    with pytest.raises(ValueError, match="exact self-comparison"):
        CrossBlockReplacementConditionMetric(
            example_id="a",
            family_id="family-a",
            condition="native",
            supervised_tokens=2,
            summed_nll=1.0,
            teacher_kl_sum_to_native=0.1,
            top1_matches_to_native=2,
        )
    with pytest.raises(ValueError, match="exact self-comparison"):
        CrossBlockReplacementConditionMetric(
            example_id="a",
            family_id="family-a",
            condition="native",
            supervised_tokens=2,
            summed_nll=1.0,
            teacher_kl_sum_to_native=0.0,
            top1_matches_to_native=1,
        )


def test_oracle_reports_undefined_recovery_when_ablation_has_no_effect() -> None:
    metrics = []
    for condition in REPLACEMENT_CONDITIONS:
        metrics.append(
            CrossBlockReplacementConditionMetric(
                example_id="a",
                family_id="family-a",
                condition=condition,
                supervised_tokens=1,
                summed_nll=2.0,
                teacher_kl_sum_to_native=0.0,
                top1_matches_to_native=1,
            )
        )

    result = aggregate_cross_block_replacement_conditions(
        metrics,
        provenance=_oracle_provenance(),
    )

    assert result.replacement_kl_recovery_vs_ablation is None
    assert (
        result.replacement_absolute_nll_recovery_vs_ablation is None
    )
