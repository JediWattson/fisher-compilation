import copy

import pytest
import torch

from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceGates,
)
from fisher_graph.state_conditioned_reference_selection import (
    FULL_REFERENCE_WIDTH,
    FullWidthCandidatePrediction,
    FullWidthCandidateScore,
    FullWidthReferenceCandidate,
    FullWidthReferenceProbe,
    FullWidthReferenceSelection,
    FullWidthStructuralMetrics,
    fit_full_width_reference_controls,
    reconstruct_full_width_prediction,
    score_full_width_reference_assessment,
    score_full_width_reference_candidate,
    select_smallest_passing_full_width_reference_candidate,
)


_GAUGE = "a" * 64
_CANDIDATE_BINDING = "b" * 64


def _target(length: int) -> torch.Tensor:
    return torch.zeros(
        1,
        length,
        FULL_REFERENCE_WIDTH,
        dtype=torch.float64,
    )


def _probe(
    probe_id: str,
    split: str,
    family: str,
    target: torch.Tensor,
    *,
    collision_group: str | None = None,
    collision_variant: str | None = None,
) -> FullWidthReferenceProbe:
    assert target.ndim == 3
    return FullWidthReferenceProbe(
        probe_id=probe_id,
        split=split,
        family=family,
        standardized_target=target,
        logical_positions=torch.arange(target.shape[1]).view(1, -1),
        valid_mask=torch.ones(
            target.shape[:2],
            dtype=torch.bool,
        ),
        standardized_gauge_sha256=_GAUGE,
        collision_group=collision_group,
        collision_variant=collision_variant,
    )


def _healthy_structure() -> FullWidthStructuralMetrics:
    return FullWidthStructuralMetrics(
        prepared_vs_analytic_relative_error=0.0,
        causality_violation=0.0,
        padding_violation=0.0,
        repeat_relative_error=0.0,
        in_support_fraction=1.0,
    )


def _candidate(
    candidate_id: str,
    probes: tuple[FullWidthReferenceProbe, ...],
    *,
    target_rank: int,
    source_rank: int | None = None,
    stored_scalars: int = 100,
    predictions: dict[str, torch.Tensor] | None = None,
    structural_metrics: FullWidthStructuralMetrics | None = None,
) -> FullWidthReferenceCandidate:
    retained = []
    for probe in probes:
        value = (
            probe.standardized_target[..., :target_rank]
            if predictions is None
            else predictions[probe.probe_id]
        )
        retained.append(
            FullWidthCandidatePrediction(
                probe_id=probe.probe_id,
                retained_standardized_prediction=value,
                standardized_gauge_sha256=_GAUGE,
            )
        )
    return FullWidthReferenceCandidate(
        candidate_id=candidate_id,
        source_rank=target_rank if source_rank is None else source_rank,
        target_rank=target_rank,
        stored_scalar_count=stored_scalars,
        predictions=tuple(retained),
        structural_metrics=(
            _healthy_structure()
            if structural_metrics is None
            else structural_metrics
        ),
        candidate_binding_sha256=_CANDIDATE_BINDING,
    )


def _controls():
    fit = _probe("fit", "fit", "rademacher", _target(8))
    return fit_full_width_reference_controls(
        fit_probes=(fit,),
        position_bin_count=4,
    )


def _separated_collisions() -> tuple[FullWidthReferenceProbe, ...]:
    first = _target(4)
    second = _target(4)
    first[..., 0] = 1.0
    second[..., 0] = 2.0
    return (
        _probe(
            "collision-a",
            "assessment",
            "radial_collision",
            first,
            collision_group="radial-0",
            collision_variant="one",
        ),
        _probe(
            "collision-b",
            "assessment",
            "radial_collision",
            second,
            collision_group="radial-0",
            collision_variant="two",
        ),
    )


def _assessment_panel() -> tuple[FullWidthReferenceProbe, ...]:
    ordinary = _target(5)
    ordinary[..., 0] = torch.linspace(0.5, 1.5, 5)
    return (
        _probe(
            "assessment-ordinary",
            "assessment",
            "sparse",
            ordinary,
        ),
        *_separated_collisions(),
    )


def test_omitted_low_rank_energy_is_counted_in_the_full_width_metric() -> None:
    controls = _controls()
    target = _target(6)
    target[..., 0] = 1.0
    target[..., 63] = 3.0
    probe = _probe("selection", "selection", "rademacher", target)
    rank_eight = _candidate("rank-8", (probe,), target_rank=8)

    reconstructed = reconstruct_full_width_prediction(
        controls=controls,
        probe=probe,
        prediction=rank_eight.predictions[0],
    )
    torch.testing.assert_close(reconstructed[..., :8], target[..., :8])
    torch.testing.assert_close(
        reconstructed[..., 63],
        torch.zeros(1, 6, dtype=torch.float64),
    )

    score = score_full_width_reference_candidate(
        controls=controls,
        selection_probes=(probe,),
        collision_probes=_separated_collisions(),
        candidate=rank_eight,
        gates=SyntheticReferenceGates(),
    )
    assert score.fisher_weighted_relative_error == pytest.approx(
        3.0 / (10.0**0.5)
    )
    assert score.probe_metrics[0].relative_error == pytest.approx(
        score.fisher_weighted_relative_error
    )
    assert not score.gate_flags.fisher_weighted_relative_error

    full_rank = _candidate("rank-64", (probe,), target_rank=64)
    full_score = score_full_width_reference_candidate(
        controls=controls,
        selection_probes=(probe,),
        collision_probes=_separated_collisions(),
        candidate=full_rank,
        gates=SyntheticReferenceGates(),
    )
    assert full_score.fisher_weighted_relative_error == 0.0
    assert full_score.passed


def test_controls_are_fit_only_and_position_bins_handle_variable_lengths() -> None:
    short = _target(2)
    short[0, :, 0] = torch.tensor([0.0, 10.0])
    long = _target(3)
    long[0, :, 0] = torch.tensor([0.0, 5.0, 10.0])
    fit_short = _probe("fit-short", "fit", "ar1", short)
    fit_long = _probe("fit-long", "fit", "ar1", long)
    controls = fit_full_width_reference_controls(
        fit_probes=(fit_long, fit_short),
        position_bin_count=3,
    )

    assert controls.fit_probe_ids == ("fit-long", "fit-short")
    assert controls.fit_target_center[0].item() == pytest.approx(5.0)
    assert controls.normalized_position_bin_counts == (2, 1, 2)
    torch.testing.assert_close(
        controls.normalized_position_bin_centers[:, 0],
        torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64),
    )

    variable = _probe("selection-variable", "selection", "ar1", _target(5))
    position_prediction = controls.position_prediction_for(variable)
    torch.testing.assert_close(
        position_prediction[0, :, 0],
        torch.tensor([0.0, 0.0, 5.0, 10.0, 10.0], dtype=torch.float64),
    )
    assert variable.probe_id not in controls.fit_probe_ids
    assert variable.artifact_sha256 not in controls.fit_probe_sha256s

    with pytest.raises(ValueError, match="only 'fit' probes"):
        fit_full_width_reference_controls(
            fit_probes=(fit_short, variable),
            position_bin_count=3,
        )


def test_p90_family_and_collision_gates_report_local_failures() -> None:
    controls = _controls()
    easy_target = _target(100)
    easy_target[..., 0] = 1.0
    hard_target = _target(10)
    hard_target[..., 0] = 1.0
    easy = _probe("easy", "selection", "rademacher", easy_target)
    hard = _probe("hard", "selection", "ar1", hard_target)
    hard_prediction = hard_target.clone()
    hard_prediction[:, -1] = 0.0
    candidate = _candidate(
        "localized-error",
        (easy, hard),
        target_rank=64,
        predictions={
            "easy": easy_target.clone(),
            "hard": hard_prediction,
        },
    )

    collision_target = _target(4)
    collision_target[..., 0] = 1.0
    identical_collisions = (
        _probe(
            "same-a",
            "assessment",
            "radial_collision",
            collision_target,
            collision_group="same",
            collision_variant="a",
        ),
        _probe(
            "same-b",
            "assessment",
            "radial_collision",
            collision_target.clone(),
            collision_group="same",
            collision_variant="b",
        ),
    )
    gates = SyntheticReferenceGates(
        maximum_fisher_weighted_relative_error=0.2,
        minimum_reference_cosine=0.9,
        minimum_error_reduction_vs_constant=0.5,
        minimum_error_reduction_vs_position_only=0.5,
        maximum_per_probe_p90_relative_error=0.05,
        maximum_worst_panel_relative_error=0.2,
        minimum_collision_target_relative_difference=0.1,
    )
    score = score_full_width_reference_candidate(
        controls=controls,
        selection_probes=(easy, hard),
        collision_probes=identical_collisions,
        candidate=candidate,
        gates=gates,
    )

    assert score.fisher_weighted_relative_error < 0.1
    assert score.gate_flags.fisher_weighted_relative_error
    assert score.gate_flags.reference_cosine
    assert not score.gate_flags.per_probe_p90_relative_error
    assert score.maximum_per_probe_p90_relative_error == pytest.approx(0.1)
    assert not score.gate_flags.worst_family_relative_error
    assert score.worst_family_relative_error == pytest.approx(10.0**-0.5)
    assert not score.gate_flags.collision_target_relative_difference
    assert score.minimum_collision_target_relative_difference == 0.0
    assert not score.passed


def test_structural_flags_are_applied_without_hiding_raw_metrics() -> None:
    controls = _controls()
    target = _target(5)
    target[..., 0] = 1.0
    probe = _probe("selection", "selection", "rademacher", target)
    structural = FullWidthStructuralMetrics(
        prepared_vs_analytic_relative_error=2e-5,
        causality_violation=2e-6,
        padding_violation=2e-6,
        repeat_relative_error=2e-7,
        in_support_fraction=0.98,
    )
    candidate = _candidate(
        "bad-structure",
        (probe,),
        target_rank=8,
        structural_metrics=structural,
    )
    score = score_full_width_reference_candidate(
        controls=controls,
        selection_probes=(probe,),
        collision_probes=_separated_collisions(),
        candidate=candidate,
        gates=SyntheticReferenceGates(),
    )

    assert score.fisher_weighted_relative_error == 0.0
    assert not score.gate_flags.prepared_vs_analytic_relative_error
    assert not score.gate_flags.causality_violation
    assert not score.gate_flags.padding_violation
    assert not score.gate_flags.repeat_relative_error
    assert not score.gate_flags.in_support_fraction
    assert score.structural_metrics == structural
    assert not score.passed


def test_selection_is_deterministic_and_uses_frozen_accounting_order() -> None:
    controls = _controls()
    target = _target(7)
    target[..., 0] = torch.linspace(1.0, 2.0, 7)
    probe = _probe("selection", "selection", "rademacher", target)
    candidates = (
        _candidate(
            "more-storage",
            (probe,),
            source_rank=4,
            target_rank=8,
            stored_scalars=101,
        ),
        _candidate(
            "wider-source",
            (probe,),
            source_rank=32,
            target_rank=8,
            stored_scalars=90,
        ),
        _candidate(
            "wider-target",
            (probe,),
            source_rank=16,
            target_rank=32,
            stored_scalars=90,
        ),
        _candidate(
            "winner",
            (probe,),
            source_rank=16,
            target_rank=8,
            stored_scalars=90,
        ),
    )
    kwargs = {
        "controls": controls,
        "selection_probes": (probe,),
        "collision_probes": _separated_collisions(),
        "gates": SyntheticReferenceGates(),
    }
    forward = select_smallest_passing_full_width_reference_candidate(
        candidates=candidates,
        **kwargs,
    )
    reverse = select_smallest_passing_full_width_reference_candidate(
        candidates=tuple(reversed(candidates)),
        **kwargs,
    )

    assert forward.selected_candidate_id == "winner"
    assert forward.selected_stored_scalar_count == 90
    assert forward.selected_source_rank == 16
    assert forward.selected_target_rank == 8
    assert forward.passed_candidate_ids == tuple(
        sorted(candidate.candidate_id for candidate in candidates)
    )
    assert reverse.selected_candidate_id == forward.selected_candidate_id
    assert reverse.artifact_sha256 == forward.artifact_sha256


def test_selection_can_defer_collision_gate_without_opening_assessment() -> None:
    controls = _controls()
    target = _target(7)
    target[..., 0] = torch.linspace(1.0, 2.0, 7)
    probe = _probe("selection", "selection", "rademacher", target)
    candidate = _candidate("candidate", (probe,), target_rank=8)
    gates = SyntheticReferenceGates(
        minimum_collision_target_relative_difference=0.0,
    )

    result = select_smallest_passing_full_width_reference_candidate(
        controls=controls,
        selection_probes=(probe,),
        collision_probes=(),
        candidates=(candidate,),
        gates=gates,
    )

    assert result.selected_candidate_id == "candidate"
    assert result.candidate_scores[0].collision_metrics == ()
    assert result.candidate_scores[
        0
    ].minimum_collision_target_relative_difference == 0.0
    assert result.candidate_scores[
        0
    ].gate_flags.collision_target_relative_difference


def test_selection_rejects_implicit_collision_gate_deferral() -> None:
    controls = _controls()
    target = _target(7)
    target[..., 0] = torch.linspace(1.0, 2.0, 7)
    probe = _probe("selection", "selection", "rademacher", target)
    candidate = _candidate("candidate", (probe,), target_rank=8)

    with pytest.raises(ValueError, match="zero deferred collision gate"):
        select_smallest_passing_full_width_reference_candidate(
            controls=controls,
            selection_probes=(probe,),
            collision_probes=(),
            candidates=(candidate,),
            gates=SyntheticReferenceGates(),
        )


def test_assessment_scores_one_frozen_candidate_on_the_complete_panel() -> None:
    controls = _controls()
    probes = _assessment_panel()
    original_splits = tuple(probe.split for probe in probes)
    original_hashes = tuple(probe.artifact_sha256 for probe in probes)
    candidate = _candidate("frozen", probes, target_rank=64)

    score = score_full_width_reference_assessment(
        controls=controls,
        assessment_probes=probes,
        candidate=candidate,
        gates=SyntheticReferenceGates(),
    )

    assert isinstance(score, FullWidthCandidateScore)
    assert score.candidate_id == "frozen"
    assert score.fisher_weighted_relative_error == 0.0
    assert score.passed
    assert {metric.probe_id for metric in score.probe_metrics} == {
        probe.probe_id for probe in probes
    }
    assert {
        metric.family: metric.probe_count for metric in score.family_metrics
    } == {"radial_collision": 2, "sparse": 1}
    assert len(score.collision_metrics) == 1
    assert score.collision_metrics[0].collision_group == "radial-0"
    assert score.collision_metrics[0].variant_count == 2
    assert tuple(probe.split for probe in probes) == original_splits
    assert tuple(probe.artifact_sha256 for probe in probes) == original_hashes


def test_assessment_rejects_relabeling_and_incomplete_predictions() -> None:
    controls = _controls()
    selection_target = _target(4)
    selection_target[..., 0] = 1.0
    selection_probe = _probe(
        "selection",
        "selection",
        "sparse",
        selection_target,
    )
    selection_candidate = _candidate(
        "selection-candidate",
        (selection_probe,),
        target_rank=64,
    )

    with pytest.raises(ValueError, match="only 'assessment' probes"):
        score_full_width_reference_assessment(
            controls=controls,
            assessment_probes=(selection_probe,),
            candidate=selection_candidate,
            gates=SyntheticReferenceGates(),
        )

    probes = _assessment_panel()
    incomplete_candidate = _candidate(
        "incomplete",
        probes[:-1],
        target_rank=64,
    )
    with pytest.raises(ValueError, match="not probe-aligned"):
        score_full_width_reference_assessment(
            controls=controls,
            assessment_probes=probes,
            candidate=incomplete_candidate,
            gates=SyntheticReferenceGates(),
        )


def test_assessment_requires_collision_tags_for_a_nonzero_gate() -> None:
    controls = _controls()
    target = _target(4)
    target[..., 0] = 1.0
    probe = _probe("assessment", "assessment", "sparse", target)
    candidate = _candidate("frozen", (probe,), target_rank=64)

    with pytest.raises(ValueError, match="no collision-tagged rows"):
        score_full_width_reference_assessment(
            controls=controls,
            assessment_probes=(probe,),
            candidate=candidate,
            gates=SyntheticReferenceGates(),
        )

    score = score_full_width_reference_assessment(
        controls=controls,
        assessment_probes=(probe,),
        candidate=candidate,
        gates=SyntheticReferenceGates(
            minimum_collision_target_relative_difference=0.0,
        ),
    )
    assert score.collision_metrics == ()
    assert score.gate_flags.collision_target_relative_difference


def _selection_result() -> FullWidthReferenceSelection:
    controls = _controls()
    target = _target(7)
    target[..., 0] = torch.linspace(1.0, 2.0, 7)
    probe = _probe("selection", "selection", "rademacher", target)
    return select_smallest_passing_full_width_reference_candidate(
        controls=controls,
        selection_probes=(probe,),
        collision_probes=_separated_collisions(),
        candidates=(
            _candidate(
                "winner",
                (probe,),
                source_rank=8,
                target_rank=8,
                stored_scalars=90,
            ),
            _candidate(
                "larger",
                (probe,),
                source_rank=16,
                target_rank=16,
                stored_scalars=120,
            ),
        ),
        gates=SyntheticReferenceGates(),
    )


def test_selection_state_round_trip_restores_nested_authenticated_values() -> None:
    original = _selection_result()

    restored = FullWidthReferenceSelection.from_state_dict(
        original.state_dict()
    )

    assert restored == original
    assert restored.artifact_sha256 == original.artifact_sha256
    assert restored.selected_candidate_id == "winner"
    assert restored.selected_candidate_artifact_sha256 == (
        original.candidate_scores[1].candidate_artifact_sha256
    )
    assert restored.selected_stored_scalar_count == 90
    assert restored.selected_source_rank == 8
    assert restored.selected_target_rank == 8
    assert tuple(
        value.artifact_sha256 for value in restored.candidate_scores
    ) == tuple(
        value.artifact_sha256 for value in original.candidate_scores
    )


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    (
        (
            lambda state: state.pop("gates_sha256"),
            ValueError,
            "fields do not match frozen format",
        ),
        (
            lambda state: state.update({"unknown": True}),
            ValueError,
            "fields do not match frozen format",
        ),
        (
            lambda state: state.update({"selected_source_rank": True}),
            ValueError,
            "positive integer",
        ),
        (
            lambda state: state["candidate_scores"][0][
                "probe_metrics"
            ][0].update({"unknown": 0}),
            ValueError,
            "fields do not match frozen format",
        ),
        (
            lambda state: state["candidate_scores"][0][
                "probe_metrics"
            ][0].update({"relative_error": 0}),
            TypeError,
            "must be a float",
        ),
        (
            lambda state: state["candidate_scores"][0][
                "structural_metrics"
            ].update({"in_support_fraction": 0.5}),
            ValueError,
            "hash mismatch",
        ),
        (
            lambda state: state["candidate_scores"][0]["gate_flags"].update(
                {"reference_cosine": False}
            ),
            ValueError,
            "pass flag does not match",
        ),
        (
            lambda state: state["candidate_score_sha256s"].__setitem__(
                0,
                "f" * 64,
            ),
            ValueError,
            "summary does not match",
        ),
        (
            lambda state: state.update({"selected_source_rank": 64}),
            ValueError,
            "accounting drifted",
        ),
        (
            lambda state: state.update({"artifact_sha256": "f" * 64}),
            ValueError,
            "hash mismatch",
        ),
    ),
)
def test_selection_state_restoration_fails_closed(
    mutation,
    error_type,
    message,
) -> None:
    state = copy.deepcopy(_selection_result().state_dict())
    mutation(state)

    with pytest.raises(error_type, match=message):
        FullWidthReferenceSelection.from_state_dict(state)


def test_selection_state_restoration_rejects_non_mapping_and_wrong_lists() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        FullWidthReferenceSelection.from_state_dict([])

    state = _selection_result().state_dict()
    state["candidate_scores"] = tuple(state["candidate_scores"])
    with pytest.raises(TypeError, match="must be a list"):
        FullWidthReferenceSelection.from_state_dict(state)
