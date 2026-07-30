from __future__ import annotations

from dataclasses import replace

import pytest

from fisher_graph.compiler.progressive import (
    CandidateEvaluation,
    DevelopmentEvaluationCoverage,
    DevelopmentCorpus,
    FrozenCalibrationAChallenger,
    FrozenProgressiveCandidateHandoff,
    MutationProposal,
    ProgressiveBehavioralFidelity,
    ProgressiveBehavioralTargets,
    ProgressiveCandidate,
    ProgressiveCompilationProtocol,
    ProgressiveFidelity,
    ProgressiveFidelityTargets,
    ProgressiveResourceBudget,
    ProgressiveResourceFootprint,
    ResidualMap,
    ResidualTarget,
    freeze_progressive_candidate,
    run_progressive_compilation,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _resources(
    total: int,
    *,
    retained: int = 100,
    complete: bool = True,
    execution_sha256: str | None = None,
    accounting_sha256: str | None = None,
) -> ProgressiveResourceFootprint:
    support = 10
    compiled = total - retained - support
    assert compiled >= 0
    return ProgressiveResourceFootprint(
        candidate_execution_sha256=execution_sha256 or _sha(200),
        accounting_artifact_sha256=accounting_sha256 or _sha(1200),
        parameter_scope="unit.full-model.parameters",
        compute_scope="unit.full-model.macs-per-token",
        runtime_id="unit.cpu",
        runtime_dtype="float32",
        sequence_scope_sha256=_sha(1201),
        compiled_learned_parameters=compiled,
        retained_source_learned_parameters=retained,
        support_learned_parameters=support,
        compiled_runtime_parameter_bytes=compiled * 4,
        retained_source_runtime_parameter_bytes=retained * 4,
        support_runtime_parameter_bytes=support * 4,
        compiled_logical_macs_per_token=compiled,
        retained_source_logical_macs_per_token=retained,
        support_logical_macs_per_token=support,
        cost_complete=complete,
        incomplete_cost_reasons=(
            () if complete else ("router_sort_comparisons",)
        ),
    )


def _fidelity(
    burden: float,
    *,
    kl: float = 0.0,
    top1: float = 1.0,
) -> ProgressiveFidelity:
    candidate_behavior = ProgressiveBehavioralFidelity(
        absolute_delta_nll_per_token=burden,
        source_to_candidate_kl_per_token=kl,
        top1_agreement_to_source=top1,
        per_prompt_p90_absolute_delta_nll_per_token=0.0,
        per_prompt_p10_top1_agreement_to_source=1.0,
    )
    passing_behavior = ProgressiveBehavioralFidelity(
        absolute_delta_nll_per_token=0.0,
        source_to_candidate_kl_per_token=0.0,
        top1_agreement_to_source=1.0,
        per_prompt_p90_absolute_delta_nll_per_token=0.0,
        per_prompt_p10_top1_agreement_to_source=1.0,
    )
    return ProgressiveFidelity(
        candidate_behavior=candidate_behavior,
        projection_oracle_behavior=passing_behavior,
        carrier_oracle_behavior=passing_behavior,
        operator_nrmse=0.0,
        boundary_relative_error=0.0,
        boundary_cosine=1.0,
        valid_target_coverage=1.0,
        worst_family_boundary_relative_error=0.0,
        worst_family_boundary_cosine=1.0,
        minimum_family_source_modal_signal_l2_norm=1.0,
        projection_full_width_relative_error=0.0,
        projection_full_width_cosine=1.0,
        worst_family_projection_relative_error=0.0,
        worst_family_projection_cosine=1.0,
        minimum_family_source_full_width_signal_l2_norm=1.0,
    )


def _protocol(
    *,
    compact_after_fidelity: bool = True,
    max_iterations: int = 8,
    max_total_fraction: float = 1.0,
    seed_resources: ProgressiveResourceFootprint | None = None,
) -> ProgressiveCompilationProtocol:
    frozen_seed_resources = seed_resources or _resources(700)
    return ProgressiveCompilationProtocol(
        protocol_id="unit.progressive",
        source_model_sha256=_sha(1),
        seed_candidate_artifact_sha256=_sha(100),
        seed_candidate_execution_sha256=_sha(200),
        seed_runtime_binding_sha256=_sha(201),
        seed_resource_receipt_sha256=(
            frozen_seed_resources.receipt_sha256
        ),
        seed_lineage_sha256s=(_sha(202), _sha(203)),
        corpus=DevelopmentCorpus(
            corpus_id="unit.corpus",
            fit_manifest_sha256=_sha(2),
            selection_manifest_sha256=_sha(3),
            guard_manifest_sha256=_sha(4),
            fit_example_count=12,
            selection_example_count=8,
            guard_example_count=6,
            fit_family_ids=("fit-a", "fit-b"),
            selection_family_ids=("selection-a", "selection-b"),
            guard_family_ids=("guard-a", "guard-b"),
        ),
        forbidden_assessment_manifest_sha256s=(_sha(5),),
        fidelity_targets=ProgressiveFidelityTargets(
            candidate_behavior=ProgressiveBehavioralTargets(
                absolute_delta_nll_per_token_max=1.0,
                source_to_candidate_kl_per_token_max=1.0,
                top1_agreement_to_source_min=0.5,
                per_prompt_p90_absolute_delta_nll_per_token_max=1.0,
                per_prompt_p10_top1_agreement_to_source_min=0.5,
            ),
            projection_oracle_behavior=ProgressiveBehavioralTargets(
                absolute_delta_nll_per_token_max=1.0,
                source_to_candidate_kl_per_token_max=1.0,
                top1_agreement_to_source_min=0.5,
                per_prompt_p90_absolute_delta_nll_per_token_max=1.0,
                per_prompt_p10_top1_agreement_to_source_min=0.5,
            ),
            carrier_oracle_behavior=ProgressiveBehavioralTargets(
                absolute_delta_nll_per_token_max=1.0,
                source_to_candidate_kl_per_token_max=1.0,
                top1_agreement_to_source_min=0.5,
                per_prompt_p90_absolute_delta_nll_per_token_max=1.0,
                per_prompt_p10_top1_agreement_to_source_min=0.5,
            ),
            operator_nrmse_max=1.0,
            boundary_relative_error_max=1.0,
            boundary_cosine_min=0.5,
            valid_target_coverage_min=0.5,
            worst_family_boundary_relative_error_max=1.0,
            worst_family_boundary_cosine_min=0.5,
            minimum_family_source_modal_signal_l2_norm=0.1,
            projection_full_width_relative_error_max=1.0,
            projection_full_width_cosine_min=0.5,
            worst_family_projection_relative_error_max=1.0,
            worst_family_projection_cosine_min=0.5,
            minimum_family_source_full_width_signal_l2_norm=0.1,
        ),
        resource_budget=ProgressiveResourceBudget(
            parameter_scope="unit.full-model.parameters",
            compute_scope="unit.full-model.macs-per-token",
            runtime_id="unit.cpu",
            runtime_dtype="float32",
            sequence_scope_sha256=_sha(1201),
            source_learned_parameters=1000,
            source_runtime_parameter_bytes=4000,
            source_logical_macs_per_token=1000,
            max_total_parameter_fraction=max_total_fraction,
            max_total_parameter_byte_fraction=max_total_fraction,
            max_total_mac_fraction=max_total_fraction,
            max_retained_source_parameter_fraction=0.5,
            max_retained_source_parameter_byte_fraction=0.5,
            max_retained_source_mac_fraction=0.5,
        ),
        max_iterations=max_iterations,
        compact_after_fidelity=compact_after_fidelity,
    )


def _seed(
    *,
    resources: ProgressiveResourceFootprint | None = None,
) -> ProgressiveCandidate:
    return ProgressiveCandidate(
        candidate_id="candidate-0",
        iteration=0,
        artifact_sha256=_sha(100),
        execution_sha256=_sha(200),
        runtime_binding_sha256=_sha(201),
        resources=resources or _resources(700),
        mutation_kind="seed",
    )


def _evaluation(
    protocol: ProgressiveCompilationProtocol,
    candidate: ProgressiveCandidate,
    fidelity: ProgressiveFidelity,
    *,
    role: str = "calibration_a_selection",
    challenger_receipt_sha256: str | None = None,
) -> CandidateEvaluation:
    manifest = (
        protocol.corpus.selection_manifest_sha256
        if role == "calibration_a_selection"
        else protocol.corpus.guard_manifest_sha256
    )
    example_count = (
        protocol.corpus.selection_example_count
        if role == "calibration_a_selection"
        else protocol.corpus.guard_example_count
    )
    family_ids = (
        protocol.corpus.selection_family_ids
        if role == "calibration_a_selection"
        else protocol.corpus.guard_family_ids
    )
    return CandidateEvaluation(
        protocol_sha256=protocol.artifact_sha256,
        development_role=role,  # type: ignore[arg-type]
        manifest_sha256=manifest,
        candidate_artifact_sha256=candidate.artifact_sha256,
        candidate_receipt_sha256=candidate.receipt_sha256,
        evaluation_artifact_sha256=_sha(
            300 + candidate.iteration + (100 if role.endswith("guard") else 0)
        ),
        coverage=DevelopmentEvaluationCoverage(
            manifest_sha256=manifest,
            expected_example_count=example_count,
            observed_example_count=example_count,
            expected_family_ids=family_ids,
            observed_family_ids=family_ids,
            supervised_token_count=example_count * 4,
            membership_receipt_sha256=_sha(
                900 + candidate.iteration
                + (100 if role.endswith("guard") else 0)
            ),
            model_inputs_receipt_sha256=_sha(
                1100 + candidate.iteration
                + (100 if role.endswith("guard") else 0)
            ),
            complete=True,
        ),
        fidelity=fidelity,
        resources=candidate.resources,
        challenger_receipt_sha256=challenger_receipt_sha256,
    )


def _guard_evaluation(
    protocol: ProgressiveCompilationProtocol,
    challenger,
    fidelity: ProgressiveFidelity,
) -> CandidateEvaluation:
    return _evaluation(
        protocol,
        challenger.candidate,
        fidelity,
        role="calibration_a_guard",
        challenger_receipt_sha256=challenger.receipt_sha256,
    )


def _map(
    candidate: ProgressiveCandidate,
    fit,
) -> ResidualMap:
    assert fit.role == "calibration_a_fit"
    assert not hasattr(fit, "selection_manifest_sha256")
    assert not hasattr(fit, "guard_manifest_sha256")
    return ResidualMap(
        protocol_sha256=fit.protocol_sha256,
        fit_manifest_sha256=fit.manifest_sha256,
        candidate_artifact_sha256=candidate.artifact_sha256,
        candidate_receipt_sha256=candidate.receipt_sha256,
        iteration=candidate.iteration,
        mapper_id="unit.mapper",
        mapper_version=1,
        analysis_artifact_sha256=_sha(400 + candidate.iteration),
        targets=(
            ResidualTarget(
                rank=0,
                location=f"block/{candidate.iteration}",
                direction_sha256=_sha(500 + candidate.iteration),
                residual_energy_fraction=0.75,
                loss_coupling=2.0,
                jvp_gain=1.0,
            ),
        ),
    )


def _proposal(
    candidate: ProgressiveCandidate,
    residual_map: ResidualMap,
    *,
    phase: str,
    mutation_kind: str,
    resources: ProgressiveResourceFootprint,
    suffix: int = 0,
) -> MutationProposal:
    child_execution_sha256 = _sha(
        800 + candidate.iteration * 10 + suffix
    )
    resources = replace(
        resources,
        candidate_execution_sha256=child_execution_sha256,
        accounting_artifact_sha256=_sha(
            1300 + candidate.iteration * 10 + suffix
        ),
    )
    return MutationProposal(
        proposal_id=f"proposal-{candidate.iteration}-{suffix}",
        phase=phase,  # type: ignore[arg-type]
        mutation_kind=mutation_kind,  # type: ignore[arg-type]
        parent_artifact_sha256=candidate.artifact_sha256,
        parent_receipt_sha256=candidate.receipt_sha256,
        residual_map_sha256=residual_map.receipt_sha256,
        recipe_sha256=_sha(600 + candidate.iteration * 10 + suffix),
        target_ranks=(0,),
        resources=resources,
    )


def _build(
    parent: ProgressiveCandidate,
    proposal: MutationProposal,
    *,
    suffix: int = 0,
) -> ProgressiveCandidate:
    return ProgressiveCandidate(
        candidate_id=f"candidate-{parent.iteration + 1}-{suffix}",
        iteration=parent.iteration + 1,
        artifact_sha256=_sha(
            700 + parent.iteration * 10 + suffix
        ),
        execution_sha256=_sha(
            800 + parent.iteration * 10 + suffix
        ),
        runtime_binding_sha256=_sha(
            850 + parent.iteration * 10 + suffix
        ),
        resources=proposal.resources,
        mutation_kind=proposal.mutation_kind,
        parent_artifact_sha256=parent.artifact_sha256,
        proposal_sha256=proposal.receipt_sha256,
    )


def test_repair_then_compact_loop_uses_one_final_guard() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(4.0))
    selection_burdens = {1: 2.0, 2: 0.8, 3: 0.9}
    totals = {0: 740, 1: 780, 2: 620}
    selection_calls = []
    guard_calls = []

    def propose(candidate, residual_map, phase):
        if candidate.iteration == 3:
            return ()
        expected_phase = (
            "repair" if candidate.iteration < 2 else "compact"
        )
        assert phase == expected_phase
        kind = (
            "widen_carrier"
            if candidate.iteration == 0
            else "add_residual_edge"
            if candidate.iteration == 1
            else "factorize_edges"
        )
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind=kind,
                resources=_resources(totals[candidate.iteration]),
            ),
        )

    def build(parent, proposal):
        return _build(parent, proposal)

    def evaluate_selection(candidate, selection):
        selection_calls.append((candidate.iteration, selection.role))
        assert selection.role == "calibration_a_selection"
        assert not hasattr(selection, "guard_manifest_sha256")
        return _evaluation(
            protocol,
            candidate,
            _fidelity(selection_burdens[candidate.iteration]),
        )

    def evaluate_guard(challenger, guard):
        candidate = challenger.candidate
        guard_calls.append((candidate.iteration, guard.role))
        assert candidate.iteration == 3
        assert guard.role == "calibration_a_guard"
        assert challenger.to_dict()["guard_opened"] is False
        return _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.95),
        )

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=seed_selection,
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=build,
        evaluate_selection=evaluate_selection,
        evaluate_guard=evaluate_guard,
    )

    assert result.status == "ready_for_candidate_binding"
    assert result.final_candidate.iteration == 3
    assert [item.phase for item in result.iterations] == [
        "repair",
        "repair",
        "compact",
        "compact",
    ]
    assert selection_calls == [
        (1, "calibration_a_selection"),
        (2, "calibration_a_selection"),
        (3, "calibration_a_selection"),
    ]
    assert guard_calls == [(3, "calibration_a_guard")]
    assert result.guard_evaluation is not None
    assert len(result.proposal_archive) == 3
    assert len(result.candidate_archive) == 4
    assert len(result.selection_evaluation_archive) == 4
    assert result.to_dict()["archive"][
        "raw_and_dominated_points_retained"
    ] is True

    handoff = freeze_progressive_candidate(
        protocol=protocol,
        result=result,
    )
    assert isinstance(handoff, FrozenProgressiveCandidateHandoff)
    assert handoff.assessment_accessed is False
    assert "assessment_manifest" not in handoff.to_dict()
    assert handoff.candidate_id == result.final_candidate.candidate_id
    assert handoff.candidate_runtime_binding_sha256 == (
        result.final_candidate.runtime_binding_sha256
    )
    assert handoff.to_dict()["candidate_runtime_binding_sha256"] == (
        result.final_candidate.runtime_binding_sha256
    )


def test_guard_is_a_single_terminal_veto_not_another_selector() -> None:
    protocol = _protocol(compact_after_fidelity=False)
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(0.8))
    guard_calls = 0

    def guard(challenger, view):
        nonlocal guard_calls
        guard_calls += 1
        return _guard_evaluation(
            protocol,
            challenger,
            _fidelity(2.0),
        )

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=seed_selection,
        map_residual=lambda *_: pytest.fail("fit loop should not run"),
        propose_mutations=lambda *_: pytest.fail(
            "proposal loop should not run"
        ),
        build_candidate=lambda *_: pytest.fail("builder should not run"),
        evaluate_selection=lambda *_: pytest.fail(
            "selection should not run"
        ),
        evaluate_guard=guard,
    )

    assert guard_calls == 1
    assert result.status == "rejected_by_guard"
    assert result.final_candidate == seed
    with pytest.raises(ValueError, match="ready_for_candidate_binding"):
        freeze_progressive_candidate(protocol=protocol, result=result)


def test_guard_must_bind_the_previously_frozen_challenger() -> None:
    protocol = _protocol(compact_after_fidelity=False)
    seed = _seed()

    def wrong_guard(challenger, _):
        return _evaluation(
            protocol,
            challenger.candidate,
            _fidelity(0.8),
            role="calibration_a_guard",
            challenger_receipt_sha256=_sha(999),
        )

    with pytest.raises(ValueError, match="challenger binding differs"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=_evaluation(
                protocol,
                seed,
                _fidelity(0.8),
            ),
            map_residual=lambda *_: pytest.fail("loop should not run"),
            propose_mutations=lambda *_: pytest.fail(
                "loop should not run"
            ),
            build_candidate=lambda *_: pytest.fail(
                "loop should not run"
            ),
            evaluate_selection=lambda *_: pytest.fail(
                "loop should not run"
            ),
            evaluate_guard=wrong_guard,
        )


def test_family_partitions_are_pairwise_disjoint() -> None:
    protocol = _protocol()
    assert protocol.metadata()["corpus"]["pairwise_family_disjoint"]

    with pytest.raises(ValueError, match="pairwise disjoint"):
        replace(
            protocol.corpus,
            selection_family_ids=("fit-a", "selection-b"),
        )


@pytest.mark.parametrize(
    "field",
    [
        "fit_manifest_sha256",
        "selection_manifest_sha256",
        "guard_manifest_sha256",
    ],
)
def test_assessment_manifest_is_rejected_from_every_development_role(
    field: str,
) -> None:
    protocol = _protocol()
    corpus = replace(
        protocol.corpus,
        **{field: protocol.forbidden_assessment_manifest_sha256s[0]},
    )
    with pytest.raises(ValueError, match="assessment manifest"):
        replace(protocol, corpus=corpus, artifact_sha256="")


def test_incomplete_cost_proposal_is_not_built_or_evaluated() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(4.0))
    built = 0
    evaluated = 0

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="widen_carrier",
                resources=_resources(600, complete=False),
            ),
        )

    def build(*_):
        nonlocal built
        built += 1
        raise AssertionError

    def evaluate(*_):
        nonlocal evaluated
        evaluated += 1
        raise AssertionError

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=seed_selection,
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=build,
        evaluate_selection=evaluate,
        evaluate_guard=evaluate,
    )

    assert result.status == "stalled_budget"
    assert built == 0
    assert evaluated == 0
    assert result.iterations[0].decision == (
        "no_budget_eligible_candidate"
    )


def test_budget_counts_retained_source_and_support_work() -> None:
    protocol = _protocol(max_total_fraction=0.5)
    resources = _resources(610, retained=510)

    violations = protocol.resource_budget.violations(resources)

    assert "total_learned_parameters" in violations
    assert "retained_source_learned_parameters" in violations
    assert "total_logical_macs_per_token" in violations


def test_resource_budget_rejects_incomparable_scopes() -> None:
    protocol = _protocol()
    resources = replace(
        _resources(600),
        compute_scope="other-model.macs-per-token",
    )
    assert protocol.resource_budget.violations(resources) == (
        "incomparable_compute_scope",
    )


def test_non_improving_repair_stalls_without_opening_guard() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(4.0))
    guard_calls = 0

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="refit_edges",
                resources=_resources(650),
            ),
        )

    def evaluate_selection(candidate, _):
        return _evaluation(protocol, candidate, _fidelity(3.95))

    def guard(*_):
        nonlocal guard_calls
        guard_calls += 1
        raise AssertionError

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=seed_selection,
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=_build,
        evaluate_selection=evaluate_selection,
        evaluate_guard=guard,
    )

    assert result.status == "stalled_fidelity"
    assert result.final_candidate == seed
    assert guard_calls == 0


def test_repair_rejects_a_large_regression_on_another_axis() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(
        protocol,
        seed,
        _fidelity(4.0, kl=0.1),
    )

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="split_generator",
                resources=_resources(650),
            ),
        )

    def evaluate_selection(candidate, _):
        return _evaluation(
            protocol,
            candidate,
            _fidelity(3.0, kl=1.0),
        )

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=seed_selection,
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=_build,
        evaluate_selection=evaluate_selection,
        evaluate_guard=lambda *_: pytest.fail("guard must stay closed"),
    )

    assert result.status == "stalled_fidelity"


def test_compaction_must_reduce_a_resource_without_breaking_fidelity() -> None:
    seed_resources = _resources(600)
    protocol = _protocol(
        max_iterations=1,
        seed_resources=seed_resources,
    )
    seed = _seed(resources=seed_resources)
    seed_selection = _evaluation(protocol, seed, _fidelity(0.8))

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="merge_generators",
                resources=_resources(600),
            ),
        )

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=seed_selection,
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=_build,
        evaluate_selection=lambda candidate, _: _evaluation(
            protocol,
            candidate,
            _fidelity(0.7),
        ),
        evaluate_guard=lambda challenger, _: _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.8),
        ),
    )

    assert result.status == "ready_for_candidate_binding"
    assert result.final_candidate == seed
    assert (
        result.iterations[0].decision
        == "no_quality_eligible_candidate"
    )


def test_wrong_selection_role_fails_before_guard() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(4.0))

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="widen_carrier",
                resources=_resources(650),
            ),
        )

    with pytest.raises(ValueError, match="development role differs"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=seed_selection,
            map_residual=_map,
            propose_mutations=propose,
            build_candidate=_build,
            evaluate_selection=lambda candidate, _: _evaluation(
                protocol,
                candidate,
                _fidelity(2.0),
                role="calibration_a_guard",
                challenger_receipt_sha256=_sha(999),
            ),
            evaluate_guard=lambda *_: pytest.fail(
                "guard must stay closed"
            ),
        )


def test_incomplete_selection_manifest_coverage_fails_closed() -> None:
    protocol = _protocol()
    seed = _seed()
    complete = _evaluation(protocol, seed, _fidelity(4.0))
    incomplete = replace(
        complete,
        coverage=replace(
            complete.coverage,
            observed_example_count=(
                complete.coverage.expected_example_count - 1
            ),
            complete=False,
        ),
    )

    with pytest.raises(ValueError, match="complete frozen role"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=incomplete,
            map_residual=lambda *_: pytest.fail("loop must not run"),
            propose_mutations=lambda *_: pytest.fail("loop must not run"),
            build_candidate=lambda *_: pytest.fail("loop must not run"),
            evaluate_selection=lambda *_: pytest.fail(
                "loop must not run"
            ),
            evaluate_guard=lambda *_: pytest.fail("guard must not run"),
        )


def test_wrong_fit_binding_is_rejected() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(4.0))

    def wrong_map(candidate, fit):
        return replace(
            _map(candidate, fit),
            fit_manifest_sha256=_sha(999),
        )

    with pytest.raises(ValueError, match="frozen fit manifest"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=seed_selection,
            map_residual=wrong_map,
            propose_mutations=lambda *_: (),
            build_candidate=lambda *_: pytest.fail("builder must not run"),
            evaluate_selection=lambda *_: pytest.fail(
                "selection must not run"
            ),
            evaluate_guard=lambda *_: pytest.fail(
                "guard must not run"
            ),
        )


def test_built_candidate_must_match_parent_proposal_and_resources() -> None:
    protocol = _protocol()
    seed = _seed()
    seed_selection = _evaluation(protocol, seed, _fidelity(4.0))

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="widen_carrier",
                resources=_resources(650),
            ),
        )

    def bad_build(parent, proposal):
        return replace(
            _build(parent, proposal),
            resources=_resources(
                640,
                execution_sha256=(
                    proposal.resources.candidate_execution_sha256
                ),
                accounting_sha256=_sha(1990),
            ),
        )

    with pytest.raises(ValueError, match="resources differ"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=seed_selection,
            map_residual=_map,
            propose_mutations=propose,
            build_candidate=bad_build,
            evaluate_selection=lambda *_: pytest.fail(
                "selection must not run"
            ),
            evaluate_guard=lambda *_: pytest.fail(
                "guard must not run"
            ),
        )


def test_protocol_and_transcript_hashes_are_deterministic() -> None:
    first = _protocol(compact_after_fidelity=False)
    second = _protocol(compact_after_fidelity=False)
    assert first.artifact_sha256 == second.artifact_sha256

    def run(protocol):
        seed = _seed()
        return run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=_evaluation(
                protocol,
                seed,
                _fidelity(0.8),
            ),
            map_residual=lambda *_: pytest.fail("loop should not run"),
            propose_mutations=lambda *_: pytest.fail(
                "loop should not run"
            ),
            build_candidate=lambda *_: pytest.fail("loop should not run"),
            evaluate_selection=lambda *_: pytest.fail(
                "loop should not run"
            ),
            evaluate_guard=lambda challenger, _: _guard_evaluation(
                protocol,
                challenger,
                _fidelity(0.9),
            ),
        )

    assert run(first).transcript_sha256 == run(second).transcript_sha256


def test_protocol_hash_rejects_tampering() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        replace(
            protocol,
            max_iterations=protocol.max_iterations + 1,
        )


def test_seed_candidate_is_frozen_by_the_protocol() -> None:
    protocol = _protocol()
    seed = replace(_seed(), artifact_sha256=_sha(999))

    with pytest.raises(ValueError, match="frozen protocol"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=_evaluation(
                protocol,
                seed,
                _fidelity(4.0),
            ),
            map_residual=lambda *_: pytest.fail("loop must not run"),
            propose_mutations=lambda *_: pytest.fail("loop must not run"),
            build_candidate=lambda *_: pytest.fail("loop must not run"),
            evaluate_selection=lambda *_: pytest.fail(
                "loop must not run"
            ),
            evaluate_guard=lambda *_: pytest.fail("guard must not run"),
        )


def test_seed_runtime_binding_is_frozen_by_the_protocol() -> None:
    protocol = _protocol()
    seed = replace(
        _seed(),
        runtime_binding_sha256=_sha(999),
    )

    with pytest.raises(ValueError, match="frozen protocol"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=_evaluation(
                protocol,
                seed,
                _fidelity(4.0),
            ),
            map_residual=lambda *_: pytest.fail("loop must not run"),
            propose_mutations=lambda *_: pytest.fail("loop must not run"),
            build_candidate=lambda *_: pytest.fail("loop must not run"),
            evaluate_selection=lambda *_: pytest.fail(
                "loop must not run"
            ),
            evaluate_guard=lambda *_: pytest.fail("guard must not run"),
        )


def test_seed_resource_receipt_is_frozen_by_the_protocol() -> None:
    protocol = _protocol()
    seed = _seed(resources=_resources(650))

    with pytest.raises(ValueError, match="frozen protocol"):
        run_progressive_compilation(
            protocol=protocol,
            seed_candidate=seed,
            seed_selection_evaluation=_evaluation(
                protocol,
                seed,
                _fidelity(4.0),
            ),
            map_residual=lambda *_: pytest.fail("loop must not run"),
            propose_mutations=lambda *_: pytest.fail("loop must not run"),
            build_candidate=lambda *_: pytest.fail("loop must not run"),
            evaluate_selection=lambda *_: pytest.fail(
                "loop must not run"
            ),
            evaluate_guard=lambda *_: pytest.fail("guard must not run"),
        )


def test_result_rejects_reordered_or_stale_active_head_receipts() -> None:
    protocol = _protocol(compact_after_fidelity=False)
    seed = _seed()

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="widen_carrier",
                resources=_resources(650),
            ),
        )

    result = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=_evaluation(
            protocol,
            seed,
            _fidelity(2.0),
        ),
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=_build,
        evaluate_selection=lambda candidate, _: _evaluation(
            protocol,
            candidate,
            _fidelity(0.8),
        ),
        evaluate_guard=lambda challenger, _: _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.9),
        ),
    )
    stale = replace(
        result.iterations[0],
        parent_candidate_receipt_sha256=_sha(999),
    )

    with pytest.raises(ValueError, match="active candidate head"):
        replace(result, iterations=(stale,))


def test_freeze_rejects_a_fabricated_protocol_unbound_seed() -> None:
    protocol = _protocol(compact_after_fidelity=False)
    seed = _seed()
    valid = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=_evaluation(
            protocol,
            seed,
            _fidelity(0.8),
        ),
        map_residual=lambda *_: pytest.fail("loop should not run"),
        propose_mutations=lambda *_: pytest.fail("loop should not run"),
        build_candidate=lambda *_: pytest.fail("loop should not run"),
        evaluate_selection=lambda *_: pytest.fail(
            "loop should not run"
        ),
        evaluate_guard=lambda challenger, _: _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.9),
        ),
    )
    fake_seed = replace(
        seed,
        artifact_sha256=_sha(997),
        execution_sha256=_sha(998),
        resources=replace(
            seed.resources,
            candidate_execution_sha256=_sha(998),
            accounting_artifact_sha256=_sha(996),
        ),
    )
    fake_selection = _evaluation(
        protocol,
        fake_seed,
        _fidelity(0.5),
    )
    fake_challenger = FrozenCalibrationAChallenger(
        protocol_sha256=protocol.artifact_sha256,
        seed_candidate_receipt_sha256=fake_seed.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=(
            fake_selection.receipt_sha256
        ),
        candidate=fake_seed,
        selection_evaluation=fake_selection,
        iteration_receipt_sha256s=(),
        residual_map_receipt_sha256s=(),
        proposal_receipt_sha256s=(),
        candidate_archive_receipt_sha256s=(
            fake_seed.receipt_sha256,
        ),
        selection_archive_receipt_sha256s=(
            fake_selection.receipt_sha256,
        ),
    )
    fake_guard = _evaluation(
        protocol,
        fake_seed,
        _fidelity(0.5),
        role="calibration_a_guard",
        challenger_receipt_sha256=fake_challenger.receipt_sha256,
    )
    forged = replace(
        valid,
        seed_candidate_receipt_sha256=fake_seed.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=(
            fake_selection.receipt_sha256
        ),
        final_candidate=fake_seed,
        final_selection_evaluation=fake_selection,
        frozen_challenger=fake_challenger,
        guard_evaluation=fake_guard,
        candidate_archive=(fake_seed,),
        selection_evaluation_archive=(fake_selection,),
    )

    with pytest.raises(ValueError, match="frozen protocol"):
        freeze_progressive_candidate(protocol=protocol, result=forged)


def test_freeze_recomputes_the_iteration_phase_from_active_fidelity() -> None:
    protocol = _protocol(max_iterations=1)
    seed = _seed()
    valid = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=_evaluation(
            protocol,
            seed,
            _fidelity(0.8),
        ),
        map_residual=_map,
        propose_mutations=lambda *_: (),
        build_candidate=lambda *_: pytest.fail("nothing should build"),
        evaluate_selection=lambda *_: pytest.fail(
            "nothing should be evaluated"
        ),
        evaluate_guard=lambda challenger, _: _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.9),
        ),
    )
    wrong_phase = replace(valid.iterations[0], phase="repair")
    forged = replace(valid, iterations=(wrong_phase,))

    with pytest.raises(ValueError, match="iteration phase differs"):
        freeze_progressive_candidate(protocol=protocol, result=forged)


def test_freeze_requires_policy_mandated_compaction_attempt() -> None:
    protocol = _protocol(max_iterations=1)
    seed = _seed()
    valid = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=_evaluation(
            protocol,
            seed,
            _fidelity(0.8),
        ),
        map_residual=_map,
        propose_mutations=lambda *_: (),
        build_candidate=lambda *_: pytest.fail("nothing should build"),
        evaluate_selection=lambda *_: pytest.fail(
            "nothing should be evaluated"
        ),
        evaluate_guard=lambda challenger, _: _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.9),
        ),
    )
    assert valid.frozen_challenger is not None
    challenger = replace(
        valid.frozen_challenger,
        iteration_receipt_sha256s=(),
        residual_map_receipt_sha256s=(),
    )
    assert valid.guard_evaluation is not None
    guard = replace(
        valid.guard_evaluation,
        challenger_receipt_sha256=challenger.receipt_sha256,
    )
    forged = replace(
        valid,
        frozen_challenger=challenger,
        guard_evaluation=guard,
        iterations=(),
        residual_map_archive=(),
    )

    with pytest.raises(
        ValueError,
        match="terminates before the loop policy permits",
    ):
        freeze_progressive_candidate(protocol=protocol, result=forged)


def test_freeze_rejects_an_accepted_child_omitted_from_archives() -> None:
    protocol = _protocol(compact_after_fidelity=False)
    seed = _seed()

    def propose(candidate, residual_map, phase):
        return (
            _proposal(
                candidate,
                residual_map,
                phase=phase,
                mutation_kind="widen_carrier",
                resources=_resources(650),
            ),
        )

    valid = run_progressive_compilation(
        protocol=protocol,
        seed_candidate=seed,
        seed_selection_evaluation=_evaluation(
            protocol,
            seed,
            _fidelity(2.0),
        ),
        map_residual=_map,
        propose_mutations=propose,
        build_candidate=_build,
        evaluate_selection=lambda candidate, _: _evaluation(
            protocol,
            candidate,
            _fidelity(0.8),
        ),
        evaluate_guard=lambda challenger, _: _guard_evaluation(
            protocol,
            challenger,
            _fidelity(0.9),
        ),
    )
    seed_evaluation = valid.selection_evaluation_archive[0]
    forged_iteration = replace(
        valid.iterations[0],
        proposal_receipt_sha256s=(),
        evaluation_receipt_sha256s=(
            seed_evaluation.receipt_sha256,
        ),
        accepted_evaluation_receipt_sha256=(
            seed_evaluation.receipt_sha256
        ),
    )
    forged = replace(
        valid,
        iterations=(forged_iteration,),
        proposal_archive=(),
        candidate_archive=(seed,),
        selection_evaluation_archive=(seed_evaluation,),
    )

    with pytest.raises(
        ValueError,
        match="evaluation archive membership|ineligible accepted child",
    ):
        freeze_progressive_candidate(protocol=protocol, result=forged)
