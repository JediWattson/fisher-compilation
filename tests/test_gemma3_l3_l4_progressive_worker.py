from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from fisher_graph.compiler.calibration import CalibrationBatch
from fisher_graph.compiler.progressive import (
    DevelopmentCorpus,
    FrozenCalibrationAChallenger,
    ProgressiveBehavioralTargets,
    ProgressiveCandidate,
    ProgressiveCompilationProtocol,
    ProgressiveFidelityTargets,
    ProgressiveResourceBudget,
    ProgressiveResourceFootprint,
)
from fisher_graph.gemma3_l3_l4_progressive_worker import (
    Gemma3L3L4ProgressiveWorker,
    GemmaGuardAuthorityRequiredError,
    GemmaL3L4DevelopmentObservation,
    GemmaMutationLoweringUnavailableError,
    GemmaTwoHeadFitSequence,
    LegacyRank64GemmaProgressiveExecutable,
    make_gemma_progressive_panel,
)


def _sha(index: int) -> str:
    return f"{index:064x}"


def _batch(example_id: str, offset: int) -> CalibrationBatch:
    input_ids = torch.tensor(
        [[offset, offset + 1, offset + 2, offset + 3]],
        dtype=torch.int64,
    )
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    targets = torch.tensor(
        [[offset + 1, offset + 2, offset + 3, -100]],
        dtype=torch.int64,
    )
    return CalibrationBatch(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": valid,
        },
        targets=targets,
        valid_positions=valid,
        example_ids=(example_id,),
    )


def _resources() -> ProgressiveResourceFootprint:
    return ProgressiveResourceFootprint(
        candidate_execution_sha256=_sha(20),
        accounting_artifact_sha256=_sha(21),
        parameter_scope="unit.full-model.parameters",
        compute_scope="unit.full-model.macs-per-token",
        runtime_id="unit.cpu",
        runtime_dtype="float64",
        sequence_scope_sha256=_sha(22),
        compiled_learned_parameters=20,
        retained_source_learned_parameters=50,
        support_learned_parameters=10,
        compiled_runtime_parameter_bytes=160,
        retained_source_runtime_parameter_bytes=400,
        support_runtime_parameter_bytes=80,
        compiled_logical_macs_per_token=20,
        retained_source_logical_macs_per_token=50,
        support_logical_macs_per_token=10,
        cost_complete=False,
        incomplete_cost_reasons=(
            "multi_pass_shadow_measurement",
            "native_boundary_fallback",
            "no_one_pass_serving_executable",
        ),
    )


def _targets() -> ProgressiveFidelityTargets:
    behavior = ProgressiveBehavioralTargets(
        absolute_delta_nll_per_token_max=1.0,
        source_to_candidate_kl_per_token_max=1.0,
        top1_agreement_to_source_min=0.5,
        per_prompt_p90_absolute_delta_nll_per_token_max=1.0,
        per_prompt_p10_top1_agreement_to_source_min=0.5,
    )
    return ProgressiveFidelityTargets(
        candidate_behavior=behavior,
        projection_oracle_behavior=behavior,
        carrier_oracle_behavior=behavior,
        operator_nrmse_max=2.0,
        boundary_relative_error_max=2.0,
        boundary_cosine_min=0.1,
        valid_target_coverage_min=0.1,
        worst_family_boundary_relative_error_max=2.0,
        worst_family_boundary_cosine_min=0.1,
        minimum_family_source_modal_signal_l2_norm=0.01,
        projection_full_width_relative_error_max=2.0,
        projection_full_width_cosine_min=0.1,
        worst_family_projection_relative_error_max=2.0,
        worst_family_projection_cosine_min=0.1,
        minimum_family_source_full_width_signal_l2_norm=0.01,
    )


def _panels():
    forbidden = (_sha(9),)
    return {
        "calibration_a_fit": make_gemma_progressive_panel(
            role="calibration_a_fit",
            manifest_sha256=_sha(2),
            batches=(_batch("fit.0", 10),),
            family_by_example={"fit.0": "fit-family"},
            forbidden_manifest_sha256s=forbidden,
        ),
        "calibration_a_selection": make_gemma_progressive_panel(
            role="calibration_a_selection",
            manifest_sha256=_sha(3),
            batches=(_batch("selection.0", 20),),
            family_by_example={"selection.0": "selection-family"},
            forbidden_manifest_sha256s=forbidden,
        ),
        "calibration_a_guard": make_gemma_progressive_panel(
            role="calibration_a_guard",
            manifest_sha256=_sha(4),
            batches=(_batch("guard.0", 30),),
            family_by_example={"guard.0": "guard-family"},
            forbidden_manifest_sha256s=forbidden,
        ),
    }


def _protocol(resources: ProgressiveResourceFootprint):
    panels = _panels()
    return ProgressiveCompilationProtocol(
        protocol_id="unit.gemma-progressive-worker",
        source_model_sha256=_sha(10),
        seed_candidate_artifact_sha256=_sha(19),
        seed_candidate_execution_sha256=_sha(20),
        seed_runtime_binding_sha256=_sha(23),
        seed_resource_receipt_sha256=resources.receipt_sha256,
        seed_lineage_sha256s=(_sha(24),),
        corpus=DevelopmentCorpus(
            corpus_id="unit.gemma-a-three-way",
            fit_manifest_sha256=_sha(2),
            selection_manifest_sha256=_sha(3),
            guard_manifest_sha256=_sha(4),
            fit_example_count=1,
            selection_example_count=1,
            guard_example_count=1,
            fit_family_ids=("fit-family",),
            selection_family_ids=("selection-family",),
            guard_family_ids=("guard-family",),
        ),
        development_role_binding_sha256s=(
            (
                "calibration_a_fit",
                panels["calibration_a_fit"].binding_sha256,
            ),
            (
                "calibration_a_guard",
                panels["calibration_a_guard"].binding_sha256,
            ),
            (
                "calibration_a_selection",
                panels["calibration_a_selection"].binding_sha256,
            ),
        ),
        forbidden_assessment_manifest_sha256s=(_sha(9),),
        fidelity_targets=_targets(),
        resource_budget=ProgressiveResourceBudget(
            parameter_scope="unit.full-model.parameters",
            compute_scope="unit.full-model.macs-per-token",
            runtime_id="unit.cpu",
            runtime_dtype="float64",
            sequence_scope_sha256=_sha(22),
            source_learned_parameters=100,
            source_runtime_parameter_bytes=800,
            source_logical_macs_per_token=100,
            max_total_parameter_fraction=1.0,
            max_total_parameter_byte_fraction=1.0,
            max_total_mac_fraction=1.0,
            max_retained_source_parameter_fraction=1.0,
            max_retained_source_parameter_byte_fraction=1.0,
            max_retained_source_mac_fraction=1.0,
        ),
        max_iterations=2,
    )


def _seed(resources: ProgressiveResourceFootprint) -> ProgressiveCandidate:
    return ProgressiveCandidate(
        candidate_id="gemma-seed",
        iteration=0,
        artifact_sha256=_sha(19),
        execution_sha256=_sha(20),
        runtime_binding_sha256=_sha(23),
        resources=resources,
        mutation_kind="seed",
    )


class _FakeExecutable:
    candidate_artifact_sha256 = _sha(19)
    candidate_execution_sha256 = _sha(20)
    runtime_binding_sha256 = _sha(23)

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def observe(
        self,
        example,
        *,
        collect_carrier_fisher: bool,
    ) -> GemmaL3L4DevelopmentObservation:
        self.calls.append((example.example_id, collect_carrier_fisher))
        source_logits = torch.tensor(
            [[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]],
            dtype=torch.float64,
        )
        source_modes = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5]],
            dtype=torch.float64,
        )
        source_full = torch.tensor(
            [[1.0, 0.0], [0.5, 0.5]],
            dtype=torch.float64,
        )
        residual = (
            torch.tensor(
                [[2.0, 0.0], [1.0, 0.0]],
                dtype=torch.float64,
            )
            if collect_carrier_fisher
            else None
        )
        gradient = (
            torch.tensor(
                [[1.0, 0.0], [2.0, 0.0]],
                dtype=torch.float64,
            )
            if collect_carrier_fisher
            else None
        )
        return GemmaL3L4DevelopmentObservation(
            example_id=example.example_id,
            family_id=example.family_id,
            model_inputs_sha256=example.model_inputs_sha256,
            runtime_binding_sha256=self.runtime_binding_sha256,
            source_logits=source_logits,
            candidate_logits=source_logits.clone(),
            projection_oracle_logits=source_logits.clone(),
            carrier_oracle_logits=source_logits.clone(),
            targets=torch.tensor([0, 1], dtype=torch.int64),
            source_target_modes=source_modes,
            candidate_target_modes=source_modes.clone(),
            source_full_width_delta=source_full,
            projection_full_width_delta=source_full.clone(),
            valid_target_rows=3,
            affected_target_rows=2,
            carrier_residual_rows=residual,
            carrier_loss_gradient_rows=gradient,
            complete_boundary_oracle_max_abs_logit_error=(
                0.0 if collect_carrier_fisher else None
            ),
        )


def _worker(*, authority=None, lowerer=None):
    resources = _resources()
    protocol = _protocol(resources)
    seed = _seed(resources)
    executable = _FakeExecutable()
    worker = Gemma3L3L4ProgressiveWorker(
        protocol=protocol,
        panels=_panels(),
        seed_candidate=seed,
        seed_executable=executable,
        max_residual_directions=2,
        mutation_lowerer=lowerer,
        guard_claim_authority=authority,
    )
    return worker, protocol, seed, executable


def test_worker_evaluates_selection_without_retaining_tensor_payloads() -> None:
    worker, protocol, seed, executable = _worker()

    evaluation = worker.evaluate_selection(seed, protocol.selection_view())

    assert evaluation.coverage.complete is True
    assert evaluation.coverage.supervised_token_count == 2
    assert evaluation.fidelity.candidate_behavior.top1_agreement_to_source == 1
    assert evaluation.fidelity.boundary_relative_error == 0
    assert evaluation.fidelity.projection_full_width_relative_error == 0
    assert executable.calls == [("selection.0", False)]
    assert evaluation.to_dict()["tensor_payload_exposed"] is False


def test_worker_maps_complete_carrier_fisher_residual() -> None:
    worker, protocol, seed, executable = _worker()

    residual_map = worker.map_residual(seed, protocol.fit_view())
    analysis = worker.residual_analysis(residual_map)

    assert residual_map.mapper_id == (
        "gemma3-l3-l4-h4-nll-vjp-residual-svd"
    )
    assert residual_map.targets[0].location == "layer.4.output"
    assert residual_map.targets[0].loss_coupling > 0
    assert residual_map.targets[0].jvp_gain == 0
    assert residual_map.targets[0].residual_energy_fraction == 1
    assert analysis.complete_boundary_oracle_max_abs_logit_error == 0
    assert analysis.family_row_counts == (("fit-family", 2),)
    assert torch.allclose(
        analysis.directions[0],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert executable.calls == [("fit.0", True)]


def test_worker_maps_distinct_x4_and_h4_fit_targets() -> None:
    class _TwoHeadExecutable(_FakeExecutable):
        def observe(self, example, *, collect_carrier_fisher: bool):
            observation = super().observe(
                example,
                collect_carrier_fisher=collect_carrier_fisher,
            )
            if not collect_carrier_fisher:
                return observation
            assert observation.carrier_residual_rows is not None
            assert observation.carrier_loss_gradient_rows is not None
            sequence = GemmaTwoHeadFitSequence(
                example_id=example.example_id,
                family_id=example.family_id,
                model_inputs_sha256=example.model_inputs_sha256,
                runtime_binding_sha256=self.runtime_binding_sha256,
                source_modes=torch.tensor(
                    [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
                logical_positions=torch.tensor([0, 1, 2]),
                valid_target_mask=torch.tensor([True, True, True]),
                source_eligible_mask=torch.tensor([True, True, False]),
                target_affected_mask=torch.tensor([True, True, False]),
                native_x4=torch.tensor(
                    [[0.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
                candidate_x4=torch.zeros(3, 2, dtype=torch.float64),
                native_h4=torch.tensor(
                    [[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
                candidate_h4=torch.zeros(3, 2, dtype=torch.float64),
                x4_loss_gradient=torch.tensor(
                    [[0.0, 1.0], [0.0, 2.0], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
                h4_loss_gradient=torch.tensor(
                    [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
                    dtype=torch.float64,
                ),
            )
            return replace(
                observation,
                two_head_fit_sequence=sequence,
            )

    resources = _resources()
    protocol = _protocol(resources)
    worker = Gemma3L3L4ProgressiveWorker(
        protocol=protocol,
        panels=_panels(),
        seed_candidate=_seed(resources),
        seed_executable=_TwoHeadExecutable(),
        max_residual_directions=2,
    )

    residual_map = worker.map_residual(
        _seed(resources),
        protocol.fit_view(),
    )
    analysis = worker.residual_analysis(residual_map)

    assert residual_map.mapper_id == (
        "gemma3-l3-l4-two-boundary-nll-vjp-residual-svd"
    )
    assert tuple(target.location for target in residual_map.targets) == (
        "layer.4.output",
        "layer.4.mlp.normalized_input",
    )
    assert len(analysis.fit_sequences) == 1
    assert analysis.x4_directions is not None
    torch.testing.assert_close(
        analysis.x4_directions[0],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )


def test_worker_refuses_to_invent_an_unlowered_mutation() -> None:
    worker, protocol, seed, _ = _worker()
    residual_map = worker.map_residual(seed, protocol.fit_view())

    with pytest.raises(
        GemmaMutationLoweringUnavailableError,
        match="candidate-bound",
    ):
        worker.propose_mutations(seed, residual_map, "repair")


def test_residual_analysis_is_isolated_from_callers_and_lowerers() -> None:
    class _MutatingLowerer:
        def propose(self, *, analysis, **_kwargs):
            analysis.directions[0, 0] = 0.25
            return ()

        def build(self, **_kwargs):
            raise AssertionError("build is not reached")

    worker, protocol, seed, _ = _worker(lowerer=_MutatingLowerer())
    residual_map = worker.map_residual(seed, protocol.fit_view())
    caller_copy = worker.residual_analysis(residual_map)
    caller_copy.directions[0, 0] = 0.5

    fresh = worker.residual_analysis(residual_map)
    assert fresh.directions[0, 0] == 1.0
    with pytest.raises(RuntimeError, match="changed its analysis"):
        worker.propose_mutations(seed, residual_map, "repair")


def test_worker_rejects_an_incomplete_carrier_boundary() -> None:
    class _IncompleteBoundaryExecutable(_FakeExecutable):
        def observe(self, example, *, collect_carrier_fisher: bool):
            observation = super().observe(
                example,
                collect_carrier_fisher=collect_carrier_fisher,
            )
            if not collect_carrier_fisher:
                return observation
            return replace(
                observation,
                complete_boundary_oracle_max_abs_logit_error=1.0e-3,
            )

    resources = _resources()
    protocol = _protocol(resources)
    worker = Gemma3L3L4ProgressiveWorker(
        protocol=protocol,
        panels=_panels(),
        seed_candidate=_seed(resources),
        seed_executable=_IncompleteBoundaryExecutable(),
        complete_boundary_oracle_atol=1.0e-8,
    )

    with pytest.raises(RuntimeError, match="complete-boundary oracle"):
        worker.map_residual(_seed(resources), protocol.fit_view())


def test_residual_energy_fraction_keeps_the_truncated_tail() -> None:
    class _TwoDirectionExecutable(_FakeExecutable):
        def observe(self, example, *, collect_carrier_fisher: bool):
            observation = super().observe(
                example,
                collect_carrier_fisher=collect_carrier_fisher,
            )
            if not collect_carrier_fisher:
                return observation
            return replace(
                observation,
                carrier_residual_rows=torch.tensor(
                    [[2.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float32,
                ),
                carrier_loss_gradient_rows=torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float32,
                ),
            )

    resources = _resources()
    protocol = _protocol(resources)
    worker = Gemma3L3L4ProgressiveWorker(
        protocol=protocol,
        panels=_panels(),
        seed_candidate=_seed(resources),
        seed_executable=_TwoDirectionExecutable(),
        max_residual_directions=1,
    )

    residual_map = worker.map_residual(_seed(resources), protocol.fit_view())
    analysis = worker.residual_analysis(residual_map)

    assert analysis.total_residual_energy == pytest.approx(5.0)
    assert analysis.residual_eigenvalues.tolist() == pytest.approx([4.0])
    assert residual_map.targets[0].residual_energy_fraction == pytest.approx(
        0.8
    )


def test_guard_requires_claim_first_authority_and_is_one_use() -> None:
    worker, protocol, seed, _ = _worker()
    selection = worker.evaluate_selection(seed, protocol.selection_view())
    challenger = FrozenCalibrationAChallenger(
        protocol_sha256=protocol.artifact_sha256,
        seed_candidate_receipt_sha256=seed.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=selection.receipt_sha256,
        candidate=seed,
        selection_evaluation=selection,
        iteration_receipt_sha256s=(),
        residual_map_receipt_sha256s=(),
        proposal_receipt_sha256s=(),
        candidate_archive_receipt_sha256s=(seed.receipt_sha256,),
        selection_archive_receipt_sha256s=(selection.receipt_sha256,),
    )

    with pytest.raises(GemmaGuardAuthorityRequiredError):
        worker.evaluate_guard(challenger, protocol.guard_view())

    class _Authority:
        def __init__(self) -> None:
            self.calls = 0

        def claim(self, **_kwargs) -> str:
            self.calls += 1
            return _sha(90)

    authority = _Authority()
    claimed_worker, claimed_protocol, claimed_seed, _ = _worker(
        authority=authority
    )
    claimed_selection = claimed_worker.evaluate_selection(
        claimed_seed,
        claimed_protocol.selection_view(),
    )
    claimed = replace(
        challenger,
        protocol_sha256=claimed_protocol.artifact_sha256,
        seed_candidate_receipt_sha256=claimed_seed.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=(
            claimed_selection.receipt_sha256
        ),
        candidate=claimed_seed,
        selection_evaluation=claimed_selection,
        candidate_archive_receipt_sha256s=(claimed_seed.receipt_sha256,),
        selection_archive_receipt_sha256s=(
            claimed_selection.receipt_sha256,
        ),
    )

    drifted_seed = replace(
        claimed_seed,
        resources=replace(
            claimed_seed.resources,
            accounting_artifact_sha256=_sha(91),
        ),
    )
    drifted_selection = replace(
        claimed_selection,
        candidate_receipt_sha256=drifted_seed.receipt_sha256,
        evaluation_artifact_sha256=_sha(92),
        resources=drifted_seed.resources,
    )
    drifted_challenger = replace(
        claimed,
        seed_candidate_receipt_sha256=drifted_seed.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=(
            drifted_selection.receipt_sha256
        ),
        candidate=drifted_seed,
        selection_evaluation=drifted_selection,
        candidate_archive_receipt_sha256s=(
            drifted_seed.receipt_sha256,
        ),
        selection_archive_receipt_sha256s=(
            drifted_selection.receipt_sha256,
        ),
    )
    with pytest.raises(RuntimeError, match="candidate receipt"):
        claimed_worker.evaluate_guard(
            drifted_challenger,
            claimed_protocol.guard_view(),
        )
    assert authority.calls == 0

    guard = claimed_worker.evaluate_guard(
        claimed,
        claimed_protocol.guard_view(),
    )
    assert guard.challenger_receipt_sha256 == claimed.receipt_sha256
    assert authority.calls == 1
    with pytest.raises(GemmaGuardAuthorityRequiredError, match="already"):
        claimed_worker.evaluate_guard(
            claimed,
            claimed_protocol.guard_view(),
        )


def test_selection_only_worker_has_no_guard_capability() -> None:
    resources = _resources()
    protocol = _protocol(resources)
    seed = _seed(resources)
    panels = _panels()
    panels.pop("calibration_a_guard")
    worker = Gemma3L3L4ProgressiveWorker(
        protocol=protocol,
        panels=panels,
        seed_candidate=seed,
        seed_executable=_FakeExecutable(),
        selection_only=True,
    )
    selection = worker.evaluate_selection(
        seed,
        protocol.selection_view(),
    )
    challenger = FrozenCalibrationAChallenger(
        protocol_sha256=protocol.artifact_sha256,
        seed_candidate_receipt_sha256=seed.receipt_sha256,
        seed_selection_evaluation_receipt_sha256=(
            selection.receipt_sha256
        ),
        candidate=seed,
        selection_evaluation=selection,
        iteration_receipt_sha256s=(),
        residual_map_receipt_sha256s=(),
        proposal_receipt_sha256s=(),
        candidate_archive_receipt_sha256s=(seed.receipt_sha256,),
        selection_archive_receipt_sha256s=(
            selection.receipt_sha256,
        ),
    )

    with pytest.raises(
        GemmaGuardAuthorityRequiredError,
        match="selection-only",
    ):
        worker.evaluate_guard(challenger, protocol.guard_view())
    with pytest.raises(ValueError, match="cannot receive guard"):
        Gemma3L3L4ProgressiveWorker(
            protocol=protocol,
            panels=panels,
            seed_candidate=seed,
            seed_executable=_FakeExecutable(),
            guard_claim_authority=object(),
            selection_only=True,
        )


def test_panel_rejects_forbidden_manifest_and_cross_role_input_replay() -> None:
    with pytest.raises(ValueError, match="forbidden assessment"):
        make_gemma_progressive_panel(
            role="calibration_a_fit",
            manifest_sha256=_sha(9),
            batches=(_batch("fit.0", 10),),
            family_by_example={"fit.0": "fit-family"},
            forbidden_manifest_sha256s=(_sha(9),),
        )

    resources = _resources()
    protocol = _protocol(resources)
    panels = _panels()
    replay_batch = _batch("selection.0", 10)
    panels["calibration_a_selection"] = make_gemma_progressive_panel(
        role="calibration_a_selection",
        manifest_sha256=_sha(3),
        batches=(replay_batch,),
        family_by_example={"selection.0": "selection-family"},
        forbidden_manifest_sha256s=(_sha(9),),
    )
    with pytest.raises(ValueError, match="contents differ"):
        Gemma3L3L4ProgressiveWorker(
            protocol=protocol,
            panels=panels,
            seed_candidate=_seed(resources),
            seed_executable=_FakeExecutable(),
        )


def test_panel_and_candidate_receipts_reject_post_binding_drift() -> None:
    worker, protocol, seed, _ = _worker()
    fit_example = worker._panels["calibration_a_fit"].examples[0]
    fit_example.batch.model_inputs["input_ids"][0, 0] += 1

    with pytest.raises(ValueError, match="changed after binding"):
        worker.map_residual(seed, protocol.fit_view())

    clean_worker, clean_protocol, clean_seed, _ = _worker()
    drifted_resources = replace(
        clean_seed.resources,
        accounting_artifact_sha256=_sha(99),
    )
    drifted_candidate = replace(
        clean_seed,
        resources=drifted_resources,
    )
    with pytest.raises(RuntimeError, match="candidate receipt"):
        clean_worker.evaluate_selection(
            drifted_candidate,
            clean_protocol.selection_view(),
        )


def test_legacy_measurement_runtime_cannot_claim_complete_costs() -> None:
    class _LegacyIdentityStub(LegacyRank64GemmaProgressiveExecutable):
        @property
        def candidate_artifact_sha256(self) -> str:
            return _sha(19)

        @property
        def candidate_execution_sha256(self) -> str:
            return _sha(20)

        @property
        def runtime_binding_sha256(self) -> str:
            return _sha(23)

        def observe(self, *_args, **_kwargs):
            raise AssertionError("construction must reject before execution")

    complete = replace(
        _resources(),
        cost_complete=True,
        incomplete_cost_reasons=(),
    )
    protocol = _protocol(complete)

    with pytest.raises(ValueError, match="incomplete resource accounting"):
        Gemma3L3L4ProgressiveWorker(
            protocol=protocol,
            panels=_panels(),
            seed_candidate=_seed(complete),
            seed_executable=object.__new__(_LegacyIdentityStub),
        )
