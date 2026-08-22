from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import pytest
import torch

import fisher_graph.gemma3_attention_transport_experiment_core as core
from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportCapture,
    AttentionTransportFitConfig,
    AttentionTransportRowBatch,
    ChartConditionedRouterConfig,
    EXACT_SPARSE_ORACLE_SOLVER,
    HELDOUT_ORACLE_COEFFICIENT_RIDGE,
    attention_transport_row_batch_from_capture,
)
from fisher_graph.gemma3_attention_transport_experiment_core import (
    AttentionTransportCandidateConfig,
    AttentionTransportSelectionGates,
    evaluate_attention_transport_selection_on_c,
    fit_attention_transport_candidates_on_a,
    run_split_locked_attention_transport_experiment,
    select_attention_transport_candidate_on_b,
)
from fisher_graph.gemma3_attention_transport_runtime import (
    AttentionTransportCapturedInvocation,
    AttentionTransportOutputNormSpec,
    sample_attention_transport_captured_invocation,
)


def _receipt(role: str, batch: int) -> str:
    return hashlib.sha256(f"transport-{role}-{batch}".encode()).hexdigest()


def _capture_record(
    role: str,
    batch: int,
    invocation: int,
    *,
    seed: int,
) -> AttentionTransportCapturedInvocation:
    hidden = 4
    heads = 2
    length = 7
    random = torch.Generator().manual_seed(seed)
    normalized = torch.randn(
        1,
        length,
        hidden,
        dtype=torch.float64,
        generator=random,
    )
    probabilities = torch.eye(length, dtype=torch.float64)[
        None,
        None,
    ].repeat(1, heads, 1, 1)
    value_weight = torch.zeros(2, hidden, dtype=torch.float64)
    value_weight[0, 0] = 1.0
    value_weight[1, 1] = 1.0
    output_weight = torch.zeros(hidden, 2, dtype=torch.float64)
    if invocation == 0:
        output_weight[0, 0] = 1.0
        output_weight[1, 1] = 1.0
    else:
        output_weight[0, 0] = 1.25
        output_weight[1, 1] = -0.75
    head_contexts = torch.einsum(
        "baqs,bsh->bqah",
        probabilities,
        normalized,
    )
    head_operators = torch.stack(
        tuple(
            output_weight[:, head : head + 1]
            @ value_weight[head : head + 1]
            for head in range(heads)
        )
    )
    outputs = torch.einsum("aoh,bqah->bqo", head_operators, head_contexts)
    gradients = torch.randn(
        outputs.shape,
        dtype=torch.float64,
        generator=random,
    )
    valid = torch.ones(1, length, dtype=torch.bool)
    capture = AttentionTransportCapture(
        invocation_id=f"mutation.{invocation}",
        normalized_inputs=normalized,
        attention_probabilities=probabilities,
        operator_outputs=outputs,
        operator_output_score_gradients=gradients,
        query_valid_mask=valid,
        key_valid_mask=valid.clone(),
        value_projection_weight=value_weight,
        output_projection_weight=output_weight,
        query_head_count=heads,
        key_value_head_count=heads,
        head_dimension=1,
    )
    norm = AttentionTransportOutputNormSpec(
        invocation_id=capture.invocation_id,
        weight=torch.zeros(hidden, dtype=torch.float64),
        epsilon=1.0e-6,
        scale_parameterization="unit_offset",
        compute_dtype="float32",
    )
    return AttentionTransportCapturedInvocation(
        role=role,
        base_capture_receipt_sha256=_receipt(role, batch),
        capture=capture,
        postnorm_targets=norm.apply(capture.valid_operator_outputs),
        output_norm=norm,
    )


def _split(
    role: str,
    *,
    batches: int,
    seed: int,
) -> tuple[AttentionTransportCapturedInvocation, ...]:
    return tuple(
        _capture_record(
            role,
            batch,
            invocation,
            seed=seed + 10 * batch + invocation,
        )
        for batch in range(batches)
        for invocation in range(2)
    )


def _candidate() -> AttentionTransportCandidateConfig:
    return AttentionTransportCandidateConfig(
        candidate_id="shared-k2-r1-top2",
        bank=AttentionTransportFitConfig(
            generator_count=2,
            generator_rank=1,
            steps=0,
            learning_rate=1.0e-3,
            euclidean_floor=0.1,
            coefficient_ridge=0.0,
        ),
        chart_count=1,
        chart_rank=1,
        chart_temperature_scale=1.0,
        chart_lloyd_iterations=2,
        router=ChartConditionedRouterConfig(
            active_count=2,
            steps=0,
            learning_rate=1.0e-3,
            euclidean_floor=0.1,
            oracle_ridge=1.0e-10,
        ),
    )


def _passing_gates() -> AttentionTransportSelectionGates:
    return AttentionTransportSelectionGates(
        max_relative_fisher_loss=1.0e-10,
        max_oracle_router_fisher_gap=1.0e-10,
        max_postnorm_normalized_rmse=1.0e-5,
        min_postnorm_cosine=0.99999,
        max_sparse_probability_renormalization_error=1.0e-12,
        max_effective_support_fraction=1.0,
        minimum_aggregate_rows=14,
        require_parameter_compression=False,
    )


def _permissive_gates() -> AttentionTransportSelectionGates:
    return AttentionTransportSelectionGates(
        max_relative_fisher_loss=1.0e6,
        max_oracle_router_fisher_gap=1.0e6,
        max_postnorm_normalized_rmse=1.0e6,
        min_postnorm_cosine=-1.0,
        max_sparse_probability_renormalization_error=1.0e-12,
        max_effective_support_fraction=1.0,
        minimum_aggregate_rows=14,
        require_parameter_compression=False,
    )


def _small_candidate() -> AttentionTransportCandidateConfig:
    base = _candidate()
    return replace(
        base,
        candidate_id="small-k1-r1-top1",
        bank=replace(base.bank, generator_count=1),
        router=replace(base.router, active_count=1),
    )


@dataclass(frozen=True, slots=True)
class _CompactRecord:
    role: str
    base_capture_receipt_sha256: str
    capture: AttentionTransportRowBatch
    postnorm_targets: torch.Tensor
    output_norm: AttentionTransportOutputNormSpec


def _compact(
    records: tuple[AttentionTransportCapturedInvocation, ...],
) -> tuple[_CompactRecord, ...]:
    return tuple(
        _CompactRecord(
            role=record.role,
            base_capture_receipt_sha256=record.base_capture_receipt_sha256,
            capture=attention_transport_row_batch_from_capture(record.capture),
            postnorm_targets=record.postnorm_targets,
            output_norm=record.output_norm,
        )
        for record in records
    )


def test_a_fit_caches_identical_bank_and_atlas_specs_without_sharing_routers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _split("A", batches=2, seed=100_000)
    base = _candidate()
    router_variant = replace(
        base,
        candidate_id="shared-k2-r1-top2-router-step1",
        router=replace(base.router, steps=1),
    )
    calls = {"bank": 0, "atlas": 0, "router": 0}
    real_bank_fit = core.fit_fisher_attention_transport_generator_bank
    real_atlas_fit = core.fit_frozen_attention_chart_atlas
    real_router_fit = core.fit_chart_conditioned_attention_transport_router

    def counted_bank_fit(*args: object, **kwargs: object) -> object:
        calls["bank"] += 1
        return real_bank_fit(*args, **kwargs)

    def counted_atlas_fit(*args: object, **kwargs: object) -> object:
        calls["atlas"] += 1
        return real_atlas_fit(*args, **kwargs)

    def counted_router_fit(*args: object, **kwargs: object) -> object:
        calls["router"] += 1
        return real_router_fit(*args, **kwargs)

    monkeypatch.setattr(
        core,
        "fit_fisher_attention_transport_generator_bank",
        counted_bank_fit,
    )
    monkeypatch.setattr(
        core,
        "fit_frozen_attention_chart_atlas",
        counted_atlas_fit,
    )
    monkeypatch.setattr(
        core,
        "fit_chart_conditioned_attention_transport_router",
        counted_router_fit,
    )
    cached = fit_attention_transport_candidates_on_a(
        records,
        (router_variant, base),
    )

    assert calls == {"bank": 1, "atlas": 1, "router": 4}
    first, second = cached.candidates
    assert first.bank_fit is second.bank_fit
    assert first.atlas is second.atlas
    assert all(
        left is not right
        for left, right in zip(first.routers, second.routers, strict=True)
    )
    assert first.artifact_sha256 != second.artifact_sha256
    assert first.bank.state_dict()["contains_teacher_outputs_or_gradients"] is False
    assert first.atlas.state_dict()["contains_teacher_outputs_or_gradients"] is False

    monkeypatch.undo()
    uncached = {
        config.candidate_id: fit_attention_transport_candidates_on_a(
            records,
            (config,),
        ).candidates[0]
        for config in (base, router_variant)
    }
    assert {
        value.config.candidate_id: value.artifact_sha256
        for value in cached.candidates
    } == {
        key: value.artifact_sha256 for key, value in uncached.items()
    }


def test_a_fit_b_lock_and_c_confirmation_report_weighted_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a_records = _split("A", batches=2, seed=101_000)
    b_records = _split("B", batches=1, seed=102_000)
    c_records = _split("C", batches=1, seed=103_000)
    development = fit_attention_transport_candidates_on_a(
        a_records,
        (_candidate(),),
    )
    candidate = development.candidates[0]
    assert candidate.bank_fit.final_fit_metrics.row_count == sum(
        record.capture.row_count for record in a_records
    )
    assert candidate.resources.chart_parameters == (
        candidate.atlas.stored_parameter_count
    )

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("B/C evaluation attempted to fit an artifact")

    monkeypatch.setattr(
        core,
        "fit_fisher_attention_transport_generator_bank",
        forbidden_fit,
    )
    monkeypatch.setattr(
        core,
        "fit_frozen_attention_chart_atlas",
        forbidden_fit,
    )
    monkeypatch.setattr(
        core,
        "fit_chart_conditioned_attention_transport_router",
        forbidden_fit,
    )
    selection = select_attention_transport_candidate_on_b(
        development,
        b_records,
        gates=_passing_gates(),
    )
    assert selection.selected_lock is not None
    assert selection.c_open_allowed
    assert len(selection.artifact_sha256) == 64
    lock = selection.selected_lock
    assert lock.selected_candidate_artifact_sha256 == candidate.artifact_sha256
    assert lock.gates_artifact_sha256 == _passing_gates().artifact_sha256
    assert lock.b_decision.passed

    confirmation = evaluate_attention_transport_selection_on_c(lock, c_records)
    metrics = confirmation.c_evaluation
    assert confirmation.c_receipt_sha256s == (_receipt("C", 0),)
    assert not set(confirmation.c_receipt_sha256s) & set(
        (*lock.a_receipt_sha256s, *lock.b_receipt_sha256s)
    )
    assert metrics.invocation_count == 2
    assert metrics.row_count == sum(
        record.capture.row_count for record in c_records
    )
    expected_fisher = sum(
        value.output_metrics.fisher_loss * value.output_metrics.row_count
        for value in metrics.per_invocation
    ) / metrics.row_count
    expected_zero = sum(
        value.zero_predictor_fisher_loss * value.output_metrics.row_count
        for value in metrics.per_invocation
    ) / metrics.row_count
    assert metrics.fisher_loss == pytest.approx(expected_fisher)
    assert metrics.zero_predictor_fisher_loss == pytest.approx(expected_zero)
    assert metrics.relative_fisher_loss == pytest.approx(
        expected_fisher / expected_zero
    )
    assert metrics.postnorm_normalized_rmse < 1.0e-5
    assert metrics.postnorm_cosine > 0.99999
    assert metrics.per_chart_active_count_minimum == 2
    assert metrics.per_chart_active_count_maximum == 2
    assert metrics.oracle_solver_kind == EXACT_SPARSE_ORACLE_SOLVER
    assert metrics.oracle_active_count_limit == 2
    assert (
        metrics.oracle_coefficient_ridge
        == HELDOUT_ORACLE_COEFFICIENT_RIDGE
        == 0.0
    )
    assert metrics.effective_support_count_maximum == 2
    assert metrics.effective_support_fraction == pytest.approx(1.0)
    assert metrics.sparse_probability_renormalization_max_error < 1.0e-12
    assert not metrics.resources.logical_speedup_claimed
    assert confirmation.scientific_status.operator_boundary_only
    assert confirmation.scientific_status.post_rmsnorm_diagnostic_measured
    assert not confirmation.scientific_status.behavioral_nll_measured
    assert not confirmation.scientific_status.behavioral_quality_claimed
    assert not confirmation.scientific_status.logical_speedup_claimed

    with pytest.raises(ValueError, match="selection lock is invalid"):
        replace(lock, selected_candidate_artifact_sha256="0" * 64)


def test_gate_rejects_negative_oracle_gap_and_global_support_over_budget() -> None:
    candidate_config = replace(
        _candidate(),
        candidate_id="shared-k2-r1-top1",
        router=replace(_candidate().router, active_count=1),
    )
    development = fit_attention_transport_candidates_on_a(
        _split("A", batches=2, seed=103_100),
        (candidate_config,),
    )
    selection = select_attention_transport_candidate_on_b(
        development,
        _split("B", batches=1, seed=103_200),
        gates=_permissive_gates(),
    )
    candidate = development.candidates[0]
    evaluation = selection.decisions[0].evaluation

    negative_gap = replace(
        evaluation,
        oracle_router_fisher_gap=-2.0e-10,
    )
    negative_decision = core._gate_candidate(
        candidate,
        negative_gap,
        _permissive_gates(),
    )
    assert not negative_decision.oracle_gap_pass
    assert not negative_decision.passed

    union_over_budget = replace(
        evaluation,
        effective_support_count_minimum=2,
        effective_support_count_mean=2.0,
        effective_support_count_maximum=2,
        effective_support_fraction=1.0,
    )
    union_decision = core._gate_candidate(
        candidate,
        union_over_budget,
        _permissive_gates(),
    )
    assert not union_decision.global_support_budget_pass
    assert not union_decision.passed


def test_no_b_candidate_pass_leaves_lazy_c_supplier_unopened() -> None:
    a_records = _split("A", batches=2, seed=104_000)
    b_records = _split("B", batches=1, seed=105_000)
    opened = False

    def c_supplier() -> tuple[AttentionTransportCapturedInvocation, ...]:
        nonlocal opened
        opened = True
        raise AssertionError("C must remain unopened after a no-pass B panel")

    underpowered_gate = replace(
        _passing_gates(),
        minimum_aggregate_rows=15,
    )
    outcome = run_split_locked_attention_transport_experiment(
        a_records,
        b_records,
        c_supplier,
        candidate_configs=(_candidate(),),
        gates=underpowered_gate,
    )

    assert not opened
    assert not outcome.c_opened
    assert outcome.confirmation is None
    assert outcome.selection.selected_lock is None
    assert not outcome.selection.c_open_allowed
    assert len(outcome.selection.artifact_sha256) == 64
    assert outcome.selection.decisions[0].effective_support_pass
    assert not outcome.selection.decisions[0].minimum_rows_pass


def test_confirmation_rejects_wrong_role_or_reused_split_receipts() -> None:
    a_records = _split("A", batches=2, seed=106_000)
    b_records = _split("B", batches=1, seed=107_000)
    development = fit_attention_transport_candidates_on_a(
        a_records,
        (_candidate(),),
    )
    selection = select_attention_transport_candidate_on_b(
        development,
        b_records,
        gates=_passing_gates(),
    )
    assert selection.selected_lock is not None

    with pytest.raises(ValueError, match="split C contains a record with role"):
        evaluate_attention_transport_selection_on_c(
            selection.selected_lock,
            b_records,
        )

    reused = tuple(
        replace(
            record,
            role="C",
            base_capture_receipt_sha256=_receipt("A", 0),
        )
        for record in _split("C", batches=1, seed=108_000)
    )
    with pytest.raises(ValueError, match="receipts must be disjoint"):
        evaluate_attention_transport_selection_on_c(
            selection.selected_lock,
            reused,
        )


def test_split_core_accepts_compact_sufficient_statistic_row_batches() -> None:
    a_records = _compact(_split("A", batches=2, seed=109_000))
    b_records = _compact(_split("B", batches=1, seed=110_000))
    c_records = _compact(_split("C", batches=1, seed=111_000))

    outcome = run_split_locked_attention_transport_experiment(
        a_records,
        b_records,
        lambda: c_records,
        candidate_configs=(_candidate(),),
        gates=_passing_gates(),
    )

    assert outcome.c_opened
    assert outcome.confirmation is not None
    assert outcome.confirmation.c_evaluation.row_count == 14


def test_split_open_detects_mutated_shared_head_operator_catalog() -> None:
    source = _split("A", batches=2, seed=104_500)
    catalogs: dict[str, object] = {}
    sampled = []
    for record in source:
        catalog = catalogs.setdefault(
            record.capture.invocation_id,
            record.head_operator_catalog,
        )
        rebound = replace(record, head_operator_catalog=catalog)
        sampled.append(
            sample_attention_transport_captured_invocation(
                rebound,
                maximum_rows=3,
            )
        )

    first_catalog = source[0].head_operator_catalog
    assert first_catalog is not None
    first_catalog.head_operators[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="catalog changed"):
        fit_attention_transport_candidates_on_a(
            tuple(sampled),
            (_candidate(),),
        )


def test_b_selection_prioritizes_relative_fisher_before_parameter_count() -> None:
    development = fit_attention_transport_candidates_on_a(
        _split("A", batches=2, seed=112_000),
        (_small_candidate(), _candidate()),
    )
    selection = select_attention_transport_candidate_on_b(
        development,
        _split("B", batches=1, seed=113_000),
        gates=_permissive_gates(),
    )

    assert all(value.passed for value in selection.decisions)
    assert selection.selected_lock is not None
    selected = selection.selected_lock.selected_candidate
    small = next(
        value
        for value in development.candidates
        if value.config.candidate_id == "small-k1-r1-top1"
    )
    assert selected.config.candidate_id == "shared-k2-r1-top2"
    assert selected.resources.compiled_parameters > small.resources.compiled_parameters
