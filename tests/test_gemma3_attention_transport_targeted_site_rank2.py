from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import hashlib
import math

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
)
from fisher_graph.gemma3_attention_transport_global_rank1 import (
    GLOBAL_ATTENTION_INVOCATION_IDS,
    GlobalRankOneAttentionTransportFitConfig,
)
from fisher_graph.gemma3_attention_transport_global_site_rank1 import (
    GlobalSiteRankOneAttentionTransportFitConfig,
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    evaluate_global_site_rank_one_coupling_diagnostic,
    fit_global_site_rank_one_attention_transport_generator_bank,
)
from fisher_graph.gemma3_attention_transport_hierarchical_edge import (
    AttentionTransportHierarchyLayout,
    HierarchicalFisherAttentionTransportGeneratorBank,
    HierarchicalSparseAttentionTransportFitConfig,
)
from fisher_graph.gemma3_attention_transport_sparse_v2 import (
    SparseAwareAttentionTransportFitConfig,
)
from fisher_graph.gemma3_attention_transport_targeted_site_rank2 import (
    TARGETED_SITE_RANK_PROFILE,
    TARGETED_SITE_RANK_TWO_INVOCATION_IDS,
    TargetedSiteRankTwoAttentionTransportFitConfig,
    TargetedSiteRankTwoFisherAttentionTransportGeneratorBank,
    _deterministic_orthogonal_second_right_vectors,
    _initialize_targeted_site_rank_two_bank,
    _targeted_bank_with_factors,
    evaluate_all_layer_attention_transport_diagnostic,
    evaluate_targeted_site_rank_two_coupling_diagnostic,
    fit_targeted_site_rank_two_attention_transport_generator_bank,
    targeted_site_rank_two_fisher_attention_transport_generator_bank_from_state_dict,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _layout() -> AttentionTransportHierarchyLayout:
    return AttentionTransportHierarchyLayout.gemma3_270m_attention(
        sorted(f"mutation.{2 * index + 1}" for index in range(18))
    )


def _hierarchical_bank(
    *,
    generator_count: int = 1,
    hidden_width: int = 2,
    generator_rank: int = 1,
    query_head_count: int = 1,
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    layout = _layout()
    factors = torch.zeros(
        generator_count,
        hidden_width,
        generator_rank,
        dtype=torch.float64,
    )
    factors[:, :generator_rank, :] = torch.eye(
        generator_rank,
        dtype=torch.float64,
    )[None]
    head_ids = tuple(
        f"{invocation_id}/query-head.{head_index}"
        for invocation_id in layout.invocation_ids
        for head_index in range(query_head_count)
    )
    return HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=factors,
        singular_scales=torch.ones(
            generator_count,
            generator_rank,
            dtype=torch.float64,
        ),
        right_factors=factors.clone(),
        head_coefficients=torch.ones(
            len(head_ids),
            generator_count,
            dtype=torch.float64,
        ),
        head_ids=head_ids,
        hierarchy=layout,
        block_scale_contrasts=torch.zeros(
            2,
            generator_count,
            generator_rank,
            dtype=torch.float64,
        ),
        site_scale_contrasts=torch.zeros(
            5,
            generator_count,
            generator_rank,
            dtype=torch.float64,
        ),
    )


def _site_rank_one_bank(
    *,
    hierarchical: HierarchicalFisherAttentionTransportGeneratorBank | None = None,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
) -> GlobalSiteRankOneFisherAttentionTransportGeneratorBank:
    parent = _hierarchical_bank() if hierarchical is None else hierarchical
    shape = (3, parent.generator_count, parent.hidden_width)
    residual_left = (
        torch.zeros(shape, dtype=torch.float64) if left is None else left
    )
    if right is None:
        residual_right = torch.zeros(shape, dtype=torch.float64)
        residual_right[:, :, 0] = 1.0
    else:
        residual_right = right
    return GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
        left_factors=parent.left_factors,
        singular_scales=parent.singular_scales,
        right_factors=parent.right_factors,
        head_coefficients=parent.head_coefficients,
        head_ids=parent.head_ids,
        hierarchy=parent.hierarchy,
        block_scale_contrasts=parent.block_scale_contrasts,
        site_scale_contrasts=parent.site_scale_contrasts,
        signed_feature_floor=parent.signed_feature_floor,
        global_site_residual_left_vectors=residual_left,
        global_site_residual_right_vectors=residual_right,
        shared_rank_one_bank_sha256=_sha("shared-rank-one"),
    )


def _capture(
    invocation_id: str,
    contexts: torch.Tensor,
    operator: torch.Tensor,
) -> AttentionTransportRowBatch:
    outputs = torch.einsum("aoh,nah->no", operator, contexts)
    return AttentionTransportRowBatch(
        invocation_id=invocation_id,
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.ones_like(outputs),
        chart_inputs=contexts[:, 0, :].clone(),
        head_operators=operator,
        native_v_o_parameter_count=2 * contexts.shape[-1] ** 2,
    )


def _tiny_captures(
    *,
    site_specific: bool = True,
) -> tuple[AttentionTransportRowBatch, ...]:
    contexts = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[1.0, -1.0]],
            [[-0.5, 2.0]],
        ],
        dtype=torch.float64,
    )
    local = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.15]]],
        dtype=torch.float64,
    )
    shared_global = torch.tensor(
        [[[1.0, 0.7], [-0.4, 0.15]]],
        dtype=torch.float64,
    )
    independent = {
        "mutation.11": torch.tensor(
            [[[1.0, 1.0], [0.0, 0.15]]],
            dtype=torch.float64,
        ),
        "mutation.23": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.15]]],
            dtype=torch.float64,
        ),
        "mutation.35": torch.tensor(
            [[[1.0, -1.0], [0.5, 0.15]]],
            dtype=torch.float64,
        ),
    }
    captures = []
    for invocation_id in _layout().invocation_ids:
        operator = local
        if invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS:
            operator = (
                independent[invocation_id]
                if site_specific
                else shared_global
            )
        captures.append(_capture(invocation_id, contexts, operator))
    return tuple(captures)


def _site_fit_config(
    *,
    learning_rate: float = 2.0e-2,
) -> GlobalSiteRankOneAttentionTransportFitConfig:
    sparse = SparseAwareAttentionTransportFitConfig(
        generator_count=1,
        generator_rank=1,
        active_count=1,
        dense_warm_start_steps=0,
        alternating_rounds=1,
        steps_per_round=1,
        learning_rate=learning_rate,
        euclidean_floor=0.1,
        oracle_coefficient_ridge=1.0e-8,
        dense_head_coefficient_ridge=1.0e-8,
        gradient_clip_norm=10.0,
        gauge_floor=1.0e-12,
    )
    shared = GlobalRankOneAttentionTransportFitConfig(
        hierarchical=HierarchicalSparseAttentionTransportFitConfig(
            shared=sparse,
            alternating_rounds=4,
        ),
        alternating_rounds=1,
        steps_per_round=2,
        learning_rate=learning_rate,
    )
    return GlobalSiteRankOneAttentionTransportFitConfig(
        shared_rank_one=shared,
        alternating_rounds=1,
        steps_per_site_per_round=2,
        learning_rate=learning_rate,
    )


def _target_config(
    site_config: GlobalSiteRankOneAttentionTransportFitConfig,
    *,
    rounds: int = 2,
) -> TargetedSiteRankTwoAttentionTransportFitConfig:
    return TargetedSiteRankTwoAttentionTransportFitConfig(
        site_rank_one=site_config,
        alternating_rounds=rounds,
        steps_per_target_site_per_round=2,
        second_component_warmup_steps=1,
        learning_rate=site_config.learning_rate,
    )


def _assert_monotone(history: tuple[float, ...]) -> None:
    assert all(
        right <= left + 1.0e-12
        for left, right in zip(history, history[1:])
    )


@pytest.fixture(scope="module")
def _fitted() -> tuple[
    tuple[AttentionTransportRowBatch, ...],
    GlobalSiteRankOneAttentionTransportFitConfig,
    object,
    object,
]:
    captures = _tiny_captures()
    site_config = _site_fit_config()
    parent = fit_global_site_rank_one_attention_transport_generator_bank(
        captures,
        config=site_config,
    )
    targeted = fit_targeted_site_rank_two_attention_transport_generator_bank(
        captures,
        config=_target_config(site_config),
        site_rank_one_fit=parent,
    )
    return captures, site_config, parent, targeted


def test_exact_zero_output_nesting_equation_and_site_isolation() -> None:
    hierarchical = _hierarchical_bank(hidden_width=3)
    first_left = torch.tensor(
        [
            [[1.0, -0.5, 0.25]],
            [[-0.75, 0.3, 1.2]],
            [[0.4, -0.2, 0.1]],
        ],
        dtype=torch.float64,
    )
    first_right = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    parent = _site_rank_one_bank(
        hierarchical=hierarchical,
        left=first_left,
        right=first_right,
    )
    initial = _initialize_targeted_site_rank_two_bank(
        parent,
        gauge_floor=parent.gauge_floor,
    )
    contexts = torch.tensor(
        [[[1.0, 2.0, -0.5]], [[-1.5, 0.25, 3.0]]],
        dtype=torch.float64,
    )

    assert TARGETED_SITE_RANK_PROFILE == (
        ("mutation.11", 2),
        ("mutation.23", 2),
        ("mutation.35", 1),
    )
    assert torch.count_nonzero(initial.second_residual_left_vectors) == 0
    for invocation_id in parent.hierarchy.invocation_ids:
        assert torch.equal(
            initial.candidate_outputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
            parent.candidate_outputs_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
        )

    target_first_left = torch.stack((first_left[0], first_left[1]))
    target_first_right = torch.stack((first_right[0], first_right[1]))
    second_left = initial.second_residual_left_vectors.clone()
    second_right = initial.second_residual_right_vectors.clone()
    second_left[0, 0] = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64)
    changed = _targeted_bank_with_factors(
        initial,
        target_first_left,
        target_first_right,
        second_left,
        second_right,
    )
    expected_second = (
        torch.einsum(
            "nkh,kh->nk",
            changed.contextual_inputs_from_head_contexts(
                invocation_id="mutation.11",
                head_contexts=contexts,
            ),
            changed.second_residual_right_vectors[0],
        )[:, :, None]
        * changed.second_residual_left_vectors[0][None]
    )
    assert torch.allclose(
        changed.second_rank_one_candidate_residuals_from_head_contexts(
            invocation_id="mutation.11",
            head_contexts=contexts,
        ),
        expected_second,
    )
    for invocation_id in parent.hierarchy.invocation_ids:
        before = parent.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        after = changed.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        if invocation_id == "mutation.11":
            assert not torch.equal(after, before)
        else:
            assert torch.equal(after, before)


def test_targeted_update_preserves_mutation_35_factors_bit_exact() -> None:
    """Rebuilding an accepted target must not recanonicalize its frozen peer."""

    hierarchical = _hierarchical_bank(hidden_width=2)
    parent = _site_rank_one_bank(
        hierarchical=hierarchical,
        left=torch.tensor(
            [
                [[1.0, 0.25]],
                [[-0.5, 0.75]],
                [[0.3, -0.4]],
            ],
            dtype=torch.float64,
        ),
        right=torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
            ],
            dtype=torch.float64,
        ),
    )
    initial = _initialize_targeted_site_rank_two_bank(
        parent,
        gauge_floor=parent.gauge_floor,
    )
    target_first_left = torch.stack(
        tuple(
            initial.global_site_residual_left_vectors[
                GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
            ]
            for invocation_id in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        )
    )
    target_first_right = torch.stack(
        tuple(
            initial.global_site_residual_right_vectors[
                GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
            ]
            for invocation_id in TARGETED_SITE_RANK_TWO_INVOCATION_IDS
        )
    )
    # Joint Adam checkpoints are not yet in canonical gauge.  A negative pivot
    # and non-unit norm force the accepted target slice through canonicalization.
    target_first_left[0, 0] = torch.tensor(
        [0.8, -0.2],
        dtype=torch.float64,
    )
    target_first_right[0, 0] = torch.tensor(
        [-2.0, 0.25],
        dtype=torch.float64,
    )
    second_left = initial.second_residual_left_vectors.clone()
    second_right = initial.second_residual_right_vectors.clone()
    second_left[0, 0] = torch.tensor([0.1, 0.2], dtype=torch.float64)

    updated = _targeted_bank_with_factors(
        initial,
        target_first_left,
        target_first_right,
        second_left,
        second_right,
    )

    mutation_11 = GLOBAL_ATTENTION_INVOCATION_IDS.index("mutation.11")
    mutation_35 = GLOBAL_ATTENTION_INVOCATION_IDS.index("mutation.35")
    assert not torch.equal(
        updated.global_site_residual_left_vectors[mutation_11],
        initial.global_site_residual_left_vectors[mutation_11],
    )
    before_left = initial.global_site_residual_left_vectors[mutation_35]
    after_left = updated.global_site_residual_left_vectors[mutation_35]
    before_right = initial.global_site_residual_right_vectors[mutation_35]
    after_right = updated.global_site_residual_right_vectors[mutation_35]
    left_drift = float((after_left - before_left).abs().max())
    right_drift = float((after_right - before_right).abs().max())
    assert torch.equal(after_left, before_left), (
        f"mutation.35 left drifted by {left_drift:.17g}; "
        f"right drift={right_drift:.17g}"
    )
    assert torch.equal(after_right, before_right), (
        f"mutation.35 right drifted by {right_drift:.17g}; "
        f"left drift={left_drift:.17g}"
    )


def test_deterministic_second_right_gauge_is_orthogonal_and_state_is_source_free() -> None:
    parent = _site_rank_one_bank(
        hierarchical=_hierarchical_bank(hidden_width=3),
        right=torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ],
            dtype=torch.float64,
        ),
    )
    first = _initialize_targeted_site_rank_two_bank(
        parent,
        gauge_floor=parent.gauge_floor,
    )
    second = _initialize_targeted_site_rank_two_bank(
        parent,
        gauge_floor=parent.gauge_floor,
    )
    assert torch.equal(
        first.second_residual_right_vectors,
        second.second_residual_right_vectors,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(first.second_residual_right_vectors, dim=-1),
        torch.ones(2, 1, dtype=torch.float64),
    )
    for target_index, invocation_id in enumerate(
        TARGETED_SITE_RANK_TWO_INVOCATION_IDS
    ):
        site_index = GLOBAL_ATTENTION_INVOCATION_IDS.index(invocation_id)
        dot = torch.einsum(
            "kh,kh->k",
            first.second_residual_right_vectors[target_index],
            parent.global_site_residual_right_vectors[site_index],
        )
        assert torch.allclose(dot, torch.zeros_like(dot), atol=1.0e-15, rtol=0.0)
    pivots = first.second_residual_right_vectors.abs().argmax(dim=-1)
    pivot_values = first.second_residual_right_vectors.gather(
        -1,
        pivots[..., None],
    ).squeeze(-1)
    assert torch.all(pivot_values > 0.0)

    state = first.state_dict()
    assert state["contains_native_value_or_output_weights"] is False
    assert state["contains_oracle_outputs_or_coefficients"] is False
    assert state["contains_raw_teacher_outputs_or_gradients"] is False
    reloaded = (
        targeted_site_rank_two_fisher_attention_transport_generator_bank_from_state_dict(
            state
        )
    )
    assert reloaded == first
    assert reloaded.artifact_sha256 == first.artifact_sha256
    assert torch.equal(
        reloaded.second_residual_right_vectors,
        first.second_residual_right_vectors,
    )

    drifted = dict(state)
    drifted["initial_site_rank_one_bank_sha256"] = _sha("wrong-parent")
    with pytest.raises(ValueError, match="lineage"):
        targeted_site_rank_two_fisher_attention_transport_generator_bank_from_state_dict(
            drifted
        )
    noncanonical = dict(state)
    noncanonical["second_residual_right_vectors"] = (
        torch.as_tensor(state["second_residual_right_vectors"]) * 2.0
    )
    with pytest.raises(ValueError, match="safety"):
        targeted_site_rank_two_fisher_attention_transport_generator_bank_from_state_dict(
            noncanonical
        )
    with pytest.raises(FrozenInstanceError):
        first.gauge_floor = 1.0  # type: ignore[misc]
    first.second_residual_left_vectors[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = first.artifact_sha256


def test_production_parameter_accounting_is_exact() -> None:
    hierarchical = _hierarchical_bank(
        generator_count=8,
        hidden_width=640,
        generator_rank=8,
        query_head_count=4,
    )
    parent = _site_rank_one_bank(hierarchical=hierarchical)
    bank = _initialize_targeted_site_rank_two_bank(
        parent,
        gauge_floor=parent.gauge_floor,
    )

    assert hierarchical.stored_parameter_count == 83_008
    assert parent.global_site_rank_one_parameter_count == 30_720
    assert parent.stored_parameter_count == 113_728
    assert bank.second_residual_parameter_count == 20_480
    assert bank.targeted_residual_parameter_count == 51_200
    assert bank.stored_parameter_count == 134_208
    hypothetical_total = bank.stored_parameter_count + 23_041 + 28_800
    assert hypothetical_total == 186_049
    assert 1.0 - hypothetical_total / 14_745_600 == pytest.approx(
        0.9873827446831597,
        abs=1.0e-15,
        rel=0.0,
    )


def test_joint_fit_is_nested_step_matched_monotone_and_beats_control(
    _fitted: tuple[
        tuple[AttentionTransportRowBatch, ...],
        GlobalSiteRankOneAttentionTransportFitConfig,
        object,
        object,
    ],
) -> None:
    _captures, _site_config, parent, fitted = _fitted
    control = fitted.continued_site_rank_one_control

    assert fitted.initial_fit_oracle == parent.final_fit_oracle
    assert fitted.initial_global_fit_oracle == parent.final_global_fit_oracle
    assert control.initial_fit_oracle == parent.final_fit_oracle
    assert control.initial_global_fit_oracle == parent.final_global_fit_oracle
    assert fitted.optimizer_steps_per_target_site_by_round == (3, 2)
    assert control.optimizer_steps_per_target_site_by_round == (3, 2)
    assert sum(fitted.optimizer_steps_per_target_site_by_round) == sum(
        control.optimizer_steps_per_target_site_by_round
    )
    _assert_monotone(fitted.fit_round_relative_fisher_losses)
    _assert_monotone(fitted.fit_round_global_relative_fisher_losses)
    _assert_monotone(control.fit_round_relative_fisher_losses)
    _assert_monotone(control.fit_round_global_relative_fisher_losses)
    for _key, history in fitted.fit_round_target_site_relative_fisher_losses:
        _assert_monotone(history)
    for _key, history in control.fit_round_target_site_relative_fisher_losses:
        _assert_monotone(history)
    assert all(
        decrease > 0.0
        for _key, decrease in fitted.warmup_fixed_support_relative_decreases
    )
    assert fitted.final_fit_oracle.relative_fisher_loss < (
        control.final_fit_oracle.relative_fisher_loss - 1.0e-4
    )
    assert fitted.final_global_fit_oracle.relative_fisher_loss < (
        control.final_global_fit_oracle.relative_fisher_loss - 1.0e-4
    )
    assert torch.count_nonzero(fitted.bank.second_residual_left_vectors) > 0
    mutation_23 = GLOBAL_ATTENTION_INVOCATION_IDS.index("mutation.23")
    assert not torch.equal(
        fitted.bank.global_site_residual_left_vectors[mutation_23],
        parent.bank.global_site_residual_left_vectors[mutation_23],
    )
    mutation_35 = GLOBAL_ATTENTION_INVOCATION_IDS.index("mutation.35")
    assert torch.equal(
        fitted.bank.global_site_residual_left_vectors[mutation_35],
        parent.bank.global_site_residual_left_vectors[mutation_35],
    )
    assert torch.equal(
        fitted.bank.global_site_residual_right_vectors[mutation_35],
        parent.bank.global_site_residual_right_vectors[mutation_35],
    )


def test_activation_warmup_recovers_when_an_adam_step_overshoots() -> None:
    captures = _tiny_captures()
    site_config = _site_fit_config(learning_rate=5.0e-2)
    parent = fit_global_site_rank_one_attention_transport_generator_bank(
        captures,
        config=site_config,
    )
    config = replace(
        _target_config(site_config, rounds=1),
        second_component_warmup_steps=8,
        steps_per_target_site_per_round=1,
    )

    fitted = fit_targeted_site_rank_two_attention_transport_generator_bank(
        captures,
        config=config,
        site_rank_one_fit=parent,
    )

    assert fitted.activation_warmup_steps_per_target_site == 8
    assert fitted.optimizer_steps_per_target_site_by_round == (9,)
    assert (
        fitted.continued_site_rank_one_control.optimizer_steps_per_target_site_by_round
        == (9,)
    )
    assert all(
        decrease > 0.0
        for _key, decrease in fitted.warmup_fixed_support_relative_decreases
    )
    accepted_learning_rates = dict(fitted.warmup_accepted_learning_rates)
    backtrack_counts = dict(fitted.warmup_backtrack_counts)
    assert backtrack_counts["mutation.11"] > 0
    assert 0.0 < accepted_learning_rates["mutation.11"] < config.learning_rate
    assert 0 <= backtrack_counts["mutation.23"] <= config.warmup_max_backtracks
    assert math.isfinite(accepted_learning_rates["mutation.23"])
    assert 0.0 < accepted_learning_rates["mutation.23"] <= config.learning_rate


def test_all_layer_diagnostic_closes_absolute_fisher_burden_and_is_metrics_only(
    _fitted: tuple[
        tuple[AttentionTransportRowBatch, ...],
        GlobalSiteRankOneAttentionTransportFitConfig,
        object,
        object,
    ],
) -> None:
    captures, _site_config, _parent, fitted = _fitted
    before = fitted.bank.artifact_sha256
    diagnostic = evaluate_all_layer_attention_transport_diagnostic(
        fitted,
        captures,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )
    assert fitted.bank.artifact_sha256 == before
    assert diagnostic.invocation_ids == _layout().invocation_ids
    banks = (
        diagnostic.hierarchical,
        diagnostic.shared_rank_one,
        diagnostic.site_rank_one,
        diagnostic.continued_site_rank_one,
        diagnostic.targeted_site_rank_two,
    )
    for bank in banks:
        assert bank.all_layers.output_metrics.row_count == 90
        assert bank.local_attention.output_metrics.row_count == 75
        assert bank.global_attention.output_metrics.row_count == 15
        assert len(bank.per_invocation) == 18
        fisher_mass = sum(
            value.output_metrics.row_count * value.output_metrics.fisher_loss
            for _key, value in bank.per_invocation
        )
        zero_mass = sum(
            value.output_metrics.row_count * value.zero_predictor_fisher_loss
            for _key, value in bank.per_invocation
        )
        assert bank.all_layers.output_metrics.fisher_loss == pytest.approx(
            fisher_mass / 90,
            abs=1.0e-12,
            rel=1.0e-11,
        )
        assert bank.all_layers.zero_predictor_fisher_loss == pytest.approx(
            zero_mass / 90,
            abs=1.0e-12,
            rel=1.0e-11,
        )
        assert sum(value for _key, value in bank.per_invocation_fisher_loss_sums) == pytest.approx(
            fisher_mass,
            abs=1.0e-12,
            rel=1.0e-11,
        )

    parent_sites = dict(diagnostic.site_rank_one.per_invocation)
    target_sites = dict(diagnostic.targeted_site_rank_two.per_invocation)
    for invocation_id in diagnostic.invocation_ids:
        if invocation_id not in TARGETED_SITE_RANK_TWO_INVOCATION_IDS:
            assert target_sites[invocation_id] == parent_sites[invocation_id]

    retained_names: set[str] = set()

    def assert_metrics_only(value: object) -> None:
        if is_dataclass(value):
            for item in fields(value):
                retained_names.add(item.name)
                assert_metrics_only(getattr(value, item.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                assert_metrics_only(item)
        else:
            assert not isinstance(value, torch.Tensor)

    assert_metrics_only(diagnostic)
    assert retained_names.isdisjoint(
        {
            "coefficients",
            "support",
            "targets",
            "score_gradients",
            "head_contexts",
            "operator_outputs",
            "raw_activations",
        }
    )


def test_total_residual_coupling_is_nested_and_zero_second_matches_parent(
    _fitted: tuple[
        tuple[AttentionTransportRowBatch, ...],
        GlobalSiteRankOneAttentionTransportFitConfig,
        object,
        object,
    ],
) -> None:
    captures, _site_config, parent, fitted = _fitted
    diagnostic = evaluate_targeted_site_rank_two_coupling_diagnostic(
        fitted.bank,
        captures,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )
    for scope in (diagnostic, *diagnostic.per_site):
        tolerance = 1.0e-11 * max(1.0, scope.tied.fisher_loss)
        assert scope.fixed_support_untied.fisher_loss <= (
            scope.tied.fisher_loss + tolerance
        )
        assert scope.reselected_support_untied.fisher_loss <= (
            scope.fixed_support_untied.fisher_loss + tolerance
        )
    assert diagnostic.active_generator_count_limit == 1
    assert diagnostic.coefficient_ridge == 0.0

    nested = _initialize_targeted_site_rank_two_bank(
        parent.bank,
        gauge_floor=parent.bank.gauge_floor,
    )
    nested_diagnostic = evaluate_targeted_site_rank_two_coupling_diagnostic(
        nested,
        captures,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )
    parent_diagnostic = evaluate_global_site_rank_one_coupling_diagnostic(
        parent.bank,
        captures,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )
    assert nested_diagnostic == parent_diagnostic


def test_stale_parent_invalid_controls_and_missing_capture_scopes_fail_closed(
    _fitted: tuple[
        tuple[AttentionTransportRowBatch, ...],
        GlobalSiteRankOneAttentionTransportFitConfig,
        object,
        object,
    ],
) -> None:
    captures, site_config, parent, fitted = _fitted
    with pytest.raises(ValueError, match="current A captures"):
        fit_targeted_site_rank_two_attention_transport_generator_bank(
            _tiny_captures(site_specific=False),
            config=_target_config(site_config, rounds=1),
            site_rank_one_fit=parent,
        )

    invalid_configs = (
        ({"alternating_rounds": 0}, "alternating_rounds"),
        ({"steps_per_target_site_per_round": 0}, "steps_per_target_site_per_round"),
        ({"second_component_warmup_steps": -1}, "warmup"),
        ({"learning_rate": 0.01}, "must match predecessor"),
        ({"residual_ridge": float("nan")}, "residual_ridge"),
        ({"gradient_clip_norm": False}, "gradient_clip_norm"),
    )
    for kwargs, message in invalid_configs:
        with pytest.raises(ValueError, match=message):
            TargetedSiteRankTwoAttentionTransportFitConfig(
                site_rank_one=site_config,
                **kwargs,
            )

    for kwargs, message in (
        ({"active_count": False}, "active_count"),
        ({"active_count": 0}, "active_count"),
        ({"active_count": 1, "euclidean_floor": 0.0}, "euclidean_floor"),
        (
            {"active_count": 1, "coefficient_ridge": -1.0},
            "coefficient_ridge",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            evaluate_targeted_site_rank_two_coupling_diagnostic(
                fitted.bank,
                captures,
                **kwargs,
            )
    incomplete_global = tuple(
        capture
        for capture in captures
        if capture.invocation_id != "mutation.35"
    )
    with pytest.raises(ValueError, match="omit.*global-attention"):
        evaluate_targeted_site_rank_two_coupling_diagnostic(
            fitted.bank,
            incomplete_global,
        )
    incomplete_all = tuple(
        capture
        for capture in captures
        if capture.invocation_id != "mutation.1"
    )
    with pytest.raises(ValueError, match="all 18"):
        evaluate_all_layer_attention_transport_diagnostic(
            fitted,
            incomplete_all,
            active_count=1,
            euclidean_floor=0.1,
        )


def test_hidden_width_one_cannot_initialize_a_second_direction() -> None:
    parent = _site_rank_one_bank(
        hierarchical=_hierarchical_bank(hidden_width=1),
    )
    with pytest.raises(ValueError, match="hidden width"):
        _initialize_targeted_site_rank_two_bank(
            parent,
            gauge_floor=parent.gauge_floor,
        )
    with pytest.raises(ValueError, match="independent second direction"):
        _deterministic_orthogonal_second_right_vectors(
            parent,
            gauge_floor=parent.gauge_floor,
        )


def test_config_defaults_bind_production_protocol() -> None:
    config = TargetedSiteRankTwoAttentionTransportFitConfig.fixed_k8_r8_top2()
    shared = config.site_rank_one.shared_rank_one.hierarchical.shared
    assert (shared.generator_count, shared.generator_rank, shared.active_count) == (
        8,
        8,
        2,
    )
    assert config.alternating_rounds == 4
    assert config.second_component_warmup_steps == 8
    assert config.steps_per_target_site_per_round == 48
    assert math.isclose(config.gauge_floor, 1.0e-12)
