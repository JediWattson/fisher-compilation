from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
    samplewise_joint_fisher_loss,
)
from fisher_graph.gemma3_attention_transport_global_rank1 import (
    GLOBAL_ATTENTION_INVOCATION_IDS,
)
from fisher_graph.gemma3_attention_transport_global_site_rank1 import (
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
)
from fisher_graph.gemma3_attention_transport_hierarchical_edge import (
    AttentionTransportHierarchyLayout,
    HierarchicalFisherAttentionTransportGeneratorBank,
)
from fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum import (
    AllLayerResidualSpectrumConfig,
    _append_allocated_component,
    _exact_fisher_amplitudes,
    _matrix_free_top_singular_pair,
    all_layer_residual_spectrum_generator_bank_from_parent,
    all_layer_residual_spectrum_generator_bank_from_state_dict,
    compute_all_layer_first_residual_spectrum,
    evaluate_allocated_residual_bank_all_layers,
    greedily_allocate_residual_spectrum,
    replay_fixed_residual_allocation_profile,
    residual_allocation_profile_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _layout() -> AttentionTransportHierarchyLayout:
    return AttentionTransportHierarchyLayout.gemma3_270m_attention(
        sorted(f"mutation.{2 * index + 1}" for index in range(18))
    )


def _parent_bank() -> GlobalSiteRankOneFisherAttentionTransportGeneratorBank:
    layout = _layout()
    factors = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)
    head_ids = tuple(
        f"{invocation_id}/query-head.0"
        for invocation_id in layout.invocation_ids
    )
    hierarchical = HierarchicalFisherAttentionTransportGeneratorBank(
        left_factors=factors,
        singular_scales=torch.ones(1, 1, dtype=torch.float64),
        right_factors=factors.clone(),
        head_coefficients=torch.ones(len(head_ids), 1, dtype=torch.float64),
        head_ids=head_ids,
        hierarchy=layout,
        block_scale_contrasts=torch.zeros(2, 1, 1, dtype=torch.float64),
        site_scale_contrasts=torch.zeros(5, 1, 1, dtype=torch.float64),
    )
    site_shape = (len(GLOBAL_ATTENTION_INVOCATION_IDS), 1, 2)
    site_right = torch.zeros(site_shape, dtype=torch.float64)
    site_right[:, :, 0] = 1.0
    return GlobalSiteRankOneFisherAttentionTransportGeneratorBank(
        left_factors=hierarchical.left_factors,
        singular_scales=hierarchical.singular_scales,
        right_factors=hierarchical.right_factors,
        head_coefficients=hierarchical.head_coefficients,
        head_ids=hierarchical.head_ids,
        hierarchy=hierarchical.hierarchy,
        block_scale_contrasts=hierarchical.block_scale_contrasts,
        site_scale_contrasts=hierarchical.site_scale_contrasts,
        signed_feature_floor=hierarchical.signed_feature_floor,
        global_site_residual_left_vectors=torch.zeros(
            site_shape,
            dtype=torch.float64,
        ),
        global_site_residual_right_vectors=site_right,
        shared_rank_one_bank_sha256=_sha("shared-rank-one"),
    )


def _capture(
    invocation_id: str,
    operator: torch.Tensor,
) -> AttentionTransportRowBatch:
    contexts = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[1.0, -1.0]],
            [[-0.5, 2.0]],
            [[2.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    outputs = torch.einsum("aoh,nah->no", operator, contexts)
    return AttentionTransportRowBatch(
        invocation_id=invocation_id,
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=torch.ones_like(outputs),
        chart_inputs=contexts[:, 0, :].clone(),
        head_operators=operator,
        native_v_o_parameter_count=8,
    )


def _all_layer_captures(
    *,
    operator: torch.Tensor | None = None,
) -> tuple[AttentionTransportRowBatch, ...]:
    value = (
        torch.eye(2, dtype=torch.float64)[None]
        if operator is None
        else operator
    )
    return tuple(
        _capture(invocation_id, value)
        for invocation_id in _layout().invocation_ids
    )


def _config(*, allocation_budget: int = 2) -> AllLayerResidualSpectrumConfig:
    return AllLayerResidualSpectrumConfig(
        generator_count=1,
        generator_rank=1,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
        amplitude_ridge=1.0e-12,
        power_iterations=24,
        deterministic_start_count=4,
        allocation_budget=allocation_budget,
        minimum_relative_fisher_gain=1.0e-12,
        tie_tolerance=1.0e-13,
    )


def test_matrix_free_singular_pair_and_fisher_amplitudes_match_dense_math() -> None:
    contextual = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]],
        dtype=torch.float64,
    )
    output_gradient = torch.tensor(
        [[3.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    coefficients = torch.ones(2, 1, dtype=torch.float64)
    left, right, singular, projection_rank = _matrix_free_top_singular_pair(
        contextual,
        output_gradient,
        coefficients,
        bank_sha256=_sha("matrix-free-bank"),
        invocation_id="mutation.1",
        generator_index=0,
        prior_rights=torch.empty(0, 2, dtype=torch.float64),
        config=_config(),
    )
    dense_gradient = torch.einsum(
        "nk,nh,nkj->hj",
        coefficients,
        output_gradient,
        contextual,
    )
    dense_left, dense_singular, dense_right_h = torch.linalg.svd(
        dense_gradient
    )
    assert projection_rank == 0
    assert singular == pytest.approx(float(dense_singular[0]), abs=1.0e-12)
    assert torch.allclose(
        left[:, None] * right[None],
        dense_left[:, :1] * dense_right_h[:1],
        atol=1.0e-12,
        rtol=0.0,
    )

    directional = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.5, 1.0], [1.0, -0.5]],
            [[-0.25, 0.75], [0.5, 1.0]],
        ],
        dtype=torch.float64,
    )
    baseline = torch.tensor(
        [[0.2, -0.1], [0.5, 0.25], [-0.4, 0.3]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[0.8, 0.3], [0.1, -0.2], [0.2, 0.9]],
        dtype=torch.float64,
    )
    score = torch.tensor(
        [[1.0, 0.5], [-0.25, 1.0], [0.75, -0.5]],
        dtype=torch.float64,
    )
    ridge = 0.2
    floor = 0.1
    observed = _exact_fisher_amplitudes(
        directional,
        baseline,
        target,
        score,
        euclidean_floor=floor,
        amplitude_ridge=ridge,
    )
    fisher = torch.stack(
        tuple(
            floor * torch.eye(2, dtype=torch.float64)
            + gradient[:, None] * gradient[None]
            for gradient in score
        )
    )
    dense_gram = torch.einsum(
        "nkh,nhj,nqj->kq",
        directional,
        fisher,
        directional,
    ) / directional.shape[0] + ridge * torch.eye(2, dtype=torch.float64)
    dense_gradient_vector = torch.einsum(
        "nkh,nhj,nj->k",
        directional,
        fisher,
        baseline - target,
    ) / directional.shape[0]
    expected = torch.linalg.solve(dense_gram, -dense_gradient_vector)
    assert torch.allclose(observed, expected, atol=1.0e-12, rtol=0.0)


def test_exhausted_h1_projection_is_an_exact_zero_product_and_not_accepted() -> None:
    contextual = torch.tensor(
        [[[1.0]], [[-0.5]], [[2.0]]],
        dtype=torch.float64,
    )
    output_gradient = torch.tensor(
        [[1.0], [-0.25], [0.5]],
        dtype=torch.float64,
    )
    coefficients = torch.ones(3, 1, dtype=torch.float64)
    left, right, singular, projection_rank = _matrix_free_top_singular_pair(
        contextual,
        output_gradient,
        coefficients,
        bank_sha256=_sha("exhausted-h1"),
        invocation_id="mutation.11",
        generator_index=0,
        prior_rights=torch.ones(1, 1, dtype=torch.float64),
        config=_config(),
    )
    assert projection_rank == 1
    assert singular == 0.0
    assert torch.equal(left, torch.zeros_like(left))
    assert torch.equal(right, torch.zeros_like(right))
    assert torch.equal(left[:, None] * right[None], torch.zeros(1, 1))

    activation = torch.einsum("nkh,kh->nk", contextual, right[None])
    directional = (
        coefficients[:, :, None]
        * activation[:, :, None]
        * left[None, None]
    )
    baseline = torch.tensor([[0.2], [-0.1], [0.4]], dtype=torch.float64)
    target = torch.tensor([[0.8], [0.3], [-0.2]], dtype=torch.float64)
    score = torch.ones_like(target)
    amplitudes = _exact_fisher_amplitudes(
        directional,
        baseline,
        target,
        score,
        euclidean_floor=0.1,
        amplitude_ridge=0.0,
    )
    candidate = baseline + torch.einsum(
        "k,nkh->nh",
        amplitudes,
        directional,
    )
    assert torch.equal(directional, torch.zeros_like(directional))
    assert torch.equal(amplitudes, torch.zeros_like(amplitudes))
    assert torch.equal(candidate, baseline)
    current_loss = samplewise_joint_fisher_loss(
        baseline,
        target,
        score,
        euclidean_floor=0.1,
    )
    candidate_loss = samplewise_joint_fisher_loss(
        candidate,
        target,
        score,
        euclidean_floor=0.1,
    )
    assert torch.equal(candidate_loss, current_loss)


def test_bank_round_trip_tamper_offsite_and_noncanonical_append_prefix() -> None:
    parent = _parent_bank()
    empty = all_layer_residual_spectrum_generator_bank_from_parent(parent)
    first = _append_allocated_component(
        empty,
        invocation_id="mutation.1",
        left=torch.tensor([[0.3, -0.7]], dtype=torch.float64),
        right=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
    )
    prefix_left = first.allocated_residual_left_vectors.clone()
    prefix_right = first.allocated_residual_right_vectors.clone()
    second = _append_allocated_component(
        first,
        invocation_id="mutation.3",
        left=torch.tensor([[0.8, -0.2]], dtype=torch.float64),
        right=torch.tensor([[-2.0, 0.25]], dtype=torch.float64),
    )
    assert torch.equal(second.allocated_residual_left_vectors[:1], prefix_left)
    assert torch.equal(second.allocated_residual_right_vectors[:1], prefix_right)
    assert torch.equal(
        second.global_site_residual_left_vectors,
        parent.global_site_residual_left_vectors,
    )
    assert torch.equal(
        second.global_site_residual_right_vectors,
        parent.global_site_residual_right_vectors,
    )

    contexts = torch.tensor(
        [[[1.0, 2.0]], [[-0.5, 0.25]]],
        dtype=torch.float64,
    )
    assert not torch.equal(
        second.candidate_outputs_from_head_contexts(
            invocation_id="mutation.1",
            head_contexts=contexts,
        ),
        parent.candidate_outputs_from_head_contexts(
            invocation_id="mutation.1",
            head_contexts=contexts,
        ),
    )
    assert torch.equal(
        second.candidate_outputs_from_head_contexts(
            invocation_id="mutation.5",
            head_contexts=contexts,
        ),
        parent.candidate_outputs_from_head_contexts(
            invocation_id="mutation.5",
            head_contexts=contexts,
        ),
    )

    state = second.state_dict()
    assert state["contains_native_value_or_output_weights"] is False
    assert state["contains_oracle_outputs_or_coefficients"] is False
    assert state["contains_raw_teacher_outputs_or_gradients"] is False
    reloaded = all_layer_residual_spectrum_generator_bank_from_state_dict(state)
    assert reloaded == second
    assert reloaded.artifact_sha256 == second.artifact_sha256

    tampered = dict(state)
    tampered["allocated_residual_right_vectors"] = (
        torch.as_tensor(state["allocated_residual_right_vectors"]) * 2.0
    )
    with pytest.raises(ValueError, match="safety"):
        all_layer_residual_spectrum_generator_bank_from_state_dict(tampered)
    reloaded.allocated_residual_left_vectors[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = reloaded.artifact_sha256


def test_all_18_first_spectrum_and_two_step_greedy_cache_tie_replay_evaluator() -> None:
    parent = _parent_bank()
    captures = _all_layer_captures()
    config = _config()
    initial = all_layer_residual_spectrum_generator_bank_from_parent(parent)
    spectrum = compute_all_layer_first_residual_spectrum(
        initial,
        captures,
        captures,
        config=config,
    )
    assert spectrum.invocation_ids == _layout().invocation_ids
    assert len(spectrum.per_site) == 18
    assert all(
        marginal.proposal_bank.allocated_invocation_ids
        == (marginal.invocation_id,)
        for marginal in spectrum.per_site
    )
    assert all(marginal.strictly_improves for marginal in spectrum.per_site)
    gains = dict(spectrum.evaluation_all_layer_normalized_fisher_gains)
    assert gains["mutation.1"] == pytest.approx(
        gains["mutation.3"],
        abs=config.tie_tolerance,
        rel=0.0,
    )

    greedy = greedily_allocate_residual_spectrum(
        parent,
        captures,
        captures,
        config=config,
    )
    assert greedy.budget_exhausted is True
    assert greedy.allocation_profile == ("mutation.1", "mutation.3")
    assert len(greedy.candidate_rounds) == len(greedy.steps) == 2
    numeric_ids = tuple(f"mutation.{2 * index + 1}" for index in range(18))
    for round_index, (candidate_round, step) in enumerate(
        zip(greedy.candidate_rounds, greedy.steps, strict=True)
    ):
        assert candidate_round.allocation_index == round_index
        assert tuple(
            marginal.invocation_id for marginal in candidate_round.marginals
        ) == numeric_ids
        assert tuple(
            invocation_id
            for invocation_id, _gain in (
                candidate_round.normalized_fisher_mass_gains
            )
        ) == numeric_ids
        assert candidate_round.selected_invocation_id == step.invocation_id
        assert candidate_round.current_bank_sha256 == step.source_bank_sha256
        gains_by_id = dict(candidate_round.normalized_fisher_mass_gains)
        eligible = tuple(
            marginal
            for marginal in candidate_round.marginals
            if marginal.strictly_improves
        )
        best_gain = max(gains_by_id[item.invocation_id] for item in eligible)
        tied = tuple(
            item
            for item in eligible
            if gains_by_id[item.invocation_id]
            >= best_gain - candidate_round.tie_tolerance
        )
        assert candidate_round.selected_invocation_id == min(
            tied,
            key=lambda item: int(item.invocation_id.split(".")[1]),
        ).invocation_id
        assert step.selection_all_layer_normalized_fisher_gain == pytest.approx(
            step.evaluation_before.relative_fisher_loss
            - step.evaluation_after.relative_fisher_loss,
            abs=1.0e-12,
            rel=0.0,
        )
    first_round, second_round = greedy.candidate_rounds
    assert all(
        marginal.source_bank_sha256 == greedy.initial_bank_sha256
        for marginal in first_round.marginals
    )
    first_by_id = {
        marginal.invocation_id: marginal for marginal in first_round.marginals
    }
    second_by_id = {
        marginal.invocation_id: marginal for marginal in second_round.marginals
    }
    first_selected = first_round.selected_invocation_id
    assert first_selected is not None
    assert second_by_id[first_selected].source_bank_sha256 == (
        second_round.current_bank_sha256
    )
    for invocation_id in numeric_ids:
        if invocation_id == first_selected:
            continue
        assert second_by_id[invocation_id] is first_by_id[invocation_id]
        assert dict(second_round.normalized_fisher_mass_gains)[invocation_id] == (
            dict(first_round.normalized_fisher_mass_gains)[invocation_id]
        )
    assert tuple(step.used_cached_off_site_marginal for step in greedy.steps) == (
        False,
        True,
    )
    assert all(
        right < left
        for history in (
            greedy.proposal_fit_relative_fisher_loss_history,
            greedy.evaluation_relative_fisher_loss_history,
        )
        for left, right in zip(history, history[1:])
    )

    replay = replay_fixed_residual_allocation_profile(
        parent,
        captures,
        greedy.allocation_profile,
        config=config,
    )
    assert replay.allocation_profile == greedy.allocation_profile
    assert replay.allocated_component_count == 2
    assert all(
        right < left
        for left, right in zip(
            replay.fit_relative_fisher_loss_history,
            replay.fit_relative_fisher_loss_history[1:],
        )
    )
    diagnostic = evaluate_allocated_residual_bank_all_layers(
        parent,
        replay.final_bank,
        captures,
        config=config,
    )
    assert diagnostic.invocation_ids == _layout().invocation_ids
    assert diagnostic.parent.all_layers.output_metrics.row_count == 18 * 6
    assert diagnostic.parent.local_attention.output_metrics.row_count == 15 * 6
    assert diagnostic.parent.global_attention.output_metrics.row_count == 3 * 6
    per_site_gain = dict(
        diagnostic.per_invocation_all_layer_normalized_fisher_gains
    )
    assert per_site_gain["mutation.1"] > 0.0
    assert per_site_gain["mutation.3"] > 0.0
    assert all(
        abs(gain) <= 1.0e-15
        for invocation_id, gain in per_site_gain.items()
        if invocation_id not in greedy.allocation_profile
    )


def test_repeated_fresh_site_reuses_exact_proposal_bank_identity() -> None:
    parent = _parent_bank()
    parent_operator = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    non_axis_operator = torch.tensor(
        [[[-3.0, -3.0], [-1.0, 0.0]]],
        dtype=torch.float64,
    )
    captures = tuple(
        _capture(
            invocation_id,
            (
                non_axis_operator
                if invocation_id == "mutation.1"
                else parent_operator
            ),
        )
        for invocation_id in _layout().invocation_ids
    )

    greedy = greedily_allocate_residual_spectrum(
        parent,
        captures,
        captures,
        config=_config(),
    )

    assert greedy.allocation_profile == ("mutation.1", "mutation.1")
    assert all(not step.used_cached_off_site_marginal for step in greedy.steps)
    assert all(
        step.result_bank_sha256 == step.marginal.proposal_bank.artifact_sha256
        for step in greedy.steps
    )
    assert greedy.final_bank.artifact_sha256 == (
        greedy.steps[-1].marginal.proposal_bank.artifact_sha256
    )


def test_greedy_stops_and_fixed_replay_aborts_when_no_component_improves() -> None:
    parent = _parent_bank()
    exactly_parent_operator = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float64,
    )
    captures = _all_layer_captures(operator=exactly_parent_operator)
    config = _config()
    greedy = greedily_allocate_residual_spectrum(
        parent,
        captures,
        captures,
        config=config,
    )
    assert greedy.allocation_profile == ()
    assert greedy.steps == ()
    assert greedy.budget_exhausted is False
    assert len(greedy.candidate_rounds) == 1
    terminal_round = greedy.candidate_rounds[0]
    assert terminal_round.selected_invocation_id is None
    assert terminal_round.current_bank_sha256 == greedy.initial_bank_sha256
    assert greedy.final_bank.artifact_sha256 == greedy.initial_bank_sha256
    assert all(
        not marginal.strictly_improves for marginal in terminal_round.marginals
    )
    assert greedy.allocation_profile_sha256 == (
        residual_allocation_profile_sha256(())
    )
    with pytest.raises(RuntimeError, match="strict improvement"):
        replay_fixed_residual_allocation_profile(
            parent,
            captures,
            ("mutation.1",),
            config=config,
        )


def test_local_positive_but_all_layer_subthreshold_candidate_stops_greedy() -> None:
    parent = _parent_bank()
    captures = _all_layer_captures()
    permissive = _config()
    initial = all_layer_residual_spectrum_generator_bank_from_parent(parent)
    spectrum = compute_all_layer_first_residual_spectrum(
        initial,
        captures,
        captures,
        config=permissive,
    )
    maximum_all_layer_gain = max(
        gain
        for _invocation_id, gain in (
            spectrum.evaluation_all_layer_normalized_fisher_gains
        )
    )
    maximum_local_protocol_floor = max(
        min(
            marginal.frozen_relative_fisher_gain,
            marginal.proposal_fit_relative_fisher_gain,
            marginal.evaluation_relative_fisher_gain,
        )
        for marginal in spectrum.per_site
    )
    assert maximum_all_layer_gain < maximum_local_protocol_floor
    threshold = (
        maximum_all_layer_gain + maximum_local_protocol_floor
    ) / 2.0
    config = replace(
        permissive,
        minimum_relative_fisher_gain=threshold,
    )
    greedy = greedily_allocate_residual_spectrum(
        parent,
        captures,
        captures,
        config=config,
    )
    assert greedy.allocation_profile == ()
    assert greedy.steps == ()
    assert greedy.final_bank.artifact_sha256 == greedy.initial_bank_sha256
    assert len(greedy.candidate_rounds) == 1
    terminal = greedy.candidate_rounds[0]
    assert terminal.selected_invocation_id is None
    assert any(marginal.strictly_improves for marginal in terminal.marginals)
    assert all(
        gain <= threshold
        for _invocation_id, gain in terminal.normalized_fisher_mass_gains
    )
