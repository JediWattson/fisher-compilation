from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportRowBatch,
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
)
import fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum as gated


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
    second_axis_scale: float,
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
    operator = torch.tensor(
        [[[1.0, 0.0], [0.0, second_axis_scale]]],
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


def _captures(
    scales: dict[str, float] | None = None,
    *,
    default_scale: float = 1.0,
) -> tuple[AttentionTransportRowBatch, ...]:
    values = scales or {}
    return tuple(
        _capture(invocation_id, values.get(invocation_id, default_scale))
        for invocation_id in _layout().invocation_ids
    )


def _config(
    *,
    minimum_relative_fisher_gain: float = 1.0e-12,
    allocation_budget: int = 2,
) -> AllLayerResidualSpectrumConfig:
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
        minimum_relative_fisher_gain=minimum_relative_fisher_gain,
        tie_tolerance=1.0e-13,
    )


def test_global_selector_uses_ratio_of_summed_mass_and_retains_all_18() -> None:
    captures = _captures(
        {
            "mutation.11": 0.8,
            "mutation.23": 1.2,
            "mutation.35": 2.0,
        },
        default_scale=0.5,
    )
    result = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
    )

    first = result.candidate_rounds[0]
    assert first.objective_mode == gated.GLOBAL_OBJECTIVE_MODE
    assert first.selected_invocation_id == "mutation.35"
    assert len(first.marginals) == 18
    assert tuple(
        invocation_id
        for invocation_id, _gain in first.global_normalized_fisher_mass_gains
    ) == _layout().invocation_ids
    global_gains = dict(first.global_normalized_fisher_mass_gains)
    assert all(
        global_gains[invocation_id] == 0.0
        for invocation_id in _layout().invocation_ids
        if invocation_id not in GLOBAL_ATTENTION_INVOCATION_IDS
    )

    selected = next(
        value
        for value in first.marginals
        if value.invocation_id == "mutation.35"
    )
    expected = (
        selected.evaluation_fisher_loss_sum_gain
        / first.global_evaluation_row_count
        / first.global_evaluation_zero_predictor_fisher_loss
    )
    assert global_gains["mutation.35"] == expected
    # A mean of per-site relative improvements is a different objective when
    # the three global sites carry different zero-predictor Fisher masses.
    assert global_gains["mutation.35"] != pytest.approx(
        selected.evaluation_relative_fisher_gain / 3.0,
        abs=1.0e-12,
        rel=0.0,
    )


def test_gate_switches_only_on_the_round_after_global_loss_crosses_0_10() -> None:
    captures = _captures(default_scale=0.8)
    result = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
    )

    assert result.allocation_profile == ("mutation.11", "mutation.1")
    assert tuple(value.objective_mode for value in result.candidate_rounds) == (
        gated.GLOBAL_OBJECTIVE_MODE,
        gated.ALL_LAYER_OBJECTIVE_MODE,
    )
    initial, after_global, after_local = (
        result.global_evaluation_relative_fisher_loss_history
    )
    assert initial > 0.10
    assert after_global <= 0.10
    assert after_local == after_global
    assert result.global_gate_passed is True


def test_no_global_candidate_above_gate_stops_despite_local_improvements() -> None:
    captures = _captures()
    result = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(minimum_relative_fisher_gain=0.005),
    )

    terminal = result.candidate_rounds[0]
    assert terminal.global_relative_fisher_loss_before > 0.10
    assert terminal.objective_mode == gated.GLOBAL_OBJECTIVE_MODE
    assert terminal.selected_invocation_id is None
    assert terminal.eligible_invocation_ids == ()
    assert any(
        marginal.strictly_improves
        for marginal in terminal.marginals
        if marginal.invocation_id not in GLOBAL_ATTENTION_INVOCATION_IDS
    )
    assert result.allocation_profile == ()
    assert result.budget_exhausted is False
    assert result.termination_reason == "no_eligible_global_candidate_above_gate"


def test_only_selected_site_is_recomputed_and_every_off_site_cache_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = _captures(
        {
            "mutation.11": 0.8,
            "mutation.23": 1.2,
            "mutation.35": 2.0,
        },
        default_scale=0.5,
    )
    original = gated.compute_site_residual_spectrum_marginal
    recomputed: list[str] = []

    def spy(*args: object, **kwargs: object) -> object:
        proposal = args[1]
        recomputed.append(tuple(proposal)[0].invocation_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(gated, "compute_site_residual_spectrum_marginal", spy)
    result = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
    )

    assert result.allocation_profile == ("mutation.35", "mutation.23")
    assert recomputed == ["mutation.35"]
    first, second = result.candidate_rounds
    first_by_id = {value.invocation_id: value for value in first.marginals}
    second_by_id = {value.invocation_id: value for value in second.marginals}
    assert second_by_id["mutation.35"].source_bank_sha256 == (
        second.current_bank_sha256
    )
    assert all(
        second_by_id[invocation_id] is first_by_id[invocation_id]
        for invocation_id in _layout().invocation_ids
        if invocation_id != "mutation.35"
    )
    assert result.steps[0].used_cached_off_site_marginal is False
    assert result.steps[1].used_cached_off_site_marginal is True


def test_gate_equality_uses_all_layers_and_budget_is_fixed_at_two() -> None:
    captures = _captures()
    baseline = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
    )
    exact_initial_global = (
        baseline.initial_global_evaluation_oracle.relative_fisher_loss
    )
    equality = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
        global_relative_fisher_threshold=exact_initial_global,
    )
    assert equality.candidate_rounds[0].objective_mode == (
        gated.ALL_LAYER_OBJECTIVE_MODE
    )
    assert equality.allocation_profile[0] == "mutation.1"

    with pytest.raises(ValueError, match="budget two"):
        gated.greedily_allocate_global_gated_residual_spectrum(
            _parent_bank(),
            captures,
            captures,
            config=_config(allocation_budget=3),
        )
    with pytest.raises(ValueError, match="zero ridge"):
        gated.greedily_allocate_global_gated_residual_spectrum(
            _parent_bank(),
            captures,
            captures,
            config=replace(_config(), coefficient_ridge=1.0e-8),
        )


def test_round_rejects_local_selection_or_nonzero_local_global_gain() -> None:
    captures = _captures()
    result = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
    )
    first = result.candidate_rounds[0]
    assert first.objective_mode == gated.GLOBAL_OBJECTIVE_MODE
    with pytest.raises(ValueError, match="numeric argmax"):
        replace(first, selected_invocation_id="mutation.1")

    gains = list(first.global_normalized_fisher_mass_gains)
    gains[0] = (gains[0][0], 1.0e-6)
    with pytest.raises(ValueError, match="does not close"):
        replace(first, global_normalized_fisher_mass_gains=tuple(gains))

    with pytest.raises(ValueError, match="aggregate Fisher mass"):
        replace(
            first,
            all_layer_relative_fisher_loss_before=(
                first.all_layer_relative_fisher_loss_before + 0.01
            ),
        )


def test_step_cache_flag_and_allocation_state_chain_are_bound() -> None:
    captures = _captures()
    result = gated.greedily_allocate_global_gated_residual_spectrum(
        _parent_bank(),
        captures,
        captures,
        config=_config(),
    )
    assert result.steps[1].used_cached_off_site_marginal is True
    with pytest.raises(ValueError, match="step is invalid"):
        replace(result.steps[1], used_cached_off_site_marginal=False)

    tampered_first = replace(
        result.steps[0],
        result_bank_sha256=_sha("unbound-result-state"),
    )
    with pytest.raises(ValueError, match="state chain|binding"):
        replace(
            result,
            steps=(tampered_first, result.steps[1]),
        )

    copied_marginal = replace(result.steps[0].marginal)
    copied_step = replace(result.steps[0], marginal=copied_marginal)
    with pytest.raises(ValueError, match="binding"):
        replace(result, steps=(copied_step, result.steps[1]))
