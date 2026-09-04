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
    GlobalRankOneFisherAttentionTransportGeneratorBank,
    fit_global_rank_one_attention_transport_generator_bank,
)
from fisher_graph.gemma3_attention_transport_global_site_rank1 import (
    GlobalSiteRankOneAttentionTransportFitConfig,
    GlobalSiteRankOneFisherAttentionTransportGeneratorBank,
    _broadcast_shared_rank_one_bank,
    _global_site_rank_one_bank_with_factors,
    evaluate_global_site_rank_one_coupling_diagnostic,
    fit_global_site_rank_one_attention_transport_generator_bank,
    global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict,
)
from fisher_graph.gemma3_attention_transport_hierarchical_edge import (
    AttentionTransportHierarchyLayout,
    HierarchicalFisherAttentionTransportGeneratorBank,
    HierarchicalSparseAttentionTransportFitConfig,
)
from fisher_graph.gemma3_attention_transport_sparse_v2 import (
    SparseAwareAttentionTransportFitConfig,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _layout() -> AttentionTransportHierarchyLayout:
    return AttentionTransportHierarchyLayout.gemma3_270m_attention(
        sorted(f"mutation.{2 * index + 1}" for index in range(18))
    )


def _hierarchical_bank(
    *,
    generator_count: int = 2,
    hidden_width: int = 3,
    generator_rank: int = 1,
    query_head_count: int = 1,
) -> HierarchicalFisherAttentionTransportGeneratorBank:
    if hidden_width < generator_rank:
        raise ValueError("test bank hidden width is below its rank")
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


def _shared_rank_one_bank(
    *,
    hierarchical: HierarchicalFisherAttentionTransportGeneratorBank | None = None,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
) -> GlobalRankOneFisherAttentionTransportGeneratorBank:
    parent = _hierarchical_bank() if hierarchical is None else hierarchical
    residual_left = (
        torch.zeros(
            parent.generator_count,
            parent.hidden_width,
            dtype=torch.float64,
        )
        if left is None
        else left
    )
    residual_right = (
        torch.eye(
            parent.generator_count,
            parent.hidden_width,
            dtype=torch.float64,
        )
        if right is None
        else right
    )
    return GlobalRankOneFisherAttentionTransportGeneratorBank(
        left_factors=parent.left_factors,
        singular_scales=parent.singular_scales,
        right_factors=parent.right_factors,
        head_coefficients=parent.head_coefficients,
        head_ids=parent.head_ids,
        hierarchy=parent.hierarchy,
        block_scale_contrasts=parent.block_scale_contrasts,
        site_scale_contrasts=parent.site_scale_contrasts,
        signed_feature_floor=parent.signed_feature_floor,
        global_residual_left_vectors=residual_left,
        global_residual_right_vectors=residual_right,
    )


def _site_rank_one_bank(
    *,
    hierarchical: HierarchicalFisherAttentionTransportGeneratorBank | None = None,
    left: torch.Tensor | None = None,
    right: torch.Tensor | None = None,
    shared_sha256: str | None = None,
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
        shared_rank_one_bank_sha256=(
            _sha("shared-rank-one")
            if shared_sha256 is None
            else shared_sha256
        ),
    )


def _capture(
    invocation_id: str,
    contexts: torch.Tensor,
    operator: torch.Tensor,
    *,
    gradients: torch.Tensor | None = None,
) -> AttentionTransportRowBatch:
    outputs = torch.einsum("aoh,nah->no", operator, contexts)
    return AttentionTransportRowBatch(
        invocation_id=invocation_id,
        head_contexts=contexts,
        operator_outputs=outputs,
        operator_output_score_gradients=(
            torch.ones_like(outputs) if gradients is None else gradients
        ),
        chart_inputs=contexts[:, 0, :].clone(),
        head_operators=operator,
        native_v_o_parameter_count=2 * contexts.shape[-1] ** 2,
    )


def _tiny_captures(
    *,
    site_specific: bool = False,
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


def _shared_fit_config(
    *,
    generator_count: int = 1,
    global_rounds: int = 1,
    global_steps: int = 4,
    learning_rate: float = 1.0e-2,
    gauge_floor: float = 1.0e-12,
) -> GlobalRankOneAttentionTransportFitConfig:
    sparse = SparseAwareAttentionTransportFitConfig(
        generator_count=generator_count,
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
        gauge_floor=gauge_floor,
    )
    return GlobalRankOneAttentionTransportFitConfig(
        hierarchical=HierarchicalSparseAttentionTransportFitConfig(
            shared=sparse,
            alternating_rounds=4,
        ),
        alternating_rounds=global_rounds,
        steps_per_round=global_steps,
        learning_rate=learning_rate,
        gauge_floor=gauge_floor,
    )


def _assert_monotone(history: tuple[float, ...]) -> None:
    assert all(
        right <= left + 1.0e-12
        for left, right in zip(history, history[1:])
    )


def test_shared_broadcast_is_bit_exact_and_local_path_is_exactly_hierarchical() -> None:
    hierarchical = _hierarchical_bank()
    shared = _shared_rank_one_bank(
        hierarchical=hierarchical,
        left=torch.tensor(
            [[1.0, -0.5, 0.25], [-0.75, 0.3, 1.2]],
            dtype=torch.float64,
        ),
        right=torch.tensor(
            [[0.2, -0.4, 0.8], [-0.5, 0.25, 0.75]],
            dtype=torch.float64,
        ),
    )
    bank = _broadcast_shared_rank_one_bank(shared, gauge_floor=shared.gauge_floor)

    assert torch.equal(
        bank.global_site_residual_left_vectors,
        shared.global_residual_left_vectors[None].expand(3, -1, -1),
    )
    assert torch.equal(
        bank.global_site_residual_right_vectors,
        shared.global_residual_right_vectors[None].expand(3, -1, -1),
    )
    contexts = torch.tensor(
        [[[1.0, 2.0, -0.5]], [[-1.5, 0.25, 3.0]]],
        dtype=torch.float64,
    )
    for invocation_id in bank.hierarchy.invocation_ids:
        site_output = bank.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        shared_output = shared.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        assert torch.equal(site_output, shared_output)
        assert torch.equal(
            bank.router_context_features_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
            hierarchical.router_context_features_from_head_contexts(
                invocation_id=invocation_id,
                head_contexts=contexts,
            ),
        )
        if invocation_id not in GLOBAL_ATTENTION_INVOCATION_IDS:
            assert torch.equal(
                site_output,
                hierarchical.candidate_outputs_from_head_contexts(
                    invocation_id=invocation_id,
                    head_contexts=contexts,
                ),
            )
            assert torch.equal(
                bank.global_site_rank_one_candidate_residuals_from_head_contexts(
                    invocation_id=invocation_id,
                    head_contexts=contexts,
                ),
                torch.zeros_like(site_output),
            )


def test_one_site_factor_perturbation_is_isolated_to_that_global_invocation() -> None:
    shared = _shared_rank_one_bank(
        left=torch.ones(2, 3, dtype=torch.float64),
        right=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    original = _broadcast_shared_rank_one_bank(
        shared,
        gauge_floor=shared.gauge_floor,
    )
    left = original.global_site_residual_left_vectors.clone()
    left[1, 0, 2] += 4.0
    changed = _global_site_rank_one_bank_with_factors(
        original,
        left,
        original.global_site_residual_right_vectors,
    )
    contexts = torch.tensor(
        [[[2.0, 1.0, -1.0]], [[-0.5, 3.0, 0.25]]],
        dtype=torch.float64,
    )

    for invocation_id in original.hierarchy.invocation_ids:
        before = original.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        after = changed.candidate_outputs_from_head_contexts(
            invocation_id=invocation_id,
            head_contexts=contexts,
        )
        if invocation_id == "mutation.23":
            assert not torch.equal(after, before)
        else:
            assert torch.equal(after, before)


def test_production_geometry_has_exact_parameter_accounting() -> None:
    hierarchical = _hierarchical_bank(
        generator_count=8,
        hidden_width=640,
        generator_rank=8,
        query_head_count=4,
    )
    shared = _shared_rank_one_bank(hierarchical=hierarchical)
    bank = _broadcast_shared_rank_one_bank(shared, gauge_floor=1.0e-12)

    assert hierarchical.stored_parameter_count == 83_008
    assert shared.global_rank_one_parameter_count == 10_240
    assert shared.stored_parameter_count == 93_248
    assert bank.global_site_rank_one_parameter_count == 30_720
    assert bank.stored_parameter_count == 113_728
    hypothetical_atlas_parameters = 23_041
    hypothetical_router_parameters = 28_800
    hypothetical_total = (
        bank.stored_parameter_count
        + hypothetical_atlas_parameters
        + hypothetical_router_parameters
    )
    assert hypothetical_total == 165_569
    assert 1.0 - hypothetical_total / 14_745_600 == pytest.approx(
        0.9887716335720486,
        abs=1.0e-15,
        rel=0.0,
    )


def test_site_factor_gauge_state_hash_and_round_trip_are_canonical() -> None:
    left = torch.tensor(
        [
            [[-8.0, 0.0, 1.0], [0.0, 0.5, -1.0]],
            [[1.0, 2.0, 3.0], [-2.0, 0.0, 0.5]],
            [[4.0, -1.0, 0.0], [9.0, 8.0, 7.0]],
        ],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [
            [[0.5, 0.0, 0.0], [0.0, -2.0, 0.0]],
            [[1.0, -2.0, 0.5], [-0.25, 0.75, 0.0]],
            [[0.0, 0.0, -3.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    original_products = torch.einsum("sko,skh->skoh", left, right)
    bank = _site_rank_one_bank(left=left, right=right)

    assert torch.allclose(
        torch.einsum(
            "sko,skh->skoh",
            bank.global_site_residual_left_vectors,
            bank.global_site_residual_right_vectors,
        ),
        original_products,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(
            bank.global_site_residual_right_vectors,
            dim=-1,
        ),
        torch.ones(3, 2, dtype=torch.float64),
    )
    pivots = torch.argmax(
        bank.global_site_residual_right_vectors.abs(),
        dim=-1,
    )
    pivot_values = bank.global_site_residual_right_vectors.gather(
        -1,
        pivots[..., None],
    ).squeeze(-1)
    assert torch.all(pivot_values > 0.0)
    assert torch.equal(
        bank.global_site_residual_left_vectors[2, 1],
        torch.zeros(3, dtype=torch.float64),
    )

    state = bank.state_dict()
    assert state["contains_native_value_or_output_weights"] is False
    assert state["contains_oracle_outputs_or_coefficients"] is False
    assert state["contains_raw_teacher_outputs_or_gradients"] is False
    reloaded = (
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            state
        )
    )
    assert reloaded == bank
    assert hash(reloaded) == hash(bank)
    assert reloaded.artifact_sha256 == bank.artifact_sha256
    reloaded_state = reloaded.state_dict()
    assert set(reloaded_state) == set(state)
    for key, expected in state.items():
        actual = reloaded_state[key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        elif isinstance(expected, dict):
            assert actual["artifact_sha256"] == expected["artifact_sha256"]
        else:
            assert actual == expected


def test_state_tensor_lineage_and_live_mutation_tampering_fail_closed() -> None:
    bank = _site_rank_one_bank(
        left=torch.arange(18, dtype=torch.float64).reshape(3, 2, 3),
        right=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float64,
        ),
    )
    state = bank.state_dict()

    extra = dict(state)
    extra["unexpected"] = False
    with pytest.raises(ValueError, match="fields"):
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            extra
        )

    unsafe = dict(state)
    unsafe["contains_native_value_or_output_weights"] = True
    with pytest.raises(ValueError, match="safety"):
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            unsafe
        )

    drifted_hierarchy = dict(state)
    drifted_hierarchy["hierarchical_bank_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="lineage"):
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            drifted_hierarchy
        )

    drifted_shared = dict(state)
    drifted_shared["shared_rank_one_bank_sha256"] = _sha("different-shared")
    with pytest.raises(ValueError, match="lineage"):
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            drifted_shared
        )

    changed_tensor = dict(state)
    changed_tensor["global_site_residual_left_vectors"] = (
        torch.as_tensor(state["global_site_residual_left_vectors"]).clone()
    )
    changed_tensor["global_site_residual_left_vectors"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="lineage"):
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            changed_tensor
        )

    noncanonical = dict(state)
    noncanonical["global_site_residual_left_vectors"] = (
        torch.as_tensor(state["global_site_residual_left_vectors"]) / 2.0
    )
    noncanonical["global_site_residual_right_vectors"] = (
        torch.as_tensor(state["global_site_residual_right_vectors"]) * 2.0
    )
    with pytest.raises(ValueError, match="gauge"):
        global_site_rank_one_fisher_attention_transport_generator_bank_from_state_dict(
            noncanonical
        )

    with pytest.raises(FrozenInstanceError):
        bank.gauge_floor = 1.0  # type: ignore[misc]
    bank.global_site_residual_left_vectors[0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="changed after construction"):
        _ = bank.artifact_sha256


@pytest.fixture(scope="module")
def _tiny_shared_fit() -> tuple[
    tuple[AttentionTransportRowBatch, ...],
    GlobalRankOneAttentionTransportFitConfig,
    object,
]:
    captures = _tiny_captures()
    config = _shared_fit_config()
    fitted = fit_global_rank_one_attention_transport_generator_bank(
        captures,
        config=config,
    )
    return captures, config, fitted


def test_tiny_site_fit_is_deterministic_and_histories_close(
    _tiny_shared_fit: tuple[
        tuple[AttentionTransportRowBatch, ...],
        GlobalRankOneAttentionTransportFitConfig,
        object,
    ],
) -> None:
    captures, shared_config, shared_fit = _tiny_shared_fit
    config = GlobalSiteRankOneAttentionTransportFitConfig(
        shared_rank_one=shared_config,
        alternating_rounds=2,
        steps_per_site_per_round=8,
    )
    first = fit_global_site_rank_one_attention_transport_generator_bank(
        captures,
        config=config,
        shared_rank_one_fit=shared_fit,
    )
    second = fit_global_site_rank_one_attention_transport_generator_bank(
        captures,
        config=config,
        shared_rank_one_fit=shared_fit,
    )

    assert first.bank.artifact_sha256 == second.bank.artifact_sha256
    assert first.fit_round_relative_fisher_losses == (
        second.fit_round_relative_fisher_losses
    )
    assert first.fit_round_global_relative_fisher_losses == (
        second.fit_round_global_relative_fisher_losses
    )
    assert first.fit_round_site_relative_fisher_losses == (
        second.fit_round_site_relative_fisher_losses
    )
    assert first.initial_fit_oracle == shared_fit.final_fit_oracle
    assert first.initial_global_fit_oracle == shared_fit.final_global_fit_oracle
    assert len(first.fit_round_relative_fisher_losses) == 3
    assert len(first.fit_round_global_relative_fisher_losses) == 3
    _assert_monotone(first.fit_round_relative_fisher_losses)
    _assert_monotone(first.fit_round_global_relative_fisher_losses)

    initial_by_site = dict(
        first.initial_global_fit_oracle.per_invocation_relative_fisher_loss
    )
    final_by_site = dict(
        first.final_global_fit_oracle.per_invocation_relative_fisher_loss
    )
    assert tuple(
        invocation_id
        for invocation_id, _history in (
            first.fit_round_site_relative_fisher_losses
        )
    ) == GLOBAL_ATTENTION_INVOCATION_IDS
    for invocation_id, history in first.fit_round_site_relative_fisher_losses:
        assert len(history) == 3
        _assert_monotone(history)
        assert history[0] == pytest.approx(
            initial_by_site[invocation_id],
            abs=1.0e-12,
            rel=1.0e-11,
        )
        assert history[-1] == pytest.approx(
            final_by_site[invocation_id],
            abs=1.0e-12,
            rel=1.0e-11,
        )

    forged_histories = tuple(
        (
            invocation_id,
            tuple(value + 0.25 for value in history),
        )
        for invocation_id, history in first.fit_round_site_relative_fisher_losses
    )
    with pytest.raises(ValueError, match="bank fit"):
        replace(
            first,
            fit_round_site_relative_fisher_losses=forged_histories,
        )


def test_supplied_shared_fit_geometry_and_gauge_mismatches_are_rejected(
    _tiny_shared_fit: tuple[
        tuple[AttentionTransportRowBatch, ...],
        GlobalRankOneAttentionTransportFitConfig,
        object,
    ],
) -> None:
    captures, shared_config, shared_fit = _tiny_shared_fit
    wrong_geometry = GlobalSiteRankOneAttentionTransportFitConfig(
        shared_rank_one=_shared_fit_config(generator_count=2),
        alternating_rounds=1,
        steps_per_site_per_round=1,
    )
    with pytest.raises(ValueError, match="shared rank-one fit"):
        fit_global_site_rank_one_attention_transport_generator_bank(
            captures,
            config=wrong_geometry,
            shared_rank_one_fit=shared_fit,
        )

    wrong_shared_gauge = replace(shared_config, gauge_floor=1.0e-6)
    wrong_gauge = GlobalSiteRankOneAttentionTransportFitConfig(
        shared_rank_one=wrong_shared_gauge,
        alternating_rounds=1,
        steps_per_site_per_round=1,
        gauge_floor=1.0e-6,
    )
    with pytest.raises(ValueError, match="shared rank-one fit"):
        fit_global_site_rank_one_attention_transport_generator_bank(
            captures,
            config=wrong_gauge,
            shared_rank_one_fit=shared_fit,
        )

    with pytest.raises(ValueError, match="must equal"):
        GlobalSiteRankOneAttentionTransportFitConfig(
            shared_rank_one=shared_config,
            gauge_floor=2.0,
        )


@pytest.fixture(scope="module")
def _planted_site_fit() -> tuple[
    tuple[AttentionTransportRowBatch, ...],
    object,
]:
    captures = _tiny_captures(site_specific=True)
    shared_config = _shared_fit_config(
        global_rounds=3,
        global_steps=32,
        learning_rate=2.0e-2,
    )
    shared_fit = fit_global_rank_one_attention_transport_generator_bank(
        captures,
        config=shared_config,
    )
    config = GlobalSiteRankOneAttentionTransportFitConfig(
        shared_rank_one=shared_config,
        alternating_rounds=4,
        steps_per_site_per_round=48,
        learning_rate=2.0e-2,
    )
    fitted = fit_global_site_rank_one_attention_transport_generator_bank(
        captures,
        config=config,
        shared_rank_one_fit=shared_fit,
    )
    return captures, fitted


def test_planted_site_specific_fit_strictly_improves_over_shared_initialization(
    _planted_site_fit: tuple[
        tuple[AttentionTransportRowBatch, ...],
        object,
    ],
) -> None:
    _captures, fitted = _planted_site_fit

    assert fitted.initial_fit_oracle == fitted.shared_rank_one_fit.final_fit_oracle
    assert (
        fitted.initial_global_fit_oracle
        == fitted.shared_rank_one_fit.final_global_fit_oracle
    )
    assert fitted.final_global_fit_oracle.relative_fisher_loss < (
        fitted.initial_global_fit_oracle.relative_fisher_loss - 1.0e-3
    )
    histories = dict(fitted.fit_round_site_relative_fisher_losses)
    assert histories["mutation.23"][-1] < histories["mutation.23"][0] - 1.0e-3
    assert not torch.equal(
        fitted.bank.global_site_residual_left_vectors[0],
        fitted.bank.global_site_residual_left_vectors[1],
    )


def test_site_coupling_diagnostic_is_nested_closed_and_metrics_only(
    _planted_site_fit: tuple[
        tuple[AttentionTransportRowBatch, ...],
        object,
    ],
) -> None:
    captures, fitted = _planted_site_fit
    diagnostic = evaluate_global_site_rank_one_coupling_diagnostic(
        fitted.bank,
        captures,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    assert tuple(site.invocation_id for site in diagnostic.per_site) == (
        GLOBAL_ATTENTION_INVOCATION_IDS
    )
    for scope in (diagnostic, *diagnostic.per_site):
        tolerance = 1.0e-11 * max(1.0, scope.tied.fisher_loss)
        assert scope.fixed_support_untied.fisher_loss <= (
            scope.tied.fisher_loss + tolerance
        )
        assert scope.reselected_support_untied.fisher_loss <= (
            scope.fixed_support_untied.fisher_loss + tolerance
        )

    total_rows = sum(site.baseline.row_count for site in diagnostic.per_site)
    leave_one = dict(
        diagnostic.leave_one_site_residual_off_relative_fisher_losses
    )
    assert tuple(leave_one) == GLOBAL_ATTENTION_INVOCATION_IDS
    for disabled_id, observed in leave_one.items():
        expected_fisher = sum(
            site.baseline.row_count
            * (
                site.baseline.fisher_loss
                if site.invocation_id == disabled_id
                else site.tied.fisher_loss
            )
            for site in diagnostic.per_site
        ) / total_rows
        assert observed == pytest.approx(
            expected_fisher / diagnostic.baseline.zero_predictor_fisher_loss,
            abs=1.0e-12,
            rel=1.0e-11,
        )

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
            "generator_support",
            "support_mask",
            "targets",
            "score_gradients",
            "head_contexts",
            "operator_outputs",
            "raw_activations",
        }
    )


def test_zero_site_residual_makes_all_diagnostic_oracles_equivalent() -> None:
    captures = tuple(
        capture
        for capture in _tiny_captures(site_specific=True)
        if capture.invocation_id in GLOBAL_ATTENTION_INVOCATION_IDS
    )
    bank = _site_rank_one_bank(
        hierarchical=_hierarchical_bank(
            generator_count=1,
            hidden_width=2,
        )
    )
    diagnostic = evaluate_global_site_rank_one_coupling_diagnostic(
        bank,
        captures,
        active_count=1,
        euclidean_floor=0.1,
        coefficient_ridge=0.0,
    )

    for scope in (diagnostic, *diagnostic.per_site):
        losses = (
            scope.baseline.fisher_loss,
            scope.tied.fisher_loss,
            scope.fixed_support_untied.fisher_loss,
            scope.reselected_support_untied.fisher_loss,
        )
        assert losses == pytest.approx(
            (losses[0],) * 4,
            abs=1.0e-12,
            rel=1.0e-11,
        )
    assert all(
        relative
        == pytest.approx(
            diagnostic.tied.relative_fisher_loss,
            abs=1.0e-12,
            rel=1.0e-11,
        )
        for _invocation_id, relative in (
            diagnostic.leave_one_site_residual_off_relative_fisher_losses
        )
    )


def test_site_coupling_diagnostic_invalid_arguments_fail_closed(
    _planted_site_fit: tuple[
        tuple[AttentionTransportRowBatch, ...],
        object,
    ],
) -> None:
    captures, fitted = _planted_site_fit
    invalid_calls = (
        ({"active_count": False}, "active_count"),
        ({"active_count": 0}, "active_count"),
        ({"active_count": fitted.bank.generator_count + 1}, "active_count"),
        ({"active_count": 1, "euclidean_floor": False}, "euclidean_floor"),
        ({"active_count": 1, "euclidean_floor": 0.0}, "euclidean_floor"),
        (
            {"active_count": 1, "euclidean_floor": float("nan")},
            "euclidean_floor",
        ),
        (
            {"active_count": 1, "coefficient_ridge": False},
            "coefficient_ridge",
        ),
        (
            {"active_count": 1, "coefficient_ridge": -1.0},
            "coefficient_ridge",
        ),
        (
            {"active_count": 1, "coefficient_ridge": float("inf")},
            "coefficient_ridge",
        ),
    )
    for kwargs, message in invalid_calls:
        with pytest.raises(ValueError, match=message):
            evaluate_global_site_rank_one_coupling_diagnostic(
                fitted.bank,
                captures,
                **kwargs,
            )

    incomplete = tuple(
        capture
        for capture in captures
        if capture.invocation_id != "mutation.35"
    )
    with pytest.raises(ValueError, match="omit.*global-attention"):
        evaluate_global_site_rank_one_coupling_diagnostic(
            fitted.bank,
            incomplete,
        )
    with pytest.raises(TypeError, match="global site rank-one"):
        evaluate_global_site_rank_one_coupling_diagnostic(
            fitted.shared_rank_one_fit.bank,  # type: ignore[arg-type]
            captures,
        )


def test_config_rejects_nonfinite_and_invalid_site_fit_controls() -> None:
    shared = _shared_fit_config()
    invalid = (
        ({"alternating_rounds": 0}, "alternating_rounds"),
        ({"steps_per_site_per_round": 0}, "steps_per_site_per_round"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"residual_ridge": float("nan")}, "residual_ridge"),
        ({"gradient_clip_norm": False}, "gradient_clip_norm"),
    )
    for kwargs, message in invalid:
        with pytest.raises(ValueError, match=message):
            GlobalSiteRankOneAttentionTransportFitConfig(
                shared_rank_one=shared,
                **kwargs,
            )

    config = GlobalSiteRankOneAttentionTransportFitConfig(
        shared_rank_one=shared
    )
    assert math.isclose(config.gauge_floor, shared.gauge_floor)
