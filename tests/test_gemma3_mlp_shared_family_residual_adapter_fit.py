from __future__ import annotations

import pytest
import torch

from fisher_graph.gemma3_mlp_shared_family_residual_adapter_fit import (
    INDEPENDENT_MEMBER_TARGET,
    PRIMARY_RESIDUAL_TARGET,
    DirectedMLPPairCandidate,
    ZeroResidualPairScore,
    fit_fisher_weighted_reduced_rank_projection,
    fit_mlp_projection_factor_catalog,
    score_fisher_weighted_zero_residual_pair,
    select_directed_mlp_family_pair,
    select_directed_mlp_pair,
)


def _projection_inputs() -> tuple[torch.Tensor, ...]:
    design = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, -1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    anchor = torch.tensor(
        [
            [0.5, -0.1, 0.3],
            [0.2, 0.4, -0.3],
        ],
        dtype=torch.float32,
    )
    residual = torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [-0.5, -1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    member = anchor + residual
    fisher = torch.tensor([1.0, 3.0, 0.5, 2.0], dtype=torch.float32)
    return design, anchor, member, fisher


def test_projection_fit_is_deterministic_canonical_and_rank_specific() -> None:
    design, anchor, member, fisher = _projection_inputs()
    first = fit_fisher_weighted_reduced_rank_projection(
        projection_name="gate_proj",
        design=design,
        member_weight=member,
        anchor_weight=anchor,
        row_fisher_weights=fisher,
        requested_rank=2,
    )
    second = fit_fisher_weighted_reduced_rank_projection(
        projection_name="gate_proj",
        design=design,
        member_weight=member,
        anchor_weight=anchor,
        row_fisher_weights=fisher,
        requested_rank=2,
    )

    assert first.U.shape == (2, 2)
    assert first.V.shape == (2, 3)
    assert first.U.dtype == first.V.dtype == member.dtype
    assert torch.equal(first.U, second.U)
    assert torch.equal(first.V, second.V)
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert first.receipt.target_kind == PRIMARY_RESIDUAL_TARGET
    assert first.receipt.relative_ridge == 1.0e-4
    assert first.receipt.solver_dtype == "torch.float64"
    assert first.receipt.factor_gauge == (
        "sqrt_singular_value_canonical_output_sign"
    )
    for component in range(first.receipt.effective_rank):
        column = first.U[:, component]
        pivot = int(column.abs().argmax().item())
        assert float(column[pivot]) > 0.0

    U, V, receipt = first
    assert U is first.U
    assert V is first.V
    assert receipt is first.receipt


def test_projection_fit_optimizes_the_rank_constrained_ridge_objective() -> None:
    generator = torch.Generator().manual_seed(164)
    design = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    design[:, 0].mul_(100.0)
    design[:, 1].mul_(0.01)
    design[:, 2].mul_(10.0)
    member = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    fisher = torch.rand(8, generator=generator, dtype=torch.float64).add_(0.1)
    relative_ridge = 1.0
    fitted = fit_fisher_weighted_reduced_rank_projection(
        projection_name="gate_proj",
        design=design,
        member_weight=member,
        anchor_weight=torch.zeros_like(member),
        row_fisher_weights=fisher,
        requested_rank=1,
        relative_ridge=relative_ridge,
    )

    weights = fisher / fisher.mean()
    weighted_design = weights.sqrt().unsqueeze(1) * design
    weighted_target = weighted_design @ member.T
    gram = weighted_design.T @ weighted_design
    ridge = relative_ridge * torch.trace(gram) / design.shape[1]

    # The superseded target-PCA path chooses output coordinates before the
    # ridge geometry is known.  It is a useful deterministic counterexample:
    # the true whitened-coefficient RRR solution must have lower objective.
    left, singular, right = torch.linalg.svd(
        weighted_target,
        full_matrices=False,
    )
    scale = singular[:1].sqrt()
    old_u = right[:1].T * scale
    old_scores = left[:, :1] * scale
    old_v = torch.linalg.solve(
        gram + ridge * torch.eye(design.shape[1], dtype=torch.float64),
        weighted_design.T @ old_scores,
    ).T

    def objective(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        coefficient = u @ v
        error = weighted_target - weighted_design @ coefficient.T
        return error.square().sum() + ridge * coefficient.square().sum()

    observed = objective(fitted.U, fitted.V)
    target_pca = objective(old_u, old_v)
    assert observed < target_pca

    ridge_gram = gram + ridge * torch.eye(
        design.shape[1],
        dtype=torch.float64,
    )
    cholesky = torch.linalg.cholesky(ridge_gram)
    full = torch.cholesky_solve(
        weighted_design.T @ weighted_target,
        cholesky,
    )
    whitened = cholesky.T @ full
    left, singular, right = torch.linalg.svd(whitened, full_matrices=False)
    expected = torch.linalg.solve_triangular(
        cholesky.T,
        left[:, :1] * singular[:1],
        upper=True,
    ) @ right[:1]
    torch.testing.assert_close(
        (fitted.U @ fitted.V).T,
        expected,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_requested_rank_above_row_and_numerical_rank_is_zero_padded() -> None:
    design = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    anchor = torch.zeros(3, 3, dtype=torch.float64)
    # All output rows are multiples, so the weighted target is rank one.
    member = torch.tensor(
        [
            [-1.0, 2.0, 0.5],
            [-2.0, 4.0, 1.0],
            [3.0, -6.0, -1.5],
        ],
        dtype=torch.float64,
    )
    fitted = fit_fisher_weighted_reduced_rank_projection(
        projection_name="up_proj",
        design=design,
        member_weight=member,
        anchor_weight=anchor,
        row_fisher_weights=torch.ones(2, dtype=torch.float64),
        requested_rank=5,
    )

    assert fitted.U.shape == (3, 5)
    assert fitted.V.shape == (5, 3)
    assert fitted.receipt.numerical_rank == 1
    assert fitted.receipt.effective_rank == 1
    assert fitted.receipt.zero_padded_rank == 4
    assert torch.count_nonzero(fitted.U[:, 1:]) == 0
    assert torch.count_nonzero(fitted.V[1:, :]) == 0
    pivot = int(fitted.U[:, 0].abs().argmax().item())
    assert float(fitted.U[pivot, 0]) > 0.0


def test_primary_and_independent_targets_are_not_conflated() -> None:
    design, anchor, member, fisher = _projection_inputs()
    primary = fit_fisher_weighted_reduced_rank_projection(
        projection_name="down_proj",
        design=design,
        member_weight=member,
        anchor_weight=anchor,
        row_fisher_weights=fisher,
        rank=2,
    )
    independent = fit_fisher_weighted_reduced_rank_projection(
        projection_name="down_proj",
        design=design,
        member_weight=member,
        anchor_weight=anchor,
        row_fisher_weights=fisher,
        rank=2,
        independent_nonsharing=True,
    )

    assert primary.receipt.target_kind == PRIMARY_RESIDUAL_TARGET
    assert independent.receipt.target_kind == INDEPENDENT_MEMBER_TARGET
    assert primary.receipt.evidence_sha256 != independent.receipt.evidence_sha256
    assert not torch.equal(primary.U @ primary.V, independent.U @ independent.V)


def test_projection_catalog_is_directly_compiler_compatible() -> None:
    design, anchor, member, fisher = _projection_inputs()
    names = ("gate_proj", "up_proj", "down_proj")
    catalog = fit_mlp_projection_factor_catalog(
        projection_designs={name: design for name in names},
        member_projection_weights={name: member for name in names},
        anchor_projection_weights={name: anchor for name in names},
        projection_row_fisher_weights=fisher,
        requested_rank=2,
    )

    assert tuple(catalog) == names
    assert set(catalog.projection_factors) == set(names)
    assert set(catalog.receipts) == set(names)
    assert catalog["gate_proj"][0].shape == (2, 2)
    assert catalog["gate_proj"][1].shape == (2, 3)
    assert catalog.to_dict()["projection_order"] == list(names)


def test_zero_residual_score_is_an_equal_family_mean() -> None:
    design = torch.arange(12, dtype=torch.float32).view(4, 3)
    member = torch.ones(4, 2)
    # Family small has relative SSE 1; family large has relative SSE 0.
    anchor = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    evidence = score_fisher_weighted_zero_residual_pair(
        aligned_design=design,
        exact_member_delta=member,
        anchor_core_delta=anchor,
        row_fisher_weights=torch.tensor([7.0, 1.0, 5.0, 9.0]),
        family_ids=("small", "large", "large", "large"),
    )

    assert evidence.score == pytest.approx(0.5)
    assert evidence.family_count == 2
    scores = {item.family_id: item for item in evidence.family_scores}
    assert scores["small"].relative_sse == pytest.approx(1.0)
    assert scores["large"].relative_sse == pytest.approx(0.0)
    # Weight normalization is within each family, not across the whole panel.
    assert scores["small"].normalized_weight_sum == pytest.approx(1.0)
    assert scores["large"].normalized_weight_sum == pytest.approx(3.0)


def _evidence(*, score: float, digest: str) -> ZeroResidualPairScore:
    baseline = score_fisher_weighted_zero_residual_pair(
        aligned_design=torch.ones(1, 1),
        exact_member_delta=torch.ones(1, 1),
        anchor_core_delta=torch.zeros(1, 1),
        row_fisher_weights=torch.ones(1),
        family_ids=("family",),
    )
    return ZeroResidualPairScore(
        score=score,
        numerical_tensor_sha256=digest,
        family_scores=baseline.family_scores,
        row_count=1,
        family_count=1,
    )


def _candidate(
    anchor: str,
    member: str,
    anchor_block: int,
    member_block: int,
    *,
    score: float,
    digest: str,
) -> DirectedMLPPairCandidate:
    return DirectedMLPPairCandidate(
        anchor_invocation_id=anchor,
        member_invocation_id=member,
        anchor_block_ordinal=anchor_block,
        member_block_ordinal=member_block,
        evidence=_evidence(score=score, digest=digest),
    )


def test_selector_uses_ordinals_only_for_eligibility_then_score_and_digest() -> None:
    adjacent_but_better = _candidate(
        "mutation.2",
        "mutation.4",
        0,
        1,
        score=0.01,
        digest="0" * 64,
    )
    digest_winner = _candidate(
        "mutation.6",
        "mutation.12",
        2,
        5,
        score=0.25,
        digest="1" * 64,
    )
    other = _candidate(
        "mutation.8",
        "mutation.16",
        3,
        7,
        score=0.25,
        digest="2" * 64,
    )

    assert select_directed_mlp_pair(
        (adjacent_but_better, other, digest_winner)
    ) is digest_winner
    assert select_directed_mlp_pair(
        (digest_winner, adjacent_but_better, other)
    ) is digest_winner
    assert select_directed_mlp_family_pair(
        (other, digest_winner)
    ) is digest_winner

    # Changing eligible distances cannot affect the numerical result.
    farther = DirectedMLPPairCandidate(
        anchor_invocation_id=digest_winner.anchor_invocation_id,
        member_invocation_id=digest_winner.member_invocation_id,
        anchor_block_ordinal=0,
        member_block_ordinal=17,
        evidence=digest_winner.evidence,
    )
    assert select_directed_mlp_pair((other, farther)) is farther


def test_selector_fails_closed_without_a_numerical_tie_break() -> None:
    first = _candidate(
        "mutation.2",
        "mutation.8",
        0,
        3,
        score=0.25,
        digest="a" * 64,
    )
    second = _candidate(
        "mutation.4",
        "mutation.10",
        1,
        4,
        score=0.25,
        digest="a" * 64,
    )
    with pytest.raises(RuntimeError, match="numerical tensor digest"):
        select_directed_mlp_pair((first, second))


def test_projection_fit_rejects_degenerate_design_and_bad_weights() -> None:
    with pytest.raises(ValueError, match="no positive finite energy"):
        fit_fisher_weighted_reduced_rank_projection(
            projection_name="gate_proj",
            design=torch.zeros(2, 2),
            member_weight=torch.eye(2),
            anchor_weight=torch.zeros(2, 2),
            row_fisher_weights=torch.ones(2),
            requested_rank=1,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        fit_fisher_weighted_reduced_rank_projection(
            projection_name="gate_proj",
            design=torch.eye(2),
            member_weight=torch.eye(2),
            anchor_weight=torch.zeros(2, 2),
            row_fisher_weights=torch.tensor([1.0, -1.0]),
            requested_rank=1,
        )
