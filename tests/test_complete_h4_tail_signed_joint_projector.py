from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from fisher_graph import complete_h4_tail_signed_joint_projector as signed_joint
from fisher_graph.complete_h4_tail_signed_joint_projector import (
    _coordinate_evidence,
    _low_rank_deflate_symmetric_operator,
    _signed_joint_operator,
    complete_h4_tail_signed_joint_prediction,
    complete_h4_tail_signed_joint_scores,
    fit_complete_h4_tail_signed_joint_held_family,
)
from fisher_graph.complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
    canonical_orthogonal_complement_rows,
)


def _supported(width: int = 3) -> torch.Tensor:
    result = torch.zeros(1, width, dtype=torch.float64)
    result[0, 0] = 1.0
    return result


def _one_token_example(
    example_id: str,
    family_id: str,
    *,
    amplitude: Sequence[float],
    gradient: Sequence[float],
    target: float,
) -> CompleteH4TailEndpointExample:
    residual = torch.tensor([amplitude], dtype=torch.float64)
    gradients = torch.tensor([[gradient]], dtype=torch.float64)
    return CompleteH4TailEndpointExample(
        example_id=example_id,
        family_id=family_id,
        residual_rows=residual,
        token_h4_gradients=gradients,
        compensation_target=torch.tensor([target], dtype=torch.float64),
    )


def _multi_token_example(
    example_id: str,
    family_id: str,
    *,
    residual_rows: Sequence[Sequence[float]],
    token_h4_gradients: Sequence[Sequence[Sequence[float]]],
    compensation_target: Sequence[float],
) -> CompleteH4TailEndpointExample:
    return CompleteH4TailEndpointExample(
        example_id=example_id,
        family_id=family_id,
        residual_rows=torch.tensor(residual_rows, dtype=torch.float64),
        token_h4_gradients=torch.tensor(
            token_h4_gradients, dtype=torch.float64
        ),
        compensation_target=torch.tensor(
            compensation_target, dtype=torch.float64
        ),
    )


def _off_diagonal_examples(*, target: float = 1.0):
    common = {
        "amplitude": (0.0, 1.0, 0.0),
        "gradient": (0.0, 0.0, 1.0),
        "target": target,
    }
    return (
        _one_token_example("a1", "a", **common),
        _one_token_example("b1", "b", **common),
        _one_token_example("held1", "held", **common),
    )


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_off_diagonal_signed_joint_rotation_beats_diagonal_and_residual_pca() -> None:
    examples = _off_diagonal_examples()
    fit = fit_complete_h4_tail_signed_joint_held_family(
        examples,
        supported_basis=_supported(),
        held_family_id="held",
        max_directions=2,
    )
    assert fit.rank == 1
    assert fit.stop_reason == "nonpositive_curvature"
    direction = fit.directions_tensor()[0]
    assert torch.allclose(
        direction,
        torch.tensor([0.0, 2.0**-0.5, 2.0**-0.5], dtype=torch.float64),
        atol=1.0e-12,
    )

    # The residual covariance/PCA direction is e1.  Both coordinate axes have
    # zero quadratic response because the useful effect is the cross term.
    diagonal_or_pca = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
    )
    baseline_scores = complete_h4_tail_signed_joint_scores(
        examples[-1], diagonal_or_pca
    )
    assert torch.equal(baseline_scores, torch.zeros_like(baseline_scores))
    prediction = complete_h4_tail_signed_joint_prediction(examples[-1], fit)
    assert fit.final_rmse < 1.0e-10
    assert torch.allclose(
        prediction, examples[-1].compensation_target, atol=1.0e-10
    )


def test_factorized_operator_matches_naive_token_outer_product_reference() -> None:
    examples = (
        _multi_token_example(
            "a1",
            "a",
            residual_rows=((0.0, 1.0, 2.0, -1.0), (0.0, -2.0, 0.5, 3.0)),
            token_h4_gradients=(
                ((0.0, 0.5, -1.0, 2.0), (0.0, 1.0, 0.25, -0.5)),
                ((0.0, -2.0, 1.5, 0.25), (0.0, 0.1, -0.5, 1.0)),
                ((0.0, 3.0, 0.0, -1.0), (0.0, -1.0, 2.0, 0.5)),
            ),
            compensation_target=(0.5, -1.25, 2.0),
        ),
        _multi_token_example(
            "a2",
            "a",
            residual_rows=((0.0, -1.0, 0.25, 2.0),),
            token_h4_gradients=(
                ((0.0, 2.0, -0.5, 1.0),),
                ((0.0, -1.0, 3.0, 0.5),),
            ),
            compensation_target=(-0.75, 1.5),
        ),
        _multi_token_example(
            "b1",
            "b",
            residual_rows=((0.0, 0.5, -3.0, 1.0),),
            token_h4_gradients=(
                ((0.0, 1.0, 2.0, -2.0),),
                ((0.0, 4.0, -1.0, 0.25),),
                ((0.0, -0.5, 0.75, 3.0),),
                ((0.0, 2.5, -2.0, 1.0),),
            ),
            compensation_target=(1.0, -0.5, 0.25, 2.0),
        ),
    )
    complement = canonical_orthogonal_complement_rows(_supported(width=4))
    evidence = _coordinate_evidence(examples, complement)
    residuals = {
        item.example.example_id: item.example.compensation_target
        for item in evidence
    }
    factorized = _signed_joint_operator(evidence, residuals)

    family_prompt_operators: dict[str, list[torch.Tensor]] = {}
    for item in evidence:
        per_token = torch.einsum(
            "ri,trj->tij", item.amplitudes, item.gradients
        )
        per_token = 0.5 * (per_token + per_token.transpose(1, 2))
        prompt = (
            residuals[item.example.example_id][:, None, None] * per_token
        ).mean(dim=0)
        family_prompt_operators.setdefault(item.example.family_id, []).append(
            prompt
        )
    family_means = tuple(
        torch.stack(family_prompt_operators[family]).mean(dim=0)
        for family in sorted(family_prompt_operators)
    )
    naive = torch.stack(family_means).mean(dim=0)
    assert torch.allclose(factorized, naive, rtol=0.0, atol=1.0e-12)


def _dense_deflation_reference(
    operator: torch.Tensor,
    prior_rows: torch.Tensor,
) -> torch.Tensor:
    projector = (
        torch.eye(operator.shape[0], dtype=torch.float64)
        - prior_rows.T @ prior_rows
    )
    result = projector @ operator @ projector
    return (0.5 * (result + result.T)).contiguous()


def test_low_rank_deflation_is_exact_for_coordinate_rows_and_close_randomly() -> None:
    exact_operator = torch.tensor(
        [
            [4.0, 1.0, 2.0, -3.0],
            [1.0, 5.0, -2.0, 1.0],
            [2.0, -2.0, 6.0, 4.0],
            [-3.0, 1.0, 4.0, 7.0],
        ],
        dtype=torch.float64,
    )
    exact_rows = torch.eye(4, dtype=torch.float64)[[0, 2]]
    assert torch.equal(
        _low_rank_deflate_symmetric_operator(exact_operator, exact_rows),
        _dense_deflation_reference(exact_operator, exact_rows),
    )

    for seed, dimension, prior_rank in (
        (1, 3, 1),
        (2, 8, 3),
        (3, 17, 0),
        (4, 17, 8),
        (5, 32, 16),
    ):
        generator = torch.Generator().manual_seed(seed)
        raw = torch.randn(
            (dimension, dimension), generator=generator, dtype=torch.float64
        )
        operator = 0.5 * (raw + raw.T)
        if prior_rank:
            frame, _ = torch.linalg.qr(
                torch.randn(
                    (dimension, prior_rank),
                    generator=generator,
                    dtype=torch.float64,
                ),
                mode="reduced",
            )
            prior = frame.T.contiguous()
        else:
            prior = torch.empty((0, dimension), dtype=torch.float64)
        optimized = _low_rank_deflate_symmetric_operator(operator, prior)
        dense = _dense_deflation_reference(operator, prior)
        scale = max(float(operator.abs().max()), 1.0)
        tolerance = (
            4096.0
            * torch.finfo(torch.float64).eps
            * dimension
            * scale
        )
        assert torch.allclose(optimized, dense, rtol=0.0, atol=tolerance)
        assert torch.equal(optimized, optimized.T)
        if prior_rank:
            assert torch.allclose(
                prior @ optimized,
                torch.zeros_like(prior),
                rtol=0.0,
                atol=tolerance,
            )


def test_low_rank_fit_matches_dense_reference_with_expected_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(9182)
    examples: list[CompleteH4TailEndpointExample] = []
    for family in ("a", "b", "held"):
        for prompt in range(2):
            residual = torch.randn(
                (3, 7), generator=generator, dtype=torch.float64
            )
            residual[:, 0] = 0.0
            gradients = torch.randn(
                (4, 3, 7), generator=generator, dtype=torch.float64
            )
            gradients[:, :, 0] = 0.0
            target = torch.randn(4, generator=generator, dtype=torch.float64)
            examples.append(
                CompleteH4TailEndpointExample(
                    example_id=f"{family}-{prompt}",
                    family_id=family,
                    residual_rows=residual,
                    token_h4_gradients=gradients,
                    compensation_target=target,
                )
            )
    optimized = fit_complete_h4_tail_signed_joint_held_family(
        examples,
        supported_basis=_supported(width=7),
        held_family_id="held",
        max_directions=5,
    )
    optimized_replay = fit_complete_h4_tail_signed_joint_held_family(
        reversed(examples),
        supported_basis=_supported(width=7),
        held_family_id="held",
        max_directions=5,
    )
    assert optimized.artifact_sha256 == optimized_replay.artifact_sha256

    monkeypatch.setattr(
        signed_joint,
        "_low_rank_deflate_symmetric_operator",
        _dense_deflation_reference,
    )
    dense = fit_complete_h4_tail_signed_joint_held_family(
        examples,
        supported_basis=_supported(width=7),
        held_family_id="held",
        max_directions=5,
    )
    assert optimized.rank == dense.rank
    assert optimized.stop_reason == dense.stop_reason
    assert torch.allclose(
        optimized.directions_tensor(),
        dense.directions_tensor(),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert torch.allclose(
        torch.tensor(optimized.gains),
        torch.tensor(dense.gains),
        rtol=0.0,
        atol=1.0e-12,
    )
    # Exact tensor hashes are intentionally not asserted across the two
    # parenthesizations: roundoff may change the final bits even though the
    # fitted subspaces and gains are numerically equivalent.


def test_signed_target_changes_the_selected_rotation_not_only_its_label() -> None:
    positive = fit_complete_h4_tail_signed_joint_held_family(
        _off_diagonal_examples(target=1.0),
        supported_basis=_supported(),
        held_family_id="held",
        max_directions=1,
    )
    negative_examples = _off_diagonal_examples(target=-1.0)
    negative = fit_complete_h4_tail_signed_joint_held_family(
        negative_examples,
        supported_basis=_supported(),
        held_family_id="held",
        max_directions=1,
    )
    assert torch.allclose(
        positive.directions_tensor()[0],
        torch.tensor([0.0, 2.0**-0.5, 2.0**-0.5], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert torch.allclose(
        negative.directions_tensor()[0],
        torch.tensor([0.0, 2.0**-0.5, -(2.0**-0.5)], dtype=torch.float64),
        atol=1.0e-12,
    )
    assert torch.allclose(
        complete_h4_tail_signed_joint_prediction(negative_examples[-1], negative),
        torch.tensor([-1.0], dtype=torch.float64),
        atol=1.0e-10,
    )


def test_held_family_perturbation_and_input_reordering_cannot_change_the_fit() -> None:
    training = _off_diagonal_examples()[:2]
    ordinary_held = _off_diagonal_examples()[-1]
    adversarial_held = _one_token_example(
        "held1",
        "held",
        amplitude=(0.0, 1.0e6, -7.0e5),
        gradient=(0.0, -9.0e4, 3.0e5),
        target=-1.0e9,
    )
    first = fit_complete_h4_tail_signed_joint_held_family(
        (*training, ordinary_held),
        supported_basis=_supported(),
        held_family_id="held",
        max_directions=2,
    )
    second = fit_complete_h4_tail_signed_joint_held_family(
        (adversarial_held, training[1], training[0]),
        supported_basis=_supported(),
        held_family_id="held",
        max_directions=2,
    )
    assert first.artifact_sha256 == second.artifact_sha256
    assert torch.equal(first.directions_tensor(), second.directions_tensor())
    assert first.steps == second.steps
    assert "held" not in first.training_family_ids
    assert "held1" not in first.training_example_ids


def _nested_examples() -> tuple[CompleteH4TailEndpointExample, ...]:
    result: list[CompleteH4TailEndpointExample] = []
    for family in ("a", "b"):
        result.extend(
            (
                _one_token_example(
                    f"{family}1",
                    family,
                    amplitude=(0.0, 1.0, 0.0, 0.0),
                    gradient=(0.0, 1.0, 0.0, 0.0),
                    target=2.0,
                ),
                _one_token_example(
                    f"{family}2",
                    family,
                    amplitude=(0.0, 0.0, 1.0, 0.0),
                    gradient=(0.0, 0.0, 1.0, 0.0),
                    target=1.0,
                ),
            )
        )
    result.append(
        _one_token_example(
            "held1",
            "held",
            amplitude=(0.0, 0.0, 0.0, 1.0),
            gradient=(0.0, 0.0, 0.0, 1.0),
            target=3.0,
        )
    )
    return tuple(result)


def test_prefixes_are_nested_orthonormal_and_stop_deterministically() -> None:
    examples = _nested_examples()
    rank_one = fit_complete_h4_tail_signed_joint_held_family(
        examples,
        supported_basis=_supported(width=4),
        held_family_id="held",
        max_directions=1,
    )
    full = fit_complete_h4_tail_signed_joint_held_family(
        reversed(examples),
        supported_basis=_supported(width=4),
        held_family_id="held",
        max_directions=3,
    )
    replay = fit_complete_h4_tail_signed_joint_held_family(
        examples,
        supported_basis=_supported(width=4),
        held_family_id="held",
        max_directions=3,
    )
    assert rank_one.rank == 1
    assert full.rank == 2
    assert full.stop_reason == "nonpositive_curvature"
    assert torch.equal(
        rank_one.directions_tensor(), full.directions_tensor()[:1]
    )
    assert rank_one.steps == full.steps[:1]
    assert torch.allclose(
        full.directions_tensor() @ full.directions_tensor().T,
        torch.eye(2, dtype=torch.float64),
        atol=1.0e-12,
    )
    assert full.artifact_sha256 == replay.artifact_sha256
    assert full.steps == replay.steps
    assert full.steps[0].rmse_after < full.steps[0].rmse_before
    assert full.steps[1].rmse_after < full.steps[1].rmse_before

    zero_examples = tuple(
        _one_token_example(
            example.example_id,
            example.family_id,
            amplitude=tuple(float(value) for value in example.residual_rows[0]),
            gradient=tuple(
                float(value) for value in example.token_h4_gradients[0, 0]
            ),
            target=0.0,
        )
        for example in examples
    )
    stopped = fit_complete_h4_tail_signed_joint_held_family(
        zero_examples,
        supported_basis=_supported(width=4),
        held_family_id="held",
        max_directions=3,
    )
    assert stopped.rank == 0
    assert stopped.stop_reason == "nonpositive_curvature"
    assert stopped.initial_rmse == stopped.final_rmse == 0.0
    assert torch.equal(
        complete_h4_tail_signed_joint_prediction(zero_examples[-1], stopped),
        torch.zeros(1, dtype=torch.float64),
    )


def test_fit_is_hash_bound_mutation_safe_and_never_serializes_raw_evidence() -> None:
    fit = fit_complete_h4_tail_signed_joint_held_family(
        _off_diagonal_examples(),
        supported_basis=_supported(),
        held_family_id="held",
        max_directions=2,
    )
    metadata = fit.metadata()
    assert not _contains_tensor(metadata)
    rendered = repr(metadata)
    assert "residual_rows" not in rendered
    assert "token_h4_gradients" not in rendered
    assert "compensation_target" not in rendered
    assert metadata["held_family_used_for_direction_gain_or_stop"] is False
    assert metadata["authorizes_serving_or_model_mutation"] is False
    assert fit.training_example_artifact_sha256s == tuple(
        example.artifact_sha256 for example in _off_diagonal_examples()[:2]
    )

    fit.ambient_directions.mul_(2.0)
    with pytest.raises(RuntimeError, match="payload drifted"):
        fit.directions_tensor()
