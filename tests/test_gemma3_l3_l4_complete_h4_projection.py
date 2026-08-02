from dataclasses import FrozenInstanceError

import pytest
import torch

from fisher_graph.gemma3_l3_l4_complete_h4_projection import (
    CompleteH4ProjectionFitSequence,
    ImmutableFloat64Matrix,
    canonical_complete_h4_rank_grid,
    canonicalize_orthonormal_basis_signs,
    fit_complete_h4_projection_basis,
    project_complete_h4_residual_rows,
    summarize_complete_h4_projection_geometry,
)


def _sequence(
    example_id: str,
    family_id: str,
    residual: list[list[float]],
    gradient: list[list[float]] | None = None,
) -> CompleteH4ProjectionFitSequence:
    return CompleteH4ProjectionFitSequence(
        example_id=example_id,
        family_id=family_id,
        residual_rows=torch.tensor(residual, dtype=torch.float64),
        gradient_rows=(
            None
            if gradient is None
            else torch.tensor(gradient, dtype=torch.float64)
        ),
    )


def _assert_tensor_free(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for item in value.values():
            _assert_tensor_free(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_tensor_free(item)


def test_fit_sequence_copies_rows_into_immutable_hash_bound_storage() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sequence = CompleteH4ProjectionFitSequence(
        example_id="example-a",
        family_id="family-a",
        residual_rows=source,
    )

    assert isinstance(sequence.residual_rows, ImmutableFloat64Matrix)
    original_receipt = sequence.sequence_sha256
    source.zero_()
    assert torch.equal(
        sequence.residual_rows.to_tensor(),
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
    )
    exported = sequence.residual_rows.to_tensor()
    exported.zero_()
    assert sequence.sequence_sha256 == original_receipt
    assert sequence.residual_rows.to_tensor().abs().sum().item() == 10.0
    _assert_tensor_free(sequence.metadata())
    with pytest.raises(FrozenInstanceError):
        sequence.example_id = "changed"  # type: ignore[misc]


def test_fit_is_canonical_family_example_macro_and_order_independent() -> None:
    # Equal-example weighting makes the single large e1 example dominate the
    # hundred-row e0 example.  Pooling rows would incorrectly pick e0.
    e0_rows = [[1.0, 0.0] for _ in range(100)]
    examples = (
        _sequence("long", "family-a", e0_rows),
        _sequence("short", "family-a", [[0.0, 2.0]]),
    )

    forward = fit_complete_h4_projection_basis(examples, max_rank=2)
    reverse = fit_complete_h4_projection_basis(reversed(examples), max_rank=2)

    assert forward.artifact_sha256 == reverse.artifact_sha256
    assert forward.source_example_ids == ("long", "short")
    first = forward.basis_tensor()[0]
    assert torch.allclose(first, torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert forward.residual_eigenvalues == pytest.approx((2.0, 0.5))
    assert forward.directional_fisher is None
    assert forward.fisher_rank_order is None


def test_established_fisher_alignment_tilt_changes_fit_direction() -> None:
    # Raw residual energy favors e1 (1.0 > 0.81), while the established
    # 1+cos^2 tilt doubles aligned e0 energy and makes e0 the first direction.
    sequence = _sequence(
        "tilted",
        "family-a",
        [[0.9, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    basis = fit_complete_h4_projection_basis((sequence,), max_rank=2)

    assert torch.allclose(
        basis.basis_tensor()[0],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert basis.residual_eigenvalues == pytest.approx((0.81, 0.5))
    assert basis.directional_residual_variance == pytest.approx((0.405, 0.5))
    assert basis.directional_fisher == pytest.approx((1.0, 0.0))
    assert basis.fisher_relevance == pytest.approx((0.405, 0.0))
    assert basis.fisher_rank_order == (0, 1)
    assert basis.metadata()["euclidean_basis_reordered_by_fisher"] is False


def test_unweighted_fit_uses_distinct_basis_and_keeps_fisher_diagnostics() -> None:
    sequence = _sequence(
        "weighted-comparison",
        "family-a",
        [[0.9, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
    )

    tilted = fit_complete_h4_projection_basis((sequence,), max_rank=2)
    unweighted = fit_complete_h4_projection_basis(
        (sequence,),
        max_rank=2,
        fit_weighting="unweighted",
    )

    assert tilted.fit_weighting == "fisher_alignment_tilted"
    assert unweighted.fit_weighting == "unweighted"
    assert torch.equal(
        tilted.basis_tensor()[0],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    assert torch.equal(
        unweighted.basis_tensor()[0],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    assert unweighted.directional_fisher is not None
    assert unweighted.fisher_relevance is not None
    assert unweighted.artifact_sha256 != tilted.artifact_sha256
    assert unweighted.metadata()["basis_fit_covariance"] == (
        "family_example_macro_unweighted_residual_second_moment"
    )
    assert tilted.metadata()["basis_fit_covariance"] == (
        "family_example_macro_residual_second_moment_with_"
        "one_plus_cosine_squared_fisher_alignment_tilt"
    )


def test_fit_rejects_unknown_weighting() -> None:
    sequence = _sequence("plain", "family", [[1.0, 0.0]])

    with pytest.raises(ValueError, match="fit_weighting"):
        fit_complete_h4_projection_basis(
            (sequence,),
            fit_weighting="unknown",  # type: ignore[arg-type]
        )


def test_fisher_ranking_is_separate_from_euclidean_basis_order() -> None:
    sequence = _sequence(
        "separate-order",
        "family-a",
        [[3.0, 0.0], [0.0, 1.0]],
        [[0.0, 10.0], [0.1, 0.0]],
    )
    basis = fit_complete_h4_projection_basis((sequence,), max_rank=2)

    assert torch.allclose(
        basis.basis_tensor(ordering="euclidean"),
        torch.eye(2, dtype=torch.float64),
    )
    assert basis.fisher_rank_order == (1, 0)
    assert torch.allclose(
        basis.basis_tensor(ordering="fisher"),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
    )
    rows = torch.tensor([[3.0, 1.0]], dtype=torch.float64)
    assert torch.equal(
        project_complete_h4_residual_rows(
            rows,
            basis,
            rank=1,
            ordering="euclidean",
        ),
        torch.tensor([[3.0, 0.0]], dtype=torch.float64),
    )
    assert torch.equal(
        project_complete_h4_residual_rows(
            rows,
            basis,
            rank=1,
            ordering="fisher",
        ),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
    )


def test_rank_grid_geometry_reports_capacity_and_first_order_error() -> None:
    sequences = (
        _sequence(
            "example-a",
            "family-a",
            [[3.0, 0.0], [3.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ),
        _sequence(
            "example-b",
            "family-b",
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ),
    )
    basis = fit_complete_h4_projection_basis(sequences, max_rank=2)

    geometry = summarize_complete_h4_projection_geometry(
        sequences,
        basis,
        ranks=(1, 2),
    )

    rank1, rank2 = geometry.rank_rows
    assert rank1.coefficient_count == 2
    assert rank1.family_balanced_residual_energy_retention == pytest.approx(0.9)
    assert rank1.row_weighted_residual_energy_retention == pytest.approx(18.0 / 19.0)
    assert rank1.family_balanced_residual_rmse == pytest.approx(0.5)
    assert rank1.row_weighted_residual_rmse == pytest.approx((1.0 / 6.0) ** 0.5)
    assert rank1.fisher_first_order_residual_coupling == pytest.approx(2.0)
    assert rank1.fisher_first_order_error_coupling == pytest.approx(0.5)
    assert rank1.fisher_absolute_first_order_residual_coupling == pytest.approx(2.0)
    assert rank1.fisher_absolute_first_order_error_coupling == pytest.approx(0.5)
    assert rank2.family_balanced_residual_energy_retention == pytest.approx(1.0)
    assert rank2.family_balanced_residual_rmse == pytest.approx(0.0, abs=1e-14)
    assert rank2.fisher_first_order_error_coupling == pytest.approx(0.0, abs=1e-14)
    _assert_tensor_free(geometry.to_dict())
    assert geometry.to_dict()["safety"] == {
        "raw_prompts_retained": False,
        "raw_token_ids_retained": False,
        "raw_logits_retained": False,
        "row_level_activations_retained": False,
        "model_weights_retained": False,
    }


def test_sign_canonicalization_uses_first_maximum_absolute_pivot() -> None:
    rows = torch.tensor(
        [[-2.0**-0.5, 2.0**-0.5], [2.0**-0.5, 2.0**-0.5]],
        dtype=torch.float64,
    )
    canonical = canonicalize_orthonormal_basis_signs(rows)
    assert canonical[0, 0] > 0.0
    assert torch.equal(canonical.abs(), rows.abs())


def test_rank_grid_caps_at_width_and_removes_duplicates() -> None:
    assert canonical_complete_h4_rank_grid(40) == (8, 16, 32, 40)
    assert canonical_complete_h4_rank_grid(4) == (4,)
    assert canonical_complete_h4_rank_grid(64) == (8, 16, 32, 64)


@pytest.mark.parametrize(
    ("residual", "gradient", "match"),
    [
        (torch.ones(2), None, "shape"),
        (torch.tensor([[float("nan")]]), None, "finite"),
        (torch.ones((2, 2), dtype=torch.int64), None, "floating"),
        (torch.ones((2, 2)), torch.ones((1, 2)), "match"),
    ],
)
def test_fit_sequence_rejects_invalid_tensor_geometry(
    residual: torch.Tensor,
    gradient: torch.Tensor | None,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        CompleteH4ProjectionFitSequence(
            example_id="example",
            family_id="family",
            residual_rows=residual,
            gradient_rows=gradient,
        )


def test_fit_rejects_duplicate_ids_mixed_gradients_and_zero_energy() -> None:
    plain = _sequence("same", "family-a", [[1.0, 0.0]])
    gradient = _sequence(
        "other",
        "family-b",
        [[0.0, 1.0]],
        [[0.0, 1.0]],
    )
    duplicate = _sequence("same", "family-b", [[0.0, 1.0]])

    with pytest.raises(ValueError, match="unique"):
        fit_complete_h4_projection_basis((plain, duplicate))
    with pytest.raises(ValueError, match="every sequence or none"):
        fit_complete_h4_projection_basis((plain, gradient))
    with pytest.raises(ValueError, match="zero family-balanced energy"):
        fit_complete_h4_projection_basis(
            (_sequence("zero", "family-a", [[0.0, 0.0]]),)
        )


def test_projection_rejects_bad_rank_width_order_and_missing_fisher() -> None:
    sequence = _sequence("plain", "family", [[1.0, 0.0]])
    basis = fit_complete_h4_projection_basis((sequence,), max_rank=2)

    with pytest.raises(ValueError, match="exceeds"):
        project_complete_h4_residual_rows(sequence.residual_rows, basis, rank=3)
    with pytest.raises(ValueError, match="width"):
        project_complete_h4_residual_rows(torch.ones((1, 3)), basis, rank=1)
    with pytest.raises(ValueError, match="Fisher ordering"):
        project_complete_h4_residual_rows(
            sequence.residual_rows,
            basis,
            rank=1,
            ordering="fisher",
        )
