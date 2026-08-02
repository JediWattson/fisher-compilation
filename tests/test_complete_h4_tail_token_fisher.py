from __future__ import annotations

import torch
import pytest

from fisher_graph.complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
    canonical_orthogonal_complement_rows,
    complete_h4_tail_gate_scores,
    fit_complete_h4_tail_held_family,
    project_complete_h4_tail_prefix,
    project_complete_h4_tail_rows,
)


def _supported() -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )


def _example(
    example_id: str,
    family_id: str,
    tail_rows: torch.Tensor,
    *,
    gradient_scale: float = 1.0,
) -> CompleteH4TailEndpointExample:
    residual = torch.zeros(tail_rows.shape[0], 4, dtype=torch.float64)
    residual[:, 2:] = tail_rows
    gradients = torch.zeros(3, tail_rows.shape[0], 4, dtype=torch.float64)
    gradients[0, :, 2] = gradient_scale
    gradients[1, :, 3] = 2.0 * gradient_scale
    gradients[2, :, 2:] = gradient_scale
    return CompleteH4TailEndpointExample(
        example_id=example_id,
        family_id=family_id,
        residual_rows=residual,
        token_h4_gradients=gradients,
        compensation_target=torch.tensor([0.2, -0.1, 0.3]),
    )


def test_canonical_complement_and_tail_projection_are_exact() -> None:
    supported = torch.tensor(
        [[2.0**-0.5, 2.0**-0.5, 0.0]], dtype=torch.float64
    )
    first = canonical_orthogonal_complement_rows(supported)
    second = canonical_orthogonal_complement_rows(supported.clone())
    assert torch.equal(first, second)
    frame = torch.cat((supported, first), dim=0)
    assert torch.allclose(
        frame @ frame.T, torch.eye(3, dtype=torch.float64), atol=1.0e-10
    )

    residual = torch.tensor([[3.0, 1.0, 5.0]], dtype=torch.float64)
    tail = project_complete_h4_tail_rows(residual, supported)
    assert torch.allclose(
        tail @ supported.T,
        torch.zeros(1, 1, dtype=torch.float64),
        atol=1.0e-10,
    )
    assert torch.allclose((tail @ first.T) @ first, tail, atol=1.0e-10)


def test_endpoint_gate_scores_contract_prompt_local_mode_fields() -> None:
    example = _example(
        "a1",
        "a",
        torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.float64),
    )
    basis = canonical_orthogonal_complement_rows(_supported())
    scores = complete_h4_tail_gate_scores(example, basis)
    assert scores.shape == (3, 2)
    assert torch.equal(
        scores,
        torch.tensor(
            [[7.0, 0.0], [0.0, 20.0], [7.0, 10.0]],
            dtype=torch.float64,
        ),
    )


def test_held_family_never_changes_training_fit_or_fisher_order() -> None:
    training = (
        _example("a1", "a", torch.tensor([[4.0, 1.0], [2.0, 0.5]])),
        _example("a2", "a", torch.tensor([[3.0, 1.0], [1.0, 0.2]])),
        _example("b1", "b", torch.tensor([[2.0, 3.0], [0.5, 2.0]])),
        _example("b2", "b", torch.tensor([[1.0, 4.0], [0.2, 3.0]])),
    )
    held = _example("c1", "c", torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    perturbed_held = _example(
        "c1",
        "c",
        torch.tensor([[1000.0, -700.0], [-300.0, 900.0]]),
        gradient_scale=100.0,
    )
    first = fit_complete_h4_tail_held_family(
        (*training, held), supported_basis=_supported(), held_family_id="c"
    )
    second = fit_complete_h4_tail_held_family(
        (*training, perturbed_held),
        supported_basis=_supported(),
        held_family_id="c",
    )
    assert torch.equal(first.fitted_basis_rows, second.fitted_basis_rows)
    assert first.token_fisher_relevance == second.token_fisher_relevance
    assert first.token_fisher_order == second.token_fisher_order
    assert first.artifact_sha256 == second.artifact_sha256
    assert "c1" not in first.training_example_ids
    assert "c" not in first.training_family_ids

    frame = torch.cat((_supported(), first.fitted_basis_rows), dim=0)
    assert torch.allclose(
        frame @ frame.T, torch.eye(4, dtype=torch.float64), atol=1.0e-9
    )
    held_tail = project_complete_h4_tail_rows(held.residual_rows, _supported())
    full = project_complete_h4_tail_prefix(held_tail, first, rank=first.rank)
    assert torch.allclose(full, held_tail, atol=1.0e-9)


def test_metadata_contains_receipts_and_scalars_but_no_raw_tensors() -> None:
    examples = (
        _example("a1", "a", torch.tensor([[2.0, 0.5]])),
        _example("b1", "b", torch.tensor([[1.0, 2.0]])),
        _example("c1", "c", torch.tensor([[0.5, 1.0]])),
    )
    fit = fit_complete_h4_tail_held_family(
        examples, supported_basis=_supported(), held_family_id="c"
    )
    example_metadata = examples[0].metadata()
    fit_metadata = fit.metadata()
    assert "residual_rows" not in example_metadata
    assert "token_h4_gradients" not in example_metadata
    assert "compensation_target" not in example_metadata
    assert "fitted_basis_rows" not in fit_metadata
    assert fit_metadata["held_family_used_for_fit_or_ordering"] is False
    assert fit_metadata["prompt_mean_fisher_used_for_ordering"] is False


def test_captured_tensors_do_not_alias_and_post_capture_mutation_fails_closed() -> None:
    residual = torch.tensor([[0.0, 0.0, 2.0, 3.0]], dtype=torch.float64)
    gradients = torch.ones(2, 1, 4, dtype=torch.float64)
    target = torch.tensor([0.1, 0.2], dtype=torch.float64)
    example = CompleteH4TailEndpointExample(
        example_id="a1",
        family_id="a",
        residual_rows=residual,
        token_h4_gradients=gradients,
        compensation_target=target,
    )
    residual.mul_(7.0)
    gradients.mul_(11.0)
    target.mul_(13.0)
    assert torch.equal(
        example.residual_rows,
        torch.tensor([[0.0, 0.0, 2.0, 3.0]], dtype=torch.float64),
    )
    example.residual_rows.mul_(7.0)
    with pytest.raises(RuntimeError, match="payload drifted"):
        complete_h4_tail_gate_scores(
            example, canonical_orthogonal_complement_rows(_supported())
        )


def test_fit_binds_training_evidence_and_post_capture_mutation_fails_closed() -> None:
    examples = (
        _example("a1", "a", torch.tensor([[2.0, 0.5]])),
        _example("b1", "b", torch.tensor([[1.0, 2.0]])),
        _example("c1", "c", torch.tensor([[0.5, 1.0]])),
    )
    fit = fit_complete_h4_tail_held_family(
        examples, supported_basis=_supported(), held_family_id="c"
    )
    assert fit.training_example_artifact_sha256s == (
        examples[0].artifact_sha256,
        examples[1].artifact_sha256,
    )
    fit.fitted_basis_rows.mul_(2.0)
    with pytest.raises(RuntimeError, match="payload drifted"):
        fit.ordered_basis_rows()
