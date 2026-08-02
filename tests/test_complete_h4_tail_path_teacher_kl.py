from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from fisher_graph.causal_edge_transport import (
    _gauss_legendre_unit_interval,
    gauss_legendre_unit_interval,
)
from fisher_graph.complete_h4_tail_path_teacher_kl import (
    CompleteH4TailPathTeacherKLAccumulator,
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
    complete_h4_tail_path_as_endpoint_example,
    complete_h4_tail_path_basis_contraction,
    complete_h4_tail_path_direct_contraction,
    complete_h4_tail_path_family_prompt_token_mean,
    complete_h4_tail_path_ftc_target,
    complete_h4_tail_path_gate_scores,
    complete_h4_tail_path_weighted_gradient,
    summarize_complete_h4_tail_path_ftc_closure,
)
from fisher_graph.complete_h4_tail_token_fisher import (
    CompleteH4TailEndpointExample,
)


_HASHES = tuple(f"{index:064x}" for index in range(1, 13))


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _evidence(
    example_id: str = "example",
    family_id: str = "family",
    *,
    residual_rows: torch.Tensor | None = None,
    integrated_gradient: torch.Tensor | None = None,
    source_kl: torch.Tensor | None = None,
    native_kl: torch.Tensor | None = None,
):
    residual = (
        torch.tensor([[2.0, -1.0, 0.5], [0.25, 3.0, -2.0]])
        if residual_rows is None
        else residual_rows
    )
    source = (
        torch.tensor([1.0, 2.0]) if source_kl is None else source_kl
    )
    native = torch.zeros_like(source) if native_kl is None else native_kl
    desired = (
        torch.tensor(
            [
                [[0.5, 1.0, -0.25], [2.0, -1.0, 0.75]],
                [[-1.0, 0.25, 2.0], [0.5, 0.125, -0.75]],
            ]
        )
        if integrated_gradient is None
        else integrated_gradient
    )
    accumulator = CompleteH4TailPathTeacherKLAccumulator(
        example_id=example_id,
        family_id=family_id,
        residual_rows=residual,
        source_token_teacher_kl=source,
        native_token_teacher_kl=native,
    )
    for index, (node, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS)
    ):
        accumulator.add_node(
            node_index=index,
            path_fraction=node,
            quadrature_weight=weight,
            token_h4_gradients=desired,
            token_teacher_kl=(1.0 - node) * source + node * native,
            vjp_artifact_sha256=_HASHES[index * 3],
            provider_artifact_sha256=_HASHES[index * 3 + 1],
            execution_artifact_sha256=_HASHES[index * 3 + 2],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    return accumulator.finalize()


def test_public_gl4_rule_has_exact_literals_and_private_compatibility() -> None:
    expected_nodes = (
        0.06943184420297371,
        0.33000947820757187,
        0.6699905217924281,
        0.9305681557970262,
    )
    expected_weights = (
        0.17392742256872692,
        0.32607257743127305,
        0.32607257743127305,
        0.17392742256872692,
    )
    assert gauss_legendre_unit_interval(4) == (expected_nodes, expected_weights)
    assert _gauss_legendre_unit_interval(4) == (
        expected_nodes,
        expected_weights,
    )
    assert GL4_UNIT_INTERVAL_NODES == expected_nodes
    assert GL4_UNIT_INTERVAL_WEIGHTS == expected_weights
    assert tuple(value.hex() for value in GL4_UNIT_INTERVAL_NODES) == tuple(
        value.hex() for value in expected_nodes
    )
    assert sum(GL4_UNIT_INTERVAL_WEIGHTS) == 1.0


def test_streaming_nodes_use_exact_weights_and_retain_only_the_integral() -> None:
    residual = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    source = torch.tensor([2.0, 4.0], dtype=torch.float32)
    native = torch.zeros(2, dtype=torch.float32)
    node_gradients = tuple(
        torch.full((2, 1, 2), float(index + 1), dtype=torch.float32)
        for index in range(4)
    )
    accumulator = CompleteH4TailPathTeacherKLAccumulator(
        example_id="stream",
        family_id="a",
        residual_rows=residual,
        source_token_teacher_kl=source,
        native_token_teacher_kl=native,
    )
    for index, gradient in enumerate(node_gradients):
        accumulator.add_node(
            node_index=index,
            path_fraction=GL4_UNIT_INTERVAL_NODES[index],
            quadrature_weight=GL4_UNIT_INTERVAL_WEIGHTS[index],
            token_h4_gradients=gradient,
            token_teacher_kl=(1.0 - GL4_UNIT_INTERVAL_NODES[index]) * source,
            vjp_artifact_sha256=_HASHES[index * 3],
            provider_artifact_sha256=_HASHES[index * 3 + 1],
            execution_artifact_sha256=_HASHES[index * 3 + 2],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    expected = sum(
        weight * gradient.double()
        for weight, gradient in zip(
            GL4_UNIT_INTERVAL_WEIGHTS, node_gradients
        )
    )
    for gradient in node_gradients:
        gradient.fill_(1000.0)
    residual.fill_(1000.0)
    source.fill_(1000.0)
    evidence = accumulator.finalize()
    assert torch.allclose(
        complete_h4_tail_path_weighted_gradient(evidence),
        expected,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert evidence.residual_rows.tolist() == [[1.0, 2.0]]
    assert evidence.source_token_teacher_kl.tolist() == [2.0, 4.0]
    assert len(evidence.node_receipts) == 4
    assert not hasattr(evidence, "path_token_h4_gradients")
    assert evidence.metadata()["full_node_gradient_banks_retained"] is False
    with pytest.raises(RuntimeError, match="sealed"):
        accumulator.finalize()


def test_direct_and_complete_basis_contractions_are_identical() -> None:
    evidence = _evidence()
    basis = torch.linalg.qr(
        torch.tensor(
            [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5], [0.0, -1.0, 2.0]],
            dtype=torch.float64,
        ).T
    ).Q.T.contiguous()
    direct = complete_h4_tail_path_direct_contraction(evidence)
    scores = complete_h4_tail_path_gate_scores(evidence, basis)
    through_basis = complete_h4_tail_path_basis_contraction(evidence, basis)
    assert scores.shape == (evidence.supervised_tokens, 3)
    assert torch.allclose(scores.sum(dim=1), direct, atol=1.0e-12, rtol=0.0)
    assert torch.allclose(through_basis, direct, atol=1.0e-12, rtol=0.0)


def test_direct_and_rectangular_spanning_basis_contractions_are_identical() -> None:
    basis = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]], dtype=torch.float64
    )
    residual_coordinates = torch.tensor(
        [[2.0, -1.0], [0.25, 3.0]], dtype=torch.float64
    )
    residual = residual_coordinates @ basis
    integrated = torch.tensor(
        [
            [[0.5, 1.0, -0.25], [2.0, -1.0, 0.75]],
            [[-1.0, 0.25, 2.0], [0.5, 0.125, -0.75]],
        ],
        dtype=torch.float64,
    )
    evidence = _evidence(
        residual_rows=residual,
        integrated_gradient=integrated,
    )

    direct = complete_h4_tail_path_direct_contraction(evidence)
    through_basis = complete_h4_tail_path_basis_contraction(evidence, basis)
    assert torch.allclose(through_basis, direct, atol=1.0e-12, rtol=0.0)


def test_ftc_target_orientation_and_exact_linear_closure() -> None:
    residual = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    source = torch.tensor([3.0, 6.0], dtype=torch.float64)
    native = torch.zeros_like(source)
    # residual dot gradient is exactly [-3, -6].
    integrated = torch.tensor(
        [[[-1.5, 0.0]], [[-3.0, 0.0]]], dtype=torch.float64
    )
    evidence = _evidence(
        residual_rows=residual,
        integrated_gradient=integrated,
        source_kl=source,
        native_kl=native,
    )
    assert torch.equal(
        complete_h4_tail_path_ftc_target(evidence),
        torch.tensor([-3.0, -6.0], dtype=torch.float64),
    )
    assert torch.allclose(
        complete_h4_tail_path_direct_contraction(evidence),
        complete_h4_tail_path_ftc_target(evidence),
        rtol=0.0,
        atol=1.0e-15,
    )
    closure = summarize_complete_h4_tail_path_ftc_closure((evidence,))
    assert closure.rmse == pytest.approx(0.0, abs=1.0e-15)
    assert closure.relative_rmse == pytest.approx(0.0, abs=1.0e-15)
    assert closure.cosine == pytest.approx(1.0, abs=1.0e-15)
    assert closure.metadata()["FTC_orientation"] == (
        "native_KL_minus_source_KL_equals_path_integral"
    )


def test_family_prompt_token_mean_does_not_pool_unequal_groups() -> None:
    def small(example_id: str, family_id: str, token_count: int):
        source = torch.ones(token_count)
        return _evidence(
            example_id,
            family_id,
            residual_rows=torch.ones(1, 1),
            integrated_gradient=torch.ones(token_count, 1, 1),
            source_kl=source,
            native_kl=torch.zeros_like(source),
        )

    a1 = small("a1", "a", 1)
    a2 = small("a2", "a", 3)
    b1 = small("b1", "b", 2)
    values = {
        "a1": torch.tensor([0.0]),
        "a2": torch.tensor([1.0, 2.0, 3.0]),
        "b1": torch.tensor([10.0, 10.0]),
    }
    # family a = mean(prompt means 0, 2) = 1; family b = 10; total = 5.5.
    nested = complete_h4_tail_path_family_prompt_token_mean(
        (b1, a2, a1), values
    )
    pooled = torch.cat(tuple(values.values())).mean()
    assert float(nested) == pytest.approx(5.5)
    assert float(nested) != pytest.approx(float(pooled))
    closure = summarize_complete_h4_tail_path_ftc_closure((b1, a2, a1))
    assert tuple(row.family_id for row in closure.family_summaries) == ("a", "b")
    assert closure.metadata()["weighting"] == (
        "equal_family_then_equal_prompt_then_equal_token"
    )


def test_inputs_and_artifacts_are_mutation_safe_and_tamper_evident() -> None:
    evidence = _evidence()
    metadata = evidence.metadata()
    assert not _contains_tensor(metadata)
    rendered = repr(metadata)
    assert "tensor([" not in rendered
    assert metadata["raw_evidence_serialized"] is False
    assert all(
        receipt["raw_node_gradient_or_token_KL_serialized"] is False
        for receipt in metadata["node_receipts"]
    )

    returned = complete_h4_tail_path_weighted_gradient(evidence)
    returned.zero_()
    evidence.validate_integrity()
    assert not torch.equal(returned, evidence.integrated_token_h4_gradients)

    evidence.integrated_token_h4_gradients.add_(1.0)
    with pytest.raises(RuntimeError, match="evidence drifted"):
        evidence.validate_integrity()

    clean = _evidence(example_id="clean")
    object.__setattr__(
        clean.node_receipts[0], "token_teacher_kl_mean", 123.0
    )
    with pytest.raises(RuntimeError, match="node receipt drifted"):
        clean.validate_integrity()


def test_path_and_endpoint_semantics_are_explicitly_separate() -> None:
    evidence = _evidence()
    endpoint = CompleteH4TailEndpointExample(
        example_id="endpoint",
        family_id="family",
        residual_rows=evidence.residual_rows,
        token_h4_gradients=evidence.integrated_token_h4_gradients,
        compensation_target=complete_h4_tail_path_ftc_target(evidence),
    )
    with pytest.raises(TypeError, match="path teacher-KL evidence"):
        complete_h4_tail_path_direct_contraction(endpoint)  # type: ignore[arg-type]

    adapted = complete_h4_tail_path_as_endpoint_example(evidence)
    assert isinstance(adapted, CompleteH4TailEndpointExample)
    assert torch.equal(
        adapted.token_h4_gradients, evidence.integrated_token_h4_gradients
    )
    assert torch.equal(
        adapted.compensation_target, complete_h4_tail_path_ftc_target(evidence)
    )
    metadata = evidence.metadata()
    assert metadata["gradient_semantics"] == (
        "GL4_path_integrated_teacher_KL_gradient"
    )
    assert metadata["endpoint_gradient_substituted_for_path_integral"] is False
    assert metadata["finite_boundary_KLs_used_as_integrand_nodes"] is False
    assert all(
        0.0 < receipt.path_fraction < 1.0
        for receipt in evidence.node_receipts
    )


def test_accumulator_rejects_noncanonical_or_incomplete_node_streams() -> None:
    accumulator = CompleteH4TailPathTeacherKLAccumulator(
        example_id="bad",
        family_id="family",
        residual_rows=torch.ones(1, 1),
        source_token_teacher_kl=torch.ones(1),
        native_token_teacher_kl=torch.zeros(1),
    )
    with pytest.raises(ValueError, match="canonical GL4 order"):
        accumulator.add_node(
            node_index=1,
            path_fraction=GL4_UNIT_INTERVAL_NODES[1],
            quadrature_weight=GL4_UNIT_INTERVAL_WEIGHTS[1],
            token_h4_gradients=torch.ones(1, 1, 1),
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_HASHES[0],
            provider_artifact_sha256=_HASHES[1],
            execution_artifact_sha256=_HASHES[2],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    with pytest.raises(RuntimeError, match="all four"):
        accumulator.finalize()

    with pytest.raises(ValueError, match="exact GL4"):
        accumulator.add_node(
            node_index=0,
            path_fraction=GL4_UNIT_INTERVAL_NODES[0] + 1.0e-12,
            quadrature_weight=GL4_UNIT_INTERVAL_WEIGHTS[0],
            token_h4_gradients=torch.ones(1, 1, 1),
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_HASHES[0],
            provider_artifact_sha256=_HASHES[1],
            execution_artifact_sha256=_HASHES[2],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
