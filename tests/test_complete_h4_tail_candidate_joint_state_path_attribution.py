from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathAccumulator,
    candidate_joint_state_finite_kl_delta,
    candidate_joint_state_held_unit_endpoint_tangent_contraction,
    candidate_joint_state_path_displacement,
    candidate_joint_state_path_integrated_contraction,
    candidate_joint_state_scalar_endpoint_tangent_contraction,
    summarize_candidate_joint_state_path_attribution,
)
from fisher_graph.complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


_HASHES = tuple(f"{index:064x}" for index in range(1, 25))


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
    scalar_h4: torch.Tensor | None = None,
    joint_h4: torch.Tensor | None = None,
    path_contraction: torch.Tensor | None = None,
    finite_delta: torch.Tensor | None = None,
    scalar_tangent_contraction: torch.Tensor | None = None,
    include_tangent: bool = True,
    tangent_future_maximum: float = 0.0,
    tangent_future_count: int = 0,
    include_unit_tangent: bool = False,
    unit_tangent_contraction: torch.Tensor | None = None,
):
    scalar = (
        torch.tensor([[2.0, -1.0]], dtype=torch.float32)
        if scalar_h4 is None
        else scalar_h4
    )
    joint = (
        torch.tensor([[3.0, -2.0]], dtype=torch.float32)
        if joint_h4 is None
        else joint_h4
    )
    delta = joint.double() - scalar.double()
    token_path = (
        torch.tensor([1.0, -2.0], dtype=torch.float64)
        if path_contraction is None
        else path_contraction.double()
    )
    token_finite = (
        token_path.clone() if finite_delta is None else finite_delta.double()
    )
    if delta.numel() == 1:
        gradient = token_path[:, None, None] / delta.reshape(1, 1, 1)
    else:
        denominator = float(delta.square().sum())
        gradient = (
            token_path[:, None, None] * delta[None, :, :] / denominator
        )
    tangent_gradient = None
    if include_tangent:
        tangent_values = (
            0.5 * token_path
            if scalar_tangent_contraction is None
            else scalar_tangent_contraction.double()
        )
        if delta.numel() == 1:
            tangent_gradient = (
                tangent_values[:, None, None] / delta.reshape(1, 1, 1)
            )
        else:
            tangent_gradient = (
                tangent_values[:, None, None] * delta[None, :, :] / denominator
            )
    unit_gradient = None
    if include_unit_tangent:
        unit_values = (
            0.75 * token_path
            if unit_tangent_contraction is None
            else unit_tangent_contraction.double()
        )
        if delta.numel() == 1:
            unit_gradient = unit_values[:, None, None] / delta.reshape(1, 1, 1)
        else:
            unit_gradient = (
                unit_values[:, None, None] * delta[None, :, :] / denominator
            )
    scalar_kl = torch.full_like(token_finite, 20.0)
    joint_kl = scalar_kl + token_finite
    accumulator = CandidateJointStatePathAccumulator(
        example_id=example_id,
        family_id=family_id,
        scalar_endpoint_h4_rows=scalar,
        joint_endpoint_h4_rows=joint,
        scalar_token_teacher_kl=scalar_kl,
        joint_token_teacher_kl=joint_kl,
        scalar_endpoint_token_h4_gradients=tangent_gradient,
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
        scalar_tangent_vjp_artifact_sha256=(
            _HASHES[17] if include_tangent else None
        ),
        scalar_tangent_provider_artifact_sha256=(
            _HASHES[18] if include_tangent else None
        ),
        scalar_tangent_execution_artifact_sha256=(
            _HASHES[19] if include_tangent else None
        ),
        scalar_tangent_maximum_future_gradient_abs=(
            tangent_future_maximum if include_tangent else None
        ),
        scalar_tangent_future_gradient_nonzero_count=(
            tangent_future_count if include_tangent else None
        ),
        held_unit_endpoint_h4_rows=(
            scalar + torch.ones_like(scalar) if include_unit_tangent else None
        ),
        held_unit_token_teacher_kl=(
            scalar_kl + 0.25 if include_unit_tangent else None
        ),
        held_unit_endpoint_token_h4_gradients=unit_gradient,
        held_unit_tangent_vjp_artifact_sha256=(
            _HASHES[20] if include_unit_tangent else None
        ),
        held_unit_tangent_provider_artifact_sha256=(
            _HASHES[21] if include_unit_tangent else None
        ),
        held_unit_tangent_execution_artifact_sha256=(
            _HASHES[22] if include_unit_tangent else None
        ),
        held_unit_tangent_maximum_future_gradient_abs=(
            0.0 if include_unit_tangent else None
        ),
        held_unit_tangent_future_gradient_nonzero_count=(
            0 if include_unit_tangent else None
        ),
    )
    for index, (node, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS)
    ):
        accumulator.add_node(
            node_index=index,
            path_fraction=node,
            quadrature_weight=weight,
            path_node_h4_rows=(
                scalar.double() + node * (joint.double() - scalar.double())
            ).to(dtype=scalar.dtype),
            token_h4_gradients=gradient,
            token_teacher_kl=(1.0 - node) * scalar_kl + node * joint_kl,
            vjp_artifact_sha256=_HASHES[5 + 3 * index],
            provider_artifact_sha256=_HASHES[6 + 3 * index],
            execution_artifact_sha256=_HASHES[7 + 3 * index],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    return accumulator.finalize()


def test_actual_scalar_to_joint_orientation_and_exact_gl4_closure() -> None:
    evidence = _evidence()
    assert evidence.scalar_endpoint_h4_rows.dtype == torch.float32
    assert evidence.joint_endpoint_h4_rows.dtype == torch.float32
    assert torch.equal(
        candidate_joint_state_path_displacement(evidence),
        torch.tensor([[1.0, -1.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        candidate_joint_state_path_integrated_contraction(evidence),
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert torch.equal(
        candidate_joint_state_finite_kl_delta(evidence),
        torch.tensor([1.0, -2.0], dtype=torch.float64),
    )
    tangent = candidate_joint_state_scalar_endpoint_tangent_contraction(evidence)
    assert tangent is not None
    assert torch.allclose(
        tangent,
        torch.tensor([0.5, -1.0], dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    summary = summarize_candidate_joint_state_path_attribution((evidence,))
    assert summary.closure_rmse == pytest.approx(0.0, abs=1.0e-15)
    assert summary.closure_cosine == pytest.approx(1.0, abs=1.0e-15)
    assert summary.family_direction_agreement_count == 1
    assert summary.mean_scalar_endpoint_tangent == pytest.approx(-0.25)
    assert summary.mean_path_minus_scalar_tangent == pytest.approx(-0.25)


def test_accumulator_streams_exact_gl4_and_does_not_retain_node_banks() -> None:
    scalar = torch.tensor([[0.0]], dtype=torch.float32)
    joint = torch.tensor([[1.0]], dtype=torch.float32)
    scalar_kl = torch.tensor([4.0], dtype=torch.float32)
    joint_kl = torch.tensor([3.0], dtype=torch.float32)
    accumulator = CandidateJointStatePathAccumulator(
        example_id="stream",
        family_id="family",
        scalar_endpoint_h4_rows=scalar,
        joint_endpoint_h4_rows=joint,
        scalar_token_teacher_kl=scalar_kl,
        joint_token_teacher_kl=joint_kl,
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
    )
    gradients = tuple(
        torch.full((1, 1, 1), float(index + 1)) for index in range(4)
    )
    for index, gradient in enumerate(gradients):
        accumulator.add_node(
            node_index=index,
            path_fraction=GL4_UNIT_INTERVAL_NODES[index],
            quadrature_weight=GL4_UNIT_INTERVAL_WEIGHTS[index],
            path_node_h4_rows=torch.tensor(
                [[GL4_UNIT_INTERVAL_NODES[index]]], dtype=torch.float32
            ),
            token_h4_gradients=gradient,
            token_teacher_kl=torch.tensor([3.5]),
            vjp_artifact_sha256=_HASHES[5 + 3 * index],
            provider_artifact_sha256=_HASHES[6 + 3 * index],
            execution_artifact_sha256=_HASHES[7 + 3 * index],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    expected = sum(
        weight * gradient.double()
        for weight, gradient in zip(GL4_UNIT_INTERVAL_WEIGHTS, gradients)
    )
    for gradient in gradients:
        gradient.fill_(999.0)
    evidence = accumulator.finalize()
    assert torch.allclose(
        evidence.integrated_token_h4_gradients,
        expected,
        atol=1.0e-15,
        rtol=0.0,
    )
    assert not hasattr(evidence, "path_token_h4_gradients")
    assert evidence.metadata()["full_node_gradient_banks_retained"] is False
    with pytest.raises(RuntimeError, match="sealed"):
        accumulator.finalize()


def test_realized_low_precision_endpoint_values_define_the_displacement() -> None:
    scalar = torch.tensor([[1.0]], dtype=torch.float16)
    joint = torch.tensor([[1.0006]], dtype=torch.float16)
    evidence = _evidence(
        scalar_h4=scalar,
        joint_h4=joint,
        path_contraction=torch.tensor([1.0, 2.0]),
    )
    realized = float(joint) - float(scalar)
    assert realized == 0.0009765625
    assert float(candidate_joint_state_path_displacement(evidence)) == realized
    assert evidence.metadata()["realized_endpoint_dtype"] == "torch.float16"


def test_optional_scalar_tangent_is_absent_without_changing_path_evidence() -> None:
    evidence = _evidence(include_tangent=False)
    assert candidate_joint_state_scalar_endpoint_tangent_contraction(evidence) is None
    summary = summarize_candidate_joint_state_path_attribution((evidence,))
    assert summary.mean_scalar_endpoint_tangent is None
    assert summary.mean_path_minus_scalar_tangent is None
    assert summary.metadata()[
        "scalar_endpoint_tangent_substituted_for_GL4_integral"
    ] is False


def test_family_prompt_token_equal_weighting_does_not_pool_tokens() -> None:
    def small(example_id: str, family_id: str, values: list[float]):
        tokens = torch.tensor(values, dtype=torch.float64)
        return _evidence(
            example_id,
            family_id,
            scalar_h4=torch.tensor([[0.0]]),
            joint_h4=torch.tensor([[1.0]]),
            path_contraction=tokens,
            finite_delta=tokens,
            include_tangent=False,
        )

    a1 = small("a1", "a", [0.0])
    a2 = small("a2", "a", [1.0, 2.0, 3.0])
    b1 = small("b1", "b", [10.0, 10.0])
    summary = summarize_candidate_joint_state_path_attribution((b1, a2, a1))
    # family a = mean(prompt means 0, 2) = 1; family b = 10; macro = 5.5.
    assert summary.mean_finite_kl_delta == pytest.approx(5.5)
    pooled = torch.cat(
        tuple(candidate_joint_state_finite_kl_delta(v) for v in (a1, a2, b1))
    ).mean()
    assert summary.mean_finite_kl_delta != pytest.approx(float(pooled))
    assert summary.metadata()["weighting"] == (
        "mean_tokens_within_prompt_then_equal_prompts_within_family_"
        "then_equal_families"
    )


def test_family_and_token_sign_summaries_are_directional_and_zero_strict() -> None:
    agree = _evidence(
        "a",
        "a",
        scalar_h4=torch.tensor([[0.0]]),
        joint_h4=torch.tensor([[1.0]]),
        path_contraction=torch.tensor([-1.0, 0.0]),
        finite_delta=torch.tensor([-2.0, 1.0]),
        include_tangent=False,
    )
    disagree = _evidence(
        "b",
        "b",
        scalar_h4=torch.tensor([[0.0]]),
        joint_h4=torch.tensor([[1.0]]),
        path_contraction=torch.tensor([1.0, 0.0]),
        finite_delta=torch.tensor([-1.0, 0.0]),
        include_tangent=False,
    )
    summary = summarize_candidate_joint_state_path_attribution((disagree, agree))
    # a: one of two token signs agrees; b: only the zero/zero token agrees.
    assert summary.family_equal_token_sign_agreement_rate == pytest.approx(0.5)
    assert summary.family_direction_agreement_count == 1
    assert summary.family_finite_joint_improvement_count == 2
    assert summary.family_path_predicts_joint_improvement_count == 1
    assert summary.metadata()["sign_zero_policy"] == "zero_agrees_only_with_zero"


def test_metadata_is_tensor_free_and_semantically_explicit() -> None:
    evidence = _evidence()
    metadata = evidence.metadata()
    summary_metadata = summarize_candidate_joint_state_path_attribution(
        (evidence,)
    ).metadata()
    assert not _contains_tensor(metadata)
    assert not _contains_tensor(summary_metadata)
    assert metadata["FTC_orientation"] == (
        "joint_KL_minus_scalar_KL_compared_with_GL4_path_integral"
    )
    assert metadata["path_geometry"] == (
        "continuous_straight_complete_H4_scalar_to_joint_path_"
        "sampled_with_one_endpoint_dtype_cast_per_node"
    )
    assert metadata["pre_cast_ideal_endpoints_used"] is False
    assert summary_metadata["native_or_D320_endpoint_schema_reused"] is False
    assert summary_metadata["fits_selects_or_ranks_gain_fields"] is False
    assert summary_metadata[
        "authorizes_serving_compression_or_model_mutation"
    ] is False


def test_inputs_outputs_and_artifacts_are_mutation_safe_and_tamper_evident() -> None:
    scalar = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    joint = torch.tensor([[3.0, -2.0]], dtype=torch.float32)
    evidence = _evidence(scalar_h4=scalar, joint_h4=joint)
    scalar.fill_(999.0)
    joint.fill_(999.0)
    assert evidence.scalar_endpoint_h4_rows.tolist() == [[2.0, -1.0]]
    assert evidence.joint_endpoint_h4_rows.tolist() == [[3.0, -2.0]]

    returned = candidate_joint_state_path_displacement(evidence)
    returned.zero_()
    evidence.validate_integrity()
    assert not torch.equal(returned, evidence.joint_endpoint_h4_rows.double())

    evidence.joint_endpoint_h4_rows.add_(1.0)
    with pytest.raises(RuntimeError, match="node H4 binding drifted"):
        evidence.validate_integrity()

    clean = _evidence(example_id="clean")
    object.__setattr__(clean.node_receipts[0], "token_teacher_kl_mean", 123.0)
    with pytest.raises(RuntimeError, match="node receipt drifted"):
        clean.validate_integrity()


def test_accumulator_rejects_noncanonical_incomplete_and_wrong_endpoint_geometry() -> None:
    kwargs = dict(
        example_id="bad",
        family_id="family",
        scalar_endpoint_h4_rows=torch.zeros(1, 1),
        joint_endpoint_h4_rows=torch.ones(1, 1),
        scalar_token_teacher_kl=torch.ones(1),
        joint_token_teacher_kl=torch.ones(1),
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
    )
    accumulator = CandidateJointStatePathAccumulator(**kwargs)
    with pytest.raises(ValueError, match="canonical GL4 order"):
        accumulator.add_node(
            node_index=1,
            path_fraction=GL4_UNIT_INTERVAL_NODES[1],
            quadrature_weight=GL4_UNIT_INTERVAL_WEIGHTS[1],
            path_node_h4_rows=torch.tensor(
                [[GL4_UNIT_INTERVAL_NODES[1]]], dtype=torch.float32
            ),
            token_h4_gradients=torch.ones(1, 1, 1),
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_HASHES[5],
            provider_artifact_sha256=_HASHES[6],
            execution_artifact_sha256=_HASHES[7],
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
            path_node_h4_rows=torch.tensor(
                [[GL4_UNIT_INTERVAL_NODES[0]]], dtype=torch.float32
            ),
            token_h4_gradients=torch.ones(1, 1, 1),
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_HASHES[5],
            provider_artifact_sha256=_HASHES[6],
            execution_artifact_sha256=_HASHES[7],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    with pytest.raises(ValueError, match="geometry differs"):
        CandidateJointStatePathAccumulator(
            **{
                **kwargs,
                "joint_endpoint_h4_rows": torch.ones(1, 2),
            }
        )
    with pytest.raises(ValueError, match="geometry differs"):
        CandidateJointStatePathAccumulator(
            **{
                **kwargs,
                "joint_endpoint_h4_rows": torch.ones(1, 1, dtype=torch.float64),
            }
        )


def test_summary_allows_variable_rows_but_rejects_width_drift_and_mixed_tangents() -> None:
    with_tangent = _evidence("same", "a", include_tangent=True)
    without_tangent = _evidence("other", "a", include_tangent=False)
    with pytest.raises(ValueError, match="uniform"):
        summarize_candidate_joint_state_path_attribution(
            (with_tangent, without_tangent)
        )
    duplicate = _evidence("same", "b", include_tangent=True)
    with pytest.raises(ValueError, match="unique"):
        summarize_candidate_joint_state_path_attribution((with_tangent, duplicate))
    variable_rows = _evidence(
        "longer",
        "b",
        scalar_h4=torch.zeros(2, 2),
        joint_h4=torch.ones(2, 2),
        include_tangent=True,
    )
    variable_summary = summarize_candidate_joint_state_path_attribution(
        (with_tangent, variable_rows)
    )
    assert variable_summary.evidence_example_ids == ("longer", "same")

    different_width = _evidence(
        "wide",
        "b",
        scalar_h4=torch.zeros(1, 1),
        joint_h4=torch.ones(1, 1),
        include_tangent=True,
    )
    with pytest.raises(ValueError, match="H4 widths differ"):
        summarize_candidate_joint_state_path_attribution(
            (with_tangent, different_width)
        )


def test_summary_artifact_detects_scalar_or_family_tampering() -> None:
    a = _evidence("a", "a", include_tangent=False)
    b = _evidence("b", "b", include_tangent=False)
    summary = summarize_candidate_joint_state_path_attribution((a, b))
    object.__setattr__(summary, "mean_path_integral", 99.0)
    with pytest.raises(RuntimeError, match="attribution drifted"):
        summary.validate_integrity()


def test_node_receipt_rejects_h4_from_a_different_gl4_fraction() -> None:
    scalar = torch.zeros(1, 1, dtype=torch.float32)
    joint = torch.ones(1, 1, dtype=torch.float32)
    accumulator = CandidateJointStatePathAccumulator(
        example_id="wrong-node",
        family_id="family",
        scalar_endpoint_h4_rows=scalar,
        joint_endpoint_h4_rows=joint,
        scalar_token_teacher_kl=torch.ones(1),
        joint_token_teacher_kl=torch.ones(1),
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
    )
    with pytest.raises(ValueError, match="frozen float64-interpolate-cast-once"):
        accumulator.add_node(
            node_index=0,
            path_fraction=GL4_UNIT_INTERVAL_NODES[0],
            quadrature_weight=GL4_UNIT_INTERVAL_WEIGHTS[0],
            # This is a valid node tensor, but for GL4 node 1 rather than node 0.
            path_node_h4_rows=torch.tensor(
                [[GL4_UNIT_INTERVAL_NODES[1]]], dtype=torch.float32
            ),
            token_h4_gradients=torch.ones(1, 1, 1),
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_HASHES[5],
            provider_artifact_sha256=_HASHES[6],
            execution_artifact_sha256=_HASHES[7],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )


def test_bfloat16_path_nodes_use_float64_interpolation_then_one_cast() -> None:
    scalar = torch.tensor([[1.0]], dtype=torch.bfloat16)
    joint = torch.tensor([[1.5]], dtype=torch.bfloat16)
    evidence = _evidence(
        "bf16",
        "family",
        scalar_h4=scalar,
        joint_h4=joint,
        include_tangent=False,
    )
    assert evidence.scalar_endpoint_h4_rows.dtype == torch.bfloat16
    assert all(
        receipt.path_node_h4_dtype == "torch.bfloat16"
        for receipt in evidence.node_receipts
    )
    assert evidence.metadata()["path_node_construction"] == (
        "interpolate_supplied_realized_endpoints_in_float64_then_cast_once_to_endpoint_dtype"
    )


def test_scalar_tangent_requires_full_provenance_and_exposes_noncausality() -> None:
    base = dict(
        example_id="tangent",
        family_id="family",
        scalar_endpoint_h4_rows=torch.zeros(1, 1),
        joint_endpoint_h4_rows=torch.ones(1, 1),
        scalar_token_teacher_kl=torch.ones(1),
        joint_token_teacher_kl=torch.ones(1),
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
        scalar_endpoint_token_h4_gradients=torch.ones(1, 1, 1),
        scalar_tangent_vjp_artifact_sha256=_HASHES[5],
        scalar_tangent_provider_artifact_sha256=_HASHES[6],
        # Deliberately omit scalar_tangent_execution_artifact_sha256.
        scalar_tangent_maximum_future_gradient_abs=0.0,
        scalar_tangent_future_gradient_nonzero_count=0,
    )
    with pytest.raises(ValueError, match="complete tangent provenance"):
        CandidateJointStatePathAccumulator(**base)

    evidence = _evidence(
        "noncausal",
        "family",
        tangent_future_maximum=0.25,
        tangent_future_count=1,
    )
    assert evidence.metadata()["scalar_endpoint_tangent_causal"] is False
    assert evidence.scalar_endpoint_tangent_receipt is not None
    assert evidence.scalar_endpoint_tangent_receipt.maximum_future_gradient_abs == 0.25
    evidence.validate_integrity()


def test_scalar_tangent_receipt_is_bound_to_scalar_endpoint_and_grid() -> None:
    evidence = _evidence("bound", "family")
    assert evidence.scalar_endpoint_tangent_receipt is not None
    wrong_endpoint = replace(
        evidence.scalar_endpoint_tangent_receipt,
        endpoint_h4_rows_sha256=_HASHES[23],
    )
    with pytest.raises(ValueError, match="scalar endpoint tangent receipt binding"):
        replace(evidence, scalar_endpoint_tangent_receipt=wrong_endpoint)

    tampered = _evidence("tampered", "family")
    assert tampered.scalar_endpoint_tangent_receipt is not None
    object.__setattr__(
        tampered.scalar_endpoint_tangent_receipt,
        "supervised_grid_sha256",
        _HASHES[23],
    )
    with pytest.raises(RuntimeError, match="tangent receipt drifted"):
        tampered.validate_integrity()


def test_held_unit_tangent_is_independently_bound_but_uses_same_displacement() -> None:
    evidence = _evidence(
        "unit",
        "family",
        include_unit_tangent=True,
    )
    unit_tangent = candidate_joint_state_held_unit_endpoint_tangent_contraction(
        evidence
    )
    assert unit_tangent is not None
    assert torch.allclose(
        unit_tangent,
        torch.tensor([0.75, -1.5], dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    assert evidence.held_unit_endpoint_tangent_receipt is not None
    assert evidence.held_unit_endpoint_tangent_receipt.endpoint_role == (
        "held_unit_endpoint"
    )
    metadata = evidence.metadata()
    assert metadata["held_unit_endpoint_tangent_causal"] is True
    assert metadata["held_unit_tangent_displacement"] == (
        "joint_actual_minus_scalar_actual_same_displacement_different_reference"
    )
    summary = summarize_candidate_joint_state_path_attribution((evidence,))
    assert summary.mean_held_unit_endpoint_tangent == pytest.approx(-0.375)
    assert summary.mean_path_minus_held_unit_tangent == pytest.approx(-0.125)

    wrong_unit = replace(
        evidence.held_unit_endpoint_tangent_receipt,
        endpoint_token_teacher_kl_sha256=_HASHES[23],
    )
    with pytest.raises(ValueError, match="held-unit endpoint tangent receipt binding"):
        replace(evidence, held_unit_endpoint_tangent_receipt=wrong_unit)


def test_summary_requires_uniform_held_unit_tangent_coverage() -> None:
    with_unit = _evidence(
        "with-unit", "a", include_tangent=False, include_unit_tangent=True
    )
    without_unit = _evidence("without-unit", "b", include_tangent=False)
    with pytest.raises(ValueError, match="held-unit endpoint tangent coverage"):
        summarize_candidate_joint_state_path_attribution((with_unit, without_unit))
