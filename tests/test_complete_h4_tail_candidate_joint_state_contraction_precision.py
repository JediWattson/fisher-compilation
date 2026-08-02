from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_joint_state_contraction_precision import (
    CONTRACTION_PUBLISHED_STAGE_ORDER,
    CONTRACTION_TELESCOPE_TRANSITIONS,
    CandidateJointStateContractionPrecisionAccumulator,
    build_candidate_joint_state_contraction_precision_evidence,
    classify_candidate_joint_state_contraction_precision,
    summarize_candidate_joint_state_contraction_precision,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_objective_precision import (
    CandidateJointStateObjectivePrecisionEvidence,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathAccumulator,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_suffix_jvp import (
    CandidateJointStateSuffixJVPEvidence,
    CandidateJointStateSuffixJVPNodeEvidence,
    summarize_candidate_joint_state_suffix_jvp,
)
from fisher_graph.complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _node_contractions(
    gradient64: torch.Tensor,
    displacement64: torch.Tensor,
    tangent32: torch.Tensor,
) -> dict[str, torch.Tensor]:
    tokens = int(gradient64.shape[0])
    g64 = gradient64.reshape(tokens, -1)
    g32 = gradient64.float().reshape(tokens, -1)
    d64 = displacement64.reshape(-1)
    d32 = tangent32.reshape(-1)
    product32 = (g32 * d32[None, :]).contiguous()
    return {
        "P64_node": torch.sum(g64 * d64[None, :], dim=1, dtype=torch.float64),
        "P_dir": torch.sum(
            g64 * d32.double()[None, :], dim=1, dtype=torch.float64
        ),
        "P_prod": torch.sum(product32.double(), dim=1, dtype=torch.float64),
        "P_live": torch.sum(product32, dim=1, dtype=torch.float32).double(),
    }


def _fixture(
    example_id: str = "example",
    family_id: str = "family",
    *,
    tokens: int = 2,
    jvp_stage: str = "P_live",
    finite_scale: float = 0.5,
) -> tuple[
    object,
    CandidateJointStateSuffixJVPEvidence,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
    dict[str, torch.Tensor],
]:
    support = torch.tensor([0, 2], dtype=torch.int64)
    full_scalar = torch.zeros((4, 3), dtype=torch.float32)
    full_joint = full_scalar.clone()
    scalar_rows = torch.tensor(
        [[2.0**-25, -1.0, 0.25], [4.0, 2.0**-20, -0.5]],
        dtype=torch.float32,
    )
    joint_rows = torch.tensor(
        [[1.0, 0.25, -0.75], [2.0**-22, 1.0, 0.75]],
        dtype=torch.float32,
    )
    full_scalar.index_copy_(0, support, scalar_rows)
    full_joint.index_copy_(0, support, joint_rows)
    full_delta64 = full_joint.double() - full_scalar.double()
    cast_tangent32 = full_delta64.float().contiguous()
    support_delta64 = full_delta64.index_select(0, support).contiguous()
    support_tangent32 = cast_tangent32.index_select(0, support).contiguous()

    gradients: list[torch.Tensor] = []
    node_values: list[dict[str, torch.Tensor]] = []
    base = torch.arange(tokens * 2 * 3, dtype=torch.float32).reshape(tokens, 2, 3)
    base = (base - float(base.numel()) / 2.0) * 0.03125
    for node_index in range(4):
        gradient32 = (
            base * float(node_index + 1)
            + torch.tensor(
                [2.0**-20, -2.0**-18, 2.0**-16], dtype=torch.float32
            )[None, None, :]
        ).contiguous()
        gradient64 = gradient32.double().contiguous()
        gradients.append(gradient64)
        node_values.append(
            _node_contractions(gradient64, support_delta64, support_tangent32)
        )

    integrated_by_stage = {
        stage: torch.zeros(tokens, dtype=torch.float64)
        for stage in ("P64_node", "P_dir", "P_prod", "P_live")
    }
    for weight, values in zip(
        GL4_UNIT_INTERVAL_WEIGHTS, node_values, strict=True
    ):
        for stage in integrated_by_stage:
            integrated_by_stage[stage] = (
                integrated_by_stage[stage] + weight * values[stage]
            ).contiguous()
    jvp_nodes = tuple(values[jvp_stage] for values in node_values)
    integrated_jvp = torch.zeros(tokens, dtype=torch.float64)
    for weight, value in zip(GL4_UNIT_INTERVAL_WEIGHTS, jvp_nodes, strict=True):
        integrated_jvp.add_(value, alpha=weight)
    finite = (finite_scale * integrated_jvp).contiguous()

    scalar_kl64 = torch.full((tokens,), 10.0, dtype=torch.float64)
    joint_kl64 = scalar_kl64 + finite
    endpoint = _hash(f"endpoint-{example_id}")
    grid = _hash(f"grid-{example_id}")
    path_accumulator = CandidateJointStatePathAccumulator(
        example_id=example_id,
        family_id=family_id,
        scalar_endpoint_h4_rows=scalar_rows,
        joint_endpoint_h4_rows=joint_rows,
        scalar_token_teacher_kl=scalar_kl64,
        joint_token_teacher_kl=joint_kl64,
        endpoint_pair_binding_sha256=endpoint,
        scalar_endpoint_execution_artifact_sha256=_hash(f"scalar-{example_id}"),
        joint_endpoint_execution_artifact_sha256=_hash(f"joint-{example_id}"),
        supervised_grid_sha256=grid,
        teacher_logits_sha256=_hash(f"teacher-{example_id}"),
        scalar_endpoint_token_h4_gradients=gradients[0],
        scalar_tangent_vjp_artifact_sha256=_hash(f"tangent-vjp-{example_id}"),
        scalar_tangent_provider_artifact_sha256=_hash(
            f"tangent-provider-{example_id}"
        ),
        scalar_tangent_execution_artifact_sha256=_hash(
            f"tangent-execution-{example_id}"
        ),
        scalar_tangent_maximum_future_gradient_abs=0.0,
        scalar_tangent_future_gradient_nonzero_count=0,
    )
    for node_index, (node, weight, gradient) in enumerate(
        zip(
            GL4_UNIT_INTERVAL_NODES,
            GL4_UNIT_INTERVAL_WEIGHTS,
            gradients,
            strict=True,
        )
    ):
        node_h4 = (
            scalar_rows.double()
            + node * (joint_rows.double() - scalar_rows.double())
        ).float()
        path_accumulator.add_node(
            node_index=node_index,
            path_fraction=node,
            quadrature_weight=weight,
            path_node_h4_rows=node_h4,
            token_h4_gradients=gradient,
            token_teacher_kl=scalar_kl64 + node * finite,
            vjp_artifact_sha256=_hash(f"vjp-{example_id}-{node_index}"),
            provider_artifact_sha256=_hash(f"provider-{example_id}-{node_index}"),
            execution_artifact_sha256=_hash(
                f"execution-{example_id}-{node_index}"
            ),
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    path = path_accumulator.finalize()
    integrated_gradient = torch.zeros_like(gradients[0])
    for weight, gradient in zip(GL4_UNIT_INTERVAL_WEIGHTS, gradients, strict=True):
        integrated_gradient = (integrated_gradient + weight * gradient).contiguous()
    integrated_by_stage["P_v10"] = torch.einsum(
        "rw,trw->t", support_delta64, integrated_gradient
    ).contiguous()

    precision = CandidateJointStateObjectivePrecisionEvidence(
        path_evidence=path,
        finite_delta_f64_direct=finite,
        scalar_token_teacher_kl_f32=torch.ones(tokens, dtype=torch.float32),
        joint_token_teacher_kl_f32=(torch.ones(tokens) + finite.float()),
        pinned_v9_evidence_artifact_sha256=_hash(f"v9-{example_id}"),
        endpoint_replay_binding_sha256=endpoint,
        f32_objective_binding_sha256=_hash(f"f32-{example_id}"),
        f64_objective_binding_sha256=_hash(f"f64-{example_id}"),
    )
    suffix_nodes = tuple(
        CandidateJointStateSuffixJVPNodeEvidence(
            node_index=receipt.node_index,
            path_fraction=receipt.path_fraction,
            quadrature_weight=receipt.quadrature_weight,
            token_directional_derivative_f64=jvp_nodes[receipt.node_index],
            pinned_v10_node_receipt_artifact_sha256=receipt.artifact_sha256,
            suffix_runtime_receipt_sha256=_hash(
                f"suffix-runtime-{example_id}-{receipt.node_index}"
            ),
            primal_token_teacher_kl_sha256=receipt.token_teacher_kl_sha256,
            provider_artifact_sha256=receipt.provider_artifact_sha256,
            execution_artifact_sha256=receipt.execution_artifact_sha256,
            path_h4_sha256=receipt.path_node_h4_sha256,
            supervised_grid_sha256=path.supervised_grid_sha256,
            endpoint_pair_binding_sha256=path.endpoint_pair_binding_sha256,
        )
        for receipt in path.node_receipts
    )
    suffix = CandidateJointStateSuffixJVPEvidence(
        precision_evidence=precision,
        nodes=suffix_nodes,
    )
    evidence = build_candidate_joint_state_contraction_precision_evidence(
        suffix_jvp_evidence=suffix,
        support_indices=support,
        full_displacement_f64=full_delta64,
        full_cast_tangent_f32=cast_tangent32,
        node_support_h4_gradients_f64=tuple(gradients),
    )
    return (
        evidence,
        suffix,
        support,
        full_delta64,
        cast_tangent32,
        tuple(gradients),
        integrated_by_stage,
    )


def test_streamed_ladder_matches_independent_typed_reductions_and_replays_v10() -> None:
    evidence, suffix, _, _, _, _, expected = _fixture()
    for stage in CONTRACTION_PUBLISHED_STAGE_ORDER:
        assert torch.equal(evidence.stage_vector_f64(stage), expected[stage])
    assert torch.equal(
        evidence.stage_vector_f64("P_v10"),
        suffix.replayed_vjp_integral_f64(),
    )
    assert all(not hasattr(node, "token_support_h4_gradients_f64") for node in evidence.nodes)
    assert evidence.metadata()["reduction_order"] == (
        "canonical_contiguous_support_row_major_width_minor_flatten_"
        "then_typed_torch_sum"
    )
    assert evidence.metadata()["P_live_is_counterfactual_not_internal_VJP_schedule_proof"] is True


def test_streaming_accumulator_releases_gradient_bank_and_seals() -> None:
    _, suffix, support, displacement, tangent, gradients, _ = _fixture("stream")
    accumulator = CandidateJointStateContractionPrecisionAccumulator(
        support_indices=support,
        full_displacement_f64=displacement,
        full_cast_tangent_f32=tangent,
    )
    for receipt, gradient in zip(
        suffix.precision_evidence.path_evidence.node_receipts,
        gradients,
        strict=True,
    ):
        accumulator.add_node(
            node_receipt=receipt,
            token_support_h4_gradients_f64=gradient,
        )
    assert getattr(accumulator, "_integrated_gradient64") is None
    assert getattr(accumulator, "_token_p_v10_f64") is not None
    evidence = accumulator.finalize(suffix_jvp_evidence=suffix)
    assert evidence.nodes[0].metadata()["transient_gradient_retained"] is False
    assert getattr(accumulator, "_integrated_gradient64") is None
    assert getattr(accumulator, "_token_p_v10_f64") is None
    with pytest.raises(RuntimeError, match="sealed"):
        accumulator.finalize(suffix_jvp_evidence=suffix)


def test_gradient_roundtrip_hash_direction_cast_and_outside_support_fail_closed() -> None:
    _, suffix, support, displacement, tangent, gradients, _ = _fixture("strict")
    bad_gradient = gradients[0].clone()
    bad_gradient[0, 0, 0] = torch.tensor(1.0 / 3.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="exact float32 lift"):
        build_candidate_joint_state_contraction_precision_evidence(
            suffix_jvp_evidence=suffix,
            support_indices=support,
            full_displacement_f64=displacement,
            full_cast_tangent_f32=tangent,
            node_support_h4_gradients_f64=(bad_gradient, *gradients[1:]),
        )
    changed_gradient = gradients[0].clone()
    changed_gradient[0, 0, 0] += torch.tensor(1.0, dtype=torch.float32).double()
    with pytest.raises(ValueError, match="differs from the V10 receipt"):
        build_candidate_joint_state_contraction_precision_evidence(
            suffix_jvp_evidence=suffix,
            support_indices=support,
            full_displacement_f64=displacement,
            full_cast_tangent_f32=tangent,
            node_support_h4_gradients_f64=(changed_gradient, *gradients[1:]),
        )
    bad_tangent = tangent.clone()
    bad_tangent[0, 0] = torch.nextafter(
        bad_tangent[0, 0], torch.tensor(float("inf"), dtype=torch.float32)
    )
    with pytest.raises(ValueError, match="exact float32 delta cast"):
        CandidateJointStateContractionPrecisionAccumulator(
            support_indices=support,
            full_displacement_f64=displacement,
            full_cast_tangent_f32=bad_tangent,
        )
    outside = displacement.clone()
    outside[1, 0] = 1.0
    with pytest.raises(ValueError, match="zero outside support"):
        CandidateJointStateContractionPrecisionAccumulator(
            support_indices=support,
            full_displacement_f64=outside,
            full_cast_tangent_f32=outside.float(),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        CandidateJointStateContractionPrecisionAccumulator(
            support_indices=torch.tensor([2, 0]),
            full_displacement_f64=displacement,
            full_cast_tangent_f32=tangent,
        )


@pytest.mark.parametrize(
    ("passes", "expected"),
    (
        ((True, True, True, True), "f64_operation_ordering_sufficient"),
        ((False, True, True, True), "direction_cast_rounding_sufficient"),
        ((False, False, True, True), "f32_product_rounding_sufficient"),
        ((False, False, False, True), "native_f32_boundary_reduction_sufficient"),
        ((False, False, False, False), "unresolved_forward_reverse_ad_kernel_mismatch"),
    ),
)
def test_classification_uses_ordered_earliest_pass(
    passes: tuple[bool, bool, bool, bool], expected: str
) -> None:
    assert classify_candidate_joint_state_contraction_precision(
        p64_node_passed=passes[0],
        p_dir_passed=passes[1],
        p_prod_passed=passes[2],
        p_live_passed=passes[3],
    ) == expected
    with pytest.raises(TypeError, match="must be boolean"):
        classify_candidate_joint_state_contraction_precision(
            p64_node_passed=1,  # type: ignore[arg-type]
            p_dir_passed=False,
            p_prod_passed=False,
            p_live_passed=False,
        )


def test_summary_replays_v11_and_publishes_finite_eligibility_separation() -> None:
    evidence, suffix, *_ = _fixture("summary", finite_scale=0.5)
    summary = summarize_candidate_joint_state_contraction_precision((evidence,))
    v11 = summarize_candidate_joint_state_suffix_jvp((suffix,))
    control = summary.metrics_for("P_v10")
    assert summary.replayed_v11_comparison_artifact_sha256 == v11.artifact_sha256
    assert control.adjoint_rmse.hex() == v11.metrics.adjoint_rmse.hex()
    assert control.closure_rmse.hex() == v11.metrics.vjp_closure_rmse.hex()
    assert summary.finite_metrics.closure_rmse.hex() == v11.metrics.jvp_closure_rmse.hex()
    assert tuple(metric.stage for metric in summary.stage_metrics) == (
        CONTRACTION_PUBLISHED_STAGE_ORDER
    )
    assert tuple(metric.transition for metric in summary.transition_metrics) == (
        CONTRACTION_TELESCOPE_TRANSITIONS
    )
    assert summary.telescope_metrics.passed is True
    assert (
        summary.telescope_metrics.maximum_absolute_residual
        <= summary.telescope_metrics.absolute_tolerance
    )
    assert all(
        family.telescope_metrics.passed for family in summary.family_summaries
    )
    assert summary.metadata()["fixed_telescope_reconstructs_D64_finite_minus_P_v10"] is True
    assert set(summary.finite_correction_eligibility_gate_results) == {
        "corrected_contraction_stage_identified",
        "remaining_adjoint_RMSE_fraction_of_finite_closure_at_most_0_01",
        "corrected_contraction_still_fails_frozen_finite_closure_suite",
        "suffix_JVP_still_fails_frozen_finite_closure_suite",
        "finite_closure_RMSE_over_remaining_adjoint_RMSE_at_least_100",
    }
    assert summary.metadata()["all_precision_stages_published_regardless_of_pass"] is True
    object.__setattr__(summary.transition_metrics[0], "mean_delta_f64", 99.0)
    with pytest.raises(RuntimeError, match="comparison drifted"):
        summary.validate_integrity()


def test_family_equal_weighting_allows_variable_token_counts() -> None:
    a1 = _fixture("a1", "a", tokens=1)[0]
    a2 = _fixture("a2", "a", tokens=3)[0]
    b1 = _fixture("b1", "b", tokens=2)[0]
    summary = summarize_candidate_joint_state_contraction_precision((b1, a2, a1))
    assert summary.supervised_token_count == 6
    assert [len(family.example_ids) for family in summary.family_summaries] == [2, 1]
    assert summary.metadata()["weighting"] == (
        "mean_tokens_within_prompt_then_equal_prompts_within_family_then_"
        "equal_families"
    )


def test_resource_algebra_defensive_copies_hashes_and_tensor_free_metadata() -> None:
    evidence = _fixture("integrity", tokens=3)[0]
    resources = evidence.resource_accounting
    node_elements = 4 * 3 * 2 * 3
    assert resources["gradient_f64_to_f32_roundtrip_validation_element_count"] == node_elements
    assert (
        resources["nodewise_contraction_coordinate_observation_count_per_stage"]
        == node_elements
    )
    assert (
        resources["nodewise_contraction_coordinate_observation_count_total"]
        == 4 * node_elements
    )
    assert resources["actual_coordinate_product_bank_count"] == 3
    assert resources["actual_coordinate_product_count_total"] == 3 * node_elements
    assert resources["full_h4_row_count"] == (
        resources["support_h4_row_count"]
        + resources["outside_support_h4_row_count"]
    )
    assert resources["fresh_model_forward_count"] == 0
    assert resources["fresh_model_backward_count"] == 0
    assert not _contains_tensor(evidence.metadata())
    summary = summarize_candidate_joint_state_contraction_precision((evidence,))
    assert not _contains_tensor(summary.metadata())

    copied = evidence.stage_vector_f64("P_live")
    copied.zero_()
    assert not torch.equal(copied, evidence.stage_vector_f64("P_live"))
    evidence.nodes[0].token_p_live_f64.add_(1.0)
    with pytest.raises(RuntimeError, match="evidence drifted"):
        evidence.validate_integrity()


def test_duplicate_and_untyped_evidence_are_rejected() -> None:
    first = _fixture("same", "a")[0]
    second = _fixture("same", "b")[0]
    with pytest.raises(ValueError, match="must be unique"):
        summarize_candidate_joint_state_contraction_precision((first, second))
    with pytest.raises(TypeError, match="typed evidence"):
        summarize_candidate_joint_state_contraction_precision(
            (first, object())  # type: ignore[arg-type]
        )


def test_comparison_rejects_cross_family_duplicate_membership() -> None:
    evidence = _fixture("cross-family", "a")[0]
    summary = summarize_candidate_joint_state_contraction_precision((evidence,))
    original = summary.family_summaries[0]
    duplicate = replace(original, family_id="b")
    with pytest.raises(ValueError, match="membership is invalid"):
        replace(
            summary,
            family_summaries=(original, duplicate),
            supervised_token_count=2 * summary.supervised_token_count,
        )
