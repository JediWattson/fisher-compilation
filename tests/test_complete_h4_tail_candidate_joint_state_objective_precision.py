from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_joint_state_objective_precision import (
    CLOSURE_COSINE_MINIMUM,
    DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE,
    FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM,
    OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateObjectivePrecisionEvidence,
    classify_candidate_joint_state_objective_precision,
    summarize_candidate_joint_state_objective_precision,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathAccumulator,
)
from fisher_graph.complete_h4_tail_path_teacher_kl import (
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
)


_HASHES = tuple(f"{index:064x}" for index in range(1, 40))


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _precision_evidence(
    example_id: str,
    family_id: str,
    *,
    d64: list[float],
    p64: list[float] | None = None,
    t64: list[float] | None = None,
    d32: list[float] | None = None,
    rows: int = 1,
    width: int = 2,
    include_tangent: bool = True,
) -> CandidateJointStateObjectivePrecisionEvidence:
    finite = torch.tensor(d64, dtype=torch.float64)
    path = torch.tensor(d64 if p64 is None else p64, dtype=torch.float64)
    tangent = torch.tensor(
        d64 if t64 is None else t64,
        dtype=torch.float64,
    )
    finite32 = torch.tensor(
        d64 if d32 is None else d32,
        dtype=torch.float32,
    )
    if not (finite.shape == path.shape == tangent.shape == finite32.shape):
        raise ValueError("test helper token vectors differ")
    scalar_h4 = torch.zeros((rows, width), dtype=torch.float32)
    joint_h4 = torch.ones((rows, width), dtype=torch.float32)
    displacement = joint_h4.double() - scalar_h4.double()
    denominator = float(displacement.square().sum())
    integrated_gradient = (
        path[:, None, None] * displacement[None, :, :] / denominator
    ).contiguous()
    tangent_gradient = (
        tangent[:, None, None] * displacement[None, :, :] / denominator
        if include_tangent
        else None
    )
    scalar_kl64 = torch.full_like(finite, 10.0)
    joint_kl64 = scalar_kl64 + finite
    accumulator = CandidateJointStatePathAccumulator(
        example_id=example_id,
        family_id=family_id,
        scalar_endpoint_h4_rows=scalar_h4,
        joint_endpoint_h4_rows=joint_h4,
        scalar_token_teacher_kl=scalar_kl64,
        joint_token_teacher_kl=joint_kl64,
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
        scalar_endpoint_token_h4_gradients=tangent_gradient,
        scalar_tangent_vjp_artifact_sha256=(
            _HASHES[5] if include_tangent else None
        ),
        scalar_tangent_provider_artifact_sha256=(
            _HASHES[6] if include_tangent else None
        ),
        scalar_tangent_execution_artifact_sha256=(
            _HASHES[7] if include_tangent else None
        ),
        scalar_tangent_maximum_future_gradient_abs=(
            0.0 if include_tangent else None
        ),
        scalar_tangent_future_gradient_nonzero_count=(
            0 if include_tangent else None
        ),
    )
    for index, (node, weight) in enumerate(
        zip(
            GL4_UNIT_INTERVAL_NODES,
            GL4_UNIT_INTERVAL_WEIGHTS,
            strict=True,
        )
    ):
        node_h4 = (
            scalar_h4.double()
            + node * (joint_h4.double() - scalar_h4.double())
        ).to(torch.float32)
        accumulator.add_node(
            node_index=index,
            path_fraction=node,
            quadrature_weight=weight,
            path_node_h4_rows=node_h4,
            token_h4_gradients=integrated_gradient,
            token_teacher_kl=scalar_kl64 + node * finite,
            vjp_artifact_sha256=_HASHES[8 + 3 * index],
            provider_artifact_sha256=_HASHES[9 + 3 * index],
            execution_artifact_sha256=_HASHES[10 + 3 * index],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    path_evidence = accumulator.finalize()
    scalar32 = torch.full(finite32.shape, 1.0, dtype=torch.float32)
    joint32 = scalar32 + finite32
    # Return the actually representable float32 finite delta.  For the integer
    # and half-integer values used by these tests it equals finite32 exactly.
    return CandidateJointStateObjectivePrecisionEvidence(
        path_evidence=path_evidence,
        finite_delta_f64_direct=finite,
        scalar_token_teacher_kl_f32=scalar32,
        joint_token_teacher_kl_f32=joint32,
        pinned_v9_evidence_artifact_sha256=_HASHES[20],
        endpoint_replay_binding_sha256=_HASHES[0],
        f32_objective_binding_sha256=_HASHES[22],
        f64_objective_binding_sha256=_HASHES[23],
    )


def test_exact_f64_closure_transport_and_precision_metrics() -> None:
    evidence = _precision_evidence(
        "example",
        "family",
        d64=[1.0, -2.0],
        p64=[1.0, -2.0],
        t64=[0.5, -1.0],
        d32=[0.5, -1.0],
        rows=3,
    )
    summary = summarize_candidate_joint_state_objective_precision((evidence,))
    metrics = summary.metrics
    assert metrics.closure_rmse == pytest.approx(0.0, abs=1.0e-15)
    assert metrics.closure_relative_rmse == pytest.approx(0.0, abs=1.0e-15)
    assert metrics.closure_cosine == pytest.approx(1.0, abs=1.0e-15)
    assert metrics.transport_rmse == pytest.approx(
        (0.5 * (0.5**2 + 1.0**2)) ** 0.5
    )
    assert metrics.transport_cosine == pytest.approx(1.0, abs=1.0e-15)
    assert metrics.finite_precision_rmse == pytest.approx(metrics.transport_rmse)
    assert summary.closure_passed is True
    assert summary.classification.endswith("balanced_or_zero_signals")


def test_family_prompt_token_weighting_resists_family_and_token_imbalance() -> None:
    # family a prompt means are 0 and 2 => family mean 1.  Family b has one
    # prompt with mean 10.  The family-equal result is 5.5, not a pooled mean.
    a1 = _precision_evidence("a1", "a", d64=[0.0], rows=1)
    a2 = _precision_evidence("a2", "a", d64=[1.0, 2.0, 3.0], rows=4)
    b1 = _precision_evidence("b1", "b", d64=[10.0, 10.0], rows=2)
    summary = summarize_candidate_joint_state_objective_precision((b1, a2, a1))
    assert summary.metrics.mean_finite_delta_f64 == pytest.approx(5.5)
    pooled = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 10.0]).mean()
    assert summary.metrics.mean_finite_delta_f64 != pytest.approx(float(pooled))
    assert summary.supervised_token_count == 6
    assert [
        family.metadata()["prompt_count"] for family in summary.family_summaries
    ] == [2, 1]
    assert {value.h4_shape[0] for value in (a1, a2, b1)} == {1, 2, 4}
    assert summary.metadata()["weighting"] == (
        "mean_tokens_within_prompt_then_equal_prompts_within_family_then_"
        "equal_families"
    )


def test_float32_endpoints_are_promoted_before_subtraction_to_replay_v9() -> None:
    evidence = _precision_evidence("round", "family", d64=[0.0])
    scalar = torch.tensor([2.0**-25], dtype=torch.float32)
    joint = torch.tensor([1.0], dtype=torch.float32)
    precise = replace(
        evidence,
        scalar_token_teacher_kl_f32=scalar,
        joint_token_teacher_kl_f32=joint,
    )
    legacy_v9_delta = joint.double() - scalar.double()
    subtraction_first_delta = (joint - scalar).double()
    assert torch.equal(precise.finite_delta_f32(), legacy_v9_delta)
    assert not torch.equal(precise.finite_delta_f32(), subtraction_first_delta)
    assert precise.metadata()[
        "f32_endpoint_operands_promoted_before_subtraction"
    ] is True


def test_direct_f64_delta_is_primary_and_endpoint_crosscheck_is_strict() -> None:
    evidence = _precision_evidence(
        "direct",
        "family",
        d64=[1.0, -2.0],
    )
    assert torch.equal(
        evidence.finite_delta_f64(),
        torch.tensor([1.0, -2.0], dtype=torch.float64),
    )
    metadata = evidence.metadata()
    assert metadata["finite_delta_f64_direct_is_primary_metrics_authority"] is True
    assert metadata["finite_delta_f64_direct_endpoint_crosscheck_passed"] is True
    assert metadata[
        "finite_delta_f64_direct_minus_endpoint_subtraction_max_abs"
    ] <= metadata["finite_delta_f64_direct_endpoint_crosscheck_tolerance"]
    assert DIRECT_FINITE_DELTA_F64_ABSOLUTE_TOLERANCE == (
        128.0 * torch.finfo(torch.float64).eps
    )
    summary = summarize_candidate_joint_state_objective_precision((evidence,))
    assert summary.maximum_direct_endpoint_crosscheck_abs_error == metadata[
        "finite_delta_f64_direct_minus_endpoint_subtraction_max_abs"
    ]
    assert summary.maximum_direct_endpoint_crosscheck_tolerance == metadata[
        "finite_delta_f64_direct_endpoint_crosscheck_tolerance"
    ]
    assert summary.metadata()["direct_endpoint_crosscheck_passed"] is True
    assert summary.family_summaries[0].metadata()[
        "direct_endpoint_crosscheck_passed"
    ] is True

    with pytest.raises(ValueError, match="differs from endpoint-KL subtraction"):
        replace(
            evidence,
            finite_delta_f64_direct=torch.tensor(
                [1.0 + 5.0e-12, -2.0], dtype=torch.float64
            ),
        )


def test_dtype_shape_and_missing_tangent_are_rejected() -> None:
    clean = _precision_evidence("clean", "family", d64=[1.0, 2.0])
    with pytest.raises(ValueError, match="float64 token vector"):
        replace(
            clean,
            finite_delta_f64_direct=torch.ones(2, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="float32 token vector"):
        replace(
            clean,
            scalar_token_teacher_kl_f32=torch.ones(2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="token geometry differs"):
        replace(
            clean,
            joint_token_teacher_kl_f32=torch.ones(3, dtype=torch.float32),
        )
    no_tangent_path = _precision_evidence(
        "temporary",
        "family",
        d64=[1.0],
    ).path_evidence
    accumulator = CandidateJointStatePathAccumulator(
        example_id="no-tangent",
        family_id="family",
        scalar_endpoint_h4_rows=no_tangent_path.scalar_endpoint_h4_rows,
        joint_endpoint_h4_rows=no_tangent_path.joint_endpoint_h4_rows,
        scalar_token_teacher_kl=no_tangent_path.scalar_token_teacher_kl,
        joint_token_teacher_kl=no_tangent_path.joint_token_teacher_kl,
        endpoint_pair_binding_sha256=_HASHES[0],
        scalar_endpoint_execution_artifact_sha256=_HASHES[1],
        joint_endpoint_execution_artifact_sha256=_HASHES[2],
        supervised_grid_sha256=_HASHES[3],
        teacher_logits_sha256=_HASHES[4],
    )
    for index, receipt in enumerate(no_tangent_path.node_receipts):
        accumulator.add_node(
            node_index=index,
            path_fraction=receipt.path_fraction,
            quadrature_weight=receipt.quadrature_weight,
            path_node_h4_rows=(
                no_tangent_path.scalar_endpoint_h4_rows.double()
                + receipt.path_fraction
                * (
                    no_tangent_path.joint_endpoint_h4_rows.double()
                    - no_tangent_path.scalar_endpoint_h4_rows.double()
                )
            ).float(),
            token_h4_gradients=no_tangent_path.integrated_token_h4_gradients,
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_HASHES[8 + 3 * index],
            provider_artifact_sha256=_HASHES[9 + 3 * index],
            execution_artifact_sha256=_HASHES[10 + 3 * index],
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    without_tangent = accumulator.finalize()
    with pytest.raises(ValueError, match="requires a scalar tangent"):
        CandidateJointStateObjectivePrecisionEvidence(
            path_evidence=without_tangent,
            finite_delta_f64_direct=torch.ones(1, dtype=torch.float64),
            scalar_token_teacher_kl_f32=torch.ones(1),
            joint_token_teacher_kl_f32=torch.ones(1),
            pinned_v9_evidence_artifact_sha256=_HASHES[20],
            endpoint_replay_binding_sha256=_HASHES[0],
            f32_objective_binding_sha256=_HASHES[22],
            f64_objective_binding_sha256=_HASHES[23],
        )


def test_input_copies_and_hashes_detect_tensor_and_scalar_tampering() -> None:
    direct = torch.tensor([1.0, 2.0], dtype=torch.float64)
    scalar = torch.ones(2, dtype=torch.float32)
    joint = torch.tensor([2.0, 3.0], dtype=torch.float32)
    base = _precision_evidence("copy", "family", d64=[1.0, 2.0])
    evidence = replace(
        base,
        finite_delta_f64_direct=direct,
        scalar_token_teacher_kl_f32=scalar,
        joint_token_teacher_kl_f32=joint,
    )
    direct.zero_()
    scalar.zero_()
    joint.zero_()
    assert evidence.finite_delta_f64_direct.tolist() == [1.0, 2.0]
    assert evidence.scalar_token_teacher_kl_f32.tolist() == [1.0, 1.0]
    assert evidence.joint_token_teacher_kl_f32.tolist() == [2.0, 3.0]
    evidence.joint_token_teacher_kl_f32.add_(1.0)
    with pytest.raises(RuntimeError, match="evidence drifted"):
        evidence.validate_integrity()

    clean = _precision_evidence("clean-summary", "family", d64=[1.0])
    summary = summarize_candidate_joint_state_objective_precision((clean,))
    object.__setattr__(summary.metrics, "closure_rmse", 99.0)
    with pytest.raises(RuntimeError, match="comparison drifted"):
        summary.validate_integrity()


def test_zero_rms_exact_closure_has_defined_cosine_and_passes() -> None:
    zero = _precision_evidence(
        "zero",
        "family",
        d64=[0.0, 0.0],
        p64=[0.0, 0.0],
        t64=[0.0, 0.0],
        d32=[0.0, 0.0],
    )
    summary = summarize_candidate_joint_state_objective_precision((zero,))
    assert summary.metrics.finite_delta_f64_rms == 0.0
    assert summary.metrics.closure_rmse == 0.0
    assert summary.metrics.closure_relative_rmse == 0.0
    assert summary.metrics.closure_cosine == 1.0
    assert summary.metrics.transport_relative_rmse_to_finite_f64 == 0.0
    assert summary.metrics.transport_cosine == 1.0
    assert summary.closure_passed is True
    assert summary.classification == (
        "f64_closure_established_balanced_or_zero_signals"
    )


def test_fixed_closure_gates_pass_and_fail_without_threshold_drift() -> None:
    assert OVERALL_CLOSURE_RELATIVE_RMSE_MAXIMUM == 0.05
    assert FAMILY_CLOSURE_RELATIVE_RMSE_MAXIMUM == 0.10
    assert CLOSURE_COSINE_MINIMUM == 0.99
    passing = _precision_evidence(
        "pass",
        "family",
        d64=[1.0, -1.0],
        p64=[1.04, -1.04],
    )
    passed = summarize_candidate_joint_state_objective_precision((passing,))
    assert passed.metrics.closure_relative_rmse == pytest.approx(0.04)
    assert all(passed.closure_gate_results.values())
    assert passed.closure_passed is True

    failing = _precision_evidence(
        "fail",
        "family",
        d64=[1.0, -1.0],
        p64=[1.2, -1.2],
    )
    failed = summarize_candidate_joint_state_objective_precision((failing,))
    assert failed.metrics.closure_relative_rmse == pytest.approx(0.2)
    assert failed.closure_gate_results[
        "overall_closure_relative_RMSE_at_most_0_05"
    ] is False
    assert failed.closure_gate_results[
        "every_family_closure_relative_RMSE_at_most_0_10"
    ] is False
    assert failed.closure_passed is False


def test_every_family_gate_can_fail_when_overall_gate_passes() -> None:
    # Seven exact families dilute one 12%-error family below the 5% overall
    # second-moment threshold, but the fixed per-family gate still rejects it.
    values = [
        _precision_evidence(f"exact-{index}", f"family-{index}", d64=[1.0])
        for index in range(7)
    ]
    values.append(
        _precision_evidence(
            "bad",
            "family-7",
            d64=[1.0],
            p64=[1.12],
        )
    )
    summary = summarize_candidate_joint_state_objective_precision(values)
    assert summary.metrics.closure_relative_rmse < 0.05
    assert summary.closure_gate_results[
        "overall_closure_relative_RMSE_at_most_0_05"
    ] is True
    assert summary.closure_gate_results[
        "overall_closure_cosine_at_least_0_99"
    ] is True
    assert summary.closure_gate_results[
        "every_family_closure_relative_RMSE_at_most_0_10"
    ] is False
    assert summary.closure_passed is False


def test_curvature_and_precision_outcomes_are_descriptive_and_distinct() -> None:
    curvature = _precision_evidence(
        "curvature",
        "family",
        d64=[1.0, -1.0],
        p64=[1.2, -1.2],
        t64=[0.0, 0.0],
        d32=[1.0, -1.0],
    )
    curvature_summary = summarize_candidate_joint_state_objective_precision(
        (curvature,)
    )
    assert curvature_summary.metrics.transport_rmse > (
        curvature_summary.metrics.finite_precision_rmse
    )
    assert curvature_summary.classification == (
        "f64_closure_unresolved_path_transport_signal_dominant"
    )

    precision = _precision_evidence(
        "precision",
        "family",
        d64=[1.0, -1.0],
        p64=[1.0, -1.0],
        t64=[1.0, -1.0],
        d32=[0.0, 0.0],
    )
    precision_summary = summarize_candidate_joint_state_objective_precision(
        (precision,)
    )
    assert precision_summary.metrics.finite_precision_rmse > (
        precision_summary.metrics.transport_rmse
    )
    assert precision_summary.classification == (
        "f64_closure_established_endpoint_precision_signal_dominant"
    )
    assert precision_summary.metadata()[
        "classification_compares_signal_RMSE_not_unique_cause"
    ] is True

    assert classify_candidate_joint_state_objective_precision(
        closure_passed=False,
        transport_rmse=1.0,
        finite_precision_rmse=1.0,
    ) == "f64_closure_unresolved_balanced_or_zero_signals"


def test_metadata_is_tensor_free_and_binds_exact_provenance() -> None:
    evidence = _precision_evidence(
        "metadata",
        "family",
        d64=[1.0, 2.0],
        rows=3,
        width=5,
    )
    summary = summarize_candidate_joint_state_objective_precision((evidence,))
    evidence_metadata = evidence.metadata()
    summary_metadata = summary.metadata()
    assert not _contains_tensor(evidence_metadata)
    assert not _contains_tensor(summary_metadata)
    assert evidence_metadata["path_evidence_artifact_sha256"] == (
        evidence.path_evidence.artifact_sha256
    )
    assert evidence_metadata["endpoint_replay_binding_sha256"] == (
        evidence.path_evidence.endpoint_pair_binding_sha256
    )
    assert evidence_metadata["h4_support_row_count"] == 3
    assert evidence_metadata["h4_width"] == 5
    assert evidence_metadata["raw_tensors_serialized"] is False
    assert summary_metadata["raw_tensors_serialized"] is False
    assert summary_metadata[
        "authorizes_serving_compression_or_model_mutation"
    ] is False


def test_replay_and_objective_authority_bindings_fail_closed() -> None:
    evidence = _precision_evidence("bindings", "family", d64=[1.0])
    with pytest.raises(ValueError, match="must equal the path endpoint-pair"):
        replace(
            evidence,
            endpoint_replay_binding_sha256=_HASHES[21],
        )
    with pytest.raises(ValueError, match="objective bindings must be distinct"):
        replace(
            evidence,
            f64_objective_binding_sha256=evidence.f32_objective_binding_sha256,
        )
    with pytest.raises(ValueError, match="distinct authorities"):
        replace(
            evidence,
            f32_objective_binding_sha256=evidence.endpoint_replay_binding_sha256,
        )


def test_duplicate_examples_and_mixed_types_are_rejected() -> None:
    first = _precision_evidence("same", "a", d64=[1.0])
    second = _precision_evidence("same", "b", d64=[1.0], rows=2)
    with pytest.raises(ValueError, match="must be unique"):
        summarize_candidate_joint_state_objective_precision((first, second))
    with pytest.raises(TypeError, match="typed evidence"):
        summarize_candidate_joint_state_objective_precision(
            (first, object())  # type: ignore[arg-type]
        )
