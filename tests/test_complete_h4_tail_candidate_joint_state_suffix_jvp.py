from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_joint_state_objective_precision import (
    CandidateJointStateObjectivePrecisionEvidence,
    summarize_candidate_joint_state_objective_precision,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_path_attribution import (
    CandidateJointStatePathAccumulator,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_suffix_jvp import (
    ADJOINT_RELATIVE_RMSE_MAXIMUM,
    CandidateJointStateSuffixJVPEvidence,
    CandidateJointStateSuffixJVPNodeEvidence,
    classify_candidate_joint_state_suffix_jvp,
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


def _precision_evidence(
    example_id: str,
    family_id: str,
    *,
    finite: list[float],
    vjp: list[float] | None = None,
    rows: int = 1,
    width: int = 2,
) -> CandidateJointStateObjectivePrecisionEvidence:
    finite64 = torch.tensor(finite, dtype=torch.float64)
    vjp64 = torch.tensor(finite if vjp is None else vjp, dtype=torch.float64)
    if finite64.shape != vjp64.shape:
        raise ValueError("test finite and VJP vectors differ")
    scalar_h4 = torch.zeros((rows, width), dtype=torch.float32)
    joint_h4 = torch.ones((rows, width), dtype=torch.float32)
    displacement = joint_h4.double() - scalar_h4.double()
    denominator = float(displacement.square().sum())
    path_gradient = (
        vjp64[:, None, None] * displacement[None, :, :] / denominator
    ).contiguous()
    tangent_gradient = path_gradient.clone()
    scalar_kl64 = torch.full_like(finite64, 10.0)
    joint_kl64 = scalar_kl64 + finite64
    endpoint = _hash(f"endpoint-{example_id}")
    grid = _hash(f"grid-{example_id}")
    accumulator = CandidateJointStatePathAccumulator(
        example_id=example_id,
        family_id=family_id,
        scalar_endpoint_h4_rows=scalar_h4,
        joint_endpoint_h4_rows=joint_h4,
        scalar_token_teacher_kl=scalar_kl64,
        joint_token_teacher_kl=joint_kl64,
        endpoint_pair_binding_sha256=endpoint,
        scalar_endpoint_execution_artifact_sha256=_hash(f"scalar-{example_id}"),
        joint_endpoint_execution_artifact_sha256=_hash(f"joint-{example_id}"),
        supervised_grid_sha256=grid,
        teacher_logits_sha256=_hash(f"teacher-{example_id}"),
        scalar_endpoint_token_h4_gradients=tangent_gradient,
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
    for index, (node, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS, strict=True)
    ):
        node_h4 = (
            scalar_h4.double() + node * (joint_h4.double() - scalar_h4.double())
        ).float()
        accumulator.add_node(
            node_index=index,
            path_fraction=node,
            quadrature_weight=weight,
            path_node_h4_rows=node_h4,
            token_h4_gradients=path_gradient,
            token_teacher_kl=scalar_kl64 + node * finite64,
            vjp_artifact_sha256=_hash(f"vjp-{example_id}-{index}"),
            provider_artifact_sha256=_hash(f"provider-{example_id}-{index}"),
            execution_artifact_sha256=_hash(f"execution-{example_id}-{index}"),
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    path = accumulator.finalize()
    scalar32 = torch.ones_like(finite64, dtype=torch.float32)
    joint32 = scalar32 + finite64.float()
    return CandidateJointStateObjectivePrecisionEvidence(
        path_evidence=path,
        finite_delta_f64_direct=finite64,
        scalar_token_teacher_kl_f32=scalar32,
        joint_token_teacher_kl_f32=joint32,
        pinned_v9_evidence_artifact_sha256=_hash(f"v9-{example_id}"),
        endpoint_replay_binding_sha256=endpoint,
        f32_objective_binding_sha256=_hash(f"f32-{example_id}"),
        f64_objective_binding_sha256=_hash(f"f64-{example_id}"),
    )


def _suffix_evidence(
    example_id: str,
    family_id: str,
    *,
    finite: list[float],
    vjp: list[float] | None = None,
    jvp: list[float] | None = None,
    rows: int = 1,
    width: int = 2,
) -> CandidateJointStateSuffixJVPEvidence:
    precision = _precision_evidence(
        example_id,
        family_id,
        finite=finite,
        vjp=vjp,
        rows=rows,
        width=width,
    )
    directional = torch.tensor(
        finite if jvp is None else jvp,
        dtype=torch.float64,
    )
    nodes = []
    path = precision.path_evidence
    for receipt in path.node_receipts:
        nodes.append(
            CandidateJointStateSuffixJVPNodeEvidence(
                node_index=receipt.node_index,
                path_fraction=receipt.path_fraction,
                quadrature_weight=receipt.quadrature_weight,
                token_directional_derivative_f64=directional,
                pinned_v10_node_receipt_artifact_sha256=receipt.artifact_sha256,
                suffix_runtime_receipt_sha256=_hash(
                    f"suffix-runtime-{example_id}-{receipt.node_index}"
                ),
                primal_token_teacher_kl_sha256=(
                    receipt.token_teacher_kl_sha256
                ),
                provider_artifact_sha256=receipt.provider_artifact_sha256,
                execution_artifact_sha256=receipt.execution_artifact_sha256,
                path_h4_sha256=receipt.path_node_h4_sha256,
                supervised_grid_sha256=path.supervised_grid_sha256,
                endpoint_pair_binding_sha256=path.endpoint_pair_binding_sha256,
            )
        )
    return CandidateJointStateSuffixJVPEvidence(
        precision_evidence=precision,
        nodes=tuple(nodes),
    )


def test_exact_gl4_jvp_vjp_and_finite_closure_passes() -> None:
    evidence = _suffix_evidence(
        "exact",
        "family",
        finite=[1.0, -2.0],
        vjp=[1.0, -2.0],
        jvp=[1.0, -2.0],
        rows=3,
        width=4,
    )
    expected = torch.tensor([1.0, -2.0], dtype=torch.float64)
    assert torch.allclose(evidence.integrated_suffix_jvp_f64(), expected)
    assert torch.allclose(evidence.replayed_vjp_integral_f64(), expected)
    assert torch.equal(evidence.finite_delta_f64(), expected)
    summary = summarize_candidate_joint_state_suffix_jvp((evidence,))
    assert summary.metrics.adjoint_relative_rmse < 1.0e-14
    assert summary.metrics.jvp_closure_relative_rmse < 1.0e-14
    assert summary.metrics.vjp_closure_relative_rmse < 1.0e-14
    assert summary.metrics.adjoint_cosine == pytest.approx(1.0)
    assert summary.adjoint_passed is True
    assert summary.jvp_closure_passed is True
    assert summary.vjp_closure_passed is True
    assert summary.classification == (
        "suffix_adjoint_passed_both_closures_established_same_a"
    )


def test_adjoint_pass_with_both_closure_misses_marks_finite_path_remainder() -> None:
    evidence = _suffix_evidence(
        "discrete",
        "family",
        finite=[1.0, -1.0],
        vjp=[1.2, -1.2],
        jvp=[1.2, -1.2],
    )
    summary = summarize_candidate_joint_state_suffix_jvp((evidence,))
    assert summary.adjoint_passed is True
    assert summary.jvp_closure_passed is False
    assert summary.vjp_closure_passed is False
    assert summary.classification == (
        "suffix_adjoint_passed_both_closures_miss_finite_path_remainder_same_a"
    )


def test_jvp_only_closure_marks_reverse_contraction_failure() -> None:
    evidence = _suffix_evidence(
        "reverse",
        "family",
        finite=[1.0, -1.0],
        vjp=[1.2, -1.2],
        jvp=[1.0, -1.0],
    )
    summary = summarize_candidate_joint_state_suffix_jvp((evidence,))
    assert summary.jvp_closure_passed is True
    assert summary.vjp_closure_passed is False
    assert summary.adjoint_passed is False
    assert summary.classification == (
        "suffix_jvp_only_closure_reverse_contraction_failure_same_a"
    )
    assert classify_candidate_joint_state_suffix_jvp(
        adjoint_passed=True,
        jvp_closure_passed=True,
        vjp_closure_passed=False,
    ) == "suffix_jvp_only_closure_reverse_contraction_failure_same_a"


def test_adjoint_failure_and_vjp_only_closure_remain_ambiguous() -> None:
    evidence = _suffix_evidence(
        "ambiguous",
        "family",
        finite=[1.0, -1.0],
        vjp=[1.0, -1.0],
        jvp=[1.2, -1.2],
    )
    summary = summarize_candidate_joint_state_suffix_jvp((evidence,))
    assert summary.adjoint_passed is False
    assert summary.jvp_closure_passed is False
    assert summary.vjp_closure_passed is True
    assert summary.classification == "suffix_adjoint_ambiguity_same_a"


def test_v10_vjp_closure_and_direct_crosscheck_replay_exactly() -> None:
    values = (
        _suffix_evidence("a1", "a", finite=[1.0], vjp=[1.04], jvp=[1.04]),
        _suffix_evidence(
            "a2",
            "a",
            finite=[2.0, -2.0, 1.0],
            vjp=[2.08, -2.08, 1.04],
            jvp=[2.08, -2.08, 1.04],
            rows=4,
        ),
        _suffix_evidence(
            "b1", "b", finite=[5.0, -5.0], vjp=[5.2, -5.2], jvp=[5.2, -5.2]
        ),
    )
    summary = summarize_candidate_joint_state_suffix_jvp(reversed(values))
    v10 = summarize_candidate_joint_state_objective_precision(
        value.precision_evidence for value in values
    )
    assert summary.replayed_v10_comparison_artifact_sha256 == v10.artifact_sha256
    assert summary.metrics.vjp_closure_rmse.hex() == v10.metrics.closure_rmse.hex()
    assert (
        summary.metrics.vjp_closure_relative_rmse.hex()
        == v10.metrics.closure_relative_rmse.hex()
    )
    assert summary.metrics.vjp_closure_cosine.hex() == v10.metrics.closure_cosine.hex()
    assert summary.vjp_closure_passed == v10.closure_passed
    assert summary.maximum_direct_endpoint_crosscheck_abs_error.hex() == (
        v10.maximum_direct_endpoint_crosscheck_abs_error.hex()
    )
    assert summary.metadata()["v10_vjp_closure_replayed_exactly"] is True


def test_nested_weighting_allows_variable_tokens_and_h4_rows() -> None:
    # Family a has prompt JVP means 0 and 2 -> 1; family b has mean 10.
    # The family-equal panel mean is therefore 5.5, not a pooled token mean.
    a1 = _suffix_evidence("a1", "a", finite=[0.0], rows=1)
    a2 = _suffix_evidence("a2", "a", finite=[1.0, 2.0, 3.0], rows=4)
    b1 = _suffix_evidence("b1", "b", finite=[10.0, 10.0], rows=2)
    summary = summarize_candidate_joint_state_suffix_jvp((b1, a2, a1))
    assert summary.metrics.mean_suffix_jvp_f64 == pytest.approx(5.5)
    assert summary.metrics.mean_vjp_path_integral_f64 == pytest.approx(5.5)
    assert summary.metrics.mean_finite_delta_f64 == pytest.approx(5.5)
    assert summary.supervised_token_count == 6
    assert [family.metadata()["prompt_count"] for family in summary.family_summaries] == [
        2,
        1,
    ]
    assert {value.h4_shape[0] for value in (a1, a2, b1)} == {1, 2, 4}
    assert summary.metadata()["weighting"] == (
        "mean_tokens_within_prompt_then_equal_prompts_within_family_then_"
        "equal_families"
    )


def test_zero_vectors_have_defined_relative_error_cosine_and_directions() -> None:
    evidence = _suffix_evidence(
        "zero", "family", finite=[0.0, 0.0], vjp=[0.0, 0.0], jvp=[0.0, 0.0]
    )
    summary = summarize_candidate_joint_state_suffix_jvp((evidence,))
    assert summary.metrics.adjoint_relative_rmse == 0.0
    assert summary.metrics.jvp_closure_relative_rmse == 0.0
    assert summary.metrics.vjp_closure_relative_rmse == 0.0
    assert summary.metrics.adjoint_cosine == 1.0
    assert summary.metrics.jvp_closure_cosine == 1.0
    assert summary.directions == {
        "suffix_jvp": "zero",
        "vjp_path_integral": "zero",
        "finite_delta": "zero",
        "jvp_minus_vjp": "zero",
        "jvp_minus_finite": "zero",
        "vjp_minus_finite": "zero",
    }


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("pinned_v10_node_receipt_artifact_sha256", "provenance differs"),
        ("primal_token_teacher_kl_sha256", "provenance differs"),
        ("provider_artifact_sha256", "provenance differs"),
        ("execution_artifact_sha256", "provenance differs"),
        ("path_h4_sha256", "provenance differs"),
        ("supervised_grid_sha256", "provenance differs"),
        ("endpoint_pair_binding_sha256", "provenance differs"),
    ),
)
def test_every_node_provenance_authority_fails_closed(
    field: str, message: str
) -> None:
    clean = _suffix_evidence("authority", "family", finite=[1.0])
    changed = replace(clean.nodes[0], **{field: _hash(f"wrong-{field}")})
    with pytest.raises(ValueError, match=message):
        replace(clean, nodes=(changed, *clean.nodes[1:]))


def test_node_order_rule_receipts_and_geometry_fail_closed() -> None:
    clean = _suffix_evidence("strict", "family", finite=[1.0, 2.0])
    with pytest.raises(ValueError, match="ordered four GL4"):
        replace(clean, nodes=tuple(reversed(clean.nodes)))
    with pytest.raises(ValueError, match="exact GL4 rule"):
        replace(clean.nodes[0], path_fraction=clean.nodes[0].path_fraction + 1.0e-6)
    with pytest.raises(ValueError, match="token geometry"):
        bad = replace(
            clean.nodes[0],
            token_directional_derivative_f64=torch.ones(3, dtype=torch.float64),
        )
        replace(clean, nodes=(bad, *clean.nodes[1:]))
    duplicate_receipt = replace(
        clean.nodes[1],
        suffix_runtime_receipt_sha256=clean.nodes[0].suffix_runtime_receipt_sha256,
    )
    with pytest.raises(ValueError, match="node-distinct"):
        replace(clean, nodes=(clean.nodes[0], duplicate_receipt, *clean.nodes[2:]))
    with pytest.raises(ValueError, match="float64 token vector"):
        replace(
            clean.nodes[0],
            token_directional_derivative_f64=torch.ones(2, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="float64 token vector"):
        replace(
            clean.nodes[0],
            token_directional_derivative_f64=torch.tensor(
                [1.0, float("nan")], dtype=torch.float64
            ),
        )


def test_defensive_copies_hash_tampering_and_tensor_free_metadata() -> None:
    clean = _suffix_evidence("tamper", "family", finite=[1.0, -1.0])
    source = torch.tensor([1.0, -1.0], dtype=torch.float64)
    copied_node = replace(clean.nodes[0], token_directional_derivative_f64=source)
    source.zero_()
    assert copied_node.directional_derivative_f64().tolist() == [1.0, -1.0]
    copied_node.token_directional_derivative_f64.add_(1.0)
    with pytest.raises(RuntimeError, match="node drifted"):
        copied_node.validate_integrity()

    summary = summarize_candidate_joint_state_suffix_jvp((clean,))
    assert not _contains_tensor(clean.nodes[0].metadata())
    assert not _contains_tensor(clean.metadata())
    assert not _contains_tensor(summary.metadata())
    assert clean.metadata()["raw_tensors_serialized"] is False
    assert summary.metadata()["raw_tensors_serialized"] is False
    assert summary.metadata()["authorizes_serving_compression_or_model_mutation"] is False
    object.__setattr__(summary.metrics, "adjoint_rmse", 99.0)
    with pytest.raises(RuntimeError, match="comparison drifted"):
        summary.validate_integrity()


def test_frozen_adjoint_gate_and_classification_boolean_validation() -> None:
    assert ADJOINT_RELATIVE_RMSE_MAXIMUM == 1.0e-4
    passing = _suffix_evidence(
        "adjoint-pass",
        "family",
        finite=[1.0],
        vjp=[1.0],
        jvp=[1.00005],
    )
    assert summarize_candidate_joint_state_suffix_jvp((passing,)).adjoint_passed
    failing = _suffix_evidence(
        "adjoint-fail",
        "family",
        finite=[1.0],
        vjp=[1.0],
        jvp=[1.0002],
    )
    assert not summarize_candidate_joint_state_suffix_jvp((failing,)).adjoint_passed
    with pytest.raises(TypeError, match="must be boolean"):
        classify_candidate_joint_state_suffix_jvp(
            adjoint_passed=1,  # type: ignore[arg-type]
            jvp_closure_passed=True,
            vjp_closure_passed=True,
        )


def test_duplicate_examples_and_untyped_evidence_are_rejected() -> None:
    first = _suffix_evidence("same", "a", finite=[1.0])
    second = _suffix_evidence("same", "b", finite=[1.0], rows=2)
    with pytest.raises(ValueError, match="must be unique"):
        summarize_candidate_joint_state_suffix_jvp((first, second))
    with pytest.raises(TypeError, match="typed evidence"):
        summarize_candidate_joint_state_suffix_jvp(
            (first, object())  # type: ignore[arg-type]
        )
