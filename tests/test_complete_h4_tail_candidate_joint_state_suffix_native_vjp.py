from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest
import torch

from fisher_graph.complete_h4_tail_candidate_joint_state_contraction_precision import (
    CONTRACTION_PUBLISHED_STAGE_ORDER,
    summarize_candidate_joint_state_contraction_precision,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_suffix_jvp import (
    summarize_candidate_joint_state_suffix_jvp,
)
from fisher_graph.complete_h4_tail_candidate_joint_state_suffix_native_vjp import (
    SUFFIX_NATIVE_VJP_TELESCOPE_POINTS,
    SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS,
    build_candidate_joint_state_suffix_native_vjp_evidence,
    classify_candidate_joint_state_suffix_native_vjp,
    summarize_candidate_joint_state_suffix_native_vjp,
)
from tests.test_complete_h4_tail_candidate_joint_state_contraction_precision import (
    _fixture as _v12_fixture,
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


def _native_fixture(
    example_id: str = "example",
    family_id: str = "family",
    *,
    tokens: int = 2,
    vectors: tuple[torch.Tensor, ...] | None = None,
    runtime_receipts: tuple[str, ...] | None = None,
):
    v12, *_ = _v12_fixture(
        example_id,
        family_id,
        tokens=tokens,
        jvp_stage="P_live",
    )
    if vectors is None:
        vectors = tuple(
            node.directional_derivative_f64()
            for node in v12.suffix_jvp_evidence.nodes
        )
    if runtime_receipts is None:
        runtime_receipts = tuple(
            _hash(f"native-runtime-{example_id}-{index}") for index in range(4)
        )
    resource_receipts = tuple(
        _hash(f"native-resource-{example_id}-{index}") for index in range(4)
    )
    evidence = build_candidate_joint_state_suffix_native_vjp_evidence(
        contraction_precision_evidence=v12,
        token_native_vjp_node_vectors_f64=vectors,
        native_suffix_runtime_receipt_sha256s=runtime_receipts,
        native_resource_receipt_sha256s=resource_receipts,
        native_suffix_forward_counts=(1, 1, 1, 1),
        native_vjp_pullback_counts=tuple((tokens + 7) // 8 for _ in range(4)),
    )
    return evidence, v12


def test_evidence_binds_exact_v12_v11_nodes_integrates_and_accounts_resources() -> None:
    evidence, v12 = _native_fixture()
    v11 = v12.suffix_jvp_evidence
    assert evidence.pinned_v12_evidence_artifact_sha256 == v12.artifact_sha256
    assert evidence.pinned_v11_evidence_artifact_sha256 == v11.artifact_sha256
    assert torch.equal(evidence.native_node_matrix_f64(), evidence.jvp_node_matrix_f64())
    assert torch.equal(
        evidence.integrated_native_vjp_f64(),
        v11.integrated_suffix_jvp_f64(),
    )
    for native_node, jvp_node in zip(evidence.nodes, v11.nodes, strict=True):
        assert native_node.pinned_v11_node_artifact_sha256 == jvp_node.artifact_sha256
        assert (
            native_node.pinned_v10_node_receipt_artifact_sha256
            == jvp_node.pinned_v10_node_receipt_artifact_sha256
        )

    resources = evidence.resource_accounting
    assert resources["quadrature_node_count"] == 4
    assert resources["native_suffix_forward_count"] == 4
    assert resources["native_vjp_pullback_count"] == 4
    assert resources["native_token_directional_derivative_count"] == 8
    assert resources["published_v11_jvp_node_token_reference_element_count"] == 8
    assert resources["canonical_output_cotangent_row_count"] == 8
    assert resources["logical_native_vjp_input_gradient_coordinate_count"] == (
        4 * 2 * 4 * 3
    )
    assert resources["fresh_full_model_forward_count"] == 0
    assert resources["fresh_full_model_backward_count"] == 0
    assert evidence.maximum_telescope_abs_error <= evidence.telescope_tolerance
    assert not _contains_tensor(evidence.metadata())


def test_exact_same_suffix_vectors_close_both_frozen_gates_and_replay_authorities() -> None:
    first, first_v12 = _native_fixture("a-1", "a", tokens=2)
    second, second_v12 = _native_fixture("a-2", "a", tokens=3)
    third, third_v12 = _native_fixture("b-1", "b", tokens=4)
    comparison = summarize_candidate_joint_state_suffix_native_vjp(
        (third, first, second)
    )
    v12 = summarize_candidate_joint_state_contraction_precision(
        (first_v12, second_v12, third_v12)
    )
    v11 = summarize_candidate_joint_state_suffix_jvp(
        tuple(value.suffix_jvp_evidence for value in (first_v12, second_v12, third_v12))
    )

    assert comparison.evidence_example_ids == ("a-1", "a-2", "b-1")
    assert comparison.replayed_v12_comparison_artifact_sha256 == v12.artifact_sha256
    assert comparison.replayed_v11_comparison_artifact_sha256 == v11.artifact_sha256
    assert comparison.nodewise_metrics.symmetric_relative_rmse == 0.0
    assert comparison.integrated_metrics.symmetric_relative_rmse == 0.0
    assert comparison.nodewise_metrics.cosine == 1.0
    assert comparison.integrated_metrics.cosine == 1.0
    assert comparison.nodewise_passed
    assert comparison.integrated_passed
    assert comparison.classification == (
        "v10_gradient_source_or_execution_path_difference_supported"
    )
    assert all(comparison.nodewise_gate_results.values())
    assert all(comparison.integrated_gate_results.values())
    assert comparison.telescope_metrics.passed
    assert comparison.telescope_metrics.maximum_absolute_residual <= (
        comparison.telescope_metrics.maximum_tolerance
    )


def test_all_nodes_and_all_v12_stages_are_published_without_selection() -> None:
    evidence, _ = _native_fixture()
    comparison = summarize_candidate_joint_state_suffix_native_vjp((evidence,))
    assert tuple(metric.node_index for metric in comparison.node_metrics) == tuple(range(4))
    assert tuple(metric.stage for metric in comparison.stage_metrics) == (
        CONTRACTION_PUBLISHED_STAGE_ORDER
    )
    assert SUFFIX_NATIVE_VJP_TELESCOPE_POINTS == (
        *CONTRACTION_PUBLISHED_STAGE_ORDER,
        "J64_suffix",
        "N64_native_vjp",
    )
    assert len(SUFFIX_NATIVE_VJP_TELESCOPE_TRANSITIONS) == (
        len(SUFFIX_NATIVE_VJP_TELESCOPE_POINTS) - 1
    )
    metadata = comparison.metadata()
    assert metadata["all_V12_stage_comparisons_published_without_selection"] is True
    assert metadata["fits_corrects_searches_or_selects_candidates"] is False
    assert metadata["authorizes_serving_compression_or_model_mutation"] is False
    assert "earliest_passing_stage" not in metadata
    assert all(
        metric.metadata()["descriptive_only"] is True
        for metric in comparison.stage_metrics
    )
    assert not _contains_tensor(metadata)


def test_nodewise_mismatch_can_remain_visible_when_gl4_integral_cancels() -> None:
    baseline, v12 = _native_fixture()
    jvp = tuple(
        node.directional_derivative_f64() for node in v12.suffix_jvp_evidence.nodes
    )
    weights = tuple(node.quadrature_weight for node in baseline.nodes)
    perturbation = torch.ones_like(jvp[0])
    vectors = (
        jvp[0] + perturbation,
        jvp[1] - (weights[0] / weights[1]) * perturbation,
        jvp[2],
        jvp[3],
    )
    evidence = build_candidate_joint_state_suffix_native_vjp_evidence(
        contraction_precision_evidence=v12,
        token_native_vjp_node_vectors_f64=vectors,
        native_suffix_runtime_receipt_sha256s=tuple(
            _hash(f"cancel-runtime-{index}") for index in range(4)
        ),
        native_resource_receipt_sha256s=tuple(
            _hash(f"cancel-resource-{index}") for index in range(4)
        ),
        native_suffix_forward_counts=(1, 1, 1, 1),
        native_vjp_pullback_counts=(1, 1, 1, 1),
    )
    comparison = summarize_candidate_joint_state_suffix_native_vjp((evidence,))
    assert not comparison.nodewise_passed
    assert comparison.integrated_passed
    assert comparison.integrated_metrics.maximum_absolute_error < 1e-12
    assert comparison.classification == (
        "persistent_same_suffix_forward_reverse_ad_or_nondifferentiable_boundary_ambiguity"
    )


def test_every_family_gate_prevents_large_family_from_hiding_small_family_failure() -> None:
    good, _ = _native_fixture("good", "large-family", tokens=8)
    bad_base, bad_v12 = _native_fixture("bad", "small-family", tokens=1)
    bad_vectors = tuple(
        -node.directional_derivative_f64()
        for node in bad_v12.suffix_jvp_evidence.nodes
    )
    bad = build_candidate_joint_state_suffix_native_vjp_evidence(
        contraction_precision_evidence=bad_v12,
        token_native_vjp_node_vectors_f64=bad_vectors,
        native_suffix_runtime_receipt_sha256s=tuple(
            _hash(f"bad-runtime-{index}") for index in range(4)
        ),
        native_resource_receipt_sha256s=tuple(
            _hash(f"bad-resource-{index}") for index in range(4)
        ),
        native_suffix_forward_counts=(1, 1, 1, 1),
        native_vjp_pullback_counts=(1, 1, 1, 1),
    )
    comparison = summarize_candidate_joint_state_suffix_native_vjp((good, bad))
    family = {value.family_id: value for value in comparison.family_summaries}
    assert family["large-family"].integrated_metrics.symmetric_relative_rmse == 0.0
    assert family["small-family"].integrated_metrics.symmetric_relative_rmse > 1e-4
    assert not comparison.integrated_gate_results[
        "every_family_integrated_symmetric_relative_RMSE_at_most_0_0001"
    ]
    assert not comparison.integrated_passed


def test_duplicate_runtime_receipt_is_rejected_across_families() -> None:
    first, _ = _native_fixture("first", "a")
    duplicate = first.nodes[0].native_suffix_runtime_receipt_sha256
    runtime = (
        duplicate,
        _hash("second-runtime-1"),
        _hash("second-runtime-2"),
        _hash("second-runtime-3"),
    )
    second, _ = _native_fixture("second", "b", runtime_receipts=runtime)
    with pytest.raises(ValueError, match="duplicate or cross-family ownership"):
        summarize_candidate_joint_state_suffix_native_vjp((first, second))


def test_builder_clones_inputs_and_internal_vector_mutation_is_detected() -> None:
    _, v12 = _native_fixture()
    vectors = tuple(
        node.directional_derivative_f64() for node in v12.suffix_jvp_evidence.nodes
    )
    evidence = build_candidate_joint_state_suffix_native_vjp_evidence(
        contraction_precision_evidence=v12,
        token_native_vjp_node_vectors_f64=vectors,
        native_suffix_runtime_receipt_sha256s=tuple(
            _hash(f"clone-runtime-{index}") for index in range(4)
        ),
        native_resource_receipt_sha256s=tuple(
            _hash(f"clone-resource-{index}") for index in range(4)
        ),
        native_suffix_forward_counts=(1, 1, 1, 1),
        native_vjp_pullback_counts=(1, 1, 1, 1),
    )
    expected = evidence.integrated_native_vjp_f64()
    vectors[0].add_(123.0)
    assert torch.equal(evidence.integrated_native_vjp_f64(), expected)
    with torch.no_grad():
        evidence.nodes[0].token_native_vjp_f64.add_(1.0)
    with pytest.raises(RuntimeError, match="native VJP evidence drifted"):
        evidence.validate_integrity()


def test_runtime_pullback_counts_are_required_accounted_and_mutation_guarded() -> None:
    evidence, _ = _native_fixture("chunked", "family", tokens=9)
    assert tuple(node.native_vjp_pullback_count for node in evidence.nodes) == (
        2,
        2,
        2,
        2,
    )
    assert evidence.resource_accounting["native_vjp_pullback_count"] == 8
    object.__setattr__(evidence.nodes[0], "native_vjp_pullback_count", 99)
    with pytest.raises(RuntimeError, match="native VJP evidence drifted"):
        evidence.validate_integrity()


def test_frozen_authority_and_summary_mutations_are_detected() -> None:
    evidence, _ = _native_fixture()
    with pytest.raises(FrozenInstanceError):
        evidence.example_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="authority pin differs"):
        replace(evidence, pinned_v12_evidence_artifact_sha256=_hash("wrong"))

    comparison = summarize_candidate_joint_state_suffix_native_vjp((evidence,))
    object.__setattr__(comparison, "supervised_token_count", 999)
    with pytest.raises(RuntimeError, match="comparison drifted"):
        comparison.validate_integrity()


def test_builder_rejects_bad_node_geometry_and_receipt_ownership() -> None:
    _, v12 = _native_fixture()
    vectors = tuple(
        node.directional_derivative_f64() for node in v12.suffix_jvp_evidence.nodes
    )
    with pytest.raises(ValueError, match="requires four values"):
        build_candidate_joint_state_suffix_native_vjp_evidence(
            contraction_precision_evidence=v12,
            token_native_vjp_node_vectors_f64=vectors[:3],
            native_suffix_runtime_receipt_sha256s=tuple(
                _hash(f"short-runtime-{index}") for index in range(4)
            ),
            native_resource_receipt_sha256s=tuple(
                _hash(f"short-resource-{index}") for index in range(4)
            ),
            native_suffix_forward_counts=(1, 1, 1, 1),
            native_vjp_pullback_counts=(1, 1, 1, 1),
        )
    duplicate = _hash("duplicate")
    with pytest.raises(ValueError, match="ownership must be node-distinct"):
        build_candidate_joint_state_suffix_native_vjp_evidence(
            contraction_precision_evidence=v12,
            token_native_vjp_node_vectors_f64=vectors,
            native_suffix_runtime_receipt_sha256s=(duplicate,) * 4,
            native_resource_receipt_sha256s=tuple(
                _hash(f"resource-{index}") for index in range(4)
            ),
            native_suffix_forward_counts=(1, 1, 1, 1),
            native_vjp_pullback_counts=(1, 1, 1, 1),
        )


def test_classification_is_total_and_type_checked() -> None:
    assert classify_candidate_joint_state_suffix_native_vjp(
        nodewise_passed=True,
        integrated_passed=True,
    ) == "v10_gradient_source_or_execution_path_difference_supported"
    for nodewise, integrated in ((False, False), (False, True), (True, False)):
        assert classify_candidate_joint_state_suffix_native_vjp(
            nodewise_passed=nodewise,
            integrated_passed=integrated,
        ) == (
            "persistent_same_suffix_forward_reverse_ad_or_nondifferentiable_boundary_ambiguity"
        )
    with pytest.raises(TypeError, match="must be boolean"):
        classify_candidate_joint_state_suffix_native_vjp(
            nodewise_passed=1,  # type: ignore[arg-type]
            integrated_passed=True,
        )
