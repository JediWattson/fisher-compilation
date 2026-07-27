from __future__ import annotations

import copy
import hashlib

import pytest

from fisher_graph.compression_rate_distortion import (
    CompressionRateDistortionCurve,
    CompressionRatePoint,
    compression_pareto_frontier,
    dense_supermode_rate_point_from_candidate,
)
from fisher_graph.structured_mlp_dense_supermode_pipeline import (
    StructuredMLPDenseSupermodeCandidate,
    _json_sha256,
    _REPORT_DOMAIN,
    build_structured_mlp_dense_supermode_candidate,
)
from fisher_graph.structured_mlp_dense_supermodes import (
    build_fisher_jacobian_dense_supermode_plan,
)

from test_structured_mlp_dense_supermodes import _activation_fixture


def _point(
    candidate_id: str,
    *,
    parameter_bytes: int,
    score: float,
    nll: float,
    latency: float | None = None,
    evaluation_id: str = "gemma3-guard-v1",
    evaluation_split_sha256: str = "a" * 64,
    task_suite: str = "gemma3-family-disjoint-v1",
    parameter_scope: str = "whole_model",
    compute_scope: str = "whole_model_forward",
    runtime_dtype: str = "torch.float32",
    runtime: str = "torch-eager-cpu",
    hardware_id: str | None = None,
    benchmark_protocol: str | None = None,
) -> CompressionRatePoint:
    return CompressionRatePoint(
        candidate_id=candidate_id,
        method="fixture",
        evaluation_id=evaluation_id,
        evaluation_split_sha256=evaluation_split_sha256,
        task_suite=task_suite,
        candidate_execution_fingerprint=hashlib.sha256(
            f"executor:{candidate_id}".encode()
        ).hexdigest(),
        candidate_report_sha256=hashlib.sha256(
            f"report:{candidate_id}".encode()
        ).hexdigest(),
        parameter_scope=parameter_scope,
        compute_scope=compute_scope,
        runtime_dtype=runtime_dtype,
        runtime=runtime,
        learned_parameters=parameter_bytes // 2,
        runtime_parameter_bytes=parameter_bytes,
        logical_macs_per_token=parameter_bytes // 4,
        downstream_score=score,
        nll=nll,
        teacher_kl=max(nll - 0.9, 0.0),
        top1_agreement=min(score, 1.0),
        operator_nrmse=max(1.0 - score, 0.0),
        measured_latency_ms=latency,
        hardware_id=(
            "apple-m5-cpu"
            if latency is not None and hardware_id is None
            else hardware_id
        ),
        benchmark_protocol=(
            "fixed-shape-cpu-latency-v1"
            if latency is not None and benchmark_protocol is None
            else benchmark_protocol
        ),
    )


def test_curve_retains_raw_points_and_frontier_excludes_dominated() -> None:
    native = _point(
        "native",
        parameter_bytes=1_000,
        score=1.0,
        nll=1.0,
        latency=2.0,
    )
    dense = _point(
        "dense-r1920",
        parameter_bytes=800,
        score=0.97,
        nll=1.02,
        latency=1.8,
    )
    deletion = _point(
        "delete-r1920",
        parameter_bytes=800,
        score=0.92,
        nll=1.08,
        latency=1.7,
    )
    aggressive = _point(
        "dense-r1536",
        parameter_bytes=650,
        score=0.90,
        nll=1.12,
        latency=1.5,
    )
    curve = CompressionRateDistortionCurve(
        (native, dense, deletion, aggressive)
    )

    assert tuple(
        point.candidate_id for point in curve.frontier()
    ) == (
        "dense-r1536",
        "dense-r1920",
        "native",
    )
    assert tuple(
        point.candidate_id
        for point in curve.frontier(quality_axis="nll")
    ) == (
        "dense-r1536",
        "dense-r1920",
        "native",
    )
    payload = curve.to_dict()
    assert len(payload["points"]) == 4
    assert payload["raw_points_retained_even_when_dominated"]
    assert "delete-r1920" not in payload["pareto"]["candidate_ids"]
    assert tuple(
        point.candidate_id
        for point in compression_pareto_frontier(
            curve.points,
            rate_axis="measured_latency_ms",
        )
    ) == (
        "dense-r1536",
        "delete-r1920",
        "dense-r1920",
        "native",
    )


def test_frontier_rejects_mixed_evaluation_and_resource_scopes() -> None:
    reference = _point(
        "reference",
        parameter_bytes=1_000,
        score=1.0,
        nll=1.0,
    )
    mixed_split = _point(
        "mixed-split",
        parameter_bytes=900,
        score=0.9,
        nll=1.1,
        evaluation_split_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="evaluation splits"):
        compression_pareto_frontier((reference, mixed_split))

    mixed_parameter_scope = _point(
        "mixed-parameter-scope",
        parameter_bytes=900,
        score=0.9,
        nll=1.1,
        parameter_scope="single_layer",
    )
    with pytest.raises(ValueError, match="parameter scopes"):
        compression_pareto_frontier(
            (reference, mixed_parameter_scope),
            rate_axis="learned_parameters",
        )
    with pytest.raises(ValueError, match="parameter scopes"):
        compression_pareto_frontier(
            (reference, mixed_parameter_scope),
            rate_axis="runtime_parameter_bytes",
        )

    mixed_compute_scope = _point(
        "mixed-compute-scope",
        parameter_bytes=900,
        score=0.9,
        nll=1.1,
        compute_scope="mlp_linear_only",
    )
    with pytest.raises(ValueError, match="compute scopes"):
        compression_pareto_frontier(
            (reference, mixed_compute_scope),
            rate_axis="logical_macs_per_token",
        )


def test_latency_frontier_requires_one_runtime_and_protocol() -> None:
    reference = _point(
        "reference-latency",
        parameter_bytes=1_000,
        score=1.0,
        nll=1.0,
        latency=2.0,
    )
    mixed_runtime = _point(
        "mixed-runtime",
        parameter_bytes=900,
        score=0.9,
        nll=1.1,
        latency=1.5,
        runtime="mlx-metal",
    )
    with pytest.raises(ValueError, match="runtimes"):
        compression_pareto_frontier(
            (reference, mixed_runtime),
            rate_axis="measured_latency_ms",
        )

    mixed_hardware = _point(
        "mixed-hardware",
        parameter_bytes=900,
        score=0.9,
        nll=1.1,
        latency=1.5,
        hardware_id="different-cpu",
    )
    with pytest.raises(ValueError, match="hardware ids"):
        compression_pareto_frontier(
            (reference, mixed_hardware),
            rate_axis="measured_latency_ms",
        )

    mixed_protocol = _point(
        "mixed-protocol",
        parameter_bytes=900,
        score=0.9,
        nll=1.1,
        latency=1.5,
        benchmark_protocol="different-shape-v2",
    )
    with pytest.raises(ValueError, match="benchmark protocols"):
        compression_pareto_frontier(
            (reference, mixed_protocol),
            rate_axis="measured_latency_ms",
        )


def test_quantized_dtype_can_share_byte_and_latency_frontiers() -> None:
    full_precision = _point(
        "full-precision",
        parameter_bytes=1_000,
        score=1.0,
        nll=1.0,
        latency=2.0,
    )
    quantized = _point(
        "packed-int4",
        parameter_bytes=300,
        score=0.95,
        nll=1.04,
        latency=1.0,
        runtime_dtype="int4-packed",
    )

    assert tuple(
        point.candidate_id
        for point in compression_pareto_frontier(
            (full_precision, quantized),
            rate_axis="runtime_parameter_bytes",
        )
    ) == ("packed-int4", "full-precision")
    assert tuple(
        point.candidate_id
        for point in compression_pareto_frontier(
            (full_precision, quantized),
            rate_axis="measured_latency_ms",
        )
    ) == ("packed-int4", "full-precision")


def _dense_candidate_bundle() -> StructuredMLPDenseSupermodeCandidate:
    parent, targets, scores = _activation_fixture()
    split = "9" * 64
    plan = build_fisher_jacobian_dense_supermode_plan(
        scores,
        source_down_weight=parent.feed_forward.down_proj.weight,
        calibration_split_sha256=split,
        activation_site="layer.0.mlp.down_input",
        parent_executor_fingerprint=parent.execution_fingerprint(),
        retained_pool_width=2,
        pool_indices=tuple(range(6)),
    )
    return build_structured_mlp_dense_supermode_candidate(
        parent,
        plan,
        targets,
        scores,
        calibration_split_sha256=split,
        generator_steps=1,
    )


def test_dense_candidate_binding_uses_integrity_checked_accounting() -> None:
    candidate = _dense_candidate_bundle()
    point = dense_supermode_rate_point_from_candidate(
        candidate,
        candidate_id="dense-k512-r384",
        evaluation_id="gemma3-guard-v1",
        evaluation_split_sha256="c" * 64,
        task_suite="gemma3-family-disjoint-v1",
        runtime="torch-eager-cpu",
        downstream_score=0.94,
        nll=1.05,
        teacher_kl=0.02,
        top1_agreement=0.96,
        operator_nrmse=0.01,
    )
    assert point.method == "dense_supermode"
    assert point.learned_parameters == candidate.executor.learned_parameter_count
    assert point.runtime_parameter_bytes == sum(
        parameter.numel() * parameter.element_size()
        for parameter in candidate.executor.parameters()
    )
    assert point.logical_macs_per_token == (
        candidate.report["resources"]["compute_per_valid_token"]["macs"][
            "candidate"
        ]
    )
    assert (
        point.candidate_execution_fingerprint
        == candidate.executor.execution_fingerprint()
    )
    assert (
        point.candidate_report_sha256
        == candidate.report["report_sha256"]
    )
    assert point.parameter_scope == (
        "compiled_structured_transformer_layer_all_learned_parameters"
    )
    assert point.compute_scope == (
        "compiled_structured_transformer_layer_mlp_gate_up_down_"
        "linear_weight_matmuls_only"
    )
    assert point.runtime_dtype == "torch.float32"
    assert point.downstream_score == pytest.approx(0.94)
    assert point.measured_latency_ms is None

    with pytest.raises(ValueError, match="no value"):
        compression_pareto_frontier(
            (point,),
            rate_axis="measured_latency_ms",
        )
    with pytest.raises(
        TypeError,
        match="StructuredMLPDenseSupermodeCandidate",
    ):
        dense_supermode_rate_point_from_candidate(
            candidate.report,
            candidate_id="bad",
            evaluation_id="gemma3-guard-v1",
            evaluation_split_sha256="c" * 64,
            task_suite="gemma3-family-disjoint-v1",
            runtime="torch-eager-cpu",
            downstream_score=0.0,
            nll=1.0,
            teacher_kl=0.0,
            top1_agreement=0.0,
            operator_nrmse=1.0,
        )


def test_dense_candidate_binding_rejects_resigned_noninteger_accounting() -> None:
    candidate = _dense_candidate_bundle()
    tampered_report = copy.deepcopy(candidate.report)
    tampered_report.pop("report_sha256")
    tampered_report["resources"]["parameters"][
        "candidate_full_layer"
    ] = float(candidate.executor.learned_parameter_count)
    tampered_report["report_sha256"] = _json_sha256(
        tampered_report,
        domain=_REPORT_DOMAIN,
    )
    tampered = StructuredMLPDenseSupermodeCandidate(
        executor=candidate.executor,
        artifact_state=candidate.artifact_state,
        report=tampered_report,
    )
    with pytest.raises(ValueError, match="positive integer"):
        dense_supermode_rate_point_from_candidate(
            tampered,
            candidate_id="tampered",
            evaluation_id="gemma3-guard-v1",
            evaluation_split_sha256="c" * 64,
            task_suite="gemma3-family-disjoint-v1",
            runtime="torch-eager-cpu",
            downstream_score=0.0,
            nll=1.0,
            teacher_kl=0.0,
            top1_agreement=0.0,
            operator_nrmse=1.0,
        )
