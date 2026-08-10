from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math

import pytest

import fisher_graph.gemma3_layer_multimode_evaluation as evaluation_module
from fisher_graph.gemma3_layer_multimode_evaluation import (
    BOOTSTRAP_REPLICATE_COUNT,
    B3TeacherForcedEvaluationReceipt,
    C3FullStackEvaluationReceipt,
    CompiledCandidateIdentityReceipt,
    D3FinalEvaluationReceipt,
    D3NoNativeMLPTraceReceipt,
    D3ResourceComparisonReceipt,
    D3SimultaneousBoundReceipt,
    FullStackExampleAggregate,
    LayerTeacherForcedAdditiveEvidence,
    LayerTeacherForcedFamilyAggregate,
    LayerTeacherForcedModeAggregate,
    d3_bootstrap_policy,
    d3_bootstrap_policy_sha256,
    evaluate_b3_teacher_forced,
    evaluate_c3_full_stack,
    evaluate_d3_final,
    heldout_evaluation_policy_sha256,
)


FAMILIES = tuple(f"family-{index}" for index in range(6))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _family(
    family_id: str,
    *,
    nrmse: float = 0.20,
    cosine: float = 0.97,
) -> LayerTeacherForcedFamilyAggregate:
    energy = 1_000.0
    return LayerTeacherForcedFamilyAggregate(
        family_id=family_id,
        scalar_count=640,
        prediction_sse=float(nrmse * nrmse * energy),
        teacher_energy=energy,
        prediction_energy=energy,
        prediction_teacher_dot=float(cosine * energy),
    )


def _mode(
    chart_ordinal: int,
    *,
    chart_weight: float,
    mode_count: int,
    masses: tuple[float, ...] | None = None,
    mutual_information: float = 0.10,
) -> LayerTeacherForcedModeAggregate:
    if mode_count == 1:
        masses = (1.0,)
        router_entropy = 0.0
    else:
        masses = masses or tuple(1.0 / mode_count for _ in range(mode_count))
        entropy = -sum(value * math.log(value) for value in masses if value)
        router_entropy = entropy - mutual_information
    return LayerTeacherForcedModeAggregate(
        chart_ordinal=chart_ordinal,
        chart_weight_sum=float(chart_weight),
        chart_mode_weight_sums=tuple(
            float(chart_weight * value) for value in masses
        ),
        weighted_router_entropy_sum_nats=float(
            chart_weight * router_entropy
        ),
    )


def _layer(
    layer: int,
    *,
    chart_count: int = 1,
    mode_count: int = 1,
    nrmse: float = 0.20,
    cosine: float = 0.97,
    masses: tuple[float, ...] | None = None,
    mutual_information: float = 0.10,
) -> LayerTeacherForcedAdditiveEvidence:
    row_count = float(len(FAMILIES))
    chart_weight = row_count / chart_count
    return LayerTeacherForcedAdditiveEvidence(
        layer_ordinal=layer,
        chart_count=chart_count,
        dynamic_mode_count=mode_count,
        tangent_rank=16,
        selected_candidate_sha256=_digest(f"selected-{layer}"),
        role_capture_receipt_sha256=_digest("B3-capture"),
        family_aggregates=tuple(
            _family(value, nrmse=nrmse, cosine=cosine)
            for value in FAMILIES
        ),
        mode_aggregates=tuple(
            _mode(
                chart,
                chart_weight=chart_weight,
                mode_count=mode_count,
                masses=masses,
                mutual_information=mutual_information,
            )
            for chart in range(chart_count)
        ),
        row_weight_sum=row_count,
    )


def _b3(
    *,
    first_layer: LayerTeacherForcedAdditiveEvidence | None = None,
) -> B3TeacherForcedEvaluationReceipt:
    layers = tuple(
        first_layer
        if layer == 0 and first_layer is not None
        else _layer(
            layer,
            chart_count=2 if layer == 0 else 1,
            mode_count=2 if layer == 0 else 1,
        )
        for layer in range(18)
    )
    return evaluate_b3_teacher_forced(
        layers,
        source_model_sha256=_digest("source-model"),
        selection_receipt_sha256=_digest("selection"),
        panel_manifest_sha256=_digest("panels"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
    )


def _candidate(nonce: str = "candidate") -> CompiledCandidateIdentityReceipt:
    return CompiledCandidateIdentityReceipt(
        selection_receipt_sha256=_digest("selection"),
        compiler_protocol_sha256=_digest("compiler-protocol"),
        compile_receipt_sha256=_digest(f"compile-{nonce}"),
        bundle_semantic_sha256=_digest(f"bundle-{nonce}"),
        compiled_model_state_sha256=_digest(f"state-{nonce}"),
    )


def _records(
    role: str,
    prefix: str,
    *,
    members_per_stratum: int = 2,
    variable: bool = True,
    delta_center: float = 0.008,
) -> tuple[FullStackExampleAggregate, ...]:
    values: list[FullStackExampleAggregate] = []
    for family in FAMILIES:
        for hard_long in (False, True):
            for member in range(members_per_stratum):
                signed = -1.0 if member % 2 == 0 else 1.0
                delta = delta_center + (0.004 * signed if variable else 0.0)
                kl = 0.007 + (0.003 * signed if variable else 0.0)
                error_count = 1 if member % 2 == 0 and variable else 2
                if variable and member % 2 == 1:
                    error_count = 3
                identity = _digest(
                    f"{prefix}-{family}-{int(hard_long)}-{member}"
                )
                values.append(
                    FullStackExampleAggregate(
                        role=role,
                        example_identity_sha256=identity,
                        family_id=family,
                        hard_long=hard_long,
                        predicted_token_count=100,
                        delta_nll_sum_nats=float(delta * 100.0),
                        native_to_candidate_kl_sum_nats=float(kl * 100.0),
                        top1_agreement_count=100 - error_count,
                        example_execution_receipt_sha256=_digest(
                            f"execution-{identity}"
                        ),
                    )
                )
    return tuple(sorted(values, key=lambda value: value.example_identity_sha256))


def _c3(
    candidate: CompiledCandidateIdentityReceipt,
) -> C3FullStackEvaluationReceipt:
    return evaluate_c3_full_stack(
        _records("C3_full_stack_source_free", "C3"),
        b3_evaluation=_b3(),
        source_model_sha256=_digest("source-model"),
        panel_manifest_sha256=_digest("panels"),
        role_capture_receipt_sha256=_digest("C3-capture"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
        candidate_identity=candidate,
    )


def _resources(
    candidate: CompiledCandidateIdentityReceipt,
    *,
    passing: bool = True,
) -> D3ResourceComparisonReceipt:
    source_boundary = 7_000_000
    candidate_boundary = 1_000_000 if passing else 8_000_000
    native_full = 10_000_000
    expected = native_full - source_boundary + candidate_boundary
    measured = expected if passing else native_full + 2_000_000
    return D3ResourceComparisonReceipt(
        candidate_identity_sha256=candidate.artifact_sha256,
        parameter_formula_receipt_sha256=_digest("parameter-formula"),
        serialization_protocol_sha256=_digest("serialization-protocol"),
        native_model_file_sha256=_digest("native-file"),
        compiled_model_file_sha256=_digest("compiled-file"),
        native_full_model_parameter_count=native_full,
        native_replaced_boundary_parameter_count=source_boundary,
        compiled_replacement_boundary_parameter_count=candidate_boundary,
        expected_compiled_full_model_parameter_count=expected,
        measured_compiled_full_model_parameter_count=measured,
        native_serialized_model_bytes=100_000_000,
        compiled_serialized_model_bytes=(
            40_000_000 if passing else 110_000_000
        ),
        formula_parameter_reduction_passed=passing,
        measured_parameter_reduction_passed=passing,
        serialized_byte_reduction_passed=passing,
        passed=passing,
    )


def _trace(
    candidate: CompiledCandidateIdentityReceipt,
    records: tuple[FullStackExampleAggregate, ...],
    *,
    passing: bool = True,
) -> D3NoNativeMLPTraceReceipt:
    forward_calls = 3
    return D3NoNativeMLPTraceReceipt(
        candidate_identity_sha256=candidate.artifact_sha256,
        role_capture_receipt_sha256=_digest("D3-capture"),
        trace_protocol_sha256=_digest("trace-protocol"),
        module_inventory_sha256=_digest("module-inventory"),
        trace_event_stream_sha256=_digest("trace-events"),
        profiler_backend="mlx-profiler-plus-module-hooks-v1",
        traced_forward_call_count=forward_calls,
        traced_example_count=len(records),
        traced_predicted_token_count=sum(
            value.predicted_token_count for value in records
        ),
        compiled_layer_call_counts=tuple(
            forward_calls for _ in range(18)
        ),
        native_mlp_module_call_count=0 if passing else 1,
        native_mlp_projection_call_count=0,
        native_mlp_profiler_op_count=0,
        native_mlp_fallback_call_count=0,
        native_mlp_parameter_storage_alias_count=0,
        passed=passing,
    )


def test_b3_additive_fidelity_multimode_gates_and_strict_round_trip() -> None:
    receipt = _b3()
    assert receipt.passed is True
    assert len(receipt.layer_decisions) == 18
    first = receipt.layer_decisions[0]
    assert first.velocity_nrmse == pytest.approx(0.20)
    assert first.aggregate_velocity_cosine == pytest.approx(0.97)
    assert first.minimum_conditional_mode_mass == pytest.approx(0.5)
    assert first.minimum_state_mode_mutual_information_nats == pytest.approx(0.1)

    encoded = json.loads(json.dumps(receipt.state_dict()))
    restored = B3TeacherForcedEvaluationReceipt.from_state_dict(encoded)
    assert restored.artifact_sha256 == receipt.artifact_sha256
    assert restored.state_dict() == receipt.state_dict()

    collapsed = _b3(
        first_layer=_layer(
            0,
            chart_count=2,
            mode_count=2,
            masses=(0.98, 0.02),
            mutual_information=0.02,
        )
    )
    assert collapsed.passed is False
    assert collapsed.layer_decisions[0].failed_gates == (
        "chart_conditional_mode_mass",
        "chart_state_mode_mutual_information",
    )


def test_c3_exact_additive_slices_point_gates_and_no_raw_state() -> None:
    receipt = _c3(_candidate())
    assert receipt.passed is True
    assert tuple(value.slice_id for value in receipt.metric_slices) == (
        "pooled",
        *(f"family:{value}" for value in FAMILIES),
        "hard_long",
    )
    pooled = receipt.metric_slices[0]
    assert pooled.predicted_token_count == 2_400
    assert pooled.delta_nll_nats_per_token == pytest.approx(0.008)
    assert pooled.native_to_candidate_kl_per_token == pytest.approx(0.007)
    assert pooled.top1_agreement == pytest.approx(0.98)
    assert pooled.top1_error == pytest.approx(0.02)
    hard_long = receipt.metric_slices[-1]
    assert hard_long.predicted_token_count == 1_200

    encoded = json.loads(json.dumps(receipt.state_dict()))
    restored = C3FullStackEvaluationReceipt.from_state_dict(encoded)
    assert restored.state_dict() == receipt.state_dict()
    serialized = json.dumps(encoded)
    assert '"prompt_text": "' not in serialized
    assert "raw_tokens" in serialized
    assert '"contains_raw_tokens": false' in serialized


def test_c3_parser_rejects_raw_fields_types_and_rehashed_derived_values() -> None:
    receipt = _c3(_candidate())

    with_raw = copy.deepcopy(receipt.state_dict())
    with_raw["prompt_text"] = "forbidden"
    with pytest.raises(ValueError, match="fields are invalid"):
        C3FullStackEvaluationReceipt.from_state_dict(with_raw)

    bool_count = copy.deepcopy(receipt.state_dict())
    bool_count["example_aggregates"][0]["predicted_token_count"] = True
    bool_count["example_aggregates"][0]["artifact_sha256"] = ""
    bool_count["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="integer"):
        C3FullStackEvaluationReceipt.from_state_dict(bool_count)

    integer_float = copy.deepcopy(receipt.state_dict())
    integer_float["example_aggregates"][0]["delta_nll_sum_nats"] = 1
    integer_float["example_aggregates"][0]["artifact_sha256"] = ""
    integer_float["artifact_sha256"] = ""
    with pytest.raises(TypeError, match="finite float"):
        C3FullStackEvaluationReceipt.from_state_dict(integer_float)

    rehashed = copy.deepcopy(receipt.state_dict())
    rehashed["metric_slices"][0]["delta_nll_nats_per_token"] += 0.1
    rehashed["metric_slices"][0]["artifact_sha256"] = ""
    rehashed["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="does not recompute"):
        C3FullStackEvaluationReceipt.from_state_dict(rehashed)

    wrong_protocol = copy.deepcopy(receipt.state_dict())
    wrong_protocol["evaluation_protocol_sha256"] = _digest("forged-policy")
    wrong_protocol["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="not frozen"):
        C3FullStackEvaluationReceipt.from_state_dict(wrong_protocol)


def test_c3_family_and_hard_long_point_failures_are_vetoes() -> None:
    candidate = _candidate()
    changed = []
    for value in _records("C3_full_stack_source_free", "C3-veto"):
        delta_sum = (
            3.0 if value.hard_long else value.delta_nll_sum_nats
        )
        kl_sum = (
            5.0
            if value.family_id == "family-0"
            else value.native_to_candidate_kl_sum_nats
        )
        changed.append(
            FullStackExampleAggregate(
                role=value.role,
                example_identity_sha256=value.example_identity_sha256,
                family_id=value.family_id,
                hard_long=value.hard_long,
                predicted_token_count=value.predicted_token_count,
                delta_nll_sum_nats=float(delta_sum),
                native_to_candidate_kl_sum_nats=float(kl_sum),
                top1_agreement_count=value.top1_agreement_count,
                example_execution_receipt_sha256=(
                    value.example_execution_receipt_sha256
                ),
            )
        )
    receipt = evaluate_c3_full_stack(
        tuple(changed),
        b3_evaluation=_b3(),
        source_model_sha256=_digest("source-model"),
        panel_manifest_sha256=_digest("panels"),
        role_capture_receipt_sha256=_digest("C3-capture"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
        candidate_identity=candidate,
    )
    assert receipt.passed is False
    assert "hard_long.delta_nll" in receipt.failed_gates
    assert "family:family-0.kl" in receipt.failed_gates


def test_c3_requires_the_exact_passing_b3_and_unchanged_selection() -> None:
    records = _records("C3_full_stack_source_free", "C3-authority")
    candidate = _candidate()
    failing_b3 = _b3(first_layer=_layer(0, nrmse=0.50))
    assert failing_b3.passed is False
    with pytest.raises(ValueError, match="until all B3 gates pass"):
        evaluate_c3_full_stack(
            records,
            b3_evaluation=failing_b3,
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("C3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=candidate,
        )

    passing_b3 = _b3()
    changed_selection = replace(
        candidate,
        selection_receipt_sha256=_digest("different-selection"),
        artifact_sha256="",
    )
    with pytest.raises(ValueError, match="candidate selections differ"):
        evaluate_c3_full_stack(
            records,
            b3_evaluation=passing_b3,
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("C3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=changed_selection,
        )

    with pytest.raises(TypeError, match="exact typed B3"):
        evaluate_c3_full_stack(
            records,
            b3_evaluation=passing_b3.state_dict(),  # type: ignore[arg-type]
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("C3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=candidate,
        )


def test_d3_policy_freezes_approximate_simultaneous_whole_example_method() -> None:
    policy = d3_bootstrap_policy()
    assert BOOTSTRAP_REPLICATE_COUNT == 10_000
    assert policy["scientific_status"] == (
        "approximate_bootstrap_not_distribution_free"
    )
    assert policy["resampling_unit"] == "whole_example"
    assert policy["required_strata"] == (
        "both_hard_long_values_within_each_of_six_families"
    )
    assert policy["max_t"] == (
        "one_shared_critical_across_every_slice_and_harm_metric"
    )
    assert len(d3_bootstrap_policy_sha256()) == 64
    evaluation_policy = evaluation_module.heldout_evaluation_policy()
    assert "same_frozen_serialization_protocol" in (
        evaluation_policy["D3"]["serialized_byte_comparison"]
    )


def test_d3_terminal_receipt_bounds_resources_trace_and_round_trip() -> None:
    candidate = _candidate()
    c3 = _c3(candidate)
    records = _records("D3_final_heldout_validation", "D3")
    receipt = evaluate_d3_final(
        records,
        c3_evaluation=c3,
        source_model_sha256=_digest("source-model"),
        panel_manifest_sha256=_digest("panels"),
        role_capture_receipt_sha256=_digest("D3-capture"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
        candidate_identity=candidate,
        resource_comparison=_resources(candidate),
        no_native_mlp_trace=_trace(candidate, records),
    )
    assert receipt.passed is True
    assert receipt.failed_gates == ()
    assert len(receipt.bootstrap_stratum_ids) == 12
    assert len(receipt.simultaneous_bounds) == 24
    assert receipt.bootstrap_max_t_critical_value > 0.0
    assert receipt.bootstrap_is_distribution_free is False
    assert receipt.resource_comparison.passed is True
    assert receipt.no_native_mlp_trace.passed is True
    pooled_top1 = next(
        value
        for value in receipt.simultaneous_bounds
        if value.slice_id == "pooled" and value.harm_metric == "top1_error"
    )
    assert pooled_top1.presentation_metric == "top1_agreement"
    assert pooled_top1.presentation_one_sided_bound == pytest.approx(
        1.0 - pooled_top1.simultaneous_upper_bound
    )

    encoded = json.loads(json.dumps(receipt.state_dict()))
    restored = D3FinalEvaluationReceipt.from_state_dict(encoded)
    assert restored.state_dict() == receipt.state_dict()

    tampered = copy.deepcopy(encoded)
    tampered["bootstrap_replicate_digest_sha256"] = _digest("forged")
    tampered["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="replicate digest does not recompute"):
        D3FinalEvaluationReceipt.from_state_dict(tampered)

    negative = evaluate_d3_final(
        records,
        c3_evaluation=c3,
        source_model_sha256=_digest("source-model"),
        panel_manifest_sha256=_digest("panels"),
        role_capture_receipt_sha256=_digest("D3-capture"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
        candidate_identity=candidate,
        resource_comparison=_resources(candidate, passing=False),
        no_native_mlp_trace=_trace(candidate, records, passing=False),
    )
    assert negative.passed is False
    assert "resource_comparison" in negative.failed_gates
    assert "no_native_mlp_trace" in negative.failed_gates


def test_d3_requires_an_exact_strictly_replayable_c3_parent() -> None:
    candidate = _candidate()
    c3 = _c3(candidate)
    records = _records("D3_final_heldout_validation", "D3-parent-type")

    class ForgedC3(C3FullStackEvaluationReceipt):
        pass

    forged = object.__new__(ForgedC3)
    with pytest.raises(TypeError, match="exact typed C3"):
        evaluate_d3_final(
            records,
            c3_evaluation=forged,
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("D3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=candidate,
            resource_comparison=_resources(candidate),
            no_native_mlp_trace=_trace(candidate, records),
        )

    with pytest.raises(TypeError, match="exact typed C3"):
        evaluate_d3_final(
            records,
            c3_evaluation=c3.state_dict(),  # type: ignore[arg-type]
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("D3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=candidate,
            resource_comparison=_resources(candidate),
            no_native_mlp_trace=_trace(candidate, records),
        )


def test_d3_pooled_point_can_pass_while_simultaneous_bound_vetoes() -> None:
    candidate = _candidate()
    c3 = _c3(candidate)
    records = _records(
        "D3_final_heldout_validation",
        "D3-near-boundary",
        delta_center=0.019,
    )
    receipt = evaluate_d3_final(
        records,
        c3_evaluation=c3,
        source_model_sha256=_digest("source-model"),
        panel_manifest_sha256=_digest("panels"),
        role_capture_receipt_sha256=_digest("D3-capture"),
        evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
        candidate_identity=candidate,
        resource_comparison=_resources(candidate),
        no_native_mlp_trace=_trace(candidate, records),
    )
    assert "pooled.delta_nll" not in receipt.point_failed_gates
    assert "pooled_bound.delta_nll" in receipt.bound_failed_gates
    assert receipt.passed is False


def test_d3_requires_unchanged_candidate_and_pairwise_disjoint_examples() -> None:
    candidate = _candidate()
    c3 = _c3(candidate)
    records = _records("D3_final_heldout_validation", "D3")
    different = _candidate("different")
    with pytest.raises(ValueError, match="not unchanged"):
        evaluate_d3_final(
            records,
            c3_evaluation=c3,
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("D3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=different,
            resource_comparison=_resources(different),
            no_native_mlp_trace=_trace(different, records),
        )

    overlapping = list(records)
    c3_first = c3.example_aggregates[0]
    first = overlapping[0]
    overlapping[0] = FullStackExampleAggregate(
        role=first.role,
        example_identity_sha256=c3_first.example_identity_sha256,
        family_id=first.family_id,
        hard_long=first.hard_long,
        predicted_token_count=first.predicted_token_count,
        delta_nll_sum_nats=first.delta_nll_sum_nats,
        native_to_candidate_kl_sum_nats=(
            first.native_to_candidate_kl_sum_nats
        ),
        top1_agreement_count=first.top1_agreement_count,
        example_execution_receipt_sha256=(
            first.example_execution_receipt_sha256
        ),
    )
    overlapping.sort(key=lambda value: value.example_identity_sha256)
    with pytest.raises(ValueError, match="not disjoint"):
        evaluate_d3_final(
            tuple(overlapping),
            c3_evaluation=c3,
            source_model_sha256=_digest("source-model"),
            panel_manifest_sha256=_digest("panels"),
            role_capture_receipt_sha256=_digest("D3-capture"),
            evaluation_protocol_sha256=heldout_evaluation_policy_sha256(),
            candidate_identity=candidate,
            resource_comparison=_resources(candidate),
            no_native_mlp_trace=_trace(candidate, tuple(overlapping)),
        )


def test_d3_zero_se_exact_path_and_inconsistent_underflow_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _records(
        "D3_final_heldout_validation",
        "D3-zero",
        members_per_stratum=1,
        variable=False,
    )
    _strata, _digest_value, critical, bounds = (
        evaluation_module._bootstrap_bounds(records)
    )
    assert critical == 0.0
    assert all(value.bootstrap_standard_error == 0.0 for value in bounds)
    assert all(
        value.simultaneous_upper_bound == value.point_estimate
        for value in bounds
    )

    slices = evaluation_module._metric_slices(records)
    coordinate_ids = tuple(
        f"{value.slice_id}|{metric}"
        for value in slices
        for metric in (
            "delta_nll",
            "native_to_candidate_kl",
            "top1_error",
        )
    )
    observed = [
        coordinate
        for value in slices
        for coordinate in value.harm_coordinates
    ]
    observed[0] = 0.0
    row_a = list(observed)
    row_b = list(observed)
    row_a[0] = 1e-200
    row_b[0] = -1e-200

    def fake_matrix(_records):
        return coordinate_ids, ("fake",), (tuple(row_a), tuple(row_b)), _digest("x")

    monkeypatch.setattr(evaluation_module, "_bootstrap_matrix", fake_matrix)
    zero_delta_records = tuple(
        FullStackExampleAggregate(
            role=value.role,
            example_identity_sha256=value.example_identity_sha256,
            family_id=value.family_id,
            hard_long=value.hard_long,
            predicted_token_count=value.predicted_token_count,
            delta_nll_sum_nats=0.0,
            native_to_candidate_kl_sum_nats=(
                value.native_to_candidate_kl_sum_nats
            ),
            top1_agreement_count=value.top1_agreement_count,
            example_execution_receipt_sha256=(
                value.example_execution_receipt_sha256
            ),
        )
        for value in records
    )
    with pytest.raises(ValueError, match="zero bootstrap SE"):
        evaluation_module._bootstrap_bounds(zero_delta_records)


def test_resource_and_trace_negative_evidence_is_terminal_not_malformed() -> None:
    candidate = _candidate()
    records = _records("D3_final_heldout_validation", "D3")
    resources = _resources(candidate, passing=False)
    trace = _trace(candidate, records, passing=False)
    assert resources.passed is False
    assert resources.formula_parameter_reduction_passed is False
    assert resources.serialized_byte_reduction_passed is False
    assert trace.passed is False
    assert trace.native_mlp_module_call_count == 1


def test_all_nested_json_parsers_reject_numeric_type_confusion() -> None:
    family = _family(FAMILIES[0])
    family_bool = family.state_dict()
    family_bool["scalar_count"] = True
    family_bool["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="integer"):
        LayerTeacherForcedFamilyAggregate.from_state_dict(family_bool)
    family_int_float = family.state_dict()
    family_int_float["teacher_energy"] = 1
    family_int_float["artifact_sha256"] = ""
    with pytest.raises(TypeError, match="finite float"):
        LayerTeacherForcedFamilyAggregate.from_state_dict(family_int_float)

    example = _records("C3_full_stack_source_free", "typed")[0]
    example_bool = example.state_dict()
    example_bool["top1_agreement_count"] = False
    example_bool["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="integer"):
        FullStackExampleAggregate.from_state_dict(example_bool)
    example_int_float = example.state_dict()
    example_int_float["native_to_candidate_kl_sum_nats"] = 1
    example_int_float["artifact_sha256"] = ""
    with pytest.raises(TypeError, match="finite float"):
        FullStackExampleAggregate.from_state_dict(example_int_float)

    bound = D3SimultaneousBoundReceipt(
        coordinate_ordinal=0,
        slice_id="pooled",
        harm_metric="delta_nll",
        point_estimate=0.01,
        bootstrap_standard_error=0.001,
        simultaneous_upper_bound=0.012,
        presentation_metric="delta_nll_nats_per_token",
        presentation_point_estimate=0.01,
        presentation_one_sided_bound=0.012,
    )
    bound_bool = bound.state_dict()
    bound_bool["coordinate_ordinal"] = True
    bound_bool["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="integer"):
        D3SimultaneousBoundReceipt.from_state_dict(bound_bool)
    bound_int_float = bound.state_dict()
    bound_int_float["point_estimate"] = 1
    bound_int_float["artifact_sha256"] = ""
    with pytest.raises(TypeError, match="finite float"):
        D3SimultaneousBoundReceipt.from_state_dict(bound_int_float)

    candidate = _candidate()
    resources = _resources(candidate)
    resource_bool = resources.state_dict()
    resource_bool["native_full_model_parameter_count"] = True
    resource_bool["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="integer"):
        D3ResourceComparisonReceipt.from_state_dict(resource_bool)
    resource_int_bool = resources.state_dict()
    resource_int_bool["passed"] = 1
    resource_int_bool["artifact_sha256"] = ""
    with pytest.raises(TypeError, match="bool"):
        D3ResourceComparisonReceipt.from_state_dict(resource_int_bool)

    records = _records("D3_final_heldout_validation", "trace-typed")
    trace = _trace(candidate, records)
    trace_bool = trace.state_dict()
    trace_bool["traced_forward_call_count"] = True
    trace_bool["artifact_sha256"] = ""
    with pytest.raises(ValueError, match="integer"):
        D3NoNativeMLPTraceReceipt.from_state_dict(trace_bool)
    trace_int_bool = trace.state_dict()
    trace_int_bool["passed"] = 1
    trace_int_bool["artifact_sha256"] = ""
    with pytest.raises(TypeError, match="bool"):
        D3NoNativeMLPTraceReceipt.from_state_dict(trace_int_bool)


def test_module_has_no_fresh_role_panel_loader_or_raw_constructor() -> None:
    public = set(evaluation_module.__all__)
    forbidden_fragments = (
        "load_b3",
        "load_c3",
        "load_d3",
        "open_b3",
        "open_c3",
        "open_d3",
        "from_panel",
        "from_logits",
        "from_tokens",
    )
    assert not any(
        fragment in name.lower()
        for name in public
        for fragment in forbidden_fragments
    )
