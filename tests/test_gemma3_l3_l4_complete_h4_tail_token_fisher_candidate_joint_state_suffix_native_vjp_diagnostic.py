from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_native_vjp_diagnostic as diagnostic,
)
from tests.test_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_contraction_precision_diagnostic import (
    _valid_v12_receipt_grid,
)


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _mock_v12_report() -> dict[str, object]:
    raw, raw_set, casts, cast_set, environment = _valid_v12_receipt_grid()
    authentication = diagnostic.v12diag._validate_contraction_receipts(
        raw_receipts=raw,
        raw_receipt_set_sha256=raw_set,
        cast_receipts=casts,
        cast_receipt_set_sha256=cast_set,
        reduction_environment=environment,
    )
    environment = {
        **environment,
        "artifact_sha256": diagnostic.V12_REDUCTION_ENVIRONMENT_SHA256,
    }
    authentication = {
        **authentication,
        "raw_gradient_receipt_set_sha256": (
            diagnostic.V12_RAW_GRADIENT_SET_SHA256
        ),
        "cast_only_jvp_receipt_set_sha256": (
            diagnostic.V12_CAST_ONLY_JVP_SET_SHA256
        ),
        "typed_reduction_environment_sha256": (
            diagnostic.V12_REDUCTION_ENVIRONMENT_SHA256
        ),
    }
    return {
        "schema": diagnostic.v12diag._SCHEMA,
        "classification": "unresolved_forward_reverse_ad_kernel_mismatch",
        "passed": False,
        "family_equal_contraction_precision_comparison": {
            "artifact_sha256": diagnostic.V12_COMPARISON_ARTIFACT_SHA256,
            "earliest_passing_stage": None,
            "finite_correction_eligible": False,
        },
        "raw_float32_gradient_selection_receipts": raw,
        "raw_float32_gradient_selection_receipt_set_sha256": (
            diagnostic.V12_RAW_GRADIENT_SET_SHA256
        ),
        "cast_only_jvp_validation_receipts": casts,
        "cast_only_jvp_validation_receipt_set_sha256": (
            diagnostic.V12_CAST_ONLY_JVP_SET_SHA256
        ),
        "typed_reduction_environment": environment,
        "v12_receipt_authentication": authentication,
        "integrity_gate_results": [[f"gate-{index}", True] for index in range(12)],
        "report_sha256": diagnostic.V12_REPORT_SHA256,
    }


def test_v12_loader_uses_exact_six_pins_rehashes_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    report = _mock_v12_report()
    original_domain_sha256 = diagnostic.v10diag.token_v1._domain_sha256

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return report

    def hash_payload(payload: object, *, domain: bytes) -> str:
        if domain == diagnostic.v12diag._REPORT_DOMAIN:
            return diagnostic.V12_REPORT_SHA256
        return original_domain_sha256(payload, domain=domain)

    monkeypatch.setattr(
        diagnostic.v10diag.token_v1, "_load_pinned_report", load
    )
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1, "_domain_sha256", hash_payload
    )
    monkeypatch.setattr(
        diagnostic.v12diag,
        "_validate_contraction_receipts",
        lambda **kwargs: report["v12_receipt_authentication"],
    )
    assert diagnostic._load_v12_report("v12.json") is report
    assert captured["expected_file_sha256"] == diagnostic.V12_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V12_REPORT_SHA256
    assert (
        report["family_equal_contraction_precision_comparison"][  # type: ignore[index]
            "artifact_sha256"
        ]
        == diagnostic.V12_COMPARISON_ARTIFACT_SHA256
    )
    assert (
        report["raw_float32_gradient_selection_receipt_set_sha256"]
        == diagnostic.V12_RAW_GRADIENT_SET_SHA256
    )
    assert (
        report["cast_only_jvp_validation_receipt_set_sha256"]
        == diagnostic.V12_CAST_ONLY_JVP_SET_SHA256
    )
    assert (
        report["typed_reduction_environment"]["artifact_sha256"]  # type: ignore[index]
        == diagnostic.V12_REDUCTION_ENVIRONMENT_SHA256
    )

    changed = {**report, "classification": "changed"}
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: changed,
    )
    with pytest.raises(RuntimeError, match="V12 suffix native-VJP anchor differs"):
        diagnostic._load_v12_report("v12.json")


def test_live_v12_section_authentication_is_complete_exact_and_fail_closed() -> None:
    keys = (
        "input_binding",
        "folds",
        "prompt_receipts",
        "replayed_v11_suffix_jvp_evidence",
        "replayed_v11_suffix_runtime_node_receipts",
        "replayed_v11_suffix_runtime_node_receipt_set_sha256",
        "replayed_v11_discrete_cast_prompt_receipts",
        "replayed_v11_discrete_cast_prompt_receipt_set_sha256",
        "replayed_v11_family_equal_suffix_jvp_comparison",
        "raw_float32_gradient_selection_receipts",
        "raw_float32_gradient_selection_receipt_set_sha256",
        "cast_only_jvp_validation_receipts",
        "cast_only_jvp_validation_receipt_set_sha256",
        "contraction_precision_evidence",
        "family_equal_contraction_precision_comparison",
        "typed_reduction_environment",
        "v12_receipt_authentication",
        "resources",
    )
    report: dict[str, object] = {key: {"key": key} for key in keys}
    report["v11_control_binding"] = {
        "live_reproduction": {"all_live_sections_canonically_equal": True}
    }
    live = {
        **{key: report[key] for key in keys},
        "v11_live_reproduction": {
            "all_live_sections_canonically_equal": True
        },
    }
    receipt = diagnostic._authenticate_live_v12_sections(
        report=report, live_sections=live
    )
    assert receipt["all_live_sections_canonically_equal"] is True
    assert receipt["section_count"] == 19
    assert receipt["v12_report_file_sha256"] == diagnostic.V12_REPORT_FILE_SHA256
    assert receipt["v12_report_sha256"] == diagnostic.V12_REPORT_SHA256

    live["resources"] = {"changed": True}
    with pytest.raises(RuntimeError, match="resources"):
        diagnostic._authenticate_live_v12_sections(
            report=report, live_sections=live
        )


def test_integrity_safety_and_atomic_no_overwrite_publication(
    tmp_path: Path,
) -> None:
    diagnostic._require_integrity_gate_results(
        {"V12_exact": True, "native_receipts_exact": True}
    )
    with pytest.raises(RuntimeError, match="before publication"):
        diagnostic._require_integrity_gate_results(
            {"V12_exact": True, "native_receipts_exact": False}
        )
    safety = diagnostic._safety_metadata()
    assert safety["contains_native_VJP_tensors"] is False
    assert safety["artifact_must_remain_outside_git"] is True
    assert not _contains_tensor(safety)

    output = tmp_path / "v13.json"
    report: dict[str, object] = {
        "schema": diagnostic._SCHEMA,
        "artifact": {"file": str(output), "committable": False},
        "passed": False,
        "classification": "test",
        "safety": safety,
    }
    published = diagnostic._publish(report, output=output)
    assert output.exists()
    assert isinstance(published["report_sha256"], str)
    assert published["artifact"]["file_bytes"] == output.stat().st_size  # type: ignore[index]
    with pytest.raises(FileExistsError):
        diagnostic._publish(
            {
                "schema": diagnostic._SCHEMA,
                "artifact": {"file": str(output), "committable": False},
            },
            output=output,
        )


def test_native_core_resources_enforce_exact_3212_token_436_chunk_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_support = (
        (35, 36),
        (37, 38),
        (34, 35),
        (37, 38),
        (42, 43),
        (37, 38),
        (34, 35),
        (34, 35),
        (61, 62),
        (65, 66),
        (60, 61),
        (67, 68),
        (72, 73),
        (65, 66),
        (62, 63),
        (61, 62),
    )
    evidence: list[object] = []
    for index, (tokens, support) in enumerate(token_support):
        full_rows = support + 8
        evidence.append(
            SimpleNamespace(
                family_id=f"family-{index // 2}",
                resource_accounting={
                    "quadrature_node_count": 4,
                    "supervised_token_count": tokens,
                    "full_h4_row_count": full_rows,
                    "h4_width": 640,
                    "native_suffix_forward_count": 4,
                    "native_vjp_pullback_count": 4 * ((tokens + 7) // 8),
                    "logical_native_vjp_input_gradient_coordinate_count": (
                        4 * tokens * full_rows * 640
                    ),
                    "canonical_output_cotangent_row_count": 4 * tokens,
                    "native_token_directional_derivative_count": 4 * tokens,
                    "published_v11_jvp_node_token_reference_element_count": (
                        4 * tokens
                    ),
                    "native_GL4_token_weight_application_count": 4 * tokens,
                    "fresh_full_model_forward_count": 0,
                    "fresh_full_model_backward_count": 0,
                },
            )
        )
    resources = diagnostic._native_core_resource_accounting(
        evidence  # type: ignore[arg-type]
    )
    assert resources["native_quadrature_node_count"] == 64
    assert resources["native_supervised_token_count"] == 803
    assert resources["native_vjp_pullback_count"] == 436
    assert resources["canonical_output_cotangent_row_count"] == 3_212
    assert (
        resources["logical_native_vjp_input_gradient_coordinate_count"]
        == 130_048_000
    )
    assert resources["fresh_full_model_forward_count"] == 0
    assert resources["resource_counts_are_not_FLOPs_or_total_model_compute"] is True

    monkeypatch.setattr(diagnostic, "_EXPECTED_NATIVE_PULLBACK_CHUNKS", 435)
    with pytest.raises(RuntimeError, match="resource accounting differs"):
        diagnostic._native_core_resource_accounting(
            evidence  # type: ignore[arg-type]
        )


def _valid_native_receipt_grid() -> tuple[
    tuple[dict[str, object], ...],
    str,
    tuple[dict[str, object], ...],
    str,
]:
    token_support = (
        (35, 36),
        (37, 38),
        (34, 35),
        (37, 38),
        (42, 43),
        (37, 38),
        (34, 35),
        (34, 35),
        (61, 62),
        (65, 66),
        (60, 61),
        (67, 68),
        (72, 73),
        (65, 66),
        (62, 63),
        (61, 62),
    )
    runtime: list[dict[str, object]] = []
    resources: list[dict[str, object]] = []
    digest_index = 1

    def digest() -> str:
        nonlocal digest_index
        value = f"{digest_index:064x}"
        digest_index += 1
        return value

    ordinal = 0
    for prompt, (tokens, support) in enumerate(token_support):
        example = f"example-{prompt:02d}"
        family = f"family-{prompt // 2}"
        full_rows = support + 8
        for node in range(4):
            ordinal += 1
            chunks: list[dict[str, object]] = []
            for start in range(0, tokens, 8):
                stop = min(start + 8, tokens)
                chunk: dict[str, object] = {
                    "token_start": start,
                    "token_stop": stop,
                    "token_count": tokens,
                    "pullback_mechanism": "torch.func.vmap(pullback)",
                    "canonical_one_hot_token_cotangents": True,
                    "vmap_pullback_call_count": 1,
                    "token_cotangent_nonzero_count": stop - start,
                    "token_cotangent_element_count": (stop - start) * tokens,
                }
                chunk["artifact_sha256"] = (
                    diagnostic.v10diag.token_v1._domain_sha256(
                        chunk, domain=diagnostic.native_runtime._CHUNK_DOMAIN
                    )
                )
                chunks.append(chunk)
            metadata = {
                "ad_mechanism": "torch.func.vjp.reverse_mode",
                "vjp_has_aux": True,
                "vjp_transform_count": 1,
                "suffix_segment_call_count": 13,
                "logit_projection_call_count": 1,
                "h4_dtype_cast_count": 1,
                "token_cotangent_chunk_size": 8,
                "pullback_vectorization": (
                    "torch.vmap_over_canonical_token_cotangents"
                ),
                "vjp_pullback_chunk_call_count": (tokens + 7) // 8,
                "vmap_pullback_call_count": (tokens + 7) // 8,
                "token_count": tokens,
                "token_cotangent_coverage_count": tokens,
                "token_cotangent_nonzero_count": tokens,
                "token_cotangent_element_count": tokens * tokens,
                "full_suffix_h4_bitwise_equal": True,
                "full_suffix_logits_bitwise_equal": True,
                "full_suffix_token_teacher_kl_bitwise_equal": True,
                "direction_is_exactly_zero_outside_support": True,
                "chunk_receipts": tuple(chunks),
            }
            runtime_artifact = diagnostic.v10diag.token_v1._domain_sha256(
                metadata, domain=diagnostic.native_runtime._RECEIPT_DOMAIN
            )
            metadata["artifact_sha256"] = runtime_artifact
            value: dict[str, object] = {
                "example_id": example,
                "family_id": family,
                "node_index": node,
                "path_fraction_hex": (
                    diagnostic.v10diag.GL4_UNIT_INTERVAL_NODES[node].hex()
                ),
                "quadrature_weight_hex": (
                    diagnostic.v10diag.GL4_UNIT_INTERVAL_WEIGHTS[node].hex()
                ),
                "live_result_ordinal": ordinal,
                "pinned_v10_core_node_receipt_artifact_sha256": digest(),
                "pinned_v12_contraction_node_artifact_sha256": digest(),
                "native_suffix_vjp_runtime_artifact_sha256": runtime_artifact,
                "provider_artifact_sha256": digest(),
                "execution_artifact_sha256": digest(),
                "path_h4_sha256": digest(),
                "supervised_grid_sha256": digest(),
                "primal_token_teacher_kl_sha256": digest(),
                "native_suffix_vjp_runtime_receipt": metadata,
                "native_vjp_primal_count": 1,
                "runtime_path_h4_matches_constructed_f64_path_bitwise": True,
                "runtime_cast_h4_matches_V10_full_node_bitwise": True,
                "runtime_full_logits_matches_V10_full_node_bitwise": True,
                "runtime_primal_token_KL_matches_V10_f64_bitwise": True,
                "same_node_direction_matches_V11_bitwise": True,
                "no_discarded_preflight_result": True,
                "raw_tensors_serialized": False,
            }
            value["receipt_sha256"] = (
                diagnostic.v10diag.token_v1._domain_sha256(
                    value, domain=diagnostic._NATIVE_RUNTIME_NODE_DOMAIN
                )
            )
            runtime.append(value)
            resource: dict[str, object] = {
                "example_id": example,
                "family_id": family,
                "node_index": node,
                "native_runtime_receipt_sha256": value["receipt_sha256"],
                "token_count": tokens,
                "full_h4_row_count": full_rows,
                "support_h4_row_count": support,
                "outside_support_h4_row_count": 8,
                "h4_width": 640,
                "native_suffix_forward_count": 1,
                "native_vjp_primal_count": 1,
                "native_vjp_pullback_count": (tokens + 7) // 8,
                "suffix_segment_call_count": 13,
                "logit_projection_call_count": 1,
                "h4_dtype_cast_count": 1,
                "canonical_output_cotangent_row_count": tokens,
                "dense_output_cotangent_coordinate_count": tokens * tokens,
                "full_input_gradient_coordinate_count": (
                    tokens * full_rows * 640
                ),
                "support_contraction_product_count": tokens * support * 640,
                "outside_support_gradient_coordinate_count": tokens * 8 * 640,
                "direction_coordinate_validation_count": full_rows * 640,
                "outside_support_direction_zero_validation_count": 8 * 640,
                "outside_direction_nonzero_count": 0,
                "maximum_outside_direction_abs": 0.0,
                "outside_direction_zero_proved_before_support_contraction": True,
                "support_indices_sha256": digest(),
                "token_basis_cotangent_sha256": digest(),
                "transient_full_cotangent_sha256": digest(),
                "transient_full_input_gradient_sha256": digest(),
                "support_input_gradient_sha256": digest(),
                "contracted_directional_vector_sha256": digest(),
                "transient_full_cotangent_hashed": True,
                "transient_full_input_gradient_hashed": True,
                "transient_tensors_retained_after_result": False,
                "raw_tensors_serialized": False,
            }
            resource["artifact_sha256"] = (
                diagnostic.v10diag.token_v1._domain_sha256(
                    resource, domain=diagnostic._NATIVE_RESOURCE_NODE_DOMAIN
                )
            )
            resources.append(resource)
    runtime_values = tuple(runtime)
    resource_values = tuple(resources)
    runtime_set = diagnostic.v10diag.token_v1._domain_sha256(
        tuple(str(value["receipt_sha256"]) for value in runtime_values),
        domain=diagnostic._NATIVE_RUNTIME_SET_DOMAIN,
    )
    resource_set = diagnostic.v10diag.token_v1._domain_sha256(
        tuple(str(value["artifact_sha256"]) for value in resource_values),
        domain=diagnostic._NATIVE_RESOURCE_SET_DOMAIN,
    )
    return runtime_values, runtime_set, resource_values, resource_set


def test_native_receipts_rehash_ownership_bitwise_proofs_and_exact_resources() -> None:
    runtime, runtime_set, resources, resource_set = _valid_native_receipt_grid()
    authentication = diagnostic._validate_native_vjp_receipts(
        runtime_receipts=runtime,
        runtime_receipt_set_sha256=runtime_set,
        resource_receipts=resources,
        resource_receipt_set_sha256=resource_set,
    )
    assert authentication["runtime_receipt_count"] == 64
    assert authentication["resource_receipt_count"] == 64
    assert authentication["first_live_result_ordinal"] == 1
    assert authentication["last_live_result_ordinal"] == 64
    assert authentication["canonical_output_cotangent_row_count"] == 3_212
    assert authentication["native_vjp_pullback_count"] == 436
    assert authentication["dense_output_cotangent_coordinate_count"] == 174_292
    assert authentication["full_input_gradient_coordinate_count"] == 130_048_000
    assert authentication["support_contraction_product_count"] == 113_602_560
    assert (
        authentication["outside_support_gradient_coordinate_count"]
        == 16_445_440
    )

    changed = [dict(value) for value in runtime]
    changed[0]["runtime_primal_token_KL_matches_V10_f64_bitwise"] = False
    with pytest.raises(RuntimeError, match="runtime receipt drifted"):
        diagnostic._validate_native_vjp_receipts(
            runtime_receipts=changed,
            runtime_receipt_set_sha256=runtime_set,
            resource_receipts=resources,
            resource_receipt_set_sha256=resource_set,
        )

    duplicated = list(resources)
    duplicated[-1] = duplicated[0]
    with pytest.raises(RuntimeError, match="order or ownership"):
        diagnostic._validate_native_vjp_receipts(
            runtime_receipts=runtime,
            runtime_receipt_set_sha256=runtime_set,
            resource_receipts=duplicated,
            resource_receipt_set_sha256=resource_set,
        )


def test_no_knob_cli_fixed_output_and_pyproject_entry() -> None:
    assert vars(diagnostic.build_parser().parse_args([])) == {}
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(["--output", "elsewhere.json"])
    assert diagnostic.DEFAULT_OUTPUT.name.endswith(
        "candidate-joint-state-suffix-native-vjp-gl4-lofo-a-fit16-dev-v13.json"
    )
    pyproject = Path("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-candidate-"
        "joint-state-suffix-native-vjp-gl4-v13-a-dev = "
        '"fisher_graph.gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_'
        'joint_state_suffix_native_vjp_diagnostic:main"'
    ) in pyproject


def test_main_prints_published_report_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_native_vjp_diagnostic",
        lambda: {
            "artifact": {"file": ".local-runs/v13.json"},
            "report_sha256": "a" * 64,
            "classification": (
                "v10_gradient_source_or_execution_path_difference_supported"
            ),
        },
    )
    diagnostic.main([])
    assert capsys.readouterr().out.splitlines() == [
        "report: .local-runs/v13.json",
        f"report sha256: {'a' * 64}",
        "classification: v10_gradient_source_or_execution_path_difference_supported",
    ]


def test_mocked_run_authenticates_v12_pre_post_then_gates_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    v12_report = {
        "schema": diagnostic.v12diag._SCHEMA,
        "classification": "unresolved_forward_reverse_ad_kernel_mismatch",
        "passed": False,
    }
    v11_report: dict[str, object] = {}
    v10_report = {"resources": {"v10": 1}}
    traces = tuple(
        SimpleNamespace(
            family_id=f"family-{index // 2}", endpoint=object()
        )
        for index in range(16)
    )
    analytic = SimpleNamespace(
        v7_phases=SimpleNamespace(
            v6_phases=SimpleNamespace(
                row_bank=SimpleNamespace(resources={})
            )
        )
    )
    live = SimpleNamespace(evidence=(), resources={})
    comparison = SimpleNamespace(
        replayed_v12_comparison_artifact_sha256=(
            diagnostic.V12_COMPARISON_ARTIFACT_SHA256
        ),
        replayed_v11_comparison_artifact_sha256=(
            diagnostic.v12diag.V11_COMPARISON_ARTIFACT_SHA256
        ),
        nodewise_gate_results={
            "overall_nodewise_symmetric_relative_RMSE_at_most_0_0001": True,
            "every_family_nodewise_symmetric_relative_RMSE_at_most_0_0001": True,
        },
        integrated_gate_results={
            "overall_integrated_symmetric_relative_RMSE_at_most_0_0001": True,
            "every_family_integrated_symmetric_relative_RMSE_at_most_0_0001": True,
        },
        nodewise_passed=True,
        integrated_passed=True,
        classification="v10_gradient_source_or_execution_path_difference_supported",
        telescope_metrics=SimpleNamespace(passed=True),
        family_summaries=(
            SimpleNamespace(telescope_metrics=SimpleNamespace(passed=True)),
        ),
        metadata=lambda: {"comparison": True},
    )
    runtime_receipts = tuple(
        {
            "runtime_path_h4_matches_constructed_f64_path_bitwise": True,
            "runtime_cast_h4_matches_V10_full_node_bitwise": True,
            "runtime_full_logits_matches_V10_full_node_bitwise": True,
            "runtime_primal_token_KL_matches_V10_f64_bitwise": True,
        }
        for _ in range(64)
    )
    resource_receipts = tuple(
        {
            "outside_direction_zero_proved_before_support_contraction": True,
            "outside_direction_nonzero_count": 0,
            "maximum_outside_direction_abs": 0.0,
        }
        for _ in range(64)
    )
    contraction = SimpleNamespace(
        suffix=SimpleNamespace(resources={}), resources={}
    )
    native = SimpleNamespace(
        contraction=contraction,
        evidence=(),
        comparison=comparison,
        runtime_receipts=runtime_receipts,
        runtime_receipt_set_sha256="1" * 64,
        resource_receipts=resource_receipts,
        resource_receipt_set_sha256="2" * 64,
        runtime_resources={
            "canonical_token_cotangent_coverage_count": 3_212,
            "vmap_pullback_call_count": 436,
        },
        core_resources={},
    )

    class Context:
        def validate_immutable_inputs(self) -> None:
            order.append("immutable")

        def close(self) -> None:
            order.append("close")

    class Collector:
        def __init__(self, *, context: object) -> None:
            self.reduction_environment = {}

        def finalize(self, evidence: object) -> object:
            order.append("native_finalize")
            return native

    monkeypatch.setattr(
        diagnostic,
        "_load_v12_report",
        lambda path: order.append("v12_load") or v12_report,
    )
    monkeypatch.setattr(
        diagnostic.v12diag, "_load_v11_report", lambda path: v11_report
    )
    monkeypatch.setattr(
        diagnostic.v11diag, "_load_v10_report", lambda path: v10_report
    )
    monkeypatch.setattr(
        diagnostic.v11diag, "_index_v10_receipts", lambda report: {}
    )
    monkeypatch.setattr(
        diagnostic.v10diag.v3diag,
        "_load_expanded_parent",
        lambda path: object(),
    )
    for owner, name in (
        (diagnostic.v10diag.v9diag.v4diag, "_load_v3_report"),
        (diagnostic.v10diag.v9diag.v5diag, "_load_v4_report"),
        (diagnostic.v10diag.v6diag, "_load_v5_report"),
        (diagnostic.v10diag.v7diag, "_load_v6_report"),
        (diagnostic.v10diag.v8diag, "_load_v7_report"),
        (diagnostic.v10diag.v9diag, "_load_v8_report"),
        (diagnostic.v10diag, "_load_v9_report"),
    ):
        monkeypatch.setattr(owner, name, lambda path: {})
    monkeypatch.setattr(
        diagnostic.v10diag.v9diag,
        "_index_v8_endpoint_observations",
        lambda report: {},
    )
    monkeypatch.setattr(
        diagnostic.v10diag, "_index_v9_receipts", lambda report: {}
    )
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1,
        "_validate_output",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        diagnostic.v10diag.expanded, "_transfer_receipts", lambda report: ()
    )
    monkeypatch.setattr(
        diagnostic.v10diag.transfer_support,
        "_load_committed_basis",
        lambda **kwargs: (object(), object(), object()),
    )
    monkeypatch.setattr(
        diagnostic.v10diag,
        "prepare_complete_h4_rank320_live_context",
        lambda **kwargs: Context(),
    )
    monkeypatch.setattr(
        diagnostic.v10diag.token_v1,
        "_collect_endpoint_traces",
        lambda **kwargs: (traces, {}),
    )
    monkeypatch.setattr(
        diagnostic.v10diag,
        "fit_complete_h4_tail_held_family",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        diagnostic.v10diag.v8diag,
        "_execute_v8_analytic_phases",
        lambda **kwargs: analytic,
    )
    monkeypatch.setattr(
        diagnostic.v10diag.v9diag,
        "_authenticate_v8_live_lineage",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        diagnostic.v10diag,
        "_authenticate_live_lineage_against_v9",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        diagnostic, "_CompositeSuffixNativeVJPCollector", Collector
    )
    monkeypatch.setattr(
        diagnostic.v10diag,
        "_execute_live_precision_grid",
        lambda **kwargs: live,
    )
    monkeypatch.setattr(
        diagnostic.v11diag,
        "_authenticate_live_v10_replay",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        diagnostic.v10diag.v8diag,
        "_row_bank_candidate_support_row_executions",
        lambda value: 0,
    )
    monkeypatch.setattr(
        diagnostic.v10diag,
        "_resource_accounting",
        lambda **kwargs: {"v10": 1},
    )
    monkeypatch.setattr(
        diagnostic.v11diag,
        "_resource_accounting",
        lambda **kwargs: {"v11": 1},
    )
    monkeypatch.setattr(
        diagnostic.v12diag,
        "_live_v11_sections",
        lambda **kwargs: {"v11": True},
    )
    monkeypatch.setattr(
        diagnostic.v12diag,
        "_authenticate_live_v11_sections",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        diagnostic.v12diag,
        "_resource_accounting",
        lambda **kwargs: {"v12": 1},
    )
    monkeypatch.setattr(
        diagnostic,
        "_live_v12_sections",
        lambda **kwargs: {"v12": True},
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_live_v12_sections",
        lambda **kwargs: order.append("v12_live_auth")
        or {
            "v12_report_file_sha256": diagnostic.V12_REPORT_FILE_SHA256,
            "v12_report_sha256": diagnostic.V12_REPORT_SHA256,
            "all_live_sections_canonically_equal": True,
            "section_count": 19,
        },
    )
    native_authentication = {
        "all_receipts_payload_set_order_and_ownership_rehashed": True,
        "runtime_receipt_count": 64,
        "resource_receipt_count": 64,
        "canonical_output_cotangent_row_count": 3_212,
        "native_vjp_pullback_count": 436,
        "first_live_result_ordinal": 1,
        "last_live_result_ordinal": 64,
    }
    monkeypatch.setattr(
        diagnostic,
        "_validate_native_vjp_receipts",
        lambda **kwargs: native_authentication,
    )
    combined_resources = {
        "native_suffix_forward_count": 64,
        "native_vjp_pullback_count": 436,
        "native_token_directional_derivative_count": 3_212,
        "additional_full_model_forward_count": 0,
        "additional_full_model_backward_count": 0,
        "total_model_forward_count": 224,
        "total_backward_call_count": 1_039,
        "no_discarded_native_preflight_evaluation": True,
    }
    monkeypatch.setattr(
        diagnostic,
        "_resource_accounting",
        lambda **kwargs: combined_resources,
    )
    original_gate = diagnostic._require_integrity_gate_results

    def gates(values: Mapping[str, bool]) -> None:
        order.append("gates")
        original_gate(values)

    monkeypatch.setattr(diagnostic, "_require_integrity_gate_results", gates)
    monkeypatch.setattr(
        diagnostic,
        "_publish",
        lambda report, **kwargs: order.append("publish") or report,
    )

    report = diagnostic.run_gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_suffix_native_vjp_diagnostic(
        output=tmp_path / "v13.json"
    )
    first_load, second_load = (
        index for index, value in enumerate(order) if value == "v12_load"
    )
    assert first_load < order.index("native_finalize")
    assert order.index("v12_live_auth") < second_load
    assert second_load < order.index("gates") < order.index("publish")
    assert report["passed"] is True
    assert (
        report["classification"]
        == "v10_gradient_source_or_execution_path_difference_supported"
    )
