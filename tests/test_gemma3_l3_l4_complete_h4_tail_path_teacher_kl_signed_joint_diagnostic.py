from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic as diagnostic,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as endpoint,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as v1,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.complete_h4_tail_signed_joint_projector import (
    complete_h4_tail_signed_joint_scores,
)
from fisher_graph.complete_h4_tail_path_teacher_kl import (
    CompleteH4TailPathTeacherKLAccumulator,
    GL4_UNIT_INTERVAL_NODES,
    GL4_UNIT_INTERVAL_WEIGHTS,
    complete_h4_tail_path_as_endpoint_example,
    summarize_complete_h4_tail_path_ftc_closure,
)


def _prefix() -> Gemma3L3L4OnePassPrefix:
    return Gemma3L3L4OnePassPrefix(
        source_modes=torch.zeros((1, 3, 2), dtype=torch.float64),
        clamped_y3=torch.zeros((1, 3, 640), dtype=torch.float32),
        predicted_target_modal_delta=torch.zeros((1, 3, 2), dtype=torch.float64),
        decoded_base_x4_delta=torch.zeros((1, 3, 640), dtype=torch.float64),
        logical_positions=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        valid_target_mask=torch.tensor([[True, True, True]]),
        source_eligible_mask=torch.tensor([[False, True, False]]),
        target_affected_mask=torch.tensor([[False, True, False]]),
        bridge_binding_sha256="c" * 64,
    )


def _path_boundaries() -> tuple[
    Gemma3L3L4OnePassPrefix,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    rows = int(support.sum())
    supported = torch.full((rows, 640), 0.25, dtype=torch.float64)
    tail = torch.full((rows, 640), 0.5, dtype=torch.float64)
    indices = torch.nonzero(support[0], as_tuple=False).flatten()
    d320 = base.clone()
    native = base.clone()
    d320[0].index_copy_(0, indices, supported.to(torch.float32))
    native[0].index_copy_(0, indices, (supported + tail).to(torch.float32))
    return prefix, base, d320, native, support, supported, tail


def test_path_node_provider_binds_quadrature_and_both_endpoints() -> None:
    prefix, base, d320, native, support, supported, tail = _path_boundaries()
    nodes, weights = diagnostic.gauss_legendre_unit_interval(4)
    provider = diagnostic._AuthenticatedPathNodeH4Provider(
        model_inputs_sha256="a" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        node_index=0,
        alpha_hex=nodes[0].hex(),
        quadrature_weight_hex=weights[0].hex(),
        base_h4=base,
        d320_h4=d320,
        native_h4=native,
        support_mask=support,
        supported_rows=supported,
        tail_rows=tail,
    )
    metadata = provider.metadata()
    assert metadata["alpha_hex"] == nodes[0].hex()
    assert metadata["quadrature_weight_hex"] == weights[0].hex()
    correction = provider.correction(prefix, base)
    expected = torch.zeros_like(correction)
    expected[support] = supported + nodes[0] * tail
    assert torch.equal(correction, expected)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)


def test_path_node_provider_rejects_noncanonical_hex_and_false_endpoints() -> None:
    prefix, base, d320, native, support, supported, tail = _path_boundaries()
    nodes, weights = diagnostic.gauss_legendre_unit_interval(4)
    with pytest.raises(ValueError, match="canonical"):
        diagnostic._AuthenticatedPathNodeH4Provider(
            model_inputs_sha256="a" * 64,
            bridge_binding_sha256="c" * 64,
            prefix_artifact_sha256=prefix.artifact_sha256,
            node_index=0,
            alpha_hex="0.5",
            quadrature_weight_hex=weights[0].hex(),
            base_h4=base,
            d320_h4=d320,
            native_h4=native,
            support_mask=support,
            supported_rows=supported,
            tail_rows=tail,
        )
    with pytest.raises(ValueError, match="exact indexed GL4 pair"):
        diagnostic._AuthenticatedPathNodeH4Provider(
            model_inputs_sha256="a" * 64,
            bridge_binding_sha256="c" * 64,
            prefix_artifact_sha256=prefix.artifact_sha256,
            node_index=1,
            alpha_hex=nodes[0].hex(),
            quadrature_weight_hex=weights[0].hex(),
            base_h4=base,
            d320_h4=d320,
            native_h4=native,
            support_mask=support,
            supported_rows=supported,
            tail_rows=tail,
        )
    bad_native = native.clone()
    bad_native[support] += 1.0
    with pytest.raises(ValueError, match="does not reconstruct native"):
        diagnostic._AuthenticatedPathNodeH4Provider(
            model_inputs_sha256="a" * 64,
            bridge_binding_sha256="c" * 64,
            prefix_artifact_sha256=prefix.artifact_sha256,
            node_index=0,
            alpha_hex=nodes[0].hex(),
            quadrature_weight_hex=weights[0].hex(),
            base_h4=base,
            d320_h4=d320,
            native_h4=bad_native,
            support_mask=support,
            supported_rows=supported,
            tail_rows=tail,
        )


def _path_evidence_for_duck_endpoint():
    tail = torch.zeros((2, 640), dtype=torch.float64)
    tail[:, 320] = torch.tensor([2.0, -1.0])
    source = torch.tensor([3.0, 1.0], dtype=torch.float64)
    gradient = torch.zeros((2, 2, 640), dtype=torch.float64)
    gradient[:, :, 320] = torch.tensor([[-1.0, 0.5], [0.25, -2.0]])
    accumulator = CompleteH4TailPathTeacherKLAccumulator(
        example_id="example",
        family_id="family",
        residual_rows=tail,
        source_token_teacher_kl=source,
        native_token_teacher_kl=torch.zeros_like(source),
    )
    for index, (node, weight) in enumerate(
        zip(GL4_UNIT_INTERVAL_NODES, GL4_UNIT_INTERVAL_WEIGHTS)
    ):
        accumulator.add_node(
            node_index=index,
            path_fraction=node,
            quadrature_weight=weight,
            token_h4_gradients=gradient,
            token_teacher_kl=(1.0 - node) * source,
            vjp_artifact_sha256=f"{index * 3 + 1:064x}",
            provider_artifact_sha256=f"{index * 3 + 2:064x}",
            execution_artifact_sha256=f"{index * 3 + 3:064x}",
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    return accumulator.finalize()


def test_typed_path_uses_E_but_finite_duck_uses_full_R_with_equal_null_scores() -> None:
    evidence = _path_evidence_for_duck_endpoint()
    basis = torch.eye(640, dtype=torch.float64)[:320]
    full_residual = evidence.residual_rows.clone()
    full_residual[:, 0] = torch.tensor([7.0, -4.0])
    finite = diagnostic._finite_endpoint_from_path_evidence(
        evidence,
        full_native_minus_base_residual_rows=full_residual,
        supported_basis=basis,
    )
    path = complete_h4_tail_path_as_endpoint_example(evidence)
    assert torch.equal(path.residual_rows, evidence.residual_rows)
    assert torch.equal(finite.residual_rows, full_residual)
    direction = torch.zeros((1, 640), dtype=torch.float64)
    direction[0, 320] = 1.0
    assert torch.equal(
        complete_h4_tail_signed_joint_scores(path, direction),
        complete_h4_tail_signed_joint_scores(finite, direction),
    )


def test_path_candidate_constructor_contains_d320_and_closes_at_native() -> None:
    _prefix_value, base, d320, native, support, supported, tail = _path_boundaries()
    support_indices = torch.nonzero(support[0], as_tuple=False).flatten()
    at_d320 = diagnostic._expected_path_candidate_h4(
        base_h4=base,
        support_indices=support_indices,
        supported_rows=supported,
        tail_rows=tail,
        alpha=0.0,
    )
    at_native = diagnostic._expected_path_candidate_h4(
        base_h4=base,
        support_indices=support_indices,
        supported_rows=supported,
        tail_rows=tail,
        alpha=1.0,
    )
    assert torch.equal(at_d320, d320)
    assert torch.equal(at_native, native)


def _endpoint_report() -> dict[str, object]:
    prompts = []
    for index in range(16):
        prompts.append(
            {
                "artifact_sha256": "0" * 64,
                "example_id": f"example-{index}",
                "family_id": f"family-{index // 2}",
                "model_inputs_sha256": "1" * 64,
                "base_x4_sha256": "2" * 64,
                "supervised_indices_sha256": "3" * 64,
                "supervised_targets_sha256": "4" * 64,
                "endpoint_support_indices_sha256": "5" * 64,
                "endpoint_support_targets_sha256": "6" * 64,
                "compensation_target_sha256": "f" * 64,
                "native_logits_sha256": "7" * 64,
                "teacher_logits_sha256": "7" * 64,
                "teacher_kl_vjp_artifact_sha256": "8" * 64,
                "endpoint_execution_artifact_sha256": "9" * 64,
                "endpoint_provider_artifact_sha256": "a" * 64,
                "causality_receipt_sha256": "b" * 64,
                "endpoint_support_supervised_token_count": 1,
                "maximum_future_gradient_abs": 0.0,
                "future_gradient_nonzero_count": 0,
            }
        )
    return {
        "schema": endpoint._SCHEMA,
        "passed": False,
        "classification": "same_a_teacher_kl_signed_joint_bounded_fidelity_not_supported",
        "protocol": {
            "requested_tail_ranks": endpoint.SIGNED_JOINT_RANKS,
            "teacher_objective": "token_KL(native_teacher||D320_candidate)",
        },
        "input_binding": {
            "materialization_report_file_sha256": v1.MATERIALIZATION_REPORT_FILE_SHA256,
            "materialization_report_sha256": v1.MATERIALIZATION_REPORT_SHA256,
            "transfer_report_file_sha256": v1.TRANSFER_REPORT_FILE_SHA256,
            "transfer_report_sha256": v1.TRANSFER_REPORT_SHA256,
        },
        "finite_observation_receipts": [{}],
        "finite_observation_set_sha256": "c" * 64,
        "primary_gate_results": (
            ("all_teacher_kl_vjps_have_zero_future_gradient", True),
            ("shared_k320_every_prompt_h4_bitwise_native", True),
            ("shared_k320_every_prompt_logits_bitwise_native", True),
        ),
        "prompt_receipts": tuple(prompts),
        "scientific_status": {
            "same_a_truth_leaking_hypothesis_use_only": True,
            "candidate_serving_authorized": False,
            "compression_claim": False,
        },
        "safety": {
            "contains_activation_tensors": False,
            "contains_gradient_tensors": False,
            "artifact_must_remain_outside_git": True,
        },
    }


def test_endpoint_loader_requires_content_pins_and_full_observation_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _endpoint_report()
    calls: list[tuple[str, str]] = []

    def load(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(
            (
                str(kwargs["expected_file_sha256"]),
                str(kwargs["expected_report_sha256"]),
            )
        )
        return report

    monkeypatch.setattr(v1, "_load_pinned_report", load)
    monkeypatch.setattr(
        endpoint, "_finite_observation_set_sha256", lambda rows: "c" * 64
    )
    loaded, prompts = diagnostic._load_endpoint_report(
        "endpoint.json",
        expected_file_sha256="d" * 64,
        expected_report_sha256="e" * 64,
    )
    assert loaded is report
    assert len(prompts) == 16
    assert calls == [("d" * 64, "e" * 64)]
    tampered = deepcopy(report)
    tampered["primary_gate_results"] = (
        ("all_teacher_kl_vjps_have_zero_future_gradient", False),
        ("shared_k320_every_prompt_h4_bitwise_native", True),
        ("shared_k320_every_prompt_logits_bitwise_native", True),
    )
    monkeypatch.setattr(v1, "_load_pinned_report", lambda *args, **kwargs: tampered)
    with pytest.raises(ValueError, match="semantics differ"):
        diagnostic._load_endpoint_report(
            "endpoint.json",
            expected_file_sha256="d" * 64,
            expected_report_sha256="e" * 64,
        )


def test_real_endpoint_pins_are_not_placeholders() -> None:
    assert diagnostic.ENDPOINT_REPORT_FILE_SHA256 == (
        "3ff52693fd863171fa5f130ac7a7ab32d3ab3e848fd51bee08b227d3a7eb635a"
    )
    assert diagnostic.ENDPOINT_REPORT_SHA256 == (
        "325b41db460435c653839b7e5c760449d6a9f1adfd02029f049873289ea8cd35"
    )


def test_closure_gates_surface_execution_invariants_and_complement_agreement() -> None:
    evidence = _path_evidence_for_duck_endpoint()
    closure = summarize_complete_h4_tail_path_ftc_closure((evidence,))
    trace = SimpleNamespace(
        maximum_future_gradient_abs=0.0,
        future_gradient_nonzero_count=0,
        prompt_path_payload={
            "full_residual_reconstruction_max_abs_error_hex": (0.0).hex(),
            "d320_h4_bitwise_pinned": True,
            "P_D_R_plus_E_H4_bitwise_native": True,
        },
    )
    basis = torch.eye(640, dtype=torch.float64)[:320]
    gates, metrics = diagnostic._closure_gate_results(
        closure,
        evidence=(evidence,),
        supported_basis=basis,
        traces=(trace,),
    )
    assert gates["every_GL4_node_has_zero_future_gradient"] is True
    assert gates["native_self_teacher_KL_is_exact_zero"] is True
    assert gates["every_prompt_D320_H4_matches_pinned_endpoint"] is True
    assert gates["every_prompt_full_tail_H4_is_bitwise_native"] is True
    assert gates["direct_and_complete_complement_contraction_agree"] is True
    assert metrics[
        "maximum_direct_minus_complete_complement_contraction_abs_error"
    ] <= 1.0e-9


def test_real_frozen_endpoint_comparison_includes_shared_k320() -> None:
    if not diagnostic.DEFAULT_ENDPOINT_REPORT.exists():
        pytest.skip("local frozen endpoint report is intentionally not committed")
    report, _prompts = diagnostic._load_endpoint_report(
        diagnostic.DEFAULT_ENDPOINT_REPORT,
        expected_file_sha256=diagnostic.ENDPOINT_REPORT_FILE_SHA256,
        expected_report_sha256=diagnostic.ENDPOINT_REPORT_SHA256,
    )
    behavior_raw = report["established_behavioral_fidelity_by_method_rank"]
    geometry_raw = report["executed_cast_once_geometry_by_method_rank"]
    assert isinstance(behavior_raw, dict)
    assert isinstance(geometry_raw, dict)
    behavior = {
        method: {int(rank): value for rank, value in ranks.items()}
        for method, ranks in behavior_raw.items()
    }
    geometry = {
        method: {int(rank): value for rank, value in ranks.items()}
        for method, ranks in geometry_raw.items()
    }
    ladders = report["finite_ladder_by_method"]
    assert isinstance(ladders, dict)
    rows = diagnostic._path_vs_frozen_endpoint_comparison(
        endpoint_report=report,
        path_arms=ladders["signed_joint"],
        path_behavioral_by_method=behavior,
        path_geometry_by_method=geometry,
    )
    assert tuple(row["rank"] for row in rows) == diagnostic.PATH_RANKS
    sentinel = rows[-1]
    assert sentinel["rank"] == 320
    assert sentinel["path_minus_endpoint_family_macro_endpoint_rmse_after"] == 0.0
    assert all(
        value["path_minus_endpoint_source_to_candidate_kl_per_token"] == 0.0
        for value in sentinel["behavioral_deltas"].values()
    )


def test_cli_pins_endpoint_parent_and_has_no_scientific_overrides() -> None:
    parser = diagnostic.build_parser()
    args = parser.parse_args([])
    assert args.endpoint_report == diagnostic.DEFAULT_ENDPOINT_REPORT
    assert args.endpoint_report_file_sha256 == diagnostic.ENDPOINT_REPORT_FILE_SHA256
    assert args.endpoint_report_sha256 == diagnostic.ENDPOINT_REPORT_SHA256
    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert not any(action.dest in {"ranks", "quadrature_order"} for action in parser._actions)


def _mock_run_prefix(
    monkeypatch: pytest.MonkeyPatch,
    *,
    closure_passed: bool,
) -> tuple[list[SimpleNamespace], list[object]]:
    context = SimpleNamespace(
        validate_immutable_inputs=lambda: None,
        close=lambda: None,
    )
    traces = [
        SimpleNamespace(example_id=f"e{index}", family_id=f"f{index // 2}")
        for index in range(16)
    ]
    evidence = [object() for _ in range(16)]
    monkeypatch.setattr(
        diagnostic,
        "_load_endpoint_report",
        lambda *args, **kwargs: ({"schema": endpoint._SCHEMA}, {}),
    )
    monkeypatch.setattr(v1, "_load_pinned_report", lambda *args, **kwargs: {})
    monkeypatch.setattr(endpoint, "_transfer_receipts", lambda value: {})
    monkeypatch.setattr(
        diagnostic,
        "_load_committed_basis",
        lambda **kwargs: (
            torch.eye(640, dtype=torch.float64)[:320],
            {"runtime_tensor_sha256": "a" * 64},
            {},
        ),
    )
    monkeypatch.setattr(
        diagnostic, "prepare_complete_h4_rank320_live_context", lambda **kwargs: context
    )
    monkeypatch.setattr(
        diagnostic,
        "_collect_path_teacher_kl_traces",
        lambda **kwargs: (traces, tuple(evidence), {"collection": 1}),
    )
    closure = SimpleNamespace(metadata=lambda: {"closure": closure_passed})
    monkeypatch.setattr(
        diagnostic,
        "summarize_complete_h4_tail_path_ftc_closure",
        lambda values: closure,
    )
    monkeypatch.setattr(
        diagnostic,
        "_closure_gate_results",
        lambda *args, **kwargs: ({"closure_gate": closure_passed}, {}),
    )
    monkeypatch.setattr(
        diagnostic,
        "_common_report",
        lambda **kwargs: {"artifact": {"file": str(kwargs["destination"])}},
    )
    monkeypatch.setattr(
        diagnostic,
        "_publish",
        lambda report, **kwargs: report,
    )
    return traces, evidence


def test_run_publishes_closure_only_and_never_fits_when_precondition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_run_prefix(monkeypatch, closure_passed=False)
    monkeypatch.setattr(
        diagnostic,
        "_path_resource_accounting",
        lambda *args, **kwargs: {"total_model_forward_count": 112},
    )
    monkeypatch.setattr(
        diagnostic,
        "fit_complete_h4_tail_signed_joint_held_family",
        lambda *args, **kwargs: pytest.fail("fit must not run before closure passes"),
    )
    result = diagnostic.run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic(
        output=".local-runs/mocked-path-closure-fail.json"
    )
    assert result["classification"] == "same_a_GL4_FTC_closure_not_supported"
    assert result["fit_and_finite_evaluation_status"]["finite_ladder_executed"] is False
    assert "finite_ladder_by_method" not in result


def test_run_executes_full_ladder_only_after_closure_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces, evidence = _mock_run_prefix(monkeypatch, closure_passed=True)
    endpoint_values = [object() for _ in evidence]
    monkeypatch.setattr(
        diagnostic,
        "complete_h4_tail_path_as_endpoint_example",
        lambda value: endpoint_values[evidence.index(value)],
    )
    fit = SimpleNamespace(
        rank=64,
        metadata=lambda: {"rank": 64},
    )
    pca = SimpleNamespace(metadata=lambda: {"rank": 64})
    monkeypatch.setattr(
        diagnostic,
        "fit_complete_h4_tail_signed_joint_held_family",
        lambda *args, **kwargs: fit,
    )
    monkeypatch.setattr(
        endpoint, "_fit_fixed_pca_signed_control", lambda *args, **kwargs: pca
    )
    ledgers = ("ordinary", "complete_h4_support", "graph_core", "causal_tail")
    behavior = {
        method: {
            rank: {ledger: {"gates": {"passed": True}} for ledger in ledgers}
            for rank in ranks
        }
        for method, ranks in (
            ("signed_joint", (8, 16, 32, 64)),
            ("fixed_pca_diagonal", (8, 16, 32, 64)),
            ("shared_exact_sentinel", (320,)),
        )
    }
    geometry = {
        method: {rank: {"gates": {"passed": True}} for rank in ranks}
        for method, ranks in (
            ("signed_joint", (8, 16, 32, 64)),
            ("fixed_pca_diagonal", (8, 16, 32, 64)),
            ("shared_exact_sentinel", (320,)),
        )
    }
    monkeypatch.setattr(
        endpoint,
        "_finite_dual_arm_observations",
        lambda **kwargs: ([], {"finite": 1}, behavior, geometry),
    )
    monkeypatch.setattr(endpoint, "_finite_observation_set_sha256", lambda rows: "a" * 64)
    sentinel = {
        "tail_rank": 320,
        "maximum_full_tail_reconstruction_abs_error": 0.0,
        "every_prompt_h4_bitwise_native": True,
        "every_prompt_logits_bitwise_native": True,
    }
    monkeypatch.setattr(
        endpoint,
        "_summarize_dual_arm_observations",
        lambda rows: (
            {
                "signed_joint": [{"tail_rank": rank} for rank in (8, 16, 32, 64)]
                + [sentinel],
                "fixed_pca_diagonal": [
                    {"tail_rank": rank} for rank in (8, 16, 32, 64)
                ]
                + [sentinel],
            },
            {},
        ),
    )
    monkeypatch.setattr(endpoint, "_paired_method_comparison", lambda **kwargs: ())
    monkeypatch.setattr(
        diagnostic,
        "_path_resource_accounting",
        lambda *args, **kwargs: {"total_model_forward_count": 272},
    )
    monkeypatch.setattr(
        diagnostic, "_path_vs_frozen_endpoint_comparison", lambda **kwargs: ()
    )
    result = diagnostic.run_gemma3_l3_l4_complete_h4_tail_path_teacher_kl_signed_joint_diagnostic(
        output=".local-runs/mocked-path-closure-pass.json"
    )
    assert result["fit_and_finite_evaluation_status"]["finite_ladder_executed"] is True
    assert result["passed"] is True
    assert result["smallest_path_signed_joint_rank_clearing_established_gates"] == 8


def test_path_resource_accounting_closes_at_112_and_272_for_exact_frozen_grid() -> None:
    traces = tuple(
        SimpleNamespace(
            family_id=f"f{index // 2}",
            endpoint=SimpleNamespace(
                residual_rows=torch.zeros((1, 640), dtype=torch.float64),
                supervised_tokens=1,
            ),
        )
        for index in range(16)
    )
    collection = {
        "base_forward_count": 16,
        "native_teacher_forward_count": 16,
        "d320_boundary_forward_count": 16,
        "path_teacher_kl_vjp_forward_count": 64,
        "path_teacher_kl_vjp_backward_call_count": 436,
        "path_quadrature_node_count": 64,
    }
    closure_only = diagnostic._path_resource_accounting(
        traces, collection_resources=collection
    )
    assert closure_only["collection_model_forward_count"] == 112
    assert closure_only["total_model_forward_count"] == 112
    assert closure_only["finite_evaluation_executed"] is False

    finite = {
        "finite_native_forward_count": 16,
        "finite_candidate_forward_count": 144,
        "finite_signed_joint_forward_count": 64,
        "finite_fixed_pca_diagonal_forward_count": 64,
        "finite_shared_exact_sentinel_forward_count": 16,
    }
    fits = {
        f"f{index}": SimpleNamespace(rank=64, stop_reason="requested_rank_reached")
        for index in range(8)
    }
    full = diagnostic._path_resource_accounting(
        traces,
        collection_resources=collection,
        finite_resources=finite,
        signed_fits=fits,
    )
    expected_deflation = 8 * sum(
        3 * step * 320 * 320 + 3 * step * step * 320
        for step in range(1, 64)
    )
    assert full["total_model_forward_count"] == 272
    assert full["finite_evaluation_model_forward_count"] == 160
    assert full["signed_joint_low_rank_U_factor_deflation_logical_macs"] == expected_deflation
    assert full["signed_joint_dense_PMP_deflation_used"] is False
    assert full["matched_PCA_symmetric_eigh_320_by_320_call_count"] == 8
