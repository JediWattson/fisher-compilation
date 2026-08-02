from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_scalar_joint_path_attribution_diagnostic as diagnostic,
)


def _sha(character: str) -> str:
    return character * 64


def _mock_v8_report() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for example_index in range(16):
        example_id = f"example-{example_index:02d}"
        family_id = f"family-{example_index // 2}"
        for arm in diagnostic.v8diag.THREE_ARM_FINITE_NAMES:
            row: dict[str, object] = {
                "example_id": example_id,
                "family_id": family_id,
                "arm": arm,
            }
            row["observation_sha256"] = diagnostic.token_v1._domain_sha256(
                row, domain=diagnostic.v8diag._OBSERVATION_DOMAIN
            )
            rows.append(row)
    ordered = tuple(
        str(row["observation_sha256"])
        for row in sorted(
            rows,
            key=lambda row: (
                str(row["example_id"]),
                diagnostic.v8diag.THREE_ARM_FINITE_NAMES.index(str(row["arm"])),
            ),
        )
    )
    return {
        "schema": diagnostic.v8diag._SCHEMA,
        "classification": "analytic_to_finite_attribution_failure_same_a",
        "passed": False,
        "resources": {
            "total_model_forward_count": 176,
            "total_backward_call_count": 494,
        },
        "finite_observation_receipts": rows,
        "finite_observation_set_sha256": diagnostic.token_v1._domain_sha256(
            ordered, domain=diagnostic.v8diag._OBSERVATION_SET_DOMAIN
        ),
    }


def test_v8_loader_uses_exact_file_and_logical_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    valid = _mock_v8_report()

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return valid

    monkeypatch.setattr(diagnostic.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v8_report("v8.json") is valid
    assert captured["expected_file_sha256"] == diagnostic.V8_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V8_REPORT_SHA256

    invalid = {**valid, "classification": "unexpected"}
    monkeypatch.setattr(
        diagnostic.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: invalid,
    )
    with pytest.raises(RuntimeError, match="V8 differs"):
        diagnostic._load_v8_report("v8.json")


def test_v8_endpoint_index_requires_exact_authenticated_16_by_3_grid() -> None:
    report = _mock_v8_report()
    indexed = diagnostic._index_v8_endpoint_observations(report)
    assert len(indexed) == 48
    assert ("example-00", "v6_exact_scalar") in indexed
    assert ("example-15", "v7_joint") in indexed

    tampered = _mock_v8_report()
    tampered["finite_observation_receipts"][0]["family_id"] = "tampered"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="receipt drifted"):
        diagnostic._index_v8_endpoint_observations(tampered)

    missing = _mock_v8_report()
    missing["finite_observation_receipts"].pop()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="grid differs"):
        diagnostic._index_v8_endpoint_observations(missing)


def _path_provider(
    *,
    node_index: int = 0,
    path_fraction: float | None = None,
    support_indices: torch.Tensor | None = None,
    scalar_dtype: torch.dtype = torch.float32,
) -> tuple[
    diagnostic._AuthenticatedV9ScalarJointPathProvider,
    object,
    torch.Tensor,
]:
    base = torch.zeros((1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    support = torch.tensor([[True, False]])
    indices = (
        torch.tensor([0], dtype=torch.int64)
        if support_indices is None
        else support_indices
    )
    scalar = torch.ones((1, diagnostic.token_v1._WIDTH), dtype=scalar_dtype)
    joint = scalar + torch.tensor(0.25, dtype=scalar_dtype)
    alpha = (
        diagnostic.GL4_UNIT_INTERVAL_NODES[node_index]
        if path_fraction is None
        else path_fraction
    )
    provider = diagnostic._AuthenticatedV9ScalarJointPathProvider(
        example_id="example",
        family_id="family",
        endpoint_pair_binding_sha256=_sha("a"),
        model_inputs_sha256=_sha("b"),
        bridge_binding_sha256=_sha("c"),
        prefix_artifact_sha256=_sha("d"),
        node_index=node_index,
        path_fraction=alpha,
        quadrature_weight=diagnostic.GL4_UNIT_INTERVAL_WEIGHTS[node_index],
        base_h4=base,
        support_mask=support,
        support_indices=indices,
        scalar_endpoint_h4_rows=scalar,
        joint_endpoint_h4_rows=joint,
    )
    prefix = SimpleNamespace(
        artifact_sha256=_sha("d"),
        bridge_binding_sha256=_sha("c"),
        validate_integrity=lambda: None,
        complete_h4_causal_support_mask=lambda: support,
    )
    return provider, prefix, base


def test_path_provider_executes_exact_cast_once_gl4_node_and_is_single_use() -> None:
    provider, prefix, base = _path_provider(node_index=2)
    expected = diagnostic._cast_once_scalar_joint_path_rows(
        scalar_h4_rows=provider._scalar_h4_rows,
        joint_h4_rows=provider._joint_h4_rows,
        path_fraction=diagnostic.GL4_UNIT_INTERVAL_NODES[2],
    )
    assert torch.equal(provider.path_h4_rows, expected)
    correction = provider.correction(prefix, base)
    replay = base.clone()
    replay[provider._support] = (
        base[provider._support].to(torch.float64)
        + correction[provider._support]
    ).to(base.dtype)
    assert torch.equal(replay[0, :1], expected)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)


def test_path_provider_rejects_wrong_node_false_endpoint_support_and_prefix() -> None:
    with pytest.raises(ValueError, match="exact GL4 rule"):
        _path_provider(node_index=0, path_fraction=0.5)
    with pytest.raises(ValueError, match="endpoint geometry"):
        _path_provider(scalar_dtype=torch.float64)
    with pytest.raises(ValueError, match="endpoint geometry"):
        _path_provider(support_indices=torch.tensor([1], dtype=torch.int64))

    provider, prefix, base = _path_provider()
    wrong_prefix = SimpleNamespace(
        artifact_sha256=_sha("0"),
        bridge_binding_sha256=prefix.bridge_binding_sha256,
        validate_integrity=lambda: None,
        complete_h4_causal_support_mask=prefix.complete_h4_causal_support_mask,
    )
    with pytest.raises(RuntimeError, match="another execution"):
        provider.correction(wrong_prefix, base)
    assert provider.used is False


def test_full_h4_node_validation_rejects_off_support_mutation() -> None:
    provider, prefix, base = _path_provider()
    correction = provider.correction(prefix, base)
    candidate_h4 = base.clone()
    candidate_h4[provider._support] = (
        base[provider._support].to(torch.float64)
        + correction[provider._support]
    ).to(base.dtype)
    candidate_x4 = torch.zeros_like(base)
    trace = SimpleNamespace(
        base_h4=base,
        base_x4_sha256=diagnostic._runtime_tensor_sha256(candidate_x4),
        model_inputs_sha256=_sha("b"),
        prefix=prefix,
    )
    execution = SimpleNamespace(
        model_forward_count=1,
        model_inputs_sha256=_sha("b"),
        bridge_binding_sha256=_sha("c"),
        prefix=prefix,
        h4_head_sha256=provider.artifact_sha256,
        candidate_x4=candidate_x4,
        candidate_h4=candidate_h4,
    )
    diagnostic._validate_v9_path_node_full_h4(
        trace=trace, provider=provider, execution=execution
    )
    execution.candidate_h4 = candidate_h4.clone()
    execution.candidate_h4[0, 1, 0] = 1.0
    with pytest.raises(RuntimeError, match="execution binding differs"):
        diagnostic._validate_v9_path_node_full_h4(
            trace=trace, provider=provider, execution=execution
        )


def test_cast_once_path_starts_from_realized_endpoint_values() -> None:
    base = torch.ones((1, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    pre_cast_scalar = base.to(torch.float64) + 1.0e-8
    scalar_actual = pre_cast_scalar.to(torch.float32)
    assert torch.equal(scalar_actual, base)
    joint_actual = torch.nextafter(
        scalar_actual,
        torch.full_like(scalar_actual, 2.0),
    )
    actual_path = diagnostic._cast_once_scalar_joint_path_rows(
        scalar_h4_rows=scalar_actual,
        joint_h4_rows=joint_actual,
        path_fraction=diagnostic.GL4_UNIT_INTERVAL_NODES[3],
    )
    expected = (
        scalar_actual.to(torch.float64)
        + diagnostic.GL4_UNIT_INTERVAL_NODES[3]
        * (joint_actual.to(torch.float64) - scalar_actual.to(torch.float64))
    ).to(torch.float32)
    assert torch.equal(actual_path, expected)


def test_path_provider_binds_live_runtime_base_hash_not_cpu_algebra_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = torch.zeros((1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    support = torch.tensor([[True, False]])
    indices = torch.tensor([0], dtype=torch.int64)
    scalar = torch.zeros((1, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    joint = torch.ones_like(scalar)
    live_storage = base.data_ptr()

    def device_sensitive_hash(value: torch.Tensor) -> str:
        return _sha("a") if value.data_ptr() == live_storage else _sha("b")

    monkeypatch.setattr(diagnostic, "_runtime_tensor_sha256", device_sensitive_hash)
    provider = diagnostic._AuthenticatedV9ScalarJointPathProvider(
        example_id="example",
        family_id="family",
        endpoint_pair_binding_sha256=_sha("c"),
        model_inputs_sha256=_sha("d"),
        bridge_binding_sha256=_sha("e"),
        prefix_artifact_sha256=_sha("f"),
        node_index=0,
        path_fraction=diagnostic.GL4_UNIT_INTERVAL_NODES[0],
        quadrature_weight=diagnostic.GL4_UNIT_INTERVAL_WEIGHTS[0],
        base_h4=base,
        support_mask=support,
        support_indices=indices,
        scalar_endpoint_h4_rows=scalar,
        joint_endpoint_h4_rows=joint,
    )
    assert provider.base_h4_sha256 == _sha("a")
    prefix = SimpleNamespace(
        artifact_sha256=_sha("f"),
        bridge_binding_sha256=_sha("e"),
        validate_integrity=lambda: None,
        complete_h4_causal_support_mask=lambda: support,
    )
    provider.correction(prefix, base)


def test_held_unit_fullfit_convention_uses_fresh_unit_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    row_scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    monkeypatch.setattr(
        diagnostic.state_field,
        "encode_candidate_conditioned_k64_state_features",
        lambda *args, **kwargs: features,
    )
    monkeypatch.setattr(
        diagnostic.state_field,
        "contract_candidate_conditioned_k64_row_direction_scores",
        lambda *args, **kwargs: row_scores,
    )
    monkeypatch.setattr(
        diagnostic.v6diag,
        "_mean_delta",
        lambda refit: torch.ones(64, dtype=torch.float64),
    )
    trace = SimpleNamespace(
        example_id="example",
        family_id="family",
        base_h4=torch.zeros(
            (1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float32
        ),
        support_indices=torch.tensor([0, 1], dtype=torch.int64),
    )
    record = SimpleNamespace(
        full_joint_fit=SimpleNamespace(
            artifact_sha256=_sha("a"),
            applied_intercept=2.0,
            applied_parameter=torch.tensor(
                [2.0, 3.0, 5.0, 0.0, 0.0], dtype=torch.float64
            ),
        ),
        full_scalar_control_fit=SimpleNamespace(
            artifact_sha256=_sha("b"), applied_coefficient=0.5
        ),
    )
    codec = SimpleNamespace(artifact_sha256=_sha("c"))
    refit = SimpleNamespace(artifact_sha256=_sha("d"))
    gradients = torch.zeros(
        (2, 2, diagnostic.token_v1._WIDTH), dtype=torch.float64
    )
    token_kl = torch.ones(2, dtype=torch.float64)
    margin, receipt = diagnostic._held_unit_fullfit_convention_tokens(
        trace=trace,
        refit=refit,
        codec=codec,
        record=record,
        directions=torch.zeros((64, diagnostic.token_v1._WIDTH)),
        tail_rows=torch.zeros((2, diagnostic.token_v1._WIDTH)),
        unit_gradients=gradients,
        unit_token_kl=token_kl,
    )
    static = row_scores.sum(dim=1)
    state = row_scores @ features
    expected = (
        static * 2.0
        + state @ torch.tensor([3.0, 5.0, 0.0, 0.0], dtype=torch.float64)
        - static * 0.5
    )
    assert torch.equal(margin, expected)
    assert receipt["held_fullfit_joint_minus_scalar_mean_derivative"] == float(
        expected.mean()
    )


def test_exact_resource_ledger_uses_actual_backward_chunks() -> None:
    endpoint = {
        "base_forward_count": 16,
        "native_forward_count": 16,
        "endpoint_token_vjp_forward_count": 16,
        "endpoint_token_vjp_backward_call_count": 109,
        "complete_h4_support_row_count": 819,
    }
    gradient = {
        "gradient_native_forward_count": 8,
        "gradient_candidate_vjp_forward_count": 56,
        "gradient_candidate_vjp_backward_call_count": 385,
    }
    live = {
        "fresh_native_teacher_forward_count": 16,
        "held_unit_endpoint_vjp_forward_count": 16,
        "held_unit_endpoint_vjp_backward_call_count": 109,
        "scalar_endpoint_vjp_forward_count": 16,
        "scalar_endpoint_vjp_backward_call_count": 109,
        "joint_boundary_forward_count": 16,
        "path_teacher_kl_vjp_forward_count": 64,
        "path_teacher_kl_vjp_backward_call_count": 436,
        "path_quadrature_node_count": 64,
    }
    resources = diagnostic._resource_accounting(
        endpoint_resources=endpoint,
        gradient_resources=gradient,
        live_resources=live,
        row_bank_candidate_support_row_executions=2842,
    )
    assert resources["total_model_forward_count"] == 240
    assert resources["total_backward_call_count"] == 1148
    assert resources["fresh_attribution_model_forward_count"] == 128
    assert resources["fresh_attribution_backward_call_count"] == 654
    assert resources[
        "fresh_unit_scalar_joint_and_GL4_candidate_support_row_executions"
    ] == 5733
    assert resources["total_candidate_support_row_executions"] == 8575

    with pytest.raises(RuntimeError, match="resource accounting differs"):
        diagnostic._resource_accounting(
            endpoint_resources=endpoint,
            gradient_resources=gradient,
            live_resources={
                **live,
                "path_teacher_kl_vjp_backward_call_count": 64,
            },
            row_bank_candidate_support_row_executions=2842,
        )


def _mock_additive_inputs() -> tuple[object, tuple[object, ...], dict[str, object]]:
    families: list[object] = []
    records: list[object] = []
    prompts: dict[str, object] = {}
    family_values: list[dict[str, float]] = []
    for index in range(8):
        family_id = f"family-{index}"
        example_ids = (f"example-{2 * index:02d}", f"example-{2 * index + 1:02d}")
        u_inner = -0.08 + index * 0.01
        values = {
            "held_unit_fullfit_convention": u_inner + 0.01,
            "held_unit_actual_displacement_tangent": u_inner + 0.03,
            "scalar_endpoint_tangent": u_inner + 0.06,
            "GL4_path_integral": u_inner + 0.10,
            "finite_joint_minus_scalar_teacher_KL": u_inner + 0.15,
        }
        family_values.append(values)
        for example_id in example_ids:
            prompts[example_id] = dict(values)
        families.append(
            SimpleNamespace(
                family_id=family_id,
                example_ids=example_ids,
                mean_finite_kl_delta=values[
                    "finite_joint_minus_scalar_teacher_KL"
                ],
                mean_path_integral=values["GL4_path_integral"],
                mean_scalar_endpoint_tangent=values["scalar_endpoint_tangent"],
                mean_held_unit_endpoint_tangent=values[
                    "held_unit_actual_displacement_tangent"
                ],
            )
        )
        records.append(
            SimpleNamespace(
                outer_held_family_id=family_id,
                joint_inner_macro_derivative=u_inner,
                scalar_inner_macro_derivative=0.0,
            )
        )

    def mean(name: str) -> float:
        return sum(value[name] for value in family_values) / len(family_values)

    attribution = SimpleNamespace(
        family_summaries=tuple(families),
        mean_finite_kl_delta=mean("finite_joint_minus_scalar_teacher_KL"),
        mean_path_integral=mean("GL4_path_integral"),
        mean_scalar_endpoint_tangent=mean("scalar_endpoint_tangent"),
        mean_held_unit_endpoint_tangent=mean(
            "held_unit_actual_displacement_tangent"
        ),
    )
    return attribution, tuple(records), prompts


def test_additive_ledger_telescopes_and_cross_binds_core_aggregation() -> None:
    attribution, records, prompts = _mock_additive_inputs()
    ledger = diagnostic._additive_attribution_ledger(
        attribution=attribution,
        joint_fold_records=records,
        prompt_additive_scalars=prompts,
    )
    assert ledger["family_count"] == 8
    for row in ledger["family_rows"]:
        assert row["additive_identity_closes"] is True
        assert abs(row["additive_identity_residual"]) <= row[
            "additive_identity_tolerance"
        ]
    aggregate = ledger["family_equal_aggregate"]
    assert all(aggregate["live_stage_core_aggregation_checks"].values())
    assert aggregate["target_D_minus_U_inner_cv"] == pytest.approx(
        aggregate["additive_component_sum"]
    )
    associations = ledger["family_association_with_finite_D"]["stages"]
    for name in (
        "U_inner_cv",
        "U_held_fullfit_convention",
        "U_held_actual_displacement",
        "T_scalar_endpoint",
        "P_GL4_path_integral",
    ):
        assert name in associations


def test_preregistered_closure_gates_and_integrity_first_classification() -> None:
    passing = SimpleNamespace(
        closure_relative_rmse=0.05,
        closure_cosine=0.99,
        family_summaries=tuple(
            SimpleNamespace(closure_relative_rmse=0.10) for _ in range(8)
        ),
    )
    assert all(diagnostic._quadrature_closure_gate_results(passing).values())
    failing = SimpleNamespace(
        closure_relative_rmse=0.051,
        closure_cosine=0.989,
        family_summaries=(
            SimpleNamespace(closure_relative_rmse=0.101),
            *(SimpleNamespace(closure_relative_rmse=0.01) for _ in range(7)),
        ),
    )
    assert not all(diagnostic._quadrature_closure_gate_results(failing).values())
    assert (
        diagnostic._classification(
            integrity_passed=False, closure_established=True
        )
        == "integrity_failure"
    )
    assert (
        diagnostic._classification(
            integrity_passed=True, closure_established=False
        )
        == "scalar_joint_path_closure_unresolved_same_a"
    )
    assert (
        diagnostic._classification(
            integrity_passed=True, closure_established=True
        )
        == "scalar_joint_GL4_closure_established_same_a"
    )


def test_finalized_evidence_grid_counts_node_receipts_not_accumulator_api() -> None:
    evidence = diagnostic.CandidateJointStatePathAccumulator(
        example_id="example",
        family_id="family",
        scalar_endpoint_h4_rows=torch.zeros(1, 1),
        joint_endpoint_h4_rows=torch.ones(1, 1),
        scalar_token_teacher_kl=torch.zeros(1),
        joint_token_teacher_kl=torch.ones(1),
        endpoint_pair_binding_sha256=_sha("1"),
        scalar_endpoint_execution_artifact_sha256=_sha("2"),
        joint_endpoint_execution_artifact_sha256=_sha("3"),
        supervised_grid_sha256=_sha("4"),
        teacher_logits_sha256=_sha("5"),
    )
    for node_index, (alpha, weight) in enumerate(
        zip(
            diagnostic.GL4_UNIT_INTERVAL_NODES,
            diagnostic.GL4_UNIT_INTERVAL_WEIGHTS,
            strict=True,
        )
    ):
        evidence.add_node(
            node_index=node_index,
            path_fraction=alpha,
            quadrature_weight=weight,
            path_node_h4_rows=torch.tensor([[alpha]], dtype=torch.float32),
            token_h4_gradients=torch.ones(1, 1, 1),
            token_teacher_kl=torch.ones(1),
            vjp_artifact_sha256=_sha("6"),
            provider_artifact_sha256=_sha("7"),
            execution_artifact_sha256=_sha("8"),
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        )
    finalized = evidence.finalize()
    assert len(finalized.node_receipts) == 4
    assert not hasattr(finalized, "node_count")
    assert diagnostic._evidence_grid_has_exact_path_nodes((finalized,) * 16)
    assert not diagnostic._evidence_grid_has_exact_path_nodes((finalized,) * 15)


def test_integrity_failure_is_fail_closed_before_any_output(tmp_path: Path) -> None:
    output = tmp_path / diagnostic.DEFAULT_OUTPUT.name
    assert not output.exists()
    with pytest.raises(RuntimeError, match="before publication"):
        diagnostic._require_integrity_gate_results(
            {"v8_authenticated": True, "full_H4_bound": False}
        )
    assert not output.exists()
    diagnostic._require_integrity_gate_results(
        {"v8_authenticated": True, "full_H4_bound": True}
    )


def test_no_knob_cli_output_suffix_and_pyproject_entry() -> None:
    assert vars(diagnostic.build_parser().parse_args([])) == {}
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(["--output", "elsewhere.json"])
    assert diagnostic.DEFAULT_OUTPUT.name.endswith(
        "candidate-joint-state-gain-scalar-joint-path-gl4-attribution-"
        "lofo-a-fit16-dev-v9.json"
    )
    pyproject = Path("pyproject.toml").read_text()
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-candidate-"
        "joint-state-gain-scalar-joint-path-gl4-attribution-v9-a-dev = "
        '"fisher_graph.gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_'
        'joint_state_gain_scalar_joint_path_attribution_diagnostic:main"'
    ) in pyproject
