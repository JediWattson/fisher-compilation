from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_finite_diagnostic as diagnostic,
)


def _sha(character: str) -> str:
    return character * 64


def _mock_v7_evidence() -> tuple[dict[str, object], object]:
    fold_records: list[object] = []
    fold_metadata: list[dict[str, object]] = []
    inner_metadata: list[dict[str, object]] = []
    for outer in range(8):
        inner_records: list[object] = []
        for inner in range(7):
            row = {
                "outer_held_family_id": f"f{outer}",
                "inner_held_family_id": f"f{(outer + inner + 1) % 8}",
                "artifact_sha256": f"{100 + outer * 7 + inner:064x}",
            }
            inner_metadata.append(row)
            inner_records.append(
                SimpleNamespace(metadata=lambda row=row: dict(row))
            )
        fold_row = {
            "outer_held_family_id": f"f{outer}",
            "artifact_sha256": f"{300 + outer:064x}",
        }
        fold_metadata.append(fold_row)
        fold_records.append(
            SimpleNamespace(
                artifact_sha256=fold_row["artifact_sha256"],
                inner_family_records=tuple(inner_records),
                metadata=lambda row=fold_row: dict(row),
            )
        )
    screen = {
        "outcome": "joint_capacity_supported_for_finite_validation",
        "capacity_screen_passed": True,
        "artifact_sha256": _sha("a"),
    }
    v6_binding = {"artifact_sha256": _sha("b")}
    scalar_binding = {"artifact_sha256": _sha("c"), "comparator_count": 64}
    phases = SimpleNamespace(
        joint_screen=SimpleNamespace(
            outcome=screen["outcome"],
            capacity_screen_passed=True,
            artifact_sha256=screen["artifact_sha256"],
            metadata=lambda: dict(screen),
        ),
        joint_fold_records=tuple(fold_records),
        v6_binding=v6_binding,
        scalar_comparator_binding=scalar_binding,
    )
    report = {
        "joint_analytic_capacity_screen": dict(screen),
        "joint_analytic_fold_records": list(fold_metadata),
        "joint_analytic_inner_family_records": list(inner_metadata),
        "v6_control_binding": {"live_evidence_reproduction": v6_binding},
        "v6_scalar_comparator_binding": scalar_binding,
        "resources": {
            "total_model_forward_count": 112,
            "total_backward_call_count": 494,
            "finite_joint_candidate_model_forward_count": 0,
        },
    }
    return report, phases


def test_v7_loader_uses_exact_pins_and_only_accepts_passing_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    valid = {
        "schema": diagnostic.v7diag._SCHEMA,
        "classification": "joint_capacity_supported_for_finite_validation",
        "passed": True,
    }

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(valid)

    monkeypatch.setattr(diagnostic.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v7_report("v7.json") == valid
    assert captured["expected_file_sha256"] == diagnostic.V7_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V7_REPORT_SHA256
    monkeypatch.setattr(
        diagnostic.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: {**valid, "passed": False},
    )
    with pytest.raises(RuntimeError, match="v7 differs"):
        diagnostic._load_v7_report("v7.json")


def test_live_v7_authentication_requires_exact_8_fold_56_inner_replay() -> None:
    report, phases = _mock_v7_evidence()
    binding = diagnostic._authenticate_live_v7_evidence(
        v7_report=report, phases=phases
    )
    assert binding["v7_inner_record_count"] == 56
    assert binding["authenticated_before_any_finite_forward"] is True
    report["joint_analytic_inner_family_records"][0] = {  # type: ignore[index]
        **report["joint_analytic_inner_family_records"][0],  # type: ignore[index]
        "outer_held_family_id": "tampered",
    }
    with pytest.raises(RuntimeError, match="inner records"):
        diagnostic._authenticate_live_v7_evidence(
            v7_report=report, phases=phases
        )


def test_analytic_phase_order_authenticates_v7_before_finite_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    phases = object()
    monkeypatch.setattr(
        diagnostic.v7diag,
        "_execute_joint_phases",
        lambda **kwargs: order.append("v7_reproduce") or phases,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_live_v7_evidence",
        lambda **kwargs: order.append("v7_auth") or {"authenticated": True},
    )
    monkeypatch.setattr(
        diagnostic,
        "_reconstruct_full_v7_codecs",
        lambda **kwargs: order.append("codec") or ({}, {"codec": True}),
    )
    monkeypatch.setattr(
        diagnostic.v5diag,
        "_authenticate_v4_unit_final_baseline",
        lambda **kwargs: order.append("unit") or ({}, {}, {}, {"unit": True}),
    )
    monkeypatch.setattr(
        diagnostic,
        "_pinned_v5_selected_plus_rows",
        lambda report: order.append("plus") or ({}, {"plus": True}),
    )
    result = diagnostic._execute_v8_analytic_phases(
        context=object(),
        parent={},
        v3_report={},
        v4_report={},
        v5_report={},
        v6_report={},
        v7_report={},
        traces=(),
        endpoint_resources={},
        basis=torch.empty((0, 0)),
        fits={},
    )
    assert order == ["v7_reproduce", "v7_auth", "codec", "unit", "plus"]
    assert result.v7_phases is phases


def _provider(
    *,
    arm: str = "v5_static_plus",
    gains: torch.Tensor | None = None,
    base_h4: torch.Tensor | None = None,
    correction: torch.Tensor | None = None,
) -> tuple[diagnostic._AuthenticatedV8GainProvider, object, torch.Tensor]:
    base = (
        base_h4
        if base_h4 is not None
        else torch.zeros((1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    )
    support = torch.tensor([[True, False]])
    delta = (
        correction
        if correction is not None
        else torch.zeros(base.shape, dtype=torch.float64)
    )
    values = gains
    if values is None:
        values = (
            torch.ones((1, 64), dtype=torch.float64)
            if arm == "v7_joint"
            else torch.ones(64, dtype=torch.float64)
        )
    provider = diagnostic._AuthenticatedV8GainProvider(
        arm=arm,
        fold_artifact_sha256=_sha("a"),
        gain_support_artifact_sha256=_sha("b"),
        ordered_directions_sha256=_sha("c"),
        gains=values,
        example_id="example",
        family_id="family",
        model_inputs_sha256=_sha("d"),
        bridge_binding_sha256=_sha("e"),
        prefix_artifact_sha256=_sha("f"),
        base_h4=base,
        support_mask=support,
        correction=delta,
    )
    prefix = SimpleNamespace(
        artifact_sha256=_sha("f"),
        bridge_binding_sha256=_sha("e"),
        validate_integrity=lambda: None,
        complete_h4_causal_support_mask=lambda: support,
    )
    return provider, prefix, base


def test_provider_rejects_cross_arm_gains_wrong_width_and_cross_execution_reuse() -> None:
    with pytest.raises(ValueError, match="gain geometry"):
        _provider(gains=torch.ones((1, 64), dtype=torch.float64))
    with pytest.raises(ValueError, match="gain geometry"):
        _provider(arm="v7_joint", gains=torch.ones(64, dtype=torch.float64))
    with pytest.raises(ValueError, match="tensor geometry"):
        _provider(base_h4=torch.zeros((1, 2, 639), dtype=torch.float32))

    provider, prefix, base = _provider()
    wrong_prefix = SimpleNamespace(
        artifact_sha256=_sha("0"),
        bridge_binding_sha256=_sha("e"),
        validate_integrity=lambda: None,
        complete_h4_causal_support_mask=prefix.complete_h4_causal_support_mask,
    )
    with pytest.raises(RuntimeError, match="another execution"):
        provider.correction(wrong_prefix, base)
    assert torch.equal(provider.correction(prefix, base), provider._correction)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)


def test_executor_rejects_v5_v6_gain_swap_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = torch.zeros((1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    support_mask = torch.tensor([[True, False]])
    support_indices = torch.tensor([0], dtype=torch.long)
    base_rows = base.to(dtype=torch.float64)[0].index_select(0, support_indices)
    directions = torch.zeros(
        (diagnostic.CANDIDATE_GAIN_RANK, diagnostic.token_v1._WIDTH),
        dtype=torch.float64,
    )
    direction_sha = diagnostic._runtime_tensor_sha256(directions)
    fold_sha = _sha("1")
    support = diagnostic.CandidateConditionedK64ThreeArmGainSupport(
        phase="held",
        held_family_id="family",
        example_id="example",
        family_id="family",
        training_family_ids=tuple(f"f{index}" for index in range(7)),
        training_example_ids=tuple(f"e{index}" for index in range(7)),
        parent_fold_artifact_sha256=fold_sha,
        refit_artifact_sha256=_sha("2"),
        codec_artifact_sha256=_sha("3"),
        scalar_fit_artifact_sha256=_sha("4"),
        joint_fit_artifact_sha256=_sha("5"),
        ordered_directions_codec_sha256=(
            diagnostic.state_field._tensor_sha256(directions)
        ),
        ordered_directions_refit_sha256=direction_sha,
        base_h4_support_rows_sha256=diagnostic._runtime_tensor_sha256(base_rows),
        standardized_state_features=torch.zeros((1, 4), dtype=torch.float64),
        static_plus_gains=torch.ones(64, dtype=torch.float64),
        exact_scalar_gains=torch.full((64,), 0.5, dtype=torch.float64),
        joint_row_gains=torch.ones((1, 64), dtype=torch.float64),
    )
    trace = SimpleNamespace(
        example_id="example",
        family_id="family",
        base_h4=base,
        support_indices=support_indices,
        prefix=SimpleNamespace(
            complete_h4_causal_support_mask=lambda: support_mask,
        ),
    )
    fit = SimpleNamespace(artifact_sha256=fold_sha)
    monkeypatch.setattr(
        diagnostic.v6diag, "_ordered_k64", lambda _fit: directions
    )
    original = diagnostic.CandidateConditionedK64ThreeArmGainSupport.gains_tensor

    def swapped_gains(
        self: diagnostic.CandidateConditionedK64ThreeArmGainSupport, arm: str
    ) -> torch.Tensor:
        if arm == "v5_static_plus":
            return self.exact_scalar_gains.clone().contiguous()
        return original(self, arm)

    monkeypatch.setattr(
        diagnostic.CandidateConditionedK64ThreeArmGainSupport,
        "gains_tensor",
        swapped_gains,
    )
    with pytest.raises(RuntimeError, match="arm gain support binding"):
        diagnostic._execute_held_finite_arm(
            context=SimpleNamespace(),
            trace=trace,
            basis=torch.empty((0, diagnostic.token_v1._WIDTH)),
            fit=fit,
            support=support,
            arm="v5_static_plus",
            model_inputs={},
            teacher_logits=torch.empty((0, 0)),
            endpoint_indices=torch.empty(0, dtype=torch.long),
        )


def test_provider_rejects_correction_outside_authenticated_support() -> None:
    correction = torch.zeros((1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float64)
    correction[0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="escapes held support"):
        _provider(correction=correction)


def test_float32_cast_once_delta_is_not_falsely_equated_to_analytic_rows() -> None:
    base = torch.ones((1, 2, diagnostic.token_v1._WIDTH), dtype=torch.float32)
    correction = torch.zeros_like(base, dtype=torch.float64)
    correction[0, 0, 0] = 1.0e-8
    executed = diagnostic._executed_cast_once_delta_rows(
        base_h4=base,
        correction=correction,
        support_mask=torch.tensor([[True, False]]),
        support_indices=torch.tensor([0], dtype=torch.long),
    )
    assert executed.dtype == torch.float64
    assert executed[0, 0].item() == 0.0
    assert executed[0, 0].item() != correction[0, 0, 0].item()


def test_held_schedule_is_exactly_16_by_3_with_two_prompts_per_family() -> None:
    traces = tuple(
        SimpleNamespace(example_id=f"e{family}-{prompt}", family_id=f"f{family}")
        for family in range(8)
        for prompt in range(2)
    )
    schedule = diagnostic._held_arm_schedule(traces)
    assert len(schedule) == 48
    assert tuple(arm for _trace, arm in schedule[:3]) == (
        diagnostic.THREE_ARM_FINITE_NAMES
    )
    assert {
        arm: sum(scheduled_arm == arm for _trace, scheduled_arm in schedule)
        for arm in diagnostic.THREE_ARM_FINITE_NAMES
    } == {arm: 16 for arm in diagnostic.THREE_ARM_FINITE_NAMES}
    with pytest.raises(RuntimeError, match="two prompts"):
        diagnostic._held_arm_schedule(traces[:-1])
    duplicate = traces[:-1] + (
        SimpleNamespace(example_id=traces[0].example_id, family_id="f7"),
    )
    with pytest.raises(RuntimeError, match="two prompts"):
        diagnostic._held_arm_schedule(duplicate)


def _nll_observations() -> tuple[dict[str, object], ...]:
    candidates = {
        "v5_static_plus": (1.10, 2.20),
        "v6_exact_scalar": (1.05, 2.10),
        "v7_joint": (1.00, 2.00),
    }
    return tuple(
        {
            "example_id": f"e{family}-{prompt}",
            "family_id": f"f{family}",
            "arm": arm,
            "native_mean_nll": 1.0,
            "d320_mean_nll": 1.2,
            "candidate_mean_nll": candidates[arm][0],
            "native_ordinary_mean_nll": 2.0,
            "d320_ordinary_mean_nll": 2.4,
            "ordinary_candidate_mean_nll": candidates[arm][1],
        }
        for family in range(8)
        for prompt in range(2)
        for arm in diagnostic.THREE_ARM_FINITE_NAMES
    )


def test_nll_summary_is_family_then_prompt_equal_and_explicitly_non_gating() -> None:
    summary = diagnostic._descriptive_nll_summary_by_arm(_nll_observations())
    joint = summary["v7_joint"]
    assert joint["prompt_count"] == 16
    assert joint["family_count"] == 8
    assert joint["family_macro_candidate_endpoint_mean_nll"] == 1.0
    assert joint["family_macro_candidate_ordinary_mean_nll"] == 2.0
    assert joint["endpoint_gap"]["family_strict_gap_improvement_count"] == 8
    assert joint["ordinary_gap"]["family_strict_gap_improvement_count"] == 8
    assert joint["descriptive_only_non_gating"] is True
    assert joint["threshold_applied"] is False


def test_resource_ledger_includes_all_analysis_only_projection_work() -> None:
    resources = diagnostic._resource_accounting(
        endpoint_resources={
            "base_forward_count": 16,
            "native_forward_count": 16,
            "endpoint_token_vjp_forward_count": 16,
            "endpoint_token_vjp_backward_call_count": 109,
            "complete_h4_support_row_count": 819,
        },
        gradient_resources={
            "gradient_native_forward_count": 8,
            "gradient_candidate_vjp_forward_count": 56,
            "gradient_candidate_vjp_backward_call_count": 385,
        },
        finite_resources={
            "held_native_forward_count": 16,
            "held_candidate_forward_count": 48,
            "finite_backward_call_count": 0,
        },
        row_bank_candidate_support_row_executions=2842,
    )
    assert resources["total_model_forward_count"] == 176
    assert resources["total_backward_call_count"] == 494
    assert resources["total_candidate_support_row_executions"] == 5299
    assert resources["analytic_and_finite_candidate_execution_count"] == 104
    assert resources["analysis_only_d320_supported_projection_logical_macs"] == 2170470400
    assert resources["analysis_only_k64_tail_projection_logical_macs"] == 434094080
    assert resources["kernel_or_serving_speed_claim"] is False
    with pytest.raises(RuntimeError, match="resource accounting"):
        diagnostic._resource_accounting(
            endpoint_resources={
                "base_forward_count": 16,
                "native_forward_count": 16,
                "endpoint_token_vjp_forward_count": 16,
                "endpoint_token_vjp_backward_call_count": 109,
                "complete_h4_support_row_count": 819,
            },
            gradient_resources={
                "gradient_native_forward_count": 8,
                "gradient_candidate_vjp_forward_count": 56,
                "gradient_candidate_vjp_backward_call_count": 385,
            },
            finite_resources={
                "held_native_forward_count": 16,
                "held_candidate_forward_count": 48,
                "finite_backward_call_count": 0,
            },
            row_bank_candidate_support_row_executions=2841,
        )


def test_row_bank_requires_exact_eight_by_seven_and_derives_2842_rows() -> None:
    counts = (50,) * 55 + (92,)
    cells = {
        f"f{fold}": tuple(
            SimpleNamespace(
                base_h4_support_rows=torch.empty(
                    (counts[fold * 7 + inner], diagnostic.token_v1._WIDTH),
                    device="meta",
                )
            )
            for inner in range(7)
        )
        for fold in range(8)
    }
    phases = SimpleNamespace(
        v6_phases=SimpleNamespace(row_bank=SimpleNamespace(cells=cells))
    )
    assert diagnostic._row_bank_candidate_support_row_executions(phases) == 2842
    cells["f0"] = cells["f0"][:-1]
    with pytest.raises(RuntimeError, match="fold count"):
        diagnostic._row_bank_candidate_support_row_executions(phases)


def _classification_inputs(
    *,
    scalar_joint: float = 0.90,
    scalar_control: float = 1.0,
    plus_joint: float = 0.90,
    plus_control: float = 1.0,
    scalar_macro: bool = True,
    scalar_breadth: bool = True,
    scalar_cap: bool = True,
    plus_macro: bool = True,
    plus_breadth: bool = True,
    plus_cap: bool = True,
) -> tuple[dict[str, object], dict[str, bool]]:
    rows = {
        "joint_vs_scalar": {
            "joint_family_equal_mean_teacher_kl": scalar_joint,
            "control_family_equal_mean_teacher_kl": scalar_control,
        },
        "joint_vs_plus": {
            "joint_family_equal_mean_teacher_kl": plus_joint,
            "control_family_equal_mean_teacher_kl": plus_control,
        },
    }
    gates = {
        "joint_vs_scalar_family_macro_teacher_kl_improves_at_least_2pct": scalar_macro,
        "joint_vs_scalar_held_family_improvement_count_at_least_6_of_8": scalar_breadth,
        "joint_vs_scalar_worst_family_regression_at_most_5pct_plus_1e_minus_8": scalar_cap,
        "joint_vs_plus_family_macro_teacher_kl_improves_at_least_2pct": plus_macro,
        "joint_vs_plus_held_family_improvement_count_at_least_6_of_8": plus_breadth,
        "joint_vs_plus_worst_family_regression_at_most_5pct_plus_1e_minus_8": plus_cap,
    }
    return rows, gates


@pytest.mark.parametrize(
    ("overrides", "top1", "expected"),
    (
        (
            {"scalar_joint": 1.0},
            True,
            "analytic_to_finite_attribution_failure_same_a",
        ),
        (
            {"plus_joint": 1.0},
            True,
            "no_improvement_over_static_plus_carrier_same_a",
        ),
        (
            {"plus_cap": False},
            True,
            "unstable_family_regression_same_a",
        ),
        (
            {"scalar_macro": False},
            True,
            "below_predeclared_useful_effect_or_insufficient_breadth_same_a",
        ),
        (
            {},
            False,
            "joint_finite_failed_approximate_top1_safety_same_a",
        ),
        (
            {},
            True,
            "joint_finite_cleared_both_controls_and_top1_same_a",
        ),
    ),
)
def test_classification_preserves_the_predeclared_failure_matrix(
    overrides: dict[str, object], top1: bool, expected: str
) -> None:
    rows, gates = _classification_inputs(**overrides)
    assert (
        diagnostic._classification(
            integrity_passed=True,
            pairwise_rows=rows,
            pairwise_gates=gates,
            top1_gates={"top1": top1},
        )
        == expected
    )


def test_classification_integrity_failure_precedes_scientific_outcomes() -> None:
    rows, gates = _classification_inputs()
    assert (
        diagnostic._classification(
            integrity_passed=False,
            pairwise_rows=rows,
            pairwise_gates=gates,
            top1_gates={"top1": True},
        )
        == "integrity_failure"
    )


def test_top1_gate_booleans_are_recomputed_from_reported_scalars() -> None:
    row = {
        "unit_aggregate_top1": 0.95,
        "candidate_aggregate_top1": 0.94,
        "unit_family_macro_top1": 0.95,
        "candidate_family_macro_top1": 0.94,
        "aggregate_at_least_point_90": True,
        "family_macro_at_least_point_90": True,
        "aggregate_no_material_regression_vs_unit": True,
        "family_macro_no_material_regression_vs_unit": True,
    }
    comparison = {
        "v7_joint": {
            ledger: dict(row)
            for ledger in ("ordinary", "complete_h4_support", "graph_core")
        }
    }
    assert all(diagnostic._joint_top1_gate_results(comparison).values())
    comparison["v7_joint"]["ordinary"][
        "aggregate_at_least_point_90"
    ] = False
    with pytest.raises(RuntimeError, match="top1 boolean drifted"):
        diagnostic._joint_top1_gate_results(comparison)


def test_cli_has_no_knobs_pyproject_exposes_v8_and_output_is_nonrepo() -> None:
    parser = diagnostic.build_parser()
    assert parser.parse_args([]).__dict__ == {}
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "elsewhere.json"])
    pyproject = Path(diagnostic.__file__).resolve().parents[2] / "pyproject.toml"
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-"
        "candidate-joint-state-gain-held-finite-v8-a-dev"
    ) in pyproject.read_text()
    assert str(diagnostic.DEFAULT_OUTPUT).endswith(
        "candidate-joint-state-gain-held-finite-lofo-a-fit16-dev-v8.json"
    )
    assert diagnostic._safety_metadata()["artifact_must_remain_outside_git"] is True
