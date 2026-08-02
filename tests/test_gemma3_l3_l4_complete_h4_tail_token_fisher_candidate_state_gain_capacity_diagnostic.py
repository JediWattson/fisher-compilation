from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_state_gain_capacity_diagnostic as diagnostic,
)
from fisher_graph.complete_h4_tail_candidate_gain_refit_v4 import (
    contract_candidate_teacher_kl_gain_scores,
)
from fisher_graph.complete_h4_tail_candidate_state_gain_field import (
    contract_candidate_conditioned_k64_row_direction_scores,
    reduce_candidate_conditioned_k64_row_mode_scores,
)


def _row_geometry() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260801)
    directions = torch.eye(64, dtype=torch.float64)
    base = torch.randn((9, 64), generator=generator, dtype=torch.float64)
    tail = torch.randn((9, 64), generator=generator, dtype=torch.float64)
    gradients = torch.randn(
        (5, 9, 64), generator=generator, dtype=torch.float64
    )
    return base, tail, directions, gradients


def test_row_bank_reconstructs_exact_v4_static_artifact_and_shared_tied_score() -> None:
    base, tail, directions, gradients = _row_geometry()
    raw, row_mode, static = diagnostic._row_resolved_contractions(
        base_rows=base,
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradients,
    )
    exact_v4 = contract_candidate_teacher_kl_gain_scores(
        tail_rows=tail,
        ordered_directions=directions,
        token_h4_gradients=gradients,
    )
    assert torch.equal(raw, base[:, :4])
    assert row_mode.shape == (5, 9, 64)
    assert torch.equal(static, exact_v4)
    assert torch.allclose(row_mode.sum(dim=1), exact_v4, rtol=0.0, atol=1.0e-12)

    delta = torch.linspace(-0.4, 0.3, 64, dtype=torch.float64)
    shared = reduce_candidate_conditioned_k64_row_mode_scores(
        token_row_mode_scores=row_mode,
        mean_gain_delta=delta,
    )
    assert torch.equal(
        diagnostic._tie_row_direction_scores(
            row_mode_scores=row_mode, mean_gain_delta=delta
        ),
        shared,
    )
    assert torch.equal(
        shared,
        contract_candidate_conditioned_k64_row_direction_scores(
            tail_rows=tail,
            ordered_directions=directions,
            token_h4_gradients=gradients,
            mean_gain_delta=delta,
        ),
    )


def _v5_carrier_fixture() -> tuple[dict[str, object], dict[str, object]]:
    refits: dict[str, object] = {}
    selections: list[dict[str, object]] = []
    for index in range(8):
        family = f"family-{index}"
        mean = torch.linspace(
            0.8 + index / 1000.0,
            1.2 + index / 1000.0,
            64,
            dtype=torch.float64,
        )
        refit = SimpleNamespace(
            artifact_sha256=f"{index + 1:064x}",
            mean_proposed_gains_tensor=lambda mean=mean: mean.clone(),
        )
        plus = 1.0 + diagnostic._BASE_STEP * (mean - 1.0)
        refits[family] = refit
        selections.append(
            {
                "held_family_id": family,
                "artifact_sha256": f"{index + 101:064x}",
                "refit_artifact_sha256": refit.artifact_sha256,
                "mean_proposed_gains_sha256": (
                    diagnostic.microstep._tensor_sha256(mean)
                ),
                "plus_gains_sha256": diagnostic.microstep._tensor_sha256(plus),
                "microstep_epsilon": diagnostic._BASE_STEP,
                "microstep_epsilon_hex": diagnostic._BASE_STEP.hex(),
                "selected_arm": "plus_epsilon" if index < 6 else "unit",
            }
        )
    return {"candidate_microstep_tune_selections": selections}, refits


def test_v5_plus_carrier_binding_authenticates_six_selected_and_two_counterfactual() -> None:
    report, refits = _v5_carrier_fixture()
    binding = diagnostic._authenticate_v5_plus_carriers(
        v5_report=report, refits=refits  # type: ignore[arg-type]
    )
    assert binding["carrier_fold_count"] == 8
    assert binding["v5_plus_selected_fold_count"] == 6
    assert binding["v5_unit_selected_fold_count"] == 2
    assert binding["counterfactual_plus_carrier_used_in_v5_unit_selected_folds"] == 2
    assert all(
        row["unit_point_linearization_incremental_to_executed_v5_plus_arm"]
        for row in binding["carrier_receipts"]
    )
    report["candidate_microstep_tune_selections"][0]["plus_gains_sha256"] = (  # type: ignore[index]
        "f" * 64
    )
    with pytest.raises(RuntimeError, match="plus carrier"):
        diagnostic._authenticate_v5_plus_carriers(
            v5_report=report, refits=refits  # type: ignore[arg-type]
        )


def test_v5_loader_uses_exact_file_report_schema_and_outcome_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    valid = {
        "schema": diagnostic.v5diag._SCHEMA,
        "classification": (
            "symmetric_microstep_static_cross_family_transfer_blocker_same_a"
        ),
        "passed": False,
    }

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(valid)

    monkeypatch.setattr(diagnostic.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v5_report("v5.json") == valid
    assert captured["expected_file_sha256"] == diagnostic.V5_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V5_REPORT_SHA256
    monkeypatch.setattr(
        diagnostic.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: {**valid, "classification": "drifted"},
    )
    with pytest.raises(RuntimeError, match="control differs"):
        diagnostic._load_v5_report("v5.json")


def test_v6_resource_ledger_is_exactly_112_forwards_494_backwards_and_no_finite_state() -> None:
    result = diagnostic._resource_accounting(
        endpoint_resources={
            "base_forward_count": 16,
            "native_forward_count": 16,
            "endpoint_token_vjp_forward_count": 16,
            "endpoint_token_vjp_backward_call_count": 109,
        },
        gradient_resources={
            "gradient_native_forward_count": 8,
            "gradient_candidate_vjp_forward_count": 56,
            "gradient_candidate_vjp_backward_call_count": 385,
            "gradient_prompt_fold_count": 56,
            "row_resolved_candidate_vjp_bank_count": 56,
            "additional_model_work_for_row_resolution": 0,
        },
    )
    assert result["total_model_forward_count"] == 112
    assert result["total_backward_call_count"] == 494
    assert result["analytic_fit_model_forward_count"] == 0
    assert result["finite_state_candidate_model_forward_count"] == 0
    assert result["phase_order"] == (
        "parent_endpoint_recollection",
        "static_unit_k64_reconstruction",
        "row_resolved_unit_candidate_vjp_recollection",
        "exact_v4_refit_and_receipt_authentication",
        "exact_v5_plus_carrier_authentication",
        "nested_held_inner_family_analytic_capacity_screen",
        "scalar_hash_only_report_publication",
    )
    with pytest.raises(RuntimeError, match="resource accounting"):
        diagnostic._resource_accounting(
            endpoint_resources={
                "base_forward_count": 16,
                "native_forward_count": 16,
                "endpoint_token_vjp_forward_count": 17,
                "endpoint_token_vjp_backward_call_count": 109,
            },
            gradient_resources={
                "gradient_native_forward_count": 8,
                "gradient_candidate_vjp_forward_count": 56,
                "gradient_candidate_vjp_backward_call_count": 385,
            },
        )


def test_phase_order_authenticates_static_v4_and_v5_carrier_before_state_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    roles = SimpleNamespace()
    bank = diagnostic._RowBankResult(
        refits={"held": object()},
        cells={},
        gradient_receipts=tuple(),
        row_bank_receipts=tuple(),
        resources={},
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_parent_recollection",
        lambda **kwargs: order.append("parent") or "a" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_static_unit_k64_replay",
        lambda **kwargs: order.append("static") or "b" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_checkerboard_prompt_roles",
        lambda traces: order.append("roles") or roles,
    )
    monkeypatch.setattr(
        diagnostic,
        "_collect_row_resolved_v4_bank",
        lambda **kwargs: order.append("row_bank") or bank,
    )
    monkeypatch.setattr(
        diagnostic.v5diag,
        "_authenticate_live_v4_refits_and_gradients",
        lambda **kwargs: order.append("v4_auth") or {},
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_v3_v4_v5_lineage",
        lambda **kwargs: order.append("lineage") or {},
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_v5_plus_carriers",
        lambda **kwargs: order.append("carrier") or {},
    )
    monkeypatch.setattr(
        diagnostic,
        "_fit_nested_capacity_screen",
        lambda **kwargs: order.append("state_fit") or (object(), tuple()),
    )
    diagnostic._execute_capacity_phases(
        context=object(),
        parent={},
        v3_report={},
        v4_report={},
        v5_report={},
        traces=tuple(),
        endpoint_resources={},
        basis=torch.zeros((1, 1)),
        fits={},
    )
    assert order == [
        "parent",
        "static",
        "roles",
        "row_bank",
        "v4_auth",
        "lineage",
        "carrier",
        "state_fit",
    ]


def test_cli_has_no_experiment_or_path_knobs_and_pyproject_exposes_it() -> None:
    parser = diagnostic.build_parser()
    assert [action.dest for action in parser._actions] == ["help"]
    assert vars(parser.parse_args([])) == {}
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "elsewhere.json"])
    pyproject = Path(diagnostic.__file__).resolve().parents[2] / "pyproject.toml"
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-"
        "candidate-state-gain-capacity-v6-a-dev"
    ) in pyproject.read_text()


def test_hash_pins_output_name_report_safety_and_write_once(
    tmp_path: Path,
) -> None:
    assert diagnostic.V5_REPORT_FILE_SHA256 == (
        "488edc027f3265a624762af7f0ad6ec0ca9e7ff81bba469862a5ef0fdc72427b"
    )
    assert diagnostic.V5_REPORT_SHA256 == (
        "dc87205ee91c0e854155de7f27acf7aacd14a90a2e16b32f9288d843dc911459"
    )
    assert str(diagnostic.DEFAULT_OUTPUT).endswith(
        "candidate-state-gain-capacity-lofo-a-fit16-dev-v6.json"
    )
    safety = diagnostic._safety_metadata()
    assert safety["contains_only_hashes_counts_and_scalar_metrics"] is True
    assert safety["artifact_must_remain_outside_git"] is True
    assert not any(
        value
        for key, value in safety.items()
        if key.startswith("contains_")
        and key != "contains_only_hashes_counts_and_scalar_metrics"
    )

    output = tmp_path / "v6.json"
    report = {
        "schema": diagnostic._SCHEMA,
        "artifact": {"file": str(output), "committable": False},
        "safety": safety,
        "passed": False,
        "classification": "degenerate_design",
    }
    published = diagnostic._publish(report, output=output)
    assert output.is_file()
    assert len(published["report_sha256"]) == 64
    with pytest.raises(FileExistsError):
        diagnostic._publish(
            {
                "schema": diagnostic._SCHEMA,
                "artifact": {"file": str(output), "committable": False},
                "safety": safety,
            },
            output=output,
        )


@pytest.mark.parametrize(
    ("flags", "expected"),
    (
        ({"capacity_screen_passed": True}, "capacity_supported"),
        ({"feature_and_design_gate_passed": False}, "degenerate_design"),
        ({"residual_energy_gate_passed": False}, "degenerate_design"),
        ({"non_noop_gate_passed": False}, "structural_no_op"),
        ({"negative_inner_global_gate_passed": False}, "structural_no_op"),
        ({"cosine_stability_gate_passed": False}, "structural_no_op"),
        ({"state_beats_scalar_gate_passed": False}, "state_not_better_than_scalar"),
    ),
)
def test_capacity_outcome_partition(flags: dict[str, bool], expected: str) -> None:
    defaults = {
        "capacity_screen_passed": False,
        "feature_and_design_gate_passed": True,
        "residual_energy_gate_passed": True,
        "non_noop_gate_passed": True,
        "negative_inner_global_gate_passed": True,
        "negative_inner_local_gate_passed": True,
        "state_beats_scalar_gate_passed": True,
        "cosine_stability_gate_passed": True,
        "outcome": "detailed",
    }
    screen = SimpleNamespace(**{**defaults, **flags})
    assert diagnostic._capacity_outcome(screen)[0] == expected
