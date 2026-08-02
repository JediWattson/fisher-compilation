from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_joint_state_gain_capacity_diagnostic as diagnostic,
)


def _mock_v6_evidence() -> tuple[dict[str, object], object]:
    folds: list[object] = []
    expected_folds: list[dict[str, object]] = []
    expected_inner: list[dict[str, object]] = []
    for outer in range(8):
        inner_records: list[object] = []
        for inner in range(7):
            metadata = {
                "outer_held_family_id": f"f{outer}",
                "inner_held_family_id": f"f{(outer + inner + 1) % 8}",
                "state_predicted_incremental_derivative": -float(inner + 1),
                "artifact_sha256": f"{100 + outer * 7 + inner:064x}",
            }
            expected_inner.append(metadata)
            inner_records.append(
                SimpleNamespace(
                    artifact_sha256=metadata["artifact_sha256"],
                    metadata=lambda metadata=metadata: dict(metadata),
                )
            )
        fold_metadata = {
            "outer_held_family_id": f"f{outer}",
            "artifact_sha256": f"{200 + outer:064x}",
        }
        expected_folds.append(fold_metadata)
        folds.append(
            SimpleNamespace(
                artifact_sha256=fold_metadata["artifact_sha256"],
                inner_family_records=tuple(inner_records),
                metadata=lambda metadata=fold_metadata: dict(metadata),
            )
        )
    screen_metadata = {
        "outcome": "fail_state_vs_scalar_attribution",
        "capacity_screen_passed": False,
        "artifact_sha256": "a" * 64,
    }
    refits = {
        f"f{index}": SimpleNamespace(
            metadata=lambda index=index: {
                "held_family_id": f"f{index}",
                "artifact_sha256": f"{300 + index:064x}",
            }
        )
        for index in range(8)
    }
    gradients = tuple(
        {"cell": index, "receipt_sha256": f"{400 + index:064x}"}
        for index in range(56)
    )
    rows = tuple(
        {"cell": index, "receipt_sha256": f"{500 + index:064x}"}
        for index in range(56)
    )
    live_v4 = {"artifact_sha256": "b" * 64}
    carriers = {"carrier_receipt_set_sha256": "c" * 64}
    phases = SimpleNamespace(
        screen=SimpleNamespace(
            artifact_sha256="a" * 64,
            outcome="fail_state_vs_scalar_attribution",
            capacity_screen_passed=False,
            metadata=lambda: dict(screen_metadata),
        ),
        fold_records=tuple(folds),
        row_bank=SimpleNamespace(
            refits=refits,
            gradient_receipts=gradients,
            row_bank_receipts=rows,
        ),
        live_v4_binding=live_v4,
        carrier_binding=carriers,
    )
    report = {
        "analytic_capacity_screen": dict(screen_metadata),
        "analytic_fold_records": [dict(row) for row in expected_folds],
        "analytic_inner_family_records": [dict(row) for row in expected_inner],
        "candidate_gain_refits": [
            refits[family].metadata() for family in sorted(refits)
        ],
        "candidate_gradient_receipts": list(gradients),
        "row_resolved_vjp_receipts": list(rows),
        "live_v4_refit_and_gradient_binding": live_v4,
        "v5_plus_carrier_binding": carriers,
        "resources": {
            "total_model_forward_count": 112,
            "total_backward_call_count": 494,
            "finite_state_candidate_model_forward_count": 0,
        },
    }
    return report, phases


def test_live_v6_binding_requires_exact_screen_folds_inner_refits_and_receipts() -> None:
    report, phases = _mock_v6_evidence()
    binding = diagnostic._authenticate_live_v6_evidence(
        v6_report=report, phases=phases
    )
    assert binding["v6_live_refit_count"] == 8
    assert binding["v6_live_gradient_receipt_count"] == 56
    assert binding["v6_live_row_receipt_count"] == 56
    assert binding["all_56_inner_metadata_rows_canonically_equal"] is True
    assert binding["authenticated_before_joint_fit"] is True

    report["analytic_inner_family_records"][0][  # type: ignore[index]
        "state_predicted_incremental_derivative"
    ] = 1.0
    with pytest.raises(RuntimeError, match="inner records"):
        diagnostic._authenticate_live_v6_evidence(
            v6_report=report, phases=phases
        )


def test_v6_loader_uses_exact_file_report_schema_and_failure_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    valid = {
        "schema": diagnostic.v6diag._SCHEMA,
        "classification": "state_not_better_than_scalar",
        "passed": False,
    }

    def load(path: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return dict(valid)

    monkeypatch.setattr(diagnostic.token_v1, "_load_pinned_report", load)
    assert diagnostic._load_v6_report("v6.json") == valid
    assert captured["expected_file_sha256"] == diagnostic.V6_REPORT_FILE_SHA256
    assert captured["expected_report_sha256"] == diagnostic.V6_REPORT_SHA256
    monkeypatch.setattr(
        diagnostic.token_v1,
        "_load_pinned_report",
        lambda *args, **kwargs: {**valid, "passed": True},
    )
    with pytest.raises(RuntimeError, match="control differs"):
        diagnostic._load_v6_report("v6.json")


def test_v7_resource_ledger_is_exact_and_performs_no_finite_joint_execution() -> None:
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
        },
    )
    assert result["total_model_forward_count"] == 112
    assert result["total_backward_call_count"] == 494
    assert result["v6_reproduction_additional_model_forward_count"] == 0
    assert result["joint_fit_model_forward_count"] == 0
    assert result["finite_joint_candidate_model_forward_count"] == 0
    assert result["phase_order"][-3:] == (
        "exact_v6_screen_fold_and_inner_reproduction",
        "nested_held_inner_family_joint5d_analytic_capacity_screen",
        "scalar_hash_only_report_publication",
    )
    with pytest.raises(RuntimeError, match="resource accounting"):
        diagnostic._resource_accounting(
            endpoint_resources={
                "base_forward_count": 16,
                "native_forward_count": 16,
                "endpoint_token_vjp_forward_count": 16,
                "endpoint_token_vjp_backward_call_count": 110,
            },
            gradient_resources={
                "gradient_native_forward_count": 8,
                "gradient_candidate_vjp_forward_count": 56,
                "gradient_candidate_vjp_backward_call_count": 385,
            },
        )


def test_phase_order_reproduces_v6_before_any_joint_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    v6_phases = object()
    joint_screen = object()
    monkeypatch.setattr(
        diagnostic.v6diag,
        "_execute_capacity_phases",
        lambda **kwargs: order.append("v6_reproduction") or v6_phases,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_live_v6_evidence",
        lambda **kwargs: order.append("v6_auth")
        or {"authenticated_before_joint_fit": True},
    )

    def joint(**kwargs: object):
        assert order[-1] == "v6_auth"
        order.append("joint_fit")
        return joint_screen, tuple(), {"artifact_sha256": "a" * 64}

    monkeypatch.setattr(diagnostic, "_fit_nested_joint_capacity_screen", joint)
    result = diagnostic._execute_joint_phases(
        context=object(),
        parent={},
        v3_report={},
        v4_report={},
        v5_report={},
        v6_report={},
        traces=tuple(),
        endpoint_resources={},
        basis=torch.zeros((1, 1)),
        fits={},
    )
    assert order == ["v6_reproduction", "v6_auth", "joint_fit"]
    assert result.v6_phases is v6_phases
    assert result.joint_screen is joint_screen


def test_mock_nested_joint_fit_uses_all_64_exact_v6_scalar_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = tuple(f"f{index}" for index in range(8))
    fits = {
        family: SimpleNamespace(held_family_id=family) for family in families
    }
    refits: dict[str, object] = {}
    cells: dict[str, tuple[object, ...]] = {}
    v6_folds: list[object] = []

    def scalar(artifact: str) -> object:
        metadata = {"artifact_sha256": artifact, "kind": "v6_scalar"}
        return SimpleNamespace(
            artifact_sha256=artifact,
            metadata=lambda metadata=metadata: dict(metadata),
        )

    for outer in families:
        training = tuple(family for family in families if family != outer)
        refits[outer] = SimpleNamespace(training_family_ids=training)
        cells[outer] = tuple(
            SimpleNamespace(
                example_id=f"e-{family}",
                family_id=family,
                held_family_id=outer,
            )
            for family in training
        )
        v6_folds.append(
            SimpleNamespace(
                outer_held_family_id=outer,
                full_static_control_fit=scalar(f"full-{outer}"),
                inner_family_records=tuple(
                    SimpleNamespace(
                        inner_held_family_id=inner,
                        inner_static_control_fit=scalar(f"inner-{outer}-{inner}"),
                    )
                    for inner in training
                ),
            )
        )
    phases = SimpleNamespace(
        fold_records=tuple(v6_folds),
        row_bank=SimpleNamespace(refits=refits, cells=cells),
    )
    monkeypatch.setattr(
        diagnostic.v6diag,
        "_ordered_k64",
        lambda fit: torch.eye(64, dtype=torch.float64),
    )
    monkeypatch.setattr(
        diagnostic.v6diag,
        "_mean_delta",
        lambda refit: torch.zeros(64, dtype=torch.float64),
    )
    monkeypatch.setattr(diagnostic.v6diag, "_feature_example", lambda cell: cell)

    def codec(examples: object, *, held_family_id: str, ordered_directions: object):
        values = tuple(examples)
        training = tuple(sorted(value.family_id for value in values))
        outer_training = set(families) - {held_family_id}
        missing = tuple(sorted(outer_training - set(training)))
        artifact = (
            f"full-{held_family_id}"
            if not missing
            else f"inner-{held_family_id}-{missing[0]}"
        )
        return SimpleNamespace(
            held_family_id=held_family_id,
            training_family_ids=training,
            expected_scalar_artifact=artifact,
        )

    monkeypatch.setattr(
        diagnostic.v6diag,
        "fit_candidate_conditioned_k64_state_feature_codec",
        codec,
    )
    monkeypatch.setattr(
        diagnostic.v6diag,
        "_state_gradient_example",
        lambda *, cell, **kwargs: SimpleNamespace(
            family_id=cell.family_id, example_id=cell.example_id
        ),
    )
    fit_calls: list[tuple[str, torch.Tensor]] = []

    def fit_joint(
        refit: object,
        codec: object,
        examples: object,
        *,
        ordered_directions: torch.Tensor,
    ):
        fit_calls.append((codec.expected_scalar_artifact, ordered_directions))
        joint = SimpleNamespace(
            artifact_sha256=f"joint-{codec.expected_scalar_artifact}"
        )
        return joint, scalar(codec.expected_scalar_artifact)

    monkeypatch.setattr(
        diagnostic,
        "fit_candidate_conditioned_k64_joint_state_gain_field_with_scalar_control",
        fit_joint,
    )
    monkeypatch.setattr(
        diagnostic,
        "build_candidate_conditioned_k64_joint_inner_family_analytic_record",
        lambda full, inner, control, held: SimpleNamespace(
            inner_held_family_id=held.family_id,
            artifact_sha256=f"record-{held.family_id}",
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_candidate_conditioned_k64_joint_state_gain_fold_analytic_record",
        lambda full, control, records: SimpleNamespace(
            outer_held_family_id=control.artifact_sha256.removeprefix("full-"),
            inner_family_records=tuple(records),
            artifact_sha256=f"fold-{control.artifact_sha256}",
        ),
    )
    screen = object()
    monkeypatch.setattr(
        diagnostic,
        "screen_candidate_conditioned_k64_joint_state_gain_capacity",
        lambda records: screen,
    )
    actual_screen, records, binding = diagnostic._fit_nested_joint_capacity_screen(
        v6_phases=phases, fits=fits
    )
    assert actual_screen is screen
    assert len(records) == 8
    assert len(fit_calls) == 64
    assert sum(value.startswith("full-") for value, _ in fit_calls) == 8
    assert sum(value.startswith("inner-") for value, _ in fit_calls) == 56
    assert all(
        torch.equal(directions, torch.eye(64, dtype=torch.float64))
        for _, directions in fit_calls
    )
    assert binding["full_scalar_comparator_count"] == 8
    assert binding["inner_scalar_comparator_count"] == 56
    assert binding["total_scalar_comparator_count"] == 64
    assert binding["every_scalar_comparator_canonically_reproduces_v6"] is True


def test_cli_has_no_knobs_and_pyproject_exposes_v7() -> None:
    parser = diagnostic.build_parser()
    assert [action.dest for action in parser._actions] == ["help"]
    assert vars(parser.parse_args([])) == {}
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "alternate.json"])
    pyproject = Path(diagnostic.__file__).resolve().parents[2] / "pyproject.toml"
    assert (
        "fisher-graph-gemma-l3-l4-complete-h4-tail-token-fisher-"
        "candidate-joint-state-gain-capacity-v7-a-dev"
    ) in pyproject.read_text()


def test_frozen_joint_and_integrity_gates_are_exact_and_tamper_sensitive() -> None:
    screen = SimpleNamespace(
        feature_and_design_gate_passed=True,
        residual_energy_gate_passed=True,
        non_noop_gate_passed=True,
        negative_inner_global_gate_passed=True,
        negative_inner_local_gate_passed=True,
        joint_beats_scalar_gate_passed=True,
        joint_beats_scalar_cell_count=0,
        cosine_stability_gate_passed=True,
    )
    joint_gates = diagnostic._joint_gate_results(screen)
    assert set(joint_gates) == {
        "augmented_feature_and_joint_design_rank_five_condition_at_most_100_all_fits",
        "residual_conditional_state_fisher_at_least_five_percent_in_six_folds",
        "at_least_six_of_eight_full_joint_fits_are_non_noop",
        "at_least_42_of_56_joint_derivatives_negative_and_4_of_7_in_six_folds",
        "joint_inner_macro_strictly_beats_exact_v6_scalar_in_six_folds",
        "median_inner_full_state_slope_cosine_at_least_point_90_in_six_folds",
    }
    assert all(joint_gates.values())
    assert not any("cell" in name for name in joint_gates)

    v6_binding = {
        "screen_metadata_canonically_equal": True,
        "fold_metadata_canonically_equal": True,
        "all_56_inner_metadata_rows_canonically_equal": True,
        "v4_refits_and_receipts_canonically_equal": True,
        "v5_carriers_canonically_equal": True,
        "authenticated_before_joint_fit": True,
    }
    phases = SimpleNamespace(
        v6_binding=v6_binding,
        scalar_comparator_binding={
            "every_scalar_comparator_canonically_reproduces_v6": True,
            "total_scalar_comparator_count": 64,
        },
        joint_fold_records=(object(),) * 8,
        v6_phases=SimpleNamespace(
            row_bank=SimpleNamespace(
                gradient_receipts=(
                    {
                        "maximum_future_gradient_abs": 0.0,
                        "future_gradient_nonzero_count": 0,
                    },
                )
                * 56
            )
        ),
    )
    traces = (
        SimpleNamespace(
            maximum_future_gradient_abs=0.0,
            future_gradient_nonzero_count=0,
        ),
    ) * 16
    resources = {
        "total_model_forward_count": 112,
        "total_backward_call_count": 494,
        "finite_joint_candidate_model_forward_count": 0,
    }
    integrity = diagnostic._integrity_gate_results(
        phases=phases,
        traces=traces,
        resources=resources,
        joint_inner_record_count=56,
    )
    assert all(integrity.values())

    v6_binding["authenticated_before_joint_fit"] = False
    tampered = diagnostic._integrity_gate_results(
        phases=phases,
        traces=traces,
        resources=resources,
        joint_inner_record_count=56,
    )
    assert tampered["all_v6_evidence_authenticated_before_any_joint_fit"] is False


def test_v6_hash_pins_output_name_safety_and_write_once(tmp_path: Path) -> None:
    assert diagnostic.V6_REPORT_FILE_SHA256 == (
        "95e49e10f802121436f081f381c8622653e79767fc2ff4d265b4c976b8193a28"
    )
    assert diagnostic.V6_REPORT_SHA256 == (
        "001738439b4c2bdd052f6220283103c592b96742b27f105778584e2529a758eb"
    )
    assert str(diagnostic.DEFAULT_OUTPUT).endswith(
        "candidate-joint-state-gain-capacity-lofo-a-fit16-dev-v7.json"
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

    output = tmp_path / "v7.json"
    report = {
        "schema": diagnostic._SCHEMA,
        "artifact": {"file": str(output), "committable": False},
        "safety": safety,
        "passed": False,
        "classification": "joint_not_better_than_scalar",
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
