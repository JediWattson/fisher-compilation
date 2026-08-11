from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_parts_first_supermode_dose_curve as subject
import fisher_graph.gemma3_parts_first_supermode_experiment as experiment
from fisher_graph.gemma3_parts_first_native_intervention import (
    NativePartGateExecutionReceipt,
)


def _receipt(alpha: float) -> NativePartGateExecutionReceipt:
    return NativePartGateExecutionReceipt(
        gate_states=subject._expected_published_gate_states(alpha),
        hook_call_counts=(1, 1, 1, 1),
        hook_execution_order=(0, 1, 2, 3),
        mask_true_counts=(1, 1, 1, 1),
        alpha_one_control=alpha == 1.0,
        hooks_removed=True,
        weights_mutated_by_hooks=False,
        later_native_computation_reran=True,
    )


def _aggregate_stub(*, router_informative: bool) -> dict[str, object]:
    return {
        "relative_l2_error": 0.01,
        "informative_row_count": 3,
        "all_informative_rows_mixed_tolerance_passed": True,
        "all_informative_rows_sign_agree": True,
        "router_switch_count": 7,
        "router_stability_informative": router_informative,
    }


def test_frozen_schedule_and_cli_have_no_download_escape_hatch() -> None:
    assert subject.FROZEN_SCHEDULE == (1.0, 0.99, 0.97, 0.9, 0.5)
    assert subject.FROZEN_FINITE_ALPHAS == (0.99, 0.97, 0.9, 0.5)
    assert not hasattr(subject._parser().parse_args([]), "allow_download")
    assert subject.PartsFirstDoseCurveConfig().output_path.suffix == ".json"


def test_preflight_cli_never_loads_the_model(monkeypatch, capsys) -> None:
    expected = {
        "ready_to_load": True,
        "model_loaded": False,
        "local_files_only": True,
    }

    monkeypatch.setattr(
        subject,
        "build_post_c_dose_curve_preflight",
        lambda _config: expected,
    )
    monkeypatch.setattr(
        subject,
        "_default_runtime_loader",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("preflight loaded the model")
        ),
    )

    assert subject.main(["--preflight-only"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_native_receipt_roundtrip_closes_its_hash() -> None:
    receipt = _receipt(0.97)

    loaded = subject._native_receipt_from_dict(
        subject._native_receipt_to_dict(receipt)
    )

    assert loaded == receipt
    assert loaded.receipt_sha256 == receipt.receipt_sha256


def test_relative_l2_uses_predicted_norm_and_router_is_tautological_at_k1() -> None:
    rows = (
        {
            "actual_delta": 2.0,
            "first_order_predicted_delta": 1.0,
            "family_id": "family",
            "informative": True,
            "mixed_tolerance_passed": False,
            "sign_agreement": True,
            "router_switched": False,
        },
    )

    aggregate = subject._aggregate_rows(
        rows,
        numerical_epsilon=5.0e-5,
        router_stability_informative=False,
        subgroup_note="test",
    )

    assert aggregate["relative_l2_error"] == 1.0
    assert aggregate["router_stability_informative"] is False
    assert aggregate["quadratic_scaled_remainder_l2"] == 1.0
    assert aggregate["family_macro_average"]["represented_family_count"] == 1
    assert tuple(aggregate["per_family"]) == ("family",)

    results = [
        {
            "alpha": alpha,
            "aggregates": {"active": _aggregate_stub(router_informative=False)},
        }
        for alpha in (0.99, 0.97)
    ]
    assert subject._stratum_local_linear_pass(
        results,
        stratum="active",
        noop_control_passed=True,
    ) is True
    for result in results:
        result["aggregates"]["active"] = _aggregate_stub(  # type: ignore[index]
            router_informative=True
        )
    assert subject._stratum_local_linear_pass(
        results,
        stratum="active",
        noop_control_passed=True,
    ) is False


def _finite_result(
    alpha: float,
) -> tuple[
    dict[str, object],
    tuple[NativePartGateExecutionReceipt, ...],
]:
    noop_receipts = tuple(_receipt(1.0) for _ in subject.PUBLISHED_C_ROWS)
    finite_receipts = tuple(_receipt(alpha) for _ in subject.PUBLISHED_C_ROWS)
    receipt_set = []
    for (row_id, _family), noop, finite in zip(
        subject.PUBLISHED_C_ROWS,
        noop_receipts,
        finite_receipts,
        strict=True,
    ):
        receipt_set.extend(
            (
                (row_id, "alpha_one", noop),
                (row_id, "finite", finite),
            )
    )
    baseline = tuple(1.0 + index / 100.0 for index in range(12))
    directional = tuple((index + 1) / 10.0 for index in range(12))
    predicted = tuple((alpha - 1.0) * value for value in directional)
    intervened = tuple(
        before + delta * 1.01
        for before, delta in zip(baseline, predicted, strict=True)
    )
    evidence = experiment.FiniteInterventionEvidence(
        candidate_sha256=subject.PUBLISHED_CANDIDATE_SHA256,
        c_row_ids_sha256=subject.PUBLISHED_C_ROW_IDS_SHA256,
        gate_alpha=alpha,
        baseline_scores=baseline,
        intervened_scores=intervened,
        first_order_predicted_score_deltas=predicted,
        baseline_chart_ids=(subject.PUBLISHED_CHART_ID,) * 12,
        online_chart_ids_after=(subject.PUBLISHED_CHART_ID,) * 12,
        native_noop_max_abs_error=0.0,
        finite_intervention_candidate_count=1,
        finite_intervention_forward_count=12,
        executor_kind="live_model_bridge",
        native_gate_execution_receipt_sha256=subject._receipt_set_sha256(
            receipt_set
        ),
        source_model_fingerprint_before=subject.PUBLISHED_SOURCE_MODEL_SHA256,
        source_model_fingerprint_after=subject.PUBLISHED_SOURCE_MODEL_SHA256,
        source_execution_fingerprint_before=(
            subject.PUBLISHED_SOURCE_EXECUTION_SHA256
        ),
        source_execution_fingerprint_after=(
            subject.PUBLISHED_SOURCE_EXECUTION_SHA256
        ),
    )
    epsilon = 5.0e-5
    rows = [
        subject._expected_raw_row(
            row_id=row_id,
            family_id=family_id,
            path_active=row_id in subject.PUBLISHED_PATH_ACTIVE_ROW_IDS,
            evidence=evidence,
            index=index,
            epsilon=epsilon,
        )
        for index, (row_id, family_id) in enumerate(subject.PUBLISHED_C_ROWS)
    ]
    aggregates = {
        "chart_selected_all_12_primary": subject._aggregate_rows(
            rows,
            numerical_epsilon=epsilon,
            router_stability_informative=False,
            attenuation_from_alpha_one=1.0 - alpha,
            subgroup_note=(
                "primary original chart-only gate mask; contains both "
                "path-active and path-inactive rows"
            ),
        ),
        "path_active_3_exploratory": subject._aggregate_rows(
            [row for row in rows if row["path_active"] is True],
            numerical_epsilon=epsilon,
            router_stability_informative=False,
            attenuation_from_alpha_one=1.0 - alpha,
            subgroup_note=(
                "exploratory reused-C subgroup; n=3 across two semantic "
                "families; not fresh confirmation"
            ),
        ),
        "path_inactive_chart_only_9_exploratory": subject._aggregate_rows(
            [row for row in rows if row["path_active"] is False],
            numerical_epsilon=epsilon,
            router_stability_informative=False,
            attenuation_from_alpha_one=1.0 - alpha,
            subgroup_note=(
                "exploratory reused-C complement; n=9 selected only by "
                "the original chart mask"
            ),
        ),
    }
    result = {
        "alpha": alpha,
        "attenuation_from_alpha_one": 1.0 - alpha,
        "finite_intervention_evidence": evidence.to_dict(),
        "native_gate_receipts": [
            {
                "row_id": row_id,
                "condition": "finite",
                **subject._native_receipt_to_dict(receipt),
            }
            for (row_id, _family), receipt in zip(
                subject.PUBLISHED_C_ROWS,
                finite_receipts,
                strict=True,
            )
        ],
        "rows": rows,
        "aggregates": aggregates,
    }
    return result, noop_receipts


def test_finite_result_validator_binds_rows_metrics_and_receipts() -> None:
    result, noop_receipts = _finite_result(0.99)

    loaded = subject._validate_finite_alpha_result(
        result,
        expected_alpha=0.99,
        epsilon=5.0e-5,
        expected_noop_error=0.0,
        alpha_one_receipts=noop_receipts,
    )

    assert loaded == result
    tampered = copy.deepcopy(result)
    tampered["rows"][0]["first_order_predicted_delta"] += 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="per-row arithmetic"):
        subject._validate_finite_alpha_result(
            tampered,
            expected_alpha=0.99,
            epsilon=5.0e-5,
            expected_noop_error=0.0,
            alpha_one_receipts=noop_receipts,
        )

    relabeled = copy.deepcopy(result)
    relabeled_rows = relabeled["rows"]
    assert isinstance(relabeled_rows, list)
    relabeled_rows[0]["family_id"] = "integer_subtraction"
    relabeled["aggregates"] = {
        "chart_selected_all_12_primary": subject._aggregate_rows(
            relabeled_rows,
            numerical_epsilon=5.0e-5,
            router_stability_informative=False,
            subgroup_note=(
                "primary original chart-only gate mask; contains both "
                "path-active and path-inactive rows"
            ),
            attenuation_from_alpha_one=0.01,
        ),
        "path_active_3_exploratory": subject._aggregate_rows(
            [row for row in relabeled_rows if row["path_active"] is True],
            numerical_epsilon=5.0e-5,
            router_stability_informative=False,
            subgroup_note=(
                "exploratory reused-C subgroup; n=3 across two semantic "
                "families; not fresh confirmation"
            ),
            attenuation_from_alpha_one=0.01,
        ),
        "path_inactive_chart_only_9_exploratory": subject._aggregate_rows(
            [row for row in relabeled_rows if row["path_active"] is False],
            numerical_epsilon=5.0e-5,
            router_stability_informative=False,
            subgroup_note=(
                "exploratory reused-C complement; n=9 selected only by "
                "the original chart mask"
            ),
            attenuation_from_alpha_one=0.01,
        ),
    }
    with pytest.raises(ValueError, match="lineage"):
        subject._validate_finite_alpha_result(
            relabeled,
            expected_alpha=0.99,
            epsilon=5.0e-5,
            expected_noop_error=0.0,
            alpha_one_receipts=noop_receipts,
        )

    boolean_alpha = copy.deepcopy(result)
    boolean_alpha["alpha"] = True
    with pytest.raises(ValueError, match="schedule drifted"):
        subject._validate_finite_alpha_result(
            boolean_alpha,
            expected_alpha=0.99,
            epsilon=5.0e-5,
            expected_noop_error=0.0,
            alpha_one_receipts=noop_receipts,
        )


def test_novel_alpha_evidence_is_anchored_to_half_baseline_and_slope() -> None:
    results = [
        _finite_result(alpha)[0]
        for alpha in subject.FROZEN_FINITE_ALPHAS
    ]

    half = subject._require_half_anchored_evidence_lineage(results)

    assert half.gate_alpha == 0.5
    tampered = copy.deepcopy(results)
    evidence_value = tampered[0]["finite_intervention_evidence"]
    assert isinstance(evidence_value, dict)
    evidence_value["baseline_scores"][0] += 0.25
    evidence_value["evidence_sha256"] = ""
    rebuilt = experiment.FiniteInterventionEvidence.from_dict(evidence_value)
    tampered[0]["finite_intervention_evidence"] = rebuilt.to_dict()
    with pytest.raises(ValueError, match="anchored to published alpha=.50"):
        subject._require_half_anchored_evidence_lineage(tampered)

    rerouted = copy.deepcopy(results)
    rerouted_value = rerouted[1]["finite_intervention_evidence"]
    assert isinstance(rerouted_value, dict)
    rerouted_value["online_chart_ids_after"][0] = "fabricated.chart"
    rerouted_value["evidence_sha256"] = ""
    rerouted[1]["finite_intervention_evidence"] = (
        experiment.FiniteInterventionEvidence.from_dict(rerouted_value).to_dict()
    )
    with pytest.raises(ValueError, match="anchored to published alpha=.50"):
        subject._require_half_anchored_evidence_lineage(rerouted)


def test_alpha_one_absolute_values_are_anchored_to_half() -> None:
    half_value = _finite_result(0.5)[0]["finite_intervention_evidence"]
    assert isinstance(half_value, dict)
    half = experiment.FiniteInterventionEvidence.from_dict(half_value)
    rows = [
        {
            "captured_baseline_score": score,
            "fresh_baseline_score": score,
            "alpha_one_score": score,
            "fresh_baseline_chart_id": chart,
            "alpha_one_chart_id": chart,
            "chart_selected": True,
        }
        for score, chart in zip(
            half.baseline_scores,
            half.baseline_chart_ids,
            strict=True,
        )
    ]
    subject._require_alpha_one_half_anchor({"rows": rows}, half)

    for row in rows:
        row["captured_baseline_score"] = 1000.0
        row["fresh_baseline_score"] = 1000.0
        row["alpha_one_score"] = 1000.0
        row["fresh_baseline_chart_id"] = "fabricated.chart"
        row["alpha_one_chart_id"] = "fabricated.chart"
    with pytest.raises(ValueError, match="alpha-one values"):
        subject._require_alpha_one_half_anchor({"rows": rows}, half)


def test_frozen_sweep_uses_one_control_and_four_finite_alphas() -> None:
    demo = experiment.build_demo_capture_bundle()
    discovery = experiment._discover_on_A(
        role=demo.role_by_id["A_discovery"],
        axis=demo.response_axis,
        catalog=experiment._canonical_catalog(),
        config=experiment.PartsFirstPilotConfig(),
    )
    artifact = discovery.candidates[0]
    rows = tuple(
        SimpleNamespace(
            row_id=f"row-{index}",
            family_id="family",
            target_score=1.0 + index / 100.0,
            chart_coordinates=(0.0,),
        )
        for index in range(12)
    )
    atlas = SimpleNamespace(
        selected_chart_count=1,
        chart_id=lambda _coordinates: artifact.candidate.chart_id,
    )
    frozen = SimpleNamespace(
        C_role=SimpleNamespace(
            rows=rows,
            row_ids_sha256="b" * 64,
        ),
        chart_atlas=atlas,
        artifact=artifact,
    )

    class Adapter:
        def model_fingerprint(self):
            return "c" * 64

        def execution_fingerprint(self):
            return "d" * 64

    class Runtime:
        adapter = Adapter()
        batches_by_example_id = {row.row_id: index for index, row in enumerate(rows)}
        calls = []

        def _score_batch(self, batch, *, gates):
            alpha = None if gates is None else gates[0].alpha
            self.calls.append((batch, alpha))
            baseline = rows[batch].target_score
            score = baseline if alpha is None else baseline + (alpha - 1.0) * 2.0
            receipt = None
            if gates is not None:
                receipt = NativePartGateExecutionReceipt(
                    gate_states=tuple(gate.state() for gate in gates),
                    hook_call_counts=(1, 1, 1, 1),
                    hook_execution_order=(0, 1, 2, 3),
                    mask_true_counts=(1, 1, 1, 1),
                    alpha_one_control=alpha == 1.0,
                    hooks_removed=True,
                    weights_mutated_by_hooks=False,
                    later_native_computation_reran=True,
                )
            return score, (0.0,), receipt

        def _first_order_delta(self, artifact, row, *, alpha):
            return (alpha - 1.0) * 2.0

    runtime = Runtime()

    sweep = subject._run_frozen_native_sweep(runtime, frozen)  # type: ignore[arg-type]

    assert tuple(item.gate_alpha for item in sweep.finite_evidences) == (
        0.99,
        0.97,
        0.9,
        0.5,
    )
    assert sweep.alpha_one_max_abs_error == 0.0
    assert len(runtime.calls) == 72


def test_atomic_writer_refuses_to_clobber(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        subject,
        "validate_post_c_dose_curve_report",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject,
        "_load_post_c_dose_curve_report",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    target = tmp_path / "dose.json"

    assert subject.write_post_c_dose_curve_report({"value": 1}, target) == {
        "value": 1
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject.write_post_c_dose_curve_report({"value": 2}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}


def test_atomic_writer_rejects_post_link_inode_replacement(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        subject,
        "validate_post_c_dose_curve_report",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject,
        "_load_post_c_dose_curve_report",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    target = tmp_path / "dose.json"

    def replace_after_link(_directory):
        if target.exists():
            target.unlink()
            target.write_text('{"value": 999}\n', encoding="utf-8")

    monkeypatch.setattr(subject, "_fsync_directory", replace_after_link)

    with pytest.raises(RuntimeError, match="target inode changed"):
        subject.write_post_c_dose_curve_report({"value": 1}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 999}


def test_strict_json_loader_rejects_duplicate_keys(tmp_path) -> None:
    source = tmp_path / "duplicate.json"
    source.write_text('{"schema":"first","schema":"last"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unavailable or invalid"):
        subject._load_strict_json_object(source, label="fixture")


def test_writer_never_unlinks_a_published_path_during_fsync_failure(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        subject,
        "validate_post_c_dose_curve_report",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject,
        "_load_post_c_dose_curve_report",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )
    monkeypatch.setattr(
        subject,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("fsync failed")),
    )
    target = tmp_path / "dose.json"

    with pytest.raises(OSError, match="fsync failed"):
        subject.write_post_c_dose_curve_report({"value": 1}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
