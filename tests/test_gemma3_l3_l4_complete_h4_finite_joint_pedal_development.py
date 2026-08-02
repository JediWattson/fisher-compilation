from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fisher_graph.complete_h4_fisher_finite_joint_pedal import (
    canonical_balanced_rank_svd_retraction,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_finite_joint_pedal_development as development,
)


_SHA = "a" * 64
_FAMILIES = tuple(f"family-{index}" for index in range(8))


def _records(
    rows: tuple[tuple[str, str], ...],
) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            sequence=SimpleNamespace(family_id=family, example_id=example)
        )
        for family, example in rows
    )


def _fidelity(
    *,
    absolute_delta_nll: float,
    kl: float,
    top1: float,
    family_absolute_delta_nll: tuple[float, ...] | None = None,
    passed: bool = True,
) -> dict[str, object]:
    family_values = family_absolute_delta_nll or (absolute_delta_nll,) * 8
    result: dict[str, object] = {}
    for ledger in development._REQUIRED_LEDGERS:
        result[ledger] = {
            "gates": {"passed": passed},
            "aggregate": {
                "delta_nll_per_token": absolute_delta_nll,
                "absolute_delta_nll_per_token": absolute_delta_nll,
                "source_to_candidate_kl_per_token": kl,
                "top1_agreement_to_source": top1,
            },
            "family_summary": {
                "macro": {
                    "absolute_delta_nll_per_token": absolute_delta_nll,
                    "source_to_candidate_kl_per_token": kl,
                    "top1_agreement_to_source": top1,
                },
                "families": tuple(
                    {
                        "family_id": family,
                        "absolute_delta_nll_per_token": value,
                    }
                    for family, value in zip(
                        _FAMILIES, family_values, strict=True
                    )
                ),
            },
        }
    return result


def _optimization_receipts() -> dict[str, object]:
    return {
        family: {
            "checkpoint_scores": (1.0, 0.80, 0.82, 0.84, 0.86),
            "selected_checkpoint": 1,
            "fit_qualification": {"passed": True},
            "capability_receipt": {
                "held_family_id": family,
                "held_family_capability_excluded": True,
            },
        }
        for family in _FAMILIES
    }


def _held_runtime(*, conditional: bool) -> dict[str, object]:
    return {
        family: {
            "pointwise_trust_passed": True,
            "pedal_nonconstant": conditional,
        }
        for family in _FAMILIES
    }


def _passing_arms() -> dict[str, dict[str, object]]:
    metrics = {
        development.PARENT_ID: (1.00, 1.00, 0.60),
        development.FISHER_START_ID: (0.90, 0.90, 0.62),
        development.FISHER_UNIT_ID: (0.85, 0.85, 0.64),
        development.FISHER_INTERCEPT_ID: (0.80, 0.80, 0.66),
        development.FISHER_CONDITIONAL_ID: (0.70, 0.70, 0.68),
        development.PCA_CONDITIONAL_ID: (0.75, 0.75, 0.67),
    }
    result: dict[str, dict[str, object]] = {}
    for arm_index, (arm, (delta, kl, top1)) in enumerate(metrics.items()):
        row: dict[str, object] = {
            "arm_id": arm,
            "fold_provider_artifact_sha256s": {
                family: f"{0x1000 + 0x10 * arm_index + index:064x}"
                for index, family in enumerate(_FAMILIES)
            },
            "fidelity": _fidelity(
                absolute_delta_nll=delta,
                kl=kl,
                top1=top1,
            ),
        }
        if arm in (
            development.FISHER_CONDITIONAL_ID,
            development.PCA_CONDITIONAL_ID,
        ):
            row["optimization_receipts"] = _optimization_receipts()
        if arm in development._JOINT_IDS:
            row["held_runtime_diagnostics"] = _held_runtime(
                conditional=arm
                in (
                    development.FISHER_CONDITIONAL_ID,
                    development.PCA_CONDITIONAL_ID,
                )
            )
        result[arm] = row
    return result


def _set_macro(
    arms: dict[str, dict[str, object]],
    arm: str,
    key: str,
    value: float,
) -> None:
    arms[arm]["fidelity"]["ordinary"]["family_summary"]["macro"][  # type: ignore[index]
        key
    ] = value


def _set_family_values(
    arms: dict[str, dict[str, object]],
    arm: str,
    values: tuple[float, ...],
) -> None:
    rows = arms[arm]["fidelity"]["ordinary"]["family_summary"][  # type: ignore[index]
        "families"
    ]
    for row, value in zip(rows, values, strict=True):
        row["absolute_delta_nll_per_token"] = value


def _report_kwargs(
    arms: dict[str, dict[str, object]],
    *,
    full_refit: dict[str, object] | None,
    candidate: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "artifact_path": Path(".local-runs/test-v19-report.json"),
        "panel": {"prompt_count": 16, "family_count": 8},
        "bridge_binding_sha256": _SHA,
        "folds": development._v14.build_outer_lofo_splits(_FAMILIES),
        "prerequisites": {
            "v18_candidate_was_null": True,
            "v18_coordinate_geometry_qualification": {"passed": True},
        },
        "fit_collection": {"prompt_count": 16, "family_count": 8},
        "base_fidelity": {},
        "arms": arms,
        "full_refit_qualification": full_refit,
        "candidate": candidate,
        "integrity": {"guard_opened": False, "calibration_b_opened": False},
    }


def _full_refit_qualification(*, passed: bool) -> dict[str, object]:
    return {
        "provider_artifact_sha256": _SHA,
        "parent_provider_artifact_sha256": "b" * 64,
        "start_provider_artifact_sha256": "c" * 64,
        "optimization_receipt": {
            "selected_provider_artifact_sha256": _SHA,
            "start_provider_artifact_sha256": "c" * 64,
            "training_family_ids": _FAMILIES,
            "training_sequence_sha256s": tuple(
                f"{index + 1:064x}" for index in range(16)
            ),
            "fit_qualification": {
                "passed": passed,
                "objective_macro_relative_improvement_at_least_threshold": passed,
            },
            "capability_receipt": {
                "held_family_id": None,
                "authorized_example_count": 16,
                "authorized_family_count": 8,
                "access_count": 80,
            },
        },
        "runtime_diagnostic": {
            "provider_artifact_sha256": _SHA,
            "held_sequence_count": 16,
            "pointwise_trust_passed": True,
            "pedal_nonconstant": True,
        },
        "serving_resources": {
            "prepared_float_scalar_count": 377_608,
            "logical_macs_per_token_upper_bound": 541_187,
        },
        "passed": passed,
    }


def test_exact_float64_teacher_kl_is_full_row_detached_and_nonmutating() -> None:
    teacher = torch.tensor(
        ((2.0, -1.0, 0.5, 4.0), (-0.5, 1.0, 3.0, -2.0)),
        dtype=torch.float32,
        requires_grad=True,
    )
    candidate = torch.tensor(
        ((1.5, -0.5, 1.0, -3.0), (0.0, 2.0, -1.0, 4.0)),
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher_before = teacher.detach().clone()
    candidate_before = candidate.detach().clone()

    actual = development.exact_float64_teacher_kl(teacher, candidate)
    teacher64 = teacher.detach().to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    teacher_logp = F.log_softmax(teacher64, dim=-1)
    expected_rows = (
        teacher_logp.exp()
        * (teacher_logp - F.log_softmax(candidate64, dim=-1))
    ).sum(dim=-1)

    assert actual.dtype == torch.float64
    assert actual.ndim == 0
    assert torch.equal(actual, expected_rows.mean())
    # Both rows and the last vocabulary column materially contribute.
    assert expected_rows[0] != expected_rows[1]
    truncated_teacher = teacher64[:, :-1]
    truncated_candidate = candidate64[:, :-1]
    truncated_logp = F.log_softmax(truncated_teacher, dim=-1)
    truncated = (
        truncated_logp.exp()
        * (truncated_logp - F.log_softmax(truncated_candidate, dim=-1))
    ).sum(dim=-1).mean()
    assert actual != truncated

    actual.backward()
    assert teacher.grad is None
    assert candidate.grad is not None
    assert bool(torch.isfinite(candidate.grad).all())
    assert torch.equal(teacher.detach(), teacher_before)
    assert torch.equal(candidate.detach(), candidate_before)


@pytest.mark.parametrize(
    "teacher_shape,candidate_shape",
    [((2, 3), (2, 4)), ((2, 3, 1), (2, 3, 1)), ((0, 3), (0, 3))],
)
def test_exact_float64_teacher_kl_rejects_non_full_row_geometry(
    teacher_shape: tuple[int, ...],
    candidate_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="matching finite"):
        development.exact_float64_teacher_kl(
            torch.zeros(teacher_shape), torch.zeros(candidate_shape)
        )


def test_canonical_balanced_factors_match_core_svd_exactly() -> None:
    generator = torch.Generator().manual_seed(1901)
    matrix = torch.randn((9, 7), generator=generator, dtype=torch.float64)

    expected_left, expected_right = canonical_balanced_rank_svd_retraction(
        matrix, rank=4
    )
    actual_left, actual_right = development.canonical_balanced_rank_factors(
        matrix, rank=4
    )

    assert torch.equal(actual_left, expected_left)
    assert torch.equal(actual_right, expected_right)
    assert torch.equal(actual_left @ actual_right, expected_left @ expected_right)
    assert torch.allclose(
        actual_left.T @ actual_left,
        actual_right @ actual_right.T,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_checkpoint_selection_uses_earliest_exact_tie_only() -> None:
    assert development.choose_earliest_checkpoint((3.0, 1.0, 1.0, 2.0)) == 1
    # A merely nearby value is not treated as a tie.
    assert development.choose_earliest_checkpoint((1.0, 1.0 - 1.0e-15)) == 1
    with pytest.raises(ValueError, match="nonempty finite"):
        development.choose_earliest_checkpoint(())
    with pytest.raises(ValueError, match="nonempty finite"):
        development.choose_earliest_checkpoint((1.0, float("nan")))


def _joint_state(value: float) -> development._JointState:
    return development._JointState(
        direction_left=torch.full((3, 16), value, dtype=torch.float64),
        direction_right=torch.full((16, 1), value, dtype=torch.float64),
        pedal_weight=torch.full((3,), value, dtype=torch.float64),
        pedal_bias=torch.full((1,), value, dtype=torch.float64),
    )


def test_adam_step_matches_bias_corrected_math_and_fixed_group_lrs() -> None:
    state = _joint_state(0.0)
    gradient = _joint_state(2.0)
    zero = development._zero_state(state)

    updated, moments = development._adam_step(
        state,
        gradient,
        development._AdamMoments(first=zero, second=zero, step=0),
    )

    direction_delta = development._DIRECTION_LEARNING_RATE * 2.0 / (
        2.0 + development._ADAM_EPSILON
    )
    pedal_delta = development._PEDAL_LEARNING_RATE * 2.0 / (
        2.0 + development._ADAM_EPSILON
    )
    assert moments.step == 1
    assert torch.allclose(
        moments.first.direction_left,
        torch.full((3, 16), 0.2, dtype=torch.float64),
    )
    assert torch.allclose(
        moments.second.direction_left,
        torch.full((3, 16), 0.004, dtype=torch.float64),
    )
    assert torch.allclose(
        updated.direction_left,
        torch.full((3, 16), -direction_delta, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert torch.allclose(
        updated.pedal_weight,
        torch.full((3,), -pedal_delta, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert pedal_delta / direction_delta == pytest.approx(25.0)


def test_teacher_vault_clones_inputs_excludes_held_and_detects_tampering() -> None:
    source_rows = {
        "train-example": torch.tensor(((1.0, 2.0, 3.0),)),
        "held-example": torch.tensor(((4.0, 5.0, 6.0),)),
    }
    families = {"train-example": "train-family", "held-example": "held-family"}
    vault = development._TeacherRowVault(source_rows, families)
    artifact = vault.artifact_sha256

    # Caller-side mutation after construction cannot alter the authenticated vault.
    source_rows["train-example"].add_(100.0)
    source_rows["held-example"].zero_()
    vault.validate_integrity()
    assert vault.artifact_sha256 == artifact

    capability = vault.capability(
        ("train-example",), held_family_id="held-family"
    )
    assert torch.equal(
        capability.get("train-example", family_id="train-family"),
        torch.tensor(((1.0, 2.0, 3.0),)),
    )
    with pytest.raises(PermissionError, match="outside"):
        capability.get("held-example", family_id="held-family")
    with pytest.raises(ValueError, match="held family"):
        vault.capability(("held-example",), held_family_id="held-family")

    # Returned rows remain authenticated; mutating one invalidates the capability.
    capability._rows["train-example"].add_(1.0)
    with pytest.raises(RuntimeError, match="mutated|drifted"):
        capability.receipt()

    vault._rows["held-example"].add_(1.0)
    with pytest.raises(RuntimeError, match="mutated"):
        vault.validate_integrity()


def test_family_equal_objective_is_not_prompt_weighted_and_rejects_coverage_drift() -> None:
    records = _records(
        (("short-family", "a"), ("long-family", "b"), ("long-family", "c"))
    )
    score, per_family = development._family_equal_mean(
        {"a": 10.0, "b": 0.0, "c": 2.0}, records
    )
    assert per_family == {"long-family": 1.0, "short-family": 10.0}
    assert score == 5.5
    assert score != pytest.approx(4.0)  # the prompt-equal mean

    with pytest.raises(ValueError, match="omitted"):
        development._family_equal_mean({"a": 10.0, "b": 0.0}, records)
    with pytest.raises(ValueError, match="unknown"):
        development._family_equal_mean(
            {"a": 10.0, "b": 0.0, "c": 2.0, "extra": 7.0}, records
        )


def test_optimizer_enforces_exact_two_prompt_family_geometry_before_execution() -> None:
    invalid = _records(
        (("family-a", "a0"), ("family-a", "a1"), ("family-b", "b0"))
    )
    with pytest.raises(RuntimeError, match="two prompts per family"):
        development._optimize_joint_provider(
            None,
            invalid,
            None,
            None,
            held_family_id="held",
            coordinate_objective="reverse_vjp_fisher",
        )


def test_protocol_geometry_and_work_accounting_are_exact() -> None:
    assert development._EXPECTED_PROMPTS == 16
    assert development._EXPECTED_FAMILIES == 8
    assert development._EXPECTED_TRAINING_PROMPTS_PER_FOLD == 14
    assert development._EXPECTED_VOCABULARY == 262_144

    outer_only = development._work_accounting(full_refit_fitted=False)
    assert outer_only["outer_provider_fit_count"] == 40
    assert outer_only["conditional_full_panel_provider_fit_count"] == 0
    assert outer_only["full_model_forward_count"] == 1_280
    assert outer_only["full_suffix_backward_traversal_count"] == 912
    assert outer_only["local_head_autograd_contraction_count"] == 896
    assert outer_only["total_autograd_grad_call_count"] == 1_808

    with_full = development._work_accounting(full_refit_fitted=True)
    assert with_full["outer_provider_fit_count"] == 40
    assert with_full["conditional_full_panel_provider_fit_count"] == 3
    assert with_full["full_model_forward_count"] == 1_360
    assert with_full["full_suffix_backward_traversal_count"] == 976
    assert with_full["local_head_autograd_contraction_count"] == 960
    assert with_full["total_autograd_grad_call_count"] == 1_936
    assert with_full["breakdown"]["conditional_full_refit_forwards"] == 80
    assert with_full["breakdown"]["conditional_full_refit_backwards"] == 64


def test_v18_prerequisite_preserves_fold_lineage_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold_hashes = {family: f"{index + 1:064x}" for index, family in enumerate(_FAMILIES)}
    payload = {
        "candidate": None,
        "coordinate_geometry_qualification": {"passed": True},
        "arms": {
            arm: {"fold_provider_artifact_sha256s": fold_hashes}
            for arm in (
                development.PARENT_ID,
                development._v18.FISHER_PEDAL_ID,
                development._v18.PCA_PEDAL_ID,
            )
        },
    }
    prerequisite = tmp_path / "v18.json"
    prerequisite.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(development, "_V18_OUTPUT", prerequisite)
    monkeypatch.setattr(
        development._v18,
        "_validate_prerequisite_report",
        lambda *args, **kwargs: {"report_sha256": "b" * 64},
    )

    receipt = development._validate_prerequisites()
    lineage = receipt["v18_fold_provider_artifact_sha256s"]
    assert lineage[development.PARENT_ID] == fold_hashes
    assert lineage[development._v18.FISHER_PEDAL_ID] == fold_hashes
    assert lineage[development._v18.PCA_PEDAL_ID] == fold_hashes

    payload["arms"][development._v18.PCA_PEDAL_ID][
        "fold_provider_artifact_sha256s"
    ] = dict(tuple(fold_hashes.items())[:-1])
    prerequisite.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fold lineage"):
        development._validate_prerequisites()


def test_output_paths_reject_escape_and_v18_aliases() -> None:
    with pytest.raises(ValueError, match="under .local-runs"):
        development._validate_output(Path(".local-runs/../escaped.json"))
    with pytest.raises(ValueError, match="under .local-runs"):
        development._validate_provider_output(
            Path(".local-runs/../escaped.provider.pt")
        )

    aliased_v18 = development._V18_OUTPUT.parent / "nested" / ".." / development._V18_OUTPUT.name
    with pytest.raises(ValueError, match="write-once V18"):
        development.run_gemma3_l3_l4_complete_h4_finite_joint_pedal_development(
            output=aliased_v18
        )
    aliased_provider = (
        development._V18_OUTPUT.parent
        / "nested"
        / ".."
        / development._V18_OUTPUT.with_suffix(".provider.pt").name
    )
    with pytest.raises(ValueError, match="write-once V18"):
        development.run_gemma3_l3_l4_complete_h4_finite_joint_pedal_development(
            output=Path(".local-runs/safe-v19.json"),
            provider_output=aliased_provider,
        )

    with pytest.raises(ValueError, match="write-once V18"):
        development._publish(
            {"candidate": None},
            output=aliased_v18,
            provider=None,
            provider_output=Path(".local-runs/safe-v19.provider.pt"),
        )
    with pytest.raises(ValueError, match="write-once V18"):
        development._publish(
            {"candidate": None},
            output=Path(".local-runs/safe-v19.json"),
            provider=None,
            provider_output=aliased_provider,
        )


def test_outer_gate_matrix_passes_only_with_all_preregistered_controls() -> None:
    result = development._evaluate_outer_gates(
        _passing_arms(), coordinate_geometry={"passed": True}
    )
    assert result["passed"] is True
    assert result["training_checkpoint_passed"] is True
    assert result["pca_training_checkpoint_diagnostic_passed"] is True
    assert all(result["checks"].values())
    assert result["thresholds"] == development._OUTER_THRESHOLDS


@pytest.mark.parametrize(
    "case,expected_check",
    (
        (
            "parent_abs",
            "fisher_absolute_delta_NLL_relative_improvement_vs_parent_at_least_5pct",
        ),
        ("parent_kl", "fisher_KL_relative_improvement_vs_parent_at_least_5pct"),
        ("parent_top1", "fisher_aggregate_top1_gain_vs_parent_at_least_0p02"),
        (
            "parent_family_wins",
            "fisher_family_absolute_delta_NLL_wins_vs_parent_at_least_6",
        ),
        (
            "parent_worst_family",
            "fisher_worst_family_relative_regression_vs_parent_at_most_2pct",
        ),
        (
            "support",
            "fisher_support_and_graph_core_no_aggregate_regression_vs_parent",
        ),
        (
            "core",
            "fisher_support_and_graph_core_no_aggregate_regression_vs_parent",
        ),
        (
            "start",
            "fisher_absolute_delta_NLL_relative_improvement_vs_v18_start_at_least_1pct",
        ),
        (
            "intercept",
            "fisher_KL_not_higher_than_intercept",
        ),
        (
            "unit",
            "fisher_absolute_delta_NLL_strictly_below_unit",
        ),
        (
            "pca",
            "fisher_absolute_delta_NLL_strictly_below_PCA",
        ),
        ("absolute", "fisher_required_absolute_ledgers_pass"),
        ("trust", "every_joint_fold_passes_pointwise_trust"),
        (
            "variation",
            "every_fisher_conditional_fold_has_nonconstant_pedal",
        ),
        (
            "fisher_fit",
            "fisher_training_every_fold_strictly_improved_checkpoint_zero",
        ),
        (
            "fisher_macro_fit",
            "fisher_eight_fold_macro_training_objective_improvement_at_least_1pct",
        ),
    ),
)
def test_outer_gate_matrix_fails_closed_per_threshold(
    case: str, expected_check: str
) -> None:
    arms = _passing_arms()
    fisher_fidelity = arms[development.FISHER_CONDITIONAL_ID]["fidelity"]
    if case == "parent_abs":
        _set_macro(
            arms,
            development.FISHER_CONDITIONAL_ID,
            "absolute_delta_nll_per_token",
            0.96,
        )
    elif case == "parent_kl":
        _set_macro(
            arms,
            development.FISHER_CONDITIONAL_ID,
            "source_to_candidate_kl_per_token",
            0.96,
        )
    elif case == "parent_top1":
        fisher_fidelity["ordinary"]["aggregate"]["top1_agreement_to_source"] = 0.619  # type: ignore[index]
    elif case == "parent_family_wins":
        _set_family_values(
            arms,
            development.FISHER_CONDITIONAL_ID,
            (1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.7, 0.7),
        )
    elif case == "parent_worst_family":
        _set_family_values(
            arms,
            development.FISHER_CONDITIONAL_ID,
            (1.03, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7),
        )
    elif case == "support":
        fisher_fidelity["complete_h4_support"]["aggregate"]["delta_nll_per_token"] = 1.01  # type: ignore[index]
    elif case == "core":
        fisher_fidelity["graph_core"]["aggregate"]["source_to_candidate_kl_per_token"] = 1.01  # type: ignore[index]
    elif case == "start":
        _set_macro(
            arms,
            development.FISHER_CONDITIONAL_ID,
            "absolute_delta_nll_per_token",
            0.895,
        )
    elif case == "intercept":
        _set_macro(
            arms,
            development.FISHER_CONDITIONAL_ID,
            "source_to_candidate_kl_per_token",
            0.81,
        )
    elif case == "unit":
        _set_macro(
            arms,
            development.FISHER_CONDITIONAL_ID,
            "absolute_delta_nll_per_token",
            0.85,
        )
    elif case == "pca":
        _set_macro(
            arms,
            development.PCA_CONDITIONAL_ID,
            "absolute_delta_nll_per_token",
            0.69,
        )
    elif case == "absolute":
        fisher_fidelity["graph_core"]["gates"]["passed"] = False  # type: ignore[index]
    elif case == "trust":
        arms[development.FISHER_UNIT_ID]["held_runtime_diagnostics"][  # type: ignore[index]
            _FAMILIES[0]
        ]["pointwise_trust_passed"] = False
    elif case == "variation":
        arms[development.FISHER_CONDITIONAL_ID]["held_runtime_diagnostics"][  # type: ignore[index]
            _FAMILIES[0]
        ]["pedal_nonconstant"] = False
    elif case == "fisher_fit":
        arms[development.FISHER_CONDITIONAL_ID]["optimization_receipts"][  # type: ignore[index]
            _FAMILIES[0]
        ]["fit_qualification"]["passed"] = False
    elif case == "fisher_macro_fit":
        for receipt in arms[development.FISHER_CONDITIONAL_ID][  # type: ignore[index]
            "optimization_receipts"
        ].values():
            receipt["checkpoint_scores"] = (1.0, 0.995, 1.0, 1.0, 1.0)
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    result = development._evaluate_outer_gates(
        arms, coordinate_geometry={"passed": True}
    )
    assert result["checks"][expected_check] is False
    if case == "fisher_macro_fit":
        assert result["checks"][
            "fisher_training_every_fold_strictly_improved_checkpoint_zero"
        ] is True
    elif case == "fisher_fit":
        assert result["checks"][
            "fisher_eight_fold_macro_training_objective_improvement_at_least_1pct"
        ] is True
    assert result["passed"] is False


def test_pca_fit_qualification_is_a_reported_diagnostic_not_a_fisher_gate() -> None:
    arms = _passing_arms()
    arms[development.PCA_CONDITIONAL_ID]["optimization_receipts"][  # type: ignore[index]
        _FAMILIES[0]
    ]["capability_receipt"]["held_family_capability_excluded"] = False
    result = development._evaluate_outer_gates(
        arms, coordinate_geometry={"passed": True}
    )
    assert result["pca_training_checkpoint_diagnostic_passed"] is False
    assert result["passed"] is True


def test_pca_pedal_variation_is_a_reported_diagnostic_not_a_fisher_gate() -> None:
    arms = _passing_arms()
    arms[development.PCA_CONDITIONAL_ID]["held_runtime_diagnostics"][  # type: ignore[index]
        _FAMILIES[0]
    ]["pedal_nonconstant"] = False
    result = development._evaluate_outer_gates(
        arms, coordinate_geometry={"passed": True}
    )
    assert result["pca_held_pedal_variation_diagnostic_passed"] is False
    assert result["passed"] is True


def test_outer_gate_rejects_failed_inherited_geometry() -> None:
    with pytest.raises(ValueError, match="coordinate geometry"):
        development._evaluate_outer_gates(
            _passing_arms(), coordinate_geometry={"passed": False}
        )


def test_report_candidate_is_conditional_on_outer_and_full_refit() -> None:
    arms = _passing_arms()
    full = _full_refit_qualification(passed=True)
    candidate = {
        "arm_id": development.FISHER_CONDITIONAL_ID,
        "provider_artifact_sha256": _SHA,
    }
    report = development.build_finite_joint_pedal_development_report(
        **_report_kwargs(arms, full_refit=full, candidate=candidate)
    )
    assert report["passed"] is True
    assert report["candidate_readiness"] is True
    assert report["candidate"] == candidate
    assert report["serving_authorized"] is False
    assert report["classification"] == (
        "finite_joint_pedal_oof_candidate_ready_for_fresh_protocol"
    )

    with pytest.raises(ValueError, match="candidate does not match"):
        development.build_finite_joint_pedal_development_report(
            **_report_kwargs(arms, full_refit=full, candidate=None)
        )
    wrong = {**candidate, "provider_artifact_sha256": "b" * 64}
    with pytest.raises(ValueError, match="binding differs"):
        development.build_finite_joint_pedal_development_report(
            **_report_kwargs(arms, full_refit=full, candidate=wrong)
        )


def test_report_fails_closed_without_outer_qualification_and_stays_scalar_only() -> None:
    failed = _passing_arms()
    _set_macro(
        failed,
        development.FISHER_CONDITIONAL_ID,
        "absolute_delta_nll_per_token",
        0.99,
    )
    report = development.build_finite_joint_pedal_development_report(
        **_report_kwargs(failed, full_refit=None, candidate=None)
    )
    assert report["passed"] is False
    assert report["candidate"] is None
    assert report["success_authorizes"] == "no_candidate_and_no_fresh_protocol"
    assert report["classification"] == "finite_joint_pedal_outer_fidelity_insufficient"

    full = _full_refit_qualification(passed=True)
    candidate = {
        "arm_id": development.FISHER_CONDITIONAL_ID,
        "provider_artifact_sha256": _SHA,
    }
    kwargs = _report_kwargs(_passing_arms(), full_refit=full, candidate=candidate)
    kwargs["panel"] = {"prompt_count": 16, "raw_logits": torch.zeros(2)}
    with pytest.raises(TypeError, match="non-scalar data Tensor"):
        development.build_finite_joint_pedal_development_report(**kwargs)


def test_report_rejects_full_refit_when_outer_did_not_authorize_it() -> None:
    failed = _passing_arms()
    failed[development.FISHER_CONDITIONAL_ID]["optimization_receipts"][  # type: ignore[index]
        _FAMILIES[0]
    ]["fit_qualification"]["passed"] = False
    with pytest.raises(ValueError, match="full refit does not match outer gates"):
        development.build_finite_joint_pedal_development_report(
            **_report_kwargs(
                failed,
                full_refit=_full_refit_qualification(passed=False),
                candidate=None,
            )
        )


def test_full_refit_qualification_recomputes_true_false_and_binds_three_hashes() -> None:
    passed = _full_refit_qualification(passed=True)
    assert development._validate_full_refit_qualification(passed) == passed

    failed = _full_refit_qualification(passed=False)
    assert development._validate_full_refit_qualification(failed) == failed

    drifted = _full_refit_qualification(passed=True)
    drifted["passed"] = False
    with pytest.raises(ValueError, match="decision drifted"):
        development._validate_full_refit_qualification(drifted)

    for key in (
        "provider_artifact_sha256",
        "parent_provider_artifact_sha256",
        "start_provider_artifact_sha256",
    ):
        malformed = _full_refit_qualification(passed=True)
        malformed[key] = "not-a-sha"
        with pytest.raises((TypeError, ValueError), match="SHA-256"):
            development._validate_full_refit_qualification(malformed)
