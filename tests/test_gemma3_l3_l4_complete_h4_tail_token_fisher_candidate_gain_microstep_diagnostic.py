from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_microstep_diagnostic as diagnostic,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
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


def _role_traces() -> tuple[SimpleNamespace, ...]:
    fit_counts = (34, 60, 37, 65, 34, 65, 42, 61)
    tune_counts = (50, 50, 50, 50, 50, 50, 50, 55)
    rows: list[SimpleNamespace] = []
    for index, (fit_count, tune_count) in enumerate(zip(fit_counts, tune_counts)):
        first_fit = index % 2 == 0
        for suffix, count in (
            ("a", fit_count if first_fit else tune_count),
            ("b", tune_count if first_fit else fit_count),
        ):
            rows.append(
                SimpleNamespace(
                    example_id=f"p{index}{suffix}",
                    family_id=f"f{index}",
                    endpoint=SimpleNamespace(
                        supervised_tokens=count,
                        residual_rows=torch.zeros((index + 2, 640)),
                    ),
                )
            )
    return tuple(rows)


def _roles(traces: tuple[SimpleNamespace, ...]) -> object:
    return diagnostic._checkerboard_prompt_roles(traces)


def _fits() -> dict[str, SimpleNamespace]:
    return {
        f"f{index}": SimpleNamespace(
            held_family_id=f"f{index}",
            artifact_sha256=f"{1000 + index:064x}",
        )
        for index in range(8)
    }


def test_v5_provider_is_exact_sign_stage_bound_and_single_use() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base, dtype=torch.float64)
    common = {
        "fold_artifact_sha256": "f" * 64,
        "ordered_directions_sha256": "d" * 64,
        "gains": torch.ones(64),
        "refit_artifact_sha256": "e" * 64,
        "example_id": "example",
        "family_id": "family",
        "model_inputs_sha256": "a" * 64,
        "bridge_binding_sha256": "c" * 64,
        "prefix_artifact_sha256": prefix.artifact_sha256,
        "base_h4": base,
        "support_mask": support,
        "correction": correction,
    }
    provider = diagnostic._AuthenticatedCandidateGainMicrostepProvider(
        stage="tune",
        candidate_variant="plus_epsilon",
        sign=1,
        selection_artifact_sha256=None,
        **common,
    )
    assert provider.sign == 1
    assert provider.step_hex == diagnostic.SYMMETRIC_GAIN_MICROSTEP_EPSILON.hex()
    assert torch.equal(provider.correction(prefix, base), correction)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)
    with pytest.raises(ValueError, match="exact integer"):
        diagnostic._AuthenticatedCandidateGainMicrostepProvider(
            stage="tune",
            candidate_variant="plus_epsilon",
            sign=True,
            selection_artifact_sha256=None,
            **common,
        )
    with pytest.raises(ValueError, match="tune provider semantics"):
        diagnostic._AuthenticatedCandidateGainMicrostepProvider(
            stage="tune",
            candidate_variant="minus_epsilon_diagnostic",
            sign=1,
            selection_artifact_sha256=None,
            **common,
        )
    with pytest.raises(ValueError, match="final provider semantics"):
        diagnostic._AuthenticatedCandidateGainMicrostepProvider(
            stage="final",
            candidate_variant="selected_plus_epsilon",
            sign=1,
            selection_artifact_sha256=None,
            **common,
        )


def test_mocked_tune_executes_plus_then_minus_without_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _role_traces()
    roles = _roles(traces)
    fits = _fits()
    refits = {
        family: SimpleNamespace(
            held_family_id=family,
            artifact_sha256=f"{2000 + index:064x}",
            mean_no_op=False,
            mean_proposed_gains_tensor=lambda: torch.full(
                (64,), 0.9, dtype=torch.float64
            ),
        )
        for index, family in enumerate(sorted(fits))
    }
    teacher = torch.zeros((1, 2, 3), dtype=torch.float64)
    grid = torch.tensor([[0, 1]], dtype=torch.int64)
    directions = torch.eye(64, 640, dtype=torch.float64)
    monkeypatch.setattr(
        diagnostic,
        "_fresh_native_teacher",
        lambda **kwargs: (
            {"input_ids": torch.zeros((1, 2), dtype=torch.int64)},
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            teacher,
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "_endpoint_indices",
        lambda trace, indices, targets: (indices, targets, grid),
    )
    monkeypatch.setattr(diagnostic, "_ordered_k64", lambda fit: directions)
    monkeypatch.setattr(
        diagnostic,
        "_microstep_gains",
        lambda refit, sign: torch.full(
            (64,), 0.9 if sign == 1 else 1.1, dtype=torch.float64
        ),
    )
    by_id = {trace.example_id: trace for trace in traces}
    unit: dict[tuple[str, str], dict[str, object]] = {}
    for example_id in roles.tune_example_ids:
        trace = by_id[example_id]
        for held, fit in fits.items():
            if held == trace.family_id:
                continue
            unit[(held, example_id)] = {
                "example_id": example_id,
                "family_id": trace.family_id,
                "held_family_id": held,
                "fold_artifact_sha256": fit.artifact_sha256,
                "ordered_directions_sha256": diagnostic._runtime_tensor_sha256(
                    directions
                ),
                "teacher_logits_sha256": diagnostic._runtime_tensor_sha256(teacher),
                "endpoint_supervised_grid_sha256": diagnostic._runtime_tensor_sha256(
                    grid
                ),
                "endpoint_supervised_token_count": 1,
                "receipt_sha256": "a" * 64,
                "dual_tune_example_artifact_sha256": "b" * 64,
                "token_teacher_kl_sha256": "c" * 64,
                "mean_teacher_kl": 1.0,
            }
    calls: list[tuple[str, str, int, str]] = []

    def execute(**kwargs: object):
        trace = kwargs["trace"]
        fit = kwargs["fit"]
        sign = int(kwargs["sign"])
        variant = str(kwargs["candidate_variant"])
        calls.append((trace.example_id, fit.held_family_id, sign, variant))
        gains = kwargs["gains"]
        return (
            torch.full(
                (trace.endpoint.supervised_tokens,),
                0.9 if sign == 1 else 1.1,
                dtype=torch.float64,
            ),
            SimpleNamespace(artifact_sha256="d" * 64),
            SimpleNamespace(
                gains_sha256=diagnostic._runtime_tensor_sha256(gains),
                artifact_sha256="e" * 64,
                model_inputs_sha256="1" * 64,
                bridge_binding_sha256="2" * 64,
                prefix_artifact_sha256="3" * 64,
                base_h4_sha256="4" * 64,
                support_mask_sha256="5" * 64,
                correction_sha256="6" * 64,
            ),
            torch.zeros((1, 640), dtype=torch.float64),
        )

    selected: dict[str, tuple[object, ...]] = {}

    def select(refit: object, examples: object):
        values = tuple(examples)
        assert len(values) == 7
        selected[refit.held_family_id] = values
        return SimpleNamespace(artifact_sha256=f"{3000 + len(selected):064x}")

    monkeypatch.setattr(
        diagnostic, "_execute_candidate_teacher_kl_forward", execute
    )
    monkeypatch.setattr(
        diagnostic, "select_candidate_conditioned_k64_symmetric_microstep", select
    )
    selections, receipts, resources = (
        diagnostic._collect_symmetric_microstep_tune_selections(
            context=object(),
            traces=traces,
            basis=torch.zeros((320, 640), dtype=torch.float64),
            fits=fits,
            refits=refits,
            roles=roles,
            unit_baselines=unit,
        )
    )
    assert len(calls) == len(receipts) == 112
    assert set(selections) == set(fits)
    for index in range(0, len(calls), 2):
        cell = calls[index : index + 2]
        assert [(row[2], row[3]) for row in cell] == [
            (1, "plus_epsilon"),
            (-1, "minus_epsilon_diagnostic"),
        ]
        assert len({row[:2] for row in cell}) == 1
    assert resources["tune_native_forward_count"] == 8
    assert resources["tune_candidate_forward_count"] == 112
    assert resources["unit_execution_count_per_prompt_fold"] == 0
    assert resources["pinned_v4_unit_baseline_reuse_count"] == 56


def _final_traces() -> tuple[SimpleNamespace, ...]:
    traces: list[SimpleNamespace] = []
    for index in range(16):
        logits = torch.zeros((1, 3, 4), dtype=torch.float64)
        traces.append(
            SimpleNamespace(
                example=SimpleNamespace(
                    example_id=f"e{index:02d}", family_id=f"f{index // 2}"
                ),
                example_id=f"e{index:02d}",
                family_id=f"f{index // 2}",
                endpoint=SimpleNamespace(
                    residual_rows=torch.zeros((1, 640), dtype=torch.float64),
                    compensation_target=torch.zeros(1, dtype=torch.float64),
                ),
                prefix=_prefix(),
                support_indices=torch.tensor([1], dtype=torch.int64),
                selected_by_ledger={
                    ledger: torch.tensor([0], dtype=torch.int64)
                    for ledger in diagnostic._LEDGERS
                },
                base_h4=torch.zeros((1, 3, 640), dtype=torch.float32),
                native_h4=torch.zeros((1, 3, 640), dtype=torch.float32),
                native_logits_sha256=diagnostic._runtime_tensor_sha256(logits),
                native_token_nll=torch.ones(1, dtype=torch.float64),
                d320_token_nll=torch.ones(1, dtype=torch.float64),
            )
        )
    return tuple(traces)


def test_mocked_final_executes_one_selection_bound_candidate_per_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _final_traces()
    directions = torch.eye(64, 640, dtype=torch.float64)
    fits = {
        f"f{index}": SimpleNamespace(
            held_family_id=f"f{index}",
            artifact_sha256=f"{4000 + index:064x}",
            ordered_basis_rows=lambda: directions,
        )
        for index in range(8)
    }
    refits = {
        family: SimpleNamespace(
            held_family_id=family,
            artifact_sha256=f"{5000 + index:064x}",
        )
        for index, family in enumerate(sorted(fits))
    }
    selected_gains = torch.full((64,), 0.9, dtype=torch.float64)
    selections = {
        family: SimpleNamespace(
            held_family_id=family,
            refit_artifact_sha256=refits[family].artifact_sha256,
            artifact_sha256=f"{6000 + index:064x}",
            selected_step=diagnostic.SYMMETRIC_GAIN_MICROSTEP_EPSILON,
            selected_arm="plus_epsilon",
            selected_gains_tensor=lambda: selected_gains,
            metadata=lambda family=family: {"held_family_id": family},
        )
        for index, family in enumerate(sorted(fits))
    }
    teacher = torch.zeros((1, 3, 4), dtype=torch.float64)
    grid = torch.tensor([[0, 1]], dtype=torch.int64)
    scores = torch.zeros((1, 64), dtype=torch.float64)
    monkeypatch.setattr(
        diagnostic,
        "_fresh_native_teacher",
        lambda **kwargs: (
            {"input_ids": torch.zeros((1, 3), dtype=torch.int64)},
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            teacher,
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "_endpoint_indices",
        lambda trace, indices, targets: (indices, targets, grid),
    )
    monkeypatch.setattr(diagnostic, "_ordered_k64", lambda fit: directions)
    monkeypatch.setattr(
        diagnostic, "complete_h4_tail_gate_scores", lambda *args: scores
    )
    unit = {
        trace.example_id: {
            "example_id": trace.example_id,
            "family_id": trace.family_id,
            "fold_artifact_sha256": fits[trace.family_id].artifact_sha256,
            "ordered_directions_sha256": diagnostic._runtime_tensor_sha256(
                directions
            ),
            "teacher_logits_sha256": diagnostic._runtime_tensor_sha256(teacher),
            "endpoint_supervised_grid_sha256": diagnostic._runtime_tensor_sha256(
                grid
            ),
            "endpoint_supervised_token_count": 1,
            "token_score_matrix_sha256": diagnostic._runtime_tensor_sha256(scores),
            "arm": "unit_k64",
            "candidate_variant": "unit",
            "observation_sha256": "a" * 64,
            "token_teacher_kl_sha256": "b" * 64,
            "complete_h4_support_mean_teacher_kl": 1.0,
        }
        for trace in traces
    }
    calls: list[tuple[str, torch.Tensor]] = []

    def execute(**kwargs: object):
        trace = kwargs["trace"]
        gains = kwargs["gains"]
        calls.append((trace.example_id, gains.clone()))
        return (
            torch.tensor([0.9], dtype=torch.float64),
            SimpleNamespace(
                logits=teacher.clone(),
                candidate_h4=torch.zeros((1, 3, 640), dtype=torch.float32),
                artifact_sha256="c" * 64,
            ),
            SimpleNamespace(
                ordered_directions_sha256=diagnostic._runtime_tensor_sha256(
                    directions
                ),
                gains_sha256=diagnostic._runtime_tensor_sha256(gains),
                artifact_sha256="d" * 64,
                model_inputs_sha256="1" * 64,
                bridge_binding_sha256="2" * 64,
                prefix_artifact_sha256="3" * 64,
                base_h4_sha256="4" * 64,
                support_mask_sha256="5" * 64,
                correction_sha256="6" * 64,
            ),
            torch.zeros((1, 640), dtype=torch.float64),
        )

    class Accumulator:
        def __init__(self, manifest: object, *, gates: object) -> None:
            self.count = 0

        def add(self, example: object) -> None:
            self.count += 1

        def finalize(self) -> dict[str, object]:
            return {"gates": {"passed": True}, "example_count": self.count}

    monkeypatch.setattr(
        diagnostic, "_execute_candidate_teacher_kl_forward", execute
    )
    monkeypatch.setattr(
        diagnostic, "SourceAuthoritativeShadowFidelityAccumulator", Accumulator
    )
    monkeypatch.setattr(
        diagnostic.ladder,
        "_geometry_with_examples",
        lambda traces, rows, *, candidate_semantics: {
            "semantics": candidate_semantics,
            "gates": {"passed": True},
            "example_count": len(rows),
        },
    )
    observations, behavior, geometry, resources = (
        diagnostic._final_selected_observations(
            context=object(),
            traces=traces,
            basis=torch.zeros((320, 640), dtype=torch.float64),
            fits=fits,
            refits=refits,
            selections=selections,
            v4_unit_observations=unit,
        )
    )
    assert len(calls) == len(observations) == 16
    assert all(torch.equal(gains, selected_gains) for _example, gains in calls)
    assert all(row["selected_arm"] == "plus_epsilon" for row in observations)
    assert resources["final_native_forward_count"] == 16
    assert resources["final_candidate_forward_count"] == 16
    assert resources["separate_pinned_v4_unit_final_baseline_reexecution_count"] == 0
    assert set(behavior) == set(diagnostic._LEDGERS)
    assert geometry["example_count"] == 16


def test_selected_unit_stable_replay_fails_closed_on_any_output_mismatch() -> None:
    keys = (
        "gains_sha256",
        "ordered_directions_sha256",
        "teacher_logits_sha256",
        "endpoint_supervised_grid_sha256",
        "endpoint_supervised_token_count",
        "token_teacher_kl_sha256",
        "complete_h4_support_mean_teacher_kl",
        "token_score_matrix_sha256",
        "native_mean_nll",
        "d320_mean_nll",
        "candidate_mean_nll",
        "ordinary_candidate_mean_nll",
        "endpoint_baseline_mse",
        "endpoint_prediction_mse",
        "candidate_h4_bitwise_native",
        "candidate_logits_bitwise_native",
        "full_tail_reconstruction_max_abs_error",
        "exact_residual_provider_used",
        "executed_correction_rows_sha256",
    )
    unit = {key: index for index, key in enumerate(keys)}
    assert diagnostic._authenticate_selected_unit_stable_replay(unit, unit)
    candidate = dict(unit)
    candidate["token_teacher_kl_sha256"] = "different"
    with pytest.raises(RuntimeError, match="token_teacher_kl_sha256"):
        diagnostic._authenticate_selected_unit_stable_replay(candidate, unit)


def test_v5_resource_ledger_closes_at_264_forwards_and_494_backwards() -> None:
    traces = _role_traces()
    roles = _roles(traces)
    endpoint = {
        "base_forward_count": 16,
        "native_forward_count": 16,
        "endpoint_token_vjp_forward_count": 16,
        "endpoint_token_vjp_backward_call_count": 109,
    }
    gradient = {
        "gradient_native_forward_count": 8,
        "gradient_candidate_vjp_forward_count": 56,
        "gradient_candidate_vjp_backward_call_count": 385,
        "gradient_prompt_fold_count": 56,
        "shared_candidate_gradient_bank_count": 56,
        "additional_backward_calls_for_second_direction": 0,
    }
    tune = {
        "tune_native_forward_count": 8,
        "tune_candidate_forward_count": 112,
        "tune_prompt_fold_count": 56,
        "tune_prompt_fold_candidate_count": 112,
        "tune_candidate_count_per_prompt_fold": 2,
        "positive_microstep_execution_count_per_prompt_fold": 1,
        "negative_microstep_execution_count_per_prompt_fold": 1,
        "unit_execution_count_per_prompt_fold": 0,
        "pinned_v4_unit_baseline_reuse_count": 56,
    }
    final = {
        "final_native_forward_count": 16,
        "final_candidate_forward_count": 16,
        "final_observation_count": 16,
        "final_arm_count": 1,
        "separate_pinned_v4_unit_final_baseline_reexecution_count": 0,
        "selected_candidate_execution_count": 16,
    }
    result = diagnostic._resource_accounting(
        traces=traces,
        roles=roles,
        endpoint_resources=endpoint,
        gradient_resources=gradient,
        tune_resources=tune,
        final_resources=final,
    )
    assert result["total_model_forward_count"] == 264
    assert result["total_backward_call_count"] == 494
    assert result["candidate_k64_execution_count"] == 184
    assert result["pinned_v4_unit_tune_model_forward_count"] == 0
    assert result["pinned_v4_unit_final_baseline_model_forward_count"] == 0


def test_phase_order_authenticates_v4_before_any_microstep_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    roles = diagnostic._PromptRoles(("fit",), ("tune",), 398, 405, "a" * 64)
    refits = {"held": object()}
    selections = {"held": object()}
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_parent_recollection",
        lambda **kwargs: order.append("parent") or "b" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_static_unit_k64_replay",
        lambda **kwargs: order.append("static_unit") or "c" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_checkerboard_prompt_roles",
        lambda traces: order.append("roles") or roles,
    )
    monkeypatch.setattr(
        diagnostic.v4diag,
        "_collect_candidate_gradient_refits",
        lambda **kwargs: order.append("gradient") or (refits, {}, tuple(), {}),
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_live_v4_refits_and_gradients",
        lambda **kwargs: order.append("v4_refit_auth")
        or {"artifact_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_v4_unit_tune_baselines",
        lambda **kwargs: order.append("v4_unit_tune_auth")
        or ({}, {"artifact_sha256": "e" * 64}),
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_v4_unit_final_baseline",
        lambda **kwargs: order.append("v4_unit_final_auth")
        or ({}, {}, {}, {"artifact_sha256": "f" * 64}),
    )

    def tune(**kwargs: object):
        assert order[-1] == "v4_unit_final_auth"
        order.append("microstep_tune")
        return selections, tuple(), {}

    def final(**kwargs: object):
        assert order[-1] == "microstep_tune"
        order.append("held_final")
        return [], {}, {}, {}

    monkeypatch.setattr(
        diagnostic, "_collect_symmetric_microstep_tune_selections", tune
    )
    monkeypatch.setattr(diagnostic, "_final_selected_observations", final)
    monkeypatch.setattr(
        diagnostic,
        "_finite_observation_set_sha256",
        lambda observations: order.append("observation_auth") or "1" * 64,
    )
    result = diagnostic._execute_candidate_phases(
        context=object(),
        parent={},
        v4_report={},
        traces=(),
        endpoint_resources={},
        basis=torch.zeros((1, 1)),
        fits={},
    )
    assert order == [
        "parent",
        "static_unit",
        "roles",
        "gradient",
        "v4_refit_auth",
        "v4_unit_tune_auth",
        "v4_unit_final_auth",
        "microstep_tune",
        "held_final",
        "observation_auth",
    ]
    assert result.selections is selections


def test_tune_receipt_set_requires_both_signs_in_all_56_cells() -> None:
    receipts: list[dict[str, object]] = []
    for index in range(56):
        for sign in (-1, 1):
            row: dict[str, object] = {
                "held_family_id": f"h{index}",
                "family_id": f"f{index}",
                "example_id": f"e{index}",
                "sign": sign,
            }
            row["receipt_sha256"] = diagnostic.token_v1._domain_sha256(
                row, domain=diagnostic._TUNE_RECEIPT_DOMAIN
            )
            receipts.append(row)
    assert len(diagnostic._receipt_set_sha256(receipts)) == 64
    malformed = list(receipts)
    malformed[-1] = dict(malformed[-2])
    with pytest.raises(ValueError, match="duplicate|coverage"):
        diagnostic._receipt_set_sha256(malformed)


def _outcome_selections(
    *, positive: int, no_op: int = 0
) -> dict[str, SimpleNamespace]:
    return {
        f"f{index}": SimpleNamespace(
            selected_step=(
                diagnostic.SYMMETRIC_GAIN_MICROSTEP_EPSILON
                if index < positive
                else 0.0
            ),
            refit_no_op_or_zero_delta=index < no_op,
        )
        for index in range(8)
    }


def _outcome_top1(*, safe: bool) -> dict[str, dict[str, bool]]:
    return {
        ledger: {
            "aggregate_at_least_point_90": safe,
            "family_macro_at_least_point_90": safe,
            "aggregate_no_material_regression_vs_unit": True,
            "family_macro_no_material_regression_vs_unit": True,
        }
        for ledger in ("ordinary", "complete_h4_support", "graph_core")
    }


@pytest.mark.parametrize(
    ("approximate", "positive", "no_op", "improved", "relative", "top1_safe", "expected"),
    (
        (
            True,
            8,
            0,
            8,
            0.03,
            True,
            "symmetric_microstep_mean_KL_OPG_supported_same_a",
        ),
        (
            False,
            5,
            0,
            8,
            0.03,
            True,
            "one_over_64_symmetric_microstep_fit_to_tune_transfer_failed_same_a",
        ),
        (
            False,
            8,
            0,
            4,
            -0.01,
            True,
            "symmetric_microstep_static_cross_family_transfer_blocker_same_a",
        ),
        (
            False,
            8,
            0,
            8,
            0.03,
            False,
            "symmetric_microstep_behavioral_top1_fidelity_failed_same_a",
        ),
        (
            False,
            8,
            0,
            8,
            0.01,
            True,
            "symmetric_microstep_transferred_but_below_useful_effect_same_a",
        ),
        (
            False,
            0,
            8,
            0,
            0.0,
            True,
            "one_over_64_symmetric_microstep_all_refits_structural_no_op_same_a",
        ),
    ),
)
def test_outcome_partition_is_exhaustive_and_fidelity_aware(
    approximate: bool,
    positive: int,
    no_op: int,
    improved: int,
    relative: float,
    top1_safe: bool,
    expected: str,
) -> None:
    result = diagnostic._outcome_matrix(
        approximate={"passed": approximate},
        comparison={
            "held_family_improvement_count": improved,
            "all_held_families_within_five_percent_plus_1e_minus_8": True,
            "family_macro_relative_improvement": relative,
        },
        top1=_outcome_top1(safe=top1_safe),
        selections=_outcome_selections(positive=positive, no_op=no_op),
    )
    assert result["outcome"] == expected


def test_cli_pins_v4_epsilon_and_local_write_once_output() -> None:
    parser = diagnostic.build_parser()
    args = parser.parse_args([])
    assert args.expanded_parent_report == diagnostic.DEFAULT_EXPANDED_PARENT_REPORT
    assert args.v4_report == diagnostic.DEFAULT_V4_REPORT
    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert diagnostic.SYMMETRIC_GAIN_MICROSTEP_EPSILON == 1.0 / 64.0
    assert not any(
        action.dest
        in {
            "rank",
            "ranks",
            "epsilon",
            "epsilons",
            "alpha",
            "alphas",
            "objective",
            "preconditioner",
        }
        for action in parser._actions
    )


def test_v4_hash_pins_and_output_name_are_exact() -> None:
    assert diagnostic.V4_REPORT_FILE_SHA256 == (
        "5cad2c81694d9a0122ffe60df1c2bc1222395ddccc661b3e0751e2d78904ed50"
    )
    assert diagnostic.V4_REPORT_SHA256 == (
        "2ace5239314d5497e1c50ef17e3820ab041e99d5c19fe9d8443b0d9505f248c2"
    )
    assert str(diagnostic.DEFAULT_OUTPUT).endswith(
        "candidate-gain-microstep-lofo-a-fit16-dev-v5.json"
    )
