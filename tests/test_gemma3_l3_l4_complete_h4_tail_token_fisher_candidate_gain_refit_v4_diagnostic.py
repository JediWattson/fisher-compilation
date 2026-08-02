from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_v4_diagnostic as diagnostic,
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
    return diagnostic.v3diag._checkerboard_prompt_roles(traces)


def _mock_fits() -> dict[str, SimpleNamespace]:
    return {
        f"f{index}": SimpleNamespace(
            held_family_id=f"f{index}", artifact_sha256=f"{index + 1:064x}"
        )
        for index in range(8)
    }


def _patch_prompt_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fresh(*, context: object, trace: object):
        count = trace.endpoint.supervised_tokens
        return (
            {"input_ids": torch.zeros((1, count), dtype=torch.int64)},
            torch.arange(count, dtype=torch.int64),
            torch.zeros(count, dtype=torch.int64),
            torch.zeros((1, count, 2), dtype=torch.float64),
        )

    def endpoint(trace: object, indices: torch.Tensor, targets: torch.Tensor):
        return (
            indices,
            targets,
            torch.stack((torch.zeros_like(indices), indices), dim=1),
        )

    monkeypatch.setattr(diagnostic, "_fresh_native_teacher", fresh)
    monkeypatch.setattr(diagnostic, "_endpoint_indices", endpoint)


def test_mocked_gradient_collector_shares_one_56_vjp_bank_between_both_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _role_traces()
    roles = _roles(traces)
    fits = _mock_fits()
    _patch_prompt_inputs(monkeypatch)
    calls: list[tuple[str, str]] = []
    v4_banks: dict[str, tuple[object, ...]] = {}
    v3_banks: dict[str, tuple[object, ...]] = {}

    def execute_vjp(**kwargs: object):
        trace = kwargs["trace"]
        fit = kwargs["fit"]
        count = trace.endpoint.supervised_tokens
        calls.append((trace.example_id, fit.held_family_id))
        return (
            torch.ones(count, dtype=torch.float64),
            torch.ones((count, 64), dtype=torch.float64),
            {
                "example_id": trace.example_id,
                "family_id": trace.family_id,
                "held_family_id": fit.held_family_id,
                "maximum_future_gradient_abs": 0.0,
                "future_gradient_nonzero_count": 0,
            },
            (count + 7) // 8,
        )

    def fit_v4(examples: object, **kwargs: object):
        held = str(kwargs["held_family_id"])
        values = tuple(examples)
        v4_banks[held] = values
        return SimpleNamespace(
            held_family_id=held, artifact_sha256=f"{100 + len(v4_banks):064x}"
        )

    def fit_v3(examples: object, **kwargs: object):
        held = str(kwargs["held_family_id"])
        values = tuple(examples)
        v3_banks[held] = values
        return SimpleNamespace(
            held_family_id=held, artifact_sha256=f"{200 + len(v3_banks):064x}"
        )

    monkeypatch.setattr(diagnostic, "_execute_candidate_teacher_kl_vjp", execute_vjp)
    monkeypatch.setattr(
        diagnostic,
        "_ordered_k64",
        lambda fit: torch.eye(64, 640, dtype=torch.float64),
    )
    monkeypatch.setattr(
        diagnostic,
        "_ordered_k64_relevance",
        lambda fit: torch.ones(64, dtype=torch.float64),
    )
    monkeypatch.setattr(
        diagnostic, "fit_candidate_conditioned_k64_mean_kl_gains", fit_v4
    )
    monkeypatch.setattr(
        diagnostic.v3diag, "fit_candidate_conditioned_k64_gains", fit_v3
    )

    refits, replayed, receipts, resources = (
        diagnostic._collect_candidate_gradient_refits(
            context=object(),
            traces=traces,
            basis=torch.zeros((320, 640), dtype=torch.float64),
            fits=fits,
            roles=roles,
        )
    )
    assert len(calls) == 56
    assert len(receipts) == 56
    assert set(refits) == set(replayed) == set(fits)
    assert resources == {
        "gradient_native_forward_count": 8,
        "gradient_candidate_vjp_forward_count": 56,
        "gradient_candidate_vjp_backward_call_count": 385,
        "gradient_prompt_fold_count": 56,
        "shared_candidate_gradient_bank_count": 56,
        "additional_backward_calls_for_second_direction": 0,
    }
    for held in sorted(fits):
        assert len(v4_banks[held]) == len(v3_banks[held]) == 7
        for v4, v3 in zip(v4_banks[held], v3_banks[held]):
            assert v4.example_id == v3.example_id
            assert v4.family_id == v3.family_id
            assert v4.family_id != held
            assert torch.equal(v4.token_gain_gradients, v3.token_gain_gradients)
            assert torch.equal(v4.token_teacher_kl, v3.token_teacher_kl)


def test_v4_provider_is_variant_stage_bound_and_single_use() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base, dtype=torch.float64)
    common = {
        "fold_artifact_sha256": "f" * 64,
        "ordered_directions_sha256": "d" * 64,
        "model_inputs_sha256": "a" * 64,
        "bridge_binding_sha256": "c" * 64,
        "prefix_artifact_sha256": prefix.artifact_sha256,
        "base_h4": base,
        "support_mask": support,
        "correction": correction,
    }
    provider = diagnostic._AuthenticatedCandidateGainProviderV4(
        stage="tune",
        candidate_variant="unit",
        gains=torch.ones(64),
        mean_refit_artifact_sha256=None,
        selection_artifact_sha256=None,
        step=0.0,
        **common,
    )
    assert torch.equal(provider.correction(prefix, base), correction)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)
    with pytest.raises(ValueError, match="mean tune provider semantics"):
        diagnostic._AuthenticatedCandidateGainProviderV4(
            stage="tune",
            candidate_variant="mean_kl_opg",
            gains=torch.ones(64),
            mean_refit_artifact_sha256="e" * 64,
            selection_artifact_sha256=None,
            step=0.0,
            **common,
        )
    with pytest.raises(ValueError, match="reverse tune provider semantics"):
        diagnostic._AuthenticatedCandidateGainProviderV4(
            stage="tune",
            candidate_variant="reverse_residual_gn",
            gains=torch.ones(64),
            mean_refit_artifact_sha256="e" * 64,
            selection_artifact_sha256=None,
            step=1.0,
            **common,
        )
    with pytest.raises(ValueError, match="final provider semantics"):
        diagnostic._AuthenticatedCandidateGainProviderV4(
            stage="final",
            candidate_variant="mean_kl_opg",
            gains=torch.ones(64),
            mean_refit_artifact_sha256="e" * 64,
            selection_artifact_sha256=None,
            step=None,
            **common,
        )


def test_mocked_tune_collector_executes_shared_unit_plus_seven_steps_per_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _role_traces()
    roles = _roles(traces)
    fits = _mock_fits()
    refits = {
        family: SimpleNamespace(
            held_family_id=family,
            artifact_sha256=f"{300 + index:064x}",
        )
        for index, family in enumerate(sorted(fits))
    }
    _patch_prompt_inputs(monkeypatch)
    calls: list[tuple[str, str, str, float]] = []
    selected_examples: dict[str, tuple[object, ...]] = {}

    monkeypatch.setattr(
        diagnostic,
        "_mean_gains",
        lambda refit, alpha: torch.ones(64, dtype=torch.float64),
    )
    monkeypatch.setattr(
        diagnostic,
        "_reverse_residual",
        lambda refit, beta: torch.ones(64, dtype=torch.float64),
    )

    def execute_forward(**kwargs: object):
        trace = kwargs["trace"]
        fit = kwargs["fit"]
        variant = str(kwargs["candidate_variant"])
        step = float(kwargs["step"])
        calls.append((trace.example_id, fit.held_family_id, variant, step))
        value = 1.0 + step
        return (
            torch.full(
                (trace.endpoint.supervised_tokens,), value, dtype=torch.float64
            ),
            SimpleNamespace(artifact_sha256="a" * 64),
            SimpleNamespace(
                ordered_directions_sha256="d" * 64,
                gains_sha256="g" * 64,
                artifact_sha256="p" * 64,
            ),
            torch.zeros((1, 640), dtype=torch.float64),
        )

    def select(refit: object, examples: object):
        values = tuple(examples)
        assert len(values) == 7
        assert all(value.family_id != refit.held_family_id for value in values)
        selected_examples[refit.held_family_id] = values
        return SimpleNamespace(
            held_family_id=refit.held_family_id,
            artifact_sha256=f"{400 + len(selected_examples):064x}",
        )

    monkeypatch.setattr(
        diagnostic, "_execute_candidate_teacher_kl_forward", execute_forward
    )
    monkeypatch.setattr(
        diagnostic, "select_candidate_conditioned_k64_dual_tune_steps", select
    )
    selections, receipts, resources = diagnostic._collect_dual_tune_selections(
        context=object(),
        traces=traces,
        basis=torch.zeros((320, 640), dtype=torch.float64),
        fits=fits,
        refits=refits,
        roles=roles,
    )
    expected_manifest = (
        ("unit", 0.0),
        ("mean_kl_opg", 0.125),
        ("mean_kl_opg", 0.25),
        ("mean_kl_opg", 0.5),
        ("mean_kl_opg", 1.0),
        ("reverse_residual_gn", 0.125),
        ("reverse_residual_gn", 0.25),
        ("reverse_residual_gn", 0.5),
    )
    assert len(calls) == len(receipts) == 448
    assert set(selections) == set(fits)
    for index in range(0, len(calls), 8):
        cell = calls[index : index + 8]
        assert tuple((variant, step) for _e, _h, variant, step in cell) == (
            expected_manifest
        )
        assert len({(example, held) for example, held, _v, _s in cell}) == 1
    assert resources == {
        "tune_native_forward_count": 8,
        "tune_candidate_forward_count": 448,
        "tune_prompt_fold_candidate_count": 448,
        "tune_unique_candidate_count_per_prompt_fold": 8,
        "unit_execution_count_per_prompt_fold": 1,
    }


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


def test_mocked_final_collector_executes_exact_16_by_3_frozen_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _final_traces()
    fits = {
        f"f{index}": SimpleNamespace(
            held_family_id=f"f{index}",
            artifact_sha256=f"{5000 + index:064x}",
            ordered_basis_rows=lambda: torch.eye(64, 640, dtype=torch.float64),
        )
        for index in range(8)
    }
    refits = {
        family: SimpleNamespace(
            held_family_id=family,
            artifact_sha256=f"{6000 + index:064x}",
        )
        for index, family in enumerate(sorted(fits))
    }
    selections = {
        family: SimpleNamespace(
            held_family_id=family,
            refit_artifact_sha256=refits[family].artifact_sha256,
            artifact_sha256=f"{7000 + index:064x}",
            selected_mean_alpha=0.25,
            selected_reverse_beta=0.125,
            selected_mean_gains_tensor=lambda: torch.full(
                (64,), 0.9, dtype=torch.float64
            ),
            selected_reverse_gains_tensor=lambda: torch.full(
                (64,), 1.1, dtype=torch.float64
            ),
        )
        for index, family in enumerate(sorted(fits))
    }
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        diagnostic,
        "_fresh_native_teacher",
        lambda **kwargs: (
            {"input_ids": torch.zeros((1, 3), dtype=torch.int64)},
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, 3, 4), dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "_endpoint_indices",
        lambda trace, indices, targets: (
            indices,
            targets,
            torch.tensor([[0, 1]], dtype=torch.int64),
        ),
    )

    def execute(**kwargs: object):
        trace = kwargs["trace"]
        variant = str(kwargs["candidate_variant"])
        calls.append((trace.example_id, variant))
        call_index = len(calls)
        return (
            torch.tensor([1.0], dtype=torch.float64),
            SimpleNamespace(
                logits=torch.zeros((1, 3, 4), dtype=torch.float64),
                candidate_h4=torch.zeros((1, 3, 640), dtype=torch.float32),
                artifact_sha256=f"{8000 + call_index:064x}",
            ),
            SimpleNamespace(
                ordered_directions_sha256="d" * 64,
                gains_sha256=f"{9000 + call_index:064x}",
                artifact_sha256=f"{10000 + call_index:064x}",
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
        diagnostic, "complete_h4_tail_gate_scores", lambda *args: torch.zeros((1, 64))
    )
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
        diagnostic._final_candidate_observations(
            context=object(),
            traces=traces,
            basis=torch.zeros((320, 640), dtype=torch.float64),
            fits=fits,
            refits=refits,
            selections=selections,
        )
    )
    expected = (
        "unit",
        "mean_kl_opg",
        "reverse_residual_gn",
    )
    assert len(calls) == len(observations) == 48
    for index in range(0, 48, 3):
        cell = calls[index : index + 3]
        assert tuple(variant for _example, variant in cell) == expected
        assert len({example for example, _variant in cell}) == 1
    assert resources == {
        "final_native_forward_count": 16,
        "final_candidate_forward_count": 48,
        "final_observation_count": 48,
        "final_arm_count": 3,
    }
    assert set(behavior) == set(geometry) == set(diagnostic._ARMS)
    assert all(row["reverse_control_can_drive_primary"] is False for row in observations)


def test_v4_resource_ledger_closes_at_632_forwards_and_494_backwards() -> None:
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
        "tune_candidate_forward_count": 448,
        "tune_prompt_fold_candidate_count": 448,
        "tune_unique_candidate_count_per_prompt_fold": 8,
        "unit_execution_count_per_prompt_fold": 1,
    }
    final = {
        "final_native_forward_count": 16,
        "final_candidate_forward_count": 48,
        "final_observation_count": 48,
        "final_arm_count": 3,
    }
    result = diagnostic._resource_accounting(
        traces=traces,
        roles=roles,
        endpoint_resources=endpoint,
        gradient_resources=gradient,
        tune_resources=tune,
        final_resources=final,
    )
    assert result["total_model_forward_count"] == 632
    assert result["total_backward_call_count"] == 494
    assert result["candidate_k64_execution_count"] == 552
    assert result["additional_model_or_backward_work_for_second_fit_direction"] == 0
    by_id = {trace.example_id: trace for trace in traces}
    fit_rows = sum(
        by_id[example].endpoint.residual_rows.shape[0]
        for example in roles.fit_example_ids
    )
    tune_rows = sum(
        by_id[example].endpoint.residual_rows.shape[0]
        for example in roles.tune_example_ids
    )
    all_rows = sum(trace.endpoint.residual_rows.shape[0] for trace in traces)
    assert result["candidate_support_row_executions"] == (
        7 * fit_rows + 56 * tune_rows + 3 * all_rows
    )


def test_reverse_only_signal_is_not_hidden_by_all_mean_fold_abstention() -> None:
    selections = {
        f"f{index}": SimpleNamespace(
            selected_mean_alpha=0.0,
            selected_reverse_beta=0.25,
        )
        for index in range(8)
    }
    result = diagnostic._outcome_matrix(
        approximate={"passed": False},
        comparison={
            "reverse_family_macro_relative_improvement": 0.03,
            "reverse_held_family_improvement_count": 6,
        },
        selections=selections,
    )
    assert result["outcome"] == (
        "reverse_residual_sign_control_signal_without_primary_support"
    )
    assert result["mean_positive_alpha_fold_count"] == 0
    assert result["all_mean_folds_abstained"] is True
    assert result["reverse_residual_mechanism_signal_report_only"] is True
    assert result["reverse_control_can_rescue_primary"] is False


def test_phase_orchestrator_authenticates_v3_before_tune_and_freezes_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    roles = diagnostic._PromptRoles(("fit",), ("tune",), 398, 405, "a" * 64)
    refits = {"held": object()}
    replayed = {"held": object()}
    selections = {"held": object()}
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_parent_recollection",
        lambda **kwargs: order.append("recollection") or "b" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_static_unit_k64_replay",
        lambda **kwargs: order.append("static_unit_replay") or "c" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_checkerboard_prompt_roles",
        lambda traces: order.append("roles") or roles,
    )

    def gradient(**kwargs: object):
        order.append("gradient_shared_bank")
        return refits, replayed, tuple(), {}

    def bind(**kwargs: object):
        assert kwargs["refits"] is refits
        assert kwargs["replayed_v3"] is replayed
        order.append("v3_residual_auth")
        return {"artifact_sha256": "d" * 64}

    def tune(**kwargs: object):
        assert kwargs["refits"] is refits
        assert order[-1] == "v3_residual_auth"
        order.append("tune_and_select")
        return selections, tuple(), {}

    def final(**kwargs: object):
        assert kwargs["refits"] is refits
        assert kwargs["selections"] is selections
        assert order[-1] == "tune_and_select"
        order.append("held_final")
        return [], {}, {}, {}

    monkeypatch.setattr(diagnostic, "_collect_candidate_gradient_refits", gradient)
    monkeypatch.setattr(diagnostic, "_authenticate_reproduced_v3_residuals", bind)
    monkeypatch.setattr(diagnostic, "_collect_dual_tune_selections", tune)
    monkeypatch.setattr(diagnostic, "_final_candidate_observations", final)
    monkeypatch.setattr(
        diagnostic,
        "_finite_observation_set_sha256",
        lambda observations: order.append("observation_set_auth") or "e" * 64,
    )
    monkeypatch.setattr(
        diagnostic.v3diag,
        "_authenticate_unit_k64_replay",
        lambda **kwargs: order.append("unit_replay") or ({"tail_rank": 64}, "f" * 64),
    )
    result = diagnostic._execute_candidate_phases(
        context=object(),
        parent={},
        v3_report={},
        traces=(),
        endpoint_resources={},
        basis=torch.zeros((1, 1)),
        fits={},
    )
    assert order == [
        "recollection",
        "static_unit_replay",
        "roles",
        "gradient_shared_bank",
        "v3_residual_auth",
        "tune_and_select",
        "held_final",
        "observation_set_auth",
        "unit_replay",
    ]
    assert result.selections is selections
    assert result.unit_replay_receipt == "f" * 64


def test_cli_has_fixed_dual_grids_v3_binding_and_local_output() -> None:
    parser = diagnostic.build_parser()
    args = parser.parse_args([])
    assert args.expanded_parent_report == diagnostic.DEFAULT_EXPANDED_PARENT_REPORT
    assert args.v3_report == diagnostic.DEFAULT_V3_REPORT
    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert not any(
        action.dest
        in {
            "rank",
            "ranks",
            "alpha",
            "alphas",
            "beta",
            "betas",
            "trust_rms",
            "objective",
            "preconditioner",
        }
        for action in parser._actions
    )
