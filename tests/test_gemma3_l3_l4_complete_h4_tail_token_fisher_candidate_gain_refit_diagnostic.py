from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_candidate_gain_refit_diagnostic as diagnostic,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_diagnostic as token_v1,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_token_fisher_expanded_diagnostic as expanded,
)
from fisher_graph import (
    gemma3_l3_l4_complete_h4_tail_teacher_kl_signed_joint_diagnostic as teacher_kl,
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
    rows = []
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


def test_checkerboard_roles_are_exact_disjoint_and_support_accounted() -> None:
    roles = diagnostic._checkerboard_prompt_roles(_role_traces())
    assert roles.fit_example_ids == (
        "p0a",
        "p1b",
        "p2a",
        "p3b",
        "p4a",
        "p5b",
        "p6a",
        "p7b",
    )
    assert roles.fit_support_tokens == 398
    assert roles.tune_support_tokens == 405
    assert set(roles.fit_example_ids).isdisjoint(roles.tune_example_ids)

    malformed = list(_role_traces())
    malformed[-1] = SimpleNamespace(
        example_id=malformed[-2].example_id,
        family_id=malformed[-2].family_id,
        endpoint=malformed[-1].endpoint,
    )
    with pytest.raises(ValueError, match="two prompts per family"):
        diagnostic._checkerboard_prompt_roles(malformed)


def _mock_fits() -> dict[str, SimpleNamespace]:
    return {
        f"f{index}": SimpleNamespace(
            held_family_id=f"f{index}", artifact_sha256=f"{index + 1:064x}"
        )
        for index in range(8)
    }


def _patch_prompt_stage_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_mocked_gradient_collector_executes_exact_8_by_7_fit_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _role_traces()
    roles = diagnostic._checkerboard_prompt_roles(traces)
    fits = _mock_fits()
    _patch_prompt_stage_inputs(monkeypatch)
    calls: list[tuple[str, str, int]] = []
    fit_examples: dict[str, tuple[object, ...]] = {}

    def execute_vjp(**kwargs: object):
        trace = kwargs["trace"]
        fit = kwargs["fit"]
        count = trace.endpoint.supervised_tokens
        calls.append((trace.example_id, fit.held_family_id, count))
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

    def fake_fit(examples: object, **kwargs: object):
        held = str(kwargs["held_family_id"])
        values = tuple(examples)
        assert len(values) == 7
        assert all(value.family_id != held for value in values)
        assert {value.example_id for value in values} == set(
            roles.fit_example_ids
        ) - {
            next(
                trace.example_id
                for trace in traces
                if trace.family_id == held
                and trace.example_id in roles.fit_example_ids
            )
        }
        fit_examples[held] = values
        return SimpleNamespace(
            held_family_id=held, artifact_sha256=f"{100 + len(fit_examples):064x}"
        )

    monkeypatch.setattr(
        diagnostic, "_execute_candidate_teacher_kl_vjp", execute_vjp
    )
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
        diagnostic, "fit_candidate_conditioned_k64_gains", fake_fit
    )

    refits, receipts, resources = diagnostic._collect_candidate_gradient_refits(
        context=object(),
        traces=traces,
        basis=torch.eye(320, 640, dtype=torch.float64),
        fits=fits,
        roles=roles,
    )
    assert len(calls) == 56
    assert len(receipts) == 56
    assert len(refits) == 8
    assert len(fit_examples) == 8
    assert resources == {
        "gradient_native_forward_count": 8,
        "gradient_candidate_vjp_forward_count": 56,
        "gradient_candidate_vjp_backward_call_count": 385,
        "gradient_prompt_fold_count": 56,
    }
    assert all(example_family != held for _example, held, _count in calls for example_family in [next(trace.family_id for trace in traces if trace.example_id == _example)])
    assert all(
        receipt["family_id"] != receipt["held_family_id"] for receipt in receipts
    )


def test_mocked_tune_collector_executes_exact_8_by_7_by_4_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _role_traces()
    roles = diagnostic._checkerboard_prompt_roles(traces)
    fits = _mock_fits()
    refits = {
        family: SimpleNamespace(
            held_family_id=family,
            artifact_sha256=f"{200 + index:064x}",
            proposed_gains_tensor=lambda: torch.zeros(64, dtype=torch.float64),
        )
        for index, family in enumerate(sorted(fits))
    }
    _patch_prompt_stage_inputs(monkeypatch)
    calls: list[tuple[str, str, float]] = []
    selected_examples: dict[str, tuple[object, ...]] = {}

    def execute_forward(**kwargs: object):
        trace = kwargs["trace"]
        fit = kwargs["fit"]
        gains = kwargs["gains"]
        alpha = 1.0 - float(gains[0])
        calls.append((trace.example_id, fit.held_family_id, alpha))
        call_index = len(calls)
        return (
            torch.full(
                (trace.endpoint.supervised_tokens,),
                1.0 - 0.1 * alpha,
                dtype=torch.float64,
            ),
            SimpleNamespace(artifact_sha256=f"{1000 + call_index:064x}"),
            SimpleNamespace(
                ordered_directions_sha256="d" * 64,
                gains_sha256=diagnostic._runtime_tensor_sha256(gains),
                artifact_sha256=f"{2000 + call_index:064x}",
            ),
            torch.zeros((1, 640), dtype=torch.float64),
        )

    def fake_select(refit: object, examples: object):
        values = tuple(examples)
        held = refit.held_family_id
        assert len(values) == 7
        assert all(value.family_id != held for value in values)
        assert all(len(value.token_teacher_kl_by_alpha) == 4 for value in values)
        selected_examples[held] = values
        return SimpleNamespace(
            held_family_id=held,
            refit_artifact_sha256=refit.artifact_sha256,
            selected_alpha=1.0,
            artifact_sha256=f"{3000 + len(selected_examples):064x}",
        )

    monkeypatch.setattr(
        diagnostic, "_execute_candidate_teacher_kl_forward", execute_forward
    )
    monkeypatch.setattr(
        diagnostic, "select_candidate_conditioned_k64_gain_alpha", fake_select
    )
    selections, receipts, resources = (
        diagnostic._collect_candidate_tune_selections(
            context=object(),
            traces=traces,
            basis=torch.eye(320, 640, dtype=torch.float64),
            fits=fits,
            refits=refits,
            roles=roles,
        )
    )
    assert len(calls) == 224
    assert len(receipts) == 224
    assert len(selections) == 8
    assert len(selected_examples) == 8
    assert resources == {
        "tune_native_forward_count": 8,
        "tune_candidate_forward_count": 224,
        "tune_prompt_fold_alpha_count": 224,
    }
    assert {
        round(alpha, 2) for _example, _held, alpha in calls
    } == set(diagnostic.CANDIDATE_GAIN_ALPHAS)
    assert all(
        receipt["family_id"] != receipt["held_family_id"] for receipt in receipts
    )


def _final_traces() -> tuple[SimpleNamespace, ...]:
    traces = []
    for index in range(16):
        prefix = _prefix()
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
                prefix=prefix,
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


def test_mocked_final_collector_executes_exact_16_by_2_after_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _final_traces()
    fits = {
        f"f{index}": SimpleNamespace(
            held_family_id=f"f{index}",
            artifact_sha256=f"{4000 + index:064x}",
            ordered_basis_rows=lambda: torch.eye(64, 640, dtype=torch.float64),
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
    selections = {
        family: SimpleNamespace(
            held_family_id=family,
            refit_artifact_sha256=refits[family].artifact_sha256,
            artifact_sha256=f"{6000 + index:064x}",
            selected_alpha=0.5,
            selected_gains_tensor=lambda: torch.full(
                (64,), 0.75, dtype=torch.float64
            ),
        )
        for index, family in enumerate(sorted(fits))
    }
    calls: list[tuple[str, str]] = []

    def fresh(*, context: object, trace: object):
        return (
            {"input_ids": torch.zeros((1, 3), dtype=torch.int64)},
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, 3, 4), dtype=torch.float64),
        )

    def endpoint(trace: object, indices: torch.Tensor, targets: torch.Tensor):
        return indices, targets, torch.tensor([[0, 1]], dtype=torch.int64)

    def execute(**kwargs: object):
        trace = kwargs["trace"]
        kind = str(kwargs["gain_kind"])
        calls.append((trace.example_id, kind))
        call_index = len(calls)
        logits = torch.zeros((1, 3, 4), dtype=torch.float64)
        return (
            torch.tensor([1.0 if kind == "unit" else 0.9]),
            SimpleNamespace(
                logits=logits,
                candidate_h4=torch.zeros((1, 3, 640), dtype=torch.float32),
                artifact_sha256=f"{7000 + call_index:064x}",
            ),
            SimpleNamespace(
                ordered_directions_sha256="d" * 64,
                gains_sha256=f"{8000 + call_index:064x}",
                artifact_sha256=f"{9000 + call_index:064x}",
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

    monkeypatch.setattr(diagnostic, "_fresh_native_teacher", fresh)
    monkeypatch.setattr(diagnostic, "_endpoint_indices", endpoint)
    monkeypatch.setattr(
        diagnostic,
        "complete_h4_tail_gate_scores",
        lambda endpoint, rows: torch.zeros((1, 64), dtype=torch.float64),
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
            basis=torch.eye(320, 640, dtype=torch.float64),
            fits=fits,
            refits=refits,
            selections=selections,
        )
    )
    assert len(calls) == 32
    assert len(observations) == 32
    assert resources == {
        "final_native_forward_count": 16,
        "final_candidate_forward_count": 32,
        "final_observation_count": 32,
    }
    assert all(
        calls[index : index + 2]
        == [(f"e{index // 2:02d}", "unit"), (f"e{index // 2:02d}", "selected_refit")]
        for index in range(0, 32, 2)
    )
    assert all(
        row["held_family_excluded_from_gain_fit_and_tune"] is True
        for row in observations
    )
    assert set(behavior) == set(diagnostic._ARMS)
    assert set(geometry) == set(diagnostic._ARMS)


def test_phase_orchestrator_freezes_selection_before_held_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    roles = diagnostic._PromptRoles(("fit",), ("tune",), 398, 405, "a" * 64)
    refits = {"held": object()}
    selections = {"held": object()}

    monkeypatch.setattr(
        diagnostic,
        "_authenticate_parent_recollection",
        lambda **kwargs: order.append("recollection") or "b" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_checkerboard_prompt_roles",
        lambda traces: order.append("roles") or roles,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_static_unit_k64_replay",
        lambda **kwargs: order.append("static_unit_replay") or "d" * 64,
    )

    def gradient(**kwargs: object):
        order.append("gradient_fit")
        return refits, tuple(), {}

    def tune(**kwargs: object):
        assert kwargs["refits"] is refits
        order.append("tune_and_select")
        return selections, tuple(), {}

    def final(**kwargs: object):
        assert kwargs["refits"] is refits
        assert kwargs["selections"] is selections
        assert order[-1] == "tune_and_select"
        order.append("held_final")
        return [], {}, {}, {}

    monkeypatch.setattr(diagnostic, "_collect_candidate_gradient_refits", gradient)
    monkeypatch.setattr(
        diagnostic, "_collect_candidate_tune_selections", tune
    )
    monkeypatch.setattr(diagnostic, "_final_candidate_observations", final)
    monkeypatch.setattr(
        diagnostic,
        "_finite_observation_set_sha256",
        lambda observations: order.append("observation_set_auth") or "c" * 64,
    )
    monkeypatch.setattr(
        diagnostic,
        "_authenticate_unit_k64_replay",
        lambda **kwargs: order.append("unit_replay") or ({"tail_rank": 64}, "d" * 64),
    )
    result = diagnostic._execute_candidate_phases(
        context=object(),
        parent={},
        traces=(),
        endpoint_resources={},
        basis=torch.zeros((1, 1)),
        fits={},
    )
    assert order == [
        "recollection",
        "static_unit_replay",
        "roles",
        "gradient_fit",
        "tune_and_select",
        "held_final",
        "observation_set_auth",
        "unit_replay",
    ]
    assert result.selections is selections
    assert result.unit_replay_receipt == "d" * 64


def test_static_unit_replay_hashes_full_parent_score_bank_before_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = _prefix()
    base_h4 = torch.zeros((1, 3, 640), dtype=torch.float32)
    support_indices = torch.nonzero(
        prefix.complete_h4_causal_support_mask()[0], as_tuple=False
    ).flatten().to(dtype=torch.int64)
    full_scores = torch.arange(
        int(support_indices.numel()) * 320, dtype=torch.float64
    ).reshape(int(support_indices.numel()), 320)
    fit = SimpleNamespace(
        artifact_sha256="f" * 64,
        ordered_basis_rows=lambda: torch.zeros((320, 640), dtype=torch.float64),
    )
    trace = SimpleNamespace(
        example_id="example",
        family_id="family",
        prefix=prefix,
        base_h4=base_h4,
        native_h4=base_h4.clone(),
        support_indices=support_indices,
        endpoint=SimpleNamespace(
            compensation_target=full_scores[:, :64].sum(dim=1)
        ),
        native_token_nll=torch.tensor([1.0], dtype=torch.float64),
        d320_token_nll=torch.tensor([2.0], dtype=torch.float64),
    )
    monkeypatch.setattr(
        diagnostic,
        "_candidate_components",
        lambda *args, **kwargs: (
            torch.zeros((64, 640), dtype=torch.float64),
            torch.zeros((support_indices.numel(), 640), dtype=torch.float64),
            torch.zeros((support_indices.numel(), 640), dtype=torch.float64),
            torch.zeros_like(base_h4, dtype=torch.float64),
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "complete_h4_tail_gate_scores",
        lambda *args, **kwargs: full_scores,
    )
    expected = {
        "example_id": "example",
        "family_id": "family",
        "rank": 64,
        "fold_artifact_sha256": "f" * 64,
        "token_score_matrix_sha256": diagnostic._runtime_tensor_sha256(
            full_scores
        ),
        "native_mean_nll": 1.0,
        "d320_mean_nll": 2.0,
        "endpoint_baseline_mse": float(
            trace.endpoint.compensation_target.square().mean()
        ),
        "endpoint_prediction_mse": 0.0,
        "candidate_h4_bitwise_native": True,
        "full_tail_reconstruction_max_abs_error": None,
        "exact_residual_provider_used": False,
        "executed_correction_rows_sha256": diagnostic._runtime_tensor_sha256(
            torch.zeros((support_indices.numel(), 640), dtype=torch.float64)
        ),
    }
    monkeypatch.setattr(token_v1, "_EXPECTED_EXAMPLES", 1)
    assert len(
        diagnostic._authenticate_static_unit_k64_replay(
            parent={"finite_observation_receipts": [expected]},
            traces=(trace,),
            basis=torch.zeros((320, 640), dtype=torch.float64),
            fits={"family": fit},
        )
    ) == 64
    expected["token_score_matrix_sha256"] = diagnostic._runtime_tensor_sha256(
        full_scores[:, :64]
    )
    with pytest.raises(RuntimeError, match="token_score_matrix_sha256"):
        diagnostic._authenticate_static_unit_k64_replay(
            parent={"finite_observation_receipts": [expected]},
            traces=(trace,),
            basis=torch.zeros((320, 640), dtype=torch.float64),
            fits={"family": fit},
        )


def test_candidate_provider_is_stage_bound_gain_bound_and_single_use() -> None:
    prefix = _prefix()
    base = torch.zeros((1, 3, 640), dtype=torch.float32)
    support = prefix.complete_h4_causal_support_mask().detach().to("cpu")
    correction = torch.zeros_like(base, dtype=torch.float64)
    correction[support] = 0.25
    provider = diagnostic._AuthenticatedCandidateGainProvider(
        stage="gradient",
        gain_kind="unit",
        fold_artifact_sha256="f" * 64,
        ordered_directions_sha256="d" * 64,
        gains=torch.ones(64),
        refit_artifact_sha256=None,
        selection_artifact_sha256=None,
        alpha=None,
        model_inputs_sha256="a" * 64,
        bridge_binding_sha256="c" * 64,
        prefix_artifact_sha256=prefix.artifact_sha256,
        base_h4=base,
        support_mask=support,
        correction=correction,
    )
    assert torch.equal(provider.correction(prefix, base), correction)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        provider.correction(prefix, base)

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
    with pytest.raises(ValueError, match="all-one gains"):
        diagnostic._AuthenticatedCandidateGainProvider(
            stage="gradient",
            gain_kind="unit",
            gains=torch.full((64,), 0.5),
            refit_artifact_sha256=None,
            selection_artifact_sha256=None,
            alpha=None,
            **common,
        )
    with pytest.raises(ValueError, match="gradient provider semantics"):
        diagnostic._AuthenticatedCandidateGainProvider(
            stage="gradient",
            gain_kind="unit",
            gains=torch.ones(64),
            refit_artifact_sha256="e" * 64,
            selection_artifact_sha256=None,
            alpha=None,
            **common,
        )
    with pytest.raises(ValueError, match="tune provider semantics"):
        diagnostic._AuthenticatedCandidateGainProvider(
            stage="tune",
            gain_kind="interpolated",
            gains=torch.ones(64),
            refit_artifact_sha256="e" * 64,
            selection_artifact_sha256="b" * 64,
            alpha=0.25,
            **common,
        )
    with pytest.raises(ValueError, match="final provider semantics"):
        diagnostic._AuthenticatedCandidateGainProvider(
            stage="final",
            gain_kind="selected_refit",
            gains=torch.ones(64),
            refit_artifact_sha256="e" * 64,
            selection_artifact_sha256=None,
            alpha=None,
            **common,
        )


def _parent_fixture() -> dict[str, object]:
    behavior = {
        ledger: {"gates": {"passed": True}} for ledger in diagnostic._LEDGERS
    }
    return {
        "schema": expanded._SCHEMA,
        "report_sha256": diagnostic.EXPANDED_PARENT_REPORT_SHA256,
        "classification": (
            "adaptive_same_a_smallest_tail_rank_256_cleared_established_gates"
        ),
        "passed": True,
        "protocol": {"tail_ranks": list(expanded.EXPANDED_TAIL_RANKS)},
        "smallest_tail_rank_below_320_clearing_established_fidelity_and_geometry_gates": 256,
        "fidelity_and_geometry_pass_by_rank": {
            "64": False,
            "96": False,
            "128": False,
            "160": False,
            "192": False,
            "256": True,
            "320": True,
        },
        "finite_observation_receipts": [],
        "finite_observation_set_sha256": "o" * 64,
        "finite_ladder": [
            {
                "tail_rank": 320,
                "every_prompt_h4_bitwise_native": True,
                "every_prompt_logits_bitwise_native": True,
                "maximum_full_tail_reconstruction_abs_error": 0.0,
            }
        ],
        "established_behavioral_fidelity_by_rank": {"320": behavior},
        "executed_cast_once_geometry_by_rank": {
            "320": {"gates": {"passed": True}}
        },
        "scientific_status": {
            "same_a_adaptive_hypothesis_use_only": True,
            "fresh_confirmation_panel_opened": False,
            "candidate_serving_authorized": False,
            "compression_claim": False,
        },
        "safety": {
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_only_hashes_counts_and_scalar_metrics": True,
        },
    }


def test_expanded_parent_loader_pins_exact_v2_and_rank_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _parent_fixture()

    def fake_load(
        path: object,
        *,
        expected_file_sha256: str,
        expected_report_sha256: str,
        label: str,
    ) -> dict[str, object]:
        assert str(path) == "parent.json"
        assert expected_file_sha256 == diagnostic.EXPANDED_PARENT_REPORT_FILE_SHA256
        assert expected_report_sha256 == diagnostic.EXPANDED_PARENT_REPORT_SHA256
        return parent

    monkeypatch.setattr(token_v1, "_load_pinned_report", fake_load)
    monkeypatch.setattr(
        teacher_kl, "_load_expanded_control_report", lambda path: parent
    )
    monkeypatch.setattr(
        token_v1,
        "_finite_observation_set_sha256",
        lambda *args, **kwargs: "o" * 64,
    )
    assert diagnostic._load_expanded_parent("parent.json") is parent
    parent["fidelity_and_geometry_pass_by_rank"] = {
        **dict(parent["fidelity_and_geometry_pass_by_rank"]),
        "192": True,
    }
    with pytest.raises(ValueError, match="parent semantics differ"):
        diagnostic._load_expanded_parent("parent.json")


def _observation(example: int, arm: str) -> dict[str, object]:
    row: dict[str, object] = {
        "example_id": f"e{example:02d}",
        "family_id": f"f{example // 2}",
        "arm": arm,
        "rank": 64,
    }
    row["observation_sha256"] = token_v1._domain_sha256(
        row, domain=diagnostic._OBSERVATION_DOMAIN
    )
    return row


def test_final_observation_receipt_requires_exact_16_by_2_grid() -> None:
    rows = [
        _observation(example, arm)
        for example in range(16)
        for arm in diagnostic._ARMS
    ]
    assert len(diagnostic._finite_observation_set_sha256(rows)) == 64
    with pytest.raises(ValueError, match="count differs"):
        diagnostic._finite_observation_set_sha256(rows[:-1])
    tampered = deepcopy(rows)
    tampered[0]["family_id"] = "changed"
    with pytest.raises(RuntimeError, match="receipt drifted"):
        diagnostic._finite_observation_set_sha256(tampered)


def _v1_replay_row(example: int, rank: int) -> dict[str, object]:
    exact = rank == 320
    return {
        "example_id": f"e{example:02d}",
        "family_id": f"f{example // 2}",
        "rank": rank,
        "fold_artifact_sha256": "a" * 64,
        "provider_artifact_sha256": "b" * 64,
        "execution_artifact_sha256": "c" * 64,
        "token_score_matrix_sha256": "d" * 64,
        "native_mean_nll": 1.0,
        "d320_mean_nll": 2.0,
        "candidate_mean_nll": 1.0 if exact else 1.5,
        "ordinary_candidate_mean_nll": 1.0 if exact else 1.5,
        "endpoint_baseline_mse": 1.0,
        "endpoint_prediction_mse": 0.0 if exact else 0.25,
        "candidate_h4_bitwise_native": exact,
        "candidate_logits_bitwise_native": exact,
        "full_tail_reconstruction_max_abs_error": 0.0 if exact else None,
        "exact_residual_provider_used": False,
        "executed_correction_rows_sha256": "e" * 64,
    }


def test_unit_replay_authenticates_exact_outputs_summary_behavior_geometry() -> None:
    parent_k64 = [_v1_replay_row(index, 64) for index in range(16)]
    parent_k320 = [_v1_replay_row(index, 320) for index in range(16)]
    summaries, _gates = token_v1._summarize_observations(
        parent_k64 + parent_k320, ranks=(64, 320)
    )
    k64_summary = next(row for row in summaries if row["tail_rank"] == 64)
    unit_behavior = {ledger: {"stable": True} for ledger in diagnostic._LEDGERS}
    unit_geometry = {
        "semantics": {
            "candidate": "actual_cast_once_d320_plus_training_only_fisher_tail_k64"
        },
        "gates": {"passed": False},
    }
    parent = {
        "finite_observation_receipts": parent_k64 + parent_k320,
        "finite_ladder": [k64_summary],
        "established_behavioral_fidelity_by_rank": {"64": unit_behavior},
        "executed_cast_once_geometry_by_rank": {"64": unit_geometry},
    }
    current = []
    for row in parent_k64:
        value = {
            **row,
            "arm": "unit_k64",
            # Provider/execution receipts are expected to differ under the
            # new provider schema; all stable outputs remain identical.
            "provider_artifact_sha256": "f" * 64,
            "execution_artifact_sha256": "1" * 64,
        }
        current.append(value)
    summary, receipt = diagnostic._authenticate_unit_k64_replay(
        parent=parent,
        observations=current,
        behavior={"unit_k64": unit_behavior},
        geometry={"unit_k64": unit_geometry},
    )
    assert summary == k64_summary
    assert len(receipt) == 64

    tampered = deepcopy(current)
    tampered[0]["candidate_mean_nll"] = 1.6
    with pytest.raises(RuntimeError, match="stable outputs differ"):
        diagnostic._authenticate_unit_k64_replay(
            parent=parent,
            observations=tampered,
            behavior={"unit_k64": unit_behavior},
            geometry={"unit_k64": unit_geometry},
        )
    with pytest.raises(RuntimeError, match="summary/behavior/geometry"):
        diagnostic._authenticate_unit_k64_replay(
            parent=parent,
            observations=current,
            behavior={"unit_k64": {"changed": True}},
            geometry={"unit_k64": unit_geometry},
        )


def test_gradient_and_tune_receipt_sets_fail_closed_on_tamper() -> None:
    gradient: list[dict[str, object]] = []
    tune: list[dict[str, object]] = []
    for held in range(8):
        for example in range(8):
            if held == example:
                continue
            row: dict[str, object] = {
                "held_family_id": f"f{held}",
                "example_id": f"e{example}",
                "family_id": f"f{example}",
                "alpha_hex": None,
            }
            row["receipt_sha256"] = token_v1._domain_sha256(
                row, domain=diagnostic._GRADIENT_RECEIPT_DOMAIN
            )
            gradient.append(row)
            for alpha in diagnostic.CANDIDATE_GAIN_ALPHAS:
                tune_row: dict[str, object] = {
                    "held_family_id": f"f{held}",
                    "example_id": f"e{example}",
                    "family_id": f"f{example}",
                    "alpha_hex": alpha.hex(),
                }
                tune_row["receipt_sha256"] = token_v1._domain_sha256(
                    tune_row, domain=diagnostic._TUNE_RECEIPT_DOMAIN
                )
                tune.append(tune_row)
    assert len(
        diagnostic._receipt_set_sha256(
            gradient,
            expected_count=56,
            receipt_domain=diagnostic._GRADIENT_RECEIPT_DOMAIN,
            set_domain=b"gradient-set\0",
        )
    ) == 64
    assert len(
        diagnostic._receipt_set_sha256(
            tune,
            expected_count=224,
            receipt_domain=diagnostic._TUNE_RECEIPT_DOMAIN,
            set_domain=b"tune-set\0",
        )
    ) == 64
    tampered = deepcopy(tune)
    tampered[0]["family_id"] = "changed"
    with pytest.raises(RuntimeError, match="receipt drifted"):
        diagnostic._receipt_set_sha256(
            tampered,
            expected_count=224,
            receipt_domain=diagnostic._TUNE_RECEIPT_DOMAIN,
            set_domain=b"tune-set\0",
        )


def test_full_resource_ledger_closes_at_392_forwards_and_494_backwards() -> None:
    traces = _role_traces()
    roles = diagnostic._checkerboard_prompt_roles(traces)
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
    }
    tune = {
        "tune_native_forward_count": 8,
        "tune_candidate_forward_count": 224,
        "tune_prompt_fold_alpha_count": 224,
    }
    final = {
        "final_native_forward_count": 16,
        "final_candidate_forward_count": 32,
        "final_observation_count": 32,
    }
    result = diagnostic._resource_accounting(
        traces=traces,
        roles=roles,
        endpoint_resources=endpoint,
        gradient_resources=gradient,
        tune_resources=tune,
        final_resources=final,
    )
    assert result["total_model_forward_count"] == 392
    assert result["total_backward_call_count"] == 494
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
        7 * fit_rows + 28 * tune_rows + 2 * all_rows
    )
    assert result["k320_model_forward_count"] == 0


def _behavior(top1_by_arm: dict[str, float]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for arm, top1 in top1_by_arm.items():
        result[arm] = {}
        for ledger in diagnostic._LEDGERS:
            value = 0.50 if ledger == "causal_tail" and arm == "selected_refit_k64" else top1
            result[arm][ledger] = {
                "aggregate": {"top1_agreement_to_source": value},
                "family_summary": {
                    "macro": {"top1_agreement_to_source": value}
                },
                "gates": {"passed": value >= 0.90},
            }
    return result


def test_approximate_gate_excludes_tiny_causal_tail_but_reports_strict_failure() -> None:
    behavior = _behavior({"unit_k64": 0.92, "selected_refit_k64": 0.91})
    geometry = {
        arm: {"gates": {"passed": True}} for arm in diagnostic._ARMS
    }
    comparison = diagnostic._top1_comparison(behavior)
    strict, approximate = diagnostic._gate_results(
        behavior=behavior,
        geometry=geometry,
        teacher_kl_comparison={
            "family_macro_relative_improvement": 0.03,
            "held_family_improvement_count": 6,
            "all_held_families_within_five_percent_plus_1e_minus_8": True,
        },
        top1_comparison=comparison,
        selections={
            f"f{index}": SimpleNamespace(selected_alpha=0.25 if index < 6 else 0.0)
            for index in range(8)
        },
    )
    assert approximate["passed"] is True
    assert approximate[
        "causal_tail_reported_but_excluded_from_approximate_decision"
    ] is True
    assert strict["selected_refit_k64"]["behavioral_only_strict_passed"] is False
    assert strict["selected_refit_k64"][
        "full_behavior_plus_geometry_strict_passed"
    ] is False
    assert comparison["ordinary"]["aggregate_delta_refit_minus_unit"] == pytest.approx(
        -0.01
    )


def test_parent_recollection_is_fail_closed_before_candidate_work() -> None:
    with pytest.raises(RuntimeError, match="did not replay"):
        diagnostic._authenticate_parent_recollection(
            parent={
                "prompt_receipts": [],
                "folds": [],
                "resources": {
                    key: 0
                    for key in (
                        "base_forward_count",
                        "native_forward_count",
                        "endpoint_token_vjp_forward_count",
                        "endpoint_token_vjp_backward_call_count",
                        "ordinary_supervised_token_count",
                        "endpoint_support_supervised_token_count",
                        "complete_h4_support_row_count",
                        "graph_core_row_count",
                        "causal_tail_row_count",
                        "graph_core_supervised_token_count",
                        "causal_tail_supervised_token_count",
                    )
                },
            },
            traces=(),
            endpoint_resources={
                "base_forward_count": 1,
                "native_forward_count": 0,
                "endpoint_token_vjp_forward_count": 0,
                "endpoint_token_vjp_backward_call_count": 0,
                "ordinary_supervised_token_count": 0,
                "endpoint_support_supervised_token_count": 0,
                "complete_h4_support_row_count": 0,
                "graph_core_row_count": 0,
                "causal_tail_row_count": 0,
                "graph_core_supervised_token_count": 0,
                "causal_tail_supervised_token_count": 0,
            },
            fits={},
        )


def test_cli_has_fixed_k64_alpha_protocol_and_local_output() -> None:
    parser = diagnostic.build_parser()
    args = parser.parse_args([])
    assert args.expanded_parent_report == diagnostic.DEFAULT_EXPANDED_PARENT_REPORT
    assert args.output == diagnostic.DEFAULT_OUTPUT
    assert not any(
        action.dest in {"rank", "ranks", "alpha", "alphas", "trust_rms"}
        for action in parser._actions
    )
