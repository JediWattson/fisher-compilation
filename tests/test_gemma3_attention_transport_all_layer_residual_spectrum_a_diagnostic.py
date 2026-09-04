from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic as runner


_LEXICAL_IDS = tuple(
    sorted(f"mutation.{2 * index + 1}" for index in range(18))
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metric_payload(
    rows: int,
    relative_fisher_loss: float,
    *,
    zero_predictor_fisher_loss: float = 10.0,
) -> dict[str, object]:
    fisher_loss = relative_fisher_loss * zero_predictor_fisher_loss
    return {
        "row_count": rows,
        "fisher_loss": fisher_loss,
        "directional_mse": fisher_loss / 2.0,
        "euclidean_mse": fisher_loss / 3.0,
        "normalized_rmse": relative_fisher_loss**0.5,
        "cosine": 0.8,
        "zero_predictor_fisher_loss": zero_predictor_fisher_loss,
        "relative_fisher_loss": relative_fisher_loss,
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "solver_kind": "exact_signed_top2_sparse_oracle_v1",
    }


def _fit_metric_payload(
    rows: int,
    relative_fisher_loss: float,
    per_invocation_relative_fisher_loss: dict[str, float],
) -> dict[str, object]:
    payload = _metric_payload(rows, relative_fisher_loss)
    payload["per_invocation_relative_fisher_loss"] = [
        [invocation_id, per_invocation_relative_fisher_loss[invocation_id]]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    return payload


def _bank_summary(
    *,
    artifact_sha256: str,
    parent_sha256: str,
    sequence: tuple[str, ...],
) -> dict[str, object]:
    allocated_rank_profile = [
        [invocation_id, sequence.count(invocation_id)]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    total_rank_profile = [
        [
            invocation_id,
            sequence.count(invocation_id)
            + (
                1
                if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
                else 0
            ),
        ]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    component_count = len(sequence)
    return {
        "artifact_sha256": artifact_sha256,
        "parent_site_rank_one_bank_sha256": parent_sha256,
        "allocated_invocation_ids": list(sequence),
        "allocated_rank_profile": allocated_rank_profile,
        "total_rank_profile": total_rank_profile,
        "allocated_component_count": component_count,
        "allocated_parameter_count": component_count * 10_240,
        "stored_parameter_count": 113_728 + component_count * 10_240,
    }


def _full_marginal(
    invocation_id: str,
    *,
    source_sha256: str,
    parent_sha256: str,
    proposal_sequence: tuple[str, ...],
    proposal_rows: int = 16,
    evaluation_rows: int = 16,
) -> dict[str, object]:
    current_fit = _metric_payload(proposal_rows, 0.5)
    candidate_fit = _metric_payload(proposal_rows, 0.4)
    current_evaluation = _metric_payload(evaluation_rows, 0.5)
    candidate_evaluation = _metric_payload(evaluation_rows, 0.4)
    return {
        "invocation_id": invocation_id,
        "source_bank_sha256": source_sha256,
        "proposal_bank": _bank_summary(
            artifact_sha256=_sha(
                "proposal-" + "-".join(proposal_sequence)
            ),
            parent_sha256=parent_sha256,
            sequence=proposal_sequence,
        ),
        "proposal_fit_current_oracle": current_fit,
        "proposal_fit_candidate_oracle": candidate_fit,
        "allocation_evaluation_current_oracle": current_evaluation,
        "allocation_evaluation_candidate_oracle": candidate_evaluation,
        "frozen_current_fisher_loss": current_fit["fisher_loss"],
        "frozen_candidate_fisher_loss": candidate_fit["fisher_loss"],
        "frozen_relative_fisher_gain": 0.1,
        "allocation_evaluation_relative_fisher_gain": 0.1,
        "generator_singular_values": [float(index + 1) for index in range(8)],
        "generator_amplitudes": [float(index + 1) / 10.0 for index in range(8)],
        "right_projection_ranks": [1 for _index in range(8)],
        "minimum_relative_fisher_gain": 1.0e-12,
        "strictly_improves_all_protocol_gates": True,
    }


def _records() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            base_capture_receipt_sha256=_sha(f"receipt-{receipt_index}"),
            capture=SimpleNamespace(
                invocation_id=invocation_id,
                row_count=16,
            ),
        )
        for receipt_index in range(45)
        for invocation_id in _LEXICAL_IDS
    )


def _sparse_evaluation(
    rows: int,
    fisher: float,
    zero: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_metrics=SimpleNamespace(
            row_count=rows,
            fisher_loss=fisher,
            directional_mse=fisher / 2.0,
            euclidean_mse=fisher / 3.0,
            normalized_rmse=(fisher / zero) ** 0.5,
            cosine=0.8,
        ),
        zero_predictor_fisher_loss=zero,
        relative_fisher_loss=fisher / zero,
        active_count_limit=2,
        coefficient_ridge=0.0,
    )


def _outer_bank_evaluation(
    *,
    allocated: bool,
) -> SimpleNamespace:
    per_invocation = []
    for index, invocation_id in enumerate(runner._NUMERIC_INVOCATION_IDS):
        zero = float(index + 1)
        relative = 0.2 + index / 100.0
        if allocated and invocation_id in ("mutation.1", "mutation.3"):
            relative -= 0.05
        per_invocation.append(
            (
                invocation_id,
                _sparse_evaluation(144, zero * relative, zero),
            )
        )

    def aggregate(
        selected: list[tuple[str, SimpleNamespace]],
    ) -> SimpleNamespace:
        rows = sum(value.output_metrics.row_count for _key, value in selected)
        fisher = sum(
            value.output_metrics.row_count * value.output_metrics.fisher_loss
            for _key, value in selected
        ) / rows
        zero = sum(
            value.output_metrics.row_count * value.zero_predictor_fisher_loss
            for _key, value in selected
        ) / rows
        return _sparse_evaluation(rows, fisher, zero)

    local = [
        item
        for item in per_invocation
        if item[0] not in runner._GLOBAL_ATTENTION_INVOCATION_IDS
    ]
    global_values = [
        item
        for item in per_invocation
        if item[0] in runner._GLOBAL_ATTENTION_INVOCATION_IDS
    ]
    return SimpleNamespace(
        all_layers=aggregate(per_invocation),
        local_attention=aggregate(local),
        global_attention=aggregate(global_values),
        per_invocation=tuple(per_invocation),
    )


def _ledger_marginal(
    invocation_id: str,
    *,
    source_sha256: str,
    normalized_gain: float,
    strictly_improves: bool,
    current_relative_fisher_loss: float = 0.5,
) -> dict[str, object]:
    current_fisher = 20.0
    candidate_fisher = current_fisher - normalized_gain * 180.0
    return {
        "invocation_id": invocation_id,
        "source_bank_sha256": source_sha256,
        "proposal_fit_current_oracle": {
            "relative_fisher_loss": current_relative_fisher_loss,
        },
        "allocation_evaluation_current_oracle": {
            "fisher_loss": current_fisher,
            "relative_fisher_loss": current_relative_fisher_loss,
        },
        "allocation_evaluation_candidate_oracle": {
            "fisher_loss": candidate_fisher,
        },
        "strictly_improves_all_protocol_gates": strictly_improves,
    }


def _ledger_candidate(
    marginal: dict[str, object],
    *,
    current_bank_sha256: str,
    normalized_gain: float,
    minimum_gain: float = 1.0e-12,
) -> dict[str, object]:
    del normalized_gain
    source_sha256 = str(marginal["source_bank_sha256"])
    current = marginal["allocation_evaluation_current_oracle"]
    candidate = marginal["allocation_evaluation_candidate_oracle"]
    assert isinstance(current, dict)
    assert isinstance(candidate, dict)
    normalized_gain = (
        runner._ALLOCATION_VALIDATION_RECEIPT_COUNT
        * 16
        * (float(current["fisher_loss"]) - float(candidate["fisher_loss"]))
        / runner._ALLOCATION_VALIDATION_ROWS
        / 10.0
    )
    return {
        "invocation_id": marginal["invocation_id"],
        "source_bank_sha256": source_sha256,
        "used_cached_off_site_marginal": (
            source_sha256 != current_bank_sha256
        ),
        "strictly_improves_all_protocol_gates": marginal[
            "strictly_improves_all_protocol_gates"
        ],
        "normalized_fisher_mass_gain": normalized_gain,
        "eligible_for_all_layer_allocation": bool(
            marginal["strictly_improves_all_protocol_gates"]
            and normalized_gain > minimum_gain
        ),
        "marginal": marginal,
    }


def _two_round_candidate_ledger() -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    str,
    str,
]:
    initial_sha = _sha("ledger-initial")
    after_first_sha = _sha("ledger-after-first")
    final_sha = _sha("ledger-final")
    numeric_ids = runner._NUMERIC_INVOCATION_IDS
    first_gains = {
        invocation_id: (
            0.05
            if invocation_id in ("mutation.1", "mutation.3")
            else 0.01
        )
        for invocation_id in numeric_ids
    }
    first_marginals = {
        invocation_id: _ledger_marginal(
            invocation_id,
            source_sha256=initial_sha,
            normalized_gain=first_gains[invocation_id],
            strictly_improves=True,
        )
        for invocation_id in numeric_ids
    }
    first_candidates = [
        _ledger_candidate(
            first_marginals[invocation_id],
            current_bank_sha256=initial_sha,
            normalized_gain=first_gains[invocation_id],
        )
        for invocation_id in numeric_ids
    ]
    first_round = {
        "allocation_index": 0,
        "current_bank_sha256": initial_sha,
        "evaluation_row_count": 1_728,
        "evaluation_zero_predictor_fisher_loss": 10.0,
        "minimum_relative_fisher_gain": 1.0e-12,
        "tie_tolerance": 1.0e-15,
        "candidates": first_candidates,
        "selected_invocation_id": "mutation.1",
        "early_stop_no_eligible_candidate": False,
    }

    second_gains = dict(first_gains)
    second_gains["mutation.1"] = 0.001
    second_marginals = dict(first_marginals)
    second_marginals["mutation.1"] = _ledger_marginal(
        "mutation.1",
        source_sha256=after_first_sha,
        normalized_gain=second_gains["mutation.1"],
        strictly_improves=True,
        current_relative_fisher_loss=0.4,
    )
    second_candidates = [
        _ledger_candidate(
            second_marginals[invocation_id],
            current_bank_sha256=after_first_sha,
            normalized_gain=second_gains[invocation_id],
        )
        for invocation_id in numeric_ids
    ]
    second_round = {
        "allocation_index": 1,
        "current_bank_sha256": after_first_sha,
        "evaluation_row_count": 1_728,
        "evaluation_zero_predictor_fisher_loss": 10.0,
        "minimum_relative_fisher_gain": 1.0e-12,
        "tie_tolerance": 1.0e-15,
        "candidates": second_candidates,
        "selected_invocation_id": "mutation.3",
        "early_stop_no_eligible_candidate": False,
    }
    steps = [
        {
            "invocation_id": "mutation.1",
            "marginal": first_marginals["mutation.1"],
            "selection_all_layer_normalized_fisher_gain": first_candidates[0][
                "normalized_fisher_mass_gain"
            ],
            "allocation_validation_before": {
                "relative_fisher_loss": 0.50,
                "zero_predictor_fisher_loss": 10.0,
            },
            "direction_train_after": {
                "per_invocation_relative_fisher_loss": [
                    [invocation_id, 0.4 if invocation_id == "mutation.1" else 0.5]
                    for invocation_id in numeric_ids
                ],
            },
            "allocation_validation_after": {
                "relative_fisher_loss": 0.45,
                "zero_predictor_fisher_loss": 10.0,
                "per_invocation_relative_fisher_loss": [
                    [invocation_id, 0.4 if invocation_id == "mutation.1" else 0.5]
                    for invocation_id in numeric_ids
                ],
            },
            "result_bank_sha256": after_first_sha,
        },
        {
            "invocation_id": "mutation.3",
            "marginal": second_marginals["mutation.3"],
            "selection_all_layer_normalized_fisher_gain": second_candidates[1][
                "normalized_fisher_mass_gain"
            ],
            "allocation_validation_before": {
                "relative_fisher_loss": 0.45,
                "zero_predictor_fisher_loss": 10.0,
            },
            "direction_train_after": {
                "per_invocation_relative_fisher_loss": [
                    [
                        invocation_id,
                        0.4
                        if invocation_id in ("mutation.1", "mutation.3")
                        else 0.5,
                    ]
                    for invocation_id in numeric_ids
                ],
            },
            "allocation_validation_after": {
                "relative_fisher_loss": 0.40,
                "zero_predictor_fisher_loss": 10.0,
                "per_invocation_relative_fisher_loss": [
                    [
                        invocation_id,
                        0.4
                        if invocation_id in ("mutation.1", "mutation.3")
                        else 0.5,
                    ]
                    for invocation_id in numeric_ids
                ],
            },
            "result_bank_sha256": final_sha,
        },
    ]
    first_spectrum = {
        "per_site": [first_marginals[value] for value in numeric_ids]
    }
    return [first_round, second_round], first_spectrum, steps, initial_sha, final_sha


def _one_step_proof_payloads() -> tuple[dict[str, object], dict[str, object]]:
    sequence = ("mutation.1",)
    initial_sha256 = _sha("step-initial")
    final_sha256 = _sha("step-final")
    parent_sha256 = _sha("step-parent")
    candidate_rounds: list[dict[str, object]] = []
    candidate_rounds_sha256 = runner._candidate_rounds_artifact_sha256(
        candidate_rounds
    )
    bank = _bank_summary(
        artifact_sha256=final_sha256,
        parent_sha256=parent_sha256,
        sequence=sequence,
    )
    profile_sha256 = runner._profile_artifact_sha256(
        sequence,
        bank["allocated_rank_profile"],
        bank["total_rank_profile"],
        candidate_rounds_sha256,
    )
    before_by_id = {
        invocation_id: 0.5
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    }
    after_by_id = dict(before_by_id)
    after_by_id["mutation.1"] = 0.4
    initial_fit = _fit_metric_payload(
        runner._DIRECTION_TRAIN_ROWS,
        0.5,
        before_by_id,
    )
    final_fit = _fit_metric_payload(
        runner._DIRECTION_TRAIN_ROWS,
        0.45,
        after_by_id,
    )
    initial_evaluation = _fit_metric_payload(
        runner._ALLOCATION_VALIDATION_ROWS,
        0.5,
        before_by_id,
    )
    final_evaluation = _fit_metric_payload(
        runner._ALLOCATION_VALIDATION_ROWS,
        0.45,
        after_by_id,
    )
    marginal = {
        "proposal_fit_current_oracle": {"relative_fisher_loss": 0.5},
        "proposal_fit_candidate_oracle": {"relative_fisher_loss": 0.4},
        "allocation_evaluation_current_oracle": {
            "relative_fisher_loss": 0.5
        },
        "allocation_evaluation_candidate_oracle": {
            "relative_fisher_loss": 0.4
        },
        "proposal_bank": {"artifact_sha256": final_sha256},
    }
    greedy_step = {
        "allocation_index": 0,
        "invocation_id": "mutation.1",
        "source_bank_sha256": initial_sha256,
        "result_bank_sha256": final_sha256,
        "marginal_source_bank_sha256": initial_sha256,
        "used_cached_off_site_marginal": False,
        "selection_all_layer_normalized_fisher_gain": 0.05,
        "marginal": marginal,
        "direction_train_before": initial_fit,
        "direction_train_after": final_fit,
        "allocation_validation_before": initial_evaluation,
        "allocation_validation_after": final_evaluation,
    }
    greedy = {
        "initial_bank_sha256": initial_sha256,
        "final_bank": bank,
        "allocation_budget": 2,
        "budget_exhausted": False,
        "frozen_allocation_sequence": list(sequence),
        "frozen_allocated_rank_profile": bank["allocated_rank_profile"],
        "frozen_total_rank_profile": bank["total_rank_profile"],
        "frozen_profile_artifact_sha256": profile_sha256,
        "first_spectrum": {
            "source_bank_sha256": initial_sha256,
            "direction_train_current_oracle": initial_fit,
            "allocation_validation_current_oracle": initial_evaluation,
        },
        "steps": [greedy_step],
        "candidate_rounds": candidate_rounds,
        "candidate_rounds_artifact_sha256": candidate_rounds_sha256,
        "initial_direction_train_oracle": initial_fit,
        "final_direction_train_oracle": final_fit,
        "initial_allocation_validation_oracle": initial_evaluation,
        "final_allocation_validation_oracle": final_evaluation,
        "direction_train_relative_fisher_loss_history": [0.5, 0.45],
        "allocation_validation_relative_fisher_loss_history": [0.5, 0.45],
    }

    initial_full_fit = _fit_metric_payload(
        runner._OUTER_FIT_ROWS,
        0.5,
        before_by_id,
    )
    final_full_fit = _fit_metric_payload(
        runner._OUTER_FIT_ROWS,
        0.45,
        after_by_id,
    )
    fixed_marginal = {
        "proposal_fit_current_oracle": {"relative_fisher_loss": 0.5},
        "proposal_fit_candidate_oracle": {"relative_fisher_loss": 0.4},
        "allocation_evaluation_current_oracle": {
            "relative_fisher_loss": 0.5
        },
        "allocation_evaluation_candidate_oracle": {
            "relative_fisher_loss": 0.4
        },
        "proposal_bank": {"artifact_sha256": final_sha256},
    }
    fixed = {
        "initial_bank_sha256": initial_sha256,
        "final_bank": bank,
        "allocation_profile": list(sequence),
        "allocated_component_count": 1,
        "selection_candidate_rounds_artifact_sha256": candidate_rounds_sha256,
        "frozen_profile_artifact_sha256": profile_sha256,
        "replay_profile_artifact_sha256": profile_sha256,
        "replay_profile_matches_frozen_selection": True,
        "steps": [
            {
                "allocation_index": 0,
                "invocation_id": "mutation.1",
                "source_bank_sha256": initial_sha256,
                "result_bank_sha256": final_sha256,
                "marginal_source_bank_sha256": initial_sha256,
                "used_cached_off_site_marginal": False,
                "all_layer_relative_fisher_gain": 0.05,
                "marginal": fixed_marginal,
                "full_fit_before": initial_full_fit,
                "full_fit_after": final_full_fit,
            }
        ],
        "initial_full_fit_oracle": initial_full_fit,
        "final_full_fit_oracle": final_full_fit,
        "full_fit_relative_fisher_loss_history": [0.5, 0.45],
    }
    return greedy, fixed


def test_module_is_stdlib_only_and_public_surface_cannot_open_B_or_C() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(
                alias.name.split(".")[0] in sys.stdlib_module_names
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            assert node.module is not None
            assert node.module.split(".")[0] in sys.stdlib_module_names

    public = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }
    assert public == {
        "run_all_layer_residual_spectrum_A_diagnostic",
        "check_all_layer_residual_spectrum_A_diagnostic",
        "main",
    }
    capture_roles = []
    called_names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        called_names.append(called)
        if called == "_capture_role":
            runtime_role = next(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "runtime_role"
            )
            assert isinstance(runtime_role, ast.Name)
            capture_roles.append(runtime_role.id)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert capture_roles == ["FIT_ROLE"]
    assert "DISCOVERY_ROLE" not in names
    assert "CONFIRMATION_ROLE" not in names
    assert "advance" not in called_names
    assert "freeze_current" not in called_names

    code = """
import sys
import fisher_graph
baseline = set(sys.modules)
import fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum_a_diagnostic
new_modules = set(sys.modules) - baseline
forbidden_heavy = ('torch', 'mlx', 'transformers')
forbidden_experiment = (
    'fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum',
    'fisher_graph.gemma3_chart_native_neighbor_fresh_panels',
)
assert not [name for name in forbidden_heavy if name in new_modules]
assert not [name for name in forbidden_experiment if name in sys.modules]
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_nested_whole_receipt_partition_is_deterministic_30_6_inside_36_9() -> None:
    records = _records()
    outer_fit, outer_dev, fit_receipts, dev_receipts = (
        runner._outer_partition_records(records)
    )
    direction, validation, direction_receipts, validation_receipts = (
        runner._inner_partition_records(outer_fit)
    )
    assert len(fit_receipts) == 36
    assert len(dev_receipts) == 9
    assert len(direction_receipts) == 30
    assert len(validation_receipts) == 6
    assert len(outer_fit) == 36 * 18
    assert len(outer_dev) == 9 * 18
    assert len(direction) == 30 * 18
    assert len(validation) == 6 * 18
    assert sum(record.capture.row_count for record in outer_fit) == 10_368
    assert sum(record.capture.row_count for record in outer_dev) == 2_592
    assert sum(record.capture.row_count for record in direction) == 8_640
    assert sum(record.capture.row_count for record in validation) == 1_728
    assert set(fit_receipts).isdisjoint(dev_receipts)
    assert set(direction_receipts).isdisjoint(validation_receipts)
    assert set(direction_receipts) | set(validation_receipts) == set(
        fit_receipts
    )
    assert set(dev_receipts).isdisjoint(direction_receipts)
    assert set(dev_receipts).isdisjoint(validation_receipts)
    assert runner._outer_partition_artifact_sha256(
        fit_receipts,
        dev_receipts,
    ) == runner._outer_partition_artifact_sha256(fit_receipts, dev_receipts)
    assert runner._inner_partition_artifact_sha256(
        direction_receipts,
        validation_receipts,
    ) == runner._inner_partition_artifact_sha256(
        direction_receipts,
        validation_receipts,
    )
    assert runner._outer_partition_records(tuple(reversed(records)))[2:] == (
        fit_receipts,
        dev_receipts,
    )


@pytest.mark.parametrize(
    ("section", "field", "wrong_value"),
    (
        ("scientific_status", "cross_run_A_split_identity_claimed", True),
        ("A_partition", "cross_run_split_identity_required", True),
        ("A_partition", "partition_policy", "cross_run_identity_v1"),
        ("protocol", "partition_policy", "cross_run_identity_v1"),
    ),
)
def test_prior_targeted_lineage_binds_current_run_split_policy(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    wrong_value: object,
) -> None:
    import fisher_graph.gemma3_attention_transport_targeted_site_rank2_a_diagnostic as targeted

    root = Path(__file__).resolve().parents[1]
    prior_path = root / runner._PRIOR_TARGETED_REPORT_PATH
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    assert prior["scientific_status"][
        "cross_run_A_split_identity_claimed"
    ] is False
    assert prior["A_partition"]["cross_run_split_identity_required"] is False
    assert prior["protocol"]["partition_policy"] == (
        "current_run_domain_hash_sorted_whole_A_receipts_36_fit_9_dev_v1"
    )

    tampered = deepcopy(prior)
    tampered[section][field] = wrong_value
    monkeypatch.setattr(targeted, "_strict_report", lambda _path: tampered)
    monkeypatch.setattr(
        targeted,
        "check_targeted_site_rank2_A_diagnostic",
        lambda *_args, **_kwargs: {},
    )
    with pytest.raises(ValueError, match="sealed targeted report lineage drifted"):
        runner._validate_prior_targeted_report(root)


def test_marginal_binds_frozen_losses_paired_zeros_parent_and_proposal() -> None:
    source_sha256 = _sha("marginal-source")
    parent_sha256 = _sha("marginal-parent")
    marginal = _full_marginal(
        "mutation.1",
        source_sha256=source_sha256,
        parent_sha256=parent_sha256,
        proposal_sequence=("mutation.1",),
    )
    assert runner._validate_marginal(
        marginal,
        proposal_rows=16,
        evaluation_rows=16,
        expected_source_sha256=source_sha256,
        expected_parent_sha256=parent_sha256,
        expected_proposal_sequence=("mutation.1",),
    ) is marginal

    frozen_tamper = deepcopy(marginal)
    frozen_tamper["frozen_current_fisher_loss"] = 5.25
    with pytest.raises(ValueError, match="marginal values drifted"):
        runner._validate_marginal(
            frozen_tamper,
            proposal_rows=16,
            evaluation_rows=16,
            expected_source_sha256=source_sha256,
            expected_parent_sha256=parent_sha256,
            expected_proposal_sequence=("mutation.1",),
        )

    for metric_name in (
        "proposal_fit_candidate_oracle",
        "allocation_evaluation_candidate_oracle",
    ):
        zero_tamper = deepcopy(marginal)
        metric = zero_tamper[metric_name]
        assert isinstance(metric, dict)
        metric["zero_predictor_fisher_loss"] = 20.0
        metric["fisher_loss"] = 8.0
        with pytest.raises(ValueError, match="marginal values drifted"):
            runner._validate_marginal(
                zero_tamper,
                proposal_rows=16,
                evaluation_rows=16,
                expected_source_sha256=source_sha256,
                expected_parent_sha256=parent_sha256,
                expected_proposal_sequence=("mutation.1",),
            )

    parent_tamper = deepcopy(marginal)
    parent_tamper["proposal_bank"][
        "parent_site_rank_one_bank_sha256"
    ] = _sha("wrong-parent")
    with pytest.raises(ValueError, match="bank summary values drifted"):
        runner._validate_marginal(
            parent_tamper,
            proposal_rows=16,
            evaluation_rows=16,
            expected_source_sha256=source_sha256,
            expected_parent_sha256=parent_sha256,
            expected_proposal_sequence=("mutation.1",),
        )

    lineage_tamper = _full_marginal(
        "mutation.1",
        source_sha256=source_sha256,
        parent_sha256=parent_sha256,
        proposal_sequence=("mutation.3", "mutation.1"),
    )
    with pytest.raises(ValueError, match="proposal lineage drifted"):
        runner._validate_marginal(
            lineage_tamper,
            proposal_rows=16,
            evaluation_rows=16,
            expected_source_sha256=source_sha256,
            expected_parent_sha256=parent_sha256,
            expected_proposal_sequence=("mutation.1",),
        )


def test_candidate_round_ledger_proves_argmax_cache_recompute_delta_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rounds, first_spectrum, steps, initial_sha, final_sha = (
        _two_round_candidate_ledger()
    )
    marginal_calls: list[tuple[str, str | None, tuple[str, ...] | None]] = []

    def record_marginal(
        value: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        sequence = kwargs.get("expected_proposal_sequence")
        marginal_calls.append(
            (
                str(value["invocation_id"]),
                None
                if kwargs.get("expected_parent_sha256") is None
                else str(kwargs["expected_parent_sha256"]),
                None if sequence is None else tuple(sequence),
            )
        )
        return value

    monkeypatch.setattr(runner, "_validate_marginal", record_marginal)
    initial_site_losses = [
        [invocation_id, 0.5]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    initial_fit = {
        "per_invocation_relative_fisher_loss": initial_site_losses,
    }
    initial_evaluation = {
        "zero_predictor_fisher_loss": 10.0,
        "per_invocation_relative_fisher_loss": initial_site_losses,
    }
    assert runner._validate_candidate_rounds(
        rounds,
        initial_bank_sha256=initial_sha,
        final_bank_sha256=final_sha,
        expected_parent_sha256=_sha("ledger-parent"),
        first_spectrum=first_spectrum,
        steps=steps,
        initial_fit=initial_fit,
        initial_evaluation=initial_evaluation,
    ) is rounds
    assert len(marginal_calls) == 36
    assert all(
        parent_sha256 == _sha("ledger-parent")
        for _invocation_id, parent_sha256, _sequence in marginal_calls
    )
    assert [sequence for _invocation_id, _parent, sequence in marginal_calls[:18]] == [
        (invocation_id,) for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    assert marginal_calls[18][2] == ("mutation.1", "mutation.1")
    assert all(
        sequence is None
        for _invocation_id, _parent, sequence in marginal_calls[19:]
    )
    assert all(len(round_value["candidates"]) == 18 for round_value in rounds)
    assert rounds[0]["selected_invocation_id"] == "mutation.1"
    assert rounds[1]["selected_invocation_id"] == "mutation.3"
    second_by_id = {
        item["invocation_id"]: item for item in rounds[1]["candidates"]
    }
    assert second_by_id["mutation.1"]["used_cached_off_site_marginal"] is False
    assert second_by_id["mutation.1"]["source_bank_sha256"] == (
        rounds[1]["current_bank_sha256"]
    )
    assert all(
        candidate["used_cached_off_site_marginal"] is True
        for invocation_id, candidate in second_by_id.items()
        if invocation_id != "mutation.1"
    )
    assert all(
        candidate["eligible_for_all_layer_allocation"] is True
        for round_value in rounds
        for candidate in round_value["candidates"]
    )
    assert steps[0]["selection_all_layer_normalized_fisher_gain"] == pytest.approx(
        steps[0]["allocation_validation_before"]["relative_fisher_loss"]
        - steps[0]["allocation_validation_after"]["relative_fisher_loss"],
        abs=1.0e-15,
        rel=0.0,
    )

    ledger_sha = runner._candidate_rounds_artifact_sha256(rounds)
    assert len(ledger_sha) == 64
    hash_tamper = deepcopy(rounds)
    hash_tamper[0]["selected_invocation_id"] = "mutation.3"
    assert runner._candidate_rounds_artifact_sha256(hash_tamper) != ledger_sha
    with pytest.raises(ValueError, match="argmax or tie-break"):
        runner._validate_candidate_rounds(
            hash_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    cache_tamper = deepcopy(rounds)
    drifted_offsite = deepcopy(
        cache_tamper[1]["candidates"][2]["marginal"]
    )
    drifted_offsite["cache_nonce"] = "drift"
    cache_tamper[1]["candidates"][2]["marginal"] = drifted_offsite
    with pytest.raises(ValueError, match="off-site cached candidate drifted"):
        runner._validate_candidate_rounds(
            cache_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    recompute_tamper = deepcopy(rounds)
    selected_site = recompute_tamper[1]["candidates"][0]
    selected_site["source_bank_sha256"] = initial_sha
    selected_site["marginal"]["source_bank_sha256"] = initial_sha
    selected_site["used_cached_off_site_marginal"] = True
    with pytest.raises(ValueError, match="selected-site candidate was not recomputed"):
        runner._validate_candidate_rounds(
            recompute_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    current_fit_tamper = deepcopy(rounds)
    current_fit_tamper[1]["candidates"][0]["marginal"][
        "proposal_fit_current_oracle"
    ]["relative_fisher_loss"] = 0.41
    with pytest.raises(ValueError, match="candidate values drifted"):
        runner._validate_candidate_rounds(
            current_fit_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    current_evaluation_tamper = deepcopy(rounds)
    current_evaluation_tamper[1]["candidates"][0]["marginal"][
        "allocation_evaluation_current_oracle"
    ]["relative_fisher_loss"] = 0.41
    with pytest.raises(ValueError, match="candidate values drifted"):
        runner._validate_candidate_rounds(
            current_evaluation_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    delta_tamper = deepcopy(steps)
    delta_tamper[0]["allocation_validation_after"]["relative_fisher_loss"] = 0.44
    with pytest.raises(ValueError, match="selected step does not close"):
        runner._validate_candidate_rounds(
            rounds,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=delta_tamper,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    eligibility_tamper = deepcopy(rounds)
    eligibility_tamper[0]["candidates"][0][
        "eligible_for_all_layer_allocation"
    ] = False
    with pytest.raises(ValueError, match="candidate values drifted"):
        runner._validate_candidate_rounds(
            eligibility_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    reported_gain_tamper = deepcopy(rounds)
    reported_gain_tamper[0]["candidates"][2][
        "normalized_fisher_mass_gain"
    ] += 1.0e-16
    with pytest.raises(ValueError, match="candidate values drifted"):
        runner._validate_candidate_rounds(
            reported_gain_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=final_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=first_spectrum,
            steps=steps,
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    empty_marginals = [
        _ledger_marginal(
            invocation_id,
            source_sha256=initial_sha,
            normalized_gain=0.0,
            strictly_improves=False,
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    empty_rounds = [
        {
            "allocation_index": 0,
            "current_bank_sha256": initial_sha,
            "evaluation_row_count": 1_728,
            "evaluation_zero_predictor_fisher_loss": 10.0,
            "minimum_relative_fisher_gain": 1.0e-12,
            "tie_tolerance": 1.0e-15,
            "candidates": [
                _ledger_candidate(
                    marginal,
                    current_bank_sha256=initial_sha,
                    normalized_gain=0.0,
                )
                for marginal in empty_marginals
            ],
            "selected_invocation_id": None,
            "early_stop_no_eligible_candidate": True,
        }
    ]
    empty_first_spectrum = {"per_site": empty_marginals}
    assert runner._validate_candidate_rounds(
        empty_rounds,
        initial_bank_sha256=initial_sha,
        final_bank_sha256=initial_sha,
        expected_parent_sha256=_sha("ledger-parent"),
        first_spectrum=empty_first_spectrum,
        steps=[],
        initial_fit=initial_fit,
        initial_evaluation=initial_evaluation,
    ) is empty_rounds
    with pytest.raises(ValueError, match="final bank lineage"):
        runner._validate_candidate_rounds(
            empty_rounds,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=_sha("wrong-empty-final"),
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum=empty_first_spectrum,
            steps=[],
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )

    subthreshold_gain = 5.0e-13
    subthreshold_marginals = [
        _ledger_marginal(
            invocation_id,
            source_sha256=initial_sha,
            normalized_gain=subthreshold_gain,
            strictly_improves=True,
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    subthreshold_rounds = [
        {
            "allocation_index": 0,
            "current_bank_sha256": initial_sha,
            "evaluation_row_count": 1_728,
            "evaluation_zero_predictor_fisher_loss": 10.0,
            "minimum_relative_fisher_gain": 1.0e-12,
            "tie_tolerance": 1.0e-15,
            "candidates": [
                _ledger_candidate(
                    marginal,
                    current_bank_sha256=initial_sha,
                    normalized_gain=subthreshold_gain,
                )
                for marginal in subthreshold_marginals
            ],
            "selected_invocation_id": None,
            "early_stop_no_eligible_candidate": True,
        }
    ]
    assert all(
        candidate["strictly_improves_all_protocol_gates"] is True
        and candidate["eligible_for_all_layer_allocation"] is False
        for candidate in subthreshold_rounds[0]["candidates"]
    )
    assert runner._validate_candidate_rounds(
        subthreshold_rounds,
        initial_bank_sha256=initial_sha,
        final_bank_sha256=initial_sha,
        expected_parent_sha256=_sha("ledger-parent"),
        first_spectrum={"per_site": subthreshold_marginals},
        steps=[],
        initial_fit=initial_fit,
        initial_evaluation=initial_evaluation,
    ) is subthreshold_rounds
    subthreshold_tamper = deepcopy(subthreshold_rounds)
    subthreshold_tamper[0]["candidates"][0][
        "eligible_for_all_layer_allocation"
    ] = True
    with pytest.raises(ValueError, match="candidate values drifted"):
        runner._validate_candidate_rounds(
            subthreshold_tamper,
            initial_bank_sha256=initial_sha,
            final_bank_sha256=initial_sha,
            expected_parent_sha256=_sha("ledger-parent"),
            first_spectrum={"per_site": subthreshold_marginals},
            steps=[],
            initial_fit=initial_fit,
            initial_evaluation=initial_evaluation,
        )


def test_greedy_and_fixed_steps_close_selected_offsite_gain_and_result_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    greedy, fixed = _one_step_proof_payloads()
    monkeypatch.setattr(
        runner,
        "_validate_first_spectrum",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "_validate_marginal",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "_validate_candidate_rounds",
        lambda value, **_kwargs: value,
    )

    assert runner._validate_greedy_allocation(greedy) is greedy
    fixed_sequence = tuple(fixed["allocation_profile"])
    fixed_profile_sha256 = str(fixed["frozen_profile_artifact_sha256"])
    fixed_candidate_sha256 = str(
        fixed["selection_candidate_rounds_artifact_sha256"]
    )
    assert runner._validate_fixed_profile_replay(
        fixed,
        frozen_sequence=fixed_sequence,
        frozen_profile_sha256=fixed_profile_sha256,
        frozen_candidate_rounds_sha256=fixed_candidate_sha256,
    ) is fixed

    for marginal_metric, changed_value in (
        ("proposal_fit_current_oracle", 0.49),
        ("proposal_fit_candidate_oracle", 0.41),
        ("allocation_evaluation_current_oracle", 0.49),
        ("allocation_evaluation_candidate_oracle", 0.41),
    ):
        tampered = deepcopy(greedy)
        tampered["steps"][0]["marginal"][marginal_metric][
            "relative_fisher_loss"
        ] = changed_value
        with pytest.raises(ValueError, match="step metrics do not close"):
            runner._validate_greedy_allocation(tampered)

        fixed_tamper = deepcopy(fixed)
        fixed_tamper["steps"][0]["marginal"][marginal_metric][
            "relative_fisher_loss"
        ] = changed_value
        with pytest.raises(ValueError, match="step metrics do not close"):
            runner._validate_fixed_profile_replay(
                fixed_tamper,
                frozen_sequence=fixed_sequence,
                frozen_profile_sha256=fixed_profile_sha256,
                frozen_candidate_rounds_sha256=fixed_candidate_sha256,
            )

    for after_metric in (
        "direction_train_after",
        "allocation_validation_after",
    ):
        offsite_tamper = deepcopy(greedy)
        pairs = offsite_tamper["steps"][0][after_metric][
            "per_invocation_relative_fisher_loss"
        ]
        next(pair for pair in pairs if pair[0] == "mutation.3")[1] = 0.49
        with pytest.raises(ValueError, match="step metrics do not close"):
            runner._validate_greedy_allocation(offsite_tamper)

    fixed_offsite_tamper = deepcopy(fixed)
    fixed_pairs = fixed_offsite_tamper["steps"][0]["full_fit_after"][
        "per_invocation_relative_fisher_loss"
    ]
    next(pair for pair in fixed_pairs if pair[0] == "mutation.3")[1] = 0.49
    with pytest.raises(ValueError, match="step metrics do not close"):
        runner._validate_fixed_profile_replay(
            fixed_offsite_tamper,
            frozen_sequence=fixed_sequence,
            frozen_profile_sha256=fixed_profile_sha256,
            frozen_candidate_rounds_sha256=fixed_candidate_sha256,
        )

    greedy_gain_tamper = deepcopy(greedy)
    greedy_gain_tamper["steps"][0][
        "selection_all_layer_normalized_fisher_gain"
    ] = 0.04
    with pytest.raises(ValueError, match="step metrics do not close"):
        runner._validate_greedy_allocation(greedy_gain_tamper)
    fixed_gain_tamper = deepcopy(fixed)
    fixed_gain_tamper["steps"][0]["all_layer_relative_fisher_gain"] = 0.04
    with pytest.raises(ValueError, match="step metrics do not close"):
        runner._validate_fixed_profile_replay(
            fixed_gain_tamper,
            frozen_sequence=fixed_sequence,
            frozen_profile_sha256=fixed_profile_sha256,
            frozen_candidate_rounds_sha256=fixed_candidate_sha256,
        )

    greedy_result_tamper = deepcopy(greedy)
    greedy_result_tamper["steps"][0]["marginal"]["proposal_bank"][
        "artifact_sha256"
    ] = _sha("wrong-greedy-proposal")
    with pytest.raises(ValueError, match="step metrics do not close"):
        runner._validate_greedy_allocation(greedy_result_tamper)
    fixed_result_tamper = deepcopy(fixed)
    fixed_result_tamper["steps"][0]["marginal"]["proposal_bank"][
        "artifact_sha256"
    ] = _sha("wrong-fixed-proposal")
    with pytest.raises(ValueError, match="step metrics do not close"):
        runner._validate_fixed_profile_replay(
            fixed_result_tamper,
            frozen_sequence=fixed_sequence,
            frozen_profile_sha256=fixed_profile_sha256,
            frozen_candidate_rounds_sha256=fixed_candidate_sha256,
        )


def test_greedy_validator_accepts_repeated_site_and_names_failed_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    greedy, _fixed = _one_step_proof_payloads()
    monkeypatch.setattr(
        runner,
        "_validate_first_spectrum",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "_validate_marginal",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(
        runner,
        "_validate_candidate_rounds",
        lambda value, **_kwargs: value,
    )

    sequence = ("mutation.1", "mutation.1")
    parent_sha256 = greedy["final_bank"][
        "parent_site_rank_one_bank_sha256"
    ]
    intermediate_sha256 = greedy["steps"][0]["result_bank_sha256"]
    final_sha256 = _sha("step-final-repeated-site")
    final_bank = _bank_summary(
        artifact_sha256=final_sha256,
        parent_sha256=parent_sha256,
        sequence=sequence,
    )
    after_by_id = {
        invocation_id: (0.3 if invocation_id == "mutation.1" else 0.5)
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    }
    final_fit = _fit_metric_payload(
        runner._DIRECTION_TRAIN_ROWS,
        0.40,
        after_by_id,
    )
    final_evaluation = _fit_metric_payload(
        runner._ALLOCATION_VALIDATION_ROWS,
        0.40,
        after_by_id,
    )
    repeated_marginal = {
        "proposal_fit_current_oracle": {"relative_fisher_loss": 0.4},
        "proposal_fit_candidate_oracle": {"relative_fisher_loss": 0.3},
        "allocation_evaluation_current_oracle": {
            "relative_fisher_loss": 0.4
        },
        "allocation_evaluation_candidate_oracle": {
            "relative_fisher_loss": 0.3
        },
        "proposal_bank": {"artifact_sha256": final_sha256},
    }
    greedy["steps"].append(
        {
            "allocation_index": 1,
            "invocation_id": "mutation.1",
            "source_bank_sha256": intermediate_sha256,
            "result_bank_sha256": final_sha256,
            "marginal_source_bank_sha256": intermediate_sha256,
            "used_cached_off_site_marginal": False,
            "selection_all_layer_normalized_fisher_gain": 0.05,
            "marginal": repeated_marginal,
            "direction_train_before": greedy["final_direction_train_oracle"],
            "direction_train_after": final_fit,
            "allocation_validation_before": greedy[
                "final_allocation_validation_oracle"
            ],
            "allocation_validation_after": final_evaluation,
        }
    )
    greedy.update(
        {
            "final_bank": final_bank,
            "budget_exhausted": True,
            "frozen_allocation_sequence": list(sequence),
            "frozen_allocated_rank_profile": final_bank[
                "allocated_rank_profile"
            ],
            "frozen_total_rank_profile": final_bank["total_rank_profile"],
            "frozen_profile_artifact_sha256": runner._profile_artifact_sha256(
                sequence,
                final_bank["allocated_rank_profile"],
                final_bank["total_rank_profile"],
                greedy["candidate_rounds_artifact_sha256"],
            ),
            "final_direction_train_oracle": final_fit,
            "final_allocation_validation_oracle": final_evaluation,
            "direction_train_relative_fisher_loss_history": [0.5, 0.45, 0.40],
            "allocation_validation_relative_fisher_loss_history": [
                0.5,
                0.45,
                0.40,
            ],
        }
    )

    assert runner._validate_greedy_allocation(greedy) is greedy

    tampered = deepcopy(greedy)
    pairs = tampered["steps"][1]["allocation_validation_after"][
        "per_invocation_relative_fisher_loss"
    ]
    next(pair for pair in pairs if pair[0] == "mutation.3")[1] = 0.49
    with pytest.raises(
        ValueError,
        match=(
            "allocation_index=1 invocation_id=mutation.1 "
            "clause=off_site_allocation_validation "
            "off_site_invocation_id=mutation.3"
        ),
    ):
        runner._validate_greedy_allocation(tampered)


def test_report_schema_and_digest_are_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.REPORT_SCHEMA == (
        "fisher_graph.gemma3_attention_transport."
        "all_layer_residual_spectrum_a_diagnostic.v1"
    )
    assert runner._TOP_LEVEL_FIELDS == {
        "schema",
        "format_version",
        "scientific_status",
        "exposure_binding",
        "model",
        "protocol",
        "A_partition",
        "selection",
        "candidate",
        "outer_A_dev_diagnostic",
        "runtime_counters",
        "runtime_resources",
        "report_sha256",
    }
    observed: dict[str, object] = {}

    def validate(value: object) -> dict[str, object]:
        assert isinstance(value, dict)
        observed.update(value)
        return value

    monkeypatch.setattr(runner, "_validate_report", validate)
    unsealed = {"schema": runner.REPORT_SCHEMA, "format_version": 1}
    sealed = runner._seal_report(unsealed)
    assert sealed["report_sha256"] == runner._digest(unsealed)
    assert observed == sealed
    with pytest.raises(ValueError, match="already has a digest"):
        runner._seal_report(sealed)


def test_outer_payload_closes_all_local_global_per_layer_burden_schema() -> None:
    parent = _outer_bank_evaluation(allocated=False)
    allocated = _outer_bank_evaluation(allocated=True)
    parent_by_id = dict(parent.per_invocation)
    allocated_by_id = dict(allocated.per_invocation)
    total_rows = parent.all_layers.output_metrics.row_count
    all_zero = parent.all_layers.zero_predictor_fisher_loss
    gains = tuple(
        (
            invocation_id,
            144
            * (
                parent_by_id[invocation_id].output_metrics.fisher_loss
                - allocated_by_id[invocation_id].output_metrics.fisher_loss
            )
            / total_rows
            / all_zero,
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    )
    payload = runner._outer_diagnostic_payload(
        SimpleNamespace(
            invocation_ids=runner._NUMERIC_INVOCATION_IDS,
            parent_bank_sha256=_sha("parent"),
            allocated_bank_sha256=_sha("allocated"),
            allocation_profile=("mutation.1", "mutation.3"),
            active_generator_count_limit=2,
            coefficient_ridge=0.0,
            parent=parent,
            allocated=allocated,
            per_invocation_all_layer_normalized_fisher_gains=gains,
        )
    )
    assert runner._validate_outer_diagnostic(payload) is payload
    assert [item["invocation_id"] for item in payload["per_layer"]] == list(
        runner._NUMERIC_INVOCATION_IDS
    )
    summary = payload["summary"]
    assert summary["current_run_outer_panel_previously_exposed_exactly"] is False
    assert summary["A_corpus_previously_exposed"] is True
    assert summary[
        "prior_A_dev_evidence_influenced_architecture_design"
    ] is True
    assert summary[
        "outer_panel_used_for_current_run_selection_or_fitting"
    ] is False
    assert summary["readout_interpretation"] == (
        "descriptive_replay_only_not_fresh_holdout_or_confirmation"
    )
    ranking = payload["absolute_burden_ranking"]
    assert [item["rank"] for item in ranking] == list(range(1, 19))
    assert sum(
        item["absolute_fisher_burden_fraction"] for item in ranking
    ) == pytest.approx(1.0)
    assert payload["retention"]["contains_residual_factor_tensors"] is False

    tampered = deepcopy(payload)
    tampered["per_layer"][0]["selected_component_count"] = 0
    with pytest.raises(ValueError, match="layer schema"):
        runner._validate_outer_diagnostic(tampered)

    scope_zero_tamper = deepcopy(payload)
    scope_metric = scope_zero_tamper["aggregate"]["all_layers"][
        "allocated_residual"
    ]
    scope_metric["zero_predictor_fisher_loss"] *= 2.0
    scope_metric["relative_fisher_loss"] = (
        scope_metric["fisher_loss"]
        / scope_metric["zero_predictor_fisher_loss"]
    )
    scope_metric["normalized_rmse"] = (
        scope_metric["relative_fisher_loss"] ** 0.5
    )
    with pytest.raises(ValueError, match="scope zero drifted"):
        runner._validate_outer_diagnostic(scope_zero_tamper)

    layer_zero_tamper = deepcopy(payload)
    layer_metric = layer_zero_tamper["per_layer"][0]["allocated_residual"]
    layer_metric["zero_predictor_fisher_loss"] *= 2.0
    layer_metric["relative_fisher_loss"] = (
        layer_metric["fisher_loss"]
        / layer_metric["zero_predictor_fisher_loss"]
    )
    layer_metric["normalized_rmse"] = (
        layer_metric["relative_fisher_loss"] ** 0.5
    )
    with pytest.raises(ValueError, match="layer zero drifted"):
        runner._validate_outer_diagnostic(layer_zero_tamper)


def test_publication_is_canonical_no_overwrite_and_preserves_foreign_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "_validate_report", lambda value: value)
    monkeypatch.setattr(
        runner,
        "_strict_report",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    report = {"schema": "test", "report_sha256": _sha("report")}
    output = tmp_path / "report.json"
    runner._publish_report(output, report, publication_guard=lambda _phase: None)
    assert output.read_bytes() == runner._canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        runner._publish_report(
            output,
            report,
            publication_guard=lambda _phase: None,
        )

    raced = tmp_path / "raced.json"
    owned = tmp_path / "owned.json"

    def replacement_guard(phase_id: str) -> None:
        if phase_id == "residual_spectrum_publication_verified":
            raced.rename(owned)
            raced.write_bytes(b"foreign replacement")

    with pytest.raises((RuntimeError, ValueError, json.JSONDecodeError)):
        runner._publish_report(
            raced,
            report,
            publication_guard=replacement_guard,
        )
    assert raced.read_bytes() == b"foreign replacement"
    assert owned.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_strict_reader_rejects_duplicate_nonfinite_and_linked_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="duplicate JSON field"):
        runner._strict_json_loads(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="non-finite"):
        runner._strict_json_loads(b'{"value":NaN}')

    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"test"}')
    symlink = tmp_path / "symlink.json"
    os.symlink(source, symlink)
    with pytest.raises(ValueError, match="regular file"):
        runner._strict_report(symlink)
    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="identity or size"):
        runner._read_no_follow_regular_bytes(source)


def test_mocked_run_captures_A_once_freezes_inner_profile_before_outer_dev(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fisher_graph.gemma3_attention_transport_all_layer_residual_spectrum as spectrum
    import fisher_graph.gemma3_attention_transport_experiment as experiment
    import fisher_graph.gemma3_attention_transport_experiment_core as experiment_core
    import fisher_graph.gemma3_attention_transport_global_site_rank1 as site_rank_one
    import fisher_graph.gemma3_attention_transport_runtime as runtime_module
    import fisher_graph.gemma3_chart_native_neighbor_experiment as neighbor
    import fisher_graph.gemma3_chart_native_neighbor_fresh_panels as panels

    fit_role = neighbor.FIT_ROLE
    records = _records()
    events: list[str] = []
    panels_seen: list[object] = []
    published: dict[str, object] = {}
    root = Path(__file__).resolve().parents[1]

    class FakePanelSource:
        def __init__(self, **_kwargs: object) -> None:
            self.opened_roles: tuple[str, ...] = ()
            self.current_role: str | None = None
            self.next_role: str | None = fit_role
            self.freeze_receipts: dict[str, object] = {}
            panels_seen.append(self)

    counters = {
        "role_batch_counts": [
            ["A_chart_transport_fit", 45],
            ["B_neighbor_discovery", 0],
            ["C_neighbor_confirmation", 0],
        ],
        "chart_backbone_forward_count": 45,
        "chart_terminal_cotangent_vjp_count": 45,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_raw_activations": False,
        "contains_raw_gradients": False,
        "contains_logits": False,
    }

    class FakeRuntime:
        counters = SimpleNamespace(payload=lambda: counters)

        def runtime_qualification_receipt(self) -> object:
            return SimpleNamespace(
                model_id="google/gemma-3-270m",
                model_revision=(
                    "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
                ),
                hidden_width=640,
                qualification_receipt_sha256=_sha("qualification"),
            )

        def close(self) -> None:
            events.append("runtime-close")

    preflight = SimpleNamespace(
        binding=SimpleNamespace(
            repository_root=root,
            storage_budget=SimpleNamespace(
                maximum_resident_memory_bytes=8 * 1024**3
            ),
        ),
        fresh_manifest={},
        fresh_receipt=object(),
        opaque_panel_receipt=object(),
        receipt_sha256=_sha("preflight"),
        validate_receipt=lambda: events.append("preflight"),
    )

    def fake_capture_role(**kwargs: object) -> tuple[tuple[object, ...], str]:
        panel = kwargs["panel_source"]
        assert kwargs["runtime_role"] == fit_role
        assert kwargs["core_role"] == "A"
        panel.opened_roles = (fit_role,)
        panel.current_role = fit_role
        panel.next_role = fit_role
        events.append("capture-A")
        return records, _sha("dense-parity")

    inner_parent_bank = SimpleNamespace(artifact_sha256=_sha("inner-parent"))
    full_parent_bank = SimpleNamespace(artifact_sha256=_sha("full-parent"))
    inner_parent_fit = SimpleNamespace(bank=inner_parent_bank)
    full_parent_fit = SimpleNamespace(bank=full_parent_bank)

    def fake_parent_fit(captures: tuple[object, ...], **_kwargs: object) -> object:
        if len(captures) == 30 * 18:
            events.append("inner-parent-fit")
            return inner_parent_fit
        assert len(captures) == 36 * 18
        events.append("full-parent-fit")
        return full_parent_fit

    frozen_profile = ("mutation.1", "mutation.3")
    greedy = SimpleNamespace(
        allocation_profile=frozen_profile,
        budget_exhausted=True,
    )

    def fake_greedy(
        bank: object,
        direction: tuple[object, ...],
        validation: tuple[object, ...],
        **_kwargs: object,
    ) -> object:
        assert bank is inner_parent_bank
        assert len(direction) == 30 * 18
        assert len(validation) == 6 * 18
        events.append("inner-profile-frozen")
        return greedy

    final_bank = SimpleNamespace(
        allocated_invocation_ids=frozen_profile,
        artifact_sha256=_sha("allocated"),
        stored_parameter_count=134_208,
        state_dict=lambda: {
            "contains_native_value_or_output_weights": False,
            "contains_oracle_outputs_or_coefficients": False,
            "contains_raw_teacher_outputs_or_gradients": False,
        },
    )
    replay = SimpleNamespace(final_bank=final_bank)

    def fake_replay(
        bank: object,
        captures: tuple[object, ...],
        profile: tuple[str, ...],
        **_kwargs: object,
    ) -> object:
        assert bank is full_parent_bank
        assert len(captures) == 36 * 18
        assert profile == frozen_profile
        assert events.index("inner-profile-frozen") < events.index(
            "full-parent-fit"
        )
        events.append("full-fit-replay")
        return replay

    def fake_outer(
        parent: object,
        allocated: object,
        captures: tuple[object, ...],
        **_kwargs: object,
    ) -> object:
        assert parent is full_parent_bank
        assert allocated is final_bank
        assert len(captures) == 9 * 18
        assert events[-1] == "full-fit-replay"
        events.append("outer-dev-once")
        return object()

    outer_payload = {
        "summary": {
            "all_layer_parent_minus_allocated_relative_fisher_loss": 0.1,
            "local_parent_minus_allocated_relative_fisher_loss": 0.08,
            "global_parent_minus_allocated_relative_fisher_loss": 0.02,
            "current_run_outer_panel_previously_exposed_exactly": False,
            "A_corpus_previously_exposed": True,
            "prior_A_dev_evidence_influenced_architecture_design": True,
            "outer_panel_used_for_current_run_selection_or_fitting": False,
            "readout_interpretation": (
                "descriptive_replay_only_not_fresh_holdout_or_confirmation"
            ),
        }
    }

    def fake_seal(value: object) -> dict[str, object]:
        assert isinstance(value, dict)
        assert set(value) == runner._TOP_LEVEL_FIELDS - {"report_sha256"}
        sealed = dict(value)
        sealed["report_sha256"] = runner._digest(value)
        return sealed

    def fake_publish(
        _path: Path,
        report: dict[str, object],
        *,
        publication_guard: object,
    ) -> None:
        publication_guard("residual_spectrum_temporary_sealed")
        publication_guard("residual_spectrum_publication_verified")
        published.update(report)
        events.append("publish")

    monkeypatch.setattr(neighbor, "static_preflight", lambda _binding: preflight)
    monkeypatch.setattr(
        neighbor,
        "validate_runtime_qualification",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        neighbor,
        "observe_phase_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            peak_resident_set_bytes=1024
        ),
    )
    monkeypatch.setattr(experiment, "_capture_role", fake_capture_role)
    monkeypatch.setattr(
        experiment,
        "_check_output_publication_headroom",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        experiment,
        "_check_resident_set_budget",
        lambda *_args, **_kwargs: 1024,
    )
    monkeypatch.setattr(
        experiment_core,
        "_open_split_panel",
        lambda _records, *, role: SimpleNamespace(
            artifact_sha256=_sha(f"split-{role}")
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_pinned_gemma3_attention_transport_capture_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(panels, "SequentialPanelSource", FakePanelSource)
    monkeypatch.setattr(
        site_rank_one,
        "fit_global_site_rank_one_attention_transport_generator_bank",
        fake_parent_fit,
    )
    monkeypatch.setattr(
        spectrum,
        "greedily_allocate_residual_spectrum",
        fake_greedy,
    )
    monkeypatch.setattr(
        spectrum,
        "replay_fixed_residual_allocation_profile",
        fake_replay,
    )
    monkeypatch.setattr(
        spectrum,
        "evaluate_allocated_residual_bank_all_layers",
        fake_outer,
    )
    monkeypatch.setattr(runner, "_validate_prior_targeted_report", lambda _root: {})
    monkeypatch.setattr(
        runner,
        "_source_sha256s",
        lambda _root: {path: _sha(path) for path in runner._SOURCE_PATHS},
    )
    monkeypatch.setattr(
        runner,
        "_outer_partition_artifact_sha256",
        lambda _fit, _dev: _sha("current-run-outer-split"),
    )
    monkeypatch.setattr(
        runner,
        "_site_rank_one_parent_fit_payload",
        lambda value: {"bank": value.bank.artifact_sha256},
    )
    monkeypatch.setattr(
        runner,
        "_greedy_allocation_payload",
        lambda _value: {
            "frozen_profile_artifact_sha256": _sha("frozen-profile"),
            "candidate_rounds_artifact_sha256": _sha("candidate-rounds"),
        },
    )
    monkeypatch.setattr(
        runner,
        "_fixed_profile_replay_payload",
        lambda _value, **_kwargs: {"profile": list(frozen_profile)},
    )
    monkeypatch.setattr(
        runner,
        "_outer_diagnostic_payload",
        lambda _value, **_kwargs: outer_payload,
    )
    monkeypatch.setattr(runner, "_seal_report", fake_seal)
    monkeypatch.setattr(runner, "_publish_report", fake_publish)

    result = runner.run_all_layer_residual_spectrum_A_diagnostic(
        binding=object(),
        report_path=tmp_path / "report.json",
    )
    assert result["B_opened_by_this_run"] is False
    assert result["C_ever_opened"] is False
    assert result["allocation_profile"] == list(frozen_profile)
    assert events.count("capture-A") == 1
    assert events.count("outer-dev-once") == 1
    assert not any("capture-B" in event or "capture-C" in event for event in events)
    assert events.index("inner-profile-frozen") < events.index("full-fit-replay")
    assert events.index("full-fit-replay") < events.index("outer-dev-once")
    assert events[-1] == "publish"
    assert len(panels_seen) == 1
    panel = panels_seen[0]
    assert panel.opened_roles == (fit_role,)
    assert panel.current_role == fit_role
    assert panel.next_role == fit_role
    assert panel.freeze_receipts == {}
    partition = published["A_partition"]
    assert len(partition["outer"]["fit_receipt_sha256s"]) == 36
    assert len(partition["outer"]["dev_receipt_sha256s"]) == 9
    assert len(partition["inner"]["direction_train_receipt_sha256s"]) == 30
    assert len(
        partition["inner"]["allocation_validation_receipt_sha256s"]
    ) == 6
    assert partition["inner_is_strict_partition_of_outer_fit"] is True
    assert partition["outer_dev_excluded_from_inner_partition"] is True
    assert partition["outer"]["matches_prior_targeted_A_split"] is False
    assert partition["outer"]["cross_run_split_identity_required"] is False
    status = published["scientific_status"]
    assert status["A_corpus_previously_exposed"] is True
    assert status[
        "prior_A_dev_evidence_influenced_architecture_design"
    ] is True
    assert status[
        "current_run_outer_A_dev_receipts_previously_exposed_exactly"
    ] is False
    assert status["outer_A_dev_readout_descriptive_replay_only"] is True
    assert status[
        "outer_A_dev_used_for_current_run_selection_or_fitting"
    ] is False
    assert status["cross_run_A_split_identity_claimed"] is False
    assert status["confirmation_claimed"] is False
    assert status["added_parameter_budget_matches_prior_targeted"] is True
    exposure = published["exposure_binding"]
    assert exposure["A_corpus_previously_exposed"] is True
    assert exposure[
        "prior_A_dev_evidence_influenced_architecture_design"
    ] is True
    assert exposure[
        "current_run_outer_A_dev_receipts_previously_exposed_exactly"
    ] is False
    assert exposure["outer_A_dev_readout_descriptive_replay_only"] is True
    assert exposure["outer_A_dev_is_fresh_holdout"] is False
    assert exposure["outer_A_dev_is_confirmation"] is False
    assert exposure[
        "current_run_A_split_reuses_prior_algorithm_without_requiring_receipt_identity"
    ] is True
    assert exposure["prior_targeted_second_residual_parameters"] == 20_480
    assert exposure["interpretation"].endswith(
        "descriptive_not_fresh_holdout_or_confirmation"
    )
    outer_evaluation = published["protocol"]["outer_evaluation"]
    spectrum_protocol = published["protocol"]["residual_spectrum"]
    assert published["protocol"]["A_corpus_policy"] == (
        "same_pinned_A_chart_transport_corpus_as_targeted_v1"
    )
    assert published["protocol"]["outer_partition_policy"] == (
        "same_partition_algorithm_as_targeted_cross_run_receipt_identity_not_required_v1"
    )
    assert spectrum_protocol["residual_component_parameters"] == 10_240
    assert spectrum_protocol["maximum_allocated_residual_parameters"] == 20_480
    assert spectrum_protocol["prior_targeted_second_residual_parameters"] == 20_480
    assert spectrum_protocol[
        "matches_prior_targeted_second_residual_parameter_budget"
    ] is True
    assert outer_evaluation["evaluation_role"] == (
        "current_run_outer_A_dev_descriptive_replay_only"
    )
    assert outer_evaluation[
        "current_run_outer_receipts_previously_exposed_exactly"
    ] is False
    assert outer_evaluation["A_corpus_previously_exposed"] is True
    assert outer_evaluation[
        "prior_A_dev_evidence_influenced_architecture_design"
    ] is True
    assert outer_evaluation[
        "used_for_current_run_selection_or_fitting"
    ] is False
    assert outer_evaluation["fresh_holdout"] is False
    assert outer_evaluation["confirmation_readout"] is False
    outer_summary = published["outer_A_dev_diagnostic"]["summary"]
    assert outer_summary[
        "current_run_outer_panel_previously_exposed_exactly"
    ] is False
    assert outer_summary["A_corpus_previously_exposed"] is True
    assert outer_summary[
        "prior_A_dev_evidence_influenced_architecture_design"
    ] is True
    assert outer_summary[
        "outer_panel_used_for_current_run_selection_or_fitting"
    ] is False
    assert outer_summary["readout_interpretation"] == (
        "descriptive_replay_only_not_fresh_holdout_or_confirmation"
    )
    candidate = published["candidate"]
    assert candidate["allocated_residual_parameters"] == 20_480
    assert candidate["maximum_allocated_residual_parameters"] == 20_480
    assert candidate[
        "matches_prior_targeted_second_residual_parameter_budget"
    ] is True

    status_tamper = deepcopy(published)
    status_tamper["scientific_status"][
        "added_parameter_budget_matches_prior_targeted"
    ] = False
    with pytest.raises(ValueError, match="scientific status drifted"):
        runner._validate_report(status_tamper)

    exposure_tamper = deepcopy(published)
    exposure_tamper["exposure_binding"][
        "prior_targeted_second_residual_parameters"
    ] = 20_479
    with pytest.raises(ValueError, match="exposure binding drifted"):
        runner._validate_report(exposure_tamper)

    for protocol_field, wrong_value in (
        ("residual_component_parameters", 10_239),
        ("maximum_allocated_residual_parameters", 20_479),
        ("prior_targeted_second_residual_parameters", 20_479),
        ("matches_prior_targeted_second_residual_parameter_budget", False),
    ):
        protocol_tamper = deepcopy(published)
        protocol_tamper["protocol"]["residual_spectrum"][
            protocol_field
        ] = wrong_value
        with pytest.raises(ValueError, match="algorithm binding drifted"):
            runner._validate_report(protocol_tamper)

    candidate_baseline = deepcopy(published)
    candidate_baseline["protocol"]["implementation_source_sha256s"][
        runner._PRIOR_TARGETED_CORE_SOURCE_PATH
    ] = runner._PRIOR_TARGETED_CORE_SOURCE_SHA256
    candidate_baseline["protocol"]["implementation_source_sha256s"][
        runner._PRIOR_TARGETED_RUNNER_SOURCE_PATH
    ] = runner._PRIOR_TARGETED_RUNNER_SOURCE_SHA256

    def validate_parent_stub(
        value: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        return {"bank_artifact_sha256": value["bank"]}

    selection_parent_sha256 = candidate_baseline["selection"][
        "inner_parent_A_fit"
    ]["bank"]
    full_parent_sha256 = candidate_baseline["candidate"][
        "full_fit_parent_bank_artifact_sha256"
    ]
    monkeypatch.setattr(runner, "_validate_parent_fit", validate_parent_stub)
    monkeypatch.setattr(
        runner,
        "_validate_greedy_allocation",
        lambda value: {
            **value,
            "final_bank": {
                "parent_site_rank_one_bank_sha256": selection_parent_sha256,
            },
            "frozen_allocation_sequence": list(frozen_profile),
        },
    )
    monkeypatch.setattr(
        runner,
        "_validate_fixed_profile_replay",
        lambda _value, **_kwargs: {
            "final_bank": {
                "parent_site_rank_one_bank_sha256": full_parent_sha256,
                "artifact_sha256": candidate_baseline["candidate"][
                    "bank_artifact_sha256"
                ],
            }
        },
    )
    for candidate_field, wrong_value in (
        ("maximum_allocated_residual_parameters", 20_479),
        ("matches_prior_targeted_second_residual_parameter_budget", False),
    ):
        candidate_tamper = deepcopy(candidate_baseline)
        candidate_tamper["candidate"][candidate_field] = wrong_value
        with pytest.raises(ValueError, match="candidate binding drifted"):
            runner._validate_report(candidate_tamper)
