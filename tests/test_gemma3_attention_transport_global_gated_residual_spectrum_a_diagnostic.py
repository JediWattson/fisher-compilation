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

from fisher_graph import (
    gemma3_attention_transport_global_gated_residual_spectrum_a_diagnostic
    as runner,
)
from fisher_graph.gemma3_attention_transport_global_gated_residual_spectrum_partition import (  # noqa: E501
    build_stable_A_batch_catalog,
    partition_stable_A_capture_records,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _pinned_partition(*, receipt_prefix: str = "runtime") -> object:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "examples/gemma3_chart_native_neighbor_fresh_manifest_v1.json")
        .read_text(encoding="utf-8")
    )
    role = manifest["roles"]["A_chart_transport_fit"]
    catalog = build_stable_A_batch_catalog(
        role_panel_binding_sha256=runner._PINNED_A_ROLE_PANEL_BINDING_SHA256,
        ordered_members=role["ordered_members"],
    )
    records = tuple(
        SimpleNamespace(
            role="A",
            base_capture_receipt_sha256=_sha(
                f"{receipt_prefix}-batch-{batch_ordinal}"
            ),
            capture=SimpleNamespace(
                invocation_id=invocation_id,
                row_count=16,
            ),
        )
        for batch_ordinal in range(45)
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    )
    return partition_stable_A_capture_records(records, catalog=catalog)


def _firewall_state() -> dict[str, object]:
    return {
        "opened_roles": ["A_chart_transport_fit"],
        "current_role": "A_chart_transport_fit",
        "next_role": "A_chart_transport_fit",
        "freeze_receipt_count": 0,
        "B_opened_by_this_run": False,
        "C_ever_opened": False,
    }


def _fit_metric(
    per_invocation_relative_fisher_loss: dict[str, float],
    *,
    row_count: int,
) -> dict[str, object]:
    ordered_ids = tuple(per_invocation_relative_fisher_loss)
    relative_fisher_loss = sum(
        per_invocation_relative_fisher_loss.values()
    ) / len(ordered_ids)
    fisher_loss = relative_fisher_loss * 10.0
    return {
        "row_count": row_count,
        "fisher_loss": fisher_loss,
        "directional_mse": fisher_loss / 2.0,
        "euclidean_mse": fisher_loss / 3.0,
        "normalized_rmse": relative_fisher_loss**0.5,
        "cosine": 0.8,
        "zero_predictor_fisher_loss": 10.0,
        "relative_fisher_loss": relative_fisher_loss,
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "solver_kind": "exact_signed_top2_sparse_oracle_v1",
        "per_invocation_relative_fisher_loss": [
            [
                invocation_id,
                per_invocation_relative_fisher_loss[invocation_id],
            ]
            for invocation_id in ordered_ids
        ],
    }


def _site_metric(relative_fisher_loss: float) -> dict[str, object]:
    fisher_loss = relative_fisher_loss * 10.0
    return {
        "row_count": runner._OUTER_FIT_RECEIPT_COUNT * 16,
        "fisher_loss": fisher_loss,
        "directional_mse": fisher_loss / 2.0,
        "euclidean_mse": fisher_loss / 3.0,
        "normalized_rmse": relative_fisher_loss**0.5,
        "cosine": 0.8,
        "zero_predictor_fisher_loss": 10.0,
        "relative_fisher_loss": relative_fisher_loss,
        "active_generator_count_limit": 2,
        "coefficient_ridge": 0.0,
        "solver_kind": "exact_signed_top2_sparse_oracle_v1",
    }


def _candidate_marginal(
    invocation_id: str,
    *,
    source_sha256: str,
    current_fisher_loss: float,
    fisher_loss_gain: float,
    proposal_current_relative_fisher_loss: float,
    proposal_relative_fisher_gain: float,
    proposal_bank_sha256: str,
    strictly_improves: bool,
) -> dict[str, object]:
    candidate_fisher_loss = current_fisher_loss - fisher_loss_gain
    proposal_candidate_relative_fisher_loss = (
        proposal_current_relative_fisher_loss
        - proposal_relative_fisher_gain
    )
    return {
        "invocation_id": invocation_id,
        "source_bank_sha256": source_sha256,
        "proposal_bank": {"artifact_sha256": proposal_bank_sha256},
        "proposal_fit_current_oracle": {
            "row_count": 480,
            "fisher_loss": proposal_current_relative_fisher_loss * 10.0,
            "zero_predictor_fisher_loss": 10.0,
            "relative_fisher_loss": proposal_current_relative_fisher_loss,
        },
        "proposal_fit_candidate_oracle": {
            "row_count": 480,
            "fisher_loss": proposal_candidate_relative_fisher_loss * 10.0,
            "zero_predictor_fisher_loss": 10.0,
            "relative_fisher_loss": proposal_candidate_relative_fisher_loss,
        },
        "allocation_evaluation_current_oracle": {
            "row_count": 96,
            "fisher_loss": current_fisher_loss,
            "zero_predictor_fisher_loss": 10.0,
            "relative_fisher_loss": current_fisher_loss / 10.0,
        },
        "allocation_evaluation_candidate_oracle": {
            "row_count": 96,
            "fisher_loss": candidate_fisher_loss,
            "zero_predictor_fisher_loss": 10.0,
            "relative_fisher_loss": candidate_fisher_loss / 10.0,
        },
        "strictly_improves_all_protocol_gates": strictly_improves,
    }


def _candidate(
    invocation_id: str,
    *,
    marginal: dict[str, object],
    current_sha256: str,
    objective_mode: str,
) -> dict[str, object]:
    current = marginal["allocation_evaluation_current_oracle"]
    proposed = marginal["allocation_evaluation_candidate_oracle"]
    assert isinstance(current, dict)
    assert isinstance(proposed, dict)
    raw_mass_gain = int(current["row_count"]) * (
        float(current["fisher_loss"]) - float(proposed["fisher_loss"])
    )
    global_gain = (
        raw_mass_gain / runner._GLOBAL_ALLOCATION_VALIDATION_ROWS / 10.0
        if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
        else 0.0
    )
    all_gain = raw_mass_gain / runner._ALLOCATION_VALIDATION_ROWS / 10.0
    in_scope = (
        invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
        if objective_mode == "global_attention"
        else True
    )
    objective_gain = global_gain if objective_mode == "global_attention" else all_gain
    strictly = bool(marginal["strictly_improves_all_protocol_gates"])
    source_sha256 = str(marginal["source_bank_sha256"])
    return {
        "invocation_id": invocation_id,
        "attention_kind": (
            "global"
            if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
            else "local"
        ),
        "source_bank_sha256": source_sha256,
        "used_cached_off_site_marginal": source_sha256 != current_sha256,
        "strictly_improves_all_protocol_gates": strictly,
        "in_objective_scope": in_scope,
        "global_normalized_fisher_mass_gain": global_gain,
        "all_layer_normalized_fisher_mass_gain": all_gain,
        "selection_objective_normalized_fisher_mass_gain": objective_gain,
        "eligible_for_selection": (
            strictly and in_scope and objective_gain > 1.0e-12
        ),
        "marginal": marginal,
    }


def _round(
    *,
    allocation_index: int,
    current_sha256: str,
    global_loss: float,
    all_loss: float,
    candidates: list[dict[str, object]],
    selected_invocation_id: str | None,
) -> dict[str, object]:
    return {
        "allocation_index": allocation_index,
        "current_bank_sha256": current_sha256,
        "objective_mode": "global_attention",
        "global_gate_active": True,
        "global_relative_fisher_threshold": 0.10,
        "global_relative_fisher_loss_before": global_loss,
        "all_layer_relative_fisher_loss_before": all_loss,
        "global_evaluation_row_count": runner._GLOBAL_ALLOCATION_VALIDATION_ROWS,
        "global_evaluation_zero_predictor_fisher_loss": 10.0,
        "all_layer_evaluation_row_count": runner._ALLOCATION_VALIDATION_ROWS,
        "all_layer_evaluation_zero_predictor_fisher_loss": 10.0,
        "minimum_relative_fisher_gain": 1.0e-12,
        "tie_tolerance": 1.0e-15,
        "candidates": candidates,
        "selected_invocation_id": selected_invocation_id,
        "early_stop_no_eligible_candidate": selected_invocation_id is None,
    }


def _valid_global_allocation() -> dict[str, object]:
    initial_sha = _sha("allocation-initial")
    final_sha = _sha("allocation-final")
    parent_sha = _sha("allocation-parent")
    sequence = ("mutation.11",)
    first_marginals = {
        invocation_id: _candidate_marginal(
            invocation_id,
            source_sha256=initial_sha,
            current_fisher_loss=(
                2.0
                if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
                else 5.6
            ),
            fisher_loss_gain=1.5 if invocation_id == "mutation.11" else 0.0,
            proposal_current_relative_fisher_loss=0.50,
            proposal_relative_fisher_gain=(
                0.18 if invocation_id == "mutation.11" else 0.0
            ),
            proposal_bank_sha256=(
                final_sha
                if invocation_id == "mutation.11"
                else _sha(f"first-proposal-{invocation_id}")
            ),
            strictly_improves=invocation_id == "mutation.11",
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    }
    first_candidates = [
        _candidate(
            invocation_id,
            marginal=first_marginals[invocation_id],
            current_sha256=initial_sha,
            objective_mode="global_attention",
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    second_marginals = dict(first_marginals)
    second_marginals["mutation.11"] = _candidate_marginal(
        "mutation.11",
        source_sha256=final_sha,
        current_fisher_loss=0.5,
        fisher_loss_gain=0.0,
        proposal_current_relative_fisher_loss=0.32,
        proposal_relative_fisher_gain=0.0,
        proposal_bank_sha256=_sha("second-proposal-mutation.11"),
        strictly_improves=False,
    )
    second_candidates = [
        _candidate(
            invocation_id,
            marginal=second_marginals[invocation_id],
            current_sha256=final_sha,
            objective_mode="global_attention",
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    rounds = [
        _round(
            allocation_index=0,
            current_sha256=initial_sha,
            global_loss=0.20,
            all_loss=0.50,
            candidates=first_candidates,
            selected_invocation_id="mutation.11",
        ),
        _round(
            allocation_index=1,
            current_sha256=final_sha,
            global_loss=0.15,
            all_loss=0.50 - 1.0 / 120.0,
            candidates=second_candidates,
            selected_invocation_id=None,
        ),
    ]
    rounds_sha = runner._candidate_rounds_artifact_sha256(rounds)
    allocated_rank_profile = [
        [invocation_id, sequence.count(invocation_id)]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    total_rank_profile = [
        [
            invocation_id,
            sequence.count(invocation_id)
            + (1 if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS else 0),
        ]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    bank = {
        "artifact_sha256": final_sha,
        "parent_site_rank_one_bank_sha256": parent_sha,
        "allocated_rank_profile": allocated_rank_profile,
        "total_rank_profile": total_rank_profile,
    }
    initial_fit_by_id = {
        invocation_id: 0.50
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    }
    final_fit_by_id = dict(initial_fit_by_id)
    final_fit_by_id["mutation.11"] = 0.32
    initial_all_by_id = {
        invocation_id: (
            0.20
            if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
            else 0.56
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    }
    final_all_by_id = dict(initial_all_by_id)
    final_all_by_id["mutation.11"] = 0.05
    initial_global_by_id = {
        invocation_id: initial_all_by_id[invocation_id]
        for invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
    }
    final_global_by_id = {
        invocation_id: final_all_by_id[invocation_id]
        for invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
    }
    initial_fit = _fit_metric(
        initial_fit_by_id,
        row_count=runner._DIRECTION_TRAIN_ROWS,
    )
    final_fit = _fit_metric(
        final_fit_by_id,
        row_count=runner._DIRECTION_TRAIN_ROWS,
    )
    initial_all = _fit_metric(
        initial_all_by_id,
        row_count=runner._ALLOCATION_VALIDATION_ROWS,
    )
    final_all = _fit_metric(
        final_all_by_id,
        row_count=runner._ALLOCATION_VALIDATION_ROWS,
    )
    initial_global = _fit_metric(
        initial_global_by_id,
        row_count=runner._GLOBAL_ALLOCATION_VALIDATION_ROWS,
    )
    final_global = _fit_metric(
        final_global_by_id,
        row_count=runner._GLOBAL_ALLOCATION_VALIDATION_ROWS,
    )
    selected = next(
        candidate
        for candidate in first_candidates
        if candidate["invocation_id"] == "mutation.11"
    )
    return {
        "initial_bank_sha256": initial_sha,
        "final_bank": bank,
        "allocation_budget": 2,
        "budget_exhausted": False,
        "global_relative_fisher_threshold": 0.10,
        "global_gate_passed": False,
        "frozen_allocation_sequence": list(sequence),
        "frozen_allocated_rank_profile": allocated_rank_profile,
        "frozen_total_rank_profile": total_rank_profile,
        "frozen_profile_artifact_sha256": runner._profile_artifact_sha256(
            sequence,
            allocated_rank_profile,
            total_rank_profile,
            rounds_sha,
        ),
        "first_spectrum": {
            "source_bank_sha256": initial_sha,
            "direction_train_current_oracle": initial_fit,
            "allocation_validation_current_oracle": initial_all,
            "per_site": [
                first_marginals[invocation_id]
                for invocation_id in runner._NUMERIC_INVOCATION_IDS
            ],
        },
        "steps": [
            {
                "allocation_index": 0,
                "invocation_id": "mutation.11",
                "objective_mode": "global_attention",
                "source_bank_sha256": initial_sha,
                "result_bank_sha256": final_sha,
                "marginal_source_bank_sha256": initial_sha,
                "used_cached_off_site_marginal": False,
                "selection_objective_normalized_fisher_gain": 0.05,
                "selection_global_normalized_fisher_gain": 0.05,
                "selection_all_layer_normalized_fisher_gain": 1.0 / 120.0,
                "marginal": selected["marginal"],
                "direction_train_before": initial_fit,
                "direction_train_after": final_fit,
                "allocation_validation_before": initial_all,
                "allocation_validation_after": final_all,
                "global_allocation_validation_before": initial_global,
                "global_allocation_validation_after": final_global,
            }
        ],
        "candidate_rounds": rounds,
        "candidate_rounds_artifact_sha256": rounds_sha,
        "initial_direction_train_oracle": initial_fit,
        "final_direction_train_oracle": final_fit,
        "initial_allocation_validation_oracle": initial_all,
        "final_allocation_validation_oracle": final_all,
        "initial_global_allocation_validation_oracle": initial_global,
        "final_global_allocation_validation_oracle": final_global,
        "direction_train_relative_fisher_loss_history": [0.50, 0.49],
        "allocation_validation_relative_fisher_loss_history": [
            0.50,
            0.50 - 1.0 / 120.0,
        ],
        "global_allocation_validation_relative_fisher_loss_history": [
            0.20,
            0.15,
        ],
    }


def _valid_fixed_profile_replay() -> tuple[
    dict[str, object],
    tuple[str, ...],
    str,
    str,
]:
    sequence = ("mutation.11", "mutation.1")
    state_sha256s = (
        _sha("replay-initial"),
        _sha("replay-after-first"),
        _sha("replay-final"),
    )
    parent_sha256 = _sha("replay-parent")
    candidate_rounds_sha256 = _sha("replay-selection-rounds")
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
    bank = {
        "artifact_sha256": state_sha256s[-1],
        "parent_site_rank_one_bank_sha256": parent_sha256,
        "allocated_rank_profile": allocated_rank_profile,
        "total_rank_profile": total_rank_profile,
    }
    profile_sha256 = runner._profile_artifact_sha256(
        sequence,
        allocated_rank_profile,
        total_rank_profile,
        candidate_rounds_sha256,
    )

    per_invocation_losses: list[dict[str, float]] = [
        {
            invocation_id: 0.50
            for invocation_id in runner._NUMERIC_INVOCATION_IDS
        }
    ]
    after_first = dict(per_invocation_losses[-1])
    after_first[sequence[0]] = 0.32
    per_invocation_losses.append(after_first)
    after_second = dict(per_invocation_losses[-1])
    after_second[sequence[1]] = 0.32
    per_invocation_losses.append(after_second)
    fit_metrics = [
        _fit_metric(
            per_invocation,
            row_count=runner._OUTER_FIT_ROWS,
        )
        for per_invocation in per_invocation_losses
    ]

    steps: list[dict[str, object]] = []
    for index, invocation_id in enumerate(sequence):
        current_site_loss = per_invocation_losses[index][invocation_id]
        candidate_site_loss = per_invocation_losses[index + 1][invocation_id]
        marginal = {
            "invocation_id": invocation_id,
            "source_bank_sha256": state_sha256s[index],
            "proposal_bank": {
                "artifact_sha256": state_sha256s[index + 1],
            },
            "proposal_fit_current_oracle": _site_metric(current_site_loss),
            "proposal_fit_candidate_oracle": _site_metric(
                candidate_site_loss
            ),
            "allocation_evaluation_current_oracle": _site_metric(
                current_site_loss
            ),
            "allocation_evaluation_candidate_oracle": _site_metric(
                candidate_site_loss
            ),
        }
        steps.append(
            {
                "allocation_index": index,
                "invocation_id": invocation_id,
                "source_bank_sha256": state_sha256s[index],
                "result_bank_sha256": state_sha256s[index + 1],
                "marginal_source_bank_sha256": state_sha256s[index],
                "used_cached_off_site_marginal": False,
                "all_layer_relative_fisher_gain": 0.01,
                "marginal": marginal,
                "full_fit_before": deepcopy(fit_metrics[index]),
                "full_fit_after": deepcopy(fit_metrics[index + 1]),
            }
        )
    payload = {
        "initial_bank_sha256": state_sha256s[0],
        "final_bank": bank,
        "allocation_profile": list(sequence),
        "allocated_component_count": len(sequence),
        "selection_candidate_rounds_artifact_sha256": (
            candidate_rounds_sha256
        ),
        "frozen_profile_artifact_sha256": profile_sha256,
        "replay_profile_artifact_sha256": profile_sha256,
        "replay_profile_matches_frozen_selection": True,
        "steps": steps,
        "initial_full_fit_oracle": deepcopy(fit_metrics[0]),
        "final_full_fit_oracle": deepcopy(fit_metrics[-1]),
        "full_fit_relative_fisher_loss_history": [0.50, 0.49, 0.48],
    }
    return payload, sequence, profile_sha256, candidate_rounds_sha256


def _reseal_allocation(value: dict[str, object]) -> None:
    rounds = value["candidate_rounds"]
    bank = value["final_bank"]
    sequence = value["frozen_allocation_sequence"]
    assert isinstance(rounds, list)
    assert isinstance(bank, dict)
    assert isinstance(sequence, list)
    rounds_sha = runner._candidate_rounds_artifact_sha256(rounds)
    value["candidate_rounds_artifact_sha256"] = rounds_sha
    value["frozen_profile_artifact_sha256"] = runner._profile_artifact_sha256(
        sequence,
        bank["allocated_rank_profile"],
        bank["total_rank_profile"],
        rounds_sha,
    )


def _stub_sealed_validator() -> SimpleNamespace:
    passthrough = lambda value, **_kwargs: value
    return SimpleNamespace(
        _validate_bank_summary=passthrough,
        _validate_first_spectrum=passthrough,
        _validate_fit_metric=passthrough,
        _validate_marginal=passthrough,
    )


def test_module_is_stdlib_only_and_public_surface_cannot_open_B_or_C() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/fisher_graph/"
        "gemma3_attention_transport_global_gated_residual_spectrum_a_diagnostic.py"
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
        "run_global_gated_residual_spectrum_A_diagnostic",
        "check_global_gated_residual_spectrum_A_diagnostic",
        "main",
    }
    capture_roles: list[str] = []
    called_names: list[str] = []
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
from fisher_graph import (
    gemma3_attention_transport_global_gated_residual_spectrum_a_diagnostic,
)
new_modules = set(sys.modules) - baseline
assert not [name for name in ('torch', 'mlx', 'transformers') if name in new_modules]
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_partition_payload_and_validator_bind_stable_and_runtime_axes() -> None:
    first = _pinned_partition(receipt_prefix="first")
    second = _pinned_partition(receipt_prefix="second")
    first_payload = runner._partition_payload(first)
    second_payload = runner._partition_payload(second)
    first_payload["final_firewall_state"] = _firewall_state()
    second_payload["final_firewall_state"] = _firewall_state()
    assert runner._validate_partition(first_payload) is first_payload
    assert runner._validate_partition(second_payload) is second_payload
    assert first_payload["stable_batch_catalog"] == second_payload[
        "stable_batch_catalog"
    ]
    assert first_payload["outer"]["outer_split_artifact_sha256"] == (
        second_payload["outer"]["outer_split_artifact_sha256"]
    )
    assert first_payload["inner"]["inner_split_artifact_sha256"] == (
        second_payload["inner"]["inner_split_artifact_sha256"]
    )
    assert first_payload["runtime_capture_binding"] != second_payload[
        "runtime_capture_binding"
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("runtime_capture_binding", "execution_receipts_drive_partition"), True),
        (("outer", "split_identity_stable_across_execution_receipt_drift"), False),
        (("outer", "outer_fit", "batch_ordinals"), []),
        (("inner", "allocation_validation", "row_count"), 0),
        (("final_firewall_state", "B_opened_by_this_run"), True),
    ),
)
def test_partition_validator_rejects_tampering(
    path: tuple[str, ...],
    value: object,
) -> None:
    payload = runner._partition_payload(_pinned_partition())
    payload["final_firewall_state"] = _firewall_state()
    target: dict[str, object] = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    with pytest.raises(ValueError):
        runner._validate_partition(payload)


def test_allocation_round_payload_enforces_global_scope() -> None:
    initial = _sha("round-initial")
    marginals = tuple(
        SimpleNamespace(
            invocation_id=invocation_id,
            source_bank_sha256=initial,
            strictly_improves=True,
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    )
    global_gains = tuple(
        (
            invocation_id,
            0.05 if invocation_id == "mutation.11" else 0.0,
        )
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    )
    all_gains = tuple(
        (invocation_id, 0.01)
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    )
    value = SimpleNamespace(
        allocation_index=0,
        current_bank_sha256=initial,
        objective_mode="global_attention",
        global_normalized_fisher_mass_gains=global_gains,
        all_layer_normalized_fisher_mass_gains=all_gains,
        marginals=marginals,
        minimum_relative_fisher_gain=1.0e-12,
        selected_invocation_id="mutation.11",
        global_relative_fisher_threshold=0.10,
        global_relative_fisher_loss_before=0.20,
        all_layer_relative_fisher_loss_before=0.50,
        global_evaluation_row_count=runner._GLOBAL_ALLOCATION_VALIDATION_ROWS,
        global_evaluation_zero_predictor_fisher_loss=10.0,
        all_layer_evaluation_row_count=runner._ALLOCATION_VALIDATION_ROWS,
        all_layer_evaluation_zero_predictor_fisher_loss=10.0,
        tie_tolerance=1.0e-15,
    )
    original = runner._marginal_payload
    try:
        runner._marginal_payload = lambda item: {
            "invocation_id": item.invocation_id
        }
        payload = runner._allocation_round_payload(value)
    finally:
        runner._marginal_payload = original
    by_id = {item["invocation_id"]: item for item in payload["candidates"]}
    assert payload["objective_mode"] == "global_attention"
    assert by_id["mutation.11"]["in_objective_scope"] is True
    assert by_id["mutation.11"]["eligible_for_selection"] is True
    assert by_id["mutation.1"]["in_objective_scope"] is False
    assert by_id["mutation.1"]["global_normalized_fisher_mass_gain"] == 0.0
    assert by_id["mutation.1"]["eligible_for_selection"] is False


def test_global_allocation_validator_accepts_closed_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload = _valid_global_allocation()
    assert runner._validate_global_gated_allocation(payload) is payload


@pytest.mark.parametrize(
    "tamper",
    ("gate_mode", "scope", "gain", "argmax", "cache", "profile"),
)
def test_global_allocation_validator_rejects_ledger_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload = _valid_global_allocation()
    rounds = payload["candidate_rounds"]
    assert isinstance(rounds, list)
    if tamper == "gate_mode":
        rounds[0]["objective_mode"] = "all_layers"
    elif tamper == "scope":
        rounds[0]["candidates"][0]["in_objective_scope"] = True
    elif tamper == "gain":
        rounds[0]["candidates"][5][
            "global_normalized_fisher_mass_gain"
        ] = 0.04
    elif tamper == "argmax":
        rounds[0]["selected_invocation_id"] = "mutation.23"
    elif tamper == "cache":
        cached_candidate = rounds[1]["candidates"][11]
        cached_candidate["marginal"] = deepcopy(cached_candidate["marginal"])
        cached_candidate["marginal"]["cache_tamper"] = True
    elif tamper == "profile":
        payload["frozen_profile_artifact_sha256"] = _sha("wrong-profile")
    if tamper != "profile":
        _reseal_allocation(payload)
    with pytest.raises(ValueError):
        runner._validate_global_gated_allocation(payload)


def test_global_allocation_rejects_resealed_sequence_step_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload = _valid_global_allocation()
    tampered_sequence = ["mutation.13"]
    allocated_rank_profile = [
        [invocation_id, tampered_sequence.count(invocation_id)]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    total_rank_profile = [
        [
            invocation_id,
            tampered_sequence.count(invocation_id)
            + (
                1
                if invocation_id in runner._GLOBAL_ATTENTION_INVOCATION_IDS
                else 0
            ),
        ]
        for invocation_id in runner._NUMERIC_INVOCATION_IDS
    ]
    payload["frozen_allocation_sequence"] = tampered_sequence
    payload["frozen_allocated_rank_profile"] = allocated_rank_profile
    payload["frozen_total_rank_profile"] = total_rank_profile
    bank = payload["final_bank"]
    assert isinstance(bank, dict)
    bank["allocated_rank_profile"] = allocated_rank_profile
    bank["total_rank_profile"] = total_rank_profile
    _reseal_allocation(payload)

    assert payload["steps"][0]["invocation_id"] == "mutation.11"
    assert payload["candidate_rounds"][0]["selected_invocation_id"] == (
        "mutation.11"
    )
    with pytest.raises(ValueError, match="frozen allocation sequence drifted"):
        runner._validate_global_gated_allocation(payload)


def test_global_allocation_rejects_denominator_rescaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload = _valid_global_allocation()
    rounds = payload["candidate_rounds"]
    assert isinstance(rounds, list)
    scaled_marginals: set[int] = set()
    for round_value in rounds:
        round_value["global_evaluation_zero_predictor_fisher_loss"] = 20.0
        round_value["all_layer_evaluation_zero_predictor_fisher_loss"] = 20.0
        for candidate in round_value["candidates"]:
            marginal = candidate["marginal"]
            if id(marginal) in scaled_marginals:
                continue
            scaled_marginals.add(id(marginal))
            for metric_name in (
                "allocation_evaluation_current_oracle",
                "allocation_evaluation_candidate_oracle",
            ):
                metric = marginal[metric_name]
                metric["fisher_loss"] *= 2.0
                metric["zero_predictor_fisher_loss"] *= 2.0
    _reseal_allocation(payload)

    with pytest.raises(ValueError):
        runner._validate_global_gated_allocation(payload)


def test_fixed_profile_replay_validator_accepts_closed_two_step_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload, sequence, profile_sha256, rounds_sha256 = (
        _valid_fixed_profile_replay()
    )
    assert runner._validate_fixed_profile_replay(
        payload,
        frozen_sequence=sequence,
        frozen_profile_sha256=profile_sha256,
        frozen_candidate_rounds_sha256=rounds_sha256,
    ) is payload


@pytest.mark.parametrize(
    "boundary",
    ("initial_to_first", "adjacent_steps", "last_to_final"),
)
def test_fixed_profile_replay_rejects_same_loss_oracle_splicing(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload, sequence, profile_sha256, rounds_sha256 = (
        _valid_fixed_profile_replay()
    )
    steps = payload["steps"]
    assert isinstance(steps, list)
    if boundary == "initial_to_first":
        spliced = steps[0]["full_fit_before"]
    elif boundary == "adjacent_steps":
        spliced = steps[1]["full_fit_before"]
    else:
        spliced = payload["final_full_fit_oracle"]
    assert isinstance(spliced, dict)
    unchanged_relative_loss = spliced["relative_fisher_loss"]
    spliced["cosine"] = 0.79
    assert spliced["relative_fisher_loss"] == unchanged_relative_loss

    with pytest.raises(ValueError):
        runner._validate_fixed_profile_replay(
            payload,
            frozen_sequence=sequence,
            frozen_profile_sha256=profile_sha256,
            frozen_candidate_rounds_sha256=rounds_sha256,
        )


def test_fixed_profile_replay_rejects_zero_baseline_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload, sequence, profile_sha256, rounds_sha256 = (
        _valid_fixed_profile_replay()
    )
    steps = payload["steps"]
    assert isinstance(steps, list)
    for metric in (
        steps[0]["full_fit_after"],
        steps[1]["full_fit_before"],
    ):
        assert isinstance(metric, dict)
        for field in ("fisher_loss", "directional_mse", "euclidean_mse"):
            metric[field] *= 1.1
        metric["zero_predictor_fisher_loss"] *= 1.1

    assert steps[0]["full_fit_after"] == steps[1]["full_fit_before"]
    with pytest.raises(ValueError):
        runner._validate_fixed_profile_replay(
            payload,
            frozen_sequence=sequence,
            frozen_profile_sha256=profile_sha256,
            frozen_candidate_rounds_sha256=rounds_sha256,
        )


@pytest.mark.parametrize(
    ("metric_name", "relative_fisher_loss"),
    (
        ("proposal_fit_current_oracle", 0.51),
        ("proposal_fit_candidate_oracle", 0.31),
        ("allocation_evaluation_current_oracle", 0.51),
        ("allocation_evaluation_candidate_oracle", 0.31),
    ),
)
def test_fixed_profile_replay_rejects_selected_marginal_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    metric_name: str,
    relative_fisher_loss: float,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload, sequence, profile_sha256, rounds_sha256 = (
        _valid_fixed_profile_replay()
    )
    metric = payload["steps"][0]["marginal"][metric_name]
    metric["relative_fisher_loss"] = relative_fisher_loss
    metric["fisher_loss"] = relative_fisher_loss * 10.0

    with pytest.raises(ValueError):
        runner._validate_fixed_profile_replay(
            payload,
            frozen_sequence=sequence,
            frozen_profile_sha256=profile_sha256,
            frozen_candidate_rounds_sha256=rounds_sha256,
        )


def test_fixed_profile_replay_rejects_offsetting_offsite_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_sealed_runner", _stub_sealed_validator)
    payload, sequence, profile_sha256, rounds_sha256 = (
        _valid_fixed_profile_replay()
    )
    steps = payload["steps"]
    assert isinstance(steps, list)
    downstream_metrics = (
        steps[0]["full_fit_after"],
        steps[1]["full_fit_before"],
        steps[1]["full_fit_after"],
        payload["final_full_fit_oracle"],
    )
    for metric in downstream_metrics:
        pairs = metric["per_invocation_relative_fisher_loss"]
        next(pair for pair in pairs if pair[0] == "mutation.3")[1] = 0.49
        next(pair for pair in pairs if pair[0] == "mutation.5")[1] = 0.51

    assert steps[0]["full_fit_after"] == steps[1]["full_fit_before"]
    assert steps[1]["full_fit_after"] == payload["final_full_fit_oracle"]
    with pytest.raises(ValueError):
        runner._validate_fixed_profile_replay(
            payload,
            frozen_sequence=sequence,
            frozen_profile_sha256=profile_sha256,
            frozen_candidate_rounds_sha256=rounds_sha256,
        )


def test_report_digest_publisher_and_strict_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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

    monkeypatch.setattr(
        runner,
        "_strict_report",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    output = tmp_path / "report.json"
    runner._publish_report(output, sealed, publication_guard=lambda _phase: None)
    assert output.read_bytes() == runner._canonical_json_bytes(sealed)
    with pytest.raises(FileExistsError):
        runner._publish_report(
            output,
            sealed,
            publication_guard=lambda _phase: None,
        )

    raced = tmp_path / "raced.json"
    owned = tmp_path / "owned.json"

    def replacement_guard(phase_id: str) -> None:
        if phase_id == "global_gated_publication_verified":
            raced.rename(owned)
            raced.write_bytes(b"foreign replacement")

    with pytest.raises((RuntimeError, ValueError, json.JSONDecodeError)):
        runner._publish_report(
            raced,
            sealed,
            publication_guard=replacement_guard,
        )
    assert raced.read_bytes() == b"foreign replacement"
    assert owned.exists()
    assert list(tmp_path.glob(".*.tmp")) == []

    with pytest.raises(ValueError, match="duplicate JSON field"):
        runner._strict_json_loads(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="non-finite"):
        runner._strict_json_loads(b'{"value":NaN}')
    source = tmp_path / "source.json"
    source.write_bytes(b'{"schema":"test"}')
    symlink = tmp_path / "symlink.json"
    os.symlink(source, symlink)
    with pytest.raises(ValueError, match="regular file"):
        runner._read_no_follow_regular_bytes(symlink)
