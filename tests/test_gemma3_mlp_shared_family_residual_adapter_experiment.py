from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fisher_graph.gemma3_mlp_shared_family_residual_adapter_experiment as worker
from fisher_graph.gemma3_mlp_shared_family_residual_adapter import (
    MLPFamilyPairManifest,
)
from fisher_graph.gemma3_mlp_shared_family_residual_adapter_fit import (
    ProjectionResidualFitReceipt,
)
from fisher_graph.gemma3_mlp_shared_family_residual_adapter_setup import (
    DEFAULT_OUTPUT as DEFAULT_SETUP_PATH,
    LOW_RANK_POINTS,
    ORDERED_ROLES,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_PANEL_PATH,
    PINNED_PREDECESSOR_REPORT_PATH,
    PINNED_SOURCE_SCALARS,
    build_gemma3_mlp_shared_family_residual_adapter_setup,
)


_ROOT = Path(__file__).resolve().parents[1]
_SETUP_PATH = _ROOT / DEFAULT_SETUP_PATH
_SETUP = build_gemma3_mlp_shared_family_residual_adapter_setup()
_ARMS = tuple(_SETUP["arm_contract"]["arm_order"])
_PRIMARY_ARMS = tuple(
    f"primary.shared_core_residual_r{rank}" for rank in LOW_RANK_POINTS
)
_FIT_ARMS = tuple(
    arm_id
    for rank in LOW_RANK_POINTS
    for arm_id in (
        f"primary.shared_core_residual_r{rank}",
        f"control.independent_member_r{rank}",
    )
)
_WINNER = "primary.shared_core_residual_r32"
_PINNED_SOURCE_MODEL_SHA256 = (
    "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
)
_PINNED_SOURCE_EXECUTION_SHA256 = (
    "b04085b6cd7c076596dd074e3693feb465e69a917c61a46cc0f7310afb444e60"
)
_VALID_REPORT_CACHE: dict[str, object] | None = None


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("message_kind", ("attention_transport", "mlp_mutation"))
def test_primary_generator_receipt_uses_live_carrier_message_kinds(
    message_kind: str,
) -> None:
    primary = SimpleNamespace(message_kind=message_kind, value_sha256=_digest("delta"))
    boundary = SimpleNamespace(
        message_receipts=(
            SimpleNamespace(message_kind="identity_carry"),
            primary,
        )
    )

    assert worker._primary_generator_message_receipt(boundary) is primary


def test_primary_generator_receipt_rejects_legacy_generic_kind() -> None:
    boundary = SimpleNamespace(
        message_receipts=(
            SimpleNamespace(message_kind="identity_carry"),
            SimpleNamespace(message_kind="primary_generator"),
        )
    )

    with pytest.raises(ValueError, match="exactly one typed primary message"):
        worker._primary_generator_message_receipt(boundary)


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    payload = deepcopy(value)
    payload.pop(field, None)
    value[field] = worker._manifest_sha256(payload)
    return value


def _typed_pair_manifest() -> dict[str, object]:
    return MLPFamilyPairManifest(
        exact_plan_protocol_sha256=_digest("exact-plan-protocol"),
        successor_setup_sha256=worker.PINNED_SETUP_SHA256,
        exact_plan_sha256=_digest("exact-plan"),
        exact_manifest_sha256=_digest("exact-manifest"),
        source_model_fingerprint=_PINNED_SOURCE_MODEL_SHA256,
        anchor_invocation_id="mutation.2",
        member_invocation_id="mutation.6",
        anchor_atom_id="atom.mlp.layer.0",
        member_atom_id="atom.mlp.layer.2",
        anchor_block_ordinal=0,
        member_block_ordinal=2,
        a_fit_manifest_sha256=_SETUP["data_firewall"]["roles"]["A_fit"][
            "role_manifest_sha256"
        ],
        selection_evidence_sha256=_digest("selection-evidence"),
    ).to_dict()


def _projection_solver_receipt(arm_id: str) -> dict[str, object]:
    rank = next(value for value in LOW_RANK_POINTS if arm_id.endswith(str(value)))
    target_kind = (
        "independent_member"
        if arm_id.startswith("control.independent")
        else "primary_member_minus_anchor"
    )
    receipts: dict[str, object] = {}
    for projection_name, input_width, output_width in (
        ("gate_proj", 640, 2_048),
        ("up_proj", 640, 2_048),
        ("down_proj", 2_048, 640),
    ):
        receipts[projection_name] = ProjectionResidualFitReceipt(
            projection_name=projection_name,
            target_kind=target_kind,
            row_count=20,
            input_width=input_width,
            output_width=output_width,
            requested_rank=rank,
            numerical_rank=rank,
            effective_rank=rank,
            zero_padded_rank=0,
            relative_ridge=1.0e-4,
            ridge_scale=1.0,
            absolute_ridge=1.0e-4,
            numerical_rank_tolerance=1.0e-12,
            weighted_target_energy=1.0,
            weighted_fit_sse=0.01,
            weighted_relative_sse=0.01,
            evidence_sha256=_digest(f"{arm_id}.{projection_name}.evidence"),
            factor_sha256=_digest(f"{arm_id}.{projection_name}.factor"),
            stored_dtype="torch.float32",
        ).to_dict()
    return {
        "projection_order": ["gate_proj", "up_proj", "down_proj"],
        "projection_receipts": receipts,
    }


def _arm_metric(arm_id: str) -> dict[str, object]:
    exact = arm_id in {
        "native",
        "exact.heterogeneous_full_rank_atom_bank_lift",
        "exact.shared_core_dense_residual",
    }
    rank = next(
        (value for value in LOW_RANK_POINTS if arm_id.endswith(str(value))),
        None,
    )
    if exact:
        delta_nll = 0.0
    elif arm_id == "ablation.shared_core_zero_residual":
        delta_nll = 0.09
    elif arm_id.startswith("primary."):
        delta_nll = {256: 0.02, 128: 0.03, 64: 0.04, 32: 0.05}[rank]
    elif arm_id.startswith("control.independent"):
        delta_nll = {256: 0.05, 128: 0.06, 64: 0.07, 32: 0.08}[rank]
    else:
        delta_nll = {256: 0.087, 128: 0.086, 64: 0.085, 32: 0.085}[rank]
    return {
        "arm_id": arm_id,
        "delta_nll_per_token": delta_nll,
        "native_to_candidate_kl_per_query": delta_nll / 2.0,
        "top1_agreement": 1.0 if exact else 0.95,
        "valid_query_logit_nrmse": delta_nll / 2.0,
        "final_carrier_nrmse": delta_nll / 2.0,
        "selected_nonanchor_MLP_delta_nrmse": (
            0.0 if exact else min(0.24, delta_nll * 2.0)
        ),
        "selected_nonanchor_MLP_delta_cosine": 1.0 if exact else 0.95,
        "source_projection_and_fallback_calls": 0,
        "passed_fidelity_gate": exact or arm_id.startswith("primary."),
    }


def _role_result(role_id: str) -> dict[str, object]:
    manifest = _SETUP["data_firewall"]["roles"][role_id]
    result: dict[str, object] = {
        "role_id": role_id,
        "role_manifest_sha256": manifest["role_manifest_sha256"],
        "example_ids_sha256": manifest["example_ids_sha256"],
        "example_count": manifest["example_count"],
        "family_count": manifest["family_count"],
        "ordered_arm_ids": list(_ARMS),
        "arm_metrics": [_arm_metric(arm_id) for arm_id in _ARMS],
        "pair_manifest_sha256": _digest("pair-manifest"),
        "fit_performed": role_id == "A_fit",
        "pair_selection_performed": role_id == "A_fit",
        "dense_exact_control_gate_passed": role_id == "A_fit",
        "threshold_or_topology_changed": False,
        "role_result_sha256": "",
    }
    return _seal(result, "role_result_sha256")


def _pair_selection() -> dict[str, object]:
    typed_pair = _typed_pair_manifest()
    result: dict[str, object] = {
        "family_id": "family.mlp.gated_gelu_h640_i2048",
        "anchor_invocation_id": "mutation.2",
        "member_invocation_id": "mutation.6",
        "anchor_atom_id": "atom.mlp.layer.0",
        "member_atom_id": "atom.mlp.layer.2",
        "anchor_block_ordinal": 0,
        "member_block_ordinal": 2,
        "selected_on": "A_fit",
        "a_fit_role_manifest_sha256": _SETUP["data_firewall"]["roles"][
            "A_fit"
        ]["role_manifest_sha256"],
        "selection_evidence_sha256": _digest("selection-evidence"),
        "shared_core_state_sha256": _digest("anchor-exact-core"),
        "activation_and_Fisher_evidence_only": True,
        "block_ordinal_used_as_numerical_feature": False,
        "nonadjacency_is_eligibility_only": True,
        "frozen_before_A_select": True,
        "dense_exact_control_gate_passed": True,
        "typed_pair_manifest": typed_pair,
        "pair_manifest_sha256": typed_pair["pair_manifest_sha256"],
    }
    return result


def _fit_receipt(arm_id: str, pair_sha256: str) -> dict[str, object]:
    rank = next(value for value in LOW_RANK_POINTS if arm_id.endswith(str(value)))
    setup_arm = next(
        item for item in _SETUP["arm_contract"]["arms"]
        if item["arm_id"] == arm_id
    )
    solver = _projection_solver_receipt(arm_id)
    result: dict[str, object] = {
        "arm_id": arm_id,
        "rank": rank,
        "fit_nonce": setup_arm["fit_nonce"],
        "fit_seed": setup_arm["fit_seed"],
        "fit_role": "A_fit",
        "a_fit_role_manifest_sha256": _SETUP["data_firewall"]["roles"][
            "A_fit"
        ]["role_manifest_sha256"],
        "pair_manifest_sha256": pair_sha256,
        "shared_core_state_sha256": _digest("anchor-exact-core"),
        "independently_fitted": True,
        "supernet_or_factor_slicing_used": False,
        "factor_catalog_sha256": _digest(f"{arm_id}.factors"),
        "projection_solver_receipt": solver,
        "projection_solver_receipt_sha256": worker._manifest_sha256(solver),
        "selected_member_input_sha256": _digest(
            "exact-selected-member-input"
        ),
        "fit_receipt_sha256": "",
    }
    return _seal(result, "fit_receipt_sha256")


def _arm_artifact(
    arm_id: str,
    pair_sha256: str,
    fit_by_arm: dict[str, dict[str, object]],
) -> dict[str, object]:
    setup_arm = next(
        item for item in _SETUP["arm_contract"]["arms"]
        if item["arm_id"] == arm_id
    )
    result: dict[str, object] = {
        "arm_id": arm_id,
        "pair_manifest_sha256": pair_sha256,
        "shared_core_state_sha256": _digest("anchor-exact-core"),
        "fit_receipt_sha256": (
            fit_by_arm[arm_id]["fit_receipt_sha256"]
            if arm_id in fit_by_arm else None
        ),
        "artifact_state_sha256": _digest(f"{arm_id}.artifact-state"),
        "candidate_stored_scalars": setup_arm["candidate_stored_scalars"],
        "learned_correction_edge_count": 0,
        "chart_or_learned_transport_scalars": 0,
        "source_projection_and_fallback_calls": 0,
        "nonanchor_dense_source_storage_present": False,
        "artifact_manifest_sha256": "",
    }
    return _seal(result, "artifact_manifest_sha256")


def _legacy_valid_report_fixture() -> dict[str, object]:
    pair = _pair_selection()
    fit_by_arm = {
        arm_id: _fit_receipt(arm_id, pair["pair_manifest_sha256"])
        for arm_id in _FIT_ARMS
    }
    artifacts = [
        _arm_artifact(arm_id, pair["pair_manifest_sha256"], fit_by_arm)
        for arm_id in _ARMS
    ]
    role_results = {role: _role_result(role) for role in ORDERED_ROLES}
    for value in role_results.values():
        value["pair_manifest_sha256"] = pair["pair_manifest_sha256"]
        _seal(value, "role_result_sha256")
    frontier: dict[str, object] = {
        "ordered_primary_arm_ids": list(_PRIMARY_ARMS),
        "promotion_order": list(reversed(_PRIMARY_ARMS)),
        "pair_manifest_sha256": pair["pair_manifest_sha256"],
        "A_select_role_result_sha256": role_results["A_select"][
            "role_result_sha256"
        ],
        "frozen_before_B_guard": True,
        "refit_or_pruning_performed": False,
        "frontier_sha256": "",
    }
    _seal(frontier, "frontier_sha256")
    report: dict[str, object] = {
        "schema": worker.GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA,
        "format_version": (
            worker.GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION
        ),
        "diagnostic_outcome": "shared_family_pair_mechanism_qualified",
        "contains_model_weights": False,
        "contains_executor_weights": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "setup": {
            "setup_sha256": worker.PINNED_SETUP_SHA256,
            "file_sha256": worker.PINNED_SETUP_FILE_SHA256,
            "path": str(DEFAULT_SETUP_PATH),
        },
        "predecessor": {
            "report_sha256": worker.PINNED_PREDECESSOR_REPORT_SHA256,
            "file_sha256": worker.PINNED_PREDECESSOR_REPORT_FILE_SHA256,
            "required_arm_id": "exact.heterogeneous_full_rank_atom_bank_lift",
            "required_arm_gate_passed": True,
            "anchor_source_atom_state_sha256": _digest("anchor-exact-core"),
        },
        "model": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "parameter_count": PINNED_SOURCE_SCALARS,
            "source_model_fingerprint_before": _PINNED_SOURCE_MODEL_SHA256,
            "source_model_fingerprint_after": _PINNED_SOURCE_MODEL_SHA256,
            "source_execution_fingerprint_before": (
                _PINNED_SOURCE_EXECUTION_SHA256
            ),
            "source_execution_fingerprint_after": (
                _PINNED_SOURCE_EXECUTION_SHA256
            ),
        },
        "data": {
            "panel_sha256": worker.PINNED_PANEL_SHA256,
            "ordered_roles": list(ORDERED_ROLES),
            "roles": deepcopy(_SETUP["data_firewall"]["roles"]),
            "role_family_overlap": False,
            "role_example_overlap": False,
        },
        "configuration": {
            "ordered_roles": list(ORDERED_ROLES),
            "ordered_arm_ids": list(_ARMS),
            "residual_rank_order": list(LOW_RANK_POINTS),
            "fit_arm_ids": list(_FIT_ARMS),
            "seeds": deepcopy(_SETUP["fit_contract"]["seeds"]),
            "acceptance_gates": deepcopy(_SETUP["acceptance_gates"]),
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 4,
        },
        "pair_selection": pair,
        "fit_receipts": [fit_by_arm[arm_id] for arm_id in _FIT_ARMS],
        "arm_artifacts": artifacts,
        "role_results": role_results,
        "frontier": frontier,
        "B_guard": {
            "opened_once": True,
            "entire_frozen_frontier_scored_once": True,
            "frontier_sha256": frontier["frontier_sha256"],
            "winner_arm_id": _WINNER,
            "bootstrap_seed": _SETUP["fit_contract"]["seeds"]["bootstrap"],
            "refit_reorder_prune_or_threshold_change": False,
            "passed": True,
        },
        "C_assessment": {
            "opened_at_most_once": True,
            "opened_after_guard_pass": True,
            "candidate_count": 1,
            "candidate_arm_id": _WINNER,
            "refit_selection_or_threshold_change": False,
            "passed": True,
        },
        "gate_chain": {
            "ordered_gates": [
                "preflight",
                "exact_predecessor",
                "dense_exact_control",
                "fit_frozen_frontier",
                "A_select_freeze",
                "B_guard",
                "C_assessment",
            ],
            "dense_exact_control_passed_before_compressed_fit": True,
            "failed_approximate_point_did_not_prune_frontier": True,
            "all_passed": True,
        },
        "execution_audit": {
            "preflight_before_live_dependencies": True,
            "source_model_unchanged": True,
            "source_execution_unchanged": True,
            "source_projection_and_fallback_calls": 0,
            "learned_correction_edge_count": 0,
            "chart_or_learned_transport_scalars": 0,
            "pair_manifest_sha256_by_role": {
                role: pair["pair_manifest_sha256"] for role in ORDERED_ROLES
            },
            "fit_call_count": 8,
            "unique_fit_nonce_count": 8,
            "selected_member_prefix_is_exact_by_single_approximation": True,
            "candidate_prefix_recapture_required": False,
            "exact_prefix_input_hash_equality_authenticated_on_A_fit": True,
            "selected_member_input_sha256_by_fit_arm": {
                arm_id: _digest("exact-selected-member-input")
                for arm_id in _FIT_ARMS
            },
            "all_39_invocations_and_75_occurrences_retained": True,
        },
        "resources": {
            "source_stored_scalars": PINNED_SOURCE_SCALARS,
            "candidate_stored_scalars_by_arm": {
                value["arm_id"]: value["candidate_stored_scalars"]
                for value in artifacts
            },
            "matched_control_budget_equal_by_rank": True,
            "nonanchor_dense_source_storage_absent": True,
            "serialized_byte_reduction_measured": False,
            "logical_MAC_reduction_measured": False,
            "peak_memory_measured": False,
            "latency_measured": False,
        },
        "claims": {
            "pair_mechanism_qualified": True,
            "twenty_percent_whole_model_compression_claimed": False,
            "learned_edge_compression_claimed": False,
            "serialized_byte_reduction_claimed": False,
            "logical_MAC_reduction_claimed": False,
            "latency_or_speed_claimed": False,
        },
    }
    report["report_sha256"] = worker._report_sha256(report)
    return report


def _mock_preflight(tmp_path: Path) -> object:
    return worker._Preflight(
        setup=_SETUP,
        predecessor_report={
            "report_sha256": worker.PINNED_PREDECESSOR_REPORT_SHA256,
        },
        role_examples={role: () for role in ORDERED_ROLES},
        setup_path=_SETUP_PATH,
        predecessor_path=tmp_path / "predecessor.json",
        panel_path=_ROOT / PINNED_PANEL_PATH,
        receipts={
            "all_authenticated_before_live_dependencies": True,
            "all_role_manifests_verified_before_model_load": True,
        },
    )


def _install_successful_public_mock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], list[str]]:
    report = _valid_report()
    pair = report["pair_selection"]
    fit_by_arm = {
        item["arm_id"]: item for item in report["fit_receipts"]
    }
    events: list[str] = []

    def preflight(**_kwargs: object) -> object:
        events.append("preflight")
        return _mock_preflight(tmp_path)

    def dependencies() -> object:
        events.append("live_dependencies")
        return SimpleNamespace()

    def prepare(**_kwargs: object) -> object:
        events.append("prepare")
        return {"runtime": "model-free"}

    def capture(**_kwargs: object) -> object:
        events.append("capture.A_fit")
        return {"capture": "A_fit-only"}

    def select(**_kwargs: object) -> object:
        events.append("select_pair.A_fit")
        return pair

    def fit(*, rank: int, independent_nonsharing: bool, **_kwargs: object) -> object:
        prefix = (
            "control.independent_member"
            if independent_nonsharing
            else "primary.shared_core_residual"
        )
        arm_id = f"{prefix}_r{rank}"
        events.append(f"fit.{arm_id}")
        return fit_by_arm[arm_id]

    def evaluate(*, role_id: str, fit_points: Mapping[str, object],
                 pair_selection: Mapping[str, object], **_kwargs: object) -> object:
        events.append(f"evaluate.{role_id}")
        assert pair_selection is pair
        if role_id == "A_fit":
            assert fit_points == {}
        elif role_id == "C_assessment":
            assert list(fit_points) == [_WINNER]
            assert fit_points[_WINNER] is fit_by_arm[_WINNER]
        else:
            assert tuple(fit_points) == _FIT_ARMS
        return report["role_results"][role_id]

    def freeze(**_kwargs: object) -> object:
        events.append("freeze.A_select")
        return report["frontier"]

    def guard(**_kwargs: object) -> object:
        events.append("guard.B_guard")
        return report["B_guard"]

    def assess(*, winner_arm_id: str, **_kwargs: object) -> object:
        events.append("assess.C_assessment")
        assert winner_arm_id == _WINNER
        return report["C_assessment"]

    def assemble(**kwargs: object) -> object:
        events.append("assemble")
        assert kwargs["pair"] is pair
        assert tuple(kwargs["fits"]) == _FIT_ARMS
        assert tuple(kwargs["role_results"]) == ORDERED_ROLES
        return report

    monkeypatch.setattr(worker, "_preflight", preflight)
    monkeypatch.setattr(worker, "_load_live_dependencies", dependencies)
    monkeypatch.setattr(worker, "_prepare_runtime", prepare)
    monkeypatch.setattr(worker, "_capture_a_fit", capture)
    monkeypatch.setattr(worker, "_select_pair", select)
    monkeypatch.setattr(worker, "_fit_rank_point", fit)
    monkeypatch.setattr(worker, "_evaluate_role", evaluate)
    monkeypatch.setattr(worker, "_freeze_a_select_frontier", freeze)
    monkeypatch.setattr(worker, "_run_b_guard", guard)
    monkeypatch.setattr(worker, "_run_c_assessment", assess)
    monkeypatch.setattr(worker, "_assemble_report", assemble)
    return report, events


class _UnchangedGuard:
    def assert_unchanged(self) -> None:
        return None


class _FingerprintAdapter:
    def model_fingerprint(self) -> str:
        return _PINNED_SOURCE_MODEL_SHA256

    def execution_fingerprint(self) -> str:
        return _PINNED_SOURCE_EXECUTION_SHA256


def _live_phase_metric(arm_id: str) -> dict[str, object]:
    report_metric = _arm_metric(arm_id)
    retention = 1.0 if arm_id.startswith(("native", "exact.")) else 0.95
    return {
        "nll_per_token": 1.0 + report_metric["delta_nll_per_token"],
        "delta_nll_per_token": report_metric["delta_nll_per_token"],
        "native_to_candidate_kl_per_query": report_metric[
            "native_to_candidate_kl_per_query"
        ],
        "top1_agreement_to_native": report_metric["top1_agreement"],
        "valid_query_logit_nrmse": report_metric["valid_query_logit_nrmse"],
        "native_correct_downstream_accuracy_retained": retention,
        "worst_family_native_correct_retention": retention,
        "selected_nonanchor_MLP_delta_nrmse": report_metric[
            "selected_nonanchor_MLP_delta_nrmse"
        ],
        "selected_nonanchor_MLP_delta_cosine": report_metric[
            "selected_nonanchor_MLP_delta_cosine"
        ],
        "final_carrier_nrmse": report_metric["final_carrier_nrmse"],
    }


def _tensor_metric_summary(*, exact: bool) -> dict[str, object]:
    return {
        "compared_elements": 640,
        "nrmse": 0.0 if exact else 0.01,
        "cosine_similarity": 1.0 if exact else 0.99,
        "maximum_absolute_error": 0.0 if exact else 0.01,
    }


def _runtime_gate_evidence(
    arm_id: str,
    *,
    batch_count: int = 1,
    scope: str = "mock",
) -> dict[str, object]:
    if arm_id == "native":
        return {
            "arm_id": arm_id,
            "source_call_audit": {
                "compiled_source_projection_call_count": 0,
                "compiled_source_fallback_call_count": 0,
                "native_reference_model_forward_count": 1,
                "passed": True,
            },
            "deterministic_replay": None,
            "runtime_gate_passed": True,
            "literal_parity": True,
            "execution_receipt_sha256s": [],
            "all_36_carrier_boundary_metrics": None,
            "first_divergence_occurrence": None,
            "all_36_mutation_input_delta_and_boundary_hash_audit": None,
            "selected_pair_input_delta_and_output_carrier_metrics": None,
        }
    exact = arm_id.startswith("exact.")
    comparisons: list[dict[str, object]] = []
    for batch_index in range(batch_count):
        for mutation_index in range(1, 37):
            reference_prefix = (
                f"{scope}.exact-reference.{batch_index}.{mutation_index}"
            )
            candidate_prefix = (
                f"{scope}.{arm_id}.{batch_index}.{mutation_index}"
            )
            exact_input = _digest(f"{reference_prefix}.input")
            exact_delta = _digest(f"{reference_prefix}.delta")
            exact_output = _digest(f"{reference_prefix}.output")
            comparisons.append(
                {
                    "batch_index": batch_index,
                    "mutation_index": mutation_index,
                    "exact_input_sha256": exact_input,
                    "candidate_input_sha256": (
                        exact_input
                        if exact
                        else _digest(f"{candidate_prefix}.candidate-input")
                    ),
                    "input_hash_equal": exact,
                    "exact_delta_sha256": exact_delta,
                    "candidate_delta_sha256": (
                        exact_delta
                        if exact
                        else _digest(f"{candidate_prefix}.candidate-delta")
                    ),
                    "delta_hash_equal": exact,
                    "exact_output_carrier_sha256": exact_output,
                    "candidate_output_carrier_sha256": (
                        exact_output
                        if exact
                        else _digest(f"{candidate_prefix}.candidate-output")
                    ),
                    "output_carrier_hash_equal": exact,
                }
            )
    mutation_audit = {
        "compared_batch_count": batch_count,
        "compared_mutation_count_per_batch": 36,
        "all_mutation_input_hashes_equal": exact,
        "all_mutation_delta_hashes_equal": exact,
        "all_output_carrier_hashes_equal": exact,
        "comparison_catalog_sha256": worker._manifest_sha256(comparisons),
        "ordered_comparisons": comparisons,
    }
    summary = _tensor_metric_summary(exact=exact)
    replay: dict[str, object] = {
        "batch_index": 0,
        "logits_bitwise_equal": True,
        "all_carrier_snapshots_bitwise_equal": True,
        "all_mutation_input_hashes_equal": True,
        "all_mutation_delta_hashes_equal": True,
        "all_output_carrier_hashes_equal": True,
        "passed": True,
    }
    replay["replay_receipt_sha256"] = worker._manifest_sha256(replay)
    return {
        "arm_id": arm_id,
        "source_call_audit": {
            "source_projection_call_count": 0,
            "source_fallback_call_count": 0,
            "passed": True,
        },
        "deterministic_replay": replay,
        "runtime_gate_passed": True,
        "literal_parity": exact,
        "execution_receipt_sha256s": [
            _digest(f"{scope}.{arm_id}.execution.{batch_index}")
            for batch_index in range(batch_count)
        ],
        "all_36_carrier_boundary_metrics": [
            {"mutation_index": index, **summary} for index in range(1, 37)
        ],
        "first_divergence_occurrence": None if exact else 1,
        "all_36_mutation_input_delta_and_boundary_hash_audit": mutation_audit,
        "selected_pair_input_delta_and_output_carrier_metrics": {
            label: {
                name: dict(summary)
                for name in ("input", "delta", "output_carrier")
            }
            for label in ("anchor", "member")
        },
    }


def _actual_assemble_inputs(tmp_path: Path) -> dict[str, object]:
    runtime = worker._Runtime(
        tokenizer=object(),
        model=object(),
        adapter=_FingerprintAdapter(),
        device="cpu",
        source_guard=_UnchangedGuard(),
        source_model_fingerprint_before=_PINNED_SOURCE_MODEL_SHA256,
        source_execution_fingerprint_before=_PINNED_SOURCE_EXECUTION_SHA256,
        exact_carrier=object(),
        exact_family=object(),
        exact_executor=object(),
        model_receipt={
            "model_id": PINNED_MODEL_ID,
            "requested_revision": PINNED_MODEL_REVISION,
            "parameter_count": PINNED_SOURCE_SCALARS,
        },
        artifact_root=tmp_path / "artifacts",
        artifact_receipts={},
    )
    typed_pair = _typed_pair_manifest()
    pair = {
        "anchor_invocation_id": "mutation.2",
        "member_invocation_id": "mutation.6",
        "anchor_block_ordinal": 0,
        "member_block_ordinal": 2,
        "selection_evidence_sha256": _digest("selection-evidence"),
        "exact_shared_core_state_sha256": _digest("anchor-exact-core"),
        "pair_manifest": typed_pair,
    }
    fits: dict[str, object] = {}
    for arm_id in _FIT_ARMS:
        setup_arm = next(
            arm for arm in _SETUP["arm_contract"]["arms"]
            if arm["arm_id"] == arm_id
        )
        fits[arm_id] = {
            "residual_rank": next(
                rank for rank in LOW_RANK_POINTS if arm_id.endswith(str(rank))
            ),
            "fit_nonce": setup_arm["fit_nonce"],
            "fit_seed": setup_arm["fit_seed"],
            "factor_catalog_sha256": _digest(arm_id),
            "fit_receipt": _projection_solver_receipt(arm_id),
            "selected_member_input_sha256": _digest(
                "exact-selected-member-input"
            ),
        }
    artifacts: dict[str, object] = {}
    shared_base_scalars = next(
        arm["candidate_stored_scalars"]
        for arm in _SETUP["arm_contract"]["arms"]
        if arm["arm_id"] == "ablation.shared_core_zero_residual"
    )
    materialized_blob_sha256s: set[str] = set()

    def component_blob(
        *,
        blob_kind: str,
        identity: str,
        stored_scalars: int,
    ) -> dict[str, object]:
        blob_sha256 = worker._manifest_sha256({"synthetic_blob": identity})
        newly_materialized = blob_sha256 not in materialized_blob_sha256s
        materialized_blob_sha256s.add(blob_sha256)
        return {
            "blob_kind": blob_kind,
            "blob_sha256": blob_sha256,
            "blob_file_basename": f"{blob_sha256}.pt",
            "blob_file_sha256": worker._manifest_sha256(
                {"synthetic_blob_file": identity}
            ),
            "blob_file_bytes": stored_scalars * 4 + 1,
            "tensor_state_sha256": worker._manifest_sha256(
                {"synthetic_tensor_state": identity}
            ),
            "ordered_state_names": [f"{identity}.tensor"],
            "stored_scalar_count": stored_scalars,
            "tensor_storage_bytes": stored_scalars * 4,
            "newly_materialized": newly_materialized,
        }

    for artifact_index, arm_id in enumerate(_ARMS[2:]):
        rank = next(
            (rank for rank in LOW_RANK_POINTS if arm_id.endswith(str(rank))),
            None,
        )
        fit_arm_id = (
            f"primary.shared_core_residual_r{rank}"
            if arm_id.startswith("control.shuffled")
            else arm_id if arm_id in fits else None
        )
        candidate_scalars = next(
            arm["candidate_stored_scalars"]
            for arm in _SETUP["arm_contract"]["arms"]
            if arm["arm_id"] == arm_id
        )
        components = [
            component_blob(
                blob_kind="shared_base",
                identity="shared-base",
                stored_scalars=shared_base_scalars,
            )
        ]
        if arm_id == "exact.shared_core_dense_residual":
            components.append(
                component_blob(
                    blob_kind="dense_exact_member",
                    identity="dense-exact-member",
                    stored_scalars=candidate_scalars - shared_base_scalars,
                )
            )
        elif arm_id.startswith("primary.") or arm_id.startswith(
            "control.shuffled"
        ):
            assert rank is not None
            components.append(
                component_blob(
                    blob_kind="primary_adapter",
                    identity=f"primary-r{rank}",
                    stored_scalars=candidate_scalars - shared_base_scalars,
                )
            )
        elif arm_id.startswith("control.independent"):
            assert rank is not None
            components.append(
                component_blob(
                    blob_kind="independent_adapter",
                    identity=f"independent-r{rank}",
                    stored_scalars=candidate_scalars - shared_base_scalars,
                )
            )
        manifest_file_bytes = 4_096 + artifact_index
        referenced_blob_file_bytes = sum(
            int(value["blob_file_bytes"]) for value in components
        )
        newly_materialized_blob_file_bytes = sum(
            int(value["blob_file_bytes"])
            for value in components
            if value["newly_materialized"] is True
        )
        newly_materialized_tensor_storage_bytes = sum(
            int(value["tensor_storage_bytes"])
            for value in components
            if value["newly_materialized"] is True
        )
        artifacts[arm_id] = {
            "candidate_state_sha256": _digest(f"{arm_id}.state"),
            "candidate_stored_scalars": candidate_scalars,
            "fit_factor_catalog_sha256": (
                fits[fit_arm_id]["factor_catalog_sha256"]
                if fit_arm_id is not None else None
            ),
            "point_manifest_sha256": _digest(f"{arm_id}.point-manifest"),
            "artifact_sha256": _digest(f"{arm_id}.point-artifact"),
            "file_sha256": _digest(f"{arm_id}.file"),
            "path_basename": f"points/{arm_id.replace('.', '--')}.json",
            "serialized_file_bytes": manifest_file_bytes,
            "tensor_storage_bytes": candidate_scalars * 4,
            "logical_tensor_storage_bytes": candidate_scalars * 4,
            "serialized_stored_scalar_count": candidate_scalars,
            "tensor_state_sha256": _digest(f"{arm_id}.tensor-state"),
            "ordered_state_names_sha256": _digest(
                f"{arm_id}.ordered-state-names"
            ),
            "storage_alias_groups_sha256": _digest(
                f"{arm_id}.storage-alias-groups"
            ),
            "storage_alias_group_count": 0,
            "content_point_manifest_sha256": _digest(
                f"{arm_id}.content-point-manifest"
            ),
            "component_blob_receipts": components,
            "referenced_blob_count": len(components),
            "referenced_blob_file_bytes": referenced_blob_file_bytes,
            "referenced_tensor_storage_bytes": candidate_scalars * 4,
            "newly_materialized_blob_count": sum(
                value["newly_materialized"] is True for value in components
            ),
            "newly_materialized_blob_file_bytes": (
                newly_materialized_blob_file_bytes
            ),
            "newly_materialized_tensor_storage_bytes": (
                newly_materialized_tensor_storage_bytes
            ),
            "serialized_file_bytes_newly_materialized": (
                manifest_file_bytes + newly_materialized_blob_file_bytes
            ),
            "content_addressed_store": True,
            "per_arm_logical_manifest_materialized": True,
            "unique_scalar_inventory_and_physical_alias_receipt_present": True,
            "source_free_candidate_runtime": True,
        }
    arm_results = {
        arm_id: {
            "metrics": _live_phase_metric(arm_id),
            **_runtime_gate_evidence(arm_id),
        }
        for arm_id in _ARMS
    }
    observed_by_role = {
        "A_fit": _ARMS[:3],
        "A_select": _ARMS,
        "B_guard": _ARMS,
        "C_assessment": (_ARMS[0], _ARMS[1], _WINNER),
    }
    role_results: dict[str, object] = {}
    for role_id in ORDERED_ROLES:
        observed = observed_by_role[role_id]
        observed_results = {
            arm_id: deepcopy(arm_results[arm_id]) for arm_id in observed
        }
        role_manifest = _SETUP["data_firewall"]["roles"][role_id]
        role_batch_count = (role_manifest["example_count"] + 3) // 4
        for arm_id, value in observed_results.items():
            value.update(
                _runtime_gate_evidence(
                    arm_id,
                    batch_count=role_batch_count,
                    scope=role_id,
                )
            )
            retention = value["metrics"][
                "worst_family_native_correct_retention"
            ]
            value["metrics"]["family_native_correct_retention"] = {
                family_id: retention
                for family_id in role_manifest["family_ids"]
            }
        role_results[role_id] = {
            "role_id": role_id,
            "role_manifest_sha256": _SETUP["data_firewall"]["roles"][
                role_id
            ]["role_manifest_sha256"],
            "opened_once": True,
            "refit_performed": False,
            "arm_results": observed_results,
            "artifact_receipts": {
                arm_id: deepcopy(artifacts[arm_id])
                for arm_id in observed if arm_id in artifacts
            },
            "dense_exact_control_gate_passed": role_id == "A_fit",
        }
    frontier = worker._freeze_a_select_frontier_live(
        role_result=role_results["A_select"],
        setup=_SETUP,
    )
    guard = worker._run_b_guard_live(
        role_result=_guard_role_result(frontier),
        frontier=frontier,
        setup=_SETUP,
    )
    assessment = worker._run_c_assessment_live(
        role_result=role_results["C_assessment"],
        winner_arm_id=_WINNER,
        setup=_SETUP,
    )
    return {
        "preflight": _mock_preflight(tmp_path),
        "runtime": runtime,
        "pair": pair,
        "fits": fits,
        "role_results": role_results,
        "frontier": frontier,
        "guard": guard,
        "assessment": assessment,
        "device_name": "cpu",
        "dtype": "float32",
        "batch_size": 4,
    }


def _valid_report() -> dict[str, object]:
    global _VALID_REPORT_CACHE
    if _VALID_REPORT_CACHE is None:
        inputs = _actual_assemble_inputs(Path("/tmp/model-free-shared-family"))
        _VALID_REPORT_CACHE = worker._assemble_report(**inputs)
    return deepcopy(_VALID_REPORT_CACHE)


def _guard_role_result(
    frontier: Mapping[str, object] | None = None,
) -> dict[str, object]:
    frozen = _guard_frontier() if frontier is None else frontier
    families = tuple(
        _SETUP["data_firewall"]["roles"]["B_guard"]["family_ids"]
    )
    arm_ids = ["ablation.shared_core_zero_residual"]
    for rank in LOW_RANK_POINTS:
        arm_ids.extend(
            (
                f"primary.shared_core_residual_r{rank}",
                f"control.independent_member_r{rank}",
                f"control.shuffled_residual_r{rank}",
            )
        )
    arms: dict[str, object] = {}
    for arm_id in arm_ids:
        metrics = _live_phase_metric(arm_id)
        family_values: dict[str, object] = {}
        for family_index, family_id in enumerate(families):
            offset = -0.001 if family_index == 0 else 0.001
            family_values[family_id] = {
                **metrics,
                "delta_nll_per_token": (
                    float(metrics["delta_nll_per_token"]) + offset
                ),
                "native_correct_downstream_accuracy_retained": 0.96,
            }
        arms[arm_id] = {
            "metrics": metrics,
            "_family_metric_values": family_values,
            **_runtime_gate_evidence(
                arm_id,
                batch_count=5,
                scope="B_guard",
            ),
        }
    return {
        "role_id": "B_guard",
        "arm_results": arms,
        "artifact_receipts": {
            point["arm_id"]: {
                "candidate_state_sha256": point["artifact_state_sha256"],
                "candidate_stored_scalars": point[
                    "candidate_stored_scalars"
                ],
            }
            for point in frozen["ordered_points"]
        },
    }


def _guard_frontier() -> dict[str, object]:
    points = [
        {
            "arm_id": arm_id,
            "metrics_sha256": _digest(f"{arm_id}.guard-metrics"),
            "artifact_state_sha256": _digest(f"{arm_id}.guard-state"),
            "candidate_stored_scalars": next(
                arm["candidate_stored_scalars"]
                for arm in _SETUP["arm_contract"]["arms"]
                if arm["arm_id"] == arm_id
            ),
        }
        for arm_id in _PRIMARY_ARMS
    ]
    payload = {
        "ordered_primary_arm_ids": list(_PRIMARY_ARMS),
        "ordered_points": points,
        "selection_role": "A_select",
        "pair_anchor_topology_and_fits_frozen": True,
        "refit_or_pruning_performed": False,
    }
    return {
        **payload,
        "frontier_sha256": worker._manifest_sha256(payload),
        "frozen_before_B_guard": True,
    }


def _guard_point(result: Mapping[str, object], arm_id: str) -> Mapping[str, object]:
    return next(
        point for point in result["point_results"]
        if point["arm_id"] == arm_id
    )


def _rehashed(report: dict[str, object]) -> dict[str, object]:
    changed = deepcopy(report)
    changed.pop("report_sha256")
    changed["report_sha256"] = worker._report_sha256(changed)
    return changed


@pytest.mark.parametrize(
    "mutation",
    ("family_metric_input", "simultaneous_bound", "dense_hash_catalog"),
)
def test_strict_loader_recomputes_guard_and_dense_hash_evidence(
    mutation: str,
) -> None:
    changed = deepcopy(_valid_report())
    if mutation == "family_metric_input":
        guard = changed["B_guard"]
        family_id = _SETUP["data_firewall"]["roles"]["B_guard"][
            "family_ids"
        ][0]
        guard["family_metric_input_catalog"][_WINNER][family_id][
            "delta_nll_per_token"
        ] += 0.001
        guard["family_metric_input_catalog_sha256"] = (
            worker._manifest_sha256(guard["family_metric_input_catalog"])
        )
        _seal(guard, "B_guard_receipt_sha256")
    elif mutation == "simultaneous_bound":
        guard = changed["B_guard"]
        point = next(
            row for row in guard["point_results"]
            if row["arm_id"] == _WINNER
        )
        point["worst_family_retention_simultaneous_lower_bound"] += 0.001
        _seal(guard, "B_guard_receipt_sha256")
    else:
        role = changed["role_results"]["B_guard"]
        dense = next(
            row for row in role["arm_metrics"]
            if row["arm_id"] == "exact.shared_core_dense_residual"
        )
        audit = dense[
            "all_36_mutation_input_delta_and_boundary_hash_audit"
        ]
        comparison = audit["ordered_comparisons"][0]
        forged = _digest("forged-dense-equal-input")
        comparison["exact_input_sha256"] = forged
        comparison["candidate_input_sha256"] = forged
        audit["comparison_catalog_sha256"] = worker._manifest_sha256(
            audit["ordered_comparisons"]
        )
        _seal(role, "role_result_sha256")

    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
            _rehashed(changed)
        )


@pytest.mark.parametrize(
    "mutation",
    ("literal_parity_metric_forgery", "nll_delta_inconsistency"),
)
def test_strict_loader_binds_identity_metrics_and_native_nll_derivation(
    mutation: str,
) -> None:
    changed = deepcopy(_valid_report())
    role = changed["role_results"]["B_guard"]
    arm_id = (
        "exact.shared_core_dense_residual"
        if mutation == "literal_parity_metric_forgery"
        else _WINNER
    )
    row = next(
        value for value in role["arm_metrics"] if value["arm_id"] == arm_id
    )
    if mutation == "literal_parity_metric_forgery":
        assert row["literal_parity"] is True
        row["nll_per_token"] += 0.01
        row["delta_nll_per_token"] = 0.01
    else:
        row["nll_per_token"] += 0.25
    _seal(role, "role_result_sha256")

    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
            _rehashed(changed)
        )


def test_exact_identity_allows_zero_denominator_family_retention_only_by_native_binding() -> None:
    changed = deepcopy(_valid_report())
    role = changed["role_results"]["A_fit"]
    zero_denominator_family = "integer_addition"
    retained_family = "integer_subtraction"
    for arm_id in (
        "native",
        "exact.heterogeneous_full_rank_atom_bank_lift",
        "exact.shared_core_dense_residual",
    ):
        row = next(
            value for value in role["arm_metrics"]
            if value["arm_id"] == arm_id
        )
        row["family_native_correct_retention"] = {
            zero_denominator_family: 0.0,
            retained_family: 1.0,
        }
        row["worst_family_native_correct_retention"] = 0.0
    _seal(role, "role_result_sha256")
    valid = _rehashed(changed)
    assert worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
        valid
    ) == valid

    drifted = deepcopy(valid)
    dense = next(
        value for value in drifted["role_results"]["A_fit"]["arm_metrics"]
        if value["arm_id"] == "exact.shared_core_dense_residual"
    )
    dense["family_native_correct_retention"][zero_denominator_family] = 1.0
    dense["worst_family_native_correct_retention"] = 1.0
    _seal(
        drifted["role_results"]["A_fit"],
        "role_result_sha256",
    )
    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
            _rehashed(drifted)
        )


def test_public_contract_is_pinned_to_the_successor_setup() -> None:
    setup = build_gemma3_mlp_shared_family_residual_adapter_setup()
    assert worker.PINNED_SETUP_SHA256 == setup["setup_sha256"]
    assert worker.GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_SCHEMA == (
        "fisher_graph.gemma3_mlp_shared_family_residual_adapter.experiment.v1"
    )
    assert worker.GEMMA3_MLP_SHARED_FAMILY_RESIDUAL_ADAPTER_EXPERIMENT_FORMAT_VERSION == 1
    assert tuple(setup["data_firewall"]["ordered_roles"]) == ORDERED_ROLES
    assert tuple(setup["arm_contract"]["residual_rank_order"][1:]) == (
        LOW_RANK_POINTS
    )


def test_worker_exposes_the_frozen_orchestration_seams() -> None:
    assert tuple(
        inspect.signature(
            worker.run_gemma3_mlp_shared_family_residual_adapter_experiment
        ).parameters
    ) == (
        "revision",
        "output",
        "setup_path",
        "predecessor_report_path",
        "panel_path",
        "model_id",
        "cache_dir",
        "device_name",
        "dtype",
        "batch_size",
    )
    for name in (
        "_preflight",
        "_load_live_dependencies",
        "_prepare_runtime",
        "_capture_a_fit",
        "_select_pair",
        "_fit_rank_point",
        "_evaluate_role",
        "_freeze_a_select_frontier",
        "_run_b_guard",
        "_run_c_assessment",
        "_report_sha256",
    ):
        assert callable(getattr(worker, name))


def test_report_hash_rejects_nonfinite_values() -> None:
    with pytest.raises((TypeError, ValueError)):
        worker._report_sha256({"metric": float("nan")})


def test_default_runner_arguments_remain_pinned() -> None:
    assert PINNED_MODEL_ID == "google/gemma-3-270m"
    assert PINNED_MODEL_REVISION == (
        "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
    )
    assert PINNED_SOURCE_SCALARS == 268_098_176
    assert PINNED_PREDECESSOR_REPORT_PATH == Path(
        ".local-runs/google--gemma-3-270m/"
        "heterogeneous-generator-family-exact-v1.json"
    )
    assert worker.DEFAULT_OUTPUT.name.endswith("residual-adapter-v1.json")


def test_strict_loader_rejects_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps({"report_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError):
        worker.load_gemma3_mlp_shared_family_residual_adapter_report(source)


def test_production_shaped_model_free_report_strictly_round_trips(
    tmp_path: Path,
) -> None:
    report = _valid_report()
    assert worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
        report
    ) == report
    destination = tmp_path / "report.json"
    assert worker.write_gemma3_mlp_shared_family_residual_adapter_report(
        report,
        destination,
    ) == report
    assert worker.load_gemma3_mlp_shared_family_residual_adapter_report(
        destination
    ) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        worker.write_gemma3_mlp_shared_family_residual_adapter_report(
            report,
            destination,
        )


def test_public_run_is_model_free_and_preserves_the_frozen_phase_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected, events = _install_successful_public_mock(monkeypatch, tmp_path)
    output = tmp_path / "public-report.json"
    observed = worker.run_gemma3_mlp_shared_family_residual_adapter_experiment(
        output=output,
    )
    assert observed == expected
    assert worker.load_gemma3_mlp_shared_family_residual_adapter_report(
        output
    ) == expected
    fit_events = [event for event in events if event.startswith("fit.")]
    assert fit_events == [f"fit.{arm_id}" for arm_id in _FIT_ARMS]
    assert events[:5] == [
        "preflight",
        "live_dependencies",
        "prepare",
        "capture.A_fit",
        "select_pair.A_fit",
    ]
    assert events[5] == "evaluate.A_fit"
    assert events[-7:] == [
        "evaluate.A_select",
        "freeze.A_select",
        "evaluate.B_guard",
        "guard.B_guard",
        "evaluate.C_assessment",
        "assess.C_assessment",
        "assemble",
    ]


def test_actual_assembler_emits_a_strict_loadable_production_report(
    tmp_path: Path,
) -> None:
    inputs = _actual_assemble_inputs(tmp_path)
    report = worker._assemble_report(**inputs)
    expected_observed = {
        "A_fit": list(_ARMS[:3]),
        "A_select": list(_ARMS),
        "B_guard": list(_ARMS),
        "C_assessment": [_ARMS[0], _ARMS[1], _WINNER],
    }
    for role_id, arm_ids in expected_observed.items():
        role = report["role_results"][role_id]
        assert role["opened"] is True
        assert role["ordered_observed_arm_ids"] == arm_ids
        assert [row["arm_id"] for row in role["arm_metrics"]] == arm_ids
        assert role["skipped_reason"] is None
    assert worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
        report
    ) == report
    assert report["model"]["source_model_fingerprint_before"] == (
        _PINNED_SOURCE_MODEL_SHA256
    )
    assert report["model"]["source_execution_fingerprint_before"] == (
        _PINNED_SOURCE_EXECUTION_SHA256
    )
    fit_by_arm = {item["arm_id"]: item for item in report["fit_receipts"]}
    for arm_id, fit in inputs["fits"].items():
        assert fit_by_arm[arm_id]["factor_catalog_sha256"] == (
            fit["factor_catalog_sha256"]
        )
    prefix_inputs = report["execution_audit"][
        "selected_member_input_sha256_by_fit_arm"
    ]
    assert set(prefix_inputs) == set(_FIT_ARMS)
    assert len(set(prefix_inputs.values())) == 1
    assert not set(prefix_inputs).intersection(
        {
            "exact.shared_core_dense_residual",
            "ablation.shared_core_zero_residual",
            *(f"control.shuffled_residual_r{rank}" for rank in LOW_RANK_POINTS),
        }
    )


def test_actual_assembler_emits_an_honest_strict_loadable_b_failure(
    tmp_path: Path,
) -> None:
    inputs = _actual_assemble_inputs(tmp_path)
    frontier = inputs["frontier"]
    guard_role = _guard_role_result(frontier)
    for arm_id in _PRIMARY_ARMS:
        for family_row in guard_role["arm_results"][arm_id][
            "_family_metric_values"
        ].values():
            family_row[
                "native_correct_downstream_accuracy_retained"
            ] = 0.70
        public_metrics = inputs["role_results"]["B_guard"]["arm_results"][
            arm_id
        ]["metrics"]
        public_metrics["native_correct_downstream_accuracy_retained"] = 0.70
        public_metrics["worst_family_native_correct_retention"] = 0.70
        public_metrics["family_native_correct_retention"] = {
            family_id: 0.70
            for family_id in _SETUP["data_firewall"]["roles"]["B_guard"][
                "family_ids"
            ]
        }
    inputs["guard"] = worker._run_b_guard_live(
        role_result=guard_role,
        frontier=frontier,
        setup=_SETUP,
    )
    assert inputs["guard"]["winner_arm_id"] is None
    assert inputs["guard"]["passed"] is False
    c_manifest = _SETUP["data_firewall"]["roles"]["C_assessment"]
    inputs["role_results"]["C_assessment"] = {
        "role_id": "C_assessment",
        "role_manifest_sha256": c_manifest["role_manifest_sha256"],
        "opened_once": False,
        "refit_performed": False,
        "arm_results": {},
        "artifact_receipts": {},
        "dense_exact_control_gate_passed": False,
        "skipped_reason": "B_guard_no_winner",
    }
    inputs["assessment"] = {
        "role_id": "C_assessment",
        "opened_at_most_once": True,
        "opened_once": False,
        "candidate_count": 0,
        "winner_arm_id": None,
        "same_point_thresholds_as_B_guard": True,
        "no_refit_selection_threshold_or_pair_change": True,
        "passed": False,
        "mechanism_qualified": False,
        "skipped_reason": "B_guard_no_winner",
    }

    report = worker._assemble_report(**inputs)
    assert report["diagnostic_outcome"] == (
        "shared_family_pair_mechanism_not_qualified"
    )
    assert report["role_results"]["C_assessment"]["opened"] is False
    assert report["role_results"]["C_assessment"][
        "ordered_observed_arm_ids"
    ] == []
    assert report["role_results"]["C_assessment"]["arm_metrics"] == []
    assert report["C_assessment"]["skipped_reason"] == "B_guard_no_winner"
    assert worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
        report
    ) == report
    destination = tmp_path / "honest-failure.json"
    worker.write_gemma3_mlp_shared_family_residual_adapter_report(
        report,
        destination,
    )
    assert worker.load_gemma3_mlp_shared_family_residual_adapter_report(
        destination
    ) == report


def test_open_c_failure_revokes_mechanism_qualification(
    tmp_path: Path,
) -> None:
    inputs = _actual_assemble_inputs(tmp_path)
    winner = inputs["role_results"]["C_assessment"]["arm_results"][
        _WINNER
    ]
    family_id = _SETUP["data_firewall"]["roles"]["C_assessment"][
        "family_ids"
    ][0]
    winner["metrics"]["native_correct_downstream_accuracy_retained"] = 0.80
    winner["metrics"]["worst_family_native_correct_retention"] = 0.80
    winner["metrics"]["family_native_correct_retention"] = {
        family_id: 0.80
    }
    inputs["assessment"] = worker._run_c_assessment_live(
        role_result=inputs["role_results"]["C_assessment"],
        winner_arm_id=_WINNER,
        setup=_SETUP,
    )
    assert inputs["assessment"]["passed"] is False

    report = worker._assemble_report(**inputs)
    assert report["C_assessment"]["opened_once"] is True
    assert report["C_assessment"]["passed"] is False
    assert report["claims"]["pair_mechanism_qualified"] is False
    assert report["gate_chain"]["all_passed"] is False
    assert worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
        report
    ) == report


def test_live_b_guard_runs_deterministic_paired_family_clustered_bootstrap() -> None:
    first = worker._run_b_guard_live(
        role_result=_guard_role_result(),
        frontier=_guard_frontier(),
        setup=_SETUP,
    )
    second = worker._run_b_guard_live(
        role_result=_guard_role_result(),
        frontier=_guard_frontier(),
        setup=_SETUP,
    )
    assert first == second
    assert first["bootstrap_seed"] == 1_733
    assert first["bootstrap_resamples"] == 10_000
    assert first["paired_family_clustered"] is True
    assert first["simultaneous_method"] == "centered_max_deviation"
    assert worker._is_sha256(first["bootstrap_draw_catalog_sha256"])
    assert first["family_metric_input_catalog_sha256"] == (
        worker._manifest_sha256(first["family_metric_input_catalog"])
    )
    assert first["runtime_source_gate_input_catalog_sha256"] == (
        worker._manifest_sha256(first["runtime_source_gate_input_catalog"])
    )
    assert all(
        row["runtime_gate_passed"] is True
        for row in first["runtime_source_gate_input_catalog"].values()
    )
    assert first["winner_arm_id"] == _WINNER
    for point in first["point_results"]:
        assert point["worst_family_retention_simultaneous_lower_bound"] >= 0.85
        assert point["real_residual_gain_simultaneous_lower_bound"] > 0.0
        assert point[
            "matched_independent_gain_simultaneous_lower_bound"
        ] > 0.0
        assert point[
            "shuffled_gain_fraction_simultaneous_upper_bound"
        ] <= 0.25
        assert point["worst_family_gate_passed"] is True
        assert point["control_lower_bound_gates_passed"] is True
        assert point["shuffled_upper_bound_gate_passed"] is True


def test_live_b_guard_worst_family_lower_bound_can_disqualify_one_point() -> None:
    role = _guard_role_result()
    family_id = _SETUP["data_firewall"]["roles"]["B_guard"]["family_ids"][0]
    role["arm_results"][_WINNER]["_family_metric_values"][family_id][
        "native_correct_downstream_accuracy_retained"
    ] = 0.70
    result = worker._run_b_guard_live(
        role_result=role,
        frontier=_guard_frontier(),
        setup=_SETUP,
    )
    point = _guard_point(result, _WINNER)
    assert point["worst_family_retention_simultaneous_lower_bound"] < 0.85
    assert point["worst_family_gate_passed"] is False
    assert point["passed"] is False
    assert result["winner_arm_id"] == "primary.shared_core_residual_r64"


@pytest.mark.parametrize("control_kind", ("independent", "shuffled"))
def test_live_b_guard_control_bounds_can_disqualify_one_point(
    control_kind: str,
) -> None:
    role = _guard_role_result()
    if control_kind == "independent":
        control_id = "control.independent_member_r32"
        primary = role["arm_results"][_WINNER]
        control = role["arm_results"][control_id]
        control["metrics"]["delta_nll_per_token"] = primary["metrics"][
            "delta_nll_per_token"
        ]
        for family_id in control["_family_metric_values"]:
            control["_family_metric_values"][family_id][
                "delta_nll_per_token"
            ] = primary["_family_metric_values"][family_id][
                "delta_nll_per_token"
            ]
    else:
        control_id = "control.shuffled_residual_r32"
        zero = role["arm_results"]["ablation.shared_core_zero_residual"]
        primary = role["arm_results"][_WINNER]
        control = role["arm_results"][control_id]
        real_gain = (
            zero["metrics"]["delta_nll_per_token"]
            - primary["metrics"]["delta_nll_per_token"]
        )
        control["metrics"]["delta_nll_per_token"] = (
            zero["metrics"]["delta_nll_per_token"] - real_gain
        )
        for family_id in control["_family_metric_values"]:
            family_zero = zero["_family_metric_values"][family_id]
            family_primary = primary["_family_metric_values"][family_id]
            control["_family_metric_values"][family_id][
                "delta_nll_per_token"
            ] = family_primary["delta_nll_per_token"]
            assert family_zero["delta_nll_per_token"] > (
                family_primary["delta_nll_per_token"]
            )
    result = worker._run_b_guard_live(
        role_result=role,
        frontier=_guard_frontier(),
        setup=_SETUP,
    )
    point = _guard_point(result, _WINNER)
    assert point["passed"] is False
    if control_kind == "independent":
        assert point["matched_independent_gain_simultaneous_lower_bound"] <= 0.0
        assert point["control_lower_bound_gates_passed"] is False
    else:
        assert point["shuffled_gain_fraction_simultaneous_upper_bound"] > 0.25
        assert point["shuffled_upper_bound_gate_passed"] is False


def test_live_b_guard_rejects_seed_or_frontier_tampering() -> None:
    changed_setup = deepcopy(_SETUP)
    changed_setup["acceptance_gates"]["B_guard_bootstrap"]["seed"] += 1
    with pytest.raises(ValueError):
        worker._run_b_guard_live(
            role_result=_guard_role_result(),
            frontier=_guard_frontier(),
            setup=changed_setup,
        )
    changed_frontier = _guard_frontier()
    changed_frontier["frontier_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        worker._run_b_guard_live(
            role_result=_guard_role_result(),
            frontier=changed_frontier,
            setup=_SETUP,
        )


def test_live_c_assessment_enforces_frozen_worst_family_retention() -> None:
    metrics = _live_phase_metric(_WINNER)
    metrics["worst_family_native_correct_retention"] = 0.85
    winner = {"metrics": metrics, **_runtime_gate_evidence(_WINNER)}
    passing = worker._run_c_assessment_live(
        role_result={
            "role_id": "C_assessment",
            "arm_results": {_WINNER: winner},
        },
        winner_arm_id=_WINNER,
        setup=_SETUP,
    )
    assert passing["passed"] is True
    assert passing["mechanism_qualified"] is True
    assert passing["runtime_and_source_call_gates_passed"] is True

    metrics["worst_family_native_correct_retention"] = 0.849
    failing = worker._run_c_assessment_live(
        role_result={
            "role_id": "C_assessment",
            "arm_results": {_WINNER: winner},
        },
        winner_arm_id=_WINNER,
        setup=_SETUP,
    )
    assert failing["passed"] is False
    assert failing["mechanism_qualified"] is False
    assert failing["worst_family_retention_gate_passed"] is False

    metrics["worst_family_native_correct_retention"] = 0.85
    winner["source_call_audit"]["passed"] = False
    runtime_failure = worker._run_c_assessment_live(
        role_result={
            "role_id": "C_assessment",
            "arm_results": {_WINNER: winner},
        },
        winner_arm_id=_WINNER,
        setup=_SETUP,
    )
    assert runtime_failure["runtime_and_source_call_gates_passed"] is False
    assert runtime_failure["passed"] is False


def test_failed_dense_exact_control_stops_before_any_compressed_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pair = _pair_selection()
    pair["dense_exact_control_gate_passed"] = False
    _seal(pair, "pair_manifest_sha256")
    monkeypatch.setattr(worker, "_preflight", lambda **_kwargs: _mock_preflight(tmp_path))
    monkeypatch.setattr(worker, "_load_live_dependencies", SimpleNamespace)
    monkeypatch.setattr(worker, "_prepare_runtime", lambda **_kwargs: object())
    monkeypatch.setattr(worker, "_capture_a_fit", lambda **_kwargs: object())
    monkeypatch.setattr(worker, "_select_pair", lambda **_kwargs: pair)
    monkeypatch.setattr(
        worker,
        "_evaluate_role",
        lambda **_kwargs: {"dense_exact_control_gate_passed": False},
    )
    monkeypatch.setattr(
        worker,
        "_fit_rank_point",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("compressed fit ran after dense exact gate failure")
        ),
    )
    with pytest.raises(RuntimeError, match="dense exact"):
        worker.run_gemma3_mlp_shared_family_residual_adapter_experiment(
            output=tmp_path / "report.json",
        )


def test_failed_b_guard_never_opens_c_assessment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _expected, events = _install_successful_public_mock(monkeypatch, tmp_path)
    monkeypatch.setattr(
        worker,
        "_run_b_guard",
        lambda **_kwargs: {
            "opened_once": True,
            "passed": False,
            "winner_arm_id": None,
        },
    )
    captured: dict[str, object] = {}

    def capture_failure(**kwargs: object) -> object:
        events.append("assemble")
        captured.update(kwargs)
        raise RuntimeError("captured honest B failure")

    monkeypatch.setattr(worker, "_assemble_report", capture_failure)
    with pytest.raises(RuntimeError, match="captured honest B failure"):
        worker.run_gemma3_mlp_shared_family_residual_adapter_experiment(
            output=tmp_path / "report.json",
        )
    assert "evaluate.C_assessment" not in events
    assert "assess.C_assessment" not in events
    assert events[-1] == "assemble"
    c_role = captured["role_results"]["C_assessment"]
    assert c_role["opened_once"] is False
    assert c_role["arm_results"] == {}
    assert c_role["skipped_reason"] == "B_guard_no_winner"
    assessment = captured["assessment"]
    assert assessment["opened_once"] is False
    assert assessment["candidate_count"] == 0
    assert assessment["winner_arm_id"] is None
    assert assessment["passed"] is False
    assert assessment["mechanism_qualified"] is False


def _rebind_pair_lineage(report: dict[str, object]) -> None:
    pair = report["pair_selection"]
    _seal(pair, "pair_manifest_sha256")
    pair_sha256 = pair["pair_manifest_sha256"]
    fit_sha256_by_arm: dict[str, object] = {}
    for receipt in report["fit_receipts"]:
        receipt["pair_manifest_sha256"] = pair_sha256
        _seal(receipt, "fit_receipt_sha256")
        fit_sha256_by_arm[receipt["arm_id"]] = receipt["fit_receipt_sha256"]
    for artifact in report["arm_artifacts"]:
        artifact["pair_manifest_sha256"] = pair_sha256
        if artifact["arm_id"] in fit_sha256_by_arm:
            artifact["fit_receipt_sha256"] = fit_sha256_by_arm[
                artifact["arm_id"]
            ]
        _seal(artifact, "artifact_manifest_sha256")
    for role_id, result in report["role_results"].items():
        result["pair_manifest_sha256"] = pair_sha256
        _seal(result, "role_result_sha256")
    frontier = report["frontier"]
    frontier["pair_manifest_sha256"] = pair_sha256
    frontier["A_select_role_result_sha256"] = report["role_results"][
        "A_select"
    ]["role_result_sha256"]
    _seal(frontier, "frontier_sha256")
    report["B_guard"]["frontier_sha256"] = frontier["frontier_sha256"]
    report["execution_audit"]["pair_manifest_sha256_by_role"] = {
        role_id: pair_sha256 for role_id in ORDERED_ROLES
    }


def test_report_authenticates_frozen_content_addressed_store_totals() -> None:
    resources = _valid_report()["resources"]
    store = resources["content_addressed_store"]
    preflight = resources["artifact_storage_preflight"]
    assert store["point_manifest_count"] == 14
    assert store["unique_blob_count"] == 10
    assert store["component_blob_reference_count"] == 27
    assert store["reused_component_blob_reference_count"] == 17
    assert store["unique_tensor_storage_bytes"] == 1_103_399_424
    assert store["logical_flat_tensor_storage_bytes"] == 14_855_469_056
    assert preflight["required_free_bytes"] == 2_177_141_248


@pytest.mark.parametrize(
    "field",
    (
        "point_manifest_count",
        "unique_blob_count",
        "component_blob_reference_count",
        "reused_component_blob_reference_count",
        "unique_tensor_storage_bytes",
        "logical_flat_tensor_storage_bytes",
    ),
)
def test_strict_loader_rejects_rehashed_content_store_total_tampering(
    field: str,
) -> None:
    changed = _valid_report()
    store = changed["resources"]["content_addressed_store"]
    store[field] += 1
    store_payload = dict(store)
    store_payload.pop("content_addressed_store_receipt_sha256")
    store["content_addressed_store_receipt_sha256"] = worker._manifest_sha256(
        store_payload
    )

    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
            _rehashed(changed)
        )


def test_strict_loader_rejects_rehashed_preflight_capacity_formula_tampering() -> None:
    changed = _valid_report()
    preflight = changed["resources"]["artifact_storage_preflight"]
    preflight["required_free_bytes"] += 1
    preflight["available_free_bytes"] += 1

    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
            _rehashed(changed)
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "setup_path",
        "predecessor_required_arm",
        "model_id",
        "source_fingerprint_equal_forgery",
        "data_role_overlap",
        "configuration_arm_order",
        "configuration_fit_seed",
        "configuration_threshold",
        "pair_adjacent_coordinated",
        "pair_anchor_source_drift",
        "pair_role_manifest",
        "role_pair_freeze",
        "fit_rank_label",
        "fit_nonce_duplicate",
        "fit_supernet",
        "fit_pair_lineage",
        "artifact_resource_coordinated",
        "artifact_learned_edge",
        "artifact_source_call",
        "role_arm_order",
        "role_manifest_swap",
        "a_select_refit",
        "exact_prefix_proof",
        "frontier_drop_point",
        "frontier_pair_lineage",
        "guard_control_winner",
        "guard_bootstrap_seed",
        "guard_refit",
        "assessment_candidate",
        "assessment_count",
        "assessment_refit",
        "dense_gate_order",
        "source_execution_changed",
        "fit_call_count",
        "matched_budget",
        "assessment_claim_incoherence",
        "nested_extra_field",
        "nested_missing_field",
    ),
)
def test_strict_loader_rejects_coordinated_rehashed_semantic_tampering(
    mutation: str,
) -> None:
    changed = deepcopy(_valid_report())
    if mutation == "setup_path":
        changed["setup"]["path"] = "forged-setup.json"
    elif mutation == "predecessor_required_arm":
        changed["predecessor"]["required_arm_id"] = "native"
    elif mutation == "model_id":
        changed["model"]["model_id"] = "forged/model"
    elif mutation == "source_fingerprint_equal_forgery":
        changed["model"]["source_model_fingerprint_before"] = _digest("forged")
        changed["model"]["source_model_fingerprint_after"] = _digest("forged")
    elif mutation == "data_role_overlap":
        changed["data"]["roles"]["A_select"] = deepcopy(
            changed["data"]["roles"]["A_fit"]
        )
        changed["data"]["role_family_overlap"] = False
        changed["data"]["role_example_overlap"] = False
    elif mutation == "configuration_arm_order":
        changed["configuration"]["ordered_arm_ids"].reverse()
    elif mutation == "configuration_fit_seed":
        changed["configuration"]["seeds"]["adapter_fit"] += 1
    elif mutation == "configuration_threshold":
        changed["configuration"]["acceptance_gates"][
            "compressed_candidate_fidelity"
        ]["delta_nll_per_token_maximum"] = 1.0
    elif mutation == "pair_adjacent_coordinated":
        pair = changed["pair_selection"]
        pair["member_invocation_id"] = "mutation.4"
        pair["member_atom_id"] = "atom.mlp.layer.1"
        pair["member_block_ordinal"] = 1
        _rebind_pair_lineage(changed)
    elif mutation == "pair_anchor_source_drift":
        changed["pair_selection"]["shared_core_state_sha256"] = _digest(
            "wrong-core"
        )
        _seal(changed["pair_selection"], "pair_manifest_sha256")
    elif mutation == "pair_role_manifest":
        changed["pair_selection"]["a_fit_role_manifest_sha256"] = changed[
            "data"
        ]["roles"]["A_select"]["role_manifest_sha256"]
        _rebind_pair_lineage(changed)
    elif mutation == "role_pair_freeze":
        role = changed["role_results"]["B_guard"]
        role["pair_manifest_sha256"] = _digest("other-pair")
        _seal(role, "role_result_sha256")
    elif mutation == "fit_rank_label":
        changed["fit_receipts"][-2]["rank"] = 64
        _seal(changed["fit_receipts"][-2], "fit_receipt_sha256")
    elif mutation == "fit_nonce_duplicate":
        changed["fit_receipts"][1]["fit_nonce"] = changed["fit_receipts"][0][
            "fit_nonce"
        ]
        _seal(changed["fit_receipts"][1], "fit_receipt_sha256")
    elif mutation == "fit_supernet":
        changed["fit_receipts"][0]["independently_fitted"] = False
        changed["fit_receipts"][0]["supernet_or_factor_slicing_used"] = True
        _seal(changed["fit_receipts"][0], "fit_receipt_sha256")
    elif mutation == "fit_pair_lineage":
        changed["fit_receipts"][0]["pair_manifest_sha256"] = _digest(
            "other-pair"
        )
        _seal(changed["fit_receipts"][0], "fit_receipt_sha256")
    elif mutation == "artifact_resource_coordinated":
        artifact = changed["arm_artifacts"][-3]
        artifact["candidate_stored_scalars"] += 1
        changed["resources"]["candidate_stored_scalars_by_arm"][
            artifact["arm_id"]
        ] = artifact["candidate_stored_scalars"]
        _seal(artifact, "artifact_manifest_sha256")
    elif mutation == "artifact_learned_edge":
        artifact = changed["arm_artifacts"][-3]
        artifact["learned_correction_edge_count"] = 1
        _seal(artifact, "artifact_manifest_sha256")
    elif mutation == "artifact_source_call":
        artifact = changed["arm_artifacts"][-3]
        artifact["source_projection_and_fallback_calls"] = 1
        _seal(artifact, "artifact_manifest_sha256")
    elif mutation == "role_arm_order":
        role = changed["role_results"]["B_guard"]
        role["ordered_observed_arm_ids"].reverse()
        role["arm_metrics"].reverse()
        _seal(role, "role_result_sha256")
    elif mutation == "role_manifest_swap":
        role = changed["role_results"]["A_select"]
        role["role_manifest_sha256"] = changed["data"]["roles"]["B_guard"][
            "role_manifest_sha256"
        ]
        _seal(role, "role_result_sha256")
    elif mutation == "a_select_refit":
        role = changed["role_results"]["A_select"]
        role["fit_performed"] = True
        _seal(role, "role_result_sha256")
    elif mutation == "exact_prefix_proof":
        changed["execution_audit"][
            "selected_member_prefix_is_exact_by_single_approximation"
        ] = False
        changed["execution_audit"]["candidate_prefix_recapture_required"] = True
        changed["execution_audit"][
            "exact_prefix_input_hash_equality_authenticated_on_A_fit"
        ] = False
        changed["execution_audit"]["selected_member_input_sha256_by_fit_arm"][
            _FIT_ARMS[-1]
        ] = _digest("forged-selected-member-input")
    elif mutation == "frontier_drop_point":
        changed["frontier"]["ordered_primary_arm_ids"].pop()
        changed["frontier"]["promotion_order"].pop(0)
        _seal(changed["frontier"], "frontier_sha256")
        changed["B_guard"]["frontier_sha256"] = changed["frontier"][
            "frontier_sha256"
        ]
    elif mutation == "frontier_pair_lineage":
        changed["frontier"]["pair_manifest_sha256"] = _digest("other-pair")
        _seal(changed["frontier"], "frontier_sha256")
        changed["B_guard"]["frontier_sha256"] = changed["frontier"][
            "frontier_sha256"
        ]
    elif mutation == "guard_control_winner":
        changed["B_guard"]["winner_arm_id"] = "control.independent_member_r32"
        changed["C_assessment"]["candidate_arm_id"] = (
            "control.independent_member_r32"
        )
    elif mutation == "guard_bootstrap_seed":
        changed["B_guard"]["bootstrap_seed"] += 1
    elif mutation == "guard_refit":
        changed["B_guard"]["refit_reorder_prune_or_threshold_change"] = True
    elif mutation == "assessment_candidate":
        changed["C_assessment"]["candidate_arm_id"] = _PRIMARY_ARMS[0]
    elif mutation == "assessment_count":
        changed["C_assessment"]["candidate_count"] = 2
    elif mutation == "assessment_refit":
        changed["C_assessment"]["refit_selection_or_threshold_change"] = True
    elif mutation == "dense_gate_order":
        changed["gate_chain"][
            "dense_exact_control_passed_before_compressed_fit"
        ] = False
    elif mutation == "source_execution_changed":
        changed["execution_audit"]["source_execution_unchanged"] = False
    elif mutation == "fit_call_count":
        changed["execution_audit"]["fit_call_count"] = 7
    elif mutation == "matched_budget":
        changed["resources"]["matched_control_budget_equal_by_rank"] = False
    elif mutation == "assessment_claim_incoherence":
        changed["C_assessment"]["passed"] = False
        changed["claims"]["pair_mechanism_qualified"] = True
    elif mutation == "nested_extra_field":
        changed["pair_selection"]["forged"] = True
        _rebind_pair_lineage(changed)
    elif mutation == "nested_missing_field":
        del changed["resources"]["latency_measured"]
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError):
        worker.validate_gemma3_mlp_shared_family_residual_adapter_report(
            _rehashed(changed)
        )


def test_preflight_failure_is_lazy_and_never_loads_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def reject_preflight(**_kwargs: object) -> object:
        calls.append("preflight")
        raise ValueError("predecessor gate failed")

    def forbid_live_dependencies() -> object:
        calls.append("live_dependencies")
        raise AssertionError("live dependencies loaded before preflight")

    monkeypatch.setattr(worker, "_preflight", reject_preflight)
    monkeypatch.setattr(
        worker,
        "_load_live_dependencies",
        forbid_live_dependencies,
    )
    with pytest.raises(ValueError, match="predecessor gate failed"):
        worker.run_gemma3_mlp_shared_family_residual_adapter_experiment(
            revision=PINNED_MODEL_REVISION,
            output=tmp_path / "report.json",
            setup_path=_SETUP_PATH,
            predecessor_report_path=tmp_path / "predecessor.json",
            panel_path=_ROOT / PINNED_PANEL_PATH,
            model_id=PINNED_MODEL_ID,
            cache_dir=None,
        )
    assert calls == ["preflight"]


def test_model_free_preflight_authenticates_roles_without_live_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predecessor = {
        "report_sha256": _digest("mocked-exact-predecessor"),
        "arm_results": [{
            "arm_id": "exact.heterogeneous_full_rank_atom_bank_lift",
            "gate": {"passed": True},
        }],
    }
    monkeypatch.setattr(
        worker,
        "_strict_predecessor_report",
        lambda _path: predecessor,
    )
    monkeypatch.setattr(
        worker,
        "_load_live_dependencies",
        lambda: (_ for _ in ()).throw(
            AssertionError("preflight loaded live dependencies")
        ),
    )
    preflight = worker._preflight(
        revision=PINNED_MODEL_REVISION,
        model_id=PINNED_MODEL_ID,
        output=tmp_path / "report.json",
        setup_path=_SETUP_PATH,
        predecessor_report_path=tmp_path / "predecessor.json",
        panel_path=_ROOT / PINNED_PANEL_PATH,
        device_name="cpu",
        dtype="float32",
        batch_size=4,
    )
    assert preflight.predecessor_report is predecessor
    assert tuple(preflight.role_examples) == ORDERED_ROLES
    assert [len(preflight.role_examples[role]) for role in ORDERED_ROLES] == [
        20,
        10,
        20,
        10,
    ]
    storage = preflight.receipts["artifact_storage_preflight"]
    assert storage["materialized_point_artifact_count"] == 14
    assert storage["unique_tensor_blob_count"] == 10
    assert storage["component_blob_reference_count"] == 27
    assert storage["reused_component_blob_reference_count"] == 17
    assert storage["logical_flat_tensor_storage_bytes"] == 14_855_469_056
    assert storage["estimated_tensor_storage_bytes"] == 1_103_399_424
    assert storage["unique_content_addressed_tensor_storage_bytes"] == (
        1_103_399_424
    )
    assert storage["content_addressed_tensor_storage_savings_bytes"] == (
        13_752_069_632
    )
    assert storage["required_free_bytes"] == 2_177_141_248
    assert storage["available_free_bytes"] >= storage["required_free_bytes"]
    assert storage["passed"] is True


def test_point_artifact_preflight_has_an_exact_one_byte_capacity_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    monkeypatch.setattr(
        worker.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1 << 62),
    )
    sizing = worker._artifact_storage_preflight(
        destination=destination,
        setup=_SETUP,
    )
    required = sizing["required_free_bytes"]

    monkeypatch.setattr(
        worker.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=required),
    )
    exact = worker._artifact_storage_preflight(
        destination=destination,
        setup=_SETUP,
    )
    assert exact["available_free_bytes"] == required
    assert exact["required_free_bytes"] == required
    assert exact["passed"] is True

    monkeypatch.setattr(
        worker.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=required - 1),
    )
    with pytest.raises(
        RuntimeError,
        match=rf"need {required} bytes, found {required - 1} bytes",
    ):
        worker._artifact_storage_preflight(
            destination=destination,
            setup=_SETUP,
        )


def test_content_addressed_storage_plan_has_frozen_full_store_totals() -> None:
    plan = worker._content_addressed_storage_plan(_SETUP)
    assert plan["materialized_point_artifact_count"] == 14
    assert plan["unique_tensor_blob_count"] == 10
    assert plan["component_blob_reference_count"] == 27
    assert plan["reused_component_blob_reference_count"] == 17
    assert plan["unique_content_addressed_tensor_storage_bytes"] == (
        1_103_399_424
    )
    assert plan["logical_flat_tensor_storage_bytes"] == 14_855_469_056
    assert plan["content_addressed_tensor_storage_savings_bytes"] == (
        13_752_069_632
    )
    assert plan["shuffled_adapter_blob_reuses_primary"] is True
    assert plan["logical_candidate_scalar_accounting_unchanged"] is True


def test_model_free_preflight_rejects_insufficient_point_artifact_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    estimated = 1_103_399_424
    required = estimated + max(1 << 30, (estimated + 9) // 10)
    available = required - 1
    monkeypatch.setattr(
        worker.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=available),
    )
    monkeypatch.setattr(
        worker,
        "_strict_predecessor_report",
        lambda _path: pytest.fail("predecessor opened after failed capacity gate"),
    )
    monkeypatch.setattr(
        worker,
        "_load_role_panel",
        lambda **_kwargs: pytest.fail("panel opened after failed capacity gate"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "insufficient free space for 14 authenticated candidate point "
            f"artifacts: need {required} bytes, found {available} bytes"
        ),
    ):
        worker._preflight(
            revision=PINNED_MODEL_REVISION,
            model_id=PINNED_MODEL_ID,
            output=tmp_path / "report.json",
            setup_path=_SETUP_PATH,
            predecessor_report_path=tmp_path / "must-not-open-predecessor.json",
            panel_path=tmp_path / "must-not-open-panel.json",
        )


def test_preflight_stops_at_predecessor_before_opening_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def reject_predecessor(_path: Path) -> object:
        calls.append("predecessor")
        raise ValueError("exact predecessor failed")

    def forbid_panel(**_kwargs: object) -> object:
        calls.append("panel")
        raise AssertionError("panel opened after predecessor failure")

    monkeypatch.setattr(
        worker,
        "_strict_predecessor_report",
        reject_predecessor,
    )
    monkeypatch.setattr(worker, "_load_role_panel", forbid_panel)
    with pytest.raises(ValueError, match="exact predecessor failed"):
        worker._preflight(
            revision=PINNED_MODEL_REVISION,
            model_id=PINNED_MODEL_ID,
            output=tmp_path / "report.json",
            setup_path=_SETUP_PATH,
            predecessor_report_path=tmp_path / "predecessor.json",
            panel_path=_ROOT / PINNED_PANEL_PATH,
            device_name="cpu",
            dtype="float32",
            batch_size=4,
        )
    assert calls == ["predecessor"]
