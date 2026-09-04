from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from fisher_graph.gemma3_attention_transport_sparse_v2_global_router_experiment import (
    REPORT_SCHEMA,
    _B_CORPUS_PATH,
    _B_CORPUS_SHA256,
    _B_MANIFEST_FILE_SHA256,
    _B_MANIFEST_KEY,
    _B_MANIFEST_PATH,
    _PRIOR_B_SPLIT_SHA256,
    _PRIOR_V1_REPORT_FILE_SHA256,
    _PRIOR_V1_REPORT_PATH,
    _PRIOR_V1_REPORT_SHA256,
    _publish_report,
    _seal_report,
    _source_sha256s,
    _strict_report,
    _validate_prior_exposure_files,
    _worst_summary,
    check_sparse_v2_global_router_B_replay,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _metric(rows: int, fisher: float) -> dict[str, object]:
    return {
        "row_count": rows,
        "fisher_loss": fisher,
        "directional_mse": fisher,
        "euclidean_mse": 0.0,
        "normalized_rmse": fisher,
        "cosine": 1.0 - fisher,
    }


def _sparse_metric(relative: float) -> dict[str, object]:
    result = _metric(12_960, relative)
    result.update(
        {
            "zero_predictor_fisher_loss": 1.0,
            "relative_fisher_loss": relative,
            "active_count_limit": 2,
            "coefficient_ridge": 1.0e-8,
            "per_invocation_relative_fisher_loss": [
                [invocation_id, relative]
                for invocation_id in sorted(
                    f"mutation.{2 * index + 1}" for index in range(18)
                )
            ],
        }
    )
    return result


def _resources() -> dict[str, object]:
    return {
        "native_v_o_parameters": 14_745_600,
        "generator_bank_parameters": 82_560,
        "chart_parameters": 23_041,
        "router_parameters": 19_584,
        "compiled_parameters": 125_185,
        "parameter_reduction_fraction": 1.0 - 125_185 / 14_745_600,
        "parameter_compression_eligible": True,
        "native_qk_retained": True,
        "native_output_rmsnorm_retained": True,
        "logical_speedup_claimed": False,
    }


def _invocation(invocation_id: str) -> dict[str, object]:
    return {
        "invocation_id": invocation_id,
        "operator": _metric(720, 0.08),
        "oracle": _metric(720, 0.03),
        "oracle_contract": {
            "solver_kind": "exact_support_enumeration_sizes_0_through_active_count",
            "active_count_limit": 2,
            "coefficient_ridge": 0.0,
            "heldout_ridge_free": True,
        },
        "zero_predictor_fisher_loss": 1.0,
        "relative_fisher_loss": 0.08,
        "oracle_relative_fisher_loss": 0.03,
        "oracle_router_fisher_gap": 0.05,
        "postnorm": {
            "row_count": 720,
            "normalized_rmse": 0.01,
            "cosine": 0.9995,
        },
        "per_chart_active_count": {
            "minimum": 2,
            "mean": 2.0,
            "maximum": 2,
        },
        "effective_support_count": {
            "minimum": 2,
            "mean": 2.0,
            "maximum": 2,
            "fraction": 0.25,
        },
        "sparse_probability_renormalization_max_error": 0.0,
        "generator_usage": [0.125] * 8,
        "dead_generator_count": 0,
    }


def _evaluation() -> dict[str, object]:
    invocation_ids = sorted(f"mutation.{2 * index + 1}" for index in range(18))
    return {
        "role": "B",
        "split_artifact_sha256": _PRIOR_B_SPLIT_SHA256,
        "row_count": 12_960,
        "invocation_count": 18,
        "fisher_loss": 0.08,
        "zero_predictor_fisher_loss": 1.0,
        "relative_fisher_loss": 0.08,
        "oracle_fisher_loss": 0.03,
        "oracle_contract": {
            "solver_kind": "exact_support_enumeration_sizes_0_through_active_count",
            "active_count_limit": 2,
            "coefficient_ridge": 0.0,
            "heldout_ridge_free": True,
        },
        "oracle_relative_fisher_loss": 0.03,
        "oracle_router_fisher_gap": 0.05,
        "operator_normalized_rmse": 0.08,
        "operator_cosine": 0.92,
        "postnorm_normalized_rmse": 0.01,
        "postnorm_cosine": 0.9995,
        "per_chart_active_count": {
            "minimum": 2,
            "mean": 2.0,
            "maximum": 2,
        },
        "effective_support_count": {
            "minimum": 2,
            "mean": 2.0,
            "maximum": 2,
            "fraction": 0.25,
        },
        "sparse_probability_renormalization_max_error": 0.0,
        "generator_usage": [0.125] * 8,
        "dead_generator_count": 0,
        "resources": _resources(),
        "per_invocation": [_invocation(value) for value in invocation_ids],
    }


def _unsealed_report(root: Path) -> dict[str, object]:
    a_receipts = sorted(_sha(f"A-{index}") for index in range(45))
    b_receipts = sorted(_sha(f"B-{index}") for index in range(45))
    candidate_sha = _sha("candidate")
    development_sha = _sha("development")
    selection_sha = _sha("selection")
    evaluation = _evaluation()
    return {
        "schema": REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "A_fitted": True,
            "B_previously_exposed_replay": True,
            "B_opened": True,
            "B_used_for_training": False,
            "B_replay_gates_passed": True,
            "C_staging_eligible": True,
            "C_open_authorized": False,
            "C_opened": False,
            "confirmation_claimed": False,
            "behavioral_nll_measured": False,
            "global_top2_support_enforced": True,
            "operator_boundary_only": True,
            "executable_artifact_published": False,
            "logical_speedup_claimed": False,
        },
        "exposure_binding": {
            "prior_v1_report_sha256": _PRIOR_V1_REPORT_SHA256,
            "prior_v1_report_file_sha256": _PRIOR_V1_REPORT_FILE_SHA256,
            "prior_v1_report_path": str(_PRIOR_V1_REPORT_PATH),
            "prior_B_split_artifact_sha256": _PRIOR_B_SPLIT_SHA256,
            "B_manifest_path": str(_B_MANIFEST_PATH),
            "B_manifest_file_sha256": _B_MANIFEST_FILE_SHA256,
            "B_manifest_key": _B_MANIFEST_KEY,
            "B_corpus_path": (
                "examples/"
                "gemma3_chart_native_neighbor_B_neighbor_discovery_corpus_v1.json"
            ),
            "B_corpus_file_sha256": _B_CORPUS_SHA256,
            "current_B_split_matches_prior": True,
            "fresh_confirmation": False,
            "interpretation": (
                "development_replay_on_previously_exposed_B_not_confirmation"
            ),
        },
        "model": {
            "model_id": "google/gemma-3-270m",
            "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "hidden_width": 640,
            "runtime_qualification_receipt_sha256": _sha("qualification"),
        },
        "protocol": {
            "protocol_id": "sparse_v2_k8_r8_global_top2_exposed_B_replay_v1",
            "A_receipt_count": 45,
            "B_receipt_count": 45,
            "rows_per_role": 12_960,
            "invocation_count": 18,
            "sparse_bank": {
                "generator_count": 8,
                "generator_rank": 8,
                "active_count": 2,
                "dense_warm_start_steps": 96,
                "alternating_rounds": 4,
                "steps_per_round": 32,
                "learning_rate": 0.01,
                "euclidean_floor": 0.001,
                "oracle_coefficient_ridge": 1.0e-8,
                "dense_head_coefficient_ridge": 1.0e-8,
                "gradient_clip_norm": 10.0,
                "gauge_floor": 1.0e-12,
            },
            "atlas": {
                "chart_count": 4,
                "chart_rank": 8,
                "temperature_scale": 1.0,
                "lloyd_iterations": 8,
            },
            "router": {
                "routing_equation": "chart_mix_then_single_global_top2",
                "active_count": 2,
                "steps": 128,
                "learning_rate": 0.03,
                "euclidean_floor": 0.001,
                "oracle_solver_kind": "exact_support_enumeration_sizes_0_through_active_count",
                "oracle_ridge": 1.0e-8,
                "route_kl_coefficient": 0.1,
                "entropy_coefficient": 0.001,
                "balance_coefficient": 0.001,
                "coefficient_mse_coefficient": 0.1,
                "gradient_clip_norm": 10.0,
            },
            "preflight_receipt_sha256": _sha("preflight"),
            "dense_lexical_parity_receipt_sha256": _sha("parity"),
            "implementation_source_sha256s": _source_sha256s(root),
        },
        "splits": {
            "A_split_artifact_sha256": _sha("A-split"),
            "B_split_artifact_sha256": _PRIOR_B_SPLIT_SHA256,
            "A_receipt_sha256s": a_receipts,
            "B_receipt_sha256s": b_receipts,
            "A_row_count": 12_960,
            "B_row_count": 12_960,
            "A_B_disjoint": True,
            "freeze_receipts": {
                "A_chart_transport_fit": {
                    "stage_kind": "chart_atlas_transport_fit",
                    "stage_receipt_sha256": development_sha,
                },
                "B_neighbor_discovery": {
                    "stage_kind": "neighbor_candidate_catalog",
                    "stage_receipt_sha256": selection_sha,
                },
            },
            "final_firewall_state": {
                "opened_roles": [
                    "A_chart_transport_fit",
                    "B_neighbor_discovery",
                ],
                "current_role": None,
                "next_role": "C_neighbor_confirmation",
                "C_opened": False,
            },
        },
        "candidate": {
            "candidate_id": "k8_r8_global_top2_sparse_v2",
            "candidate_artifact_sha256": candidate_sha,
            "A_development_artifact_sha256": development_sha,
            "bank_artifact_sha256": _sha("bank"),
            "dense_warm_start_bank_artifact_sha256": _sha("warm-bank"),
            "atlas_artifact_sha256": _sha("atlas"),
            "router_artifact_sha256s": [
                [invocation_id, _sha(f"router-{invocation_id}")]
                for invocation_id in sorted(
                    f"mutation.{2 * index + 1}" for index in range(18)
                )
            ],
            "A_sparse_fit": {
                "initial_sparse_oracle": _sparse_metric(0.2),
                "final_sparse_oracle": _sparse_metric(0.05),
                "round_best_relative_fisher_losses": [
                    0.2,
                    0.1,
                    0.075,
                    0.05,
                    0.05,
                ],
            },
            "resources": _resources(),
            "executable_artifact_published": False,
        },
        "gates": {
            "gates_artifact_sha256": _sha("gates"),
            "max_relative_fisher_loss": 0.1,
            "max_oracle_router_fisher_gap": 0.05,
            "oracle_gap_numerical_tolerance": 1.0e-10,
            "max_postnorm_normalized_rmse": 0.02,
            "min_postnorm_cosine": 0.999,
            "max_sparse_probability_renormalization_error": 1.0e-10,
            "max_effective_support_fraction": 0.25,
            "minimum_aggregate_rows": 12_960,
            "require_parameter_compression": True,
        },
        "B_replay": {
            "B_selection_artifact_sha256": selection_sha,
            "decision": {
                "candidate_id": "k8_r8_global_top2_sparse_v2",
                "candidate_artifact_sha256": candidate_sha,
                "evaluation_artifact_sha256": _sha("evaluation"),
                "relative_fisher_pass": True,
                "oracle_gap_pass": True,
                "postnorm_nrmse_pass": True,
                "postnorm_cosine_pass": True,
                "sparse_renormalization_pass": True,
                "effective_support_pass": True,
                "global_support_budget_pass": True,
                "resource_pass": True,
                "minimum_rows_pass": True,
                "passed": True,
            },
            "evaluation": evaluation,
            "worst_per_invocation": _worst_summary(evaluation),
            "candidate_staged_for_separate_C_decision": True,
            "C_open_authorized": False,
            "C_opened": False,
        },
        "runtime_counters": {
            "role_batch_counts": [
                ["A_chart_transport_fit", 45],
                ["B_neighbor_discovery", 45],
                ["C_neighbor_confirmation", 0],
            ],
            "chart_backbone_forward_count": 90,
            "chart_terminal_cotangent_vjp_count": 90,
            "dense_parity_smoke_count": 1,
            "parity_backbone_forward_count": 1,
            "parity_terminal_cotangent_vjp_count": 1,
            "captured_example_count": 96,
            "supplemental_route_receipt_count": 1728,
            "canonical_gradient_targets_per_vjp": 73,
            "auxiliary_gradient_targets_per_vjp": 19,
            "gradient_targets_per_vjp": 92,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_raw_activations": False,
            "contains_raw_gradients": False,
            "contains_logits": False,
        },
        "runtime_resources": {
            "prepublication_peak_resident_set_bytes": 1024**3,
            "resident_set_cap_bytes": 8 * 1024**3,
            "publication_guarded": True,
        },
    }


def _noop_guard(_phase_id: str) -> None:
    return None


def test_cli_import_defers_model_transport_and_panel_modules() -> None:
    code = """
import sys
import fisher_graph.gemma3_attention_transport_sparse_v2_global_router_experiment
forbidden = (
    'fisher_graph.gemma3_attention_transport_compiler',
    'fisher_graph.gemma3_attention_transport_runtime',
    'fisher_graph.gemma3_attention_transport_experiment_core',
    'fisher_graph.gemma3_attention_transport_sparse_v2',
    'fisher_graph.gemma3_attention_transport_sparse_v2_global_router',
    'fisher_graph.gemma3_chart_native_neighbor_fresh_panels',
    'transformers',
)
assert not [name for name in forbidden if name in sys.modules]
"""
    subprocess.run(
        [str(Path(".venv/bin/python")), "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )


def test_report_publication_is_canonical_source_bound_and_no_overwrite(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"
    _publish_report(output, report, publication_guard=_noop_guard)

    assert _strict_report(output) == report
    assert check_sparse_v2_global_router_B_replay(
        output,
        repository_root=root,
    )["report_sha256"] == report["report_sha256"]
    with pytest.raises(FileExistsError):
        _publish_report(output, report, publication_guard=_noop_guard)


def test_report_rejects_reauthored_exposure_and_C_firewall() -> None:
    root = Path(__file__).resolve().parents[1]
    exposure = _unsealed_report(root)
    exposure["exposure_binding"]["prior_v1_report_sha256"] = _sha("other")
    with pytest.raises(ValueError, match="exposure"):
        _seal_report(exposure)

    firewall = _unsealed_report(root)
    firewall["scientific_status"]["C_opened"] = True
    with pytest.raises(ValueError, match="scientific status"):
        _seal_report(firewall)

    authorization = _unsealed_report(root)
    authorization["scientific_status"]["C_open_authorized"] = True
    with pytest.raises(ValueError, match="scientific status"):
        _seal_report(authorization)

    authorization = _unsealed_report(root)
    authorization["B_replay"]["C_open_authorized"] = True
    with pytest.raises(ValueError, match="replay lineage"):
        _seal_report(authorization)

    firewall = _unsealed_report(root)
    firewall["splits"]["final_firewall_state"]["next_role"] = None
    with pytest.raises(ValueError, match="firewall"):
        _seal_report(firewall)

    freeze = _unsealed_report(root)
    freeze["splits"]["freeze_receipts"]["A_chart_transport_fit"][
        "stage_receipt_sha256"
    ] = _sha("foreign-development")
    with pytest.raises(ValueError, match="freeze receipt"):
        _seal_report(freeze)


def test_prior_exposure_files_require_pinned_canonical_bytes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        _PRIOR_V1_REPORT_PATH,
        _B_MANIFEST_PATH,
        _B_CORPUS_PATH,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / relative).read_bytes())

    _validate_prior_exposure_files(tmp_path)
    prior = tmp_path / _PRIOR_V1_REPORT_PATH
    reauthored = json.loads(prior.read_bytes())
    prior.write_text(json.dumps(reauthored, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="prior v1 exposure binding"):
        _validate_prior_exposure_files(tmp_path)


def test_report_rejects_metric_gate_worst_counter_and_resource_tampering() -> None:
    root = Path(__file__).resolve().parents[1]
    metric = _unsealed_report(root)
    metric["B_replay"]["evaluation"]["relative_fisher_loss"] = 0.09
    with pytest.raises(ValueError, match="Fisher equations"):
        _seal_report(metric)

    gate = _unsealed_report(root)
    gate["B_replay"]["decision"]["passed"] = False
    with pytest.raises(ValueError, match="gate equations"):
        _seal_report(gate)

    relaxed_support = _unsealed_report(root)
    relaxed_support["gates"]["max_effective_support_fraction"] = 0.5
    with pytest.raises(ValueError, match="gates"):
        _seal_report(relaxed_support)

    expanded_support = _unsealed_report(root)
    expanded_support["B_replay"]["evaluation"]["per_invocation"][0][
        "effective_support_count"
    ]["maximum"] = 3
    with pytest.raises(ValueError, match="support summary"):
        _seal_report(expanded_support)

    contradictory_postnorm = _unsealed_report(root)
    contradictory_evaluation = contradictory_postnorm["B_replay"]["evaluation"]
    for invocation in contradictory_evaluation["per_invocation"]:
        invocation["postnorm"]["normalized_rmse"] = 0.5
    contradictory_postnorm["B_replay"]["worst_per_invocation"] = (
        _worst_summary(contradictory_evaluation)
    )
    with pytest.raises(ValueError, match="normalized-RMSE bounds"):
        _seal_report(contradictory_postnorm)

    contradictory_cosine = _unsealed_report(root)
    contradictory_evaluation = contradictory_cosine["B_replay"]["evaluation"]
    for invocation in contradictory_evaluation["per_invocation"]:
        invocation["postnorm"]["cosine"] = 0.1
    contradictory_cosine["B_replay"]["worst_per_invocation"] = (
        _worst_summary(contradictory_evaluation)
    )
    with pytest.raises(ValueError, match="cosine upper bound"):
        _seal_report(contradictory_cosine)

    worst = _unsealed_report(root)
    worst["B_replay"]["worst_per_invocation"][
        "maximum_relative_fisher_loss"
    ]["value"] = 0.07
    with pytest.raises(ValueError, match="worst"):
        _seal_report(worst)

    counter = _unsealed_report(root)
    counter["runtime_counters"]["chart_backbone_forward_count"] = 90.0
    with pytest.raises(ValueError, match="counters"):
        _seal_report(counter)

    peak = _unsealed_report(root)
    peak["runtime_resources"]["prepublication_peak_resident_set_bytes"] = (
        8 * 1024**3 + 1
    )
    with pytest.raises(ValueError, match="runtime resources"):
        _seal_report(peak)


def test_report_rejects_duplicate_json_field_and_symlink(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(canonical.replace("{", '{"schema":"duplicate",', 1))
    with pytest.raises(ValueError, match="JSON"):
        _strict_report(duplicate)

    source = tmp_path / "source.json"
    source.write_text(canonical)
    linked = tmp_path / "linked.json"
    os.symlink(source, linked)
    with pytest.raises(ValueError, match="regular file"):
        _strict_report(linked)


def test_publication_rejects_symlink_parent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    os.symlink(real, linked)
    with pytest.raises(ValueError, match="real directory"):
        _publish_report(
            linked / "replay.json",
            report,
            publication_guard=_noop_guard,
        )

    nested = real / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="real directory chain"):
        _publish_report(
            linked / "nested" / "replay.json",
            report,
            publication_guard=_noop_guard,
        )


def test_publication_guard_rolls_back_only_owned_inode(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"
    displaced = tmp_path / "owned.json"
    phases: list[str] = []

    def guard(phase_id: str) -> None:
        phases.append(phase_id)
        if phase_id == "global_router_publication_verified":
            output.rename(displaced)
            output.write_bytes(b"foreign replacement")
            raise RuntimeError("resident_cap_crossed")

    with pytest.raises(RuntimeError, match="resident_cap_crossed"):
        _publish_report(output, report, publication_guard=guard)

    assert phases == [
        "global_router_temporary_sealed",
        "global_router_publication_verified",
    ]
    assert output.read_bytes() == b"foreign replacement"
    assert displaced.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_publication_preserves_replaced_foreign_temporary_path(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"
    displaced = tmp_path / "owned-temporary.json"
    foreign_bytes = b"foreign temporary replacement"

    def guard(phase_id: str) -> None:
        if phase_id == "global_router_temporary_sealed":
            temporary = next(tmp_path.glob(".*.tmp"))
            temporary.rename(displaced)
            temporary.write_bytes(foreign_bytes)

    with pytest.raises(RuntimeError, match="temporary report identity"):
        _publish_report(output, report, publication_guard=guard)

    remaining = list(tmp_path.glob(".*.tmp"))
    assert len(remaining) == 1
    assert remaining[0].read_bytes() == foreign_bytes
    assert displaced.exists()
    assert not output.exists()


def test_publication_rejects_same_inode_corruption_after_final_guard(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "replay.json"

    def guard(phase_id: str) -> None:
        if phase_id == "global_router_publication_verified":
            with output.open("r+b") as handle:
                handle.seek(0)
                handle.write(b"corrupt")
                handle.flush()
                os.fsync(handle.fileno())

    with pytest.raises((RuntimeError, ValueError)):
        _publish_report(output, report, publication_guard=guard)

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp")) == []
