from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

import fisher_graph.gemma3_attention_transport_artifact as artifact_module
from fisher_graph.gemma3_attention_transport_artifact import (
    ATTENTION_TRANSPORT_ARTIFACT_SCHEMA,
    ATTENTION_TRANSPORT_REPORT_SCHEMA,
    build_attention_transport_payload,
    canonical_json_bytes,
    load_attention_transport_artifact,
    save_attention_transport_artifact,
    strict_load_attention_transport_report,
)
from fisher_graph.gemma3_attention_transport_compiler import (
    AttentionTransportResourceAccounting,
    ChartConditionedAttentionTransportRouter,
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    chart_conditioned_attention_transport_router_from_state_dict,
    fisher_attention_transport_generator_bank_from_state_dict,
    frozen_attention_chart_atlas_from_state_dict,
)


def _components() -> tuple[
    FisherAttentionTransportGeneratorBank,
    FrozenAttentionChartAtlas,
    tuple[ChartConditionedAttentionTransportRouter, ...],
    AttentionTransportResourceAccounting,
]:
    left = torch.zeros(2, 4, 1, dtype=torch.float64)
    right = torch.zeros(2, 4, 1, dtype=torch.float64)
    left[0, 0, 0] = 1.0
    left[1, 1, 0] = 1.0
    right[0, 2, 0] = 1.0
    right[1, 3, 0] = 1.0
    bank = FisherAttentionTransportGeneratorBank(
        left_factors=left,
        singular_scales=torch.tensor([[2.0], [1.5]], dtype=torch.float64),
        right_factors=right,
        head_coefficients=torch.eye(2, dtype=torch.float64),
        head_ids=(
            "layer.0/query-head.0",
            "layer.0/query-head.1",
        ),
    )
    centers = torch.zeros(2, 4, dtype=torch.float64)
    centers[1, 1] = 1.0
    bases = torch.zeros(2, 4, 1, dtype=torch.float64)
    bases[:, 2, 0] = 1.0
    atlas = FrozenAttentionChartAtlas(
        centers=centers,
        bases=bases,
        temperature=0.25,
    )
    feature_width = atlas.chart_rank + bank.generator_count
    router = ChartConditionedAttentionTransportRouter(
        invocation_id="layer.0",
        generator_bank_sha256=bank.artifact_sha256,
        atlas=atlas,
        selection_weight=torch.zeros(
            atlas.chart_count,
            bank.generator_count,
            feature_width,
            dtype=torch.float64,
        ),
        selection_bias=torch.tensor(
            [[3.0, -3.0], [-3.0, 3.0]],
            dtype=torch.float64,
        ),
        amplitude_weight=torch.zeros(
            atlas.chart_count,
            bank.generator_count,
            feature_width,
            dtype=torch.float64,
        ),
        amplitude_bias=torch.ones(
            atlas.chart_count,
            bank.generator_count,
            dtype=torch.float64,
        ),
        active_count=1,
    )
    compiled = (
        bank.stored_parameter_count
        + atlas.stored_parameter_count
        + router.stored_parameter_count
    )
    native = compiled + 100
    resources = AttentionTransportResourceAccounting(
        native_v_o_parameters=native,
        generator_bank_parameters=bank.stored_parameter_count,
        chart_parameters=atlas.stored_parameter_count,
        router_parameters=router.stored_parameter_count,
        compiled_parameters=compiled,
        parameter_reduction_fraction=1.0 - compiled / native,
        parameter_compression_eligible=True,
    )
    return bank, atlas, (router,), resources


def _metric(*, fisher_loss: float) -> dict[str, object]:
    return {
        "row_count": 12_960,
        "fisher_loss": fisher_loss,
        "directional_mse": fisher_loss / 2.0,
        "euclidean_mse": fisher_loss / 2.0,
        "normalized_rmse": fisher_loss**0.5,
        "cosine": 0.9999,
    }


def _aggregate_evaluation(
    *,
    role: str,
    split_sha256: str,
    router: ChartConditionedAttentionTransportRouter,
    resources: AttentionTransportResourceAccounting,
) -> dict[str, object]:
    fisher_loss = 0.02 if role == "B" else 0.025
    oracle_fisher_loss = 0.01
    usage = [0.5, 0.5]
    invocation = {
        "invocation_id": router.invocation_id,
        "operator": _metric(fisher_loss=fisher_loss),
        "oracle": _metric(fisher_loss=oracle_fisher_loss),
        "oracle_contract": {
            "solver_kind": (
                "exact_support_enumeration_sizes_0_through_active_count"
            ),
            "active_count_limit": 1,
            "coefficient_ridge": 0.0,
            "heldout_ridge_free": True,
        },
        "zero_predictor_fisher_loss": 1.0,
        "relative_fisher_loss": fisher_loss,
        "oracle_relative_fisher_loss": oracle_fisher_loss,
        "oracle_router_fisher_gap": fisher_loss - oracle_fisher_loss,
        "postnorm": {
            "row_count": 12_960,
            "normalized_rmse": 0.01,
            "cosine": 0.9999,
        },
        "per_chart_active_count": {"minimum": 1, "mean": 1.0, "maximum": 1},
        "effective_support_count": {
            "minimum": 1,
            "mean": 1.0,
            "maximum": 1,
            "fraction": 0.5,
        },
        "sparse_probability_renormalization_max_error": 0.0,
        "generator_usage": usage,
        "dead_generator_count": 0,
    }
    return {
        "role": role,
        "split_artifact_sha256": split_sha256,
        "row_count": 12_960,
        "invocation_count": 1,
        "fisher_loss": fisher_loss,
        "zero_predictor_fisher_loss": 1.0,
        "relative_fisher_loss": fisher_loss,
        "oracle_fisher_loss": oracle_fisher_loss,
        "oracle_contract": dict(invocation["oracle_contract"]),
        "oracle_relative_fisher_loss": oracle_fisher_loss,
        "oracle_router_fisher_gap": fisher_loss - oracle_fisher_loss,
        "operator_normalized_rmse": fisher_loss**0.5,
        "operator_cosine": 0.9999,
        "postnorm_normalized_rmse": 0.01,
        "postnorm_cosine": 0.9999,
        "per_chart_active_count": {"minimum": 1, "mean": 1.0, "maximum": 1},
        "effective_support_count": {
            "minimum": 1,
            "mean": 1.0,
            "maximum": 1,
            "fraction": 0.5,
        },
        "sparse_probability_renormalization_max_error": 0.0,
        "generator_usage": usage,
        "dead_generator_count": 0,
        "resources": asdict(resources),
        "per_invocation": [invocation],
    }


def _decision(
    *,
    role: str,
    split_sha256: str,
    router: ChartConditionedAttentionTransportRouter,
    resources: AttentionTransportResourceAccounting,
) -> dict[str, object]:
    return {
        "candidate_id": "K2-r1-top1",
        "candidate_artifact_sha256": "d" * 64,
        "evaluation_artifact_sha256": ("e" if role == "B" else "f") * 64,
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
        "evaluation": _aggregate_evaluation(
            role=role,
            split_sha256=split_sha256,
            router=router,
            resources=resources,
        ),
    }


def _metadata(
    *,
    bank: FisherAttentionTransportGeneratorBank,
    atlas: FrozenAttentionChartAtlas,
    routers: tuple[ChartConditionedAttentionTransportRouter, ...],
    resources: AttentionTransportResourceAccounting,
) -> dict[str, dict[str, object]]:
    router = routers[0]
    b_decision = _decision(
        role="B",
        split_sha256="2" * 64,
        router=router,
        resources=resources,
    )
    c_decision = _decision(
        role="C",
        split_sha256="3" * 64,
        router=router,
        resources=resources,
    )
    return {
        "model": {
            "model_id": "google/gemma-3-1b-it",
            "model_revision": "a" * 40,
            "hidden_width": bank.hidden_width,
            "runtime_qualification_receipt_sha256": "0" * 64,
            "runtime_execution_receipt_sha256": "1" * 64,
        },
        "protocol": {
            "protocol_id": "all_length_uniform_row_canary_v1",
            "a_fit_only": True,
            "b_selection_only": True,
            "c_confirmation_only": True,
            "no_post_c_refit_or_selection": True,
            "rows_per_batch_and_invocation": 16,
            "expected_rows_per_role": 12_960,
            "candidate_configs": [
                {
                    "candidate_id": "K2-r1-top1",
                    "config_artifact_sha256": "4" * 64,
                    "generator_count": bank.generator_count,
                    "generator_rank": bank.generator_rank,
                    "bank_steps": 1,
                    "bank_learning_rate": 0.01,
                    "euclidean_floor": 0.001,
                    "chart_count": atlas.chart_count,
                    "chart_rank": atlas.chart_rank,
                    "chart_temperature_scale": 1.0,
                    "chart_lloyd_iterations": 1,
                    "active_count": router.active_count,
                    "router_training_oracle": {
                        "split": "A_only",
                        "solver_kind": (
                            "exact_support_enumeration_sizes_0_through_active_count"
                        ),
                        "coefficient_ridge": 1.0e-8,
                        "heldout_metric_used": False,
                    },
                    "router_steps": 1,
                    "router_learning_rate": 0.03,
                }
            ],
            "gates": {
                "gates_artifact_sha256": "5" * 64,
                "max_relative_fisher_loss": 0.1,
                "max_oracle_router_fisher_gap": 0.05,
                "oracle_gap_numerical_tolerance": 1.0e-10,
                "max_postnorm_normalized_rmse": 0.02,
                "min_postnorm_cosine": 0.999,
                "max_sparse_probability_renormalization_error": 1.0e-10,
                "max_effective_support_fraction": 0.75,
                "require_parameter_compression": True,
                "minimum_aggregate_rows": 12_960,
            },
            "implementation_source_sha256s": {
                path: hashlib.sha256(path.encode()).hexdigest()
                for path in artifact_module._IMPLEMENTATION_SOURCE_PATHS
            },
            "preflight_receipt_sha256": "6" * 64,
            "dense_lexical_parity_receipt_sha256": "7" * 64,
        },
        "splits": {
            "A_split_artifact_sha256": "1" * 64,
            "B_split_artifact_sha256": "2" * 64,
            "C_split_artifact_sha256": "3" * 64,
            "A_receipt_sha256s": ["8" * 64],
            "B_receipt_sha256s": ["9" * 64],
            "C_receipt_sha256s": ["a" * 64],
            "A_B_C_receipts_disjoint": True,
        },
        "selection": {
            "selected_candidate_id": "K2-r1-top1",
            "selected_candidate_artifact_sha256": "d" * 64,
            "selection_lock_artifact_sha256": "b" * 64,
            "B_selection_artifact_sha256": "c" * 64,
            "B_decision_artifact_sha256": "e" * 64,
            "C_confirmation_artifact_sha256": "f" * 64,
            "B_passed": True,
            "C_passed": True,
            "selection_rule": (
                "minimum_B_relative_fisher_then_compiled_parameters_then_K_rank_"
                "active_count_then_candidate_id"
            ),
        },
        "evaluations": {
            "A": {
                "initial_bank": _metric(fisher_loss=0.03),
                "final_bank": _metric(fisher_loss=0.02),
            },
            "B": b_decision,
            "C": c_decision,
            "all_B_candidates": [b_decision],
            "runtime": {
                "counters": {
                    "role_batch_counts": [
                        ["A_chart_transport_fit", 45],
                        ["B_neighbor_discovery", 45],
                        ["C_neighbor_confirmation", 45],
                    ],
                    "chart_backbone_forward_count": 135,
                    "chart_terminal_cotangent_vjp_count": 135,
                    "dense_parity_smoke_count": 1,
                    "parity_backbone_forward_count": 1,
                    "parity_terminal_cotangent_vjp_count": 1,
                    "captured_example_count": 1_080,
                    "supplemental_route_receipt_count": 1,
                    "canonical_gradient_targets_per_vjp": 73,
                    "auxiliary_gradient_targets_per_vjp": 19,
                    "gradient_targets_per_vjp": 92,
                    "contains_prompt_text": False,
                    "contains_token_ids": False,
                    "contains_raw_activations": False,
                    "contains_raw_gradients": False,
                    "contains_logits": False,
                },
                "prepublication_peak_resident_set_bytes": 1,
                "resident_set_cap_bytes": 2,
                "publication_guarded": True,
                "logical_speedup_claimed": False,
            },
        },
    }


def _save(
    path: Path,
    *,
    publication_guard: Callable[[str], None] | None = None,
) -> dict[str, object]:
    bank, atlas, routers, resources = _components()
    guard = (
        (lambda _phase_id: None)
        if publication_guard is None
        else publication_guard
    )
    return save_attention_transport_artifact(
        path,
        **_metadata(
            bank=bank,
            atlas=atlas,
            routers=routers,
            resources=resources,
        ),
        bank=bank,
        atlas=atlas,
        routers=routers,
        resource_accounting=resources,
        publication_guard=guard,
    )


def test_attention_transport_artifact_round_trip_and_report_commit_marker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attention-transport.pt"
    report = _save(output)
    report_path = output.with_suffix(".json")

    payload = load_attention_transport_artifact(output)
    loaded_report = strict_load_attention_transport_report(report_path)

    assert payload["schema"] == ATTENTION_TRANSPORT_ARTIFACT_SCHEMA
    assert loaded_report == report
    assert loaded_report["schema"] == ATTENTION_TRANSPORT_REPORT_SCHEMA
    assert report_path.read_bytes() == canonical_json_bytes(report)
    assert report["artifact"]["tensor_file_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    bank = fisher_attention_transport_generator_bank_from_state_dict(
        payload["generator_bank"]
    )
    atlas = frozen_attention_chart_atlas_from_state_dict(payload["atlas"])
    router = chart_conditioned_attention_transport_router_from_state_dict(
        payload["routers"][0],
        atlas=atlas,
    )
    assert bank.artifact_sha256 == report["generator_bank"]["artifact_sha256"]
    assert atlas.artifact_sha256 == report["atlas"]["artifact_sha256"]
    assert router.artifact_sha256 == report["routers"][0]["artifact_sha256"]
    assert list(tmp_path.glob(".*.tmp")) == []


def test_attention_transport_publication_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "attention-transport.pt"
    expected_report = _save(output)
    tensor_before = output.read_bytes()
    report_before = output.with_suffix(".json").read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _save(output)

    assert output.read_bytes() == tensor_before
    assert output.with_suffix(".json").read_bytes() == report_before
    assert strict_load_attention_transport_report(
        output.with_suffix(".json")
    ) == expected_report


def test_payload_rejects_forbidden_teacher_or_capture_fields() -> None:
    bank, atlas, routers, resources = _components()
    metadata = _metadata(
        bank=bank,
        atlas=atlas,
        routers=routers,
        resources=resources,
    )
    metadata["evaluations"]["C"]["operator_outputs"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="forbidden field"):
        build_attention_transport_payload(
            **metadata,
            bank=bank,
            atlas=atlas,
            routers=routers,
            resource_accounting=resources,
        )


def test_tensor_tampering_fails_logical_digest_validation(tmp_path: Path) -> None:
    output = tmp_path / "attention-transport.pt"
    _save(output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["generator_bank"]["singular_scales"][0, 0] += 0.25
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)

    with pytest.raises(ValueError, match="payload digest drifted"):
        load_attention_transport_artifact(forged)


def test_report_requires_canonical_bytes_and_cross_file_digest(tmp_path: Path) -> None:
    output = tmp_path / "attention-transport.pt"
    _save(output)
    report_path = output.with_suffix(".json")
    canonical = report_path.read_bytes()
    report_path.write_bytes(canonical + b"\n")

    with pytest.raises(ValueError, match="canonical JSON"):
        strict_load_attention_transport_report(report_path)

    report_path.write_bytes(canonical)
    tensor = bytearray(output.read_bytes())
    tensor[-1] ^= 1
    output.write_bytes(tensor)
    with pytest.raises(ValueError, match="tensor file digest drifted"):
        strict_load_attention_transport_report(report_path)


def test_failed_report_commit_rolls_back_tensor_and_temporaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fisher_graph.gemma3_attention_transport_artifact as artifact_module

    output = tmp_path / "attention-transport.pt"

    def fail_strict_reload(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("injected_cross_reload_failure")

    monkeypatch.setattr(
        artifact_module,
        "strict_load_attention_transport_report",
        fail_strict_reload,
    )
    with pytest.raises(RuntimeError, match="injected_cross_reload_failure"):
        _save(output)

    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_verified_publication_guard_failure_rolls_back_both_outputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attention-transport.pt"
    phases: list[str] = []

    def fail_after_verified(phase_id: str) -> None:
        phases.append(phase_id)
        if phase_id == "artifact_publication_verified":
            raise RuntimeError("injected_verified_publication_guard_failure")

    with pytest.raises(
        RuntimeError,
        match="injected_verified_publication_guard_failure",
    ):
        _save(output, publication_guard=fail_after_verified)

    assert phases == [
        "artifact_temporaries_sealed",
        "artifact_publication_verified",
    ]
    assert not output.exists()
    assert not output.with_suffix(".json").exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_rehashed_payload_rejects_overlapping_split_receipts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attention-transport.pt"
    _save(output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["splits"]["B_receipt_sha256s"] = list(
        payload["splits"]["A_receipt_sha256s"]
    )
    unsigned = dict(payload)
    unsigned.pop("scientific_payload_sha256")
    payload["scientific_payload_sha256"] = artifact_module._payload_sha256(
        unsigned
    )
    forged = tmp_path / "rehashed-overlap.pt"
    torch.save(payload, forged)

    with pytest.raises(ValueError, match="not disjoint"):
        load_attention_transport_artifact(forged)


def test_rehashed_payload_rejects_false_b_terminal_status(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attention-transport.pt"
    _save(output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["selection"]["B_passed"] = False
    unsigned = dict(payload)
    unsigned.pop("scientific_payload_sha256")
    payload["scientific_payload_sha256"] = artifact_module._payload_sha256(
        unsigned
    )
    forged = tmp_path / "rehashed-false-b.pt"
    torch.save(payload, forged)

    with pytest.raises(ValueError, match="terminal contract"):
        load_attention_transport_artifact(forged)


def test_rehashed_report_contradiction_fails_success_check(
    tmp_path: Path,
) -> None:
    from fisher_graph.gemma3_attention_transport_experiment import (
        check_attention_transport_outputs,
    )

    output = tmp_path / "attention-transport.pt"
    _save(output)
    report_path = output.with_suffix(".json")
    report = json.loads(report_path.read_bytes())
    report["selection"]["C_passed"] = False
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    report["report_sha256"] = artifact_module._report_sha256(unsigned)
    report_path.write_bytes(canonical_json_bytes(report))

    with pytest.raises(ValueError, match="bindings are inconsistent"):
        check_attention_transport_outputs(
            artifact_path=output,
            report_path=report_path,
        )


def test_strict_loader_uses_cpu_weights_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fisher_graph.gemma3_attention_transport_artifact as artifact_module

    output = tmp_path / "attention-transport.pt"
    _save(output)
    calls: list[tuple[object, object]] = []
    real_load = artifact_module.torch.load

    def audited_load(*args: object, **kwargs: object) -> object:
        calls.append((kwargs.get("map_location"), kwargs.get("weights_only")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(artifact_module.torch, "load", audited_load)
    load_attention_transport_artifact(output)

    assert calls == [("cpu", True)]


def test_report_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    output = tmp_path / "attention-transport.pt"
    _save(output)
    report_path = output.with_suffix(".json")
    raw = report_path.read_text(encoding="utf-8")
    attacked = raw.replace(
        '{"artifact":',
        '{"schema":"duplicate","artifact":',
        1,
    )
    # Ensure the injected key actually duplicates the canonical top-level key.
    assert json.loads(attacked)["schema"] == ATTENTION_TRANSPORT_REPORT_SCHEMA
    report_path.write_text(attacked, encoding="utf-8")

    with pytest.raises(ValueError, match="report JSON is invalid"):
        strict_load_attention_transport_report(report_path)
