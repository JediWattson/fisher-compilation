from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import pytest

from fisher_graph.gemma3_attention_transport_sparse_v2_experiment import (
    REPORT_SCHEMA,
    _PARTITION_DOMAIN,
    _partition_records,
    _publish_report,
    _seal_report,
    _source_sha256s,
    _strict_report,
    check_sparse_v2_A_diagnostic,
)


def _noop_publication_guard(_phase_id: str) -> None:
    return None


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _metric(
    rows: int,
    relative: float,
    *,
    coefficient_ridge: float,
) -> dict[str, object]:
    return {
        "row_count": rows,
        "fisher_loss": relative,
        "directional_mse": relative,
        "euclidean_mse": relative,
        "normalized_rmse": relative,
        "cosine": 1.0 - relative,
        "zero_predictor_fisher_loss": 1.0,
        "relative_fisher_loss": relative,
        "active_count_limit": 2,
        "coefficient_ridge": coefficient_ridge,
        "per_invocation_relative_fisher_loss": [
            [invocation_id, relative]
            for invocation_id in sorted(
                f"mutation.{2 * index + 1}" for index in range(18)
            )
        ],
    }


def _unsealed_report(root: Path) -> dict[str, object]:
    receipts = sorted(
        [_sha(f"receipt-{index}") for index in range(45)],
        key=lambda receipt: hashlib.sha256(
            _PARTITION_DOMAIN + receipt.encode("ascii")
        ).hexdigest(),
    )
    fit_receipts = receipts[:36]
    dev_receipts = receipts[36:]
    return {
        "schema": REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "A_only_development_diagnostic": True,
            "B_opened": False,
            "C_opened": False,
            "behavioral_nll_measured": False,
            "confirmation_claimed": False,
            "executable_artifact_published": False,
            "global_sparse_oracle_gate_targeted": True,
            "literal_chart_union_runtime_optimized": False,
            "logical_speedup_claimed": False,
        },
        "model": {
            "model_id": "google/gemma-3-270m",
            "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
            "hidden_width": 640,
            "runtime_qualification_receipt_sha256": _sha("qualification"),
        },
        "protocol": {
            "protocol_id": "A_only_sparse_oracle_alternation_k8_r8_top2_v1",
            "fit_receipt_count": 36,
            "dev_receipt_count": 9,
            "B_and_C_forbidden": True,
            "generator_count": 8,
            "generator_rank": 8,
            "active_count": 2,
            "dense_warm_start_steps": 96,
            "alternating_rounds": 4,
            "steps_per_round": 32,
            "learning_rate": 1.0e-2,
            "euclidean_floor": 1.0e-3,
            "oracle_coefficient_ridge": 1.0e-8,
            "heldout_oracle_ridge": 0.0,
            "dense_head_coefficient_ridge": 1.0e-8,
            "gauge_floor": 1.0e-12,
            "preflight_receipt_sha256": _sha("preflight"),
            "dense_lexical_parity_receipt_sha256": _sha("dense-parity"),
            "implementation_source_sha256s": _source_sha256s(root),
        },
        "A_partition": {
            "A_split_artifact_sha256": _sha("A-split"),
            "partition_policy": (
                "domain_hash_sorted_whole_capture_receipts_36_fit_9_dev"
            ),
            "fit_receipt_sha256s": fit_receipts,
            "dev_receipt_sha256s": dev_receipts,
            "fit_row_count": 10_368,
            "dev_row_count": 2_592,
            "fit_and_dev_disjoint": True,
        },
        "metrics": {
            "dense_warm_start_fit_sparse_oracle": _metric(
                10_368,
                0.5,
                coefficient_ridge=1.0e-8,
            ),
            "dense_warm_start_dev_sparse_oracle": _metric(
                2_592,
                0.6,
                coefficient_ridge=0.0,
            ),
            "sparse_aware_fit_sparse_oracle": _metric(
                10_368,
                0.3,
                coefficient_ridge=1.0e-8,
            ),
            "sparse_aware_dev_sparse_oracle": _metric(
                2_592,
                0.4,
                coefficient_ridge=0.0,
            ),
            "fit_round_best_relative_fisher_losses": [0.5, 0.4, 0.35, 0.3, 0.3],
            "dev_relative_fisher_loss_absolute_change": 0.2,
            "dev_relative_fisher_loss_fractional_improvement": 1.0 / 3.0,
            "dev_oracle_threshold": 0.1,
            "dev_oracle_threshold_cleared": False,
        },
        "resources": {
            "native_v_o_parameters": 14_745_600,
            "generator_bank_parameters": 82_560,
            "compiled_parameters": 82_560,
            "parameter_reduction_fraction": 1.0 - 82_560 / 14_745_600,
            "native_qk_retained": True,
            "native_output_rmsnorm_retained": True,
            "logical_speedup_claimed": False,
            "parameter_compression_eligible": True,
            "bank_artifact_sha256": _sha("bank"),
            "dense_warm_start_bank_artifact_sha256": _sha("warm-bank"),
            "executable_bank_published": False,
        },
        "runtime_counters": {
            "role_batch_counts": [
                ["A_chart_transport_fit", 45],
                ["B_neighbor_discovery", 0],
                ["C_neighbor_confirmation", 0],
            ],
            "chart_backbone_forward_count": 45,
            "chart_terminal_cotangent_vjp_count": 45,
            "dense_parity_smoke_count": 1,
            "parity_backbone_forward_count": 1,
            "parity_terminal_cotangent_vjp_count": 1,
            "captured_example_count": 48,
            "supplemental_route_receipt_count": 864,
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


@dataclass(frozen=True)
class _Capture:
    row_count: int = 16


@dataclass(frozen=True)
class _Record:
    base_capture_receipt_sha256: str
    capture: _Capture


def test_A_partition_keeps_every_receipt_whole() -> None:
    records = tuple(
        _Record(_sha(f"receipt-{receipt}"), _Capture())
        for receipt in range(45)
        for _ in range(18)
    )

    fit, dev, fit_receipts, dev_receipts = _partition_records(records)

    assert len(fit) == 36 * 18
    assert len(dev) == 9 * 18
    assert set(fit_receipts).isdisjoint(dev_receipts)
    assert {record.base_capture_receipt_sha256 for record in fit} == set(
        fit_receipts
    )
    assert {record.base_capture_receipt_sha256 for record in dev} == set(
        dev_receipts
    )
    reversed_partition = _partition_records(tuple(reversed(records)))
    assert reversed_partition[2:] == (fit_receipts, dev_receipts)


def test_report_publication_is_canonical_no_overwrite_and_source_bound(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "diagnostic.json"

    _publish_report(
        output,
        report,
        publication_guard=_noop_publication_guard,
    )

    assert _strict_report(output) == report
    assert check_sparse_v2_A_diagnostic(
        output,
        repository_root=root,
    )["report_sha256"] == report["report_sha256"]
    with pytest.raises(FileExistsError):
        _publish_report(
            output,
            report,
            publication_guard=_noop_publication_guard,
        )


def test_report_rejects_rehashed_claim_that_B_was_opened(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    value = _unsealed_report(root)
    value["scientific_status"]["B_opened"] = True

    with pytest.raises(ValueError, match="scientific status"):
        _seal_report(value)


def test_report_rejects_metric_tampering_without_rehash(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "diagnostic.json"
    _publish_report(
        output,
        report,
        publication_guard=_noop_publication_guard,
    )
    value = json.loads(output.read_text())
    value["metrics"]["sparse_aware_dev_sparse_oracle"][
        "relative_fisher_loss"
    ] = 0.01
    output.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))

    with pytest.raises(ValueError, match="equations"):
        _strict_report(output)


def test_report_rejects_duplicate_or_reassigned_A_receipts() -> None:
    root = Path(__file__).resolve().parents[1]
    duplicate = _unsealed_report(root)
    duplicate["A_partition"]["fit_receipt_sha256s"][0] = duplicate[
        "A_partition"
    ]["fit_receipt_sha256s"][1]
    with pytest.raises(ValueError, match="receipt partition"):
        _seal_report(duplicate)

    reassigned = _unsealed_report(root)
    fit_receipts = reassigned["A_partition"]["fit_receipt_sha256s"]
    dev_receipts = reassigned["A_partition"]["dev_receipt_sha256s"]
    fit_receipts[-1], dev_receipts[0] = dev_receipts[0], fit_receipts[-1]
    with pytest.raises(ValueError, match="reauthored"):
        _seal_report(reassigned)


def test_report_rejects_metric_equation_drift_float_counter_and_peak_cap() -> None:
    root = Path(__file__).resolve().parents[1]
    equation_drift = _unsealed_report(root)
    equation_drift["metrics"]["sparse_aware_dev_sparse_oracle"][
        "fisher_loss"
    ] = 0.41
    with pytest.raises(ValueError, match="equations"):
        _seal_report(equation_drift)

    float_counter = _unsealed_report(root)
    float_counter["runtime_counters"]["chart_backbone_forward_count"] = 45.0
    with pytest.raises(ValueError, match="runtime counters"):
        _seal_report(float_counter)

    excessive_peak = _unsealed_report(root)
    excessive_peak["runtime_resources"][
        "prepublication_peak_resident_set_bytes"
    ] = 8 * 1024**3 + 1
    with pytest.raises(ValueError, match="runtime resources"):
        _seal_report(excessive_peak)


def test_strict_report_rejects_symlink(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    source = tmp_path / "source.json"
    source.write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    )
    linked = tmp_path / "linked.json"
    os.symlink(source, linked)

    with pytest.raises(ValueError, match="regular file"):
        _strict_report(linked)


def test_publication_rejects_symlink_parent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    os.symlink(real_parent, linked_parent)

    with pytest.raises(ValueError, match="real directory"):
        _publish_report(
            linked_parent / "diagnostic.json",
            report,
            publication_guard=_noop_publication_guard,
        )


def test_publication_guard_failure_rolls_back_only_owned_inode(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    report = _seal_report(_unsealed_report(root))
    output = tmp_path / "diagnostic.json"
    displaced = tmp_path / "owned-report.json"
    phases: list[str] = []

    def guard(phase_id: str) -> None:
        phases.append(phase_id)
        if phase_id == "sparse_v2_publication_verified":
            output.rename(displaced)
            output.write_bytes(b"foreign replacement")
            raise RuntimeError("resident_cap_crossed")

    with pytest.raises(RuntimeError, match="resident_cap_crossed"):
        _publish_report(output, report, publication_guard=guard)

    assert phases == [
        "sparse_v2_temporary_sealed",
        "sparse_v2_publication_verified",
    ]
    assert output.read_bytes() == b"foreign replacement"
    assert displaced.exists()
    assert list(tmp_path.glob(".*.tmp")) == []
