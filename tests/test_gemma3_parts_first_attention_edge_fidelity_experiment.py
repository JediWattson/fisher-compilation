from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from fisher_graph import gemma3_parts_first_attention_edge_fidelity_experiment as fidelity
from fisher_graph.gemma3_parts_first_attention_edge_experiment import (
    build_deterministic_attention_edge_fixture,
)

from fisher_graph.gemma3_parts_first_attention_edge_fidelity_experiment import (
    ATTENTION_EDGE_FIDELITY_ABI,
    ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS,
    ATTENTION_EDGE_FIDELITY_K_VALUES,
    ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS,
    ATTENTION_EDGE_FIDELITY_RANDOM_SALTS,
    ATTENTION_EDGE_FIDELITY_ROLE,
    ATTENTION_EDGE_FIDELITY_SCHEMA,
    ATTENTION_EDGE_FIDELITY_VARIANTS,
    DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT,
    PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256,
    PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256,
    AttentionEdgeFidelityCondition,
    AttentionEdgeFidelityLiveConfig,
    AttentionFidelityVectorStats,
    build_attention_edge_fidelity_preflight,
    build_deterministic_attention_edge_fidelity_fixture,
    load_attention_edge_fidelity,
    main,
    parse_attention_edge_fidelity_json,
    validate_attention_edge_fidelity,
    write_attention_edge_fidelity,
)
from fisher_graph.gemma3_parts_first_supermode_model_runner import (
    PINNED_PANEL_PATH,
)


REPORT_DOMAIN = b"fisher_graph:parts-first-attention-edge-fidelity:v1\0"
ROW_DOMAIN = b"fisher_graph:parts-first-attention-edge-fidelity-row:v1\0"
SELECTION_DOMAIN = b"fisher_graph:parts-first-attention-edge-selection:v1\0"
TENSOR_DOMAIN = b"fisher_graph:parts-first-attention-edge-fidelity-tensor:v1\0"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _rehash_report(value: dict[str, object]) -> None:
    payload = dict(value)
    payload.pop("report_sha256")
    value["report_sha256"] = hashlib.sha256(
        REPORT_DOMAIN + _canonical(payload)
    ).hexdigest()


def _rehash_row(value: dict[str, object]) -> None:
    measurements = value["measurements"]
    assert isinstance(measurements, list)
    value["measurement_catalog_sha256"] = hashlib.sha256(
        SELECTION_DOMAIN + _canonical(measurements)
    ).hexdigest()
    payload = dict(value)
    payload.pop("row_sha256")
    value["row_sha256"] = hashlib.sha256(
        ROW_DOMAIN + _canonical(payload)
    ).hexdigest()


def test_frozen_condition_inventory_and_claim_boundary() -> None:
    report = build_deterministic_attention_edge_fidelity_fixture()

    assert ATTENTION_EDGE_FIDELITY_K_VALUES == (1, 2, 4, 8, 16)
    assert ATTENTION_EDGE_FIDELITY_PRIMARY_RANKERS == (
        "fisher",
        "probability",
        "recency",
        "deterministic_random",
    )
    assert ATTENTION_EDGE_FIDELITY_DIAGNOSTIC_RANKERS == (
        "route_output_energy",
    )
    assert ATTENTION_EDGE_FIDELITY_VARIANTS == (
        "unnormalized_contribution",
        "renormalized_sparse_attention",
    )
    assert ATTENTION_EDGE_FIDELITY_RANDOM_SALTS == (
        "primary",
        "replicate_1",
        "replicate_2",
        "replicate_3",
    )
    assert len(report.cells) == 53
    assert report.role == ATTENTION_EDGE_FIDELITY_ROLE
    assert report.atlas_extraction_identity_count == 18
    assert report.atlas_fisher_k8_coordinate_identity_count == 18
    assert report.full_atlas_1080_identity_passed is False
    assert report.self_proves_live_fidelity is False
    assert report.selection_optimality_claimed is False
    assert report.fresh_panel_used is False
    assert report.compression_qualification_claimed is False


def test_fixture_strict_json_round_trip_and_derived_metrics() -> None:
    report = build_deterministic_attention_edge_fidelity_fixture()
    encoded = json.dumps(report.to_dict(), allow_nan=False)

    parsed = parse_attention_edge_fidelity_json(encoded)

    assert parsed == report
    assert validate_attention_edge_fidelity(parsed) is parsed
    fisher = next(
        cell
        for cell in parsed.cells
        if cell.condition
        == AttentionEdgeFidelityCondition(
            "fisher", 8, "unnormalized_contribution"
        )
    )
    assert fisher.starved_head_query_count == 432
    assert fisher.operator_nrmse > 0
    assert fisher.final_carrier_nrmse > 0
    assert fisher.terminal_nrmse > 0
    assert 0 <= fisher.top1_retention_rate <= 1
    assert len(parsed.family_cells) == 53
    assert len(parsed.row_receipts[0].measurements) == 53
    assert len(
        parsed.row_receipts[0].measurements[0].metrics.propagated_mutations
    ) == 36
    assert len(parsed.row_receipts[0].mutation_delta_sha256s) == 36
    assert len(parsed.row_receipts[0].mutation_carrier_sha256s) == 36
    assert parsed.cells[0].metrics.valid_logits.maximum_abs_error >= 0
    assert parsed.cells[0].metrics.centered_valid_logits.element_count > 0
    assert 0 <= parsed.cells[0].baseline_target_accuracy <= 1
    assert 0 <= parsed.cells[0].sparse_target_accuracy <= 1
    persisted = parsed.to_dict()
    assert "target_id" not in json.dumps(persisted)
    assert "target_token" not in json.dumps(persisted)


def test_parser_rejects_duplicate_nonfinite_and_unknown_fields() -> None:
    report = build_deterministic_attention_edge_fidelity_fixture()
    encoded = json.dumps(report.to_dict(), allow_nan=False)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_attention_edge_fidelity_json('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError, match="non-finite"):
        parse_attention_edge_fidelity_json('{"x":NaN}')
    value = report.to_dict()
    value["unknown"] = True
    with pytest.raises(ValueError, match="field inventory"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_derived_metric_tamper_fails_even_after_outer_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    cells = value["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    cell["operator_nrmse"] = float(cell["operator_nrmse"]) + 0.1
    _rehash_report(value)

    with pytest.raises(ValueError, match="derived operator_nrmse"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_vector_algebra_tamper_fails_after_coordinated_row_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    rows = value["row_receipts"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    measurements = row["measurements"]
    assert isinstance(measurements, list)
    measurement = measurements[0]
    assert isinstance(measurement, dict)
    metrics = measurement["metrics"]
    assert isinstance(metrics, dict)
    operator = metrics["local_operator"]
    assert isinstance(operator, dict)
    operator["error_sum_squares"] = (
        float(operator["error_sum_squares"]) + 1.0
    )
    operator["nrmse"] = (
        float(operator["error_sum_squares"])
        / float(operator["reference_sum_squares"])
    ) ** 0.5
    operator["rmse"] = (
        float(operator["error_sum_squares"])
        / int(operator["element_count"])
    ) ** 0.5
    measurement["operator_error_sum_squares"] = operator[
        "error_sum_squares"
    ]
    _rehash_row(row)
    _rehash_report(value)

    with pytest.raises(ValueError, match="quadratic identity"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_vector_stats_reject_cauchy_and_maxabs_impossibilities() -> None:
    with pytest.raises(ValueError, match="Cauchy"):
        AttentionFidelityVectorStats(
            element_count=6,
            reference_sum_squares=1.0,
            sparse_sum_squares=1.0,
            error_sum_squares=6.0,
            dot_product=-2.0,
            maximum_abs_error=1.0,
        )
    with pytest.raises(ValueError, match="maximum exceeds total error"):
        AttentionFidelityVectorStats(
            element_count=2,
            reference_sum_squares=1.0,
            sparse_sum_squares=0.0,
            error_sum_squares=1.0,
            dot_product=0.0,
            maximum_abs_error=2.0,
        )


def test_selection_variant_catalog_tamper_fails_after_coordinated_hashes() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    rows = value["row_receipts"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    measurements = row["measurements"]
    assert isinstance(measurements, list)
    renormalized = next(
        item
        for item in measurements
        if isinstance(item, dict)
        and isinstance(item.get("condition"), dict)
        and item["condition"]["ranker"] == "fisher"
        and item["condition"]["k"] == 1
        and item["condition"]["variant"]
        == "renormalized_sparse_attention"
    )
    renormalized["selection_catalog_sha256"] = "f" * 64
    _rehash_row(row)
    _rehash_report(value)

    with pytest.raises(ValueError, match="frozen selection"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_baseline_drift_fails_after_algebra_preserving_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    rows = value["row_receipts"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    measurements = row["measurements"]
    assert isinstance(measurements, list)
    measurement = measurements[0]
    assert isinstance(measurement, dict)
    metrics = measurement["metrics"]
    assert isinstance(metrics, dict)
    operator = metrics["local_operator"]
    assert isinstance(operator, dict)
    for field in (
        "reference_sum_squares",
        "sparse_sum_squares",
        "error_sum_squares",
        "dot_product",
    ):
        operator[field] = float(operator[field]) * 4.0
    operator["maximum_abs_error"] = (
        float(operator["maximum_abs_error"]) * 2.0
    )
    operator["rmse"] = float(operator["rmse"]) * 2.0
    for scalar, metric in (
        ("operator_reference_sum_squares", "reference_sum_squares"),
        ("operator_sparse_sum_squares", "sparse_sum_squares"),
        ("operator_error_sum_squares", "error_sum_squares"),
        ("operator_dot_product", "dot_product"),
        ("operator_max_abs_error", "maximum_abs_error"),
    ):
        measurement[scalar] = operator[metric]
    _rehash_row(row)
    _rehash_report(value)

    with pytest.raises(ValueError, match="row-invariant fidelity baselines"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_family_counts_must_close_over_row_receipts_after_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    value["family_counts"] = [["counterfeit-family", 1]]
    _rehash_report(value)

    with pytest.raises(ValueError, match="family counts do not close"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_attention_delta_must_bind_odd_mutation_after_coordinated_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    rows = value["row_receipts"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    attention_hashes = row["attention_delta_sha256s"]
    assert isinstance(attention_hashes, list)
    attention_hashes[0] = "f" * 64
    row["attention_delta_catalog_sha256"] = hashlib.sha256(
        TENSOR_DOMAIN
        + _canonical(
            {
                "label": "attention_delta",
                "tensor_sha256s": attention_hashes,
            }
        )
    ).hexdigest()
    _rehash_row(row)
    _rehash_report(value)

    with pytest.raises(ValueError, match="odd mutations"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_final_carrier_must_bind_mutation_36_after_coordinated_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    rows = value["row_receipts"]
    assert isinstance(rows, list)
    row = rows[0]
    assert isinstance(row, dict)
    row["final_carrier_sha256"] = "e" * 64
    _rehash_row(row)
    _rehash_report(value)

    with pytest.raises(ValueError, match="final carrier.*mutation 36"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_false_claim_boundary_fails_even_after_outer_rehash() -> None:
    value = build_deterministic_attention_edge_fidelity_fixture().to_dict()
    value["compression_qualification_claimed"] = True
    _rehash_report(value)

    with pytest.raises(ValueError, match="claim boundary"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_report_hash_binds_every_persisted_field() -> None:
    report = build_deterministic_attention_edge_fidelity_fixture()
    value = report.to_dict()
    value["role"] = "wrong"

    with pytest.raises(ValueError, match="schema/ABI|hash mismatch"):
        parse_attention_edge_fidelity_json(json.dumps(value))


def test_writer_is_0600_no_clobber_and_strict_reload(tmp_path: Path) -> None:
    report = build_deterministic_attention_edge_fidelity_fixture()
    destination = tmp_path / "nested" / "fidelity.json"

    written = write_attention_edge_fidelity(report, destination)

    assert written == report
    assert load_attention_edge_fidelity(destination) == report
    metadata = destination.lstat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_attention_edge_fidelity(report, destination)
    assert not list(destination.parent.glob(".*.tmp"))


def test_loader_rejects_symlink_and_unsafe_mode(tmp_path: Path) -> None:
    report = build_deterministic_attention_edge_fidelity_fixture()
    destination = tmp_path / "fidelity.json"
    write_attention_edge_fidelity(report, destination)
    alias = tmp_path / "alias.json"
    alias.symlink_to(destination)

    with pytest.raises(ValueError, match="unavailable or unsafe"):
        load_attention_edge_fidelity(alias)
    os.chmod(destination, 0o644)
    with pytest.raises(ValueError, match="identity is unsafe"):
        load_attention_edge_fidelity(destination)


def test_preflight_is_model_inert_and_binds_source_inventory() -> None:
    preflight = build_attention_edge_fidelity_preflight()

    assert preflight["schema"] == ATTENTION_EDGE_FIDELITY_SCHEMA
    assert preflight["abi"] == ATTENTION_EDGE_FIDELITY_ABI
    assert preflight["role"] == ATTENTION_EDGE_FIDELITY_ROLE
    assert preflight["panel_path"] == str(PINNED_PANEL_PATH)
    assert preflight["condition_count"] == 53
    assert preflight["model_loaded"] is False
    assert preflight["atlas_opened"] is False
    assert preflight["panel_opened"] is False
    assert preflight["live_execution_performed"] is False
    assert (
        preflight["pinned_atlas_report_sha256"]
        == PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256
    )
    assert (
        preflight["pinned_atlas_file_sha256"]
        == PINNED_ATTENTION_EDGE_ATLAS_FILE_SHA256
    )
    assert preflight["post_execution_source_recheck_required"] is True
    source_fields = [
        key
        for key in preflight
        if key.endswith("implementation_sha256")
    ]
    assert len(source_fields) == 7
    assert all(len(str(preflight[key])) == 64 for key in source_fields)


def test_import_and_default_cli_do_not_import_transformers() -> None:
    script = """
import json, sys
import fisher_graph.gemma3_parts_first_attention_edge_fidelity_experiment as m
print(json.dumps({"transformers": "transformers" in sys.modules, "preflight": m.main([])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    lines = result.stdout.strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload == {"transformers": False, "preflight": 0}


def test_demo_cli_is_compact_and_write_is_opt_in(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["condition_count"] == 53
    assert payload["source_kind"] == "model_free_fixture"
    assert "cells" not in payload


def test_canonical_output_rejects_fixture_and_smoke_poisoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    canonical = tmp_path / DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT

    with pytest.raises(ValueError, match="noncanonical output"):
        main(["--demo", "--write"])
    assert not canonical.exists()

    with pytest.raises(ValueError, match="full 60-row model_capture"):
        write_attention_edge_fidelity(
            build_deterministic_attention_edge_fidelity_fixture(),
            DEFAULT_ATTENTION_EDGE_FIDELITY_OUTPUT,
        )
    assert not canonical.exists()


def test_wrong_but_valid_atlas_is_not_accepted_as_published_A() -> None:
    wrong_valid_atlas = build_deterministic_attention_edge_fixture()
    assert wrong_valid_atlas.report_sha256 != (
        PINNED_ATTENTION_EDGE_ATLAS_REPORT_SHA256
    )

    with pytest.raises(ValueError, match="frozen live A atlas"):
        fidelity._validate_source_atlas(wrong_valid_atlas)


def test_wrong_atlas_file_bytes_fail_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    atlas = tmp_path / "wrong-atlas.json"
    atlas.write_bytes(b"{}")
    os.chmod(atlas, 0o600)
    monkeypatch.setattr(
        fidelity,
        "_capture_committed_source_state",
        lambda: object(),
    )

    def forbidden_model_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("model load must not run for wrong atlas bytes")

    monkeypatch.setattr(
        fidelity.AttentionEdgeLiveRuntime,
        "load",
        forbidden_model_load,
    )
    with pytest.raises(ValueError, match="file SHA-256 drifted"):
        fidelity.AttentionEdgeFidelityRuntime.load(
            AttentionEdgeFidelityLiveConfig(
                atlas_path=atlas,
                output=tmp_path / "unused.json",
            )
        )


def test_cli_rejects_existing_output_before_execution(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="before an existing"):
        main(["--demo", "--write", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "do not replace"


def test_live_config_freezes_scientific_parameters() -> None:
    assert AttentionEdgeFidelityLiveConfig().panel_path == PINNED_PANEL_PATH
    with pytest.raises(ValueError, match="maximum_prompt_tokens"):
        AttentionEdgeFidelityLiveConfig(maximum_prompt_tokens=64)
    with pytest.raises(ValueError, match="local-only"):
        AttentionEdgeFidelityLiveConfig(local_files_only=False)
    with pytest.raises(ValueError, match="must use .json"):
        AttentionEdgeFidelityLiveConfig(output=Path("bad.txt"))


def test_condition_rejects_nonrandom_replicate_and_nonprimary_shape() -> None:
    with pytest.raises(ValueError, match="random replicates"):
        AttentionEdgeFidelityCondition(
            "fisher", 8, "unnormalized_contribution", "replicate_1"
        )
    with pytest.raises(ValueError, match="frozen at K8"):
        AttentionEdgeFidelityCondition(
            "deterministic_random",
            4,
            "unnormalized_contribution",
            "replicate_1",
        )


def test_typed_report_rejects_subclass() -> None:
    with pytest.raises(TypeError, match="exact typed report"):
        validate_attention_edge_fidelity(object())  # type: ignore[arg-type]
