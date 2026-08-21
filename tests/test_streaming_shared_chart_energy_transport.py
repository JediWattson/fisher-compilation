from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import pytest
import torch

import fisher_graph.streaming_shared_chart_energy_transport as streaming
from fisher_graph.shared_chart_energy_transport import (
    FIT_ROLE,
    ChartRowKey,
    SharedChartNumericsConfig,
    SharedChartRowBatch,
    apply_shared_chart_atlas,
    balanced_atlas_row_weights,
    build_within_family_row_shuffle,
    evaluate_directed_chart_transport,
    fit_directed_chart_transport,
    fit_shared_chart_atlas,
    load_directed_chart_transport,
)
from fisher_graph.streaming_shared_chart_energy_transport import (
    RawModeRowSpoolWriter,
    StreamingChartResourceConfig,
    apply_shared_chart_atlas_streaming,
    estimate_streaming_chart_resources,
    evaluate_directed_chart_transport_streaming,
    fit_all_ordered_directed_chart_transports_streaming,
    fit_directed_chart_transport_streaming,
    fit_shared_chart_atlas_streaming,
    open_raw_mode_row_spool,
    open_streaming_applied_chart_store,
    streaming_chart_kernel_receipt,
    write_raw_mode_row_spool,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _keys(
    *, families: int = 2, examples: int = 3, rows: int = 5
) -> tuple[ChartRowKey, ...]:
    return tuple(
        ChartRowKey(
            f"family.{family}",
            f"example.{family}.{example}",
            row,
            row + 1,
        )
        for family in range(families)
        for example in range(examples)
        for row in range(rows)
    )


def _batches(
    role: str,
    label: str,
    *,
    mode_count: int = 3,
    width: int = 4,
    contribution_shift: float = 0.0,
) -> tuple[SharedChartRowBatch, ...]:
    keys = _keys()
    result = []
    for mode in range(mode_count):
        generator = torch.Generator().manual_seed(
            91_000 + sum(label.encode()) * 19 + mode
        )
        context = torch.randn(
            len(keys), width, generator=generator, dtype=torch.float32
        )
        midpoint = len(keys) // 2
        context[:midpoint, 0] -= 4.0
        context[midpoint:, 0] += 4.0
        values = (0.2 * context + float(mode + 1)).to(torch.float32)
        score = torch.randn(
            len(keys), width, generator=generator, dtype=torch.float32
        )
        contribution = (values * score + contribution_shift).to(torch.float32)
        result.append(
            SharedChartRowBatch(
                role=role,
                panel_binding_sha256=_sha(label),
                mode_id=f"mode.{mode:02d}",
                row_keys=keys,
                context_values=context,
                comparison_values=values,
                generalized_contributions=contribution,
            )
        )
    return tuple(result)


@pytest.fixture
def resources() -> StreamingChartResourceConfig:
    return StreamingChartResourceConfig(
        row_chunk_size=5,
        atlas_chunk_size=7,
        max_resident_bytes=256 * 1024 * 1024,
        max_artifact_bytes=64 * 1024 * 1024,
    )


def _write_spools(
    tmp_path: Path,
    batches: tuple[SharedChartRowBatch, ...],
    resources: StreamingChartResourceConfig,
    *,
    prefix: str,
):
    return tuple(
        write_raw_mode_row_spool(
            batch,
            tmp_path / f"{prefix}-{index}.fgstream",
            resources=resources,
        )
        for index, batch in enumerate(batches)
    )


def test_raw_spool_append_roundtrip_binds_catalog_and_float32_semantics(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    batch = _batches(FIT_ROLE, "A")[0]
    path = tmp_path / "raw.fgstream"
    writer = RawModeRowSpoolWriter.create(
        path,
        role=batch.role,
        panel_binding_sha256=batch.panel_binding_sha256,
        mode_id=batch.mode_id,
        row_keys=batch.row_keys,
        hidden_width=batch.width,
        resources=resources,
    )
    for start in range(0, len(batch.row_keys), 3):
        end = min(len(batch.row_keys), start + 3)
        writer.append(
            batch.row_keys[start:end],
            batch.context_values[start:end],
            batch.comparison_values[start:end],
            batch.generalized_contributions[start:end],
        )
    spool = writer.seal()
    loaded = open_raw_mode_row_spool(path)
    assert loaded.artifact_sha256 == spool.artifact_sha256
    assert loaded.row_keys == batch.row_keys
    assert loaded.tensor_catalog_sha256 == batch.tensor_catalog_sha256
    chunks = list(loaded.iter_float64_chunks(chunk_size=4))
    context = torch.cat(
        [chunk.tensors["context_values"] for chunk in chunks], dim=0
    )
    assert torch.equal(context, batch.context_values)
    assert all(chunk.row_keys == batch.row_keys[chunk.start_row : chunk.end_row] for chunk in chunks)
    with pytest.raises(FileExistsError):
        write_raw_mode_row_spool(batch, path, resources=resources)


def test_raw_spool_rejects_duplicate_rows_nonfinite_wrong_order_and_incomplete(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    batch = _batches(FIT_ROLE, "A")[0]
    with pytest.raises(ValueError, match="unique"):
        RawModeRowSpoolWriter.create(
            tmp_path / "duplicate.fgstream",
            role=batch.role,
            panel_binding_sha256=batch.panel_binding_sha256,
            mode_id=batch.mode_id,
            row_keys=(batch.row_keys[0], batch.row_keys[0]),
            hidden_width=batch.width,
            resources=resources,
        )
    writer = RawModeRowSpoolWriter.create(
        tmp_path / "wrong-order.fgstream",
        role=batch.role,
        panel_binding_sha256=batch.panel_binding_sha256,
        mode_id=batch.mode_id,
        row_keys=batch.row_keys,
        hidden_width=batch.width,
        resources=resources,
    )
    with pytest.raises(ValueError, match="catalog/order"):
        writer.append(
            batch.row_keys[1:2],
            batch.context_values[1:2],
            batch.comparison_values[1:2],
            batch.generalized_contributions[1:2],
        )
    bad = batch.context_values[:1].clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        writer.append(
            batch.row_keys[:1],
            bad,
            batch.comparison_values[:1],
            batch.generalized_contributions[:1],
        )
    with pytest.raises(ValueError, match="incomplete"):
        writer.seal()
    assert not (tmp_path / "wrong-order.fgstream").exists()


@pytest.mark.parametrize("attack", ["flip", "truncate", "trailing", "duplicate-json"])
def test_raw_spool_strict_artifact_attacks_fail_closed(
    tmp_path: Path,
    resources: StreamingChartResourceConfig,
    attack: str,
) -> None:
    batch = _batches(FIT_ROLE, "A")[0]
    original = tmp_path / "original.fgstream"
    write_raw_mode_row_spool(batch, original, resources=resources)
    raw = original.read_bytes()
    attacked = tmp_path / f"{attack}.fgstream"
    if attack == "flip":
        changed = bytearray(raw)
        changed[-1] ^= 1
        attacked.write_bytes(changed)
    elif attack == "truncate":
        attacked.write_bytes(raw[:-1])
    elif attack == "trailing":
        attacked.write_bytes(raw + b"x")
    else:
        position = len(streaming._STORE_MAGIC)
        length = struct.unpack(">Q", raw[position : position + 8])[0]
        start = position + 8
        end = start + length
        header = raw[start:end]
        duplicate = header.replace(
            b'"kind":', b'"kind":"forged","kind":', 1
        )
        attacked.write_bytes(
            raw[:position]
            + struct.pack(">Q", len(duplicate))
            + duplicate
            + raw[end:]
        )
    with pytest.raises(ValueError):
        open_raw_mode_row_spool(attacked)
    if hasattr(os, "symlink"):
        linked = tmp_path / f"{attack}-link.fgstream"
        linked.symlink_to(original)
        with pytest.raises(ValueError, match="regular file"):
            open_raw_mode_row_spool(linked)


def test_spool_rejects_mutation_after_open_and_forged_resource_receipt(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    batch = _batches(FIT_ROLE, "A")[0]
    original = tmp_path / "mutable.fgstream"
    spool = write_raw_mode_row_spool(batch, original, resources=resources)
    raw = bytearray(original.read_bytes())
    raw[-1] ^= 1
    original.write_bytes(raw)
    with pytest.raises(ValueError, match="changed after authentication"):
        next(spool.iter_float64_chunks(fields=("context_values",)))

    clean = tmp_path / "clean.fgstream"
    write_raw_mode_row_spool(batch, clean, resources=resources)
    encoded = clean.read_bytes()
    position = len(streaming._STORE_MAGIC)
    length = struct.unpack(">Q", encoded[position : position + 8])[0]
    start = position + 8
    end = start + length
    header = json.loads(encoded[start:end])
    header["metadata"]["resource_config"]["max_resident_bytes"] += 1
    unsigned = dict(header)
    del unsigned["artifact_sha256"]
    header["artifact_sha256"] = streaming._digest(
        streaming._STORE_ARTIFACT_DOMAIN, unsigned
    )
    changed_header = streaming._canonical_json_bytes(header)
    forged = tmp_path / "forged-resource.fgstream"
    forged.write_bytes(
        encoded[:position]
        + struct.pack(">Q", len(changed_header))
        + changed_header
        + encoded[end:]
    )
    with pytest.raises(ValueError, match="resource/kernel"):
        open_raw_mode_row_spool(forged)


def test_failed_strict_reload_rolls_back_publication_and_allows_retry(
    tmp_path: Path,
    resources: StreamingChartResourceConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _batches(FIT_ROLE, "A")[0]
    path = tmp_path / "rollback.fgstream"
    real_open = streaming._open_array_store

    def fail_reload(*args, **kwargs):
        raise RuntimeError("injected streaming strict reload failure")

    monkeypatch.setattr(streaming, "_open_array_store", fail_reload)
    with pytest.raises(RuntimeError, match="injected streaming"):
        write_raw_mode_row_spool(batch, path, resources=resources)
    assert not path.exists()
    monkeypatch.setattr(streaming, "_open_array_store", real_open)
    recovered = write_raw_mode_row_spool(batch, path, resources=resources)
    assert recovered.tensor_catalog_sha256 == batch.tensor_catalog_sha256


def test_streaming_atlas_matches_reference_topology_and_strict_tolerance(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    batches = _batches(FIT_ROLE, "A")
    config = SharedChartNumericsConfig(
        hidden_width=4, chart_count=2, tangent_rank=2
    )
    reference = fit_shared_chart_atlas(batches, config=config)
    spools = _write_spools(tmp_path, batches, resources, prefix="A")
    actual = fit_shared_chart_atlas_streaming(
        tuple(reversed(spools)), config=config, resources=resources
    )
    assert actual.fit_mode_ids == reference.fit_mode_ids
    assert actual.fit_context_catalog_sha256 == reference.fit_context_catalog_sha256
    assert actual.fit_balanced_row_weight_catalog_sha256 == reference.fit_balanced_row_weight_catalog_sha256
    assert torch.equal(actual.hard_occupancy_count, reference.hard_occupancy_count)
    torch.testing.assert_close(actual.centers, reference.centers, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual.radius_squared, reference.radius_squared, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(actual.tangent_bases, reference.tangent_bases, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(actual.tangent_variance, reference.tangent_variance, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(actual.orthogonal_alpha, reference.orthogonal_alpha, rtol=1e-9, atol=1e-9)
    assert actual.fit_solver_receipt_sha256 != reference.fit_solver_receipt_sha256


def test_streaming_atlas_is_context_only(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    first = _batches(FIT_ROLE, "same")
    changed = tuple(
        SharedChartRowBatch(
            role=batch.role,
            panel_binding_sha256=batch.panel_binding_sha256,
            mode_id=batch.mode_id,
            row_keys=batch.row_keys,
            context_values=batch.context_values,
            comparison_values=-batch.comparison_values,
            generalized_contributions=batch.generalized_contributions * 13 + 9,
        )
        for batch in first
    )
    config = SharedChartNumericsConfig(
        hidden_width=4, chart_count=2, tangent_rank=2
    )
    first_spools = _write_spools(tmp_path, first, resources, prefix="one")
    changed_spools = _write_spools(tmp_path, changed, resources, prefix="two")
    atlas_a = fit_shared_chart_atlas_streaming(
        first_spools, config=config, resources=resources
    )
    atlas_b = fit_shared_chart_atlas_streaming(
        changed_spools, config=config, resources=resources
    )
    assert atlas_a.atlas_sha256 == atlas_b.atlas_sha256
    assert torch.equal(atlas_a.tangent_bases, atlas_b.tangent_bases)


def test_applied_store_is_compact_and_matches_reference(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    batches = _batches(FIT_ROLE, "A")
    config = SharedChartNumericsConfig(
        hidden_width=4, chart_count=2, tangent_rank=2
    )
    spools = _write_spools(tmp_path, batches, resources, prefix="raw")
    atlas = fit_shared_chart_atlas_streaming(
        spools, config=config, resources=resources
    )
    store = apply_shared_chart_atlas_streaming(
        atlas, spools[0], tmp_path / "applied.fgstream", resources=resources
    )
    loaded = open_streaming_applied_chart_store(store.path)
    assert set(loaded._store.blocks) == streaming._APPLIED_FIELD_NAMES
    assert "context_values" not in loaded._store.blocks
    assert "comparison_values" not in loaded._store.blocks
    reference = apply_shared_chart_atlas(atlas, batches[0])
    features = torch.cat(
        [
            chunk.tensors["features"]
            for chunk in loaded.iter_float64_chunks(
                fields=("features",), chunk_size=4
            )
        ]
    )
    torch.testing.assert_close(features, reference.feature_matrix(), rtol=1e-9, atol=1e-9)
    fisher = torch.cat(
        [
            chunk.tensors["weighted_fisher_energy"]
            for chunk in loaded.iter_float64_chunks(
                fields=("weighted_fisher_energy",), chunk_size=4
            )
        ]
    )
    torch.testing.assert_close(fisher, reference.weighted_fisher_energy, rtol=1e-9, atol=1e-9)
    with pytest.raises(FileExistsError):
        apply_shared_chart_atlas_streaming(
            atlas, spools[0], store.path, resources=resources
        )


def test_streaming_transport_fit_eval_and_global_shuffle_match_reference(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    config = SharedChartNumericsConfig(
        hidden_width=4, chart_count=2, tangent_rank=2
    )
    fit_batches = _batches(FIT_ROLE, "A")
    fit_spools = _write_spools(tmp_path, fit_batches, resources, prefix="fit")
    atlas = fit_shared_chart_atlas_streaming(
        fit_spools, config=config, resources=resources
    )
    fit_stores = tuple(
        apply_shared_chart_atlas_streaming(
            atlas,
            spool,
            tmp_path / f"fit-applied-{index}.fgstream",
            resources=resources,
        )
        for index, spool in enumerate(fit_spools)
    )
    actual_transport = fit_directed_chart_transport_streaming(
        atlas, fit_stores[0], fit_stores[1], resources=resources
    )
    reference_applied = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in fit_batches
    )
    reference_transport = fit_directed_chart_transport(
        atlas,
        reference_applied[0],
        reference_applied[1],
        row_weights=balanced_atlas_row_weights(fit_batches)["mode.00"],
    )
    torch.testing.assert_close(
        actual_transport.coefficient,
        reference_transport.coefficient,
        rtol=1e-8,
        atol=1e-8,
    )

    heldout_batches = _batches("B_neighbor_discovery", "B")
    heldout_spools = _write_spools(
        tmp_path, heldout_batches, resources, prefix="heldout"
    )
    heldout_stores = tuple(
        apply_shared_chart_atlas_streaming(
            atlas,
            spool,
            tmp_path / f"heldout-applied-{index}.fgstream",
            resources=resources,
        )
        for index, spool in enumerate(heldout_spools)
    )
    shuffle = build_within_family_row_shuffle(
        heldout_batches[0].row_keys, role="B_neighbor_discovery"
    )
    all_features = torch.cat(
        [
            chunk.tensors["features"]
            for chunk in heldout_stores[0].iter_float64_chunks(
                fields=("features",), chunk_size=4
            )
        ]
    )
    gathered = heldout_stores[0].read_rows(
        "features", shuffle.permutation[:13]
    )
    assert torch.equal(gathered, all_features[shuffle.permutation[:13]])
    actual = evaluate_directed_chart_transport_streaming(
        atlas,
        actual_transport,
        heldout_stores[0],
        heldout_stores[1],
        row_shuffle=shuffle,
        resources=resources,
    )
    heldout_reference = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout_batches
    )
    expected = evaluate_directed_chart_transport(
        atlas,
        actual_transport,
        heldout_reference[0],
        heldout_reference[1],
        row_shuffle=shuffle,
    )
    assert actual.row_shuffle_catalog_sha256 == expected.row_shuffle_catalog_sha256
    assert actual.prediction_backend_id == streaming.STREAMING_PREDICTION_BACKEND_ID
    assert len(actual.examples) == len(expected.examples)
    assert len(actual.examples) == 2 * 3
    assert all(item.row_count == 5 for item in actual.examples)
    for left, right in zip(actual.examples, expected.examples, strict=True):
        assert (left.family_id, left.example_id, left.row_count) == (
            right.family_id,
            right.example_id,
            right.row_count,
        )
        for field in (
            "weighted_transport_residual_energy",
            "weighted_predicted_fisher_energy",
            "weighted_target_fisher_energy",
            "weighted_predicted_target_cross_inner_product",
            "weighted_zero_prediction_residual_energy",
            "weighted_row_shuffled_transport_residual_energy",
        ):
            assert getattr(left, field) == pytest.approx(
                getattr(right, field), rel=1e-8, abs=1e-8
            )


def test_all_ordered_fit_persists_each_coefficient_immediately(
    tmp_path: Path, resources: StreamingChartResourceConfig
) -> None:
    batches = _batches(FIT_ROLE, "A")
    config = SharedChartNumericsConfig(
        hidden_width=4, chart_count=2, tangent_rank=2
    )
    spools = _write_spools(tmp_path, batches, resources, prefix="all")
    atlas = fit_shared_chart_atlas_streaming(
        spools, config=config, resources=resources
    )
    stores = tuple(
        apply_shared_chart_atlas_streaming(
            atlas,
            spool,
            tmp_path / f"all-applied-{index}.fgstream",
            resources=resources,
        )
        for index, spool in enumerate(spools)
    )
    progress: list[tuple[int, int]] = []
    receipts = fit_all_ordered_directed_chart_transports_streaming(
        atlas,
        stores,
        tmp_path / "transports",
        resources=resources,
        progress=lambda current, total, _source, _target: progress.append(
            (current, total)
        ),
    )
    assert len(receipts) == 3 * 2
    assert progress[-1] == (6, 6)
    assert len({receipt.transport_sha256 for receipt in receipts}) == 6
    for receipt in receipts:
        assert receipt.path.is_file()
        assert load_directed_chart_transport(receipt.path).transport_sha256 == receipt.transport_sha256


def test_resource_budget_is_enforced_and_kernel_is_authenticated(
    tmp_path: Path,
) -> None:
    resources = StreamingChartResourceConfig(
        row_chunk_size=2,
        atlas_chunk_size=2,
        max_resident_bytes=1_000,
        max_artifact_bytes=16 * 1024 * 1024,
    )
    batches = _batches(FIT_ROLE, "A")
    spools = _write_spools(tmp_path, batches, resources, prefix="budget")
    config = SharedChartNumericsConfig(
        hidden_width=4, chart_count=2, tangent_rank=2
    )
    with pytest.raises(MemoryError, match="streaming atlas fit"):
        fit_shared_chart_atlas_streaming(
            spools, config=config, resources=resources
        )
    first = streaming_chart_kernel_receipt(resources)
    second = streaming_chart_kernel_receipt(resources)
    assert first == second
    assert len(first["receipt_sha256"]) == 64


def test_realistic_k8_h640_resource_smoke_allocates_no_row_tensors() -> None:
    config = SharedChartNumericsConfig()
    resources = StreamingChartResourceConfig()
    estimate = estimate_streaming_chart_resources(
        row_count_per_mode=24_732,
        mode_count=38,
        config=config,
        resources=resources,
    )
    assert estimate.source_feature_width == 136
    assert estimate.raw_spool_payload_bytes_per_mode == 24_732 * 3 * 640 * 4
    assert estimate.raw_spool_payload_bytes_per_role == 24_732 * 3 * 640 * 4 * 38
    assert estimate.all_ordered_coefficient_bytes == 979_025_920
    assert estimate.atlas_peak_resident_bytes < resources.max_resident_bytes
    assert estimate.apply_peak_resident_bytes < resources.max_resident_bytes
    assert estimate.transport_fit_peak_resident_bytes < resources.max_resident_bytes
    assert estimate.transport_eval_peak_resident_bytes < resources.max_resident_bytes
    assert len(estimate.receipt_sha256) == 64
