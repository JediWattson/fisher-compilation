from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

import pytest
import torch

import fisher_graph.shared_chart_energy_transport as shared_runtime

from fisher_graph.shared_chart_energy_transport import (
    FIT_ROLE,
    AppliedChartRows,
    ChartRowKey,
    RowShuffleCatalog,
    SharedChartNumericsConfig,
    SharedChartRowBatch,
    apply_shared_chart_atlas,
    balanced_atlas_row_weights,
    build_within_family_row_shuffle,
    evaluate_directed_chart_transport,
    fit_directed_chart_transport,
    fit_shared_chart_atlas,
    load_directed_chart_transport,
    load_shared_chart_atlas,
    save_directed_chart_transport,
    save_shared_chart_atlas,
    target_fisher_balanced_transport_row_weights,
    transport_coefficient_storage_bytes,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _keys(*, families: int = 2, examples: int = 3, rows: int = 4):
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
    keys: tuple[ChartRowKey, ...] | None = None,
    contribution_offset: int = 0,
) -> tuple[SharedChartRowBatch, ...]:
    row_keys = _keys() if keys is None else keys
    result = []
    for mode in range(mode_count):
        generator = torch.Generator().manual_seed(
            10_000 + sum(label.encode()) * 17 + mode
        )
        context = torch.randn(
            len(row_keys), width, generator=generator, dtype=torch.float64
        )
        midpoint = len(row_keys) // 2
        context[:midpoint, 0] -= 3.0
        context[midpoint:, 0] += 3.0
        value = 0.25 * context + float(mode + 1)
        score = torch.randn(
            len(row_keys), width, generator=generator, dtype=torch.float64
        )
        contribution = value * score + contribution_offset
        result.append(
            SharedChartRowBatch(
                role=role,
                panel_binding_sha256=_sha(label),
                mode_id=f"mode.{mode:02d}",
                row_keys=row_keys,
                context_values=context,
                comparison_values=value,
                generalized_contributions=contribution,
            )
        )
    return tuple(result)


def _fit_three():
    config = SharedChartNumericsConfig(
        hidden_width=4,
        chart_count=2,
        tangent_rank=2,
    )
    batches = _batches(FIT_ROLE, "A")
    atlas = fit_shared_chart_atlas(batches, config=config)
    applied = tuple(apply_shared_chart_atlas(atlas, batch) for batch in batches)
    return config, batches, atlas, applied


def test_row_key_binds_prediction_query_and_next_label_positions() -> None:
    key = ChartRowKey("family", "example", 7, 8)
    assert key.payload() == ["family", "example", 7, 8]
    with pytest.raises(ValueError, match=r"t=q\+1"):
        ChartRowKey("family", "example", 7, 7)


def test_balanced_weights_are_equal_mode_family_example_then_row() -> None:
    batches = _batches(FIT_ROLE, "A")
    weights = balanced_atlas_row_weights(batches)
    assert set(weights) == {"mode.00", "mode.01", "mode.02"}
    assert torch.equal(weights["mode.00"], weights["mode.01"])
    expected = 1.0 / (3 * 2 * 3 * 4)
    assert torch.allclose(
        weights["mode.00"],
        torch.full((24,), expected, dtype=torch.float64),
        atol=0,
        rtol=0,
    )
    assert float(sum(value.sum() for value in weights.values())) == pytest.approx(1.0)


def test_atlas_is_deterministic_and_canonicalizes_every_basis_sign() -> None:
    config, batches, first, _ = _fit_three()
    second = fit_shared_chart_atlas(tuple(reversed(batches)), config=config)
    assert first.atlas_sha256 == second.atlas_sha256
    assert torch.equal(first.centers, second.centers)
    assert torch.equal(first.tangent_bases, second.tangent_bases)
    for chart in range(config.chart_count):
        for column in range(config.tangent_rank):
            vector = first.tangent_bases[chart, :, column]
            pivot = int(torch.argmax(vector.abs()))
            assert float(vector[pivot]) > 0.0


def test_atlas_geometry_uses_context_not_generalized_contribution() -> None:
    config = SharedChartNumericsConfig(
        hidden_width=4,
        chart_count=2,
        tangent_rank=2,
    )
    first = _batches(FIT_ROLE, "same-context")
    changed = tuple(
        SharedChartRowBatch(
            role=batch.role,
            panel_binding_sha256=batch.panel_binding_sha256,
            mode_id=batch.mode_id,
            row_keys=batch.row_keys,
            context_values=batch.context_values,
            comparison_values=batch.comparison_values,
            generalized_contributions=batch.generalized_contributions * 17.0 + 31.0,
        )
        for batch in first
    )
    atlas_a = fit_shared_chart_atlas(first, config=config)
    atlas_b = fit_shared_chart_atlas(changed, config=config)
    assert atlas_a.atlas_sha256 == atlas_b.atlas_sha256
    assert torch.equal(atlas_a.tangent_bases, atlas_b.tangent_bases)
    assert torch.equal(atlas_a.tangent_variance, atlas_b.tangent_variance)
    assert (
        atlas_a.fit_context_catalog_sha256
        == atlas_b.fit_context_catalog_sha256
    )


def test_applied_receipt_is_sign_complete_even_when_energies_match() -> None:
    _, batches, atlas, applied = _fit_three()
    original = applied[0]
    flipped_batch = replace(
        batches[0],
        generalized_contributions=-batches[0].generalized_contributions,
    )
    flipped = apply_shared_chart_atlas(atlas, flipped_batch)
    assert torch.equal(
        original.weighted_fisher_energy,
        flipped.weighted_fisher_energy,
    )
    assert (
        original.batch_tensor_catalog_sha256
        != flipped.batch_tensor_catalog_sha256
    )
    assert original.vector_catalog_sha256 != flipped.vector_catalog_sha256


def test_applied_energy_has_full_width_tangent_plus_orthogonal_closure() -> None:
    _, _, atlas, applied = _fit_three()
    rows: AppliedChartRows = applied[0]
    assert torch.all(rows.memberships > 0)
    assert torch.allclose(
        rows.memberships.sum(dim=1),
        torch.ones(len(rows.batch.row_keys), dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.all(rows.weighted_chart_orthogonal_energy > 0)
    assert torch.allclose(
        rows.weighted_chart_tangent_energy
        + rows.weighted_chart_orthogonal_energy,
        rows.weighted_fisher_energy,
        atol=1e-10,
        rtol=1e-10,
    )
    assert rows.feature_matrix().shape == (
        len(rows.batch.row_keys),
        atlas.config.chart_count * (atlas.config.tangent_rank + 1),
    )


def test_transport_binds_pair_design_target_weights_solver_and_ridge() -> None:
    _, batches, atlas, applied = _fit_three()
    weights = balanced_atlas_row_weights(batches)["mode.00"]
    first = fit_directed_chart_transport(
        atlas, applied[0], applied[1], row_weights=weights
    )
    second = fit_directed_chart_transport(
        atlas, applied[0], applied[2], row_weights=weights
    )
    assert first.fit_catalog_sha256 == second.fit_catalog_sha256
    assert first.fit_design_sha256 == second.fit_design_sha256
    assert first.fit_target_sha256 != second.fit_target_sha256
    assert first.fit_row_weights_sha256 != second.fit_row_weights_sha256
    assert first.solver_id == "balanced_float64_primal_ridge_solve_v1"
    assert first.absolute_ridge > 0
    assert first.transport_sha256 != second.transport_sha256


def test_transport_fit_rejects_skewed_weights_and_wrong_atlas_fit_panel() -> None:
    _, batches, atlas, applied = _fit_three()
    weights = balanced_atlas_row_weights(batches)["mode.00"]
    skewed = weights.clone()
    skewed[0] *= 2.0
    with pytest.raises(ValueError, match="balanced formula"):
        fit_directed_chart_transport(
            atlas,
            applied[0],
            applied[1],
            row_weights=skewed,
        )
    wrong_panel = apply_shared_chart_atlas(
        atlas,
        replace(batches[0], panel_binding_sha256=_sha("other-A")),
    )
    with pytest.raises(ValueError, match="exact panel/catalog"):
        fit_directed_chart_transport(
            atlas,
            wrong_panel,
            applied[1],
            row_weights=weights,
        )


def test_transport_weights_retain_zero_fisher_rows_with_exact_floor() -> None:
    _, batches, atlas, applied = _fit_three()
    target_batch = batches[1]
    contribution = target_batch.generalized_contributions.clone()
    contribution[0].zero_()
    zero_target = apply_shared_chart_atlas(
        atlas,
        SharedChartRowBatch(
            role=target_batch.role,
            panel_binding_sha256=target_batch.panel_binding_sha256,
            mode_id=target_batch.mode_id,
            row_keys=target_batch.row_keys,
            context_values=target_batch.context_values,
            comparison_values=target_batch.comparison_values,
            generalized_contributions=contribution,
        ),
    )
    assert float(zero_target.weighted_fisher_energy[0]) == 0.0
    weights = target_fisher_balanced_transport_row_weights(
        atlas,
        zero_target,
        balanced_row_weights=balanced_atlas_row_weights(batches)["mode.00"],
    )
    assert float(weights[0]) > 0.0
    assert float(weights.sum()) == pytest.approx(1.0)
    transport = fit_directed_chart_transport(
        atlas,
        applied[0],
        zero_target,
        row_weights=balanced_atlas_row_weights(batches)["mode.00"],
    )
    assert transport.fit_row_weights_sha256


def test_streaming_and_batch_evaluation_are_exactly_identical() -> None:
    _, batches, atlas, applied = _fit_three()
    transport = fit_directed_chart_transport(
        atlas,
        applied[0],
        applied[1],
        row_weights=balanced_atlas_row_weights(batches)["mode.00"],
    )
    heldout = _batches("B_neighbor_discovery", "B")
    heldout_applied = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout
    )
    shuffle = build_within_family_row_shuffle(
        heldout[0].row_keys, role="B_neighbor_discovery"
    )
    streamed = evaluate_directed_chart_transport(
        atlas,
        transport,
        heldout_applied[0],
        heldout_applied[1],
        row_shuffle=shuffle,
        chunk_size=3,
    )
    batch = evaluate_directed_chart_transport(
        atlas,
        transport,
        heldout_applied[0],
        heldout_applied[1],
        row_shuffle=shuffle,
        chunk_size=len(heldout[0].row_keys),
    )
    assert streamed == batch
    for receipt in streamed.examples:
        assert receipt.weighted_zero_prediction_residual_energy == (
            receipt.weighted_target_fisher_energy
        )
        expected = (
            receipt.weighted_predicted_fisher_energy
            + receipt.weighted_target_fisher_energy
            - 2.0 * receipt.weighted_predicted_target_cross_inner_product
        )
        assert receipt.weighted_transport_residual_energy == pytest.approx(expected)
        assert receipt.real_gain_over_zero == pytest.approx(
            1.0
            - receipt.weighted_transport_residual_energy
            / receipt.weighted_zero_prediction_residual_energy
        )
        if receipt.weighted_row_shuffled_transport_residual_energy > 0.0:
            assert receipt.real_gain_over_row_shuffle == pytest.approx(
                1.0
                - receipt.weighted_transport_residual_energy
                / receipt.weighted_row_shuffled_transport_residual_energy
            )
    assert streamed.comparison_abi_name == "carrier_chart_energy.h640.v1"
    assert streamed.prediction_backend_id == (
        "torch_mm_cpu_float64_fixed_rows_v1"
    )
    assert streamed.prediction_row_chunk_size == 256
    assert len(streamed.prediction_backend_receipt_sha256) == 64


def test_fixed_chunk_prediction_matches_rowwise_reference() -> None:
    generator = torch.Generator().manual_seed(810_2026)
    design = torch.randn(513, 7, generator=generator, dtype=torch.float64)
    coefficient = torch.randn(7, 11, generator=generator, dtype=torch.float64)
    actual = shared_runtime._fixed_chunk_linear(design, coefficient)
    expected = shared_runtime._rowwise_linear_reference(design, coefficient)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    receipt = shared_runtime.prediction_backend_receipt()
    assert receipt["backend_id"] == shared_runtime.PREDICTION_BACKEND_ID
    assert receipt["row_chunk_size"] == 256
    assert len(receipt["receipt_sha256"]) == 64


def test_fit_eval_reuse_and_wrong_shuffle_fail_closed() -> None:
    _, batches, atlas, applied = _fit_three()
    transport = fit_directed_chart_transport(
        atlas,
        applied[0],
        applied[1],
        row_weights=balanced_atlas_row_weights(batches)["mode.00"],
    )
    fit_shuffle = build_within_family_row_shuffle(
        batches[0].row_keys, role=FIT_ROLE
    )
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            applied[0],
            applied[1],
            row_shuffle=fit_shuffle,
        )
    heldout = _batches("B_neighbor_discovery", "B")
    heldout_applied = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout
    )
    with pytest.raises(ValueError, match="row-shuffle"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_applied[0],
            heldout_applied[1],
            row_shuffle=fit_shuffle,
        )


def test_evaluation_rejects_swapped_endpoints_atlas_and_forged_shuffle() -> None:
    _, batches, atlas, applied = _fit_three()
    transport = fit_directed_chart_transport(
        atlas,
        applied[0],
        applied[1],
        row_weights=balanced_atlas_row_weights(batches)["mode.00"],
    )
    heldout = _batches("B_neighbor_discovery", "B")
    heldout_applied = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout
    )
    shuffle = build_within_family_row_shuffle(
        heldout[0].row_keys,
        role="B_neighbor_discovery",
    )
    with pytest.raises(ValueError, match="source endpoint"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_applied[1],
            heldout_applied[0],
            row_shuffle=shuffle,
        )
    with pytest.raises(ValueError, match="target endpoint"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_applied[0],
            heldout_applied[2],
            row_shuffle=shuffle,
        )
    other_batches = _batches(FIT_ROLE, "other-A")
    other_atlas = fit_shared_chart_atlas(
        other_batches,
        config=atlas.config,
    )
    other_heldout = tuple(
        apply_shared_chart_atlas(other_atlas, batch) for batch in heldout
    )
    with pytest.raises(ValueError, match="atlas/config"):
        evaluate_directed_chart_transport(
            other_atlas,
            transport,
            other_heldout[0],
            other_heldout[1],
            row_shuffle=shuffle,
        )
    identity = RowShuffleCatalog(
        role="B_neighbor_discovery",
        row_catalog_sha256=shuffle.row_catalog_sha256,
        seed=shuffle.seed,
        permutation=torch.arange(len(heldout[0].row_keys)),
        catalog_sha256=_sha("forged-identity"),
    )
    with pytest.raises(ValueError, match="deterministic within-example"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_applied[0],
            heldout_applied[1],
            row_shuffle=identity,
        )
    across_family = shuffle.permutation.clone()
    across_family[0], across_family[-1] = (
        across_family[-1].clone(),
        across_family[0].clone(),
    )
    forged = replace(
        shuffle,
        permutation=across_family,
        catalog_sha256=_sha("forged-cross-family"),
    )
    with pytest.raises(ValueError, match="deterministic within-example"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_applied[0],
            heldout_applied[1],
            row_shuffle=forged,
        )
    changed_seed = build_within_family_row_shuffle(
        heldout[0].row_keys,
        role="B_neighbor_discovery",
        seed=6_608,
    )
    with pytest.raises(ValueError, match="seed"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout_applied[0],
            heldout_applied[1],
            row_shuffle=changed_seed,
        )
def test_row_shuffle_is_one_deterministic_within_example_derangement() -> None:
    keys = _keys()
    first = build_within_family_row_shuffle(
        keys, role="B_neighbor_discovery"
    )
    second = build_within_family_row_shuffle(
        keys, role="B_neighbor_discovery"
    )
    assert first.catalog_sha256 == second.catalog_sha256
    assert torch.equal(first.permutation, second.permutation)
    assert all(index != source for index, source in enumerate(first.permutation.tolist()))
    assert all(
        (
            keys[index].family_id,
            keys[index].example_id,
        )
        == (
            keys[source].family_id,
            keys[source].example_id,
        )
        for index, source in enumerate(first.permutation.tolist())
    )
    assert all(
        keys[index].prediction_query_position
        != keys[source].prediction_query_position
        for index, source in enumerate(first.permutation.tolist())
    )


def test_row_shuffle_mapping_and_residual_are_stable_when_examples_removed() -> None:
    _, fit_batches, atlas, fit_applied = _fit_three()
    transport = fit_directed_chart_transport(
        atlas,
        fit_applied[0],
        fit_applied[1],
        row_weights=balanced_atlas_row_weights(fit_batches)["mode.00"],
    )
    heldout = _batches("B_neighbor_discovery", "B")
    full_applied = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout
    )
    full_shuffle = build_within_family_row_shuffle(
        heldout[0].row_keys,
        role="B_neighbor_discovery",
    )
    removed_examples = {"example.0.1", "example.1.1"}
    keep = torch.tensor(
        [
            index
            for index, key in enumerate(heldout[0].row_keys)
            if key.example_id not in removed_examples
        ],
        dtype=torch.int64,
    )
    subset_batches = tuple(
        SharedChartRowBatch(
            role=batch.role,
            panel_binding_sha256=batch.panel_binding_sha256,
            mode_id=batch.mode_id,
            row_keys=tuple(batch.row_keys[index] for index in keep.tolist()),
            context_values=batch.context_values[keep],
            comparison_values=batch.comparison_values[keep],
            generalized_contributions=batch.generalized_contributions[keep],
        )
        for batch in heldout
    )
    subset_applied = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in subset_batches
    )
    subset_shuffle = build_within_family_row_shuffle(
        subset_batches[0].row_keys,
        role="B_neighbor_discovery",
    )

    full_mapping = {
        key: heldout[0].row_keys[source]
        for key, source in zip(
            heldout[0].row_keys,
            full_shuffle.permutation.tolist(),
        )
        if key.example_id not in removed_examples
    }
    subset_mapping = {
        key: subset_batches[0].row_keys[source]
        for key, source in zip(
            subset_batches[0].row_keys,
            subset_shuffle.permutation.tolist(),
        )
    }
    assert subset_mapping == full_mapping

    full_evaluation = evaluate_directed_chart_transport(
        atlas,
        transport,
        full_applied[0],
        full_applied[1],
        row_shuffle=full_shuffle,
    )
    subset_evaluation = evaluate_directed_chart_transport(
        atlas,
        transport,
        subset_applied[0],
        subset_applied[1],
        row_shuffle=subset_shuffle,
    )
    full_by_example = {
        (row.family_id, row.example_id): row
        for row in full_evaluation.examples
        if row.example_id not in removed_examples
    }
    subset_by_example = {
        (row.family_id, row.example_id): row
        for row in subset_evaluation.examples
    }
    assert subset_by_example == full_by_example


def test_strict_atlas_and_transport_bundles_roundtrip_and_refuse_clobber(
    tmp_path: Path,
) -> None:
    _, batches, atlas, applied = _fit_three()
    transport = fit_directed_chart_transport(
        atlas,
        applied[0],
        applied[1],
        row_weights=balanced_atlas_row_weights(batches)["mode.00"],
    )
    atlas_path = tmp_path / "atlas.fgchart"
    transport_path = tmp_path / "transport.fgchart"
    save_shared_chart_atlas(atlas, atlas_path)
    save_directed_chart_transport(transport, transport_path)
    loaded_atlas = load_shared_chart_atlas(atlas_path)
    loaded_transport = load_directed_chart_transport(transport_path)
    assert loaded_atlas.atlas_sha256 == atlas.atlas_sha256
    assert loaded_transport.transport_sha256 == transport.transport_sha256
    assert loaded_atlas.centers.data_ptr() != atlas.centers.data_ptr()
    assert loaded_transport.coefficient.data_ptr() != transport.coefficient.data_ptr()
    with pytest.raises(FileExistsError):
        save_shared_chart_atlas(atlas, atlas_path)


def test_one_byte_tamper_and_symlink_are_rejected(tmp_path: Path) -> None:
    _, _, atlas, _ = _fit_three()
    path = tmp_path / "atlas.fgchart"
    save_shared_chart_atlas(atlas, path)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    tampered = tmp_path / "tampered.fgchart"
    tampered.write_bytes(raw)
    with pytest.raises(ValueError, match="digest"):
        load_shared_chart_atlas(tampered)
    if hasattr(os, "symlink"):
        linked = tmp_path / "linked.fgchart"
        linked.symlink_to(path)
        with pytest.raises(ValueError, match="regular file"):
            load_shared_chart_atlas(linked)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("offset", False),
        ("byte_count", False),
        ("name", 7),
        ("sha256", 7),
        ("shape", [True, 4]),
        ("shape", [-1, 4]),
    ],
)
def test_bundle_tensor_catalog_exact_types_fail_closed(
    tmp_path: Path,
    field: str,
    bad_value: object,
) -> None:
    _, _, atlas, _ = _fit_three()
    original = tmp_path / "atlas.fgchart"
    save_shared_chart_atlas(atlas, original)
    raw = original.read_bytes()
    magic = shared_runtime._BUNDLE_MAGIC
    position = len(magic)
    header_length = struct.unpack(">Q", raw[position : position + 8])[0]
    header_start = position + 8
    header_end = header_start + header_length
    header = json.loads(raw[header_start:header_end])
    header["tensors"][0][field] = bad_value
    encoded = shared_runtime._canonical_json_bytes(header)
    attacked = tmp_path / f"bad-{field}-{type(bad_value).__name__}.fgchart"
    attacked.write_bytes(
        magic + struct.pack(">Q", len(encoded)) + encoded + raw[header_end:]
    )
    with pytest.raises(ValueError):
        load_shared_chart_atlas(attacked)


def test_failed_strict_reload_removes_publication_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, atlas, _ = _fit_three()
    path = tmp_path / "atlas.fgchart"
    real_load = shared_runtime._load_bundle

    def fail_reload(*args, **kwargs):
        raise RuntimeError("injected strict reload failure")

    monkeypatch.setattr(shared_runtime, "_load_bundle", fail_reload)
    with pytest.raises(RuntimeError, match="injected"):
        save_shared_chart_atlas(atlas, path)
    assert not path.exists()
    monkeypatch.setattr(shared_runtime, "_load_bundle", real_load)
    save_shared_chart_atlas(atlas, path)
    assert load_shared_chart_atlas(path).atlas_sha256 == atlas.atlas_sha256


def test_failed_directory_fsync_removes_publication_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, atlas, _ = _fit_three()
    path = tmp_path / "atlas.fgchart"
    real_fsync = shared_runtime.os.fsync

    def fail_directory(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(shared_runtime.os, "fsync", fail_directory)
    with pytest.raises(OSError, match="injected"):
        save_shared_chart_atlas(atlas, path)
    assert not path.exists()
    monkeypatch.setattr(shared_runtime.os, "fsync", real_fsync)
    save_shared_chart_atlas(atlas, path)
    assert load_shared_chart_atlas(path).atlas_sha256 == atlas.atlas_sha256


def test_production_mode_pair_shape_fits_all_1406_small_width_transports() -> None:
    keys = _keys(families=2, examples=2, rows=3)
    batches = _batches(
        FIT_ROLE,
        "A-production-shape",
        mode_count=38,
        width=3,
        keys=keys,
    )
    config = SharedChartNumericsConfig(
        hidden_width=3,
        chart_count=2,
        tangent_rank=1,
    )
    atlas = fit_shared_chart_atlas(batches, config=config)
    applied = {
        batch.mode_id: apply_shared_chart_atlas(atlas, batch)
        for batch in batches
    }
    weights = balanced_atlas_row_weights(batches)
    transports = [
        fit_directed_chart_transport(
            atlas,
            applied[source.mode_id],
            applied[target.mode_id],
            row_weights=weights[source.mode_id],
        )
        for source in batches
        for target in batches
        if source.mode_id != target.mode_id
    ]
    assert len(transports) == 38 * 37 == 1_406
    assert len({item.transport_sha256 for item in transports}) == 1_406
    assert len({item.fit_catalog_sha256 for item in transports}) == 1


def test_production_coefficient_memory_accounting_uses_actual_k136_h640() -> None:
    config = SharedChartNumericsConfig()
    assert config.source_feature_width == 136
    exact = 38 * 37 * 136 * 640 * 8
    assert transport_coefficient_storage_bytes(config, mode_count=38) == exact
    assert exact == 979_025_920
    assert exact / (1024**2) == pytest.approx(933.671875)
