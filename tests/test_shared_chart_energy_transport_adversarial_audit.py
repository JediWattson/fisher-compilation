from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct

import pytest
import torch

import fisher_graph.gemma3_chart_native_neighbor_setup as setup_runtime
import fisher_graph.shared_chart_energy_transport as transport_runtime
from fisher_graph.gemma3_chart_native_neighbor_setup import (
    build_setup,
    canonical_json_bytes,
    load_setup,
    write_setup,
)
from fisher_graph.shared_chart_energy_transport import (
    FIT_ROLE,
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
    load_shared_chart_atlas,
    save_shared_chart_atlas,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _row_keys() -> tuple[ChartRowKey, ...]:
    return tuple(
        ChartRowKey(
            f"family.{family}",
            f"example.{family}.{example}",
            position,
            position + 1,
        )
        for family in range(2)
        for example in range(3)
        for position in range(4)
    )


def _batches(
    role: str,
    panel_label: str,
    *,
    row_keys: tuple[ChartRowKey, ...] | None = None,
) -> tuple[SharedChartRowBatch, ...]:
    keys = _row_keys() if row_keys is None else row_keys
    batches = []
    for mode in range(3):
        generator = torch.Generator().manual_seed(
            20_000 + 31 * sum(panel_label.encode("utf-8")) + mode
        )
        context = torch.randn(
            len(keys), 4, generator=generator, dtype=torch.float64
        )
        midpoint = len(keys) // 2
        context[:midpoint, 0] -= 3.0
        context[midpoint:, 0] += 3.0
        value = 0.25 * context + mode + 1.0
        score = torch.randn(
            len(keys), 4, generator=generator, dtype=torch.float64
        )
        batches.append(
            SharedChartRowBatch(
                role=role,
                panel_binding_sha256=_sha(panel_label),
                mode_id=f"mode.{mode:02d}",
                row_keys=keys,
                context_values=context,
                comparison_values=value,
                generalized_contributions=value * score,
            )
        )
    return tuple(batches)


@pytest.fixture
def runtime_fixture():
    fit_batches = _batches(FIT_ROLE, "fit-panel")
    config = SharedChartNumericsConfig(
        hidden_width=4,
        chart_count=2,
        tangent_rank=2,
    )
    atlas = fit_shared_chart_atlas(fit_batches, config=config)
    fit_rows = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in fit_batches
    )
    transport = fit_directed_chart_transport(
        atlas,
        fit_rows[0],
        fit_rows[1],
        row_weights=balanced_atlas_row_weights(fit_batches)["mode.00"],
    )
    heldout_batches = _batches(
        "B_neighbor_discovery",
        "heldout-panel",
        row_keys=fit_batches[0].row_keys,
    )
    heldout_rows = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in heldout_batches
    )
    shuffle = build_within_family_row_shuffle(
        heldout_batches[0].row_keys,
        role="B_neighbor_discovery",
    )
    return fit_batches, atlas, fit_rows, transport, heldout_rows, shuffle


def test_frozen_transport_rejects_wrong_target_mode(runtime_fixture) -> None:
    _, atlas, _, transport, heldout, shuffle = runtime_fixture
    assert transport.target_mode_id == "mode.01"
    assert heldout[2].batch.mode_id == "mode.02"
    with pytest.raises(ValueError, match="target mode|endpoint"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout[0],
            heldout[2],
            row_shuffle=shuffle,
        )


def test_membership_mismatch_residual_matches_direct_spd_square(
    runtime_fixture,
) -> None:
    _, atlas, _, _, _, _ = runtime_fixture
    generator = torch.Generator().manual_seed(991)
    predicted = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    target = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    predicted_memberships = torch.tensor(
        [[0.9, 0.1], [0.25, 0.75], [0.6, 0.4]], dtype=torch.float64
    )
    target_memberships = torch.tensor(
        [[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]], dtype=torch.float64
    )
    _, _, _, observed = transport_runtime._pair_energy_rows(
        atlas,
        predicted,
        predicted_memberships,
        target,
        target_memberships,
    )
    expected = torch.zeros(3, dtype=torch.float64)
    for chart in range(atlas.config.chart_count):
        difference = (
            torch.sqrt(predicted_memberships[:, chart, None]) * predicted
            - torch.sqrt(target_memberships[:, chart, None]) * target
        )
        basis = atlas.tangent_bases[chart]
        projection = difference @ basis
        orthogonal = difference - projection @ basis.T
        expected += (
            projection.square() / atlas.tangent_variance[chart]
        ).sum(dim=1) + atlas.orthogonal_alpha[chart] * orthogonal.square().sum(
            dim=1
        )
    assert torch.allclose(observed, expected, atol=1.0e-12, rtol=1.0e-12)


def test_applied_row_receipt_binds_signed_generalized_contribution(
    runtime_fixture,
) -> None:
    _, _, _, _, heldout, _ = runtime_fixture
    original = heldout[1]
    sign_flipped_batch = replace(
        original.batch,
        generalized_contributions=-original.batch.generalized_contributions,
    )
    assert (
        sign_flipped_batch.tensor_catalog_sha256
        != original.batch.tensor_catalog_sha256
    )
    with pytest.raises(ValueError, match="authenticate source rows"):
        replace(original, batch=sign_flipped_batch)


def test_applied_receipt_rejects_all_finite_geometry_and_energy_mutations(
    runtime_fixture,
) -> None:
    _, _, _, _, heldout, _ = runtime_fixture
    original = heldout[0]
    coordinates = original.tangent_coordinates.clone()
    coordinates[0, 0, 0] += 0.125
    with pytest.raises(ValueError, match="authenticate applied"):
        replace(original, tangent_coordinates=coordinates)
    memberships = original.memberships.clone()
    transfer = min(float(memberships[0, 0]) / 2.0, 1.0e-3)
    memberships[0, 0] -= transfer
    memberships[0, 1] += transfer
    with pytest.raises(ValueError, match="authenticate applied"):
        replace(original, memberships=memberships)
    orthogonal = original.weighted_chart_orthogonal_energy.clone()
    fisher = original.weighted_fisher_energy.clone()
    orthogonal[0] += 0.25
    fisher[0] += 0.25
    with pytest.raises(ValueError, match="authenticate applied"):
        replace(
            original,
            weighted_chart_orthogonal_energy=orthogonal,
            weighted_fisher_energy=fisher,
        )
    changed_context_batch = replace(
        original.batch,
        context_values=original.batch.context_values + 0.001,
    )
    with pytest.raises(ValueError, match="authenticate source rows"):
        replace(original, batch=changed_context_batch)


def test_evaluation_rejects_unauthenticated_identity_shuffle(
    runtime_fixture,
) -> None:
    _, atlas, _, transport, heldout, valid_shuffle = runtime_fixture
    forged = RowShuffleCatalog(
        role=valid_shuffle.role,
        row_catalog_sha256=valid_shuffle.row_catalog_sha256,
        seed=valid_shuffle.seed,
        permutation=torch.arange(
            len(valid_shuffle.permutation), dtype=torch.int64
        ),
        catalog_sha256=valid_shuffle.catalog_sha256,
    )
    with pytest.raises(ValueError, match="row-shuffle|derangement|catalog"):
        evaluate_directed_chart_transport(
            atlas,
            transport,
            heldout[0],
            heldout[1],
            row_shuffle=forged,
        )


def test_evaluation_fails_closed_on_nonfinite_derived_energy(
    runtime_fixture,
) -> None:
    _, atlas, _, transport, heldout, shuffle = runtime_fixture
    with pytest.raises(
        ValueError,
        match="non-finite|finite energy|authenticate applied",
    ):
        explosive = replace(
            heldout[0],
            tangent_coordinates=torch.full_like(
                heldout[0].tangent_coordinates,
                1.0e308,
            ),
        )
        evaluate_directed_chart_transport(
            atlas,
            transport,
            explosive,
            heldout[1],
            row_shuffle=shuffle,
        )


def test_applied_rows_reject_finite_coordinate_tamper_under_stale_receipt(
    runtime_fixture,
) -> None:
    _, _, _, _, heldout, _ = runtime_fixture
    original = heldout[0]
    tampered_coordinates = original.tangent_coordinates.clone()
    tampered_coordinates[0, 0, 0] += 1.0e-3
    with pytest.raises(ValueError, match="vector catalog|authenticate"):
        replace(
            original,
            tangent_coordinates=tampered_coordinates,
        )


def test_transport_fit_rejects_rows_from_a_different_a_panel(
    runtime_fixture,
) -> None:
    fit_batches, atlas, _, _, _, _ = runtime_fixture
    foreign_batches = _batches(
        FIT_ROLE,
        "different-fit-panel",
        row_keys=fit_batches[0].row_keys,
    )
    foreign_rows = tuple(
        apply_shared_chart_atlas(atlas, batch) for batch in foreign_batches
    )
    with pytest.raises(ValueError, match="panel/catalog|atlas fit"):
        fit_directed_chart_transport(
            atlas,
            foreign_rows[0],
            foreign_rows[1],
            row_weights=balanced_atlas_row_weights(foreign_batches)["mode.00"],
        )


def test_transport_fit_rejects_noncanonical_base_row_weights(
    runtime_fixture,
) -> None:
    fit_batches, atlas, fit_rows, _, _, _ = runtime_fixture
    weights = balanced_atlas_row_weights(fit_batches)["mode.00"]
    noncanonical = weights.clone()
    noncanonical[0] *= 100.0
    with pytest.raises(ValueError, match="balanced|canonical.*weight"):
        fit_directed_chart_transport(
            atlas,
            fit_rows[0],
            fit_rows[1],
            row_weights=noncanonical,
        )


def test_setup_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    raw = canonical_json_bytes(build_setup())
    needle = b'"artifact_role":"frozen_execution_setup_without_measurement",'
    duplicate = raw.replace(
        needle,
        needle + needle,
        1,
    )
    assert duplicate != raw
    path = tmp_path / "duplicate-setup.json"
    path.write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate|invalid JSON"):
        load_setup(path)


def test_setup_attention_route_identity_includes_context_and_head() -> None:
    route_row_key = build_setup()["attention_cross_token_route_chart"][
        "route_row_key"
    ]
    assert "context_fingerprint_sha256" in route_row_key
    assert "query_head" in route_row_key


def test_setup_write_failure_rolls_back_partial_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "setup.json"
    real_dump = setup_runtime.json.dump

    def fail_after_partial_write(payload, handle, **kwargs) -> None:
        del payload, kwargs
        handle.write('{"partial"')
        raise OSError("injected setup serialization failure")

    monkeypatch.setattr(setup_runtime.json, "dump", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        write_setup(destination)
    assert not destination.exists()
    monkeypatch.setattr(setup_runtime.json, "dump", real_dump)
    write_setup(destination)
    assert load_setup(destination) == build_setup()


def test_setup_postpublication_reload_failure_rolls_back_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "setup.json"
    real_load = setup_runtime.load_setup

    def fail_reload(path: Path) -> dict[str, object]:
        del path
        raise RuntimeError("injected strict reload failure")

    monkeypatch.setattr(setup_runtime, "load_setup", fail_reload)
    with pytest.raises(RuntimeError, match="injected"):
        write_setup(destination)
    assert not destination.exists()
    monkeypatch.setattr(setup_runtime, "load_setup", real_load)
    write_setup(destination)
    assert load_setup(destination) == build_setup()


def test_bundle_loader_rejects_bool_for_zero_tensor_offset(
    runtime_fixture,
    tmp_path: Path,
) -> None:
    _, atlas, _, _, _, _ = runtime_fixture
    canonical_path = tmp_path / "atlas.fgchart"
    save_shared_chart_atlas(atlas, canonical_path)
    raw = canonical_path.read_bytes()
    magic = b"FGCHART1\n"
    header_start = len(magic) + 8
    header_length = struct.unpack(">Q", raw[len(magic) : header_start])[0]
    header = json.loads(raw[header_start : header_start + header_length])
    payload = raw[header_start + header_length :]
    assert header["tensors"][0]["offset"] == 0
    header["tensors"][0]["offset"] = False
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    forged_path = tmp_path / "bool-offset-atlas.fgchart"
    forged_path.write_bytes(
        magic + struct.pack(">Q", len(header_bytes)) + header_bytes + payload
    )
    with pytest.raises(ValueError, match="integer|tensor catalog"):
        load_shared_chart_atlas(forged_path)
