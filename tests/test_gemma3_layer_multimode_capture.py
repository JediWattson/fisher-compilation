from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import fisher_graph.gemma3_layer_multimode_capture as capture
from fisher_graph.gemma3_chart_native_neighbor_experiment import (
    CONFIRMATION_ROLE as PREDECESSOR_C_ROLE,
    DISCOVERY_ROLE as PREDECESSOR_B_ROLE,
    FIT_ROLE as PREDECESSOR_A_ROLE,
    CapturedBatch,
    CapturedExampleRows,
    CapturedModeRows,
    RoleCaptureReceipt,
)
from fisher_graph.shared_chart_energy_transport import (
    ChartRowKey,
    row_catalog_sha256,
)
from fisher_graph.streaming_shared_chart_energy_transport import RawModeRowSpool


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _mode_ids() -> tuple[str, ...]:
    return (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )


def _example(
    *,
    family: str,
    example: str,
    tokens: int,
) -> CapturedExampleRows:
    rows = tokens - 1
    modes = tuple(
        CapturedModeRows(
            mode_id=mode_id,
            context_values=torch.full(
                (rows, capture.GEMMA3_HIDDEN_WIDTH),
                float(index) + 0.25,
                dtype=torch.float32,
            ),
            comparison_values=torch.full(
                (rows, capture.GEMMA3_HIDDEN_WIDTH),
                -float(index) - 0.5,
                dtype=torch.float32,
            ),
            generalized_contributions=torch.zeros(
                rows,
                capture.GEMMA3_HIDDEN_WIDTH,
                dtype=torch.float32,
            ),
        )
        for index, mode_id in enumerate(_mode_ids())
    )
    return CapturedExampleRows(
        family_id=family,
        example_id=example,
        token_count=tokens,
        row_keys=tuple(
            (family, example, query, query + 1)
            for query in range(rows)
        ),
        modes=modes,
        context_fingerprint_sha256=_sha(f"context:{example}"),
        capture_receipt_sha256=_sha(f"capture:{example}"),
    )


def _batch(role: str) -> CapturedBatch:
    return CapturedBatch(
        role=role,
        examples=(
            _example(family="family.a", example="a1", tokens=3),
            _example(family="family.b", example="b1", tokens=4),
        ),
        lexical_nll_receipt_sha256=_sha("lexical"),
        execution_receipt_sha256=_sha("execution"),
    )


def _batch_with_examples(
    role: str,
    examples: tuple[CapturedExampleRows, ...],
) -> CapturedBatch:
    return CapturedBatch(
        role=role,
        examples=examples,
        lexical_nll_receipt_sha256=_sha("lexical-custom"),
        execution_receipt_sha256=_sha("execution-custom"),
    )


class _FakeOpenedStore:
    def __init__(
        self,
        *,
        path: Path,
        artifact_sha256: str,
        row_keys: tuple[ChartRowKey, ...],
        layer_ordinal: int,
    ) -> None:
        self.path = path
        self.artifact_sha256 = artifact_sha256
        self.row_keys = row_keys
        self._layer_ordinal = layer_ordinal
        self.read_calls = 0
        self.blocks = {
            name: {
                "semantic_float64_tensor_sha256": _sha(
                    f"semantic:{layer_ordinal}:{name}"
                )
            }
            for name in (
                "comparison_values",
                "context_values",
                "generalized_contributions",
            )
        }

    def read_slice(self, name: str, start: int, end: int) -> torch.Tensor:
        self.read_calls += 1
        values = {
            "context_values": float(self._layer_ordinal) + 0.25,
            "comparison_values": -float(self._layer_ordinal) - 0.5,
            "generalized_contributions": 0.0,
        }
        return torch.full(
            (end - start, capture.GEMMA3_HIDDEN_WIDTH),
            values[name],
            dtype=torch.float32,
        )


def _raw_spool(
    *,
    layer_ordinal: int,
    row_keys: tuple[ChartRowKey, ...],
    row_catalog: str,
    panel_binding_sha256: str,
    role: str = PREDECESSOR_A_ROLE,
    path: Path | None = None,
    artifact_sha256: str | None = None,
    tensor_catalog_sha256: str | None = None,
) -> RawModeRowSpool:
    mode_id = f"mode.mutation.{2 + 2 * layer_ordinal}"
    resolved_path = path or Path(f"/tmp/full-A-layer-{layer_ordinal}.spool")
    resolved_artifact = artifact_sha256 or _sha(f"artifact:{mode_id}")
    spool = object.__new__(RawModeRowSpool)
    values = {
        "_store": _FakeOpenedStore(
            path=resolved_path,
            artifact_sha256=resolved_artifact,
            row_keys=row_keys,
            layer_ordinal=layer_ordinal,
        ),
        "role": role,
        "panel_binding_sha256": panel_binding_sha256,
        "mode_id": mode_id,
        "row_catalog_sha256": row_catalog,
        "tensor_catalog_sha256": tensor_catalog_sha256
        or _sha(f"tensor-catalog:{mode_id}"),
        "hidden_width": capture.GEMMA3_HIDDEN_WIDTH,
        "row_chunk_size": 257,
        "kernel_receipt_sha256": _sha("kernel"),
    }
    for name, value in values.items():
        object.__setattr__(spool, name, value)
    return spool


@dataclass(frozen=True)
class _ArtifactWitness:
    mode_id: str
    artifact_sha256: str
    row_catalog_sha256: str
    tensor_catalog_sha256: str


@dataclass(frozen=True)
class _FullAFixture:
    role_manifest: dict[str, object]
    receipt: RoleCaptureReceipt
    mlp_spools: tuple[RawModeRowSpool, ...]
    all_mode_witnesses: tuple[object, ...]
    batch_example_ids: tuple[tuple[str, ...], ...]
    lexical_receipts: tuple[str, ...]
    execution_receipts: tuple[str, ...]
    example_receipts: tuple[tuple[str, str], ...]
    route_evidence: tuple[tuple[str, str, str, str], ...]
    runtime_session_binding_sha256: str
    panel_binding_sha256: str


@pytest.fixture(scope="module")
def full_A_fixture() -> _FullAFixture:
    lengths = (96, 511, 512, 513, 514, 576, 640, 768)
    members = [
        {
            "family_id": f"family.{family}",
            "example_id": f"family.{family}.length.{length}",
            "target_token_count": length,
        }
        for family in range(6)
        for length in lengths
    ]
    manifest: dict[str, object] = {
        "role": PREDECESSOR_A_ROLE,
        "ordered_members": members,
        "ordered_objective_row_catalog_sha256": _sha("A-objective-rows"),
        "score_objective_receipt_sha256": _sha("A-score-objective"),
    }
    chart_keys = tuple(
        ChartRowKey(
            member["family_id"],
            member["example_id"],
            query,
            query + 1,
        )
        for member in members
        for query in range(member["target_token_count"] - 1)
    )
    assert len(chart_keys) == capture.PRODUCTION_A_ROW_COUNT
    panel = _sha("A-panel")
    session = _sha("runtime-session")
    catalog = row_catalog_sha256(chart_keys, role=PREDECESSOR_A_ROLE)
    mlp_spools = tuple(
        _raw_spool(
            layer_ordinal=layer,
            row_keys=chart_keys,
            row_catalog=catalog,
            panel_binding_sha256=panel,
        )
        for layer in range(capture.GEMMA3_LAYER_COUNT)
    )
    mlp_by_mode = {spool.mode_id: spool for spool in mlp_spools}
    all_witnesses = tuple(
        mlp_by_mode.get(mode_id)
        or _ArtifactWitness(
            mode_id=mode_id,
            artifact_sha256=_sha(f"artifact:{mode_id}"),
            row_catalog_sha256=catalog,
            tensor_catalog_sha256=_sha(f"tensor-catalog:{mode_id}"),
        )
        for mode_id in _mode_ids()
    )
    plan = capture._predecessor_capture._capture_batch_plan(
        {"examples": members}
    )
    batch_example_ids = tuple(
        tuple(row["example_id"] for row in batch) for batch in plan
    )
    lexical = tuple(_sha(f"lexical:{index}") for index in range(45))
    executions = tuple(_sha(f"execution:{index}") for index in range(45))
    example_receipts = tuple(
        (member["example_id"], _sha(f"example:{member['example_id']}"))
        for member in members
    )
    route_evidence = tuple(
        (
            f"mode.mutation.{mutation}",
            member["example_id"],
            _sha(f"route-catalog:{mutation}:{member['example_id']}"),
            _sha(f"route-receipt:{mutation}:{member['example_id']}"),
        )
        for mutation in range(1, 37, 2)
        for member in members
    )
    receipt = capture._predecessor_capture._build_role_capture_receipt(
        role=PREDECESSOR_A_ROLE,
        runtime_session_binding_sha256=session,
        panel_binding_sha256=panel,
        role_manifest=manifest,
        spools=all_witnesses,
        batch_example_ids=batch_example_ids,
        lexical_receipts=lexical,
        execution_receipts=executions,
        example_receipts=example_receipts,
        route_evidence=route_evidence,
        dense_lexical_parity_receipt_sha256=_sha("dense-parity"),
        row_count=len(chart_keys),
    )
    return _FullAFixture(
        role_manifest=manifest,
        receipt=receipt,
        mlp_spools=mlp_spools,
        all_mode_witnesses=all_witnesses,
        batch_example_ids=batch_example_ids,
        lexical_receipts=lexical,
        execution_receipts=executions,
        example_receipts=example_receipts,
        route_evidence=route_evidence,
        runtime_session_binding_sha256=session,
        panel_binding_sha256=panel,
    )


def _install_fake_spool_opener(
    monkeypatch: pytest.MonkeyPatch,
    fixture: _FullAFixture,
    *,
    replacements: dict[Path, RawModeRowSpool] | None = None,
) -> list[Path]:
    by_path = {spool.path: spool for spool in fixture.mlp_spools}
    by_path.update(replacements or {})
    calls: list[Path] = []

    def _open(
        source: Path | str,
        *,
        max_artifact_bytes: int,
    ) -> RawModeRowSpool:
        del max_artifact_bytes
        path = Path(source)
        calls.append(path)
        return by_path[path]

    monkeypatch.setattr(capture, "open_raw_mode_row_spool", _open)
    return calls


def _open_full_A_bridge(
    fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[capture.FullALayerMLPSpoolBridge, list[Path]]:
    calls = _install_fake_spool_opener(monkeypatch, fixture)
    bridge = capture.open_full_A_layer_mlp_spool_bridge(
        capture_receipt=fixture.receipt,
        mlp_spools=fixture.mlp_spools,
        role_manifest=fixture.role_manifest,
        expected_capture_receipt_sha256=fixture.receipt.receipt_sha256,
        expected_runtime_session_binding_sha256=(
            fixture.runtime_session_binding_sha256
        ),
        expected_role_panel_binding_sha256=fixture.panel_binding_sha256,
    )
    return bridge, calls


def _unsafe_role_receipt(
    receipt: RoleCaptureReceipt,
    **changes: object,
) -> RoleCaptureReceipt:
    clone = object.__new__(RoleCaptureReceipt)
    for field in fields(RoleCaptureReceipt):
        object.__setattr__(
            clone,
            field.name,
            changes.get(field.name, getattr(receipt, field.name)),
        )
    return clone


def _rebuild_role_receipt(
    fixture: _FullAFixture,
    *,
    role: str,
    role_manifest: dict[str, object] | None = None,
) -> RoleCaptureReceipt:
    manifest = fixture.role_manifest if role_manifest is None else role_manifest
    return capture._predecessor_capture._build_role_capture_receipt(
        role=role,
        runtime_session_binding_sha256=fixture.runtime_session_binding_sha256,
        panel_binding_sha256=fixture.panel_binding_sha256,
        role_manifest=manifest,
        spools=fixture.all_mode_witnesses,
        batch_example_ids=fixture.batch_example_ids,
        lexical_receipts=fixture.lexical_receipts,
        execution_receipts=fixture.execution_receipts,
        example_receipts=fixture.example_receipts,
        route_evidence=fixture.route_evidence,
        dense_lexical_parity_receipt_sha256=(
            _sha("dense-parity") if role == PREDECESSOR_A_ROLE else None
        ),
        row_count=capture.PRODUCTION_A_ROW_COUNT,
    )


def test_predecessor_A_lowers_exact_mlp_states_and_teacher_velocities() -> None:
    batch = _batch(PREDECESSOR_A_ROLE)
    rows = capture.build_layer_mlp_flow_rows_from_batch(
        batch,
        layer_ordinal=0,
        development_only=True,
    )

    assert rows.role == capture.A_DEVELOPMENT_ROLE
    assert rows.mode_id == "mode.mutation.2"
    assert rows.row_count == 5
    assert rows.row_keys == batch.examples[0].row_keys + batch.examples[1].row_keys
    assert torch.equal(
        rows.chart_states,
        torch.full((5, 640), 2.25, dtype=torch.float32),
    )
    assert torch.equal(
        rows.teacher_velocities,
        torch.full((5, 640), -2.5, dtype=torch.float32),
    )
    assert rows.metadata()["attention_replaced"] is False
    assert rows.metadata()["contains_token_ids"] is False
    assert rows.predecessor_source_role == PREDECESSOR_A_ROLE
    assert (
        rows.source_batch_execution_receipt_sha256
        == batch.execution_receipt_sha256
    )
    assert (
        rows.source_batch_lexical_nll_receipt_sha256
        == batch.lexical_nll_receipt_sha256
    )


def test_capture_catalog_is_exact_ordered_18_layer_axis() -> None:
    catalog = capture.build_layer_multimode_capture_catalog(
        _batch(PREDECESSOR_A_ROLE),
        development_only=True,
    )

    assert len(catalog.layers) == 18
    assert tuple(row.layer_ordinal for row in catalog.layers) == tuple(range(18))
    assert tuple(row.mode_id for row in catalog.layers) == tuple(
        f"mode.mutation.{2 + 2 * layer}" for layer in range(18)
    )
    assert len({row.receipt_sha256 for row in catalog.layers}) == 18
    assert all(row.row_keys == catalog.layers[0].row_keys for row in catalog.layers)


def test_spent_predecessor_B_is_rejected_from_v3_development() -> None:
    with pytest.raises(ValueError, match="spent predecessor B/C"):
        capture.build_layer_mlp_flow_rows_from_batch(
            _batch(PREDECESSOR_B_ROLE),
            layer_ordinal=0,
            development_only=True,
        )
    with pytest.raises(ValueError, match="spent predecessor B/C"):
        capture.build_layer_mlp_flow_rows(
            _batch(PREDECESSOR_B_ROLE),
            layer_ordinal=0,
            development_only=True,
        )
    with pytest.raises(ValueError, match="spent predecessor B/C"):
        capture.build_layer_multimode_capture_catalog(
            _batch(PREDECESSOR_B_ROLE),
            development_only=True,
        )


def test_balanced_weights_equalize_family_example_and_rows() -> None:
    batch = _batch_with_examples(
        PREDECESSOR_A_ROLE,
        (
            _example(family="family.a", example="a1", tokens=3),
            _example(family="family.a", example="a2", tokens=4),
            _example(family="family.b", example="b1", tokens=5),
        ),
    )
    rows = capture.build_layer_mlp_flow_rows(
        batch,
        layer_ordinal=2,
        development_only=True,
    )
    weights = capture.family_example_balanced_row_weights(rows)

    assert weights.dtype == torch.float64
    assert float(weights.sum()) == pytest.approx(1.0)
    assert float(weights[0:2].sum()) == pytest.approx(0.25)
    assert float(weights[2:5].sum()) == pytest.approx(0.25)
    assert float(weights[5:9].sum()) == pytest.approx(0.50)


@pytest.mark.parametrize(
    "role",
    (
        capture.B3_DISCOVERY_ROLE,
        capture.C3_CONFIRMATION_ROLE,
        capture.D3_FINAL_ROLE,
    ),
)
def test_fresh_roles_cannot_be_synthesized_from_bare_examples(
    role: str,
) -> None:
    with pytest.raises(
        capture.FreshRoleCaptureNotAuthorizedError,
        match="capture is not authorized",
    ):
        replace(
            capture.build_layer_mlp_flow_rows(
                _batch(PREDECESSOR_A_ROLE),
                layer_ordinal=0,
                development_only=True,
            ),
            role=role,
            receipt_sha256="",
        )


def test_fresh_role_relabel_fails_even_after_valid_A_construction() -> None:
    rows = capture.build_layer_mlp_flow_rows_from_batch(
        _batch(PREDECESSOR_A_ROLE),
        layer_ordinal=0,
        development_only=True,
    )
    with pytest.raises(
        capture.FreshRoleCaptureNotAuthorizedError,
        match="B3_teacher_forced_layerwise capture is not authorized",
    ):
        replace(rows, role=capture.B3_DISCOVERY_ROLE, receipt_sha256="")


def test_fixed_A_split_uses_only_preregistered_target_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture, "A_FIT_TARGET_TOKEN_COUNTS", (2, 4))
    monkeypatch.setattr(capture, "A_VALIDATION_TARGET_TOKEN_COUNTS", (3, 5))
    examples = tuple(
        _example(
            family=f"family.{family}",
            example=f"f{family}.length{tokens}",
            tokens=tokens,
        )
        for family in range(6)
        for tokens in (2, 4, 3, 5)
    )
    rows = capture.build_layer_mlp_flow_rows(
        _batch_with_examples(PREDECESSOR_A_ROLE, examples),
        layer_ordinal=0,
        development_only=True,
    )

    fit, validation, receipt = capture.fixed_A_development_split_masks(rows)
    assert int(fit.sum()) == sum(
        6 * (value - 1) for value in capture.A_FIT_TARGET_TOKEN_COUNTS
    )
    assert int(validation.sum()) == sum(
        6 * (value - 1) for value in capture.A_VALIDATION_TARGET_TOKEN_COUNTS
    )
    assert not bool((fit & validation).any())
    assert bool((fit | validation).all())
    assert len(receipt) == 64

    partial = capture.build_layer_mlp_flow_rows(
        _batch_with_examples(PREDECESSOR_A_ROLE, examples[:-4]),
        layer_ordinal=0,
        development_only=True,
    )
    with pytest.raises(ValueError, match="did not close"):
        capture.fixed_A_development_split_masks(partial)


def test_tensor_and_receipt_tampering_fail_closed() -> None:
    rows = capture.build_layer_mlp_flow_rows(
        _batch(PREDECESSOR_A_ROLE),
        layer_ordinal=0,
        development_only=True,
    )
    with pytest.raises(ValueError, match="tensor hash drifted"):
        replace(rows, chart_state_sha256=_sha("forged-state"))
    with pytest.raises(ValueError, match="receipt digest drifted"):
        replace(rows, receipt_sha256=_sha("forged-receipt"))
    with pytest.raises(ValueError, match="predecessor A role witness"):
        replace(
            rows,
            predecessor_source_role=PREDECESSOR_B_ROLE,
            receipt_sha256="",
        )
    with pytest.raises(ValueError, match="finite detached contiguous"):
        replace(rows, chart_states=rows.chart_states.to(torch.float64))


def test_single_batch_helpers_require_explicit_development_scope() -> None:
    batch = _batch(PREDECESSOR_A_ROLE)
    with pytest.raises(ValueError, match="development-only"):
        capture.build_layer_mlp_flow_rows(batch, layer_ordinal=0)
    with pytest.raises(ValueError, match="development-only"):
        capture.build_layer_mlp_flow_rows_from_batch(batch, layer_ordinal=0)
    with pytest.raises(ValueError, match="development-only"):
        capture.build_layer_multimode_capture_catalog(batch)
    with pytest.raises(ValueError, match="development-only"):
        capture.build_layer_mlp_flow_rows(
            batch,
            layer_ordinal=0,
            development_only=1,  # type: ignore[arg-type]
        )


def test_full_A_bridge_reopens_sequentially_and_materializes_one_layer(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, calls = _open_full_A_bridge(full_A_fixture, monkeypatch)
    expected_paths = tuple(spool.path for spool in full_A_fixture.mlp_spools)

    assert tuple(calls) == expected_paths
    assert bridge.spool_paths == expected_paths
    assert not hasattr(bridge, "mlp_spools")
    assert bridge.receipt.batch_count == 45
    assert bridge.receipt.example_count == 48
    assert bridge.receipt.row_count == 24_732
    assert tuple(source[1] for source in bridge.receipt.layer_sources) == tuple(
        f"mode.mutation.{2 + 2 * layer}" for layer in range(18)
    )
    metadata = bridge.receipt.metadata()
    assert metadata["captures_normalized_x"] is True
    assert metadata["captures_teacher_velocity_v_T"] is True
    assert metadata["captures_residual_carrier_h"] is False
    assert metadata["captures_output_or_h_plus_v_closure"] is False
    assert metadata["unchanged_decoder_layer_owns_residual_add"] is True
    assert metadata["contains_raw_tensors"] is False
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    serialized = json.loads(
        json.dumps(bridge.receipt.state_dict(), allow_nan=False, sort_keys=True)
    )
    restored = capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
        serialized,
        expected_receipt_sha256=bridge.receipt.receipt_sha256,
    )
    assert restored == bridge.receipt
    assert restored.state_dict() == serialized

    layer_input = bridge.open_layer(2)
    assert bridge.live_full_A_authorized is True
    assert copy.copy(bridge).live_full_A_authorized is False
    assert copy.deepcopy(bridge).live_full_A_authorized is False
    assert layer_input.live_full_A_authorized is True
    assert copy.copy(layer_input).live_full_A_authorized is False
    assert copy.deepcopy(layer_input).live_full_A_authorized is False
    assert calls[-1] == expected_paths[2]
    assert layer_input.mode_id == "mode.mutation.6"
    assert layer_input.row_count == 24_732
    first = next(layer_input.iter_float64_chunks(chunk_size=11))
    assert (first.start_row, first.end_row) == (0, 11)
    assert first.chart_states.shape == (11, 640)
    assert first.chart_states.dtype == torch.float64
    assert first.teacher_velocities.dtype == torch.float64
    assert first.chart_states[0, 0].item() == pytest.approx(2.25)
    assert first.teacher_velocities[-1, -1].item() == pytest.approx(-2.5)

    rows = layer_input.materialize_rows()
    assert rows.row_count == 24_732
    assert rows.chart_states.shape == (24_732, 640)
    assert rows.teacher_velocities.shape == (24_732, 640)
    assert rows.chart_states.dtype == torch.float32
    assert rows.teacher_velocities.dtype == torch.float32
    assert rows.chart_states[0, 0].item() == pytest.approx(2.25)
    assert rows.chart_states[-1, -1].item() == pytest.approx(2.25)
    assert rows.teacher_velocities[0, 0].item() == pytest.approx(-2.5)
    assert rows.source_capture_scope == (
        "production_full_A_RoleCaptureReceipt_spools"
    )
    assert (
        rows.source_role_capture_receipt_sha256
        == full_A_fixture.receipt.receipt_sha256
    )
    assert rows.source_batch_lexical_nll_receipt_sha256 is None
    assert rows.source_batch_execution_receipt_sha256 is None
    assert rows.metadata()["captured_residual_carrier_h"] is False
    assert rows.metadata()["captured_output_or_h_plus_v_closure"] is False
    bridge_arguments = {
        item.name: getattr(bridge, item.name)
        for item in fields(capture.FullALayerMLPSpoolBridge)
        if item.name != "_construction_authority"
    }
    unissued_bridge = capture.FullALayerMLPSpoolBridge(**bridge_arguments)
    with pytest.raises(PermissionError, match="strict role opener"):
        unissued_bridge.open_layer(0)
    with pytest.raises(PermissionError, match="strict role opener"):
        replace(bridge).open_layer(0)
    input_arguments = {
        item.name: getattr(layer_input, item.name)
        for item in fields(capture.LayerMLPFlowSpoolInput)
        if item.name != "_construction_authority"
    }
    unissued_input = capture.LayerMLPFlowSpoolInput(**input_arguments)
    assert unissued_input.live_full_A_authorized is False
    with pytest.raises(PermissionError, match="strict full-A bridge"):
        next(unissued_input.iter_float64_chunks(chunk_size=11))
    replaced_input = replace(layer_input)
    assert replaced_input.live_full_A_authorized is False
    with pytest.raises(PermissionError, match="strict full-A bridge"):
        next(replaced_input.iter_float64_chunks(chunk_size=11))
    with pytest.raises(ValueError, match="row/family catalog drifted"):
        replace(
            bridge,
            family_ids=("forged.family", *bridge.family_ids[1:]),
        )
    with pytest.raises(ValueError, match="example capture receipt"):
        replace(
            layer_input,
            source_capture_receipt_sha256s=(
                "not-a-digest",
                *layer_input.source_capture_receipt_sha256s[1:],
            ),
        )


@pytest.mark.parametrize("role", (PREDECESSOR_B_ROLE, PREDECESSOR_C_ROLE))
def test_full_A_bridge_rejects_spent_predecessor_roles(
    role: str,
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _rebuild_role_receipt(full_A_fixture, role=role)
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    with pytest.raises(ValueError, match="role/session/panel lineage"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=full_A_fixture.role_manifest,
            expected_capture_receipt_sha256=receipt.receipt_sha256,
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


@pytest.mark.parametrize("inventory", ("batches", "examples"))
def test_full_A_bridge_rejects_incomplete_production_inventories(
    inventory: str,
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if inventory == "batches":
        receipt = _unsafe_role_receipt(
            full_A_fixture.receipt,
            lexical_nll_receipt_sha256s=(
                full_A_fixture.receipt.lexical_nll_receipt_sha256s[:-1]
            ),
        )
    else:
        receipt = _unsafe_role_receipt(
            full_A_fixture.receipt,
            example_capture_receipt_sha256s=(
                full_A_fixture.receipt.example_capture_receipt_sha256s[:-1]
            ),
        )
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    with pytest.raises(ValueError, match="production ledger drifted"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=full_A_fixture.role_manifest,
            expected_capture_receipt_sha256=(
                full_A_fixture.receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


def test_full_A_bridge_rejects_spool_order_type_and_duplicate_paths(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    common: dict[str, Any] = {
        "capture_receipt": full_A_fixture.receipt,
        "role_manifest": full_A_fixture.role_manifest,
        "expected_capture_receipt_sha256": (
            full_A_fixture.receipt.receipt_sha256
        ),
        "expected_runtime_session_binding_sha256": (
            full_A_fixture.runtime_session_binding_sha256
        ),
        "expected_role_panel_binding_sha256": (
            full_A_fixture.panel_binding_sha256
        ),
    }
    with pytest.raises(ValueError, match="incomplete or out of order"):
        capture.open_full_A_layer_mlp_spool_bridge(
            **common,
            mlp_spools=tuple(reversed(full_A_fixture.mlp_spools)),
        )
    with pytest.raises(TypeError, match="typed MLP spools"):
        capture.open_full_A_layer_mlp_spool_bridge(
            **common,
            mlp_spools=(
                SimpleNamespace(mode_id="mode.mutation.2"),
                *full_A_fixture.mlp_spools[1:],
            ),
        )
    original = full_A_fixture.mlp_spools[1]
    duplicate_path = _raw_spool(
        layer_ordinal=1,
        row_keys=original.row_keys,
        row_catalog=original.row_catalog_sha256,
        panel_binding_sha256=full_A_fixture.panel_binding_sha256,
        path=full_A_fixture.mlp_spools[0].path,
        artifact_sha256=original.artifact_sha256,
        tensor_catalog_sha256=original.tensor_catalog_sha256,
    )
    with pytest.raises(ValueError, match="paths must be unique"):
        capture.open_full_A_layer_mlp_spool_bridge(
            **common,
            mlp_spools=(
                full_A_fixture.mlp_spools[0],
                duplicate_path,
                *full_A_fixture.mlp_spools[2:],
            ),
        )
    assert calls == []


def test_full_A_bridge_rejects_provided_spool_role_before_reopen(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = full_A_fixture.mlp_spools[0]
    wrong_role = _raw_spool(
        layer_ordinal=0,
        row_keys=original.row_keys,
        row_catalog=original.row_catalog_sha256,
        panel_binding_sha256=full_A_fixture.panel_binding_sha256,
        role=PREDECESSOR_B_ROLE,
        path=original.path,
        artifact_sha256=original.artifact_sha256,
        tensor_catalog_sha256=original.tensor_catalog_sha256,
    )
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    with pytest.raises(ValueError, match="role/panel/row lineage"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=full_A_fixture.receipt,
            mlp_spools=(wrong_role, *full_A_fixture.mlp_spools[1:]),
            role_manifest=full_A_fixture.role_manifest,
            expected_capture_receipt_sha256=(
                full_A_fixture.receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("pin", "message"),
    (
        ("capture", "role/session/panel lineage"),
        ("session", "role/session/panel lineage"),
        ("panel", "role/session/panel lineage"),
    ),
)
def test_full_A_bridge_rejects_wrong_pinned_lineage_hashes(
    pin: str,
    message: str,
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    values = {
        "capture": full_A_fixture.receipt.receipt_sha256,
        "session": full_A_fixture.runtime_session_binding_sha256,
        "panel": full_A_fixture.panel_binding_sha256,
    }
    values[pin] = _sha(f"wrong:{pin}")
    with pytest.raises(ValueError, match=message):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=full_A_fixture.receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=full_A_fixture.role_manifest,
            expected_capture_receipt_sha256=values["capture"],
            expected_runtime_session_binding_sha256=values["session"],
            expected_role_panel_binding_sha256=values["panel"],
        )
    assert calls == []


def test_full_A_bridge_rejects_wrong_manifest_order(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    manifest = {
        **full_A_fixture.role_manifest,
        "ordered_members": tuple(
            reversed(full_A_fixture.role_manifest["ordered_members"])
        ),
    }
    with pytest.raises(ValueError, match="example order differs"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=full_A_fixture.receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=manifest,
            expected_capture_receipt_sha256=(
                full_A_fixture.receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


def test_full_A_bridge_directly_crossbinds_manifest_objective_hashes(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_manifest = {
        **full_A_fixture.role_manifest,
        "ordered_objective_row_catalog_sha256": _sha(
            "coordinated-changed-objective-rows"
        ),
        "score_objective_receipt_sha256": _sha(
            "coordinated-changed-score-objective"
        ),
    }
    coordinated_receipt = _rebuild_role_receipt(
        full_A_fixture,
        role=PREDECESSOR_A_ROLE,
        role_manifest=changed_manifest,
    )
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    with pytest.raises(ValueError, match="differs from frozen manifest"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=coordinated_receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=full_A_fixture.role_manifest,
            expected_capture_receipt_sha256=(
                coordinated_receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


def test_full_A_bridge_manifest_objective_hash_types_are_exact(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DigestSubclass(str):
        pass

    manifest = {
        **full_A_fixture.role_manifest,
        "score_objective_receipt_sha256": _DigestSubclass(
            full_A_fixture.role_manifest["score_objective_receipt_sha256"]
        ),
    }
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    with pytest.raises(TypeError, match="must be exact strings"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=full_A_fixture.receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=manifest,
            expected_capture_receipt_sha256=(
                full_A_fixture.receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


def test_full_A_bridge_rejects_wrong_manifest_role(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {
        **full_A_fixture.role_manifest,
        "role": PREDECESSOR_B_ROLE,
    }
    calls = _install_fake_spool_opener(monkeypatch, full_A_fixture)
    with pytest.raises(ValueError, match="frozen A role manifest"):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=full_A_fixture.receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=manifest,
            expected_capture_receipt_sha256=(
                full_A_fixture.receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == []


@pytest.mark.parametrize(
    "attack",
    (
        "role",
        "panel",
        "artifact",
        "row_catalog",
        "tensor_catalog",
        "rows",
        "semantic",
    ),
)
def test_full_A_bridge_rejects_reopen_lineage_substitutions(
    attack: str,
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = full_A_fixture.mlp_spools[0]
    values: dict[str, object] = {
        "role": PREDECESSOR_A_ROLE,
        "panel_binding_sha256": full_A_fixture.panel_binding_sha256,
        "artifact_sha256": original.artifact_sha256,
        "row_catalog": original.row_catalog_sha256,
        "tensor_catalog_sha256": original.tensor_catalog_sha256,
        "row_keys": original.row_keys,
    }
    if attack == "role":
        values["role"] = PREDECESSOR_B_ROLE
    elif attack == "panel":
        values["panel_binding_sha256"] = _sha("wrong-panel")
    elif attack == "artifact":
        values["artifact_sha256"] = _sha("wrong-artifact")
    elif attack == "row_catalog":
        values["row_catalog"] = _sha("wrong-row-catalog")
    elif attack == "tensor_catalog":
        values["tensor_catalog_sha256"] = _sha("wrong-tensor-catalog")
    elif attack == "rows":
        values["row_keys"] = tuple(reversed(original.row_keys))
    replacement = _raw_spool(
        layer_ordinal=0,
        row_keys=values["row_keys"],  # type: ignore[arg-type]
        row_catalog=values["row_catalog"],  # type: ignore[arg-type]
        panel_binding_sha256=values["panel_binding_sha256"],  # type: ignore[arg-type]
        role=values["role"],  # type: ignore[arg-type]
        path=original.path,
        artifact_sha256=values["artifact_sha256"],  # type: ignore[arg-type]
        tensor_catalog_sha256=values["tensor_catalog_sha256"],  # type: ignore[arg-type]
    )
    if attack == "semantic":
        replacement._store.blocks["context_values"] = {
            "semantic_float64_tensor_sha256": _sha("wrong-context-semantic")
        }
    calls = _install_fake_spool_opener(
        monkeypatch,
        full_A_fixture,
        replacements={original.path: replacement},
    )
    with pytest.raises(
        ValueError,
        match="spool.*lineage|differs from role|semantic lineage",
    ):
        capture.open_full_A_layer_mlp_spool_bridge(
            capture_receipt=full_A_fixture.receipt,
            mlp_spools=full_A_fixture.mlp_spools,
            role_manifest=full_A_fixture.role_manifest,
            expected_capture_receipt_sha256=(
                full_A_fixture.receipt.receipt_sha256
            ),
            expected_runtime_session_binding_sha256=(
                full_A_fixture.runtime_session_binding_sha256
            ),
            expected_role_panel_binding_sha256=(
                full_A_fixture.panel_binding_sha256
            ),
        )
    assert calls == [original.path]


def test_full_A_bridge_receipt_hash_closure_detects_tampering(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _ = _open_full_A_bridge(full_A_fixture, monkeypatch)
    tampered = object.__new__(capture.FullALayerMLPSpoolBridgeReceipt)
    for field in fields(capture.FullALayerMLPSpoolBridgeReceipt):
        value = getattr(bridge.receipt, field.name)
        if field.name == "source_runtime_session_binding_sha256":
            value = _sha("tampered-runtime-session")
        object.__setattr__(tampered, field.name, value)
    with pytest.raises(ValueError, match="receipt digest drifted"):
        tampered.assert_integrity()


def test_full_A_bridge_receipt_state_parser_fails_closed(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _ = _open_full_A_bridge(full_A_fixture, monkeypatch)
    receipt = bridge.receipt
    state = receipt.state_dict()

    extra = {**state, "raw_tensor_payload": []}
    with pytest.raises(ValueError, match="state fields differ"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            extra,
            expected_receipt_sha256=receipt.receipt_sha256,
        )

    tuple_sources = {**state, "layer_sources": tuple(state["layer_sources"])}
    with pytest.raises(TypeError, match="exact 18-row list"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            tuple_sources,
            expected_receipt_sha256=receipt.receipt_sha256,
        )

    inner_tuple = json.loads(json.dumps(state))
    inner_tuple["layer_sources"][0] = tuple(inner_tuple["layer_sources"][0])
    with pytest.raises(TypeError, match="seven-item lists"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            inner_tuple,
            expected_receipt_sha256=receipt.receipt_sha256,
        )

    bool_count = {**state, "batch_count": True}
    with pytest.raises(ValueError, match="fixed field batch_count differs"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            bool_count,
            expected_receipt_sha256=receipt.receipt_sha256,
        )

    false_raw_claim = {**state, "contains_raw_tensors": True}
    with pytest.raises(ValueError, match="contains_raw_tensors differs"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            false_raw_claim,
            expected_receipt_sha256=receipt.receipt_sha256,
        )

    with pytest.raises(ValueError, match="trusted hash pin"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            state,
            expected_receipt_sha256=_sha("wrong-bridge-pin"),
        )

    coordinated = replace(
        receipt,
        source_runtime_session_binding_sha256=_sha("coordinated-session"),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="trusted hash pin"):
        capture.FullALayerMLPSpoolBridgeReceipt.from_state_dict(
            coordinated.state_dict(),
            expected_receipt_sha256=receipt.receipt_sha256,
        )


def test_A_partition_is_shared_across_all_layers_without_tensor_reads(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _ = _open_full_A_bridge(full_A_fixture, monkeypatch)
    reads_before = tuple(
        spool._store.read_calls for spool in full_A_fixture.mlp_spools
    )
    receipt = capture.build_A_development_partition_receipt(
        bridge_receipt=bridge.receipt,
        role_manifest=full_A_fixture.role_manifest,
        expected_bridge_receipt_sha256=bridge.receipt.receipt_sha256,
    )
    reads_after = tuple(
        spool._store.read_calls for spool in full_A_fixture.mlp_spools
    )

    assert reads_after == reads_before
    assert receipt.fit_row_count == 10_548
    assert receipt.validation_row_count == 14_184
    assert receipt.row_count == 24_732
    assert receipt.fit_row_count + receipt.validation_row_count == 24_732
    assert receipt.layer_count == 18
    assert receipt.canonical_layer_mode_axis == tuple(
        (layer, f"mode.mutation.{2 + 2 * layer}") for layer in range(18)
    )
    assert receipt.fit_mask_sha256 != receipt.validation_mask_sha256
    assert receipt.fit_partition_sha256 != receipt.validation_partition_sha256
    assert (
        receipt.source_full_A_bridge_receipt_sha256
        == bridge.receipt.receipt_sha256
    )
    metadata = receipt.metadata()
    assert metadata["fit_validation_masks_disjoint"] is True
    assert metadata["fit_validation_masks_exhaustive"] is True
    assert metadata["layer_independent_partition"] is True
    assert metadata["masks_regenerated_without_tensor_reads"] is True
    assert metadata["contains_raw_tensors"] is False
    serialized = json.loads(json.dumps(receipt.state_dict(), sort_keys=True))
    restored = capture.ADevelopmentPartitionReceipt.from_state_dict(
        serialized,
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    assert restored == receipt
    assert restored.state_dict() == serialized
    repeated = capture.build_A_development_partition_receipt(
        bridge_receipt=bridge.receipt,
        role_manifest=full_A_fixture.role_manifest,
        expected_bridge_receipt_sha256=bridge.receipt.receipt_sha256,
    )
    assert repeated == receipt


def test_A_partition_builder_rejects_manifest_and_bridge_attacks(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _ = _open_full_A_bridge(full_A_fixture, monkeypatch)
    common = {
        "bridge_receipt": bridge.receipt,
        "expected_bridge_receipt_sha256": bridge.receipt.receipt_sha256,
    }
    with pytest.raises(ValueError, match="trusted hash pin"):
        capture.build_A_development_partition_receipt(
            bridge_receipt=bridge.receipt,
            role_manifest=full_A_fixture.role_manifest,
            expected_bridge_receipt_sha256=_sha("wrong-bridge-pin"),
        )
    with pytest.raises(ValueError, match="frozen A role manifest"):
        capture.build_A_development_partition_receipt(
            **common,
            role_manifest={
                **full_A_fixture.role_manifest,
                "role": PREDECESSOR_B_ROLE,
            },
        )
    with pytest.raises(ValueError, match="manifest/bridge lineage"):
        capture.build_A_development_partition_receipt(
            **common,
            role_manifest={
                **full_A_fixture.role_manifest,
                "score_objective_receipt_sha256": _sha("wrong-score"),
            },
        )
    with pytest.raises(TypeError, match="exact 48-row list"):
        capture.build_A_development_partition_receipt(
            **common,
            role_manifest={
                **full_A_fixture.role_manifest,
                "ordered_members": tuple(
                    full_A_fixture.role_manifest["ordered_members"]
                ),
            },
        )
    reordered = list(full_A_fixture.role_manifest["ordered_members"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="row catalog differs"):
        capture.build_A_development_partition_receipt(
            **common,
            role_manifest={
                **full_A_fixture.role_manifest,
                "ordered_members": reordered,
            },
        )


def test_A_partition_receipt_parser_rejects_type_and_rehash_attacks(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _ = _open_full_A_bridge(full_A_fixture, monkeypatch)
    receipt = capture.build_A_development_partition_receipt(
        bridge_receipt=bridge.receipt,
        role_manifest=full_A_fixture.role_manifest,
        expected_bridge_receipt_sha256=bridge.receipt.receipt_sha256,
    )
    state = receipt.state_dict()
    with pytest.raises(ValueError, match="state fields differ"):
        capture.ADevelopmentPartitionReceipt.from_state_dict(
            {**state, "raw_payload": []},
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    with pytest.raises(TypeError, match="layer axis must be a list"):
        capture.ADevelopmentPartitionReceipt.from_state_dict(
            {
                **state,
                "canonical_layer_mode_axis": tuple(
                    state["canonical_layer_mode_axis"]
                ),
            },
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    with pytest.raises(TypeError, match="exact JSON list"):
        capture.ADevelopmentPartitionReceipt.from_state_dict(
            {
                **state,
                "fit_target_token_counts": tuple(
                    state["fit_target_token_counts"]
                ),
            },
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    with pytest.raises(ValueError, match="fixed field fit_row_count differs"):
        capture.ADevelopmentPartitionReceipt.from_state_dict(
            {**state, "fit_row_count": True},
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    coordinated = replace(
        receipt,
        source_full_A_bridge_receipt_sha256=_sha("coordinated-bridge"),
        fit_partition_sha256="",
        validation_partition_sha256="",
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="trusted hash pin"):
        capture.ADevelopmentPartitionReceipt.from_state_dict(
            coordinated.state_dict(),
            expected_receipt_sha256=receipt.receipt_sha256,
        )
