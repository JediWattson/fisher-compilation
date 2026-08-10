from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import json
import math

import pytest
import torch

from fisher_graph import gemma3_chart_native_neighbor_experiment as predecessor
from fisher_graph import gemma3_layer_multimode_capture as capture
from fisher_graph import gemma3_layer_multimode_materialization as materialization
from fisher_graph.gemma3_layer_multimode_materialization import (
    LayerMLPFlowMaterializationInventoryReceipt,
    LayerMLPFlowMaterializationReceipt,
    build_layer_mlp_flow_materialization_inventory,
    materialize_layer_mlp_flow_rows,
)
from test_gemma3_layer_multimode_capture import (
    PREDECESSOR_A_ROLE,
    _FullAFixture,
    _install_fake_spool_opener,
    _sha,
    full_A_fixture,
)


def _direct_copy(value):
    return type(value)(
        **{
            descriptor.name: getattr(value, descriptor.name)
            for descriptor in fields(value)
            if descriptor.init
        }
    )


def _lineage() -> tuple[predecessor.RuntimeModeLineage, ...]:
    mode_ids = (
        "mode.lexical.ingress",
        *(f"mode.mutation.{index}" for index in range(1, 37)),
        "mode.lexical.egress",
    )
    return tuple(
        predecessor.RuntimeModeLineage(
            mode_id=mode_id,
            state_fingerprint_sha256=_sha(f"state:{mode_id}"),
            site_placement_sha256=_sha(f"site:{mode_id}"),
            comparison_adapter_receipt_sha256=_sha(f"adapter:{mode_id}"),
        )
        for mode_id in mode_ids
    )


def _qualification() -> predecessor.RuntimeQualificationReceipt:
    lineage = _lineage()
    return predecessor.build_runtime_qualification_receipt(
        schema=(
            "fisher_graph.gemma3_chart_native_neighbor."
            "runtime_qualification.v1"
        ),
        format_version=1,
        model_id="google/gemma-3-270m",
        model_revision="materialization-test-revision",
        source_model_fingerprint_before_compilation_sha256=_sha(
            "source-model"
        ),
        source_model_fingerprint_after_compilation_sha256=_sha(
            "source-model"
        ),
        source_execution_fingerprint_before_compilation_sha256=_sha(
            "source-execution"
        ),
        source_execution_fingerprint_after_compilation_sha256=_sha(
            "source-execution"
        ),
        source_model_mutated_during_compilation=False,
        source_parameter_count=268_098_176,
        hidden_width=640,
        compiled_execution_fingerprint_sha256=_sha("compiled-execution"),
        compiler_receipt_sha256=_sha("compiler"),
        tied_readout_binding_receipt_sha256=_sha("tied-readout"),
        tied_lexical_lineage_sha256=_sha("tied-lexical"),
        final_logit_softcap=4.0,
        mode_lineage=lineage,
        mode_lineage_catalog_sha256=(
            predecessor.runtime_mode_lineage_catalog_sha256(lineage)
        ),
        lexical_kernel_receipt_sha256=_sha("lexical-kernel"),
        streaming_kernel_receipt_sha256=_sha("streaming-kernel"),
        factorized_route_kernel_receipt_sha256=_sha("route-kernel"),
        contains_prompt_text=False,
        contains_token_ids=False,
        contains_raw_activations=False,
        contains_raw_gradients=False,
        contains_logits=False,
    )


def _semantic_hash(
    *,
    value: float,
    row_count: int,
    width: int,
    chunk_size: int,
) -> str:
    hasher = materialization._float64_semantic_hasher((row_count, width))
    for start in range(0, row_count, chunk_size):
        rows = min(chunk_size, row_count - start)
        chunk = torch.full((rows, width), value, dtype=torch.float64)
        hasher.update(materialization._float64_le_bytes(chunk))
    return hasher.hexdigest()


def _role_receipt_with_session(
    fixture: _FullAFixture,
    session_sha256: str,
):
    return capture._predecessor_capture._build_role_capture_receipt(
        role=PREDECESSOR_A_ROLE,
        runtime_session_binding_sha256=session_sha256,
        panel_binding_sha256=fixture.panel_binding_sha256,
        role_manifest=fixture.role_manifest,
        spools=fixture.all_mode_witnesses,
        batch_example_ids=fixture.batch_example_ids,
        lexical_receipts=fixture.lexical_receipts,
        execution_receipts=fixture.execution_receipts,
        example_receipts=fixture.example_receipts,
        route_evidence=fixture.route_evidence,
        dense_lexical_parity_receipt_sha256=_sha("dense-parity"),
        row_count=capture.PRODUCTION_A_ROW_COUNT,
    )


def _authority(
    fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layer_ordinal: int = 0,
):
    qualification = _qualification()
    preflight_sha256 = _sha("preflight")
    session = predecessor.derive_runtime_session_binding_sha256(
        preflight_receipt_sha256=preflight_sha256,
        runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        mode_lineage=qualification.mode_lineage,
    )
    spool = fixture.mlp_spools[layer_ordinal]
    row_count = spool.row_count
    chunk_size = spool.row_chunk_size
    spool._store.blocks["context_values"][
        "semantic_float64_tensor_sha256"
    ] = _semantic_hash(
        value=float(layer_ordinal) + 0.25,
        row_count=row_count,
        width=spool.hidden_width,
        chunk_size=chunk_size,
    )
    spool._store.blocks["comparison_values"][
        "semantic_float64_tensor_sha256"
    ] = _semantic_hash(
        value=-float(layer_ordinal) - 0.5,
        row_count=row_count,
        width=spool.hidden_width,
        chunk_size=chunk_size,
    )
    role_receipt = _role_receipt_with_session(fixture, session)
    _install_fake_spool_opener(monkeypatch, fixture)
    bridge = capture.open_full_A_layer_mlp_spool_bridge(
        capture_receipt=role_receipt,
        mlp_spools=fixture.mlp_spools,
        role_manifest=fixture.role_manifest,
        expected_capture_receipt_sha256=role_receipt.receipt_sha256,
        expected_runtime_session_binding_sha256=session,
        expected_role_panel_binding_sha256=fixture.panel_binding_sha256,
    )
    partition = capture.build_A_development_partition_receipt(
        bridge_receipt=bridge.receipt,
        role_manifest=fixture.role_manifest,
        expected_bridge_receipt_sha256=bridge.receipt.receipt_sha256,
    )
    layer_input = bridge.open_layer(layer_ordinal)
    return (
        layer_input,
        partition,
        qualification,
        preflight_sha256,
    )


def _materialized(
    fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    layer_input, partition, qualification, preflight = _authority(
        fixture,
        monkeypatch,
    )
    result = materialize_layer_mlp_flow_rows(
        layer_input,
        a_development_partition_receipt=partition,
        runtime_qualification_receipt=qualification,
        predecessor_preflight_receipt_sha256=preflight,
    )
    return result, layer_input, partition, qualification, preflight


def _all_live_receipts(
    fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    qualification = _qualification()
    preflight_sha256 = _sha("preflight")
    session = predecessor.derive_runtime_session_binding_sha256(
        preflight_receipt_sha256=preflight_sha256,
        runtime_qualification_receipt_sha256=(
            qualification.qualification_receipt_sha256
        ),
        source_model_fingerprint_sha256=(
            qualification.source_model_fingerprint_after_compilation_sha256
        ),
        source_execution_fingerprint_sha256=(
            qualification.source_execution_fingerprint_after_compilation_sha256
        ),
        compiled_execution_fingerprint_sha256=(
            qualification.compiled_execution_fingerprint_sha256
        ),
        tied_readout_binding_receipt_sha256=(
            qualification.tied_readout_binding_receipt_sha256
        ),
        mode_lineage=qualification.mode_lineage,
    )
    for layer, spool in enumerate(fixture.mlp_spools):
        spool._store.blocks["context_values"][
            "semantic_float64_tensor_sha256"
        ] = _semantic_hash(
            value=float(layer) + 0.25,
            row_count=spool.row_count,
            width=spool.hidden_width,
            chunk_size=spool.row_chunk_size,
        )
        spool._store.blocks["comparison_values"][
            "semantic_float64_tensor_sha256"
        ] = _semantic_hash(
            value=-float(layer) - 0.5,
            row_count=spool.row_count,
            width=spool.hidden_width,
            chunk_size=spool.row_chunk_size,
        )
    role_receipt = _role_receipt_with_session(fixture, session)
    _install_fake_spool_opener(monkeypatch, fixture)
    bridge = capture.open_full_A_layer_mlp_spool_bridge(
        capture_receipt=role_receipt,
        mlp_spools=fixture.mlp_spools,
        role_manifest=fixture.role_manifest,
        expected_capture_receipt_sha256=role_receipt.receipt_sha256,
        expected_runtime_session_binding_sha256=session,
        expected_role_panel_binding_sha256=fixture.panel_binding_sha256,
    )
    partition = capture.build_A_development_partition_receipt(
        bridge_receipt=bridge.receipt,
        role_manifest=fixture.role_manifest,
        expected_bridge_receipt_sha256=bridge.receipt.receipt_sha256,
    )
    receipts = []
    for layer in range(capture.GEMMA3_LAYER_COUNT):
        result = materialize_layer_mlp_flow_rows(
            bridge.open_layer(layer),
            a_development_partition_receipt=partition,
            runtime_qualification_receipt=qualification,
            predecessor_preflight_receipt_sha256=preflight_sha256,
        )
        receipts.append(result.receipt)
    return tuple(receipts), partition, qualification


def _eighteen(
    first: LayerMLPFlowMaterializationReceipt,
) -> tuple[LayerMLPFlowMaterializationReceipt, ...]:
    values = []
    for layer in range(18):
        values.append(
            replace(
                first,
                layer_ordinal=layer,
                mode_id=f"mode.mutation.{2 + 2 * layer}",
                source_spool_artifact_sha256=_sha(f"artifact:{layer}"),
                source_spool_tensor_catalog_sha256=_sha(
                    f"tensor-catalog:{layer}"
                ),
                source_spool_context_float64_semantic_sha256=_sha(
                    f"context-semantic:{layer}"
                ),
                source_spool_comparison_float64_semantic_sha256=_sha(
                    f"comparison-semantic:{layer}"
                ),
                verified_context_float64_semantic_sha256=_sha(
                    f"context-semantic:{layer}"
                ),
                verified_comparison_float64_semantic_sha256=_sha(
                    f"comparison-semantic:{layer}"
                ),
                materialized_chart_state_float32_sha256=_sha(
                    f"x-float32:{layer}"
                ),
                verified_cast_chart_state_float32_sha256=_sha(
                    f"x-float32:{layer}"
                ),
                materialized_teacher_velocity_float32_sha256=_sha(
                    f"v-float32:{layer}"
                ),
                verified_cast_teacher_velocity_float32_sha256=_sha(
                    f"v-float32:{layer}"
                ),
                flow_rows_receipt_sha256=_sha(f"flow-rows:{layer}"),
                receipt_sha256="",
            )
        )
    return tuple(values)


def test_layer_builder_proves_exact_cast_and_binds_runtime_source(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, layer_input, partition, qualification, preflight = _materialized(
        full_A_fixture,
        monkeypatch,
    )
    receipt = result.receipt
    assert result.live_materialization_authorized is True
    assert replace(result).live_materialization_authorized is False
    assert copy.copy(result).live_materialization_authorized is False
    assert copy.deepcopy(result).live_materialization_authorized is False
    assert _direct_copy(result).live_materialization_authorized is False
    assert result.rows.row_count == 24_732
    assert receipt.layer_ordinal == 0
    assert receipt.mode_id == "mode.mutation.2"
    assert receipt.flow_rows_receipt_sha256 == result.rows.receipt_sha256
    assert (
        receipt.materialized_chart_state_float32_sha256
        == result.rows.chart_state_sha256
        == receipt.verified_cast_chart_state_float32_sha256
    )
    assert (
        receipt.materialized_teacher_velocity_float32_sha256
        == result.rows.teacher_velocity_sha256
        == receipt.verified_cast_teacher_velocity_float32_sha256
    )
    assert (
        receipt.source_spool_context_float64_semantic_sha256
        == receipt.verified_context_float64_semantic_sha256
    )
    assert receipt.source_full_A_bridge_receipt_sha256 == (
        layer_input.bridge_receipt.receipt_sha256
    )
    assert receipt.a_development_partition_receipt_sha256 == (
        partition.receipt_sha256
    )
    assert receipt.predecessor_preflight_receipt_sha256 == preflight
    assert receipt.predecessor_runtime_qualification_receipt_sha256 == (
        qualification.qualification_receipt_sha256
    )
    assert receipt.source_model_fingerprint_sha256 == (
        qualification.source_model_fingerprint_after_compilation_sha256
    )
    assert receipt.exact_x_cast_equality_verified is True
    assert receipt.exact_teacher_velocity_cast_equality_verified is True
    assert receipt.live_materialization_authorized is True
    assert copy.copy(receipt).live_materialization_authorized is False
    assert copy.deepcopy(receipt).live_materialization_authorized is False
    assert _direct_copy(receipt).live_materialization_authorized is False
    assert receipt.contains_raw_tensors is False
    state = receipt.state_dict()
    assert not any(isinstance(value, torch.Tensor) for value in state.values())
    restored = LayerMLPFlowMaterializationReceipt.from_state_dict(
        json.loads(json.dumps(state)),
        expected_receipt_sha256=receipt.receipt_sha256,
    )
    assert restored == receipt
    assert restored.live_materialization_authorized is False


def test_layer_builder_rejects_cast_drift_and_wrong_runtime_lineage(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_input, partition, qualification, preflight = _authority(
        full_A_fixture,
        monkeypatch,
    )
    forged_partition = replace(
        partition,
        fit_mask_sha256=_sha("coordinated-forged-fit-mask"),
        fit_partition_sha256="",
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="live spool-derived partition"):
        materialize_layer_mlp_flow_rows(
            layer_input,
            a_development_partition_receipt=forged_partition,
            runtime_qualification_receipt=qualification,
            predecessor_preflight_receipt_sha256=preflight,
        )

    cloned_input = replace(layer_input)
    assert cloned_input.live_full_A_authorized is False
    with pytest.raises(ValueError, match="live strict full-A opener authority"):
        materialize_layer_mlp_flow_rows(
            cloned_input,
            a_development_partition_receipt=partition,
            runtime_qualification_receipt=qualification,
            predecessor_preflight_receipt_sha256=preflight,
        )

    good_rows = layer_input.materialize_rows()
    bad_states = good_rows.chart_states.clone()
    bad_states[0, 0] += 1.0
    bad_rows = replace(
        good_rows,
        chart_states=bad_states,
        chart_state_sha256=capture._tensor_sha256(bad_states),
        receipt_sha256="",
    )
    monkeypatch.setattr(
        capture.LayerMLPFlowSpoolInput,
        "materialize_rows",
        lambda _self: bad_rows,
    )
    with pytest.raises(ValueError, match="materialized x differs"):
        materialize_layer_mlp_flow_rows(
            layer_input,
            a_development_partition_receipt=partition,
            runtime_qualification_receipt=qualification,
            predecessor_preflight_receipt_sha256=preflight,
        )

    with pytest.raises(ValueError, match="does not derive"):
        materialize_layer_mlp_flow_rows(
            layer_input,
            a_development_partition_receipt=partition,
            runtime_qualification_receipt=qualification,
            predecessor_preflight_receipt_sha256=_sha("wrong-preflight"),
        )


def test_layer_parser_rejects_types_raw_fields_and_rehash_attacks(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, *_ = _materialized(full_A_fixture, monkeypatch)
    receipt = result.receipt
    state = receipt.state_dict()
    with pytest.raises(ValueError, match="state fields differ"):
        LayerMLPFlowMaterializationReceipt.from_state_dict(
            {**state, "raw_tensor": []},
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    bool_layer = copy.deepcopy(state)
    bool_layer["layer_ordinal"] = True
    bool_layer["receipt_sha256"] = ""
    with pytest.raises(ValueError, match="layer_ordinal"):
        LayerMLPFlowMaterializationReceipt.from_state_dict(
            bool_layer,
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    int_bool = copy.deepcopy(state)
    int_bool["exact_x_cast_equality_verified"] = 1
    int_bool["receipt_sha256"] = ""
    with pytest.raises(TypeError, match="exact bool"):
        LayerMLPFlowMaterializationReceipt.from_state_dict(
            int_bool,
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    coordinated = replace(
        receipt,
        source_model_fingerprint_sha256=_sha("forged-model"),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="trusted hash pin differs"):
        LayerMLPFlowMaterializationReceipt.from_state_dict(
            coordinated.state_dict(),
            expected_receipt_sha256=receipt.receipt_sha256,
        )
    cloned = replace(receipt)
    assert cloned == receipt
    assert cloned.live_materialization_authorized is False
    with pytest.raises(TypeError, match="init=False"):
        replace(receipt, _construction_authority=object())


def test_ordered_inventory_is_tensor_free_strict_and_source_bound(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers, partition, qualification = _all_live_receipts(
        full_A_fixture,
        monkeypatch,
    )
    inventory = build_layer_mlp_flow_materialization_inventory(
        layers,
        a_development_partition_receipt=partition,
    )
    assert inventory.layer_flow_rows_receipt_sha256s == tuple(
        receipt.flow_rows_receipt_sha256 for receipt in layers
    )
    assert inventory.source_model_fingerprint_sha256 == (
        qualification.source_model_fingerprint_after_compilation_sha256
    )
    assert inventory.a_development_partition_receipt_sha256 == (
        partition.receipt_sha256
    )
    assert inventory.exact_cast_equality_all_layers is True
    state = inventory.state_dict()
    encoded = json.dumps(state)
    assert "torch.Tensor" not in encoded
    assert '"contains_raw_tensors": false' in encoded
    restored = LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
        json.loads(encoded),
        expected_receipt_sha256=inventory.receipt_sha256,
    )
    assert inventory.live_materialization_authorized is True
    assert copy.copy(inventory).live_materialization_authorized is False
    assert copy.deepcopy(inventory).live_materialization_authorized is False
    assert _direct_copy(inventory).live_materialization_authorized is False
    assert restored.live_materialization_authorized is False
    assert all(
        receipt.live_materialization_authorized is False
        for receipt in restored.layer_receipts
    )
    assert restored == inventory
    assert restored.state_dict() == state
    cloned_inventory = replace(inventory)
    assert cloned_inventory == inventory
    assert cloned_inventory.live_materialization_authorized is False
    with pytest.raises(ValueError, match="live builder-authority"):
        build_layer_mlp_flow_materialization_inventory(
            restored.layer_receipts,
            a_development_partition_receipt=partition,
        )
    cloned_layers = _eighteen(layers[0])
    assert all(
        receipt.live_materialization_authorized is False
        for receipt in cloned_layers
    )
    with pytest.raises(ValueError, match="live builder-authority"):
        build_layer_mlp_flow_materialization_inventory(
            cloned_layers,
            a_development_partition_receipt=partition,
        )


def test_inventory_parser_rejects_order_lists_types_and_coordinated_rehash(
    full_A_fixture: _FullAFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers, partition, _qualification_receipt = _all_live_receipts(
        full_A_fixture,
        monkeypatch,
    )
    inventory = build_layer_mlp_flow_materialization_inventory(
        layers,
        a_development_partition_receipt=partition,
    )
    state = inventory.state_dict()
    with pytest.raises(ValueError, match="state fields differ"):
        LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
            {**state, "prompt_text": "forbidden"},
            expected_receipt_sha256=inventory.receipt_sha256,
        )
    tuple_layers = copy.deepcopy(state)
    tuple_layers["layer_receipts"] = tuple(tuple_layers["layer_receipts"])
    with pytest.raises(TypeError, match="18-row list"):
        LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
            tuple_layers,
            expected_receipt_sha256=inventory.receipt_sha256,
        )
    bool_count = copy.deepcopy(state)
    bool_count["layer_count"] = True
    bool_count["receipt_sha256"] = ""
    with pytest.raises(ValueError, match="dimensions drifted"):
        LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
            bool_count,
            expected_receipt_sha256=inventory.receipt_sha256,
        )
    reordered = copy.deepcopy(state)
    reordered["layer_receipts"][0], reordered["layer_receipts"][1] = (
        reordered["layer_receipts"][1],
        reordered["layer_receipts"][0],
    )
    with pytest.raises(ValueError, match="does not recompute|layers 0..17"):
        LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
            reordered,
            expected_receipt_sha256=inventory.receipt_sha256,
        )
    coordinated = replace(
        inventory,
        source_model_fingerprint_sha256=_sha("forged-model"),
        layer_receipts=tuple(
            replace(
                value,
                source_model_fingerprint_sha256=_sha("forged-model"),
                receipt_sha256="",
            )
            for value in inventory.layer_receipts
        ),
        receipt_sha256="",
    )
    with pytest.raises(ValueError, match="trusted hash pin differs"):
        LayerMLPFlowMaterializationInventoryReceipt.from_state_dict(
            coordinated.state_dict(),
            expected_receipt_sha256=inventory.receipt_sha256,
        )


def test_public_surface_has_no_loader_worker_model_or_artifact_api() -> None:
    forbidden = ("load", "open", "save", "write", "model", "worker", "panel")
    assert not any(
        fragment in name.lower()
        for name in materialization.__all__
        for fragment in forbidden
    )
