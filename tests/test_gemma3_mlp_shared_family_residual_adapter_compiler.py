from __future__ import annotations

import copy
import json

import pytest
import torch

import fisher_graph.gemma3_mlp_shared_family_residual_adapter as compression

from fisher_graph.gemma3_mlp_shared_family_residual_adapter import (
    FITTED_ASSIGNMENT,
    INDEPENDENT_LOW_RANK_NONSHARING,
    SHARED_CORE_LOW_RANK,
    SHUFFLED_ASSIGNMENT,
    SWAPPED_ASSIGNMENT,
    ZERO_RESIDUAL_SHARED_CORE,
    compile_gemma3_mlp_shared_family_residual_adapter,
    compile_no_refit_residual_assignment_control,
    content_addressed_mlp_shared_family_point_manifest_path,
    initialize_mlp_projection_factors,
    load_mlp_shared_family_content_addressed_point_artifact,
    load_mlp_shared_family_point_artifact,
    projection_factors_from_compilation,
    validate_mlp_shared_family_point_artifact,
    validate_mlp_shared_family_content_addressed_store_inventory,
    write_mlp_shared_family_content_addressed_point_artifact,
    write_mlp_shared_family_content_addressed_store_inventory,
    write_mlp_shared_family_point_artifact,
)
from fisher_graph.heterogeneous_generator_atom_bank import (
    LIFT_SCOPE_FULL_HETEROGENEOUS,
    module_state_sha256,
)
from test_gemma3_mlp_shared_family_residual_adapter import (
    _WIDTH,
    _exact_compilation,
    _pair,
)
from test_gemma3_causal_carrier_transport_compiler import _sequence


def _primary():
    exact = _exact_compilation()
    pair = _pair(exact)
    factors = initialize_mlp_projection_factors(
        exact,
        pair_manifest=pair,
        rank=1,
    )
    candidate = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=SHARED_CORE_LOW_RANK,
        residual_rank=1,
        projection_factors=factors,
    )
    return exact, pair, factors, candidate


def test_compiler_preserves_topology_and_only_replaces_one_mlp_provider() -> None:
    exact, pair, _factors, candidate = _primary()
    metadata = candidate.metadata()
    assert metadata["invocation_count"] == 39
    assert metadata["edge_occurrence_count"] == 75
    assert metadata["carrier_boundary_count"] == 36
    assert metadata["identity_carry_count"] == 36
    assert metadata["learned_cross_event_correction_count"] == 0
    assert metadata["topology_preserved"] is True
    assert metadata["core_invocation_id"] == pair.anchor_invocation_id
    assert metadata["adapted_invocation_id"] == pair.member_invocation_id
    assert len(candidate.executor.attention_atoms) == 18
    assert len(candidate.executor.exact_mlp_atoms) == 16
    assert candidate.executor.adapter.stored_scalar_count == 2 * _WIDTH + 24
    assert candidate.candidate_stored_scalars == (
        candidate.topology_manifest.source_stored_scalar_count
        - candidate.exact_adapted_atom_stored_scalars
        + candidate.adapter_stored_scalars
    )
    assert candidate.source_independence_audit == {
        "passed": True,
        "candidate_predecessor_storage_overlap_count": 0,
        "candidate_owns_exact_unmodified_components": True,
        "live_source_model_reference_present": False,
    }

    source = exact.assemble(LIFT_SCOPE_FULL_HETEROGENEOUS)
    source_invocations = {
        item.invocation_id: item for item in source.manifest.invocations
    }
    for invocation_id in candidate.executor._attention_ids:
        assert module_state_sha256(candidate.executor.attention_atom(invocation_id)) == (
            module_state_sha256(source.atom_for(source_invocations[invocation_id]))
        )
    for invocation_id in candidate.executor._exact_mlp_ids:
        assert module_state_sha256(candidate.executor.exact_mlp_atom(invocation_id)) == (
            module_state_sha256(source.atom_for(source_invocations[invocation_id]))
        )
    assert module_state_sha256(candidate.executor.lexical_atom) == module_state_sha256(
        source.atom_for(source.manifest.invocations[0])
    )
    assert module_state_sha256(candidate.executor.terminal_runtime) == (
        module_state_sha256(source.terminal_runtime)
    )


def test_candidate_executes_one_adapted_provider_without_graph_corrections() -> None:
    _exact, pair, _factors, candidate = _primary()
    input_ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    with torch.inference_mode():
        execution = candidate.executor.execute(
            input_ids,
            _sequence(),
            execution_nonce="candidate.primary.r1",
        )
    receipt = execution.receipt
    assert len(receipt.invocation_receipts) == 39
    assert receipt.carrier_execution_receipt.aggregate_commit_count == 36
    assert receipt.carrier_execution_receipt.correction_message_count == 0
    assert receipt.learned_cross_event_correction_count == 0
    assert receipt.source_projection_call_count == 0
    assert receipt.source_fallback_call_count == 0
    assert receipt.candidate_state_before_sha256 == (
        receipt.candidate_state_after_sha256
    )
    providers = {
        row.invocation_id: row.provider_kind for row in receipt.invocation_receipts
    }
    assert providers[pair.member_invocation_id] == SHARED_CORE_LOW_RANK
    assert providers[pair.anchor_invocation_id] == "exact_shared_mlp_core"
    assert sum(value == SHARED_CORE_LOW_RANK for value in providers.values()) == 1
    assert torch.isfinite(execution.logits).all()


def test_zero_and_independent_controls_have_honest_matched_resources() -> None:
    exact, pair, _shared_factors, primary = _primary()
    zero = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=ZERO_RESIDUAL_SHARED_CORE,
    )
    assert zero.adapter_stored_scalars == 0
    assert zero.candidate_stored_scalars == (
        zero.topology_manifest.source_stored_scalar_count
        - zero.exact_adapted_atom_stored_scalars
    )
    zero_metadata = zero.executor.adapter.metadata()
    assert zero_metadata["residual_rank"] == 0
    assert zero_metadata["norm_delta_stored_scalars"] == 0
    assert zero_metadata["anchor_norm_plus_delta_live"] is False

    independent_factors = initialize_mlp_projection_factors(
        exact,
        pair_manifest=pair,
        rank=1,
        independent_nonsharing=True,
    )
    independent = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=INDEPENDENT_LOW_RANK_NONSHARING,
        residual_rank=1,
        projection_factors=independent_factors,
    )
    assert independent.adapter_stored_scalars == primary.adapter_stored_scalars
    assert independent.candidate_stored_scalars == primary.candidate_stored_scalars
    assert independent.executor.adapter.metadata()[
        "parameter_matched_to_shared_residual"
    ] is True
    assert independent.control_only is True


def test_swapped_and_seeded_shuffled_controls_reuse_frozen_factors() -> None:
    exact, pair, _factors, primary = _primary()
    original_factors = projection_factors_from_compilation(primary)
    swapped = compile_no_refit_residual_assignment_control(
        exact,
        fitted_compilation=primary,
        assignment_mode=SWAPPED_ASSIGNMENT,
    )
    shuffled_one = compile_no_refit_residual_assignment_control(
        exact,
        fitted_compilation=primary,
        assignment_mode=SHUFFLED_ASSIGNMENT,
        shuffle_seed=71,
    )
    shuffled_two = compile_no_refit_residual_assignment_control(
        exact,
        fitted_compilation=primary,
        assignment_mode=SHUFFLED_ASSIGNMENT,
        shuffle_seed=71,
    )
    assert primary.assignment_mode == FITTED_ASSIGNMENT
    assert swapped.executor.core_invocation_id == pair.anchor_invocation_id
    assert swapped.executor.adapted_invocation_id == pair.anchor_invocation_id
    assert swapped.fit_factor_catalog_sha256 == primary.fit_factor_catalog_sha256
    assert shuffled_one.fit_factor_catalog_sha256 == primary.fit_factor_catalog_sha256
    assert shuffled_one.executor.core_invocation_id == shuffled_two.executor.core_invocation_id
    assert shuffled_one.executor.adapted_invocation_id == (
        shuffled_two.executor.adapted_invocation_id
    )
    assert shuffled_one.shuffle_seed == shuffled_two.shuffle_seed == 71
    assert swapped.candidate_stored_scalars == primary.candidate_stored_scalars
    assert shuffled_one.candidate_stored_scalars == primary.candidate_stored_scalars
    for name, (left, right) in original_factors.items():
        swapped_left, swapped_right = projection_factors_from_compilation(swapped)[name]
        assert torch.equal(left, swapped_left)
        assert torch.equal(right, swapped_right)


def test_source_free_artifact_round_trip_and_strict_tamper_rejection(
    tmp_path,
) -> None:
    _exact, _pair_manifest, _factors, candidate = _primary()
    artifact = candidate.export_artifact()
    validated = validate_mlp_shared_family_point_artifact(
        artifact,
        expected_artifact_sha256=artifact.artifact_sha256,
    )
    assert validated.serialized_stored_scalar_count == candidate.candidate_stored_scalars
    assert validated.point_manifest["source_free_candidate_runtime"] is True
    assert validated.point_manifest["fused_dense_projection_buffer_present"] is False

    path = tmp_path / "candidate-r1.pt"
    loaded = write_mlp_shared_family_point_artifact(artifact, path)
    assert loaded.artifact_sha256 == artifact.artifact_sha256
    assert load_mlp_shared_family_point_artifact(
        path,
        expected_artifact_sha256=artifact.artifact_sha256,
    ).point_manifest == artifact.point_manifest
    with pytest.raises(FileExistsError):
        write_mlp_shared_family_point_artifact(artifact, path)

    tampered = copy.deepcopy(artifact.to_payload())
    first_name = tampered["ordered_state_names"][0]
    tampered["tensor_state"][first_name].view(-1)[0].add_(1.0)
    with pytest.raises(ValueError, match="tensor authentication"):
        validate_mlp_shared_family_point_artifact(tampered)

    stale_claim = copy.deepcopy(artifact.to_payload())
    point = stale_claim["point_manifest"]
    point["candidate_state_sha256"] = "0" * 64
    point_payload = dict(point)
    point_payload.pop("point_manifest_sha256")
    point["point_manifest_sha256"] = compression._fingerprint(
        compression._POINT_DOMAIN,
        point_payload,
    )
    metadata = dict(stale_claim)
    metadata.pop("tensor_state")
    metadata.pop("artifact_sha256")
    stale_claim["artifact_sha256"] = compression._fingerprint(
        compression._ARTIFACT_DOMAIN,
        metadata,
    )
    with pytest.raises(ValueError, match="candidate-state receipt"):
        validate_mlp_shared_family_point_artifact(stale_claim)

    with pytest.raises(ValueError, match="frozen frontier"):
        validate_mlp_shared_family_point_artifact(
            artifact,
            expected_artifact_sha256="0" * 64,
        )


def test_content_addressed_store_materializes_each_point_and_reuses_components(
    tmp_path,
) -> None:
    exact, pair, factors, primary = _primary()
    shuffled = compile_no_refit_residual_assignment_control(
        exact,
        fitted_compilation=primary,
        assignment_mode=SHUFFLED_ASSIGNMENT,
        shuffle_seed=1_734,
    )
    independent = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=INDEPENDENT_LOW_RANK_NONSHARING,
        residual_rank=1,
        projection_factors=factors,
    )
    dense = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=compression.DENSE_EXACT_CONTROL,
    )
    zero = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=ZERO_RESIDUAL_SHARED_CORE,
    )
    store = tmp_path / "point-store"
    compilations = (primary, shuffled, independent, dense, zero)
    receipts = {
        item.arm_id: write_mlp_shared_family_content_addressed_point_artifact(
            item.export_artifact(),
            store,
        )
        for item in compilations
    }

    assert len(list((store / "points").glob("*.json"))) == len(compilations)
    assert len(list((store / "blobs").glob("*.pt"))) == 4
    primary_blobs = receipts[primary.arm_id]["component_blob_receipts"]
    shuffled_blobs = receipts[shuffled.arm_id]["component_blob_receipts"]
    assert [item["blob_sha256"] for item in shuffled_blobs] == [
        item["blob_sha256"] for item in primary_blobs
    ]
    assert shuffled_blobs[1]["blob_kind"] == "primary_adapter"
    assert receipts[shuffled.arm_id]["newly_materialized_blob_count"] == 0
    assert (
        receipts[independent.arm_id]["component_blob_receipts"][1]["blob_kind"]
        == "independent_adapter"
    )
    assert (
        receipts[independent.arm_id]["component_blob_receipts"][1]["blob_sha256"]
        != primary_blobs[1]["blob_sha256"]
    )
    shared_base_shas = {
        receipt["component_blob_receipts"][0]["blob_sha256"]
        for receipt in receipts.values()
    }
    assert len(shared_base_shas) == 1

    ordered_arm_ids = [item.arm_id for item in compilations]
    inventory = write_mlp_shared_family_content_addressed_store_inventory(
        store,
        expected_arm_ids=ordered_arm_ids,
    )
    assert inventory["point_manifest_count"] == len(compilations)
    assert inventory["unique_blob_count"] == 4
    assert inventory["logical_flat_tensor_storage_bytes"] == sum(
        item.candidate_stored_scalars * 4 for item in compilations
    )
    assert inventory["tensor_storage_savings_bytes"] == (
        inventory["logical_flat_tensor_storage_bytes"]
        - inventory["unique_tensor_storage_bytes"]
    )
    assert inventory["serialized_store_file_bytes"] == sum(
        path.stat().st_size for path in store.rglob("*") if path.is_file()
    )
    assert validate_mlp_shared_family_content_addressed_store_inventory(
        store,
        expected_arm_ids=ordered_arm_ids,
        expected_inventory_sha256=inventory["store_inventory_sha256"],
    ) == inventory

    for compilation in compilations:
        receipt = receipts[compilation.arm_id]
        manifest_path = content_addressed_mlp_shared_family_point_manifest_path(
            store,
            arm_id=compilation.arm_id,
        )
        loaded = load_mlp_shared_family_content_addressed_point_artifact(
            manifest_path,
            expected_artifact_sha256=receipt["artifact_sha256"],
            expected_manifest_sha256=receipt[
                "content_point_manifest_sha256"
            ],
        )
        assert loaded.point_manifest == compilation.metadata()
        assert loaded.serialized_stored_scalar_count == (
            compilation.candidate_stored_scalars
        )
        compilation.restore_artifact_(loaded)
        assert compilation.executor.state_sha256 == loaded.point_manifest[
            "candidate_state_sha256"
        ]

    extra = store / "blobs" / f"{'0' * 64}.pt"
    extra.write_bytes(b"not-a-tensor-blob")
    with pytest.raises(ValueError, match="blob inventory"):
        validate_mlp_shared_family_content_addressed_store_inventory(
            store,
            expected_arm_ids=ordered_arm_ids,
        )


def test_content_addressed_store_rejects_manifest_and_blob_tampering(tmp_path) -> None:
    _exact, _pair_manifest, _factors, candidate = _primary()
    store = tmp_path / "point-store"
    receipt = write_mlp_shared_family_content_addressed_point_artifact(
        candidate.export_artifact(),
        store,
    )
    manifest_path = content_addressed_mlp_shared_family_point_manifest_path(
        store,
        arm_id=candidate.arm_id,
    )
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original_manifest)
    manifest["logical_point_materialized_separately"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="authentication"):
        load_mlp_shared_family_content_addressed_point_artifact(manifest_path)

    manifest_path.write_text(original_manifest, encoding="utf-8")
    adapter_ref = receipt["component_blob_receipts"][1]
    blob_path = store / "blobs" / adapter_ref["blob_file_basename"]
    blob = torch.load(blob_path, map_location="cpu", weights_only=True)
    first_name = blob["ordered_state_names"][0]
    blob["tensor_state"][first_name].view(-1)[0].add_(1.0)
    torch.save(blob, blob_path)
    with pytest.raises(ValueError, match="metadata drifted"):
        load_mlp_shared_family_content_addressed_point_artifact(manifest_path)


def test_adapter_blob_hash_is_namespaced_by_scientific_lineage() -> None:
    _exact, _pair_manifest, _factors, candidate = _primary()
    state = {
        name: value
        for name, value in candidate.export_artifact().tensor_state.items()
        if name.startswith("adapter.")
    }
    primary = compression._content_blob_payload(
        state,
        blob_kind="primary_adapter",
    )
    independent = compression._content_blob_payload(
        state,
        blob_kind="independent_adapter",
    )
    assert primary["tensor_state_sha256"] == independent["tensor_state_sha256"]
    assert primary["blob_sha256"] != independent["blob_sha256"]
