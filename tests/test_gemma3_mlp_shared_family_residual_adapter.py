from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import fisher_graph.gemma3_mlp_shared_family_residual_adapter as compression
from fisher_graph import gemma3_causal_carrier_transport_compiler as transport
from fisher_graph import gemma3_joint_cross_block_chart_executor as joint
from fisher_graph.gemma3_heterogeneous_generator_family_compiler import (
    Gemma3HeterogeneousGeneratorCompilation,
    compile_gemma3_exact_heterogeneous_generator_plan,
)
from fisher_graph.gemma3_layer_agnostic_whole_model_compression_worker import (
    SharedTiedLexicalFactor,
)
from fisher_graph.gemma3_mlp_shared_family_residual_adapter import (
    DENSE_EXACT_CONTROL,
    INDEPENDENT_LOW_RANK_NONSHARING,
    SHARED_CORE_LOW_RANK,
    SHUFFLED_ASSIGNMENT,
    LowRankProjectionDelta,
    MLPFamilyPairManifest,
    build_mlp_family_pair_manifest,
    compile_gemma3_mlp_shared_family_residual_adapter,
    compile_no_refit_residual_assignment_control,
    content_addressed_mlp_shared_family_point_manifest_path,
    initialize_mlp_projection_factors,
    load_mlp_shared_family_content_addressed_point_artifact,
    pinned_mlp_pair_resource_ladder,
    write_mlp_shared_family_content_addressed_point_artifact,
)
from test_gemma3_causal_carrier_transport_compiler import (
    _attention,
    _sequence,
)


_WIDTH = 4
_VOCABULARY = 11
_SETUP_SHA = "9" * 64
_A_FIT_SHA = "8" * 64
_SELECTION_SHA = "7" * 64


class _ToyRMSNorm(nn.Module):
    def __init__(self, width: int, *, offset: float = 0.0) -> None:
        super().__init__()
        self.epsilon = 1.0e-6
        self.weight = nn.Parameter(
            torch.full((width,), offset),
            requires_grad=False,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * (1.0 + self.weight.float())).to(values.dtype)


class _ToyGatedMLP(nn.Module):
    activation_name = "gelu_pytorch_tanh"

    def __init__(self, width: int, *, seed: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for projection in (self.gate_proj, self.up_proj, self.down_proj):
                projection.weight.copy_(
                    torch.randn(width, width, generator=generator) / 5.0
                )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(values), approximate="tanh")
            * self.up_proj(values)
        )


def _toy_exact_executor() -> joint.Gemma3JointCrossBlockChartExecutor:
    schedule = joint.build_joint_cross_block_event_schedule()
    mutations: list[joint.JointGraphGeneratorMutation] = []
    for event in schedule[1:-2]:
        if event.event_kind == "attention_mutation":
            mutations.append(
                joint.JointGraphGeneratorMutation(
                    event_spec=event,
                    compiled_operator=_attention(event),
                    source_checkpoint_coefficients_retained=True,
                )
            )
        else:
            block = (event.event_index - 2) // 2
            mutations.append(
                joint.JointGraphGeneratorMutation(
                    event_spec=event,
                    input_norm=_ToyRMSNorm(_WIDTH, offset=block / 100.0),
                    chart_generator=joint.ExactTokenLocalChartGenerator(
                        operator=_ToyGatedMLP(
                            _WIDTH,
                            seed=41_000 + event.event_index,
                        ),
                        hidden_width=_WIDTH,
                        chart_count=1,
                        generator_rank=_WIDTH,
                    ),
                    output_norm=_ToyRMSNorm(
                        _WIDTH,
                        offset=-block / 200.0,
                    ),
                    source_checkpoint_coefficients_retained=True,
                )
            )
    lexical_weight = torch.arange(
        _VOCABULARY * _WIDTH,
        dtype=torch.float32,
    ).view(_VOCABULARY, _WIDTH) / 100.0
    return joint.Gemma3JointCrossBlockChartExecutor(
        lexical=SharedTiedLexicalFactor(exact_weight=lexical_weight),
        mutations=tuple(mutations),
        transports=(),
        terminal_norm=_ToyRMSNorm(_WIDTH),
        final_logit_softcap=None,
        arm_id=joint.JOINT_EXACT_CONTROL_ARM_ID,
    )


def _exact_compilation() -> Gemma3HeterogeneousGeneratorCompilation:
    exact = _toy_exact_executor()
    carrier = transport.assemble_gemma3_exact_causal_carrier_transport_executor(
        exact_executor=exact,
        source_model_parameter_count=exact.deployed_scalar_count,
        source_model_fingerprint="a" * 64,
        source_independence_audit={"passed": True},
    )
    return compile_gemma3_exact_heterogeneous_generator_plan(
        carrier,
        require_pinned_gemma=False,
    )


def _pair(
    compilation: Gemma3HeterogeneousGeneratorCompilation,
) -> MLPFamilyPairManifest:
    return build_mlp_family_pair_manifest(
        compilation,
        anchor_invocation_id="mutation.2",
        member_invocation_id="mutation.6",
        successor_setup_sha256=_SETUP_SHA,
        a_fit_manifest_sha256=_A_FIT_SHA,
        selection_evidence_sha256=_SELECTION_SHA,
    )


def test_pinned_pair_resource_ladder_has_exact_frozen_counts() -> None:
    rows = pinned_mlp_pair_resource_ladder()
    assert tuple(row["residual_rank"] for row in rows) == (
        None,
        256,
        128,
        64,
        32,
    )
    assert tuple(row["per_residual_adapter_stored_scalars"] for row in rows) == (
        3_933_440,
        2_065_664,
        1_033_472,
        517_376,
        259_328,
    )
    assert tuple(row["candidate_stored_scalars"] for row in rows) == (
        268_098_176,
        266_230_400,
        265_198_208,
        264_682_112,
        264_424_064,
    )
    assert all(row["learned_correction_edge_count"] == 0 for row in rows)


def test_low_rank_projection_delta_matches_materialized_matrix_without_buffer() -> None:
    generator = torch.Generator().manual_seed(120)
    left = torch.randn(5, 2, generator=generator)
    right = torch.randn(2, 3, generator=generator)
    values = torch.randn(7, 3, generator=generator)
    delta = LowRankProjectionDelta(
        left=left,
        right=right,
        output_width=5,
        input_width=3,
        rank=2,
    )
    torch.testing.assert_close(
        delta(values),
        F.linear(values, left @ right),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert set(delta.state_dict()) == {"left", "right"}
    assert delta.stored_scalar_count == 16
    assert delta.metadata()["fused_dense_buffer_present"] is False


def test_pair_manifest_binds_both_protocols_and_rejects_adjacent_selection() -> None:
    compilation = _exact_compilation()
    pair = _pair(compilation)
    assert pair.exact_plan_protocol_sha256 == compilation.plan.protocol_sha256
    assert pair.successor_setup_sha256 == _SETUP_SHA
    assert MLPFamilyPairManifest.from_dict(pair.to_dict()) == pair

    with pytest.raises(ValueError, match="non-adjacent"):
        build_mlp_family_pair_manifest(
            compilation,
            anchor_invocation_id="mutation.2",
            member_invocation_id="mutation.4",
            successor_setup_sha256=_SETUP_SHA,
            a_fit_manifest_sha256=_A_FIT_SHA,
            selection_evidence_sha256=_SELECTION_SHA,
        )

    tampered = copy.deepcopy(pair.to_dict())
    tampered["successor_setup_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="authentication"):
        MLPFamilyPairManifest.from_dict(tampered)


def test_full_rank_factorized_residual_matches_exact_member_algebra() -> None:
    compilation = _exact_compilation()
    pair = _pair(compilation)
    factors = initialize_mlp_projection_factors(
        compilation,
        pair_manifest=pair,
        rank=_WIDTH,
    )
    candidate = compile_gemma3_mlp_shared_family_residual_adapter(
        compilation,
        pair_manifest=pair,
        artifact_kind=SHARED_CORE_LOW_RANK,
        residual_rank=_WIDTH,
        projection_factors=factors,
    )
    exact = compilation.assemble("heterogeneous_full_rank_lift")
    invocations = {item.invocation_id: item for item in exact.manifest.invocations}
    member = exact.atom_for(invocations[pair.member_invocation_id])
    core = candidate.executor.exact_mlp_atom(pair.anchor_invocation_id)
    carrier = torch.randn(2, 3, _WIDTH, generator=torch.Generator().manual_seed(99))
    with torch.inference_mode():
        observed = candidate.executor.adapter.emit_with_core(
            carrier,
            _sequence(),
            core,
        ).value
        expected = member.emit(carrier, _sequence()).value
    torch.testing.assert_close(observed, expected, rtol=2.0e-5, atol=2.0e-6)
    assert candidate.executor.adapter.stored_scalar_count == (
        2 * _WIDTH + 3 * _WIDTH * (2 * _WIDTH)
    )
    assert all("dense" not in name for name in candidate.executor.adapter.state_dict())


def test_dense_control_uses_exact_nonanchor_and_is_bit_identical() -> None:
    compilation = _exact_compilation()
    dense = compile_gemma3_mlp_shared_family_residual_adapter(
        compilation,
        pair_manifest=_pair(compilation),
        artifact_kind=DENSE_EXACT_CONTROL,
    )
    input_ids = torch.tensor([[1, 2, 3], [0, 4, 5]])
    with torch.inference_mode():
        reference = compilation.assemble("heterogeneous_full_rank_lift").execute(
            input_ids,
            _sequence(),
            execution_nonce="reference.dense",
        )
        candidate = dense.executor.execute(
            input_ids,
            _sequence(),
            execution_nonce="candidate.dense",
        )
    assert torch.equal(candidate.output, reference.output)
    assert torch.equal(candidate.normalized_output, reference.normalized_output)
    assert torch.equal(candidate.logits, reference.logits)
    assert dense.candidate_stored_scalars == dense.topology_manifest.source_stored_scalar_count
    metadata = dense.executor.adapter.metadata()
    assert metadata["dense_core_residual_aggregation_live"] is False
    assert metadata["dense_core_residual_relationship_observational"] is True
    assert candidate.receipt.compression_attempted is False


def _toy_primary_point():
    exact = _exact_compilation()
    pair = _pair(exact)
    factors = initialize_mlp_projection_factors(
        exact,
        pair_manifest=pair,
        rank=1,
    )
    primary = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=SHARED_CORE_LOW_RANK,
        residual_rank=1,
        projection_factors=factors,
    )
    return exact, pair, factors, primary


def _point_manifest_path(store: Path, arm_id: str) -> Path:
    return content_addressed_mlp_shared_family_point_manifest_path(
        store,
        arm_id=arm_id,
    )


def _reseal_content_manifest(path: Path, manifest: dict[str, object]) -> None:
    payload = copy.deepcopy(manifest)
    payload.pop("content_point_manifest_sha256", None)
    manifest["content_point_manifest_sha256"] = compression._fingerprint(
        compression._CONTENT_POINT_DOMAIN,
        payload,
    )
    path.write_bytes(compression._canonical_json_bytes(manifest))


def test_content_blob_round_trip_preserves_true_aliases_but_not_equal_values(
    tmp_path: Path,
) -> None:
    shared = torch.arange(8, dtype=torch.float32)
    receipt = compression._write_content_blob(
        state={
            "shared.left": shared,
            "shared.right": shared,
            "equal.nonalias": shared.clone(),
        },
        blob_kind="shared_base",
        blob_root=tmp_path / "blobs",
    )
    loaded, metadata = compression._load_content_blob(
        tmp_path / "blobs" / receipt["blob_file_basename"],
        expected_blob_sha256=receipt["blob_sha256"],
        store_root=tmp_path,
    )
    left_pointer = loaded["shared.left"].untyped_storage().data_ptr()
    assert loaded["shared.right"].untyped_storage().data_ptr() == left_pointer
    assert loaded["equal.nonalias"].untyped_storage().data_ptr() != left_pointer
    assert metadata["storage_alias_groups"] == [
        ["shared.left", "shared.right"]
    ]
    assert metadata["stored_scalar_count"] == 16


def test_content_addressed_reconstruction_preserves_point_identity_and_is_source_free(
    tmp_path: Path,
) -> None:
    exact, _pair_manifest, _factors, primary = _toy_primary_point()
    artifact = primary.export_artifact()
    store = tmp_path / "store"
    receipt = write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        store,
    )
    loaded = load_mlp_shared_family_content_addressed_point_artifact(
        _point_manifest_path(store, primary.arm_id),
        expected_artifact_sha256=receipt["artifact_sha256"],
        expected_manifest_sha256=receipt[
            "content_point_manifest_sha256"
        ],
    )

    assert loaded.artifact_sha256 == artifact.artifact_sha256
    assert loaded.point_manifest["candidate_state_sha256"] == (
        artifact.point_manifest["candidate_state_sha256"]
    )
    assert loaded.serialized_stored_scalar_count == (
        artifact.serialized_stored_scalar_count
    )
    assert loaded.ordered_state_names == artifact.ordered_state_names
    assert loaded.storage_alias_groups == artifact.storage_alias_groups
    for group in loaded.storage_alias_groups:
        pointers = {
            loaded.tensor_state[name].untyped_storage().data_ptr()
            for name in group
        }
        assert len(pointers) == 1

    primary.restore_artifact_(loaded)
    del exact
    with torch.inference_mode():
        execution = primary.executor.execute(
            torch.tensor([[1, 2, 3], [0, 4, 5]]),
            _sequence(),
            execution_nonce="content-addressed.source-free",
        )
    assert execution.receipt.candidate_state_before_sha256 == (
        artifact.point_manifest["candidate_state_sha256"]
    )
    assert execution.receipt.candidate_state_after_sha256 == (
        artifact.point_manifest["candidate_state_sha256"]
    )
    assert execution.receipt.source_projection_call_count == 0
    assert execution.receipt.source_fallback_call_count == 0
    assert receipt["source_free_candidate_runtime"] is True


def test_content_addressed_loader_rejects_missing_bit_flipped_and_extra_blobs(
    tmp_path: Path,
) -> None:
    _exact, _pair_manifest, _factors, primary = _toy_primary_point()
    artifact = primary.export_artifact()

    missing_store = tmp_path / "missing"
    missing_receipt = write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        missing_store,
    )
    missing_blob = (
        missing_store
        / "blobs"
        / missing_receipt["component_blob_receipts"][1]["blob_file_basename"]
    )
    missing_blob.unlink()
    with pytest.raises(ValueError, match="not a regular file"):
        load_mlp_shared_family_content_addressed_point_artifact(
            _point_manifest_path(missing_store, primary.arm_id)
        )

    flipped_store = tmp_path / "bit-flipped"
    flipped_receipt = write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        flipped_store,
    )
    flipped_blob = (
        flipped_store
        / "blobs"
        / flipped_receipt["component_blob_receipts"][1]["blob_file_basename"]
    )
    flipped_payload = torch.load(
        flipped_blob,
        map_location="cpu",
        weights_only=True,
    )
    flipped_name = flipped_payload["ordered_state_names"][0]
    original_tensor = flipped_payload["tensor_state"][flipped_name]
    flipped_flat = original_tensor.reshape(-1).clone().contiguous()
    flipped_bytes = flipped_flat.view(torch.uint8)
    flipped_bytes.view(-1)[0].bitwise_xor_(1)
    flipped_payload["tensor_state"][flipped_name] = flipped_flat.reshape(
        original_tensor.shape
    )
    torch.save(flipped_payload, flipped_blob)
    with pytest.raises(ValueError, match="metadata drifted|authentication"):
        load_mlp_shared_family_content_addressed_point_artifact(
            _point_manifest_path(flipped_store, primary.arm_id)
        )

    extra_store = tmp_path / "extra-reference"
    write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        extra_store,
    )
    extra_manifest_path = _point_manifest_path(extra_store, primary.arm_id)
    extra_manifest = json.loads(extra_manifest_path.read_text(encoding="utf-8"))
    extra_manifest["component_blobs"].append(
        copy.deepcopy(extra_manifest["component_blobs"][-1])
    )
    _reseal_content_manifest(extra_manifest_path, extra_manifest)
    with pytest.raises(ValueError, match="overlap|role inventory"):
        load_mlp_shared_family_content_addressed_point_artifact(
            extra_manifest_path
        )


def test_content_addressed_loader_rejects_manifest_escape_and_symlinks(
    tmp_path: Path,
) -> None:
    _exact, _pair_manifest, _factors, primary = _toy_primary_point()
    artifact = primary.export_artifact()
    store = tmp_path / "store"
    receipt = write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        store,
    )
    manifest_path = _point_manifest_path(store, primary.arm_id)

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["logical_point_materialized_separately"] = False
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="claim drifted|authentication"):
        load_mlp_shared_family_content_addressed_point_artifact(manifest_path)

    traversal_store = tmp_path / "traversal"
    write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        traversal_store,
    )
    traversal_path = _point_manifest_path(traversal_store, primary.arm_id)
    traversal = json.loads(traversal_path.read_text(encoding="utf-8"))
    traversal["component_blobs"][0]["blob_file_basename"] = "../../escape.pt"
    _reseal_content_manifest(traversal_path, traversal)
    with pytest.raises(ValueError, match="filename drifted|path"):
        load_mlp_shared_family_content_addressed_point_artifact(traversal_path)
    with pytest.raises(ValueError, match="point-manifest name|identifier"):
        content_addressed_mlp_shared_family_point_manifest_path(
            traversal_store,
            arm_id="../../escape",
        )

    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(_point_manifest_path(store, primary.arm_id))
    with pytest.raises(ValueError, match="symlink"):
        load_mlp_shared_family_content_addressed_point_artifact(manifest_link)

    symlink_store = tmp_path / "blob-symlink"
    symlink_receipt = write_mlp_shared_family_content_addressed_point_artifact(
        artifact,
        symlink_store,
    )
    blob_path = (
        symlink_store
        / "blobs"
        / symlink_receipt["component_blob_receipts"][1]["blob_file_basename"]
    )
    outside_blob = tmp_path / "outside-blob.pt"
    blob_path.rename(outside_blob)
    blob_path.symlink_to(outside_blob)
    with pytest.raises(ValueError, match="symlink"):
        load_mlp_shared_family_content_addressed_point_artifact(
            _point_manifest_path(symlink_store, primary.arm_id)
        )


def test_content_addressed_loader_rejects_base_overlay_name_collision(
    tmp_path: Path,
) -> None:
    _exact, _pair_manifest, _factors, primary = _toy_primary_point()
    store = tmp_path / "store"
    receipt = write_mlp_shared_family_content_addressed_point_artifact(
        primary.export_artifact(),
        store,
    )
    manifest_path = _point_manifest_path(store, primary.arm_id)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_receipt, adapter_receipt = receipt["component_blob_receipts"]
    base_payload = torch.load(
        store / "blobs" / base_receipt["blob_file_basename"],
        map_location="cpu",
        weights_only=True,
    )
    adapter_payload = torch.load(
        store / "blobs" / adapter_receipt["blob_file_basename"],
        map_location="cpu",
        weights_only=True,
    )
    colliding_state = dict(adapter_payload["tensor_state"])
    base_name = base_payload["ordered_state_names"][0]
    colliding_state[base_name] = base_payload["tensor_state"][base_name]
    colliding_payload = compression._content_blob_payload(
        colliding_state,
        blob_kind="primary_adapter",
    )
    colliding_sha = colliding_payload["blob_sha256"]
    torch.save(
        colliding_payload,
        store / "blobs" / f"{colliding_sha}.pt",
    )
    manifest["component_blobs"][1] = {
        "blob_kind": colliding_payload["blob_kind"],
        "blob_sha256": colliding_sha,
        "blob_file_basename": f"{colliding_sha}.pt",
        "tensor_state_sha256": colliding_payload["tensor_state_sha256"],
        "ordered_state_names": colliding_payload["ordered_state_names"],
        "stored_scalar_count": colliding_payload["stored_scalar_count"],
        "tensor_storage_bytes": colliding_payload["tensor_storage_bytes"],
    }
    _reseal_content_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="state names overlap"):
        load_mlp_shared_family_content_addressed_point_artifact(manifest_path)


def test_content_store_namespaces_identical_independent_fit_and_reuses_shuffled_factors(
    tmp_path: Path,
) -> None:
    exact, pair, identical_factors, primary = _toy_primary_point()
    independent = compile_gemma3_mlp_shared_family_residual_adapter(
        exact,
        pair_manifest=pair,
        artifact_kind=INDEPENDENT_LOW_RANK_NONSHARING,
        residual_rank=1,
        projection_factors=identical_factors,
    )
    shuffled = compile_no_refit_residual_assignment_control(
        exact,
        fitted_compilation=primary,
        assignment_mode=SHUFFLED_ASSIGNMENT,
        shuffle_seed=1_734,
    )
    store = tmp_path / "store"
    primary_receipt = write_mlp_shared_family_content_addressed_point_artifact(
        primary.export_artifact(),
        store,
    )
    independent_receipt = write_mlp_shared_family_content_addressed_point_artifact(
        independent.export_artifact(),
        store,
    )
    shuffled_receipt = write_mlp_shared_family_content_addressed_point_artifact(
        shuffled.export_artifact(),
        store,
    )
    primary_adapter = primary_receipt["component_blob_receipts"][1]
    independent_adapter = independent_receipt["component_blob_receipts"][1]
    shuffled_adapter = shuffled_receipt["component_blob_receipts"][1]

    assert independent_adapter["blob_sha256"] != primary_adapter["blob_sha256"]
    assert independent_adapter["newly_materialized"] is True
    assert independent_adapter["blob_kind"] == "independent_adapter"
    identical_adapter_state = {
        name: tensor
        for name, tensor in primary.export_artifact().tensor_state.items()
        if name.startswith("adapter.")
    }
    identical_primary_payload = compression._content_blob_payload(
        identical_adapter_state,
        blob_kind="primary_adapter",
    )
    identical_independent_payload = compression._content_blob_payload(
        identical_adapter_state,
        blob_kind="independent_adapter",
    )
    assert identical_independent_payload["tensor_state_sha256"] == (
        identical_primary_payload["tensor_state_sha256"]
    )
    assert identical_independent_payload["blob_sha256"] != (
        identical_primary_payload["blob_sha256"]
    )
    assert shuffled.fit_factor_catalog_sha256 == primary.fit_factor_catalog_sha256
    assert shuffled_adapter["blob_sha256"] == primary_adapter["blob_sha256"]
    assert shuffled_adapter["newly_materialized"] is False
    assert shuffled_receipt["newly_materialized_blob_count"] == 0

    loaded_primary = load_mlp_shared_family_content_addressed_point_artifact(
        _point_manifest_path(store, primary.arm_id)
    )
    loaded_shuffled = load_mlp_shared_family_content_addressed_point_artifact(
        _point_manifest_path(store, shuffled.arm_id)
    )
    assert loaded_primary.tensor_state_sha256 == loaded_shuffled.tensor_state_sha256
    isolated_name = loaded_primary.ordered_state_names[0]
    assert torch.equal(
        loaded_primary.tensor_state[isolated_name],
        loaded_shuffled.tensor_state[isolated_name],
    )
    assert (
        loaded_primary.tensor_state[isolated_name].untyped_storage().data_ptr()
        != loaded_shuffled.tensor_state[isolated_name].untyped_storage().data_ptr()
    )
    shuffled_before = loaded_shuffled.tensor_state[isolated_name].clone()
    loaded_primary.tensor_state[isolated_name].add_(1.0)
    assert torch.equal(
        loaded_shuffled.tensor_state[isolated_name],
        shuffled_before,
    )
