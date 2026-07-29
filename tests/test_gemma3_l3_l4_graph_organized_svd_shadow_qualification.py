from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import Tensor

from fisher_graph import (
    gemma3_l3_l4_graph_organized_svd_shadow_qualification as
    qualification_module,
)
from fisher_graph.adapters import Gemma3CausalLMAdapter
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
    derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt,
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    assemble_gemma3_l3_l4_graph_organized_svd_shadow_evidence_observation,
    derive_gemma3_l3_l4_supervised_boundary,
    prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    AuthenticatedOracleSuffixResult,
    Gemma3L3L4GraphOrganizedSVDShadowAccounting,
    Gemma3L3L4GraphOrganizedSVDShadowResult,
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    gemma3_l3_l4_shadow_model_inputs_sha256,
)
from fisher_graph.graph_organized_svd import (
    GraphOrganizedSVDExecutionAccounting,
)


_RUNTIME_RESULT_SHA256 = "ab" * 32
_PROMPT_UTF8 = b"qualification bridge fixture"
_EXAMPLE_ID = gemma3_l3_l4_graph_organized_svd_prompt_sha256(
    _PROMPT_UTF8
)
_FAMILY_ID = "qualification_fixture"
_GEMMA_VOCAB_SIZE = 262_144
_RUNTIME_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-svd-shadow-runtime-tensor:v1\0"
)
_MODEL_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
_PRIVATE_FIVE_PASS = getattr(
    qualification_module,
    "_execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation",
)


class GemmaTokenizer:
    """Small callable stand-in with the frozen public tokenizer identity."""

    name_or_path = "google/gemma-3-270m"
    vocab_size = _GEMMA_VOCAB_SIZE
    model_max_length = 1_000_000_000_000_000_000_000_000_000_000
    padding_side = "left"
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 1
    eos_token = "<eos>"
    init_kwargs = {"_commit_hash": _MODEL_REVISION}

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, prompts: object, **kwargs: object) -> dict[str, Tensor]:
        self.calls.append((prompts, kwargs))
        return {
            "input_ids": torch.tensor([[7, 8, 9, 0]], dtype=torch.int64),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 0]],
                dtype=torch.int64,
            ),
        }


GemmaTokenizer.__module__ = (
    "transformers.models.gemma.tokenization_gemma"
)


def _frozen_tokenizer_provenance(protocol: object) -> dict[str, object]:
    contract = protocol.metadata()["tokenizer"]  # type: ignore[attr-defined]
    fields = (
        "tokenizer_class",
        "name_or_path",
        "configuration_sha256",
        "backend_serialized_bytes",
        "backend_serialized_sha256",
        "canonical_vocab_count",
        "canonical_vocab_sha256",
        "added_token_count",
        "added_tokens_sha256",
        "special_tokens_map_sha256",
        "transformers_version",
        "tokenizers_version",
        "sentencepiece_version",
    )
    return {name: contract[name] for name in fields}


def _backend_identity_sequence(protocol: object) -> list[dict[str, object]]:
    contract = protocol.metadata()["tokenizer"]  # type: ignore[attr-defined]
    return [
        {
            "bytes": contract["backend_serialized_bytes"],
            "sha256": contract["backend_serialized_sha256"],
        },
        {
            "bytes": contract[
                "post_tokenization_backend_serialized_bytes"
            ],
            "sha256": contract[
                "post_tokenization_backend_serialized_sha256"
            ],
        },
    ]


def _runtime_stub() -> Mock:
    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    binding = dict(protocol.metadata()["runtime_binding_contract"])
    binding.pop("artifact_sha256")
    runtime = Mock(spec=Gemma3L3L4GraphOrganizedSVDShadowRuntime)
    runtime.candidate_artifact_sha256 = binding[
        "candidate_logical_artifact_sha256"
    ]
    runtime.basis_payload_sha256 = binding["basis_payload_sha256"]
    runtime.plan_artifact_sha256 = binding["signed_plan_sha256"]
    runtime.source_model_sha256 = binding["source_model_sha256"]
    runtime.live_model_sha256 = binding[
        "factorized_live_execution_sha256"
    ]
    runtime.adapter_execution_sha256 = binding[
        "adapter_execution_fingerprint"
    ]
    runtime.runtime_binding_sha256 = _RUNTIME_RESULT_SHA256
    runtime.source_modes = 64
    runtime.target_modes = 64
    runtime.residual_width = 640

    def validate_result(
        result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    ) -> None:
        result.validate_integrity()
        if result.runtime_binding_sha256 != _RUNTIME_RESULT_SHA256:
            raise ValueError("shadow result belongs to a different runtime")

    def encode(full_delta: Tensor) -> Tensor:
        if not bool(torch.isfinite(full_delta).all()):
            raise AssertionError("qualification failed to sanitize padding")
        return full_delta.to(dtype=torch.float64)[..., :64].clone()

    def decode(modes: Tensor) -> Tensor:
        if not bool(torch.isfinite(modes).all()):
            raise AssertionError("qualification passed nonfinite modes")
        result = torch.zeros(
            (*modes.shape[:-1], 640),
            dtype=torch.float64,
            device=modes.device,
        )
        result[..., :64] = modes
        return result

    runtime.validate_result_binding.side_effect = validate_result
    runtime.encode_target_delta.side_effect = encode
    runtime.decode_target_modal_delta.side_effect = decode
    return runtime


def _model_inputs() -> dict[str, Tensor]:
    return {
        "attention_mask": torch.tensor(
            [[False, True, True, False]],
            dtype=torch.bool,
        ),
        "input_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
        "position_ids": torch.tensor([[0, 8, 9, 10]], dtype=torch.int64),
    }


def _three_pass_result(
    *,
    invalid_padding_nan: bool = True,
    vocab_size: int = 11,
) -> Gemma3L3L4GraphOrganizedSVDShadowResult:
    model_inputs = _model_inputs()
    valid = model_inputs["attention_mask"].clone()
    source = torch.tensor([[False, True, True, False]])
    affected = source.clone()
    reference = torch.zeros((1, 4, 640), dtype=torch.float32)
    authoritative = reference.clone()
    authoritative[:, 1, :64] = 1.0
    authoritative[:, 2, :64] = 2.0
    authoritative[:, 1:3, 64:] = 0.25
    if invalid_padding_nan:
        reference[:, (0, 3)] = torch.nan
        authoritative[:, (0, 3)] = torch.nan
    candidate = authoritative.clone()
    candidate[:, 1:3, 64:] = 0.0
    native_y3 = torch.zeros_like(authoritative)
    clamped_y3 = torch.zeros_like(authoritative)
    if invalid_padding_nan:
        native_y3[:, (0, 3)] = torch.nan
        clamped_y3[:, (0, 3)] = torch.nan
    predicted = torch.zeros((1, 4, 64), dtype=torch.float64)
    predicted[:, 1] = 0.9
    predicted[:, 2] = 1.8
    source_modes = torch.zeros((1, 4, 64), dtype=torch.float64)
    graph = GraphOrganizedSVDExecutionAccounting(
        batch_size=1,
        sequence_length=4,
        valid_source_rows=2,
        valid_target_rows=2,
        admitted_causal_pairs=3,
        active_pack_instances=4,
        active_rank_instances=8,
        interpolated_active_rank_instances=0,
        admitted_active_rank_pairs=12,
        admitted_active_pack_pairs=6,
        source_modes=64,
        target_modes=64,
        source_rank=4,
        pack_count=2,
        lag_count=32,
    )
    accounting = Gemma3L3L4GraphOrganizedSVDShadowAccounting(
        arm="all_on",
        batch_size=1,
        sequence_length=4,
        residual_width=640,
        source_modes=64,
        target_modes=64,
        valid_target_rows=2,
        source_eligible_rows=2,
        target_affected_rows=2,
        target_fallback_rows=2,
        model_forward_count=3,
        graph=graph,
    )
    logits = torch.arange(
        4 * vocab_size,
        dtype=torch.float32,
    ).reshape(1, 4, vocab_size)
    return Gemma3L3L4GraphOrganizedSVDShadowResult(
        arm="all_on",
        authoritative_logits=logits,
        candidate_logits=logits + 0.125,
        authoritative_x4=authoritative,
        candidate_x4=candidate,
        reference_x4=reference,
        native_y3=native_y3,
        clamped_y3=clamped_y3,
        source_modes=source_modes,
        predicted_target_modal_delta=predicted,
        logical_positions=model_inputs["position_ids"].clone(),
        valid_target_mask=valid,
        source_eligible_mask=source,
        target_affected_mask=affected,
        pack_mask=source.unsqueeze(-1).expand(-1, -1, 2).clone(),
        route_scores=torch.zeros((1, 4, 2), dtype=torch.float64),
        runtime_binding_sha256=_RUNTIME_RESULT_SHA256,
        model_inputs_sha256=gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs
        ),
        layer3_reconstruction_max_abs_error=0.0,
        target_dual_reconstruction_max_abs_error=0.0,
        accounting=accounting,
    )


def _runtime_tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_RUNTIME_TENSOR_DOMAIN)
    digest.update(str(value.device).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        str(tuple(int(width) for width in value.shape)).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(canonical.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _oracle_result(
    *,
    runtime: Mock,
    three_pass: Gemma3L3L4GraphOrganizedSVDShadowResult,
    role: str,
    injected_x4: Tensor,
    logit_offset: float,
) -> AuthenticatedOracleSuffixResult:
    assert three_pass.authoritative_logits is not None
    return AuthenticatedOracleSuffixResult(
        role=role,  # type: ignore[arg-type]
        logits=three_pass.authoritative_logits + logit_offset,
        injected_x4_sha256=_runtime_tensor_sha256(injected_x4),
        shadow_result_artifact_sha256=(
            three_pass.result_artifact_sha256
        ),
        runtime_binding_sha256=three_pass.runtime_binding_sha256,
        execution_grid_sha256=three_pass.execution_grid_sha256,
        adapter_execution_sha256=runtime.adapter_execution_sha256,
    )


def _qualification_evidence(
    *,
    vocab_size: int = _GEMMA_VOCAB_SIZE,
    invalid_padding_nan: bool = True,
) -> tuple[
    object,
    Mock,
    dict[str, Tensor],
    Gemma3L3L4GraphOrganizedSVDShadowResult,
    object,
    AuthenticatedOracleSuffixResult,
    AuthenticatedOracleSuffixResult,
]:
    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    runtime = _runtime_stub()
    model_inputs = _model_inputs()
    three_pass = _three_pass_result(
        invalid_padding_nan=invalid_padding_nan,
        vocab_size=vocab_size,
    )
    injections = (
        prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
            runtime,
            three_pass,
        )
    )
    projection = _oracle_result(
        runtime=runtime,
        three_pass=three_pass,
        role="projection_64",
        injected_x4=injections.projection_x4,
        logit_offset=0.25,
    )
    carrier = _oracle_result(
        runtime=runtime,
        three_pass=three_pass,
        role="exact_x4_carrier",
        injected_x4=injections.carrier_x4,
        logit_offset=0.5,
    )
    return (
        protocol,
        runtime,
        model_inputs,
        three_pass,
        injections,
        projection,
        carrier,
    )


def _top_level_evidence() -> tuple[
    object,
    Mock,
    GemmaTokenizer,
    dict[str, Tensor],
    Gemma3L3L4GraphOrganizedSVDShadowResult,
    AuthenticatedOracleSuffixResult,
    AuthenticatedOracleSuffixResult,
]:
    protocol = (
        default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    )
    runtime = _runtime_stub()
    tokenizer = GemmaTokenizer()
    model_inputs = {
        "input_ids": torch.tensor([[7, 8, 9, 0]], dtype=torch.int64),
        "attention_mask": torch.tensor(
            [[True, True, True, False]],
            dtype=torch.bool,
        ),
    }
    base = _three_pass_result(
        invalid_padding_nan=False,
        vocab_size=_GEMMA_VOCAB_SIZE,
    )
    accounting = replace(base.accounting, valid_target_rows=3)
    three_pass = replace(
        base,
        logical_positions=torch.tensor(
            [[7, 8, 9, 10]],
            dtype=torch.int64,
        ),
        valid_target_mask=model_inputs["attention_mask"].clone(),
        model_inputs_sha256=gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs
        ),
        accounting=accounting,
        execution_grid_sha256="",
        result_artifact_sha256="",
    )
    injections = (
        prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
            runtime,
            three_pass,
        )
    )
    projection = _oracle_result(
        runtime=runtime,
        three_pass=three_pass,
        role="projection_64",
        injected_x4=injections.projection_x4,
        logit_offset=0.25,
    )
    carrier = _oracle_result(
        runtime=runtime,
        three_pass=three_pass,
        role="exact_x4_carrier",
        injected_x4=injections.carrier_x4,
        logit_offset=0.5,
    )
    return (
        protocol,
        runtime,
        tokenizer,
        model_inputs,
        three_pass,
        projection,
        carrier,
    )


def _build_observation(
    *,
    protocol: object,
    runtime: Mock,
    model_inputs: dict[str, Tensor],
    three_pass: Gemma3L3L4GraphOrganizedSVDShadowResult,
    projection: AuthenticatedOracleSuffixResult,
    carrier: AuthenticatedOracleSuffixResult,
    prompt_utf8: bytes = _PROMPT_UTF8,
    example_id: str = _EXAMPLE_ID,
    family_id: str = _FAMILY_ID,
):
    with patch.object(
        qualification_module,
        "frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest",
        return_value={_EXAMPLE_ID: _FAMILY_ID},
    ):
        return (
            assemble_gemma3_l3_l4_graph_organized_svd_shadow_evidence_observation(
            protocol=protocol,
            runtime=runtime,
            three_pass_result=three_pass,
            projection_oracle=projection,
            carrier_oracle=carrier,
            model_inputs=model_inputs,
            prompt_utf8=prompt_utf8,
            example_id=example_id,
            family_id=family_id,
            )
        )


def test_supervised_boundary_uses_every_adjacent_valid_pair() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    valid = torch.tensor([[False, True, True, False, True, True]])
    original_ids = input_ids.clone()
    original_valid = valid.clone()

    indices, targets = derive_gemma3_l3_l4_supervised_boundary(
        input_ids,
        valid,
    )

    assert torch.equal(indices, torch.tensor([1, 4]))
    assert torch.equal(targets, torch.tensor([3, 6]))
    assert indices.device.type == targets.device.type == "cpu"
    assert indices.dtype == targets.dtype == torch.int64
    assert torch.equal(input_ids, original_ids)
    assert torch.equal(valid, original_valid)


@pytest.mark.parametrize(
    ("input_ids", "valid"),
    (
        (
            torch.tensor([[1]]),
            torch.tensor([[True]]),
        ),
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[True, True], [True, True]]),
        ),
        (
            torch.tensor([[1, 2]]),
            torch.tensor([[True, False]]),
        ),
    ),
)
def test_supervised_boundary_rejects_nonqualifying_geometry(
    input_ids: Tensor,
    valid: Tensor,
) -> None:
    with pytest.raises(ValueError):
        derive_gemma3_l3_l4_supervised_boundary(input_ids, valid)


def test_oracle_injections_are_semantic_and_preserve_padding() -> None:
    runtime = _runtime_stub()
    three_pass = _three_pass_result()
    authoritative_before = three_pass.authoritative_x4.clone()
    reference_before = three_pass.reference_x4.clone()

    injections = (
        prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
            runtime,
            three_pass,
        )
    )

    affected = three_pass.target_affected_mask
    fallback = ~affected
    torch.testing.assert_close(
        injections.carrier_x4,
        three_pass.authoritative_x4,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        injections.projection_x4[fallback],
        three_pass.authoritative_x4[fallback],
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    expected_projection = three_pass.reference_x4.clone()
    expected_projection[..., :64] += torch.nan_to_num(
        three_pass.authoritative_x4 - three_pass.reference_x4
    )[..., :64]
    assert torch.equal(
        injections.projection_x4[affected],
        expected_projection[affected],
    )
    assert bool(
        torch.isnan(
            injections.source_target_full_width_delta[fallback]
        ).all()
    )
    assert bool(torch.isfinite(injections.source_target_modes).all())
    torch.testing.assert_close(
        three_pass.authoritative_x4,
        authoritative_before,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        three_pass.reference_x4,
        reference_before,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )


def test_oracle_injections_reject_nonfinite_valid_rows() -> None:
    runtime = _runtime_stub()
    three_pass = _three_pass_result(invalid_padding_nan=False)
    authoritative = three_pass.authoritative_x4.clone()
    authoritative[:, 1] = torch.nan
    tampered = replace(
        three_pass,
        authoritative_x4=authoritative,
        result_artifact_sha256="",
    )

    with pytest.raises(ValueError, match="valid source target deltas"):
        prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
            runtime,
            tampered,
        )


def test_builder_creates_exact_authenticated_five_pass_observation() -> None:
    (
        protocol,
        runtime,
        model_inputs,
        three_pass,
        _,
        projection,
        carrier,
    ) = _qualification_evidence()
    inputs_before = {
        name: value.clone() for name, value in model_inputs.items()
    }

    with patch("builtins.open", side_effect=AssertionError("file access")):
        observation = _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
        )

    observation.validate_integrity()
    assert observation.example_id == _EXAMPLE_ID
    assert observation.prompt_identity_sha256 == _EXAMPLE_ID
    assert observation.family_id == _FAMILY_ID
    assert observation.model_inputs_sha256 == (
        three_pass.model_inputs_sha256
    )
    assert observation.shadow_result_artifact_sha256 == (
        three_pass.result_artifact_sha256
    )
    assert observation.execution_grid_sha256 == (
        three_pass.execution_grid_sha256
    )
    assert observation.projection_oracle_artifact_sha256 == (
        projection.artifact_sha256
    )
    assert observation.carrier_oracle_artifact_sha256 == (
        carrier.artifact_sha256
    )
    assert observation.model_forward_count == 5
    assert observation.shadow_model_forward_count == 3
    assert observation.projection_oracle_model_forward_count == 1
    assert observation.carrier_oracle_model_forward_count == 1
    assert torch.equal(
        observation.supervised_boundary_indices,
        torch.tensor([1]),
    )
    assert torch.equal(observation.targets, torch.tensor([2]))
    assert observation.source_logits.shape == (1, _GEMMA_VOCAB_SIZE)
    assert observation.source_logits.dtype == torch.float64
    assert observation.source_logits.device.type == "cpu"
    assert bool(
        torch.isnan(
            observation.source_target_full_width_delta[[0, 3]]
        ).all()
    )
    expected_receipt = (
        derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt(
            protocol_sha256=observation.protocol_sha256,
            assessment_claim_sha256=(
                observation.assessment_claim_sha256
            ),
            runtime_binding_sha256=observation.runtime_binding_sha256,
            example_id=observation.example_id,
            family_id=observation.family_id,
            prompt_identity_sha256=(
                observation.prompt_identity_sha256
            ),
            model_inputs_sha256=observation.model_inputs_sha256,
            shadow_result_artifact_sha256=(
                observation.shadow_result_artifact_sha256
            ),
            execution_grid_sha256=observation.execution_grid_sha256,
            projection_oracle_artifact_sha256=(
                observation.projection_oracle_artifact_sha256
            ),
            projection_injected_x4_sha256=(
                observation.projection_injected_x4_sha256
            ),
            carrier_oracle_artifact_sha256=(
                observation.carrier_oracle_artifact_sha256
            ),
            carrier_injected_x4_sha256=(
                observation.carrier_injected_x4_sha256
            ),
            evidence_payload_sha256=(
                observation.evidence_payload_sha256
            ),
            shadow_model_forward_count=3,
            projection_oracle_model_forward_count=1,
            carrier_oracle_model_forward_count=1,
            projection_oracle_role="projection_64",
            carrier_oracle_role="exact_x4_carrier",
        )
    )
    assert observation.input_provenance_sha256 == (
        expected_receipt["input_provenance_sha256"]
    )
    assert observation.five_pass_receipt_sha256 == (
        expected_receipt["five_pass_receipt_sha256"]
    )
    assert not hasattr(observation, "prompt_utf8")
    assert _PROMPT_UTF8 not in repr(observation.state_dict()).encode()
    for name, before in inputs_before.items():
        assert torch.equal(model_inputs[name], before)


def test_builder_rejects_prompt_identity_and_same_shape_input_swap() -> None:
    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )

    with pytest.raises(ValueError, match="prompt identity differs"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
            prompt_utf8=b"a different prompt",
        )

    substituted = {
        name: value.clone() for name, value in model_inputs.items()
    }
    substituted["input_ids"][0, 2] += 1
    with pytest.raises(ValueError, match="model_inputs SHA-256 differs"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=substituted,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
        )


@pytest.mark.parametrize(
    "field",
    (
        "shadow_result_artifact_sha256",
        "execution_grid_sha256",
        "role",
    ),
)
def test_builder_rejects_oracle_binding_and_role_drift(
    field: str,
) -> None:
    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )
    replacement: dict[str, object] = {
        "artifact_sha256": "",
        field: (
            "exact_x4_carrier"
            if field == "role"
            else "cd" * 32
        ),
    }
    drifted = replace(projection, **replacement)

    with pytest.raises(ValueError, match="binding or role differs"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=drifted,
            carrier=carrier,
        )


def test_builder_rejects_semantically_wrong_projection_injection() -> None:
    evidence = _qualification_evidence()
    (
        protocol,
        runtime,
        model_inputs,
        three_pass,
        injections,
        _,
        carrier,
    ) = evidence
    wrong_x4 = injections.projection_x4.clone()
    wrong_x4[three_pass.target_affected_mask] += 1.0
    wrong_projection = _oracle_result(
        runtime=runtime,
        three_pass=three_pass,
        role="projection_64",
        injected_x4=wrong_x4,
        logit_offset=0.25,
    )

    with pytest.raises(ValueError, match="injected X4 hash mismatch"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=wrong_projection,
            carrier=carrier,
        )


def test_builder_rejects_tampered_or_replayed_oracle_artifacts() -> None:
    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )
    object.__setattr__(projection, "artifact_sha256", "cd" * 32)
    with pytest.raises(ValueError, match="oracle suffix artifact hash"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
        )

    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )
    assert three_pass.candidate_logits is not None
    next_result = replace(
        three_pass,
        candidate_logits=three_pass.candidate_logits + 0.01,
        result_artifact_sha256="",
    )
    with pytest.raises(ValueError, match="binding or role differs"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=next_result,
            projection=projection,
            carrier=carrier,
        )


def test_builder_rejects_receipt_and_evidence_self_attestation() -> None:
    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )
    false_receipt = {
        "input_provenance_sha256": "cd" * 32,
        "five_pass_receipt_sha256": "ef" * 32,
    }

    with (
        patch.object(
            qualification_module,
            "derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt",
            return_value=false_receipt,
        ),
        pytest.raises(ValueError, match="five-pass receipt differs"),
    ):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
        )


def test_builder_rejects_non_all_boundaries_and_wrong_vocab() -> None:
    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )
    with (
        patch.object(
            qualification_module,
            "derive_gemma3_l3_l4_supervised_boundary",
            return_value=(torch.tensor([0]), torch.tensor([1])),
        ),
        pytest.raises(ValueError, match="every adjacent valid"),
    ):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
        )

    small = _qualification_evidence(vocab_size=11)
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        small
    )
    with pytest.raises(ValueError, match="tensor geometry differs"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=three_pass,
            projection=projection,
            carrier=carrier,
        )


def test_builder_rejects_mask_holes_position_gaps_and_affected_nans() -> None:
    evidence = _qualification_evidence()
    protocol, runtime, model_inputs, three_pass, _, projection, carrier = (
        evidence
    )
    hole_inputs = {
        name: value.clone() for name, value in model_inputs.items()
    }
    hole_inputs["attention_mask"] = torch.tensor(
        [[True, False, True, True]],
        dtype=torch.bool,
    )
    hole_valid = hole_inputs["attention_mask"]
    hole_source = torch.tensor([[False, False, True, True]])
    hole_accounting = replace(
        three_pass.accounting,
        valid_target_rows=3,
        source_eligible_rows=2,
        target_affected_rows=2,
    )
    hole_result = replace(
        three_pass,
        valid_target_mask=hole_valid,
        source_eligible_mask=hole_source,
        target_affected_mask=hole_source,
        pack_mask=hole_source.unsqueeze(-1).expand(-1, -1, 2).clone(),
        model_inputs_sha256=gemma3_l3_l4_shadow_model_inputs_sha256(
            hole_inputs
        ),
        accounting=hole_accounting,
        execution_grid_sha256="",
        result_artifact_sha256="",
    )
    with pytest.raises(ValueError, match="contiguous"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=hole_inputs,
            three_pass=hole_result,
            projection=projection,
            carrier=carrier,
        )

    gap_inputs = {
        name: value.clone() for name, value in model_inputs.items()
    }
    gap_inputs["position_ids"][0, 2] = 10
    gap_result = replace(
        three_pass,
        logical_positions=gap_inputs["position_ids"],
        model_inputs_sha256=gemma3_l3_l4_shadow_model_inputs_sha256(
            gap_inputs
        ),
        execution_grid_sha256="",
        result_artifact_sha256="",
    )
    with pytest.raises(ValueError, match="contiguous"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=gap_inputs,
            three_pass=gap_result,
            projection=projection,
            carrier=carrier,
        )

    authoritative = three_pass.authoritative_x4.clone()
    authoritative[:, 1] = torch.nan
    nonfinite = replace(
        three_pass,
        authoritative_x4=authoritative,
        result_artifact_sha256="",
    )
    with pytest.raises(ValueError, match="valid source target deltas"):
        _build_observation(
            protocol=protocol,
            runtime=runtime,
            model_inputs=model_inputs,
            three_pass=nonfinite,
            projection=projection,
            carrier=carrier,
        )


def test_top_level_wrapper_executes_exactly_five_metrics_only_passes() -> None:
    evidence = _top_level_evidence()
    (
        protocol,
        runtime,
        tokenizer,
        model_inputs,
        three_pass,
        projection,
        carrier,
    ) = evidence
    adapter = Mock(spec=Gemma3CausalLMAdapter)

    def execute_shadow(
        adapter_arg: Mock,
        inputs_arg: dict[str, Tensor],
        *,
        arm: str,
    ) -> Gemma3L3L4GraphOrganizedSVDShadowResult:
        assert adapter_arg is adapter
        assert set(inputs_arg) == set(model_inputs)
        for name in model_inputs:
            assert torch.equal(inputs_arg[name], model_inputs[name])
            assert inputs_arg[name].device.type == "cpu"
        assert arm == "all_on"
        for _ in range(3):
            adapter.forward()
        return three_pass

    def execute_oracle(
        adapter_arg: Mock,
        inputs_arg: dict[str, Tensor],
        result_arg: Gemma3L3L4GraphOrganizedSVDShadowResult,
        injected_x4: Tensor,
        *,
        role: str,
    ) -> AuthenticatedOracleSuffixResult:
        assert adapter_arg is adapter
        for name in model_inputs:
            assert torch.equal(inputs_arg[name], model_inputs[name])
        assert result_arg is three_pass
        adapter.forward()
        expected = projection if role == "projection_64" else carrier
        assert _runtime_tensor_sha256(injected_x4) == (
            expected.injected_x4_sha256
        )
        return expected

    runtime.execute_model_shadow.side_effect = execute_shadow
    runtime.execute_oracle_suffix.side_effect = execute_oracle
    with (
        patch.object(
            qualification_module,
            "frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest",
            return_value={_EXAMPLE_ID: _FAMILY_ID},
        ),
        patch.object(
            qualification_module,
            "gemma3_l3_l4_graph_organized_svd_tokenizer_provenance",
            return_value=_frozen_tokenizer_provenance(protocol),
        ),
        patch.object(
            qualification_module,
            "_tokenizer_backend_identity",
            side_effect=_backend_identity_sequence(protocol),
        ),
    ):
        observation = (
            _PRIVATE_FIVE_PASS(
                protocol=protocol,
                runtime=runtime,
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_utf8=_PROMPT_UTF8,
                example_id=_EXAMPLE_ID,
                family_id=_FAMILY_ID,
            )
        )

    observation.validate_integrity()
    assert observation.model_forward_count == 5
    assert observation.candidate_served is False
    assert adapter.forward.call_count == 5
    runtime.execute_model_shadow.assert_called_once()
    assert runtime.execute_oracle_suffix.call_count == 2
    assert [
        call.kwargs["role"]
        for call in runtime.execute_oracle_suffix.call_args_list
    ] == ["projection_64", "exact_x4_carrier"]
    assert tokenizer.padding_side == "right"
    assert tokenizer.calls == [
        (
            [_PROMPT_UTF8.decode("utf-8")],
            {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": 256,
                "add_special_tokens": True,
                "return_attention_mask": True,
            },
        )
    ]


def test_top_level_signature_and_prompt_prevent_input_bundle_swap() -> None:
    public_name = (
        "execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation"
    )
    assert public_name not in qualification_module.__all__
    assert not hasattr(qualification_module, public_name)
    assert (
        "_execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation"
        not in qualification_module.__all__
    )
    signature = inspect.signature(
        _PRIVATE_FIVE_PASS
    )
    assert "tokenizer" in signature.parameters
    assert "model_inputs" not in signature.parameters

    (
        protocol,
        runtime,
        tokenizer,
        model_inputs,
        _,
        _,
        _,
    ) = _top_level_evidence()
    adapter = Mock(spec=Gemma3CausalLMAdapter)
    with pytest.raises(TypeError, match="unexpected keyword.*model_inputs"):
        _PRIVATE_FIVE_PASS(
            protocol=protocol,
            runtime=runtime,
            adapter=adapter,
            tokenizer=tokenizer,
            model_inputs=model_inputs,  # type: ignore[call-arg]
            prompt_utf8=_PROMPT_UTF8,
            example_id=_EXAMPLE_ID,
            family_id=_FAMILY_ID,
        )
    with pytest.raises(ValueError, match="prompt identity differs"):
        _PRIVATE_FIVE_PASS(
            protocol=protocol,
            runtime=runtime,
            adapter=adapter,
            tokenizer=tokenizer,
            prompt_utf8=b"permuted prompt",
            example_id=_EXAMPLE_ID,
            family_id=_FAMILY_ID,
        )
    assert tokenizer.calls == []
    runtime.execute_model_shadow.assert_not_called()
    assert adapter.forward.call_count == 0


@pytest.mark.parametrize(
    "drift",
    (
        "vocab",
        "name",
        "revision",
        "configuration",
        "backend",
        "token_map",
    ),
)
def test_wrong_tokenizer_fails_before_any_model_forward(drift: str) -> None:
    (
        protocol,
        runtime,
        tokenizer,
        _,
        _,
        _,
        _,
    ) = _top_level_evidence()
    adapter = Mock(spec=Gemma3CausalLMAdapter)
    provenance = _frozen_tokenizer_provenance(protocol)
    expected_match = "vocab_size differs"
    if drift == "vocab":
        tokenizer.vocab_size = 11
    elif drift == "name":
        tokenizer.name_or_path = "google/not-the-frozen-gemma"
        expected_match = "name_or_path differs"
    elif drift == "revision":
        tokenizer.init_kwargs = {"_commit_hash": "0" * 40}
        expected_match = "revision differs"
    else:
        field = {
            "configuration": "configuration_sha256",
            "backend": "backend_serialized_sha256",
            "token_map": "canonical_vocab_sha256",
        }[drift]
        provenance[field] = "0" * 64
        expected_match = "implementation differs"
    with (
        patch.object(
            qualification_module,
            "frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest",
            return_value={_EXAMPLE_ID: _FAMILY_ID},
        ),
        patch.object(
            qualification_module,
            "gemma3_l3_l4_graph_organized_svd_tokenizer_provenance",
            return_value=provenance,
        ),
        pytest.raises(ValueError, match=expected_match),
    ):
        _PRIVATE_FIVE_PASS(
            protocol=protocol,
            runtime=runtime,
            adapter=adapter,
            tokenizer=tokenizer,
            prompt_utf8=_PROMPT_UTF8,
            example_id=_EXAMPLE_ID,
            family_id=_FAMILY_ID,
        )
    assert tokenizer.calls == []
    runtime.execute_model_shadow.assert_not_called()
    assert adapter.forward.call_count == 0


@pytest.mark.parametrize(
    ("failure_stage", "expected_forward_count"),
    (
        ("shadow", 2),
        ("projection", 4),
        ("carrier", 5),
    ),
)
def test_top_level_wrapper_is_fail_closed_on_interrupted_pass(
    failure_stage: str,
    expected_forward_count: int,
) -> None:
    evidence = _top_level_evidence()
    (
        protocol,
        runtime,
        tokenizer,
        model_inputs,
        three_pass,
        projection,
        carrier,
    ) = evidence
    adapter = Mock(spec=Gemma3CausalLMAdapter)

    def execute_shadow(*args: object, **kwargs: object):
        del args, kwargs
        if failure_stage == "shadow":
            adapter.forward()
            adapter.forward()
            raise RuntimeError("interrupted shadow")
        for _ in range(3):
            adapter.forward()
        return three_pass

    def execute_oracle(
        *args: object,
        role: str,
        **kwargs: object,
    ) -> AuthenticatedOracleSuffixResult:
        del args, kwargs
        adapter.forward()
        if failure_stage == "projection" and role == "projection_64":
            raise RuntimeError("interrupted projection")
        if failure_stage == "carrier" and role == "exact_x4_carrier":
            raise RuntimeError("interrupted carrier")
        return projection if role == "projection_64" else carrier

    runtime.execute_model_shadow.side_effect = execute_shadow
    runtime.execute_oracle_suffix.side_effect = execute_oracle
    with (
        patch.object(
            qualification_module,
            "frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest",
            return_value={_EXAMPLE_ID: _FAMILY_ID},
        ),
        patch.object(
            qualification_module,
            "gemma3_l3_l4_graph_organized_svd_tokenizer_provenance",
            return_value=_frozen_tokenizer_provenance(protocol),
        ),
        patch.object(
            qualification_module,
            "_tokenizer_backend_identity",
            side_effect=_backend_identity_sequence(protocol),
        ),
        patch.object(
            qualification_module,
            "assemble_gemma3_l3_l4_graph_organized_svd_shadow_evidence_observation",
        ) as builder,
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        _PRIVATE_FIVE_PASS(
            protocol=protocol,
            runtime=runtime,
            adapter=adapter,
            tokenizer=tokenizer,
            prompt_utf8=_PROMPT_UTF8,
            example_id=_EXAMPLE_ID,
            family_id=_FAMILY_ID,
        )

    builder.assert_not_called()
    assert adapter.forward.call_count == expected_forward_count
    assert len(tokenizer.calls) == 1
