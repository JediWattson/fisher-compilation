import copy
from dataclasses import replace
import hashlib
from itertools import chain, islice
from typing import Callable, Iterable, Iterator

import pytest
import torch

from fisher_graph import (
    gemma3_l3_l4_graph_organized_svd_shadow_protocol as protocol_module,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    Gemma3L3L4GraphOrganizedSVDShadowObservation,
    Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
    derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt,
    derive_gemma3_l3_l4_graph_organized_svd_shadow_masks,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
    gemma3_l3_l4_graph_organized_svd_evidence_payload_sha256,
    gemma3_l3_l4_graph_organized_svd_model_inputs_sha256,
)


_GEMMA_VOCAB_SIZE = 262_144
_PRIVATE_EVALUATE = getattr(
    protocol_module,
    "_evaluate_gemma3_l3_l4_graph_organized_svd_shadow",
)


def _sha256(*parts: object) -> str:
    return hashlib.sha256(
        ":".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()


def _source_logits(
    targets: torch.Tensor | None = None,
    *,
    vocab_size: int = _GEMMA_VOCAB_SIZE,
) -> torch.Tensor:
    target_values = (
        torch.tensor([0], dtype=torch.int64)
        if targets is None
        else targets
    )
    result = torch.full(
        (target_values.numel(), vocab_size),
        -8.0,
        dtype=torch.float32,
    )
    for row, target in enumerate(target_values.tolist()):
        result[row, int(target)] = 3.0
        result[row, 1 - int(target)] = 0.0
    return result


def _reversed_logits(
    supervised_tokens: int = 1,
    *,
    vocab_size: int = _GEMMA_VOCAB_SIZE,
) -> torch.Tensor:
    targets = torch.arange(supervised_tokens, dtype=torch.int64) % 2
    result = _source_logits(targets, vocab_size=vocab_size)
    for row, target in enumerate(targets.tolist()):
        result[row, int(target)] = 0.0
        result[row, 1 - int(target)] = 3.0
    return result


def _observation(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    example_id: str,
    family_id: str,
    claim: str | None = None,
    runtime_binding_sha256: str | None = None,
    arm: str = "all_on",
    modal_scale: float = 0.99,
    projection_scale: float = 1.0,
    modal_signal_scale: float = 1.0,
    full_signal_scale: float = 1.0,
    source_logits: torch.Tensor | None = None,
    candidate_logits: torch.Tensor | None = None,
    projection_oracle_logits: torch.Tensor | None = None,
    carrier_oracle_logits: torch.Tensor | None = None,
    logical_positions: torch.Tensor | None = None,
    valid_target_mask: torch.Tensor | None = None,
    supervised_boundary_indices: torch.Tensor | None = None,
    source_eligible_mask: torch.Tensor | None = None,
    prompt_identity_sha256: str | None = None,
    model_inputs_sha256: str | None = None,
    evidence_payload_sha256: str | None = None,
    five_pass_receipt_sha256: str | None = None,
    role: str = "calibration_b_one_shot",
) -> Gemma3L3L4GraphOrganizedSVDShadowObservation:
    positions = (
        torch.tensor([8, 9], dtype=torch.int64)
        if logical_positions is None
        else logical_positions
    )
    valid = (
        torch.ones(positions.numel(), dtype=torch.bool)
        if valid_target_mask is None
        else valid_target_mask
    )
    boundaries = (
        torch.nonzero(valid[:-1] & valid[1:], as_tuple=False).flatten()
        if supervised_boundary_indices is None
        else supervised_boundary_indices
    )
    derived_masks = derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
        positions,
        valid,
        boundaries,
    )
    source_eligible = (
        derived_masks["source_eligible_mask"]
        if source_eligible_mask is None
        else source_eligible_mask
    )
    targets = torch.arange(boundaries.numel(), dtype=torch.int64) % 2
    source_logits_value = (
        _source_logits(targets)
        if source_logits is None
        else source_logits
    )
    candidate_logits_value = (
        source_logits_value.clone()
        if candidate_logits is None
        else candidate_logits
    )
    projection_oracle_logits_value = (
        source_logits_value.clone()
        if projection_oracle_logits is None
        else projection_oracle_logits
    )
    carrier_oracle_logits_value = (
        source_logits_value.clone()
        if carrier_oracle_logits is None
        else carrier_oracle_logits
    )
    rows = positions.numel()
    source_modes = modal_signal_scale * (
        torch.arange(
            1,
            rows * 64 + 1,
            dtype=torch.float64,
        ).reshape(rows, 64)
        / 100.0
    )
    source_full = full_signal_scale * (
        torch.arange(
            1,
            rows * 640 + 1,
            dtype=torch.float64,
        ).reshape(rows, 640)
        / 100.0
    )
    evidence_tensors = {
        "source_logits": source_logits_value,
        "candidate_logits": candidate_logits_value,
        "projection_oracle_logits": projection_oracle_logits_value,
        "carrier_oracle_logits": carrier_oracle_logits_value,
        "targets": targets,
        "source_target_modes": source_modes,
        "candidate_target_modes": modal_scale * source_modes,
        "source_target_full_width_delta": source_full,
        "projection_target_full_width_delta": (
            projection_scale * source_full
        ),
        "logical_positions": positions,
        "supervised_boundary_indices": boundaries,
        "valid_target_mask": valid,
        "source_eligible_mask": source_eligible,
    }
    evidence_sha256 = (
        gemma3_l3_l4_graph_organized_svd_evidence_payload_sha256(
            evidence_tensors
        )
        if evidence_payload_sha256 is None
        else evidence_payload_sha256
    )
    prompt_identity = (
        example_id
        if prompt_identity_sha256 is None
        else prompt_identity_sha256
    )
    input_sha256 = (
        gemma3_l3_l4_graph_organized_svd_model_inputs_sha256(
            {
                "attention_mask": torch.ones(
                    (1, 32),
                    dtype=torch.int64,
                ),
                "input_ids": torch.tensor(
                    list(bytes.fromhex(example_id)),
                    dtype=torch.int64,
                ).unsqueeze(0),
            }
        )
        if model_inputs_sha256 is None
        else model_inputs_sha256
    )
    claim_sha256 = (
        protocol.calibration_b_assessment_claim_sha256()
        if claim is None
        else claim
    )
    runtime_sha256 = (
        protocol.metadata()["runtime_binding_contract"]["artifact_sha256"]
        if runtime_binding_sha256 is None
        else runtime_binding_sha256
    )
    shadow_result_sha256 = _sha256("shadow", example_id)
    execution_grid_sha256 = _sha256("grid", example_id)
    projection_oracle_sha256 = _sha256("projection-oracle", example_id)
    projection_injected_sha256 = _sha256("projection-x4", example_id)
    carrier_oracle_sha256 = _sha256("carrier-oracle", example_id)
    carrier_injected_sha256 = _sha256("carrier-x4", example_id)
    receipt = derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt(
        protocol_sha256=protocol.artifact_sha256,
        assessment_claim_sha256=claim_sha256,
        runtime_binding_sha256=runtime_sha256,
        example_id=example_id,
        family_id=family_id,
        prompt_identity_sha256=prompt_identity,
        model_inputs_sha256=input_sha256,
        shadow_result_artifact_sha256=shadow_result_sha256,
        execution_grid_sha256=execution_grid_sha256,
        projection_oracle_artifact_sha256=projection_oracle_sha256,
        projection_injected_x4_sha256=projection_injected_sha256,
        carrier_oracle_artifact_sha256=carrier_oracle_sha256,
        carrier_injected_x4_sha256=carrier_injected_sha256,
        evidence_payload_sha256=evidence_sha256,
        shadow_model_forward_count=3,
        projection_oracle_model_forward_count=1,
        carrier_oracle_model_forward_count=1,
        projection_oracle_role="projection_64",
        carrier_oracle_role="exact_x4_carrier",
    )
    return Gemma3L3L4GraphOrganizedSVDShadowObservation(
        protocol_sha256=protocol.artifact_sha256,
        assessment_claim_sha256=claim_sha256,
        runtime_binding_sha256=runtime_sha256,
        role=role,
        arm=arm,  # type: ignore[arg-type]
        example_id=example_id,
        family_id=family_id,
        prompt_identity_sha256=prompt_identity,
        model_inputs_sha256=input_sha256,
        input_provenance_sha256=receipt["input_provenance_sha256"],
        shadow_result_artifact_sha256=shadow_result_sha256,
        execution_grid_sha256=execution_grid_sha256,
        projection_oracle_artifact_sha256=projection_oracle_sha256,
        projection_injected_x4_sha256=projection_injected_sha256,
        carrier_oracle_artifact_sha256=carrier_oracle_sha256,
        carrier_injected_x4_sha256=carrier_injected_sha256,
        evidence_payload_sha256=evidence_sha256,
        five_pass_receipt_sha256=(
            receipt["five_pass_receipt_sha256"]
            if five_pass_receipt_sha256 is None
            else five_pass_receipt_sha256
        ),
        **evidence_tensors,
    )


class _ObservationPanel:
    def __init__(
        self,
        entries: Iterable[tuple[str, str]],
        factory: Callable[
            [str, str],
            Gemma3L3L4GraphOrganizedSVDShadowObservation,
        ],
    ) -> None:
        self._entries = tuple(entries)
        self._factory = factory

    def __iter__(
        self,
    ) -> Iterator[Gemma3L3L4GraphOrganizedSVDShadowObservation]:
        for example_id, family_id in self._entries:
            yield self._factory(example_id, family_id)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(
        self,
        index: int | slice,
    ) -> (
        Gemma3L3L4GraphOrganizedSVDShadowObservation
        | "_ObservationPanel"
    ):
        if isinstance(index, slice):
            return _ObservationPanel(
                self._entries[index],
                self._factory,
            )
        example_id, family_id = self._entries[index]
        return self._factory(example_id, family_id)


def _panel(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    modal_scale: float = 0.99,
    projection_scale: float = 1.0,
    modal_scale_by_family: dict[str, float] | None = None,
    projection_scale_by_family: dict[str, float] | None = None,
    modal_signal_scale_by_family: dict[str, float] | None = None,
    full_signal_scale_by_family: dict[str, float] | None = None,
    candidate_logits: torch.Tensor | None = None,
    projection_oracle_logits: torch.Tensor | None = None,
    carrier_oracle_logits: torch.Tensor | None = None,
    logical_positions: torch.Tensor | None = None,
    valid_target_mask: torch.Tensor | None = None,
    supervised_boundary_indices: torch.Tensor | None = None,
) -> tuple[
    _ObservationPanel,
    dict[str, str],
]:
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )

    def factory(
        example_id: str,
        family_id: str,
    ) -> Gemma3L3L4GraphOrganizedSVDShadowObservation:
        return _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            modal_scale=(
                modal_scale
                if modal_scale_by_family is None
                else modal_scale_by_family.get(family_id, modal_scale)
            ),
            projection_scale=(
                projection_scale
                if projection_scale_by_family is None
                else projection_scale_by_family.get(
                    family_id,
                    projection_scale,
                )
            ),
            modal_signal_scale=(
                1.0
                if modal_signal_scale_by_family is None
                else modal_signal_scale_by_family.get(family_id, 1.0)
            ),
            full_signal_scale=(
                1.0
                if full_signal_scale_by_family is None
                else full_signal_scale_by_family.get(family_id, 1.0)
            ),
            candidate_logits=candidate_logits,
            projection_oracle_logits=projection_oracle_logits,
            carrier_oracle_logits=carrier_oracle_logits,
            logical_positions=logical_positions,
            valid_target_mask=valid_target_mask,
            supervised_boundary_indices=supervised_boundary_indices,
        )

    rows = _ObservationPanel(manifest.items(), factory)
    return rows, manifest


def _evaluate(
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    rows: Iterable[Gemma3L3L4GraphOrganizedSVDShadowObservation],
    manifest: dict[str, str],
) -> dict[str, object]:
    return _PRIVATE_EVALUATE(
        protocol,
        rows,
        assessment_claim_sha256=(
            protocol.calibration_b_assessment_claim_sha256()
        ),
        expected_family_by_example=manifest,
    )


def test_protocol_binds_every_identity_role_gate_and_nonclaim() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    metadata = protocol.metadata()

    assert protocol.artifact_sha256 == (
        "7a79087fe4ea90b383bd98f787bede4131c457533a9eef91e6903b2e9c5ea3c8"
    )
    assert protocol.calibration_b_assessment_claim_sha256() == (
        "405719406dc6ee6293e816d76ba20fab93ff9bf7a6a840ec2f80fd000a17bf16"
    )
    assert metadata["model"] == {
        "model_id": "google/gemma-3-270m",
        "requested_revision": (
            "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
        ),
        "resolved_commit": (
            "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
        ),
        "source_model_sha256": (
            "7b083050fa3ae98fde3f193cdf84c91b27ce40a68b3117e9cc38260ca945d4b9"
        ),
        "vocab_size": _GEMMA_VOCAB_SIZE,
        "local_files_only": True,
    }
    assert metadata["tokenizer"] == {
        "tokenizer_class": (
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer"
        ),
        "name_or_path": "google/gemma-3-270m",
        "configuration_sha256": (
            "b02c42b40d0c95c70024c617c8774cde360991e2c949de1d35b51288ded31372"
        ),
        "backend_serialized_bytes": 14_386_244,
        "backend_serialized_sha256": (
            "c1a087240686a7d141101217051f76d5cd4cbe2b6093e3c3553fb26dcc4d0e9a"
        ),
        "post_tokenization_backend_serialized_bytes": 14_386_431,
        "post_tokenization_backend_serialized_sha256": (
            "09afbc35a2fa856bf2baf6f3d140ac7ccddb97179b616b797b5688a96763c189"
        ),
        "canonical_vocab_count": 262_145,
        "canonical_vocab_sha256": (
            "8a2dcfa056d1a48a1cfcb752524bf3a19ff7c996c4f5d4625ad331ca5e0b6eb1"
        ),
        "added_token_count": 6_415,
        "added_tokens_sha256": (
            "7e24459f9c42fe138dfc7ee71cf68a1b1e3b8098d18690c4c28645bed3a5360d"
        ),
        "special_tokens_map_sha256": (
            "a237afa0a3964f4db32b59e1031adcc948cf01b552badfdf5a96092422a19884"
        ),
        "transformers_version": "5.14.1",
        "tokenizers_version": "0.22.2",
        "sentencepiece_version": "0.2.2",
        "vocab_size": _GEMMA_VOCAB_SIZE,
        "model_revision": (
            "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
        ),
        "local_files_only": True,
        "max_length": 256,
        "tokenization_batch_size": 1,
        "device": "cpu",
        "padding_side": "right",
        "padding": True,
        "truncation": True,
        "add_special_tokens": True,
        "return_attention_mask": True,
    }
    graph = metadata["graph_candidate"]
    assert graph["tensor_file_sha256"] == (
        "d77a60532b660160413331ceddbe8d970c2828d53ff5788642250ff3c5d49fa1"
    )
    assert graph["logical_artifact_sha256"] == (
        "b3e011d8067ff3538888851c476fba03c57f4e9f172f923c20fdd90ac0799f84"
    )
    assert graph["factorized_live_execution_sha256"] == (
        "ead03074b87898c9e6c5b068b738420ab0dcf178f07603e885a71964b94ebb7a"
    )
    assert graph["factorized_refit_execution_sha256"] == (
        "911f9869077be1fec2f8610f2f2cbe4c5c6e01a8d632573bec52f2fcc12d1df9"
    )
    assert metadata["prompt_blind_basis"] == {
        "tensor_file_sha256": (
            "359c9659358cbaf97232848a10bdf0e2261d95820ad5effda9bdafeead6a7605"
        ),
        "logical_payload_sha256": (
            "b2217153911436673f2ff7475c658c928112e802f5999619393287d2b0803c01"
        ),
        "target_modal_width": 64,
        "target_full_width": 640,
    }
    corpus = metadata["corpus"]
    assert corpus["prompt_file_sha256"] == (
        "d03a514287afd6b607f4307db58092edfa1c75c2927779b01595eaa0ca106c07"
    )
    assert corpus["family_file_sha256"] == (
        "db8dfafb23c249067f18aaf413b0fe077f4dfc132be666469568e633693d21dd"
    )
    assert corpus["audit_file_sha256"] == (
        "ff11401b61562e02854654657a7c9a46470032e99c43d7697e6dfe1ef536df52"
    )
    assert corpus["calibration_a_fit"] == "development_only"
    assert corpus["calibration_a_guard"] == "development_only"
    assert corpus["calibration_b"] == "unopened_one_shot"
    assert corpus["validation"] == "unopened"
    assert corpus["test"] == "unopened"
    assert corpus["calibration_b_manifest"] == {
        "role": "calibration_b_one_shot",
        "artifact_sha256": (
            "986ee9da505fb056853f4fc7ed4f5eee6e9313f0419f2ca9ebc54e0df8607bdd"
        ),
        "example_count": 96,
        "family_count": 8,
        "derivation": (
            "canonical_sorted_zip_of_audit_prompt_sha256_by_role_"
            "calibration_b_and_family_file_calibration_b"
        ),
        "prompt_file_opened_for_derivation": False,
    }
    assert metadata["arms"] == {
        "primary": "all_on",
        "routed": "locked_disabled_and_rejected",
    }
    assert metadata["runtime_binding_contract"]["artifact_sha256"] == (
        "fe5297e550a96b29cb1cc4d811028c1764df4e583d07970f60997c0112a5b3a6"
    )
    assert metadata["causal_geometry"] == {
        "source_origin_min_inclusive": 8,
        "source_origin_max_inclusive": 40,
        "lag_count": 32,
        "producer_target_affected_mask_trusted": False,
        "producer_affected_supervised_mask_trusted": False,
        "valid_mask_scope": "one_contiguous_span",
        "valid_logical_position_stride": 1,
        "supervised_boundary_index_scope": (
            "all_adjacent_valid_next_token_rows"
        ),
    }
    assert metadata["boundary_gates"] == {
        "pooled_target_modal_relative_error_max": 0.25,
        "pooled_target_modal_cosine_min": 0.95,
        "valid_target_coverage_min": 0.80,
        "worst_family_target_modal_relative_error_max": 0.35,
        "worst_family_target_modal_cosine_min": 0.90,
        "minimum_family_source_modal_signal_l2_norm": 1e-12,
    }
    projection_gates = metadata["projection_capacity_gates"]
    assert projection_gates[
        "pooled_full_width_delta_relative_error_max"
    ] == 0.05
    assert projection_gates["pooled_full_width_delta_cosine_min"] == 0.995
    assert projection_gates[
        "worst_family_full_width_delta_relative_error_max"
    ] == 0.10
    assert projection_gates[
        "worst_family_full_width_delta_cosine_min"
    ] == 0.99
    development = metadata["calibration_a_development_evidence"]
    assert development["selection_or_assessment_eligible"] is False
    assert development["corrected_all_on"]["passed"] is False
    assert development["projection_capacity"]["passed"] is False
    assert development["carrier_completeness"]["passed"] is False
    assert development["deployment_authorized"] is False
    assert development["routing_authorized"] is False
    scope = metadata["scope"]
    assert scope["partial_edge_only"] is True
    assert scope["reference_pass_oracle_fallback_required"] is True
    assert scope["candidate_outputs_must_not_be_served"] is True
    assert scope["standalone_deployment_claim"] is False
    assert scope["full_model_claim"] is False
    assert scope["parameter_reduction_claim"] is False
    assert scope["latency_or_speed_claim"] is False


def test_calibration_b_evaluator_is_private_and_not_exported() -> None:
    public_name = "evaluate_gemma3_l3_l4_graph_organized_svd_shadow"
    assert public_name not in protocol_module.__all__
    assert not hasattr(protocol_module, public_name)
    assert (
        "_evaluate_gemma3_l3_l4_graph_organized_svd_shadow"
        not in protocol_module.__all__
    )


def test_frozen_manifest_is_exact_prompt_blind_and_independent() -> None:
    first = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    second = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )

    assert len(first) == 96
    assert len(set(first)) == 96
    assert len(set(first.values())) == 8
    assert all(len(example_id) == 64 for example_id in first)
    assert all("calibration_b" in family for family in first.values())
    first.clear()
    assert len(second) == 96


def test_runtime_binding_authenticates_exact_all_on_five_pass_bridge() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    frozen = protocol.metadata()["runtime_binding_contract"]
    binding = {
        key: value
        for key, value in frozen.items()
        if key != "artifact_sha256"
    }

    assert protocol.validate_runtime_binding(binding) == (
        "fe5297e550a96b29cb1cc4d811028c1764df4e583d07970f60997c0112a5b3a6"
    )

    routed = {**binding, "routing_enabled": True}
    with pytest.raises(ValueError, match="differs from freeze"):
        protocol.validate_runtime_binding(routed)

    wrong_adapter = {
        **binding,
        "adapter_execution_fingerprint": "0" * 64,
    }
    with pytest.raises(ValueError, match="differs from freeze"):
        protocol.validate_runtime_binding(wrong_adapter)

    unknown = {**binding, "extra": True}
    with pytest.raises(ValueError, match="fields differ"):
        protocol.validate_runtime_binding(unknown)


def test_protocol_round_trip_and_any_freeze_tamper_fail_closed() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    restored = Gemma3L3L4GraphOrganizedSVDShadowProtocol.from_state_dict(
        protocol.state_dict()
    )
    assert restored == protocol
    assert (
        restored.calibration_b_assessment_claim_sha256()
        == protocol.calibration_b_assessment_claim_sha256()
    )

    changed = copy.deepcopy(protocol.state_dict())
    changed["graph_candidate"]["factorized_refit_execution_sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="differs from the freeze"):
        Gemma3L3L4GraphOrganizedSVDShadowProtocol.from_state_dict(changed)

    changed_hash = copy.deepcopy(protocol.state_dict())
    changed_hash["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        Gemma3L3L4GraphOrganizedSVDShadowProtocol.from_state_dict(changed_hash)

    unknown = copy.deepcopy(protocol.state_dict())
    unknown["extra"] = True
    with pytest.raises(ValueError, match="fields differ"):
        Gemma3L3L4GraphOrganizedSVDShadowProtocol.from_state_dict(unknown)


def test_mask_derivation_uses_frozen_origins_lags_and_boundaries() -> None:
    positions = torch.arange(6, 12, dtype=torch.int64)
    valid = torch.ones(6, dtype=torch.bool)
    boundaries = torch.arange(5, dtype=torch.int64)

    masks = derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
        positions,
        valid,
        boundaries,
    )

    assert masks["source_eligible_mask"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert masks["target_affected_mask"].tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert masks["affected_supervised_mask"].tolist() == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_causal_geometry_rejects_boundary_subsets_and_sequence_gaps() -> None:
    with pytest.raises(ValueError, match="equal every adjacent valid"):
        derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
            torch.tensor([8, 9, 10], dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([0], dtype=torch.int64),
        )

    with pytest.raises(ValueError, match="logical positions.*contiguous"):
        derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
            torch.tensor([8, 10, 11], dtype=torch.int64),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([0, 1], dtype=torch.int64),
        )

    with pytest.raises(ValueError, match="valid mask.*contiguous span"):
        derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
            torch.tensor([8, 99, 9], dtype=torch.int64),
            torch.tensor([True, False, True]),
            torch.tensor([0], dtype=torch.int64),
        )


def test_complete_passing_panel_only_qualifies_partial_shadow() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    rows, manifest = _panel(protocol=protocol)

    report = _evaluate(protocol, rows, manifest)

    assert report["assessment_claim_sha256"] == (
        protocol.calibration_b_assessment_claim_sha256()
    )
    assert report["assessment_claim_identity"]["role"] == (
        "calibration_b_one_shot"
    )
    assert report["assessment_claim_identity"]["tokenizer"] == (
        protocol.metadata()["tokenizer"]
    )
    assert report["manifest"] == {
        "role": "calibration_b_one_shot",
        "example_identity": "prompt_sha256_only",
        "artifact_sha256": (
            "986ee9da505fb056853f4fc7ed4f5eee6e9313f0419f2ca9ebc54e0df8607bdd"
        ),
        "example_count": 96,
        "family_count": 8,
        "complete": True,
        "matches_frozen_role": True,
        "derivation": (
            "canonical_sorted_zip_of_audit_prompt_sha256_by_role_"
            "calibration_b_and_family_file_calibration_b"
        ),
        "prompt_file_opened_by_evaluator": False,
    }
    assert report["all_on"]["observation_count"] == 96
    assert report["all_on"]["behavioral"]["aggregate"][
        "example_count"
    ] == 96
    assert report["all_on"]["behavioral"]["aggregate"][
        "supervised_tokens"
    ] == 96
    boundary = report["all_on"]["boundary"]
    assert boundary["valid_target_coverage"] == 1.0
    assert boundary["pooled_target_modal_relative_error"] == pytest.approx(
        0.01
    )
    assert boundary["pooled_target_modal_cosine"] == pytest.approx(1.0)
    assert boundary["worst_family_target_modal_relative_error"] == (
        pytest.approx(0.01)
    )
    assert boundary["worst_family_target_modal_cosine"] == pytest.approx(
        1.0
    )
    assert boundary["gates"]["passed"] is True
    projection = report["all_on"]["projection_capacity"]
    assert projection["pooled_full_width_delta_relative_error"] == 0.0
    assert projection["pooled_full_width_delta_cosine"] == pytest.approx(1.0)
    assert projection["worst_family_full_width_delta_relative_error"] == 0.0
    assert projection["gates"]["passed"] is True
    assert report["all_on"]["carrier_completeness"]["gates"]["passed"] is True
    assert report["all_on"]["passed"] is True
    assert report["routed"] == {
        "allowed": False,
        "evaluated": False,
        "reason": "locked_protocol_all_on_only",
    }
    assert report["authorization"] == {
        "partial_shadow_qualified": True,
        "partial_shadow_scope": "partial_edge_reference_oracle_shadow",
        "all_on_passed": True,
        "deployment_authorized": False,
        "deployment_scope": "none",
        "routing_authorized": False,
        "routing_qualification_available": False,
        "non_authorization_reason": (
            "reference_oracle_required_and_candidate_outputs_metrics_only"
        ),
        "standalone_deployment_authorized": False,
        "full_model_deployment_authorized": False,
    }


def test_one_example_manifest_and_arbitrary_claim_cannot_qualify() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    rows, manifest = _panel(protocol=protocol)
    one = {rows[0].example_id: rows[0].family_id}

    with pytest.raises(ValueError, match="exact full frozen"):
        _PRIVATE_EVALUATE(
            protocol,
            rows[:1],
            assessment_claim_sha256=(
                protocol.calibration_b_assessment_claim_sha256()
            ),
            expected_family_by_example=one,
        )

    with pytest.raises(ValueError, match="claim differs"):
        _PRIVATE_EVALUATE(
            protocol,
            rows,
            assessment_claim_sha256="0" * 64,
            expected_family_by_example=manifest,
        )

    wrong_binding = _observation(
        protocol=protocol,
        example_id=rows[0].example_id,
        family_id=rows[0].family_id,
        runtime_binding_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="binding or arm differs"):
        _evaluate(
            protocol,
            chain((wrong_binding,), islice(rows, 1, None)),
            manifest,
        )


def test_routed_arm_and_routed_panel_are_rejected_unconditionally() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    example_id, family_id = next(iter(manifest.items()))

    with pytest.raises(ValueError, match="must be all_on"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            arm="routed",
        )

    rows, manifest = _panel(protocol=protocol)
    with pytest.raises(ValueError, match="routed observations are disabled"):
        _PRIVATE_EVALUATE(
            protocol,
            rows,
            assessment_claim_sha256=(
                protocol.calibration_b_assessment_claim_sha256()
            ),
            expected_family_by_example=manifest,
            routed_observations=(),
        )


def test_causal_masks_are_recomputed_and_unaffected_logits_excluded() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    positions = torch.arange(7, 12, dtype=torch.int64)
    boundaries = torch.arange(4, dtype=torch.int64)
    prefix_only_mismatch = _source_logits(
        torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    )
    prefix_only_mismatch[0, 0] = 0.0
    prefix_only_mismatch[0, 1] = 3.0
    rows, manifest = _panel(
        protocol=protocol,
        modal_scale=1.0,
        candidate_logits=prefix_only_mismatch,
        logical_positions=positions,
        supervised_boundary_indices=boundaries,
    )

    report = _evaluate(protocol, rows, manifest)

    assert report["all_on"]["behavioral_scope"] == {
        "token_scope": "causally_affected_supervised_tokens_only",
        "total_supervised_tokens": 384,
        "affected_supervised_tokens": 288,
        "affected_supervised_coverage": 0.75,
        "unaffected_prefix_tokens_excluded": True,
    }
    assert report["all_on"]["behavioral"]["aggregate"][
        "supervised_tokens"
    ] == 288
    assert report["all_on"]["behavioral"]["gates"]["passed"] is True
    assert report["authorization"]["partial_shadow_qualified"] is True


def test_modal_coverage_direction_and_behavior_fail_independently() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    positions = torch.arange(6, 11, dtype=torch.int64)
    boundaries = torch.arange(4, dtype=torch.int64)
    coverage_rows, manifest = _panel(
        protocol=protocol,
        logical_positions=positions,
        supervised_boundary_indices=boundaries,
    )
    coverage_report = _evaluate(protocol, coverage_rows, manifest)
    assert coverage_report["all_on"]["boundary"][
        "valid_target_coverage"
    ] == 0.6
    assert coverage_report["all_on"]["boundary"]["gates"][
        "valid_target_coverage"
    ] is False
    assert coverage_report["authorization"][
        "partial_shadow_qualified"
    ] is False

    direction_rows, manifest = _panel(
        protocol=protocol,
        modal_scale=-1.0,
    )
    direction_report = _evaluate(protocol, direction_rows, manifest)
    assert direction_report["all_on"]["boundary"][
        "pooled_target_modal_relative_error"
    ] == pytest.approx(2.0)
    assert direction_report["all_on"]["boundary"][
        "pooled_target_modal_cosine"
    ] == pytest.approx(-1.0)
    assert direction_report["all_on"]["boundary"]["gates"]["passed"] is False

    behavioral_rows, manifest = _panel(
        protocol=protocol,
        modal_scale=1.0,
        candidate_logits=_reversed_logits(),
    )
    behavioral_report = _evaluate(protocol, behavioral_rows, manifest)
    assert behavioral_report["all_on"]["boundary"]["gates"]["passed"] is True
    assert behavioral_report["all_on"]["behavioral"]["aggregate"][
        "top1_agreement_to_source"
    ] == 0.0
    assert behavioral_report["all_on"]["behavioral"]["gates"][
        "passed"
    ] is False


def test_projection_and_carrier_are_separate_required_gates() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    projection_rows, manifest = _panel(
        protocol=protocol,
        modal_scale=1.0,
        projection_scale=0.5,
    )
    projection_report = _evaluate(protocol, projection_rows, manifest)
    projection = projection_report["all_on"]["projection_capacity"]
    assert projection["pooled_full_width_delta_relative_error"] == (
        pytest.approx(0.5)
    )
    assert projection["behavioral"]["gates"]["passed"] is True
    assert projection["gates"]["passed"] is False
    assert projection_report["all_on"]["carrier_completeness"]["gates"][
        "passed"
    ] is True

    carrier_rows, manifest = _panel(
        protocol=protocol,
        modal_scale=1.0,
        carrier_oracle_logits=_reversed_logits(),
    )
    carrier_report = _evaluate(protocol, carrier_rows, manifest)
    assert carrier_report["all_on"]["boundary"]["gates"]["passed"] is True
    assert carrier_report["all_on"]["projection_capacity"]["gates"][
        "passed"
    ] is True
    carrier = carrier_report["all_on"]["carrier_completeness"]
    assert carrier["interpretation"] == (
        "incomplete_replacement_not_isolated_boundary_fidelity"
    )
    assert carrier["behavioral"]["aggregate"][
        "top1_agreement_to_source"
    ] == 0.0
    assert carrier["gates"]["passed"] is False


def test_worst_family_and_nondegenerate_signal_gates_prevent_dilution() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    family = next(iter(manifest.values()))
    modal_rows, manifest = _panel(
        protocol=protocol,
        modal_scale_by_family={family: 0.5},
    )
    modal_report = _evaluate(protocol, modal_rows, manifest)
    boundary = modal_report["all_on"]["boundary"]
    assert boundary["pooled_target_modal_relative_error"] < 0.25
    assert boundary["worst_family_target_modal_relative_error"] == (
        pytest.approx(0.5)
    )
    assert boundary["gates"][
        "pooled_target_modal_relative_error"
    ] is True
    assert boundary["gates"][
        "worst_family_target_modal_relative_error"
    ] is False

    projection_rows, manifest = _panel(
        protocol=protocol,
        projection_scale_by_family={family: 0.88},
    )
    projection_report = _evaluate(protocol, projection_rows, manifest)
    projection = projection_report["all_on"]["projection_capacity"]
    assert projection["pooled_full_width_delta_relative_error"] < 0.05
    assert projection["worst_family_full_width_delta_relative_error"] == (
        pytest.approx(0.12)
    )
    assert projection["gates"][
        "pooled_full_width_delta_relative_error"
    ] is True
    assert projection["gates"][
        "worst_family_full_width_delta_relative_error"
    ] is False

    degenerate_rows, manifest = _panel(
        protocol=protocol,
        modal_signal_scale_by_family={family: 0.0},
        full_signal_scale_by_family={family: 0.0},
    )
    degenerate_report = _evaluate(protocol, degenerate_rows, manifest)
    assert degenerate_report["all_on"]["boundary"]["gates"][
        "nondegenerate_every_family_source_modal_signal"
    ] is False
    assert degenerate_report["all_on"]["projection_capacity"]["gates"][
        "nondegenerate_every_family_source_full_width_signal"
    ] is False
    assert degenerate_report["authorization"][
        "partial_shadow_qualified"
    ] is False


def test_exact_observation_membership_duplicates_and_families_fail_closed() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    rows, manifest = _panel(protocol=protocol)

    first = rows[0]
    duplicate = chain((first, first), islice(rows, 2, None))
    with pytest.raises(ValueError, match="duplicate shadow observation"):
        _evaluate(protocol, duplicate, manifest)

    with pytest.raises(ValueError, match="missing examples"):
        _evaluate(protocol, rows[:-1], manifest)

    wrong_family = _observation(
        protocol=protocol,
        example_id=rows[0].example_id,
        family_id="wrong-family",
    )
    with pytest.raises(ValueError, match="belongs to family"):
        _evaluate(
            protocol,
            chain((wrong_family,), islice(rows, 1, None)),
            manifest,
        )

    undeclared = _observation(
        protocol=protocol,
        example_id="0" * 64,
        family_id=rows[0].family_id,
    )
    with pytest.raises(ValueError, match="undeclared"):
        _evaluate(
            protocol,
            chain((undeclared,), islice(rows, 1, None)),
            manifest,
        )

    relabelled_manifest = {
        **manifest,
        rows[0].example_id: "wrong-family",
    }
    with pytest.raises(ValueError, match="exact full frozen"):
        _evaluate(protocol, rows, relabelled_manifest)


def test_observation_round_trip_and_tensor_tamper_fail_closed() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    example_id, family_id = next(iter(manifest.items()))
    observation = _observation(
        protocol=protocol,
        example_id=example_id,
        family_id=family_id,
    )
    restored = (
        Gemma3L3L4GraphOrganizedSVDShadowObservation.from_state_dict(
            observation.state_dict()
        )
    )
    assert restored.artifact_sha256 == observation.artifact_sha256
    assert torch.equal(restored.logical_positions, observation.logical_positions)
    assert torch.equal(
        restored.source_eligible_mask,
        observation.source_eligible_mask,
    )

    changed_tensor = copy.deepcopy(observation.state_dict())
    changed_tensor["candidate_target_modes"][0, 0] += 1.0
    with pytest.raises(ValueError, match="evidence payload differs"):
        Gemma3L3L4GraphOrganizedSVDShadowObservation.from_state_dict(
            changed_tensor
        )

    changed_digest = copy.deepcopy(observation.state_dict())
    changed_digest["tensor_sha256s"]["logical_positions"] = "0" * 64
    with pytest.raises(ValueError, match="serialized observation tensors"):
        Gemma3L3L4GraphOrganizedSVDShadowObservation.from_state_dict(
            changed_digest
        )

    unknown = copy.deepcopy(observation.state_dict())
    unknown["prompt_text"] = "must never be accepted"
    with pytest.raises(ValueError, match="fields differ"):
        Gemma3L3L4GraphOrganizedSVDShadowObservation.from_state_dict(unknown)

    observation.logical_positions[0] = 7
    with pytest.raises(ValueError, match="hash mismatch"):
        observation.validate_integrity()


def test_receipts_bind_inputs_targets_logits_and_exact_pass_shape() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    entries = list(manifest.items())
    first_id, first_family = entries[0]
    second_id, second_family = entries[1]
    first = _observation(
        protocol=protocol,
        example_id=first_id,
        family_id=first_family,
        candidate_logits=_reversed_logits(),
    )
    second = _observation(
        protocol=protocol,
        example_id=second_id,
        family_id=second_family,
    )

    with pytest.raises(ValueError, match="evidence payload differs"):
        replace(second, targets=torch.tensor([1], dtype=torch.int64))

    with pytest.raises(ValueError, match="evidence payload differs"):
        replace(second, candidate_logits=first.candidate_logits)

    with pytest.raises(ValueError, match="five-pass receipt differs"):
        replace(
            second,
            five_pass_receipt_sha256=first.five_pass_receipt_sha256,
        )

    with pytest.raises(ValueError, match="prompt identity.*equal"):
        _observation(
            protocol=protocol,
            example_id=second_id,
            family_id=second_family,
            prompt_identity_sha256="0" * 64,
        )

    with pytest.raises(ValueError, match="provenance drifted"):
        replace(
            second,
            shadow_model_forward_count=2,
            model_forward_count=4,
        )


def test_model_input_replay_across_two_prompt_ids_is_rejected() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    rows, manifest = _panel(protocol=protocol)
    entries = list(manifest.items())
    first_id, first_family = entries[0]
    second_id, second_family = entries[1]
    first = _observation(
        protocol=protocol,
        example_id=first_id,
        family_id=first_family,
    )
    replay = _observation(
        protocol=protocol,
        example_id=second_id,
        family_id=second_family,
        model_inputs_sha256=first.model_inputs_sha256,
    )

    with pytest.raises(
        ValueError,
        match="replayed shadow observation model inputs",
    ):
        _evaluate(
            protocol,
            chain((first, replay), islice(rows, 2, None)),
            manifest,
        )


def test_observation_role_source_mask_and_geometry_are_verified() -> None:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    example_id, family_id = next(iter(manifest.items()))

    with pytest.raises(ValueError, match="role must be"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            role="calibration_a_development",
        )

    with pytest.raises(ValueError, match="source-eligible mask differs"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            source_eligible_mask=torch.zeros(2, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="supervised boundary indices"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            supervised_boundary_indices=torch.tensor([0, 9]),
        )

    with pytest.raises(ValueError, match="contiguous"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            logical_positions=torch.tensor([8, 10, 9, 11, 12]),
        )

    with pytest.raises(ValueError, match="affected supervised"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            logical_positions=torch.arange(5),
        )

    with pytest.raises(ValueError, match="tensor geometry differs"):
        _observation(
            protocol=protocol,
            example_id=example_id,
            family_id=family_id,
            source_logits=_source_logits(vocab_size=3),
        )
