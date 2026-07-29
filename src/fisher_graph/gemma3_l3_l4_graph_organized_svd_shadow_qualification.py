"""Locked prompt-to-five-pass qualification and pure evidence assembly.

The host-global held-out transaction privately loads and authenticates the
frozen local Gemma tokenizer, then passes ephemeral strict UTF-8 prompt bytes
to this module's private issuer.  That issuer verifies prompt membership,
tokenizes exactly one prompt on CPU, and owns the exact all-on ``3 + 1 + 1``
execution.  Callers cannot supply model-input tensors or invoke a supported
Calibration-B issuer directly.

The lower-level assembly helper remains useful for deterministic tests and
offline evidence inspection, but it accepts caller-supplied evidence and is
therefore explicitly *not* a held-out evidence issuer.  SHA-256 fields in this
module provide integrity and audit provenance; they do not claim hostile
in-process cryptographic attestation.  Prompts and serving outputs are never
retained by either path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .gemma3_experiment import make_causal_lm_calibration_batches
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    Gemma3L3L4GraphOrganizedSVDShadowObservation,
    Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt,
    derive_gemma3_l3_l4_graph_organized_svd_shadow_masks,
    frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest,
    gemma3_l3_l4_graph_organized_svd_evidence_payload_sha256,
    gemma3_l3_l4_graph_organized_svd_model_inputs_sha256,
    gemma3_l3_l4_graph_organized_svd_prompt_sha256,
)
from .gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    AuthenticatedOracleSuffixResult,
    Gemma3L3L4GraphOrganizedSVDShadowResult,
    Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    validate_gemma3_l3_l4_shadow_model_inputs_sha256,
)


__all__ = [
    "Gemma3L3L4GraphOrganizedSVDOracleInjections",
    "assemble_gemma3_l3_l4_graph_organized_svd_shadow_evidence_observation",
    "derive_gemma3_l3_l4_supervised_boundary",
    "gemma3_l3_l4_graph_organized_svd_tokenizer_provenance",
    "prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections",
]


_QUALIFICATION_BINDING_SCOPE = (
    "all_on_partial_edge_reference_oracle_shadow_metrics_only"
)


def _json_compatible(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return str(value)


def _tokenizer_backend_identity(tokenizer: object) -> dict[str, object]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        raise ValueError("tokenizer must expose a serializable backend")
    backend_text = to_str()
    if not isinstance(backend_text, str):
        raise ValueError("tokenizer backend serialization must be text")
    backend_bytes = backend_text.encode("utf-8")
    return {
        "bytes": len(backend_bytes),
        "sha256": hashlib.sha256(backend_bytes).hexdigest(),
    }


def gemma3_l3_l4_graph_organized_svd_tokenizer_provenance(
    tokenizer: object,
) -> dict[str, object]:
    """Hash the live tokenizer configuration and semantic implementation."""

    config = {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "init_kwargs": getattr(tokenizer, "init_kwargs", None),
    }
    serialized = json.dumps(
        _json_compatible(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    backend_identity = _tokenizer_backend_identity(tokenizer)

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise ValueError("tokenizer must expose get_vocab")
    vocab = get_vocab()
    if (
        not isinstance(vocab, Mapping)
        or not all(
            isinstance(token, str)
            and type(token_id) is int
            and token_id >= 0
            for token, token_id in vocab.items()
        )
    ):
        raise ValueError("tokenizer vocabulary must map text to token IDs")
    canonical_vocab = sorted(
        ((token, token_id) for token, token_id in vocab.items()),
        key=lambda item: (item[1], item[0]),
    )
    vocab_bytes = json.dumps(
        canonical_vocab,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")

    added = getattr(tokenizer, "added_tokens_encoder", None)
    if (
        not isinstance(added, Mapping)
        or not all(
            isinstance(token, str)
            and type(token_id) is int
            and token_id >= 0
            for token, token_id in added.items()
        )
    ):
        raise ValueError("tokenizer added-token map is malformed")
    canonical_added = sorted(
        ((token, token_id) for token, token_id in added.items()),
        key=lambda item: (item[0], item[1]),
    )
    added_bytes = json.dumps(
        canonical_added,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")

    special = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(special, Mapping):
        raise ValueError("tokenizer special-token map is malformed")
    special_bytes = json.dumps(
        {str(key): str(value) for key, value in special.items()},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")

    def installed(distribution: str) -> str | None:
        try:
            return version(distribution)
        except PackageNotFoundError:
            return None

    return {
        "tokenizer_class": (
            f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
        ),
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "configuration_sha256": hashlib.sha256(serialized).hexdigest(),
        "backend_serialized_bytes": backend_identity["bytes"],
        "backend_serialized_sha256": backend_identity["sha256"],
        "canonical_vocab_count": len(canonical_vocab),
        "canonical_vocab_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
        "added_token_count": len(canonical_added),
        "added_tokens_sha256": hashlib.sha256(added_bytes).hexdigest(),
        "special_tokens_map_sha256": hashlib.sha256(
            special_bytes
        ).hexdigest(),
        "transformers_version": installed("transformers"),
        "tokenizers_version": installed("tokenizers"),
        "sentencepiece_version": installed("sentencepiece"),
    }


def _canonical_float(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if (
        result.ndim != ndim
        or any(int(width) <= 0 for width in result.shape)
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError(f"{label} has invalid geometry or values")
    return result


def _runtime_protocol_binding_metadata(
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
) -> dict[str, object]:
    """Translate public runtime identities into the protocol binding ABI."""

    return {
        "candidate_logical_artifact_sha256": (
            runtime.candidate_artifact_sha256
        ),
        "basis_payload_sha256": runtime.basis_payload_sha256,
        "signed_plan_sha256": runtime.plan_artifact_sha256,
        "source_model_sha256": runtime.source_model_sha256,
        "factorized_live_execution_sha256": runtime.live_model_sha256,
        "adapter_execution_fingerprint": (
            runtime.adapter_execution_sha256
        ),
        "binding_scope": _QUALIFICATION_BINDING_SCOPE,
        "routing_enabled": False,
    }


def _validate_protocol_runtime(
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
) -> str:
    if not isinstance(
        protocol,
        Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    ):
        raise TypeError("protocol must use the strict shadow protocol type")
    if not isinstance(
        runtime,
        Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    ):
        raise TypeError("runtime must use the strict L3/L4 shadow type")
    protocol.validate_integrity()
    runtime.validate_integrity()
    return protocol.validate_runtime_binding(
        _runtime_protocol_binding_metadata(runtime)
    )


def _validate_live_adapter(
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    adapter: Gemma3CausalLMAdapter,
) -> None:
    """Authenticate the exact eval-mode adapter without running a forward."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    runtime._authenticate_adapter(adapter)


def _validate_prompt_manifest(
    *,
    prompt_utf8: bytes,
    example_id: str,
    family_id: str,
) -> str:
    if not isinstance(prompt_utf8, bytes):
        raise TypeError("prompt_utf8 must be exact strict UTF-8 bytes")
    prompt_identity_sha256 = (
        gemma3_l3_l4_graph_organized_svd_prompt_sha256(prompt_utf8)
    )
    if prompt_identity_sha256 != example_id:
        raise ValueError("prompt identity differs from example_id")
    manifest = (
        frozen_gemma3_l3_l4_graph_organized_svd_calibration_b_manifest()
    )
    if (
        not isinstance(example_id, str)
        or not isinstance(family_id, str)
        or manifest.get(example_id) != family_id
    ):
        raise ValueError(
            "example and family must match the frozen Calibration-B manifest"
        )
    return prompt_identity_sha256


def _frozen_tokenizer_contract(
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
) -> Mapping[str, object]:
    metadata = protocol.metadata()
    contract = metadata.get("tokenizer")
    if not isinstance(contract, Mapping):
        raise ValueError("protocol must contain the frozen tokenizer contract")
    expected_fields = {
        "tokenizer_class",
        "name_or_path",
        "configuration_sha256",
        "backend_serialized_bytes",
        "backend_serialized_sha256",
        "post_tokenization_backend_serialized_bytes",
        "post_tokenization_backend_serialized_sha256",
        "canonical_vocab_count",
        "canonical_vocab_sha256",
        "added_token_count",
        "added_tokens_sha256",
        "special_tokens_map_sha256",
        "transformers_version",
        "tokenizers_version",
        "sentencepiece_version",
        "vocab_size",
        "model_revision",
        "local_files_only",
        "max_length",
        "tokenization_batch_size",
        "device",
        "padding_side",
        "padding",
        "truncation",
        "add_special_tokens",
        "return_attention_mask",
    }
    if set(contract) != expected_fields:
        raise ValueError("protocol frozen tokenizer contract fields differ")
    return contract


def _normalize_and_validate_frozen_tokenizer(
    tokenizer: object,
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
) -> Mapping[str, object]:
    """Normalize padding, then bind the live tokenizer to the frozen V9 ABI."""

    contract = _frozen_tokenizer_contract(protocol)
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if (
        type(vocab_size) is not int
        or vocab_size != contract["vocab_size"]
    ):
        raise ValueError("tokenizer vocab_size differs from frozen Gemma")
    tokenizer_type = (
        f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    )
    if tokenizer_type != contract["tokenizer_class"]:
        raise ValueError("tokenizer class differs from frozen Gemma")
    if getattr(tokenizer, "name_or_path", None) != contract["name_or_path"]:
        raise ValueError("tokenizer name_or_path differs from frozen Gemma")
    direct_local_only = getattr(tokenizer, "local_files_only", None)
    if direct_local_only is not None and direct_local_only is not True:
        raise ValueError("tokenizer is not marked local_files_only")

    if not hasattr(tokenizer, "padding_side"):
        raise ValueError("tokenizer must expose padding_side")
    try:
        setattr(tokenizer, "padding_side", contract["padding_side"])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "tokenizer padding_side cannot be normalized"
        ) from error
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("tokenizer padding_side must normalize to right")

    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError("tokenizer must define a pad or EOS token")
        try:
            setattr(tokenizer, "pad_token", eos_token)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("tokenizer pad token cannot be normalized") from error
        if getattr(tokenizer, "pad_token_id", None) is None:
            raise ValueError("tokenizer pad token normalization failed")

    revisions: list[object] = []
    for attribute in ("revision", "_commit_hash"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            revisions.append(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if init_kwargs is not None and not isinstance(init_kwargs, Mapping):
        raise ValueError("tokenizer init_kwargs must be a mapping when exposed")
    if isinstance(init_kwargs, Mapping):
        for name in ("revision", "_commit_hash", "commit_hash"):
            value = init_kwargs.get(name)
            if value is not None:
                revisions.append(value)
        local_only = init_kwargs.get("local_files_only")
        if local_only is not None and local_only is not True:
            raise ValueError("tokenizer is not marked local_files_only")
    if any(value != contract["model_revision"] for value in revisions):
        raise ValueError("tokenizer revision differs from frozen Gemma")

    provenance = (
        gemma3_l3_l4_graph_organized_svd_tokenizer_provenance(tokenizer)
    )
    expected_provenance = {
        "tokenizer_class": contract["tokenizer_class"],
        "name_or_path": contract["name_or_path"],
        "configuration_sha256": contract["configuration_sha256"],
        "backend_serialized_bytes": contract[
            "backend_serialized_bytes"
        ],
        "backend_serialized_sha256": contract[
            "backend_serialized_sha256"
        ],
        "canonical_vocab_count": contract["canonical_vocab_count"],
        "canonical_vocab_sha256": contract["canonical_vocab_sha256"],
        "added_token_count": contract["added_token_count"],
        "added_tokens_sha256": contract["added_tokens_sha256"],
        "special_tokens_map_sha256": contract[
            "special_tokens_map_sha256"
        ],
        "transformers_version": contract["transformers_version"],
        "tokenizers_version": contract["tokenizers_version"],
        "sentencepiece_version": contract["sentencepiece_version"],
    }
    if provenance != expected_provenance:
        raise ValueError(
            "tokenizer implementation differs from frozen V9 provenance"
        )
    return contract


def _load_and_validate_frozen_local_tokenizer(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
) -> tuple[object, Mapping[str, object]]:
    """Load the pinned tokenizer from local files and authenticate semantics."""

    contract = _frozen_tokenizer_contract(protocol)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "the frozen Gemma tokenizer requires the optional gemma extras"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(contract["name_or_path"]),
        revision=str(contract["model_revision"]),
        local_files_only=True,
        use_fast=False,
    )
    validated = _normalize_and_validate_frozen_tokenizer(
        tokenizer,
        protocol=protocol,
    )
    return tokenizer, validated


def _tokenize_verified_prompt(
    *,
    tokenizer: object,
    prompt_utf8: bytes,
    contract: Mapping[str, object],
) -> dict[str, Tensor]:
    """Tokenize exactly one verified prompt under the frozen CPU contract."""

    backend_before = _tokenizer_backend_identity(tokenizer)
    allowed_before = (
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
    )
    if backend_before not in allowed_before:
        raise ValueError("tokenizer backend drifted before tokenization")
    prompt = prompt_utf8.decode("utf-8", errors="strict")
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        (prompt,),
        max_length=int(contract["max_length"]),
        tokenization_batch_size=int(
            contract["tokenization_batch_size"]
        ),
        device=torch.device(str(contract["device"])),
    )
    iterator = iter(batches)
    try:
        batch = next(iterator)
    except StopIteration as error:
        raise RuntimeError("one-prompt tokenization returned no batch") from error
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise RuntimeError("one-prompt tokenization returned multiple batches")
    backend_after = _tokenizer_backend_identity(tokenizer)
    if backend_after != {
        "bytes": contract["post_tokenization_backend_serialized_bytes"],
        "sha256": contract[
            "post_tokenization_backend_serialized_sha256"
        ],
    }:
        raise ValueError("tokenizer backend drifted during tokenization")
    if set(batch.model_inputs) != {"input_ids", "attention_mask"}:
        raise ValueError("tokenizer contract produced unexpected model inputs")
    model_inputs = {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in batch.model_inputs.items()
    }
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    if (
        input_ids.dtype not in (torch.int32, torch.int64)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or attention_mask.dtype != torch.bool
        or attention_mask.shape != input_ids.shape
        or input_ids.shape[1] > int(contract["max_length"])
        or input_ids.shape[1] < 2
    ):
        raise ValueError("tokenizer produced invalid one-prompt CPU tensors")
    if (
        bool((input_ids < 0).any())
        or bool((input_ids >= int(contract["vocab_size"])).any())
    ):
        raise ValueError("tokenizer produced an ID outside frozen vocabulary")
    mask = attention_mask[0]
    if not bool(mask.any()):
        raise ValueError("tokenizer produced no valid prompt tokens")
    first_padding = torch.nonzero(~mask, as_tuple=False)
    if (
        first_padding.numel()
        and bool(mask[int(first_padding[0, 0]) :].any())
    ):
        raise ValueError("tokenizer output is not right padded")
    return model_inputs


def derive_gemma3_l3_l4_supervised_boundary(
    input_ids: Tensor,
    valid_target_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return local logit-row indices and causal next-token targets.

    The bridge follows the repository's causal-language-model convention:
    physical row ``j`` is supervised by token ``input_ids[j + 1]`` exactly
    when both rows are valid.  Invalid padding never creates a cross-padding
    pair.  The result is canonical CPU int64 data.
    """

    if (
        not isinstance(input_ids, Tensor)
        or input_ids.dtype not in (torch.int32, torch.int64)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.shape[1] <= 1
    ):
        raise ValueError(
            "input_ids must be integer [1, sequence] data with sequence > 1"
        )
    if (
        not isinstance(valid_target_mask, Tensor)
        or valid_target_mask.dtype != torch.bool
        or valid_target_mask.shape != input_ids.shape
    ):
        raise ValueError(
            "valid_target_mask must be boolean data matching input_ids"
        )
    valid = valid_target_mask.detach().to(device="cpu")
    tokens = input_ids.detach().to(device="cpu", dtype=torch.int64)
    supervised = valid[0, :-1] & valid[0, 1:]
    boundary_indices = torch.nonzero(
        supervised,
        as_tuple=False,
    ).flatten().to(dtype=torch.int64).contiguous()
    if boundary_indices.numel() == 0:
        raise ValueError("example has no valid causal next-token boundary")
    targets = tokens[0].index_select(
        0,
        boundary_indices + 1,
    ).contiguous()
    return boundary_indices, targets


@dataclass(frozen=True, slots=True)
class Gemma3L3L4GraphOrganizedSVDOracleInjections:
    """The two exact X4 tensors and their boundary-analysis ingredients."""

    projection_x4: Tensor
    carrier_x4: Tensor
    source_target_full_width_delta: Tensor
    source_target_modes: Tensor
    projection_target_full_width_delta: Tensor
    valid_target_mask: Tensor
    target_affected_mask: Tensor
    runtime_binding_sha256: str
    shadow_result_artifact_sha256: str
    execution_grid_sha256: str

    def __post_init__(self) -> None:
        boundary = self.projection_x4
        if (
            not isinstance(boundary, Tensor)
            or not boundary.is_floating_point()
            or boundary.ndim != 3
            or self.carrier_x4.shape != boundary.shape
            or self.carrier_x4.dtype != boundary.dtype
            or self.carrier_x4.device != boundary.device
            or self.source_target_full_width_delta.shape != boundary.shape
            or self.projection_target_full_width_delta.shape != boundary.shape
            or self.source_target_modes.shape != (*boundary.shape[:2], 64)
            or self.valid_target_mask.shape != boundary.shape[:2]
            or self.valid_target_mask.dtype != torch.bool
            or self.target_affected_mask.shape != boundary.shape[:2]
            or self.target_affected_mask.dtype != torch.bool
            or bool(
                (
                    self.target_affected_mask
                    & ~self.valid_target_mask
                ).any()
            )
        ):
            raise ValueError("oracle injection tensor geometry differs")
        valid = self.valid_target_mask
        affected = self.target_affected_mask
        if (
            not bool(torch.isfinite(self.projection_x4[valid]).all())
            or not bool(torch.isfinite(self.carrier_x4[valid]).all())
            or not bool(
                torch.isfinite(
                    self.source_target_full_width_delta[valid]
                ).all()
            )
            or not bool(
                torch.isfinite(self.source_target_modes[affected]).all()
            )
            or not bool(
                torch.isfinite(
                    self.projection_target_full_width_delta[affected]
                ).all()
            )
        ):
            raise ValueError(
                "valid oracle rows and affected modal rows must be finite"
            )
        for value, label in (
            (self.runtime_binding_sha256, "runtime binding"),
            (
                self.shadow_result_artifact_sha256,
                "shadow result artifact",
            ),
            (self.execution_grid_sha256, "execution grid"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
            ):
                raise ValueError(f"{label} must be a lowercase SHA-256")


def prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
) -> Gemma3L3L4GraphOrganizedSVDOracleInjections:
    """Derive the projection and exact-carrier X4 oracle interventions."""

    if not isinstance(
        runtime,
        Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    ):
        raise TypeError("runtime must use the strict L3/L4 shadow type")
    if not isinstance(
        three_pass_result,
        Gemma3L3L4GraphOrganizedSVDShadowResult,
    ):
        raise TypeError("three_pass_result must use the strict result type")
    runtime.validate_result_binding(three_pass_result)
    if (
        three_pass_result.arm != "all_on"
        or three_pass_result.accounting.model_forward_count != 3
        or three_pass_result.authoritative_logits is None
        or three_pass_result.candidate_logits is None
    ):
        raise ValueError(
            "qualification requires a completed all-on three-pass result"
        )
    full_delta = (
        three_pass_result.authoritative_x4
        - three_pass_result.reference_x4
    )
    valid = three_pass_result.valid_target_mask.to(device=full_delta.device)
    if not bool(torch.isfinite(full_delta[valid]).all()):
        raise ValueError("valid source target deltas must be finite")
    encoded_delta = torch.where(
        valid.unsqueeze(-1),
        full_delta,
        torch.zeros_like(full_delta),
    )
    source_modes = runtime.encode_target_delta(encoded_delta)
    projection_delta = runtime.decode_target_modal_delta(source_modes)
    reference = three_pass_result.reference_x4.to(
        device=projection_delta.device,
        dtype=projection_delta.dtype,
    )
    authoritative = three_pass_result.authoritative_x4.to(
        device=projection_delta.device,
        dtype=projection_delta.dtype,
    )
    affected = three_pass_result.target_affected_mask.to(
        device=projection_delta.device,
    )
    projection_analysis = authoritative.clone()
    projection_analysis[affected] = (
        reference + projection_delta
    )[affected]
    projection_x4 = projection_analysis.to(
        device=three_pass_result.authoritative_x4.device,
        dtype=three_pass_result.authoritative_x4.dtype,
    ).contiguous()
    carrier_x4 = three_pass_result.authoritative_x4.detach().clone().contiguous()
    result = Gemma3L3L4GraphOrganizedSVDOracleInjections(
        projection_x4=projection_x4,
        carrier_x4=carrier_x4,
        source_target_full_width_delta=full_delta.detach().clone().contiguous(),
        source_target_modes=source_modes.detach().clone().contiguous(),
        projection_target_full_width_delta=(
            projection_delta.detach().clone().contiguous()
        ),
        valid_target_mask=(
            three_pass_result.valid_target_mask.detach().clone().contiguous()
        ),
        target_affected_mask=(
            three_pass_result.target_affected_mask.detach()
            .clone()
            .contiguous()
        ),
        runtime_binding_sha256=three_pass_result.runtime_binding_sha256,
        shadow_result_artifact_sha256=(
            three_pass_result.result_artifact_sha256
        ),
        execution_grid_sha256=three_pass_result.execution_grid_sha256,
    )
    runtime.validate_result_binding(three_pass_result)
    return result


def _validate_oracle(
    oracle: AuthenticatedOracleSuffixResult,
    *,
    role: str,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    injected_x4: Tensor,
) -> None:
    if not isinstance(oracle, AuthenticatedOracleSuffixResult):
        raise TypeError("oracle results must use the authenticated runtime type")
    oracle.validate_integrity()
    if (
        oracle.role != role
        or oracle.runtime_binding_sha256
        != three_pass_result.runtime_binding_sha256
        or oracle.shadow_result_artifact_sha256
        != three_pass_result.result_artifact_sha256
        or oracle.execution_grid_sha256
        != three_pass_result.execution_grid_sha256
        or oracle.adapter_execution_sha256
        != runtime.adapter_execution_sha256
    ):
        raise ValueError("oracle suffix binding or role differs")
    oracle.validate_injected_x4(injected_x4)


def _gather_supervised_logits(
    logits: Tensor,
    *,
    boundary_indices: Tensor,
    expected_sequence_length: int,
    label: str,
) -> Tensor:
    if (
        not isinstance(logits, Tensor)
        or not logits.is_floating_point()
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != expected_sequence_length
        or logits.shape[2] < 2
    ):
        raise ValueError(f"{label} must have aligned [1, sequence, vocab] data")
    selected = logits[0].index_select(
        0,
        boundary_indices.to(device=logits.device),
    )
    return _canonical_float(selected, label=label, ndim=2)


def assemble_gemma3_l3_l4_graph_organized_svd_shadow_evidence_observation(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    three_pass_result: Gemma3L3L4GraphOrganizedSVDShadowResult,
    projection_oracle: AuthenticatedOracleSuffixResult,
    carrier_oracle: AuthenticatedOracleSuffixResult,
    model_inputs: Mapping[str, Tensor],
    prompt_utf8: bytes,
    example_id: str,
    family_id: str,
) -> Gemma3L3L4GraphOrganizedSVDShadowObservation:
    """Assemble supplied evidence; this is not a held-out evidence issuer.

    Callers provide every model input and runtime result to this pure helper.
    Only the fused ledger transaction may call the private five-pass issuer
    that derives model inputs from a claimed prompt.
    """

    qualification_binding_sha256 = _validate_protocol_runtime(
        protocol,
        runtime,
    )
    if not isinstance(model_inputs, Mapping):
        raise TypeError("model_inputs must be a mapping")
    if "input_ids" not in model_inputs or "inputs_embeds" in model_inputs:
        raise ValueError(
            "qualification requires already-tokenized input_ids, not embeds"
        )
    if any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        for name, value in model_inputs.items()
    ):
        raise TypeError("model_inputs must map string names to Tensors")
    runtime.validate_result_binding(three_pass_result)
    model_inputs_sha256 = (
        validate_gemma3_l3_l4_shadow_model_inputs_sha256(
            model_inputs,
            three_pass_result.model_inputs_sha256,
        )
    )
    if (
        gemma3_l3_l4_graph_organized_svd_model_inputs_sha256(
            model_inputs
        )
        != model_inputs_sha256
    ):
        raise RuntimeError("protocol and runtime model-input identities differ")
    prompt_identity_sha256 = _validate_prompt_manifest(
        prompt_utf8=prompt_utf8,
        example_id=example_id,
        family_id=family_id,
    )
    assessment_claim_sha256 = (
        protocol.calibration_b_assessment_claim_sha256()
    )
    validate_gemma3_l3_l4_shadow_model_inputs_sha256(
        model_inputs,
        model_inputs_sha256,
    )
    bound_input_ids = model_inputs["input_ids"].detach().clone()
    if (
        three_pass_result.arm != "all_on"
        or three_pass_result.accounting.model_forward_count != 3
        or three_pass_result.authoritative_logits is None
        or three_pass_result.candidate_logits is None
    ):
        raise ValueError(
            "qualification requires a completed all-on three-pass result"
        )
    if (
        three_pass_result.authoritative_x4.shape[0] != 1
        or three_pass_result.authoritative_x4.shape[-1] != 640
        or runtime.target_modes != 64
        or runtime.residual_width != 640
    ):
        raise ValueError(
            "qualification requires one example with frozen 64/640 geometry"
        )
    boundary_indices, targets = derive_gemma3_l3_l4_supervised_boundary(
        bound_input_ids,
        three_pass_result.valid_target_mask,
    )
    positions = (
        three_pass_result.logical_positions[0]
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
    )
    valid = (
        three_pass_result.valid_target_mask[0]
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    derived_masks = derive_gemma3_l3_l4_graph_organized_svd_shadow_masks(
        positions,
        valid,
        boundary_indices,
    )
    source_eligible = (
        three_pass_result.source_eligible_mask[0]
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    target_affected = (
        three_pass_result.target_affected_mask[0]
        .detach()
        .to(device="cpu")
        .contiguous()
    )
    if (
        not torch.equal(
            source_eligible,
            derived_masks["source_eligible_mask"],
        )
        or not torch.equal(
            target_affected,
            derived_masks["target_affected_mask"],
        )
    ):
        raise ValueError(
            "runtime grid differs from protocol causal recomputation"
        )
    injections = (
        prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
            runtime,
            three_pass_result,
        )
    )
    _validate_oracle(
        projection_oracle,
        role="projection_64",
        runtime=runtime,
        three_pass_result=three_pass_result,
        injected_x4=injections.projection_x4,
    )
    _validate_oracle(
        carrier_oracle,
        role="exact_x4_carrier",
        runtime=runtime,
        three_pass_result=three_pass_result,
        injected_x4=injections.carrier_x4,
    )
    sequence_length = int(three_pass_result.authoritative_x4.shape[1])
    source_logits = _gather_supervised_logits(
        three_pass_result.authoritative_logits,
        boundary_indices=boundary_indices,
        expected_sequence_length=sequence_length,
        label="authoritative_logits",
    )
    candidate_logits = _gather_supervised_logits(
        three_pass_result.candidate_logits,
        boundary_indices=boundary_indices,
        expected_sequence_length=sequence_length,
        label="candidate_logits",
    )
    projection_logits = _gather_supervised_logits(
        projection_oracle.logits,
        boundary_indices=boundary_indices,
        expected_sequence_length=sequence_length,
        label="projection_oracle_logits",
    )
    carrier_logits = _gather_supervised_logits(
        carrier_oracle.logits,
        boundary_indices=boundary_indices,
        expected_sequence_length=sequence_length,
        label="carrier_oracle_logits",
    )
    evidence_tensors = {
        "source_logits": source_logits,
        "candidate_logits": candidate_logits,
        "projection_oracle_logits": projection_logits,
        "carrier_oracle_logits": carrier_logits,
        "targets": targets,
        "source_target_modes": (
            injections.source_target_modes[0]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        ),
        "candidate_target_modes": (
            three_pass_result.predicted_target_modal_delta[0]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        ),
        "source_target_full_width_delta": (
            injections.source_target_full_width_delta[0]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        ),
        "projection_target_full_width_delta": (
            injections.projection_target_full_width_delta[0]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        ),
        "logical_positions": positions,
        "supervised_boundary_indices": boundary_indices,
        "valid_target_mask": valid,
        "source_eligible_mask": source_eligible,
    }
    evidence_payload_sha256 = (
        gemma3_l3_l4_graph_organized_svd_evidence_payload_sha256(
            evidence_tensors
        )
    )
    receipt = derive_gemma3_l3_l4_graph_organized_svd_five_pass_receipt(
        protocol_sha256=protocol.artifact_sha256,
        assessment_claim_sha256=assessment_claim_sha256,
        runtime_binding_sha256=qualification_binding_sha256,
        example_id=example_id,
        family_id=family_id,
        prompt_identity_sha256=prompt_identity_sha256,
        model_inputs_sha256=model_inputs_sha256,
        shadow_result_artifact_sha256=(
            three_pass_result.result_artifact_sha256
        ),
        execution_grid_sha256=three_pass_result.execution_grid_sha256,
        projection_oracle_artifact_sha256=(
            projection_oracle.artifact_sha256
        ),
        projection_injected_x4_sha256=(
            projection_oracle.injected_x4_sha256
        ),
        carrier_oracle_artifact_sha256=carrier_oracle.artifact_sha256,
        carrier_injected_x4_sha256=carrier_oracle.injected_x4_sha256,
        evidence_payload_sha256=evidence_payload_sha256,
        shadow_model_forward_count=(
            three_pass_result.accounting.model_forward_count
        ),
        projection_oracle_model_forward_count=1,
        carrier_oracle_model_forward_count=1,
        projection_oracle_role=projection_oracle.role,
        carrier_oracle_role=carrier_oracle.role,
    )
    observation = Gemma3L3L4GraphOrganizedSVDShadowObservation(
        protocol_sha256=protocol.artifact_sha256,
        assessment_claim_sha256=assessment_claim_sha256,
        runtime_binding_sha256=qualification_binding_sha256,
        role="calibration_b_one_shot",
        arm="all_on",
        example_id=example_id,
        family_id=family_id,
        prompt_identity_sha256=prompt_identity_sha256,
        model_inputs_sha256=model_inputs_sha256,
        input_provenance_sha256=receipt[
            "input_provenance_sha256"
        ],
        shadow_result_artifact_sha256=(
            three_pass_result.result_artifact_sha256
        ),
        execution_grid_sha256=three_pass_result.execution_grid_sha256,
        projection_oracle_artifact_sha256=(
            projection_oracle.artifact_sha256
        ),
        projection_injected_x4_sha256=(
            projection_oracle.injected_x4_sha256
        ),
        carrier_oracle_artifact_sha256=carrier_oracle.artifact_sha256,
        carrier_injected_x4_sha256=carrier_oracle.injected_x4_sha256,
        evidence_payload_sha256=evidence_payload_sha256,
        five_pass_receipt_sha256=receipt[
            "five_pass_receipt_sha256"
        ],
        **evidence_tensors,
    )
    observation.validate_integrity()
    runtime.validate_result_binding(three_pass_result)
    projection_oracle.validate_integrity()
    carrier_oracle.validate_integrity()
    validate_gemma3_l3_l4_shadow_model_inputs_sha256(
        model_inputs,
        model_inputs_sha256,
    )
    return observation


def _execute_gemma3_l3_l4_graph_organized_svd_five_pass_observation(
    *,
    protocol: Gemma3L3L4GraphOrganizedSVDShadowProtocol,
    runtime: Gemma3L3L4GraphOrganizedSVDShadowRuntime,
    adapter: Gemma3CausalLMAdapter,
    tokenizer: object,
    prompt_utf8: bytes,
    example_id: str,
    family_id: str,
    _validated_tokenizer_contract: Mapping[str, object] | None = None,
) -> Gemma3L3L4GraphOrganizedSVDShadowObservation:
    """Execute the locked all-on ``3 + 1 + 1`` qualification transaction.

    The returned observation is metrics-only.  This wrapper never returns a
    candidate boundary, candidate logits, or either oracle output as a serving
    result.
    """

    _validate_protocol_runtime(protocol, runtime)
    _validate_live_adapter(runtime, adapter)
    _validate_prompt_manifest(
        prompt_utf8=prompt_utf8,
        example_id=example_id,
        family_id=family_id,
    )
    if _validated_tokenizer_contract is None:
        tokenizer_contract = _normalize_and_validate_frozen_tokenizer(
            tokenizer,
            protocol=protocol,
        )
    else:
        tokenizer_contract = _frozen_tokenizer_contract(protocol)
        if dict(_validated_tokenizer_contract) != dict(tokenizer_contract):
            raise ValueError(
                "prevalidated tokenizer contract differs from protocol"
            )
    model_inputs = _tokenize_verified_prompt(
        tokenizer=tokenizer,
        prompt_utf8=prompt_utf8,
        contract=tokenizer_contract,
    )
    model_inputs_sha256 = (
        gemma3_l3_l4_graph_organized_svd_model_inputs_sha256(
            model_inputs
        )
    )
    three_pass_result = runtime.execute_model_shadow(
        adapter,
        model_inputs,
        arm="all_on",
    )
    runtime.validate_result_binding(three_pass_result)
    validate_gemma3_l3_l4_shadow_model_inputs_sha256(
        model_inputs,
        model_inputs_sha256,
    )
    injections = (
        prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
            runtime,
            three_pass_result,
        )
    )
    projection_oracle = runtime.execute_oracle_suffix(
        adapter,
        model_inputs,
        three_pass_result,
        injections.projection_x4,
        role="projection_64",
    )
    carrier_oracle = runtime.execute_oracle_suffix(
        adapter,
        model_inputs,
        three_pass_result,
        injections.carrier_x4,
        role="exact_x4_carrier",
    )
    return (
        assemble_gemma3_l3_l4_graph_organized_svd_shadow_evidence_observation(
            protocol=protocol,
            runtime=runtime,
            three_pass_result=three_pass_result,
            projection_oracle=projection_oracle,
            carrier_oracle=carrier_oracle,
            model_inputs=model_inputs,
            prompt_utf8=prompt_utf8,
            example_id=example_id,
            family_id=family_id,
        )
    )
