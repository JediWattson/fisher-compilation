"""Streaming development evaluation for the conditional spectral shadow.

The production shadow runtime owns model execution and authentication.  This
module owns only development measurement: one prompt is tokenized, executed
through the all-on three-pass shadow, reduced to scalar statistics, and then
discarded before the next prompt is evaluated.

The source path remains authoritative.  Candidate boundaries and logits are
never returned and are consumed only by the metric accumulators.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
from torch import Tensor

from .gemma3_experiment import make_causal_lm_calibration_batches
from .gemma3_l3_l4_graph_organized_svd_shadow_qualification import (
    prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections,
)
from .shadow_fidelity import (
    ESTABLISHED_SHADOW_FIDELITY_GATES,
    ShadowFidelityExample,
    ShadowFidelityGates,
    SourceAuthoritativeShadowFidelityAccumulator,
)


_REPORT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_conditional_spectral_development_shadow"
)
_PROMPT_DOMAIN = b"fisher-graph:conditional-spectral-shadow-prompt:v1\0"
_MANIFEST_DOMAIN = b"fisher-graph:conditional-spectral-shadow-manifest:v1\0"
_ORACLE_RECEIPT_DOMAIN = (
    b"fisher-graph:conditional-spectral-shadow-oracle-receipt:v1\0"
)
_COMPLETE_H4_AUDIT_RECEIPT_DOMAIN = (
    b"fisher-graph:conditional-spectral-shadow-complete-h4-audit-receipt:v1\0"
)
_ORACLE_SUFFIX_ORDER = ("projection_64", "exact_x4_carrier")
_COMPLETE_H4_AUDIT_ORDER = (
    "native_h4_replay",
    "partial_exact_x4_replay",
    "complete_h4_identity",
)
_COMPLETE_H4_BOUNDARY_CALLBACK_ORDER = (
    "partial_exact_x4.y3",
    "partial_exact_x4.x4",
    "complete_h4.y3",
    "complete_h4.x4",
    "complete_h4.h4",
)
_SHA256_HEX = frozenset("0123456789abcdef")
_RUNTIME_BINDING_SCALAR_FIELDS = (
    "runtime_binding_sha256",
    "candidate_artifact_sha256",
    "candidate_method",
    "basis_payload_sha256",
    "plan_artifact_sha256",
    "raw_source_model_sha256",
    "live_factorized_model_sha256",
    "adapter_execution_sha256",
    "adapter_execution_binding_scope",
    "analysis_device",
    "residual_width",
    "source_modes",
    "source_rank",
    "target_modes",
    "lag_count",
    "executor_kind",
    "routing_supported",
    "candidate_serving_authorized",
    "all_on_only",
)


@dataclass(frozen=True, slots=True)
class Gemma3L3L4ConditionalSpectralShadowExample:
    """One named prompt in a family-disjoint development panel."""

    example_id: str
    family_id: str
    prompt: str

    def __post_init__(self) -> None:
        _identifier(self.example_id, label="example_id")
        _identifier(self.family_id, label="family_id")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be nonempty text")


ShadowExampleInput = (
    Gemma3L3L4ConditionalSpectralShadowExample | Mapping[str, object]
)


@dataclass(slots=True)
class _VectorMoments:
    rows: int = 0
    elements: int = 0
    residual_square: float = 0.0
    source_square: float = 0.0
    candidate_square: float = 0.0
    dot: float = 0.0

    def add(self, source: Tensor, candidate: Tensor) -> None:
        if (
            not isinstance(source, Tensor)
            or not isinstance(candidate, Tensor)
            or source.shape != candidate.shape
            or source.ndim != 2
            or source.shape[0] == 0
            or source.shape[1] == 0
            or not source.is_floating_point()
            or not candidate.is_floating_point()
        ):
            raise ValueError(
                "geometry rows must be aligned nonempty floating matrices"
            )
        with torch.inference_mode():
            source_cpu = source.detach().to(device="cpu", dtype=torch.float64)
            candidate_cpu = candidate.detach().to(
                device="cpu",
                dtype=torch.float64,
            )
            if not bool(torch.isfinite(source_cpu).all()) or not bool(
                torch.isfinite(candidate_cpu).all()
            ):
                raise ValueError("geometry rows must be finite")
            residual = candidate_cpu - source_cpu
            self.rows += source_cpu.shape[0]
            self.elements += source_cpu.numel()
            self.residual_square += float(residual.square().sum().item())
            self.source_square += float(source_cpu.square().sum().item())
            self.candidate_square += float(candidate_cpu.square().sum().item())
            self.dot += float((source_cpu * candidate_cpu).sum().item())

    def summary(self) -> dict[str, object]:
        source_l2 = math.sqrt(self.source_square)
        candidate_l2 = math.sqrt(self.candidate_square)
        residual_l2 = math.sqrt(self.residual_square)
        if self.source_square > 0.0:
            relative: float | None = math.sqrt(
                self.residual_square / self.source_square
            )
        elif self.residual_square == 0.0:
            relative = 0.0
        else:
            relative = None
        denominator = math.sqrt(self.source_square * self.candidate_square)
        if denominator > 0.0:
            cosine: float | None = self.dot / denominator
            # Keep harmless roundoff within the mathematical cosine range.
            cosine = max(-1.0, min(1.0, cosine))
        elif self.source_square == 0.0 and self.candidate_square == 0.0:
            cosine = 1.0
        else:
            cosine = None
        return {
            "affected_rows": self.rows,
            "scalar_elements": self.elements,
            "source_signal_l2_norm": source_l2,
            "candidate_signal_l2_norm": candidate_l2,
            "residual_l2_norm": residual_l2,
            "relative_l2_error": relative,
            "cosine": cosine,
            "source_signal_nondegenerate": self.source_square > 0.0,
        }


@dataclass(slots=True)
class _Coverage:
    examples: int = 0
    valid_rows: int = 0
    source_eligible_rows: int = 0
    affected_rows: int = 0
    supervised_tokens: int = 0
    affected_supervised_tokens: int = 0
    model_forwards: int = 0
    local_factorized_linear_macs: int = 0

    def summary(self) -> dict[str, int | float]:
        return {
            "example_count": self.examples,
            "valid_target_rows": self.valid_rows,
            "source_eligible_rows": self.source_eligible_rows,
            "affected_target_rows": self.affected_rows,
            "valid_target_coverage": self.affected_rows / self.valid_rows,
            "supervised_tokens": self.supervised_tokens,
            "affected_supervised_tokens": self.affected_supervised_tokens,
            "affected_supervised_coverage": (
                self.affected_supervised_tokens / self.supervised_tokens
            ),
            "model_forward_count": self.model_forwards,
            "local_factorized_linear_macs": (
                self.local_factorized_linear_macs
            ),
        }


@dataclass(slots=True)
class _FamilyState:
    coverage: _Coverage = field(default_factory=_Coverage)
    modal: _VectorMoments = field(default_factory=_VectorMoments)
    full_width: _VectorMoments = field(default_factory=_VectorMoments)


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return value


def _coerce_example(value: ShadowExampleInput) -> (
    Gemma3L3L4ConditionalSpectralShadowExample
):
    if isinstance(value, Gemma3L3L4ConditionalSpectralShadowExample):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("shadow examples must be strict examples or mappings")
    if set(value) != {"example_id", "family_id", "prompt"}:
        raise ValueError(
            "shadow example mappings require exactly example_id, family_id, "
            "and prompt"
        )
    return Gemma3L3L4ConditionalSpectralShadowExample(
        example_id=value["example_id"],  # type: ignore[arg-type]
        family_id=value["family_id"],  # type: ignore[arg-type]
        prompt=value["prompt"],  # type: ignore[arg-type]
    )


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(
        _PROMPT_DOMAIN + prompt.encode("utf-8", errors="strict")
    ).hexdigest()


def _manifest_sha256(
    rows: Iterable[tuple[str, str, str]],
) -> str:
    payload = json.dumps(
        sorted(rows),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(_MANIFEST_DOMAIN + payload).hexdigest()


def _oracle_receipt_sha256(payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(_ORACLE_RECEIPT_DOMAIN + serialized).hexdigest()


def _complete_h4_audit_receipt_sha256(
    payload: Mapping[str, object],
) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(
        _COMPLETE_H4_AUDIT_RECEIPT_DOMAIN + serialized
    ).hexdigest()


def _runtime_metadata(runtime: object) -> dict[str, object]:
    metadata_method = getattr(runtime, "metadata", None)
    if not callable(metadata_method):
        raise TypeError("runtime must expose metadata()")
    raw = metadata_method()
    if not isinstance(raw, Mapping):
        raise TypeError("runtime metadata must be a mapping")
    result: dict[str, object] = {}
    for name in _RUNTIME_BINDING_SCALAR_FIELDS:
        if name not in raw:
            continue
        value = raw[name]
        if value is None or isinstance(value, (str, bool, int)):
            result[name] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[name] = value
        else:
            raise ValueError(
                f"runtime metadata field {name!r} is not a finite scalar"
            )
    binding = _sha256(
        result.get("runtime_binding_sha256"),
        label="runtime binding",
    )
    result["runtime_binding_sha256"] = binding
    return result


def _tokenize_one(
    tokenizer: object,
    prompt: str,
    *,
    max_length: int,
    model_input_device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    batches = make_causal_lm_calibration_batches(
        tokenizer,
        (prompt,),
        max_length=max_length,
        tokenization_batch_size=1,
        device=model_input_device,
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
    if set(batch.model_inputs) != {"input_ids", "attention_mask"}:
        raise ValueError("tokenizer produced unexpected model inputs")
    model_inputs = dict(batch.model_inputs)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    targets = batch.targets
    if (
        input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or input_ids.dtype not in (torch.int32, torch.int64)
        or attention_mask.shape != input_ids.shape
        or attention_mask.dtype != torch.bool
        or targets.shape != input_ids.shape
        or targets.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("tokenizer produced invalid one-prompt tensors")
    supervised_indices = torch.nonzero(
        targets[0] != -100,
        as_tuple=False,
    ).flatten().to(device="cpu", dtype=torch.int64)
    if supervised_indices.numel() == 0:
        raise ValueError("prompt has no valid causal next-token boundary")
    supervised_targets = targets[0].index_select(
        0,
        supervised_indices.to(targets.device),
    ).to(device="cpu", dtype=torch.int64)
    return model_inputs, supervised_indices, supervised_targets


def _select_sequence_rows(value: Tensor, indices: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError("shadow tensors must have shape [1, sequence, width]")
    return value[0].index_select(0, indices.to(value.device))


def _result_tensors(
    runtime: object,
    result: object,
    *,
    model_inputs: Mapping[str, Tensor],
    supervised_indices: Tensor,
    supervised_targets: Tensor,
    expected_runtime_binding_sha256: str,
) -> dict[str, Tensor | int | str]:
    validate_result = getattr(runtime, "validate_result_binding", None)
    if not callable(validate_result):
        raise TypeError("runtime must expose validate_result_binding()")
    validate_result(result)
    if getattr(result, "arm", None) != "all_on":
        raise ValueError("development evaluation requires the all_on arm")
    result_binding = _sha256(
        getattr(result, "runtime_binding_sha256", None),
        label="result runtime binding",
    )
    if result_binding != expected_runtime_binding_sha256:
        raise ValueError("shadow result runtime binding differs")
    for name in (
        "model_inputs_sha256",
        "execution_grid_sha256",
        "result_artifact_sha256",
    ):
        _sha256(getattr(result, name, None), label=name)

    accounting = getattr(result, "accounting", None)
    if accounting is None or getattr(accounting, "model_forward_count", None) != 3:
        raise ValueError("every prompt must own exactly three model forwards")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    sequence_length = input_ids.shape[1]
    authoritative_logits = getattr(result, "authoritative_logits", None)
    candidate_logits = getattr(result, "candidate_logits", None)
    if (
        not isinstance(authoritative_logits, Tensor)
        or not isinstance(candidate_logits, Tensor)
        or authoritative_logits.shape != candidate_logits.shape
        or authoritative_logits.ndim != 3
        or authoritative_logits.shape[:2] != (1, sequence_length)
        or authoritative_logits.shape[2] < 2
        or not authoritative_logits.is_floating_point()
        or not candidate_logits.is_floating_point()
    ):
        raise ValueError("source and candidate logits must align [1, S, vocab]")
    if int(supervised_targets.max()) >= authoritative_logits.shape[2]:
        raise ValueError("supervised target lies outside the logits vocabulary")

    valid_target_mask = getattr(result, "valid_target_mask", None)
    source_eligible_mask = getattr(result, "source_eligible_mask", None)
    target_affected_mask = getattr(result, "target_affected_mask", None)
    for value, name in (
        (valid_target_mask, "valid_target_mask"),
        (source_eligible_mask, "source_eligible_mask"),
        (target_affected_mask, "target_affected_mask"),
    ):
        if (
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or value.shape != (1, sequence_length)
        ):
            raise ValueError(f"{name} must be boolean [1, sequence]")
    if not torch.equal(
        valid_target_mask.detach().to(device="cpu"),
        attention_mask.detach().to(device="cpu"),
    ):
        raise ValueError("runtime valid-target mask differs from tokenization")
    affected_cpu = target_affected_mask.detach().to(device="cpu")
    valid_cpu = valid_target_mask.detach().to(device="cpu")
    source_cpu = source_eligible_mask.detach().to(device="cpu")
    if bool((affected_cpu & ~valid_cpu).any()) or bool(
        (source_cpu & ~valid_cpu).any()
    ):
        raise ValueError("source or affected rows escape the valid mask")

    affected_indices = torch.nonzero(
        affected_cpu[0],
        as_tuple=False,
    ).flatten().to(dtype=torch.int64)
    if affected_indices.numel() == 0:
        raise ValueError("shadow prompt has no affected target rows")
    supervised_affected = affected_cpu[0].index_select(
        0,
        supervised_indices,
    )
    if not bool(supervised_affected.any()):
        raise ValueError("shadow prompt has no affected supervised token")

    authoritative_x4 = getattr(result, "authoritative_x4", None)
    candidate_x4 = getattr(result, "candidate_x4", None)
    reference_x4 = getattr(result, "reference_x4", None)
    predicted_modes = getattr(result, "predicted_target_modal_delta", None)
    if (
        not isinstance(authoritative_x4, Tensor)
        or not isinstance(candidate_x4, Tensor)
        or not isinstance(reference_x4, Tensor)
        or authoritative_x4.shape != candidate_x4.shape
        or authoritative_x4.shape != reference_x4.shape
        or authoritative_x4.ndim != 3
        or authoritative_x4.shape[:2] != (1, sequence_length)
        or authoritative_x4.shape[2] == 0
        or not authoritative_x4.is_floating_point()
        or not candidate_x4.is_floating_point()
        or not reference_x4.is_floating_point()
        or not isinstance(predicted_modes, Tensor)
        or predicted_modes.ndim != 3
        or predicted_modes.shape[:2] != (1, sequence_length)
        or predicted_modes.shape[2] == 0
        or not predicted_modes.is_floating_point()
    ):
        raise ValueError("shadow boundary or modal geometry differs")

    source_full = _select_sequence_rows(
        authoritative_x4 - reference_x4,
        affected_indices,
    )
    candidate_full = _select_sequence_rows(
        candidate_x4 - reference_x4,
        affected_indices,
    )
    encode_target_delta = getattr(runtime, "encode_target_delta", None)
    if not callable(encode_target_delta):
        raise TypeError("runtime must expose encode_target_delta()")
    source_modal = encode_target_delta(source_full)
    candidate_modal = _select_sequence_rows(predicted_modes, affected_indices)
    if (
        not isinstance(source_modal, Tensor)
        or source_modal.shape != candidate_modal.shape
        or source_modal.ndim != 2
    ):
        raise ValueError("encoded source and predicted target modes differ")

    source_logits = _select_sequence_rows(
        authoritative_logits,
        supervised_indices,
    )
    candidate_logits = _select_sequence_rows(
        candidate_logits,
        supervised_indices,
    )
    affected_supervised_indices = torch.nonzero(
        supervised_affected,
        as_tuple=False,
    ).flatten().to(dtype=torch.int64)
    return {
        "source_logits": source_logits,
        "candidate_logits": candidate_logits,
        "targets": supervised_targets,
        "affected_source_logits": source_logits.index_select(
            0,
            affected_supervised_indices.to(source_logits.device),
        ),
        "affected_candidate_logits": candidate_logits.index_select(
            0,
            affected_supervised_indices.to(candidate_logits.device),
        ),
        "affected_targets": supervised_targets.index_select(
            0,
            affected_supervised_indices,
        ),
        "affected_supervised_indices": affected_supervised_indices,
        "source_modal": source_modal,
        "candidate_modal": candidate_modal,
        "source_full": source_full,
        "candidate_full": candidate_full,
        "valid_rows": int(valid_cpu.sum().item()),
        "source_eligible_rows": int(source_cpu.sum().item()),
        "affected_rows": int(affected_cpu.sum().item()),
        "supervised_tokens": supervised_indices.numel(),
        "affected_supervised_tokens": affected_supervised_indices.numel(),
        "model_inputs_sha256": getattr(result, "model_inputs_sha256"),
        "execution_grid_sha256": getattr(result, "execution_grid_sha256"),
        "result_artifact_sha256": getattr(result, "result_artifact_sha256"),
        "local_factorized_linear_macs": int(
            getattr(accounting, "local_factorized_linear_macs", 0)
        ),
    }


def _add_coverage(
    coverage: _Coverage,
    measured: Mapping[str, Tensor | int | str],
    *,
    model_forward_count: int = 3,
) -> None:
    coverage.examples += 1
    coverage.valid_rows += int(measured["valid_rows"])
    coverage.source_eligible_rows += int(measured["source_eligible_rows"])
    coverage.affected_rows += int(measured["affected_rows"])
    coverage.supervised_tokens += int(measured["supervised_tokens"])
    coverage.affected_supervised_tokens += int(
        measured["affected_supervised_tokens"]
    )
    if type(model_forward_count) is not int or model_forward_count <= 0:
        raise ValueError("model_forward_count must be a positive integer")
    coverage.model_forwards += model_forward_count
    macs = measured["local_factorized_linear_macs"]
    if isinstance(macs, bool) or not isinstance(macs, int) or macs < 0:
        raise ValueError("local factorized linear MACs must be nonnegative")
    coverage.local_factorized_linear_macs += macs


def _oracle_suffix_metadata(
    oracle: object,
    *,
    role: str,
    injected_x4: Tensor,
    result: object,
    runtime_binding: Mapping[str, object],
) -> dict[str, object]:
    validate_integrity = getattr(oracle, "validate_integrity", None)
    validate_injected_x4 = getattr(oracle, "validate_injected_x4", None)
    metadata_method = getattr(oracle, "metadata", None)
    if (
        not callable(validate_integrity)
        or not callable(validate_injected_x4)
        or not callable(metadata_method)
    ):
        raise TypeError("oracle suffix must expose authenticated result methods")
    validate_integrity()
    validate_injected_x4(injected_x4)
    metadata = metadata_method()
    if not isinstance(metadata, Mapping):
        raise TypeError("oracle suffix metadata must be a mapping")
    normalized = dict(metadata)
    required = {
        "role",
        "execution_mode",
        "metrics_only",
        "serving_authorized",
        "model_forward_count",
        "injected_x4_sha256",
        "shadow_result_artifact_sha256",
        "runtime_binding_sha256",
        "execution_grid_sha256",
        "adapter_execution_sha256",
        "logits_sha256",
        "artifact_sha256",
    }
    if set(normalized) != required:
        raise ValueError("oracle suffix metadata fields differ")
    for name in (
        "injected_x4_sha256",
        "shadow_result_artifact_sha256",
        "runtime_binding_sha256",
        "execution_grid_sha256",
        "adapter_execution_sha256",
        "logits_sha256",
        "artifact_sha256",
    ):
        normalized[name] = _sha256(
            normalized[name],
            label=f"oracle suffix {name}",
        )
    if (
        normalized["role"] != role
        or normalized["execution_mode"] != "authenticated_oracle_suffix"
        or normalized["metrics_only"] is not True
        or normalized["serving_authorized"] is not False
        or normalized["model_forward_count"] != 1
        or normalized["runtime_binding_sha256"]
        != runtime_binding["runtime_binding_sha256"]
        or normalized["shadow_result_artifact_sha256"]
        != getattr(result, "result_artifact_sha256", None)
        or normalized["execution_grid_sha256"]
        != getattr(result, "execution_grid_sha256", None)
    ):
        raise ValueError("oracle suffix binding, role, or safety metadata differs")
    expected_execution = runtime_binding.get("adapter_execution_sha256")
    if (
        expected_execution is not None
        and normalized["adapter_execution_sha256"] != expected_execution
    ):
        raise ValueError("oracle suffix adapter execution binding differs")
    logits = getattr(oracle, "logits", None)
    authoritative_logits = getattr(result, "authoritative_logits", None)
    if (
        not isinstance(logits, Tensor)
        or not isinstance(authoritative_logits, Tensor)
        or logits.shape != authoritative_logits.shape
        or logits.dtype != authoritative_logits.dtype
        or logits.device != authoritative_logits.device
        or not logits.is_floating_point()
    ):
        raise ValueError("oracle suffix logits differ from the source geometry")
    validate_integrity()
    validate_injected_x4(injected_x4)
    return normalized


def _complete_h4_identity_audit_metadata(
    audit: object,
    *,
    result: object,
    runtime_binding: Mapping[str, object],
) -> dict[str, object]:
    validate_integrity = getattr(audit, "validate_integrity", None)
    validate_difference_mask = getattr(
        audit,
        "validate_incomplete_h4_difference_mask",
        None,
    )
    metadata_method = getattr(audit, "metadata", None)
    if (
        not callable(validate_integrity)
        or not callable(validate_difference_mask)
        or not callable(metadata_method)
    ):
        raise TypeError(
            "complete-H4 identity audit must expose authenticated result "
            "methods"
        )
    validate_integrity()
    difference_mask = getattr(audit, "incomplete_h4_difference_mask", None)
    if not isinstance(difference_mask, Tensor):
        raise TypeError(
            "complete-H4 identity audit must expose its difference mask"
        )
    validate_difference_mask(difference_mask)
    metadata = metadata_method()
    if not isinstance(metadata, Mapping):
        raise TypeError("complete-H4 identity audit metadata must be a mapping")
    normalized = dict(metadata)
    required = {
        "execution_mode",
        "metrics_only",
        "serving_authorized",
        "model_forward_count",
        "native_h4_sha256",
        "incomplete_carrier_h4_sha256",
        "injected_h4_sha256",
        "shadow_result_artifact_sha256",
        "runtime_binding_sha256",
        "model_inputs_sha256",
        "execution_grid_sha256",
        "adapter_execution_sha256",
        "target_affected_rows",
        "incomplete_h4_difference_mask_sha256",
        "incomplete_h4_difference_rows",
        "incomplete_h4_difference_valid_rows",
        "incomplete_h4_difference_padding_rows",
        "incomplete_h4_difference_target_rows",
        "incomplete_h4_difference_outside_target_rows",
        "target_affected_h4_difference_observed",
        "incomplete_h4_difference_nonvacuous",
        "boundary_callbacks_exactly_once",
        "boundary_callback_order",
        "complete_h4_logits_bitwise_authoritative",
        "complete_h4_max_abs_logit_error",
        "partial_exact_x4_logits_sha256",
        "complete_h4_logits_sha256",
        "artifact_sha256",
    }
    if set(normalized) != required:
        raise ValueError("complete-H4 identity audit metadata fields differ")
    hash_fields = (
        "native_h4_sha256",
        "incomplete_carrier_h4_sha256",
        "injected_h4_sha256",
        "shadow_result_artifact_sha256",
        "runtime_binding_sha256",
        "model_inputs_sha256",
        "execution_grid_sha256",
        "adapter_execution_sha256",
        "incomplete_h4_difference_mask_sha256",
        "partial_exact_x4_logits_sha256",
        "complete_h4_logits_sha256",
        "artifact_sha256",
    )
    for name in hash_fields:
        normalized[name] = _sha256(
            normalized[name],
            label=f"complete-H4 identity audit {name}",
        )
    valid_target_mask = getattr(result, "valid_target_mask", None)
    target_affected_mask = getattr(result, "target_affected_mask", None)
    if (
        not isinstance(valid_target_mask, Tensor)
        or not isinstance(target_affected_mask, Tensor)
        or valid_target_mask.dtype != torch.bool
        or target_affected_mask.dtype != torch.bool
        or valid_target_mask.shape != target_affected_mask.shape
        or difference_mask.dtype != torch.bool
        or difference_mask.shape != valid_target_mask.shape
        or difference_mask.device != valid_target_mask.device
        or target_affected_mask.device != valid_target_mask.device
    ):
        raise ValueError(
            "complete-H4 difference mask geometry differs from the shadow"
        )
    if bool((target_affected_mask & ~valid_target_mask).any()):
        raise ValueError("shadow target-affected rows escape the valid mask")
    expected_affected_rows = int(target_affected_mask.sum().item())
    difference_rows = int(difference_mask.sum().item())
    difference_valid_rows = int(
        (difference_mask & valid_target_mask).sum().item()
    )
    difference_padding_rows = int(
        (difference_mask & ~valid_target_mask).sum().item()
    )
    difference_target_rows = int(
        (difference_mask & target_affected_mask).sum().item()
    )
    difference_outside_target_rows = int(
        (difference_mask & ~target_affected_mask).sum().item()
    )
    expected_difference_counts = {
        "incomplete_h4_difference_rows": difference_rows,
        "incomplete_h4_difference_valid_rows": difference_valid_rows,
        "incomplete_h4_difference_padding_rows": difference_padding_rows,
        "incomplete_h4_difference_target_rows": difference_target_rows,
        "incomplete_h4_difference_outside_target_rows": (
            difference_outside_target_rows
        ),
    }
    if any(
        type(normalized[name]) is not int
        or normalized[name] != expected
        for name, expected in expected_difference_counts.items()
    ):
        raise ValueError(
            "complete-H4 identity audit difference-mask counts differ"
        )
    maximum_error = normalized["complete_h4_max_abs_logit_error"]
    bitwise_authoritative = normalized[
        "complete_h4_logits_bitwise_authoritative"
    ]
    if (
        normalized["execution_mode"]
        != "authenticated_complete_h4_identity_audit"
        or normalized["metrics_only"] is not True
        or normalized["serving_authorized"] is not False
        or normalized["model_forward_count"] != 3
        or normalized["runtime_binding_sha256"]
        != runtime_binding["runtime_binding_sha256"]
        or normalized["shadow_result_artifact_sha256"]
        != getattr(result, "result_artifact_sha256", None)
        or normalized["model_inputs_sha256"]
        != getattr(result, "model_inputs_sha256", None)
        or normalized["execution_grid_sha256"]
        != getattr(result, "execution_grid_sha256", None)
        or type(normalized["target_affected_rows"]) is not int
        or normalized["target_affected_rows"] != expected_affected_rows
        or expected_affected_rows <= 0
        or difference_rows <= 0
        or difference_target_rows <= 0
        or normalized["target_affected_h4_difference_observed"] is not True
        or normalized["incomplete_h4_difference_nonvacuous"] is not True
        or normalized["boundary_callbacks_exactly_once"] is not True
        or normalized["boundary_callback_order"]
        != _COMPLETE_H4_BOUNDARY_CALLBACK_ORDER
        or type(bitwise_authoritative) is not bool
        or isinstance(maximum_error, bool)
        or not isinstance(maximum_error, (int, float))
        or not math.isfinite(float(maximum_error))
        or float(maximum_error) < 0.0
        or normalized["native_h4_sha256"]
        != normalized["injected_h4_sha256"]
        or normalized["native_h4_sha256"]
        == normalized["incomplete_carrier_h4_sha256"]
    ):
        raise ValueError(
            "complete-H4 identity audit binding, callbacks, or safety "
            "metadata differs"
        )
    expected_execution = runtime_binding.get("adapter_execution_sha256")
    if (
        expected_execution is not None
        and normalized["adapter_execution_sha256"] != expected_execution
    ):
        raise ValueError(
            "complete-H4 identity audit adapter execution binding differs"
        )
    partial_logits = getattr(audit, "partial_exact_x4_logits", None)
    complete_logits = getattr(audit, "complete_h4_logits", None)
    authoritative_logits = getattr(result, "authoritative_logits", None)
    if (
        not isinstance(partial_logits, Tensor)
        or not isinstance(complete_logits, Tensor)
        or not isinstance(authoritative_logits, Tensor)
        or partial_logits.shape != authoritative_logits.shape
        or complete_logits.shape != authoritative_logits.shape
        or partial_logits.dtype != authoritative_logits.dtype
        or complete_logits.dtype != authoritative_logits.dtype
        or partial_logits.device != authoritative_logits.device
        or complete_logits.device != authoritative_logits.device
        or not partial_logits.is_floating_point()
        or not complete_logits.is_floating_point()
        or not bool(torch.isfinite(partial_logits).all())
        or not bool(torch.isfinite(complete_logits).all())
    ):
        raise ValueError(
            "complete-H4 identity audit logits differ from source geometry"
        )
    logits_are_bitwise_authoritative = torch.equal(
        complete_logits.contiguous().view(torch.uint8),
        authoritative_logits.contiguous().view(torch.uint8),
    )
    observed_maximum_error = float(
        (
            complete_logits.detach().to(dtype=torch.float64)
            - authoritative_logits.detach().to(dtype=torch.float64)
        )
        .abs()
        .max()
    )
    if (
        bitwise_authoritative is not logits_are_bitwise_authoritative
        or float(maximum_error) != observed_maximum_error
    ):
        raise ValueError(
            "complete-H4 identity audit outcome metadata differs from logits"
        )
    validate_integrity()
    validate_difference_mask(difference_mask)
    return normalized


def _scalar_report(value: object, *, path: str = "report") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite scalar")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _scalar_report(child, path=f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _scalar_report(child, path=f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains non-scalar data {type(value).__name__}")


def evaluate_gemma3_l3_l4_conditional_spectral_development_shadow(
    *,
    runtime: object,
    adapter: object,
    tokenizer: object,
    examples: Iterable[ShadowExampleInput],
    max_length: int,
    model_input_device: torch.device | str = "cpu",
    gates: ShadowFidelityGates = ESTABLISHED_SHADOW_FIDELITY_GATES,
    vocab_chunk_size: int = 16_384,
    tokenizer_integrity_check: Callable[[str], None] | None = None,
    include_oracle_suffixes: bool = False,
    include_complete_h4_identity_audit: bool = False,
) -> dict[str, object]:
    """Run and reduce a family-disjoint all-on shadow panel.

    Exactly one prompt's full-vocabulary logits and boundary tensors are live
    at a time.  Both behavioral views are streamed through
    :mod:`fisher_graph.shadow_fidelity`; only scalar statistics and
    authenticated hashes survive in the returned report.
    """

    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if not isinstance(gates, ShadowFidelityGates):
        raise TypeError("gates must be a ShadowFidelityGates instance")
    if (
        isinstance(vocab_chunk_size, bool)
        or not isinstance(vocab_chunk_size, int)
        or vocab_chunk_size <= 0
    ):
        raise ValueError("vocab_chunk_size must be a positive integer")
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")
    if (
        tokenizer_integrity_check is not None
        and not callable(tokenizer_integrity_check)
    ):
        raise TypeError("tokenizer_integrity_check must be callable")
    if type(include_oracle_suffixes) is not bool:
        raise TypeError("include_oracle_suffixes must be boolean")
    if type(include_complete_h4_identity_audit) is not bool:
        raise TypeError("include_complete_h4_identity_audit must be boolean")
    if include_oracle_suffixes and include_complete_h4_identity_audit:
        raise ValueError(
            "oracle suffixes and the complete-H4 identity audit are "
            "mutually exclusive"
        )
    device = torch.device(model_input_device)

    materialized = tuple(_coerce_example(value) for value in examples)
    if not materialized:
        raise ValueError("development shadow examples cannot be empty")
    manifest: dict[str, str] = {}
    prompt_identities: dict[str, str] = {}
    for example in materialized:
        if example.example_id in manifest:
            raise ValueError(f"duplicate shadow example: {example.example_id!r}")
        manifest[example.example_id] = example.family_id
        prompt_identities[example.example_id] = _prompt_sha256(example.prompt)

    validate_runtime = getattr(runtime, "validate_integrity", None)
    if not callable(validate_runtime):
        raise TypeError("runtime must expose validate_integrity()")
    validate_runtime()
    runtime_binding = _runtime_metadata(runtime)
    binding_sha256 = str(runtime_binding["runtime_binding_sha256"])
    behavioral = SourceAuthoritativeShadowFidelityAccumulator(
        manifest,
        gates=gates,
        vocab_chunk_size=vocab_chunk_size,
    )
    affected_behavioral = SourceAuthoritativeShadowFidelityAccumulator(
        manifest,
        gates=gates,
        vocab_chunk_size=vocab_chunk_size,
    )
    projection_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_oracle_suffixes
        else None
    )
    projection_affected_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_oracle_suffixes
        else None
    )
    carrier_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_oracle_suffixes
        else None
    )
    carrier_affected_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_oracle_suffixes
        else None
    )
    partial_exact_x4_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_complete_h4_identity_audit
        else None
    )
    partial_exact_x4_affected_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_complete_h4_identity_audit
        else None
    )
    complete_h4_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_complete_h4_identity_audit
        else None
    )
    complete_h4_affected_behavioral = (
        SourceAuthoritativeShadowFidelityAccumulator(
            manifest,
            gates=gates,
            vocab_chunk_size=vocab_chunk_size,
        )
        if include_complete_h4_identity_audit
        else None
    )
    projection_full_width = _VectorMoments()
    pooled_coverage = _Coverage()
    pooled_modal = _VectorMoments()
    pooled_full_width = _VectorMoments()
    families = {
        family_id: _FamilyState()
        for family_id in sorted(set(manifest.values()))
    }
    receipts: list[dict[str, object]] = []
    oracle_receipts: list[dict[str, object]] = []
    complete_h4_audit_receipts: list[dict[str, object]] = []
    model_forwards_per_prompt = (
        6
        if include_complete_h4_identity_audit
        else 5
        if include_oracle_suffixes
        else 3
    )

    # Lexical execution makes a report independent of caller iteration order.
    for example in sorted(materialized, key=lambda value: value.example_id):
        if tokenizer_integrity_check is not None:
            tokenizer_integrity_check("before")
        model_inputs, supervised_indices, supervised_targets = _tokenize_one(
            tokenizer,
            example.prompt,
            max_length=max_length,
            model_input_device=device,
        )
        if tokenizer_integrity_check is not None:
            tokenizer_integrity_check("after")
        execute = getattr(runtime, "execute_model_shadow", None)
        if not callable(execute):
            raise TypeError("runtime must expose execute_model_shadow()")
        with torch.inference_mode():
            result = execute(adapter, model_inputs, arm="all_on")
        measured = _result_tensors(
            runtime,
            result,
            model_inputs=model_inputs,
            supervised_indices=supervised_indices,
            supervised_targets=supervised_targets,
            expected_runtime_binding_sha256=binding_sha256,
        )
        if include_oracle_suffixes:
            execute_oracle = getattr(runtime, "execute_oracle_suffix", None)
            if not callable(execute_oracle):
                raise TypeError("runtime must expose execute_oracle_suffix()")
            injections = (
                prepare_gemma3_l3_l4_graph_organized_svd_oracle_injections(
                    runtime,
                    result,
                )
            )
            if (
                getattr(injections, "runtime_binding_sha256", None)
                != binding_sha256
                or getattr(
                    injections,
                    "shadow_result_artifact_sha256",
                    None,
                )
                != measured["result_artifact_sha256"]
                or getattr(injections, "execution_grid_sha256", None)
                != measured["execution_grid_sha256"]
            ):
                raise ValueError("oracle injection binding differs from shadow")
            authoritative_x4 = getattr(result, "authoritative_x4", None)
            if (
                not isinstance(authoritative_x4, Tensor)
                or not torch.equal(injections.carrier_x4, authoritative_x4)
            ):
                raise RuntimeError("exact-X4 carrier injection differs from source")
            projection_oracle = execute_oracle(
                adapter,
                model_inputs,
                result,
                injections.projection_x4,
                role="projection_64",
            )
            carrier_oracle = execute_oracle(
                adapter,
                model_inputs,
                result,
                injections.carrier_x4,
                role="exact_x4_carrier",
            )
            projection_metadata = _oracle_suffix_metadata(
                projection_oracle,
                role="projection_64",
                injected_x4=injections.projection_x4,
                result=result,
                runtime_binding=runtime_binding,
            )
            carrier_metadata = _oracle_suffix_metadata(
                carrier_oracle,
                role="exact_x4_carrier",
                injected_x4=injections.carrier_x4,
                result=result,
                runtime_binding=runtime_binding,
            )
            affected_supervised_indices = measured[
                "affected_supervised_indices"
            ]
            if not isinstance(affected_supervised_indices, Tensor):
                raise TypeError("affected supervised indices must be a Tensor")
            projection_logits = _select_sequence_rows(
                projection_oracle.logits,
                supervised_indices,
            )
            carrier_logits = _select_sequence_rows(
                carrier_oracle.logits,
                supervised_indices,
            )
            source_logits = measured["source_logits"]
            affected_source_logits = measured["affected_source_logits"]
            targets = measured["targets"]
            affected_targets = measured["affected_targets"]
            if not all(
                isinstance(value, Tensor)
                for value in (
                    source_logits,
                    affected_source_logits,
                    targets,
                    affected_targets,
                )
            ):
                raise TypeError("measured shadow logits and targets must be Tensors")
            affected_projection_logits = projection_logits.index_select(
                0,
                affected_supervised_indices.to(projection_logits.device),
            )
            affected_carrier_logits = carrier_logits.index_select(
                0,
                affected_supervised_indices.to(carrier_logits.device),
            )
            assert projection_behavioral is not None
            assert projection_affected_behavioral is not None
            assert carrier_behavioral is not None
            assert carrier_affected_behavioral is not None
            projection_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=source_logits,
                    candidate_logits=projection_logits,
                    targets=targets,
                )
            )
            projection_affected_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=affected_source_logits,
                    candidate_logits=affected_projection_logits,
                    targets=affected_targets,
                )
            )
            carrier_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=source_logits,
                    candidate_logits=carrier_logits,
                    targets=targets,
                )
            )
            carrier_affected_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=affected_source_logits,
                    candidate_logits=affected_carrier_logits,
                    targets=affected_targets,
                )
            )
            affected_rows = getattr(result, "target_affected_mask")[0]
            projection_full_width.add(
                injections.source_target_full_width_delta[0][affected_rows],
                injections.projection_target_full_width_delta[0][affected_rows],
            )
            oracle_receipt_payload = {
                "example_id": example.example_id,
                "family_id": example.family_id,
                "prompt_sha256": prompt_identities[example.example_id],
                "model_inputs_sha256": measured["model_inputs_sha256"],
                "execution_grid_sha256": measured["execution_grid_sha256"],
                "shadow_result_artifact_sha256": measured[
                    "result_artifact_sha256"
                ],
                "execution_order": _ORACLE_SUFFIX_ORDER,
                "model_forward_count": 5,
                "projection_64": projection_metadata,
                "exact_x4_carrier": carrier_metadata,
            }
            oracle_receipts.append(
                {
                    **oracle_receipt_payload,
                    "oracle_suffix_receipt_sha256": _oracle_receipt_sha256(
                        oracle_receipt_payload
                    ),
                }
            )
        elif include_complete_h4_identity_audit:
            execute_audit = getattr(
                runtime,
                "execute_complete_h4_identity_audit",
                None,
            )
            if not callable(execute_audit):
                raise TypeError(
                    "runtime must expose "
                    "execute_complete_h4_identity_audit()"
                )
            with torch.inference_mode():
                audit = execute_audit(adapter, model_inputs, result)
            audit_metadata = _complete_h4_identity_audit_metadata(
                audit,
                result=result,
                runtime_binding=runtime_binding,
            )
            affected_supervised_indices = measured[
                "affected_supervised_indices"
            ]
            source_logits = measured["source_logits"]
            affected_source_logits = measured["affected_source_logits"]
            targets = measured["targets"]
            affected_targets = measured["affected_targets"]
            if (
                not isinstance(affected_supervised_indices, Tensor)
                or not isinstance(source_logits, Tensor)
                or not isinstance(affected_source_logits, Tensor)
                or not isinstance(targets, Tensor)
                or not isinstance(affected_targets, Tensor)
            ):
                raise TypeError(
                    "measured complete-H4 audit rows must be Tensors"
                )
            partial_logits = _select_sequence_rows(
                audit.partial_exact_x4_logits,
                supervised_indices,
            )
            complete_logits = _select_sequence_rows(
                audit.complete_h4_logits,
                supervised_indices,
            )
            affected_partial_logits = partial_logits.index_select(
                0,
                affected_supervised_indices.to(partial_logits.device),
            )
            affected_complete_logits = complete_logits.index_select(
                0,
                affected_supervised_indices.to(complete_logits.device),
            )
            assert partial_exact_x4_behavioral is not None
            assert partial_exact_x4_affected_behavioral is not None
            assert complete_h4_behavioral is not None
            assert complete_h4_affected_behavioral is not None
            partial_exact_x4_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=source_logits,
                    candidate_logits=partial_logits,
                    targets=targets,
                )
            )
            partial_exact_x4_affected_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=affected_source_logits,
                    candidate_logits=affected_partial_logits,
                    targets=affected_targets,
                )
            )
            complete_h4_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=source_logits,
                    candidate_logits=complete_logits,
                    targets=targets,
                )
            )
            complete_h4_affected_behavioral.add(
                ShadowFidelityExample(
                    example_id=example.example_id,
                    family_id=example.family_id,
                    source_logits=affected_source_logits,
                    candidate_logits=affected_complete_logits,
                    targets=affected_targets,
                )
            )
            complete_h4_audit_receipt_payload = {
                "example_id": example.example_id,
                "family_id": example.family_id,
                "prompt_sha256": prompt_identities[example.example_id],
                "model_inputs_sha256": measured["model_inputs_sha256"],
                "execution_grid_sha256": measured["execution_grid_sha256"],
                "shadow_result_artifact_sha256": measured[
                    "result_artifact_sha256"
                ],
                "audit": audit_metadata,
            }
            complete_h4_audit_receipts.append(
                {
                    **complete_h4_audit_receipt_payload,
                    "complete_h4_audit_receipt_sha256": (
                        _complete_h4_audit_receipt_sha256(
                            complete_h4_audit_receipt_payload
                        )
                    ),
                }
            )
            # Function locals otherwise keep the previous prompt's audit-owned
            # full-vocabulary logits and their selected rows alive until this
            # branch is reached again.  Release them before the next prompt's
            # three-pass shadow can begin; only scalar reductions and hashes
            # survive in the accumulators and receipt above.
            del (
                audit,
                audit_metadata,
                affected_supervised_indices,
                source_logits,
                affected_source_logits,
                targets,
                affected_targets,
                partial_logits,
                complete_logits,
                affected_partial_logits,
                affected_complete_logits,
                complete_h4_audit_receipt_payload,
            )
        behavioral.add(
            ShadowFidelityExample(
                example_id=example.example_id,
                family_id=example.family_id,
                source_logits=measured["source_logits"],  # type: ignore[arg-type]
                candidate_logits=measured["candidate_logits"],  # type: ignore[arg-type]
                targets=measured["targets"],  # type: ignore[arg-type]
            )
        )
        affected_behavioral.add(
            ShadowFidelityExample(
                example_id=example.example_id,
                family_id=example.family_id,
                source_logits=measured[  # type: ignore[arg-type]
                    "affected_source_logits"
                ],
                candidate_logits=measured[  # type: ignore[arg-type]
                    "affected_candidate_logits"
                ],
                targets=measured["affected_targets"],  # type: ignore[arg-type]
            )
        )
        pooled_modal.add(
            measured["source_modal"],  # type: ignore[arg-type]
            measured["candidate_modal"],  # type: ignore[arg-type]
        )
        pooled_full_width.add(
            measured["source_full"],  # type: ignore[arg-type]
            measured["candidate_full"],  # type: ignore[arg-type]
        )
        family = families[example.family_id]
        family.modal.add(
            measured["source_modal"],  # type: ignore[arg-type]
            measured["candidate_modal"],  # type: ignore[arg-type]
        )
        family.full_width.add(
            measured["source_full"],  # type: ignore[arg-type]
            measured["candidate_full"],  # type: ignore[arg-type]
        )
        _add_coverage(
            pooled_coverage,
            measured,
            model_forward_count=model_forwards_per_prompt,
        )
        _add_coverage(
            family.coverage,
            measured,
            model_forward_count=model_forwards_per_prompt,
        )
        receipts.append(
            {
                "example_id": example.example_id,
                "family_id": example.family_id,
                "prompt_sha256": prompt_identities[example.example_id],
                "tokenized_tokens": int(
                    model_inputs["attention_mask"].sum().item()
                ),
                "supervised_tokens": measured["supervised_tokens"],
                "affected_supervised_tokens": measured[
                    "affected_supervised_tokens"
                ],
                "model_forward_count": model_forwards_per_prompt,
                "model_inputs_sha256": measured["model_inputs_sha256"],
                "execution_grid_sha256": measured[
                    "execution_grid_sha256"
                ],
                "result_artifact_sha256": measured[
                    "result_artifact_sha256"
                ],
            }
        )
        del measured, result, model_inputs

    validate_runtime()
    family_rows = tuple(
        {
            "family_id": family_id,
            "coverage": state.coverage.summary(),
            "target_modal": state.modal.summary(),
            "full_width_boundary": state.full_width.summary(),
        }
        for family_id, state in sorted(families.items())
    )
    report: dict[str, object] = {
        "schema": _REPORT_SCHEMA,
        "format_version": 1,
        "semantics": {
            "execution_mode": "development_shadow",
            "arm": "all_on",
            "authoritative_path": "source",
            "source_outputs_authoritative": True,
            "candidate_outputs_authoritative": False,
            "candidate_outputs_must_not_be_served": True,
            "candidate_logits_used_for_metrics_only": True,
            "candidate_boundaries_used_for_metrics_only": True,
            "behavioral_scope": "all_valid_supervised_tokens",
            "affected_behavioral_scope": (
                "causally_affected_supervised_tokens_only"
            ),
            "boundary_scope": "target_affected_rows_only",
            "calibration_b_protocol_used": False,
            "qualification_ledger_used": False,
            "tokenizer_integrity_checked_per_prompt": (
                tokenizer_integrity_check is not None
            ),
        },
        "runtime_binding": runtime_binding,
        "manifest": {
            "manifest_sha256": _manifest_sha256(
                (
                    example.example_id,
                    example.family_id,
                    prompt_identities[example.example_id],
                )
                for example in materialized
            ),
            "example_count": len(materialized),
            "family_count": len(families),
            "strict_example_membership": True,
            "strict_family_membership": True,
            "prompt_text_retained": False,
            "token_ids_retained": False,
        },
        "execution": {
            "prompt_streaming": True,
            "one_prompt_live_at_a_time": True,
            "model_forwards_per_prompt": model_forwards_per_prompt,
            "total_model_forward_count": pooled_coverage.model_forwards,
            "full_vocabulary_materialized_one_prompt_at_a_time": True,
        },
        "behavioral": behavioral.finalize(),
        "affected_behavioral": affected_behavioral.finalize(),
        "coverage": pooled_coverage.summary(),
        "target_modal": {
            "scope": "target_affected_rows_only",
            "pooled": pooled_modal.summary(),
        },
        "full_width_boundary": {
            "scope": "target_affected_rows_only",
            "pooled": pooled_full_width.summary(),
        },
        "families": family_rows,
        "receipts": tuple(receipts),
        "safety": {
            "scalar_only_report": True,
            "prompt_text_retained": False,
            "token_ids_retained": False,
            "logits_retained": False,
            "activations_retained": False,
            "candidate_serving_authorized": False,
        },
    }
    if include_oracle_suffixes:
        assert projection_behavioral is not None
        assert projection_affected_behavioral is not None
        assert carrier_behavioral is not None
        assert carrier_affected_behavioral is not None
        report_semantics = report["semantics"]
        if not isinstance(report_semantics, dict):
            raise RuntimeError("development shadow semantics must be an object")
        report_semantics["oracle_suffixes_metrics_only"] = True
        report["oracle_suffixes"] = {
            "semantics": {
                "execution_order": _ORACLE_SUFFIX_ORDER,
                "truth_leaking_analysis_controls": True,
                "candidate_source_rank_and_projection_target_width_are_distinct": (
                    True
                ),
                "source_outputs_authoritative": True,
                "oracle_outputs_must_not_be_served": True,
            },
            "projection_64": {
                "role": "true_target_modes_decoded_through_authenticated_dual",
                "behavioral": projection_behavioral.finalize(),
                "affected_behavioral": (
                    projection_affected_behavioral.finalize()
                ),
                "full_width_boundary": {
                    "scope": "target_affected_rows_only",
                    "pooled": projection_full_width.summary(),
                },
            },
            "exact_x4_carrier": {
                "role": "exact_full_width_x4_on_clamped_y3_carrier",
                "behavioral": carrier_behavioral.finalize(),
                "affected_behavioral": carrier_affected_behavioral.finalize(),
                "injected_boundary_equals_authoritative_x4": True,
            },
            "execution": {
                "oracle_forwards_per_prompt": 2,
                "total_oracle_model_forward_count": 2 * len(materialized),
                "total_fused_model_forward_count": (
                    model_forwards_per_prompt * len(materialized)
                ),
            },
            "receipts": tuple(oracle_receipts),
        }
    if include_complete_h4_identity_audit:
        assert partial_exact_x4_behavioral is not None
        assert partial_exact_x4_affected_behavioral is not None
        assert complete_h4_behavioral is not None
        assert complete_h4_affected_behavioral is not None
        report_semantics = report["semantics"]
        if not isinstance(report_semantics, dict):
            raise RuntimeError("development shadow semantics must be an object")
        report_semantics["complete_h4_identity_audit_metrics_only"] = True
        report["complete_h4_identity_audit"] = {
            "semantics": {
                "execution_order": _COMPLETE_H4_AUDIT_ORDER,
                "truth_leaking_identity_control": True,
                "source_outputs_authoritative": True,
                "audit_outputs_must_not_be_served": True,
                "complete_boundary": "layer.4.output",
                "graph_target_affected_mask_semantics": (
                    "finite_lag_prediction_support"
                ),
                "observed_h4_difference_mask_semantics": (
                    "bitwise_full_row_native_vs_incomplete_carrier_support"
                ),
                "graph_target_support_is_distinct_from_observed_h4_"
                "difference_support": True,
                "outside_graph_target_difference_is_not_integrity_failure": (
                    True
                ),
            },
            "partial_exact_x4_replay": {
                "role": "exact_native_x4_on_incomplete_clamped_y3_carrier",
                "behavioral": partial_exact_x4_behavioral.finalize(),
                "affected_behavioral": (
                    partial_exact_x4_affected_behavioral.finalize()
                ),
            },
            "complete_h4_identity": {
                "role": "exact_native_h4_at_complete_layer4_output",
                "behavioral": complete_h4_behavioral.finalize(),
                "affected_behavioral": (
                    complete_h4_affected_behavioral.finalize()
                ),
                "complete_h4_logits_bitwise_authoritative": all(
                    receipt["audit"][  # type: ignore[index]
                        "complete_h4_logits_bitwise_authoritative"
                    ]
                    is True
                    for receipt in complete_h4_audit_receipts
                ),
                "complete_h4_max_abs_logit_error": max(
                    float(
                        receipt["audit"][  # type: ignore[index]
                            "complete_h4_max_abs_logit_error"
                        ]
                    )
                    for receipt in complete_h4_audit_receipts
                ),
            },
            "execution": {
                "audit_forwards_per_prompt": 3,
                "total_audit_model_forward_count": 3 * len(materialized),
                "total_fused_model_forward_count": (
                    model_forwards_per_prompt * len(materialized)
                ),
            },
            "receipts": tuple(complete_h4_audit_receipts),
        }
    _scalar_report(report)
    return report


__all__ = [
    "Gemma3L3L4ConditionalSpectralShadowExample",
    "ShadowExampleInput",
    "evaluate_gemma3_l3_l4_conditional_spectral_development_shadow",
]
