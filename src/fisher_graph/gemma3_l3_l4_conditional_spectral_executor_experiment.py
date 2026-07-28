"""Compile and assess the frozen Gemma L3-to-L4 spectral generator.

This development rung has two deliberately separate entry points:

* compilation may read only the pinned five-origin interior measurement and
  uses origins 8/24/40 for fitting and 16/32 for selection;
* assessment may read only an already-frozen candidate and one separately
  measured single-origin spectral artifact.

The compiled object predicts fixed-reference modal deltas.  It does not
provide the prompt-conditioned base/reference state required to replace a
Gemma block, and it makes no NLL, task-fidelity, compression, or speed claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from torch import Tensor

from .external_models import find_git_worktree
from .modal_spectral_mapping import ModalSpectralMapping


__all__ = [
    "DEFAULT_ASSESSMENT_ORIGIN",
    "DEFAULT_INTERIOR_ARTIFACT",
    "DEFAULT_INTERIOR_ARTIFACT_SHA256",
    "DEFAULT_INTERIOR_REPORT_SHA256",
    "DEFAULT_OUTPUT",
    "Gemma3ConditionalSpectralCandidate",
    "Gemma3SpectralSource",
    "assess_gemma3_l3_l4_conditional_spectral_executor",
    "build_parser",
    "compile_gemma3_l3_l4_conditional_spectral_executor",
    "compile_gemma3_conditional_spectral_candidate",
    "load_gemma3_conditional_spectral_candidate",
    "load_gemma3_spectral_source",
    "main",
]


DEFAULT_INTERIOR_ARTIFACT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-spectral-map-interior-dev-v1.pt"
)
DEFAULT_INTERIOR_ARTIFACT_SHA256 = (
    "a80b9ce1a5e433724e74cb7c29143d18442805a7b05fcb419ede6ad1e23686b3"
)
DEFAULT_INTERIOR_REPORT_SHA256 = (
    "a3330bcd75c637811be62dd33a53b6a2329edf66d586b73f157176400462e7b5"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-conditional-spectral-executor-dev-v1.pt"
)
DEFAULT_ASSESSMENT_ORIGIN = 20

FIT_ORIGINS = (8, 24, 40)
SELECTION_ORIGINS = (16, 32)
INTERIOR_ORIGINS = (8, 16, 24, 32, 40)
LINEAR_RANK_PAIRS = (
    (8, 8),
    (12, 10),
    (16, 14),
    (18, 16),
    (20, 18),
    (24, 20),
    (32, 24),
    (40, 32),
    (64, 64),
)
QUADRATIC_RANK_PAIRS = (
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (4, 6),
    (6, 4),
    (6, 6),
    (8, 8),
    (12, 10),
)

LINEAR_MAX_MACRO_RELATIVE_ERROR = 0.20
LINEAR_MAX_WORST_RELATIVE_ERROR = 0.20
LINEAR_MIN_WORST_COSINE = 0.98
QUADRATIC_MIN_FIT_EVEN_ENERGY = 0.85
QUADRATIC_MIN_WORST_FINITE_ERROR_REDUCTION = 0.10

_SOURCE_SCHEMA = "fisher_graph.gemma3_l3_l4_spectral_mapping_development"
_CANDIDATE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_conditional_spectral_executor_development"
)
_ASSESSMENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_conditional_spectral_assessment_development"
)
_FORMAT_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-conditional-spectral-candidate:v1\0"
)
_REPORT_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-conditional-spectral-report:v1\0"
)
_SOURCE_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-spectral-report:v1\0"

_EXPECTED_SOURCE_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_structural_reference_rows": True,
    "contains_spectral_response_tensors": True,
    "artifact_must_remain_outside_git": True,
}


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields mismatch: expected {sorted(expected)}, "
            f"got {sorted(value)}"
        )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(width) for width in tensor.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_sha256(value: object, *, domain: bytes) -> str:
    return hashlib.sha256(domain + _canonical_json_bytes(value)).hexdigest()


def _canonical_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
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
        raise ValueError(f"{label} must be a finite nonempty rank-{ndim} tensor")
    return result


def _origin_ordinals(
    available: Sequence[int],
    requested: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    available_tuple = tuple(available)
    requested_tuple = tuple(requested)
    if (
        not requested_tuple
        or tuple(sorted(set(requested_tuple))) != requested_tuple
        or not set(requested_tuple).issubset(available_tuple)
    ):
        raise ValueError(f"{label} are not a strict subset of source origins")
    return tuple(available_tuple.index(origin) for origin in requested_tuple)


def _validate_compile_split(origins: Sequence[int]) -> None:
    values = tuple(origins)
    if values != INTERIOR_ORIGINS:
        raise ValueError("compile source must contain only the frozen origins")
    if set(FIT_ORIGINS) & set(SELECTION_ORIGINS):
        raise RuntimeError("compile fit and selection origins overlap")
    if set(FIT_ORIGINS) | set(SELECTION_ORIGINS) != set(values):
        raise RuntimeError("compile split does not exhaust the source origins")
    if DEFAULT_ASSESSMENT_ORIGIN in values:
        raise RuntimeError("assessment origin leaked into compilation")


@dataclass(frozen=True, slots=True)
class Gemma3SpectralSource:
    """Strict-loaded spectral response tensors and their source binding."""

    file_sha256: str
    report_file_sha256: str
    report_payload_sha256: str
    binding: Mapping[str, object]
    model: Mapping[str, object]
    protocol: Mapping[str, object]
    mapping: ModalSpectralMapping
    source_mode_standard_deviations: Tensor

    def __post_init__(self) -> None:
        _sha256(self.file_sha256, label="source file")
        _sha256(self.report_file_sha256, label="source report file")
        _sha256(self.report_payload_sha256, label="source report payload")
        if not isinstance(self.mapping, ModalSpectralMapping):
            raise TypeError("mapping must be a ModalSpectralMapping")
        self.mapping.validate_integrity()
        sigma = _canonical_tensor(
            self.source_mode_standard_deviations,
            label="source mode standard deviations",
            ndim=1,
        )
        if (
            sigma.numel() != self.mapping.source_rank
            or bool((sigma <= 0).any())
        ):
            raise ValueError("source standard deviations do not match mapping")
        for name in ("binding", "model", "protocol"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(value))
        object.__setattr__(self, "source_mode_standard_deviations", sigma)


def load_gemma3_spectral_source(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
    expected_origins: Sequence[int],
    expected_binding: Mapping[str, object] | None = None,
) -> Gemma3SpectralSource:
    """Strict-load one prompt-free spectral measurement artifact."""

    source = Path(path)
    expected_digest = _sha256(
        expected_file_sha256,
        label="expected source file",
    )
    actual_digest = _file_sha256(source)
    if actual_digest != expected_digest:
        raise ValueError("spectral source file differs from expected SHA-256")
    expected_report_digest = _sha256(
        expected_report_sha256,
        label="expected source report payload",
    )
    report_path = source.with_suffix(".json")
    with report_path.open("r", encoding="utf-8") as handle:
        report_raw = json.load(handle)
    report = _mapping(report_raw, label="spectral source report")
    claimed_report_digest = _sha256(
        report.get("report_sha256"),
        label="source report payload",
    )
    report_payload = dict(report)
    report_payload.pop("report_sha256")
    if (
        claimed_report_digest != expected_report_digest
        or _json_sha256(report_payload, domain=_SOURCE_REPORT_DOMAIN)
        != claimed_report_digest
    ):
        raise ValueError("spectral source report payload hash mismatch")
    report_artifact = _mapping(
        report.get("artifact"),
        label="source report artifact",
    )
    if report_artifact.get("tensor_file_sha256") != actual_digest:
        raise ValueError("source report does not bind the tensor file")
    report_file_digest = _file_sha256(report_path)
    raw = torch.load(source, map_location="cpu", weights_only=True)
    state = _mapping(raw, label="spectral source")
    _strict_keys(
        state,
        expected={
            "schema",
            "format_version",
            "scientific_status",
            "binding",
            "model",
            "protocol",
            "canonical_reference",
            "spectral_mapping",
            "safe_analysis",
            "safety",
        },
        label="spectral source",
    )
    if state["schema"] != _SOURCE_SCHEMA or state["format_version"] != 1:
        raise ValueError("spectral source schema or version drifted")
    safety = _mapping(state["safety"], label="source safety")
    if dict(safety) != _EXPECTED_SOURCE_SAFETY:
        raise ValueError("spectral source safety declaration drifted")
    status = _mapping(state["scientific_status"], label="source status")
    for field in (
        "shift_invariant_convolution_claim",
        "semantic_equivalence_claim",
        "prompt_distribution_fidelity_claim",
        "compression_claim",
        "latency_or_speed_claim",
        "cached_decode_claim",
    ):
        if status.get(field) is not False:
            raise ValueError("spectral source overclaims scientific status")
    if (
        status.get("fixed_reference_interventional_causal_influence") is not True
        or status.get("development_only") is not True
    ):
        raise ValueError("spectral source scientific scope drifted")
    binding = dict(_mapping(state["binding"], label="source binding"))
    if _canonical_json_bytes(
        _mapping(report.get("binding"), label="report binding")
    ) != _canonical_json_bytes(binding):
        raise ValueError("spectral source report binding drifted")
    if expected_binding is not None and binding != dict(expected_binding):
        raise ValueError("assessment source binding differs from candidate")
    model = dict(_mapping(state["model"], label="source model"))
    if (
        model.get("tokenizer_loaded") is not False
        or model.get("local_files_only") is not True
        or model.get("source_model_sha256")
        != binding.get("source_model_sha256")
    ):
        raise ValueError("spectral source model binding drifted")
    protocol = dict(_mapping(state["protocol"], label="source protocol"))
    if _canonical_json_bytes(
        _mapping(report.get("protocol"), label="report protocol")
    ) != _canonical_json_bytes(protocol):
        raise ValueError("spectral source report protocol drifted")
    mapping = ModalSpectralMapping.from_state_dict(
        _mapping(state["spectral_mapping"], label="spectral mapping")
    )
    origins = tuple(expected_origins)
    if (
        mapping.impulse_logical_positions != origins
        or tuple(protocol.get("impulse_logical_positions", ())) != origins
        or protocol.get("modal_rank") != 64
        or tuple(protocol.get("source_mode_indices", ())) != tuple(range(64))
        or mapping.source_rank != 64
        or mapping.target_rank != 64
        or mapping.source_mode_indices != tuple(range(64))
        or protocol.get("new_prompt_text_loaded") is not False
        or protocol.get("new_token_ids_loaded") is not False
        or protocol.get("prefill_only") is not True
        or protocol.get("cache_state") != "none"
    ):
        raise ValueError("spectral source protocol differs from frozen ABI")
    if set(mapping.symmetric_by_label) != {
        "local_fraction_sigma",
        "operating_1_sigma",
    }:
        raise ValueError("spectral source response regimes drifted")
    canonical = _mapping(
        state["canonical_reference"],
        label="canonical reference",
    )
    _strict_keys(
        canonical,
        expected={
            "metadata",
            "l3_post_attention_preimage",
            "x3_mean_target",
            "source_mode_standard_deviations",
            "baseline_x4_reference",
        },
        label="canonical reference",
    )
    sigma = _canonical_tensor(
        canonical["source_mode_standard_deviations"],
        label="source mode standard deviations",
        ndim=1,
    )
    return Gemma3SpectralSource(
        file_sha256=actual_digest,
        report_file_sha256=report_file_digest,
        report_payload_sha256=claimed_report_digest,
        binding=binding,
        model=model,
        protocol=protocol,
        mapping=mapping,
        source_mode_standard_deviations=sigma,
    )


def _cosine(first: Tensor, second: Tensor) -> float:
    left = first.detach().to(dtype=torch.float64).reshape(-1)
    right = second.detach().to(dtype=torch.float64).reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    epsilon = torch.finfo(torch.float64).eps
    if left_norm <= epsilon:
        return 1.0 if right_norm <= epsilon else 0.0
    if right_norm <= epsilon:
        return 0.0
    return max(
        -1.0,
        min(1.0, float(torch.dot(left, right)) / (left_norm * right_norm)),
    )


def _relative_error(prediction: Tensor, target: Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(target)),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.vector_norm(prediction - target)) / denominator


def _linear_candidate_passes(row: Mapping[str, object]) -> bool:
    return (
        float(row["selection_macro_weighted_relative_error"])
        <= LINEAR_MAX_MACRO_RELATIVE_ERROR
        and float(row["selection_worst_weighted_relative_error"])
        <= LINEAR_MAX_WORST_RELATIVE_ERROR
        and float(row["selection_worst_cosine"])
        >= LINEAR_MIN_WORST_COSINE
    )


def _quadratic_candidate_passes(row: Mapping[str, object]) -> bool:
    # Do not round either value before applying the preregistered gate.
    return (
        float(row["fit_even_energy_retained"])
        >= QUADRATIC_MIN_FIT_EVEN_ENERGY
        and float(row["selection_worst_finite_error_reduction_fraction"])
        >= QUADRATIC_MIN_WORST_FINITE_ERROR_REDUCTION
    )


def _select_minimal(
    rows: Sequence[Mapping[str, object]],
    *,
    predicate: Callable[[Mapping[str, object]], bool],
    label: str,
) -> Mapping[str, object]:
    passing = [row for row in rows if predicate(row)]
    if not passing:
        raise RuntimeError(f"no {label} rate candidate passed its frozen gate")
    return min(
        passing,
        key=lambda row: (
            int(row["stored_coefficient_count"]),
            int(row["source_rank"]),
            int(row["target_rank"]),
        ),
    )


def _generic_api() -> tuple[Any, Callable[..., Any], Callable[..., Any]]:
    """Isolate the model-agnostic compiler API from this experiment wrapper."""

    from .conditional_spectral_generator import (  # type: ignore[import-not-found]
        ConditionalSpectralGeneratorPlan,
        evaluate_conditional_spectral_generator,
        fit_conditional_spectral_generator,
    )

    return (
        ConditionalSpectralGeneratorPlan,
        fit_conditional_spectral_generator,
        evaluate_conditional_spectral_generator,
    )


def _plan_state(plan: object) -> Mapping[str, object]:
    method = getattr(plan, "state_dict", None)
    if not callable(method):
        raise TypeError("conditional spectral plan lacks state_dict")
    return _mapping(method(), label="conditional spectral plan state")


def _plan_metadata(plan: object) -> Mapping[str, object]:
    method = getattr(plan, "metadata", None)
    if not callable(method):
        raise TypeError("conditional spectral plan lacks metadata")
    return _mapping(method(), label="conditional spectral plan metadata")


def _plan_hash(plan: object) -> str:
    return _sha256(
        getattr(plan, "artifact_sha256", None),
        label="conditional spectral plan",
    )


def _plan_stored_coefficients(plan: object) -> int:
    value = getattr(plan, "stored_coefficient_count", None)
    if type(value) is not int or value <= 0:
        raise ValueError("conditional spectral plan coefficient count is invalid")
    return value


def _plan_accounting(plan: object) -> Mapping[str, object]:
    method = getattr(plan, "accounting", None)
    if not callable(method):
        raise TypeError("conditional spectral plan lacks accounting")
    accounting = method()
    metadata = getattr(accounting, "metadata", None)
    if not callable(metadata):
        raise TypeError("conditional spectral accounting lacks metadata")
    return _mapping(metadata(), label="conditional spectral accounting")


def _plan_predict(plan: object, origin: int) -> Tensor:
    for name in (
        "weighted_kernel_at_origin",
        "reconstruct_weighted_response",
        "weighted_response_at_origin",
        "reconstruct_at_origin",
    ):
        method = getattr(plan, name, None)
        if callable(method):
            result = method(origin)
            return _canonical_tensor(
                result,
                label="conditional spectral prediction",
                ndim=3,
            )
    raise TypeError("conditional spectral plan lacks frozen reconstruction")


def _fit_generic_plan(
    fit_function: Callable[..., object],
    *,
    responses: Tensor,
    source_scales: Tensor,
    origins: Sequence[int],
    fit_origins: Sequence[int],
    source_rank: int,
    target_rank: int,
    response_binding_sha256: str,
    input_transform: str,
    fft_length: int,
) -> object:
    """One adapter point for the model-agnostic compiler implementation."""

    return fit_function(
        responses=responses,
        source_scales=source_scales,
        origins=tuple(origins),
        fit_origins=tuple(fit_origins),
        source_rank=source_rank,
        target_rank=target_rank,
        response_binding_sha256=response_binding_sha256,
        input_transform=input_transform,
        fft_length=fft_length,
    )


def _metric_rows(
    plan: object,
    *,
    truth: Tensor,
    available_origins: Sequence[int],
    measured_origins: Sequence[int],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    available = tuple(available_origins)
    for origin in measured_origins:
        ordinal = available.index(origin)
        target = truth[:, ordinal]
        prediction = _plan_predict(plan, origin)
        rows.append(
            {
                "origin": origin,
                "weighted_relative_error": _relative_error(
                    prediction,
                    target,
                ),
                "cosine": _cosine(prediction, target),
            }
        )
    return tuple(rows)


def _response_binding(
    source: Gemma3SpectralSource,
    *,
    component: str,
) -> str:
    return _json_sha256(
        {
            "source_binding": dict(source.binding),
            "component": component,
            "source_rank": source.mapping.source_rank,
            "target_rank": source.mapping.target_rank,
            "source_mode_indices": source.mapping.source_mode_indices,
            "max_lag": source.mapping.max_lag,
            "fft_length": source.mapping.fft_length,
            "source_scales_sha256": _tensor_sha256(
                source.source_mode_standard_deviations
            ),
            "interpolation_coordinate": "source_logical_position",
            "response_tensor_order": "source_origin_lag_target",
        },
        domain=_CANDIDATE_DOMAIN,
    )


def _global_relative_error(
    prediction: Sequence[Tensor],
    target: Sequence[Tensor],
) -> float:
    return _relative_error(
        torch.stack(tuple(prediction)),
        torch.stack(tuple(target)),
    )


def _linear_rate_row(
    plan: object,
    *,
    weighted_truth: Tensor,
    origins: tuple[int, ...],
    source_rank: int,
    target_rank: int,
) -> dict[str, object]:
    fit_rows = _metric_rows(
        plan,
        truth=weighted_truth,
        available_origins=origins,
        measured_origins=FIT_ORIGINS,
    )
    selection_rows = _metric_rows(
        plan,
        truth=weighted_truth,
        available_origins=origins,
        measured_origins=SELECTION_ORIGINS,
    )
    fit_prediction = tuple(
        _plan_predict(plan, origin) for origin in FIT_ORIGINS
    )
    fit_target = tuple(
        weighted_truth[:, origins.index(origin)] for origin in FIT_ORIGINS
    )
    fit_error = _global_relative_error(fit_prediction, fit_target)
    macro = sum(
        float(row["weighted_relative_error"]) for row in selection_rows
    ) / len(selection_rows)
    worst = max(
        float(row["weighted_relative_error"]) for row in selection_rows
    )
    worst_cosine = min(float(row["cosine"]) for row in selection_rows)
    row: dict[str, object] = {
        "source_rank": source_rank,
        "target_rank": target_rank,
        "stored_coefficient_count": _plan_stored_coefficients(plan),
        "plan_artifact_sha256": _plan_hash(plan),
        "fit_weighted_relative_error": fit_error,
        "fit_weighted_energy_retained": max(0.0, 1.0 - fit_error**2),
        "fit_by_origin": fit_rows,
        "selection_by_origin": selection_rows,
        "selection_macro_weighted_relative_error": macro,
        "selection_worst_weighted_relative_error": worst,
        "selection_worst_cosine": worst_cosine,
    }
    row["passes_frozen_gate"] = _linear_candidate_passes(row)
    return row


def _quadratic_rate_row(
    plan: object,
    *,
    linear_plan: object,
    local_weighted_truth: Tensor,
    operating_odd_weighted_truth: Tensor,
    operating_even_weighted_truth: Tensor,
    origins: tuple[int, ...],
    source_rank: int,
    target_rank: int,
) -> dict[str, object]:
    fit_prediction = tuple(
        _plan_predict(plan, origin) for origin in FIT_ORIGINS
    )
    fit_target = tuple(
        operating_even_weighted_truth[:, origins.index(origin)]
        for origin in FIT_ORIGINS
    )
    fit_error = _global_relative_error(fit_prediction, fit_target)
    selection_rows: list[dict[str, object]] = []
    for origin in SELECTION_ORIGINS:
        ordinal = origins.index(origin)
        linear_prediction = _plan_predict(linear_plan, origin)
        even_prediction = _plan_predict(plan, origin)
        odd_target = operating_odd_weighted_truth[:, ordinal]
        even_target = operating_even_weighted_truth[:, ordinal]
        plus_minus_target = torch.stack(
            (odd_target + even_target, -odd_target + even_target)
        )
        linear_only = torch.stack(
            (linear_prediction, -linear_prediction)
        )
        corrected = torch.stack(
            (
                linear_prediction + even_prediction,
                -linear_prediction + even_prediction,
            )
        )
        baseline_error = _relative_error(linear_only, plus_minus_target)
        corrected_error = _relative_error(corrected, plus_minus_target)
        reduction = (
            (baseline_error - corrected_error) / baseline_error
            if baseline_error > torch.finfo(torch.float64).eps
            else 0.0
        )
        selection_rows.append(
            {
                "origin": origin,
                "linear_only_finite_weighted_relative_error": baseline_error,
                "corrected_finite_weighted_relative_error": corrected_error,
                "finite_error_reduction_fraction": reduction,
                "corrected_finite_cosine": _cosine(
                    corrected,
                    plus_minus_target,
                ),
                "linear_local_tangent_target_relative_error": (
                    _relative_error(
                        linear_prediction,
                        local_weighted_truth[:, ordinal],
                    )
                ),
            }
        )
    worst_reduction = min(
        float(row["finite_error_reduction_fraction"])
        for row in selection_rows
    )
    fit_energy = max(0.0, 1.0 - fit_error**2)
    row: dict[str, object] = {
        "source_rank": source_rank,
        "target_rank": target_rank,
        "stored_coefficient_count": _plan_stored_coefficients(plan),
        "plan_artifact_sha256": _plan_hash(plan),
        "fit_even_weighted_relative_error": fit_error,
        "fit_even_energy_retained": fit_energy,
        "selection_by_origin": tuple(selection_rows),
        "selection_worst_finite_error_reduction_fraction": worst_reduction,
    }
    row["passes_frozen_gate"] = _quadratic_candidate_passes(row)
    return row


_CANDIDATE_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_spectral_response_tensors": False,
    "contains_compiled_conditional_generator_plans": True,
    "assessment_artifact_opened_during_compilation": False,
    "artifact_must_remain_outside_git": True,
}


@dataclass(frozen=True, slots=True)
class Gemma3ConditionalSpectralCandidate:
    """Frozen linear plus diagonal-even modal-delta candidate."""

    source_artifact_file_sha256: str
    source_report_file_sha256: str
    source_report_payload_sha256: str
    source_mapping_artifact_sha256: str
    binding: Mapping[str, object]
    model: Mapping[str, object]
    linear_plan: object
    quadratic_plan: object
    linear_rate_curve: tuple[Mapping[str, object], ...]
    quadratic_rate_curve: tuple[Mapping[str, object], ...]
    selected_linear_rate_row: Mapping[str, object]
    selected_quadratic_rate_row: Mapping[str, object]
    accounting: Mapping[str, object]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _sha256(self.source_artifact_file_sha256, label="source artifact file")
        _sha256(self.source_report_file_sha256, label="source report file")
        _sha256(
            self.source_report_payload_sha256,
            label="source report payload",
        )
        _sha256(
            self.source_mapping_artifact_sha256,
            label="source mapping artifact",
        )
        for plan, label in (
            (self.linear_plan, "linear"),
            (self.quadratic_plan, "quadratic"),
        ):
            validate = getattr(plan, "validate_integrity", None)
            if not callable(validate):
                raise TypeError(f"{label} plan lacks integrity validation")
            validate()
            _plan_hash(plan)
        for name in (
            "binding",
            "model",
            "selected_linear_rate_row",
            "selected_quadratic_rate_row",
            "accounting",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(value))
        for name in ("linear_rate_curve", "quadratic_rate_curve"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            canonical = tuple(dict(row) for row in values)
            _canonical_json_bytes(canonical)
            object.__setattr__(self, name, canonical)
        if not _linear_candidate_passes(self.selected_linear_rate_row):
            raise ValueError("selected linear plan does not pass frozen gate")
        if not _quadratic_candidate_passes(self.selected_quadratic_rate_row):
            raise ValueError("selected quadratic plan does not pass frozen gate")
        if (
            self.selected_linear_rate_row.get("plan_artifact_sha256")
            != _plan_hash(self.linear_plan)
            or self.selected_quadratic_rate_row.get("plan_artifact_sha256")
            != _plan_hash(self.quadratic_plan)
        ):
            raise ValueError("selected rate rows do not bind their plans")
        computed = _json_sha256(self._hash_payload(), domain=_CANDIDATE_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("conditional spectral candidate hash mismatch")

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": _CANDIDATE_SCHEMA,
            "format_version": _FORMAT_VERSION,
            "source_artifact_file_sha256": self.source_artifact_file_sha256,
            "source_report_file_sha256": self.source_report_file_sha256,
            "source_report_payload_sha256": self.source_report_payload_sha256,
            "source_mapping_artifact_sha256": (
                self.source_mapping_artifact_sha256
            ),
            "binding": dict(self.binding),
            "model": dict(self.model),
            "fit_origins": FIT_ORIGINS,
            "selection_origins": SELECTION_ORIGINS,
            "assessment_origins_used": (),
            "linear_plan_artifact_sha256": _plan_hash(self.linear_plan),
            "quadratic_plan_artifact_sha256": _plan_hash(self.quadratic_plan),
            "linear_rate_curve": self.linear_rate_curve,
            "quadratic_rate_curve": self.quadratic_rate_curve,
            "selected_linear_rate_row": dict(self.selected_linear_rate_row),
            "selected_quadratic_rate_row": dict(
                self.selected_quadratic_rate_row
            ),
            "accounting": dict(self.accounting),
            "claim_boundaries": _claim_boundaries(),
            "safety": _CANDIDATE_SAFETY,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "linear_plan": dict(_plan_state(self.linear_plan)),
            "quadratic_plan": dict(_plan_state(self.quadratic_plan)),
            "artifact_sha256": self.artifact_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "linear_plan": dict(_plan_metadata(self.linear_plan)),
            "quadratic_plan": dict(_plan_metadata(self.quadratic_plan)),
            "artifact_sha256": self.artifact_sha256,
        }


def _claim_boundaries() -> dict[str, object]:
    return {
        "fixed_reference_modal_delta_executor_only": True,
        "prompt_conditioned_reference_provider_compiled": False,
        "baseline_provider_compiled": False,
        "full_gemma_block_replacement_authorized": False,
        "linear_path": "local_central_tangent_tucker",
        "finite_correction": "operating_scale_diagonal_square_even_tucker",
        "finite_correction_has_linear_or_bias_terms": False,
        "finite_correction_has_cross_mode_terms": False,
        "finite_correction_has_cross_lag_terms": False,
        "general_quadratic_hessian_identified": False,
        "heldout_prompt_fidelity_claim": False,
        "nll_claim": False,
        "task_accuracy_claim": False,
        "compression_claim": False,
        "speed_or_latency_claim": False,
        "cached_decode_claim": False,
    }


def compile_gemma3_conditional_spectral_candidate(
    source: Gemma3SpectralSource,
    *,
    fit_function: Callable[..., object] | None = None,
) -> Gemma3ConditionalSpectralCandidate:
    """Compile and select plans without accepting an assessment input."""

    if not isinstance(source, Gemma3SpectralSource):
        raise TypeError("source must be a Gemma3SpectralSource")
    _validate_compile_split(source.mapping.impulse_logical_positions)
    _, default_fit, _ = _generic_api()
    fit = default_fit if fit_function is None else fit_function
    if not callable(fit):
        raise TypeError("fit_function must be callable")
    mapping = source.mapping
    origins = mapping.impulse_logical_positions
    sigma = source.source_mode_standard_deviations
    local = mapping.symmetric_by_label["local_fraction_sigma"]
    operating = mapping.symmetric_by_label["operating_1_sigma"]
    if (
        local.even_residual_impulse_responses is None
        or operating.even_residual_impulse_responses is None
    ):
        raise ValueError("central spectral responses lack even residuals")
    local_binding = _response_binding(
        source,
        component="local_central_odd_tangent",
    )
    even_binding = _response_binding(
        source,
        component="operating_diagonal_even_residual",
    )
    local_truth = (
        local.impulse_responses
        * sigma.reshape(-1, 1, 1, 1)
    )
    operating_odd_truth = (
        operating.impulse_responses
        * sigma.reshape(-1, 1, 1, 1)
    )
    operating_even_truth = (
        operating.even_residual_impulse_responses
        * sigma.reshape(-1, 1, 1, 1)
    )

    linear_plans: list[object] = []
    linear_rows: list[dict[str, object]] = []
    for source_rank, target_rank in LINEAR_RANK_PAIRS:
        plan = _fit_generic_plan(
            fit,
            responses=local.impulse_responses,
            source_scales=sigma,
            origins=origins,
            fit_origins=FIT_ORIGINS,
            source_rank=source_rank,
            target_rank=target_rank,
            response_binding_sha256=local_binding,
            input_transform="standardized_linear",
            fft_length=mapping.fft_length,
        )
        linear_plans.append(plan)
        linear_rows.append(
            _linear_rate_row(
                plan,
                weighted_truth=local_truth,
                origins=origins,
                source_rank=source_rank,
                target_rank=target_rank,
            )
        )
    selected_linear_row = _select_minimal(
        linear_rows,
        predicate=_linear_candidate_passes,
        label="linear",
    )
    selected_linear = linear_plans[
        linear_rows.index(selected_linear_row)  # type: ignore[arg-type]
    ]

    quadratic_plans: list[object] = []
    quadratic_rows: list[dict[str, object]] = []
    for source_rank, target_rank in QUADRATIC_RANK_PAIRS:
        plan = _fit_generic_plan(
            fit,
            responses=operating.even_residual_impulse_responses,
            source_scales=sigma,
            origins=origins,
            fit_origins=FIT_ORIGINS,
            source_rank=source_rank,
            target_rank=target_rank,
            response_binding_sha256=even_binding,
            input_transform="standardized_square",
            fft_length=mapping.fft_length,
        )
        quadratic_plans.append(plan)
        quadratic_rows.append(
            _quadratic_rate_row(
                plan,
                linear_plan=selected_linear,
                local_weighted_truth=local_truth,
                operating_odd_weighted_truth=operating_odd_truth,
                operating_even_weighted_truth=operating_even_truth,
                origins=origins,
                source_rank=source_rank,
                target_rank=target_rank,
            )
        )
    selected_quadratic_row = _select_minimal(
        quadratic_rows,
        predicate=_quadratic_candidate_passes,
        label="quadratic",
    )
    selected_quadratic = quadratic_plans[
        quadratic_rows.index(selected_quadratic_row)  # type: ignore[arg-type]
    ]
    linear_accounting = dict(_plan_accounting(selected_linear))
    quadratic_accounting = dict(_plan_accounting(selected_quadratic))
    accounting = {
        "linear_plan": linear_accounting,
        "quadratic_plan": quadratic_accounting,
        "linear_plan_stored_coefficient_count": (
            _plan_stored_coefficients(selected_linear)
        ),
        "quadratic_plan_stored_coefficient_count": (
            _plan_stored_coefficients(selected_quadratic)
        ),
        "sum_of_plan_stored_coefficient_counts": (
            _plan_stored_coefficients(selected_linear)
            + _plan_stored_coefficients(selected_quadratic)
        ),
        "source_scale_count_per_strict_plan_state": sigma.numel(),
        "strict_plan_state_count": 2,
        "source_scales_are_identical_but_not_deduplicated_in_plan_states": True,
        "strict_plan_artifact_float_scalar_count": (
            int(linear_accounting["artifact_float_scalar_count"])
            + int(quadratic_accounting["artifact_float_scalar_count"])
        ),
        "deduplicated_prepared_runtime_float_scalar_count": (
            _plan_stored_coefficients(selected_linear)
            + _plan_stored_coefficients(selected_quadratic)
            + sigma.numel()
        ),
        "deduplicated_prepared_runtime_float32_bytes": 4
        * (
            _plan_stored_coefficients(selected_linear)
            + _plan_stored_coefficients(selected_quadratic)
            + sigma.numel()
        ),
        "deduplicated_prepared_runtime_float64_bytes": 8
        * (
            _plan_stored_coefficients(selected_linear)
            + _plan_stored_coefficients(selected_quadratic)
            + sigma.numel()
        ),
        "dense_linear_anchor_kernel_coefficients": (
            len(FIT_ORIGINS)
            * (mapping.max_lag + 1)
            * mapping.source_rank
            * mapping.target_rank
        ),
        "dense_even_anchor_kernel_coefficients": (
            len(FIT_ORIGINS)
            * (mapping.max_lag + 1)
            * mapping.source_rank
            * mapping.target_rank
        ),
        "parameter_count_is_not_model_compression": True,
        "runtime_macs_measured": False,
        "latency_measured": False,
    }
    return Gemma3ConditionalSpectralCandidate(
        source_artifact_file_sha256=source.file_sha256,
        source_report_file_sha256=source.report_file_sha256,
        source_report_payload_sha256=source.report_payload_sha256,
        source_mapping_artifact_sha256=mapping.artifact_sha256,
        binding=source.binding,
        model=source.model,
        linear_plan=selected_linear,
        quadratic_plan=selected_quadratic,
        linear_rate_curve=tuple(linear_rows),
        quadratic_rate_curve=tuple(quadratic_rows),
        selected_linear_rate_row=selected_linear_row,
        selected_quadratic_rate_row=selected_quadratic_row,
        accounting=accounting,
    )


_CANDIDATE_STATE_KEYS = {
    "schema",
    "format_version",
    "source_artifact_file_sha256",
    "source_report_file_sha256",
    "source_report_payload_sha256",
    "source_mapping_artifact_sha256",
    "binding",
    "model",
    "fit_origins",
    "selection_origins",
    "assessment_origins_used",
    "linear_plan_artifact_sha256",
    "quadratic_plan_artifact_sha256",
    "linear_rate_curve",
    "quadratic_rate_curve",
    "selected_linear_rate_row",
    "selected_quadratic_rate_row",
    "accounting",
    "claim_boundaries",
    "safety",
    "linear_plan",
    "quadratic_plan",
    "artifact_sha256",
}


def _candidate_from_state(
    value: Mapping[str, object],
) -> Gemma3ConditionalSpectralCandidate:
    _strict_keys(value, expected=_CANDIDATE_STATE_KEYS, label="candidate")
    if (
        value["schema"] != _CANDIDATE_SCHEMA
        or value["format_version"] != _FORMAT_VERSION
        or tuple(value["fit_origins"]) != FIT_ORIGINS  # type: ignore[arg-type]
        or tuple(value["selection_origins"])
        != SELECTION_ORIGINS  # type: ignore[arg-type]
        or tuple(value["assessment_origins_used"]) != ()  # type: ignore[arg-type]
        or dict(_mapping(value["claim_boundaries"], label="claim boundaries"))
        != _claim_boundaries()
        or dict(_mapping(value["safety"], label="candidate safety"))
        != _CANDIDATE_SAFETY
    ):
        raise ValueError("conditional spectral candidate protocol drifted")
    plan_class, _, _ = _generic_api()
    restore = getattr(plan_class, "from_state_dict", None)
    if not callable(restore):
        raise TypeError("generic plan class lacks from_state_dict")
    linear = restore(_mapping(value["linear_plan"], label="linear plan"))
    quadratic = restore(
        _mapping(value["quadratic_plan"], label="quadratic plan")
    )
    if (
        _plan_hash(linear) != value["linear_plan_artifact_sha256"]
        or _plan_hash(quadratic) != value["quadratic_plan_artifact_sha256"]
    ):
        raise ValueError("candidate plan hash binding drifted")
    linear_curve_raw = value["linear_rate_curve"]
    quadratic_curve_raw = value["quadratic_rate_curve"]
    if (
        isinstance(linear_curve_raw, (str, bytes))
        or not isinstance(linear_curve_raw, Sequence)
        or isinstance(quadratic_curve_raw, (str, bytes))
        or not isinstance(quadratic_curve_raw, Sequence)
    ):
        raise TypeError("candidate rate curves must be sequences")
    return Gemma3ConditionalSpectralCandidate(
        source_artifact_file_sha256=value[
            "source_artifact_file_sha256"
        ],  # type: ignore[arg-type]
        source_report_file_sha256=value[
            "source_report_file_sha256"
        ],  # type: ignore[arg-type]
        source_report_payload_sha256=value[
            "source_report_payload_sha256"
        ],  # type: ignore[arg-type]
        source_mapping_artifact_sha256=value[
            "source_mapping_artifact_sha256"
        ],  # type: ignore[arg-type]
        binding=_mapping(value["binding"], label="candidate binding"),
        model=_mapping(value["model"], label="candidate model"),
        linear_plan=linear,
        quadratic_plan=quadratic,
        linear_rate_curve=tuple(
            _mapping(row, label="linear rate row") for row in linear_curve_raw
        ),
        quadratic_rate_curve=tuple(
            _mapping(row, label="quadratic rate row")
            for row in quadratic_curve_raw
        ),
        selected_linear_rate_row=_mapping(
            value["selected_linear_rate_row"],
            label="selected linear row",
        ),
        selected_quadratic_rate_row=_mapping(
            value["selected_quadratic_rate_row"],
            label="selected quadratic row",
        ),
        accounting=_mapping(value["accounting"], label="candidate accounting"),
        artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
    )


def load_gemma3_conditional_spectral_candidate(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str | None = None,
) -> Gemma3ConditionalSpectralCandidate:
    """Strict-load one frozen candidate and authenticate its complete file."""

    source = Path(path)
    expected = _sha256(
        expected_file_sha256,
        label="expected candidate file",
    )
    actual = _file_sha256(source)
    if actual != expected:
        raise ValueError("candidate tensor file differs from expected SHA-256")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    candidate = _candidate_from_state(
        _mapping(raw, label="candidate tensor file")
    )
    if expected_report_sha256 is not None:
        expected_report = _sha256(
            expected_report_sha256,
            label="expected candidate report",
        )
        with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
            report_raw = json.load(handle)
        report = _mapping(report_raw, label="candidate report")
        claimed = _sha256(
            report.get("report_sha256"),
            label="candidate report",
        )
        payload = dict(report)
        payload.pop("report_sha256")
        artifact = _mapping(
            report.get("artifact"),
            label="candidate report artifact",
        )
        metadata = _mapping(
            report.get("candidate"),
            label="candidate report metadata",
        )
        if (
            claimed != expected_report
            or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed
            or artifact.get("tensor_file_sha256") != actual
            or metadata.get("artifact_sha256") != candidate.artifact_sha256
        ):
            raise ValueError("candidate report binding mismatch")
    return candidate


def _validate_output_path(path: Path | str, *, suffix: str) -> Path:
    destination = Path(path)
    if destination.suffix != suffix:
        raise ValueError(f"output must use a {suffix} suffix")
    if destination.exists():
        raise FileExistsError("refusing to overwrite conditional spectral output")
    worktree = find_git_worktree(Path(__file__))
    resolved = destination.expanduser().resolve()
    if worktree is not None:
        root = worktree.resolve()
        if resolved == root or root in resolved.parents:
            relative = resolved.relative_to(root)
            if not relative.parts or relative.parts[0] not in (
                ".local-runs",
                "local-runs",
            ):
                raise ValueError(
                    "conditional spectral outputs inside the worktree must "
                    "remain under an ignored local-runs directory"
                )
    return destination


def _path_identity(path: Path) -> tuple[int, int] | None:
    try:
        state = path.stat()
    except FileNotFoundError:
        return None
    return state.st_dev, state.st_ino


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    if _path_identity(path) == identity:
        path.unlink(missing_ok=True)


@dataclass(slots=True)
class _OutputReservation:
    destinations: tuple[Path, ...]
    claim: Path
    claim_identity: tuple[int, int]
    released: bool = False

    def __enter__(self) -> _OutputReservation:
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.release()

    def release(self) -> None:
        if not self.released:
            _unlink_if_identity(self.claim, self.claim_identity)
            self.released = True

    def publish(self, staged: Sequence[Path]) -> None:
        if self.released or len(staged) != len(self.destinations):
            raise RuntimeError("output reservation is not publishable")
        if _path_identity(self.claim) != self.claim_identity:
            raise RuntimeError("output reservation ownership was lost")
        identities = tuple(_path_identity(path) for path in staged)
        if any(identity is None for identity in identities):
            raise FileNotFoundError("staged conditional spectral output missing")
        published: list[tuple[Path, tuple[int, int]]] = []
        try:
            for stage, destination, identity in zip(
                staged,
                self.destinations,
                identities,
                strict=True,
            ):
                assert identity is not None
                os.link(stage, destination)
                published.append((destination, identity))
        except FileExistsError as error:
            for destination, identity in published:
                _unlink_if_identity(destination, identity)
            raise FileExistsError(
                "refusing to overwrite conditional spectral output"
            ) from error
        except BaseException:
            for destination, identity in published:
                _unlink_if_identity(destination, identity)
            raise


def _reserve_outputs(destinations: Sequence[Path]) -> _OutputReservation:
    values = tuple(destinations)
    if not values or len(set(values)) != len(values):
        raise ValueError("output reservation destinations are invalid")
    for destination in values:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(
                "refusing to overwrite conditional spectral output"
            )
    claim = values[0].with_name(f".{values[0].name}.publish.lock")
    try:
        descriptor = os.open(
            claim,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError("conditional spectral output is reserved") from error
    try:
        state = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reservation = _OutputReservation(
        destinations=values,
        claim=claim,
        claim_identity=(state.st_dev, state.st_ino),
    )
    if any(destination.exists() for destination in values):
        reservation.release()
        raise FileExistsError("refusing to overwrite conditional spectral output")
    return reservation


def _stage_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _stage_torch(value: object, destination: Path) -> Path:
    stage = _stage_path(destination)
    try:
        torch.save(value, stage)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    return stage


def _stage_json(value: Mapping[str, object], destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    stage = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    return stage


def _publish_candidate(
    candidate: Gemma3ConditionalSpectralCandidate,
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    reservation = _reserve_outputs((output, report_path))
    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch(candidate.state_dict(), output)
        tensor_digest = _file_sha256(tensor_stage)
        tensor_bytes = tensor_stage.stat().st_size
        report: dict[str, object] = {
            "schema": _CANDIDATE_SCHEMA,
            "format_version": _FORMAT_VERSION,
            "candidate": candidate.metadata(),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": tensor_digest,
                "tensor_file_bytes": tensor_bytes,
                "report_file": str(report_path),
                "committable": False,
            },
            "scientific_status": _claim_boundaries(),
            "safety": _CANDIDATE_SAFETY,
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        _canonical_json_bytes(report)
        report_stage = _stage_json(report, report_path)
        reservation.publish((tensor_stage, report_stage))
        return report
    finally:
        reservation.release()
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def compile_gemma3_l3_l4_conditional_spectral_executor(
    *,
    source_artifact_path: Path | str = DEFAULT_INTERIOR_ARTIFACT,
    source_artifact_sha256: str = DEFAULT_INTERIOR_ARTIFACT_SHA256,
    source_report_sha256: str = DEFAULT_INTERIOR_REPORT_SHA256,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Compile, select, freeze, and exclusively publish the candidate."""

    if source_artifact_sha256 != DEFAULT_INTERIOR_ARTIFACT_SHA256:
        raise ValueError("compile source must equal the pinned interior tensor")
    if source_report_sha256 != DEFAULT_INTERIOR_REPORT_SHA256:
        raise ValueError("compile source must equal the pinned interior report")
    destination = _validate_output_path(output, suffix=".pt")
    report_path = destination.with_suffix(".json")
    if report_path.exists():
        raise FileExistsError("refusing to overwrite candidate report")
    source = load_gemma3_spectral_source(
        source_artifact_path,
        expected_file_sha256=source_artifact_sha256,
        expected_report_sha256=source_report_sha256,
        expected_origins=INTERIOR_ORIGINS,
    )
    candidate = compile_gemma3_conditional_spectral_candidate(source)
    return _publish_candidate(candidate, output=destination)


def _plan_field(plan: object, name: str) -> object:
    value = getattr(plan, name, None)
    if value is None:
        metadata = _plan_metadata(plan)
        value = metadata.get(name)
    return value


def _validate_assessment_plan_abi(
    candidate: Gemma3ConditionalSpectralCandidate,
    source: Gemma3SpectralSource,
    *,
    origin: int,
) -> None:
    if origin in FIT_ORIGINS or origin in SELECTION_ORIGINS:
        raise ValueError("assessment origin overlaps compilation origins")
    if source.mapping.impulse_logical_positions != (origin,):
        raise ValueError("assessment artifact must contain one planned origin")
    if source.mapping.artifact_sha256 == candidate.source_mapping_artifact_sha256:
        raise ValueError("assessment mapping must be separately measured")
    if source.binding != candidate.binding or source.model != candidate.model:
        raise ValueError("assessment model or hierarchy binding drifted")
    for plan, component in (
        (candidate.linear_plan, "local_central_odd_tangent"),
        (
            candidate.quadratic_plan,
            "operating_diagonal_even_residual",
        ),
    ):
        expected_binding = _response_binding(source, component=component)
        if _plan_field(plan, "response_binding_sha256") != expected_binding:
            raise ValueError("assessment response ABI differs from frozen plan")
        if (
            tuple(_plan_field(plan, "fit_knot_origins"))  # type: ignore[arg-type]
            != FIT_ORIGINS
        ):
            raise ValueError("frozen plan fit origins drifted")
        if _plan_field(plan, "fft_length") != source.mapping.fft_length:
            raise ValueError("assessment FFT protocol differs from frozen plan")
        plan_scales = _canonical_tensor(
            _plan_field(plan, "source_scales"),
            label="plan source scales",
            ndim=1,
        )
        if not torch.equal(
            plan_scales,
            source.source_mode_standard_deviations,
        ):
            raise ValueError("assessment source scales differ from frozen plan")


def _assessment_metrics(
    candidate: Gemma3ConditionalSpectralCandidate,
    source: Gemma3SpectralSource,
    *,
    origin: int,
) -> dict[str, object]:
    _validate_assessment_plan_abi(candidate, source, origin=origin)
    local = source.mapping.symmetric_by_label["local_fraction_sigma"]
    operating = source.mapping.symmetric_by_label["operating_1_sigma"]
    if operating.even_residual_impulse_responses is None:
        raise ValueError("assessment operating response lacks even residual")
    sigma = source.source_mode_standard_deviations.reshape(-1, 1, 1)
    local_target = local.impulse_responses[:, 0] * sigma
    operating_odd = operating.impulse_responses[:, 0] * sigma
    operating_even = (
        operating.even_residual_impulse_responses[:, 0] * sigma
    )
    linear_prediction = _plan_predict(candidate.linear_plan, origin)
    even_prediction = _plan_predict(candidate.quadratic_plan, origin)
    true_plus_minus = torch.stack(
        (
            operating_odd + operating_even,
            -operating_odd + operating_even,
        )
    )
    linear_plus_minus = torch.stack(
        (linear_prediction, -linear_prediction)
    )
    corrected_plus_minus = torch.stack(
        (
            linear_prediction + even_prediction,
            -linear_prediction + even_prediction,
        )
    )
    linear_error = _relative_error(
        linear_plus_minus,
        true_plus_minus,
    )
    corrected_error = _relative_error(
        corrected_plus_minus,
        true_plus_minus,
    )
    return {
        "assessment_origin": origin,
        "linear_local_weighted_relative_error": _relative_error(
            linear_prediction,
            local_target,
        ),
        "linear_local_cosine": _cosine(
            linear_prediction,
            local_target,
        ),
        "even_weighted_relative_error": _relative_error(
            even_prediction,
            operating_even,
        ),
        "even_energy_retained": max(
            0.0,
            1.0
            - _relative_error(even_prediction, operating_even) ** 2,
        ),
        "linear_only_operating_finite_weighted_relative_error": linear_error,
        "corrected_operating_finite_weighted_relative_error": corrected_error,
        "operating_finite_error_reduction_fraction": (
            (linear_error - corrected_error) / linear_error
            if linear_error > torch.finfo(torch.float64).eps
            else 0.0
        ),
        "corrected_operating_finite_cosine": _cosine(
            corrected_plus_minus,
            true_plus_minus,
        ),
        "assessment_refit_performed": False,
        "assessment_changed_frozen_plan": False,
    }


_ASSESSMENT_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_spectral_response_tensors": False,
    "contains_compiled_plan_tensors": False,
    "assessment_refit_performed": False,
    "committable": False,
}


def _publish_assessment(
    report: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    reservation = _reserve_outputs((output,))
    stage: Path | None = None
    try:
        result = dict(report)
        if "report_sha256" in result:
            raise ValueError("assessment report already has a hash")
        result["report_sha256"] = _json_sha256(
            result,
            domain=_REPORT_DOMAIN,
        )
        _canonical_json_bytes(result)
        stage = _stage_json(result, output)
        reservation.publish((stage,))
        return result
    finally:
        reservation.release()
        if stage is not None:
            stage.unlink(missing_ok=True)


def assess_gemma3_l3_l4_conditional_spectral_executor(
    *,
    candidate_path: Path | str,
    candidate_file_sha256: str,
    candidate_report_sha256: str,
    assessment_artifact_path: Path | str,
    assessment_artifact_sha256: str,
    assessment_report_sha256: str,
    output: Path | str,
    assessment_origin: int = DEFAULT_ASSESSMENT_ORIGIN,
) -> dict[str, object]:
    """Evaluate one frozen candidate without exposing any fitting operation."""

    if type(assessment_origin) is not int or assessment_origin < 0:
        raise ValueError("assessment origin must be a nonnegative integer")
    destination = _validate_output_path(output, suffix=".json")
    candidate_file = Path(candidate_path)
    candidate = load_gemma3_conditional_spectral_candidate(
        candidate_file,
        expected_file_sha256=candidate_file_sha256,
        expected_report_sha256=candidate_report_sha256,
    )
    assessment_file = Path(assessment_artifact_path)
    source = load_gemma3_spectral_source(
        assessment_file,
        expected_file_sha256=assessment_artifact_sha256,
        expected_report_sha256=assessment_report_sha256,
        expected_origins=(assessment_origin,),
        expected_binding=candidate.binding,
    )
    metrics = _assessment_metrics(
        candidate,
        source,
        origin=assessment_origin,
    )
    report = {
        "schema": _ASSESSMENT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "binding": {
            "candidate_tensor_file_sha256": _sha256(
                candidate_file_sha256,
                label="candidate tensor file",
            ),
            "candidate_report_payload_sha256": _sha256(
                candidate_report_sha256,
                label="candidate report",
            ),
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "linear_plan_artifact_sha256": _plan_hash(
                candidate.linear_plan
            ),
            "quadratic_plan_artifact_sha256": _plan_hash(
                candidate.quadratic_plan
            ),
            "assessment_tensor_file_sha256": source.file_sha256,
            "assessment_report_file_sha256": source.report_file_sha256,
            "assessment_report_payload_sha256": (
                source.report_payload_sha256
            ),
            "assessment_mapping_artifact_sha256": (
                source.mapping.artifact_sha256
            ),
            "source_model_sha256": candidate.binding.get(
                "source_model_sha256"
            ),
        },
        "split": {
            "compile_fit_origins": FIT_ORIGINS,
            "compile_selection_origins": SELECTION_ORIGINS,
            "assessment_origin": assessment_origin,
            "assessment_origin_was_opened_during_compilation": False,
            "assessment_refit_performed": False,
        },
        "metrics": metrics,
        "claim_boundaries": _claim_boundaries(),
        "safety": _ASSESSMENT_SAFETY,
        "artifact": {
            "report_file": str(destination),
            "committable": False,
        },
    }
    return _publish_assessment(report, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile or assess the development-only Gemma L3-L4 "
            "conditional spectral modal-delta executor."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser(
        "compile",
        help="compile only the pinned interior development measurement",
    )
    compile_parser.add_argument(
        "--source-artifact",
        type=Path,
        default=DEFAULT_INTERIOR_ARTIFACT,
    )
    compile_parser.add_argument(
        "--source-artifact-sha256",
        default=DEFAULT_INTERIOR_ARTIFACT_SHA256,
    )
    compile_parser.add_argument(
        "--source-report-sha256",
        default=DEFAULT_INTERIOR_REPORT_SHA256,
    )
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    assess_parser = commands.add_parser(
        "assess",
        help="assess a frozen candidate on one separately measured origin",
    )
    assess_parser.add_argument("--candidate", type=Path, required=True)
    assess_parser.add_argument(
        "--candidate-file-sha256",
        required=True,
    )
    assess_parser.add_argument(
        "--candidate-report-sha256",
        required=True,
    )
    assess_parser.add_argument(
        "--assessment-artifact",
        type=Path,
        required=True,
    )
    assess_parser.add_argument(
        "--assessment-artifact-sha256",
        required=True,
    )
    assess_parser.add_argument(
        "--assessment-report-sha256",
        required=True,
    )
    assess_parser.add_argument("--output", type=Path, required=True)
    assess_parser.add_argument(
        "--assessment-origin",
        type=int,
        default=DEFAULT_ASSESSMENT_ORIGIN,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "compile":
        report = compile_gemma3_l3_l4_conditional_spectral_executor(
            source_artifact_path=arguments.source_artifact,
            source_artifact_sha256=arguments.source_artifact_sha256,
            source_report_sha256=arguments.source_report_sha256,
            output=arguments.output,
        )
    elif arguments.command == "assess":
        report = assess_gemma3_l3_l4_conditional_spectral_executor(
            candidate_path=arguments.candidate,
            candidate_file_sha256=arguments.candidate_file_sha256,
            candidate_report_sha256=arguments.candidate_report_sha256,
            assessment_artifact_path=arguments.assessment_artifact,
            assessment_artifact_sha256=(
                arguments.assessment_artifact_sha256
            ),
            assessment_report_sha256=arguments.assessment_report_sha256,
            output=arguments.output,
            assessment_origin=arguments.assessment_origin,
        )
    else:  # pragma: no cover - argparse enforces this.
        raise AssertionError("unreachable command")
    print(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
