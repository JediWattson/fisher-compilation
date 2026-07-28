"""Prompt-free spectral mapping at the frozen Gemma L3-to-L4 boundary.

This experiment deliberately does not load text, a tokenizer, or token IDs.
It restores the frozen full-stack refit lineage, authenticates the v3 L3/L4
hierarchy artifact, and constructs one deterministic internal reference:

* the stored mean L3 MLP input is inverted through Gemma's unit-offset
  pre-feedforward RMSNorm using its minimum-norm preimage;
* modal perturbations are decoded around the stored mean L3 MLP output;
* only the L3 post-feedforward norm and the exact L4 attention prefix run;
* the resulting L4 pre-feedforward input displacement is projected into the
  frozen L4 modal basis.

The resulting spectra are structural fingerprints of this fixed reference.
They demonstrate interventional influence through the known L3-to-L4 causal
boundary, but do not establish prompt-distribution fidelity, shift-invariant
convolution, semantic equivalence, compression, or speed.
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
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .external_models import find_git_worktree
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    Gemma3RefitRuntimeCatalog,
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .modal_spectral_mapping import (
    ModalSpectralMapping,
    ModalSpectralResponse,
    analyze_modal_spectral_mapping,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_HIERARCHY_ARTIFACT",
    "DEFAULT_HIERARCHY_ARTIFACT_SHA256",
    "DEFAULT_OUTPUT",
    "FixedRMSNormReference",
    "Gemma3L3L4SpectralAnalysis",
    "Gemma3L3L4SpectralReference",
    "analyze_prompt_free_gemma3_l3_l4_spectral_mapping",
    "build_parser",
    "invert_unit_offset_rmsnorm_reference",
    "load_gemma3_l3_l4_spectral_reference",
    "main",
    "run_gemma3_l3_l4_spectral_mapping_experiment",
]


DEFAULT_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DEFAULT_HIERARCHY_ARTIFACT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-hierarchy-measurement-dev-v3.pt"
)
DEFAULT_HIERARCHY_ARTIFACT_SHA256 = (
    "2e35cbd0e54a1db4b483f11ebbc3b1f9cdd3472d55aac5f1a046c434477b08ff"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-spectral-map-dev-v2.pt"
)
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_MODAL_RANK = 64
DEFAULT_SOURCE_MODE_COUNT = 8
DEFAULT_IMPULSE_POSITIONS = (0, 8)
DEFAULT_LOCAL_SIGMA_FRACTION = 0.05
DEFAULT_SIMILARITY_THRESHOLD = 0.9

_SCHEMA = "fisher_graph.gemma3_l3_l4_spectral_mapping_development"
_FORMAT_VERSION = 1
_HIERARCHY_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_hierarchy_measurement_development"
)
_SOURCE_SCOPE = "factorized_refit"
_X3 = "layer.3.mlp.normalized_input"
_Y3 = "layer.3.mlp.operator_output"
_X4 = "layer.4.mlp.normalized_input"
_Y4 = "layer.4.mlp.operator_output"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-spectral-report:v1\0"


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _report_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("report_sha256", None)
    return hashlib.sha256(
        _REPORT_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()


def _canonical_tensor(
    value: object,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{label} must be a floating Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if result.ndim != ndim or any(int(size) <= 0 for size in result.shape):
        raise ValueError(f"{label} must be nonempty and rank {ndim}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _strict_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match the frozen format")


@dataclass(frozen=True, slots=True)
class FixedRMSNormReference:
    """One authenticated minimum-norm preimage of a unit-offset RMSNorm."""

    value: Tensor
    target: Tensor
    null_gain_indices: tuple[int, ...]
    normalized_second_moment: float
    radial_scale: float
    reconstruction_max_abs: float
    reconstruction_relative_l2: float
    norm_module_sha256: str
    convention: str = "minimum_norm_zero_on_unit_offset_gain_nullspace"

    def __post_init__(self) -> None:
        value = _canonical_tensor(self.value, label="value", ndim=1)
        target = _canonical_tensor(self.target, label="target", ndim=1)
        if value.shape != target.shape:
            raise ValueError("RMSNorm reference and target widths differ")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "target", target)
        if (
            type(self.null_gain_indices) is not tuple
            or tuple(sorted(set(self.null_gain_indices)))
            != self.null_gain_indices
            or any(
                type(index) is not int
                or index < 0
                or index >= value.numel()
                for index in self.null_gain_indices
            )
        ):
            raise ValueError("null RMSNorm gain indices are invalid")
        for name in (
            "normalized_second_moment",
            "radial_scale",
            "reconstruction_max_abs",
            "reconstruction_relative_l2",
        ):
            number = getattr(self, name)
            if (
                isinstance(number, bool)
                or not isinstance(number, (float, int))
                or not math.isfinite(float(number))
                or float(number) < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.normalized_second_moment >= 1.0 or self.radial_scale <= 0.0:
            raise ValueError("RMSNorm preimage radial solution is invalid")
        _require_sha256(self.norm_module_sha256, label="norm module")

    def metadata(self) -> dict[str, object]:
        return {
            "width": self.value.numel(),
            "null_gain_indices": self.null_gain_indices,
            "null_gain_count": len(self.null_gain_indices),
            "normalized_second_moment": self.normalized_second_moment,
            "radial_scale": self.radial_scale,
            "reference_l2": float(torch.linalg.vector_norm(self.value)),
            "reconstruction_max_abs": self.reconstruction_max_abs,
            "reconstruction_relative_l2": self.reconstruction_relative_l2,
            "norm_module_sha256": self.norm_module_sha256,
            "convention": self.convention,
            "is_data_derived_mean_post_attention": False,
            "preimage_is_unique": not self.null_gain_indices,
        }


@dataclass(frozen=True, slots=True)
class Gemma3L3L4SpectralReference:
    """Strict factor/reference view of the frozen hierarchy v3 artifact."""

    hierarchy_artifact_sha256: str
    source_model_sha256: str
    base_artifact_file_sha256: str
    base_scientific_payload_sha256: str
    refit_artifact_file_sha256: str
    refit_scientific_payload_sha256: str
    generator_plan_sha256s: tuple[str, ...]
    layer3_factor_sha256: str
    layer4_factor_sha256: str
    x3_mean: Tensor
    y3_mean: Tensor
    x4_mean: Tensor
    y4_mean: Tensor
    R3: Tensor
    P3: Tensor
    R4: Tensor
    P4: Tensor
    S4: Tensor
    x3_covariance: Tensor
    upstream_mean_prompt_local_kernel: Tensor

    def __post_init__(self) -> None:
        for name in (
            "hierarchy_artifact_sha256",
            "source_model_sha256",
            "base_artifact_file_sha256",
            "base_scientific_payload_sha256",
            "refit_artifact_file_sha256",
            "refit_scientific_payload_sha256",
            "layer3_factor_sha256",
            "layer4_factor_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if (
            type(self.generator_plan_sha256s) is not tuple
            or not self.generator_plan_sha256s
        ):
            raise ValueError("generator plan hashes must be a nonempty tuple")
        for digest in self.generator_plan_sha256s:
            _require_sha256(digest, label="generator plan")
        for name in ("x3_mean", "y3_mean", "x4_mean", "y4_mean"):
            object.__setattr__(
                self,
                name,
                _canonical_tensor(getattr(self, name), label=name, ndim=1),
            )
        for name in ("R3", "P3", "R4", "P4", "x3_covariance"):
            object.__setattr__(
                self,
                name,
                _canonical_tensor(getattr(self, name), label=name, ndim=2),
            )
        object.__setattr__(
            self,
            "S4",
            _canonical_tensor(self.S4, label="S4", ndim=1),
        )
        object.__setattr__(
            self,
            "upstream_mean_prompt_local_kernel",
            _canonical_tensor(
                self.upstream_mean_prompt_local_kernel,
                label="upstream_mean_prompt_local_kernel",
                ndim=3,
            ),
        )
        width = self.x3_mean.numel()
        if any(
            value.numel() != width
            for value in (self.y3_mean, self.x4_mean, self.y4_mean)
        ):
            raise ValueError("L3/L4 means must share the residual width")
        if (
            self.R3.shape != (width, width)
            or self.P3.shape != (width, width)
            or self.R4.shape != (width, width)
            or self.P4.shape != (width, width)
            or self.S4.shape != (width,)
            or self.x3_covariance.shape != (width, width)
        ):
            raise ValueError("frozen L3/L4 factor geometry is invalid")
        singular_tolerance = max(float(self.S4.abs().max()), 1.0) * 1e-12
        if (
            bool((self.S4 < 0.0).any())
            or bool((self.S4[1:] > self.S4[:-1] + singular_tolerance).any())
        ):
            raise ValueError(
                "L4 balanced modal singular spectrum is invalid"
            )
        kernel = self.upstream_mean_prompt_local_kernel
        if (
            kernel.shape[1] > width
            or kernel.shape[2] > width
            or kernel.shape[1] != kernel.shape[2]
        ):
            raise ValueError("upstream prompt-local kernel geometry is invalid")

    @property
    def residual_width(self) -> int:
        return self.x3_mean.numel()

    @property
    def upstream_edge_rank(self) -> int:
        return int(self.upstream_mean_prompt_local_kernel.shape[1])

    def source_mode_standard_deviations(self, rank: int) -> Tensor:
        if type(rank) is not int or rank <= 0 or rank > self.residual_width:
            raise ValueError("rank is outside the frozen L3 basis")
        restriction = self.R3[:rank]
        variances = torch.diagonal(
            restriction @ self.x3_covariance @ restriction.T
        )
        tolerance = max(float(variances.abs().max()), 1.0) * 1e-10
        if float(variances.min()) < -tolerance:
            raise ValueError("projected source covariance has negative energy")
        result = variances.clamp_min(0.0).sqrt()
        if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
            raise ValueError("source modal standard deviations are degenerate")
        return result.contiguous()

    def metadata(self) -> dict[str, object]:
        return {
            "hierarchy_artifact_sha256": self.hierarchy_artifact_sha256,
            "source_model_sha256": self.source_model_sha256,
            "base_artifact_file_sha256": self.base_artifact_file_sha256,
            "base_scientific_payload_sha256": (
                self.base_scientific_payload_sha256
            ),
            "refit_artifact_file_sha256": self.refit_artifact_file_sha256,
            "refit_scientific_payload_sha256": (
                self.refit_scientific_payload_sha256
            ),
            "generator_plan_sha256s": self.generator_plan_sha256s,
            "layer3_factor_sha256": self.layer3_factor_sha256,
            "layer4_factor_sha256": self.layer4_factor_sha256,
            "layer4_modal_singular_count": int(self.S4.numel()),
            "residual_width": self.residual_width,
            "upstream_edge_rank": self.upstream_edge_rank,
        }


def _factor_state(
    raw: object,
    *,
    label: str,
    input_site: str,
    output_site: str,
) -> dict[str, object]:
    state = _require_mapping(raw, label=label)
    _strict_keys(
        state,
        expected={
            "artifact_sha256",
            "input_site",
            "output_site",
            "singular_values",
            "singular_tolerance",
            "restriction",
            "prolongation",
            "input_mean",
            "output_mean",
            "input_support_rank",
            "output_support_rank",
        },
        label=label,
    )
    if state["input_site"] != input_site or state["output_site"] != output_site:
        raise ValueError(f"{label} activation sites drifted")
    _require_sha256(state["artifact_sha256"], label=f"{label} artifact")
    tensors = {
        "singular_values": _canonical_tensor(
            state["singular_values"],
            label=f"{label}.singular_values",
            ndim=1,
        ),
        "restriction": _canonical_tensor(
            state["restriction"],
            label=f"{label}.restriction",
            ndim=2,
        ),
        "prolongation": _canonical_tensor(
            state["prolongation"],
            label=f"{label}.prolongation",
            ndim=2,
        ),
        "input_mean": _canonical_tensor(
            state["input_mean"],
            label=f"{label}.input_mean",
            ndim=1,
        ),
        "output_mean": _canonical_tensor(
            state["output_mean"],
            label=f"{label}.output_mean",
            ndim=1,
        ),
    }
    width = tensors["input_mean"].numel()
    if (
        tensors["output_mean"].numel() != width
        or tensors["restriction"].shape != (width, width)
        or tensors["prolongation"].shape != (width, width)
        or tensors["singular_values"].numel() != width
        or state["input_support_rank"] != width
        or state["output_support_rank"] != width
    ):
        raise ValueError(f"{label} is not the frozen full-rank factor")
    tolerance = state["singular_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (float, int))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0.0
    ):
        raise ValueError(f"{label} singular tolerance is invalid")
    return {**dict(state), **tensors}


def load_gemma3_l3_l4_spectral_reference(
    path: Path | str,
    *,
    expected_file_sha256: str,
    catalog: Gemma3RefitRuntimeCatalog,
) -> Gemma3L3L4SpectralReference:
    """Strict-load only factor/reference state from hierarchy v3."""

    if not isinstance(catalog, Gemma3RefitRuntimeCatalog):
        raise TypeError("catalog must be a Gemma3RefitRuntimeCatalog")
    source = Path(path)
    expected_digest = _require_sha256(
        expected_file_sha256,
        label="expected hierarchy artifact",
    )
    if expected_digest != DEFAULT_HIERARCHY_ARTIFACT_SHA256:
        raise ValueError(
            "expected hierarchy artifact digest must equal the canonical "
            "frozen v3 digest"
        )
    actual_digest = _file_sha256(source)
    if actual_digest != expected_digest:
        raise ValueError("hierarchy artifact file hash differs from frozen v3")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping):
        raise TypeError("hierarchy artifact must contain a mapping")
    _strict_keys(
        raw,
        expected={
            "schema",
            "format_version",
            "scientific_status",
            "binding",
            "protocol",
            "moments",
            "factors",
            "edge_jvp_states",
            "mean_prompt_local_kernel",
            "safe_analysis",
            "safety",
        },
        label="hierarchy artifact",
    )
    if raw["schema"] != _HIERARCHY_SCHEMA or raw["format_version"] != 1:
        raise ValueError("hierarchy artifact is not frozen format v3")
    safety = _require_mapping(raw["safety"], label="hierarchy safety")
    required_safety = {
        "contains_source_model_state_dict": False,
        "contains_tokenizer": False,
        "contains_prompt_text": False,
        "contains_token_ids": False,
        "contains_activation_rows": False,
        "contains_score_gradient_rows": False,
        "contains_executable_low_rank_factors": True,
        "artifact_must_remain_outside_git": True,
    }
    if dict(safety) != required_safety:
        raise ValueError("hierarchy artifact safety declaration drifted")
    status = _require_mapping(
        raw["scientific_status"],
        label="hierarchy scientific status",
    )
    for field in (
        "authorizes_compilation",
        "authorizes_execution",
        "compression_claim",
        "latency_claim",
        "cached_decode_claim",
    ):
        if status.get(field) is not False:
            raise ValueError("hierarchy artifact overclaims scientific status")
    protocol = _require_mapping(raw["protocol"], label="hierarchy protocol")
    if (
        protocol.get("source_scope") != _SOURCE_SCOPE
        or protocol.get("prefill_only") is not True
        or protocol.get("cache_state") != "none"
        or protocol.get("tear_source_site") != _Y3
        or protocol.get("tear_target_site") != _X4
        or tuple(protocol.get("fit_sites", ())) != (_X3, _Y3, _X4, _Y4)
    ):
        raise ValueError("hierarchy protocol is not the frozen L3/L4 rung")
    binding = _require_mapping(raw["binding"], label="hierarchy binding")
    expected_binding = {
        "base_tensor_file_sha256": catalog.base_artifact_file_sha256,
        "base_scientific_payload_sha256": (
            catalog.base_scientific_payload_sha256
        ),
        "refit_tensor_file_sha256": catalog.refit_artifact_file_sha256,
        "refit_scientific_payload_sha256": (
            catalog.refit_scientific_payload_sha256
        ),
        "source_model_sha256": catalog.source_model_sha256,
        "generator_plan_sha256s": catalog.generator_plan_sha256s,
    }
    for field, expected in expected_binding.items():
        if binding.get(field) != expected:
            raise ValueError(f"hierarchy {field} differs from refit runtime")
    factors = _require_mapping(raw["factors"], label="hierarchy factors")
    _strict_keys(
        factors,
        expected={"layer_3", "layer_4"},
        label="hierarchy factors",
    )
    layer3 = _factor_state(
        factors["layer_3"],
        label="layer_3 factor",
        input_site=_X3,
        output_site=_Y3,
    )
    layer4 = _factor_state(
        factors["layer_4"],
        label="layer_4 factor",
        input_site=_X4,
        output_site=_Y4,
    )
    if layer3["input_mean"].shape != layer4["input_mean"].shape:
        raise ValueError("L3/L4 frozen factor widths differ")
    moments = _require_mapping(raw["moments"], label="hierarchy moments")
    sites = _require_mapping(moments.get("sites"), label="hierarchy sites")
    if set(sites) != {_X3, _Y3, _X4, _Y4}:
        raise ValueError("hierarchy moment sites drifted")
    x3_moments = _require_mapping(sites[_X3], label="x3 moments")
    y3_moments = _require_mapping(sites[_Y3], label="y3 moments")
    covariance = _canonical_tensor(
        x3_moments.get("covariance"),
        label="x3 covariance",
        ndim=2,
    )
    measured_y3_mean = _canonical_tensor(
        y3_moments.get("mean"),
        label="measured y3 mean",
        ndim=1,
    )
    if not torch.allclose(
        layer3["output_mean"],
        measured_y3_mean,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError("L3 factor mean differs from frozen y3 moments")
    kernel = _canonical_tensor(
        raw["mean_prompt_local_kernel"],
        label="upstream mean prompt-local kernel",
        ndim=3,
    )
    edge_rank = protocol.get("edge_rank")
    logical_lags = tuple(protocol.get("logical_lags", ()))
    if (
        type(edge_rank) is not int
        or edge_rank <= 0
        or kernel.shape != (len(logical_lags), edge_rank, edge_rank)
        or logical_lags != tuple(range(len(logical_lags)))
    ):
        raise ValueError("upstream prompt-local kernel binding drifted")
    # Prompt-local JVP states and their kernel are not used to construct the
    # prompt-free function.  The aggregate kernel is retained only for the
    # explicitly labeled post-hoc diagnostic comparison.
    return Gemma3L3L4SpectralReference(
        hierarchy_artifact_sha256=actual_digest,
        source_model_sha256=catalog.source_model_sha256,
        base_artifact_file_sha256=catalog.base_artifact_file_sha256,
        base_scientific_payload_sha256=(
            catalog.base_scientific_payload_sha256
        ),
        refit_artifact_file_sha256=catalog.refit_artifact_file_sha256,
        refit_scientific_payload_sha256=(
            catalog.refit_scientific_payload_sha256
        ),
        generator_plan_sha256s=catalog.generator_plan_sha256s,
        layer3_factor_sha256=layer3["artifact_sha256"],  # type: ignore[arg-type]
        layer4_factor_sha256=layer4["artifact_sha256"],  # type: ignore[arg-type]
        x3_mean=layer3["input_mean"],  # type: ignore[arg-type]
        y3_mean=layer3["output_mean"],  # type: ignore[arg-type]
        x4_mean=layer4["input_mean"],  # type: ignore[arg-type]
        y4_mean=layer4["output_mean"],  # type: ignore[arg-type]
        R3=layer3["restriction"],  # type: ignore[arg-type]
        P3=layer3["prolongation"],  # type: ignore[arg-type]
        R4=layer4["restriction"],  # type: ignore[arg-type]
        P4=layer4["prolongation"],  # type: ignore[arg-type]
        S4=layer4["singular_values"],  # type: ignore[arg-type]
        x3_covariance=covariance,
        upstream_mean_prompt_local_kernel=kernel,
    )


def invert_unit_offset_rmsnorm_reference(
    module: nn.Module,
    target: Tensor,
    *,
    epsilon: float,
) -> FixedRMSNormReference:
    """Return the deterministic minimum-norm preimage of Gemma RMSNorm."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor) or weight.ndim != 1:
        raise TypeError("Gemma RMSNorm must expose a one-dimensional weight")
    target64 = _canonical_tensor(target, label="target", ndim=1)
    if target64.shape != weight.shape:
        raise ValueError("RMSNorm target width differs from its weight")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (float, int))
        or not math.isfinite(float(epsilon))
        or float(epsilon) <= 0.0
    ):
        raise ValueError("RMSNorm epsilon must be finite and positive")
    gain = 1.0 + weight.detach().to(device="cpu", dtype=torch.float64)
    gain_tolerance = torch.finfo(torch.float64).eps * 32.0
    null = gain.abs() <= gain_tolerance
    target_scale = max(float(target64.abs().max()), 1.0)
    if bool((target64[null].abs() > target_scale * 1e-10).any()):
        raise ValueError("RMSNorm target is nonzero on a zero-gain coordinate")
    normalized = torch.zeros_like(target64)
    normalized[~null] = target64[~null] / gain[~null]
    q = float(normalized.square().mean())
    if not math.isfinite(q) or q >= 1.0:
        raise ValueError("RMSNorm target has no finite radial preimage")
    alpha = math.sqrt(float(epsilon) / (1.0 - q))
    reference = normalized * alpha
    runtime = reference.to(device=weight.device, dtype=weight.dtype).view(1, -1)
    with torch.no_grad():
        reconstructed = module(runtime)
    if not isinstance(reconstructed, Tensor):
        raise TypeError("Gemma RMSNorm did not return a Tensor")
    reconstructed64 = reconstructed[0].detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    difference = reconstructed64 - target64
    denominator = max(
        float(torch.linalg.vector_norm(target64)),
        torch.finfo(torch.float64).eps,
    )
    relative = float(torch.linalg.vector_norm(difference)) / denominator
    maximum = float(difference.abs().max())
    dtype_tolerance = (
        5e-3
        if weight.dtype in (torch.float16, torch.bfloat16)
        else 5e-6
    )
    if maximum > dtype_tolerance * target_scale:
        raise RuntimeError(
            "canonical RMSNorm preimage failed live reconstruction: "
            f"max_abs={maximum:.6e}"
        )
    return FixedRMSNormReference(
        value=reference,
        target=target64,
        null_gain_indices=tuple(
            int(index)
            for index in torch.nonzero(null, as_tuple=False).flatten().tolist()
        ),
        normalized_second_moment=q,
        radial_scale=alpha,
        reconstruction_max_abs=maximum,
        reconstruction_relative_l2=relative,
        norm_module_sha256=module_state_fingerprint(module),
    )


@dataclass(frozen=True, slots=True)
class Gemma3L3L4SpectralAnalysis:
    """In-memory result before serialization to the ignored artifact."""

    mapping: ModalSpectralMapping
    canonical_l3_post_attention_preimage: FixedRMSNormReference
    source_mode_standard_deviations: Tensor
    baseline_x4_reference: Tensor
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.mapping, ModalSpectralMapping):
            raise TypeError("mapping must be a ModalSpectralMapping")
        self.mapping.validate_integrity()
        if not isinstance(
            self.canonical_l3_post_attention_preimage,
            FixedRMSNormReference,
        ):
            raise TypeError("canonical preimage has an invalid type")
        sigma = _canonical_tensor(
            self.source_mode_standard_deviations,
            label="source_mode_standard_deviations",
            ndim=1,
        )
        baseline = _canonical_tensor(
            self.baseline_x4_reference,
            label="baseline_x4_reference",
            ndim=3,
        )
        if sigma.numel() != self.mapping.source_rank:
            raise ValueError("source sigma width differs from spectral mapping")
        if (
            baseline.shape[0] != 1
            or baseline.shape[1] != len(self.mapping.logical_positions)
        ):
            raise ValueError("baseline x4 does not match the spectral grid")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(self, "source_mode_standard_deviations", sigma)
        object.__setattr__(self, "baseline_x4_reference", baseline)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _finite_ratio(numerator: float, denominator: float) -> float:
    epsilon = torch.finfo(torch.float64).eps
    return numerator / max(denominator, epsilon)


def _cosine(first: Tensor, second: Tensor) -> float:
    first64 = first.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    second64 = second.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    first_norm = float(torch.linalg.vector_norm(first64))
    second_norm = float(torch.linalg.vector_norm(second64))
    if first_norm <= torch.finfo(torch.float64).eps:
        return 1.0 if second_norm <= torch.finfo(torch.float64).eps else 0.0
    if second_norm <= torch.finfo(torch.float64).eps:
        return 0.0
    return max(
        -1.0,
        min(1.0, float(torch.dot(first64, second64)) / (first_norm * second_norm)),
    )


def _float64_vector_sha256(value: Tensor, *, domain: bytes) -> str:
    tensor = _canonical_tensor(value, label="hashed vector", ndim=1)
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": "float64",
                "shape": tuple(int(size) for size in tensor.shape),
            }
        )
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _rank_at_squared_singular_energy(
    singular_values: Tensor,
    fraction: float,
) -> int:
    energy = singular_values.square()
    total = float(energy.sum())
    if total <= torch.finfo(torch.float64).eps:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cumulative, fraction).item()) + 1


def _source_sigma_weighted_spectral_rank_diagnostic(
    *,
    responses: Mapping[str, ModalSpectralResponse],
    source_mode_standard_deviations: Tensor,
) -> dict[str, object]:
    """Measure source-distribution-weighted joint ranks for compression."""

    sigma = _canonical_tensor(
        source_mode_standard_deviations,
        label="source_mode_standard_deviations",
        ndim=1,
    )
    if bool((sigma <= 0).any()):
        raise ValueError("source modal standard deviations must be positive")
    if not responses:
        raise ValueError("at least one spectral response is required")
    response_items = tuple(responses.items())
    if any(
        not isinstance(label, str)
        or not label
        or not isinstance(response, ModalSpectralResponse)
        for label, response in response_items
    ):
        raise TypeError("weighted rank responses are invalid")
    source_modes = response_items[0][1].source_mode_indices
    if any(
        response.source_mode_indices != source_modes
        for _, response in response_items
    ):
        raise ValueError("weighted rank responses use different source modes")
    if any(index >= sigma.numel() for index in source_modes):
        raise ValueError("weighted rank source mode exceeds frozen sigma")
    selected_sigma = sigma[list(source_modes)].contiguous()

    ranks: dict[str, object] = {}
    for label, response in response_items:
        mean_spectrum = torch.complex(
            response.mean_spectral_fingerprint_real,
            response.mean_spectral_fingerprint_imag,
        )
        weighted = mean_spectrum * selected_sigma.reshape(-1, 1, 1)
        singular_values = torch.linalg.svdvals(
            weighted.reshape(weighted.shape[0], -1)
        )
        ranks[label] = {
            "response_label": response.label,
            "joint_rank_90": _rank_at_squared_singular_energy(
                singular_values,
                0.90,
            ),
            "joint_rank_95": _rank_at_squared_singular_energy(
                singular_values,
                0.95,
            ),
            "joint_rank_99": _rank_at_squared_singular_energy(
                singular_values,
                0.99,
            ),
        }

    minimum = float(selected_sigma.min())
    maximum = float(selected_sigma.max())
    return {
        "weight_semantics": (
            "multiply_each_selected_source_row_of_the_mean_complex_"
            "spectrum_by_its_frozen_X3_modal_standard_deviation_before_"
            "joint_svd"
        ),
        "weight_site": _X3,
        "weights_are_probe_amplitudes": False,
        "weights_are_frozen_distribution_scales": True,
        "rank_semantics": (
            "smallest_singular_prefix_reaching_squared_singular_value_"
            "energy_fraction_after_source_sigma_weighting"
        ),
        "rank_is_response_energy_not_downstream_task_accuracy": True,
        "source_mode_indices": source_modes,
        "selected_weight_count": selected_sigma.numel(),
        "selected_weight_minimum": minimum,
        "selected_weight_maximum": maximum,
        "selected_weight_maximum_to_minimum_ratio": maximum / minimum,
        "selected_weights_sha256": _float64_vector_sha256(
            selected_sigma,
            domain=(
                b"fisher-graph:gemma3-l3-l4-source-sigma-weights:v1\0"
            ),
        ),
        "responses": ranks,
    }


def _relative_difference_from_reference(
    measured: Tensor,
    comparison_reference: Tensor,
) -> float:
    measured64 = measured.detach().to(device="cpu", dtype=torch.float64)
    reference64 = comparison_reference.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    return _finite_ratio(
        float(torch.linalg.vector_norm(measured64 - reference64)),
        float(torch.linalg.vector_norm(reference64)),
    )


def _cross_estimator_kernel_comparison(
    canonical_finite_secant: Tensor,
    upstream_prompt_mean_jvp: Tensor,
    *,
    fft_length: int,
) -> dict[str, object]:
    if (
        canonical_finite_secant.ndim != 3
        or canonical_finite_secant.shape != upstream_prompt_mean_jvp.shape
        or fft_length < canonical_finite_secant.shape[0]
    ):
        raise ValueError("kernel comparison geometry is invalid")
    per_lag = tuple(
        {
            "lag": lag,
            "cross_estimator_cosine": _cosine(
                canonical_finite_secant[lag],
                upstream_prompt_mean_jvp[lag],
            ),
            "canonical_finite_secant_relative_difference_from_upstream": (
                _relative_difference_from_reference(
                    canonical_finite_secant[lag],
                    upstream_prompt_mean_jvp[lag],
                )
            ),
            "canonical_finite_secant_frobenius": float(
                torch.linalg.vector_norm(canonical_finite_secant[lag])
            ),
            "upstream_prompt_mean_jvp_frobenius": float(
                torch.linalg.vector_norm(upstream_prompt_mean_jvp[lag])
            ),
        }
        for lag in range(canonical_finite_secant.shape[0])
    )
    canonical_spectrum = torch.fft.rfft(
        canonical_finite_secant.to(dtype=torch.float64),
        n=fft_length,
        dim=0,
    )
    upstream_spectrum = torch.fft.rfft(
        upstream_prompt_mean_jvp.to(dtype=torch.float64),
        n=fft_length,
        dim=0,
    )
    canonical_parts = torch.view_as_real(canonical_spectrum)
    upstream_parts = torch.view_as_real(upstream_spectrum)
    return {
        "lag_count": canonical_finite_secant.shape[0],
        "source_mode_count": canonical_finite_secant.shape[1],
        "target_mode_count": canonical_finite_secant.shape[2],
        "cross_estimator_global_cosine": _cosine(
            canonical_finite_secant,
            upstream_prompt_mean_jvp,
        ),
        "canonical_finite_secant_global_relative_difference_from_upstream": (
            _relative_difference_from_reference(
                canonical_finite_secant,
                upstream_prompt_mean_jvp,
            )
        ),
        "per_lag": per_lag,
        "rfft_length": fft_length,
        "cross_estimator_rfft_cosine": _cosine(
            canonical_parts,
            upstream_parts,
        ),
        "canonical_finite_secant_rfft_relative_difference_from_upstream": (
            _relative_difference_from_reference(
                canonical_parts,
                upstream_parts,
            )
        ),
    }


def _prompt_local_kernel_diagnostic(
    mapping: ModalSpectralMapping,
    reference: Gemma3L3L4SpectralReference,
    *,
    local_label: str,
) -> dict[str, object]:
    local = mapping.symmetric_by_label[local_label]
    lag_count = min(
        5,
        int(local.impulse_responses.shape[2]),
        int(reference.upstream_mean_prompt_local_kernel.shape[0]),
    )
    source_indices = torch.tensor(
        mapping.source_mode_indices,
        dtype=torch.long,
    )
    upstream_prompt_mean_jvp = (
        reference.upstream_mean_prompt_local_kernel[:lag_count]
        .index_select(1, source_indices)
        [..., : mapping.target_rank]
    )
    by_origin: list[dict[str, object]] = []
    kernels: list[Tensor] = []
    for origin_index, origin in enumerate(mapping.impulse_logical_positions):
        kernel = local.impulse_responses[
            :,
            origin_index,
            :lag_count,
            :,
        ].permute(1, 0, 2)
        kernels.append(kernel)
        by_origin.append(
            {
                "impulse_logical_position": origin,
                **_cross_estimator_kernel_comparison(
                    kernel,
                    upstream_prompt_mean_jvp,
                    fft_length=mapping.fft_length,
                ),
            }
        )
    pooled = torch.stack(kernels, dim=0).mean(dim=0)
    return {
        "comparison_scope": (
            "selected_source_modes_and_first_target_rank_coordinates"
        ),
        "source_mode_indices": mapping.source_mode_indices,
        "target_mode_indices": tuple(range(mapping.target_rank)),
        "comparison_kind": "cross_estimator_descriptive_similarity",
        "canonical_probe_estimator": {
            "kind": "symmetric_finite_central_secant",
            "reference_state": (
                "one_repeated_canonical_minimum_norm_rmsnorm_preimage"
            ),
            "amplitude_label": local.label,
            "source_mode_amplitudes": local.source_mode_amplitudes,
        },
        "upstream_artifact_estimator": {
            "kind": (
                "arithmetic_mean_of_prompt_local_ridge_fitted_randomized_"
                "exact_jvp_causal_lag_kernels"
            ),
            "reference_state": (
                "prompt_conditioned_sequence_states_with_mean_l3_source_path"
            ),
            "stationarity_aggregation": (
                "one_stationary_logical_lag_kernel_per_probe_prompt_then_"
                "arithmetic_mean_across_probe_prompts"
            ),
            "ridge_fitted": True,
        },
        "estimator_mismatch": True,
        "reference_state_mismatch": True,
        "not_ground_truth_or_accuracy_validation": True,
        "relative_difference_denominator": (
            "upstream_prompt_mean_jvp_frobenius_with_machine_floor"
        ),
        "upstream_artifact_role": (
            "existing_in_sample_prompt_conditioned_diagnostic_only"
        ),
        "upstream_kernel_used_to_construct_prompt_free_map": False,
        "heldout_validation": False,
        "prompt_distribution_fidelity_claim": False,
        "per_origin": tuple(by_origin),
        "origin_pooled": _cross_estimator_kernel_comparison(
            pooled,
            upstream_prompt_mean_jvp,
            fft_length=mapping.fft_length,
        ),
    }


def _unique_parameter_count(modules: Sequence[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return total


def _projection_parameter_counts(module: nn.Module) -> tuple[int, int]:
    weights = 0
    biases = 0
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        projection = getattr(module, name, None)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"L4 attention must expose linear {name}")
        weights += projection.weight.numel()
        if projection.bias is not None:
            biases += projection.bias.numel()
    return weights, biases


def analyze_prompt_free_gemma3_l3_l4_spectral_mapping(
    adapter: Gemma3CausalLMAdapter,
    reference: Gemma3L3L4SpectralReference,
    *,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    modal_rank: int = DEFAULT_MODAL_RANK,
    source_mode_indices: Sequence[int] | None = None,
    impulse_logical_positions: Sequence[int] = DEFAULT_IMPULSE_POSITIONS,
    max_lag: int | None = None,
    fft_length: int | None = None,
    local_sigma_fraction: float = DEFAULT_LOCAL_SIGMA_FRACTION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    analyzer: Callable[..., ModalSpectralMapping] = (
        analyze_modal_spectral_mapping
    ),
) -> Gemma3L3L4SpectralAnalysis:
    """Measure the fixed-reference structural map without model inputs."""

    if not isinstance(adapter, Gemma3CausalLMAdapter):
        raise TypeError("adapter must be a Gemma3CausalLMAdapter")
    if not isinstance(reference, Gemma3L3L4SpectralReference):
        raise TypeError("reference must be a Gemma3L3L4SpectralReference")
    if (
        type(sequence_length) is not int
        or sequence_length <= 1
        or type(modal_rank) is not int
        or modal_rank <= 0
        or modal_rank > reference.upstream_edge_rank
        or modal_rank > reference.residual_width
    ):
        raise ValueError("sequence length or modal rank is invalid")
    if (
        isinstance(local_sigma_fraction, bool)
        or not isinstance(local_sigma_fraction, (float, int))
        or not math.isfinite(float(local_sigma_fraction))
        or not 0.0 < float(local_sigma_fraction) < 1.0
    ):
        raise ValueError("local sigma fraction must be between zero and one")
    if not callable(analyzer):
        raise TypeError("analyzer must be callable")
    modes = (
        tuple(range(min(DEFAULT_SOURCE_MODE_COUNT, modal_rank)))
        if source_mode_indices is None
        else tuple(source_mode_indices)
    )
    if (
        not modes
        or tuple(sorted(set(modes))) != modes
        or any(
            type(index) is not int
            or index < 0
            or index >= modal_rank
            for index in modes
        )
    ):
        raise ValueError("source mode indices must be unique sorted rank indices")
    origins = tuple(impulse_logical_positions)
    if (
        not origins
        or tuple(sorted(set(origins))) != origins
        or any(
            type(position) is not int
            or position < 0
            or position >= sequence_length
            for position in origins
        )
    ):
        raise ValueError("impulse positions must be unique valid positions")
    maximum_fully_observed_lag = sequence_length - 1 - max(origins)
    if max_lag is None:
        max_lag = maximum_fully_observed_lag
    if (
        type(max_lag) is not int
        or max_lag < 0
        or max_lag > maximum_fully_observed_lag
    ):
        raise ValueError(
            "max_lag must be observed from every impulse origin; increase "
            "sequence_length or move the latest origin earlier"
        )
    if len(adapter.layers) <= 4:
        raise ValueError("Gemma adapter does not contain layers 3 and 4")
    if adapter.module.training or any(
        parameter.requires_grad for parameter in adapter.module.parameters()
    ):
        raise ValueError("spectral mapping requires a frozen eval Gemma model")

    layer3_spec = adapter.layer("layer.3")
    layer4_spec = adapter.layer("layer.4")
    layer3 = adapter.source_module(layer3_spec.id)
    layer4 = adapter.source_module(layer4_spec.id)
    required3 = ("pre_feedforward_layernorm", "post_feedforward_layernorm")
    required4 = (
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
    )
    if any(
        not isinstance(getattr(layer3, name, None), nn.Module)
        for name in required3
    ):
        raise TypeError("live Gemma L3 lacks required normalization modules")
    if any(
        not isinstance(getattr(layer4, name, None), nn.Module)
        for name in required4
    ):
        raise TypeError("live Gemma L4 lacks its exact attention-prefix modules")
    pre_ff3 = layer3.pre_feedforward_layernorm
    post_ff3 = layer3.post_feedforward_layernorm
    transformer3 = layer3_spec.transformer
    if (
        transformer3 is None
        or transformer3.feed_forward_input_norm.kind != "rms_norm"
        or transformer3.feed_forward_input_norm.scale_parameterization
        != "unit_offset"
    ):
        raise ValueError("Gemma L3 pre-feedforward norm semantics drifted")
    fixed_reference = invert_unit_offset_rmsnorm_reference(
        pre_ff3,
        reference.x3_mean,
        epsilon=transformer3.feed_forward_input_norm.epsilon,
    )
    first_parameter = next(adapter.module.parameters(), None)
    if first_parameter is None or not first_parameter.is_floating_point():
        raise TypeError("live Gemma model has no floating parameters")
    device = first_parameter.device
    dtype = first_parameter.dtype
    if device != next(pre_ff3.parameters()).device:
        raise ValueError("Gemma L3 normalization is on a different device")

    # These zero embeddings exist solely to ask the adapter for its canonical
    # cache-free prefill grid. They are deleted before any model computation.
    placeholder = torch.zeros(
        (1, sequence_length, reference.residual_width),
        dtype=dtype,
        device=device,
    )
    attention_mask = torch.ones(
        (1, sequence_length),
        dtype=torch.bool,
        device=device,
    )
    logical_positions = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    sequence = adapter.prepare_sequence(
        {
            "inputs_embeds": placeholder,
            "attention_mask": attention_mask,
            "position_ids": logical_positions,
        }
    )
    del placeholder

    preimage = fixed_reference.value.to(device=device, dtype=dtype).view(1, 1, -1)
    preimage = preimage.expand(1, sequence_length, -1)
    y3_mean = reference.y3_mean.to(device=device, dtype=dtype).view(1, 1, -1)
    y3_mean = y3_mean.expand(1, sequence_length, -1)
    P3 = reference.P3[:, :modal_rank].to(device=device, dtype=dtype)
    R4 = reference.R4[:modal_rank].to(device=device, dtype=dtype)
    sigma = reference.source_mode_standard_deviations(modal_rank)
    sigma_runtime = sigma.to(device=device, dtype=dtype)
    segment4 = adapter.segment("layer.4")
    with torch.no_grad():
        hidden3_reference = preimage + post_ff3(y3_mean)
        baseline_prefix = adapter.run_attention_prefix(
            segment4,
            hidden3_reference,
            sequence,
        )
        baseline_x4 = baseline_prefix.normalized_mlp_input.detach()
    call_count = 0

    def structural_map(source_modes: Tensor) -> Tensor:
        nonlocal call_count
        if (
            not isinstance(source_modes, Tensor)
            or source_modes.shape
            != (1, sequence_length, modal_rank)
            or source_modes.device != device
            or source_modes.dtype != dtype
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError("source modes differ from the structural map ABI")
        call_count += 1
        with torch.no_grad():
            y3 = y3_mean + source_modes @ P3.T
            hidden3 = preimage + post_ff3(y3)
            x4 = adapter.run_attention_prefix(
                segment4,
                hidden3,
                sequence,
            ).normalized_mlp_input
            return (x4 - baseline_x4) @ R4.T

    baseline_modes = torch.zeros(
        (1, sequence_length, modal_rank),
        dtype=dtype,
        device=device,
    )
    mapping = analyzer(
        structural_map,
        baseline_modes=baseline_modes,
        logical_positions=logical_positions,
        valid_mask=attention_mask,
        source_mode_indices=modes,
        impulse_logical_positions=origins,
        max_lag=max_lag,
        fft_length=fft_length,
        finite_impulse_amplitudes=sigma_runtime,
        symmetric_amplitude_sets={
            "local_fraction_sigma": (
                sigma_runtime * float(local_sigma_fraction)
            ),
            "operating_1_sigma": sigma_runtime,
        },
        similarity_threshold=similarity_threshold,
    )
    mapping.validate_integrity()
    if call_count != mapping.function_evaluation_count:
        raise RuntimeError("spectral function evaluation accounting drifted")
    if mapping.source_rank != modal_rank or mapping.target_rank != modal_rank:
        raise RuntimeError("spectral mapping rank differs from the frozen basis")

    local = mapping.symmetric_by_label["local_fraction_sigma"]
    operating = mapping.symmetric_by_label["operating_1_sigma"]
    scale_similarity = mapping.scale_similarity(
        "local_fraction_sigma",
        "operating_1_sigma",
    )
    source_sigma_weighted_ranks = (
        _source_sigma_weighted_spectral_rank_diagnostic(
            responses={
                "local": local,
                "operating": operating,
            },
            source_mode_standard_deviations=sigma,
        )
    )
    exercised_modules = (
        post_ff3,
        layer4.input_layernorm,
        layer4.self_attn,
        layer4.post_attention_layernorm,
        layer4.pre_feedforward_layernorm,
    )
    (
        projection_weight_parameters,
        projection_bias_parameters,
    ) = _projection_parameter_counts(layer4.self_attn)
    prefix_parameters = _unique_parameter_count(exercised_modules)
    p3_decode_macs_per_evaluation = sequence_length * P3.numel()
    r4_project_macs_per_evaluation = sequence_length * R4.numel()
    attention_projection_macs_per_prefix = (
        sequence_length * projection_weight_parameters
    )
    artifact_macs_per_evaluation = (
        p3_decode_macs_per_evaluation + r4_project_macs_per_evaluation
    )
    per_evaluation_linear_macs = (
        artifact_macs_per_evaluation + attention_projection_macs_per_prefix
    )
    mapping_linear_macs = (
        per_evaluation_linear_macs * mapping.function_evaluation_count
    )
    baseline_linear_macs = attention_projection_macs_per_prefix
    total_counted_linear_macs = (
        mapping_linear_macs + baseline_linear_macs
    )
    diagnostics = {
        "reference": {
            **fixed_reference.metadata(),
            "baseline_x4_rms": float(
                baseline_x4.detach().float().square().mean().sqrt()
            ),
            "canonical_preimage_is_not_mean_post_attention": True,
            "nullspace_choice_can_change_downstream_response": bool(
                fixed_reference.null_gain_indices
            ),
        },
        "spectral_findings": {
            "local": local.metadata(),
            "operating": operating.metadata(),
            "local_vs_operating_per_source_similarity": tuple(
                float(value) for value in scale_similarity.tolist()
            ),
            "local_vs_operating_mean_similarity": float(
                scale_similarity.mean()
            ),
            "local_vs_operating_minimum_similarity": float(
                scale_similarity.min()
            ),
            "source_sigma_weighted_spectral_ranks": (
                source_sigma_weighted_ranks
            ),
            "canonical_finite_secant_vs_upstream_prompt_mean_jvp_lag0_to_lag4": (
                _prompt_local_kernel_diagnostic(
                    mapping,
                    reference,
                    local_label="local_fraction_sigma",
                )
            ),
        },
        "resource_accounting": {
            "learned_live_prefix_parameters_exercised": prefix_parameters,
            "l4_attention_projection_weight_parameters_exercised": (
                projection_weight_parameters
            ),
            "l4_attention_projection_bias_parameters_exercised": (
                projection_bias_parameters
            ),
            "l4_attention_projection_parameters_exercised_including_bias": (
                projection_weight_parameters + projection_bias_parameters
            ),
            "l4_attention_projection_bias_free": (
                projection_bias_parameters == 0
            ),
            "artifact_P3_coefficients_used": P3.numel(),
            "artifact_R4_coefficients_used": R4.numel(),
            "artifact_modal_coefficients_used": P3.numel() + R4.numel(),
            "l3_pre_ff_norm_parameters_accessed_for_reference_only": sum(
                parameter.numel() for parameter in pre_ff3.parameters()
            ),
            "counted_linear_macs_per_function_evaluation": (
                per_evaluation_linear_macs
            ),
            "artifact_P3_decode_linear_macs_per_function_evaluation": (
                p3_decode_macs_per_evaluation
            ),
            "artifact_R4_projection_linear_macs_per_function_evaluation": (
                r4_project_macs_per_evaluation
            ),
            "artifact_decode_and_projection_linear_macs_per_function_evaluation": (
                artifact_macs_per_evaluation
            ),
            "l4_attention_projection_weight_macs_per_prefix_execution": (
                attention_projection_macs_per_prefix
            ),
            "counted_linear_macs_mapping_function_evaluations": (
                mapping_linear_macs
            ),
            "counted_linear_macs_baseline_prefix": baseline_linear_macs,
            "counted_linear_macs_total_experiment": (
                total_counted_linear_macs
            ),
            "mac_count_excludes": (
                "linear_bias_additions",
                "normalization",
                "RoPE",
                "attention_score_and_value_matmuls",
                "softmax",
                "elementwise_and_residual_ops",
            ),
            "native_or_compiled_l3_mlp_body_executions": 0,
            "native_or_compiled_l4_mlp_body_executions": 0,
            "l4_attention_prefix_executions": (
                mapping.function_evaluation_count + 1
            ),
            "baseline_prefix_execution_is_outside_mapping_eval_count": True,
            "runtime_speedup_claim": False,
        },
        "scientific_scope": {
            "fixed_reference_interventional_causal_influence": True,
            "shift_invariant_convolution_proven": False,
            "one_origin_fft_is_lti_proof": False,
            "multi_origin_shift_stability_is_diagnostic_only": True,
            "semantic_equivalence_proven": False,
            "clustered_mode_interchangeability_proven": False,
            "prompt_distribution_fidelity_proven": False,
            "compression_claim": False,
            "speed_claim": False,
            "spectral_sparsity_is_only_an_optimization_opportunity": True,
            "unstable_origin_similarity_next_step": (
                "use conditional spectra or an STFT-style state-conditioned map"
            ),
        },
    }
    return Gemma3L3L4SpectralAnalysis(
        mapping=mapping,
        canonical_l3_post_attention_preimage=fixed_reference,
        source_mode_standard_deviations=sigma,
        baseline_x4_reference=baseline_x4,
        diagnostics=diagnostics,
    )


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    report = destination.with_suffix(".json")
    if destination.exists() or report.exists():
        raise FileExistsError("refusing to overwrite spectral-map output")
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
                    "spectral-map output inside the worktree must remain "
                    "under an ignored .local-runs directory"
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
class _OutputPairReservation:
    destination: Path
    report: Path
    claim: Path
    claim_identity: tuple[int, int]
    released: bool = False
    published: bool = False

    def __enter__(self) -> _OutputPairReservation:
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

    def publish(self, tensor_stage: Path, report_stage: Path) -> None:
        if self.released or self.published:
            raise RuntimeError("output reservation is not publishable")
        if _path_identity(self.claim) != self.claim_identity:
            raise RuntimeError("output reservation ownership was lost")
        if self.destination.exists() or self.report.exists():
            raise FileExistsError("refusing to overwrite spectral-map output")
        tensor_identity = _path_identity(tensor_stage)
        report_identity = _path_identity(report_stage)
        if tensor_identity is None or report_identity is None:
            raise FileNotFoundError("staged spectral-map output is missing")
        tensor_published = False
        try:
            # Hard links publish complete same-filesystem staging files while
            # preserving O_EXCL-style no-overwrite behavior.
            os.link(tensor_stage, self.destination)
            tensor_published = True
            os.link(report_stage, self.report)
        except FileExistsError as error:
            if tensor_published:
                _unlink_if_identity(self.destination, tensor_identity)
            raise FileExistsError(
                "refusing to overwrite spectral-map output"
            ) from error
        except BaseException:
            if tensor_published:
                _unlink_if_identity(self.destination, tensor_identity)
            _unlink_if_identity(self.report, report_identity)
            raise
        self.published = True


def _reserve_output_pair(path: Path | str) -> _OutputPairReservation:
    destination = _validate_output_path(path)
    report = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.with_name(f".{destination.name}.publish.lock")
    try:
        descriptor = os.open(
            claim,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "spectral-map output pair is already reserved"
        ) from error
    try:
        state = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reservation = _OutputPairReservation(
        destination=destination,
        report=report,
        claim=claim,
        claim_identity=(state.st_dev, state.st_ino),
    )
    if destination.exists() or report.exists():
        reservation.release()
        raise FileExistsError("refusing to overwrite spectral-map output")
    return reservation


def _new_staging_path(destination: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _stage_torch_save(value: object, destination: Path) -> Path:
    temporary = _new_staging_path(destination)
    try:
        torch.save(value, temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_json_save(
    value: Mapping[str, object],
    destination: Path,
) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
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
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_and_publish_output_pair(
    *,
    reservation: _OutputPairReservation,
    artifact: Mapping[str, object],
    report_builder: Callable[[str, int], Mapping[str, object]],
) -> dict[str, object]:
    """Validate and exclusively publish one matched tensor/report pair."""

    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch_save(
            dict(artifact),
            reservation.destination,
        )
        tensor_sha256 = _file_sha256(tensor_stage)
        tensor_bytes = tensor_stage.stat().st_size
        report = dict(report_builder(tensor_sha256, tensor_bytes))
        if "report_sha256" in report:
            raise ValueError("report builder must not set report_sha256")
        report["report_sha256"] = _report_sha256(report)
        # Validate the complete report before either final file is visible.
        _canonical_json_bytes(report)
        report_stage = _stage_json_save(report, reservation.report)
        reservation.publish(tensor_stage, report_stage)
        return report
    finally:
        if tensor_stage is not None:
            tensor_stage.unlink(missing_ok=True)
        if report_stage is not None:
            report_stage.unlink(missing_ok=True)


def _load_local_gemma3_model_only(
    *,
    model_id: str,
    revision: str,
    cache_dir: Path,
    device: torch.device,
    dtype: str,
) -> nn.Module:
    """Load only the local causal LM; no tokenizer object is constructed."""

    try:
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "Gemma support is optional; install it with "
            '`pip install -e ".[gemma]"`'
        ) from error
    dtype_values: dict[str, str | torch.dtype] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        runtime_dtype = dtype_values[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {dtype!r}") from error
    match = re.match(r"^(\d+)", transformers.__version__)
    if match is None:
        raise RuntimeError("could not determine Transformers major version")
    dtype_keyword = "dtype" if int(match.group(1)) >= 5 else "torch_dtype"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            attn_implementation="eager",
            **{dtype_keyword: runtime_dtype},
        )
    except OSError as error:
        raise RuntimeError(
            f"could not load the pinned local model {model_id!r}"
        ) from error
    if not isinstance(model, nn.Module):
        raise TypeError("AutoModelForCausalLM did not return an nn.Module")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    disable_checkpointing = getattr(
        model,
        "gradient_checkpointing_disable",
        None,
    )
    if callable(disable_checkpointing):
        disable_checkpointing()
    return model


def _analysis_artifact(
    *,
    analysis: Gemma3L3L4SpectralAnalysis,
    reference: Gemma3L3L4SpectralReference,
    model_id: str,
    revision: str,
    sequence_length: int,
    modal_rank: int,
    local_sigma_fraction: float,
) -> dict[str, object]:
    mapping = analysis.mapping
    return {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "scientific_status": {
            "scope": "prompt_free_fixed_reference_l3_l4_structural_map",
            "fixed_reference_interventional_causal_influence": True,
            "shift_invariant_convolution_claim": False,
            "semantic_equivalence_claim": False,
            "prompt_distribution_fidelity_claim": False,
            "compression_claim": False,
            "latency_or_speed_claim": False,
            "cached_decode_claim": False,
            "development_only": True,
        },
        "binding": reference.metadata(),
        "model": {
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_commit": revision,
            "source_model_sha256": reference.source_model_sha256,
            "local_files_only": True,
            "tokenizer_loaded": False,
        },
        "protocol": {
            "source_scope": _SOURCE_SCOPE,
            "sequence_length": sequence_length,
            "modal_rank": modal_rank,
            "source_mode_indices": mapping.source_mode_indices,
            "impulse_logical_positions": (
                mapping.impulse_logical_positions
            ),
            "logical_positions": mapping.logical_positions,
            "valid_mask": mapping.valid_mask,
            "prefill_only": True,
            "cache_state": "none",
            "placeholder_inputs_embeds_used_for_context_only": True,
            "placeholder_inputs_embeds_forwarded": False,
            "finite_amplitude": "one_source_modal_standard_deviation",
            "symmetric_amplitudes": {
                "local_fraction_sigma": local_sigma_fraction,
                "operating_1_sigma": 1.0,
            },
            "upstream_fisher_modes_are_data_conditioned": True,
            "new_prompt_text_loaded": False,
            "new_token_ids_loaded": False,
            "upstream_prompt_fitted_kernel_used_to_construct_map": False,
        },
        "canonical_reference": {
            "metadata": (
                analysis.canonical_l3_post_attention_preimage.metadata()
            ),
            "l3_post_attention_preimage": (
                analysis.canonical_l3_post_attention_preimage.value.clone()
            ),
            "x3_mean_target": (
                analysis.canonical_l3_post_attention_preimage.target.clone()
            ),
            "source_mode_standard_deviations": (
                analysis.source_mode_standard_deviations.clone()
            ),
            "baseline_x4_reference": (
                analysis.baseline_x4_reference.clone()
            ),
        },
        "spectral_mapping": mapping.state_dict(),
        "safe_analysis": dict(analysis.diagnostics),
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_prompt_activation_rows": False,
            "contains_score_gradient_rows": False,
            "contains_structural_reference_rows": True,
            "contains_spectral_response_tensors": True,
            "artifact_must_remain_outside_git": True,
        },
    }


def run_gemma3_l3_l4_spectral_mapping_experiment(
    *,
    hierarchy_artifact_path: Path | str = DEFAULT_HIERARCHY_ARTIFACT,
    hierarchy_artifact_sha256: str = DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    modal_rank: int = DEFAULT_MODAL_RANK,
    source_mode_indices: Sequence[int] | None = None,
    impulse_logical_positions: Sequence[int] = DEFAULT_IMPULSE_POSITIONS,
    max_lag: int | None = None,
    fft_length: int | None = None,
    local_sigma_fraction: float = DEFAULT_LOCAL_SIGMA_FRACTION,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, object]:
    """Run and serialize the prompt-free fixed-reference structural map."""

    destination = _validate_output_path(output)
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a nonempty string")
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    reference = load_gemma3_l3_l4_spectral_reference(
        hierarchy_artifact_path,
        expected_file_sha256=hierarchy_artifact_sha256,
        catalog=catalog,
    )
    metadata = dict(catalog.model_metadata)
    if (
        metadata.get("model_id") != model_id
        or metadata.get("requested_revision") != revision
        or metadata.get("resolved_commit") != revision
        or metadata.get("local_files_only") is not True
    ):
        raise ValueError("requested model differs from the frozen refit")
    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    model = _load_local_gemma3_model_only(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
    )
    adapter = Gemma3CausalLMAdapter(model)
    source_fingerprint = adapter.model_fingerprint()
    if source_fingerprint != reference.source_model_sha256:
        raise ValueError("live Gemma fingerprint differs from frozen hierarchy")

    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_SOURCE_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_SOURCE_SCOPE)
        analysis = analyze_prompt_free_gemma3_l3_l4_spectral_mapping(
            adapter,
            reference,
            sequence_length=sequence_length,
            modal_rank=modal_rank,
            source_mode_indices=source_mode_indices,
            impulse_logical_positions=impulse_logical_positions,
            max_lag=max_lag,
            fft_length=fft_length,
            local_sigma_fraction=local_sigma_fraction,
            similarity_threshold=similarity_threshold,
        )
    finally:
        switcher.close()
    if adapter.model_fingerprint() != source_fingerprint:
        raise RuntimeError("spectral experiment did not restore the Gemma model")

    artifact = _analysis_artifact(
        analysis=analysis,
        reference=reference,
        model_id=model_id,
        revision=revision,
        sequence_length=sequence_length,
        modal_rank=modal_rank,
        local_sigma_fraction=float(local_sigma_fraction),
    )
    reservation = _reserve_output_pair(destination)
    try:
        def build_report(
            artifact_file_sha256: str,
            artifact_file_bytes: int,
        ) -> Mapping[str, object]:
            return {
                "schema": _SCHEMA,
                "format_version": _FORMAT_VERSION,
                "scientific_status": artifact["scientific_status"],
                "binding": reference.metadata(),
                "model": artifact["model"],
                "protocol": artifact["protocol"],
                "analysis": {
                    "spectral_mapping": analysis.mapping.metadata(),
                    **dict(analysis.diagnostics),
                },
                "artifact": {
                    "tensor_file": str(destination),
                    "tensor_file_sha256": artifact_file_sha256,
                    "tensor_file_bytes": artifact_file_bytes,
                    "report_file": str(destination.with_suffix(".json")),
                    "committable": False,
                },
                "safety": artifact["safety"],
                "interpretation": {
                    "finding_scope": (
                        "fixed-reference interventional L3-to-L4 "
                        "structural influence"
                    ),
                    "spectral_sparsity_interpretation": (
                        "an optimization opportunity, not compression "
                        "by itself"
                    ),
                    "prompt_local_comparison_role": (
                        "cross-estimator in-sample diagnostic with "
                        "different reference states; not ground truth "
                        "or accuracy validation"
                    ),
                    "not_proven": (
                        "shift-invariant convolution, semantic equivalence, "
                        "clustered-mode interchangeability, "
                        "prompt-distribution fidelity, compression, or speed"
                    ),
                },
            }

        return _stage_and_publish_output_pair(
            reservation=reservation,
            artifact=artifact,
            report_builder=build_report,
        )
    finally:
        reservation.release()


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a prompt-free fixed-reference Gemma L3-to-L4 "
            "modal spectral map."
        )
    )
    parser.add_argument(
        "--hierarchy-artifact",
        type=Path,
        default=DEFAULT_HIERARCHY_ARTIFACT,
    )
    parser.add_argument(
        "--hierarchy-artifact-sha256",
        default=DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    )
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
    )
    parser.add_argument("--modal-rank", type=int, default=DEFAULT_MODAL_RANK)
    source_modes = parser.add_mutually_exclusive_group()
    source_modes.add_argument(
        "--source-mode-indices",
        type=_comma_separated_ints,
        default=None,
        help=(
            "comma-separated source modes; default selects up to the first "
            f"{DEFAULT_SOURCE_MODE_COUNT} modes within --modal-rank"
        ),
    )
    source_modes.add_argument(
        "--all-source-modes",
        action="store_true",
        help="probe every source mode through --modal-rank",
    )
    parser.add_argument(
        "--impulse-positions",
        type=_comma_separated_ints,
        default=DEFAULT_IMPULSE_POSITIONS,
    )
    parser.add_argument("--max-lag", type=int)
    parser.add_argument("--fft-length", type=int)
    parser.add_argument(
        "--local-sigma-fraction",
        type=float,
        default=DEFAULT_LOCAL_SIGMA_FRACTION,
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_spectral_mapping_experiment(
        hierarchy_artifact_path=arguments.hierarchy_artifact,
        hierarchy_artifact_sha256=arguments.hierarchy_artifact_sha256,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        output=arguments.output,
        model_id=arguments.model_id,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        sequence_length=arguments.sequence_length,
        modal_rank=arguments.modal_rank,
        source_mode_indices=(
            tuple(range(arguments.modal_rank))
            if arguments.all_source_modes
            else arguments.source_mode_indices
        ),
        impulse_logical_positions=arguments.impulse_positions,
        max_lag=arguments.max_lag,
        fft_length=arguments.fft_length,
        local_sigma_fraction=arguments.local_sigma_fraction,
        similarity_threshold=arguments.similarity_threshold,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
