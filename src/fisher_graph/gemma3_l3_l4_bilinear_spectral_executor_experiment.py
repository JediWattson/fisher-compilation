"""Compile and independently assess a Gemma L3->L4 bilinear correction.

This development protocol adds one deliberately narrow branch to the frozen
linear plus diagonal-square conditional spectral executor.  The branch sees
all 28 explicit products over the eight source modes selected *before* these
measurements from quadratic-source-basis leverage:

``phi_ij(m) = 2 * (m_i / sigma_i) * (m_j / sigma_j)``.

A two-mode chord whose components are
``(+/- rho * sigma / sqrt(2))`` therefore emits exactly
``(+/- rho**2)`` in its matching feature.  Singleton axes and every pair
outside the frozen set are structural zeros.

Compilation and assessment are separate commands.  Compilation:

1. measures only fit origins 8/24/40;
2. fits the complete preregistered rate ladder;
3. only then opens selection origins 16/32; and
4. publishes a candidate only when a frozen selection row passes.

Assessment strict-authenticates that candidate before opening origin 20 and
contains no fitting or gate-relaxation path.  A failed selection publishes a
source-safe JSON diagnosis and no candidate tensor.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter
from .conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
)
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
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    Gemma3ConditionalSpectralCandidate,
    _reserve_outputs,
    _stage_json,
    _stage_path,
    _stage_torch,
    _validate_output_path,
    load_gemma3_conditional_spectral_candidate,
)
from .gemma3_l3_l4_mixed_mode_falsification_experiment import (
    DEFAULT_CANDIDATE,
    DEFAULT_CANDIDATE_FILE_SHA256,
    DEFAULT_CANDIDATE_REPORT_SHA256,
    _prepare_runtime,
    _response_slice,
    _runtime_response,
    _source_row,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    DEFAULT_HIERARCHY_ARTIFACT,
    DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    DEFAULT_REVISION,
    _load_local_gemma3_model_only,
    invert_unit_offset_rmsnorm_reference,
    load_gemma3_l3_l4_spectral_reference,
)
from .off_diagonal_bilinear_modal import (
    ExplicitPairProductFeatureMap,
    build_explicit_pair_product_feature_map,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "ASSESSMENT_CONTROL_PAIRS",
    "ASSESSMENT_ORIGIN",
    "DEFAULT_ASSESSMENT_OUTPUT",
    "DEFAULT_OUTPUT",
    "FIT_ORIGINS",
    "FrozenBilinearSpectralProtocol",
    "Gemma3BilinearSpectralCandidate",
    "MeasuredBilinearPanel",
    "POSITIVE_PAIRS",
    "PreparedGemma3BilinearSpectralBranch",
    "SELECTION_CONTROL_PAIRS",
    "SELECTION_ORIGINS",
    "SENSITIVE_MODES",
    "assess_gemma3_l3_l4_bilinear_spectral_executor",
    "build_parser",
    "compile_bilinear_spectral_candidate",
    "compile_gemma3_l3_l4_bilinear_spectral_executor",
    "default_bilinear_spectral_protocol",
    "evaluate_bilinear_plan",
    "load_gemma3_bilinear_spectral_candidate",
    "main",
    "measure_bilinear_panel",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-bilinear-spectral-executor-dev-v1.pt"
)
DEFAULT_ASSESSMENT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-bilinear-spectral-assessment-dev-v1.pt"
)

FIT_ORIGINS = (8, 24, 40)
SELECTION_ORIGINS = (16, 32)
ASSESSMENT_ORIGIN = 20
SEQUENCE_LENGTH = 72
MODAL_RANK = 64
TARGET_RANK = 64
MAX_LAG = 31
LAG_COUNT = MAX_LAG + 1
FFT_LENGTH = 32
RADII = (0.5, 1.0)
SIGN_ROWS = (
    ("++", 1, 1),
    ("+-", 1, -1),
    ("-+", -1, 1),
    ("--", -1, -1),
)
SENSITIVE_MODES = (0, 1, 2, 7, 15, 28, 42, 43)
POSITIVE_PAIRS = tuple(
    (left, right)
    for left_ordinal, left in enumerate(SENSITIVE_MODES)
    for right in SENSITIVE_MODES[left_ordinal + 1 :]
)
SELECTION_CONTROL_PAIRS = (
    (3, 21),
    (3, 47),
    (16, 41),
    (18, 47),
    (37, 44),
    (41, 44),
)
ASSESSMENT_CONTROL_PAIRS = (
    (3, 37),
    (16, 18),
    (16, 21),
    (18, 37),
    (21, 44),
    (41, 47),
)
SPECTRAL_RANK_LADDER = (
    (4, 6),
    (8, 8),
    (12, 12),
    (16, 16),
    (20, 20),
    (24, 24),
    (28, 32),
    (28, 48),
)
RATE_LADDER = (
    ("zero", 0, 0),
    *(("spectral", source, target) for source, target in SPECTRAL_RANK_LADDER),
    ("dense", 28, 64),
)
FROZEN_STORED_COEFFICIENT_COUNTS = (
    0,
    2800,
    6880,
    14928,
    26048,
    40240,
    57504,
    88848,
    132880,
    172032,
)

GATES = MappingProxyType({
    "operating_radius": 1.0,
    "maximum_pooled_c11_relative_error": 0.30,
    "maximum_origin_c11_relative_error": 0.35,
    "minimum_c11_cosine": 0.95,
    "maximum_truth_scale_defect": 0.25,
    "minimum_truth_scale_cosine": 0.95,
    "maximum_augmented_full_mixed_relative_error": 0.225,
    "minimum_augmented_full_mixed_cosine": 0.975,
    "minimum_selection_pooled_error_reduction": 0.10,
    "minimum_selection_origin_error_reduction": 0.05,
    "minimum_assessment_error_reduction": 0.10,
    "minimum_c11_oracle_headroom": 0.10,
    "minimum_oracle_recovery_fraction": 0.50,
    "maximum_control_pooled_e11": 0.075,
    "maximum_reliable_control_pair_e11": 0.15,
    "control_reliability_minimum_panel_median_rms_fraction": 0.25,
    "control_reliability_minimum_two_c11_noise_multiple": 10.0,
})

_SCHEMA = "fisher_graph.gemma3_l3_l4_bilinear_spectral_executor_development"
_ASSESSMENT_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_bilinear_spectral_assessment_development"
)
_FAILURE_SCHEMA = (
    "fisher_graph.gemma3_l3_l4_bilinear_spectral_selection_failure"
)
_DENSE_PLAN_KIND = "fisher_graph.dense_position_bilinear_plan"
_FORMAT_VERSION = 1
_SOURCE_SCOPE = "factorized_refit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-protocol:v1\0"
_PANEL_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-panel:v1\0"
_DENSE_PLAN_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-dense-plan:v1\0"
_CANDIDATE_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-candidate:v1\0"
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-report:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l3-l4-bilinear-tensor:v1\0"


def _code_sha256s() -> dict[str, str]:
    runner = Path(__file__)
    return {
        "gemma_bilinear_runner": _file_sha256(runner),
        "generic_bilinear_features": _file_sha256(
            runner.with_name("off_diagonal_bilinear_modal.py")
        ),
        "conditional_spectral_generator": _file_sha256(
            runner.with_name("conditional_spectral_generator.py")
        ),
        "mixed_mode_measurement_helpers": _file_sha256(
            runner.with_name(
                "gemma3_l3_l4_mixed_mode_falsification_experiment.py"
            )
        ),
        "spectral_reference_runner": _file_sha256(
            runner.with_name(
                "gemma3_l3_l4_spectral_mapping_experiment.py"
            )
        ),
        "gemma3_adapter": _file_sha256(
            runner.with_name("adapters").joinpath("gemma3.py")
        ),
    }


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


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


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
    if (
        result.ndim != ndim
        or any(int(width) <= 0 for width in result.shape)
        or not bool(torch.isfinite(result).all())
    ):
        raise ValueError(f"{label} must be finite, nonempty, and rank {ndim}")
    return result


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(
        device="cpu", dtype=torch.float64
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(int(width) for width in tensor.shape)).encode())
    digest.update(b"\0float64\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


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


def _relative_error(prediction: Tensor, target: Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(prediction - target))
    denominator = float(torch.linalg.vector_norm(target))
    if denominator <= torch.finfo(torch.float64).eps:
        return 0.0 if numerator <= torch.finfo(torch.float64).eps else math.inf
    return numerator / denominator


def _cosine(first: Tensor, second: Tensor) -> float:
    left = first.reshape(-1).to(dtype=torch.float64)
    right = second.reshape(-1).to(dtype=torch.float64)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    epsilon = torch.finfo(torch.float64).eps
    if left_norm <= epsilon or right_norm <= epsilon:
        return 1.0 if left_norm <= epsilon and right_norm <= epsilon else 0.0
    return float(torch.dot(left, right) / (left_norm * right_norm))


def _error_reduction(
    baseline: Tensor,
    corrected: Tensor,
    truth: Tensor,
) -> float:
    base_error = _relative_error(baseline, truth)
    corrected_error = _relative_error(corrected, truth)
    if base_error <= torch.finfo(torch.float64).eps:
        return 0.0
    return (base_error - corrected_error) / base_error


@dataclass(frozen=True, slots=True)
class FrozenBilinearSpectralProtocol:
    """Complete preregistration constructed before any live response."""

    fit_origins: tuple[int, ...]
    selection_origins: tuple[int, ...]
    assessment_origin: int
    sequence_length: int
    modal_rank: int
    target_rank: int
    max_lag: int
    fft_length: int
    radii: tuple[float, ...]
    sign_rows: tuple[tuple[str, int, int], ...]
    sensitive_modes: tuple[int, ...]
    positive_pairs: tuple[tuple[int, int], ...]
    selection_control_pairs: tuple[tuple[int, int], ...]
    assessment_control_pairs: tuple[tuple[int, int], ...]
    rank_ladder: tuple[tuple[str, int, int], ...]
    gates: Mapping[str, float]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        expected = (
            FIT_ORIGINS,
            SELECTION_ORIGINS,
            ASSESSMENT_ORIGIN,
            SEQUENCE_LENGTH,
            MODAL_RANK,
            TARGET_RANK,
            MAX_LAG,
            FFT_LENGTH,
            RADII,
            SIGN_ROWS,
            SENSITIVE_MODES,
            POSITIVE_PAIRS,
            SELECTION_CONTROL_PAIRS,
            ASSESSMENT_CONTROL_PAIRS,
            RATE_LADDER,
            GATES,
        )
        actual = (
            self.fit_origins,
            self.selection_origins,
            self.assessment_origin,
            self.sequence_length,
            self.modal_rank,
            self.target_rank,
            self.max_lag,
            self.fft_length,
            self.radii,
            self.sign_rows,
            self.sensitive_modes,
            self.positive_pairs,
            self.selection_control_pairs,
            self.assessment_control_pairs,
            self.rank_ladder,
            dict(self.gates),
        )
        if actual != expected:
            raise ValueError("bilinear protocol differs from frozen declaration")
        all_controls = (
            self.selection_control_pairs + self.assessment_control_pairs
        )
        if (
            len(self.positive_pairs) != 28
            or self.positive_pairs != tuple(sorted(self.positive_pairs))
            or len(set(all_controls)) != len(all_controls)
            or set(all_controls) & set(self.positive_pairs)
            or any(
                left >= right
                or left < 0
                or right >= self.modal_rank
                or left in self.sensitive_modes
                or right in self.sensitive_modes
                for left, right in all_controls
            )
            or self.assessment_origin
            in self.fit_origins + self.selection_origins
            or max(
                *self.fit_origins,
                *self.selection_origins,
                self.assessment_origin,
            )
            + self.max_lag
            >= self.sequence_length
        ):
            raise ValueError("bilinear protocol split or pair panel is invalid")
        object.__setattr__(
            self,
            "gates",
            MappingProxyType(dict(self.gates)),
        )
        computed = _json_sha256(self._payload(), domain=_PROTOCOL_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("bilinear protocol hash mismatch")

    def _payload(self) -> dict[str, object]:
        return {
            "fit_origins": self.fit_origins,
            "selection_origins": self.selection_origins,
            "assessment_origin": self.assessment_origin,
            "sequence_length": self.sequence_length,
            "modal_rank": self.modal_rank,
            "target_rank": self.target_rank,
            "max_lag": self.max_lag,
            "lag_count": self.max_lag + 1,
            "fft_length": self.fft_length,
            "radii": self.radii,
            "sign_rows": self.sign_rows,
            "sensitive_modes": self.sensitive_modes,
            "positive_pairs": self.positive_pairs,
            "selection_control_pairs": self.selection_control_pairs,
            "assessment_control_pairs": self.assessment_control_pairs,
            "rank_ladder": self.rank_ladder,
            "feature_semantics": (
                "phi_ij=2*(m_i/sigma_i)*(m_j/sigma_j)"
            ),
            "chord_semantics": (
                "components=sign*rho*sigma/sqrt(2);"
                "matching_feature=sign_left*sign_right*rho^2"
            ),
            "fit_kernel_estimator": (
                "sum_rho(rho^2*C11_residual)/sum_rho(rho^4)"
            ),
            "interpolation": "piecewise_linear_source_origin",
            "selection_rule": (
                "minimal_stored_coefficients_then_source_rank_then_target_rank"
            ),
            "gates": dict(self.gates),
            "gate_values_applied_without_rounding": True,
            "fit_all_ladder_rows_before_selection_measurement": True,
            "assessment_requires_published_passing_candidate": True,
            "zero_control_response_energy_denominator_fails_closed": True,
            "response_measurement_used_for_protocol_selection": False,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def default_bilinear_spectral_protocol() -> FrozenBilinearSpectralProtocol:
    return FrozenBilinearSpectralProtocol(
        fit_origins=FIT_ORIGINS,
        selection_origins=SELECTION_ORIGINS,
        assessment_origin=ASSESSMENT_ORIGIN,
        sequence_length=SEQUENCE_LENGTH,
        modal_rank=MODAL_RANK,
        target_rank=TARGET_RANK,
        max_lag=MAX_LAG,
        fft_length=FFT_LENGTH,
        radii=RADII,
        sign_rows=SIGN_ROWS,
        sensitive_modes=SENSITIVE_MODES,
        positive_pairs=POSITIVE_PAIRS,
        selection_control_pairs=SELECTION_CONTROL_PAIRS,
        assessment_control_pairs=ASSESSMENT_CONTROL_PAIRS,
        rank_ladder=RATE_LADDER,
        gates=GATES,
    )


@dataclass(frozen=True, slots=True)
class MeasuredBilinearPanel:
    """Canonical structural responses for one declared split."""

    split: str
    origins: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    positive_pair_count: int
    radii: tuple[float, ...]
    truth: Tensor
    base_prediction: Tensor
    zero_sentinel: Tensor
    repeat_sentinel_first: Tensor
    repeat_sentinel_second: Tensor
    protocol_sha256: str
    base_candidate_sha256: str
    measurement: Mapping[str, object]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.split not in ("fit", "selection", "assessment"):
            raise ValueError("panel split is invalid")
        if (
            not self.origins
            or tuple(sorted(set(self.origins))) != self.origins
            or not self.pairs
            or len(set(self.pairs)) != len(self.pairs)
            or self.positive_pair_count <= 0
            or self.positive_pair_count > len(self.pairs)
            or self.radii != RADII
        ):
            raise ValueError("panel axes are invalid")
        truth = _canonical_tensor(self.truth, label="truth", ndim=6)
        base = _canonical_tensor(
            self.base_prediction,
            label="base_prediction",
            ndim=6,
        )
        expected = (
            len(self.origins),
            len(self.pairs),
            len(self.radii),
            len(SIGN_ROWS),
            LAG_COUNT,
            TARGET_RANK,
        )
        if tuple(truth.shape) != expected or base.shape != truth.shape:
            raise ValueError("panel response geometry differs from protocol")
        sentinels: dict[str, Tensor] = {}
        for name in (
            "zero_sentinel",
            "repeat_sentinel_first",
            "repeat_sentinel_second",
        ):
            value = _canonical_tensor(
                getattr(self, name),
                label=name,
                ndim=3,
            )
            if tuple(value.shape) != (
                len(self.origins),
                LAG_COUNT,
                TARGET_RANK,
            ):
                raise ValueError(f"{name} geometry differs from protocol")
            sentinels[name] = value
        _sha256(self.protocol_sha256, label="panel protocol")
        _sha256(self.base_candidate_sha256, label="panel base candidate")
        if not isinstance(self.measurement, Mapping):
            raise TypeError("panel measurement must be a mapping")
        _canonical_json_bytes(self.measurement)
        object.__setattr__(self, "truth", truth)
        object.__setattr__(self, "base_prediction", base)
        for name, value in sentinels.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "measurement", dict(self.measurement))
        computed = _json_sha256(self._hash_payload(), domain=_PANEL_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("bilinear panel hash mismatch")

    @property
    def control_pairs(self) -> tuple[tuple[int, int], ...]:
        return self.pairs[self.positive_pair_count :]

    def _hash_payload(self) -> dict[str, object]:
        return {
            "split": self.split,
            "origins": self.origins,
            "pairs": self.pairs,
            "positive_pair_count": self.positive_pair_count,
            "radii": self.radii,
            "truth_sha256": _tensor_sha256(self.truth),
            "base_prediction_sha256": _tensor_sha256(
                self.base_prediction
            ),
            "zero_sentinel_sha256": _tensor_sha256(self.zero_sentinel),
            "repeat_sentinel_first_sha256": _tensor_sha256(
                self.repeat_sentinel_first
            ),
            "repeat_sentinel_second_sha256": _tensor_sha256(
                self.repeat_sentinel_second
            ),
            "response_shape": tuple(self.truth.shape),
            "protocol_sha256": self.protocol_sha256,
            "base_candidate_sha256": self.base_candidate_sha256,
            "measurement": dict(self.measurement),
        }

    def metadata(self) -> dict[str, object]:
        return {**self._hash_payload(), "artifact_sha256": self.artifact_sha256}

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "truth": self.truth.clone(),
            "base_prediction": self.base_prediction.clone(),
            "zero_sentinel": self.zero_sentinel.clone(),
            "repeat_sentinel_first": self.repeat_sentinel_first.clone(),
            "repeat_sentinel_second": self.repeat_sentinel_second.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(
        cls, value: Mapping[str, object]
    ) -> MeasuredBilinearPanel:
        expected = {
            "split",
            "origins",
            "pairs",
            "positive_pair_count",
            "radii",
            "truth_sha256",
            "base_prediction_sha256",
            "zero_sentinel_sha256",
            "repeat_sentinel_first_sha256",
            "repeat_sentinel_second_sha256",
            "response_shape",
            "protocol_sha256",
            "base_candidate_sha256",
            "measurement",
            "truth",
            "base_prediction",
            "zero_sentinel",
            "repeat_sentinel_first",
            "repeat_sentinel_second",
            "artifact_sha256",
        }
        _strict_keys(value, expected=expected, label="measured panel")
        tensor_names = (
            "truth",
            "base_prediction",
            "zero_sentinel",
            "repeat_sentinel_first",
            "repeat_sentinel_second",
        )
        for name in tensor_names:
            raw = value[name]
            if (
                not isinstance(raw, Tensor)
                or raw.dtype != torch.float64
                or raw.device.type != "cpu"
                or not raw.is_contiguous()
                or not bool(torch.isfinite(raw).all())
                or _tensor_sha256(raw) != value[f"{name}_sha256"]
            ):
                raise ValueError(f"serialized panel {name} is not canonical")
        result = cls(
            split=value["split"],  # type: ignore[arg-type]
            origins=tuple(value["origins"]),  # type: ignore[arg-type]
            pairs=tuple(
                tuple(pair) for pair in value["pairs"]  # type: ignore[union-attr]
            ),
            positive_pair_count=value[
                "positive_pair_count"
            ],  # type: ignore[arg-type]
            radii=tuple(value["radii"]),  # type: ignore[arg-type]
            truth=value["truth"],  # type: ignore[arg-type]
            base_prediction=value[
                "base_prediction"
            ],  # type: ignore[arg-type]
            zero_sentinel=value["zero_sentinel"],  # type: ignore[arg-type]
            repeat_sentinel_first=value[
                "repeat_sentinel_first"
            ],  # type: ignore[arg-type]
            repeat_sentinel_second=value[
                "repeat_sentinel_second"
            ],  # type: ignore[arg-type]
            protocol_sha256=value["protocol_sha256"],  # type: ignore[arg-type]
            base_candidate_sha256=value[
                "base_candidate_sha256"
            ],  # type: ignore[arg-type]
            measurement=_mapping(
                value["measurement"], label="panel measurement"
            ),
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
        )
        if tuple(value["response_shape"]) != tuple(result.truth.shape):  # type: ignore[arg-type]
            raise ValueError("serialized panel response shape drifted")
        return result


@dataclass(frozen=True, slots=True)
class DensePositionBilinearPlan:
    """Exact dense fit-knot kernels with piecewise-linear interpolation."""

    fit_knot_origins: tuple[int, ...]
    feature_kernels: Tensor
    response_binding_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _DENSE_PLAN_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.fit_knot_origins != FIT_ORIGINS:
            raise ValueError("dense plan fit knots differ from protocol")
        kernels = _canonical_tensor(
            self.feature_kernels,
            label="dense feature kernels",
            ndim=4,
        )
        if tuple(kernels.shape) != (
            len(FIT_ORIGINS),
            len(POSITIVE_PAIRS),
            LAG_COUNT,
            TARGET_RANK,
        ):
            raise ValueError("dense feature kernel geometry is invalid")
        _sha256(self.response_binding_sha256, label="response binding")
        if (
            self.artifact_kind != _DENSE_PLAN_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("dense plan header is invalid")
        object.__setattr__(self, "feature_kernels", kernels)
        computed = _json_sha256(
            self._hash_payload(), domain=_DENSE_PLAN_DOMAIN
        )
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("dense bilinear plan hash mismatch")

    @property
    def stored_coefficient_count(self) -> int:
        return int(self.feature_kernels.numel())

    @property
    def source_rank(self) -> int:
        return len(POSITIVE_PAIRS)

    @property
    def target_rank(self) -> int:
        return TARGET_RANK

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "fit_knot_origins": self.fit_knot_origins,
            "feature_kernels_sha256": _tensor_sha256(
                self.feature_kernels
            ),
            "feature_kernels_shape": tuple(self.feature_kernels.shape),
            "response_binding_sha256": self.response_binding_sha256,
            "interpolation": "piecewise_linear_source_origin",
            "stored_coefficient_count": self.stored_coefficient_count,
        }

    def validate_integrity(self) -> None:
        if (
            self.feature_kernels.dtype != torch.float64
            or self.feature_kernels.device.type != "cpu"
            or not self.feature_kernels.is_contiguous()
            or _json_sha256(
                self._hash_payload(), domain=_DENSE_PLAN_DOMAIN
            )
            != self.artifact_sha256
        ):
            raise ValueError("dense bilinear plan integrity check failed")

    def linear_kernel_at_origin(self, origin: int) -> Tensor:
        self.validate_integrity()
        if origin < FIT_ORIGINS[0] or origin > FIT_ORIGINS[-1]:
            raise ValueError("origin lies outside dense fit knots")
        right = min(
            max(bisect_right(FIT_ORIGINS, origin), 1),
            len(FIT_ORIGINS) - 1,
        )
        left = right - 1
        alpha = (
            (origin - FIT_ORIGINS[left])
            / (FIT_ORIGINS[right] - FIT_ORIGINS[left])
        )
        return (
            self.feature_kernels[left] * (1.0 - alpha)
            + self.feature_kernels[right] * alpha
        ).contiguous()

    def state_dict(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._hash_payload(),
            "feature_kernels": self.feature_kernels.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {**self._hash_payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_state_dict(
        cls, value: Mapping[str, object]
    ) -> DensePositionBilinearPlan:
        expected = {
            "artifact_kind",
            "format_version",
            "fit_knot_origins",
            "feature_kernels_sha256",
            "feature_kernels_shape",
            "response_binding_sha256",
            "interpolation",
            "stored_coefficient_count",
            "feature_kernels",
            "artifact_sha256",
        }
        _strict_keys(value, expected=expected, label="dense plan")
        if value["interpolation"] != "piecewise_linear_source_origin":
            raise ValueError("dense plan interpolation drifted")
        raw_kernels = value["feature_kernels"]
        if (
            not isinstance(raw_kernels, Tensor)
            or raw_kernels.dtype != torch.float64
            or raw_kernels.device.type != "cpu"
            or not raw_kernels.is_contiguous()
            or not bool(torch.isfinite(raw_kernels).all())
            or value["feature_kernels_sha256"]
            != _tensor_sha256(raw_kernels)
            or tuple(value["feature_kernels_shape"])
            != tuple(raw_kernels.shape)  # type: ignore[arg-type]
        ):
            raise ValueError("serialized dense kernels are not canonical")
        result = cls(
            fit_knot_origins=tuple(value["fit_knot_origins"]),  # type: ignore[arg-type]
            feature_kernels=raw_kernels,
            response_binding_sha256=value[
                "response_binding_sha256"
            ],  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
            artifact_kind=value["artifact_kind"],  # type: ignore[arg-type]
            format_version=value["format_version"],  # type: ignore[arg-type]
        )
        if (
            value["feature_kernels_sha256"]
            != _tensor_sha256(result.feature_kernels)
            or tuple(value["feature_kernels_shape"])
            != tuple(result.feature_kernels.shape)  # type: ignore[arg-type]
            or value["stored_coefficient_count"]
            != result.stored_coefficient_count
        ):
            raise ValueError("dense plan serialized tensor binding drifted")
        return result


def _feature_source_binding(
    *,
    protocol_sha256: str,
    base_candidate_artifact_sha256: str,
    hierarchy_artifact_sha256: str,
    source_model_sha256: str,
) -> str:
    return _json_sha256(
        {
            "protocol_sha256": _sha256(
                protocol_sha256, label="protocol"
            ),
            "base_candidate_artifact_sha256": _sha256(
                base_candidate_artifact_sha256,
                label="base candidate artifact",
            ),
            "hierarchy_artifact_sha256": _sha256(
                hierarchy_artifact_sha256,
                label="hierarchy artifact",
            ),
            "source_model_sha256": _sha256(
                source_model_sha256, label="source model"
            ),
            "source_coordinate_system": (
                "frozen_hierarchy_P3_modal_coordinates"
            ),
            "source_scales": "frozen_hierarchy_modal_standard_deviations",
        },
        domain=_CANDIDATE_DOMAIN,
    )


def _plan_hash(
    plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan,
) -> str:
    validate = getattr(plan, "validate_integrity", None)
    if not callable(validate):
        raise TypeError("bilinear plan lacks integrity validation")
    validate()
    return _sha256(
        getattr(plan, "artifact_sha256", None),
        label="bilinear plan",
    )


def _plan_kernel(
    plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan,
    origin: int,
) -> Tensor:
    method = getattr(plan, "linear_kernel_at_origin", None)
    if not callable(method):
        raise TypeError("bilinear plan lacks a linear kernel accessor")
    result = _canonical_tensor(
        method(origin),
        label="bilinear plan kernel",
        ndim=3,
    )
    if tuple(result.shape) != (
        len(POSITIVE_PAIRS),
        LAG_COUNT,
        TARGET_RANK,
    ):
        raise ValueError("bilinear plan kernel geometry drifted")
    return result


def _plan_state(
    plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan,
) -> Mapping[str, object]:
    method = getattr(plan, "state_dict", None)
    if not callable(method):
        raise TypeError("bilinear plan lacks state_dict")
    return _mapping(method(), label="bilinear plan state")


def _plan_metadata(
    plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan,
) -> Mapping[str, object]:
    method = getattr(plan, "metadata", None)
    if not callable(method):
        raise TypeError("bilinear plan lacks metadata")
    return _mapping(method(), label="bilinear plan metadata")


def _plan_stored_coefficients(
    plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan,
) -> int:
    value = getattr(plan, "stored_coefficient_count", None)
    if type(value) is not int or value <= 0:
        raise ValueError("bilinear plan coefficient count is invalid")
    return value


def _row_passes(
    row: Mapping[str, object],
    *,
    assessment: bool = False,
) -> bool:
    gates = GATES
    origin_gain_gate = (
        gates["minimum_assessment_error_reduction"]
        if assessment
        else gates["minimum_selection_origin_error_reduction"]
    )
    by_origin = row.get("by_origin")
    if (
        not isinstance(by_origin, Sequence)
        or isinstance(by_origin, (str, bytes))
        or not by_origin
    ):
        return False
    return bool(
        float(row.get("pooled_c11_relative_error", math.inf))
        <= gates["maximum_pooled_c11_relative_error"]
        and float(row.get("pooled_c11_cosine", -math.inf))
        >= gates["minimum_c11_cosine"]
        and float(row.get("truth_scale_defect", math.inf))
        <= gates["maximum_truth_scale_defect"]
        and float(row.get("truth_scale_cosine", -math.inf))
        >= gates["minimum_truth_scale_cosine"]
        and float(
            row.get(
                "augmented_full_mixed_relative_error",
                math.inf,
            )
        )
        <= gates["maximum_augmented_full_mixed_relative_error"]
        and float(row.get("augmented_full_mixed_cosine", -math.inf))
        >= gates["minimum_augmented_full_mixed_cosine"]
        and float(row.get("pooled_error_reduction", -math.inf))
        >= gates["minimum_selection_pooled_error_reduction"]
        and float(row.get("c11_oracle_headroom", -math.inf))
        >= gates["minimum_c11_oracle_headroom"]
        and float(row.get("oracle_recovery_fraction", -math.inf))
        >= gates["minimum_oracle_recovery_fraction"]
        and float(row.get("control_pooled_e11", math.inf))
        < gates["maximum_control_pooled_e11"]
        and float(row.get("control_worst_reliable_pair_e11", math.inf))
        < gates["maximum_reliable_control_pair_e11"]
        and row.get("control_branch_exact_zero") is True
        and row.get(
            "control_pooled_response_energy_denominator_positive"
        )
        is True
        and all(
            isinstance(origin_row, Mapping)
            and float(
                origin_row.get("c11_relative_error", math.inf)
            )
            <= gates["maximum_origin_c11_relative_error"]
            and float(
                origin_row.get("error_reduction", -math.inf)
            )
            >= origin_gain_gate
            for origin_row in by_origin
        )
    )


def _claim_boundaries(*, selected_plan_kind: str | None) -> dict[str, object]:
    return {
        "fixed_reference_modal_delta_correction_only": True,
        "base_linear_and_diagonal_executor_reused": True,
        "explicit_cross_mode_products": True,
        "positive_pair_identity_generalization_claim": False,
        "origin_interpolation_claim_only": True,
        "control_pairs_are_structural_zero_negative_controls": True,
        "prompt_conditioned_reference_provider_compiled": False,
        "full_gemma_block_replacement_authorized": False,
        "heldout_prompt_fidelity_claim": False,
        "nll_claim": False,
        "task_accuracy_claim": False,
        "model_parameter_compression_claim": False,
        "speed_or_latency_claim": False,
        "selected_plan_kind": selected_plan_kind,
        "dense_only_result_has_compaction_claim": False,
    }


_CANDIDATE_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_structural_response_tensors": False,
    "contains_compiled_bilinear_plan": True,
    "assessment_origin_opened_during_compilation": False,
    "artifact_must_remain_outside_git": True,
}


@dataclass(frozen=True, slots=True)
class Gemma3BilinearSpectralCandidate:
    """Strict compiled explicit-pair feature map plus selected plan."""

    base_candidate_file_sha256: str
    base_candidate_report_sha256: str
    base_candidate_artifact_sha256: str
    hierarchy_artifact_sha256: str
    source_model_sha256: str
    binding: Mapping[str, object]
    model: Mapping[str, object]
    protocol_sha256: str
    feature_map: ExplicitPairProductFeatureMap
    plan_kind: str
    plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan
    response_binding_sha256: str
    fit_panel_sha256: str
    selection_panel_sha256: str
    compile_evidence_artifact_sha256: str
    compile_evidence_file_sha256: str
    code_sha256s: Mapping[str, str]
    rate_curve: tuple[Mapping[str, object], ...]
    selected_rate_row: Mapping[str, object]
    accounting: Mapping[str, object]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "base_candidate_file_sha256",
            "base_candidate_report_sha256",
            "base_candidate_artifact_sha256",
            "hierarchy_artifact_sha256",
            "source_model_sha256",
            "protocol_sha256",
            "response_binding_sha256",
            "fit_panel_sha256",
            "selection_panel_sha256",
            "compile_evidence_artifact_sha256",
            "compile_evidence_file_sha256",
        ):
            _sha256(getattr(self, name), label=name)
        if self.protocol_sha256 != default_bilinear_spectral_protocol().artifact_sha256:
            raise ValueError("candidate protocol is not the frozen protocol")
        if not isinstance(self.feature_map, ExplicitPairProductFeatureMap):
            raise TypeError("candidate feature_map has the wrong type")
        self.feature_map.validate_integrity()
        if (
            self.feature_map.source_pairs != POSITIVE_PAIRS
            or self.feature_map.source_modes != MODAL_RANK
        ):
            raise ValueError("candidate feature map differs from protocol")
        expected_feature_binding = _feature_source_binding(
            protocol_sha256=self.protocol_sha256,
            base_candidate_artifact_sha256=(
                self.base_candidate_artifact_sha256
            ),
            hierarchy_artifact_sha256=self.hierarchy_artifact_sha256,
            source_model_sha256=self.source_model_sha256,
        )
        if (
            self.feature_map.source_binding_sha256
            != expected_feature_binding
        ):
            raise ValueError(
                "candidate feature map source binding differs from hierarchy"
            )
        if self.plan_kind not in ("spectral", "dense"):
            raise ValueError("candidate plan kind is invalid")
        if (
            self.plan_kind == "spectral"
            and not isinstance(self.plan, ConditionalSpectralGeneratorPlan)
        ) or (
            self.plan_kind == "dense"
            and not isinstance(self.plan, DensePositionBilinearPlan)
        ):
            raise TypeError("candidate plan type does not match plan kind")
        plan_hash = _plan_hash(self.plan)
        if (
            getattr(self.plan, "response_binding_sha256", None)
            != self.response_binding_sha256
        ):
            raise ValueError("candidate response binding differs from plan")
        if isinstance(self.plan, ConditionalSpectralGeneratorPlan):
            if (
                self.plan.fit_knot_origins != FIT_ORIGINS
                or self.plan.source_modes != len(POSITIVE_PAIRS)
                or self.plan.target_modes != TARGET_RANK
                or self.plan.lag_count != LAG_COUNT
                or self.plan.fft_length != FFT_LENGTH
                or self.plan.input_transform != "standardized_linear"
                or not torch.equal(
                    self.plan.source_scales,
                    torch.ones(
                        len(POSITIVE_PAIRS),
                        dtype=torch.float64,
                    ),
                )
            ):
                raise ValueError("candidate spectral plan ABI drifted")
        for name in (
            "binding",
            "model",
            "code_sha256s",
            "selected_rate_row",
            "accounting",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, dict(value))
        if (
            set(self.code_sha256s) != set(_code_sha256s())
            or any(
                _sha256(digest, label=f"code digest {name}") != digest
                for name, digest in self.code_sha256s.items()
            )
        ):
            raise ValueError("candidate code digest declaration is invalid")
        if not self.rate_curve:
            raise ValueError("candidate rate curve cannot be empty")
        rate_curve = tuple(dict(row) for row in self.rate_curve)
        _canonical_json_bytes(rate_curve)
        object.__setattr__(self, "rate_curve", rate_curve)
        rate_declarations = tuple(
            (
                row.get("plan_kind"),
                row.get("source_rank"),
                row.get("target_rank"),
            )
            for row in rate_curve
        )
        if (
            rate_declarations != RATE_LADDER
            or tuple(
                row.get("stored_coefficient_count")
                for row in rate_curve
            )
            != FROZEN_STORED_COEFFICIENT_COUNTS
            or any(
                row.get("passes_frozen_gate") != _row_passes(row)
                for row in rate_curve
            )
        ):
            raise ValueError("candidate rate curve protocol drifted")
        matching_rows = [
            (index, row)
            for index, row in enumerate(rate_curve)
            if row == dict(self.selected_rate_row)
        ]
        passing_rows = [
            (index, row)
            for index, row in enumerate(rate_curve)
            if bool(row.get("passes_frozen_gate"))
            and row.get("plan_kind") != "zero"
        ]
        minimal_passing = (
            min(
                passing_rows,
                key=lambda item: (
                    int(item[1]["stored_coefficient_count"]),
                    int(item[1]["source_rank"]),
                    int(item[1]["target_rank"]),
                ),
            )
            if passing_rows
            else None
        )
        actual_source_rank = self.plan.source_rank
        actual_target_rank = self.plan.target_rank
        if (
            not _row_passes(self.selected_rate_row)
            or self.selected_rate_row.get("plan_artifact_sha256")
            != plan_hash
            or self.selected_rate_row.get("plan_kind") != self.plan_kind
            or len(matching_rows) != 1
            or minimal_passing is None
            or matching_rows[0][0] != minimal_passing[0]
            or self.selected_rate_row.get("stored_coefficient_count")
            != _plan_stored_coefficients(self.plan)
            or self.selected_rate_row.get("source_rank")
            != actual_source_rank
            or self.selected_rate_row.get("target_rank")
            != actual_target_rank
        ):
            raise ValueError("selected row does not bind a passing plan")
        computed = _json_sha256(
            self._hash_payload(), domain=_CANDIDATE_DOMAIN
        )
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("bilinear candidate hash mismatch")

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "base_candidate_file_sha256": self.base_candidate_file_sha256,
            "base_candidate_report_sha256": (
                self.base_candidate_report_sha256
            ),
            "base_candidate_artifact_sha256": (
                self.base_candidate_artifact_sha256
            ),
            "hierarchy_artifact_sha256": self.hierarchy_artifact_sha256,
            "source_model_sha256": self.source_model_sha256,
            "binding": dict(self.binding),
            "model": dict(self.model),
            "protocol_sha256": self.protocol_sha256,
            "feature_map_artifact_sha256": (
                self.feature_map.artifact_sha256
            ),
            "plan_kind": self.plan_kind,
            "plan_artifact_sha256": _plan_hash(self.plan),
            "response_binding_sha256": self.response_binding_sha256,
            "fit_panel_sha256": self.fit_panel_sha256,
            "selection_panel_sha256": self.selection_panel_sha256,
            "compile_evidence_artifact_sha256": (
                self.compile_evidence_artifact_sha256
            ),
            "compile_evidence_file_sha256": (
                self.compile_evidence_file_sha256
            ),
            "code_sha256s": dict(self.code_sha256s),
            "fit_origins": FIT_ORIGINS,
            "selection_origins": SELECTION_ORIGINS,
            "assessment_origins_used": (),
            "rate_curve": self.rate_curve,
            "selected_rate_row": dict(self.selected_rate_row),
            "accounting": dict(self.accounting),
            "claim_boundaries": _claim_boundaries(
                selected_plan_kind=self.plan_kind
            ),
            "safety": _CANDIDATE_SAFETY,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "feature_map": self.feature_map.state_dict(),
            "plan": dict(_plan_state(self.plan)),
            "artifact_sha256": self.artifact_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "feature_map": self.feature_map.metadata(),
            "plan": dict(_plan_metadata(self.plan)),
            "artifact_sha256": self.artifact_sha256,
        }

    def prepare(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PreparedGemma3BilinearSpectralBranch:
        return PreparedGemma3BilinearSpectralBranch(
            self,
            device=device,
            dtype=dtype,
        )


class PreparedGemma3BilinearSpectralBranch(nn.Module):
    """Raw-modal executable: explicit products then causal spectral branch."""

    def __init__(
        self,
        candidate: Gemma3BilinearSpectralCandidate,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not isinstance(candidate, Gemma3BilinearSpectralCandidate):
            raise TypeError("candidate has the wrong type")
        if dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("dtype must be a supported floating dtype")
        runtime_device = torch.device(device)
        candidate.feature_map.validate_integrity()
        _plan_hash(candidate.plan)
        self.candidate_sha256 = candidate.artifact_sha256
        self.feature_map_sha256 = candidate.feature_map.artifact_sha256
        self.plan_sha256 = _plan_hash(candidate.plan)
        self.plan_kind = candidate.plan_kind
        self.feature_runtime = candidate.feature_map.prepare(
            device=runtime_device,
            dtype=dtype,
        )
        if isinstance(candidate.plan, ConditionalSpectralGeneratorPlan):
            self.spectral_runtime: nn.Module | None = candidate.plan.prepare(
                device=runtime_device,
                dtype=dtype,
            )
            self.register_buffer(
                "dense_knot_kernels",
                torch.empty(
                    0,
                    device=runtime_device,
                    dtype=dtype,
                ),
            )
        else:
            self.spectral_runtime = None
            self.register_buffer(
                "dense_knot_kernels",
                candidate.plan.feature_kernels.to(
                    device=runtime_device,
                    dtype=dtype,
                ).contiguous().clone(),
            )
        self.fit_knot_origins = FIT_ORIGINS
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return self.feature_runtime.device

    @property
    def dtype(self) -> torch.dtype:
        return self.feature_runtime.dtype

    def _validate_inputs(
        self,
        source_modes: Tensor,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor,
    ) -> None:
        if (
            not isinstance(source_modes, Tensor)
            or source_modes.ndim != 3
            or source_modes.shape[-1] != MODAL_RANK
            or source_modes.device != self.device
            or source_modes.dtype != self.dtype
            or not bool(torch.isfinite(source_modes).all())
            or not isinstance(logical_positions, Tensor)
            or logical_positions.shape != source_modes.shape[:2]
            or logical_positions.device != self.device
            or logical_positions.dtype not in (torch.int32, torch.int64)
            or not isinstance(valid_mask, Tensor)
            or valid_mask.shape != logical_positions.shape
            or valid_mask.device != self.device
            or valid_mask.dtype != torch.bool
            or not isinstance(source_mask, Tensor)
            or source_mask.shape != logical_positions.shape
            or source_mask.device != self.device
            or source_mask.dtype != torch.bool
            or bool((source_mask & ~valid_mask).any())
        ):
            raise ValueError(
                "prepared bilinear inputs differ from the frozen runtime ABI"
            )

    def _dense_kernel(self, origin: int) -> Tensor:
        if origin < FIT_ORIGINS[0] or origin > FIT_ORIGINS[-1]:
            raise ValueError("source origin lies outside dense fit knots")
        right = min(
            max(bisect_right(FIT_ORIGINS, origin), 1),
            len(FIT_ORIGINS) - 1,
        )
        left = right - 1
        alpha = (
            (origin - FIT_ORIGINS[left])
            / (FIT_ORIGINS[right] - FIT_ORIGINS[left])
        )
        return (
            self.dense_knot_kernels[left] * (1.0 - alpha)
            + self.dense_knot_kernels[right] * alpha
        )

    def _forward_dense(
        self,
        features: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        result = torch.zeros(
            (*features.shape[:2], TARGET_RANK),
            device=self.device,
            dtype=self.dtype,
        )
        batch, sequence_length = logical_positions.shape
        for batch_ordinal in range(batch):
            for source_ordinal in range(sequence_length):
                if not bool(source_mask[batch_ordinal, source_ordinal]):
                    continue
                origin = int(
                    logical_positions[batch_ordinal, source_ordinal]
                )
                kernel = self._dense_kernel(origin)
                for target_ordinal in range(sequence_length):
                    if not bool(
                        valid_mask[batch_ordinal, target_ordinal]
                    ):
                        continue
                    lag = int(
                        logical_positions[batch_ordinal, target_ordinal]
                    ) - origin
                    if 0 <= lag < LAG_COUNT:
                        result[batch_ordinal, target_ordinal] += (
                            features[batch_ordinal, source_ordinal]
                            @ kernel[:, lag]
                        )
        return result

    def forward(
        self,
        source_modes: Tensor,
        *,
        logical_positions: Tensor,
        valid_mask: Tensor,
        source_mask: Tensor,
    ) -> Tensor:
        self._validate_inputs(
            source_modes,
            logical_positions,
            valid_mask,
            source_mask,
        )
        features = self.feature_runtime(source_modes)
        if self.spectral_runtime is not None:
            result = self.spectral_runtime(
                features,
                logical_positions=logical_positions,
                valid_mask=valid_mask,
                source_mask=source_mask,
            )
            if (
                not isinstance(result, Tensor)
                or result.shape
                != (*source_modes.shape[:2], TARGET_RANK)
            ):
                raise ValueError("nested spectral runtime output ABI drifted")
            return result
        return self._forward_dense(
            features,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )


_CANDIDATE_STATE_KEYS = {
    "schema",
    "format_version",
    "base_candidate_file_sha256",
    "base_candidate_report_sha256",
    "base_candidate_artifact_sha256",
    "hierarchy_artifact_sha256",
    "source_model_sha256",
    "binding",
    "model",
    "protocol_sha256",
    "feature_map_artifact_sha256",
    "plan_kind",
    "plan_artifact_sha256",
    "response_binding_sha256",
    "fit_panel_sha256",
    "selection_panel_sha256",
    "compile_evidence_artifact_sha256",
    "compile_evidence_file_sha256",
    "code_sha256s",
    "fit_origins",
    "selection_origins",
    "assessment_origins_used",
    "rate_curve",
    "selected_rate_row",
    "accounting",
    "claim_boundaries",
    "safety",
    "feature_map",
    "plan",
    "artifact_sha256",
}


def _candidate_from_state(
    value: Mapping[str, object],
) -> Gemma3BilinearSpectralCandidate:
    _strict_keys(value, expected=_CANDIDATE_STATE_KEYS, label="candidate")
    if (
        value["schema"] != _SCHEMA
        or value["format_version"] != _FORMAT_VERSION
        or tuple(value["fit_origins"]) != FIT_ORIGINS  # type: ignore[arg-type]
        or tuple(value["selection_origins"])  # type: ignore[arg-type]
        != SELECTION_ORIGINS
        or tuple(value["assessment_origins_used"]) != ()  # type: ignore[arg-type]
        or dict(_mapping(value["safety"], label="candidate safety"))
        != _CANDIDATE_SAFETY
    ):
        raise ValueError("bilinear candidate protocol drifted")
    feature_map = ExplicitPairProductFeatureMap.from_state_dict(
        _mapping(value["feature_map"], label="feature map")
    )
    if (
        feature_map.artifact_sha256
        != value["feature_map_artifact_sha256"]
    ):
        raise ValueError("candidate feature-map hash binding drifted")
    plan_kind = value["plan_kind"]
    plan_state = _mapping(value["plan"], label="candidate plan")
    if plan_kind == "spectral":
        plan: ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan
        plan = ConditionalSpectralGeneratorPlan.from_state_dict(plan_state)
    elif plan_kind == "dense":
        plan = DensePositionBilinearPlan.from_state_dict(plan_state)
    else:
        raise ValueError("candidate plan kind is invalid")
    if _plan_hash(plan) != value["plan_artifact_sha256"]:
        raise ValueError("candidate plan hash binding drifted")
    rate_curve = value["rate_curve"]
    if (
        isinstance(rate_curve, (str, bytes))
        or not isinstance(rate_curve, Sequence)
    ):
        raise TypeError("candidate rate curve must be a sequence")
    result = Gemma3BilinearSpectralCandidate(
        base_candidate_file_sha256=value[
            "base_candidate_file_sha256"
        ],  # type: ignore[arg-type]
        base_candidate_report_sha256=value[
            "base_candidate_report_sha256"
        ],  # type: ignore[arg-type]
        base_candidate_artifact_sha256=value[
            "base_candidate_artifact_sha256"
        ],  # type: ignore[arg-type]
        hierarchy_artifact_sha256=value[
            "hierarchy_artifact_sha256"
        ],  # type: ignore[arg-type]
        source_model_sha256=value[
            "source_model_sha256"
        ],  # type: ignore[arg-type]
        binding=_mapping(value["binding"], label="candidate binding"),
        model=_mapping(value["model"], label="candidate model"),
        protocol_sha256=value["protocol_sha256"],  # type: ignore[arg-type]
        feature_map=feature_map,
        plan_kind=plan_kind,  # type: ignore[arg-type]
        plan=plan,
        response_binding_sha256=value[
            "response_binding_sha256"
        ],  # type: ignore[arg-type]
        fit_panel_sha256=value["fit_panel_sha256"],  # type: ignore[arg-type]
        selection_panel_sha256=value[
            "selection_panel_sha256"
        ],  # type: ignore[arg-type]
        compile_evidence_artifact_sha256=value[
            "compile_evidence_artifact_sha256"
        ],  # type: ignore[arg-type]
        compile_evidence_file_sha256=value[
            "compile_evidence_file_sha256"
        ],  # type: ignore[arg-type]
        code_sha256s=_mapping(
            value["code_sha256s"], label="candidate code digests"
        ),  # type: ignore[arg-type]
        rate_curve=tuple(
            _mapping(row, label="rate row") for row in rate_curve
        ),
        selected_rate_row=_mapping(
            value["selected_rate_row"],
            label="selected rate row",
        ),
        accounting=_mapping(
            value["accounting"], label="candidate accounting"
        ),
        artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
    )
    if (
        dict(
            _mapping(
                value["claim_boundaries"],
                label="claim boundaries",
            )
        )
        != _claim_boundaries(selected_plan_kind=result.plan_kind)
    ):
        raise ValueError("candidate claim boundaries drifted")
    return result


def load_gemma3_bilinear_spectral_candidate(
    path: Path | str,
    *,
    expected_file_sha256: str,
    expected_report_sha256: str,
) -> Gemma3BilinearSpectralCandidate:
    """Authenticate the complete candidate and its source-safe report."""

    source = Path(path)
    expected_file = _sha256(
        expected_file_sha256, label="expected candidate file"
    )
    actual_file = _file_sha256(source)
    if actual_file != expected_file:
        raise ValueError("candidate tensor differs from expected SHA-256")
    raw = torch.load(source, map_location="cpu", weights_only=True)
    candidate = _candidate_from_state(
        _mapping(raw, label="candidate tensor")
    )
    evidence_path = source.with_name(
        f"{source.stem}-compile-evidence.pt"
    )
    if (
        not evidence_path.is_file()
        or _file_sha256(evidence_path)
        != candidate.compile_evidence_file_sha256
    ):
        raise ValueError("candidate compile-evidence file binding mismatch")
    evidence_raw = torch.load(
        evidence_path, map_location="cpu", weights_only=True
    )
    evidence = _mapping(evidence_raw, label="compile evidence")
    if (
        evidence.get("artifact_sha256")
        != candidate.compile_evidence_artifact_sha256
        or evidence.get("response_binding_sha256")
        != candidate.response_binding_sha256
    ):
        raise ValueError("candidate compile-evidence artifact drifted")
    expected_report = _sha256(
        expected_report_sha256, label="expected candidate report"
    )
    with source.with_suffix(".json").open("r", encoding="utf-8") as handle:
        report_raw = json.load(handle)
    report = _mapping(report_raw, label="candidate report")
    claimed = _sha256(report.get("report_sha256"), label="candidate report")
    payload = dict(report)
    payload.pop("report_sha256")
    artifact = _mapping(report.get("artifact"), label="report artifact")
    metadata = _mapping(report.get("candidate"), label="report candidate")
    if (
        claimed != expected_report
        or _json_sha256(payload, domain=_REPORT_DOMAIN) != claimed
        or artifact.get("tensor_file_sha256") != actual_file
        or metadata.get("artifact_sha256") != candidate.artifact_sha256
    ):
        raise ValueError("candidate report binding mismatch")
    return candidate


def _publish_compilation(
    result: BilinearCompilationResult,
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    evidence_path = output.with_name(
        f"{output.stem}-compile-evidence.pt"
    )
    destinations = (
        (output, evidence_path, report_path)
        if result.candidate is not None
        else (evidence_path, report_path)
    )
    reservation = _reserve_outputs(destinations)
    stages: list[Path] = []
    try:
        evidence_stage = _stage_path(evidence_path)
        stages.append(evidence_stage)
        evidence_stage.write_bytes(result.evidence_bytes)
        if (
            _file_sha256(evidence_stage) != result.evidence_file_sha256
        ):
            raise RuntimeError("staged evidence bytes changed")
        tensor_stage: Path | None = None
        if result.candidate is not None:
            tensor_stage = _stage_torch(
                result.candidate.state_dict(), output
            )
            stages.append(tensor_stage)
        report: dict[str, object] = {
            "schema": (
                _SCHEMA if result.candidate is not None else _FAILURE_SCHEMA
            ),
            "format_version": _FORMAT_VERSION,
            "selection_passed": result.selection_passed,
            "candidate": (
                result.candidate.metadata()
                if result.candidate is not None
                else None
            ),
            "rate_curve": result.rate_curve,
            "prepared_runtime_crosscheck": (
                dict(result.prepared_runtime_crosscheck)
                if result.prepared_runtime_crosscheck is not None
                else None
            ),
            "live_measurement": (
                dict(result.orchestration)
                if result.orchestration is not None
                else None
            ),
            "compile_evidence": {
                "tensor_file": str(evidence_path),
                "tensor_file_sha256": result.evidence_file_sha256,
                "artifact_sha256": result.evidence_artifact_sha256,
                "fit_panel_sha256": result.fit_panel.artifact_sha256,
                "selection_panel_sha256": (
                    result.selection_panel.artifact_sha256
                ),
                "committable": False,
            },
            "artifact": {
                "tensor_file": (
                    str(output) if result.candidate is not None else None
                ),
                "tensor_file_sha256": (
                    _file_sha256(tensor_stage)
                    if tensor_stage is not None
                    else None
                ),
                "report_file": str(report_path),
                "committable": False,
            },
            "claim_boundaries": _claim_boundaries(
                selected_plan_kind=(
                    result.candidate.plan_kind
                    if result.candidate is not None
                    else None
                )
            ),
            "safety": {
                **_CANDIDATE_SAFETY,
                "candidate_tensor_published": (
                    result.candidate is not None
                ),
                "selection_failure_did_not_open_assessment_origin": True,
            },
        }
        payload = dict(report)
        report["report_sha256"] = _json_sha256(
            payload, domain=_REPORT_DOMAIN
        )
        _canonical_json_bytes(report)
        report_stage = _stage_json(report, report_path)
        stages.append(report_stage)
        ordered_stages = (
            (tensor_stage, evidence_stage, report_stage)
            if tensor_stage is not None
            else (evidence_stage, report_stage)
        )
        reservation.publish(ordered_stages)  # type: ignore[arg-type]
        return report
    finally:
        reservation.release()
        for stage in stages:
            stage.unlink(missing_ok=True)


def _c11(values: Tensor) -> Tensor:
    if values.shape[-3] != len(SIGN_ROWS):
        raise ValueError("C11 input sign axis differs from protocol")
    return (
        values[..., 0, :, :]
        - values[..., 1, :, :]
        - values[..., 2, :, :]
        + values[..., 3, :, :]
    ) * 0.25


def _signed_c11(value: Tensor) -> Tensor:
    signs = torch.tensor(
        (1.0, -1.0, -1.0, 1.0),
        dtype=torch.float64,
    )
    return value.unsqueeze(-3) * signs.view(
        *((1,) * (value.ndim - 2)),
        4,
        1,
        1,
    )


def _panel_noise_floor(panel: MeasuredBilinearPanel) -> float:
    repeat = panel.repeat_sentinel_first - panel.repeat_sentinel_second
    zero = panel.zero_sentinel
    return max(
        float(torch.linalg.vector_norm(repeat)),
        float(torch.linalg.vector_norm(zero)),
        torch.finfo(torch.float64).eps
        * math.sqrt(float(panel.zero_sentinel.numel())),
    )


def _branch_predictions(
    panel: MeasuredBilinearPanel,
    plan: (
        ConditionalSpectralGeneratorPlan
        | DensePositionBilinearPlan
        | None
    ),
) -> Tensor:
    result = torch.zeros_like(panel.truth)
    if plan is None:
        return result
    pair_to_feature = {
        pair: ordinal for ordinal, pair in enumerate(POSITIVE_PAIRS)
    }
    for origin_ordinal, origin in enumerate(panel.origins):
        kernel = _plan_kernel(plan, origin)
        for pair_ordinal, pair in enumerate(panel.pairs):
            feature_ordinal = pair_to_feature.get(pair)
            if feature_ordinal is None:
                continue
            for radius_ordinal, radius in enumerate(panel.radii):
                for sign_ordinal, (
                    _label,
                    left_sign,
                    right_sign,
                ) in enumerate(SIGN_ROWS):
                    result[
                        origin_ordinal,
                        pair_ordinal,
                        radius_ordinal,
                        sign_ordinal,
                    ] = (
                        left_sign
                        * right_sign
                        * radius**2
                        * kernel[feature_ordinal]
                    )
    return result


def _control_metrics(
    panel: MeasuredBilinearPanel,
    *,
    truth_c11: Tensor,
    base_c11: Tensor,
    branch: Tensor,
) -> dict[str, object]:
    controls = truth_c11[:, panel.positive_pair_count :, 1]
    base_controls = base_c11[:, panel.positive_pair_count :, 1]
    control_truth = panel.truth[:, panel.positive_pair_count :, 1]
    if controls.shape[1] == 0:
        return {
            "control_pooled_e11": 0.0,
            "control_worst_reliable_pair_e11": 0.0,
            "control_reliable_pair_count": 0,
            "control_by_pair": (),
            "control_branch_exact_zero": True,
        }
    pooled_numerator = 4.0 * float(controls.square().sum())
    pooled_denominator = float(control_truth.square().sum())
    metric_epsilon = torch.finfo(torch.float64).eps
    pooled = math.sqrt(
        pooled_numerator / max(pooled_denominator, metric_epsilon)
    )
    all_pair_truth = panel.truth[:, :, 1]
    pair_rms = all_pair_truth.square().mean(
        dim=(0, 2, 3, 4)
    ).sqrt()
    panel_median = float(pair_rms.median())
    noise = _panel_noise_floor(panel)
    rows: list[dict[str, object]] = []
    reliable_e11: list[float] = []
    for control_ordinal, pair in enumerate(panel.control_pairs):
        values = controls[:, control_ordinal]
        responses = control_truth[:, control_ordinal]
        numerator = 4.0 * float(values.square().sum())
        denominator = float(responses.square().sum())
        e11 = math.sqrt(numerator / max(denominator, metric_epsilon))
        response_rms = float(responses.square().mean().sqrt())
        c11_l2 = float(torch.linalg.vector_norm(values))
        rms_fraction = GATES[
            "control_reliability_minimum_panel_median_rms_fraction"
        ]
        noise_multiple = GATES[
            "control_reliability_minimum_two_c11_noise_multiple"
        ]
        reliable = bool(
            response_rms >= rms_fraction * panel_median
            and 2.0 * c11_l2 >= noise_multiple * noise
        )
        if reliable:
            reliable_e11.append(e11)
        rows.append(
            {
                "pair": pair,
                "e11": e11,
                "response_rms": response_rms,
                "c11_l2": c11_l2,
                "reliable": reliable,
                "response_energy_denominator_positive": denominator > 0.0,
            }
        )
    control_branch = branch[:, panel.positive_pair_count :]
    return {
        "control_pooled_e11": pooled,
        "control_worst_reliable_pair_e11": (
            max(reliable_e11) if reliable_e11 else 0.0
        ),
        "control_reliable_pair_count": len(reliable_e11),
        "control_by_pair": tuple(rows),
        "control_pooled_response_energy_denominator_positive": (
            pooled_denominator > 0.0
        ),
        "control_branch_exact_zero": torch.equal(
            control_branch, torch.zeros_like(control_branch)
        ),
        "control_base_c11_l2": float(
            torch.linalg.vector_norm(base_controls)
        ),
        "control_base_c11_max_abs": float(base_controls.abs().max()),
        "control_base_c11_numerically_zero": bool(
            float(torch.linalg.vector_norm(base_controls)) <= noise
        ),
        "control_reliability_response_rms_floor": (
            GATES[
                "control_reliability_minimum_panel_median_rms_fraction"
            ]
            * panel_median
        ),
        "control_reliability_two_c11_noise_multiple": GATES[
            "control_reliability_minimum_two_c11_noise_multiple"
        ],
    }


def evaluate_bilinear_plan(
    panel: MeasuredBilinearPanel,
    plan: (
        ConditionalSpectralGeneratorPlan
        | DensePositionBilinearPlan
        | None
    ),
    *,
    assessment: bool = False,
    branch_prediction: Tensor | None = None,
) -> dict[str, object]:
    """Score one already-frozen plan; this function performs no fitting."""

    if not isinstance(panel, MeasuredBilinearPanel):
        raise TypeError("panel must be a MeasuredBilinearPanel")
    expected_split = "assessment" if assessment else "selection"
    if panel.split != expected_split:
        raise ValueError(f"evaluation requires the {expected_split} panel")
    branch = (
        _branch_predictions(panel, plan)
        if branch_prediction is None
        else _canonical_tensor(
            branch_prediction,
            label="prepared branch prediction",
            ndim=6,
        )
    )
    if branch.shape != panel.truth.shape:
        raise ValueError("prepared branch prediction geometry drifted")
    truth_c11 = _c11(panel.truth)
    base_c11 = _c11(panel.base_prediction)
    residual_c11 = truth_c11 - base_c11
    branch_c11 = _c11(branch)
    positive = slice(0, panel.positive_pair_count)
    target_operating = residual_c11[:, positive, 1]
    prediction_operating = branch_c11[:, positive, 1]
    scale_low = 4.0 * residual_c11[:, positive, 0]
    scale_high = residual_c11[:, positive, 1]
    truth_operating = panel.truth[:, positive, 1]
    base_operating = panel.base_prediction[:, positive, 1]
    branch_operating = branch[:, positive, 1]
    augmented = base_operating + branch_operating
    oracle = base_operating + _signed_c11(target_operating)
    base_residual_norm = float(
        torch.linalg.vector_norm(base_operating - truth_operating)
    )
    augmented_residual_norm = float(
        torch.linalg.vector_norm(augmented - truth_operating)
    )
    oracle_residual_norm = float(
        torch.linalg.vector_norm(oracle - truth_operating)
    )
    pooled_gain = (
        1.0 - augmented_residual_norm / base_residual_norm
        if base_residual_norm > 0.0
        else 0.0
    )
    oracle_headroom = (
        1.0 - oracle_residual_norm / base_residual_norm
        if base_residual_norm > 0.0
        else 0.0
    )
    oracle_denominator = base_residual_norm - oracle_residual_norm
    oracle_recovery = (
        (base_residual_norm - augmented_residual_norm)
        / oracle_denominator
        if oracle_denominator > 0.0
        else -1.0
    )
    by_origin: list[dict[str, object]] = []
    for ordinal, origin in enumerate(panel.origins):
        origin_truth = truth_operating[ordinal]
        origin_base = base_operating[ordinal]
        origin_augmented = augmented[ordinal]
        by_origin.append(
            {
                "origin": origin,
                "c11_relative_error": _relative_error(
                    prediction_operating[ordinal],
                    target_operating[ordinal],
                ),
                "c11_cosine": _cosine(
                    prediction_operating[ordinal],
                    target_operating[ordinal],
                ),
                "base_full_mixed_relative_error": _relative_error(
                    origin_base, origin_truth
                ),
                "augmented_full_mixed_relative_error": _relative_error(
                    origin_augmented, origin_truth
                ),
                "augmented_full_mixed_cosine": _cosine(
                    origin_augmented, origin_truth
                ),
                "error_reduction": _error_reduction(
                    origin_base,
                    origin_augmented,
                    origin_truth,
                ),
            }
        )
    control = _control_metrics(
        panel,
        truth_c11=truth_c11,
        base_c11=base_c11,
        branch=branch,
    )
    result: dict[str, object] = {
        "pooled_c11_relative_error": _relative_error(
            prediction_operating, target_operating
        ),
        "pooled_c11_cosine": _cosine(
            prediction_operating, target_operating
        ),
        "truth_scale_defect": _relative_error(scale_low, scale_high),
        "truth_scale_cosine": _cosine(scale_low, scale_high),
        "base_full_mixed_relative_error": _relative_error(
            base_operating, truth_operating
        ),
        "augmented_full_mixed_relative_error": _relative_error(
            augmented, truth_operating
        ),
        "augmented_full_mixed_cosine": _cosine(
            augmented, truth_operating
        ),
        "pooled_error_reduction": pooled_gain,
        "c11_oracle_headroom": oracle_headroom,
        "oracle_recovery_fraction": oracle_recovery,
        "oracle_recovery_denominator_positive": (
            oracle_denominator > 0.0
        ),
        "by_origin": tuple(by_origin),
        **control,
    }
    result["passes_frozen_gate"] = _row_passes(
        result, assessment=assessment
    )
    return result


def _fit_kernel_from_panel(panel: MeasuredBilinearPanel) -> Tensor:
    if (
        panel.split != "fit"
        or panel.origins != FIT_ORIGINS
        or panel.pairs != POSITIVE_PAIRS
        or panel.positive_pair_count != len(POSITIVE_PAIRS)
    ):
        raise ValueError("fit panel differs from frozen compile split")
    residual_c11 = _c11(panel.truth) - _c11(panel.base_prediction)
    radii = torch.tensor(panel.radii, dtype=torch.float64)
    numerator = (
        residual_c11
        * radii.square().view(1, 1, -1, 1, 1)
    ).sum(dim=2)
    denominator = float(radii.pow(4).sum())
    # Conditional plans require [feature, origin, lag, target].
    return (numerator / denominator).permute(1, 0, 2, 3).contiguous()


def _response_binding(
    *,
    protocol: FrozenBilinearSpectralProtocol,
    feature_map: ExplicitPairProductFeatureMap,
    base_candidate: Gemma3ConditionalSpectralCandidate,
    fit_panel: MeasuredBilinearPanel,
) -> str:
    return _json_sha256(
        {
            "protocol_sha256": protocol.artifact_sha256,
            "feature_map_sha256": feature_map.artifact_sha256,
            "base_candidate_artifact_sha256": (
                base_candidate.artifact_sha256
            ),
            "fit_panel_sha256": fit_panel.artifact_sha256,
            "fit_origins": FIT_ORIGINS,
            "source_features": len(POSITIVE_PAIRS),
            "lag_count": LAG_COUNT,
            "target_modes": TARGET_RANK,
            "kernel_semantics": (
                "C11_residual_per_unit_explicit_pair_product"
            ),
        },
        domain=_CANDIDATE_DOMAIN,
    )


def _fit_plan_ladder(
    kernels: Tensor,
    *,
    response_binding_sha256: str,
    fit_function: Callable[..., ConditionalSpectralGeneratorPlan] | None,
) -> tuple[
    tuple[
        ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan | None,
        ...
    ],
    tuple[dict[str, object], ...],
]:
    """Fit every nonzero row.  Selection data is not accepted here."""

    if fit_function is None:
        from .conditional_spectral_generator import (
            fit_conditional_spectral_generator,
        )

        fit = fit_conditional_spectral_generator
    else:
        fit = fit_function
    plans: list[
        ConditionalSpectralGeneratorPlan | DensePositionBilinearPlan | None
    ] = [None]
    declarations: list[dict[str, object]] = [
        {
            "plan_kind": "zero",
            "source_rank": 0,
            "target_rank": 0,
            "stored_coefficient_count": 0,
            "plan_artifact_sha256": None,
            "dense_only_no_compaction": False,
        }
    ]
    unit_scales = torch.ones(len(POSITIVE_PAIRS), dtype=torch.float64)
    for source_rank, target_rank in SPECTRAL_RANK_LADDER:
        plan = fit(
            responses=kernels,
            source_scales=unit_scales,
            origins=FIT_ORIGINS,
            fit_origins=FIT_ORIGINS,
            source_rank=source_rank,
            target_rank=target_rank,
            response_binding_sha256=response_binding_sha256,
            input_transform="standardized_linear",
            fft_length=FFT_LENGTH,
        )
        if not isinstance(plan, ConditionalSpectralGeneratorPlan):
            raise TypeError("fit function returned the wrong plan type")
        plans.append(plan)
        declarations.append(
            {
                "plan_kind": "spectral",
                "source_rank": source_rank,
                "target_rank": target_rank,
                "stored_coefficient_count": (
                    plan.stored_coefficient_count
                ),
                "plan_artifact_sha256": plan.artifact_sha256,
                "dense_only_no_compaction": False,
            }
        )
    dense = DensePositionBilinearPlan(
        fit_knot_origins=FIT_ORIGINS,
        feature_kernels=kernels.permute(1, 0, 2, 3),
        response_binding_sha256=response_binding_sha256,
    )
    plans.append(dense)
    declarations.append(
        {
            "plan_kind": "dense",
            "source_rank": len(POSITIVE_PAIRS),
            "target_rank": TARGET_RANK,
            "stored_coefficient_count": dense.stored_coefficient_count,
            "plan_artifact_sha256": dense.artifact_sha256,
            "dense_only_no_compaction": True,
        }
    )
    if tuple(
        (
            row["plan_kind"],
            row["source_rank"],
            row["target_rank"],
        )
        for row in declarations
    ) != RATE_LADDER:
        raise RuntimeError("fitted rate ladder order drifted")
    if tuple(
        row["stored_coefficient_count"] for row in declarations
    ) != FROZEN_STORED_COEFFICIENT_COUNTS:
        raise RuntimeError("fitted rate ladder storage accounting drifted")
    return tuple(plans), tuple(declarations)


@dataclass(frozen=True, slots=True)
class BilinearCompilationResult:
    candidate: Gemma3BilinearSpectralCandidate | None
    fit_panel: MeasuredBilinearPanel
    selection_panel: MeasuredBilinearPanel
    rate_curve: tuple[Mapping[str, object], ...]
    response_binding_sha256: str
    selection_passed: bool
    evidence_artifact_sha256: str
    evidence_file_sha256: str
    evidence_bytes: bytes
    prepared_runtime_crosscheck: Mapping[str, object] | None
    orchestration: Mapping[str, object] | None = None


def _compile_evidence(
    *,
    fit_panel: MeasuredBilinearPanel,
    selection_panel: MeasuredBilinearPanel,
    response_binding_sha256: str,
    rate_curve: Sequence[Mapping[str, object]],
    feature_map: ExplicitPairProductFeatureMap,
    code_sha256s: Mapping[str, str],
) -> tuple[str, str, bytes]:
    common = {
        "schema": f"{_SCHEMA}.compile_evidence",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": fit_panel.protocol_sha256,
        "base_candidate_artifact_sha256": (
            fit_panel.base_candidate_sha256
        ),
        "feature_map_artifact_sha256": feature_map.artifact_sha256,
        "response_binding_sha256": response_binding_sha256,
        "fit_panel": fit_panel.metadata(),
        "selection_panel": selection_panel.metadata(),
        "rate_curve": tuple(dict(row) for row in rate_curve),
        "code_sha256s": dict(code_sha256s),
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_structural_response_tensors": True,
            "committable": False,
        },
    }
    artifact_sha256 = _json_sha256(common, domain=_ARTIFACT_DOMAIN)
    state = {
        **common,
        "fit_panel_state": fit_panel.state_dict(),
        "selection_panel_state": selection_panel.state_dict(),
        "artifact_sha256": artifact_sha256,
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    payload = buffer.getvalue()
    file_sha256 = hashlib.sha256(payload).hexdigest()
    return artifact_sha256, file_sha256, payload


def compile_bilinear_spectral_candidate(
    *,
    fit_panel_factory: Callable[[], MeasuredBilinearPanel],
    selection_panel_factory: Callable[[], MeasuredBilinearPanel],
    base_candidate: Gemma3ConditionalSpectralCandidate,
    feature_map: ExplicitPairProductFeatureMap,
    base_candidate_file_sha256: str,
    base_candidate_report_sha256: str,
    hierarchy_artifact_sha256: str,
    source_model_sha256: str,
    binding: Mapping[str, object],
    model: Mapping[str, object],
    protocol: FrozenBilinearSpectralProtocol | None = None,
    code_sha256s: Mapping[str, str] | None = None,
    prepared_runtime_dtype: torch.dtype = torch.float32,
    prepared_runtime_device: torch.device | str = "cpu",
    fit_function: Callable[..., ConditionalSpectralGeneratorPlan]
    | None = None,
) -> BilinearCompilationResult:
    """Fit the complete ladder before invoking ``selection_panel_factory``."""

    frozen = (
        default_bilinear_spectral_protocol()
        if protocol is None
        else protocol
    )
    frozen_code_sha256s = (
        _code_sha256s() if code_sha256s is None else dict(code_sha256s)
    )
    if set(frozen_code_sha256s) != set(_code_sha256s()):
        raise ValueError("compile code digest keys differ from protocol")
    for name, digest in frozen_code_sha256s.items():
        _sha256(digest, label=f"compile code digest {name}")
    if (
        not isinstance(frozen, FrozenBilinearSpectralProtocol)
        or frozen.artifact_sha256
        != default_bilinear_spectral_protocol().artifact_sha256
    ):
        raise ValueError("compile protocol is not the frozen declaration")
    if not callable(fit_panel_factory) or not callable(
        selection_panel_factory
    ):
        raise TypeError("panel factories must be callable")
    if not isinstance(base_candidate, Gemma3ConditionalSpectralCandidate):
        raise TypeError("base candidate has the wrong type")
    if not isinstance(feature_map, ExplicitPairProductFeatureMap):
        raise TypeError("feature map has the wrong type")
    feature_map.validate_integrity()
    if feature_map.source_pairs != POSITIVE_PAIRS:
        raise ValueError("feature map pairs differ from frozen protocol")
    expected_feature_binding = _feature_source_binding(
        protocol_sha256=frozen.artifact_sha256,
        base_candidate_artifact_sha256=base_candidate.artifact_sha256,
        hierarchy_artifact_sha256=hierarchy_artifact_sha256,
        source_model_sha256=source_model_sha256,
    )
    if feature_map.source_binding_sha256 != expected_feature_binding:
        raise ValueError("feature map source binding differs from compile ABI")

    # FIREWALL: there is no selection object until every plan exists.
    fit_panel = fit_panel_factory()
    if (
        not isinstance(fit_panel, MeasuredBilinearPanel)
        or fit_panel.protocol_sha256 != frozen.artifact_sha256
        or fit_panel.base_candidate_sha256
        != base_candidate.artifact_sha256
    ):
        raise ValueError("fit panel binding differs from frozen compile ABI")
    kernels = _fit_kernel_from_panel(fit_panel)
    response_binding = _response_binding(
        protocol=frozen,
        feature_map=feature_map,
        base_candidate=base_candidate,
        fit_panel=fit_panel,
    )
    plans, declarations = _fit_plan_ladder(
        kernels,
        response_binding_sha256=response_binding,
        fit_function=fit_function,
    )

    # Only this line is allowed to open the held-out selection responses.
    selection_panel = selection_panel_factory()
    if (
        not isinstance(selection_panel, MeasuredBilinearPanel)
        or selection_panel.split != "selection"
        or selection_panel.origins != SELECTION_ORIGINS
        or selection_panel.pairs
        != POSITIVE_PAIRS + SELECTION_CONTROL_PAIRS
        or selection_panel.protocol_sha256 != frozen.artifact_sha256
        or selection_panel.base_candidate_sha256
        != base_candidate.artifact_sha256
    ):
        raise ValueError("selection panel differs from frozen compile ABI")
    rows: list[dict[str, object]] = []
    for plan, declaration in zip(plans, declarations, strict=True):
        metrics = evaluate_bilinear_plan(selection_panel, plan)
        row = {**declaration, **metrics}
        row["passes_frozen_gate"] = _row_passes(row)
        rows.append(row)
    passing = [
        (index, plan, row)
        for index, (plan, row) in enumerate(
            zip(plans, rows, strict=True)
        )
        if plan is not None and bool(row["passes_frozen_gate"])
    ]
    if _code_sha256s() != frozen_code_sha256s:
        raise RuntimeError("compiler code changed during fit or selection")
    if not passing:
        (
            evidence_artifact_sha256,
            evidence_file_sha256,
            evidence_bytes,
        ) = _compile_evidence(
            fit_panel=fit_panel,
            selection_panel=selection_panel,
            response_binding_sha256=response_binding,
            rate_curve=rows,
            feature_map=feature_map,
            code_sha256s=frozen_code_sha256s,
        )
        return BilinearCompilationResult(
            candidate=None,
            fit_panel=fit_panel,
            selection_panel=selection_panel,
            rate_curve=tuple(rows),
            response_binding_sha256=response_binding,
            selection_passed=False,
            evidence_artifact_sha256=evidence_artifact_sha256,
            evidence_file_sha256=evidence_file_sha256,
            evidence_bytes=evidence_bytes,
            prepared_runtime_crosscheck=None,
        )
    selected_index, selected_plan, selected_row = min(
        passing,
        key=lambda item: (
            int(item[2]["stored_coefficient_count"]),
            int(item[2]["source_rank"]),
            int(item[2]["target_rank"]),
        ),
    )
    assert selected_plan is not None
    dense_coefficients = (
        len(FIT_ORIGINS)
        * LAG_COUNT
        * len(POSITIVE_PAIRS)
        * TARGET_RANK
    )
    accounting = {
        "selected_rate_ladder_ordinal": selected_index,
        "selected_plan_kind": selected_row["plan_kind"],
        "selected_plan_stored_coefficient_count": (
            selected_row["stored_coefficient_count"]
        ),
        "direct_dense_kernel_coefficient_count": dense_coefficients,
        "selected_fraction_of_direct_dense": (
            float(selected_row["stored_coefficient_count"])
            / dense_coefficients
        ),
        "feature_map_accounting": feature_map.accounting().metadata(),
        "dense_identity_bases_are_implicit": True,
        "dense_only_result_has_no_compaction_claim": (
            selected_row["plan_kind"] == "dense"
        ),
        "base_candidate_coefficients_excluded": True,
        "model_parameter_compression_claim": False,
        "runtime_macs_measured": False,
        "latency_measured": False,
    }
    (
        evidence_artifact_sha256,
        evidence_file_sha256,
        evidence_bytes,
    ) = _compile_evidence(
        fit_panel=fit_panel,
        selection_panel=selection_panel,
        response_binding_sha256=response_binding,
        rate_curve=rows,
        feature_map=feature_map,
        code_sha256s=frozen_code_sha256s,
    )
    candidate = Gemma3BilinearSpectralCandidate(
        base_candidate_file_sha256=_sha256(
            base_candidate_file_sha256,
            label="base candidate file",
        ),
        base_candidate_report_sha256=_sha256(
            base_candidate_report_sha256,
            label="base candidate report",
        ),
        base_candidate_artifact_sha256=base_candidate.artifact_sha256,
        hierarchy_artifact_sha256=_sha256(
            hierarchy_artifact_sha256,
            label="hierarchy artifact",
        ),
        source_model_sha256=_sha256(
            source_model_sha256,
            label="source model",
        ),
        binding=binding,
        model=model,
        protocol_sha256=frozen.artifact_sha256,
        feature_map=feature_map,
        plan_kind=str(selected_row["plan_kind"]),
        plan=selected_plan,
        response_binding_sha256=response_binding,
        fit_panel_sha256=fit_panel.artifact_sha256,
        selection_panel_sha256=selection_panel.artifact_sha256,
        compile_evidence_artifact_sha256=evidence_artifact_sha256,
        compile_evidence_file_sha256=evidence_file_sha256,
        code_sha256s=frozen_code_sha256s,
        rate_curve=tuple(rows),
        selected_rate_row=selected_row,
        accounting=accounting,
    )
    prepared_prediction, prepared_crosscheck = _prepared_crosscheck(
        candidate,
        selection_panel,
        dtype=prepared_runtime_dtype,
        device=prepared_runtime_device,
    )
    prepared_metrics = evaluate_bilinear_plan(
        selection_panel,
        selected_plan,
        branch_prediction=prepared_prediction,
    )
    prepared_crosscheck = {
        **prepared_crosscheck,
        "prepared_metrics_pass_frozen_gate": _row_passes(
            prepared_metrics
        ),
        "prepared_metrics": prepared_metrics,
    }
    if (
        not bool(prepared_crosscheck["prepared_control_exact_zero"])
        or not bool(
            prepared_crosscheck["prepared_metrics_pass_frozen_gate"]
        )
    ):
        return BilinearCompilationResult(
            candidate=None,
            fit_panel=fit_panel,
            selection_panel=selection_panel,
            rate_curve=tuple(rows),
            response_binding_sha256=response_binding,
            selection_passed=False,
            evidence_artifact_sha256=evidence_artifact_sha256,
            evidence_file_sha256=evidence_file_sha256,
            evidence_bytes=evidence_bytes,
            prepared_runtime_crosscheck=prepared_crosscheck,
        )
    return BilinearCompilationResult(
        candidate=candidate,
        fit_panel=fit_panel,
        selection_panel=selection_panel,
        rate_curve=tuple(rows),
        response_binding_sha256=response_binding,
        selection_passed=True,
        evidence_artifact_sha256=evidence_artifact_sha256,
        evidence_file_sha256=evidence_file_sha256,
        evidence_bytes=evidence_bytes,
        prepared_runtime_crosscheck=prepared_crosscheck,
    )


def measure_bilinear_panel(
    structural_map: Callable[[Tensor], Tensor],
    *,
    base_candidate: Gemma3ConditionalSpectralCandidate,
    source_sigmas: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
    split: str,
    origins: Sequence[int],
    pairs: Sequence[tuple[int, int]],
    protocol: FrozenBilinearSpectralProtocol,
) -> MeasuredBilinearPanel:
    """Measure one split without exposing a fitting operation."""

    if not callable(structural_map):
        raise TypeError("structural_map must be callable")
    if not isinstance(base_candidate, Gemma3ConditionalSpectralCandidate):
        raise TypeError("base_candidate has the wrong type")
    if (
        not isinstance(protocol, FrozenBilinearSpectralProtocol)
        or protocol.artifact_sha256
        != default_bilinear_spectral_protocol().artifact_sha256
    ):
        raise ValueError("measurement protocol is not frozen")
    origin_values = tuple(origins)
    pair_values = tuple(pairs)
    expected: tuple[tuple[int, ...], tuple[tuple[int, int], ...]]
    if split == "fit":
        expected = (FIT_ORIGINS, POSITIVE_PAIRS)
    elif split == "selection":
        expected = (
            SELECTION_ORIGINS,
            POSITIVE_PAIRS + SELECTION_CONTROL_PAIRS,
        )
    elif split == "assessment":
        expected = (
            (ASSESSMENT_ORIGIN,),
            POSITIVE_PAIRS + ASSESSMENT_CONTROL_PAIRS,
        )
    else:
        raise ValueError("measurement split is invalid")
    if (origin_values, pair_values) != expected:
        raise ValueError("measurement axes differ from frozen split")
    if (
        not isinstance(logical_positions, Tensor)
        or logical_positions.shape != (1, SEQUENCE_LENGTH)
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or not isinstance(valid_mask, Tensor)
        or valid_mask.shape != logical_positions.shape
        or valid_mask.dtype != torch.bool
        or valid_mask.device != logical_positions.device
        or not bool(valid_mask.all())
    ):
        raise ValueError("logical grid differs from frozen no-cache ABI")
    expected_positions = torch.arange(
        SEQUENCE_LENGTH,
        device=logical_positions.device,
        dtype=logical_positions.dtype,
    ).unsqueeze(0)
    if not torch.equal(logical_positions, expected_positions):
        raise ValueError("logical positions must be contiguous")
    sigma = _canonical_tensor(
        source_sigmas, label="source sigmas", ndim=1
    )
    if (
        sigma.numel() != MODAL_RANK
        or bool((sigma <= 0.0).any())
    ):
        raise ValueError("source sigmas differ from modal ABI")
    for plan in (
        base_candidate.linear_plan,
        base_candidate.quadratic_plan,
    ):
        plan_scales = _canonical_tensor(
            getattr(plan, "source_scales", None),
            label="base plan scales",
            ndim=1,
        )
        if not torch.equal(plan_scales, sigma):
            raise ValueError("base candidate scales differ from hierarchy")

    device = logical_positions.device
    dtype = getattr(structural_map, "runtime_dtype", None)
    if dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        dtype = torch.float32
    linear_runtime = _prepare_runtime(
        base_candidate.linear_plan,
        device=device,
        dtype=dtype,
    )
    quadratic_runtime = _prepare_runtime(
        base_candidate.quadratic_plan,
        device=device,
        dtype=dtype,
    )
    sigma_runtime = sigma.to(device=device, dtype=dtype)
    truth_by_origin: list[Tensor] = []
    base_by_origin: list[Tensor] = []
    zero_by_origin: list[Tensor] = []
    repeat_first_by_origin: list[Tensor] = []
    repeat_second_by_origin: list[Tensor] = []
    for origin in origin_values:
        source_mask = torch.zeros_like(valid_mask)
        source_mask[:, origin] = True
        zero_source = _source_row(
            sequence_length=SEQUENCE_LENGTH,
            modal_rank=MODAL_RANK,
            origin=origin,
            device=device,
            dtype=dtype,
            components=(),
        )
        zero = _response_slice(
            structural_map,
            zero_source,
            origin=origin,
            lag_count=LAG_COUNT,
        )
        sentinel_pair = pair_values[0]
        sentinel_scale = sigma_runtime / math.sqrt(2.0)
        sentinel_source = _source_row(
            sequence_length=SEQUENCE_LENGTH,
            modal_rank=MODAL_RANK,
            origin=origin,
            device=device,
            dtype=dtype,
            components=(
                (
                    sentinel_pair[0],
                    float(sentinel_scale[sentinel_pair[0]]),
                ),
                (
                    sentinel_pair[1],
                    float(sentinel_scale[sentinel_pair[1]]),
                ),
            ),
        )
        repeat_first = (
            _response_slice(
                structural_map,
                sentinel_source,
                origin=origin,
                lag_count=LAG_COUNT,
            )
            - zero
        )
        truth_by_pair: list[Tensor] = []
        base_by_pair: list[Tensor] = []
        for left, right in pair_values:
            truth_by_radius: list[Tensor] = []
            base_by_radius: list[Tensor] = []
            for radius in RADII:
                component = sigma_runtime * (
                    radius / math.sqrt(2.0)
                )
                truth_by_sign: list[Tensor] = []
                base_by_sign: list[Tensor] = []
                for _label, left_sign, right_sign in SIGN_ROWS:
                    source = _source_row(
                        sequence_length=SEQUENCE_LENGTH,
                        modal_rank=MODAL_RANK,
                        origin=origin,
                        device=device,
                        dtype=dtype,
                        components=(
                            (
                                left,
                                left_sign * float(component[left]),
                            ),
                            (
                                right,
                                right_sign * float(component[right]),
                            ),
                        ),
                    )
                    truth_by_sign.append(
                        _response_slice(
                            structural_map,
                            source,
                            origin=origin,
                            lag_count=LAG_COUNT,
                        )
                        - zero
                    )
                    base_by_sign.append(
                        _runtime_response(
                            linear_runtime,
                            source,
                            logical_positions=logical_positions,
                            valid_mask=valid_mask,
                            source_mask=source_mask,
                            origin=origin,
                            lag_count=LAG_COUNT,
                        )
                        + _runtime_response(
                            quadratic_runtime,
                            source,
                            logical_positions=logical_positions,
                            valid_mask=valid_mask,
                            source_mask=source_mask,
                            origin=origin,
                            lag_count=LAG_COUNT,
                        )
                    )
                truth_by_radius.append(
                    torch.stack(truth_by_sign)[:, 0]
                )
                base_by_radius.append(
                    torch.stack(base_by_sign)[:, 0]
                )
            truth_by_pair.append(torch.stack(truth_by_radius))
            base_by_pair.append(torch.stack(base_by_radius))
        repeat_second = (
            _response_slice(
                structural_map,
                sentinel_source,
                origin=origin,
                lag_count=LAG_COUNT,
            )
            - zero
        )
        truth_by_origin.append(torch.stack(truth_by_pair))
        base_by_origin.append(torch.stack(base_by_pair))
        zero_by_origin.append(zero[0])
        repeat_first_by_origin.append(repeat_first[0])
        repeat_second_by_origin.append(repeat_second[0])
    truth = torch.stack(truth_by_origin)
    base = torch.stack(base_by_origin)
    zero_tensor = torch.stack(zero_by_origin)
    repeat_first_tensor = torch.stack(repeat_first_by_origin)
    repeat_second_tensor = torch.stack(repeat_second_by_origin)
    core_calls = len(origin_values) * len(pair_values) * len(RADII) * len(
        SIGN_ROWS
    )
    structural_calls = core_calls + 3 * len(origin_values)
    measurement = {
        "fixed_reference_function_evaluation_count": structural_calls,
        "structural_response_row_count": structural_calls,
        "structural_map_batch_size": 1,
        "batched_live_measurement": False,
        "core_chord_function_evaluation_count": core_calls,
        "zero_reference_evaluation_count": len(origin_values),
        "repeat_sentinel_evaluation_count": 2 * len(origin_values),
        "repeat_sentinel_brackets_each_origin_panel": True,
        "shared_zero_subtracted_from_every_truth_response": True,
        "base_runtime_prediction_count": core_calls,
        "linear_base_runtime_call_count": core_calls,
        "quadratic_base_runtime_call_count": core_calls,
        "no_cache_full_sequence": True,
        "zero_response_l2": float(
            torch.linalg.vector_norm(zero_tensor)
        ),
        "repeat_response_l2_difference": float(
            torch.linalg.vector_norm(
                repeat_first_tensor - repeat_second_tensor
            )
        ),
        "repeat_response_max_abs_difference": float(
            (repeat_first_tensor - repeat_second_tensor).abs().max()
        ),
    }
    return MeasuredBilinearPanel(
        split=split,
        origins=origin_values,
        pairs=pair_values,
        positive_pair_count=len(POSITIVE_PAIRS),
        radii=RADII,
        truth=truth,
        base_prediction=base,
        zero_sentinel=zero_tensor,
        repeat_sentinel_first=repeat_first_tensor,
        repeat_sentinel_second=repeat_second_tensor,
        protocol_sha256=protocol.artifact_sha256,
        base_candidate_sha256=base_candidate.artifact_sha256,
        measurement=measurement,
    )


def _prepared_candidate_panel_predictions(
    candidate: Gemma3BilinearSpectralCandidate,
    panel: MeasuredBilinearPanel,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Execute the published outer graph on the exact chord ABI."""

    runtime_device = torch.device(device)
    runtime = candidate.prepare(device=runtime_device, dtype=dtype)
    positions = torch.arange(
        SEQUENCE_LENGTH,
        device=runtime_device,
        dtype=torch.long,
    ).unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)
    scales = candidate.feature_map.source_scales.to(
        device=runtime_device,
        dtype=dtype,
    )
    rows_by_origin: list[Tensor] = []
    for origin in panel.origins:
        source_mask = torch.zeros_like(valid)
        source_mask[:, origin] = True
        rows_by_pair: list[Tensor] = []
        for left, right in panel.pairs:
            rows_by_radius: list[Tensor] = []
            for radius in panel.radii:
                component = scales * (radius / math.sqrt(2.0))
                rows_by_sign: list[Tensor] = []
                for _label, left_sign, right_sign in SIGN_ROWS:
                    source = _source_row(
                        sequence_length=SEQUENCE_LENGTH,
                        modal_rank=MODAL_RANK,
                        origin=origin,
                        device=runtime_device,
                        dtype=dtype,
                        components=(
                            (
                                left,
                                left_sign * float(component[left]),
                            ),
                            (
                                right,
                                right_sign * float(component[right]),
                            ),
                        ),
                    )
                    with torch.no_grad():
                        response = runtime(
                            source,
                            logical_positions=positions,
                            valid_mask=valid,
                            source_mask=source_mask,
                        )
                    rows_by_sign.append(
                        _canonical_tensor(
                            response[
                                0, origin : origin + LAG_COUNT
                            ],
                            label="prepared candidate response",
                            ndim=2,
                        )
                    )
                rows_by_radius.append(torch.stack(rows_by_sign))
            rows_by_pair.append(torch.stack(rows_by_radius))
        rows_by_origin.append(torch.stack(rows_by_pair))
    return torch.stack(rows_by_origin)


def _prepared_crosscheck(
    candidate: Gemma3BilinearSpectralCandidate,
    panel: MeasuredBilinearPanel,
    *,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, dict[str, object]]:
    prepared = _prepared_candidate_panel_predictions(
        candidate,
        panel,
        dtype=dtype,
        device=device,
    )
    analytic = _branch_predictions(panel, candidate.plan)
    difference = prepared - analytic
    control = prepared[:, panel.positive_pair_count :]
    return prepared, {
        "runtime_dtype": str(dtype),
        "prepared_vs_analytic_relative_error": _relative_error(
            prepared, analytic
        ),
        "prepared_vs_analytic_max_abs_difference": float(
            difference.abs().max()
        ),
        "prepared_control_exact_zero": torch.equal(
            control, torch.zeros_like(control)
        ),
        "prepared_runtime_was_executed": True,
        "analytic_formula_was_not_the_only_executor_validation": True,
    }


def _build_live_structural_map(
    adapter: Gemma3CausalLMAdapter,
    reference: Any,
    protocol: FrozenBilinearSpectralProtocol,
) -> tuple[
    Callable[[Tensor], Tensor],
    Tensor,
    Tensor,
    Tensor,
    dict[str, int],
    dict[str, object],
]:
    if adapter.module.training or any(
        parameter.requires_grad for parameter in adapter.module.parameters()
    ):
        raise ValueError("bilinear measurement requires frozen eval Gemma")
    layer3_spec = adapter.layer("layer.3")
    layer4_spec = adapter.layer("layer.4")
    layer3 = adapter.source_module(layer3_spec.id)
    layer4 = adapter.source_module(layer4_spec.id)
    pre_ff3 = getattr(layer3, "pre_feedforward_layernorm", None)
    post_ff3 = getattr(layer3, "post_feedforward_layernorm", None)
    if not isinstance(pre_ff3, nn.Module) or not isinstance(
        post_ff3, nn.Module
    ):
        raise TypeError("live Gemma L3 normalization modules are missing")
    for name in (
        "input_layernorm",
        "self_attn",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
    ):
        if not isinstance(getattr(layer4, name, None), nn.Module):
            raise TypeError("live Gemma L4 attention prefix is incomplete")
    transformer3 = layer3_spec.transformer
    if (
        transformer3 is None
        or transformer3.feed_forward_input_norm.kind != "rms_norm"
        or transformer3.feed_forward_input_norm.scale_parameterization
        != "unit_offset"
    ):
        raise ValueError("Gemma L3 RMSNorm semantics drifted")
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
    logical_positions = torch.arange(
        protocol.sequence_length,
        device=device,
        dtype=torch.long,
    ).unsqueeze(0)
    valid_mask = torch.ones_like(logical_positions, dtype=torch.bool)
    placeholder = torch.zeros(
        (1, protocol.sequence_length, reference.residual_width),
        device=device,
        dtype=dtype,
    )
    sequence = adapter.prepare_sequence(
        {
            "inputs_embeds": placeholder,
            "attention_mask": valid_mask,
            "position_ids": logical_positions,
        }
    )
    preimage = fixed_reference.value.to(
        device=device, dtype=dtype
    ).view(1, 1, -1).expand(
        1, protocol.sequence_length, -1
    )
    y3_mean = reference.y3_mean.to(
        device=device, dtype=dtype
    ).view(1, 1, -1).expand(
        1, protocol.sequence_length, -1
    )
    p3 = reference.P3[:, : protocol.modal_rank].to(
        device=device, dtype=dtype
    )
    r4 = reference.R4[: protocol.target_rank].to(
        device=device, dtype=dtype
    )
    segment4 = adapter.segment("layer.4")
    with torch.no_grad():
        hidden3_reference = preimage + post_ff3(y3_mean)
        baseline_x4 = adapter.run_attention_prefix(
            segment4,
            hidden3_reference,
            sequence,
        ).normalized_mlp_input.detach()
    counter = {"calls": 0}

    def structural_map(source_modes: Tensor) -> Tensor:
        if (
            not isinstance(source_modes, Tensor)
            or source_modes.shape
            != (
                1,
                protocol.sequence_length,
                protocol.modal_rank,
            )
            or source_modes.device != device
            or source_modes.dtype != dtype
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError("source modes differ from fixed-reference ABI")
        counter["calls"] += 1
        with torch.no_grad():
            y3 = y3_mean + source_modes @ p3.T
            hidden3 = preimage + post_ff3(y3)
            x4 = adapter.run_attention_prefix(
                segment4,
                hidden3,
                sequence,
            ).normalized_mlp_input
            return (x4 - baseline_x4) @ r4.T

    setattr(structural_map, "runtime_dtype", dtype)
    sigma = reference.source_mode_standard_deviations(
        protocol.modal_rank
    )
    p3_macs = protocol.sequence_length * p3.numel()
    r4_macs = protocol.sequence_length * r4.numel()
    attention_projection_macs = (
        protocol.sequence_length
        * sum(
            getattr(layer4.self_attn, name).weight.numel()
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
    )
    metadata = {
        "canonical_reference": fixed_reference.metadata(),
        "baseline_x4_rms": float(
            baseline_x4.detach().float().square().mean().sqrt()
        ),
        "baseline_prefix_evaluation_count": 1,
        "native_or_compiled_l3_mlp_body_executions": 0,
        "native_or_compiled_l4_mlp_body_executions": 0,
        "no_cache_full_sequence": True,
        "partial_analytic_live_model_macs_per_structural_call": {
            "P3_decode_macs": p3_macs,
            "R4_projection_macs": r4_macs,
            "l4_attention_projection_weight_macs": (
                attention_projection_macs
            ),
            "counted_macs_per_structural_call": (
                p3_macs + r4_macs + attention_projection_macs
            ),
            "baseline_prefix_attention_projection_macs": (
                attention_projection_macs
            ),
            "excluded": (
                "linear_bias_additions",
                "normalization",
                "RoPE",
                "attention_score_and_value_matmuls",
                "softmax",
                "elementwise_and_residual_ops",
                "memory_traffic",
                "compiled_branch_runtime",
            ),
            "latency_claim": False,
        },
    }
    return (
        structural_map,
        logical_positions,
        valid_mask,
        sigma,
        counter,
        metadata,
    )


def _load_live_dependencies(
    *,
    base_candidate_path: Path | str,
    base_candidate_file_sha256: str,
    base_candidate_report_sha256: str,
    hierarchy_artifact_path: Path | str,
    hierarchy_artifact_sha256: str,
    base_artifact_path: Path | str,
    refit_artifact_path: Path | str,
    model_id: str,
    revision: str,
    cache_dir: Path | str | None,
    device_name: str,
    dtype: str,
) -> tuple[
    Gemma3ConditionalSpectralCandidate,
    Any,
    Gemma3CausalLMAdapter,
    PreparedGemma3FullMLPStackSwitcher,
]:
    base_candidate = load_gemma3_conditional_spectral_candidate(
        base_candidate_path,
        expected_file_sha256=base_candidate_file_sha256,
        expected_report_sha256=base_candidate_report_sha256,
    )
    catalog = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    reference = load_gemma3_l3_l4_spectral_reference(
        hierarchy_artifact_path,
        expected_file_sha256=hierarchy_artifact_sha256,
        catalog=catalog,
    )
    if dict(reference.metadata()) != dict(base_candidate.binding):
        raise ValueError("base candidate and hierarchy bindings differ")
    if (
        base_candidate.model.get("model_id") != model_id
        or base_candidate.model.get("requested_revision") != revision
        or base_candidate.model.get("resolved_commit") != revision
        or base_candidate.model.get("local_files_only") is not True
        or base_candidate.model.get("tokenizer_loaded") is not False
    ):
        raise ValueError("base candidate model metadata drifted")
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
    if adapter.model_fingerprint() != reference.source_model_sha256:
        raise ValueError("live Gemma fingerprint differs from hierarchy")
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_SOURCE_SCOPE: catalog.replacements},
    )
    return base_candidate, reference, adapter, switcher


def compile_gemma3_l3_l4_bilinear_spectral_executor(
    *,
    base_candidate_path: Path | str = DEFAULT_CANDIDATE,
    base_candidate_file_sha256: str = DEFAULT_CANDIDATE_FILE_SHA256,
    base_candidate_report_sha256: str = DEFAULT_CANDIDATE_REPORT_SHA256,
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
) -> dict[str, object]:
    """Measure fit, freeze every plan, then open selection and publish."""

    protocol = default_bilinear_spectral_protocol()
    protocol_before = protocol.artifact_sha256
    code_before = _code_sha256s()
    if (
        base_candidate_file_sha256
        != DEFAULT_CANDIDATE_FILE_SHA256
        or base_candidate_report_sha256
        != DEFAULT_CANDIDATE_REPORT_SHA256
    ):
        raise ValueError("compile requires the pinned base candidate")
    if hierarchy_artifact_sha256 != DEFAULT_HIERARCHY_ARTIFACT_SHA256:
        raise ValueError("compile requires the pinned hierarchy")
    destination = _validate_output_path(output, suffix=".pt")
    for extra in (
        destination.with_suffix(".json"),
        destination.with_name(
            f"{destination.stem}-compile-evidence.pt"
        ),
    ):
        if extra.exists():
            raise FileExistsError("refusing to overwrite bilinear output")
    base_file_before = _file_sha256(base_candidate_path)
    (
        base_candidate,
        reference,
        adapter,
        switcher,
    ) = _load_live_dependencies(
        base_candidate_path=base_candidate_path,
        base_candidate_file_sha256=base_candidate_file_sha256,
        base_candidate_report_sha256=base_candidate_report_sha256,
        hierarchy_artifact_path=hierarchy_artifact_path,
        hierarchy_artifact_sha256=hierarchy_artifact_sha256,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    source_model_before = adapter.model_fingerprint()
    base_plan_hashes_before = (
        _sha256(
            base_candidate.linear_plan.artifact_sha256,
            label="base linear plan",
        ),
        _sha256(
            base_candidate.quadratic_plan.artifact_sha256,
            label="base quadratic plan",
        ),
    )
    try:
        switcher.switch(_SOURCE_SCOPE)
        (
            structural_map,
            positions,
            valid,
            sigma,
            counter,
            live_metadata,
        ) = _build_live_structural_map(adapter, reference, protocol)
        feature_binding = _feature_source_binding(
            protocol_sha256=protocol.artifact_sha256,
            base_candidate_artifact_sha256=(
                base_candidate.artifact_sha256
            ),
            hierarchy_artifact_sha256=hierarchy_artifact_sha256,
            source_model_sha256=source_model_before,
        )
        feature_map = build_explicit_pair_product_feature_map(
            sigma,
            source_pairs=POSITIVE_PAIRS,
            source_binding_sha256=feature_binding,
        )

        def fit_factory() -> MeasuredBilinearPanel:
            return measure_bilinear_panel(
                structural_map,
                base_candidate=base_candidate,
                source_sigmas=sigma,
                logical_positions=positions,
                valid_mask=valid,
                split="fit",
                origins=FIT_ORIGINS,
                pairs=POSITIVE_PAIRS,
                protocol=protocol,
            )

        def selection_factory() -> MeasuredBilinearPanel:
            return measure_bilinear_panel(
                structural_map,
                base_candidate=base_candidate,
                source_sigmas=sigma,
                logical_positions=positions,
                valid_mask=valid,
                split="selection",
                origins=SELECTION_ORIGINS,
                pairs=POSITIVE_PAIRS + SELECTION_CONTROL_PAIRS,
                protocol=protocol,
            )

        result = compile_bilinear_spectral_candidate(
            fit_panel_factory=fit_factory,
            selection_panel_factory=selection_factory,
            base_candidate=base_candidate,
            feature_map=feature_map,
            base_candidate_file_sha256=base_candidate_file_sha256,
            base_candidate_report_sha256=(
                base_candidate_report_sha256
            ),
            hierarchy_artifact_sha256=hierarchy_artifact_sha256,
            source_model_sha256=source_model_before,
            binding=reference.metadata(),
            model=base_candidate.model,
            protocol=protocol,
            code_sha256s=code_before,
            prepared_runtime_dtype=getattr(
                structural_map, "runtime_dtype"
            ),
            prepared_runtime_device=positions.device,
        )
        if counter["calls"] != 1231:
            raise RuntimeError(
                "compile structural call count differs from 1231"
            )
    finally:
        switcher.close()
    source_model_after = adapter.model_fingerprint()
    base_file_after = _file_sha256(base_candidate_path)
    base_plan_hashes_after = (
        base_candidate.linear_plan.artifact_sha256,
        base_candidate.quadratic_plan.artifact_sha256,
    )
    if (
        source_model_after != source_model_before
        or base_file_after != base_file_before
        or base_plan_hashes_after != base_plan_hashes_before
        or protocol.artifact_sha256 != protocol_before
        or _code_sha256s() != code_before
    ):
        raise RuntimeError(
            "model, base candidate, protocol, or code changed during compile"
        )
    result = replace(
        result,
        orchestration={
            **live_metadata,
            "compile_structural_function_evaluations": 1231,
            "compile_l4_attention_prefix_evaluations": 1232,
            "fit_core_calls": 672,
            "selection_core_calls": 544,
            "per_origin_zero_and_repeat_calls": 15,
            "source_model_sha256_before": source_model_before,
            "source_model_sha256_after": source_model_after,
            "partial_analytic_live_model_macs_total": (
                1231
                * int(
                    live_metadata[
                        "partial_analytic_live_model_macs_per_structural_call"
                    ]["counted_macs_per_structural_call"]  # type: ignore[index]
                )
                + int(
                    live_metadata[
                        "partial_analytic_live_model_macs_per_structural_call"
                    ][  # type: ignore[index]
                        "baseline_prefix_attention_projection_macs"
                    ]
                )
            ),
        },
    )
    return _publish_compilation(result, output=destination)


def _publish_assessment(
    artifact: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    reservation = _reserve_outputs((output, report_path))
    tensor_stage: Path | None = None
    report_stage: Path | None = None
    try:
        tensor_stage = _stage_torch(dict(artifact), output)
        report: dict[str, object] = {
            **dict(report_payload),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": _file_sha256(tensor_stage),
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "committable": False,
            },
        }
        report["report_sha256"] = _json_sha256(
            report, domain=_REPORT_DOMAIN
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


def assess_gemma3_l3_l4_bilinear_spectral_executor(
    *,
    candidate_path: Path | str,
    candidate_file_sha256: str,
    candidate_report_sha256: str,
    base_candidate_path: Path | str = DEFAULT_CANDIDATE,
    hierarchy_artifact_path: Path | str = DEFAULT_HIERARCHY_ARTIFACT,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_REFIT_ARTIFACT,
    output: Path | str = DEFAULT_ASSESSMENT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Open origin 20 only after strict candidate authentication; never fit."""

    destination = _validate_output_path(output, suffix=".pt")
    if destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite assessment report")
    protocol = default_bilinear_spectral_protocol()
    protocol_before = protocol.artifact_sha256
    code_before = _code_sha256s()

    # FIREWALL: candidate, evidence, report, and code authenticate before any
    # hierarchy or model object capable of opening origin 20 is loaded.
    candidate_file = Path(candidate_path)
    candidate = load_gemma3_bilinear_spectral_candidate(
        candidate_file,
        expected_file_sha256=candidate_file_sha256,
        expected_report_sha256=candidate_report_sha256,
    )
    if dict(candidate.code_sha256s) != code_before:
        raise ValueError("assessment code differs from compiled candidate")
    if (
        candidate.protocol_sha256 != protocol.artifact_sha256
        or candidate.base_candidate_file_sha256
        != DEFAULT_CANDIDATE_FILE_SHA256
        or candidate.base_candidate_report_sha256
        != DEFAULT_CANDIDATE_REPORT_SHA256
        or candidate.hierarchy_artifact_sha256
        != DEFAULT_HIERARCHY_ARTIFACT_SHA256
    ):
        raise ValueError("assessment candidate source bindings drifted")
    candidate_file_before = _file_sha256(candidate_file)
    candidate_plan_before = _plan_hash(candidate.plan)
    base_file_before = _file_sha256(base_candidate_path)
    (
        base_candidate,
        reference,
        adapter,
        switcher,
    ) = _load_live_dependencies(
        base_candidate_path=base_candidate_path,
        base_candidate_file_sha256=(
            candidate.base_candidate_file_sha256
        ),
        base_candidate_report_sha256=(
            candidate.base_candidate_report_sha256
        ),
        hierarchy_artifact_path=hierarchy_artifact_path,
        hierarchy_artifact_sha256=(
            candidate.hierarchy_artifact_sha256
        ),
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    if (
        base_candidate.artifact_sha256
        != candidate.base_candidate_artifact_sha256
        or dict(reference.metadata()) != dict(candidate.binding)
        or dict(base_candidate.model) != dict(candidate.model)
    ):
        raise ValueError("assessment base candidate or hierarchy drifted")
    base_plan_hashes_before = (
        base_candidate.linear_plan.artifact_sha256,
        base_candidate.quadratic_plan.artifact_sha256,
    )
    feature_map_hash_before = candidate.feature_map.artifact_sha256
    source_model_before = adapter.model_fingerprint()
    if source_model_before != candidate.source_model_sha256:
        raise ValueError("assessment live model differs from candidate")
    try:
        switcher.switch(_SOURCE_SCOPE)
        (
            structural_map,
            positions,
            valid,
            sigma,
            counter,
            live_metadata,
        ) = _build_live_structural_map(adapter, reference, protocol)
        if not torch.equal(
            candidate.feature_map.source_scales,
            _canonical_tensor(sigma, label="live source scales", ndim=1),
        ):
            raise ValueError(
                "assessment modal scales differ from candidate feature map"
            )
        panel = measure_bilinear_panel(
            structural_map,
            base_candidate=base_candidate,
            source_sigmas=sigma,
            logical_positions=positions,
            valid_mask=valid,
            split="assessment",
            origins=(ASSESSMENT_ORIGIN,),
            pairs=POSITIVE_PAIRS + ASSESSMENT_CONTROL_PAIRS,
            protocol=protocol,
        )
        if counter["calls"] != 275:
            raise RuntimeError(
                "assessment structural call count differs from 275"
            )
        runtime_dtype = getattr(structural_map, "runtime_dtype")
        prepared, prepared_crosscheck = _prepared_crosscheck(
            candidate,
            panel,
            dtype=runtime_dtype,
            device=positions.device,
        )
        metrics = evaluate_bilinear_plan(
            panel,
            candidate.plan,
            assessment=True,
            branch_prediction=prepared,
        )
        if not bool(prepared_crosscheck["prepared_control_exact_zero"]):
            raise RuntimeError(
                "prepared assessment branch is nonzero on controls"
            )
    finally:
        switcher.close()
    source_model_after = adapter.model_fingerprint()
    candidate_file_after = _file_sha256(candidate_file)
    base_file_after = _file_sha256(base_candidate_path)
    candidate_plan_after = _plan_hash(candidate.plan)
    base_plan_hashes_after = (
        base_candidate.linear_plan.artifact_sha256,
        base_candidate.quadratic_plan.artifact_sha256,
    )
    candidate.feature_map.validate_integrity()
    feature_map_hash_after = candidate.feature_map.artifact_sha256
    code_after = _code_sha256s()
    if (
        source_model_after != source_model_before
        or candidate_file_after != candidate_file_before
        or base_file_after != base_file_before
        or candidate_plan_after != candidate_plan_before
        or base_plan_hashes_after != base_plan_hashes_before
        or feature_map_hash_after != feature_map_hash_before
        or protocol.artifact_sha256 != protocol_before
        or code_after != code_before
        or dict(candidate.code_sha256s) != code_after
    ):
        raise RuntimeError(
            "model, candidate, base, protocol, or code changed in assessment"
        )
    common = {
        "schema": _ASSESSMENT_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "binding": {
            "candidate_tensor_file_sha256": candidate_file_before,
            "candidate_report_payload_sha256": _sha256(
                candidate_report_sha256,
                label="candidate report",
            ),
            "candidate_artifact_sha256": candidate.artifact_sha256,
            "candidate_plan_artifact_sha256": candidate_plan_before,
            "candidate_feature_map_artifact_sha256": (
                candidate.feature_map.artifact_sha256
            ),
            "candidate_feature_map_artifact_sha256_after": (
                feature_map_hash_after
            ),
            "base_linear_plan_artifact_sha256_before": (
                base_plan_hashes_before[0]
            ),
            "base_linear_plan_artifact_sha256_after": (
                base_plan_hashes_after[0]
            ),
            "base_quadratic_plan_artifact_sha256_before": (
                base_plan_hashes_before[1]
            ),
            "base_quadratic_plan_artifact_sha256_after": (
                base_plan_hashes_after[1]
            ),
            "compile_evidence_artifact_sha256": (
                candidate.compile_evidence_artifact_sha256
            ),
            "compile_evidence_file_sha256": (
                candidate.compile_evidence_file_sha256
            ),
            "assessment_panel_artifact_sha256": panel.artifact_sha256,
            "source_model_sha256_before": source_model_before,
            "source_model_sha256_after": source_model_after,
            "code_sha256s_before": code_before,
            "code_sha256s_after": code_after,
        },
        "split": {
            "compile_fit_origins": FIT_ORIGINS,
            "compile_selection_origins": SELECTION_ORIGINS,
            "assessment_origin": ASSESSMENT_ORIGIN,
            "assessment_origin_was_opened_during_compilation": False,
            "candidate_authenticated_before_assessment_origin_opened": True,
            "assessment_refit_performed": False,
            "assessment_gate_relaxation_performed": False,
        },
        "protocol": protocol.metadata(),
        "candidate": candidate.metadata(),
        "assessment_panel": panel.metadata(),
        "metrics": metrics,
        "prepared_runtime_crosscheck": prepared_crosscheck,
        "prepared_branch_tensor_manifest": {
            "sha256": _tensor_sha256(prepared),
            "shape": tuple(prepared.shape),
            "dtype": "float64_canonical_storage",
        },
        "decision": (
            "passes_frozen_assessment"
            if bool(metrics["passes_frozen_gate"])
            else "fails_frozen_assessment"
        ),
        "live_measurement": {
            **live_metadata,
            "assessment_structural_function_evaluations": 275,
            "assessment_l4_attention_prefix_evaluations": 276,
            "assessment_core_calls": 272,
            "per_origin_zero_and_repeat_calls": 3,
            "partial_analytic_live_model_macs_total": (
                275
                * int(
                    live_metadata[
                        "partial_analytic_live_model_macs_per_structural_call"
                    ]["counted_macs_per_structural_call"]  # type: ignore[index]
                )
                + int(
                    live_metadata[
                        "partial_analytic_live_model_macs_per_structural_call"
                    ][  # type: ignore[index]
                        "baseline_prefix_attention_projection_macs"
                    ]
                )
            ),
            "compile_plus_assessment_partial_analytic_live_model_macs": (
                1506
                * int(
                    live_metadata[
                        "partial_analytic_live_model_macs_per_structural_call"
                    ]["counted_macs_per_structural_call"]  # type: ignore[index]
                )
                + 2
                * int(
                    live_metadata[
                        "partial_analytic_live_model_macs_per_structural_call"
                    ][  # type: ignore[index]
                        "baseline_prefix_attention_projection_macs"
                    ]
                )
            ),
        },
        "claim_boundaries": _claim_boundaries(
            selected_plan_kind=candidate.plan_kind
        ),
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_tokenizer": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "contains_structural_response_tensors": True,
            "contains_compiled_candidate_plan": False,
            "assessment_refit_performed": False,
            "committable": False,
        },
    }
    logical_artifact_sha256 = _json_sha256(
        common, domain=_ARTIFACT_DOMAIN
    )
    artifact = {
        **common,
        "assessment_panel_state": panel.state_dict(),
        "prepared_branch_prediction": prepared,
        "artifact_sha256": logical_artifact_sha256,
    }
    report_payload = {
        **common,
        "artifact_sha256": logical_artifact_sha256,
    }
    return _publish_assessment(
        artifact,
        report_payload,
        output=destination,
    )


def _add_live_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-candidate",
        type=Path,
        default=DEFAULT_CANDIDATE,
    )
    parser.add_argument(
        "--hierarchy-artifact",
        type=Path,
        default=DEFAULT_HIERARCHY_ARTIFACT,
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
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile or independently assess the development-only Gemma "
            "L3-L4 explicit-pair bilinear spectral correction."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser(
        "compile",
        help=(
            "measure fit origins, freeze the complete ladder, then open "
            "selection origins"
        ),
    )
    _add_live_model_arguments(compile_parser)
    compile_parser.add_argument(
        "--base-candidate-file-sha256",
        default=DEFAULT_CANDIDATE_FILE_SHA256,
    )
    compile_parser.add_argument(
        "--base-candidate-report-sha256",
        default=DEFAULT_CANDIDATE_REPORT_SHA256,
    )
    compile_parser.add_argument(
        "--hierarchy-artifact-sha256",
        default=DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    )
    compile_parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )

    assess_parser = commands.add_parser(
        "assess",
        help=(
            "authenticate a published passing candidate before opening "
            "fresh origin 20"
        ),
    )
    _add_live_model_arguments(assess_parser)
    assess_parser.add_argument("--candidate", type=Path, required=True)
    assess_parser.add_argument(
        "--candidate-file-sha256", required=True
    )
    assess_parser.add_argument(
        "--candidate-report-sha256", required=True
    )
    assess_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ASSESSMENT_OUTPUT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "compile":
        report = compile_gemma3_l3_l4_bilinear_spectral_executor(
            base_candidate_path=arguments.base_candidate,
            base_candidate_file_sha256=(
                arguments.base_candidate_file_sha256
            ),
            base_candidate_report_sha256=(
                arguments.base_candidate_report_sha256
            ),
            hierarchy_artifact_path=arguments.hierarchy_artifact,
            hierarchy_artifact_sha256=(
                arguments.hierarchy_artifact_sha256
            ),
            base_artifact_path=arguments.base_artifact,
            refit_artifact_path=arguments.refit_artifact,
            output=arguments.output,
            model_id=arguments.model_id,
            revision=arguments.revision,
            cache_dir=arguments.cache_dir,
            device_name=arguments.device,
            dtype=arguments.dtype,
        )
    elif arguments.command == "assess":
        report = assess_gemma3_l3_l4_bilinear_spectral_executor(
            candidate_path=arguments.candidate,
            candidate_file_sha256=arguments.candidate_file_sha256,
            candidate_report_sha256=arguments.candidate_report_sha256,
            base_candidate_path=arguments.base_candidate,
            hierarchy_artifact_path=arguments.hierarchy_artifact,
            base_artifact_path=arguments.base_artifact,
            refit_artifact_path=arguments.refit_artifact,
            output=arguments.output,
            model_id=arguments.model_id,
            revision=arguments.revision,
            cache_dir=arguments.cache_dir,
            device_name=arguments.device,
            dtype=arguments.dtype,
        )
    else:  # pragma: no cover - argparse enforces the command.
        raise AssertionError("unreachable command")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
