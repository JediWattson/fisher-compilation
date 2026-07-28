"""Prompt-free mixed-mode falsification for the frozen Gemma L3->L4 map.

This rung asks whether the already-frozen linear plus diagonal-square
conditional spectral executor remains additive when two source modes are
played together.  It does not fit or modify either generator.

The protocol is frozen before the live reference is evaluated:

* one previously unopened source origin (28);
* a balanced 24-pair, 16-mode panel;
* radial standardized magnitudes 0.5 and 1.0, with each component scaled by
  ``rho / sqrt(2)``;
* four ordered signs (++,+-,-+,--);
* singleton +/- controls at the identical component magnitudes; and
* causal output lags 0..31 only.

For signs ``s,t``, exact chord nonadditivity is

``I_st = f(s*u + t*v) - f(s*u) - f(t*v)``

because every response uses the same explicitly measured zero reference.
The Walsh ``C11`` component is retained as an odd/odd diagnostic, but is not
mistaken for the complete interaction.  A full-interaction oracle reports
how much candidate error could be removed if ``I`` were known exactly.

The ignored tensor artifact contains structural response tensors.  Its JSON
companion contains only authenticated manifests, aggregate metrics, protocol
declarations, and conservative claim boundaries.  Neither artifact contains
model state, prompts, token IDs, or a tokenizer.
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

from .adapters import Gemma3CausalLMAdapter
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
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_l3_l4_conditional_spectral_executor_experiment import (
    Gemma3ConditionalSpectralCandidate,
    load_gemma3_conditional_spectral_candidate,
)
from .gemma3_l3_l4_spectral_mapping_experiment import (
    DEFAULT_HIERARCHY_ARTIFACT,
    DEFAULT_HIERARCHY_ARTIFACT_SHA256,
    DEFAULT_REVISION,
    _load_local_gemma3_model_only,
    invert_unit_offset_rmsnorm_reference,
    load_gemma3_l3_l4_spectral_reference,
)
from .mixed_modal_interaction import (
    build_mixed_modal_interaction_artifact,
    pairwise_all_nonadditivity,
)
from .prepared_gemma3_full_mlp_stack import (
    PreparedGemma3FullMLPStackSwitcher,
)


__all__ = [
    "DEFAULT_CANDIDATE",
    "DEFAULT_CANDIDATE_FILE_SHA256",
    "DEFAULT_CANDIDATE_REPORT_SHA256",
    "DEFAULT_OUTPUT",
    "FrozenMixedModeProtocol",
    "build_parser",
    "default_mixed_mode_protocol",
    "evaluate_mixed_mode_falsification",
    "main",
    "run_gemma3_l3_l4_mixed_mode_falsification_experiment",
]


DEFAULT_CANDIDATE = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-conditional-spectral-executor-dev-v1.pt"
)
DEFAULT_CANDIDATE_FILE_SHA256 = (
    "9be7c0345acfaef8d77c273b1b69e3d83c930b807fd33756f955b5eef3fe2d2a"
)
DEFAULT_CANDIDATE_REPORT_SHA256 = (
    "ce0649eb1d4559524243e8ad7b10dd9482dea31ca9ace35bde4e8568f2f49abc"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-mixed-mode-falsification-dev-v1.pt"
)

ORIGIN = 28
SEQUENCE_LENGTH = 60
MODAL_RANK = 64
MAX_LAG = 31
LAG_COUNT = MAX_LAG + 1
RADII = (0.5, 1.0)
SIGN_ROWS = (
    ("++", 1, 1),
    ("+-", 1, -1),
    ("-+", -1, 1),
    ("--", -1, -1),
)
PAIR_FAMILIES = (
    (
        "candidate_leverage_stress",
        (
            (0, 2),
            (1, 2),
            (1, 15),
            (15, 43),
            (7, 43),
            (7, 42),
            (28, 42),
            (0, 28),
            (0, 43),
            (2, 7),
            (1, 42),
            (15, 28),
        ),
    ),
    (
        "rank_coverage",
        (
            (3, 4),
            (4, 16),
            (16, 17),
            (17, 32),
            (32, 33),
            (33, 62),
            (62, 63),
            (3, 63),
            (3, 32),
            (4, 33),
            (16, 62),
            (17, 63),
        ),
    ),
)
UNIQUE_MODES = (
    0,
    1,
    2,
    3,
    4,
    7,
    15,
    16,
    17,
    28,
    32,
    33,
    42,
    43,
    62,
    63,
)

# All decisions are applied to unrounded float64 measurements.
GATES = {
    "operating_radius": 1.0,
    "maximum_corrected_global_relative_error": 0.225,
    "minimum_corrected_global_cosine": 0.975,
    "maximum_pooled_nonadditivity_relative_norm": 0.05,
    "maximum_family_nonadditivity_relative_norm": 0.075,
    "maximum_reliable_pair_nonadditivity_relative_norm": 0.10,
    "maximum_full_interaction_oracle_gain": 0.05,
    "c11_total_energy_fraction_diagnostic_maximum": 0.05,
    "bilinear_branch_minimum_interaction_parity_c11_energy_fraction": 0.75,
    "bilinear_branch_minimum_c11_hessian_scaling_cosine": 0.95,
    "bilinear_branch_maximum_c11_hessian_scaling_relative_error": 0.25,
    "material_pooled_nonadditivity_relative_norm": 0.10,
    "material_family_nonadditivity_relative_norm": 0.15,
    "material_full_interaction_oracle_gain": 0.10,
    "material_pair_nonadditivity_relative_norm": 0.20,
    "material_pair_minimum_panel_median_response_rms_fraction": 0.25,
}

_SCHEMA = "fisher_graph.gemma3_l3_l4_mixed_mode_falsification_development"
_FORMAT_VERSION = 1
_SOURCE_SCOPE = "factorized_refit"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROTOCOL_DOMAIN = b"fisher-graph:gemma3-l3-l4-mixed-protocol:v1\0"
_ARTIFACT_DOMAIN = b"fisher-graph:gemma3-l3-l4-mixed-artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:gemma3-l3-l4-mixed-report:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:gemma3-l3-l4-mixed-tensor:v1\0"
_COMPONENT_NORMALIZATION = (
    "each_modal_component=rho*source_sigma/sqrt(2);"
    "two_mode_standardized_radial_norm=rho"
)


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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical_tensor(value: object, *, label: str, ndim: int) -> Tensor:
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
        device="cpu",
        dtype=torch.float64,
    ).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": "float64",
                "shape": tuple(int(width) for width in tensor.shape),
            }
        )
    )
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _relative_error(prediction: Tensor, target: Tensor) -> float:
    left = prediction.detach().to(dtype=torch.float64).reshape(-1)
    right = target.detach().to(dtype=torch.float64).reshape(-1)
    denominator = max(
        float(torch.linalg.vector_norm(right)),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.vector_norm(left - right)) / denominator


def _relative_norm(value: Tensor, reference: Tensor) -> float:
    denominator = max(
        float(torch.linalg.vector_norm(reference)),
        torch.finfo(torch.float64).eps,
    )
    return float(torch.linalg.vector_norm(value)) / denominator


def _cosine(first: Tensor, second: Tensor) -> float:
    left = first.detach().to(dtype=torch.float64).reshape(-1)
    right = second.detach().to(dtype=torch.float64).reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    floor = torch.finfo(torch.float64).eps
    if left_norm <= floor:
        return 1.0 if right_norm <= floor else 0.0
    if right_norm <= floor:
        return 0.0
    return max(
        -1.0,
        min(1.0, float(torch.dot(left, right)) / (left_norm * right_norm)),
    )


def _fractional_error_reduction(
    original_prediction: Tensor,
    oracle_prediction: Tensor,
    truth: Tensor,
) -> float:
    original = _relative_error(original_prediction, truth)
    oracle = _relative_error(oracle_prediction, truth)
    if original <= torch.finfo(torch.float64).eps:
        return 0.0
    return (original - oracle) / original


@dataclass(frozen=True, slots=True)
class FrozenMixedModeProtocol:
    """Authenticated declaration that is constructed before inference."""

    origin: int
    sequence_length: int
    modal_rank: int
    max_lag: int
    radii: tuple[float, ...]
    sign_rows: tuple[tuple[str, int, int], ...]
    pair_families: tuple[
        tuple[str, tuple[tuple[int, int], ...]],
        ...,
    ]
    unique_modes: tuple[int, ...]
    gates: Mapping[str, float]
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if (
            self.origin != ORIGIN
            or self.sequence_length != SEQUENCE_LENGTH
            or self.modal_rank != MODAL_RANK
            or self.max_lag != MAX_LAG
            or self.radii != RADII
            or self.sign_rows != SIGN_ROWS
            or self.pair_families != PAIR_FAMILIES
            or self.unique_modes != UNIQUE_MODES
            or dict(self.gates) != GATES
        ):
            raise ValueError("mixed-mode protocol differs from the frozen panel")
        pairs = tuple(
            pair
            for _family, family_pairs in self.pair_families
            for pair in family_pairs
        )
        if (
            len(pairs) != 24
            or len(set(pairs)) != len(pairs)
            or any(
                left >= right
                or left < 0
                or right >= self.modal_rank
                for left, right in pairs
            )
            or tuple(sorted({mode for pair in pairs for mode in pair}))
            != self.unique_modes
            or self.origin + self.max_lag >= self.sequence_length
        ):
            raise ValueError("mixed-mode pair/grid declaration is invalid")
        object.__setattr__(self, "gates", dict(self.gates))
        computed = _json_sha256(self._payload(), domain=_PROTOCOL_DOMAIN)
        if self.artifact_sha256 == "":
            object.__setattr__(self, "artifact_sha256", computed)
        elif self.artifact_sha256 != computed:
            raise ValueError("mixed-mode protocol hash mismatch")

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            pair
            for _family, family_pairs in self.pair_families
            for pair in family_pairs
        )

    @property
    def family_by_pair(self) -> tuple[str, ...]:
        return tuple(
            family
            for family, pairs in self.pair_families
            for _pair in pairs
        )

    def _payload(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "sequence_length": self.sequence_length,
            "modal_rank": self.modal_rank,
            "max_lag": self.max_lag,
            "lag_count": self.max_lag + 1,
            "radii": self.radii,
            "sign_rows": self.sign_rows,
            "pair_families": self.pair_families,
            "pair_count": len(self.pairs),
            "unique_modes": self.unique_modes,
            "unique_mode_count": len(self.unique_modes),
            "component_normalization": _COMPONENT_NORMALIZATION,
            "pair_canonicalization": "unordered_pairs_stored_as_left_lt_right",
            "stress_pair_selection": (
                "exact_ring_order_over_quadratic_source_basis_row_leverage_"
                "top8_selected_before_live_origin28_response_measurement"
            ),
            "rank_coverage_selection": (
                "fixed_balanced_chain_and_long_range_rank_coverage"
            ),
            "singleton_control": (
                "same_origin_same_component_amplitude_same_shared_zero_"
                "reference"
            ),
            "gates": dict(self.gates),
            "gate_values_applied_without_rounding": True,
            "response_measurement_used_for_protocol_selection": False,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}


def default_mixed_mode_protocol() -> FrozenMixedModeProtocol:
    return FrozenMixedModeProtocol(
        origin=ORIGIN,
        sequence_length=SEQUENCE_LENGTH,
        modal_rank=MODAL_RANK,
        max_lag=MAX_LAG,
        radii=RADII,
        sign_rows=SIGN_ROWS,
        pair_families=PAIR_FAMILIES,
        unique_modes=UNIQUE_MODES,
        gates=GATES,
    )


def _plan_hash(plan: object) -> str:
    return _require_sha256(
        getattr(plan, "artifact_sha256", None),
        label="conditional generator plan",
    )


def _plan_scales(plan: object) -> Tensor:
    return _canonical_tensor(
        getattr(plan, "source_scales", None),
        label="conditional generator source scales",
        ndim=1,
    )


def _prepare_runtime(
    plan: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    method = getattr(plan, "prepare", None)
    if not callable(method):
        raise TypeError("frozen conditional plan has no prepare method")
    runtime = method(device=device, dtype=dtype)
    if not isinstance(runtime, nn.Module):
        raise TypeError("prepared conditional runtime must be an nn.Module")
    runtime.eval()
    runtime.requires_grad_(False)
    return runtime


def _validate_stress_modes_from_quadratic_leverage(
    candidate: Gemma3ConditionalSpectralCandidate,
    protocol: FrozenMixedModeProtocol,
) -> dict[str, object]:
    """Authenticate the stress-mode set without reading a live response."""

    basis = _canonical_tensor(
        getattr(candidate.quadratic_plan, "source_basis", None),
        label="quadratic source basis",
        ndim=2,
    )
    if basis.shape[0] != protocol.modal_rank:
        raise ValueError("quadratic source basis width differs from protocol")
    leverage = basis.square().sum(dim=1)
    descending = tuple(
        sorted(
            range(protocol.modal_rank),
            key=lambda mode: (-float(leverage[mode]), mode),
        )
    )
    expected_top8 = tuple(descending[:8])
    stress_pairs = protocol.pair_families[0][1]
    stress_modes = tuple(sorted({mode for pair in stress_pairs for mode in pair}))
    if set(stress_modes) != set(expected_top8):
        raise ValueError(
            "stress panel is not the quadratic source-basis leverage top-8"
        )
    return {
        "selection_statistic": (
            "quadratic_source_basis_row_leverage=sum_r_Uq_i_r_squared"
        ),
        "top8_descending_leverage_order": expected_top8,
        "stress_ring_mode_set_sorted": stress_modes,
        "set_equality_verified_before_live_response_measurement": True,
        "stress_pair_order_is_exact_frozen_ring_not_score_order": True,
        "quadratic_source_basis_sha256": _tensor_sha256(basis),
        "top8_leverage_values": tuple(
            float(leverage[mode]) for mode in expected_top8
        ),
    }


def _runtime_response(
    runtime: nn.Module,
    source_modes: Tensor,
    *,
    logical_positions: Tensor,
    valid_mask: Tensor,
    source_mask: Tensor,
    origin: int,
    lag_count: int,
) -> Tensor:
    with torch.no_grad():
        value = runtime(
            source_modes,
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
    if not isinstance(value, Tensor):
        raise TypeError("prepared conditional runtime did not return a Tensor")
    return _canonical_tensor(
        value[:, origin : origin + lag_count],
        label="prepared conditional response",
        ndim=3,
    )


def _source_row(
    *,
    sequence_length: int,
    modal_rank: int,
    origin: int,
    device: torch.device,
    dtype: torch.dtype,
    components: Sequence[tuple[int, float]],
) -> Tensor:
    result = torch.zeros(
        (1, sequence_length, modal_rank),
        device=device,
        dtype=dtype,
    )
    for mode, amplitude in components:
        result[0, origin, mode] = amplitude
    return result


def _response_slice(
    structural_map: Callable[[Tensor], Tensor],
    source: Tensor,
    *,
    origin: int,
    lag_count: int,
) -> Tensor:
    value = structural_map(source)
    if (
        not isinstance(value, Tensor)
        or value.ndim != 3
        or value.shape[0] != 1
        or value.shape[1] < origin + lag_count
    ):
        raise ValueError("fixed-reference structural map response ABI drifted")
    return _canonical_tensor(
        value[:, origin : origin + lag_count],
        label="fixed-reference mixed response",
        ndim=3,
    )


def measure_mixed_mode_responses(
    structural_map: Callable[[Tensor], Tensor],
    *,
    candidate: Gemma3ConditionalSpectralCandidate,
    source_sigmas: Tensor,
    logical_positions: Tensor,
    valid_mask: Tensor,
    protocol: FrozenMixedModeProtocol,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Measure the frozen panel; this function exposes no fitting operation."""

    if not callable(structural_map):
        raise TypeError("structural_map must be callable")
    if not isinstance(candidate, Gemma3ConditionalSpectralCandidate):
        raise TypeError("candidate must be a frozen Gemma candidate")
    if not isinstance(protocol, FrozenMixedModeProtocol):
        raise TypeError("protocol must be frozen before measurement")
    if protocol.artifact_sha256 != default_mixed_mode_protocol().artifact_sha256:
        raise ValueError("measurement protocol hash is not the frozen manifest")
    stress_selection = _validate_stress_modes_from_quadratic_leverage(
        candidate,
        protocol,
    )
    if (
        not isinstance(logical_positions, Tensor)
        or logical_positions.shape != (1, protocol.sequence_length)
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or not isinstance(valid_mask, Tensor)
        or valid_mask.shape != logical_positions.shape
        or valid_mask.dtype != torch.bool
        or logical_positions.device != valid_mask.device
        or not bool(valid_mask.all())
    ):
        raise ValueError("mixed-mode logical grid differs from the frozen ABI")
    expected_positions = torch.arange(
        protocol.sequence_length,
        device=logical_positions.device,
        dtype=logical_positions.dtype,
    ).unsqueeze(0)
    if not torch.equal(logical_positions, expected_positions):
        raise ValueError("mixed-mode logical positions must be contiguous")
    sigma = _canonical_tensor(
        source_sigmas,
        label="source modal sigmas",
        ndim=1,
    )
    if sigma.numel() != protocol.modal_rank or bool((sigma <= 0.0).any()):
        raise ValueError("source modal sigmas differ from the frozen rank")
    for plan in (candidate.linear_plan, candidate.quadratic_plan):
        if not torch.equal(_plan_scales(plan), sigma):
            raise ValueError("candidate source scales differ from live reference")

    device = logical_positions.device
    dtype = next(
        (
            parameter.dtype
            for parameter in candidate_runtime_parameters(candidate)
        ),
        torch.float32,
    )
    # Candidate plans are canonical CPU float64 objects; the live structural
    # map ABI determines the actual model dtype.  A zero call is also the
    # shared-baseline sentinel, so use its input dtype as the runtime dtype.
    source_mask = torch.zeros_like(valid_mask)
    source_mask[:, protocol.origin] = True
    zero_source = _source_row(
        sequence_length=protocol.sequence_length,
        modal_rank=protocol.modal_rank,
        origin=protocol.origin,
        device=device,
        dtype=dtype,
        components=(),
    )
    try:
        zero_response = _response_slice(
            structural_map,
            zero_source,
            origin=protocol.origin,
            lag_count=protocol.max_lag + 1,
        )
    except ValueError as error:
        # The common case is a live model dtype other than float32.  Structural
        # maps are permitted to advertise it on a bound ``runtime_dtype``
        # attribute without exposing model state.
        runtime_dtype = getattr(structural_map, "runtime_dtype", None)
        if runtime_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise error
        dtype = runtime_dtype
        zero_source = zero_source.to(dtype=dtype)
        zero_response = _response_slice(
            structural_map,
            zero_source,
            origin=protocol.origin,
            lag_count=protocol.max_lag + 1,
        )

    linear_runtime = _prepare_runtime(
        candidate.linear_plan,
        device=device,
        dtype=dtype,
    )
    quadratic_runtime = _prepare_runtime(
        candidate.quadratic_plan,
        device=device,
        dtype=dtype,
    )
    execution_accounting: dict[str, object] = {}
    for label, runtime in (
        ("linear", linear_runtime),
        ("quadratic", quadratic_runtime),
    ):
        method = getattr(runtime, "execution_accounting", None)
        if not callable(method):
            raise TypeError("prepared runtime lacks execution accounting")
        accounting = method(
            logical_positions=logical_positions,
            valid_mask=valid_mask,
            source_mask=source_mask,
        )
        metadata = getattr(accounting, "metadata", None)
        if not callable(metadata):
            raise TypeError("prepared execution accounting lacks metadata")
        execution_accounting[label] = dict(metadata())
    sigma_runtime = sigma.to(device=device, dtype=dtype)
    component_scale = {
        radius: sigma_runtime * (radius / math.sqrt(2.0))
        for radius in protocol.radii
    }

    sentinel_source = _source_row(
        sequence_length=protocol.sequence_length,
        modal_rank=protocol.modal_rank,
        origin=protocol.origin,
        device=device,
        dtype=dtype,
        components=((protocol.unique_modes[0], float(component_scale[1.0][0])),),
    )
    repeat_a = _response_slice(
        structural_map,
        sentinel_source,
        origin=protocol.origin,
        lag_count=protocol.max_lag + 1,
    ) - zero_response

    singleton_truth: list[Tensor] = []
    singleton_linear: list[Tensor] = []
    singleton_quadratic: list[Tensor] = []
    for radius in protocol.radii:
        truth_by_mode: list[Tensor] = []
        linear_by_mode: list[Tensor] = []
        quadratic_by_mode: list[Tensor] = []
        for mode in protocol.unique_modes:
            truth_by_sign: list[Tensor] = []
            linear_by_sign: list[Tensor] = []
            quadratic_by_sign: list[Tensor] = []
            for sign in (1, -1):
                source = _source_row(
                    sequence_length=protocol.sequence_length,
                    modal_rank=protocol.modal_rank,
                    origin=protocol.origin,
                    device=device,
                    dtype=dtype,
                    components=(
                        (mode, sign * float(component_scale[radius][mode])),
                    ),
                )
                truth_by_sign.append(
                    _response_slice(
                        structural_map,
                        source,
                        origin=protocol.origin,
                        lag_count=protocol.max_lag + 1,
                    )
                    - zero_response
                )
                linear_by_sign.append(
                    _runtime_response(
                        linear_runtime,
                        source,
                        logical_positions=logical_positions,
                        valid_mask=valid_mask,
                        source_mask=source_mask,
                        origin=protocol.origin,
                        lag_count=protocol.max_lag + 1,
                    )
                )
                quadratic_by_sign.append(
                    _runtime_response(
                        quadratic_runtime,
                        source,
                        logical_positions=logical_positions,
                        valid_mask=valid_mask,
                        source_mask=source_mask,
                        origin=protocol.origin,
                        lag_count=protocol.max_lag + 1,
                    )
                )
            truth_by_mode.append(torch.stack(truth_by_sign))
            linear_by_mode.append(torch.stack(linear_by_sign))
            quadratic_by_mode.append(torch.stack(quadratic_by_sign))
        singleton_truth.append(torch.stack(truth_by_mode))
        singleton_linear.append(torch.stack(linear_by_mode))
        singleton_quadratic.append(torch.stack(quadratic_by_mode))

    mixed_truth: list[Tensor] = []
    mixed_linear: list[Tensor] = []
    mixed_quadratic: list[Tensor] = []
    for radius in protocol.radii:
        truth_by_pair: list[Tensor] = []
        linear_by_pair: list[Tensor] = []
        quadratic_by_pair: list[Tensor] = []
        for left, right in protocol.pairs:
            truth_by_sign: list[Tensor] = []
            linear_by_sign: list[Tensor] = []
            quadratic_by_sign: list[Tensor] = []
            for _label, left_sign, right_sign in protocol.sign_rows:
                source = _source_row(
                    sequence_length=protocol.sequence_length,
                    modal_rank=protocol.modal_rank,
                    origin=protocol.origin,
                    device=device,
                    dtype=dtype,
                    components=(
                        (
                            left,
                            left_sign
                            * float(component_scale[radius][left]),
                        ),
                        (
                            right,
                            right_sign
                            * float(component_scale[radius][right]),
                        ),
                    ),
                )
                truth_by_sign.append(
                    _response_slice(
                        structural_map,
                        source,
                        origin=protocol.origin,
                        lag_count=protocol.max_lag + 1,
                    )
                    - zero_response
                )
                linear_by_sign.append(
                    _runtime_response(
                        linear_runtime,
                        source,
                        logical_positions=logical_positions,
                        valid_mask=valid_mask,
                        source_mask=source_mask,
                        origin=protocol.origin,
                        lag_count=protocol.max_lag + 1,
                    )
                )
                quadratic_by_sign.append(
                    _runtime_response(
                        quadratic_runtime,
                        source,
                        logical_positions=logical_positions,
                        valid_mask=valid_mask,
                        source_mask=source_mask,
                        origin=protocol.origin,
                        lag_count=protocol.max_lag + 1,
                    )
                )
            truth_by_pair.append(torch.stack(truth_by_sign))
            linear_by_pair.append(torch.stack(linear_by_sign))
            quadratic_by_pair.append(torch.stack(quadratic_by_sign))
        mixed_truth.append(torch.stack(truth_by_pair))
        mixed_linear.append(torch.stack(linear_by_pair))
        mixed_quadratic.append(torch.stack(quadratic_by_pair))

    # The second sentinel is intentionally after all 256 panel responses so
    # the pair measures whole-run drift, not merely consecutive determinism.
    repeat_b = _response_slice(
        structural_map,
        sentinel_source,
        origin=protocol.origin,
        lag_count=protocol.max_lag + 1,
    ) - zero_response
    tensors = {
        "mixed_truth": torch.stack(mixed_truth),
        "mixed_linear_prediction": torch.stack(mixed_linear),
        "mixed_quadratic_prediction": torch.stack(mixed_quadratic),
        "singleton_truth": torch.stack(singleton_truth),
        "singleton_linear_prediction": torch.stack(singleton_linear),
        "singleton_quadratic_prediction": torch.stack(singleton_quadratic),
        "zero_sentinel": zero_response,
        "repeat_sentinel_first": repeat_a,
        "repeat_sentinel_second": repeat_b,
    }
    tensors = {
        name: _canonical_tensor(
            value,
            label=name,
            ndim=value.ndim,
        )
        for name, value in tensors.items()
    }
    call_count = 1 + 2 + len(RADII) * (
        len(UNIQUE_MODES) * 2 + len(protocol.pairs) * len(SIGN_ROWS)
    )
    sentinels = {
        "stress_selection": stress_selection,
        "shared_zero_reference_used_for_every_response": True,
        "shared_zero_response_explicitly_subtracted_from_every_truth_row": True,
        "zero_response_l2": float(torch.linalg.vector_norm(zero_response)),
        "zero_response_max_abs": float(zero_response.abs().max()),
        "repeat_response_l2_difference": float(
            torch.linalg.vector_norm(repeat_a - repeat_b)
        ),
        "repeat_response_max_abs_difference": float(
            (repeat_a - repeat_b).abs().max()
        ),
        "empirical_noise_floor_l2": max(
            float(torch.linalg.vector_norm(zero_response)),
            float(torch.linalg.vector_norm(repeat_a - repeat_b)),
            torch.finfo(torch.float64).eps
            * math.sqrt(float(repeat_a.numel())),
        ),
        "repeat_sentinel_brackets_all_panel_measurements": True,
        "fixed_reference_function_evaluation_count": call_count,
        "baseline_prefix_evaluation_count": 1,
        "l4_attention_prefix_evaluation_count": call_count + 1,
        "prepared_runtime_execution_accounting_per_panel_call": (
            execution_accounting
        ),
    }
    return tensors, sentinels


def candidate_runtime_parameters(
    _candidate: Gemma3ConditionalSpectralCandidate,
) -> tuple[Tensor, ...]:
    """Compatibility seam: plans contain coefficients, not nn.Parameters."""

    return ()


def _walsh_c11(values: Tensor) -> Tensor:
    # values: [pair, sign, ...] in frozen (++,+-,-+,--) order.
    weights = values.new_tensor((1.0, -1.0, -1.0, 1.0))
    return torch.einsum("ps...,s->p...", values, weights) / 4.0


def _signed_c11(c11: Tensor) -> Tensor:
    signs = c11.new_tensor((1.0, -1.0, -1.0, 1.0))
    return c11.unsqueeze(1) * signs.view(1, 4, *([1] * (c11.ndim - 1)))


def _interaction_tensors(
    tensors: Mapping[str, Tensor],
    *,
    protocol: FrozenMixedModeProtocol,
) -> dict[str, Tensor]:
    mode_ordinal = {
        mode: ordinal for ordinal, mode in enumerate(protocol.unique_modes)
    }
    singleton_truth = tensors["singleton_truth"]
    singleton_corrected = (
        tensors["singleton_linear_prediction"]
        + tensors["singleton_quadratic_prediction"]
    )
    mixed_corrected = (
        tensors["mixed_linear_prediction"]
        + tensors["mixed_quadratic_prediction"]
    )
    truth_interactions: list[Tensor] = []
    prediction_interactions: list[Tensor] = []
    additive_predictions: list[Tensor] = []
    for radius_ordinal in range(len(protocol.radii)):
        truth_by_pair: list[Tensor] = []
        predicted_by_pair: list[Tensor] = []
        additive_by_pair: list[Tensor] = []
        for pair_ordinal, (left, right) in enumerate(protocol.pairs):
            left_controls = singleton_truth[
                radius_ordinal, mode_ordinal[left]
            ]
            right_controls = singleton_truth[
                radius_ordinal, mode_ordinal[right]
            ]
            left_prediction = singleton_corrected[
                radius_ordinal, mode_ordinal[left]
            ]
            right_prediction = singleton_corrected[
                radius_ordinal, mode_ordinal[right]
            ]
            truth_axis: list[Tensor] = []
            predicted_axis: list[Tensor] = []
            for _label, left_sign, right_sign in protocol.sign_rows:
                left_ordinal = 0 if left_sign == 1 else 1
                right_ordinal = 0 if right_sign == 1 else 1
                truth_axis.append(
                    left_controls[left_ordinal]
                    + right_controls[right_ordinal]
                )
                predicted_axis.append(
                    left_prediction[left_ordinal]
                    + right_prediction[right_ordinal]
                )
            truth_axis_tensor = torch.stack(truth_axis)
            predicted_axis_tensor = torch.stack(predicted_axis)
            truth_by_pair.append(
                tensors["mixed_truth"][radius_ordinal, pair_ordinal]
                - truth_axis_tensor
            )
            predicted_by_pair.append(
                mixed_corrected[radius_ordinal, pair_ordinal]
                - predicted_axis_tensor
            )
            additive_by_pair.append(predicted_axis_tensor)
        truth_interactions.append(torch.stack(truth_by_pair))
        prediction_interactions.append(torch.stack(predicted_by_pair))
        additive_predictions.append(torch.stack(additive_by_pair))
    return {
        "truth_interaction": torch.stack(truth_interactions),
        "candidate_interaction": torch.stack(prediction_interactions),
        "candidate_additive_prediction": torch.stack(additive_predictions),
    }


def _pool_metrics(
    *,
    truth: Tensor,
    corrected: Tensor,
    additive_prediction: Tensor,
    interaction: Tensor,
    candidate_interaction: Tensor,
    empirical_noise_floor_l2: float,
) -> dict[str, object]:
    c11 = _walsh_c11(interaction)
    c11_signed = _signed_c11(c11)
    full_oracle = additive_prediction + interaction
    odd_odd_oracle = additive_prediction + c11_signed
    truth_energy = float(truth.square().sum())
    interaction_energy = float(interaction.square().sum())
    interaction_walsh_energy = max(
        interaction_energy / int(interaction.shape[1]),
        torch.finfo(torch.float64).eps,
    )
    response_walsh_energy = max(
        truth_energy / int(truth.shape[1]),
        torch.finfo(torch.float64).eps,
    )
    candidate_interaction_relative = _relative_norm(
        candidate_interaction,
        corrected,
    )
    return {
        "corrected_relative_error": _relative_error(corrected, truth),
        "corrected_cosine": _cosine(corrected, truth),
        "nonadditivity_relative_norm": _relative_norm(interaction, truth),
        "nonadditivity_energy_fraction": (
            interaction_energy
            / max(truth_energy, torch.finfo(torch.float64).eps)
        ),
        "interaction_l2": math.sqrt(interaction_energy),
        "response_l2": math.sqrt(truth_energy),
        "response_rms": math.sqrt(truth_energy / float(truth.numel())),
        "interaction_reliable_above_empirical_floor": (
            math.sqrt(interaction_energy)
            > 10.0 * empirical_noise_floor_l2
        ),
        "interaction_parity_c11_energy_fraction": (
            float(c11.square().sum()) / interaction_walsh_energy
        ),
        "c11_full_response_energy_fraction": (
            float(c11.square().sum()) / response_walsh_energy
        ),
        "full_interaction_oracle_relative_error": _relative_error(
            full_oracle,
            truth,
        ),
        "full_interaction_oracle_gain": _fractional_error_reduction(
            corrected,
            full_oracle,
            truth,
        ),
        "odd_odd_oracle_relative_error": _relative_error(
            odd_odd_oracle,
            truth,
        ),
        "odd_odd_oracle_gain": _fractional_error_reduction(
            corrected,
            odd_odd_oracle,
            truth,
        ),
        "candidate_nonadditivity_relative_norm": (
            candidate_interaction_relative
        ),
        "candidate_c11_relative_norm": _relative_norm(
            _walsh_c11(candidate_interaction),
            corrected,
        ),
    }


def _radius_metrics(
    *,
    radius_ordinal: int,
    tensors: Mapping[str, Tensor],
    derived: Mapping[str, Tensor],
    protocol: FrozenMixedModeProtocol,
    empirical_noise_floor_l2: float,
) -> dict[str, object]:
    truth = tensors["mixed_truth"][radius_ordinal]
    linear = tensors["mixed_linear_prediction"][radius_ordinal]
    corrected = linear + tensors["mixed_quadratic_prediction"][radius_ordinal]
    interaction = derived["truth_interaction"][radius_ordinal]
    candidate_interaction = derived["candidate_interaction"][radius_ordinal]
    additive = derived["candidate_additive_prediction"][radius_ordinal]
    pooled = _pool_metrics(
        truth=truth,
        corrected=corrected,
        additive_prediction=additive,
        interaction=interaction,
        candidate_interaction=candidate_interaction,
        empirical_noise_floor_l2=empirical_noise_floor_l2,
    )
    pooled["linear_only_relative_error"] = _relative_error(linear, truth)
    pooled["linear_only_cosine"] = _cosine(linear, truth)

    pair_rows: list[dict[str, object]] = []
    for ordinal, ((left, right), family) in enumerate(
        zip(protocol.pairs, protocol.family_by_pair, strict=True)
    ):
        row = _pool_metrics(
            truth=truth[ordinal : ordinal + 1],
            corrected=corrected[ordinal : ordinal + 1],
            additive_prediction=additive[ordinal : ordinal + 1],
            interaction=interaction[ordinal : ordinal + 1],
            candidate_interaction=candidate_interaction[
                ordinal : ordinal + 1
            ],
            empirical_noise_floor_l2=empirical_noise_floor_l2,
        )
        row.update(
            {
                "pair_ordinal": ordinal,
                "left_mode": left,
                "right_mode": right,
                "family": family,
            }
        )
        pair_rows.append(row)

    family_rows: list[dict[str, object]] = []
    for family, family_pairs in protocol.pair_families:
        ordinals = tuple(
            ordinal
            for ordinal, candidate_family in enumerate(
                protocol.family_by_pair
            )
            if candidate_family == family
        )
        index = torch.tensor(ordinals, dtype=torch.long)
        row = _pool_metrics(
            truth=truth.index_select(0, index),
            corrected=corrected.index_select(0, index),
            additive_prediction=additive.index_select(0, index),
            interaction=interaction.index_select(0, index),
            candidate_interaction=candidate_interaction.index_select(0, index),
            empirical_noise_floor_l2=empirical_noise_floor_l2,
        )
        row.update(
            {
                "family": family,
                "pair_count": len(family_pairs),
                "macro_corrected_relative_error": sum(
                    float(pair_rows[ordinal]["corrected_relative_error"])
                    for ordinal in ordinals
                )
                / len(ordinals),
                "macro_nonadditivity_relative_norm": sum(
                    float(
                        pair_rows[ordinal][
                            "nonadditivity_relative_norm"
                        ]
                    )
                    for ordinal in ordinals
                )
                / len(ordinals),
            }
        )
        family_rows.append(row)
    reliable_pair_values = tuple(
        float(row["nonadditivity_relative_norm"])
        for row in pair_rows
        if row["interaction_reliable_above_empirical_floor"] is True
    )
    pair_response_rms = tuple(
        float(row["response_rms"]) for row in pair_rows
    )
    return {
        "radius": protocol.radii[radius_ordinal],
        "pooled": pooled,
        "families": tuple(family_rows),
        "pairs": tuple(pair_rows),
        "maximum_reliable_pair_nonadditivity_relative_norm": (
            max(reliable_pair_values) if reliable_pair_values else 0.0
        ),
        "reliable_pair_count": len(reliable_pair_values),
        "panel_median_pair_response_rms": float(
            torch.tensor(pair_response_rms, dtype=torch.float64).median()
        ),
    }


def _scaling_metrics(
    interaction_low: Tensor,
    interaction_high: Tensor,
) -> dict[str, object]:
    low_c11 = _walsh_c11(interaction_low)
    high_c11 = _walsh_c11(interaction_high)
    quadratic_scale = (RADII[1] / RADII[0]) ** 2
    return {
        "low_radius": RADII[0],
        "high_radius": RADII[1],
        "quadratic_scale_factor": quadratic_scale,
        "full_interaction_quadratic_scaling_relative_error": _relative_error(
            interaction_low * quadratic_scale,
            interaction_high,
        ),
        "full_interaction_quadratic_scaling_cosine": _cosine(
            interaction_low,
            interaction_high,
        ),
        "c11_hessian_scaling_relative_error": _relative_error(
            low_c11 * quadratic_scale,
            high_c11,
        ),
        "c11_hessian_scaling_cosine": _cosine(low_c11, high_c11),
    }


def evaluate_mixed_mode_falsification(
    tensors: Mapping[str, Tensor],
    *,
    sentinels: Mapping[str, object],
    protocol: FrozenMixedModeProtocol,
) -> tuple[dict[str, object], dict[str, Tensor]]:
    """Compute preregistered metrics without modifying the candidate."""

    expected = {
        "mixed_truth": (2, 24, 4, 1, 32, 64),
        "mixed_linear_prediction": (2, 24, 4, 1, 32, 64),
        "mixed_quadratic_prediction": (2, 24, 4, 1, 32, 64),
        "singleton_truth": (2, 16, 2, 1, 32, 64),
        "singleton_linear_prediction": (2, 16, 2, 1, 32, 64),
        "singleton_quadratic_prediction": (2, 16, 2, 1, 32, 64),
        "zero_sentinel": (1, 32, 64),
        "repeat_sentinel_first": (1, 32, 64),
        "repeat_sentinel_second": (1, 32, 64),
    }
    canonical: dict[str, Tensor] = {}
    if set(tensors) != set(expected):
        raise ValueError("mixed-mode tensor fields differ from frozen protocol")
    for name, shape in expected.items():
        value = _canonical_tensor(
            tensors[name],
            label=name,
            ndim=len(shape),
        )
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} shape differs from frozen protocol")
        canonical[name] = value
    floor = sentinels.get("empirical_noise_floor_l2")
    if (
        isinstance(floor, bool)
        or not isinstance(floor, (int, float))
        or not math.isfinite(float(floor))
        or float(floor) <= 0.0
    ):
        raise ValueError("empirical noise floor is invalid")

    derived = _interaction_tensors(canonical, protocol=protocol)
    operating_ordinal = protocol.radii.index(1.0)
    operating_corrected = (
        canonical["mixed_linear_prediction"][operating_ordinal]
        + canonical["mixed_quadratic_prediction"][operating_ordinal]
    )
    candidate_interaction_relative = _relative_norm(
        derived["candidate_interaction"][operating_ordinal],
        operating_corrected,
    )
    candidate_c11_relative = _relative_norm(
        _walsh_c11(derived["candidate_interaction"][operating_ordinal]),
        operating_corrected,
    )
    if (
        candidate_interaction_relative > 1e-6
        or candidate_c11_relative > 1e-6
    ):
        raise ValueError(
            "frozen diagonal candidate is not numerically additive; "
            "interaction-oracle gain would be ill-defined"
        )
    by_radius = tuple(
        _radius_metrics(
            radius_ordinal=ordinal,
            tensors=canonical,
            derived=derived,
            protocol=protocol,
            empirical_noise_floor_l2=float(floor),
        )
        for ordinal in range(len(protocol.radii))
    )
    operating = by_radius[protocol.radii.index(1.0)]
    pooled = operating["pooled"]
    families = operating["families"]
    maximum_family_nonadditivity = max(
        float(row["nonadditivity_relative_norm"]) for row in families
    )
    maximum_family_macro_error = max(
        float(row["macro_corrected_relative_error"]) for row in families
    )
    support_checks = {
        "corrected_global_relative_error": (
            float(pooled["corrected_relative_error"])
            <= GATES["maximum_corrected_global_relative_error"]
        ),
        "corrected_global_cosine": (
            float(pooled["corrected_cosine"])
            >= GATES["minimum_corrected_global_cosine"]
        ),
        "pooled_nonadditivity": (
            float(pooled["nonadditivity_relative_norm"])
            < GATES["maximum_pooled_nonadditivity_relative_norm"]
        ),
        "family_nonadditivity": (
            maximum_family_nonadditivity
            < GATES["maximum_family_nonadditivity_relative_norm"]
        ),
        "reliable_pair_nonadditivity": (
            float(
                operating[
                    "maximum_reliable_pair_nonadditivity_relative_norm"
                ]
            )
            < GATES[
                "maximum_reliable_pair_nonadditivity_relative_norm"
            ]
        ),
        "full_interaction_oracle_gain": (
            float(pooled["full_interaction_oracle_gain"])
            < GATES["maximum_full_interaction_oracle_gain"]
        ),
    }
    diagnostic_checks = {
        "c11_full_response_energy_fraction": (
            float(pooled["c11_full_response_energy_fraction"])
            <= GATES["c11_total_energy_fraction_diagnostic_maximum"]
        ),
        "candidate_interaction_numerically_zero": (
            float(pooled["candidate_nonadditivity_relative_norm"]) <= 1e-6
            and float(pooled["candidate_c11_relative_norm"]) <= 1e-6
        ),
    }
    material_checks = {
        "pooled_nonadditivity": (
            float(pooled["nonadditivity_relative_norm"])
            >= GATES["material_pooled_nonadditivity_relative_norm"]
        ),
        "family_nonadditivity": (
            maximum_family_nonadditivity
            >= GATES["material_family_nonadditivity_relative_norm"]
        ),
        "full_interaction_oracle_gain": (
            float(pooled["full_interaction_oracle_gain"])
            >= GATES["material_full_interaction_oracle_gain"]
        ),
        "reliable_pair_nonadditivity": any(
            (
                float(row["nonadditivity_relative_norm"])
                >= GATES["material_pair_nonadditivity_relative_norm"]
                and float(row["response_rms"])
                >= GATES[
                    "material_pair_minimum_panel_median_response_rms_fraction"
                ]
                * float(operating["panel_median_pair_response_rms"])
                and row["interaction_reliable_above_empirical_floor"] is True
            )
            for row in operating["pairs"]
        ),
    }
    if any(material_checks.values()):
        decision = "material_cross_interaction_failure"
    elif all(support_checks.values()):
        decision = "panel_supports_additive_diagonal_executor"
    else:
        decision = "inconclusive_between_support_and_material_failure"
    interaction_magnitude_material = any(
        material_checks[name]
        for name in (
            "pooled_nonadditivity",
            "family_nonadditivity",
            "reliable_pair_nonadditivity",
        )
    )
    scaling = _scaling_metrics(
        derived["truth_interaction"][0],
        derived["truth_interaction"][1],
    )
    bilinear_pi11_passes = (
        float(pooled["interaction_parity_c11_energy_fraction"])
        >= GATES[
            "bilinear_branch_minimum_interaction_parity_c11_energy_fraction"
        ]
    )
    bilinear_scaling_cosine_passes = (
        float(scaling["c11_hessian_scaling_cosine"])
        >= GATES["bilinear_branch_minimum_c11_hessian_scaling_cosine"]
    )
    bilinear_scaling_error_passes = (
        float(scaling["c11_hessian_scaling_relative_error"])
        <= GATES[
            "bilinear_branch_maximum_c11_hessian_scaling_relative_error"
        ]
    )
    metrics = {
        "by_radius": by_radius,
        "rho_0_5_to_1_0_scaling": scaling,
        "operating_radius": 1.0,
        "operating_maximum_family_nonadditivity_relative_norm": (
            maximum_family_nonadditivity
        ),
        "operating_maximum_family_macro_corrected_relative_error": (
            maximum_family_macro_error
        ),
        "operating_family_macro_corrected_relative_error_is_descriptive_only": (
            True
        ),
        "bilinear_branch_suitability": {
            "interaction_magnitude_is_material": (
                interaction_magnitude_material
            ),
            "oracle_gain_is_not_sufficient_without_material_magnitude": True,
            "interaction_parity_c11_energy_fraction": float(
                pooled["interaction_parity_c11_energy_fraction"]
            ),
            "minimum_required_interaction_parity_c11_energy_fraction": GATES[
                "bilinear_branch_minimum_interaction_parity_c11_energy_fraction"
            ],
            "interaction_parity_c11_gate_passes": bilinear_pi11_passes,
            "c11_hessian_scaling_cosine": float(
                scaling["c11_hessian_scaling_cosine"]
            ),
            "minimum_required_c11_hessian_scaling_cosine": GATES[
                "bilinear_branch_minimum_c11_hessian_scaling_cosine"
            ],
            "c11_hessian_scaling_cosine_gate_passes": (
                bilinear_scaling_cosine_passes
            ),
            "c11_hessian_scaling_relative_error": float(
                scaling["c11_hessian_scaling_relative_error"]
            ),
            "maximum_c11_hessian_scaling_relative_error": GATES[
                "bilinear_branch_maximum_c11_hessian_scaling_relative_error"
            ],
            "c11_hessian_scaling_relative_error_gate_passes": (
                bilinear_scaling_error_passes
            ),
            "suitable": (
                interaction_magnitude_material
                and bilinear_pi11_passes
                and bilinear_scaling_cosine_passes
                and bilinear_scaling_error_passes
            ),
            "interpretation": (
                "only material predominantly odd_odd interaction supports_"
                "a_bilinear_branch; small pure_c11 remains harmless"
            ),
        },
        "support_checks": support_checks,
        "diagnostic_checks": diagnostic_checks,
        "material_failure_checks": material_checks,
        "decision": decision,
        "gate_values_applied_without_rounding": True,
    }
    return metrics, {
        **derived,
        "truth_c11": torch.stack(
            (
                _walsh_c11(derived["truth_interaction"][0]),
                _walsh_c11(derived["truth_interaction"][1]),
            )
        ),
    }


def _claim_boundaries() -> dict[str, object]:
    return {
        "scope": "prompt_free_fixed_reference_origin28_pair_panel_only",
        "panel_can_falsify_additivity": True,
        "panel_can_prove_global_additivity": False,
        "singleton_subtracted_full_interaction_measured": True,
        "c11_is_complete_interaction_measure": False,
        "candidate_was_refit_or_modified": False,
        "candidate_has_cross_mode_terms": False,
        "prompt_conditioned_reference_provider_compiled": False,
        "full_gemma_block_replacement_authorized": False,
        "heldout_prompt_fidelity_claim": False,
        "nll_claim": False,
        "task_accuracy_claim": False,
        "model_compression_claim": False,
        "runtime_speed_or_latency_claim": False,
        "cached_decode_claim": False,
    }


def _generic_interaction_crosscheck(
    tensors: Mapping[str, Tensor],
    derived: Mapping[str, Tensor],
    *,
    candidate: Gemma3ConditionalSpectralCandidate,
    protocol: FrozenMixedModeProtocol,
) -> tuple[object, dict[str, object]]:
    """Build the generic authenticated artifact from the measured panel."""

    responses = (
        tensors["mixed_truth"].squeeze(3).permute(1, 0, 2, 3, 4)
    )
    predictions = (
        tensors["mixed_linear_prediction"]
        + tensors["mixed_quadratic_prediction"]
    ).squeeze(3).permute(1, 0, 2, 3, 4)
    singleton = (
        tensors["singleton_truth"].squeeze(3).permute(1, 0, 2, 3, 4)
    )
    sigma = _plan_scales(candidate.linear_plan)
    pair_amplitudes = torch.tensor(
        [
            (
                float(sigma[left]) / math.sqrt(2.0),
                float(sigma[right]) / math.sqrt(2.0),
            )
            for left, right in protocol.pairs
        ],
        dtype=torch.float64,
    )
    generic = build_mixed_modal_interaction_artifact(
        responses,
        predictions,
        pair_labels=tuple(
            f"mode_{left:02d}_mode_{right:02d}"
            for left, right in protocol.pairs
        ),
        pair_indices=protocol.pairs,
        pair_families=protocol.family_by_pair,
        pair_amplitudes=pair_amplitudes,
        scales=torch.tensor(protocol.radii, dtype=torch.float64),
        origin=protocol.origin,
        origin_binding_sha256=protocol.artifact_sha256,
        candidate_binding_sha256=candidate.artifact_sha256,
        shared_baseline_sha256=_tensor_sha256(tensors["zero_sentinel"]),
        singleton_responses=singleton,
        singleton_mode_indices=protocol.unique_modes,
    )
    generic_nonadditivity = pairwise_all_nonadditivity(generic)
    expected = (
        derived["truth_interaction"]
        .squeeze(3)
        .permute(1, 0, 2, 3, 4)
    )
    if not torch.equal(generic_nonadditivity, expected):
        difference = _relative_error(generic_nonadditivity, expected)
        if difference > 1e-12:
            raise RuntimeError(
                "generic and runner singleton interaction math differs"
            )
    analysis = generic.analyze()
    metadata = analysis.metadata()
    parity = analysis.per_scale_interaction_parity_energy
    if len(parity) != len(protocol.radii):
        raise RuntimeError("generic interaction scale analysis drifted")
    return generic, {
        "artifact": generic.metadata(),
        "analysis": metadata,
        "crosscheck": {
            "singleton_nonadditivity_matches_runner": True,
            "relative_error": _relative_error(
                generic_nonadditivity,
                expected,
            ),
            "operating_interaction_parity_c11_energy_fraction": (
                parity[protocol.radii.index(1.0)].c11_energy_fraction
            ),
            "c11_field_is_interaction_parity_not_full_response": True,
        },
    }


_SAFETY = {
    "contains_source_model_state_dict": False,
    "contains_compiled_candidate_plan_tensors": False,
    "contains_tokenizer": False,
    "contains_prompt_text": False,
    "contains_token_ids": False,
    "contains_prompt_activation_rows": False,
    "contains_score_gradient_rows": False,
    "contains_fixed_reference_mixed_response_tensors": True,
    "tensor_artifact_must_remain_outside_git": True,
    "json_report_is_source_safe": True,
}


def _tensor_manifest(tensors: Mapping[str, Tensor]) -> dict[str, object]:
    return {
        name: {
            "shape": tuple(int(width) for width in value.shape),
            "dtype": "float64",
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(tensors.items())
    }


def _validate_output(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("mixed-mode output must use a .pt suffix")
    report = destination.with_suffix(".json")
    if destination.exists() or report.exists():
        raise FileExistsError("refusing to overwrite mixed-mode output")
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
                    "mixed-mode tensor outputs in the worktree must stay "
                    "under an ignored local-runs directory"
                )
    return destination


def _stage_file(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish(
    artifact: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    report_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite mixed-mode output")
    tensor_stage = _stage_file(output)
    report_stage = _stage_file(report_path)
    published: list[Path] = []
    try:
        torch.save(dict(artifact), tensor_stage)
        tensor_digest = _file_sha256(tensor_stage)
        report = {
            **dict(report_payload),
            "artifact": {
                "tensor_file": str(output),
                "tensor_file_sha256": tensor_digest,
                "tensor_file_bytes": tensor_stage.stat().st_size,
                "report_file": str(report_path),
                "tensor_file_committable": False,
                "json_report_source_safe": True,
            },
        }
        report["report_sha256"] = _json_sha256(
            report,
            domain=_REPORT_DOMAIN,
        )
        with report_stage.open("w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.link(tensor_stage, output)
        published.append(output)
        os.link(report_stage, report_path)
        published.append(report_path)
        return report
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        tensor_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)


def _run_live_measurement(
    adapter: Gemma3CausalLMAdapter,
    reference: Any,
    candidate: Gemma3ConditionalSpectralCandidate,
    protocol: FrozenMixedModeProtocol,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    if adapter.module.training or any(
        parameter.requires_grad for parameter in adapter.module.parameters()
    ):
        raise ValueError("mixed-mode measurement requires frozen eval Gemma")
    layer3_spec = adapter.layer("layer.3")
    layer4_spec = adapter.layer("layer.4")
    layer3 = adapter.source_module(layer3_spec.id)
    layer4 = adapter.source_module(layer4_spec.id)
    pre_ff3 = getattr(layer3, "pre_feedforward_layernorm", None)
    post_ff3 = getattr(layer3, "post_feedforward_layernorm", None)
    if not isinstance(pre_ff3, nn.Module) or not isinstance(post_ff3, nn.Module):
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
    del placeholder
    preimage = fixed_reference.value.to(
        device=device,
        dtype=dtype,
    ).view(1, 1, -1).expand(1, protocol.sequence_length, -1)
    y3_mean = reference.y3_mean.to(
        device=device,
        dtype=dtype,
    ).view(1, 1, -1).expand(1, protocol.sequence_length, -1)
    p3 = reference.P3[:, : protocol.modal_rank].to(
        device=device,
        dtype=dtype,
    )
    r4 = reference.R4[: protocol.modal_rank].to(
        device=device,
        dtype=dtype,
    )
    segment4 = adapter.segment("layer.4")
    with torch.no_grad():
        hidden3_reference = preimage + post_ff3(y3_mean)
        baseline_x4 = adapter.run_attention_prefix(
            segment4,
            hidden3_reference,
            sequence,
        ).normalized_mlp_input.detach()

    call_count = 0

    def structural_map(source_modes: Tensor) -> Tensor:
        nonlocal call_count
        if (
            not isinstance(source_modes, Tensor)
            or source_modes.shape
            != (1, protocol.sequence_length, protocol.modal_rank)
            or source_modes.device != device
            or source_modes.dtype != dtype
            or not bool(torch.isfinite(source_modes).all())
        ):
            raise ValueError("source modes differ from fixed-reference ABI")
        call_count += 1
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
    sigma = reference.source_mode_standard_deviations(protocol.modal_rank)
    tensors, sentinels = measure_mixed_mode_responses(
        structural_map,
        candidate=candidate,
        source_sigmas=sigma,
        logical_positions=logical_positions,
        valid_mask=valid_mask,
        protocol=protocol,
    )
    if call_count != sentinels["fixed_reference_function_evaluation_count"]:
        raise RuntimeError("fixed-reference function accounting drifted")
    sentinels.update(
        {
            "canonical_reference": fixed_reference.metadata(),
            "baseline_x4_rms": float(
                baseline_x4.detach().float().square().mean().sqrt()
            ),
            "native_or_compiled_l3_mlp_body_executions": 0,
            "native_or_compiled_l4_mlp_body_executions": 0,
            "partial_analytic_live_model_macs": {
                "P3_decode_macs_per_function_evaluation": (
                    protocol.sequence_length * p3.numel()
                ),
                "R4_projection_macs_per_function_evaluation": (
                    protocol.sequence_length * r4.numel()
                ),
                "l4_attention_projection_weight_macs_per_prefix": (
                    protocol.sequence_length
                    * sum(
                        getattr(layer4.self_attn, name).weight.numel()
                        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                    )
                ),
                "counted_linear_macs_total_experiment": (
                    call_count
                    * (
                        protocol.sequence_length * p3.numel()
                        + protocol.sequence_length * r4.numel()
                        + protocol.sequence_length
                        * sum(
                            getattr(layer4.self_attn, name).weight.numel()
                            for name in (
                                "q_proj",
                                "k_proj",
                                "v_proj",
                                "o_proj",
                            )
                        )
                    )
                    + protocol.sequence_length
                    * sum(
                        getattr(layer4.self_attn, name).weight.numel()
                        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                    )
                ),
                "excluded": (
                    "linear_bias_additions",
                    "normalization",
                    "RoPE",
                    "attention_score_and_value_matmuls",
                    "softmax",
                    "elementwise_and_residual_ops",
                ),
            },
        }
    )
    return tensors, sentinels


def run_gemma3_l3_l4_mixed_mode_falsification_experiment(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE,
    candidate_file_sha256: str = DEFAULT_CANDIDATE_FILE_SHA256,
    candidate_report_sha256: str = DEFAULT_CANDIDATE_REPORT_SHA256,
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
    """Run the frozen origin-28 panel without any candidate fitting."""

    protocol = default_mixed_mode_protocol()
    # The exact manifest and all gates exist before candidate/model loading.
    protocol_sha256_before_inference = protocol.artifact_sha256
    runner_code_path = Path(__file__)
    generic_code_path = runner_code_path.with_name(
        "mixed_modal_interaction.py"
    )
    code_sha256s_before_model_load = {
        "gemma_mixed_mode_runner": _file_sha256(runner_code_path),
        "generic_mixed_modal_interaction": _file_sha256(generic_code_path),
    }
    if candidate_file_sha256 != DEFAULT_CANDIDATE_FILE_SHA256:
        raise ValueError("mixed-mode run requires the pinned candidate tensor")
    if candidate_report_sha256 != DEFAULT_CANDIDATE_REPORT_SHA256:
        raise ValueError("mixed-mode run requires the pinned candidate report")
    if hierarchy_artifact_sha256 != DEFAULT_HIERARCHY_ARTIFACT_SHA256:
        raise ValueError("mixed-mode run requires the pinned hierarchy")
    destination = _validate_output(output)
    candidate_file = Path(candidate_path)
    candidate = load_gemma3_conditional_spectral_candidate(
        candidate_file,
        expected_file_sha256=candidate_file_sha256,
        expected_report_sha256=candidate_report_sha256,
    )
    candidate_file_before = _file_sha256(candidate_file)
    plan_hashes_before = (
        _plan_hash(candidate.linear_plan),
        _plan_hash(candidate.quadratic_plan),
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
    if dict(reference.metadata()) != dict(candidate.binding):
        raise ValueError("candidate and live hierarchy bindings differ")
    if (
        candidate.model.get("model_id") != model_id
        or candidate.model.get("requested_revision") != revision
        or candidate.model.get("resolved_commit") != revision
        or candidate.model.get("local_files_only") is not True
        or candidate.model.get("tokenizer_loaded") is not False
    ):
        raise ValueError("candidate model metadata differs from requested model")
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
    source_model_before = adapter.model_fingerprint()
    if source_model_before != reference.source_model_sha256:
        raise ValueError("live Gemma fingerprint differs from frozen hierarchy")
    switcher = PreparedGemma3FullMLPStackSwitcher(
        adapter,
        {_SOURCE_SCOPE: catalog.replacements},
    )
    try:
        switcher.switch(_SOURCE_SCOPE)
        tensors, sentinels = _run_live_measurement(
            adapter,
            reference,
            candidate,
            protocol,
        )
    finally:
        switcher.close()
    source_model_after = adapter.model_fingerprint()
    candidate_file_after = _file_sha256(candidate_file)
    code_sha256s_after_inference = {
        "gemma_mixed_mode_runner": _file_sha256(runner_code_path),
        "generic_mixed_modal_interaction": _file_sha256(generic_code_path),
    }
    plan_hashes_after = (
        _plan_hash(candidate.linear_plan),
        _plan_hash(candidate.quadratic_plan),
    )
    if (
        source_model_after != source_model_before
        or candidate_file_after != candidate_file_before
        or plan_hashes_after != plan_hashes_before
        or protocol.artifact_sha256 != protocol_sha256_before_inference
        or code_sha256s_after_inference != code_sha256s_before_model_load
    ):
        raise RuntimeError(
            "model, candidate, code, or frozen protocol changed"
        )

    metrics, derived = evaluate_mixed_mode_falsification(
        tensors,
        sentinels=sentinels,
        protocol=protocol,
    )
    generic_artifact, generic_crosscheck = _generic_interaction_crosscheck(
        tensors,
        derived,
        candidate=candidate,
        protocol=protocol,
    )
    complete_tensors = {**tensors, **derived}
    manifest = _tensor_manifest(complete_tensors)
    binding = {
        "candidate_tensor_file_sha256": candidate_file_before,
        "candidate_report_payload_sha256": candidate_report_sha256,
        "candidate_artifact_sha256": candidate.artifact_sha256,
        "linear_plan_artifact_sha256": plan_hashes_before[0],
        "quadratic_plan_artifact_sha256": plan_hashes_before[1],
        "hierarchy_artifact_sha256": hierarchy_artifact_sha256,
        "source_model_sha256_before": source_model_before,
        "source_model_sha256_after": source_model_after,
        "candidate_tensor_file_sha256_after": candidate_file_after,
        "linear_plan_artifact_sha256_after": plan_hashes_after[0],
        "quadratic_plan_artifact_sha256_after": plan_hashes_after[1],
        "protocol_sha256_before_inference": protocol_sha256_before_inference,
        "protocol_sha256_after_inference": protocol.artifact_sha256,
        "candidate_changed_during_measurement": False,
        "source_model_changed_during_measurement": False,
        "protocol_changed_during_measurement": False,
        "code_sha256s_before_model_load": code_sha256s_before_model_load,
        "code_sha256s_after_inference": code_sha256s_after_inference,
        "code_changed_during_measurement": False,
    }
    common = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "binding": binding,
        "model": dict(candidate.model),
        "protocol": protocol.metadata(),
        "metrics": metrics,
        "generic_interaction_crosscheck": generic_crosscheck,
        "sentinels": sentinels,
        "tensor_manifest": manifest,
        "resource_accounting": {
            "fixed_reference_function_evaluations": sentinels[
                "fixed_reference_function_evaluation_count"
            ],
            "l4_attention_prefix_executions": sentinels[
                "l4_attention_prefix_evaluation_count"
            ],
            "linear_prepared_runtime_panel_executions": (
                len(RADII)
                * (
                    len(UNIQUE_MODES) * 2
                    + len(protocol.pairs) * len(SIGN_ROWS)
                )
            ),
            "quadratic_prepared_runtime_panel_executions": (
                len(RADII)
                * (
                    len(UNIQUE_MODES) * 2
                    + len(protocol.pairs) * len(SIGN_ROWS)
                )
            ),
            "prepared_runtime_execution_accounting_per_panel_call": (
                sentinels[
                    "prepared_runtime_execution_accounting_per_panel_call"
                ]
            ),
            "linear_prepared_runtime_factorized_macs_total": (
                sentinels[
                    "prepared_runtime_execution_accounting_per_panel_call"
                ]["linear"]["factorized_linear_macs"]
                * (
                    len(RADII)
                    * (
                        len(UNIQUE_MODES) * 2
                        + len(protocol.pairs) * len(SIGN_ROWS)
                    )
                )
            ),
            "quadratic_prepared_runtime_factorized_macs_total": (
                sentinels[
                    "prepared_runtime_execution_accounting_per_panel_call"
                ]["quadratic"]["factorized_linear_macs"]
                * (
                    len(RADII)
                    * (
                        len(UNIQUE_MODES) * 2
                        + len(protocol.pairs) * len(SIGN_ROWS)
                    )
                )
            ),
            "partial_analytic_live_model_macs": sentinels[
                "partial_analytic_live_model_macs"
            ],
            "native_or_compiled_l3_mlp_body_executions": 0,
            "native_or_compiled_l4_mlp_body_executions": 0,
            "runtime_speed_claim": False,
        },
        "claim_boundaries": _claim_boundaries(),
        "safety": _SAFETY,
    }
    logical_artifact_sha256 = _json_sha256(
        common,
        domain=_ARTIFACT_DOMAIN,
    )
    artifact = {
        **common,
        "response_tensors": complete_tensors,
        "generic_mixed_modal_interaction_artifact": (
            generic_artifact.state_dict()
        ),
        "artifact_sha256": logical_artifact_sha256,
    }
    report_payload = {
        **common,
        "artifact_sha256": logical_artifact_sha256,
        "interpretation": {
            "support_decision_is_panel_scoped": True,
            "full_interaction_uses_singleton_subtraction": True,
            "c11_is_odd_odd_diagnostic_only": True,
            "rho_0_5_is_scaling_diagnostic_only": True,
            "rho_1_0_is_preregistered_decision_radius": True,
            "candidate_contains_no_learned_cross_mode_correction": True,
        },
    }
    return _publish(artifact, report_payload, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen prompt-free Gemma L3-L4 mixed-mode "
            "falsification panel."
        )
    )
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--candidate-file-sha256",
        default=DEFAULT_CANDIDATE_FILE_SHA256,
    )
    parser.add_argument(
        "--candidate-report-sha256",
        default=DEFAULT_CANDIDATE_REPORT_SHA256,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_l3_l4_mixed_mode_falsification_experiment(
        candidate_path=arguments.candidate,
        candidate_file_sha256=arguments.candidate_file_sha256,
        candidate_report_sha256=arguments.candidate_report_sha256,
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
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
