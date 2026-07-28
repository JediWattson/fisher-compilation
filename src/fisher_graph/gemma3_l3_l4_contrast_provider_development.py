"""Develop a contrast-aware packed Gemma L3-to-L4 reference provider.

This is an open-development rung, not a sealed assessment.  It keeps the
prompt-conditioned Fisher basis frozen, constructs fresh prompt-blind
synthetic panels, and tests a materially different compression mechanism:

```
all 64 source modes -> learned 64-to-r packer -> causal rank-r executor
                    -> learned r-to-64 unpacker -> all 64 target modes
```

Ranks 8, 16, and 32 therefore measure a genuine dense modal bottleneck rather
than prefix deletion.  Fit-only finite contrasts and midpoint teacher JVPs
train the candidates.  Every candidate is frozen before the disjoint
selection panel is materialized.  The smallest candidate must pass both the
ordinary full-width fidelity gates and the contrast-family gates.

The exact Gemma gain-null coordinate is removed structurally.  The only gain
feature is the RMS of the non-null coordinates, reconstructed from row RMS
and the normalized null coordinate.  A pure gain-null displacement therefore
cannot change the provider input.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import Tensor, nn

from .adapters import Gemma3CausalLMAdapter, module_state_fingerprint
from .external_models import find_git_worktree
from .gated_executor import GatedCausalModalExecutorConfig
from .gemma3_experiment import DEFAULT_MODEL_ID
from .gemma3_l3_l4_basis_package import (
    DEFAULT_BASIS_PACKAGE,
    Gemma3L3L4BasisPackage,
)
from .gemma3_l3_l4_contrast_provider_development_materialization import (
    MaterializedDevelopmentBatch,
    materialize_development_role,
)
from .gemma3_l3_l4_contrast_provider_development_protocol import (
    DEFAULT_DEVELOPMENT_PROTOCOL_SHA256,
    CalibrationPilotMetric,
    ContrastProviderDevelopmentProtocol,
    DevelopmentCalibrationBinding,
    DevelopmentContrastGroupSpec,
    DevelopmentProbeSpec,
    FrozenDevelopmentCandidateSet,
    default_contrast_provider_development_protocol,
    freeze_development_candidates,
    select_global_calibration_amplitude,
)
from .gemma3_l3_l4_manifold_lift import (
    _authenticate_basis,
    _live_tolerance,
    _maximum_or_zero,
    _validate_live_unit_offset_norm,
)
from .gemma3_l3_l4_reference_provider_experiment import (
    DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256,
    _deferred_collision_gates,
    _fisher_metric_weight,
    _load_live_dependencies,
)
from .gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceGates,
)
from .gemma3_l3_l4_spectral_mapping_experiment import DEFAULT_REVISION
from .state_conditioned_contrast_assessment import (
    ContrastAssessmentGates,
    ContrastAssessmentResult,
    ContrastDefinition,
    ContrastObservation,
    assess_state_conditioned_contrasts,
)
from .state_conditioned_contrast_fit import (
    ContrastAwareObjective,
    ContrastAwareReferenceProviderPlan,
    IndexedReferenceBatch,
    ReferenceProviderContrastPair,
    fit_contrast_aware_reference_provider,
)
from .state_conditioned_reference_provider import SyntheticReferenceBatch
from .state_conditioned_reference_selection import (
    FullWidthCandidatePrediction,
    FullWidthCandidateScore,
    FullWidthReferenceCandidate,
    FullWidthReferenceControls,
    FullWidthReferenceProbe,
    FullWidthStructuralMetrics,
    fit_full_width_reference_controls,
    score_full_width_reference_candidate,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "compile_contrast_provider_development",
    "describe_contrast_provider_development",
    "build_parser",
    "main",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-l3-l4-contrast-packed-provider-dev-c2.pt"
)

_SCHEMA = "fisher_graph.gemma3_l3_l4_contrast_packed_provider_development.c2"
_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_RANKS = (8, 16, 32)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRAINING_STEPS = 300
_LEARNING_RATE = 1e-3
_BASE_SEED = 20_260_728_401
_EXPERT_COUNT = 2
_EXPERT_RANK_CAP = 16
_ROUTER_WIDTH = 16
_TARGET_SCALE_FLOOR = 1e-8
_GAIN_LOG_SCALE_FLOOR = 1e-8
_SUPPORT_RELATIVE_MARGIN = 1e-6
_SUPPORT_ABSOLUTE_MARGIN = 1e-9
_TENSOR_DOMAIN = b"fisher-graph:contrast-packed-development:tensor:v1\0"
_TRAINING_DOMAIN = b"fisher-graph:contrast-packed-development:training:v1\0"
_GAUGE_DOMAIN = b"fisher-graph:contrast-packed-development:gauge:v1\0"
_BINDING_DOMAIN = b"fisher-graph:contrast-packed-development:binding:v1\0"
_CODE_DOMAIN = b"fisher-graph:contrast-packed-development:code:v1\0"
_ARTIFACT_DOMAIN = b"fisher-graph:contrast-packed-development:artifact:v1\0"
_REPORT_DOMAIN = b"fisher-graph:contrast-packed-development:report:v1\0"
_LIFT_DOMAIN = b"fisher-graph:contrast-packed-development:lift:v1\0"
_CODE_FILES = (
    "gated_executor.py",
    "gemma3_l3_l4_contrast_provider_development_protocol.py",
    "gemma3_l3_l4_contrast_provider_development_materialization.py",
    "gemma3_l3_l4_manifold_lift.py",
    "gemma3_l3_l4_reference_provider_experiment.py",
    "state_conditioned_contrast_assessment.py",
    "state_conditioned_contrast_fit.py",
    "state_conditioned_reference_provider.py",
    "state_conditioned_reference_selection.py",
    "gemma3_l3_l4_contrast_provider_development.py",
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


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = _canonical_json_bytes(
        {
            "dtype": str(tensor.dtype),
            "shape": tuple(int(size) for size in tensor.shape),
        }
    )
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + header
        + b"\0"
        + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256s() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    result = {name: _file_sha256(directory / name) for name in _CODE_FILES}
    if set(result) != set(_CODE_FILES):
        raise RuntimeError("development code manifest is incomplete")
    return result


def _code_bundle_sha256(values: Mapping[str, str]) -> str:
    if set(values) != set(_CODE_FILES):
        raise ValueError("development code manifest is incomplete")
    for name, value in values.items():
        _require_sha256(value, label=f"code digest {name}")
    return _json_sha256(dict(values), domain=_CODE_DOMAIN)


def _training_spec() -> dict[str, object]:
    return {
        "steps": _TRAINING_STEPS,
        "learning_rate": _LEARNING_RATE,
        "base_seed": _BASE_SEED,
        "expert_count": _EXPERT_COUNT,
        "expert_rank_cap": _EXPERT_RANK_CAP,
        "router_width": _ROUTER_WIDTH,
        "router_activation": "tanh",
        "source_normalized_routing": True,
        "target_scale_floor": _TARGET_SCALE_FLOOR,
        "gain_log_scale_floor": _GAIN_LOG_SCALE_FLOOR,
        "support_relative_margin": _SUPPORT_RELATIVE_MARGIN,
        "support_absolute_margin": _SUPPORT_ABSOLUTE_MARGIN,
        "fit_schedule": "deterministic_full_batch_fixed_steps",
        "early_stopping": False,
        "selection_data_can_change_training": False,
        "visible_source_modes": _MODAL_WIDTH,
        "visible_target_modes": _MODAL_WIDTH,
        "bottleneck_semantics": "learned_64_to_r_to_64_modal_packing",
        "gain_semantics": (
            "standardized_log_nonnull_rms_exact_gain_null_omitted"
        ),
    }


def _training_sha256() -> str:
    return _json_sha256(_training_spec(), domain=_TRAINING_DOMAIN)


def _objective() -> ContrastAwareObjective:
    return ContrastAwareObjective(
        pointwise_weight=1.0,
        sensitivity_relative_delta_weight=2.0,
        sensitivity_direction_weight=0.5,
        midpoint_jvp_weight=1.0,
        intended_null_weight=1.0,
        sensitivity_relative_floor=1e-6,
        direction_norm_floor=1e-8,
        jvp_relative_floor=1e-6,
    )


def _executor_config(rank: int) -> GatedCausalModalExecutorConfig:
    if rank not in _RANKS:
        raise ValueError("latent rank is not in the frozen ladder")
    return GatedCausalModalExecutorConfig(
        input_modes=rank + 2,
        output_modes=rank,
        expert_count=_EXPERT_COUNT,
        expert_rank=min(_EXPERT_RANK_CAP, rank),
        router_width=_ROUTER_WIDTH,
        same_position_skip=False,
        max_positive_lag=None,
        router_activation="tanh",
        source_normalized_routing=True,
    )


@dataclass(frozen=True, slots=True)
class _LiftedDevelopmentBatch:
    batch: MaterializedDevelopmentBatch
    hidden_states: Tensor
    modal_coordinates: Tensor
    null_coordinates: Tensor
    row_rms: Tensor
    active_mask: Tensor
    norm_module_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        batch_size = len(self.batch.probe_ids)
        sequence = self.batch.sequence_length
        width = int(self.hidden_states.shape[-1])
        expected = {
            "hidden_states": (batch_size, sequence, width),
            "modal_coordinates": (batch_size, sequence, _MODAL_WIDTH),
            "null_coordinates": (batch_size, sequence, 1),
            "row_rms": (batch_size, sequence),
            "active_mask": (batch_size, sequence),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"development lift {name} geometry is invalid")
        if self.active_mask.dtype != torch.bool:
            raise TypeError("development lift active mask must be boolean")
        for value in (
            self.hidden_states,
            self.modal_coordinates,
            self.null_coordinates,
            self.row_rms,
        ):
            if not value.is_floating_point() or not bool(
                torch.isfinite(value).all()
            ):
                raise ValueError("development lift tensors must be finite")
        _require_sha256(
            self.norm_module_sha256,
            label="development lift norm hash",
        )
        _require_sha256(
            self.artifact_sha256,
            label="development lift artifact hash",
        )


@dataclass(frozen=True, slots=True)
class _MeasuredDevelopmentProbe:
    probe: DevelopmentProbeSpec
    hidden_states: Tensor
    modal_coordinates: Tensor
    null_coordinates: Tensor
    row_rms: Tensor
    target_modes: Tensor
    target_replays: tuple[Tensor, ...]
    logical_positions: Tensor
    valid_mask: Tensor
    materialized_tensor_sha256: str
    lift_artifact_sha256: str

    def __post_init__(self) -> None:
        length = self.probe.sequence_length
        expected = {
            "modal_coordinates": (1, length, _MODAL_WIDTH),
            "null_coordinates": (1, length, 1),
            "row_rms": (1, length),
            "target_modes": (1, length, _MODAL_WIDTH),
            "logical_positions": (1, length),
            "valid_mask": (1, length),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"measured {name} geometry is invalid")
        if (
            type(self.target_replays) is not tuple
            or not self.target_replays
            or any(
                tuple(value.shape) != (1, length, _MODAL_WIDTH)
                for value in self.target_replays
            )
            or not torch.equal(self.target_modes, self.target_replays[0])
        ):
            raise ValueError("measured target replays are invalid")
        for value in (
            self.materialized_tensor_sha256,
            self.lift_artifact_sha256,
        ):
            _require_sha256(value, label="measured probe hash")


@dataclass(frozen=True, slots=True)
class _ProviderChartMidpointJVP:
    """Exact provider-chart primal and push-forward at a hidden midpoint."""

    modal_primal: Tensor
    null_primal: Tensor
    row_rms_primal: Tensor
    modal_tangent: Tensor
    null_tangent: Tensor
    row_rms_tangent: Tensor

    def __post_init__(self) -> None:
        sequence = int(self.modal_primal.shape[0])
        expected = {
            "modal_primal": (sequence, _MODAL_WIDTH),
            "null_primal": (sequence, 1),
            "row_rms_primal": (sequence,),
            "modal_tangent": (sequence, _MODAL_WIDTH),
            "null_tangent": (sequence, 1),
            "row_rms_tangent": (sequence,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or tuple(value.shape) != shape
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"provider chart {name} geometry or values are invalid"
                )
        if bool((self.row_rms_primal <= 0.0).any()):
            raise ValueError("provider chart row-RMS primal must be positive")


def _lift_development_batch(
    basis: Gemma3L3L4BasisPackage,
    rmsnorm: nn.Module,
    *,
    epsilon: float,
    batch: MaterializedDevelopmentBatch,
    probes: Sequence[DevelopmentProbeSpec],
) -> _LiftedDevelopmentBatch:
    """Apply the authenticated option-B manifold lift to a development batch."""

    authenticated_basis = _authenticate_basis(basis)
    if not isinstance(batch, MaterializedDevelopmentBatch):
        raise TypeError("batch must be a MaterializedDevelopmentBatch")
    batch.validate_integrity()
    specifications = tuple(probes)
    if (
        len(specifications) != len(batch.probe_ids)
        or tuple(value.probe_id for value in specifications)
        != batch.probe_ids
        or tuple(value.artifact_sha256 for value in specifications)
        != batch.probe_artifact_sha256s
    ):
        raise ValueError("development lift probes do not match batch identity")
    gain, null_indices, norm_sha256 = _validate_live_unit_offset_norm(
        rmsnorm,
        epsilon=epsilon,
        width=authenticated_basis.residual_width,
    )
    if len(null_indices) != 1:
        raise ValueError("development lift requires one exact gain null")
    null = list(null_indices)
    nonnull = torch.ones(authenticated_basis.residual_width, dtype=torch.bool)
    nonnull[null] = False
    requested = batch.values.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    active = torch.linalg.vector_norm(requested, dim=-1) > 0.0
    sigma = authenticated_basis.source_mode_standard_deviations(_MODAL_WIDTH)
    decoded = (
        requested * sigma.view(1, 1, -1)
    ) @ authenticated_basis.P3[:, :_MODAL_WIDTH].T
    requested_x3 = authenticated_basis.x3_mean.view(1, 1, -1) + decoded
    pre_norm = torch.zeros_like(requested_x3)
    pre_norm[..., nonnull] = requested_x3[..., nonnull] / gain[nonnull]
    q = pre_norm.square().mean(dim=-1)
    if bool((q <= torch.finfo(torch.float64).tiny).any()):
        raise ValueError("development Fisher direction has degenerate RMS")
    directions = pre_norm / q.sqrt().unsqueeze(-1)

    neutral = torch.zeros_like(authenticated_basis.x3_mean)
    neutral[nonnull] = authenticated_basis.x3_mean[nonnull] / gain[nonnull]
    neutral_q = float(neutral.square().mean())
    if not math.isfinite(neutral_q) or neutral_q <= 0.0:
        raise ValueError("development neutral direction has degenerate RMS")
    neutral = neutral / math.sqrt(neutral_q)
    neutral[null] = 0.0

    hidden = directions * batch.radial_scales.view(-1, 1, 1)
    hidden[..., null] = batch.null_coordinates.view(-1, 1, 1)
    hidden = torch.where(
        active.unsqueeze(-1),
        hidden,
        neutral.view(1, 1, -1).expand_as(hidden),
    )
    weight = getattr(rmsnorm, "weight")
    if not isinstance(weight, Tensor):
        raise TypeError("live RMSNorm lacks a tensor weight")
    runtime_hidden = hidden.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        live_x3 = rmsnorm(runtime_hidden)
    if module_state_fingerprint(rmsnorm) != norm_sha256:
        raise RuntimeError("RMSNorm changed during development lift")
    hidden64 = runtime_hidden.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    x3 = live_x3.detach().to(device="cpu", dtype=torch.float64)
    denominator = (
        hidden64.square().mean(dim=-1, keepdim=True) + float(epsilon)
    ).sqrt()
    analytic = gain.view(1, 1, -1) * hidden64 / denominator
    tolerance = _live_tolerance(weight.dtype)
    if _maximum_or_zero(x3 - analytic) > tolerance * max(
        float(x3.abs().max()),
        1.0,
    ):
        raise ValueError("live module is not the declared unit-offset RMSNorm")
    if bool((x3[..., null] != 0.0).any()):
        raise ValueError("live RMSNorm is nonzero on the exact gain null")
    modal = (
        (
            x3 - authenticated_basis.x3_mean.view(1, 1, -1)
        )
        @ authenticated_basis.R3[:_MODAL_WIDTH].T
    ) / sigma.view(1, 1, -1)
    row_rms = hidden64.square().mean(dim=-1).sqrt()
    normalized_null = hidden64[..., null] / denominator
    payload = {
        "batch_sha256": batch.artifact_sha256,
        "basis_sha256": authenticated_basis.basis_payload_sha256,
        "norm_sha256": norm_sha256,
        "epsilon": float(epsilon),
        "hidden_sha256": _tensor_sha256(hidden64),
        "modal_sha256": _tensor_sha256(modal),
        "null_sha256": _tensor_sha256(normalized_null),
        "row_rms_sha256": _tensor_sha256(row_rms),
        "active_mask_sha256": _tensor_sha256(active),
        "formula": "option_b_unit_rms_direction_then_radial_and_null",
    }
    return _LiftedDevelopmentBatch(
        batch=batch,
        hidden_states=hidden64.contiguous(),
        modal_coordinates=modal.contiguous(),
        null_coordinates=normalized_null.contiguous(),
        row_rms=row_rms.contiguous(),
        active_mask=active.contiguous(),
        norm_module_sha256=norm_sha256,
        artifact_sha256=_json_sha256(payload, domain=_LIFT_DOMAIN),
    )


def _prepare_sequence(
    *,
    adapter: Gemma3CausalLMAdapter,
    basis: Gemma3L3L4BasisPackage,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[object, Tensor, Tensor]:
    positions = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(batch_size, -1)
    valid = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.bool,
        device=device,
    )
    placeholder = torch.zeros(
        batch_size,
        sequence_length,
        basis.residual_width,
        dtype=dtype,
        device=device,
    )
    sequence = adapter.prepare_sequence(
        {
            "inputs_embeds": placeholder,
            "attention_mask": valid,
            "position_ids": positions,
        }
    )
    return sequence, positions, valid


def _teacher_target_function(
    *,
    adapter: Gemma3CausalLMAdapter,
    basis: Gemma3L3L4BasisPackage,
    post_ff3: nn.Module,
    sequence: object,
    device: torch.device,
    dtype: torch.dtype,
):
    y3_mean = basis.y3_mean.to(device=device, dtype=dtype)
    x4_mean = basis.x4_mean.to(device=device, dtype=dtype)
    r4 = basis.R4[:_MODAL_WIDTH].to(device=device, dtype=dtype)
    with torch.no_grad():
        post_ff_delta = post_ff3(
            y3_mean.view(1, 1, -1)
        ).detach()
    segment4 = adapter.segment("layer.4")

    def teacher(hidden: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden.shape
        reference = hidden + post_ff_delta.expand(
            batch_size,
            sequence_length,
            -1,
        )
        x4 = adapter.run_attention_prefix(
            segment4,
            reference,
            sequence,  # type: ignore[arg-type]
        ).normalized_mlp_input
        return (
            x4 - x4_mean.view(1, 1, -1)
        ) @ r4.T

    return teacher


def _measure_role(
    *,
    role: str,
    protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding | None,
    frozen_candidates: FrozenDevelopmentCandidateSet | None,
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    post_ff3: nn.Module,
    epsilon: float,
    replay_count: int,
) -> tuple[tuple[_MeasuredDevelopmentProbe, ...], dict[str, object]]:
    if replay_count <= 0:
        raise ValueError("teacher replay count must be positive")
    materialized = materialize_development_role(
        protocol,
        role,  # type: ignore[arg-type]
        calibration=calibration,
        frozen_candidates=frozen_candidates,
    )
    specifications = {
        probe.probe_id: probe for probe in protocol.probes_for_role(role)
    }
    first_parameter = next(adapter.module.parameters(), None)
    if first_parameter is None or not first_parameter.is_floating_point():
        raise TypeError("live Gemma model has no floating parameters")
    device = first_parameter.device
    dtype = first_parameter.dtype
    measured: list[_MeasuredDevelopmentProbe] = []
    lift_hashes: list[str] = []
    target_hashes: list[tuple[str, ...]] = []
    model_before = adapter.model_fingerprint()
    norm_before = module_state_fingerprint(pre_ff3)

    for request in materialized:
        probes = tuple(specifications[value] for value in request.probe_ids)
        lifted = _lift_development_batch(
            basis,
            pre_ff3,
            epsilon=epsilon,
            batch=request,
            probes=probes,
        )
        lift_hashes.append(lifted.artifact_sha256)
        sequence, positions, valid = _prepare_sequence(
            adapter=adapter,
            basis=basis,
            batch_size=len(probes),
            sequence_length=request.sequence_length,
            device=device,
            dtype=dtype,
        )
        teacher = _teacher_target_function(
            adapter=adapter,
            basis=basis,
            post_ff3=post_ff3,
            sequence=sequence,
            device=device,
            dtype=dtype,
        )
        runtime_hidden = lifted.hidden_states.to(
            device=device,
            dtype=dtype,
        )
        replays: list[Tensor] = []
        with torch.no_grad():
            for _ in range(replay_count):
                replays.append(
                    teacher(runtime_hidden)
                    .detach()
                    .to(device="cpu", dtype=torch.float64)
                    .contiguous()
                )
        positions64 = positions.detach().to(device="cpu")
        valid64 = valid.detach().to(device="cpu")
        for index, probe in enumerate(probes):
            probe_replays = tuple(
                value[index : index + 1] for value in replays
            )
            target_hashes.append(
                tuple(_tensor_sha256(value) for value in probe_replays)
            )
            measured.append(
                _MeasuredDevelopmentProbe(
                    probe=probe,
                    hidden_states=lifted.hidden_states[index : index + 1],
                    modal_coordinates=(
                        lifted.modal_coordinates[index : index + 1]
                    ),
                    null_coordinates=(
                        lifted.null_coordinates[index : index + 1]
                    ),
                    row_rms=lifted.row_rms[index : index + 1],
                    target_modes=probe_replays[0],
                    target_replays=probe_replays,
                    logical_positions=positions64[index : index + 1],
                    valid_mask=valid64[index : index + 1],
                    materialized_tensor_sha256=(
                        request.probe_tensor_sha256s[index]
                    ),
                    lift_artifact_sha256=lifted.artifact_sha256,
                )
            )

    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_before
    ):
        raise RuntimeError("model or norm changed while measuring development")
    measured.sort(key=lambda value: value.probe.ordinal)
    expected = protocol.probes_for_role(role)
    if tuple(value.probe.probe_id for value in measured) != tuple(
        value.probe_id for value in expected
    ):
        raise RuntimeError("measured development probe order drifted")
    family_counts = {
        family: sum(value.probe.family == family for value in measured)
        for family in sorted({value.probe.family for value in measured})
    }
    return tuple(measured), {
        "role": role,
        "probe_count": len(measured),
        "family_counts": family_counts,
        "materialized_batch_sha256s": tuple(
            value.artifact_sha256 for value in materialized
        ),
        "lift_artifact_sha256s": tuple(lift_hashes),
        "ordered_target_replay_sha256s": tuple(target_hashes),
        "teacher_replay_count": replay_count,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
    }


def _population_center_scale(
    values: Sequence[Tensor],
    masks: Sequence[Tensor],
    *,
    floor: float,
) -> tuple[Tensor, Tensor]:
    if len(values) != len(masks) or not values:
        raise ValueError("population tensors and masks are not aligned")
    rows = torch.cat(
        [
            value.detach().to(device="cpu", dtype=torch.float64)[mask]
            for value, mask in zip(values, masks, strict=True)
        ],
        dim=0,
    )
    if rows.ndim != 2 or rows.shape[0] <= 0:
        raise ValueError("population contains no valid rows")
    center = rows.mean(dim=0)
    scale = rows.std(dim=0, correction=0).clamp_min(float(floor))
    return center.contiguous(), scale.contiguous()


def _nonnull_log_rms(
    *,
    row_rms: Tensor,
    normalized_null: Tensor,
    residual_width: int,
    epsilon: float,
) -> Tensor:
    denominator = (row_rms.square() + float(epsilon)).sqrt()
    hidden_null = normalized_null[..., 0] * denominator
    nonnull_mean_square = (
        row_rms.square() - hidden_null.square() / residual_width
    ).clamp_min(float(epsilon))
    return 0.5 * torch.log(nonnull_mean_square)


def _fit_gauges(
    measured: Sequence[_MeasuredDevelopmentProbe],
    *,
    residual_width: int,
    epsilon: float,
) -> tuple[Tensor, float, float, Tensor, Tensor]:
    masks = [value.valid_mask for value in measured]
    modal_center, _ = _population_center_scale(
        [value.modal_coordinates for value in measured],
        masks,
        floor=1.0,
    )
    target_center, target_scale = _population_center_scale(
        [value.target_modes for value in measured],
        masks,
        floor=_TARGET_SCALE_FLOOR,
    )
    log_gain_rows = torch.cat(
        [
            _nonnull_log_rms(
                row_rms=value.row_rms,
                normalized_null=value.null_coordinates,
                residual_width=residual_width,
                epsilon=epsilon,
            )[value.valid_mask]
            for value in measured
        ]
    )
    gain_center = float(log_gain_rows.mean())
    gain_scale = max(
        float(log_gain_rows.std(correction=0)),
        _GAIN_LOG_SCALE_FLOOR,
    )
    return (
        modal_center,
        gain_center,
        gain_scale,
        target_center,
        target_scale,
    )


def _provider_binding_sha256(
    *,
    basis: Gemma3L3L4BasisPackage,
    protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
    objective: ContrastAwareObjective,
    norm_sha256: str,
    metric_weight: Tensor,
) -> str:
    return _json_sha256(
        {
            "schema": "fisher_graph.contrast_packed_provider_binding.c2",
            "basis_payload_sha256": basis.basis_payload_sha256,
            "source_model_sha256": basis.source_model_sha256,
            "protocol_sha256": protocol.protocol_sha256,
            "calibration_sha256": calibration.artifact_sha256,
            "objective_sha256": objective.artifact_sha256,
            "training_sha256": _training_sha256(),
            "norm_sha256": norm_sha256,
            "metric_weight_sha256": _tensor_sha256(metric_weight),
            "visible_source_modes": _MODAL_WIDTH,
            "visible_target_modes": _MODAL_WIDTH,
        },
        domain=_BINDING_DOMAIN,
    )


def _indexed_batches(
    measured: Sequence[_MeasuredDevelopmentProbe],
    *,
    split: str,
    binding_sha256: str,
) -> tuple[IndexedReferenceBatch, ...]:
    by_geometry: dict[int, list[_MeasuredDevelopmentProbe]] = {}
    for value in measured:
        by_geometry.setdefault(value.probe.sequence_length, []).append(value)
    result: list[IndexedReferenceBatch] = []
    for sequence_length in sorted(by_geometry):
        rows = by_geometry[sequence_length]
        result.append(
            IndexedReferenceBatch(
                batch=SyntheticReferenceBatch(
                    split=split,  # type: ignore[arg-type]
                    modal_coordinates=torch.cat(
                        [value.modal_coordinates for value in rows],
                        dim=0,
                    ),
                    null_coordinates=torch.cat(
                        [value.null_coordinates for value in rows],
                        dim=0,
                    ),
                    row_rms=torch.cat(
                        [value.row_rms for value in rows],
                        dim=0,
                    ),
                    target_modes=torch.cat(
                        [value.target_modes for value in rows],
                        dim=0,
                    ),
                    logical_positions=torch.cat(
                        [value.logical_positions for value in rows],
                        dim=0,
                    ),
                    valid_mask=torch.cat(
                        [value.valid_mask for value in rows],
                        dim=0,
                    ),
                    synthetic_binding_sha256=binding_sha256,
                ),
                endpoint_ids=tuple(value.probe.probe_id for value in rows),
            )
        )
    return tuple(result)


def _exact_hidden_chart_midpoint_pushforward(
    chart: Callable[[Tensor], tuple[Tensor, Tensor, Tensor]],
    *,
    left_hidden: Tensor,
    right_hidden: Tensor,
) -> tuple[
    tuple[Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]:
    """Evaluate ``z(Hmid)`` and ``J_z(Hmid) (Hright - Hleft)`` exactly."""

    if (
        not isinstance(left_hidden, Tensor)
        or not isinstance(right_hidden, Tensor)
        or left_hidden.shape != right_hidden.shape
        or not left_hidden.is_floating_point()
        or not right_hidden.is_floating_point()
    ):
        raise ValueError("hidden chart endpoints must have equal float geometry")
    midpoint = 0.5 * (left_hidden + right_hidden)
    hidden_tangent = right_hidden - left_hidden
    if float(torch.linalg.vector_norm(hidden_tangent)) <= 0.0:
        raise ValueError("hidden chart tangent must be nonzero")
    primals, tangents = torch.autograd.functional.jvp(
        chart,
        midpoint,
        hidden_tangent,
        create_graph=False,
        strict=True,
    )
    if (
        type(primals) is not tuple
        or type(tangents) is not tuple
        or len(primals) != 3
        or len(tangents) != 3
    ):
        raise TypeError("provider chart must return modal, null, and row RMS")
    return primals, tangents


def _provider_chart_midpoint_jvp(
    *,
    left: _MeasuredDevelopmentProbe,
    right: _MeasuredDevelopmentProbe,
    basis: Gemma3L3L4BasisPackage,
    pre_ff3: nn.Module,
    epsilon: float,
) -> _ProviderChartMidpointJVP:
    """Push a hidden-space chord through the live nonlinear provider chart."""

    if left.probe.sequence_length != right.probe.sequence_length:
        raise ValueError("chart JVP contrast endpoints must have equal length")
    authenticated_basis = _authenticate_basis(basis)
    _, null_indices, norm_sha256 = _validate_live_unit_offset_norm(
        pre_ff3,
        epsilon=epsilon,
        width=authenticated_basis.residual_width,
    )
    null_index = null_indices[0]
    weight = getattr(pre_ff3, "weight")
    if not isinstance(weight, Tensor):
        raise TypeError("live RMSNorm lacks a tensor weight")
    device = weight.device
    dtype = weight.dtype
    source_mean = authenticated_basis.x3_mean.to(
        device=device,
        dtype=dtype,
    )
    source_basis = authenticated_basis.R3[:_MODAL_WIDTH].to(
        device=device,
        dtype=dtype,
    )
    source_sigma = authenticated_basis.source_mode_standard_deviations(
        _MODAL_WIDTH
    ).to(device=device, dtype=dtype)

    def chart(hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x3 = pre_ff3(hidden)
        denominator = (
            hidden.square().mean(dim=-1, keepdim=True) + float(epsilon)
        ).sqrt()
        modal = (
            (x3 - source_mean.view(1, 1, -1)) @ source_basis.T
        ) / source_sigma.view(1, 1, -1)
        normalized_null = hidden[..., null_index : null_index + 1] / (
            denominator
        )
        row_rms = hidden.square().mean(dim=-1).sqrt()
        return modal, normalized_null, row_rms

    primals, tangents = _exact_hidden_chart_midpoint_pushforward(
        chart,
        left_hidden=left.hidden_states.to(device=device, dtype=dtype),
        right_hidden=right.hidden_states.to(device=device, dtype=dtype),
    )
    if module_state_fingerprint(pre_ff3) != norm_sha256:
        raise RuntimeError("RMSNorm changed during provider-chart JVP")

    def canonical(value: Tensor) -> Tensor:
        return (
            value[0]
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
        )

    return _ProviderChartMidpointJVP(
        modal_primal=canonical(primals[0]),
        null_primal=canonical(primals[1]),
        row_rms_primal=canonical(primals[2]),
        modal_tangent=canonical(tangents[0]),
        null_tangent=canonical(tangents[1]),
        row_rms_tangent=canonical(tangents[2]),
    )


def _chart_comparison_metrics(
    exact: Tensor,
    endpoint_approximation: Tensor,
) -> dict[str, float]:
    exact64 = exact.detach().to(device="cpu", dtype=torch.float64).flatten()
    approximation64 = (
        endpoint_approximation.detach()
        .to(device="cpu", dtype=torch.float64)
        .flatten()
    )
    if exact64.shape != approximation64.shape or exact64.numel() <= 0:
        raise ValueError("chart comparison geometry is invalid")
    exact_norm = float(torch.linalg.vector_norm(exact64))
    approximation_norm = float(torch.linalg.vector_norm(approximation64))
    difference_norm = float(
        torch.linalg.vector_norm(approximation64 - exact64)
    )
    floor = torch.finfo(torch.float64).eps
    if exact_norm <= floor and approximation_norm <= floor:
        cosine = 1.0
        gain = 1.0
    elif exact_norm <= floor or approximation_norm <= floor:
        cosine = 0.0
        gain = 0.0 if approximation_norm <= floor else (
            approximation_norm / floor
        )
    else:
        dot = float(torch.dot(approximation64, exact64))
        cosine = max(
            -1.0,
            min(1.0, dot / (approximation_norm * exact_norm)),
        )
        gain = dot / (exact_norm * exact_norm)
    return {
        "endpoint_approximation_relative_error_vs_exact": (
            difference_norm / max(exact_norm, floor)
        ),
        "endpoint_approximation_cosine_vs_exact": cosine,
        "endpoint_approximation_gain_along_exact": gain,
    }


def _provider_chart_mismatch_diagnostic(
    *,
    pair_id: str,
    family: str,
    rank_stratum: str,
    left: _MeasuredDevelopmentProbe,
    right: _MeasuredDevelopmentProbe,
    exact: _ProviderChartMidpointJVP,
) -> dict[str, object]:
    """Compare exact hidden-chart data with the old endpoint arithmetic."""

    endpoint_primals = {
        "modal": 0.5 * (
            left.modal_coordinates[0] + right.modal_coordinates[0]
        ),
        "null": 0.5 * (
            left.null_coordinates[0] + right.null_coordinates[0]
        ),
        "row_rms": 0.5 * (left.row_rms[0] + right.row_rms[0]),
    }
    endpoint_tangents = {
        "modal": right.modal_coordinates[0] - left.modal_coordinates[0],
        "null": right.null_coordinates[0] - left.null_coordinates[0],
        "row_rms": right.row_rms[0] - left.row_rms[0],
    }
    exact_primals = {
        "modal": exact.modal_primal,
        "null": exact.null_primal,
        "row_rms": exact.row_rms_primal,
    }
    exact_tangents = {
        "modal": exact.modal_tangent,
        "null": exact.null_tangent,
        "row_rms": exact.row_rms_tangent,
    }
    return {
        "pair_id": pair_id,
        "family": family,
        "rank_stratum": rank_stratum,
        "sequence_length": left.probe.sequence_length,
        "primal_exact_vs_endpoint_midpoint": {
            name: _chart_comparison_metrics(
                exact_primals[name],
                endpoint_primals[name],
            )
            for name in ("modal", "null", "row_rms")
        },
        "tangent_exact_pushforward_vs_endpoint_chord": {
            name: _chart_comparison_metrics(
                exact_tangents[name],
                endpoint_tangents[name],
            )
            for name in ("modal", "null", "row_rms")
        },
        "exact_semantics": (
            "z_of_hidden_midpoint_and_J_z_at_hidden_midpoint_times_"
            "right_minus_left_hidden"
        ),
        "endpoint_arithmetic_used_for_fit": False,
    }


def _teacher_midpoint_jvp(
    *,
    left: _MeasuredDevelopmentProbe,
    right: _MeasuredDevelopmentProbe,
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    post_ff3: nn.Module,
) -> Tensor:
    if left.probe.sequence_length != right.probe.sequence_length:
        raise ValueError("JVP contrast endpoints must have equal length")
    first_parameter = next(adapter.module.parameters())
    device = first_parameter.device
    dtype = first_parameter.dtype
    sequence, _, _ = _prepare_sequence(
        adapter=adapter,
        basis=basis,
        batch_size=1,
        sequence_length=left.probe.sequence_length,
        device=device,
        dtype=dtype,
    )
    teacher = _teacher_target_function(
        adapter=adapter,
        basis=basis,
        post_ff3=post_ff3,
        sequence=sequence,
        device=device,
        dtype=dtype,
    )
    left_hidden = left.hidden_states.to(device=device, dtype=dtype)
    right_hidden = right.hidden_states.to(device=device, dtype=dtype)
    midpoint = 0.5 * (left_hidden + right_hidden)
    # Match the packed fitter's chord convention: the midpoint is evaluated
    # with the complete left-to-right displacement as its tangent.
    tangent = right_hidden - left_hidden
    if float(torch.linalg.vector_norm(tangent)) <= 0.0:
        raise ValueError("JVP contrast tangent must be nonzero")
    _, jvp = torch.autograd.functional.jvp(
        teacher,
        midpoint,
        tangent,
        create_graph=False,
        strict=True,
    )
    return (
        jvp[0]
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
    )


def _training_contrast_pairs(
    *,
    protocol: ContrastProviderDevelopmentProtocol,
    measured: Sequence[_MeasuredDevelopmentProbe],
    basis: Gemma3L3L4BasisPackage,
    adapter: Gemma3CausalLMAdapter,
    pre_ff3: nn.Module,
    post_ff3: nn.Module,
    epsilon: float,
) -> tuple[
    tuple[ReferenceProviderContrastPair, ...],
    tuple[dict[str, object], ...],
]:
    measured_by_id = {value.probe.probe_id: value for value in measured}
    result: list[ReferenceProviderContrastPair] = []
    diagnostics: list[dict[str, object]] = []
    for group in protocol.groups_for_role("fit"):
        for index, (left_id, right_id) in enumerate(
            group.canonical_variant_pairs
        ):
            pair_id = f"{group.group_id}.pair.{index:02d}"
            left = measured_by_id[left_id]
            right = measured_by_id[right_id]
            jvp = (
                _teacher_midpoint_jvp(
                    left=left,
                    right=right,
                    basis=basis,
                    adapter=adapter,
                    post_ff3=post_ff3,
                )
                if group.intent == "sensitivity"
                else None
            )
            chart = (
                _provider_chart_midpoint_jvp(
                    left=left,
                    right=right,
                    basis=basis,
                    pre_ff3=pre_ff3,
                    epsilon=epsilon,
                )
                if group.intent == "sensitivity"
                else None
            )
            if chart is not None:
                diagnostics.append(
                    _provider_chart_mismatch_diagnostic(
                        pair_id=pair_id,
                        family=group.family,
                        rank_stratum=group.rank_band,
                        left=left,
                        right=right,
                        exact=chart,
                    )
                )
            result.append(
                ReferenceProviderContrastPair(
                    pair_id=pair_id,
                    family=group.family,
                    role=(
                        "expected_sensitivity"
                        if group.intent == "sensitivity"
                        else "intended_null"
                    ),
                    left_endpoint_id=left_id,
                    right_endpoint_id=right_id,
                    rank_stratum=group.rank_band,
                    teacher_midpoint_jvp=jvp,
                    provider_chart_modal_primal=(
                        None if chart is None else chart.modal_primal
                    ),
                    provider_chart_null_primal=(
                        None if chart is None else chart.null_primal
                    ),
                    provider_chart_row_rms_primal=(
                        None if chart is None else chart.row_rms_primal
                    ),
                    provider_chart_modal_tangent=(
                        None if chart is None else chart.modal_tangent
                    ),
                    provider_chart_null_tangent=(
                        None if chart is None else chart.null_tangent
                    ),
                    provider_chart_row_rms_tangent=(
                        None if chart is None else chart.row_rms_tangent
                    ),
                )
            )
    return tuple(result), tuple(diagnostics)


def _masked_metric_target(
    value: _MeasuredDevelopmentProbe,
    *,
    replay: int,
    metric_weight: Tensor,
) -> Tensor:
    target = value.target_replays[replay] * metric_weight.view(1, 1, -1)
    return target[value.valid_mask].contiguous()


def _calibration_metrics(
    *,
    protocol: ContrastProviderDevelopmentProtocol,
    measured: Sequence[_MeasuredDevelopmentProbe],
    metric_weight: Tensor,
) -> tuple[CalibrationPilotMetric, ...]:
    """Construct the 20 fit-only amplitude metrics.

    A half-step is another preregistered pilot pair for the same modal base.
    The frozen grid contains exact halves for 4.0, 8.0, and 12.0.  The 2.0
    and 6.0 rows remain deliberately ineligible because their half-step was
    not opened.
    """

    measured_by_id = {value.probe.probe_id: value for value in measured}
    groups = protocol.groups_for_role("pilot")
    if len(groups) != 20:
        raise RuntimeError("pilot protocol must contain 20 groups")
    by_base_and_amplitude: dict[
        tuple[str, float],
        tuple[DevelopmentContrastGroupSpec, Tensor, Tensor, float, float],
    ] = {}
    epsilon = torch.finfo(torch.float64).eps
    for group in groups:
        if len(group.canonical_variant_pairs) != 1:
            raise RuntimeError("pilot groups must contain one signed pair")
        left_id, right_id = group.canonical_variant_pairs[0]
        left = measured_by_id[left_id]
        right = measured_by_id[right_id]
        amplitude = left.probe.modal_amplitude
        if amplitude != right.probe.modal_amplitude:
            raise RuntimeError("pilot pair amplitudes differ")
        left_target = _masked_metric_target(
            left,
            replay=0,
            metric_weight=metric_weight,
        )
        right_target = _masked_metric_target(
            right,
            replay=0,
            metric_weight=metric_weight,
        )
        delta = right_target - left_target
        baseline = 0.5 * (
            float(torch.linalg.vector_norm(left_target))
            + float(torch.linalg.vector_norm(right_target))
        )
        if baseline <= 0.0:
            raise ValueError("pilot teacher baseline is zero")
        repeated_delta = (
            _masked_metric_target(
                right,
                replay=1,
                metric_weight=metric_weight,
            )
            - _masked_metric_target(
                left,
                replay=1,
                metric_weight=metric_weight,
            )
        )
        repeat_noise = float(
            torch.linalg.vector_norm(delta - repeated_delta)
        )
        uncertainty = 8.0 * max(
            repeat_noise,
            4.0 * epsilon * baseline,
        )
        signal = float(torch.linalg.vector_norm(delta))
        lower = max(signal - uncertainty, 0.0) / baseline
        upper = (signal + uncertainty) / baseline
        base_id = group.group_id.rsplit(".h_", 1)[0]
        by_base_and_amplitude[(base_id, amplitude)] = (
            group,
            delta / (2.0 * amplitude),
            repeated_delta / (2.0 * amplitude),
            lower,
            upper,
        )

    result: list[CalibrationPilotMetric] = []
    for (base_id, amplitude), (
        group,
        full_fd,
        _repeated_fd,
        lower,
        upper,
    ) in sorted(by_base_and_amplitude.items()):
        half = by_base_and_amplitude.get((base_id, amplitude / 2.0))
        if half is None:
            cosine = None
            gain = None
        else:
            half_fd = half[1]
            full_flat = full_fd.reshape(-1)
            half_flat = half_fd.reshape(-1)
            full_norm = float(torch.linalg.vector_norm(full_flat))
            half_norm = float(torch.linalg.vector_norm(half_flat))
            if full_norm <= 0.0 or half_norm <= 0.0:
                cosine = -1.0
                gain = 0.0
            else:
                dot = float(torch.dot(full_flat, half_flat))
                cosine = max(
                    -1.0,
                    min(1.0, dot / (full_norm * half_norm)),
                )
                gain = dot / max(
                    float(torch.dot(half_flat, half_flat)),
                    torch.finfo(torch.float64).tiny,
                )
        label = str(amplitude).replace(".", "p")
        result.append(
            CalibrationPilotMetric(
                metric_id=(
                    "development_c2.pilot.metric."
                    f"{group.rank_band}.h_{label}"
                ),
                rank_band=group.rank_band,
                amplitude=amplitude,
                teacher_relative_effect_lower=lower,
                teacher_relative_effect_upper=upper,
                half_full_fd_cosine=cosine,
                half_full_fd_gain=gain,
            )
        )
    if len(result) != 20:
        raise RuntimeError("pilot metric construction did not produce 20 rows")
    return tuple(result)


def _standardized_gauge_sha256(
    *,
    basis: Gemma3L3L4BasisPackage,
    protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
    objective: ContrastAwareObjective,
    metric_weight: Tensor,
) -> str:
    return _json_sha256(
        {
            "basis_payload_sha256": basis.basis_payload_sha256,
            "source_model_sha256": basis.source_model_sha256,
            "protocol_sha256": protocol.protocol_sha256,
            "calibration_sha256": calibration.artifact_sha256,
            "training_sha256": _training_sha256(),
            "objective_sha256": objective.artifact_sha256,
            "metric_weight_sha256": _tensor_sha256(metric_weight),
            "metric_width": _MODAL_WIDTH,
            "metric_semantics": (
                "raw_l4_modal_coordinates_times_sqrt_frozen_singular_values"
            ),
        },
        domain=_GAUGE_DOMAIN,
    )


def _ordinary_full_width_probes(
    measured: Sequence[_MeasuredDevelopmentProbe],
    *,
    split: str,
    metric_weight: Tensor,
    standardized_gauge_sha256: str,
) -> tuple[FullWidthReferenceProbe, ...]:
    ordinary = tuple(
        value
        for value in measured
        if value.probe.family in {"multitone", "block_sparse"}
    )
    if len(ordinary) != 16:
        raise RuntimeError("ordinary development panel must contain 16 probes")
    return tuple(
        FullWidthReferenceProbe(
            probe_id=value.probe.probe_id,
            split=split,  # type: ignore[arg-type]
            family=value.probe.family,
            standardized_target=(
                value.target_modes * metric_weight.view(1, 1, -1)
            ),
            logical_positions=value.logical_positions,
            valid_mask=value.valid_mask,
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        for value in ordinary
    )


def _runtime_predictions(
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[_MeasuredDevelopmentProbe],
    *,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    runtime = plan.prepare(dtype=dtype, device="cpu")
    result: dict[str, Tensor] = {}
    with torch.no_grad():
        for value in measured:
            result[value.probe.probe_id] = (
                runtime(
                    value.modal_coordinates.to(dtype=dtype),
                    value.null_coordinates.to(dtype=dtype),
                    value.row_rms.to(dtype=dtype),
                    valid_mask=value.valid_mask,
                    logical_positions=value.logical_positions,
                )
                .detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            )
    return result


def _relative_tensor_map_error(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
) -> float:
    if set(left) != set(right) or not left:
        raise ValueError("prediction maps are not aligned")
    numerator = sum(
        float((left[key] - right[key]).square().sum()) for key in left
    )
    denominator = sum(float(right[key].square().sum()) for key in right)
    return math.sqrt(numerator / max(denominator, 1e-24))


def _feature_radius(
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[_MeasuredDevelopmentProbe],
) -> float:
    runtime = plan.prepare(dtype=torch.float64, device="cpu")
    maximum = 0.0
    with torch.no_grad():
        for value in measured:
            features = runtime.encode_features(
                value.modal_coordinates,
                value.null_coordinates,
                value.row_rms,
                value.valid_mask,
            )
            radius = torch.linalg.vector_norm(features[..., 1:], dim=-1)
            maximum = max(
                maximum,
                float(radius[value.valid_mask].max()),
            )
    return (
        maximum * (1.0 + _SUPPORT_RELATIVE_MARGIN)
        + _SUPPORT_ABSOLUTE_MARGIN
    )


def _in_support_fraction(
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[_MeasuredDevelopmentProbe],
    *,
    support_radius: float,
) -> float:
    runtime = plan.prepare(dtype=torch.float64, device="cpu")
    supported = 0
    total = 0
    with torch.no_grad():
        for value in measured:
            features = runtime.encode_features(
                value.modal_coordinates,
                value.null_coordinates,
                value.row_rms,
                value.valid_mask,
            )
            radius = torch.linalg.vector_norm(features[..., 1:], dim=-1)
            selected = radius[value.valid_mask]
            supported += int((selected <= support_radius).sum())
            total += int(selected.numel())
    if total <= 0:
        raise ValueError("support panel contains no valid rows")
    return supported / total


def _structural_metrics(
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[_MeasuredDevelopmentProbe],
    *,
    support_radius: float,
    raw64: Mapping[str, Tensor],
    raw32: Mapping[str, Tensor],
) -> tuple[FullWidthStructuralMetrics, dict[str, object]]:
    runtime = plan.prepare(dtype=torch.float64, device="cpu")
    repeat_numerator = 0.0
    causality_numerator = 0.0
    padding_numerator = 0.0
    denominator = 0.0
    invalid_row_count = 0
    with torch.no_grad():
        for value in measured:
            modal = value.modal_coordinates
            null = value.null_coordinates
            rms = value.row_rms
            mask = value.valid_mask
            positions = value.logical_positions
            baseline = runtime(
                modal,
                null,
                rms,
                valid_mask=mask,
                logical_positions=positions,
            )
            repeated = runtime(
                modal,
                null,
                rms,
                valid_mask=mask,
                logical_positions=positions,
            )
            repeat_numerator += float((baseline - repeated).square().sum())
            denominator += float(baseline[mask].square().sum())

            if value.probe.sequence_length > 1:
                cut = max(1, value.probe.sequence_length // 2)
                changed_modal = modal.clone()
                changed_null = null.clone()
                changed_rms = rms.clone()
                changed_modal[:, cut:] = -1.25 * changed_modal[:, cut:] + 7.0
                changed_null[:, cut:] = 1.75 * changed_null[:, cut:] - 5.0
                changed_rms[:, cut:] = 1.5 * changed_rms[:, cut:] + 0.75
                changed = runtime(
                    changed_modal,
                    changed_null,
                    changed_rms,
                    valid_mask=mask,
                    logical_positions=positions,
                )
                causality_numerator += float(
                    (baseline[:, :cut] - changed[:, :cut]).square().sum()
                )

            prefix = 3
            padded_shape = (
                1,
                value.probe.sequence_length + prefix,
            )
            padded_mask = torch.zeros(padded_shape, dtype=torch.bool)
            padded_mask[:, prefix:] = mask
            padded_positions = torch.full(
                padded_shape,
                -777,
                dtype=torch.int64,
            )
            padded_positions[:, prefix:] = positions
            padded_modal = torch.full(
                (*padded_shape, _MODAL_WIDTH),
                9_973.0,
                dtype=torch.float64,
            )
            padded_modal[:, prefix:] = modal
            padded_null = torch.full(
                (*padded_shape, 1),
                -4_113.0,
                dtype=torch.float64,
            )
            padded_null[:, prefix:] = null
            padded_rms = torch.full(
                padded_shape,
                31.0,
                dtype=torch.float64,
            )
            padded_rms[:, prefix:] = rms
            padded_output = runtime(
                padded_modal,
                padded_null,
                padded_rms,
                valid_mask=padded_mask,
                logical_positions=padded_positions,
            )
            padding_numerator += float(
                (
                    padded_output[:, prefix:] - baseline
                ).square().sum()
            )
            padding_numerator += float(
                padded_output[:, :prefix].square().sum()
            )
            invalid_row_count += prefix

    scale = max(denominator, 1e-24)
    structural = FullWidthStructuralMetrics(
        prepared_vs_analytic_relative_error=_relative_tensor_map_error(
            raw32,
            raw64,
        ),
        causality_violation=math.sqrt(causality_numerator / scale),
        padding_violation=math.sqrt(padding_numerator / scale),
        repeat_relative_error=math.sqrt(repeat_numerator / scale),
        in_support_fraction=_in_support_fraction(
            plan,
            measured,
            support_radius=support_radius,
        ),
    )
    return structural, {
        "support_radius": support_radius,
        "support_rule": (
            "fit_max_l2_radius_of_encoded_nonconstant_features_plus_margin"
        ),
        "invalid_padding_rows_tested": invalid_row_count,
        "nonvacuous_padding_test": invalid_row_count > 0,
    }


def _ordinary_candidate_and_score(
    *,
    candidate_id: str,
    plan: ContrastAwareReferenceProviderPlan,
    measured: Sequence[_MeasuredDevelopmentProbe],
    ordinary_probes: Sequence[FullWidthReferenceProbe],
    controls: FullWidthReferenceControls,
    metric_weight: Tensor,
    standardized_gauge_sha256: str,
    support_radius: float,
    gates: SyntheticReferenceGates,
) -> tuple[
    FullWidthReferenceCandidate,
    FullWidthCandidateScore,
    dict[str, Tensor],
    dict[str, object],
]:
    raw64 = _runtime_predictions(plan, measured, dtype=torch.float64)
    raw32 = _runtime_predictions(plan, measured, dtype=torch.float32)
    structural, structural_metadata = _structural_metrics(
        plan,
        measured,
        support_radius=support_radius,
        raw64=raw64,
        raw32=raw32,
    )
    ordinary_ids = {value.probe_id for value in ordinary_probes}
    predictions = tuple(
        FullWidthCandidatePrediction(
            probe_id=probe_id,
            retained_standardized_prediction=(
                raw64[probe_id] * metric_weight.view(1, 1, -1)
            ),
            standardized_gauge_sha256=standardized_gauge_sha256,
        )
        for probe_id in sorted(ordinary_ids)
    )
    candidate = FullWidthReferenceCandidate(
        candidate_id=candidate_id,
        source_rank=_MODAL_WIDTH,
        target_rank=_MODAL_WIDTH,
        stored_scalar_count=plan.accounting().total_stored_scalar_count,
        predictions=predictions,
        structural_metrics=structural,
        candidate_binding_sha256=plan.artifact_sha256,
    )
    score = score_full_width_reference_candidate(
        controls=controls,
        selection_probes=ordinary_probes,
        collision_probes=(),
        candidate=candidate,
        gates=_deferred_collision_gates(gates),
    )
    return candidate, score, raw64, structural_metadata


def _contrast_assessment(
    *,
    protocol: ContrastProviderDevelopmentProtocol,
    measured: Sequence[_MeasuredDevelopmentProbe],
    predictions: Mapping[str, Tensor],
    metric_weight: Tensor,
    gates: ContrastAssessmentGates,
) -> tuple[
    ContrastAssessmentResult,
    dict[str, dict[str, str]],
    dict[str, object],
]:
    measured_by_id = {value.probe.probe_id: value for value in measured}
    if set(predictions) != set(measured_by_id):
        raise ValueError("contrast predictions do not cover the selection panel")
    observations: list[ContrastObservation] = []
    identities: dict[str, dict[str, str]] = {}
    for group in protocol.groups_for_role("selection"):
        for index, (left_id, right_id) in enumerate(
            group.canonical_variant_pairs
        ):
            contrast_id = f"{group.group_id}.pair.{index:02d}"
            left = measured_by_id[left_id]
            right = measured_by_id[right_id]
            teacher = (
                left.target_replays[0]
                * metric_weight.view(1, 1, -1),
                right.target_replays[0]
                * metric_weight.view(1, 1, -1),
            )
            repeated_teacher = (
                left.target_replays[1]
                * metric_weight.view(1, 1, -1),
                right.target_replays[1]
                * metric_weight.view(1, 1, -1),
            )
            candidate = (
                predictions[left_id] * metric_weight.view(1, 1, -1),
                predictions[right_id] * metric_weight.view(1, 1, -1),
            )
            observations.append(
                ContrastObservation(
                    definition=ContrastDefinition(
                        contrast_id=contrast_id,
                        family=group.family,
                        role=(
                            "expected_sensitivity"
                            if group.intent == "sensitivity"
                            else "intended_null"
                        ),
                        coefficients=(-1.0, 1.0),
                    ),
                    teacher_endpoints=teacher,
                    repeated_teacher_endpoints=repeated_teacher,
                    candidate_endpoints=candidate,
                    repeated_candidate_endpoints=tuple(
                        value.clone() for value in candidate
                    ),
                )
            )
            identities[contrast_id] = {
                "group_id": group.group_id,
                "family": group.family,
                "intent": group.intent,
                "rank_band": group.rank_band,
                "left_probe_id": left_id,
                "right_probe_id": right_id,
            }
    result = assess_state_conditioned_contrasts(
        observations,
        gates=gates,
    )
    scores = {value.contrast_id: value for value in result.contrast_scores}
    if set(scores) != set(identities):
        raise RuntimeError("contrast score identities drifted")
    coverage_rows: dict[str, dict[str, object]] = {}
    for family, intent in (
        ("radial_sensitivity", "sensitivity"),
        ("signed_sensitivity", "sensitivity"),
        ("null_invariance", "invariance"),
    ):
        ids = tuple(
            key
            for key, value in identities.items()
            if value["family"] == family
        )
        qualified_status = (
            "eligible_sensitivity"
            if intent == "sensitivity"
            else "valid_intended_null"
        )
        qualified = tuple(
            key for key in ids if scores[key].teacher_status == qualified_status
        )
        bands = tuple(
            sorted({identities[key]["rank_band"] for key in qualified})
        )
        coverage_rows[family] = {
            "intent": intent,
            "planned_contrast_count": len(ids),
            "teacher_qualified_contrast_count": len(qualified),
            "qualified_rank_bands": bands,
            "all_four_rank_bands_covered": len(bands) == 4,
        }
    coverage = {
        "family_coverage": coverage_rows,
        "all_families_cover_all_four_rank_bands": all(
            bool(value["all_four_rank_bands_covered"])
            for value in coverage_rows.values()
        ),
    }
    return result, identities, coverage


def _mode_packing_diagnostics(
    plan: ContrastAwareReferenceProviderPlan,
) -> dict[str, object]:
    encoder = plan.encoder_weight
    decoder = plan.decoder_weight
    source_energy = encoder.square().sum(dim=1)
    target_energy = decoder.square().sum(dim=0)
    source_norm = source_energy.sqrt()
    target_norm = target_energy.sqrt()
    source_threshold = max(float(source_norm.max()) * 1e-6, 1e-12)
    target_threshold = max(float(target_norm.max()) * 1e-6, 1e-12)
    bands = {
        "band_00_07": (0, 8),
        "band_08_15": (8, 16),
        "band_16_31": (16, 32),
        "band_32_63": (32, 64),
    }
    source_total = max(float(source_energy.sum()), 1e-24)
    target_total = max(float(target_energy.sum()), 1e-24)
    encoder_fan_in: list[int] = []
    for latent in range(plan.latent_rank):
        column = encoder[:, latent].abs()
        threshold = max(float(column.max()) * 0.10, 1e-12)
        encoder_fan_in.append(int((column >= threshold).sum()))
    decoder_fan_out: list[int] = []
    for latent in range(plan.latent_rank):
        row = decoder[latent].abs()
        threshold = max(float(row.max()) * 0.10, 1e-12)
        decoder_fan_out.append(int((row >= threshold).sum()))
    encoder_singular = torch.linalg.svdvals(encoder)
    decoder_singular = torch.linalg.svdvals(decoder)
    return {
        "visible_source_modes": _MODAL_WIDTH,
        "visible_target_modes": _MODAL_WIDTH,
        "latent_rank": plan.latent_rank,
        "active_source_modes_exact_nonzero": (
            plan.active_encoder_source_modes
        ),
        "active_target_modes_exact_nonzero": (
            plan.active_decoder_target_modes
        ),
        "active_source_modes_relative_threshold": int(
            (source_norm >= source_threshold).sum()
        ),
        "active_target_modes_relative_threshold": int(
            (target_norm >= target_threshold).sum()
        ),
        "source_energy_fraction_by_rank_band": {
            name: float(source_energy[start:end].sum()) / source_total
            for name, (start, end) in bands.items()
        },
        "target_energy_fraction_by_rank_band": {
            name: float(target_energy[start:end].sum()) / target_total
            for name, (start, end) in bands.items()
        },
        "encoder_effective_singular_values": tuple(
            float(value) for value in encoder_singular
        ),
        "decoder_effective_singular_values": tuple(
            float(value) for value in decoder_singular
        ),
        "encoder_fan_in_at_10pct_of_latent_peak": {
            "minimum": min(encoder_fan_in),
            "mean": sum(encoder_fan_in) / len(encoder_fan_in),
            "maximum": max(encoder_fan_in),
        },
        "decoder_fan_out_at_10pct_of_latent_peak": {
            "minimum": min(decoder_fan_out),
            "mean": sum(decoder_fan_out) / len(decoder_fan_out),
            "maximum": max(decoder_fan_out),
        },
        "prefix_deletion_used": False,
        "nonadjacent_modal_packing_available": True,
    }


def _execution_accounting(
    plan: ContrastAwareReferenceProviderPlan,
    batches: Sequence[IndexedReferenceBatch],
) -> dict[str, object]:
    runtime = plan.prepare(dtype=torch.float64, device="cpu")
    fields = (
        "encoder_mac_count",
        "decoder_mac_count",
        "target_destandardization_mac_count",
        "total_mac_count",
    )
    totals = {name: 0 for name in fields}
    core_total = 0
    valid_rows = 0
    for indexed in batches:
        batch = indexed.batch
        accounting = runtime.execution_accounting(
            valid_mask=batch.valid_mask,
            logical_positions=batch.logical_positions,
        )
        valid_rows += accounting.valid_rows
        core_total += accounting.core.total_mac_count
        for name in fields:
            totals[name] += int(getattr(accounting, name))
    canonical_mask = torch.ones(1, 128, dtype=torch.bool)
    canonical_positions = torch.arange(128, dtype=torch.int64).view(1, -1)
    canonical = runtime.execution_accounting(
        valid_mask=canonical_mask,
        logical_positions=canonical_positions,
    )
    return {
        "selection_panel_valid_rows": valid_rows,
        "selection_panel_core_mac_count": core_total,
        "selection_panel_encoder_mac_count": totals["encoder_mac_count"],
        "selection_panel_decoder_mac_count": totals["decoder_mac_count"],
        "selection_panel_target_destandardization_mac_count": totals[
            "target_destandardization_mac_count"
        ],
        "selection_panel_total_mac_count": totals["total_mac_count"],
        "macs_per_valid_row_over_selection_panel": (
            totals["total_mac_count"] / valid_rows
        ),
        "canonical_sequence_length": 128,
        "canonical_batch_size": 1,
        "canonical_total_mac_count": canonical.total_mac_count,
        "canonical_core_mac_count": canonical.core.total_mac_count,
        "canonical_encoder_mac_count": canonical.encoder_mac_count,
        "canonical_decoder_mac_count": canonical.decoder_mac_count,
        "canonical_target_destandardization_mac_count": (
            canonical.target_destandardization_mac_count
        ),
        "semantics": (
            "ideal_sparse_mathematical_MACs_not_wall_clock_or_kernel_latency"
        ),
    }


def _validate_output_path(path: Path | str) -> Path:
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError("development output must use a .pt suffix")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite development output")
    worktree = find_git_worktree(Path(__file__))
    resolved = destination.expanduser().resolve()
    if worktree is not None:
        root = worktree.resolve()
        if resolved == root or root in resolved.parents:
            relative = resolved.relative_to(root)
            if not relative.parts or relative.parts[0] not in {
                ".local-runs",
                "local-runs",
            }:
                raise ValueError(
                    "worktree outputs must remain under ignored local-runs"
                )
    return resolved


def _stage_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish_artifact(
    state: Mapping[str, object],
    report_payload: Mapping[str, object],
    *,
    output: Path,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite development output")
    tensor_stage = _stage_path(output)
    report_stage = _stage_path(report_path)
    published: list[Path] = []
    try:
        torch.save(dict(state), tensor_stage)
        report = {
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
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tensor_stage, output)
        published.append(output)
        os.link(report_stage, report_path)
        published.append(report_path)
        return report
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        tensor_stage.unlink(missing_ok=True)
        report_stage.unlink(missing_ok=True)


def describe_contrast_provider_development() -> dict[str, object]:
    """Describe the complete development ladder without opening live targets."""

    protocol = default_contrast_provider_development_protocol()
    objective = _objective()
    code_sha256s = _code_sha256s()
    return {
        "schema": f"{_SCHEMA}.description",
        "format_version": _FORMAT_VERSION,
        "protocol_sha256": protocol.protocol_sha256,
        "protocol_trust_anchor": DEFAULT_DEVELOPMENT_PROTOCOL_SHA256,
        "role_panel_sha256s": {
            role: protocol.panel_sha256(role)  # type: ignore[arg-type]
            for role in ("pilot", "fit", "selection")
        },
        "role_probe_counts": {
            role: len(protocol.probes_for_role(role))  # type: ignore[arg-type]
            for role in ("pilot", "fit", "selection")
        },
        "rank_ladder": protocol.rank_ladder,
        "candidate_ids": protocol.candidate_ids,
        "rank_semantics": "all_64_modes_to_latent_r_to_all_64_modes",
        "objective": objective.state_dict(),
        "training": _training_spec(),
        "training_sha256": _training_sha256(),
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": _code_bundle_sha256(code_sha256s),
        "model_loaded": False,
        "pilot_materialized": False,
        "fit_materialized": False,
        "selection_materialized": False,
        "teacher_target_opened": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
    }


def _candidate_failure_reasons(
    *,
    ordinary_score: FullWidthCandidateScore,
    contrast_result: ContrastAssessmentResult,
    coverage: Mapping[str, object],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not ordinary_score.passed:
        reasons.extend(
            f"ordinary:{name}"
            for name, passed in ordinary_score.gate_flags.state_dict().items()
            if name != "all_passed" and passed is False
        )
    if contrast_result.overall_status != "pass":
        reasons.append(f"contrast:{contrast_result.overall_status}")
        reasons.extend(
            f"contrast:{value}" for value in contrast_result.reason_codes
        )
    if not bool(coverage["all_families_cover_all_four_rank_bands"]):
        reasons.append("contrast:teacher_coverage_missing_rank_band")
    return tuple(sorted(set(reasons)))


def compile_contrast_provider_development(
    *,
    basis_package_path: Path | str = DEFAULT_BASIS_PACKAGE,
    basis_package_file_sha256: str = DEFAULT_BASIS_PACKAGE_FILE_SHA256,
    basis_package_payload_sha256: str = (
        DEFAULT_BASIS_PACKAGE_PAYLOAD_SHA256
    ),
    output: Path | str = DEFAULT_OUTPUT,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
) -> dict[str, object]:
    """Fit all ranks, freeze them, and run disjoint development selection."""

    protocol = default_contrast_provider_development_protocol()
    if (
        not DEFAULT_DEVELOPMENT_PROTOCOL_SHA256
        or protocol.protocol_sha256 != DEFAULT_DEVELOPMENT_PROTOCOL_SHA256
    ):
        raise ValueError("development protocol trust anchor drifted")
    objective = _objective()
    fidelity_gates = SyntheticReferenceGates()
    contrast_gates = ContrastAssessmentGates()
    destination = _validate_output_path(output)
    code_sha256s = _code_sha256s()
    code_bundle_sha256 = _code_bundle_sha256(code_sha256s)

    (
        basis,
        adapter,
        pre_ff3,
        post_ff3,
        epsilon,
    ) = _load_live_dependencies(
        basis_package_path=basis_package_path,
        basis_package_file_sha256=basis_package_file_sha256,
        basis_package_payload_sha256=basis_package_payload_sha256,
        model_id=DEFAULT_MODEL_ID,
        revision=DEFAULT_REVISION,
        cache_dir=cache_dir,
        device_name=device_name,
        dtype=dtype,
    )
    model_before = adapter.model_fingerprint()
    norm_sha256 = module_state_fingerprint(pre_ff3)
    metric_weight = _fisher_metric_weight(basis)

    # Fit-only pilot: its sole authority is choosing h.  It is not used to fit
    # any candidate coefficient.
    pilot, pilot_measurement = _measure_role(
        role="pilot",
        protocol=protocol,
        calibration=None,
        frozen_candidates=None,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
        replay_count=2,
    )
    pilot_metrics = _calibration_metrics(
        protocol=protocol,
        measured=pilot,
        metric_weight=metric_weight,
    )
    calibration = select_global_calibration_amplitude(
        protocol,
        pilot_metrics,
    )

    fit, fit_measurement = _measure_role(
        role="fit",
        protocol=protocol,
        calibration=calibration,
        frozen_candidates=None,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
        replay_count=1,
    )
    (
        modal_center,
        gain_log_center,
        gain_log_scale,
        target_center,
        target_scale,
    ) = _fit_gauges(
        fit,
        residual_width=basis.residual_width,
        epsilon=epsilon,
    )
    standardized_gauge_sha256 = _standardized_gauge_sha256(
        basis=basis,
        protocol=protocol,
        calibration=calibration,
        objective=objective,
        metric_weight=metric_weight,
    )
    binding_sha256 = _provider_binding_sha256(
        basis=basis,
        protocol=protocol,
        calibration=calibration,
        objective=objective,
        norm_sha256=norm_sha256,
        metric_weight=metric_weight,
    )
    fit_batches = _indexed_batches(
        fit,
        split="fit",
        binding_sha256=binding_sha256,
    )
    fit_pairs, fit_chart_mismatch_diagnostics = _training_contrast_pairs(
        protocol=protocol,
        measured=fit,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
    )

    # FIREWALL: all three candidates are fit and frozen before selection is
    # materialized or any selection teacher target exists.
    plans: dict[str, ContrastAwareReferenceProviderPlan] = {}
    support_radii: dict[str, float] = {}
    for index, (candidate_id, rank) in enumerate(
        zip(protocol.candidate_ids, protocol.rank_ladder, strict=True)
    ):
        plan = fit_contrast_aware_reference_provider(
            modal_center=modal_center,
            gain_log_center=gain_log_center,
            gain_log_scale=gain_log_scale,
            residual_width=basis.residual_width,
            rms_epsilon=epsilon,
            target_center=target_center,
            target_scale=target_scale,
            fit_batches=fit_batches,
            contrast_pairs=fit_pairs,
            executor_config=_executor_config(rank),
            objective=objective,
            fisher_metric_weight=metric_weight,
            steps=_TRAINING_STEPS,
            learning_rate=_LEARNING_RATE,
            seed=_BASE_SEED + index,
        )
        plans[candidate_id] = plan
        support_radii[candidate_id] = _feature_radius(plan, fit)
    frozen_candidates = freeze_development_candidates(
        protocol,
        calibration,
        tuple(
            plans[candidate_id].artifact_sha256
            for candidate_id in protocol.candidate_ids
        ),
    )

    selection, selection_measurement = _measure_role(
        role="selection",
        protocol=protocol,
        calibration=calibration,
        frozen_candidates=frozen_candidates,
        basis=basis,
        adapter=adapter,
        pre_ff3=pre_ff3,
        post_ff3=post_ff3,
        epsilon=epsilon,
        replay_count=2,
    )
    fit_ordinary = _ordinary_full_width_probes(
        fit,
        split="fit",
        metric_weight=metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
    )
    controls = fit_full_width_reference_controls(
        fit_probes=fit_ordinary,
        position_bin_count=16,
    )
    selection_ordinary = _ordinary_full_width_probes(
        selection,
        split="selection",
        metric_weight=metric_weight,
        standardized_gauge_sha256=standardized_gauge_sha256,
    )
    selection_batches = _indexed_batches(
        selection,
        split="selection",
        binding_sha256=binding_sha256,
    )

    candidate_rows: dict[str, dict[str, object]] = {}
    ordinary_scores: dict[str, FullWidthCandidateScore] = {}
    contrast_results: dict[str, ContrastAssessmentResult] = {}
    passing: list[str] = []
    for candidate_id in protocol.candidate_ids:
        plan = plans[candidate_id]
        candidate, ordinary_score, predictions, structural_metadata = (
            _ordinary_candidate_and_score(
                candidate_id=candidate_id,
                plan=plan,
                measured=selection,
                ordinary_probes=selection_ordinary,
                controls=controls,
                metric_weight=metric_weight,
                standardized_gauge_sha256=standardized_gauge_sha256,
                support_radius=support_radii[candidate_id],
                gates=fidelity_gates,
            )
        )
        contrast_result, identities, coverage = _contrast_assessment(
            protocol=protocol,
            measured=selection,
            predictions=predictions,
            metric_weight=metric_weight,
            gates=contrast_gates,
        )
        combined_pass = (
            ordinary_score.passed
            and contrast_result.overall_status == "pass"
            and bool(coverage["all_families_cover_all_four_rank_bands"])
        )
        if combined_pass:
            passing.append(candidate_id)
        accounting = asdict(plan.accounting())
        execution = _execution_accounting(plan, selection_batches)
        candidate_rows[candidate_id] = {
            "candidate_id": candidate_id,
            "latent_rank": plan.latent_rank,
            "visible_source_modes": _MODAL_WIDTH,
            "visible_target_modes": _MODAL_WIDTH,
            "stored_scalar_count": accounting["total_stored_scalar_count"],
            "accounting": accounting,
            "execution_accounting": execution,
            "initial_training_metrics": plan.initial_metrics.state_dict(),
            "final_training_metrics": plan.final_metrics.state_dict(),
            "ordinary_score": ordinary_score.state_dict(),
            "contrast_result": contrast_result.state_dict(),
            "contrast_coverage": coverage,
            "contrast_identities": identities,
            "structural_metadata": structural_metadata,
            "mode_packing": _mode_packing_diagnostics(plan),
            "combined_selection_pass": combined_pass,
            "failure_reasons": _candidate_failure_reasons(
                ordinary_score=ordinary_score,
                contrast_result=contrast_result,
                coverage=coverage,
            ),
            "candidate_binding_sha256": candidate.artifact_sha256,
            "plan_sha256": plan.artifact_sha256,
        }
        ordinary_scores[candidate_id] = ordinary_score
        contrast_results[candidate_id] = contrast_result

    selected_id = (
        min(
            passing,
            key=lambda value: (
                plans[value].accounting().total_stored_scalar_count,
                plans[value].latent_rank,
                value,
            ),
        )
        if passing
        else None
    )
    if (
        adapter.model_fingerprint() != model_before
        or module_state_fingerprint(pre_ff3) != norm_sha256
        or _code_sha256s() != code_sha256s
    ):
        raise RuntimeError(
            "model, normalization, or code changed during development run"
        )

    manifest = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "basis_package_file_sha256": basis_package_file_sha256,
        "basis_package_payload_sha256": basis.basis_payload_sha256,
        "source_model_sha256": basis.source_model_sha256,
        "protocol_sha256": protocol.protocol_sha256,
        "pilot_panel_sha256": protocol.panel_sha256("pilot"),
        "calibrated_fit_panel_sha256": (
            protocol.calibrated_panel_sha256("fit", calibration)
        ),
        "calibrated_selection_panel_sha256": (
            protocol.calibrated_panel_sha256("selection", calibration)
        ),
        "calibration_sha256": calibration.artifact_sha256,
        "selected_calibration_amplitude": calibration.selected_amplitude,
        "training_sha256": _training_sha256(),
        "objective_sha256": objective.artifact_sha256,
        "standardized_gauge_sha256": standardized_gauge_sha256,
        "metric_weight_sha256": _tensor_sha256(metric_weight),
        "provider_binding_sha256": binding_sha256,
        "pre_feedforward_norm_sha256": norm_sha256,
        "candidate_set_sha256": frozen_candidates.artifact_sha256,
        "candidate_plan_sha256s": {
            value: plans[value].artifact_sha256
            for value in protocol.candidate_ids
        },
        "candidate_stored_scalar_counts": {
            value: plans[value].accounting().total_stored_scalar_count
            for value in protocol.candidate_ids
        },
        "controls_sha256": controls.artifact_sha256,
        "ordinary_score_sha256s": {
            value: ordinary_scores[value].artifact_sha256
            for value in protocol.candidate_ids
        },
        "contrast_result_sha256s": {
            value: contrast_results[value].artifact_sha256
            for value in protocol.candidate_ids
        },
        "selected_candidate_id": selected_id,
        "selected_plan_sha256": (
            None if selected_id is None else plans[selected_id].artifact_sha256
        ),
        "all_candidates_frozen_before_selection_materialization": True,
        "selection_data_changed_training": False,
        "v2_targets_loaded": False,
        "v3_targets_loaded": False,
        "prompt_text_loaded": False,
        "token_ids_loaded": False,
        "tokenizer_loaded": False,
        "natural_activation_rows_loaded": False,
        "code_sha256s": code_sha256s,
        "code_bundle_sha256": code_bundle_sha256,
        "scientific_scope": (
            "open_development_prompt_blind_synthetic_provider_selection"
        ),
    }
    logical_artifact_sha256 = _json_sha256(
        manifest,
        domain=_ARTIFACT_DOMAIN,
    )
    state = {
        "manifest": manifest,
        "artifact_sha256": logical_artifact_sha256,
        "metric_weight": metric_weight,
        "protocol_state": protocol.state_dict(),
        "calibration_state": calibration.state_dict(),
        "frozen_candidate_set_state": frozen_candidates.state_dict(),
        "objective_state": objective.state_dict(),
        "training_spec": _training_spec(),
        "controls_state": controls.state_dict(),
        "plan_states": {
            value: plans[value].state_dict()
            for value in protocol.candidate_ids
        },
        "candidate_results": candidate_rows,
    }
    ordered_rows = [
        candidate_rows[value] for value in protocol.candidate_ids
    ]
    largest_storage = max(
        int(value["stored_scalar_count"]) for value in ordered_rows
    )
    largest_macs = max(
        int(
            value["execution_accounting"]["canonical_total_mac_count"]  # type: ignore[index]
        )
        for value in ordered_rows
    )
    rate_curve = [
        {
            **value,
            "stored_scalar_reduction_vs_largest_tested_rank": (
                1.0 - int(value["stored_scalar_count"]) / largest_storage
            ),
            "canonical_mac_reduction_vs_largest_tested_rank": (
                1.0
                - int(
                    value["execution_accounting"][  # type: ignore[index]
                        "canonical_total_mac_count"
                    ]
                )
                / largest_macs
            ),
        }
        for value in ordered_rows
    ]
    report_payload = {
        **manifest,
        "artifact_sha256": logical_artifact_sha256,
        "training": _training_spec(),
        "objective": objective.state_dict(),
        "calibration": calibration.state_dict(),
        "pilot_metrics": [
            value.state_dict() for value in pilot_metrics
        ],
        "pilot_measurement": pilot_measurement,
        "fit_measurement": fit_measurement,
        "fit_provider_chart_mismatch_diagnostics": (
            fit_chart_mismatch_diagnostics
        ),
        "selection_measurement": selection_measurement,
        "rate_curve": rate_curve,
        "selection_outcome": (
            "selected_smallest_combined_passer"
            if selected_id is not None
            else "no_candidate_passed_combined_gates"
        ),
        "interpretation": {
            "prefix_deletion_used": False,
            "all_64_source_modes_visible_to_every_candidate": True,
            "all_64_target_modes_reconstructed_by_every_candidate": True,
            "latent_rank_is_dense_modal_packing_width": True,
            "exact_gain_null_omitted_structurally": True,
            "selection_is_held_out_from_fit": True,
            "v4_assessment_opened": False,
            "natural_prompt_fidelity_claim": False,
            "whole_model_replacement_claim": False,
            "wall_clock_speed_claim": False,
            "whole_model_compression_claim": False,
        },
        "safety": {
            "contains_source_model_state_dict": False,
            "contains_provider_parameters": True,
            "contains_raw_teacher_targets": False,
            "contains_teacher_jvp_tensors": False,
            "contains_provider_chart_jvp_tensors": False,
            "contains_prompt_text": False,
            "contains_token_ids": False,
            "committable": False,
        },
    }
    return _publish_artifact(state, report_payload, output=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "describe",
        help="describe the frozen development ladder without opening targets",
    )
    compile_parser = commands.add_parser(
        "compile",
        help="run fit-only calibration, fit all ranks, then open selection",
    )
    compile_parser.add_argument(
        "--basis-package",
        type=Path,
        default=DEFAULT_BASIS_PACKAGE,
    )
    compile_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    compile_parser.add_argument("--cache-dir", type=Path)
    compile_parser.add_argument("--device", default="cpu")
    compile_parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe":
        report = describe_contrast_provider_development()
    else:
        report = compile_contrast_provider_development(
            basis_package_path=args.basis_package,
            output=args.output,
            cache_dir=args.cache_dir,
            device_name=args.device,
            dtype=args.dtype,
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
