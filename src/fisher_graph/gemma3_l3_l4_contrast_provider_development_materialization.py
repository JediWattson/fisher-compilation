"""Authenticated tensor materialization for the contrast-provider development.

The protocol module is specification-only.  This module is the first place
Torch is imported.  It deterministically expands those specifications into
standardized 64-mode direction tensors and binds the fit-only calibration
choice into every fit/selection batch.

Selection is fail-closed: its tensors cannot be requested without both the
calibration binding and the exact three-candidate frozen-set binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor

from .gemma3_l3_l4_contrast_provider_development_protocol import (
    ContrastProviderDevelopmentProtocol,
    DevelopmentCalibrationBinding,
    DevelopmentProbeSpec,
    FrozenDevelopmentCandidateSet,
    calibrated_role_panel_sha256,
    default_contrast_provider_development_protocol,
)


__all__ = [
    "MaterializedDevelopmentBatch",
    "materialize_development_probe",
    "materialize_development_role",
]


DevelopmentRole = Literal["pilot", "fit", "selection"]
_MODAL_WIDTH = 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROW_NORM_ATOL = 1e-12
_TENSOR_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"materialized-tensor:v1\0"
)
_BATCH_DOMAIN = (
    b"fisher-graph:gemma3-l3-l4-contrast-provider-development-c2:"
    b"materialized-batch:v1\0"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest(value: object, *, domain: bytes) -> str:
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
            "shape": tuple(int(width) for width in tensor.shape),
        }
    )
    return hashlib.sha256(
        _TENSOR_DOMAIN
        + header
        + b"\0"
        + tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def _canonical_float_tensor(
    value: Tensor,
    *,
    label: str,
    ndim: int,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.ndim != ndim
        or value.numel() == 0
    ):
        raise TypeError(f"{label} must be a nonempty floating rank-{ndim} Tensor")
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite")
    return result


def _authenticate_protocol(
    protocol: ContrastProviderDevelopmentProtocol,
) -> ContrastProviderDevelopmentProtocol:
    if not isinstance(protocol, ContrastProviderDevelopmentProtocol):
        raise TypeError("protocol must be a development protocol")
    expected = default_contrast_provider_development_protocol()
    if protocol.protocol_sha256 != expected.protocol_sha256 or protocol != expected:
        raise ValueError("materializer requires the default development protocol")
    return expected


def _authenticate_probe(
    protocol: ContrastProviderDevelopmentProtocol,
    probe: DevelopmentProbeSpec,
) -> DevelopmentProbeSpec:
    exact_protocol = _authenticate_protocol(protocol)
    if not isinstance(probe, DevelopmentProbeSpec):
        raise TypeError("probe must be a DevelopmentProbeSpec")
    matches = tuple(
        value
        for value in exact_protocol.probes
        if value.artifact_sha256 == probe.artifact_sha256
    )
    if len(matches) != 1 or matches[0] != probe:
        raise ValueError("probe is not an exact development protocol member")
    return matches[0]


def _authenticate_calibration(
    protocol: ContrastProviderDevelopmentProtocol,
    calibration: DevelopmentCalibrationBinding,
) -> None:
    if not isinstance(calibration, DevelopmentCalibrationBinding):
        raise TypeError("fit/selection requires a calibration binding")
    if (
        calibration.protocol_sha256 != protocol.protocol_sha256
        or calibration.pilot_panel_sha256 != protocol.panel_sha256("pilot")
        or calibration.calibration_rule_sha256
        != protocol.calibration_rule.artifact_sha256
    ):
        raise ValueError("calibration binding does not match the protocol")


def _authenticate_context(
    protocol: ContrastProviderDevelopmentProtocol,
    role: DevelopmentRole,
    *,
    calibration: DevelopmentCalibrationBinding | None,
    frozen_candidates: FrozenDevelopmentCandidateSet | None,
) -> tuple[str, str | None, float | None]:
    if role == "pilot":
        if calibration is not None or frozen_candidates is not None:
            raise ValueError("pilot materialization accepts no later-role binding")
        return protocol.panel_sha256("pilot"), None, None
    if calibration is None:
        raise ValueError(f"{role} materialization requires calibration")
    _authenticate_calibration(protocol, calibration)
    panel_sha = calibrated_role_panel_sha256(protocol, role, calibration)
    if role == "fit":
        if frozen_candidates is not None:
            raise ValueError("fit materialization precedes candidate freezing")
        return panel_sha, None, calibration.selected_amplitude
    if not isinstance(frozen_candidates, FrozenDevelopmentCandidateSet):
        raise ValueError(
            "selection materialization requires all candidates to be frozen"
        )
    expected_fit_panel = calibrated_role_panel_sha256(
        protocol,
        "fit",
        calibration,
    )
    if (
        frozen_candidates.protocol_sha256 != protocol.protocol_sha256
        or frozen_candidates.calibration_sha256
        != calibration.artifact_sha256
        or frozen_candidates.calibrated_fit_panel_sha256
        != expected_fit_panel
        or frozen_candidates.rank_ladder != protocol.rank_ladder
        or frozen_candidates.candidate_ids != protocol.candidate_ids
    ):
        raise ValueError("frozen candidate set does not match development fit")
    return (
        panel_sha,
        frozen_candidates.artifact_sha256,
        calibration.selected_amplitude,
    )


def _raw_multitone(probe: DevelopmentProbeSpec) -> Tensor:
    temporal = probe.multitone_temporal_frequencies
    modal = probe.multitone_modal_frequencies
    phase = probe.multitone_phase_quadrants
    if temporal is None or modal is None or phase is None:
        raise ValueError("multitone probe lacks frozen frequency metadata")
    raw = torch.zeros(
        probe.sequence_length,
        _MODAL_WIDTH,
        dtype=torch.float64,
    )
    modal_positions = (
        torch.arange(_MODAL_WIDTH, dtype=torch.float64) + 0.5
    ) / _MODAL_WIDTH
    active_length = probe.sequence_length - probe.source_offset
    for position in range(probe.source_offset, probe.sequence_length):
        relative = position - probe.source_offset
        tau = (relative + 0.5) / active_length
        first = (
            0.5 * temporal[0] * tau * tau
            + modal[0] * modal_positions
            + phase[0] / 4.0
        )
        second = (
            0.5 * temporal[1] * tau * tau
            + modal[1] * modal_positions
            + phase[1] / 4.0
        )
        raw[position] = torch.cos(2.0 * math.pi * first) + (
            0.5 * torch.cos(2.0 * math.pi * second)
        )
    return raw


def _raw_block_sparse(probe: DevelopmentProbeSpec) -> Tensor:
    raw = torch.zeros(
        probe.sequence_length,
        _MODAL_WIDTH,
        dtype=torch.float64,
    )
    for block in probe.sparse_blocks:
        for position in range(block.start, block.start + block.length):
            for mode, sign in block.mode_signs:
                raw[position, mode] = float(sign)
    return raw


def _raw_axis(probe: DevelopmentProbeSpec) -> Tensor:
    if probe.axis_mode is None or probe.axis_sign is None:
        raise ValueError("axis probe lacks a mode/sign coordinate")
    raw = torch.zeros(
        probe.sequence_length,
        _MODAL_WIDTH,
        dtype=torch.float64,
    )
    end = probe.source_offset + probe.active_block_length
    raw[probe.source_offset:end, probe.axis_mode] = float(probe.axis_sign)
    return raw


def _raw_direction(probe: DevelopmentProbeSpec) -> Tensor:
    if probe.family == "multitone":
        return _raw_multitone(probe)
    if probe.family == "block_sparse":
        return _raw_block_sparse(probe)
    return _raw_axis(probe)


def _effective_amplitude(
    probe: DevelopmentProbeSpec,
    calibration: DevelopmentCalibrationBinding | None,
) -> float:
    if not probe.uses_calibrated_amplitude:
        return probe.modal_amplitude
    if calibration is None:
        raise ValueError("calibrated probe requires a calibration binding")
    return probe.modal_amplitude * calibration.selected_amplitude


def materialize_development_probe(
    protocol: ContrastProviderDevelopmentProtocol,
    probe: DevelopmentProbeSpec,
    *,
    calibration: DevelopmentCalibrationBinding | None = None,
    frozen_candidates: FrozenDevelopmentCandidateSet | None = None,
) -> Tensor:
    """Materialize one authenticated endpoint as float64 ``[1, S, 64]``."""

    exact = _authenticate_probe(protocol, probe)
    _authenticate_context(
        protocol,
        exact.role,
        calibration=calibration,
        frozen_candidates=frozen_candidates,
    )
    raw = _raw_direction(exact)
    norms = torch.linalg.vector_norm(raw, dim=-1)
    active = norms > 0.0
    if not bool(active.any()):
        raise ValueError("development probe materialized no active rows")
    amplitude = _effective_amplitude(exact, calibration)
    normalized = torch.zeros_like(raw)
    normalized[active] = (
        raw[active] / norms[active].unsqueeze(-1) * amplitude
    )
    values = normalized.unsqueeze(0).contiguous()
    if bool((values[:, : exact.source_offset] != 0.0).any()):
        raise ValueError("development probe is active before source offset")
    row_norms = torch.linalg.vector_norm(values[0], dim=-1)
    if not torch.allclose(
        row_norms[active],
        torch.full_like(row_norms[active], amplitude),
        rtol=0.0,
        atol=_ROW_NORM_ATOL,
    ):
        raise ValueError("development probe violates its effective amplitude")
    if bool((row_norms[~active] != 0.0).any()):
        raise ValueError("development probe inactive rows are nonzero")
    return values


@dataclass(frozen=True, slots=True)
class MaterializedDevelopmentBatch:
    """One authenticated equal-length development batch."""

    role: DevelopmentRole
    protocol_sha256: str
    role_panel_sha256: str
    calibration_sha256: str | None
    selected_amplitude: float | None
    candidate_set_sha256: str | None
    sequence_length: int
    probe_ids: tuple[str, ...]
    probe_artifact_sha256s: tuple[str, ...]
    probe_tensor_sha256s: tuple[str, ...]
    modal_amplitudes: Tensor
    radial_scales: Tensor
    null_coordinates: Tensor
    values: Tensor
    tensor_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"pilot", "fit", "selection"}:
            raise ValueError("materialized development role is invalid")
        _require_sha256(self.protocol_sha256, label="protocol hash")
        _require_sha256(self.role_panel_sha256, label="role panel hash")
        if self.role == "pilot":
            if (
                self.calibration_sha256 is not None
                or self.selected_amplitude is not None
                or self.candidate_set_sha256 is not None
            ):
                raise ValueError("pilot batch carries a later-role binding")
        else:
            _require_sha256(
                self.calibration_sha256,
                label="calibration hash",
            )
            if (
                isinstance(self.selected_amplitude, bool)
                or not isinstance(self.selected_amplitude, (int, float))
                or not math.isfinite(float(self.selected_amplitude))
                or float(self.selected_amplitude) <= 0.0
            ):
                raise ValueError("calibrated batch amplitude is invalid")
            if self.role == "fit" and self.candidate_set_sha256 is not None:
                raise ValueError("fit batch carries a frozen candidate set")
            if self.role == "selection":
                _require_sha256(
                    self.candidate_set_sha256,
                    label="candidate-set hash",
                )
        if type(self.sequence_length) is not int or self.sequence_length <= 0:
            raise ValueError("batch sequence length must be positive")
        count = len(self.probe_ids)
        if (
            count == 0
            or type(self.probe_ids) is not tuple
            or len(set(self.probe_ids)) != count
            or len(self.probe_artifact_sha256s) != count
            or len(self.probe_tensor_sha256s) != count
        ):
            raise ValueError("batch probe identities are invalid")
        for values, label in (
            (self.probe_artifact_sha256s, "probe artifact hash"),
            (self.probe_tensor_sha256s, "probe tensor hash"),
        ):
            for value in values:
                _require_sha256(value, label=label)
        amplitudes = _canonical_float_tensor(
            self.modal_amplitudes,
            label="modal amplitudes",
            ndim=1,
        )
        radial = _canonical_float_tensor(
            self.radial_scales,
            label="radial scales",
            ndim=1,
        )
        null = _canonical_float_tensor(
            self.null_coordinates,
            label="null coordinates",
            ndim=1,
        )
        values = _canonical_float_tensor(
            self.values,
            label="materialized values",
            ndim=3,
        )
        if (
            amplitudes.shape != (count,)
            or radial.shape != (count,)
            or null.shape != (count,)
            or values.shape != (count, self.sequence_length, _MODAL_WIDTH)
            or bool((amplitudes <= 0.0).any())
            or bool((radial <= 0.0).any())
        ):
            raise ValueError("batch tensor geometry is invalid")
        row_hashes = tuple(
            _tensor_sha256(values[index : index + 1])
            for index in range(count)
        )
        if row_hashes != self.probe_tensor_sha256s:
            raise ValueError("batch per-probe tensor hashes do not match values")
        tensor_hash = _tensor_sha256(values)
        if self.tensor_sha256 and self.tensor_sha256 != tensor_hash:
            raise ValueError("batch tensor hash mismatch")
        object.__setattr__(self, "modal_amplitudes", amplitudes)
        object.__setattr__(self, "radial_scales", radial)
        object.__setattr__(self, "null_coordinates", null)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "tensor_sha256", tensor_hash)
        computed = _digest(self._payload(), domain=_BATCH_DOMAIN)
        if self.artifact_sha256 and self.artifact_sha256 != computed:
            raise ValueError("materialized batch artifact hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed)

    @property
    def batch_size(self) -> int:
        return len(self.probe_ids)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": "fisher_graph.materialized_development_c2_batch.v1",
            "format_version": 1,
            "role": self.role,
            "protocol_sha256": self.protocol_sha256,
            "role_panel_sha256": self.role_panel_sha256,
            "calibration_sha256": self.calibration_sha256,
            "selected_amplitude": self.selected_amplitude,
            "candidate_set_sha256": self.candidate_set_sha256,
            "sequence_length": self.sequence_length,
            "probe_ids": list(self.probe_ids),
            "probe_artifact_sha256s": list(
                self.probe_artifact_sha256s
            ),
            "probe_tensor_sha256s": list(self.probe_tensor_sha256s),
            "modal_amplitudes_sha256": _tensor_sha256(
                self.modal_amplitudes
            ),
            "radial_scales_sha256": _tensor_sha256(self.radial_scales),
            "null_coordinates_sha256": _tensor_sha256(
                self.null_coordinates
            ),
            "tensor_sha256": self.tensor_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def validate_integrity(self) -> None:
        if _tensor_sha256(self.values) != self.tensor_sha256:
            raise ValueError("materialized batch tensor was mutated")
        if _digest(self._payload(), domain=_BATCH_DOMAIN) != self.artifact_sha256:
            raise ValueError("materialized batch metadata was mutated")


def materialize_development_role(
    protocol: ContrastProviderDevelopmentProtocol,
    role: DevelopmentRole,
    *,
    calibration: DevelopmentCalibrationBinding | None = None,
    frozen_candidates: FrozenDevelopmentCandidateSet | None = None,
) -> tuple[MaterializedDevelopmentBatch, ...]:
    """Materialize a complete role, grouped into equal-length batches."""

    exact = _authenticate_protocol(protocol)
    if role not in {"pilot", "fit", "selection"}:
        raise ValueError("development role is invalid")
    panel_sha, candidate_sha, selected_amplitude = _authenticate_context(
        exact,
        role,
        calibration=calibration,
        frozen_candidates=frozen_candidates,
    )
    by_length: dict[int, list[DevelopmentProbeSpec]] = {}
    for probe in exact.probes_for_role(role):
        by_length.setdefault(probe.sequence_length, []).append(probe)
    result: list[MaterializedDevelopmentBatch] = []
    for length in sorted(by_length):
        probes = tuple(
            sorted(by_length[length], key=lambda value: value.ordinal)
        )
        rows = tuple(
            materialize_development_probe(
                exact,
                probe,
                calibration=calibration,
                frozen_candidates=frozen_candidates,
            )
            for probe in probes
        )
        values = torch.cat(rows, dim=0).contiguous()
        amplitudes = torch.tensor(
            [
                _effective_amplitude(probe, calibration)
                for probe in probes
            ],
            dtype=torch.float64,
        )
        result.append(
            MaterializedDevelopmentBatch(
                role=role,
                protocol_sha256=exact.protocol_sha256,
                role_panel_sha256=panel_sha,
                calibration_sha256=(
                    None
                    if calibration is None
                    else calibration.artifact_sha256
                ),
                selected_amplitude=selected_amplitude,
                candidate_set_sha256=candidate_sha,
                sequence_length=length,
                probe_ids=tuple(probe.probe_id for probe in probes),
                probe_artifact_sha256s=tuple(
                    probe.artifact_sha256 for probe in probes
                ),
                probe_tensor_sha256s=tuple(
                    _tensor_sha256(value) for value in rows
                ),
                modal_amplitudes=amplitudes,
                radial_scales=torch.tensor(
                    [probe.radial_scale for probe in probes],
                    dtype=torch.float64,
                ),
                null_coordinates=torch.tensor(
                    [probe.null_coordinate for probe in probes],
                    dtype=torch.float64,
                ),
                values=values,
            )
        )
    expected_count = {"pilot": 40, "fit": 80, "selection": 80}[role]
    if sum(value.batch_size for value in result) != expected_count:
        raise RuntimeError("materialized development role count drifted")
    return tuple(result)
