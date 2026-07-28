"""Deterministic tensors for the frozen L3-to-L4 synthetic probe protocol.

This module materializes only standardized modal directions.  It does not
load a model, tokenizer, prompt, or artifact.  Pre-normalization radial scale
and RMSNorm-null coordinates remain authenticated scalar fields on the probe;
the model-specific runner is responsible for applying them when lifting the
modal direction into a residual-state tensor.

Every returned tensor is CPU float64 with shape ``[1, length, 64]``.  Rows
before ``source_offset`` are exactly zero.  Every nonzero row is normalized
to unit L2 and then multiplied by the probe's declared modal amplitude.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct

import torch
from torch import Tensor

from .gemma3_l3_l4_synthetic_reference_protocol import (
    DEFAULT_PROTOCOL_SHA256,
    SyntheticReferenceProbe,
    SyntheticReferenceProtocol,
)


__all__ = [
    "MaterializedSyntheticReferenceBatch",
    "MaterializedSyntheticReferenceProbe",
    "materialize_synthetic_reference_batches",
    "materialize_synthetic_reference_probe",
]


_MODAL_WIDTH = 64
_MASK64 = (1 << 64) - 1
_PROBE_MATERIALIZATION_DOMAIN = (
    b"fisher-graph:l3-l4-synthetic-reference-materialization:v1\0"
)
_BATCH_MATERIALIZATION_DOMAIN = (
    b"fisher-graph:l3-l4-synthetic-reference-batch:v1\0"
)
_TENSOR_DOMAIN = b"fisher-graph:canonical-little-endian-float64-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE_ORDER = {"fit": 0, "selection": 1, "assessment": 2}
_ROW_NORM_ATOL = 2e-12


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


def _splitmix64(value: int) -> int:
    z = (value + 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _rademacher(seed: int, logical_position: int, mode: int) -> float:
    value = (
        seed
        ^ ((logical_position + 1) * 0xD1342543DE82EF95)
        ^ ((mode + 1) * 0x9E3779B97F4A7C15)
    )
    return 1.0 if _splitmix64(value) & 1 else -1.0


def _tensor_sha256(value: Tensor) -> str:
    canonical = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": "float64",
                "shape": list(canonical.shape),
                "byte_order": "little",
            }
        )
    )
    for number in canonical.reshape(-1).tolist():
        digest.update(struct.pack("<d", float(number)))
    return digest.hexdigest()


def _authenticate_protocol(
    protocol: SyntheticReferenceProtocol,
) -> SyntheticReferenceProtocol:
    if not isinstance(protocol, SyntheticReferenceProtocol):
        raise TypeError("protocol must be a SyntheticReferenceProtocol")
    restored = SyntheticReferenceProtocol.from_state_dict(
        protocol.state_dict()
    )
    if (
        restored != protocol
        or restored.protocol_sha256 != DEFAULT_PROTOCOL_SHA256
    ):
        raise ValueError("synthetic reference protocol authentication failed")
    return restored


def _authenticate_probe_membership(
    protocol: SyntheticReferenceProtocol,
    probe: SyntheticReferenceProbe,
) -> SyntheticReferenceProbe:
    if not isinstance(probe, SyntheticReferenceProbe):
        raise TypeError("probe must be a SyntheticReferenceProbe")
    members = {
        value.artifact_sha256: value for value in protocol.probes
    }
    member = members.get(probe.artifact_sha256)
    if member is None or member != probe:
        raise ValueError("probe is not an authenticated protocol member")
    return member


def _normalize_rows(raw: Tensor, *, amplitude: float) -> Tensor:
    if raw.ndim != 2 or raw.shape[1] != _MODAL_WIDTH:
        raise ValueError("raw modal direction geometry is invalid")
    norms = torch.linalg.vector_norm(raw, dim=-1)
    active = norms > 0.0
    result = torch.zeros_like(raw)
    if bool(active.any()):
        result[active] = raw[active] / norms[active, None]
        result[active] *= amplitude
    return result


def _materialize_raw(probe: SyntheticReferenceProbe) -> Tensor:
    length = probe.sequence_length
    offset = probe.source_offset
    raw = torch.zeros((length, _MODAL_WIDTH), dtype=torch.float64)

    if probe.family == "rademacher":
        for position in range(offset, length):
            raw[position] = torch.tensor(
                [
                    _rademacher(probe.direction_seed, position, mode)
                    for mode in range(_MODAL_WIDTH)
                ],
                dtype=torch.float64,
            )
    elif probe.family == "ar1":
        coefficient = probe.ar_coefficient
        if coefficient is None:
            raise ValueError("AR probe lacks its coefficient")
        innovation_scale = math.sqrt(1.0 - coefficient * coefficient)
        previous = [
            _rademacher(probe.direction_seed, offset, mode)
            for mode in range(_MODAL_WIDTH)
        ]
        raw[offset] = torch.tensor(previous, dtype=torch.float64)
        for position in range(offset + 1, length):
            current = []
            for mode, prior in enumerate(previous):
                innovation = _rademacher(
                    probe.direction_seed,
                    position,
                    mode,
                )
                current.append(
                    coefficient * prior + innovation_scale * innovation
                )
            raw[position] = torch.tensor(current, dtype=torch.float64)
            previous = current
    elif probe.family == "sparse":
        for position, mode, sign in probe.sparse_coordinates:
            raw[position, mode] = float(sign)
    elif probe.family == "chirp":
        temporal_frequency = probe.chirp_temporal_frequency
        modal_frequency = probe.chirp_modal_frequency
        phase_quadrant = probe.chirp_phase_quadrant
        if (
            temporal_frequency is None
            or modal_frequency is None
            or phase_quadrant is None
        ):
            raise ValueError("chirp probe lacks its parameters")
        active_length = length - offset
        for position in range(offset, length):
            relative = position - offset
            tau = (relative + 0.5) / active_length
            row = []
            for mode in range(_MODAL_WIDTH):
                modal_phase = (mode + 0.5) / _MODAL_WIDTH
                cycles = (
                    0.5 * temporal_frequency * tau * tau
                    + modal_frequency * modal_phase
                    + phase_quadrant / 4.0
                )
                row.append(math.cos(2.0 * math.pi * cycles))
            raw[position] = torch.tensor(row, dtype=torch.float64)
    elif probe.family in {
        "axis",
        "radial_collision",
        "null_collision",
    }:
        if probe.axis_mode is None or probe.axis_sign is None:
            raise ValueError("axis-family probe lacks its coordinate")
        raw[offset, probe.axis_mode] = float(probe.axis_sign)
    else:  # pragma: no cover - protocol validation makes this unreachable.
        raise ValueError(f"unsupported synthetic probe family: {probe.family}")
    return raw


def _materialize_authenticated_probe(
    protocol: SyntheticReferenceProtocol,
    probe: SyntheticReferenceProbe,
) -> "MaterializedSyntheticReferenceProbe":
    raw = _materialize_raw(probe)
    values = _normalize_rows(raw, amplitude=probe.modal_amplitude).unsqueeze(0)
    return MaterializedSyntheticReferenceProbe(
        protocol_sha256=protocol.protocol_sha256,
        probe_id=probe.probe_id,
        probe_artifact_sha256=probe.artifact_sha256,
        role=probe.role,
        family=probe.family,
        source_offset=probe.source_offset,
        radial_scale=probe.radial_scale,
        null_coordinate=probe.null_coordinate,
        modal_amplitude=probe.modal_amplitude,
        values=values,
    )


@dataclass(frozen=True, slots=True)
class MaterializedSyntheticReferenceProbe:
    """One authenticated standardized modal-direction tensor."""

    protocol_sha256: str
    probe_id: str
    probe_artifact_sha256: str
    role: str
    family: str
    source_offset: int
    radial_scale: float
    null_coordinate: float
    modal_amplitude: float
    values: Tensor
    tensor_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="protocol_sha256")
        _require_sha256(
            self.probe_artifact_sha256,
            label="probe_artifact_sha256",
        )
        if (
            not isinstance(self.probe_id, str)
            or not self.probe_id
            or self.role not in _ROLE_ORDER
            or not isinstance(self.family, str)
            or not self.family
        ):
            raise ValueError("materialized probe identity is invalid")
        for label, number in (
            ("radial_scale", self.radial_scale),
            ("null_coordinate", self.null_coordinate),
            ("modal_amplitude", self.modal_amplitude),
        ):
            if (
                isinstance(number, bool)
                or not isinstance(number, (float, int))
                or not math.isfinite(float(number))
            ):
                raise TypeError(f"{label} must be finite")
        if float(self.modal_amplitude) <= 0.0:
            raise ValueError("modal_amplitude must be positive")
        if not isinstance(self.values, Tensor):
            raise TypeError("materialized probe values must be a Tensor")
        values = (
            self.values.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        if (
            values.ndim != 3
            or values.shape[0] != 1
            or values.shape[1] <= 0
            or values.shape[2] != _MODAL_WIDTH
            or type(self.source_offset) is not int
            or not 0 <= self.source_offset < values.shape[1]
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("materialized probe tensor is invalid")
        object.__setattr__(self, "values", values)
        if bool((values[:, : self.source_offset] != 0.0).any()):
            raise ValueError("materialized probe is nonzero before source offset")
        row_norms = torch.linalg.vector_norm(values[0], dim=-1)
        active = row_norms > 0.0
        if bool(active.any()) and not bool(
            torch.allclose(
                row_norms[active],
                torch.full_like(row_norms[active], self.modal_amplitude),
                rtol=0.0,
                atol=_ROW_NORM_ATOL,
            )
        ):
            raise ValueError("materialized probe rows violate amplitude norm")
        computed_tensor = _tensor_sha256(values)
        if self.tensor_sha256:
            if self.tensor_sha256 != computed_tensor:
                raise ValueError("materialized probe tensor hash mismatch")
        else:
            object.__setattr__(self, "tensor_sha256", computed_tensor)
        computed_artifact = _digest(
            self._payload(),
            domain=_PROBE_MATERIALIZATION_DOMAIN,
        )
        if self.artifact_sha256:
            if self.artifact_sha256 != computed_artifact:
                raise ValueError("materialized probe artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    @property
    def sequence_length(self) -> int:
        return int(self.values.shape[1])

    @property
    def active_row_count(self) -> int:
        return int(
            (
                torch.linalg.vector_norm(self.values[0], dim=-1) > 0.0
            ).sum()
        )

    def _payload(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "probe_id": self.probe_id,
            "probe_artifact_sha256": self.probe_artifact_sha256,
            "role": self.role,
            "family": self.family,
            "source_offset": self.source_offset,
            "sequence_length": self.sequence_length,
            "modal_width": _MODAL_WIDTH,
            "radial_scale": self.radial_scale,
            "null_coordinate": self.null_coordinate,
            "modal_amplitude": self.modal_amplitude,
            "dtype": "float64",
            "normalization": (
                "per_nonzero_row_unit_L2_then_modal_amplitude"
            ),
            "tensor_sha256": self.tensor_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {
            **self._payload(),
            "active_row_count": self.active_row_count,
            "artifact_sha256": self.artifact_sha256,
        }

    def validate_integrity(self) -> None:
        if _tensor_sha256(self.values) != self.tensor_sha256:
            raise ValueError("materialized probe tensor was mutated")
        if (
            _digest(self._payload(), domain=_PROBE_MATERIALIZATION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("materialized probe metadata was mutated")


@dataclass(frozen=True, slots=True)
class MaterializedSyntheticReferenceBatch:
    """One deterministic equal-length batch of materialized probes."""

    protocol_sha256: str
    sequence_length: int
    probe_ids: tuple[str, ...]
    probe_artifact_sha256s: tuple[str, ...]
    probe_tensor_sha256s: tuple[str, ...]
    values: Tensor
    tensor_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.protocol_sha256, label="protocol_sha256")
        if (
            type(self.sequence_length) is not int
            or self.sequence_length <= 0
            or type(self.probe_ids) is not tuple
            or not self.probe_ids
            or len(set(self.probe_ids)) != len(self.probe_ids)
            or type(self.probe_artifact_sha256s) is not tuple
            or type(self.probe_tensor_sha256s) is not tuple
            or len(self.probe_ids) != len(self.probe_artifact_sha256s)
            or len(self.probe_ids) != len(self.probe_tensor_sha256s)
        ):
            raise ValueError("materialized batch identity is invalid")
        for digest in (
            *self.probe_artifact_sha256s,
            *self.probe_tensor_sha256s,
        ):
            _require_sha256(digest, label="batch member SHA-256")
        if not isinstance(self.values, Tensor):
            raise TypeError("materialized batch values must be a Tensor")
        values = (
            self.values.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .clone()
        )
        if (
            values.shape
            != (len(self.probe_ids), self.sequence_length, _MODAL_WIDTH)
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("materialized batch tensor is invalid")
        object.__setattr__(self, "values", values)
        for index, expected in enumerate(self.probe_tensor_sha256s):
            if _tensor_sha256(values[index : index + 1]) != expected:
                raise ValueError("materialized batch member hash mismatch")
        computed_tensor = _tensor_sha256(values)
        if self.tensor_sha256:
            if self.tensor_sha256 != computed_tensor:
                raise ValueError("materialized batch tensor hash mismatch")
        else:
            object.__setattr__(self, "tensor_sha256", computed_tensor)
        computed_artifact = _digest(
            self._payload(),
            domain=_BATCH_MATERIALIZATION_DOMAIN,
        )
        if self.artifact_sha256:
            if self.artifact_sha256 != computed_artifact:
                raise ValueError("materialized batch artifact hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed_artifact)

    def _payload(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "sequence_length": self.sequence_length,
            "modal_width": _MODAL_WIDTH,
            "batch_size": len(self.probe_ids),
            "probe_ids": list(self.probe_ids),
            "probe_artifact_sha256s": list(self.probe_artifact_sha256s),
            "probe_tensor_sha256s": list(self.probe_tensor_sha256s),
            "dtype": "float64",
            "padding": "none_equal_length_only",
            "tensor_sha256": self.tensor_sha256,
        }

    def metadata(self) -> dict[str, object]:
        return {**self._payload(), "artifact_sha256": self.artifact_sha256}

    def validate_integrity(self) -> None:
        if _tensor_sha256(self.values) != self.tensor_sha256:
            raise ValueError("materialized batch tensor was mutated")
        if (
            _digest(self._payload(), domain=_BATCH_MATERIALIZATION_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("materialized batch metadata was mutated")


def materialize_synthetic_reference_probe(
    protocol: SyntheticReferenceProtocol,
    probe: SyntheticReferenceProbe,
) -> MaterializedSyntheticReferenceProbe:
    """Authenticate and materialize one frozen probe on CPU in float64."""

    authenticated = _authenticate_protocol(protocol)
    member = _authenticate_probe_membership(authenticated, probe)
    return _materialize_authenticated_probe(authenticated, member)


def materialize_synthetic_reference_batches(
    protocol: SyntheticReferenceProtocol,
    probes: Sequence[SyntheticReferenceProbe] | None = None,
) -> tuple[MaterializedSyntheticReferenceBatch, ...]:
    """Materialize unique members and return deterministic equal-length batches.

    The output order is ascending sequence length.  Within a batch, members
    follow frozen role order, role ordinal, and probe id.  No padding is added.
    """

    authenticated = _authenticate_protocol(protocol)
    requested = authenticated.probes if probes is None else tuple(probes)
    if not requested:
        raise ValueError("at least one synthetic probe is required")
    members = [
        _authenticate_probe_membership(authenticated, probe)
        for probe in requested
    ]
    if len({probe.artifact_sha256 for probe in members}) != len(members):
        raise ValueError("batch request contains duplicate probes")
    ordered = sorted(
        members,
        key=lambda probe: (
            probe.sequence_length,
            _ROLE_ORDER[probe.role],
            probe.ordinal,
            probe.probe_id,
        ),
    )
    grouped: dict[int, list[SyntheticReferenceProbe]] = defaultdict(list)
    for probe in ordered:
        grouped[probe.sequence_length].append(probe)

    batches = []
    for length in sorted(grouped):
        materialized = [
            _materialize_authenticated_probe(authenticated, probe)
            for probe in grouped[length]
        ]
        batches.append(
            MaterializedSyntheticReferenceBatch(
                protocol_sha256=authenticated.protocol_sha256,
                sequence_length=length,
                probe_ids=tuple(value.probe_id for value in materialized),
                probe_artifact_sha256s=tuple(
                    value.probe_artifact_sha256 for value in materialized
                ),
                probe_tensor_sha256s=tuple(
                    value.tensor_sha256 for value in materialized
                ),
                values=torch.cat(
                    [value.values for value in materialized],
                    dim=0,
                ),
            )
        )
    return tuple(batches)
