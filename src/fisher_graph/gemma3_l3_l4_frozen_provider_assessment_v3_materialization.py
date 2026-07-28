"""Deterministic tensor materialization for the fresh V3 assessment panel.

The protocol module is tensor-free.  This module is the only bridge from its
authenticated specifications to standardized 64-mode direction tensors.  It
does not load a model, basis, provider, prompt, token, or measured activation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
import sys

import torch
from torch import Tensor

from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    DEFAULT_V3_PANEL_SPEC_SHA256,
    DEFAULT_V3_PROTOCOL_SHA256,
    V3AssessmentProtocol,
    V3ProbeSpec,
)


__all__ = [
    "MaterializedV3Batch",
    "materialize_v3_panel",
    "materialize_v3_probe",
]


_SCHEMA = "fisher_graph.gemma3_l3_l4_v3_materialized_batch.v1"
_FORMAT_VERSION = 1
_MODAL_WIDTH = 64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_DOMAIN = b"fisher-graph:l3-l4-v3-materialized-tensor:v1\0"
_BATCH_DOMAIN = b"fisher-graph:l3-l4-v3-materialized-batch:v1\0"
_ROW_NORM_ATOL = 1e-12


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


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return value


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


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_int(
    value: object,
    *,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _canonical_float_tensor(
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


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    header = {
        "dtype": "float64",
        "shape": [int(size) for size in tensor.shape],
        "byte_order": "little",
    }
    array = tensor.numpy()
    if sys.byteorder == "big":
        array = array.byteswap()
    digest = hashlib.sha256(
        _TENSOR_DOMAIN + _canonical_json_bytes(header) + b"\0"
    )
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _authenticate_protocol(
    protocol: V3AssessmentProtocol,
) -> V3AssessmentProtocol:
    if not isinstance(protocol, V3AssessmentProtocol):
        raise TypeError("protocol must be a V3AssessmentProtocol")
    if (
        protocol.protocol_sha256 != DEFAULT_V3_PROTOCOL_SHA256
        or protocol.panel_spec_sha256 != DEFAULT_V3_PANEL_SPEC_SHA256
    ):
        raise ValueError("V3 materialization requires the frozen default panel")
    # Round-trip validation rejects mutated nested objects or stale hashes.
    restored = V3AssessmentProtocol.from_state_dict(protocol.state_dict())
    if restored != protocol:
        raise ValueError("V3 protocol failed its strict round trip")
    return protocol


def _authenticate_probe(
    protocol: V3AssessmentProtocol,
    probe: V3ProbeSpec,
) -> V3ProbeSpec:
    _authenticate_protocol(protocol)
    if not isinstance(probe, V3ProbeSpec):
        raise TypeError("probe must be a V3ProbeSpec")
    matches = tuple(
        value for value in protocol.probes if value.probe_id == probe.probe_id
    )
    if len(matches) != 1 or matches[0] != probe:
        raise ValueError("V3 probe is not an exact member of the protocol")
    return probe


def _raw_multitone(probe: V3ProbeSpec) -> Tensor:
    temporal = probe.multitone_temporal_frequencies
    modal = probe.multitone_modal_frequencies
    phase = probe.multitone_phase_quadrants
    if temporal is None or modal is None or phase is None:
        raise ValueError("V3 multitone probe lacks its frozen parameters")
    raw = torch.zeros(
        probe.sequence_length,
        _MODAL_WIDTH,
        dtype=torch.float64,
    )
    active_length = probe.sequence_length - probe.source_offset
    modal_positions = (
        torch.arange(_MODAL_WIDTH, dtype=torch.float64) + 0.5
    ) / _MODAL_WIDTH
    for position in range(probe.source_offset, probe.sequence_length):
        relative = position - probe.source_offset
        tau = (relative + 0.5) / active_length
        first_cycles = (
            0.5 * temporal[0] * tau * tau
            + modal[0] * modal_positions
            + phase[0] / 4.0
        )
        second_cycles = (
            0.5 * temporal[1] * tau * tau
            + modal[1] * modal_positions
            + phase[1] / 4.0
        )
        raw[position] = torch.cos(2.0 * math.pi * first_cycles) + (
            0.5 * torch.cos(2.0 * math.pi * second_cycles)
        )
    return raw


def _raw_block_sparse(probe: V3ProbeSpec) -> Tensor:
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


def _raw_axis_block(probe: V3ProbeSpec) -> Tensor:
    if probe.axis_mode is None or probe.axis_sign is None:
        raise ValueError("V3 axis-family probe lacks its coordinate")
    raw = torch.zeros(
        probe.sequence_length,
        _MODAL_WIDTH,
        dtype=torch.float64,
    )
    end = probe.source_offset + probe.active_block_length
    raw[probe.source_offset:end, probe.axis_mode] = float(probe.axis_sign)
    return raw


def _expected_active_mask(probe: V3ProbeSpec) -> Tensor:
    expected = torch.zeros(probe.sequence_length, dtype=torch.bool)
    if probe.family == "multitone":
        expected[probe.source_offset:] = True
    elif probe.family == "block_sparse":
        for block in probe.sparse_blocks:
            expected[block.start : block.start + block.length] = True
    else:
        expected[
            probe.source_offset : probe.source_offset
            + probe.active_block_length
        ] = True
    return expected


def _normalize_rows(raw: Tensor, *, probe: V3ProbeSpec) -> Tensor:
    norms = torch.linalg.vector_norm(raw, dim=-1)
    active = norms > 0.0
    expected = _expected_active_mask(probe)
    if not torch.equal(active, expected):
        raise ValueError("V3 raw materialization differs from declared support")
    normalized = torch.zeros_like(raw)
    normalized[active] = (
        raw[active] / norms[active].unsqueeze(-1) * probe.modal_amplitude
    )
    return normalized


def materialize_v3_probe(
    protocol: V3AssessmentProtocol,
    probe: V3ProbeSpec,
) -> Tensor:
    """Materialize one authenticated probe as float64 ``[1, S, 64]``."""

    exact = _authenticate_probe(protocol, probe)
    if exact.family == "multitone":
        raw = _raw_multitone(exact)
    elif exact.family == "block_sparse":
        raw = _raw_block_sparse(exact)
    elif exact.family in {
        "radial_block_sensitivity",
        "signed_block_sensitivity",
        "null_single_invariance",
    }:
        raw = _raw_axis_block(exact)
    else:  # pragma: no cover - protocol validation makes this unreachable.
        raise ValueError(f"unsupported V3 probe family: {exact.family}")
    values = _normalize_rows(raw, probe=exact).unsqueeze(0).contiguous()
    if bool((values[:, : exact.source_offset] != 0.0).any()):
        raise ValueError("V3 materialization is nonzero before source offset")
    row_norms = torch.linalg.vector_norm(values[0], dim=-1)
    active = _expected_active_mask(exact)
    if not torch.allclose(
        row_norms[active],
        torch.full_like(row_norms[active], exact.modal_amplitude),
        rtol=0.0,
        atol=_ROW_NORM_ATOL,
    ):
        raise ValueError("V3 materialized rows violate amplitude norm")
    if bool((row_norms[~active] != 0.0).any()):
        raise ValueError("V3 materialized inactive rows are nonzero")
    return values


@dataclass(frozen=True, slots=True)
class MaterializedV3Batch:
    """Authenticated equal-length V3 directions and lift-side scalar state."""

    protocol_sha256: str
    panel_spec_sha256: str
    sequence_length: int
    probe_ids: tuple[str, ...]
    probe_artifact_sha256s: tuple[str, ...]
    probe_tensor_sha256s: tuple[str, ...]
    radial_scales: Tensor
    null_coordinates: Tensor
    values: Tensor
    tensor_sha256: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        protocol_sha = _require_sha256(
            self.protocol_sha256,
            label="materialized V3 protocol SHA",
        )
        panel_sha = _require_sha256(
            self.panel_spec_sha256,
            label="materialized V3 panel SHA",
        )
        if (
            protocol_sha != DEFAULT_V3_PROTOCOL_SHA256
            or panel_sha != DEFAULT_V3_PANEL_SPEC_SHA256
        ):
            raise ValueError("materialized V3 batch uses a nondefault protocol")
        sequence_length = _exact_int(
            self.sequence_length,
            label="materialized V3 sequence length",
            minimum=1,
        )
        if (
            type(self.probe_ids) is not tuple
            or not self.probe_ids
            or any(not isinstance(value, str) or not value for value in self.probe_ids)
            or len(set(self.probe_ids)) != len(self.probe_ids)
        ):
            raise ValueError("materialized V3 probe ids are invalid")
        batch_size = len(self.probe_ids)
        if (
            type(self.probe_artifact_sha256s) is not tuple
            or len(self.probe_artifact_sha256s) != batch_size
            or type(self.probe_tensor_sha256s) is not tuple
            or len(self.probe_tensor_sha256s) != batch_size
        ):
            raise ValueError("materialized V3 probe hashes are misaligned")
        for value in (
            *self.probe_artifact_sha256s,
            *self.probe_tensor_sha256s,
        ):
            _require_sha256(value, label="materialized V3 probe SHA")
        if len(set(self.probe_artifact_sha256s)) != batch_size:
            raise ValueError("materialized V3 probe artifacts must be unique")

        radial = _canonical_float_tensor(
            self.radial_scales,
            label="materialized V3 radial scales",
            ndim=1,
        )
        null = _canonical_float_tensor(
            self.null_coordinates,
            label="materialized V3 null coordinates",
            ndim=1,
        )
        values = _canonical_float_tensor(
            self.values,
            label="materialized V3 values",
            ndim=3,
        )
        if (
            radial.shape != (batch_size,)
            or null.shape != (batch_size,)
            or values.shape
            != (batch_size, sequence_length, _MODAL_WIDTH)
            or bool((radial <= 0.0).any())
        ):
            raise ValueError("materialized V3 batch tensor geometry is invalid")
        computed_probe_hashes = tuple(
            _tensor_sha256(values[index : index + 1])
            for index in range(batch_size)
        )
        if computed_probe_hashes != self.probe_tensor_sha256s:
            raise ValueError("materialized V3 per-probe tensor hash mismatch")
        computed_tensor = _tensor_sha256(values)
        if self.tensor_sha256:
            if _require_sha256(
                self.tensor_sha256,
                label="materialized V3 batch tensor SHA",
            ) != computed_tensor:
                raise ValueError("materialized V3 batch tensor hash mismatch")

        object.__setattr__(self, "protocol_sha256", protocol_sha)
        object.__setattr__(self, "panel_spec_sha256", panel_sha)
        object.__setattr__(self, "sequence_length", sequence_length)
        object.__setattr__(self, "radial_scales", radial)
        object.__setattr__(self, "null_coordinates", null)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "tensor_sha256", computed_tensor)
        computed_artifact = _digest(self._payload(), domain=_BATCH_DOMAIN)
        if self.artifact_sha256:
            if _require_sha256(
                self.artifact_sha256,
                label="materialized V3 artifact SHA",
            ) != computed_artifact:
                raise ValueError("materialized V3 batch artifact hash mismatch")
        object.__setattr__(self, "artifact_sha256", computed_artifact)

    @property
    def batch_size(self) -> int:
        return len(self.probe_ids)

    @property
    def modal_width(self) -> int:
        return int(self.values.shape[-1])

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "format_version": _FORMAT_VERSION,
            "protocol_sha256": self.protocol_sha256,
            "panel_spec_sha256": self.panel_spec_sha256,
            "sequence_length": self.sequence_length,
            "batch_size": len(self.probe_ids),
            "modal_width": _MODAL_WIDTH,
            "probe_ids": list(self.probe_ids),
            "probe_artifact_sha256s": list(self.probe_artifact_sha256s),
            "probe_tensor_sha256s": list(self.probe_tensor_sha256s),
            "radial_scales_sha256": _tensor_sha256(self.radial_scales),
            "null_coordinates_sha256": _tensor_sha256(self.null_coordinates),
            "tensor_sha256": self.tensor_sha256,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "radial_scales": self.radial_scales.clone(),
            "null_coordinates": self.null_coordinates.clone(),
            "values": self.values.clone(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_state_dict(cls, raw: object) -> "MaterializedV3Batch":
        state = _mapping(raw, label="materialized V3 batch")
        _strict_keys(
            state,
            expected={
                "schema",
                "format_version",
                "protocol_sha256",
                "panel_spec_sha256",
                "sequence_length",
                "batch_size",
                "modal_width",
                "probe_ids",
                "probe_artifact_sha256s",
                "probe_tensor_sha256s",
                "radial_scales_sha256",
                "null_coordinates_sha256",
                "tensor_sha256",
                "radial_scales",
                "null_coordinates",
                "values",
                "artifact_sha256",
            },
            label="materialized V3 batch",
        )
        if (
            state["schema"] != _SCHEMA
            or state["format_version"] != _FORMAT_VERSION
            or state["modal_width"] != _MODAL_WIDTH
        ):
            raise ValueError("materialized V3 schema, version, or width is invalid")
        probe_ids = tuple(
            str(value)
            for value in _sequence(
                state["probe_ids"],
                label="materialized V3 probe ids",
            )
        )
        if state["batch_size"] != len(probe_ids):
            raise ValueError("materialized V3 batch size drifted")
        radial = _canonical_float_tensor(
            state["radial_scales"],
            label="materialized V3 radial scales",
            ndim=1,
        )
        null = _canonical_float_tensor(
            state["null_coordinates"],
            label="materialized V3 null coordinates",
            ndim=1,
        )
        if _require_sha256(
            state["radial_scales_sha256"],
            label="materialized V3 radial SHA",
        ) != _tensor_sha256(radial):
            raise ValueError("materialized V3 radial scale hash mismatch")
        if _require_sha256(
            state["null_coordinates_sha256"],
            label="materialized V3 null SHA",
        ) != _tensor_sha256(null):
            raise ValueError("materialized V3 null coordinate hash mismatch")
        return cls(
            protocol_sha256=_require_sha256(
                state["protocol_sha256"],
                label="materialized V3 protocol SHA",
            ),
            panel_spec_sha256=_require_sha256(
                state["panel_spec_sha256"],
                label="materialized V3 panel SHA",
            ),
            sequence_length=_exact_int(
                state["sequence_length"],
                label="materialized V3 sequence length",
                minimum=1,
            ),
            probe_ids=probe_ids,
            probe_artifact_sha256s=tuple(
                _require_sha256(value, label="materialized V3 probe artifact SHA")
                for value in _sequence(
                    state["probe_artifact_sha256s"],
                    label="materialized V3 probe artifact hashes",
                )
            ),
            probe_tensor_sha256s=tuple(
                _require_sha256(value, label="materialized V3 probe tensor SHA")
                for value in _sequence(
                    state["probe_tensor_sha256s"],
                    label="materialized V3 probe tensor hashes",
                )
            ),
            radial_scales=radial,
            null_coordinates=null,
            values=_canonical_float_tensor(
                state["values"],
                label="materialized V3 values",
                ndim=3,
            ),
            tensor_sha256=_require_sha256(
                state["tensor_sha256"],
                label="materialized V3 tensor SHA",
            ),
            artifact_sha256=_require_sha256(
                state["artifact_sha256"],
                label="materialized V3 artifact SHA",
            ),
        )

    def validate_integrity(self) -> None:
        restored = MaterializedV3Batch.from_state_dict(self.state_dict())
        if (
            restored.artifact_sha256 != self.artifact_sha256
            or restored.tensor_sha256 != self.tensor_sha256
            or restored.probe_tensor_sha256s != self.probe_tensor_sha256s
            or not torch.equal(restored.values, self.values)
            or not torch.equal(restored.radial_scales, self.radial_scales)
            or not torch.equal(restored.null_coordinates, self.null_coordinates)
        ):
            raise ValueError("materialized V3 batch failed strict round trip")


def materialize_v3_panel(
    protocol: V3AssessmentProtocol,
) -> tuple[MaterializedV3Batch, ...]:
    """Materialize all 48 probes into deterministic equal-length batches."""

    exact = _authenticate_protocol(protocol)
    by_length: OrderedDict[int, list[V3ProbeSpec]] = OrderedDict()
    for probe in exact.probes:
        by_length.setdefault(probe.sequence_length, []).append(probe)
    batches: list[MaterializedV3Batch] = []
    emitted_ids: list[str] = []
    for sequence_length, probes in by_length.items():
        rows = tuple(materialize_v3_probe(exact, probe) for probe in probes)
        values = torch.cat(rows, dim=0)
        tensor_hashes = tuple(_tensor_sha256(value) for value in rows)
        batch = MaterializedV3Batch(
            protocol_sha256=exact.protocol_sha256,
            panel_spec_sha256=exact.panel_spec_sha256,
            sequence_length=sequence_length,
            probe_ids=tuple(probe.probe_id for probe in probes),
            probe_artifact_sha256s=tuple(
                probe.artifact_sha256 for probe in probes
            ),
            probe_tensor_sha256s=tensor_hashes,
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
        batch.validate_integrity()
        batches.append(batch)
        emitted_ids.extend(batch.probe_ids)
    if tuple(emitted_ids) != tuple(probe.probe_id for probe in exact.probes):
        # First-seen grouping preserves family order only when lengths are
        # unique across earlier groups.  Compare sets and then use a separate
        # ordinal check so a future repeated length cannot silently reorder.
        ordinal = {
            probe.probe_id: probe.ordinal for probe in exact.probes
        }
        flattened = tuple(
            probe_id for batch in batches for probe_id in batch.probe_ids
        )
        if (
            set(flattened) != set(emitted_ids)
            or sorted(flattened, key=ordinal.__getitem__)
            != list(probe.probe_id for probe in exact.probes)
        ):
            raise RuntimeError("V3 panel batching lost or duplicated probes")
    if sum(batch.batch_size for batch in batches) != 48:
        raise RuntimeError("V3 panel batching did not emit 48 probes")
    return tuple(batches)
