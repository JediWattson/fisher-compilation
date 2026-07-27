"""Recursive causal graph contraction around modal-generator leaf graphs.

This module is a strict linear reference IR for the hierarchy rung.  A graph
contains affine multi-port components connected by signed *direct Jacobian*
maps.  It can be executed directly or symbolically reduced to an exact causal
boundary transfer.  That transfer is then eligible for the Fisher-weighted
connectivity factorization in :mod:`fisher_graph.modal_connectivity_modes`.

The separation is intentional:

* finite ablation responses may nominate a group but are not direct edges;
* causal containment is restricted to contiguous component intervals;
* every edge crossing a contraction cut is surfaced as its own boundary port;
* noncontiguous parameter sharing is metadata, never causal containment.

A :class:`HierarchicalModalGenerator` binds the child graph and one
connectivity decomposition.  A reduced candidate is lowered into a fine
encoder ``R``, an explicit modal core, and a fine decoder ``P``.  The default
core is only a zero-cost identity handoff.  A real recursive child is formed
after signed ``R_j @ J_ji @ P_i`` interactions and fresh output moments have
been measured; it never refactors a materialized ``P @ R`` matrix.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re

import torch
from torch import Tensor

from .modal_connectivity_modes import (
    CausalBoundaryTransfer,
    ConnectivityModeMoments,
    MessageMoments,
    ModalBoundaryPort,
    ModalConnectivityDecomposition,
    ModalConnectivityFactor,
    factor_modal_connectivity,
)


__all__ = [
    "AffineModalComponent",
    "BoundaryInputInjection",
    "BoundaryOutputReadout",
    "CausalCoarseningGroup",
    "DirectModalConnection",
    "HierarchicalModalGenerator",
    "HierarchicalModeExpansion",
    "IdentityModalComponent",
    "ImplicitIdentityMap",
    "LinearModalGraphLevel",
    "MeasuredModalCore",
    "ModalParameterSharingFamily",
    "ProjectedModalConnection",
    "affine_modal_component",
    "extract_coarsening_group",
    "identity_modal_component",
    "project_modal_jacobian",
]


_FORMAT_VERSION = 1
_COMPONENT_KIND = "fisher_graph.affine_modal_component"
_IDENTITY_COMPONENT_KIND = "fisher_graph.identity_modal_component"
_CONNECTION_KIND = "fisher_graph.direct_modal_connection"
_INJECTION_KIND = "fisher_graph.modal_boundary_input_injection"
_READOUT_KIND = "fisher_graph.modal_boundary_output_readout"
_GRAPH_KIND = "fisher_graph.linear_modal_graph_level"
_GROUP_KIND = "fisher_graph.causal_coarsening_group"
_GENERATOR_KIND = "fisher_graph.hierarchical_modal_generator"
_MODE_EXPANSION_KIND = "fisher_graph.hierarchical_mode_expansion"
_MEASURED_MODAL_CORE_KIND = "fisher_graph.measured_modal_core"
_PROJECTED_CONNECTION_KIND = "fisher_graph.projected_modal_connection"
_IDENTITY_MAP_KIND = "fisher_graph.implicit_identity_map"
_SHARING_KIND = "fisher_graph.modal_parameter_sharing_family"
_COMPONENT_DOMAIN = b"fisher_graph.affine_modal_component.v1\0"
_IDENTITY_COMPONENT_DOMAIN = (
    b"fisher_graph.identity_modal_component.v1\0"
)
_CONNECTION_DOMAIN = b"fisher_graph.direct_modal_connection.v1\0"
_INJECTION_DOMAIN = b"fisher_graph.modal_boundary_input_injection.v1\0"
_READOUT_DOMAIN = b"fisher_graph.modal_boundary_output_readout.v1\0"
_GRAPH_DOMAIN = b"fisher_graph.linear_modal_graph_level.v1\0"
_GROUP_DOMAIN = b"fisher_graph.causal_coarsening_group.v1\0"
_GENERATOR_DOMAIN = b"fisher_graph.hierarchical_modal_generator.v1\0"
_MODE_EXPANSION_DOMAIN = (
    b"fisher_graph.hierarchical_mode_expansion.v1\0"
)
_MEASURED_MODAL_CORE_DOMAIN = b"fisher_graph.measured_modal_core.v1\0"
_PROJECTED_CONNECTION_DOMAIN = (
    b"fisher_graph.projected_modal_connection.v1\0"
)
_IDENTITY_MAP_DOMAIN = b"fisher_graph.implicit_identity_map.v1\0"
_SHARING_DOMAIN = b"fisher_graph.modal_parameter_sharing_family.v1\0"
_TENSOR_DOMAIN = b"fisher_graph.modal_graph_hierarchy.tensor.v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _require_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _cpu_float64(
    value: Tensor,
    *,
    label: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point")
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result.clone()


def _tensor_sha256(value: Tensor, *, label: str) -> str:
    tensor = _cpu_float64(value, label=label)
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0float64\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_sha256(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _canonical_edge_key(
    value: DirectModalConnection,
) -> tuple[str, str, str, str]:
    return (
        value.source_component,
        value.source_port,
        value.target_component,
        value.target_port,
    )


@dataclass(frozen=True, slots=True)
class ImplicitIdentityMap:
    """A typed zero-storage, zero-MAC identity wire."""

    width: int
    artifact_sha256: str = ""
    artifact_kind: str = _IDENTITY_MAP_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_int(self.width, label="identity width", minimum=1)
        if (
            self.artifact_kind != _IDENTITY_MAP_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("implicit identity map header is invalid")
        computed = _json_sha256(
            {
                "artifact_kind": self.artifact_kind,
                "format_version": self.format_version,
                "width": self.width,
            },
            domain=_IDENTITY_MAP_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="identity map artifact_sha256",
                )
                != computed
            ):
                raise ValueError("implicit identity map hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.width, self.width)

    def validate_integrity(self) -> None:
        self.__post_init__()


LinearMap = Tensor | ImplicitIdentityMap


def _canonical_linear_map(
    value: LinearMap,
    *,
    label: str,
) -> LinearMap:
    if isinstance(value, ImplicitIdentityMap):
        value.validate_integrity()
        return value
    return _cpu_float64(value, label=label, ndim=2)


def _linear_map_shape(value: LinearMap) -> tuple[int, int]:
    if isinstance(value, ImplicitIdentityMap):
        return value.shape
    return tuple(value.shape)  # type: ignore[return-value]


def _linear_map_stored_scalar_count(value: LinearMap) -> int:
    return 0 if isinstance(value, ImplicitIdentityMap) else value.numel()


def _linear_map_macs_per_row(value: LinearMap) -> int:
    return 0 if isinstance(value, ImplicitIdentityMap) else value.numel()


def _linear_map_payload(value: LinearMap, *, label: str) -> dict[str, object]:
    if isinstance(value, ImplicitIdentityMap):
        value.validate_integrity()
        return {
            "kind": "implicit_identity",
            "identity_map_sha256": value.artifact_sha256,
        }
    return {
        "kind": "dense",
        "tensor_sha256": _tensor_sha256(value, label=label),
    }


def _apply_linear_map(value: Tensor, mapping: LinearMap) -> Tensor:
    if isinstance(mapping, ImplicitIdentityMap):
        return value
    return value @ mapping.to(device=value.device, dtype=value.dtype).T


def _left_multiply(mapping: LinearMap, value: Tensor) -> Tensor:
    if isinstance(mapping, ImplicitIdentityMap):
        return value
    return mapping @ value


def _linear_map_scale(value: LinearMap) -> float:
    if isinstance(value, ImplicitIdentityMap):
        return 1.0
    return float(value.abs().max().item())


def project_modal_jacobian(
    *,
    upstream_factor: ModalConnectivityFactor,
    downstream_factor: ModalConnectivityFactor,
    downstream_input_port_name: str,
    fine_jacobian: Tensor,
) -> Tensor:
    """Project one signed fine edge into the two retained mode bases.

    For a fine edge ``J: y_i -> x_j`` this returns
    ``R_j[target block] @ J @ P_i``.  The caller must still bind the result to
    direct-Jacobian evidence before inserting it into an executable graph.
    """

    if not isinstance(upstream_factor, ModalConnectivityFactor) or not isinstance(
        downstream_factor,
        ModalConnectivityFactor,
    ):
        raise TypeError("modal projection requires connectivity factors")
    upstream_factor.validate_integrity()
    downstream_factor.validate_integrity()
    _require_name(
        downstream_input_port_name,
        label="downstream_input_port_name",
    )
    input_names = tuple(
        port.name for port in downstream_factor.input_ports
    )
    if downstream_input_port_name not in input_names:
        raise ValueError("downstream factor does not contain target input port")
    input_port = downstream_factor.input_ports[
        input_names.index(downstream_input_port_name)
    ]
    if (
        upstream_factor.output_port.causal_order
        >= input_port.causal_order
    ):
        raise ValueError(
            "fine modal Jacobian must connect an earlier output to a later "
            "input"
        )
    jacobian = _cpu_float64(
        fine_jacobian,
        label="fine modal Jacobian",
        ndim=2,
    )
    if jacobian.shape != (
        input_port.width,
        upstream_factor.output_port.width,
    ):
        raise ValueError("fine Jacobian shape does not match projected ports")
    start = sum(
        port.width
        for port in downstream_factor.input_ports[
            : input_names.index(downstream_input_port_name)
        ]
    )
    stop = start + input_port.width
    return (
        downstream_factor.restriction[:, start:stop]
        @ jacobian
        @ upstream_factor.prolongation
    ).contiguous()


@dataclass(frozen=True, slots=True)
class AffineModalComponent:
    """One causal component whose local semantics are a boundary transfer."""

    component_id: str
    level_index: int
    causal_start: int
    causal_end: int
    transfer: CausalBoundaryTransfer
    child_ids: tuple[str, ...]
    source_artifact_sha256s: tuple[str, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _COMPONENT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.component_id, label="component_id")
        _require_int(self.level_index, label="level_index", minimum=0)
        _require_int(self.causal_start, label="causal_start", minimum=0)
        _require_int(self.causal_end, label="causal_end", minimum=0)
        if self.causal_end < self.causal_start:
            raise ValueError("component causal interval is reversed")
        if not isinstance(self.transfer, CausalBoundaryTransfer):
            raise TypeError("component transfer must be CausalBoundaryTransfer")
        self.transfer.validate_integrity()
        if (
            type(self.child_ids) is not tuple
            or not self.child_ids
            or any(
                not isinstance(value, str)
                or _NAME.fullmatch(value) is None
                for value in self.child_ids
            )
            or len(self.child_ids) != len(set(self.child_ids))
        ):
            raise ValueError("component child_ids must be unique names")
        if (
            type(self.source_artifact_sha256s) is not tuple
            or not self.source_artifact_sha256s
        ):
            raise ValueError(
                "component source_artifact_sha256s must be nonempty"
            )
        for digest in self.source_artifact_sha256s:
            _require_sha256(digest, label="component source artifact digest")
        for port in self.transfer.input_ports + self.transfer.output_ports:
            if not self.causal_start <= port.causal_order <= self.causal_end:
                raise ValueError(
                    "component port lies outside its causal interval"
                )
        if (
            self.artifact_kind != _COMPONENT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("affine modal component header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_COMPONENT_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="component artifact_sha256",
                )
                != computed
            ):
                raise ValueError("affine modal component hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def input_ports(self) -> dict[str, ModalBoundaryPort]:
        return {port.name: port for port in self.transfer.input_ports}

    @property
    def output_ports(self) -> dict[str, ModalBoundaryPort]:
        return {port.name: port for port in self.transfer.output_ports}

    @property
    def stored_scalar_count(self) -> int:
        return self.transfer.stored_scalar_count

    @property
    def macs_per_row(self) -> int:
        return self.transfer.macs_per_row

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "component_id": self.component_id,
            "level_index": self.level_index,
            "causal_start": self.causal_start,
            "causal_end": self.causal_end,
            "transfer_sha256": self.transfer.artifact_sha256,
            "child_ids": self.child_ids,
            "source_artifact_sha256s": self.source_artifact_sha256s,
        }

    def validate_integrity(self) -> None:
        self.transfer.validate_integrity()
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_COMPONENT_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("affine modal component hash mismatch")


def affine_modal_component(
    *,
    component_id: str,
    causal_order: int,
    matrix: Tensor,
    bias: Tensor,
    source_artifact_sha256: str,
    level_index: int = 0,
) -> AffineModalComponent:
    """Build a one-input, one-output primitive affine component."""

    canonical_matrix = _cpu_float64(
        matrix,
        label="component matrix",
        ndim=2,
    )
    canonical_bias = _cpu_float64(
        bias,
        label="component bias",
        ndim=1,
    )
    if canonical_matrix.shape[0] != canonical_bias.shape[0]:
        raise ValueError("component matrix output width does not match bias")
    if canonical_matrix.shape[1] == 0 or canonical_matrix.shape[0] == 0:
        raise ValueError("component matrix dimensions must be nonempty")
    input_port = ModalBoundaryPort(
        name=f"{component_id}.input",
        direction="input",
        causal_order=causal_order,
        width=canonical_matrix.shape[1],
        owner_id=component_id,
    )
    output_port = ModalBoundaryPort(
        name=f"{component_id}.output",
        direction="output",
        causal_order=causal_order,
        width=canonical_matrix.shape[0],
        owner_id=component_id,
    )
    transfer = CausalBoundaryTransfer(
        source_level_sha256=source_artifact_sha256,
        input_ports=(input_port,),
        output_ports=(output_port,),
        input_prefixes=((input_port.name,),),
        transfer_matrices=(canonical_matrix,),
        affine_offsets=(canonical_bias,),
    )
    return AffineModalComponent(
        component_id=component_id,
        level_index=level_index,
        causal_start=causal_order,
        causal_end=causal_order,
        transfer=transfer,
        child_ids=(component_id,),
        source_artifact_sha256s=(source_artifact_sha256,),
    )


@dataclass(frozen=True, slots=True)
class IdentityModalComponent:
    """A true alias component with no parameter tensor or linear MAC."""

    component_id: str
    level_index: int
    causal_order: int
    input_port: ModalBoundaryPort
    output_port: ModalBoundaryPort
    child_ids: tuple[str, ...]
    source_artifact_sha256s: tuple[str, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _IDENTITY_COMPONENT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.component_id, label="identity component_id")
        _require_int(self.level_index, label="level_index", minimum=0)
        _require_int(
            self.causal_order,
            label="identity causal_order",
            minimum=0,
        )
        for port, direction in (
            (self.input_port, "input"),
            (self.output_port, "output"),
        ):
            if not isinstance(port, ModalBoundaryPort):
                raise TypeError(
                    "identity component ports must be ModalBoundaryPort"
                )
            port.validate_integrity()
            if (
                port.direction != direction
                or port.owner_id != self.component_id
                or port.causal_order != self.causal_order
            ):
                raise ValueError(
                    "identity component port contract is invalid"
                )
        if self.input_port.width != self.output_port.width:
            raise ValueError("identity component widths must match")
        if (
            type(self.child_ids) is not tuple
            or not self.child_ids
            or len(self.child_ids) != len(set(self.child_ids))
        ):
            raise ValueError(
                "identity component child_ids must be nonempty and unique"
            )
        for child_id in self.child_ids:
            _require_name(child_id, label="identity child_id")
        if (
            type(self.source_artifact_sha256s) is not tuple
            or not self.source_artifact_sha256s
        ):
            raise ValueError(
                "identity source artifact digests must be nonempty"
            )
        for digest in self.source_artifact_sha256s:
            _require_sha256(
                digest,
                label="identity source artifact digest",
            )
        if (
            self.artifact_kind != _IDENTITY_COMPONENT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("identity modal component header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_IDENTITY_COMPONENT_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="identity component artifact_sha256",
                )
                != computed
            ):
                raise ValueError("identity modal component hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def causal_start(self) -> int:
        return self.causal_order

    @property
    def causal_end(self) -> int:
        return self.causal_order

    @property
    def input_ports(self) -> dict[str, ModalBoundaryPort]:
        return {self.input_port.name: self.input_port}

    @property
    def output_ports(self) -> dict[str, ModalBoundaryPort]:
        return {self.output_port.name: self.output_port}

    @property
    def stored_scalar_count(self) -> int:
        return 0

    @property
    def macs_per_row(self) -> int:
        return 0

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "component_id": self.component_id,
            "level_index": self.level_index,
            "causal_order": self.causal_order,
            "input_port_sha256": self.input_port.artifact_sha256,
            "output_port_sha256": self.output_port.artifact_sha256,
            "child_ids": self.child_ids,
            "source_artifact_sha256s": self.source_artifact_sha256s,
        }

    def validate_integrity(self) -> None:
        self.input_port.validate_integrity()
        self.output_port.validate_integrity()
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_IDENTITY_COMPONENT_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("identity modal component hash mismatch")


def identity_modal_component(
    *,
    component_id: str,
    causal_order: int,
    width: int,
    source_artifact_sha256: str,
    level_index: int = 0,
) -> IdentityModalComponent:
    """Build a typed alias component without materializing ``eye(width)``."""

    _require_int(width, label="identity component width", minimum=1)
    input_port = ModalBoundaryPort(
        name=f"{component_id}.input",
        direction="input",
        causal_order=causal_order,
        width=width,
        owner_id=component_id,
    )
    output_port = ModalBoundaryPort(
        name=f"{component_id}.output",
        direction="output",
        causal_order=causal_order,
        width=width,
        owner_id=component_id,
    )
    return IdentityModalComponent(
        component_id=component_id,
        level_index=level_index,
        causal_order=causal_order,
        input_port=input_port,
        output_port=output_port,
        child_ids=(component_id,),
        source_artifact_sha256s=(source_artifact_sha256,),
    )


ModalComponent = AffineModalComponent | IdentityModalComponent


@dataclass(frozen=True, slots=True)
class DirectModalConnection:
    """One signed, direct-Jacobian connection between component ports."""

    source_component: str
    source_port: str
    target_component: str
    target_port: str
    matrix: LinearMap
    evidence_kind: str
    evidence_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _CONNECTION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        for field in (
            "source_component",
            "source_port",
            "target_component",
            "target_port",
        ):
            _require_name(getattr(self, field), label=field)
        if self.source_component == self.target_component:
            raise ValueError("direct modal connection cannot be a self-edge")
        if self.evidence_kind != "direct_jacobian":
            raise ValueError(
                "hierarchy execution accepts direct_jacobian edges only; "
                "finite ablation responses are nomination evidence"
            )
        _require_sha256(
            self.evidence_sha256,
            label="connection evidence_sha256",
        )
        matrix = _canonical_linear_map(
            self.matrix,
            label="connection matrix",
        )
        if 0 in _linear_map_shape(matrix):
            raise ValueError("connection matrix dimensions must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        if (
            self.artifact_kind != _CONNECTION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("direct modal connection header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_CONNECTION_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="connection artifact_sha256",
                )
                != computed
            ):
                raise ValueError("direct modal connection hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def stored_scalar_count(self) -> int:
        return _linear_map_stored_scalar_count(self.matrix)

    @property
    def macs_per_row(self) -> int:
        return _linear_map_macs_per_row(self.matrix)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_component": self.source_component,
            "source_port": self.source_port,
            "target_component": self.target_component,
            "target_port": self.target_port,
            "linear_map": _linear_map_payload(
                self.matrix,
                label="connection matrix",
            ),
            "evidence_kind": self.evidence_kind,
            "evidence_sha256": self.evidence_sha256,
        }

    def validate_integrity(self) -> None:
        if isinstance(self.matrix, ImplicitIdentityMap):
            self.matrix.validate_integrity()
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_CONNECTION_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("direct modal connection hash mismatch")


@dataclass(frozen=True, slots=True)
class ProjectedModalConnection:
    """Proof-carrying ``R_j J_ji P_i`` interaction for one expansion."""

    source_expansion_sha256: str
    upstream_factor: ModalConnectivityFactor
    downstream_factor: ModalConnectivityFactor
    downstream_input_port_name: str
    fine_jacobian: Tensor
    direct_jacobian_evidence_sha256: str
    source_component: str
    source_port: str
    target_component: str
    target_port: str
    matrix: Tensor
    artifact_sha256: str = ""
    artifact_kind: str = _PROJECTED_CONNECTION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_expansion_sha256,
            label="projected connection source expansion digest",
        )
        if not isinstance(self.upstream_factor, ModalConnectivityFactor):
            raise TypeError("projected upstream factor is invalid")
        if not isinstance(self.downstream_factor, ModalConnectivityFactor):
            raise TypeError("projected downstream factor is invalid")
        self.upstream_factor.validate_integrity()
        self.downstream_factor.validate_integrity()
        _require_name(
            self.downstream_input_port_name,
            label="projected downstream input port",
        )
        _require_sha256(
            self.direct_jacobian_evidence_sha256,
            label="projected direct Jacobian evidence digest",
        )
        for field in (
            "source_component",
            "source_port",
            "target_component",
            "target_port",
        ):
            _require_name(getattr(self, field), label=field)
        if self.source_component == self.target_component:
            raise ValueError("projected modal connection cannot be a self-edge")
        fine_jacobian = _cpu_float64(
            self.fine_jacobian,
            label="projected fine Jacobian",
            ndim=2,
        )
        matrix = _cpu_float64(
            self.matrix,
            label="projected modal matrix",
            ndim=2,
        )
        expected = project_modal_jacobian(
            upstream_factor=self.upstream_factor,
            downstream_factor=self.downstream_factor,
            downstream_input_port_name=self.downstream_input_port_name,
            fine_jacobian=fine_jacobian,
        )
        if not torch.equal(matrix, expected):
            raise ValueError(
                "projected modal matrix does not equal R_j J_ji P_i"
            )
        object.__setattr__(self, "fine_jacobian", fine_jacobian)
        object.__setattr__(self, "matrix", matrix)
        if (
            self.artifact_kind != _PROJECTED_CONNECTION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("projected modal connection header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_PROJECTED_CONNECTION_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="projected connection artifact_sha256",
                )
                != computed
            ):
                raise ValueError("projected modal connection hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_expansion_sha256": self.source_expansion_sha256,
            "upstream_factor_sha256": (
                self.upstream_factor.artifact_sha256
            ),
            "downstream_factor_sha256": (
                self.downstream_factor.artifact_sha256
            ),
            "downstream_input_port_name": (
                self.downstream_input_port_name
            ),
            "fine_jacobian_sha256": _tensor_sha256(
                self.fine_jacobian,
                label="projected fine Jacobian",
            ),
            "direct_jacobian_evidence_sha256": (
                self.direct_jacobian_evidence_sha256
            ),
            "source_component": self.source_component,
            "source_port": self.source_port,
            "target_component": self.target_component,
            "target_port": self.target_port,
            "projected_matrix_sha256": _tensor_sha256(
                self.matrix,
                label="projected modal matrix",
            ),
        }

    def validate_integrity(self) -> None:
        self.__post_init__()

    def as_direct_connection(self) -> DirectModalConnection:
        """Lower the verified projected edge into the executable graph IR."""

        self.validate_integrity()
        return DirectModalConnection(
            source_component=self.source_component,
            source_port=self.source_port,
            target_component=self.target_component,
            target_port=self.target_port,
            matrix=self.matrix,
            evidence_kind="direct_jacobian",
            evidence_sha256=self.artifact_sha256,
        )


@dataclass(frozen=True, slots=True)
class BoundaryInputInjection:
    """A signed graph-boundary input map into a component input port."""

    boundary_port: str
    target_component: str
    target_port: str
    matrix: LinearMap
    cut_edge_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _INJECTION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.boundary_port, label="boundary_port")
        _require_name(self.target_component, label="target_component")
        _require_name(self.target_port, label="target_port")
        _require_sha256(
            self.cut_edge_sha256,
            label="injection cut_edge_sha256",
        )
        matrix = _canonical_linear_map(
            self.matrix,
            label="boundary input injection matrix",
        )
        if 0 in _linear_map_shape(matrix):
            raise ValueError("injection matrix dimensions must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        if (
            self.artifact_kind != _INJECTION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("boundary input injection header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_INJECTION_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="injection artifact_sha256",
                )
                != computed
            ):
                raise ValueError("boundary input injection hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def stored_scalar_count(self) -> int:
        return _linear_map_stored_scalar_count(self.matrix)

    @property
    def macs_per_row(self) -> int:
        return _linear_map_macs_per_row(self.matrix)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "boundary_port": self.boundary_port,
            "target_component": self.target_component,
            "target_port": self.target_port,
            "linear_map": _linear_map_payload(
                self.matrix,
                label="injection matrix",
            ),
            "cut_edge_sha256": self.cut_edge_sha256,
        }

    def validate_integrity(self) -> None:
        if isinstance(self.matrix, ImplicitIdentityMap):
            self.matrix.validate_integrity()
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_INJECTION_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("boundary input injection hash mismatch")


@dataclass(frozen=True, slots=True)
class BoundaryOutputReadout:
    """A signed component-output map onto one graph boundary output."""

    source_component: str
    source_port: str
    boundary_port: str
    matrix: LinearMap
    cut_edge_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _READOUT_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.source_component, label="source_component")
        _require_name(self.source_port, label="source_port")
        _require_name(self.boundary_port, label="boundary_port")
        _require_sha256(
            self.cut_edge_sha256,
            label="readout cut_edge_sha256",
        )
        matrix = _canonical_linear_map(
            self.matrix,
            label="boundary output readout matrix",
        )
        if 0 in _linear_map_shape(matrix):
            raise ValueError("readout matrix dimensions must be nonempty")
        object.__setattr__(self, "matrix", matrix)
        if (
            self.artifact_kind != _READOUT_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("boundary output readout header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_READOUT_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="readout artifact_sha256",
                )
                != computed
            ):
                raise ValueError("boundary output readout hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def stored_scalar_count(self) -> int:
        return _linear_map_stored_scalar_count(self.matrix)

    @property
    def macs_per_row(self) -> int:
        return _linear_map_macs_per_row(self.matrix)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_component": self.source_component,
            "source_port": self.source_port,
            "boundary_port": self.boundary_port,
            "linear_map": _linear_map_payload(
                self.matrix,
                label="readout matrix",
            ),
            "cut_edge_sha256": self.cut_edge_sha256,
        }

    def validate_integrity(self) -> None:
        if isinstance(self.matrix, ImplicitIdentityMap):
            self.matrix.validate_integrity()
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_READOUT_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("boundary output readout hash mismatch")


@dataclass(frozen=True, slots=True)
class LinearModalGraphLevel:
    """A strict causal DAG of affine components at one hierarchy level."""

    graph_id: str
    level_index: int
    source_artifact_sha256: str
    components: tuple[ModalComponent, ...]
    connections: tuple[DirectModalConnection, ...]
    boundary_inputs: tuple[ModalBoundaryPort, ...]
    boundary_outputs: tuple[ModalBoundaryPort, ...]
    input_injections: tuple[BoundaryInputInjection, ...]
    output_readouts: tuple[BoundaryOutputReadout, ...]
    output_offsets: tuple[Tensor, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _GRAPH_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.graph_id, label="graph_id")
        _require_int(self.level_index, label="level_index", minimum=0)
        _require_sha256(
            self.source_artifact_sha256,
            label="graph source_artifact_sha256",
        )
        if (
            self.artifact_kind != _GRAPH_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("linear modal graph header is invalid")
        if (
            type(self.components) is not tuple
            or not self.components
            or any(
                not isinstance(
                    value,
                    (AffineModalComponent, IdentityModalComponent),
                )
                for value in self.components
            )
        ):
            raise ValueError("components must be a nonempty component tuple")
        expected_components = tuple(
            sorted(
                self.components,
                key=lambda value: (
                    value.causal_start,
                    value.causal_end,
                    value.component_id,
                ),
            )
        )
        if self.components != expected_components:
            raise ValueError("components must be in canonical causal order")
        names = tuple(value.component_id for value in self.components)
        if len(names) != len(set(names)):
            raise ValueError("component IDs must be unique")
        for component in self.components:
            component.validate_integrity()
            if component.level_index != self.level_index:
                raise ValueError("component level does not match graph level")
        for left, right in zip(
            self.components,
            self.components[1:],
        ):
            if left.causal_end >= right.causal_start:
                raise ValueError(
                    "component causal intervals must be disjoint and ordered"
                )

        if (
            type(self.connections) is not tuple
            or any(
                not isinstance(value, DirectModalConnection)
                for value in self.connections
            )
            or self.connections
            != tuple(sorted(self.connections, key=_canonical_edge_key))
        ):
            raise ValueError(
                "connections must be a canonical connection tuple"
            )
        edge_keys = tuple(_canonical_edge_key(edge) for edge in self.connections)
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("direct modal connections must be unique")
        if (
            type(self.boundary_inputs) is not tuple
            or not self.boundary_inputs
            or type(self.boundary_outputs) is not tuple
            or not self.boundary_outputs
        ):
            raise ValueError("graph boundary ports must be nonempty tuples")
        expected_inputs = tuple(
            sorted(
                self.boundary_inputs,
                key=lambda port: (port.causal_order, port.name),
            )
        )
        expected_outputs = tuple(
            sorted(
                self.boundary_outputs,
                key=lambda port: (port.causal_order, port.name),
            )
        )
        if self.boundary_inputs != expected_inputs:
            raise ValueError("boundary inputs must be in canonical order")
        if self.boundary_outputs != expected_outputs:
            raise ValueError("boundary outputs must be in canonical order")
        if any(port.direction != "input" for port in self.boundary_inputs):
            raise ValueError("boundary_inputs contain a non-input port")
        if any(port.direction != "output" for port in self.boundary_outputs):
            raise ValueError("boundary_outputs contain a non-output port")
        boundary_names = tuple(
            port.name
            for port in self.boundary_inputs + self.boundary_outputs
        )
        if len(boundary_names) != len(set(boundary_names)):
            raise ValueError("graph boundary port names must be unique")
        for port in self.boundary_inputs + self.boundary_outputs:
            port.validate_integrity()
            if port.owner_id != self.graph_id:
                raise ValueError("graph boundary port owner does not match graph")
        if any(
            not any(
                input_port.causal_order <= output_port.causal_order
                for input_port in self.boundary_inputs
            )
            for output_port in self.boundary_outputs
        ):
            raise ValueError(
                "every boundary output requires a nonempty causal input "
                "prefix"
            )

        if (
            type(self.input_injections) is not tuple
            or any(
                not isinstance(value, BoundaryInputInjection)
                for value in self.input_injections
            )
        ):
            raise ValueError("input_injections must be an injection tuple")
        if (
            type(self.output_readouts) is not tuple
            or any(
                not isinstance(value, BoundaryOutputReadout)
                for value in self.output_readouts
            )
        ):
            raise ValueError("output_readouts must be a readout tuple")
        expected_injections = tuple(
            sorted(
                self.input_injections,
                key=lambda value: (
                    value.boundary_port,
                    value.target_component,
                    value.target_port,
                    value.artifact_sha256,
                ),
            )
        )
        expected_readouts = tuple(
            sorted(
                self.output_readouts,
                key=lambda value: (
                    value.source_component,
                    value.source_port,
                    value.boundary_port,
                    value.artifact_sha256,
                ),
            )
        )
        if self.input_injections != expected_injections:
            raise ValueError("input_injections must be in canonical order")
        if self.output_readouts != expected_readouts:
            raise ValueError("output_readouts must be in canonical order")
        injection_keys = tuple(
            (
                value.boundary_port,
                value.target_component,
                value.target_port,
            )
            for value in self.input_injections
        )
        if len(injection_keys) != len(set(injection_keys)):
            raise ValueError("boundary input injections must be unique")
        readout_keys = tuple(
            (
                value.source_component,
                value.source_port,
                value.boundary_port,
            )
            for value in self.output_readouts
        )
        if len(readout_keys) != len(set(readout_keys)):
            raise ValueError("boundary output readouts must be unique")
        for value in self.input_injections:
            value.validate_integrity()
        for value in self.output_readouts:
            value.validate_integrity()

        if (
            type(self.output_offsets) is not tuple
            or len(self.output_offsets) != len(self.boundary_outputs)
        ):
            raise ValueError("output_offsets must match boundary outputs")
        offsets: list[Tensor] = []
        for port, value in zip(
            self.boundary_outputs,
            self.output_offsets,
            strict=True,
        ):
            canonical = _cpu_float64(
                value,
                label=f"graph output offset {port.name}",
                ndim=1,
            )
            if canonical.shape != (port.width,):
                raise ValueError("graph output offset width is invalid")
            offsets.append(canonical)
        object.__setattr__(self, "output_offsets", tuple(offsets))
        self._validate_wiring()

        computed = _json_sha256(self._hash_payload(), domain=_GRAPH_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="graph artifact_sha256",
                )
                != computed
            ):
                raise ValueError("linear modal graph hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def components_by_id(self) -> dict[str, ModalComponent]:
        return {value.component_id: value for value in self.components}

    @property
    def boundary_inputs_by_name(self) -> dict[str, ModalBoundaryPort]:
        return {value.name: value for value in self.boundary_inputs}

    @property
    def boundary_outputs_by_name(self) -> dict[str, ModalBoundaryPort]:
        return {value.name: value for value in self.boundary_outputs}

    def _validate_wiring(self) -> None:
        components = self.components_by_id
        inputs = self.boundary_inputs_by_name
        outputs = self.boundary_outputs_by_name
        for connection in self.connections:
            if (
                connection.source_component not in components
                or connection.target_component not in components
            ):
                raise ValueError("connection references an unknown component")
            source = components[connection.source_component]
            target = components[connection.target_component]
            if source.causal_end >= target.causal_start:
                raise ValueError(
                    "connection is backward or violates strict causal order"
                )
            if connection.source_port not in source.output_ports:
                raise ValueError("connection source port is unknown")
            if connection.target_port not in target.input_ports:
                raise ValueError("connection target port is unknown")
            source_port = source.output_ports[connection.source_port]
            target_port = target.input_ports[connection.target_port]
            if _linear_map_shape(connection.matrix) != (
                target_port.width,
                source_port.width,
            ):
                raise ValueError("connection matrix width is invalid")
            if source_port.causal_order > target_port.causal_order:
                raise ValueError("connection contains a future-to-past edge")
        seen_inputs: set[str] = set()
        for injection in self.input_injections:
            if injection.boundary_port not in inputs:
                raise ValueError("injection boundary port is unknown")
            if injection.target_component not in components:
                raise ValueError("injection target component is unknown")
            target = components[injection.target_component]
            if injection.target_port not in target.input_ports:
                raise ValueError("injection target port is unknown")
            boundary = inputs[injection.boundary_port]
            target_port = target.input_ports[injection.target_port]
            if _linear_map_shape(injection.matrix) != (
                target_port.width,
                boundary.width,
            ):
                raise ValueError("injection matrix width is invalid")
            if boundary.causal_order > target_port.causal_order:
                raise ValueError("boundary injection is future-to-past")
            seen_inputs.add(boundary.name)
        if seen_inputs != set(inputs):
            raise ValueError("every boundary input must have an injection")

        seen_outputs: set[str] = set()
        for readout in self.output_readouts:
            if readout.source_component not in components:
                raise ValueError("readout source component is unknown")
            if readout.boundary_port not in outputs:
                raise ValueError("readout boundary port is unknown")
            source = components[readout.source_component]
            if readout.source_port not in source.output_ports:
                raise ValueError("readout source port is unknown")
            source_port = source.output_ports[readout.source_port]
            boundary = outputs[readout.boundary_port]
            if _linear_map_shape(readout.matrix) != (
                boundary.width,
                source_port.width,
            ):
                raise ValueError("readout matrix width is invalid")
            if source_port.causal_order > boundary.causal_order:
                raise ValueError("boundary readout is future-to-past")
            seen_outputs.add(boundary.name)
        if seen_outputs != set(outputs):
            raise ValueError("every boundary output must have a readout")

    @property
    def stored_scalar_count(self) -> int:
        return (
            sum(value.stored_scalar_count for value in self.components)
            + sum(value.stored_scalar_count for value in self.connections)
            + sum(
                value.stored_scalar_count for value in self.input_injections
            )
            + sum(
                value.stored_scalar_count for value in self.output_readouts
            )
            + sum(
                value.numel()
                for value in self.output_offsets
                if bool(torch.count_nonzero(value).item())
            )
        )

    @property
    def macs_per_row(self) -> int:
        return (
            sum(value.macs_per_row for value in self.components)
            + sum(value.macs_per_row for value in self.connections)
            + sum(value.macs_per_row for value in self.input_injections)
            + sum(value.macs_per_row for value in self.output_readouts)
        )

    def _validate_runtime_inputs(
        self,
        inputs: Mapping[str, Tensor],
    ) -> tuple[tuple[int, ...], torch.device, torch.dtype]:
        if not isinstance(inputs, Mapping):
            raise TypeError("graph inputs must be a mapping")
        if set(inputs) != {port.name for port in self.boundary_inputs}:
            raise ValueError("graph input names do not match boundary")
        leading: tuple[int, ...] | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for port in self.boundary_inputs:
            value = inputs[port.name]
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("graph inputs must be floating Tensors")
            if value.shape[-1:] != (port.width,):
                raise ValueError(f"graph input {port.name} has wrong width")
            if not torch.isfinite(value).all():
                raise ValueError("graph inputs must be finite")
            if leading is None:
                leading = tuple(value.shape[:-1])
                device = value.device
                dtype = value.dtype
            elif (
                tuple(value.shape[:-1]) != leading
                or value.device != device
                or value.dtype != dtype
            ):
                raise ValueError(
                    "graph inputs must share leading shape, device, and dtype"
                )
        assert leading is not None
        assert device is not None
        assert dtype is not None
        return leading, device, dtype

    def execute(self, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self.validate_integrity()
        leading, device, dtype = self._validate_runtime_inputs(inputs)
        incoming_connections: dict[
            tuple[str, str],
            list[DirectModalConnection],
        ] = defaultdict(list)
        for edge in self.connections:
            incoming_connections[
                (edge.target_component, edge.target_port)
            ].append(edge)
        incoming_injections: dict[
            tuple[str, str],
            list[BoundaryInputInjection],
        ] = defaultdict(list)
        for injection in self.input_injections:
            incoming_injections[
                (injection.target_component, injection.target_port)
            ].append(injection)

        component_outputs: dict[tuple[str, str], Tensor] = {}
        for component in self.components:
            local_inputs: dict[str, Tensor] = {}
            for port in component.input_ports.values():
                value = torch.zeros(
                    (*leading, port.width),
                    device=device,
                    dtype=dtype,
                )
                for injection in incoming_injections[
                    (component.component_id, port.name)
                ]:
                    value = value + _apply_linear_map(
                        inputs[injection.boundary_port],
                        injection.matrix,
                    )
                for edge in incoming_connections[
                    (component.component_id, port.name)
                ]:
                    value = value + _apply_linear_map(
                        component_outputs[
                            (edge.source_component, edge.source_port)
                        ],
                        edge.matrix,
                    )
                local_inputs[port.name] = value
            if isinstance(component, IdentityModalComponent):
                local_outputs = {
                    component.output_port.name: local_inputs[
                        component.input_port.name
                    ]
                }
            else:
                local_outputs = component.transfer.execute(local_inputs)
            for name, value in local_outputs.items():
                component_outputs[(component.component_id, name)] = value

        result: dict[str, Tensor] = {}
        readouts_by_output: dict[str, list[BoundaryOutputReadout]] = (
            defaultdict(list)
        )
        for readout in self.output_readouts:
            readouts_by_output[readout.boundary_port].append(readout)
        for port, offset in zip(
            self.boundary_outputs,
            self.output_offsets,
            strict=True,
        ):
            value = offset.to(device=device, dtype=dtype).expand(
                *leading,
                port.width,
            )
            for readout in readouts_by_output[port.name]:
                value = value + _apply_linear_map(
                    component_outputs[
                        (readout.source_component, readout.source_port)
                    ],
                    readout.matrix,
                )
            result[port.name] = value
        return result

    def boundary_transfer(self) -> CausalBoundaryTransfer:
        """Symbolically compose the graph into its exact affine boundary map."""

        self.validate_integrity()
        total_input_width = sum(port.width for port in self.boundary_inputs)
        input_slices: dict[str, slice] = {}
        start = 0
        for port in self.boundary_inputs:
            input_slices[port.name] = slice(start, start + port.width)
            start += port.width

        incoming_connections: dict[
            tuple[str, str],
            list[DirectModalConnection],
        ] = defaultdict(list)
        for edge in self.connections:
            incoming_connections[
                (edge.target_component, edge.target_port)
            ].append(edge)
        incoming_injections: dict[
            tuple[str, str],
            list[BoundaryInputInjection],
        ] = defaultdict(list)
        for injection in self.input_injections:
            incoming_injections[
                (injection.target_component, injection.target_port)
            ].append(injection)

        output_linear: dict[tuple[str, str], Tensor] = {}
        output_bias: dict[tuple[str, str], Tensor] = {}
        for component in self.components:
            local_linear: dict[str, Tensor] = {}
            local_bias: dict[str, Tensor] = {}
            for port in component.input_ports.values():
                coefficient = torch.zeros(
                    (port.width, total_input_width),
                    dtype=torch.float64,
                )
                bias = torch.zeros(port.width, dtype=torch.float64)
                for injection in incoming_injections[
                    (component.component_id, port.name)
                ]:
                    source_slice = input_slices[injection.boundary_port]
                    if isinstance(
                        injection.matrix,
                        ImplicitIdentityMap,
                    ):
                        coefficient[:, source_slice] += torch.eye(
                            injection.matrix.width,
                            dtype=torch.float64,
                        )
                    else:
                        coefficient[:, source_slice] += injection.matrix
                for edge in incoming_connections[
                    (component.component_id, port.name)
                ]:
                    coefficient += _left_multiply(
                        edge.matrix,
                        output_linear[
                            (edge.source_component, edge.source_port)
                        ],
                    )
                    bias += _left_multiply(
                        edge.matrix,
                        output_bias[
                            (edge.source_component, edge.source_port)
                        ],
                    )
                local_linear[port.name] = coefficient
                local_bias[port.name] = bias

            if isinstance(component, IdentityModalComponent):
                output_linear[
                    (component.component_id, component.output_port.name)
                ] = local_linear[component.input_port.name]
                output_bias[
                    (component.component_id, component.output_port.name)
                ] = local_bias[component.input_port.name]
            else:
                for index, (port, prefix) in enumerate(
                    zip(
                        component.transfer.output_ports,
                        component.transfer.input_prefixes,
                        strict=True,
                    )
                ):
                    joined_linear = torch.cat(
                        tuple(local_linear[name] for name in prefix),
                        dim=0,
                    )
                    joined_bias = torch.cat(
                        tuple(local_bias[name] for name in prefix),
                        dim=0,
                    )
                    matrix = component.transfer.transfer_matrices[index]
                    output_linear[(component.component_id, port.name)] = (
                        matrix @ joined_linear
                    )
                    output_bias[(component.component_id, port.name)] = (
                        component.transfer.affine_offsets[index]
                        + matrix @ joined_bias
                    )

        readouts_by_output: dict[str, list[BoundaryOutputReadout]] = (
            defaultdict(list)
        )
        for readout in self.output_readouts:
            readouts_by_output[readout.boundary_port].append(readout)
        transfer_matrices: list[Tensor] = []
        offsets: list[Tensor] = []
        prefixes: list[tuple[str, ...]] = []
        numerical_scale = max(
            (
                _linear_map_scale(value.matrix)
                for value in (
                    self.connections
                    + self.input_injections
                    + self.output_readouts
                )
            ),
            default=1.0,
        )
        causal_tolerance = (
            max(numerical_scale, 1.0)
            * torch.finfo(torch.float64).eps
            * max(total_input_width, 1)
            * 512
        )
        for output, base_offset in zip(
            self.boundary_outputs,
            self.output_offsets,
            strict=True,
        ):
            coefficient = torch.zeros(
                (output.width, total_input_width),
                dtype=torch.float64,
            )
            offset = base_offset.clone()
            for readout in readouts_by_output[output.name]:
                coefficient += _left_multiply(
                    readout.matrix,
                    output_linear[
                        (readout.source_component, readout.source_port)
                    ],
                )
                offset += _left_multiply(
                    readout.matrix,
                    output_bias[
                        (readout.source_component, readout.source_port)
                    ],
                )
            legal = tuple(
                port
                for port in self.boundary_inputs
                if port.causal_order <= output.causal_order
            )
            illegal = tuple(
                port
                for port in self.boundary_inputs
                if port.causal_order > output.causal_order
            )
            if illegal:
                future = torch.cat(
                    tuple(
                        coefficient[:, input_slices[port.name]]
                        for port in illegal
                    ),
                    dim=1,
                )
                if (
                    future.numel()
                    and float(future.abs().max().item()) > causal_tolerance
                ):
                    raise ValueError(
                        f"graph induces future-to-past boundary transfer "
                        f"for {output.name}"
                    )
            prefixes.append(tuple(port.name for port in legal))
            transfer_matrices.append(
                torch.cat(
                    tuple(
                        coefficient[:, input_slices[port.name]]
                        for port in legal
                    ),
                    dim=1,
                )
            )
            offsets.append(offset)
        return CausalBoundaryTransfer(
            source_level_sha256=self.artifact_sha256,
            input_ports=self.boundary_inputs,
            output_ports=self.boundary_outputs,
            input_prefixes=tuple(prefixes),
            transfer_matrices=tuple(transfer_matrices),
            affine_offsets=tuple(offsets),
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "graph_id": self.graph_id,
            "level_index": self.level_index,
            "source_artifact_sha256": self.source_artifact_sha256,
            "component_sha256s": tuple(
                value.artifact_sha256 for value in self.components
            ),
            "connection_sha256s": tuple(
                value.artifact_sha256 for value in self.connections
            ),
            "boundary_input_sha256s": tuple(
                value.artifact_sha256 for value in self.boundary_inputs
            ),
            "boundary_output_sha256s": tuple(
                value.artifact_sha256 for value in self.boundary_outputs
            ),
            "input_injection_sha256s": tuple(
                value.artifact_sha256 for value in self.input_injections
            ),
            "output_readout_sha256s": tuple(
                value.artifact_sha256 for value in self.output_readouts
            ),
            "output_offset_sha256s": tuple(
                _tensor_sha256(value, label="graph output offset")
                for value in self.output_offsets
            ),
            "stored_scalar_count": self.stored_scalar_count,
            "macs_per_row": self.macs_per_row,
        }

    def validate_integrity(self) -> None:
        for component in self.components:
            component.validate_integrity()
        for value in (
            self.connections + self.input_injections + self.output_readouts
        ):
            value.validate_integrity()
        self._validate_wiring()
        if (
            _json_sha256(self._hash_payload(), domain=_GRAPH_DOMAIN)
            != self.artifact_sha256
        ):
            raise ValueError("linear modal graph hash mismatch")


@dataclass(frozen=True, slots=True)
class CausalCoarseningGroup:
    """A contiguous child interval with every crossing edge enumerated."""

    parent_graph_sha256: str
    parent_id: str
    child_component_ids: tuple[str, ...]
    internal_connection_sha256s: tuple[str, ...]
    incoming_connection_sha256s: tuple[str, ...]
    outgoing_connection_sha256s: tuple[str, ...]
    boundary_injection_sha256s: tuple[str, ...]
    boundary_readout_sha256s: tuple[str, ...]
    extracted_child_sha256: str
    artifact_sha256: str = ""
    artifact_kind: str = _GROUP_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.parent_graph_sha256,
            label="group parent_graph_sha256",
        )
        _require_name(self.parent_id, label="group parent_id")
        if (
            type(self.child_component_ids) is not tuple
            or not self.child_component_ids
            or len(self.child_component_ids)
            != len(set(self.child_component_ids))
        ):
            raise ValueError(
                "group child_component_ids must be nonempty and unique"
            )
        for child in self.child_component_ids:
            _require_name(child, label="group child component")
        digest_fields = (
            "internal_connection_sha256s",
            "incoming_connection_sha256s",
            "outgoing_connection_sha256s",
            "boundary_injection_sha256s",
            "boundary_readout_sha256s",
        )
        all_edges: list[str] = []
        for field in digest_fields:
            values = getattr(self, field)
            if type(values) is not tuple or values != tuple(sorted(set(values))):
                raise ValueError(f"{field} must be a sorted unique tuple")
            for digest in values:
                _require_sha256(digest, label=field)
            all_edges.extend(values)
        if len(all_edges) != len(set(all_edges)):
            raise ValueError("group edge classifications overlap")
        _require_sha256(
            self.extracted_child_sha256,
            label="group extracted_child_sha256",
        )
        if (
            self.artifact_kind != _GROUP_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("causal coarsening group header is invalid")
        computed = _json_sha256(self._hash_payload(), domain=_GROUP_DOMAIN)
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="group artifact_sha256",
                )
                != computed
            ):
                raise ValueError("causal coarsening group hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @classmethod
    def from_graph(
        cls,
        graph: LinearModalGraphLevel,
        *,
        parent_id: str,
        child_component_ids: tuple[str, ...],
    ) -> CausalCoarseningGroup:
        if not isinstance(graph, LinearModalGraphLevel):
            raise TypeError("graph must be a LinearModalGraphLevel")
        graph.validate_integrity()
        selected = set(child_component_ids)
        ordered = tuple(
            component.component_id
            for component in graph.components
            if component.component_id in selected
        )
        if ordered != child_component_ids or len(ordered) != len(selected):
            raise ValueError(
                "group children must follow graph causal order exactly"
            )
        positions = tuple(
            index
            for index, component in enumerate(graph.components)
            if component.component_id in selected
        )
        if not positions or positions != tuple(
            range(positions[0], positions[-1] + 1)
        ):
            raise ValueError(
                "causal coarsening groups must be contiguous intervals"
            )
        internal: list[str] = []
        incoming: list[str] = []
        outgoing: list[str] = []
        for edge in graph.connections:
            source_inside = edge.source_component in selected
            target_inside = edge.target_component in selected
            if source_inside and target_inside:
                internal.append(edge.artifact_sha256)
            elif not source_inside and target_inside:
                incoming.append(edge.artifact_sha256)
            elif source_inside and not target_inside:
                outgoing.append(edge.artifact_sha256)
        injections = tuple(
            value.artifact_sha256
            for value in graph.input_injections
            if value.target_component in selected
        )
        readouts = tuple(
            value.artifact_sha256
            for value in graph.output_readouts
            if value.source_component in selected
        )
        extracted_child = _extract_coarsening_group_unchecked(
            graph,
            parent_id=parent_id,
            child_component_ids=child_component_ids,
        )
        return cls(
            parent_graph_sha256=graph.artifact_sha256,
            parent_id=parent_id,
            child_component_ids=child_component_ids,
            internal_connection_sha256s=tuple(sorted(internal)),
            incoming_connection_sha256s=tuple(sorted(incoming)),
            outgoing_connection_sha256s=tuple(sorted(outgoing)),
            boundary_injection_sha256s=tuple(sorted(injections)),
            boundary_readout_sha256s=tuple(sorted(readouts)),
            extracted_child_sha256=extracted_child.artifact_sha256,
        )

    def validate_against(self, graph: LinearModalGraphLevel) -> None:
        expected = CausalCoarseningGroup.from_graph(
            graph,
            parent_id=self.parent_id,
            child_component_ids=self.child_component_ids,
        )
        if self._hash_payload() != expected._hash_payload():
            raise ValueError(
                "coarsening group omits or misclassifies a cut edge"
            )
        if graph.artifact_sha256 != self.parent_graph_sha256:
            raise ValueError("coarsening group parent graph digest is stale")

    def validate_extracted_child(
        self,
        child_graph: LinearModalGraphLevel,
    ) -> None:
        """Authenticate the complete cut certificate on an extracted child."""

        if not isinstance(child_graph, LinearModalGraphLevel):
            raise TypeError("child_graph must be a LinearModalGraphLevel")
        child_graph.validate_integrity()
        if (
            child_graph.source_artifact_sha256
            != self.parent_graph_sha256
            or child_graph.graph_id != self.parent_id
            or tuple(
                component.component_id
                for component in child_graph.components
            )
            != self.child_component_ids
        ):
            raise ValueError(
                "extracted child does not bind the coarsening group"
            )
        if child_graph.artifact_sha256 != self.extracted_child_sha256:
            raise ValueError(
                "extracted child graph differs from certified extraction"
            )
        if tuple(
            sorted(
                connection.artifact_sha256
                for connection in child_graph.connections
            )
        ) != self.internal_connection_sha256s:
            raise ValueError(
                "extracted child internal connections differ from group"
            )
        expected_inputs = tuple(
            sorted(
                self.incoming_connection_sha256s
                + self.boundary_injection_sha256s
            )
        )
        actual_inputs = tuple(
            sorted(
                injection.cut_edge_sha256
                for injection in child_graph.input_injections
            )
        )
        expected_outputs = tuple(
            sorted(
                self.outgoing_connection_sha256s
                + self.boundary_readout_sha256s
            )
        )
        actual_outputs = tuple(
            sorted(
                readout.cut_edge_sha256
                for readout in child_graph.output_readouts
            )
        )
        if (
            actual_inputs != expected_inputs
            or actual_outputs != expected_outputs
        ):
            raise ValueError(
                "extracted child omits or rewrites a certified cut edge"
            )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "parent_graph_sha256": self.parent_graph_sha256,
            "parent_id": self.parent_id,
            "child_component_ids": self.child_component_ids,
            "internal_connection_sha256s": (
                self.internal_connection_sha256s
            ),
            "incoming_connection_sha256s": (
                self.incoming_connection_sha256s
            ),
            "outgoing_connection_sha256s": (
                self.outgoing_connection_sha256s
            ),
            "boundary_injection_sha256s": (
                self.boundary_injection_sha256s
            ),
            "boundary_readout_sha256s": (
                self.boundary_readout_sha256s
            ),
            "extracted_child_sha256": self.extracted_child_sha256,
        }


def _extract_coarsening_group_unchecked(
    graph: LinearModalGraphLevel,
    *,
    parent_id: str,
    child_component_ids: tuple[str, ...],
) -> LinearModalGraphLevel:
    """Deterministically build the child whose digest the group certifies."""

    graph.validate_integrity()
    selected = set(child_component_ids)
    components = tuple(
        value for value in graph.components if value.component_id in selected
    )
    internal = tuple(
        value
        for value in graph.connections
        if value.source_component in selected
        and value.target_component in selected
    )
    by_component = graph.components_by_id

    boundary_inputs: list[ModalBoundaryPort] = []
    injections: list[BoundaryInputInjection] = []
    for original in graph.input_injections:
        if original.target_component not in selected:
            continue
        source = graph.boundary_inputs_by_name[original.boundary_port]
        target_component = by_component[original.target_component]
        target_port = target_component.input_ports[original.target_port]
        port = ModalBoundaryPort(
            name=f"{parent_id}.in.boundary-{original.artifact_sha256[:16]}",
            direction="input",
            # The cut value is consumed at the selected child's input stage.
            # Keeping the surfaced port inside the child interval makes the
            # contracted parent recursively reusable without introducing a
            # fictitious earlier component stage.
            causal_order=target_port.causal_order,
            width=source.width,
            owner_id=parent_id,
        )
        boundary_inputs.append(port)
        injections.append(
            BoundaryInputInjection(
                boundary_port=port.name,
                target_component=original.target_component,
                target_port=original.target_port,
                matrix=original.matrix,
                cut_edge_sha256=original.artifact_sha256,
            )
        )
    for edge in graph.connections:
        if (
            edge.source_component in selected
            or edge.target_component not in selected
        ):
            continue
        target_component = by_component[edge.target_component]
        target_port = target_component.input_ports[edge.target_port]
        source_component = by_component[edge.source_component]
        source_port = source_component.output_ports[edge.source_port]
        port = ModalBoundaryPort(
            name=f"{parent_id}.in.cut-{edge.artifact_sha256[:16]}",
            direction="input",
            causal_order=target_port.causal_order,
            width=source_port.width,
            owner_id=parent_id,
        )
        boundary_inputs.append(port)
        injections.append(
            BoundaryInputInjection(
                boundary_port=port.name,
                target_component=edge.target_component,
                target_port=edge.target_port,
                matrix=edge.matrix,
                cut_edge_sha256=edge.artifact_sha256,
            )
        )

    boundary_outputs: list[ModalBoundaryPort] = []
    readouts: list[BoundaryOutputReadout] = []
    output_offsets: list[Tensor] = []
    graph_output_offsets = {
        port.name: offset
        for port, offset in zip(
            graph.boundary_outputs,
            graph.output_offsets,
            strict=True,
        )
    }
    offset_owner_by_output: dict[str, str] = {}
    for readout in graph.output_readouts:
        offset_owner_by_output.setdefault(
            readout.boundary_port,
            readout.artifact_sha256,
        )
    for original in graph.output_readouts:
        if original.source_component not in selected:
            continue
        target = graph.boundary_outputs_by_name[original.boundary_port]
        source_component = by_component[original.source_component]
        source_port = source_component.output_ports[original.source_port]
        port = ModalBoundaryPort(
            name=f"{parent_id}.out.boundary-{original.artifact_sha256[:16]}",
            direction="output",
            # The readout contribution is available when its selected source
            # produces it; the enclosing graph may delay accumulation.
            causal_order=source_port.causal_order,
            width=target.width,
            owner_id=parent_id,
        )
        boundary_outputs.append(port)
        readouts.append(
            BoundaryOutputReadout(
                source_component=original.source_component,
                source_port=original.source_port,
                boundary_port=port.name,
                matrix=original.matrix,
                cut_edge_sha256=original.artifact_sha256,
            )
        )
        if (
            offset_owner_by_output[original.boundary_port]
            == original.artifact_sha256
        ):
            output_offsets.append(
                graph_output_offsets[original.boundary_port]
            )
        else:
            output_offsets.append(
                torch.zeros(target.width, dtype=torch.float64)
            )
    for edge in graph.connections:
        if (
            edge.source_component not in selected
            or edge.target_component in selected
        ):
            continue
        target_component = by_component[edge.target_component]
        target_port = target_component.input_ports[edge.target_port]
        source_component = by_component[edge.source_component]
        source_port = source_component.output_ports[edge.source_port]
        port = ModalBoundaryPort(
            name=f"{parent_id}.out.cut-{edge.artifact_sha256[:16]}",
            direction="output",
            causal_order=source_port.causal_order,
            width=target_port.width,
            owner_id=parent_id,
        )
        boundary_outputs.append(port)
        readouts.append(
            BoundaryOutputReadout(
                source_component=edge.source_component,
                source_port=edge.source_port,
                boundary_port=port.name,
                matrix=edge.matrix,
                cut_edge_sha256=edge.artifact_sha256,
            )
        )
        output_offsets.append(torch.zeros(target_port.width, dtype=torch.float64))

    if not boundary_inputs or not boundary_outputs:
        raise ValueError(
            "coarsening group must expose at least one input and output port"
        )
    sorted_inputs = tuple(
        sorted(boundary_inputs, key=lambda port: (port.causal_order, port.name))
    )
    sorted_outputs_with_offsets = sorted(
        zip(boundary_outputs, output_offsets, strict=True),
        key=lambda item: (item[0].causal_order, item[0].name),
    )
    sorted_outputs = tuple(item[0] for item in sorted_outputs_with_offsets)
    sorted_offsets = tuple(item[1] for item in sorted_outputs_with_offsets)
    return LinearModalGraphLevel(
        graph_id=parent_id,
        level_index=graph.level_index,
        source_artifact_sha256=graph.artifact_sha256,
        components=components,
        connections=tuple(sorted(internal, key=_canonical_edge_key)),
        boundary_inputs=sorted_inputs,
        boundary_outputs=sorted_outputs,
        input_injections=tuple(
            sorted(
                injections,
                key=lambda value: (
                    value.boundary_port,
                    value.target_component,
                    value.target_port,
                    value.artifact_sha256,
                ),
            )
        ),
        output_readouts=tuple(
            sorted(
                readouts,
                key=lambda value: (
                    value.source_component,
                    value.source_port,
                    value.boundary_port,
                    value.artifact_sha256,
                ),
            )
        ),
        output_offsets=sorted_offsets,
    )


def extract_coarsening_group(
    graph: LinearModalGraphLevel,
    group: CausalCoarseningGroup,
) -> LinearModalGraphLevel:
    """Extract and authenticate a complete multi-port contraction child."""

    if not isinstance(graph, LinearModalGraphLevel):
        raise TypeError("graph must be a LinearModalGraphLevel")
    if not isinstance(group, CausalCoarseningGroup):
        raise TypeError("group must be a CausalCoarseningGroup")
    graph.validate_integrity()
    group.validate_against(graph)
    child = _extract_coarsening_group_unchecked(
        graph,
        parent_id=group.parent_id,
        child_component_ids=group.child_component_ids,
    )
    group.validate_extracted_child(child)
    return child


@dataclass(frozen=True, slots=True)
class ModalParameterSharingFamily:
    """A non-causal sharing relation that never owns child components."""

    family_id: str
    member_component_ids: tuple[str, ...]
    nomination_source_sha256: str
    observational_only: bool = True
    authorizes_containment: bool = False
    authorizes_execution: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _SHARING_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.family_id, label="sharing family_id")
        if (
            type(self.member_component_ids) is not tuple
            or len(self.member_component_ids) < 2
            or len(self.member_component_ids)
            != len(set(self.member_component_ids))
        ):
            raise ValueError(
                "sharing family requires at least two unique members"
            )
        for value in self.member_component_ids:
            _require_name(value, label="sharing family member")
        _require_sha256(
            self.nomination_source_sha256,
            label="sharing nomination source digest",
        )
        if (
            self.observational_only is not True
            or self.authorizes_containment is not False
            or self.authorizes_execution is not False
        ):
            raise ValueError(
                "sharing families are observational non-containment metadata"
            )
        if (
            self.artifact_kind != _SHARING_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("sharing family header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_SHARING_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="sharing family artifact_sha256",
                )
                != computed
            ):
                raise ValueError("sharing family hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "family_id": self.family_id,
            "member_component_ids": self.member_component_ids,
            "nomination_source_sha256": self.nomination_source_sha256,
            "observational_only": self.observational_only,
            "authorizes_containment": self.authorizes_containment,
            "authorizes_execution": self.authorizes_execution,
        }


@dataclass(frozen=True, slots=True)
class MeasuredModalCore:
    """A signed mode-to-mode graph with measured boundary moments."""

    source_expansion_sha256: str
    graph: LinearModalGraphLevel
    input_moments: tuple[MessageMoments, ...]
    output_moments: tuple[MessageMoments, ...]
    analysis_only: bool = True
    authorizes_replacement: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _MEASURED_MODAL_CORE_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_expansion_sha256,
            label="measured modal core source expansion digest",
        )
        if not isinstance(self.graph, LinearModalGraphLevel):
            raise TypeError("measured modal core graph is invalid")
        self.graph.validate_integrity()
        if (
            self.graph.source_artifact_sha256
            != self.source_expansion_sha256
        ):
            raise ValueError(
                "measured modal core does not bind its source expansion"
            )
        if not self.graph.connections:
            raise ValueError(
                "measured modal core requires at least one signed "
                "mode-to-mode interaction"
            )
        if (
            type(self.input_moments) is not tuple
            or type(self.output_moments) is not tuple
            or len(self.input_moments) != len(self.graph.boundary_inputs)
            or len(self.output_moments) != len(self.graph.boundary_outputs)
            or any(
                not isinstance(moments, MessageMoments)
                for moments in self.input_moments + self.output_moments
            )
        ):
            raise ValueError(
                "measured modal moments must cover the graph boundary"
            )
        for moments, port in zip(
            self.input_moments,
            self.graph.boundary_inputs,
            strict=True,
        ):
            moments.validate_integrity()
            if (
                moments.port.artifact_sha256 != port.artifact_sha256
                or moments.source_level_sha256
                != self.graph.artifact_sha256
            ):
                raise ValueError(
                    "measured modal input moments are stale"
                )
        for moments, port in zip(
            self.output_moments,
            self.graph.boundary_outputs,
            strict=True,
        ):
            moments.validate_integrity()
            if (
                moments.port.artifact_sha256 != port.artifact_sha256
                or moments.source_level_sha256
                != self.graph.artifact_sha256
            ):
                raise ValueError(
                    "measured modal output moments are stale"
                )
        lineages = {
            (moments.reduction_id, moments.sample_count)
            for moments in self.input_moments + self.output_moments
        }
        if len(lineages) != 1:
            raise ValueError(
                "measured modal moments must share reduction_id and "
                "sample_count"
            )
        expected_means = self.graph.execute(
            {
                moments.port.name: moments.mean.unsqueeze(0)
                for moments in self.input_moments
            }
        )
        for moments in self.output_moments:
            if not torch.allclose(
                moments.mean,
                expected_means[moments.port.name].squeeze(0),
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError(
                    "measured modal output mean is inconsistent with graph"
                )
        if (
            self.analysis_only is not True
            or self.authorizes_replacement is not False
        ):
            raise ValueError(
                "v1 measured modal cores are analysis-only and do not "
                "authorize source replacement"
            )
        if (
            self.artifact_kind != _MEASURED_MODAL_CORE_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("measured modal core header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_MEASURED_MODAL_CORE_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="measured modal core artifact_sha256",
                )
                != computed
            ):
                raise ValueError("measured modal core hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_expansion_sha256": self.source_expansion_sha256,
            "graph_sha256": self.graph.artifact_sha256,
            "input_moment_sha256s": tuple(
                moments.artifact_sha256 for moments in self.input_moments
            ),
            "output_moment_sha256s": tuple(
                moments.artifact_sha256 for moments in self.output_moments
            ),
            "analysis_only": self.analysis_only,
            "authorizes_replacement": self.authorizes_replacement,
        }

    def validate_integrity(self) -> None:
        self.__post_init__()

    def factor_connectivity(
        self,
        *,
        retained_ranks: (
            int | Sequence[int] | Mapping[str, int] | None
        ) = None,
        relative_eigenvalue_cutoff: float = 1e-12,
        relative_singular_value_cutoff: float = 1e-12,
        assume_block_diagonal_input_covariance: bool = False,
    ) -> ModalConnectivityDecomposition:
        """Factor the measured signed interactions at the next rung."""

        self.validate_integrity()
        return factor_modal_connectivity(
            self.graph.boundary_transfer(),
            self.input_moments,
            self.output_moments,
            retained_ranks=retained_ranks,
            relative_eigenvalue_cutoff=relative_eigenvalue_cutoff,
            relative_singular_value_cutoff=(
                relative_singular_value_cutoff
            ),
            assume_block_diagonal_input_covariance=(
                assume_block_diagonal_input_covariance
            ),
        )


@dataclass(frozen=True, slots=True)
class HierarchicalModeExpansion:
    """Fine encoder/decoder plus a typed modal-core handoff scaffold."""

    source_generator_sha256: str
    source_decomposition: ModalConnectivityDecomposition
    graph: LinearModalGraphLevel
    encoder_graph: LinearModalGraphLevel
    recursive_graph: LinearModalGraphLevel
    mode_input_ports: tuple[ModalBoundaryPort, ...]
    mode_ports: tuple[ModalBoundaryPort, ...]
    mode_moments: tuple[ConnectivityModeMoments, ...]
    recursive_input_moments: tuple[MessageMoments, ...]
    recursive_output_moments: tuple[MessageMoments, ...]
    artifact_sha256: str = ""
    artifact_kind: str = _MODE_EXPANSION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_generator_sha256,
            label="mode expansion source generator digest",
        )
        if not isinstance(
            self.source_decomposition,
            ModalConnectivityDecomposition,
        ):
            raise TypeError(
                "mode expansion source must be a connectivity decomposition"
            )
        self.source_decomposition.validate_integrity()
        if not isinstance(self.graph, LinearModalGraphLevel):
            raise TypeError("mode expansion graph must be LinearModalGraphLevel")
        self.graph.validate_integrity()
        if not isinstance(self.encoder_graph, LinearModalGraphLevel):
            raise TypeError("mode encoder must be LinearModalGraphLevel")
        self.encoder_graph.validate_integrity()
        if not isinstance(self.recursive_graph, LinearModalGraphLevel):
            raise TypeError(
                "recursive mode graph must be LinearModalGraphLevel"
            )
        self.recursive_graph.validate_integrity()
        if any(
            value.source_artifact_sha256
            != self.source_generator_sha256
            for value in (
                self.graph,
                self.encoder_graph,
                self.recursive_graph,
            )
        ):
            raise ValueError(
                "mode expansion graphs do not bind the source generator"
            )
        if (
            type(self.mode_input_ports) is not tuple
            or type(self.mode_ports) is not tuple
            or not self.mode_ports
            or type(self.mode_moments) is not tuple
            or len(self.mode_input_ports) != len(self.mode_ports)
            or len(self.mode_ports) != len(self.mode_moments)
        ):
            raise ValueError(
                "mode expansion ports and moments must be aligned nonempty "
                "tuples"
            )
        if (
            self.mode_input_ports
            != self.recursive_graph.boundary_inputs
            or self.mode_ports != self.recursive_graph.boundary_outputs
        ):
            raise ValueError(
                "mode ports must be the recursive graph boundary"
            )
        if (
            type(self.recursive_input_moments) is not tuple
            or type(self.recursive_output_moments) is not tuple
            or len(self.recursive_input_moments)
            != len(self.recursive_graph.boundary_inputs)
            or len(self.recursive_output_moments)
            != len(self.recursive_graph.boundary_outputs)
        ):
            raise ValueError(
                "recursive moments must cover the modal graph boundary"
            )
        names: list[str] = []
        aligned_modes = zip(
            self.mode_input_ports,
            self.mode_ports,
            self.mode_moments,
            self.recursive_input_moments,
            self.recursive_output_moments,
            self.source_decomposition.factors,
            strict=True,
        )
        for (
            input_port,
            port,
            modal_moments,
            input_moments,
            output_moments,
            factor,
        ) in aligned_modes:
            if not isinstance(port, ModalBoundaryPort):
                raise TypeError("mode expansion ports must be boundary ports")
            port.validate_integrity()
            if port.direction != "output":
                raise ValueError("mode coordinates must be graph outputs")
            if not isinstance(modal_moments, ConnectivityModeMoments):
                raise TypeError(
                    "mode expansion moments must be ConnectivityModeMoments"
                )
            modal_moments.validate_integrity()
            if (
                modal_moments.mode_id != port.name
                or modal_moments.causal_order != port.causal_order
                or modal_moments.salience.numel() != port.width
            ):
                raise ValueError(
                    "mode moments do not match their explicit graph port"
                )
            source_modal_moments = factor.mode_moments
            if (
                modal_moments.source_factor_sha256
                != factor.artifact_sha256
                or modal_moments.reduction_id
                != source_modal_moments.reduction_id
                or modal_moments.sample_count
                != source_modal_moments.sample_count
                or not torch.equal(
                    modal_moments.covariance,
                    source_modal_moments.covariance,
                )
                or not torch.equal(
                    modal_moments.fisher,
                    source_modal_moments.fisher,
                )
                or not torch.equal(
                    modal_moments.salience,
                    source_modal_moments.salience,
                )
            ):
                raise ValueError(
                    "mode moments do not match source decomposition"
                )
            output_moments.validate_integrity()
            expected_input_moments = modal_moments.bind_port(
                input_port,
                source_level_sha256=self.recursive_graph.artifact_sha256,
            )
            expected_output_moments = modal_moments.bind_port(
                port,
                source_level_sha256=self.recursive_graph.artifact_sha256,
            )
            if (
                input_moments.artifact_sha256
                != expected_input_moments.artifact_sha256
                or output_moments.artifact_sha256
                != expected_output_moments.artifact_sha256
            ):
                raise ValueError(
                    "recursive output moments do not bind modal moments"
                )
            names.append(port.name)
        for port, moments in zip(
            self.recursive_graph.boundary_inputs,
            self.recursive_input_moments,
            strict=True,
        ):
            moments.validate_integrity()
            if (
                moments.port.artifact_sha256 != port.artifact_sha256
                or moments.source_level_sha256
                != self.recursive_graph.artifact_sha256
            ):
                raise ValueError(
                    "recursive input moments do not bind modal graph"
                )
        fine_input_contract = tuple(
            (port.name, port.width)
            for port in self.graph.boundary_inputs
        )
        encoder_input_contract = tuple(
            (port.name, port.width)
            for port in self.encoder_graph.boundary_inputs
        )
        if fine_input_contract != encoder_input_contract:
            raise ValueError(
                "fine and encoder graphs must share their input contract"
            )
        encoder_output_contract = tuple(
            (port.name, port.width)
            for port in self.encoder_graph.boundary_outputs
        )
        recursive_input_contract = tuple(
            (port.name, port.width)
            for port in self.recursive_graph.boundary_inputs
        )
        if encoder_output_contract != recursive_input_contract:
            raise ValueError(
                "encoder outputs must feed recursive modal inputs"
            )
        fine_output_contract = tuple(
            (port.name, port.width)
            for port in self.graph.boundary_outputs
        )
        source_output_contract = tuple(
            (port.name, port.width)
            for port in self.source_decomposition.source_transfer.output_ports
        )
        if fine_output_contract != source_output_contract:
            raise ValueError(
                "fine graph does not preserve source output contract"
            )
        fine_transfer = self.graph.boundary_transfer()
        expected_fine_transfer = (
            self.source_decomposition.candidate_transfer()
        )
        if (
            fine_transfer.input_prefixes
            != expected_fine_transfer.input_prefixes
            or any(
                not torch.allclose(
                    actual,
                    expected,
                    rtol=2e-10,
                    atol=2e-11
                    * max(
                        float(expected.abs().max().item()),
                        1.0,
                    ),
                )
                for actual, expected in zip(
                    fine_transfer.transfer_matrices,
                    expected_fine_transfer.transfer_matrices,
                    strict=True,
                )
            )
            or any(
                not torch.allclose(
                    actual,
                    expected,
                    rtol=2e-10,
                    atol=2e-11
                    * max(
                        float(expected.abs().max().item()),
                        1.0,
                    ),
                )
                for actual, expected in zip(
                    fine_transfer.affine_offsets,
                    expected_fine_transfer.affine_offsets,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "fine graph does not execute the source decomposition"
            )
        encoder_transfer = self.encoder_graph.boundary_transfer()
        for factor, matrix, offset in zip(
            self.source_decomposition.factors,
            encoder_transfer.transfer_matrices,
            encoder_transfer.affine_offsets,
            strict=True,
        ):
            if (
                not torch.allclose(
                    matrix,
                    factor.restriction,
                    rtol=2e-10,
                    atol=2e-11
                    * max(
                        float(factor.restriction.abs().max().item()),
                        1.0,
                    ),
                )
                or not torch.allclose(
                    offset,
                    -(factor.restriction @ factor.input_mean),
                    rtol=2e-10,
                    atol=2e-11
                    * max(float(offset.abs().max().item()), 1.0),
                )
            ):
                raise ValueError(
                    "encoder graph does not expose source restrictions"
                )
        recursive_transfer = self.recursive_graph.boundary_transfer()
        recursive_inputs_by_name = {
            port.name: port
            for port in recursive_transfer.input_ports
        }
        for output_index, (input_port, output_port) in enumerate(
            zip(self.mode_input_ports, self.mode_ports, strict=True)
        ):
            prefix = recursive_transfer.input_prefixes[output_index]
            expected = torch.zeros(
                (
                    output_port.width,
                    sum(
                        recursive_inputs_by_name[name].width
                        for name in prefix
                    ),
                ),
                dtype=torch.float64,
            )
            start = sum(
                recursive_inputs_by_name[name].width
                for name in prefix[: prefix.index(input_port.name)]
            )
            expected[
                :,
                start : start + input_port.width,
            ] = torch.eye(input_port.width, dtype=torch.float64)
            if (
                not torch.equal(
                    recursive_transfer.transfer_matrices[output_index],
                    expected,
                )
                or bool(
                    torch.count_nonzero(
                        recursive_transfer.affine_offsets[output_index]
                    ).item()
                )
            ):
                raise ValueError(
                    "recursive modal core must preserve its mode coordinates"
                )
        if len(names) != len(set(names)):
            raise ValueError("mode expansion port names must be unique")
        if (
            self.artifact_kind != _MODE_EXPANSION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("hierarchical mode expansion header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_MODE_EXPANSION_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="mode expansion artifact_sha256",
                )
                != computed
            ):
                raise ValueError("hierarchical mode expansion hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "source_generator_sha256": self.source_generator_sha256,
            "source_decomposition_sha256": (
                self.source_decomposition.artifact_sha256
            ),
            "graph_sha256": self.graph.artifact_sha256,
            "encoder_graph_sha256": self.encoder_graph.artifact_sha256,
            "recursive_graph_sha256": self.recursive_graph.artifact_sha256,
            "mode_input_port_sha256s": tuple(
                port.artifact_sha256 for port in self.mode_input_ports
            ),
            "mode_port_sha256s": tuple(
                port.artifact_sha256 for port in self.mode_ports
            ),
            "mode_moment_sha256s": tuple(
                moments.artifact_sha256
                for moments in self.mode_moments
            ),
            "recursive_input_moment_sha256s": tuple(
                moments.artifact_sha256
                for moments in self.recursive_input_moments
            ),
            "recursive_output_moment_sha256s": tuple(
                moments.artifact_sha256
                for moments in self.recursive_output_moments
            ),
        }

    def validate_integrity(self) -> None:
        self.__post_init__()

    def factor_recursive_connectivity(
        self,
        *,
        retained_ranks: (
            int | Sequence[int] | Mapping[str, int] | None
        ) = None,
        relative_eigenvalue_cutoff: float = 1e-12,
        relative_singular_value_cutoff: float = 1e-12,
        assume_block_diagonal_input_covariance: bool = False,
    ) -> ModalConnectivityDecomposition:
        """Factor the identity handoff scaffold for contract-level checks.

        Use :meth:`measured_modal_core` and
        :meth:`MeasuredModalCore.factor_connectivity` for a nontrivial next
        rung.
        """

        self.validate_integrity()
        return factor_modal_connectivity(
            self.recursive_graph.boundary_transfer(),
            self.recursive_input_moments,
            self.recursive_output_moments,
            retained_ranks=retained_ranks,
            relative_eigenvalue_cutoff=relative_eigenvalue_cutoff,
            relative_singular_value_cutoff=(
                relative_singular_value_cutoff
            ),
            assume_block_diagonal_input_covariance=(
                assume_block_diagonal_input_covariance
            ),
        )

    def projected_connection(
        self,
        *,
        upstream_mode_index: int,
        downstream_mode_index: int,
        downstream_input_port_name: str,
        fine_jacobian: Tensor,
        evidence_sha256: str,
    ) -> ProjectedModalConnection:
        """Build one authenticated ``R_j J P_i`` modal interaction edge."""

        self.validate_integrity()
        for value, label in (
            (upstream_mode_index, "upstream_mode_index"),
            (downstream_mode_index, "downstream_mode_index"),
        ):
            _require_int(value, label=label, minimum=0)
            if value >= len(self.source_decomposition.factors):
                raise ValueError(f"{label} is outside the modal catalog")
        upstream_component = self.recursive_graph.components[
            upstream_mode_index
        ]
        downstream_component = self.recursive_graph.components[
            downstream_mode_index
        ]
        if upstream_component.causal_end >= downstream_component.causal_start:
            raise ValueError(
                "projected modal interaction must follow causal order"
            )
        upstream_factor = self.source_decomposition.factors[
            upstream_mode_index
        ]
        downstream_factor = self.source_decomposition.factors[
            downstream_mode_index
        ]
        matrix = project_modal_jacobian(
            upstream_factor=upstream_factor,
            downstream_factor=downstream_factor,
            downstream_input_port_name=downstream_input_port_name,
            fine_jacobian=fine_jacobian,
        )
        return ProjectedModalConnection(
            source_expansion_sha256=self.artifact_sha256,
            upstream_factor=upstream_factor,
            downstream_factor=downstream_factor,
            downstream_input_port_name=downstream_input_port_name,
            fine_jacobian=fine_jacobian,
            direct_jacobian_evidence_sha256=evidence_sha256,
            source_component=upstream_component.component_id,
            source_port=next(iter(upstream_component.output_ports)),
            target_component=downstream_component.component_id,
            target_port=next(iter(downstream_component.input_ports)),
            matrix=matrix,
        )

    def modal_core_graph(
        self,
        *,
        connections: Sequence[
            DirectModalConnection | ProjectedModalConnection
        ],
    ) -> LinearModalGraphLevel:
        """Build the signed mode-to-mode graph to instrument and measure.

        Generic direct edges represent independently measured modal
        Jacobians.  Projected edges additionally carry their factors and fine
        Jacobian, and are revalidated against this exact expansion before
        being lowered.
        """

        self.validate_integrity()
        factor_digests = tuple(
            factor.artifact_sha256
            for factor in self.source_decomposition.factors
        )
        lowered: list[DirectModalConnection] = []
        for connection in connections:
            if isinstance(connection, ProjectedModalConnection):
                connection.validate_integrity()
                if (
                    connection.source_expansion_sha256
                    != self.artifact_sha256
                    or connection.upstream_factor.artifact_sha256
                    not in factor_digests
                    or connection.downstream_factor.artifact_sha256
                    not in factor_digests
                ):
                    raise ValueError(
                        "projected edge is stale for this modal expansion"
                    )
                upstream_index = factor_digests.index(
                    connection.upstream_factor.artifact_sha256
                )
                downstream_index = factor_digests.index(
                    connection.downstream_factor.artifact_sha256
                )
                upstream = self.recursive_graph.components[upstream_index]
                downstream = self.recursive_graph.components[
                    downstream_index
                ]
                if (
                    connection.source_component
                    != upstream.component_id
                    or connection.source_port
                    != next(iter(upstream.output_ports))
                    or connection.target_component
                    != downstream.component_id
                    or connection.target_port
                    != next(iter(downstream.input_ports))
                ):
                    raise ValueError(
                        "projected edge endpoints do not match its factors"
                    )
                lowered.append(connection.as_direct_connection())
            elif isinstance(connection, DirectModalConnection):
                connection.validate_integrity()
                lowered.append(connection)
            else:
                raise TypeError(
                    "modal core connections must be direct or projected edges"
                )
        connection_tuple = tuple(sorted(lowered, key=_canonical_edge_key))
        return LinearModalGraphLevel(
            graph_id=self.recursive_graph.graph_id,
            level_index=self.recursive_graph.level_index,
            source_artifact_sha256=self.artifact_sha256,
            components=self.recursive_graph.components,
            connections=connection_tuple,
            boundary_inputs=self.recursive_graph.boundary_inputs,
            boundary_outputs=self.recursive_graph.boundary_outputs,
            input_injections=self.recursive_graph.input_injections,
            output_readouts=self.recursive_graph.output_readouts,
            output_offsets=self.recursive_graph.output_offsets,
        )

    def measured_modal_core(
        self,
        *,
        connections: Sequence[
            DirectModalConnection | ProjectedModalConnection
        ],
        input_moments: Sequence[MessageMoments],
        output_moments: Sequence[MessageMoments],
    ) -> MeasuredModalCore:
        """Bind measured signed modal interactions for the next factorization.

        Both boundary sides must have been collected on the exact graph
        returned by :meth:`modal_core_graph`.  Unlike the initial balanced
        encoder coordinates, an interacting core may change downstream input
        Fisher and may produce arbitrary PSD covariance and Fisher matrices.
        """

        graph = self.modal_core_graph(connections=connections)
        input_tuple = tuple(input_moments)
        output_tuple = tuple(output_moments)
        if (
            len(input_tuple) != len(self.mode_input_ports)
            or any(
                not isinstance(moments, MessageMoments)
                for moments in input_tuple
            )
            or len(output_tuple) != len(self.mode_ports)
            or any(
                not isinstance(moments, MessageMoments)
                for moments in output_tuple
            )
        ):
            raise ValueError(
                "measured modal moments must cover both graph boundaries"
            )
        for port, moments in zip(
            graph.boundary_inputs,
            input_tuple,
            strict=True,
        ):
            moments.validate_integrity()
            if (
                moments.port.artifact_sha256 != port.artifact_sha256
                or moments.source_level_sha256 != graph.artifact_sha256
            ):
                raise ValueError(
                    "measured input modal moments are not bound to graph"
                )
        for port, moments in zip(
            graph.boundary_outputs,
            output_tuple,
            strict=True,
        ):
            moments.validate_integrity()
            if (
                moments.port.artifact_sha256 != port.artifact_sha256
                or moments.source_level_sha256 != graph.artifact_sha256
            ):
                raise ValueError(
                    "measured output modal moments are not bound to graph"
                )
        return MeasuredModalCore(
            source_expansion_sha256=self.artifact_sha256,
            graph=graph,
            input_moments=input_tuple,
            output_moments=output_tuple,
        )


@dataclass(frozen=True, slots=True)
class HierarchicalModalGenerator:
    """One recursively reusable parent bound to an exact child graph."""

    parent_id: str
    level_index: int
    child_graph: LinearModalGraphLevel
    decomposition: ModalConnectivityDecomposition
    coarsening_group: CausalCoarseningGroup
    exact_child_fallback_sha256: str
    analysis_only: bool = True
    authorizes_source_removal: bool = False
    authorizes_transitive_leaf_fallback: bool = False
    artifact_sha256: str = ""
    artifact_kind: str = _GENERATOR_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        _require_name(self.parent_id, label="hierarchical parent_id")
        _require_int(self.level_index, label="level_index", minimum=1)
        if not isinstance(self.child_graph, LinearModalGraphLevel):
            raise TypeError("child_graph must be a LinearModalGraphLevel")
        self.child_graph.validate_integrity()
        if self.level_index != self.child_graph.level_index + 1:
            raise ValueError(
                "hierarchical generator level must follow its child graph"
            )
        if not isinstance(
            self.decomposition,
            ModalConnectivityDecomposition,
        ):
            raise TypeError(
                "decomposition must be ModalConnectivityDecomposition"
            )
        self.decomposition.validate_integrity()
        exact_transfer = self.child_graph.boundary_transfer()
        if (
            exact_transfer.artifact_sha256
            != self.decomposition.source_transfer.artifact_sha256
        ):
            raise ValueError(
                "connectivity decomposition is stale for the child graph"
            )
        if not isinstance(self.coarsening_group, CausalCoarseningGroup):
            raise TypeError(
                "coarsening_group must be a CausalCoarseningGroup"
            )
        self._validate_coarsening_binding()
        if (
            self.coarsening_group.parent_id != self.parent_id
            or self.coarsening_group.child_component_ids
            != tuple(
                component.component_id
                for component in self.child_graph.components
            )
        ):
            raise ValueError(
                "coarsening group must bind this parent and complete child "
                "graph"
            )
        _require_sha256(
            self.exact_child_fallback_sha256,
            label="exact_child_fallback_sha256",
        )
        if (
            self.exact_child_fallback_sha256
            != self.child_graph.artifact_sha256
        ):
            raise ValueError(
                "exact child fallback digest must bind the immediate graph "
                "that the reference executor actually runs"
            )
        if (
            self.analysis_only is not True
            or self.authorizes_source_removal is not False
            or self.authorizes_transitive_leaf_fallback is not False
        ):
            raise ValueError(
                "v1 hierarchical generators remain analysis-only with "
                "only an exact immediate-child fallback"
            )
        if (
            self.artifact_kind != _GENERATOR_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("hierarchical generator header is invalid")
        computed = _json_sha256(
            self._hash_payload(),
            domain=_GENERATOR_DOMAIN,
        )
        if self.artifact_sha256:
            if (
                _require_sha256(
                    self.artifact_sha256,
                    label="hierarchical generator artifact_sha256",
                )
                != computed
            ):
                raise ValueError("hierarchical generator hash mismatch")
        else:
            object.__setattr__(self, "artifact_sha256", computed)

    @property
    def causal_start(self) -> int:
        return self.child_graph.components[0].causal_start

    @property
    def causal_end(self) -> int:
        return self.child_graph.components[-1].causal_end

    def _validate_coarsening_binding(self) -> None:
        if (
            self.coarsening_group.parent_graph_sha256
            == self.child_graph.artifact_sha256
        ):
            self.coarsening_group.validate_against(self.child_graph)
        elif (
            self.coarsening_group.parent_graph_sha256
            == self.child_graph.source_artifact_sha256
        ):
            self.coarsening_group.validate_extracted_child(
                self.child_graph
            )
        else:
            raise ValueError(
                "coarsening group is unrelated to the child graph"
            )

    def as_component(self, *, candidate: bool) -> AffineModalComponent:
        """Return an exact dense boundary view for analysis-only composition.

        A reduced candidate may not use this path: multiplying ``P @ R`` would
        hide the connectivity modes and re-densify their execution.  Use
        :meth:`factorized_expansion` or :meth:`as_factorized_graph` instead.
        """

        if candidate:
            raise ValueError(
                "reduced candidates must remain factorized; use "
                "as_factorized_graph()"
            )
        return AffineModalComponent(
            component_id=self.parent_id,
            level_index=self.level_index,
            causal_start=self.causal_start,
            causal_end=self.causal_end,
            transfer=self.decomposition.source_transfer,
            child_ids=tuple(
                component.component_id
                for component in self.child_graph.components
            ),
            source_artifact_sha256s=(
                self.child_graph.artifact_sha256,
                self.decomposition.artifact_sha256,
            ),
        )

    def factorized_expansion(self) -> HierarchicalModeExpansion:
        """Lower every retained factor into explicit modal graph messages."""

        self.validate_integrity()
        if any(
            factor.retained_rank == 0
            for factor in self.decomposition.factors
        ):
            raise ValueError(
                "rank-zero outputs have no explicit mode port; keep them in "
                "the shadow executor rather than a factorized graph"
            )
        graph_id = f"{self.parent_id}.factorized"
        encoder_graph_id = f"{self.parent_id}.modal-encoder"
        recursive_graph_id = f"{self.parent_id}.recursive-modal"
        outputs_per_stage: dict[int, int] = defaultdict(int)
        for factor in self.decomposition.factors:
            outputs_per_stage[factor.output_port.causal_order] += 1
        stride = 2 * (max(outputs_per_stage.values()) + 1)
        boundary_inputs = tuple(
            ModalBoundaryPort(
                name=port.name,
                direction="input",
                causal_order=port.causal_order * stride,
                width=port.width,
                owner_id=graph_id,
            )
            for port in self.decomposition.source_transfer.input_ports
        )
        boundary_input_by_name = {
            port.name: port for port in boundary_inputs
        }
        encoder_boundary_inputs = tuple(
            ModalBoundaryPort(
                name=port.name,
                direction="input",
                causal_order=port.causal_order,
                width=port.width,
                owner_id=encoder_graph_id,
            )
            for port in boundary_inputs
        )
        encoder_boundary_input_by_name = {
            port.name: port for port in encoder_boundary_inputs
        }

        components: list[AffineModalComponent] = []
        restriction_components: list[AffineModalComponent] = []
        connections: list[DirectModalConnection] = []
        injections: list[BoundaryInputInjection] = []
        encoder_injections: list[BoundaryInputInjection] = []
        readouts: list[BoundaryOutputReadout] = []
        encoder_readouts: list[BoundaryOutputReadout] = []
        boundary_outputs: list[ModalBoundaryPort] = []
        encoder_boundary_outputs: list[ModalBoundaryPort] = []
        factor_mode_moments: list[ConnectivityModeMoments] = []
        occurrence_by_stage: dict[int, int] = defaultdict(int)
        used_boundary_inputs: set[str] = set()

        for factor_index, factor in enumerate(
            self.decomposition.factors
        ):
            source_stage = factor.output_port.causal_order
            occurrence = occurrence_by_stage[source_stage]
            occurrence_by_stage[source_stage] += 1
            restriction_stage = source_stage * stride + 2 * occurrence
            prolongation_stage = restriction_stage + 1
            restriction_id = (
                f"{self.parent_id}.restrict.{factor_index}"
            )
            prolongation_id = (
                f"{self.parent_id}.prolong.{factor_index}"
            )
            local_inputs = tuple(
                ModalBoundaryPort(
                    name=f"{restriction_id}.input.{input_index:04d}",
                    direction="input",
                    causal_order=restriction_stage,
                    width=source_port.width,
                    owner_id=restriction_id,
                )
                for input_index, source_port in enumerate(
                    factor.input_ports
                )
            )
            mode_port = ModalBoundaryPort(
                name=f"{restriction_id}.mode",
                direction="output",
                causal_order=restriction_stage,
                width=factor.retained_rank,
                owner_id=restriction_id,
            )
            restriction_transfer = CausalBoundaryTransfer(
                source_level_sha256=factor.artifact_sha256,
                input_ports=local_inputs,
                output_ports=(mode_port,),
                input_prefixes=(
                    tuple(port.name for port in local_inputs),
                ),
                transfer_matrices=(factor.restriction,),
                affine_offsets=(
                    -(factor.restriction @ factor.input_mean),
                ),
            )
            restriction_component = AffineModalComponent(
                component_id=restriction_id,
                level_index=self.level_index,
                causal_start=restriction_stage,
                causal_end=restriction_stage,
                transfer=restriction_transfer,
                child_ids=(f"factor.{factor_index}.restriction",),
                source_artifact_sha256s=(factor.artifact_sha256,),
            )
            components.append(restriction_component)
            restriction_components.append(restriction_component)

            mode_input = ModalBoundaryPort(
                name=f"{prolongation_id}.mode-input",
                direction="input",
                causal_order=prolongation_stage,
                width=factor.retained_rank,
                owner_id=prolongation_id,
            )
            fine_output = ModalBoundaryPort(
                name=f"{prolongation_id}.fine-output",
                direction="output",
                causal_order=prolongation_stage,
                width=factor.output_port.width,
                owner_id=prolongation_id,
            )
            prolongation_transfer = CausalBoundaryTransfer(
                source_level_sha256=factor.artifact_sha256,
                input_ports=(mode_input,),
                output_ports=(fine_output,),
                input_prefixes=((mode_input.name,),),
                transfer_matrices=(factor.prolongation,),
                affine_offsets=(factor.output_mean,),
            )
            components.append(
                AffineModalComponent(
                    component_id=prolongation_id,
                    level_index=self.level_index,
                    causal_start=prolongation_stage,
                    causal_end=prolongation_stage,
                    transfer=prolongation_transfer,
                    child_ids=(f"factor.{factor_index}.prolongation",),
                    source_artifact_sha256s=(factor.artifact_sha256,),
                )
            )
            connections.append(
                DirectModalConnection(
                    source_component=restriction_id,
                    source_port=mode_port.name,
                    target_component=prolongation_id,
                    target_port=mode_input.name,
                    matrix=ImplicitIdentityMap(factor.retained_rank),
                    evidence_kind="direct_jacobian",
                    evidence_sha256=factor.artifact_sha256,
                )
            )

            for local_port, source_port in zip(
                local_inputs,
                factor.input_ports,
                strict=True,
            ):
                boundary_port = boundary_input_by_name[source_port.name]
                if boundary_port.causal_order > local_port.causal_order:
                    raise ValueError(
                        "factorized expansion would introduce a future input"
                    )
                cut_digest = _json_sha256(
                    {
                        "factor_sha256": factor.artifact_sha256,
                        "source_port_sha256": source_port.artifact_sha256,
                        "local_port_sha256": local_port.artifact_sha256,
                    },
                    domain=_MODE_EXPANSION_DOMAIN,
                )
                injections.append(
                    BoundaryInputInjection(
                        boundary_port=boundary_port.name,
                        target_component=restriction_id,
                        target_port=local_port.name,
                        matrix=ImplicitIdentityMap(source_port.width),
                        cut_edge_sha256=cut_digest,
                    )
                )
                encoder_boundary_port = (
                    encoder_boundary_input_by_name[source_port.name]
                )
                encoder_cut_digest = _json_sha256(
                    {
                        "factor_sha256": factor.artifact_sha256,
                        "source_port_sha256": (
                            source_port.artifact_sha256
                        ),
                        "encoder_boundary_port_sha256": (
                            encoder_boundary_port.artifact_sha256
                        ),
                        "local_port_sha256": (
                            local_port.artifact_sha256
                        ),
                    },
                    domain=_MODE_EXPANSION_DOMAIN,
                )
                encoder_injections.append(
                    BoundaryInputInjection(
                        boundary_port=encoder_boundary_port.name,
                        target_component=restriction_id,
                        target_port=local_port.name,
                        matrix=ImplicitIdentityMap(source_port.width),
                        cut_edge_sha256=encoder_cut_digest,
                    )
                )
                used_boundary_inputs.add(boundary_port.name)

            boundary_output = ModalBoundaryPort(
                name=factor.output_port.name,
                direction="output",
                causal_order=prolongation_stage,
                width=factor.output_port.width,
                owner_id=graph_id,
            )
            boundary_outputs.append(boundary_output)
            readout_digest = _json_sha256(
                {
                    "factor_sha256": factor.artifact_sha256,
                    "fine_output_sha256": fine_output.artifact_sha256,
                    "boundary_output_sha256": (
                        boundary_output.artifact_sha256
                    ),
                },
                domain=_MODE_EXPANSION_DOMAIN,
            )
            readouts.append(
                BoundaryOutputReadout(
                    source_component=prolongation_id,
                    source_port=fine_output.name,
                    boundary_port=boundary_output.name,
                    matrix=ImplicitIdentityMap(factor.output_port.width),
                    cut_edge_sha256=readout_digest,
                )
            )
            source_mode_moments = factor.mode_moments
            encoder_boundary_output = ModalBoundaryPort(
                name=f"{self.parent_id}.mode-in.{factor_index:04d}",
                direction="output",
                causal_order=restriction_stage,
                width=factor.retained_rank,
                owner_id=encoder_graph_id,
            )
            encoder_boundary_outputs.append(encoder_boundary_output)
            encoder_readout_digest = _json_sha256(
                {
                    "factor_sha256": factor.artifact_sha256,
                    "mode_port_sha256": mode_port.artifact_sha256,
                    "encoder_boundary_output_sha256": (
                        encoder_boundary_output.artifact_sha256
                    ),
                },
                domain=_MODE_EXPANSION_DOMAIN,
            )
            encoder_readouts.append(
                BoundaryOutputReadout(
                    source_component=restriction_id,
                    source_port=mode_port.name,
                    boundary_port=encoder_boundary_output.name,
                    matrix=ImplicitIdentityMap(factor.retained_rank),
                    cut_edge_sha256=encoder_readout_digest,
                )
            )
            factor_mode_moments.append(source_mode_moments)

        expected_boundary_inputs = {
            port.name for port in boundary_inputs
        }
        if used_boundary_inputs != expected_boundary_inputs:
            raise ValueError(
                "factorized expansion contains a boundary input unused by "
                "every causal output"
            )
        graph = LinearModalGraphLevel(
            graph_id=graph_id,
            level_index=self.level_index,
            source_artifact_sha256=self.artifact_sha256,
            components=tuple(
                sorted(
                    components,
                    key=lambda value: (
                        value.causal_start,
                        value.causal_end,
                        value.component_id,
                    ),
                )
            ),
            connections=tuple(
                sorted(connections, key=_canonical_edge_key)
            ),
            boundary_inputs=tuple(
                sorted(
                    boundary_inputs,
                    key=lambda port: (
                        port.causal_order,
                        port.name,
                    ),
                )
            ),
            boundary_outputs=tuple(
                sorted(
                    boundary_outputs,
                    key=lambda port: (
                        port.causal_order,
                        port.name,
                    ),
                )
            ),
            input_injections=tuple(
                sorted(
                    injections,
                    key=lambda value: (
                        value.boundary_port,
                        value.target_component,
                        value.target_port,
                        value.artifact_sha256,
                    ),
                )
            ),
            output_readouts=tuple(
                sorted(
                    readouts,
                    key=lambda value: (
                        value.source_component,
                        value.source_port,
                        value.boundary_port,
                        value.artifact_sha256,
                    ),
                )
            ),
            output_offsets=tuple(
                torch.zeros(port.width, dtype=torch.float64)
                for port in sorted(
                    boundary_outputs,
                    key=lambda value: (
                        value.causal_order,
                        value.name,
                    ),
                )
            ),
        )
        encoder_graph = LinearModalGraphLevel(
            graph_id=encoder_graph_id,
            level_index=self.level_index,
            source_artifact_sha256=self.artifact_sha256,
            components=tuple(
                sorted(
                    restriction_components,
                    key=lambda value: (
                        value.causal_start,
                        value.causal_end,
                        value.component_id,
                    ),
                )
            ),
            connections=(),
            boundary_inputs=tuple(
                sorted(
                    encoder_boundary_inputs,
                    key=lambda port: (
                        port.causal_order,
                        port.name,
                    ),
                )
            ),
            boundary_outputs=tuple(
                sorted(
                    encoder_boundary_outputs,
                    key=lambda port: (
                        port.causal_order,
                        port.name,
                    ),
                )
            ),
            input_injections=tuple(
                sorted(
                    encoder_injections,
                    key=lambda value: (
                        value.boundary_port,
                        value.target_component,
                        value.target_port,
                        value.artifact_sha256,
                    ),
                )
            ),
            output_readouts=tuple(
                sorted(
                    encoder_readouts,
                    key=lambda value: (
                        value.source_component,
                        value.source_port,
                        value.boundary_port,
                        value.artifact_sha256,
                    ),
                )
            ),
            output_offsets=tuple(
                torch.zeros(port.width, dtype=torch.float64)
                for port in sorted(
                    encoder_boundary_outputs,
                    key=lambda value: (
                        value.causal_order,
                        value.name,
                    ),
                )
            ),
        )
        ordered_encoder_pairs = tuple(
            sorted(
                zip(
                    encoder_boundary_outputs,
                    factor_mode_moments,
                    strict=True,
                ),
                key=lambda pair: (
                    pair[0].causal_order,
                    pair[0].name,
                ),
            )
        )
        modal_input_ports = tuple(
            ModalBoundaryPort(
                name=encoder_port.name,
                direction="input",
                causal_order=encoder_port.causal_order,
                width=encoder_port.width,
                owner_id=recursive_graph_id,
            )
            for encoder_port, _ in ordered_encoder_pairs
        )
        modal_output_ports = tuple(
            ModalBoundaryPort(
                name=f"{self.parent_id}.mode-out.{index:04d}",
                direction="output",
                causal_order=input_port.causal_order,
                width=input_port.width,
                owner_id=recursive_graph_id,
            )
            for index, input_port in enumerate(modal_input_ports)
        )
        modal_components: list[IdentityModalComponent] = []
        modal_injections: list[BoundaryInputInjection] = []
        modal_readouts: list[BoundaryOutputReadout] = []
        for index, (input_port, output_port, (_, source_moments)) in enumerate(
            zip(
                modal_input_ports,
                modal_output_ports,
                ordered_encoder_pairs,
                strict=True,
            )
        ):
            component_id = f"{self.parent_id}.modal-core.{index:04d}"
            component = identity_modal_component(
                component_id=component_id,
                causal_order=input_port.causal_order,
                width=input_port.width,
                source_artifact_sha256=(
                    source_moments.source_factor_sha256
                ),
                level_index=self.level_index,
            )
            modal_components.append(component)
            modal_injections.append(
                BoundaryInputInjection(
                    boundary_port=input_port.name,
                    target_component=component_id,
                    target_port=f"{component_id}.input",
                    matrix=ImplicitIdentityMap(input_port.width),
                    cut_edge_sha256=_json_sha256(
                        {
                            "source_factor_sha256": (
                                source_moments.source_factor_sha256
                            ),
                            "modal_input_port_sha256": (
                                input_port.artifact_sha256
                            ),
                        },
                        domain=_MODE_EXPANSION_DOMAIN,
                    ),
                )
            )
            modal_readouts.append(
                BoundaryOutputReadout(
                    source_component=component_id,
                    source_port=f"{component_id}.output",
                    boundary_port=output_port.name,
                    matrix=ImplicitIdentityMap(output_port.width),
                    cut_edge_sha256=_json_sha256(
                        {
                            "source_factor_sha256": (
                                source_moments.source_factor_sha256
                            ),
                            "modal_output_port_sha256": (
                                output_port.artifact_sha256
                            ),
                        },
                        domain=_MODE_EXPANSION_DOMAIN,
                    ),
                )
            )
        recursive_graph = LinearModalGraphLevel(
            graph_id=recursive_graph_id,
            level_index=self.level_index,
            source_artifact_sha256=self.artifact_sha256,
            components=tuple(
                sorted(
                    modal_components,
                    key=lambda value: (
                        value.causal_start,
                        value.causal_end,
                        value.component_id,
                    ),
                )
            ),
            connections=(),
            boundary_inputs=modal_input_ports,
            boundary_outputs=modal_output_ports,
            input_injections=tuple(
                sorted(
                    modal_injections,
                    key=lambda value: (
                        value.boundary_port,
                        value.target_component,
                        value.target_port,
                        value.artifact_sha256,
                    ),
                )
            ),
            output_readouts=tuple(
                sorted(
                    modal_readouts,
                    key=lambda value: (
                        value.source_component,
                        value.source_port,
                        value.boundary_port,
                        value.artifact_sha256,
                    ),
                )
            ),
            output_offsets=tuple(
                torch.zeros(port.width, dtype=torch.float64)
                for port in modal_output_ports
            ),
        )
        ordered_mode_moments = tuple(
            ConnectivityModeMoments(
                mode_id=port.name,
                causal_order=port.causal_order,
                source_factor_sha256=source_moments.source_factor_sha256,
                reduction_id=source_moments.reduction_id,
                sample_count=source_moments.sample_count,
                covariance=source_moments.covariance,
                fisher=source_moments.fisher,
                salience=source_moments.salience,
            )
            for port, (_, source_moments) in zip(
                recursive_graph.boundary_outputs,
                ordered_encoder_pairs,
                strict=True,
            )
        )
        recursive_input_moments = tuple(
            moments.bind_port(
                port,
                source_level_sha256=recursive_graph.artifact_sha256,
            )
            for port, moments in zip(
                recursive_graph.boundary_inputs,
                ordered_mode_moments,
                strict=True,
            )
        )
        recursive_output_moments = tuple(
            moments.bind_port(
                port,
                source_level_sha256=recursive_graph.artifact_sha256,
            )
            for port, moments in zip(
                recursive_graph.boundary_outputs,
                ordered_mode_moments,
                strict=True,
            )
        )
        return HierarchicalModeExpansion(
            source_generator_sha256=self.artifact_sha256,
            source_decomposition=self.decomposition,
            graph=graph,
            encoder_graph=encoder_graph,
            recursive_graph=recursive_graph,
            mode_input_ports=recursive_graph.boundary_inputs,
            mode_ports=recursive_graph.boundary_outputs,
            mode_moments=ordered_mode_moments,
            recursive_input_moments=recursive_input_moments,
            recursive_output_moments=recursive_output_moments,
        )

    def as_factorized_graph(self) -> LinearModalGraphLevel:
        """Return the executable candidate graph with explicit mode edges."""

        return self.factorized_expansion().graph

    def as_recursive_modal_graph(self) -> LinearModalGraphLevel:
        """Return the mode-input to mode-output recursive core graph."""

        return self.factorized_expansion().recursive_graph

    def as_modal_encoder_graph(self) -> LinearModalGraphLevel:
        """Return the fine-boundary to mode-input encoder graph."""

        return self.factorized_expansion().encoder_graph

    def prolong_modal_outputs(
        self,
        modal_outputs: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        """Decode this rung's named mode coordinates to its fine interface."""

        self.validate_integrity()
        expansion = self.factorized_expansion()
        if set(modal_outputs) != {
            port.name for port in expansion.mode_ports
        }:
            raise ValueError("modal outputs do not match this generator")
        result: dict[str, Tensor] = {}
        for port, factor in zip(
            expansion.mode_ports,
            self.decomposition.factors,
            strict=True,
        ):
            value = modal_outputs[port.name]
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.shape[-1:] != (port.width,)
                or not torch.isfinite(value).all()
            ):
                raise ValueError("modal output tensor is invalid")
            result[factor.output_port.name] = (
                value
                @ factor.prolongation.to(
                    device=value.device,
                    dtype=value.dtype,
                ).T
                + factor.output_mean.to(
                    device=value.device,
                    dtype=value.dtype,
                )
            )
        return result

    def _hash_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
            "parent_id": self.parent_id,
            "level_index": self.level_index,
            "child_graph_sha256": self.child_graph.artifact_sha256,
            "decomposition_sha256": self.decomposition.artifact_sha256,
            "coarsening_group_sha256": (
                self.coarsening_group.artifact_sha256
            ),
            "exact_child_fallback_sha256": (
                self.exact_child_fallback_sha256
            ),
            "analysis_only": self.analysis_only,
            "authorizes_source_removal": self.authorizes_source_removal,
            "authorizes_transitive_leaf_fallback": (
                self.authorizes_transitive_leaf_fallback
            ),
        }

    def validate_integrity(self) -> None:
        self.child_graph.validate_integrity()
        self.decomposition.validate_integrity()
        self._validate_coarsening_binding()
        if (
            self.coarsening_group.parent_id != self.parent_id
            or self.coarsening_group.child_component_ids
            != tuple(
                component.component_id
                for component in self.child_graph.components
            )
        ):
            raise ValueError("coarsening group is stale for generator")
        if (
            self.exact_child_fallback_sha256
            != self.child_graph.artifact_sha256
        ):
            raise ValueError(
                "exact child fallback digest does not bind the child graph"
            )
        if (
            self.child_graph.boundary_transfer().artifact_sha256
            != self.decomposition.source_transfer.artifact_sha256
        ):
            raise ValueError(
                "connectivity decomposition is stale for the child graph"
            )
        if (
            _json_sha256(
                self._hash_payload(),
                domain=_GENERATOR_DOMAIN,
            )
            != self.artifact_sha256
        ):
            raise ValueError("hierarchical generator hash mismatch")
