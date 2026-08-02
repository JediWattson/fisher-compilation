"""Fit-only grouped bases inside a frozen graph-wavelet parent subspace.

This module tests two deliberately constrained alternatives to unconstrained
global SVD:

``wavelet_local_svd``
    Partition the frozen parent directions using graph topology, then permit
    response-derived SVD rotations only inside each partition block.

``wavelet_cluster_gfa``
    Use the same topology partition, block-diagonalize the projected graph
    Laplacian, and rank the resulting local graph-Fourier directions by
    fit-response energy.

Both methods return a full orthonormal basis ordered by fit-only energy.  A
caller can therefore compare equal-rank prefixes without changing the runtime
executor or its coefficient accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Literal

import torch
from torch import Tensor


GroupedBasisKind = Literal[
    "wavelet_local_svd",
    "wavelet_cluster_gfa",
    "global_svd_control",
]

__all__ = [
    "FitOnlyGraphWaveletGroupedBasis",
    "GraphWaveletTopologyMerge",
    "GraphWaveletTopologyPartition",
    "GroupedBasisKind",
    "fit_graph_wavelet_grouped_basis",
    "fit_graph_wavelet_topology_partition",
    "grouped_basis_one_hot_control",
    "grouped_basis_projector_overlap",
]


_PARTITION_KIND = "fisher_graph.graph_wavelet_topology_partition"
_BASIS_KIND = "fisher_graph.fit_only_graph_wavelet_grouped_basis"
_FORMAT_VERSION = 1
_PARTITION_DOMAIN = b"fisher-graph:graph-wavelet-topology-partition:v1\0"
_BASIS_DOMAIN = b"fisher-graph:graph-wavelet-grouped-basis:v1\0"
_TENSOR_DOMAIN = b"fisher-graph:graph-wavelet-grouped-basis-tensor:v1\0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARTITION_ALGORITHM = (
    "deterministic_balanced_dyadic_average_linkage_with_top_k_preferred_"
    "seed_pairs_on_abs_parent_basis_transpose_abs_offdiag_laplacian_"
    "abs_parent_basis"
)
_LOCAL_SVD_SEMANTICS = (
    "fit_response_svd_rotations_restricted_to_frozen_topology_blocks"
)
_CLUSTER_GFA_SEMANTICS = (
    "fit_energy_ordered_eigenvectors_of_topology_block_restricted_"
    "parent_projected_laplacian"
)
_GLOBAL_SVD_CONTROL_SEMANTICS = (
    "explicit_unrestricted_fit_response_svd_control_not_a_local_method"
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: object, *, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(_TENSOR_DOMAIN)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json_bytes(tuple(tensor.shape)))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _float_tensor(value: Tensor, *, label: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{label} must be a Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}")
    if (
        value.dtype != torch.float64
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be finite CPU float64")
    return value.detach().clone().contiguous()


def _canonicalize_columns(value: Tensor) -> Tensor:
    result = value.clone()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(torch.argmax(vector.abs()).item())
        if float(vector[pivot]) < 0.0:
            result[:, column] *= -1.0
    return result.contiguous()


def _orthonormal_error(value: Tensor) -> float:
    identity = torch.eye(value.shape[1], dtype=torch.float64)
    return float(torch.linalg.matrix_norm(value.T @ value - identity))


def _canonical_subspace_basis(projector: Tensor, rank: int) -> Tensor:
    node_count = int(projector.shape[0])
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(1, node_count)
    columns: list[Tensor] = []
    for coordinate in range(node_count):
        candidate = projector[:, coordinate].clone()
        for column in columns:
            candidate -= torch.dot(column, candidate) * column
        norm = float(torch.linalg.vector_norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(torch.argmax(candidate.abs()))
        if float(candidate[pivot]) < 0.0:
            candidate *= -1.0
        columns.append(candidate)
        if len(columns) == rank:
            break
    if len(columns) != rank:
        raise RuntimeError("could not canonicalize a grouped eigenspace")
    return torch.stack(columns, dim=1).contiguous()


def _deterministic_psd_eigh(value: Tensor) -> tuple[Tensor, Tensor]:
    symmetric = ((value + value.T) * 0.5).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    scale = max(1.0, float(torch.linalg.matrix_norm(symmetric, ord=2)))
    tolerance = 1.0e-11 * scale
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("grouped operator must be positive semidefinite")
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    canonical_values = eigenvalues.clone()
    canonical_vectors = torch.empty_like(eigenvectors)
    start = 0
    while start < eigenvalues.numel():
        end = start + 1
        while (
            end < eigenvalues.numel()
            and abs(float(eigenvalues[end] - eigenvalues[start]))
            <= tolerance
        ):
            end += 1
        width = end - start
        source = eigenvectors[:, start:end]
        if width == 1:
            canonical_vectors[:, start:end] = _canonicalize_columns(source)
        else:
            canonical_vectors[:, start:end] = _canonical_subspace_basis(
                source @ source.T,
                width,
            )
            canonical_values[start:end] = eigenvalues[start:end].mean()
        start = end
    return canonical_values.contiguous(), canonical_vectors.contiguous()


@dataclass(frozen=True, slots=True)
class GraphWaveletTopologyMerge:
    """One deterministic average-linkage merge in parent-coordinate space."""

    left_members: tuple[int, ...]
    right_members: tuple[int, ...]
    merged_members: tuple[int, ...]
    average_coupling: float

    def __post_init__(self) -> None:
        if (
            not self.left_members
            or not self.right_members
            or set(self.left_members).intersection(self.right_members)
            or self.merged_members
            != tuple(sorted((*self.left_members, *self.right_members)))
            or not math.isfinite(self.average_coupling)
            or self.average_coupling < 0.0
        ):
            raise ValueError("topology merge is invalid")

    def metadata(self) -> dict[str, object]:
        return {
            "left_members": self.left_members,
            "right_members": self.right_members,
            "merged_members": self.merged_members,
            "average_coupling": self.average_coupling,
        }


def _agglomerate_balanced(
    coupling: Tensor,
    *,
    group_count: int,
    topology_top_k: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[GraphWaveletTopologyMerge, ...],
]:
    parent_rank = int(coupling.shape[0])
    groups: list[tuple[int, ...]] = [
        (index,) for index in range(parent_rank)
    ]
    history: list[GraphWaveletTopologyMerge] = []
    scale = max(1.0, float(coupling.max()))
    positive_tolerance = 1.0e-13 * scale
    neighbors = []
    for index in range(parent_rank):
        ordered = sorted(
            (other for other in range(parent_rank) if other != index),
            key=lambda other: (-float(coupling[index, other]), other),
        )
        neighbors.append(frozenset(ordered[:topology_top_k]))
    while len(groups) > group_count:
        smallest_size = min(len(group) for group in groups)
        choices: list[
            tuple[
                int,
                float,
                tuple[int, ...],
                tuple[int, ...],
                int,
                int,
            ]
        ] = []
        for left_index, left in enumerate(groups):
            if len(left) != smallest_size:
                continue
            left_tensor = torch.tensor(left, dtype=torch.int64)
            for right_index in range(left_index + 1, len(groups)):
                right = groups[right_index]
                if len(right) != smallest_size:
                    continue
                right_tensor = torch.tensor(right, dtype=torch.int64)
                block = coupling.index_select(
                    0,
                    left_tensor,
                ).index_select(1, right_tensor)
                score = float(block.mean())
                top_k_penalty = (
                    int(
                        right[0] not in neighbors[left[0]]
                        and left[0] not in neighbors[right[0]]
                    )
                    if smallest_size == 1
                    else 0
                )
                choices.append(
                    (
                        top_k_penalty,
                        -score,
                        left,
                        right,
                        left_index,
                        right_index,
                    )
                )
        if not choices:
            raise RuntimeError("partition agglomeration exhausted choices")
        choice = min(choices)
        score = -choice[1]
        if score <= positive_tolerance:
            raise ValueError(
                "topology has no positive coupling for a requested merge"
            )
        _, _, left, right, left_index, right_index = choice
        merged = tuple(sorted((*left, *right)))
        history.append(
            GraphWaveletTopologyMerge(
                left_members=left,
                right_members=right,
                merged_members=merged,
                average_coupling=score,
            )
        )
        groups = [
            group
            for index, group in enumerate(groups)
            if index not in (left_index, right_index)
        ]
        groups.append(merged)
        groups.sort(key=lambda group: group[0])
    return tuple(groups), tuple(history)


def _topology_matrices(
    parent_basis: Tensor,
    signed_laplacian: Tensor,
    *,
    topology_top_k: int,
) -> tuple[Tensor, Tensor]:
    projected = (parent_basis.T @ signed_laplacian @ parent_basis)
    projected = ((projected + projected.T) * 0.5).contiguous()
    off_diagonal = signed_laplacian.clone()
    off_diagonal.fill_diagonal_(0.0)
    absolute_basis = parent_basis.abs()
    raw_coupling = (
        absolute_basis.T @ off_diagonal.abs() @ absolute_basis
    ).contiguous()
    raw_coupling.fill_diagonal_(0.0)
    raw_coupling = ((raw_coupling + raw_coupling.T) * 0.5).contiguous()
    parent_rank = int(parent_basis.shape[1])
    return projected, raw_coupling


@dataclass(frozen=True, slots=True)
class GraphWaveletTopologyPartition:
    """A topology-only partition of frozen graph-wavelet parent directions."""

    groups: tuple[tuple[int, ...], ...]
    merge_history: tuple[GraphWaveletTopologyMerge, ...]
    topology_coupling: Tensor
    projected_laplacian: Tensor
    topology_top_k: int
    parent_basis_sha256: str
    signed_laplacian_sha256: str
    artifact_sha256: str
    algorithm: str = _PARTITION_ALGORITHM
    artifact_kind: str = _PARTITION_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        coupling = _float_tensor(
            self.topology_coupling,
            label="topology_coupling",
            ndim=2,
        )
        projected = _float_tensor(
            self.projected_laplacian,
            label="projected_laplacian",
            ndim=2,
        )
        object.__setattr__(self, "topology_coupling", coupling)
        object.__setattr__(self, "projected_laplacian", projected)
        self.validate_integrity()

    def validate_integrity(self) -> None:
        coupling = self.topology_coupling
        projected = self.projected_laplacian
        width = coupling.shape[0]
        flat = tuple(member for group in self.groups for member in group)
        if (
            coupling.shape != (width, width)
            or projected.shape != (width, width)
            or tuple(sorted(flat)) != tuple(range(width))
            or len(flat) != len(set(flat))
            or any(group != tuple(sorted(group)) for group in self.groups)
            or self.groups
            != tuple(sorted(self.groups, key=lambda group: group[0]))
            or len(self.merge_history) != width - len(self.groups)
            or isinstance(self.topology_top_k, bool)
            or not isinstance(self.topology_top_k, int)
            or not 1 <= self.topology_top_k < width
            or self.algorithm != _PARTITION_ALGORITHM
            or self.artifact_kind != _PARTITION_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("topology partition fields are invalid")
        tolerance = 2.0e-11
        if (
            float(torch.linalg.matrix_norm(coupling - coupling.T))
            > tolerance
            or float(coupling.abs().diag().max()) > tolerance
            or bool((coupling < 0.0).any())
            or float(torch.linalg.matrix_norm(projected - projected.T))
            > tolerance
        ):
            raise ValueError("topology partition matrices are invalid")
        replay_groups, replay_history = _agglomerate_balanced(
            coupling,
            group_count=len(self.groups),
            topology_top_k=self.topology_top_k,
        )
        if (
            replay_groups != self.groups
            or replay_history != self.merge_history
        ):
            raise ValueError("topology partition replay differs")
        payload = self._payload()
        if self.artifact_sha256 != _json_sha256(
            payload,
            domain=_PARTITION_DOMAIN,
        ):
            raise ValueError("topology partition artifact hash differs")

    def validate_against(
        self,
        parent_basis: Tensor,
        signed_laplacian: Tensor,
    ) -> None:
        self.validate_integrity()
        basis = _float_tensor(parent_basis, label="parent_basis", ndim=2)
        laplacian = _float_tensor(
            signed_laplacian,
            label="signed_laplacian",
            ndim=2,
        )
        projected, coupling = _topology_matrices(
            basis,
            laplacian,
            topology_top_k=self.topology_top_k,
        )
        if (
            _tensor_sha256(basis) != self.parent_basis_sha256
            or _tensor_sha256(laplacian)
            != self.signed_laplacian_sha256
            or not torch.equal(projected, self.projected_laplacian)
            or not torch.equal(coupling, self.topology_coupling)
        ):
            raise ValueError("topology partition input replay differs")

    @property
    def parent_rank(self) -> int:
        return int(self.topology_coupling.shape[0])

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(len(group) for group in self.groups)

    def _payload(self) -> dict[str, object]:
        return {
            "groups": self.groups,
            "merge_history": tuple(
                merge.metadata() for merge in self.merge_history
            ),
            "topology_coupling_sha256": _tensor_sha256(
                self.topology_coupling
            ),
            "projected_laplacian_sha256": _tensor_sha256(
                self.projected_laplacian
            ),
            "topology_top_k": self.topology_top_k,
            "parent_basis_sha256": self.parent_basis_sha256,
            "signed_laplacian_sha256": self.signed_laplacian_sha256,
            "algorithm": self.algorithm,
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "parent_rank": self.parent_rank,
            "group_count": self.group_count,
            "group_sizes": self.group_sizes,
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class FitOnlyGraphWaveletGroupedBasis:
    """A full, fit-energy-ordered orthonormal grouped parent basis."""

    method: GroupedBasisKind
    basis: Tensor
    component_scores: Tensor
    component_group_ordinals: tuple[int, ...]
    component_local_ordinals: tuple[int, ...]
    component_frequencies: tuple[float | None, ...]
    partition_artifact_sha256: str
    fit_weighted_response_sha256: str
    fit_weighted_response_shape: tuple[int, int, int, int]
    fit_origins: tuple[int, ...]
    response_binding_sha256: str
    parent_subspace_artifact_sha256: str
    parent_basis_sha256: str
    artifact_sha256: str
    fit_scope: str = "declared_fit_origins_only"
    heldout_input_used: bool = False
    artifact_kind: str = _BASIS_KIND
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        basis = _float_tensor(self.basis, label="basis", ndim=2)
        scores = _float_tensor(
            self.component_scores,
            label="component_scores",
            ndim=1,
        )
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "component_scores", scores)
        self.validate_integrity()

    def validate_integrity(self) -> None:
        basis = self.basis
        scores = self.component_scores
        width = basis.shape[1]
        if (
            self.method
            not in (
                "wavelet_local_svd",
                "wavelet_cluster_gfa",
                "global_svd_control",
            )
            or basis.shape[0] < width
            or scores.shape != (width,)
            or len(self.component_group_ordinals) != width
            or len(self.component_local_ordinals) != width
            or len(self.component_frequencies) != width
            or len(self.fit_weighted_response_shape) != 4
            or self.fit_weighted_response_shape[0] != basis.shape[0]
            or self.fit_weighted_response_shape[1] != len(self.fit_origins)
            or any(width <= 0 for width in self.fit_weighted_response_shape)
            or not self.fit_origins
            or tuple(sorted(set(self.fit_origins))) != self.fit_origins
            or any(
                isinstance(origin, bool) or not isinstance(origin, int)
                for origin in self.fit_origins
            )
            or _SHA256.fullmatch(self.response_binding_sha256) is None
            or _SHA256.fullmatch(
                self.parent_subspace_artifact_sha256
            )
            is None
            or bool((scores < -1.0e-12).any())
            or any(
                frequency is not None and not math.isfinite(frequency)
                for frequency in self.component_frequencies
            )
            or _orthonormal_error(basis) > 2.0e-10
            or self.fit_scope != "declared_fit_origins_only"
            or self.heldout_input_used is not False
            or self.artifact_kind != _BASIS_KIND
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError("grouped basis fields are invalid")
        if any(
            float(scores[index]) + 1.0e-12 < float(scores[index + 1])
            for index in range(width - 1)
        ):
            raise ValueError("grouped basis components are not score ordered")
        if self.artifact_sha256 != _json_sha256(
            self._payload(),
            domain=_BASIS_DOMAIN,
        ):
            raise ValueError("grouped basis artifact hash differs")

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def semantics(self) -> str:
        if self.method == "wavelet_local_svd":
            return _LOCAL_SVD_SEMANTICS
        if self.method == "wavelet_cluster_gfa":
            return _CLUSTER_GFA_SEMANTICS
        return _GLOBAL_SVD_CONTROL_SEMANTICS

    def prefix(self, rank: int) -> Tensor:
        self.validate_integrity()
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if not 1 <= rank <= self.rank:
            raise ValueError("rank must lie within the grouped basis")
        return self.basis[:, :rank].clone()

    def _payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "basis_sha256": _tensor_sha256(self.basis),
            "component_scores_sha256": _tensor_sha256(
                self.component_scores
            ),
            "component_group_ordinals": self.component_group_ordinals,
            "component_local_ordinals": self.component_local_ordinals,
            "component_frequencies": self.component_frequencies,
            "partition_artifact_sha256": self.partition_artifact_sha256,
            "fit_weighted_response_sha256": (
                self.fit_weighted_response_sha256
            ),
            "fit_weighted_response_shape": (
                self.fit_weighted_response_shape
            ),
            "fit_origins": self.fit_origins,
            "response_binding_sha256": self.response_binding_sha256,
            "parent_subspace_artifact_sha256": (
                self.parent_subspace_artifact_sha256
            ),
            "parent_basis_sha256": self.parent_basis_sha256,
            "fit_scope": self.fit_scope,
            "heldout_input_used": self.heldout_input_used,
            "semantics": self.semantics,
            "artifact_kind": self.artifact_kind,
            "format_version": self.format_version,
        }

    def metadata(self) -> dict[str, object]:
        self.validate_integrity()
        return {
            **self._payload(),
            "rank": self.rank,
            "artifact_sha256": self.artifact_sha256,
        }

    def validate_against(
        self,
        parent_basis: Tensor,
        partition: GraphWaveletTopologyPartition,
        fit_weighted_response: Tensor,
    ) -> None:
        self.validate_integrity()
        partition.validate_integrity()
        parent = _float_tensor(parent_basis, label="parent_basis", ndim=2)
        response = _float_tensor(
            fit_weighted_response,
            label="fit_weighted_response",
            ndim=4,
        )
        if (
            _tensor_sha256(parent) != self.parent_basis_sha256
            or _tensor_sha256(parent) != partition.parent_basis_sha256
            or _tensor_sha256(response)
            != self.fit_weighted_response_sha256
            or tuple(response.shape) != self.fit_weighted_response_shape
            or partition.artifact_sha256
            != self.partition_artifact_sha256
            or response.shape[0] != parent.shape[0]
            or self.rank != parent.shape[1]
            or (
                self.method == "global_svd_control"
                and partition.group_count != 1
            )
            or (
                self.method != "global_svd_control"
                and partition.group_count < 2
            )
        ):
            raise ValueError("grouped basis provenance or locality differs")
        if self.method in ("wavelet_local_svd", "global_svd_control"):
            if any(
                frequency is not None
                for frequency in self.component_frequencies
            ):
                raise ValueError("SVD grouped basis has graph frequencies")
        elif any(
            frequency is None or frequency < -1.0e-11
            for frequency in self.component_frequencies
        ):
            raise ValueError("cluster GFA frequencies are invalid")
        coordinates = parent.T @ self.basis
        for group_ordinal, group in enumerate(partition.groups):
            members = [
                index
                for index, ordinal in enumerate(
                    self.component_group_ordinals
                )
                if ordinal == group_ordinal
            ]
            local = tuple(
                self.component_local_ordinals[index] for index in members
            )
            if tuple(sorted(local)) != tuple(range(len(group))):
                raise ValueError("grouped basis local ordinals differ")
            outside = tuple(
                index
                for index in range(partition.parent_rank)
                if index not in group
            )
            if outside and members:
                outside_tensor = torch.tensor(outside, dtype=torch.int64)
                member_tensor = torch.tensor(members, dtype=torch.int64)
                leakage = coordinates.index_select(
                    0,
                    outside_tensor,
                ).index_select(1, member_tensor)
                if float(leakage.abs().max()) > 2.0e-10:
                    raise ValueError("grouped basis crosses topology blocks")
        if any(
            not 0 <= ordinal < partition.group_count
            for ordinal in self.component_group_ordinals
        ):
            raise ValueError("grouped basis group ordinal is invalid")
        measured_scores = (
            self.basis.T
            @ response.reshape(response.shape[0], -1)
        ).square().sum(dim=1)
        if not torch.allclose(
            measured_scores,
            self.component_scores,
            atol=2.0e-9,
            rtol=2.0e-9,
        ):
            raise ValueError("grouped basis component scores differ")


def _partition_payload(
    *,
    groups: tuple[tuple[int, ...], ...],
    merge_history: tuple[GraphWaveletTopologyMerge, ...],
    coupling: Tensor,
    projected: Tensor,
    topology_top_k: int,
    parent_basis: Tensor,
    signed_laplacian: Tensor,
) -> dict[str, object]:
    return {
        "groups": groups,
        "merge_history": tuple(
            merge.metadata() for merge in merge_history
        ),
        "topology_coupling_sha256": _tensor_sha256(coupling),
        "projected_laplacian_sha256": _tensor_sha256(projected),
        "topology_top_k": topology_top_k,
        "parent_basis_sha256": _tensor_sha256(parent_basis),
        "signed_laplacian_sha256": _tensor_sha256(signed_laplacian),
        "algorithm": _PARTITION_ALGORITHM,
        "artifact_kind": _PARTITION_KIND,
        "format_version": _FORMAT_VERSION,
    }


def fit_graph_wavelet_topology_partition(
    parent_basis: Tensor,
    signed_laplacian: Tensor,
    *,
    group_count: int,
    topology_top_k: int = 8,
) -> GraphWaveletTopologyPartition:
    """Partition parent directions without reading response values."""

    basis = _float_tensor(parent_basis, label="parent_basis", ndim=2)
    laplacian = _float_tensor(
        signed_laplacian,
        label="signed_laplacian",
        ndim=2,
    )
    source_modes, parent_rank = basis.shape
    if (
        laplacian.shape != (source_modes, source_modes)
        or isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or not 1 <= group_count <= parent_rank
        or isinstance(topology_top_k, bool)
        or not isinstance(topology_top_k, int)
        or not 1 <= topology_top_k < parent_rank
        or parent_rank % group_count != 0
        or (parent_rank // group_count) & (
            parent_rank // group_count - 1
        )
        or _orthonormal_error(basis) > 2.0e-10
        or float(torch.linalg.matrix_norm(laplacian - laplacian.T))
        > 2.0e-10
    ):
        raise ValueError("partition inputs are invalid")
    _deterministic_psd_eigh(laplacian)
    projected, coupling = _topology_matrices(
        basis,
        laplacian,
        topology_top_k=topology_top_k,
    )
    _deterministic_psd_eigh(projected)
    frozen_groups, frozen_history = _agglomerate_balanced(
        coupling,
        group_count=group_count,
        topology_top_k=topology_top_k,
    )
    payload = _partition_payload(
        groups=frozen_groups,
        merge_history=frozen_history,
        coupling=coupling,
        projected=projected,
        topology_top_k=topology_top_k,
        parent_basis=basis,
        signed_laplacian=laplacian,
    )
    result = GraphWaveletTopologyPartition(
        groups=frozen_groups,
        merge_history=frozen_history,
        topology_coupling=coupling,
        projected_laplacian=projected,
        topology_top_k=topology_top_k,
        parent_basis_sha256=_tensor_sha256(basis),
        signed_laplacian_sha256=_tensor_sha256(laplacian),
        artifact_sha256=_json_sha256(payload, domain=_PARTITION_DOMAIN),
    )
    result.validate_against(basis, laplacian)
    return result


def _grouped_basis_payload(
    *,
    method: GroupedBasisKind,
    basis: Tensor,
    scores: Tensor,
    group_ordinals: tuple[int, ...],
    local_ordinals: tuple[int, ...],
    frequencies: tuple[float | None, ...],
    partition: GraphWaveletTopologyPartition,
    weighted_response: Tensor,
    parent_basis: Tensor,
    fit_origins: tuple[int, ...],
    response_binding_sha256: str,
    parent_subspace_artifact_sha256: str,
) -> dict[str, object]:
    if method == "wavelet_local_svd":
        semantics = _LOCAL_SVD_SEMANTICS
    elif method == "wavelet_cluster_gfa":
        semantics = _CLUSTER_GFA_SEMANTICS
    else:
        semantics = _GLOBAL_SVD_CONTROL_SEMANTICS
    return {
        "method": method,
        "basis_sha256": _tensor_sha256(basis),
        "component_scores_sha256": _tensor_sha256(scores),
        "component_group_ordinals": group_ordinals,
        "component_local_ordinals": local_ordinals,
        "component_frequencies": frequencies,
        "partition_artifact_sha256": partition.artifact_sha256,
        "fit_weighted_response_sha256": _tensor_sha256(weighted_response),
        "fit_weighted_response_shape": tuple(weighted_response.shape),
        "fit_origins": fit_origins,
        "response_binding_sha256": response_binding_sha256,
        "parent_subspace_artifact_sha256": (
            parent_subspace_artifact_sha256
        ),
        "parent_basis_sha256": _tensor_sha256(parent_basis),
        "fit_scope": "declared_fit_origins_only",
        "heldout_input_used": False,
        "semantics": semantics,
        "artifact_kind": _BASIS_KIND,
        "format_version": _FORMAT_VERSION,
    }


def fit_graph_wavelet_grouped_basis(
    parent_basis: Tensor,
    fit_weighted_response: Tensor,
    partition: GraphWaveletTopologyPartition,
    *,
    method: GroupedBasisKind,
    fit_origins: Sequence[int],
    response_binding_sha256: str,
    parent_subspace_artifact_sha256: str,
) -> FitOnlyGraphWaveletGroupedBasis:
    """Fit one full grouped basis using response values from fit origins only."""

    basis = _float_tensor(parent_basis, label="parent_basis", ndim=2)
    response = _float_tensor(
        fit_weighted_response,
        label="fit_weighted_response",
        ndim=4,
    )
    origins = tuple(fit_origins)
    if method not in (
        "wavelet_local_svd",
        "wavelet_cluster_gfa",
        "global_svd_control",
    ):
        raise ValueError("grouped basis method is invalid")
    partition.validate_integrity()
    if (
        response.shape[0] != basis.shape[0]
        or basis.shape[1] != partition.parent_rank
        or _tensor_sha256(basis) != partition.parent_basis_sha256
        or len(origins) != response.shape[1]
        or tuple(sorted(set(origins))) != origins
        or any(
            isinstance(origin, bool) or not isinstance(origin, int)
            for origin in origins
        )
        or _SHA256.fullmatch(response_binding_sha256) is None
        or _SHA256.fullmatch(parent_subspace_artifact_sha256) is None
        or (
            method == "global_svd_control"
            and partition.group_count != 1
        )
        or (
            method != "global_svd_control"
            and partition.group_count < 2
        )
    ):
        raise ValueError("grouped basis geometry or parent binding differs")

    parent_coordinates = (
        basis.T @ response.reshape(response.shape[0], -1)
    ).contiguous()
    columns: list[Tensor] = []
    records: list[tuple[float, int, int, float | None]] = []
    for group_ordinal, group in enumerate(partition.groups):
        indices = torch.tensor(group, dtype=torch.int64)
        group_parent = basis.index_select(1, indices)
        group_response = parent_coordinates.index_select(0, indices)
        if method in ("wavelet_local_svd", "global_svd_control"):
            eigenvalues, local_vectors = _deterministic_psd_eigh(
                group_response @ group_response.T
            )
            reverse = torch.arange(
                len(group) - 1,
                -1,
                -1,
                dtype=torch.int64,
            )
            local_vectors = local_vectors.index_select(1, reverse)
            score_values = eigenvalues.index_select(0, reverse)
            frequencies: tuple[float | None, ...] = (None,) * len(group)
        else:
            block = partition.projected_laplacian.index_select(
                0,
                indices,
            ).index_select(1, indices)
            eigenvalues, local_vectors = _deterministic_psd_eigh(
                (block + block.T) * 0.5
            )
            local_responses = local_vectors.T @ group_response
            score_values = local_responses.square().sum(dim=1)
            frequencies = tuple(float(value) for value in eigenvalues)
        lifted = _canonicalize_columns(group_parent @ local_vectors)
        for local_ordinal in range(len(group)):
            columns.append(lifted[:, local_ordinal])
            records.append(
                (
                    float(score_values[local_ordinal]),
                    group_ordinal,
                    local_ordinal,
                    frequencies[local_ordinal],
                )
            )

    order = tuple(
        sorted(
            range(len(records)),
            key=lambda index: (
                -records[index][0],
                records[index][1],
                records[index][2],
            ),
        )
    )
    ordered_basis = _canonicalize_columns(
        torch.stack([columns[index] for index in order], dim=1)
    )
    ordered_scores = torch.tensor(
        [records[index][0] for index in order],
        dtype=torch.float64,
    )
    group_ordinals = tuple(records[index][1] for index in order)
    local_ordinals = tuple(records[index][2] for index in order)
    component_frequencies = tuple(records[index][3] for index in order)
    payload = _grouped_basis_payload(
        method=method,
        basis=ordered_basis,
        scores=ordered_scores,
        group_ordinals=group_ordinals,
        local_ordinals=local_ordinals,
        frequencies=component_frequencies,
        partition=partition,
        weighted_response=response,
        parent_basis=basis,
        fit_origins=origins,
        response_binding_sha256=response_binding_sha256,
        parent_subspace_artifact_sha256=(
            parent_subspace_artifact_sha256
        ),
    )
    result = FitOnlyGraphWaveletGroupedBasis(
        method=method,
        basis=ordered_basis,
        component_scores=ordered_scores,
        component_group_ordinals=group_ordinals,
        component_local_ordinals=local_ordinals,
        component_frequencies=component_frequencies,
        partition_artifact_sha256=partition.artifact_sha256,
        fit_weighted_response_sha256=_tensor_sha256(response),
        fit_weighted_response_shape=tuple(response.shape),
        fit_origins=origins,
        response_binding_sha256=response_binding_sha256,
        parent_subspace_artifact_sha256=(
            parent_subspace_artifact_sha256
        ),
        parent_basis_sha256=_tensor_sha256(basis),
        artifact_sha256=_json_sha256(payload, domain=_BASIS_DOMAIN),
    )
    result.validate_against(basis, partition, response)
    return result


def grouped_basis_one_hot_control(
    parent_basis: Tensor,
    fit_weighted_response: Tensor,
    partition: GraphWaveletTopologyPartition,
    grouped_basis: FitOnlyGraphWaveletGroupedBasis,
    *,
    rank: int,
) -> Tensor:
    """Retain original parent columns under a grouped basis' rank allocation."""

    basis = _float_tensor(parent_basis, label="parent_basis", ndim=2)
    response = _float_tensor(
        fit_weighted_response,
        label="fit_weighted_response",
        ndim=4,
    )
    partition.validate_integrity()
    grouped_basis.validate_against(basis, partition, response)
    if (
        not 1 <= rank <= grouped_basis.rank
        or _tensor_sha256(basis) != partition.parent_basis_sha256
        or _tensor_sha256(basis) != grouped_basis.parent_basis_sha256
        or _tensor_sha256(response)
        != grouped_basis.fit_weighted_response_sha256
        or grouped_basis.partition_artifact_sha256
        != partition.artifact_sha256
    ):
        raise ValueError("one-hot control binding or rank differs")
    allocations = [
        grouped_basis.component_group_ordinals[:rank].count(group)
        for group in range(partition.group_count)
    ]
    coordinates = basis.T @ response.reshape(response.shape[0], -1)
    columns: list[tuple[float, int, int, Tensor]] = []
    for group_ordinal, (group, retained) in enumerate(
        zip(partition.groups, allocations, strict=True)
    ):
        ordered = sorted(
            group,
            key=lambda index: (
                -float(coordinates[index].square().sum()),
                index,
            ),
        )
        for parent_index in ordered[:retained]:
            columns.append(
                (
                    float(coordinates[parent_index].square().sum()),
                    group_ordinal,
                    parent_index,
                    basis[:, parent_index],
                )
            )
    columns.sort(key=lambda item: (-item[0], item[1], item[2]))
    if len(columns) != rank:
        raise RuntimeError("one-hot control rank allocation is inconsistent")
    return _canonicalize_columns(
        torch.stack([item[3] for item in columns], dim=1)
    )


def grouped_basis_projector_overlap(left: Tensor, right: Tensor) -> float:
    """Return mean squared canonical correlation for equal-rank bases."""

    first = _float_tensor(left, label="left", ndim=2)
    second = _float_tensor(right, label="right", ndim=2)
    if (
        first.shape != second.shape
        or _orthonormal_error(first) > 2.0e-10
        or _orthonormal_error(second) > 2.0e-10
    ):
        raise ValueError("projector overlap requires equal orthonormal bases")
    rank = first.shape[1]
    return float((first.T @ second).square().sum() / rank)
