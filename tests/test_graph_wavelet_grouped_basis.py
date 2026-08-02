from __future__ import annotations

import pytest
import torch

from fisher_graph.graph_wavelet_grouped_basis import (
    fit_graph_wavelet_grouped_basis as _fit_graph_wavelet_grouped_basis,
    fit_graph_wavelet_topology_partition,
    grouped_basis_one_hot_control,
    grouped_basis_projector_overlap,
)


def fit_graph_wavelet_grouped_basis(
    parent_basis: torch.Tensor,
    response: torch.Tensor,
    partition: object,
    *,
    method: str,
):
    return _fit_graph_wavelet_grouped_basis(
        parent_basis,
        response,
        partition,  # type: ignore[arg-type]
        method=method,  # type: ignore[arg-type]
        fit_origins=(8, 24, 40),
        response_binding_sha256="7" * 64,
        parent_subspace_artifact_sha256="8" * 64,
    )


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260802)
    raw = torch.randn((8, 8), generator=generator, dtype=torch.float64)
    parent, _ = torch.linalg.qr(raw)
    adjacency = torch.zeros((8, 8), dtype=torch.float64)
    for index in range(7):
        adjacency[index, index + 1] = 1.0 + 0.1 * index
        adjacency[index + 1, index] = 1.0 + 0.1 * index
    adjacency[0, 4] = adjacency[4, 0] = 0.35
    adjacency[2, 7] = adjacency[7, 2] = 0.20
    laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency
    response = torch.randn(
        (8, 3, 4, 5),
        generator=generator,
        dtype=torch.float64,
    )
    response[1] += 0.7 * response[0]
    response[3] -= 0.5 * response[2]
    return parent.contiguous(), laplacian.contiguous(), response.contiguous()


def _assert_group_support(
    parent: torch.Tensor,
    family: object,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    coordinates = parent.T @ family.basis  # type: ignore[attr-defined]
    for column, group_ordinal in enumerate(
        family.component_group_ordinals  # type: ignore[attr-defined]
    ):
        allowed = set(groups[group_ordinal])
        outside = [
            index for index in range(parent.shape[1]) if index not in allowed
        ]
        if outside:
            assert float(coordinates[outside, column].abs().max()) < 2.0e-12


def test_partition_is_deterministic_balanced_and_topology_only() -> None:
    parent, laplacian, _ = _fixture()
    first = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )
    second = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.groups == second.groups
    assert first.group_sizes == (4, 4)
    assert len(first.merge_history) == 6
    assert first.topology_top_k == 3
    torch.testing.assert_close(
        first.topology_coupling,
        second.topology_coupling,
    )


def test_explicit_one_group_control_reaches_global_svd_ceiling() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=1,
        topology_top_k=3,
    )
    grouped = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="global_svd_control",
    )
    global_left, _, _ = torch.linalg.svd(
        response.reshape(8, -1),
        full_matrices=True,
    )

    for rank in (1, 3, 7, 8):
        grouped_prefix = grouped.prefix(rank)
        global_prefix = global_left[:, :rank]
        torch.testing.assert_close(
            grouped_prefix @ grouped_prefix.T,
            global_prefix @ global_prefix.T,
            atol=3.0e-11,
            rtol=3.0e-11,
        )


def test_local_svd_is_orthonormal_energy_ordered_and_block_confined() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=4,
        topology_top_k=3,
    )
    family = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_local_svd",
    )

    torch.testing.assert_close(
        family.basis.T @ family.basis,
        torch.eye(8, dtype=torch.float64),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert tuple(family.component_scores) == tuple(
        sorted(family.component_scores, reverse=True)
    )
    assert set(family.component_frequencies) == {None}
    _assert_group_support(parent, family, partition.groups)


def test_cluster_gfa_is_deterministic_and_block_confined() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )
    first = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_cluster_gfa",
    )
    second = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_cluster_gfa",
    )

    assert first.artifact_sha256 == second.artifact_sha256
    torch.testing.assert_close(first.basis, second.basis)
    assert all(
        frequency is not None for frequency in first.component_frequencies
    )
    _assert_group_support(parent, first, partition.groups)


def test_one_hot_control_matches_allocation_without_local_rotations() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=4,
        topology_top_k=3,
    )
    family = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_local_svd",
    )
    control = grouped_basis_one_hot_control(
        parent,
        response,
        partition,
        family,
        rank=5,
    )

    torch.testing.assert_close(
        control.T @ control,
        torch.eye(5, dtype=torch.float64),
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    assert grouped_basis_projector_overlap(
        family.prefix(5),
        family.prefix(5),
    ) == pytest.approx(1.0)
    assert 0.0 <= grouped_basis_projector_overlap(
        family.prefix(5),
        control,
    ) <= 1.0 + 1.0e-12


def test_response_changes_basis_but_not_topology_partition() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )
    changed = response.clone()
    changed[0] *= 7.0
    changed[4] -= 3.0 * changed[5]
    first = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_local_svd",
    )
    second = fit_graph_wavelet_grouped_basis(
        parent,
        changed,
        partition,
        method="wavelet_local_svd",
    )

    assert first.partition_artifact_sha256 == second.partition_artifact_sha256
    assert first.artifact_sha256 != second.artifact_sha256


def test_invalid_unbalanced_partition_and_parent_binding_are_rejected() -> None:
    parent, laplacian, response = _fixture()
    with pytest.raises(ValueError, match="partition inputs are invalid"):
        fit_graph_wavelet_topology_partition(
            parent,
            laplacian,
            group_count=3,
            topology_top_k=3,
        )
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )
    changed_parent = parent.roll(1, dims=0).contiguous()
    with pytest.raises(
        ValueError,
        match="geometry or parent binding differs",
    ):
        fit_graph_wavelet_grouped_basis(
            changed_parent,
            response,
            partition,
            method="wavelet_local_svd",
        )


def test_local_label_rejects_one_group_and_zero_topology_merges() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=1,
        topology_top_k=3,
    )
    with pytest.raises(
        ValueError,
        match="geometry or parent binding differs",
    ):
        fit_graph_wavelet_grouped_basis(
            parent,
            response,
            partition,
            method="wavelet_local_svd",
        )

    with pytest.raises(
        ValueError,
        match="no positive coupling",
    ):
        fit_graph_wavelet_topology_partition(
            torch.eye(8, dtype=torch.float64),
            torch.eye(8, dtype=torch.float64),
            group_count=2,
            topology_top_k=3,
        )


def test_mutation_is_detected_before_basis_use() -> None:
    parent, laplacian, response = _fixture()
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )
    family = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_local_svd",
    )
    family.basis[0, 0].add_(1.0)
    with pytest.raises(ValueError, match="fields are invalid|hash differs"):
        family.prefix(3)


def test_rectangular_parent_is_supported_and_indefinite_graph_rejected() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    parent, _ = torch.linalg.qr(
        torch.randn((10, 8), generator=generator, dtype=torch.float64)
    )
    adjacency = torch.zeros((10, 10), dtype=torch.float64)
    for index in range(9):
        adjacency[index, index + 1] = adjacency[index + 1, index] = 1.0
    laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency
    response = torch.randn(
        (10, 3, 2, 4),
        generator=generator,
        dtype=torch.float64,
    )
    partition = fit_graph_wavelet_topology_partition(
        parent,
        laplacian,
        group_count=2,
        topology_top_k=3,
    )
    family = fit_graph_wavelet_grouped_basis(
        parent,
        response,
        partition,
        method="wavelet_local_svd",
    )
    assert family.basis.shape == (10, 8)
    assert family.prefix(5).shape == (10, 5)

    indefinite = laplacian.clone()
    indefinite[0, 0] = -100.0
    with pytest.raises(
        ValueError,
        match="positive semidefinite",
    ):
        fit_graph_wavelet_topology_partition(
            parent,
            indefinite,
            group_count=2,
            topology_top_k=3,
        )
