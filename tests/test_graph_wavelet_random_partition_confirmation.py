from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from fisher_graph.graph_wavelet_grouped_basis import (
    fit_graph_wavelet_grouped_basis as _fit_graph_wavelet_grouped_basis,
)
from fisher_graph.graph_wavelet_random_partition_confirmation import (
    CONFIRMATION_GROUP_COUNT,
    CONFIRMATION_GROUP_SIZE,
    CONFIRMATION_PARENT_RANK,
    PRIMARY_NULL_GATES,
    RANDOM_CONTROL_COUNT,
    derive_balanced_random_mode_partition,
    derive_balanced_random_partition_panel,
    evaluate_random_partition_null_panel,
    grouped_parent_basis_sha256,
)


_CANDIDATE_SHA256 = "a" * 64
_PARENT_SHA256 = "b" * 64
_NATIVE_PARTITION_SHA256 = "c" * 64
_NATIVE_GROUPS = tuple(
    tuple(range(start, start + CONFIRMATION_GROUP_SIZE))
    for start in range(
        0,
        CONFIRMATION_PARENT_RANK,
        CONFIRMATION_GROUP_SIZE,
    )
)
_FAMILY_IDS = tuple(f"family-{ordinal}" for ordinal in range(8))


def _panel():
    return derive_balanced_random_partition_panel(
        candidate_artifact_sha256=_CANDIDATE_SHA256,
        parent_basis_sha256=_PARENT_SHA256,
        native_partition_artifact_sha256=_NATIVE_PARTITION_SHA256,
        native_groups=_NATIVE_GROUPS,
    )


def _evaluate(
    native_family_sses: tuple[float, ...],
    control_family_sses: tuple[tuple[float, ...], ...],
):
    return evaluate_random_partition_null_panel(
        _panel(),
        family_ids=_FAMILY_IDS,
        native_pooled_sse=math.fsum(native_family_sses),
        control_pooled_sses=tuple(
            math.fsum(row) for row in control_family_sses
        ),
        native_family_sses=native_family_sses,
        control_family_sses=control_family_sses,
    )


def test_panel_derives_63_unique_deterministic_balanced_controls() -> None:
    first = _panel()
    second = _panel()

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.controls == second.controls
    assert len(first.controls) == RANDOM_CONTROL_COUNT == 63
    assert len(set(first.control_artifact_sha256s)) == 63
    assert len({control.groups for control in first.controls}) == 63
    assert tuple(control.control_ordinal for control in first.controls) == tuple(
        range(63)
    )
    for control in first.controls:
        assert len(control.groups) == CONFIRMATION_GROUP_COUNT == 8
        assert control.group_sizes == (CONFIRMATION_GROUP_SIZE,) * 8
        assert tuple(
            sorted(member for group in control.groups for member in group)
        ) == tuple(range(CONFIRMATION_PARENT_RANK))
        assert control.random_control is True
        assert control.topology_partition is False
        assert control.topology_used is False
        assert control.response_values_used is False
        assert control.metadata()["graph_topology_claim"] is False


def test_controls_are_seeded_by_candidate_parent_and_ordinal() -> None:
    baseline = derive_balanced_random_mode_partition(
        candidate_artifact_sha256=_CANDIDATE_SHA256,
        parent_basis_sha256=_PARENT_SHA256,
        control_ordinal=0,
    )
    different_candidate = derive_balanced_random_mode_partition(
        candidate_artifact_sha256="d" * 64,
        parent_basis_sha256=_PARENT_SHA256,
        control_ordinal=0,
    )
    different_parent = derive_balanced_random_mode_partition(
        candidate_artifact_sha256=_CANDIDATE_SHA256,
        parent_basis_sha256="e" * 64,
        control_ordinal=0,
    )
    different_ordinal = derive_balanced_random_mode_partition(
        candidate_artifact_sha256=_CANDIDATE_SHA256,
        parent_basis_sha256=_PARENT_SHA256,
        control_ordinal=1,
    )

    assert baseline.groups != different_candidate.groups
    assert baseline.groups != different_parent.groups
    assert baseline.groups != different_ordinal.groups
    assert len(
        {
            baseline.artifact_sha256,
            different_candidate.artifact_sha256,
            different_parent.artifact_sha256,
            different_ordinal.artifact_sha256,
        }
    ) == 4


def test_controls_do_not_depend_on_native_topology_groups() -> None:
    interleaved_native_groups = tuple(
        tuple(range(offset, CONFIRMATION_PARENT_RANK, 8))
        for offset in range(8)
    )
    contiguous = _panel()
    interleaved = derive_balanced_random_partition_panel(
        candidate_artifact_sha256=_CANDIDATE_SHA256,
        parent_basis_sha256=_PARENT_SHA256,
        native_partition_artifact_sha256="f" * 64,
        native_groups=interleaved_native_groups,
    )

    assert (
        contiguous.control_artifact_sha256s
        == interleaved.control_artifact_sha256s
    )
    assert contiguous.artifact_sha256 != interleaved.artifact_sha256


def test_partition_and_panel_reject_tampering_or_native_collision() -> None:
    control = _panel().controls[0]
    swapped = [list(group) for group in control.groups]
    swapped[0][0], swapped[1][0] = swapped[1][0], swapped[0][0]
    changed_groups = tuple(
        sorted(tuple(sorted(group)) for group in swapped)
    )

    with pytest.raises(ValueError, match="deterministic replay"):
        replace(control, groups=changed_groups)
    with pytest.raises(ValueError, match="fields are invalid"):
        replace(control, topology_used=True)
    with pytest.raises(ValueError, match="artifact hash differs"):
        replace(control, artifact_sha256="0" * 64)
    with pytest.raises(ValueError, match="distinct from native groups"):
        derive_balanced_random_partition_panel(
            candidate_artifact_sha256=_CANDIDATE_SHA256,
            parent_basis_sha256=_PARENT_SHA256,
            native_partition_artifact_sha256=(
                _NATIVE_PARTITION_SHA256
            ),
            native_groups=control.groups,
        )


def test_random_control_is_duck_compatible_with_local_svd_only() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    parent = torch.eye(
        CONFIRMATION_PARENT_RANK,
        dtype=torch.float64,
    )
    response = torch.randn(
        (CONFIRMATION_PARENT_RANK, 3, 2, 3),
        generator=generator,
        dtype=torch.float64,
    )
    control = derive_balanced_random_mode_partition(
        candidate_artifact_sha256=_CANDIDATE_SHA256,
        parent_basis_sha256=grouped_parent_basis_sha256(parent),
        control_ordinal=0,
    )
    fitted = _fit_graph_wavelet_grouped_basis(
        parent,
        response,
        control,  # type: ignore[arg-type]
        method="wavelet_local_svd",
        fit_origins=(8, 24, 40),
        response_binding_sha256="1" * 64,
        parent_subspace_artifact_sha256="2" * 64,
    )

    assert fitted.partition_artifact_sha256 == control.artifact_sha256
    assert fitted.rank == CONFIRMATION_PARENT_RANK
    torch.testing.assert_close(
        fitted.basis.T @ fitted.basis,
        torch.eye(CONFIRMATION_PARENT_RANK, dtype=torch.float64),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    with pytest.raises(ValueError, match="no graph Laplacian"):
        _fit_graph_wavelet_grouped_basis(
            parent,
            response,
            control,  # type: ignore[arg-type]
            method="wavelet_cluster_gfa",
            fit_origins=(8, 24, 40),
            response_binding_sha256="1" * 64,
            parent_subspace_artifact_sha256="2" * 64,
        )


def test_primary_null_panel_passes_all_three_pre_registered_gates() -> None:
    native = (1.0,) * 8
    controls = ((0.9,) * 8,) * 2 + ((1.2,) * 8,) * 61

    result = _evaluate(native, controls)
    replay = _evaluate(native, controls)

    assert result.artifact_sha256 == replay.artifact_sha256
    assert result.better_or_tied_control_count == 2
    assert result.empirical_p_value == pytest.approx(3.0 / 64.0)
    assert result.median_control_pooled_sse == pytest.approx(9.6)
    assert result.median_sse_recovery_fraction == pytest.approx(1.0 / 6.0)
    assert result.family_wins == (True,) * 8
    assert result.family_win_count == 8
    assert all(result.gate_results.values())
    assert result.passed is True
    assert PRIMARY_NULL_GATES.maximum_empirical_p_value == 0.05
    assert PRIMARY_NULL_GATES.minimum_median_sse_recovery_fraction == 0.05
    assert PRIMARY_NULL_GATES.minimum_family_win_count == 7


def test_empirical_p_gate_counts_ties_against_the_native_partition() -> None:
    native = (1.0,) * 8
    controls = ((0.9,) * 8,) * 2 + ((1.0,) * 8,) + ((1.2,) * 8,) * 60

    result = _evaluate(native, controls)

    assert result.better_or_tied_control_count == 3
    assert result.empirical_p_value == pytest.approx(4.0 / 64.0)
    assert result.empirical_p_gate_passed is False
    assert result.median_sse_recovery_gate_passed is True
    assert result.family_win_gate_passed is True
    assert result.passed is False


def test_recovery_and_family_gates_fail_independently() -> None:
    native = (1.0,) * 8
    low_recovery_controls = ((0.9,) * 8,) * 2 + ((1.04,) * 8,) * 61
    low_recovery = _evaluate(native, low_recovery_controls)

    assert low_recovery.empirical_p_gate_passed is True
    assert low_recovery.median_sse_recovery_fraction == pytest.approx(
        0.04 / 1.04
    )
    assert low_recovery.median_sse_recovery_gate_passed is False
    assert low_recovery.family_win_gate_passed is True

    six_win_native = (1.3, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    six_win_controls = ((0.9,) * 8,) * 2 + ((1.2,) * 8,) * 61
    six_wins = _evaluate(six_win_native, six_win_controls)

    assert six_wins.empirical_p_gate_passed is True
    assert six_wins.median_sse_recovery_gate_passed is True
    assert six_wins.family_wins == (
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    assert six_wins.family_win_count == 6
    assert six_wins.family_win_gate_passed is False
    assert six_wins.passed is False


def test_null_panel_rejects_bad_geometry_aggregates_and_tampering() -> None:
    panel = _panel()
    controls = ((1.2,) * 8,) * 63
    pooled = tuple(math.fsum(row) for row in controls)

    with pytest.raises(ValueError, match="geometry is invalid"):
        evaluate_random_partition_null_panel(
            panel,
            family_ids=_FAMILY_IDS[:-1],
            native_pooled_sse=8.0,
            control_pooled_sses=pooled,
            native_family_sses=(1.0,) * 8,
            control_family_sses=controls,
        )
    with pytest.raises(ValueError, match="does not equal the sum"):
        evaluate_random_partition_null_panel(
            panel,
            family_ids=_FAMILY_IDS,
            native_pooled_sse=8.1,
            control_pooled_sses=pooled,
            native_family_sses=(1.0,) * 8,
            control_family_sses=controls,
        )
    result = _evaluate((1.0,) * 8, controls)
    with pytest.raises(ValueError, match="differ from replay"):
        replace(result, family_win_count=7)
    with pytest.raises(ValueError, match="artifact hash differs"):
        replace(result, artifact_sha256="9" * 64)
