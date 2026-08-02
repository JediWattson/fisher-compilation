from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4TrainingSequence,
    fit_autonomous_complete_h4_residual,
)
from fisher_graph.complete_h4_fisher_conditional_residual import (
    FISHER_XY_OPERATOR_NORM_BOUND,
    autonomous_complete_h4_fisher_xy_provider_from_state_dict,
    autonomous_complete_h4_fisher_xy_provider_state_dict,
    fisher_xy_bounded_coordinates,
    fit_autonomous_complete_h4_fisher_xy_residual,
    load_autonomous_complete_h4_fisher_xy_provider,
    project_fisher_xy_conditional_factors,
    replay_autonomous_complete_h4_fisher_xy_bounded_coordinates,
    save_autonomous_complete_h4_fisher_xy_provider,
    summarize_fisher_xy_bounded_coordinate_geometry,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)


_BRIDGE = "b" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 18) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(7100 + index)
    source = torch.randn(length, _SOURCE_RANK, generator=generator, dtype=torch.float64)
    base = torch.randn(length, _WIDTH, generator=generator, dtype=torch.float64)
    source_kernel = torch.randn(
        _SOURCE_RANK, _RANK, generator=generator, dtype=torch.float64
    ) * 0.025
    state_kernel = torch.tensor(
        [
            [0.16, -0.03, 0.02, 0.01],
            [0.02, 0.13, -0.04, 0.00],
            [-0.01, 0.03, 0.12, 0.02],
            [0.00, -0.02, 0.04, 0.11],
        ],
        dtype=torch.float64,
    )
    parent_modal = source @ source_kernel + base[:, :_RANK] @ state_kernel
    raw = torch.stack(
        (
            1.2 * parent_modal[:, 0] + 0.4 * parent_modal[:, 2],
            -0.9 * parent_modal[:, 1] + 0.3 * parent_modal[:, 3],
        ),
        dim=1,
    )
    coordinates = raw / (0.75 + raw.abs())
    a_x = torch.tensor(
        [
            [0.07, 0.01, 0.00, 0.00],
            [0.00, -0.05, 0.01, 0.00],
            [0.01, 0.00, 0.04, 0.00],
            [0.00, 0.01, 0.00, -0.03],
        ],
        dtype=torch.float64,
    )
    a_y = torch.tensor(
        [
            [-0.04, 0.00, 0.01, 0.00],
            [0.01, 0.06, 0.00, 0.00],
            [0.00, 0.01, -0.03, 0.01],
            [0.00, 0.00, 0.01, 0.04],
        ],
        dtype=torch.float64,
    )
    a_xy = torch.tensor(
        [
            [0.025, 0.00, 0.00, 0.01],
            [0.00, -0.020, 0.01, 0.00],
            [0.01, 0.00, 0.018, 0.00],
            [0.00, 0.01, 0.00, -0.015],
        ],
        dtype=torch.float64,
    )
    conditional = (
        coordinates[:, :1] * (parent_modal @ a_x)
        + coordinates[:, 1:] * (parent_modal @ a_y)
        + coordinates[:, :1] * coordinates[:, 1:] * (parent_modal @ a_xy)
    )
    decoder = _decoder()
    native = base + (parent_modal + conditional) @ decoder

    # The two reverse-VJP directions are deterministic, independent, and
    # predictable from the parent modal.  Remaining H4 coordinates stay zero.
    modal_gradient = torch.stack(
        (
            parent_modal[:, 0] + 0.2 * parent_modal[:, 2],
            parent_modal[:, 1] - 0.3 * parent_modal[:, 3],
            0.2 * parent_modal[:, 0] + 0.1 * parent_modal[:, 2],
            -0.1 * parent_modal[:, 1] + 0.2 * parent_modal[:, 3],
        ),
        dim=1,
    )
    gradients = torch.zeros((length, _WIDTH), dtype=torch.float64)
    gradients[:, :_RANK] = modal_gradient
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"example-{index}",
        family_id=f"family-{index % 3}",
        source_modes=source,
        logical_positions=torch.arange(length, dtype=torch.int64),
        valid_mask=mask,
        source_mask=mask,
        support_mask=mask,
        base_h4=base,
        native_h4=native,
        reverse_vjp_gradients=gradients,
    )


def _collapsed_router_sequence(
    index: int,
    *,
    family_id: str | None = None,
) -> AutonomousCompleteH4TrainingSequence:
    length = 4
    first = torch.tensor((-1.0, -1.0, 1.0, 1.0), dtype=torch.float64)
    independent = torch.tensor((-1.0, 1.0, -1.0, 1.0), dtype=torch.float64)
    source = torch.zeros((length, _SOURCE_RANK), dtype=torch.float64)
    source[:, 0] = first
    base = torch.zeros((length, _WIDTH), dtype=torch.float64)
    target_modal = torch.stack((first, torch.zeros_like(first)), dim=1)
    native = base + target_modal @ torch.eye(
        _WIDTH,
        dtype=torch.float64,
    )[:2]
    gradients = torch.zeros((length, _WIDTH), dtype=torch.float64)
    gradients[:, 0] = first
    gradients[:, 1] = 2.0 * first + independent
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"collapsed-example-{index}",
        family_id=(
            f"collapsed-family-{index % 3}"
            if family_id is None
            else family_id
        ),
        source_modes=source,
        logical_positions=torch.arange(length, dtype=torch.int64),
        valid_mask=mask,
        source_mask=mask,
        support_mask=mask,
        base_h4=base,
        native_h4=native,
        reverse_vjp_gradients=gradients,
    )


def _prefix(sequence: AutonomousCompleteH4TrainingSequence) -> Gemma3L3L4OnePassPrefix:
    length = sequence.source_modes.shape[0]
    return Gemma3L3L4OnePassPrefix(
        source_modes=sequence.source_modes.unsqueeze(0),
        clamped_y3=torch.zeros((1, length, _WIDTH), dtype=torch.float64),
        predicted_target_modal_delta=torch.zeros((1, length, 1), dtype=torch.float64),
        decoded_base_x4_delta=torch.zeros((1, length, _WIDTH), dtype=torch.float64),
        logical_positions=sequence.logical_positions.unsqueeze(0),
        valid_target_mask=sequence.valid_mask.unsqueeze(0),
        source_eligible_mask=sequence.source_mask.unsqueeze(0),
        target_affected_mask=sequence.support_mask.unsqueeze(0),
        bridge_binding_sha256=_BRIDGE,
    )


@pytest.fixture(scope="module")
def fitted():
    sequences = tuple(_sequence(index) for index in range(6))
    parent = fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=_decoder(),
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-7,
    )
    fisher = fit_autonomous_complete_h4_fisher_xy_residual(
        sequences=sequences,
        parent_provider=parent,
        conditional_rank=_RANK,
        coordinate_objective="reverse_vjp_fisher",
        router_ridge=1.0e-7,
        conditional_ridge=1.0e-7,
    )
    pca = fit_autonomous_complete_h4_fisher_xy_residual(
        sequences=sequences,
        parent_provider=parent,
        conditional_rank=_RANK,
        coordinate_objective="activation_pca",
        router_ridge=1.0e-7,
        conditional_ridge=1.0e-7,
    )
    return sequences, parent, fisher, pca


def test_parent_modal_decode_path_is_semantics_preserving(fitted) -> None:
    sequences, parent, _fisher, _pca = fitted
    sequence = sequences[0]
    prefix = _prefix(sequence)
    realized = sequence.base_h4.unsqueeze(0)
    direct = parent.correction(prefix, realized)
    modal = parent.modal_correction(prefix, realized)
    decoded = parent.decode_modal(prefix, modal, like=realized)
    assert direct.dtype == torch.float64
    torch.testing.assert_close(decoded, direct, rtol=0.0, atol=0.0)


def test_rational_coordinates_preserve_device_and_remain_open_bounded() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raw = torch.tensor(
        [[-1.0e300, -2.0], [0.0, 3.0], [1.0e300, 8.0]],
        dtype=torch.float64,
        device=device,
    )
    scales = torch.tensor([0.5, 2.0], dtype=torch.float64, device=device)
    bounded = fisher_xy_bounded_coordinates(raw, scales)
    assert bounded.device == raw.device
    assert bounded.dtype == torch.float64
    assert bool((bounded.abs() < 1.0).all())
    assert bounded[1, 0] == 0.0


def test_four_corner_projection_bounds_every_sampled_interior_operator() -> None:
    generator = torch.Generator().manual_seed(88)
    left = torch.randn(3 * _RANK, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(3, _RANK, generator=generator, dtype=torch.float64)
    projected_left, projected_right, pre, post, scale = (
        project_fisher_xy_conditional_factors(left, right)
    )
    assert max(pre) > FISHER_XY_OPERATOR_NORM_BOUND
    assert max(post) <= FISHER_XY_OPERATOR_NORM_BOUND
    assert 0.0 < scale < 1.0
    blocks = projected_left.reshape(3, _RANK, 3)
    a_x, a_y, a_xy = (block @ projected_right for block in blocks)
    for coordinates in torch.rand((128, 2), generator=generator) * 2.0 - 1.0:
        c_x, c_y = (float(value) for value in coordinates)
        operator = c_x * a_x + c_y * a_y + c_x * c_y * a_xy
        assert float(torch.linalg.svdvals(operator).max()) <= max(post) + 1.0e-12


def test_fisher_and_pca_controls_are_parameter_matched_and_improve_fit(fitted) -> None:
    _sequences, parent, fisher, pca = fitted
    assert fisher.coordinate_objective == "reverse_vjp_fisher"
    assert pca.coordinate_objective == "activation_pca"
    assert fisher.coordinate_axes_sha256 != pca.coordinate_axes_sha256
    assert fisher.incremental_prepared_float_scalar_count == 2 * _RANK + 4 + 4 * _RANK * _RANK
    assert fisher.incremental_logical_macs_per_token_upper_bound == 2 * _RANK + 4 * _RANK * _RANK
    assert fisher.incremental_prepared_float_scalar_count == pca.incremental_prepared_float_scalar_count
    assert fisher.incremental_logical_macs_per_token_upper_bound == pca.incremental_logical_macs_per_token_upper_bound
    assert fisher.prepared_float_scalar_count == parent.prepared_float_scalar_count + fisher.incremental_prepared_float_scalar_count
    assert fisher.weighted_residual_rmse_after <= fisher.weighted_residual_rmse_before + 1.0e-12
    assert pca.weighted_residual_rmse_after <= pca.weighted_residual_rmse_before + 1.0e-12


def test_actual_bounded_router_coordinates_report_genuine_two_dimensionality(
    fitted,
) -> None:
    _sequences, _parent, fisher, pca = fitted
    for provider in (fisher, pca):
        first, second = provider.bounded_coordinate_covariance_eigenvalues
        assert first >= second > 0.0
        assert provider.bounded_coordinate_lambda2_over_lambda1 > 0.5
        assert provider.bounded_coordinate_abs_correlation < 0.5
        assert all(value > 0.8 for value in provider.bounded_coordinate_target_r2)
        assert provider.residual_second_coordinate_energy_fraction > 0.9
        metadata = provider.metadata()
        assert metadata["corner_certificate_scope"] == (
            "pointwise_conditional_correction_amplitude_operator_bound_"
            "not_full_nonlinear_jacobian_or_lipschitz_bound"
        )
        assert metadata["runtime_parameter_bytes_float64_scope"] == (
            "prepared_float_payload_not_peak_runtime_memory"
        )
        assert metadata["router_bias_additions_per_token"] == 2


def test_collapsed_runtime_router_is_diagnosed_without_rejecting_provider() -> None:
    sequences = tuple(_collapsed_router_sequence(index) for index in range(6))
    decoder = torch.eye(_WIDTH, dtype=torch.float64)[:2].contiguous()
    parent = fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=decoder,
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-7,
    )
    provider = fit_autonomous_complete_h4_fisher_xy_residual(
        sequences=sequences,
        parent_provider=parent,
        conditional_rank=2,
        coordinate_objective="reverse_vjp_fisher",
        router_ridge=1.0e-7,
        conditional_ridge=1.0e-7,
    )

    first, second = provider.bounded_coordinate_covariance_eigenvalues
    assert first > 0.0
    assert second <= first * 1.0e-12
    assert provider.bounded_coordinate_lambda2_over_lambda1 <= 1.0e-12
    assert provider.bounded_coordinate_abs_correlation >= 1.0 - 1.0e-12
    assert provider.residual_second_coordinate_energy_fraction <= 1.0e-12
    assert 0.0 < provider.bounded_coordinate_target_r2[0] < 1.0
    assert provider.bounded_coordinate_target_r2[1] < 0.0
    provider.validate_integrity()


def test_offline_replay_exactly_matches_the_allowed_runtime_formula(fitted) -> None:
    sequences, parent, fisher, _pca = fitted
    sequence = sequences[2]
    replayed = replay_autonomous_complete_h4_fisher_xy_bounded_coordinates(
        fisher,
        sequence,
    )
    prefix = _prefix(sequence)
    parent_modal = parent.modal_correction(
        prefix,
        sequence.base_h4.unsqueeze(0),
    )
    expected = fisher.bounded_coordinates(parent_modal)[0][sequence.support_mask]
    assert replayed.device.type == "cpu"
    assert replayed.dtype == torch.float64
    torch.testing.assert_close(replayed, expected, rtol=0.0, atol=0.0)

    geometry = summarize_fisher_xy_bounded_coordinate_geometry(replayed)
    geometry.validate_integrity()
    assert geometry.row_count == int(sequence.support_mask.sum())
    assert geometry.metadata()["artifact_sha256"] == geometry.artifact_sha256
    assert "bounded_coordinate_target_r2" not in geometry.metadata()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        replace(
            geometry,
            artifact_sha256="0" * 64,
        )


def test_offline_replay_does_not_read_native_h4_or_reverse_vjp(fitted) -> None:
    sequences, _parent, fisher, _pca = fitted
    sequence = sequences[3]
    reference = replay_autonomous_complete_h4_fisher_xy_bounded_coordinates(
        fisher,
        sequence,
    )
    generator = torch.Generator().manual_seed(9951)
    forbidden_fields_changed = AutonomousCompleteH4TrainingSequence(
        example_id="forbidden-fields-changed",
        family_id="held-family",
        source_modes=sequence.source_modes,
        logical_positions=sequence.logical_positions,
        valid_mask=sequence.valid_mask,
        source_mask=sequence.source_mask,
        support_mask=sequence.support_mask,
        base_h4=sequence.base_h4,
        native_h4=torch.randn(
            sequence.native_h4.shape,
            generator=generator,
            dtype=torch.float64,
        ),
        reverse_vjp_gradients=None,
    )
    replayed = replay_autonomous_complete_h4_fisher_xy_bounded_coordinates(
        fisher,
        forbidden_fields_changed,
    )
    torch.testing.assert_close(replayed, reference, rtol=0.0, atol=0.0)


def test_held_replay_geometry_detects_a_collapsed_coordinate_square() -> None:
    training = tuple(_collapsed_router_sequence(index) for index in range(6))
    decoder = torch.eye(_WIDTH, dtype=torch.float64)[:2].contiguous()
    parent = fit_autonomous_complete_h4_residual(
        sequences=training,
        output_decoder=decoder,
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-7,
    )
    provider = fit_autonomous_complete_h4_fisher_xy_residual(
        sequences=training,
        parent_provider=parent,
        conditional_rank=2,
        coordinate_objective="reverse_vjp_fisher",
        router_ridge=1.0e-7,
        conditional_ridge=1.0e-7,
    )
    held = _collapsed_router_sequence(200, family_id="held-family")
    assert held.family_id not in provider.fit_family_ids
    held_coordinates = (
        replay_autonomous_complete_h4_fisher_xy_bounded_coordinates(
            provider,
            held,
        )
    )
    geometry = summarize_fisher_xy_bounded_coordinate_geometry(
        held_coordinates
    )

    assert geometry.covariance_eigenvalues[0] > 0.0
    assert geometry.covariance_eigenvalues[1] == 0.0
    assert geometry.lambda2_over_lambda1 == 0.0
    assert geometry.abs_correlation == 1.0
    assert geometry.residual_second_coordinate_energy_fraction == 0.0
    geometry.validate_integrity()


def test_runtime_is_source_free_causal_and_confined_to_complete_support(fitted) -> None:
    sequences, _parent, fisher, _pca = fitted
    sequence = sequences[1]
    prefix = _prefix(sequence)
    realized = sequence.base_h4.unsqueeze(0)
    correction = fisher.correction(prefix, realized)
    assert correction.dtype == torch.float64
    support = prefix.complete_h4_causal_support_mask()
    assert torch.count_nonzero(correction[~support]) == 0
    assert fisher.metadata()["runtime_forbidden_inputs"] == (
        "native_h4",
        "targets",
        "logits",
        "gradients",
        "coordinate_axes",
        "family_ids",
    )

    cutoff = 7
    changed_source = sequence.source_modes.clone()
    changed_source[cutoff + 1 :] += 1000.0
    changed_base = sequence.base_h4.clone()
    changed_base[cutoff + 1 :] -= 1000.0
    changed = AutonomousCompleteH4TrainingSequence(
        example_id="changed",
        family_id="held-out",
        source_modes=changed_source,
        logical_positions=sequence.logical_positions,
        valid_mask=sequence.valid_mask,
        source_mask=sequence.source_mask,
        support_mask=sequence.support_mask,
        base_h4=changed_base,
        native_h4=changed_base,
        reverse_vjp_gradients=sequence.reverse_vjp_gradients,
    )
    changed_correction = fisher.correction(
        _prefix(changed), changed_base.unsqueeze(0)
    )
    torch.testing.assert_close(
        changed_correction[:, : cutoff + 1],
        correction[:, : cutoff + 1],
        rtol=0.0,
        atol=0.0,
    )


def test_state_roundtrip_tamper_rejection_and_secure_file_roundtrip(
    fitted,
    tmp_path,
) -> None:
    _sequences, _parent, fisher, _pca = fitted
    state = autonomous_complete_h4_fisher_xy_provider_state_dict(fisher)
    restored = autonomous_complete_h4_fisher_xy_provider_from_state_dict(
        state,
        expected_artifact_sha256=fisher.artifact_sha256,
        expected_bridge_binding_sha256=_BRIDGE,
    )
    assert restored.metadata() == fisher.metadata()

    tampered = copy.deepcopy(state)
    tampered["tensors"]["conditional_left"][0, 0] += 1.0
    with pytest.raises(ValueError, match="trust receipt|artifact hash mismatch"):
        autonomous_complete_h4_fisher_xy_provider_from_state_dict(
            tampered,
            expected_artifact_sha256=fisher.artifact_sha256,
            expected_bridge_binding_sha256=_BRIDGE,
        )

    tampered_diagnostic = copy.deepcopy(state)
    tampered_diagnostic["bounded_coordinate_lambda2_over_lambda1"] *= 0.5
    with pytest.raises(
        ValueError,
        match="covariance ratio differs|artifact hash mismatch",
    ):
        autonomous_complete_h4_fisher_xy_provider_from_state_dict(
            tampered_diagnostic,
            expected_artifact_sha256=fisher.artifact_sha256,
            expected_bridge_binding_sha256=_BRIDGE,
        )

    path = tmp_path / "fisher-xy.pt"
    receipt = save_autonomous_complete_h4_fisher_xy_provider(fisher, path)
    loaded = load_autonomous_complete_h4_fisher_xy_provider(
        path,
        expected_artifact_sha256=fisher.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_BRIDGE,
    )
    assert loaded.metadata() == fisher.metadata()
    with pytest.raises(FileExistsError, match="overwrite"):
        save_autonomous_complete_h4_fisher_xy_provider(fisher, path)
