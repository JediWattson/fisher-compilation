from __future__ import annotations

import copy
from dataclasses import replace
import stat

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4TrainingSequence,
    fit_autonomous_complete_h4_residual,
)
from fisher_graph.complete_h4_fisher_conditional_pedal import (
    fit_autonomous_complete_h4_fisher_xy_pedal,
)
from fisher_graph.complete_h4_fisher_finite_joint_pedal import (
    FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION,
    autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict,
    autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict,
    canonical_balanced_rank_svd_retraction,
    dense_direction_descent_proposal,
    fisher_finite_joint_direction_features,
    fisher_finite_joint_matched_resource_geometry,
    fisher_finite_joint_modal_terms,
    fisher_finite_joint_pedal_control,
    initialize_autonomous_complete_h4_fisher_finite_joint_pedal,
    interpolate_fisher_finite_joint_pedal_parameters,
    load_autonomous_complete_h4_fisher_finite_joint_pedal_provider,
    refit_autonomous_complete_h4_fisher_finite_joint_pedal,
    replay_autonomous_complete_h4_fisher_finite_joint_pedal,
    save_autonomous_complete_h4_fisher_finite_joint_pedal_provider,
    validate_fisher_finite_joint_pedal_runtime_replay_metadata,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)


_BRIDGE = "d" * 64
_PROTOCOL = "a" * 64
_EVIDENCE = "b" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 12) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(53100 + index)
    source = torch.randn(length, _SOURCE_RANK, generator=generator, dtype=torch.float64)
    base = torch.randn(length, _WIDTH, generator=generator, dtype=torch.float64)
    source_kernel = torch.randn(
        _SOURCE_RANK,
        _RANK,
        generator=generator,
        dtype=torch.float64,
    ) * 0.02
    state_kernel = torch.tensor(
        [
            [0.18, -0.03, 0.02, 0.01],
            [0.02, 0.15, -0.04, 0.00],
            [-0.01, 0.03, 0.14, 0.02],
            [0.00, -0.02, 0.04, 0.13],
        ],
        dtype=torch.float64,
    )
    parent = source @ source_kernel + base[:, :_RANK] @ state_kernel
    raw = torch.stack(
        (
            parent[:, 0] + 0.3 * parent[:, 2],
            -0.8 * parent[:, 1] + 0.25 * parent[:, 3],
        ),
        dim=1,
    )
    coordinates = raw / (0.7 + raw.abs())
    direction = torch.stack(
        (
            0.22 * coordinates[:, 0] * parent[:, 0]
            - 0.12 * coordinates[:, 1] * parent[:, 1],
            0.18 * coordinates[:, 1] * parent[:, 1]
            + 0.09 * coordinates[:, 0] * parent[:, 2],
            -0.16 * coordinates[:, 0] * parent[:, 2]
            + 0.08 * coordinates.prod(dim=1) * parent[:, 0],
            0.14 * coordinates[:, 1] * parent[:, 3],
        ),
        dim=1,
    )
    pedal = torch.sigmoid(
        0.2
        + 0.6 * coordinates[:, 0]
        - 0.4 * coordinates[:, 1]
        + 0.2 * coordinates.prod(dim=1)
    )
    native = base + (parent + pedal.unsqueeze(1) * direction) @ _decoder()
    gradients = torch.zeros((length, _WIDTH), dtype=torch.float64)
    gradients[:, :_RANK] = torch.stack(
        (
            parent[:, 0] + 0.1 * parent[:, 2],
            parent[:, 1] - 0.2 * parent[:, 3],
            0.2 * parent[:, 0] + 0.1 * parent[:, 2],
            -0.1 * parent[:, 1] + 0.2 * parent[:, 3],
        ),
        dim=1,
    )
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"finite-joint-example-{index}",
        family_id=f"finite-joint-family-{index % 3}",
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
    start = fit_autonomous_complete_h4_fisher_xy_pedal(
        sequences=sequences,
        parent_provider=parent,
        conditional_rank=_RANK,
        coordinate_objective="reverse_vjp_fisher",
        pedal_mode="conditional",
        router_ridge=1.0e-7,
        direction_ridge=1.0e-7,
        pedal_ridge=1.0e-7,
    )
    initialized = initialize_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        fit_protocol_sha256=_PROTOCOL,
        fit_evidence_sha256=_EVIDENCE,
    )
    provider = refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        direction_left=0.93 * initialized.direction_left,
        direction_right=initialized.direction_right,
        pedal_weight=torch.tensor([0.7, -0.4, 0.25], dtype=torch.float64),
        pedal_bias=torch.tensor([-0.15], dtype=torch.float64),
        fit_protocol_sha256=_PROTOCOL,
        fit_evidence_sha256=_EVIDENCE,
    )
    return sequences, parent, start, initialized, provider


def test_balanced_retraction_proposal_and_interpolation_are_deterministic() -> None:
    generator = torch.Generator().manual_seed(771)
    dense = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    gradient = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    proposal = dense_direction_descent_proposal(
        dense,
        gradient,
        step_size=0.125,
    )
    torch.testing.assert_close(
        proposal,
        dense - 0.125 * gradient,
        rtol=0.0,
        atol=0.0,
    )
    left_a, right_a = canonical_balanced_rank_svd_retraction(proposal, rank=4)
    left_b, right_b = canonical_balanced_rank_svd_retraction(proposal, rank=4)
    torch.testing.assert_close(left_a, left_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(right_a, right_b, rtol=0.0, atol=0.0)
    torch.testing.assert_close(left_a @ right_a, proposal, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        left_a.T @ left_a,
        right_a @ right_a.T,
        rtol=1e-12,
        atol=1e-12,
    )
    weight, bias = interpolate_fisher_finite_joint_pedal_parameters(
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        fraction=0.25,
    )
    torch.testing.assert_close(weight, torch.full((3,), 0.25, dtype=torch.float64))
    torch.testing.assert_close(bias, torch.tensor([0.5], dtype=torch.float64))


def test_initialization_is_balanced_double_direction_with_half_pedal(fitted) -> None:
    _sequences, _parent, start, initialized, _provider = fitted
    torch.testing.assert_close(
        initialized.direction_left @ initialized.direction_right,
        2.0 * (start.direction_left @ start.direction_right),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        initialized.direction_left.T @ initialized.direction_left,
        initialized.direction_right @ initialized.direction_right.T,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.count_nonzero(initialized.pedal_weight) == 0
    assert torch.count_nonzero(initialized.pedal_bias) == 0
    coordinates = torch.tensor(
        [[-0.3, 0.2], [0.4, -0.1]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        initialized.pedal_values(coordinates),
        torch.full((2,), 0.5, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    for name in ("router_weight", "router_bias", "coordinate_scales"):
        torch.testing.assert_close(
            getattr(initialized, name),
            getattr(start, name),
            rtol=0.0,
            atol=0.0,
        )


def test_pure_terms_match_provider_replay_and_remain_differentiable(fitted) -> None:
    sequences, _parent, _start, _initialized, provider = fitted
    replay = replay_autonomous_complete_h4_fisher_finite_joint_pedal(
        provider,
        sequences[0],
    )
    left = provider.direction_left.clone().requires_grad_()
    right = provider.direction_right.clone().requires_grad_()
    beta = provider.pedal_weight.clone().requires_grad_()
    bias = provider.pedal_bias.clone().requires_grad_()
    direction, bounded, pedal, delta = fisher_finite_joint_modal_terms(
        replay.parent_modal,
        replay.bounded_coordinates,
        left,
        right,
        beta,
        bias,
    )
    torch.testing.assert_close(direction, replay.unbounded_direction, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bounded, replay.bounded_direction, rtol=0.0, atol=0.0)
    torch.testing.assert_close(pedal, replay.pedal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(delta, replay.emitted_delta, rtol=0.0, atol=0.0)
    features = fisher_finite_joint_direction_features(
        replay.parent_modal,
        replay.bounded_coordinates,
    )
    torch.testing.assert_close(
        direction,
        (features @ left) @ right,
        rtol=0.0,
        atol=0.0,
    )
    expected_pedal = torch.sigmoid(
        torch.cat(
            (
                replay.bounded_coordinates,
                replay.bounded_coordinates[:, :1]
                * replay.bounded_coordinates[:, 1:],
            ),
            dim=1,
        )
        @ beta
        + bias[0]
    )
    torch.testing.assert_close(pedal, expected_pedal, rtol=0.0, atol=0.0)
    delta.square().sum().backward()
    for value in (left.grad, right.grad, beta.grad, bias.grad):
        assert value is not None
        assert bool(torch.isfinite(value).all())
        assert torch.count_nonzero(value) > 0


def test_controls_are_direction_and_resource_matched(fitted) -> None:
    _sequences, _parent, start, _initialized, conditional = fitted
    intercept = fisher_finite_joint_pedal_control(
        conditional,
        pedal_mode="intercept",
    )
    unit = fisher_finite_joint_pedal_control(conditional, pedal_mode="unit")
    for control in (intercept, unit):
        for name in (
            "router_weight",
            "router_bias",
            "coordinate_scales",
            "direction_left",
            "direction_right",
        ):
            torch.testing.assert_close(
                getattr(control, name),
                getattr(conditional, name),
                rtol=0.0,
                atol=0.0,
            )
        assert control.incremental_prepared_float_scalar_count == (
            conditional.incremental_prepared_float_scalar_count
        )
        assert control.incremental_logical_macs_per_token_upper_bound == (
            conditional.incremental_logical_macs_per_token_upper_bound
        )
    expected_scalars = 2 * _RANK + 8 + 4 * _RANK * _RANK
    expected_macs = 2 * _RANK + 4 * _RANK * _RANK + 3
    for value in (conditional, intercept, unit):
        assert value.incremental_prepared_float_scalar_count == expected_scalars
        assert value.incremental_logical_macs_per_token_upper_bound == expected_macs
        assert value.incremental_prepared_float_scalar_count == (
            start.incremental_prepared_float_scalar_count
        )
        assert value.incremental_logical_macs_per_token_upper_bound == (
            start.incremental_logical_macs_per_token_upper_bound
        )
    for mode in ("conditional", "intercept", "unit"):
        k256 = fisher_finite_joint_matched_resource_geometry(
            parent_prepared_float_scalar_count=360_704,
            parent_logical_macs_per_token_upper_bound=524_288,
            rank=256,
            conditional_rank=16,
            pedal_mode=mode,
        )
        assert k256["prepared_float_scalar_count"] == 377_608
        assert k256["logical_macs_per_token_upper_bound"] == 541_187
    coordinates = torch.tensor(
        [[-0.8, 0.7], [0.4, -0.6]],
        dtype=torch.float64,
    )
    expected_intercept = torch.sigmoid(conditional.pedal_bias[0])
    torch.testing.assert_close(
        intercept.pedal_values(coordinates),
        torch.full((2,), expected_intercept, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        unit.pedal_values(coordinates),
        torch.ones(2, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_pointwise_trust_and_zero_direction_no_op(fitted) -> None:
    sequences, parent, start, _initialized, provider = fitted
    for sequence in sequences:
        replay = replay_autonomous_complete_h4_fisher_finite_joint_pedal(
            provider,
            sequence,
        )
        metadata = replay.metadata()
        assert validate_fisher_finite_joint_pedal_runtime_replay_metadata(metadata) == metadata
        assert metadata["pointwise_trust_certificate_passed"] is True
        assert metadata["max_bounded_direction_to_parent_norm_ratio"] <= (
            FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION + 1.0e-14
        )
        assert metadata["max_emitted_delta_to_parent_norm_ratio"] <= (
            FISHER_FINITE_JOINT_PEDAL_TRUST_FRACTION + 1.0e-14
        )
    no_op = refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        direction_left=torch.zeros_like(provider.direction_left),
        direction_right=provider.direction_right,
        pedal_weight=provider.pedal_weight,
        pedal_bias=provider.pedal_bias,
        fit_protocol_sha256=_PROTOCOL,
        fit_evidence_sha256="c" * 64,
    )
    sequence = sequences[0]
    replay = replay_autonomous_complete_h4_fisher_finite_joint_pedal(no_op, sequence)
    assert torch.count_nonzero(replay.unbounded_direction) == 0
    assert torch.count_nonzero(replay.bounded_direction) == 0
    assert torch.count_nonzero(replay.emitted_delta) == 0
    prefix = _prefix(sequence)
    realized = sequence.base_h4.unsqueeze(0)
    torch.testing.assert_close(
        no_op.modal_correction(prefix, realized),
        parent.modal_correction(prefix, realized),
        rtol=0.0,
        atol=0.0,
    )


def test_state_is_serving_only_and_roundtrips_fail_closed(fitted, tmp_path) -> None:
    _sequences, _parent, start, _initialized, provider = fitted
    state = autonomous_complete_h4_fisher_finite_joint_pedal_provider_state_dict(
        provider
    )
    assert set(state["tensors"]) == {
        "router_weight",
        "router_bias",
        "coordinate_scales",
        "direction_left",
        "direction_right",
        "pedal_weight",
        "pedal_bias",
    }
    assert "fit_tensors" not in state
    restored = (
        autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
            state,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_bridge_binding_sha256=_BRIDGE,
            expected_start_provider_artifact_sha256=start.artifact_sha256,
        )
    )
    assert restored.metadata() == provider.metadata()

    tampered = copy.deepcopy(state)
    tampered["tensors"]["direction_left"][0, 0] += 1.0
    with pytest.raises(ValueError, match="fit receipt hash mismatch"):
        autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
            tampered,
            expected_artifact_sha256=provider.artifact_sha256,
        )
    tampered = copy.deepcopy(state)
    tampered["fit_evidence_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fit receipt hash mismatch"):
        autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
            tampered,
            expected_artifact_sha256=provider.artifact_sha256,
        )
    with pytest.raises(ValueError, match="start provider differs"):
        autonomous_complete_h4_fisher_finite_joint_pedal_provider_from_state_dict(
            state,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_start_provider_artifact_sha256="e" * 64,
        )

    destination = tmp_path / "finite-joint.pt"
    receipt = save_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
        provider,
        destination,
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    loaded = load_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
        destination,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_BRIDGE,
        expected_start_provider_artifact_sha256=start.artifact_sha256,
    )
    assert loaded.metadata() == provider.metadata()
    with pytest.raises(FileExistsError, match="overwrite"):
        save_autonomous_complete_h4_fisher_finite_joint_pedal_provider(
            provider,
            destination,
        )


def test_hashes_are_deterministic_and_bind_evidence(fitted) -> None:
    _sequences, _parent, start, _initialized, provider = fitted
    repeated = refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        direction_left=provider.direction_left,
        direction_right=provider.direction_right,
        pedal_weight=provider.pedal_weight,
        pedal_bias=provider.pedal_bias,
        fit_protocol_sha256=_PROTOCOL,
        fit_evidence_sha256=_EVIDENCE,
    )
    assert repeated.artifact_sha256 == provider.artifact_sha256
    assert repeated.fit_receipt_sha256 == provider.fit_receipt_sha256
    changed_evidence = refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        direction_left=provider.direction_left,
        direction_right=provider.direction_right,
        pedal_weight=provider.pedal_weight,
        pedal_bias=provider.pedal_bias,
        fit_protocol_sha256=_PROTOCOL,
        fit_evidence_sha256="9" * 64,
    )
    assert changed_evidence.artifact_sha256 != provider.artifact_sha256
    assert changed_evidence.fit_receipt_sha256 != provider.fit_receipt_sha256
    for name in (
        "router_weight",
        "router_bias",
        "coordinate_scales",
        "direction_left",
        "direction_right",
        "pedal_weight",
        "pedal_bias",
    ):
        torch.testing.assert_close(
            getattr(changed_evidence, name),
            getattr(provider, name),
            rtol=0.0,
            atol=0.0,
        )


def test_control_tensor_invariants_are_fail_closed(fitted) -> None:
    _sequences, _parent, _start, _initialized, conditional = fitted
    intercept = fisher_finite_joint_pedal_control(
        conditional,
        pedal_mode="intercept",
    )
    unit = fisher_finite_joint_pedal_control(conditional, pedal_mode="unit")
    with pytest.raises(ValueError, match="intercept pedal slopes"):
        replace(
            intercept,
            pedal_weight=torch.ones_like(intercept.pedal_weight),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="unit pedal tensors"):
        replace(
            unit,
            pedal_bias=torch.ones_like(unit.pedal_bias),
            artifact_sha256="",
        )
