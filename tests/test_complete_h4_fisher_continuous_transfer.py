from __future__ import annotations

import copy
import math

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4TrainingSequence,
    fit_autonomous_complete_h4_residual,
)
from fisher_graph.complete_h4_fisher_conditional_pedal import (
    fit_autonomous_complete_h4_fisher_xy_pedal,
)
from fisher_graph.complete_h4_fisher_continuous_transfer import (
    AutonomousCompleteH4FisherContinuousTransferProvider,
    build_autonomous_complete_h4_fisher_continuous_axis_response,
    build_autonomous_complete_h4_fisher_continuous_constant_control,
    build_autonomous_complete_h4_fisher_continuous_transfer,
    fisher_continuous_bilinear_box_max_abs,
    fisher_continuous_bilinear_corner_values,
    fisher_continuous_factor_direction,
    fisher_continuous_pedal_logit,
    fisher_continuous_response_features,
    fisher_continuous_response_gain,
    fisher_continuous_transfer_modal_terms,
)
from fisher_graph.complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    fisher_finite_joint_direction_features,
    initialize_autonomous_complete_h4_fisher_finite_joint_pedal,
    refit_autonomous_complete_h4_fisher_finite_joint_pedal,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
)
from fisher_graph.gemma3_l3_l4_complete_h4_finite_joint_pedal_development import (
    _held_runtime_diagnostics,
)


_BRIDGE = "d" * 64
_V19_PROTOCOL = "a" * 64
_V20C_PROTOCOL = "1" * 64
_V20C_EVIDENCE = "2" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 8) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(81200 + index)
    source = torch.randn(
        length, _SOURCE_RANK, generator=generator, dtype=torch.float64
    )
    base = torch.randn(length, _WIDTH, generator=generator, dtype=torch.float64)
    parent = 0.15 * base[:, :_RANK]
    raw = torch.stack(
        (
            parent[:, 0] + 0.2 * parent[:, 2],
            -0.8 * parent[:, 1] + 0.25 * parent[:, 3],
        ),
        dim=1,
    )
    coordinates = raw / (0.5 + raw.abs())
    direction = torch.stack(
        (
            0.20 * coordinates[:, 0] * parent[:, 0],
            -0.16 * coordinates[:, 1] * parent[:, 1],
            0.13 * coordinates[:, 0] * parent[:, 2]
            + 0.05 * coordinates.prod(dim=1) * parent[:, 0],
            0.11 * coordinates[:, 1] * parent[:, 3],
        ),
        dim=1,
    )
    pedal = torch.sigmoid(
        0.2
        + 0.45 * coordinates[:, 0]
        - 0.35 * coordinates[:, 1]
        + 0.15 * coordinates.prod(dim=1)
    )
    native = base + (parent + pedal.unsqueeze(1) * direction) @ _decoder()
    gradients = torch.zeros_like(base)
    gradients[:, :_RANK] = parent
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"continuous-transfer-example-{index}",
        family_id=f"continuous-transfer-family-{index // 2}",
        source_modes=source,
        logical_positions=torch.arange(length, dtype=torch.int64),
        valid_mask=mask,
        source_mask=mask,
        support_mask=mask,
        base_h4=base,
        native_h4=native,
        reverse_vjp_gradients=gradients,
    )


@pytest.fixture(scope="module")
def endpoints() -> tuple[
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
]:
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
    base_provider = initialize_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        fit_protocol_sha256=_V19_PROTOCOL,
        fit_evidence_sha256="b" * 64,
    )
    proposal_provider = refit_autonomous_complete_h4_fisher_finite_joint_pedal(
        start,
        direction_left=base_provider.direction_left
        + torch.linspace(
            -0.02,
            0.025,
            base_provider.direction_left.numel(),
            dtype=torch.float64,
        ).reshape_as(base_provider.direction_left),
        direction_right=base_provider.direction_right
        + torch.linspace(
            0.018,
            -0.012,
            base_provider.direction_right.numel(),
            dtype=torch.float64,
        ).reshape_as(base_provider.direction_right),
        pedal_weight=torch.tensor((0.75, -0.45, 0.28), dtype=torch.float64),
        pedal_bias=torch.tensor((0.18,), dtype=torch.float64),
        fit_protocol_sha256=_V19_PROTOCOL,
        fit_evidence_sha256="c" * 64,
    )
    return base_provider, proposal_provider


def test_raw_axis_linear_signed_log_and_exact_mirrors() -> None:
    coordinates = torch.tensor(
        [[-0.8, 0.25], [-0.2, -0.6], [0.0, 0.4], [0.7, -0.1]],
        dtype=torch.float64,
    )
    first_axis = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
    linear = fisher_continuous_response_gain(
        coordinates,
        first_axis,
        response_source="direct",
        response_law="linear",
        polarity=1,
    )
    log_gain = fisher_continuous_response_gain(
        coordinates,
        first_axis,
        response_source="direct",
        response_law="signed_log",
        polarity=1,
    )
    mirror = fisher_continuous_response_gain(
        coordinates,
        first_axis,
        response_source="direct",
        response_law="signed_log",
        polarity=-1,
    )
    torch.testing.assert_close(linear, coordinates[:, 0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        log_gain,
        torch.asinh(9.0 * coordinates[:, 0]) / math.asinh(9.0),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(mirror, -log_gain, rtol=0.0, atol=0.0)
    assert not torch.equal(linear, log_gain)


def test_bilinear_direct_box_certificate_is_exact_and_global() -> None:
    weight = torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64)
    assert fisher_continuous_bilinear_corner_values(weight) == pytest.approx(
        (0.5, -0.9, 0.1, 0.3), rel=0.0, abs=1.0e-15
    )
    assert fisher_continuous_bilinear_box_max_abs(weight) == pytest.approx(0.9)

    grid = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    c1, c2 = torch.meshgrid(grid, grid, indexing="ij")
    values = weight[0] * c1 + weight[1] * c2 + weight[2] * c1 * c2
    assert float(values.abs().max()) <= fisher_continuous_bilinear_box_max_abs(
        weight
    )


def test_response_and_factor_terms_preserve_autograd() -> None:
    generator = torch.Generator().manual_seed(9181)
    coordinates = (
        0.7
        * torch.tanh(torch.randn(5, 2, generator=generator, dtype=torch.float64))
    ).requires_grad_()
    response_weight = torch.tensor(
        (0.35, -0.22, 0.11), dtype=torch.float64, requires_grad=True
    )
    gain = fisher_continuous_response_gain(
        coordinates,
        response_weight,
        response_source="tanh_projection",
        response_law="signed_log",
        polarity=1,
    )
    features = torch.randn(5, 12, generator=generator, dtype=torch.float64)
    left0 = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = left0 + 0.1
    right1 = right0 - 0.08
    direction = fisher_continuous_factor_direction(
        features, left0, right0, left1, right1, gain
    )
    direction.square().mean().backward()
    assert response_weight.grad is not None
    assert coordinates.grad is not None
    assert bool(torch.isfinite(response_weight.grad).all())
    assert float(response_weight.grad.abs().sum()) > 0.0


def test_row_factor_expansion_matches_explicit_interpolation() -> None:
    generator = torch.Generator().manual_seed(1128)
    features = torch.randn(6, 12, generator=generator, dtype=torch.float64)
    left0 = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = left0 + torch.randn(
        12, 3, generator=generator, dtype=torch.float64
    ) * 0.1
    right1 = right0 + torch.randn(
        3, 4, generator=generator, dtype=torch.float64
    ) * 0.1
    gains = torch.tensor((-1.0, -0.4, 0.0, 0.25, 0.8, 1.0), dtype=torch.float64)
    actual = fisher_continuous_factor_direction(
        features, left0, right0, left1, right1, gains
    )
    expected = torch.stack(
        tuple(
            (features[row : row + 1] @ (left0 + alpha * (left1 - left0)))
            @ (right0 + alpha * (right1 - right0))
            for row, alpha in enumerate(gains)
        )
    ).squeeze(1)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(actual[-1], expected[-1], rtol=2e-14, atol=2e-14)


def test_pedal_logit_is_exact_row_interpolation() -> None:
    coordinates = torch.tensor(
        [[-0.3, 0.4], [0.2, -0.5], [0.7, 0.1]], dtype=torch.float64
    )
    weight0 = torch.tensor((0.2, -0.1, 0.3), dtype=torch.float64)
    bias0 = torch.tensor((-0.15,), dtype=torch.float64)
    weight1 = torch.tensor((0.8, -0.4, 0.1), dtype=torch.float64)
    bias1 = torch.tensor((0.25,), dtype=torch.float64)
    gains = torch.tensor((-1.0, 0.35, 1.0), dtype=torch.float64)
    actual = fisher_continuous_pedal_logit(
        coordinates, weight0, bias0, weight1, bias1, gains
    )
    features = fisher_continuous_response_features(coordinates)
    expected = torch.stack(
        tuple(
            features[row]
            @ (weight0 + gains[row] * (weight1 - weight0))
            + bias0[0]
            + gains[row] * (bias1[0] - bias0[0])
            for row in range(coordinates.shape[0])
        )
    )
    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)


def test_constant_controls_match_endpoint_and_negative_microstep(endpoints) -> None:
    base, proposal = endpoints
    positive = build_autonomous_complete_h4_fisher_continuous_constant_control(
        base,
        proposal,
        alpha=1,
        transfer_protocol_sha256=_V20C_PROTOCOL,
        transfer_evidence_sha256=_V20C_EVIDENCE,
    )
    negative = build_autonomous_complete_h4_fisher_continuous_constant_control(
        base,
        proposal,
        alpha=-1,
        transfer_protocol_sha256=_V20C_PROTOCOL,
        transfer_evidence_sha256="3" * 64,
    )
    parent = torch.tensor(
        [
            [0.4, -0.2, 0.1, 0.3],
            [-0.1, 0.5, -0.3, 0.2],
            [0.2, 0.1, 0.4, -0.5],
        ],
        dtype=torch.float64,
    )
    coordinates = base.bounded_coordinates(parent)
    positive_terms = positive.terms_from_parent(parent, coordinates)
    negative_terms = negative.terms_from_parent(parent, coordinates)
    torch.testing.assert_close(
        positive_terms[0], torch.ones(3, dtype=torch.float64), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        negative_terms[0], -torch.ones(3, dtype=torch.float64), rtol=0.0, atol=0.0
    )
    expected_positive_direction = proposal.unbounded_direction(parent, coordinates)
    torch.testing.assert_close(
        positive_terms[1], expected_positive_direction, rtol=2e-13, atol=2e-13
    )
    expected_negative_features = fisher_finite_joint_direction_features(
        parent, coordinates
    )
    expected_negative_direction = (
        expected_negative_features
        @ (2.0 * base.direction_left - proposal.direction_left)
    ) @ (2.0 * base.direction_right - proposal.direction_right)
    torch.testing.assert_close(
        negative_terms[1], expected_negative_direction, rtol=2e-13, atol=2e-13
    )
    pedal_features = fisher_continuous_response_features(coordinates)
    proposal_logit = (
        pedal_features @ proposal.pedal_weight + proposal.pedal_bias[0]
    )
    torch.testing.assert_close(
        positive_terms[3], proposal_logit, rtol=2e-13, atol=2e-13
    )
    torch.testing.assert_close(
        positive.pedal_logits(coordinates), proposal_logit, rtol=2e-13, atol=2e-13
    )
    torch.testing.assert_close(
        positive.pedal_values(coordinates),
        torch.sigmoid(proposal_logit),
        rtol=2e-13,
        atol=2e-13,
    )


def test_axis_builders_and_pointwise_trust_certificate(endpoints) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_continuous_axis_response(
        base,
        proposal,
        coordinate_index=1,
        response_law="signed_log",
        polarity=1,
        transfer_protocol_sha256=_V20C_PROTOCOL,
        transfer_evidence_sha256=_V20C_EVIDENCE,
    )
    assert isinstance(provider, Gemma3L3L4CorrectionProvider)
    assert isinstance(provider, AutonomousCompleteH4FisherContinuousTransferProvider)
    parent = torch.tensor(
        [
            [1.0e-4, -2.0e-4, 3.0e-4, 4.0e-4],
            [5.0, -4.0, 3.0, -2.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    coordinates = base.bounded_coordinates(parent)
    gain, _direction, bounded, _logit, pedal, delta = provider.terms_from_parent(
        parent, coordinates
    )
    expected_gain = torch.asinh(9.0 * coordinates[:, 1]) / math.asinh(9.0)
    torch.testing.assert_close(gain, expected_gain, rtol=0.0, atol=0.0)
    parent_norm = torch.linalg.vector_norm(parent, dim=1)
    bounded_norm = torch.linalg.vector_norm(bounded, dim=1)
    delta_norm = torch.linalg.vector_norm(delta, dim=1)
    tolerance = 1.0e-14 * torch.maximum(parent_norm, torch.ones_like(parent_norm))
    assert bool((bounded_norm <= 0.25 * parent_norm + tolerance).all())
    assert bool((delta_norm <= 0.25 * parent_norm + tolerance).all())
    assert torch.equal(bounded[2], torch.zeros(_RANK, dtype=torch.float64))
    assert torch.equal(delta[2], torch.zeros(_RANK, dtype=torch.float64))
    assert bool(((pedal >= 0.0) & (pedal <= 1.0)).all())

    # The live V20c smoke deliberately reuses the authenticated V19 runtime
    # diagnostic.  Keep this ABI integration under test, especially the
    # coordinate-only ``pedal_values`` call.
    diagnostic = _held_runtime_diagnostics(provider, (_sequence(0),))
    assert diagnostic["pointwise_trust_passed"] is True
    assert diagnostic["provider_artifact_sha256"] == provider.artifact_sha256


def test_pure_modal_terms_enforce_trust_under_large_factors() -> None:
    generator = torch.Generator().manual_seed(419)
    parent = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    coordinates = 0.7 * torch.tanh(
        torch.randn(7, 2, generator=generator, dtype=torch.float64)
    )
    left0 = 1.0e4 * torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = 1.0e4 * torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = -0.8 * left0
    right1 = 1.2 * right0
    gain, _q, bounded, _logit, _pedal, delta = (
        fisher_continuous_transfer_modal_terms(
            parent,
            coordinates,
            left0,
            right0,
            left1,
            right1,
            torch.zeros(3, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64),
            response_source="direct",
            response_law="linear",
            polarity=1,
        )
    )
    assert bool((gain.abs() < 1.0).all())
    parent_norm = torch.linalg.vector_norm(parent, dim=1)
    assert bool(
        (
            torch.linalg.vector_norm(bounded, dim=1)
            <= 0.25 * parent_norm + 1.0e-13
        ).all()
    )
    assert bool(
        (
            torch.linalg.vector_norm(delta, dim=1)
            <= 0.25 * parent_norm + 1.0e-13
        ).all()
    )


def test_provider_hash_lineage_forbidden_inputs_and_resource_accounting(
    endpoints,
) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_continuous_transfer(
        base,
        proposal,
        response_law="linear",
        response_source="tanh_projection",
        response_weight=torch.tensor((0.4, -0.2, 0.1), dtype=torch.float64),
        polarity=1,
        transfer_protocol_sha256=_V20C_PROTOCOL,
        transfer_evidence_sha256=_V20C_EVIDENCE,
    )
    metadata = provider.metadata()
    assert metadata["base_provider_artifact_sha256"] == base.artifact_sha256
    assert metadata["proposal_provider_artifact_sha256"] == proposal.artifact_sha256
    assert metadata["runtime_forbidden_inputs"] == (
        "native_h4",
        "targets",
        "logits",
        "gradients",
        "family_ids",
        "fit_examples",
        "optimizer_state",
    )
    expected_incremental_macs = (
        2 * _RANK + 3 + 10 * _RANK * base.conditional_rank + 6
    )
    assert (
        metadata["incremental_logical_macs_per_token_upper_bound"]
        == expected_incremental_macs
    )
    assert "two_endpoints" in metadata["experimental_serving_status"]
    assert "both_endpoint_factor_paths" in metadata["logical_macs_accounting_scope"]
    tampered = copy.deepcopy(provider)
    tampered.response_weight[0] += 0.01
    with pytest.raises(RuntimeError, match="payload drifted"):
        tampered.validate_integrity()


def test_invalid_direct_projection_and_incompatible_lineage_are_rejected(
    endpoints,
) -> None:
    base, proposal = endpoints
    coordinates = torch.tensor([[0.8, 0.7]], dtype=torch.float64)
    with pytest.raises(ValueError, match="escaped"):
        fisher_continuous_response_gain(
            coordinates,
            torch.tensor((1.0, 1.0, 0.0), dtype=torch.float64),
            response_source="direct",
            response_law="linear",
            polarity=1,
        )
    with pytest.raises(ValueError, match=r"global \[-1,1\] box certificate"):
        build_autonomous_complete_h4_fisher_continuous_transfer(
            base,
            proposal,
            response_law="linear",
            response_source="direct",
            response_weight=torch.tensor((1.0, 1.0, 0.0), dtype=torch.float64),
            polarity=1,
            transfer_protocol_sha256=_V20C_PROTOCOL,
            transfer_evidence_sha256=_V20C_EVIDENCE,
        )
    wrong = copy.deepcopy(proposal)
    wrong.router_bias[0] += 0.01
    with pytest.raises(RuntimeError, match="payload drifted"):
        build_autonomous_complete_h4_fisher_continuous_constant_control(
            base,
            wrong,
            alpha=1,
            transfer_protocol_sha256=_V20C_PROTOCOL,
            transfer_evidence_sha256=_V20C_EVIDENCE,
        )
