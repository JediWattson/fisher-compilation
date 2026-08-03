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
    fisher_xy_pointwise_bounded_direction,
    fisher_xy_pedal_features,
    fit_autonomous_complete_h4_fisher_xy_pedal,
)
from fisher_graph.complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    initialize_autonomous_complete_h4_fisher_finite_joint_pedal,
    refit_autonomous_complete_h4_fisher_finite_joint_pedal,
)
from fisher_graph.complete_h4_fisher_soft_polarity import (
    FISHER_SOFT_POLARITY_ETA_COUNT,
    FISHER_SOFT_POLARITY_ETA_MAX_ABS,
    FISHER_SOFT_POLARITY_FEATURE_NAMES,
    AutonomousCompleteH4FisherSoftPolarityProvider,
    build_autonomous_complete_h4_fisher_soft_polarity,
    build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control,
    build_autonomous_complete_h4_fisher_soft_polarity_zero_control,
    fisher_soft_polarity_box_certificate,
    fisher_soft_polarity_constant_tensor_sha256s,
    fisher_soft_polarity_envelope,
    fisher_soft_polarity_features,
    fisher_soft_polarity_gain,
    fisher_soft_polarity_modal_terms,
    fisher_soft_polarity_value,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
)


_BRIDGE = "d" * 64
_V19_PROTOCOL = "a" * 64
_V20F_PROTOCOL = "4" * 64
_V20F_EVIDENCE = "5" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 8) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(94100 + index)
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
        example_id=f"soft-polarity-example-{index}",
        family_id=f"soft-polarity-family-{index // 2}",
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


def test_exact_frozen_formula_and_feature_order() -> None:
    coordinates = torch.tensor(
        [[-0.8, 0.25], [-0.2, -0.6], [0.0, 0.4], [0.7, -0.1]],
        dtype=torch.float64,
    )
    eta = torch.tensor((0.3, -0.4, 0.2, 0.5), dtype=torch.float64)
    features = fisher_soft_polarity_features(coordinates)
    expected_features = torch.stack(
        (
            torch.ones(4, dtype=torch.float64),
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 0] * coordinates[:, 1],
        ),
        dim=1,
    )
    expected_envelope = (
        torch.asinh(9.0 * coordinates[:, 1]) / math.asinh(9.0)
    )
    expected_polarity = torch.tanh(expected_features @ eta)
    torch.testing.assert_close(features, expected_features, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        fisher_soft_polarity_envelope(coordinates),
        expected_envelope,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fisher_soft_polarity_value(coordinates, eta),
        expected_polarity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fisher_soft_polarity_gain(coordinates, eta),
        expected_envelope * expected_polarity,
        rtol=0.0,
        atol=0.0,
    )
    assert FISHER_SOFT_POLARITY_FEATURE_NAMES == (
        "one",
        "c1",
        "c2",
        "c1_times_c2",
    )


def test_pure_formula_ignores_rebound_public_semantic_aliases(monkeypatch) -> None:
    import fisher_graph.complete_h4_fisher_soft_polarity as module

    coordinates = torch.tensor(
        [[-0.6, -0.4], [0.2, 0.5], [0.8, 0.9]], dtype=torch.float64
    )
    eta = torch.tensor((0.3, -0.2, 0.1, 0.4), dtype=torch.float64)
    expected_envelope = fisher_soft_polarity_envelope(coordinates)
    expected_gain = fisher_soft_polarity_gain(coordinates, eta)
    expected_certificate = fisher_soft_polarity_box_certificate(eta)

    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_ETA_COUNT", 3)
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_ETA_MAX_ABS", 0.01)
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_FEATURE_NAMES", ("drift",))
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_SIGNED_LOG_KAPPA", 1.0)
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_TRUST_FRACTION", 0.5)

    torch.testing.assert_close(
        fisher_soft_polarity_envelope(coordinates),
        expected_envelope,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fisher_soft_polarity_gain(coordinates, eta),
        expected_gain,
        rtol=0.0,
        atol=0.0,
    )
    assert fisher_soft_polarity_box_certificate(eta) == expected_certificate


def test_zero_and_controlled_constant_polarity_are_exact() -> None:
    coordinates = torch.tensor(
        [[-0.6, -0.8], [0.1, -0.2], [0.4, 0.3], [0.9, 0.75]],
        dtype=torch.float64,
    )
    zero = fisher_soft_polarity_gain(
        coordinates, torch.zeros(4, dtype=torch.float64)
    )
    torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0.0, atol=0.0)

    target = 0.5
    eta = torch.tensor((math.atanh(target), 0.0, 0.0, 0.0), dtype=torch.float64)
    expected = target * fisher_soft_polarity_envelope(coordinates)
    torch.testing.assert_close(
        fisher_soft_polarity_gain(coordinates, eta),
        expected,
        rtol=2.0e-16,
        atol=2.0e-16,
    )


def test_analytic_box_certificate_corners_grid_and_random_large_eta() -> None:
    eta = torch.tensor((40.0, -31.0, 28.0, 36.0), dtype=torch.float64)
    corners = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    grid = torch.linspace(-1.0, 1.0, 81, dtype=torch.float64)
    c1, c2 = torch.meshgrid(grid, grid, indexing="ij")
    dense = torch.stack((c1.reshape(-1), c2.reshape(-1)), dim=1)
    generator = torch.Generator().manual_seed(7801)
    random = 2.0 * torch.rand(
        10000, 2, generator=generator, dtype=torch.float64
    ) - 1.0
    for coordinates in (corners, dense, random):
        gain = fisher_soft_polarity_gain(coordinates, eta)
        assert bool(torch.isfinite(gain).all())
        assert float(gain.abs().max()) <= 1.0
    certificate = fisher_soft_polarity_box_certificate(eta)
    assert certificate["gain_max_abs"] == 1.0
    assert certificate["proof"] == (
        "abs_normalized_asinh_at_most_one_times_abs_tanh_at_most_one"
    )


def test_eta_numerical_domain_prevents_overflow_cancellation() -> None:
    corners = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    accepted = torch.full(
        (4,), -FISHER_SOFT_POLARITY_ETA_MAX_ABS, dtype=torch.float64
    )
    value = fisher_soft_polarity_value(corners, accepted)
    assert bool(torch.isfinite(value).all())
    certificate = fisher_soft_polarity_box_certificate(accepted)
    assert certificate["eta_count"] == 4
    assert certificate["eta_max_abs"] == FISHER_SOFT_POLARITY_ETA_MAX_ABS
    assert certificate["router_projection_max_abs_upper_bound"] < (
        torch.finfo(torch.float64).max
    )

    rejected = torch.full(
        (4,), -2.0 * FISHER_SOFT_POLARITY_ETA_MAX_ABS, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="numerical magnitude limit"):
        fisher_soft_polarity_value(corners, rejected)
    with pytest.raises(ValueError, match="numerical magnitude limit"):
        fisher_soft_polarity_box_certificate(rejected)


def test_formula_preserves_coordinate_and_eta_gradients() -> None:
    coordinates = torch.tensor(
        [[-0.5, 0.3], [0.2, -0.4], [0.65, 0.55]],
        dtype=torch.float64,
        requires_grad=True,
    )
    eta = torch.tensor(
        (0.2, -0.35, 0.18, 0.27), dtype=torch.float64, requires_grad=True
    )
    gain = fisher_soft_polarity_gain(coordinates, eta)
    gain.square().sum().backward()
    assert coordinates.grad is not None and eta.grad is not None
    assert bool(torch.isfinite(coordinates.grad).all())
    assert bool(torch.isfinite(eta.grad).all())
    assert float(coordinates.grad.abs().sum()) > 0.0
    assert float(eta.grad.abs().sum()) > 0.0

    gradcheck_coordinates = coordinates.detach().clone().requires_grad_(True)
    gradcheck_eta = eta.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        fisher_soft_polarity_gain,
        (gradcheck_coordinates, gradcheck_eta),
        eps=1.0e-6,
        atol=2.0e-5,
        rtol=2.0e-4,
    )


def test_modal_terms_preserve_eta_gradient_and_pointwise_trust() -> None:
    generator = torch.Generator().manual_seed(991)
    parent = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    coordinates = 0.8 * torch.tanh(
        torch.randn(7, 2, generator=generator, dtype=torch.float64)
    )
    left0 = 1.0e3 * torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = 1.0e3 * torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = -0.7 * left0
    right1 = 1.1 * right0
    eta = torch.tensor(
        (0.1, -0.3, 0.25, 0.2), dtype=torch.float64, requires_grad=True
    )
    gain, _q, bounded, _logit, _pedal, delta = fisher_soft_polarity_modal_terms(
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
        eta,
    )
    parent_norm = torch.linalg.vector_norm(parent, dim=1)
    tolerance = 1.0e-13 * torch.maximum(parent_norm, torch.ones_like(parent_norm))
    assert bool(
        (torch.linalg.vector_norm(bounded, dim=1) <= 0.25 * parent_norm + tolerance).all()
    )
    assert bool(
        (torch.linalg.vector_norm(delta, dim=1) <= 0.25 * parent_norm + tolerance).all()
    )
    assert bool((gain.abs() <= 1.0).all())
    delta.square().mean().backward()
    assert eta.grad is not None
    assert bool(torch.isfinite(eta.grad).all())
    assert float(eta.grad.abs().sum()) > 0.0


def test_zero_provider_is_exact_base_endpoint_and_controls_are_exact(endpoints) -> None:
    base, proposal = endpoints
    zero = build_autonomous_complete_h4_fisher_soft_polarity_zero_control(
        base,
        proposal,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256=_V20F_EVIDENCE,
    )
    positive = build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
        base,
        proposal,
        polarity=1,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256="6" * 64,
    )
    negative = build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
        base,
        proposal,
        polarity=-1,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256="7" * 64,
    )
    parent = torch.tensor(
        [[0.4, -0.2, 0.1, 0.3], [-0.1, 0.5, -0.3, 0.2]],
        dtype=torch.float64,
    )
    coordinates = base.bounded_coordinates(parent)
    terms = zero.terms_from_parent(parent, coordinates)
    base_direction = base.unbounded_direction(parent, coordinates)
    base_bounded = fisher_xy_pointwise_bounded_direction(
        parent,
        base_direction,
        trust_fraction=base.trust_fraction,
    )
    base_pedal = base.pedal_values(coordinates)
    base_delta = base_pedal.unsqueeze(1) * base_bounded
    assert torch.equal(terms[0], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(terms[1], base_direction)
    assert torch.equal(terms[2], base_bounded)
    assert torch.equal(terms[4], base_pedal)
    assert torch.equal(terms[5], base_delta)
    envelope = fisher_soft_polarity_envelope(coordinates)
    torch.testing.assert_close(positive.response_gain(coordinates), envelope)
    torch.testing.assert_close(negative.response_gain(coordinates), -envelope)
    assert isinstance(zero, Gemma3L3L4CorrectionProvider)
    assert isinstance(zero, AutonomousCompleteH4FisherSoftPolarityProvider)


def test_provider_hash_constants_forbidden_inputs_and_resource_accounting(
    endpoints,
) -> None:
    base, proposal = endpoints
    source_eta = torch.tensor((0.4, -0.2, 0.1, 0.3), dtype=torch.float32)
    provider = build_autonomous_complete_h4_fisher_soft_polarity(
        base,
        proposal,
        eta=source_eta,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256=_V20F_EVIDENCE,
    )
    source_eta[0] = 99.0
    assert provider.eta.dtype == torch.float64
    assert float(provider.eta[0]) == pytest.approx(0.4)
    metadata = provider.metadata()
    assert metadata["base_provider_artifact_sha256"] == base.artifact_sha256
    assert metadata["proposal_provider_artifact_sha256"] == proposal.artifact_sha256
    assert metadata["eta_float64_scalar_count"] == 4
    assert metadata["soft_polarity_fitted_float_scalar_count"] == 4
    assert metadata["constant_tensor_sha256s"] == (
        fisher_soft_polarity_constant_tensor_sha256s()
    )
    assert metadata["runtime_forbidden_inputs"] == (
        "native_h4",
        "targets",
        "logits",
        "gradients",
        "family_ids",
        "prompt_text",
        "token_ids",
        "fit_examples",
        "optimizer_state",
    )
    expected_incremental_macs = (
        2 * _RANK + 4 + 10 * _RANK * base.conditional_rank + 6
    )
    assert (
        metadata["incremental_logical_macs_per_token_upper_bound"]
        == expected_incremental_macs
    )
    assert metadata["soft_polarity_router_dot_macs_per_token"] == 4
    assert metadata["soft_polarity_elementwise_scalar_arithmetic_per_token"] == 4
    assert metadata["soft_polarity_nonlinear_scalar_ops_per_token"] == 2
    assert metadata["routing_control_flow"] == "none_validation_guards_only"
    assert metadata["global_gain_certificate"] == "absolute_gain_at_most_one"

    tampered = copy.deepcopy(provider)
    tampered.eta[0] += 0.01
    with pytest.raises(RuntimeError, match="payload drifted"):
        tampered.validate_integrity()


def test_provider_construction_ignores_rebound_public_semantic_aliases(
    endpoints,
    monkeypatch,
) -> None:
    import fisher_graph.complete_h4_fisher_soft_polarity as module

    base, proposal = endpoints
    eta = torch.tensor((0.4, -0.2, 0.1, 0.3), dtype=torch.float64)
    expected = build_autonomous_complete_h4_fisher_soft_polarity(
        base,
        proposal,
        eta=eta,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256=_V20F_EVIDENCE,
    )
    coordinates = torch.tensor(
        [[-0.5, -0.4], [0.25, 0.6]], dtype=torch.float64
    )
    expected_gain = expected.response_gain(coordinates)

    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_ETA_COUNT", 3)
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_ETA_MAX_ABS", 0.01)
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_FEATURE_NAMES", ("drift",))
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_SIGNED_LOG_KAPPA", 1.0)
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_TRUST_FRACTION", 0.5)
    observed = build_autonomous_complete_h4_fisher_soft_polarity(
        base,
        proposal,
        eta=eta,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256=_V20F_EVIDENCE,
    )

    assert observed.artifact_sha256 == expected.artifact_sha256
    metadata = observed.metadata()
    assert metadata["eta_float64_scalar_count"] == 4
    assert metadata["eta_max_abs"] == FISHER_SOFT_POLARITY_ETA_MAX_ABS
    assert metadata["feature_names"] == (
        "one",
        "c1",
        "c2",
        "c1_times_c2",
    )
    assert metadata["signed_log_kappa"] == 9.0
    assert metadata["trust_fraction"] == 0.25
    torch.testing.assert_close(
        observed.response_gain(coordinates), expected_gain, rtol=0.0, atol=0.0
    )
    zero = build_autonomous_complete_h4_fisher_soft_polarity_zero_control(
        base,
        proposal,
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256="8" * 64,
    )
    assert zero.eta.shape == (4,)
    with pytest.raises(ValueError, match="frozen at 0.25"):
        AutonomousCompleteH4FisherSoftPolarityProvider(
            base_provider=base,
            proposal_provider=proposal,
            eta=eta,
            transfer_protocol_sha256=_V20F_PROTOCOL,
            transfer_evidence_sha256="9" * 64,
            trust_fraction=0.5,
        )


@pytest.mark.parametrize(
    ("coordinates", "eta", "message"),
    (
        (torch.zeros(2, 3, dtype=torch.float64), torch.zeros(4), r"shape \[N,2\]"),
        (torch.tensor([[1.01, 0.0]]), torch.zeros(4), r"inside \[-1,1\]"),
        (torch.tensor([[0.0, float("nan")]]), torch.zeros(4), "finite"),
        (torch.zeros(2, 2), torch.zeros(3), "exactly four"),
        (torch.zeros(2, 2), torch.tensor([0.0, 0.0, 0.0, float("inf")]), "finite"),
    ),
)
def test_shape_bound_and_finite_fail_closed(coordinates, eta, message) -> None:
    with pytest.raises(ValueError, match=message):
        fisher_soft_polarity_gain(coordinates, eta)


def test_provider_rejects_bad_eta_hashes_and_control_polarity(endpoints) -> None:
    base, proposal = endpoints
    with pytest.raises(ValueError, match="exactly four"):
        build_autonomous_complete_h4_fisher_soft_polarity(
            base,
            proposal,
            eta=torch.zeros(3, dtype=torch.float64),
            transfer_protocol_sha256=_V20F_PROTOCOL,
            transfer_evidence_sha256=_V20F_EVIDENCE,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        build_autonomous_complete_h4_fisher_soft_polarity(
            base,
            proposal,
            eta=torch.zeros(4, dtype=torch.float64),
            transfer_protocol_sha256="not-a-hash",
            transfer_evidence_sha256=_V20F_EVIDENCE,
        )
    with pytest.raises(ValueError, match="exactly -1 or 1"):
        build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
            base,
            proposal,
            polarity=0,
            transfer_protocol_sha256=_V20F_PROTOCOL,
            transfer_evidence_sha256=_V20F_EVIDENCE,
        )


def test_provider_supports_batched_leading_dimensions(endpoints) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity(
        base,
        proposal,
        eta=torch.tensor((0.2, -0.3, 0.1, 0.25), dtype=torch.float64),
        transfer_protocol_sha256=_V20F_PROTOCOL,
        transfer_evidence_sha256=_V20F_EVIDENCE,
    )
    parent = torch.linspace(-0.7, 0.8, 24, dtype=torch.float64).reshape(2, 3, 4)
    coordinates = base.bounded_coordinates(parent)
    gain = provider.response_gain(coordinates)
    terms = provider.terms_from_parent(parent, coordinates)
    assert gain.shape == (2, 3)
    assert terms[0].shape == (2, 3)
    assert terms[1].shape == parent.shape
    logit = provider.pedal_logits(coordinates)
    assert logit.shape == (2, 3)
    flat_features = fisher_xy_pedal_features(coordinates.reshape(-1, 2))
    flat_gain = gain.reshape(-1)
    expected_logit = (
        flat_features @ base.pedal_weight
        + base.pedal_bias[0]
        + flat_gain
        * (
            flat_features @ (proposal.pedal_weight - base.pedal_weight)
            + proposal.pedal_bias[0]
            - base.pedal_bias[0]
        )
    )
    assert expected_logit.shape == (6,)
    torch.testing.assert_close(terms[0], gain, rtol=0.0, atol=0.0)
    torch.testing.assert_close(logit.reshape(-1), expected_logit)
    torch.testing.assert_close(terms[3].reshape(-1), expected_logit)
