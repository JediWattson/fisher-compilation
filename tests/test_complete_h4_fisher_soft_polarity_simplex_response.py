from __future__ import annotations

import copy
import json

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
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    initialize_autonomous_complete_h4_fisher_finite_joint_pedal,
    refit_autonomous_complete_h4_fisher_finite_joint_pedal,
)
from fisher_graph.complete_h4_fisher_soft_polarity import (
    fisher_soft_polarity_envelope,
    fisher_soft_polarity_features,
)
from fisher_graph.complete_h4_fisher_soft_polarity_confidence import (
    fisher_soft_polarity_confidence_gain,
)
from fisher_graph.complete_h4_fisher_soft_polarity_log_response import (
    fisher_soft_polarity_log_response_gain,
)
from fisher_graph.complete_h4_fisher_soft_polarity_signed_stack import (
    fisher_soft_polarity_signed_stack_gain,
)
from fisher_graph.complete_h4_fisher_soft_polarity_simplex_response import (
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX,
    FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX,
    AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_response,
    fisher_soft_polarity_simplex_response_box_certificate,
    fisher_soft_polarity_simplex_response_calibrator,
    fisher_soft_polarity_simplex_response_constant_tensor_sha256s,
    fisher_soft_polarity_simplex_response_direction_sha256,
    fisher_soft_polarity_simplex_response_gain,
    fisher_soft_polarity_simplex_response_modal_terms,
    fisher_soft_polarity_simplex_response_projection,
    fisher_soft_polarity_simplex_response_provider_artifact_sha256,
    fisher_soft_polarity_simplex_response_value,
    normalize_fisher_soft_polarity_simplex_response_direction,
    validate_fisher_soft_polarity_simplex_response_provider_evidence,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
)


_BRIDGE = "d" * 64
_V19_PROTOCOL = "a" * 64
_PROTOCOL = "e" * 64
_EVIDENCE = "f" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 8) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(97600 + index)
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
        example_id=f"simplex-response-example-{index}",
        family_id=f"simplex-response-family-{index // 2}",
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


def _raw_direction() -> torch.Tensor:
    return torch.tensor((0.35, -0.55, 0.20, 0.45), dtype=torch.float64)


def _normalized_direction() -> torch.Tensor:
    return normalize_fisher_soft_polarity_simplex_response_direction(
        _raw_direction()
    )


def _provider_evidence(endpoints):
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=_normalized_direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    return provider, provider.artifact_payload(), provider.metadata()


def test_direction_normalization_certifies_box_projection() -> None:
    direction = _normalized_direction()
    corners = torch.tensor(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
        dtype=torch.float64,
    )
    projection = fisher_soft_polarity_simplex_response_projection(
        corners, direction
    )
    assert float(projection.abs().max()) == pytest.approx(1.0, abs=1.0e-15)
    assert fisher_soft_polarity_simplex_response_direction_sha256(direction)
    with pytest.raises(ValueError, match="box-corner normalized"):
        fisher_soft_polarity_simplex_response_direction_sha256(_raw_direction())


@pytest.mark.parametrize(
    ("radius", "u", "v"),
    (
        (0.0, 0.0, 0.0),
        (0.125, 0.125, -0.125),
        (0.125, 0.125, -0.0625),
        (0.125, 0.125, 0.0),
        (0.25, 0.25, 0.125),
        (1.5, 0.5, 0.5),
    ),
)
def test_exact_formula_and_simplex_bound(radius: float, u: float, v: float) -> None:
    generator = torch.Generator().manual_seed(8119)
    coordinates = 2.0 * torch.rand(
        20000, 2, generator=generator, dtype=torch.float64
    ) - 1.0
    direction = _normalized_direction()
    z = fisher_soft_polarity_features(coordinates) @ direction
    expected_value = (1.0 - u * z.square()) * torch.tanh(radius * z) + v * z.square()
    expected_gain = fisher_soft_polarity_envelope(coordinates) * expected_value
    torch.testing.assert_close(
        fisher_soft_polarity_simplex_response_value(
            coordinates, direction, radius, u, v
        ),
        expected_value,
        rtol=0.0,
        atol=0.0,
    )
    observed = fisher_soft_polarity_simplex_response_gain(
        coordinates, direction, radius, u, v
    )
    torch.testing.assert_close(observed, expected_gain, rtol=0.0, atol=0.0)
    assert float(observed.abs().max()) <= 1.0
    assert fisher_soft_polarity_simplex_response_calibrator(
        torch.zeros(1, dtype=torch.float64), radius, u, v
    ).item() == 0.0


def test_simplex_weights_are_nonnegative_and_sum_to_one() -> None:
    z = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)
    for u, v in ((0.125, -0.125), (0.125, -0.0625), (0.25, 0.0), (0.5, 0.5)):
        z2 = z.square()
        w0 = 1.0 - u * z2
        w_plus = 0.5 * (u + v) * z2
        w_minus = 0.5 * (u - v) * z2
        assert bool((w0 >= 0.0).all())
        assert bool((w_plus >= 0.0).all())
        assert bool((w_minus >= 0.0).all())
        torch.testing.assert_close(
            w0 + w_plus + w_minus,
            torch.ones_like(z),
            rtol=0.0,
            atol=torch.finfo(torch.float64).eps,
        )


def test_oddness_only_when_polarity_bias_is_zero() -> None:
    z = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    odd_positive = fisher_soft_polarity_simplex_response_calibrator(
        z, 0.25, 0.25, 0.0
    )
    odd_negative = fisher_soft_polarity_simplex_response_calibrator(
        -z, 0.25, 0.25, 0.0
    )
    torch.testing.assert_close(odd_negative, -odd_positive, rtol=0.0, atol=0.0)
    biased_positive = fisher_soft_polarity_simplex_response_calibrator(
        z, 0.25, 0.25, 0.125
    )
    biased_negative = fisher_soft_polarity_simplex_response_calibrator(
        -z, 0.25, 0.25, 0.125
    )
    assert not torch.equal(biased_negative, -biased_positive)


def test_linear_and_v20l_boundary_identities_are_bit_exact() -> None:
    coordinates = torch.tensor(
        ((-0.8, 0.25), (-0.2, -0.6), (0.0, 0.4), (0.7, -0.1)),
        dtype=torch.float64,
    )
    direction = _normalized_direction()
    radius = torch.tensor(0.25, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    linear = fisher_soft_polarity_simplex_response_gain(
        coordinates, direction, radius, zero, zero
    )
    assert torch.equal(
        linear,
        fisher_soft_polarity_confidence_gain(
            coordinates, direction, radius, zero
        ),
    )
    assert torch.equal(
        linear,
        fisher_soft_polarity_log_response_gain(
            coordinates, direction, radius, zero
        ),
    )
    for signed_mix in (-0.125, 0.125):
        simplex = fisher_soft_polarity_simplex_response_gain(
            coordinates,
            direction,
            radius,
            abs(signed_mix),
            signed_mix,
        )
        stack = fisher_soft_polarity_signed_stack_gain(
            coordinates, direction, radius, signed_mix
        )
        assert torch.equal(simplex, stack)


def test_certificate_pins_convexity_and_shape_claims() -> None:
    certificate = fisher_soft_polarity_simplex_response_box_certificate(
        _normalized_direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
    )
    assert certificate["projection_max_abs"] == 1.0
    assert certificate["simplex_weights_nonnegative"] is True
    assert certificate["simplex_weights_sum_to_one"] is True
    assert certificate["base_weight_min_lower_bound"] == 0.75
    assert certificate["calibrator_center_value"] == 0.0
    assert certificate["calibrator_odd_when_polarity_bias_zero"] is True
    assert certificate["calibrator_oddness_claim_when_polarity_bias_nonzero"] == "none"
    assert certificate["calibrator_monotonicity_claim"] == "none"
    assert certificate["gain_max_abs"] == 1.0
    assert certificate["pointwise_trust_fraction"] == 0.25


def test_modal_terms_preserve_gradients_and_pointwise_trust() -> None:
    generator = torch.Generator().manual_seed(2991)
    parent = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    coordinates = 0.8 * torch.tanh(
        torch.randn(7, 2, generator=generator, dtype=torch.float64)
    )
    left0 = 1.0e3 * torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = 1.0e3 * torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = -0.7 * left0
    right1 = 1.1 * right0
    raw_direction = _raw_direction().requires_grad_(True)
    direction = normalize_fisher_soft_polarity_simplex_response_direction(
        raw_direction
    )
    radius = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    shrink_mass = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    polarity_bias = torch.tensor(-0.125, dtype=torch.float64, requires_grad=True)
    gain, _q, bounded, _logit, _pedal, delta = (
        fisher_soft_polarity_simplex_response_modal_terms(
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
            direction,
            radius,
            shrink_mass,
            polarity_bias,
        )
    )
    parent_norm = torch.linalg.vector_norm(parent, dim=1)
    tolerance = 1.0e-13 * torch.maximum(parent_norm, torch.ones_like(parent_norm))
    assert bool(
        (
            torch.linalg.vector_norm(bounded, dim=1)
            <= 0.25 * parent_norm + tolerance
        ).all()
    )
    assert bool(
        (
            torch.linalg.vector_norm(delta, dim=1)
            <= 0.25 * parent_norm + tolerance
        ).all()
    )
    assert bool((gain.abs() <= 1.0).all())
    delta.square().mean().backward()
    for gradient in (
        raw_direction.grad,
        radius.grad,
        shrink_mass.grad,
        polarity_bias.grad,
    ):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())


def test_provider_accounting_artifact_and_mutation_safety(endpoints) -> None:
    base, proposal = endpoints
    source_direction = _normalized_direction()
    provider = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=source_direction,
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    original = provider.direction.clone()
    source_direction[0] = 99.0
    assert torch.equal(provider.direction, original)
    assert provider.radius == 0.25
    assert provider.shrink_mass == 0.25
    assert provider.polarity_bias == -0.125
    assert isinstance(provider, Gemma3L3L4CorrectionProvider)
    assert isinstance(
        provider, AutonomousCompleteH4FisherSoftPolaritySimplexResponseProvider
    )
    metadata = provider.metadata()
    assert metadata["base_provider_artifact_sha256"] == base.artifact_sha256
    assert metadata["proposal_provider_artifact_sha256"] == proposal.artifact_sha256
    assert metadata["direction_float64_scalar_count"] == 4
    assert metadata["response_float64_scalar_count"] == 3
    assert metadata["fitted_float64_scalar_count"] == 7
    assert metadata["simplex_response_fitted_float_scalar_count"] == 7
    assert metadata["constant_tensor_sha256s"] == (
        fisher_soft_polarity_simplex_response_constant_tensor_sha256s()
    )
    expected_scalars = (
        base.incremental_prepared_float_scalar_count
        + proposal.incremental_prepared_float_scalar_count
        + 7
    )
    assert provider.incremental_prepared_float_scalar_count == expected_scalars
    expected_macs = 2 * _RANK + 4 + 10 * _RANK * base.conditional_rank + 6
    assert provider.incremental_logical_macs_per_token_upper_bound == expected_macs
    assert metadata["simplex_response_calibrator_scalar_arithmetic_per_token"] == 7
    assert metadata["simplex_response_elementwise_scalar_arithmetic_per_token"] == 10
    assert metadata["simplex_response_nonlinear_scalar_ops_per_token"] == 2
    assert metadata["experimental_serving_status"].startswith("analysis_only")

    equivalent = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=original.clone(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert equivalent.artifact_sha256 == provider.artifact_sha256
    changed = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=original.clone(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=0.0,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert changed.artifact_sha256 != provider.artifact_sha256

    tampered = copy.deepcopy(provider)
    object.__setattr__(tampered, "polarity_bias", 0.0)
    with pytest.raises(RuntimeError, match="payload drifted"):
        tampered.validate_integrity()
    tampered_direction = copy.deepcopy(provider)
    tampered_direction.direction[0] += 0.01
    with pytest.raises((ValueError, RuntimeError)):
        tampered_direction.validate_integrity()


def test_provider_evidence_roundtrips_model_free(endpoints) -> None:
    provider, payload, metadata = _provider_evidence(endpoints)
    validated = validate_fisher_soft_polarity_simplex_response_provider_evidence(
        payload, metadata
    )
    assert validated.artifact_sha256 == provider.artifact_sha256
    assert (
        fisher_soft_polarity_simplex_response_provider_artifact_sha256(payload)
        == provider.artifact_sha256
    )
    serialized = json.loads(json.dumps({"payload": payload, "metadata": metadata}))
    replayed = validate_fisher_soft_polarity_simplex_response_provider_evidence(
        serialized["payload"], serialized["metadata"]
    )
    assert replayed == validated

    def contains_tensor(value: object) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(payload)
    assert not contains_tensor(metadata)
    payload["constant_tensor_sha256s"]["box_corner_features"] = "0" * 64
    assert provider.artifact_payload()["constant_tensor_sha256s"] == (
        fisher_soft_polarity_simplex_response_constant_tensor_sha256s()
    )


def test_evidence_rejects_payload_and_metadata_tampering(endpoints) -> None:
    _provider, payload, metadata = _provider_evidence(endpoints)
    payload_attacks = []
    stale_radius = copy.deepcopy(payload)
    stale_radius["radius"] = 0.125
    payload_attacks.append(stale_radius)
    stale_u = copy.deepcopy(payload)
    stale_u["shrink_mass"] = 0.125
    payload_attacks.append(stale_u)
    invalid_pair = copy.deepcopy(payload)
    invalid_pair["polarity_bias"] = -0.3
    payload_attacks.append(invalid_pair)
    formula = copy.deepcopy(payload)
    formula["gain_formula"] = "different"
    payload_attacks.append(formula)
    serving = copy.deepcopy(payload)
    serving["experimental_serving_status"] = "serving_ready"
    payload_attacks.append(serving)
    for attacked in payload_attacks:
        with pytest.raises(ValueError):
            fisher_soft_polarity_simplex_response_provider_artifact_sha256(attacked)
    missing = copy.deepcopy(payload)
    missing.pop("polarity_bias")
    with pytest.raises(ValueError, match="key set differs"):
        fisher_soft_polarity_simplex_response_provider_artifact_sha256(missing)

    metadata_attacks = []
    wrong_hash = copy.deepcopy(metadata)
    wrong_hash["artifact_sha256"] = "0" * 64
    metadata_attacks.append(wrong_hash)
    wrong_count = copy.deepcopy(metadata)
    wrong_count["incremental_prepared_float_scalar_count"] += 1
    wrong_count["incremental_runtime_parameter_bytes_float64"] += 8
    metadata_attacks.append(wrong_count)
    wrong_macs = copy.deepcopy(metadata)
    wrong_macs["incremental_logical_macs_per_token_upper_bound"] += 1
    metadata_attacks.append(wrong_macs)
    wrong_certificate = copy.deepcopy(metadata)
    wrong_certificate["box_certificate"]["simplex_weights_sum_to_one"] = False
    metadata_attacks.append(wrong_certificate)
    false_oddness = copy.deepcopy(metadata)
    false_oddness["box_certificate"][
        "calibrator_oddness_claim_when_polarity_bias_nonzero"
    ] = "odd"
    metadata_attacks.append(false_oddness)
    for attacked in metadata_attacks:
        with pytest.raises(ValueError):
            validate_fisher_soft_polarity_simplex_response_provider_evidence(
                payload, attacked
            )


def test_provider_supports_batched_leading_dimensions(endpoints) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=_normalized_direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=0.125,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    parent = torch.linspace(-0.7, 0.8, 24, dtype=torch.float64).reshape(2, 3, 4)
    coordinates = base.bounded_coordinates(parent)
    terms = provider.terms_from_parent(parent, coordinates)
    assert provider.response_gain(coordinates).shape == (2, 3)
    assert terms[0].shape == (2, 3)
    assert terms[1].shape == parent.shape
    assert terms[3].shape == (2, 3)
    assert provider.pedal_logits(coordinates).shape == (2, 3)


@pytest.mark.parametrize(
    ("radius", "u", "v", "message"),
    (
        (-0.1, 0.0, 0.0, "nonnegative"),
        (0.25, -0.1, 0.0, r"inside \[0.0,0.5\]"),
        (0.25, 0.51, 0.0, r"inside \[0.0,0.5\]"),
        (0.25, 0.125, 0.126, r"abs\(polarity_bias\) <= shrink_mass"),
        (0.25, 0.125, -0.126, r"abs\(polarity_bias\) <= shrink_mass"),
        (True, 0.0, 0.0, "floating scalar"),
        (0.25, True, 0.0, "floating scalar"),
        (0.25, 0.0, True, "floating scalar"),
        (float("nan"), 0.0, 0.0, "finite"),
        (0.25, float("inf"), 0.0, "finite"),
        (0.25, 0.0, float("nan"), "finite"),
        (-0.0, 0.0, 0.0, "signed negative zero"),
        (0.25, -0.0, 0.0, "signed negative zero"),
        (0.25, 0.0, -0.0, "signed negative zero"),
    ),
)
def test_provider_parameter_validation_fails_closed(
    endpoints, radius, u, v, message
) -> None:
    base, proposal = endpoints
    with pytest.raises(ValueError, match=message):
        build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
            base,
            proposal,
            direction=_normalized_direction(),
            radius=radius,
            shrink_mass=u,
            polarity_bias=v,
            transfer_protocol_sha256=_PROTOCOL,
            transfer_evidence_sha256=_EVIDENCE,
        )


@pytest.mark.parametrize(
    ("u", "v"),
    ((0.0, 0.0), (0.125, -0.125), (0.125, 0.0), (0.25, 0.125), (0.5, -0.5)),
)
def test_provider_accepts_closed_simplex_parameter_domain(endpoints, u, v) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=_normalized_direction(),
        radius=0.25,
        shrink_mass=u,
        polarity_bias=v,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert provider.shrink_mass == u
    assert provider.polarity_bias == v


def test_exported_constants_are_pinned() -> None:
    assert FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_DIRECTION_COUNT == 4
    assert FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_FITTED_SCALAR_COUNT == 7
    assert FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_SHRINK_MASS_MAX == 0.5
    assert FISHER_SOFT_POLARITY_SIMPLEX_RESPONSE_RADIUS_MAX > 1.0e300
