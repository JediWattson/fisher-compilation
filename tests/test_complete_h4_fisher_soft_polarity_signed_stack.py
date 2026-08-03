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
    FISHER_SOFT_POLARITY_SIGNED_STACK_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_SIGNED_STACK_FITTED_SCALAR_COUNT,
    FISHER_SOFT_POLARITY_SIGNED_STACK_RADIUS_MAX,
    FISHER_SOFT_POLARITY_SIGNED_STACK_SIGNED_MIX_MAX_ABS,
    AutonomousCompleteH4FisherSoftPolaritySignedStackProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_signed_stack,
    fisher_soft_polarity_signed_stack_box_certificate,
    fisher_soft_polarity_signed_stack_calibrator,
    fisher_soft_polarity_signed_stack_constant_tensor_sha256s,
    fisher_soft_polarity_signed_stack_direction_sha256,
    fisher_soft_polarity_signed_stack_gain,
    fisher_soft_polarity_signed_stack_modal_terms,
    fisher_soft_polarity_signed_stack_projection,
    fisher_soft_polarity_signed_stack_provider_artifact_sha256,
    fisher_soft_polarity_signed_stack_value,
    normalize_fisher_soft_polarity_signed_stack_direction,
    validate_fisher_soft_polarity_signed_stack_provider_evidence,
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
    generator = torch.Generator().manual_seed(96400 + index)
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
        example_id=f"signed-stack-example-{index}",
        family_id=f"signed-stack-family-{index // 2}",
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
    return normalize_fisher_soft_polarity_signed_stack_direction(_raw_direction())


def _provider_evidence(endpoints):
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
        base,
        proposal,
        direction=_normalized_direction(),
        radius=0.9,
        signed_mix=-0.3,
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
    projection = fisher_soft_polarity_signed_stack_projection(corners, direction)
    assert float(projection.abs().max()) == pytest.approx(1.0, abs=1.0e-15)
    assert fisher_soft_polarity_signed_stack_direction_sha256(direction)
    with pytest.raises(ValueError, match="box-corner normalized"):
        fisher_soft_polarity_signed_stack_direction_sha256(_raw_direction())


@pytest.mark.parametrize("signed_mix", (-0.5, -0.2, 0.0, 0.2, 0.5))
def test_exact_formula_center_and_analytic_bound(signed_mix: float) -> None:
    generator = torch.Generator().manual_seed(7781)
    coordinates = 2.0 * torch.rand(
        20000, 2, generator=generator, dtype=torch.float64
    ) - 1.0
    direction = _normalized_direction()
    radius = torch.tensor(1.4, dtype=torch.float64)
    mix = torch.tensor(signed_mix, dtype=torch.float64)
    z = fisher_soft_polarity_features(coordinates) @ direction
    z_squared = z.square()
    m = torch.tanh(radius * z)
    w = torch.abs(mix) * z_squared
    expected_value = (1.0 - w) * m + mix * z_squared
    expected_gain = fisher_soft_polarity_envelope(coordinates) * expected_value

    torch.testing.assert_close(
        fisher_soft_polarity_signed_stack_value(
            coordinates, direction, radius, mix
        ),
        expected_value,
        rtol=0.0,
        atol=0.0,
    )
    observed = fisher_soft_polarity_signed_stack_gain(
        coordinates, direction, radius, mix
    )
    torch.testing.assert_close(observed, expected_gain, rtol=0.0, atol=0.0)
    assert float(observed.abs().max()) <= 1.0
    center = fisher_soft_polarity_signed_stack_calibrator(
        torch.zeros(1, dtype=torch.float64), radius, mix
    )
    assert torch.equal(center, torch.zeros_like(center))


def test_signed_stack_has_positive_and_negative_fixed_polarity_bias() -> None:
    z = torch.tensor((-1.0, -0.8, 0.8, 1.0), dtype=torch.float64)
    base = fisher_soft_polarity_signed_stack_calibrator(z, 1.2, 0.0)
    positive = fisher_soft_polarity_signed_stack_calibrator(z, 1.2, 0.4)
    negative = fisher_soft_polarity_signed_stack_calibrator(z, 1.2, -0.4)
    assert bool((positive > base).all())
    assert bool((negative < base).all())
    # Nonzero stacking is intentionally not odd.
    assert not torch.equal(positive, -torch.flip(positive, dims=(0,)))


def test_certificate_proves_convex_bound_without_nonzero_mix_shape_claims() -> None:
    certificate = fisher_soft_polarity_signed_stack_box_certificate(
        _normalized_direction(), radius=3.0, signed_mix=-0.5
    )
    assert certificate["projection_max_abs"] == 1.0
    assert certificate["stack_weight_nonnegative"] is True
    assert certificate["stack_weight_max_upper_bound"] == 0.5
    assert certificate["calibrator_center_value"] == 0.0
    assert certificate["calibrator_odd_when_signed_mix_zero"] is True
    assert certificate["calibrator_oddness_claim_when_signed_mix_nonzero"] == "none"
    assert (
        certificate["calibrator_monotonicity_claim_when_signed_mix_nonzero"]
        == "none"
    )
    assert certificate["calibrator_max_abs_upper_bound"] == 1.0
    assert certificate["gain_max_abs"] == 1.0
    assert certificate["pointwise_trust_fraction"] == 0.25


def test_signed_mix_zero_is_bit_exact_linear_v20j_and_v20k() -> None:
    coordinates = torch.tensor(
        ((-0.8, 0.25), (-0.2, -0.6), (0.0, 0.4), (0.7, -0.1)),
        dtype=torch.float64,
    )
    direction = _normalized_direction()
    radius = torch.tensor(1.75, dtype=torch.float64)
    zero = torch.tensor(0.0, dtype=torch.float64)
    signed = fisher_soft_polarity_signed_stack_gain(
        coordinates, direction, radius, zero
    )
    v20j = fisher_soft_polarity_confidence_gain(
        coordinates, direction, radius, zero
    )
    v20k = fisher_soft_polarity_log_response_gain(
        coordinates, direction, radius, zero
    )
    assert torch.equal(signed, v20j)
    assert torch.equal(signed, v20k)


def test_modal_terms_preserve_gradients_and_pointwise_trust() -> None:
    generator = torch.Generator().manual_seed(1779)
    parent = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    coordinates = 0.8 * torch.tanh(
        torch.randn(7, 2, generator=generator, dtype=torch.float64)
    )
    left0 = 1.0e3 * torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = 1.0e3 * torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = -0.7 * left0
    right1 = 1.1 * right0
    raw_direction = _raw_direction().requires_grad_(True)
    direction = normalize_fisher_soft_polarity_signed_stack_direction(raw_direction)
    radius = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    signed_mix = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    gain, _q, bounded, _logit, _pedal, delta = (
        fisher_soft_polarity_signed_stack_modal_terms(
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
            signed_mix,
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
    for gradient in (raw_direction.grad, radius.grad, signed_mix.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())


def test_provider_artifact_accounting_and_mutation_safety(endpoints) -> None:
    base, proposal = endpoints
    source_direction = _normalized_direction()
    provider = build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
        base,
        proposal,
        direction=source_direction,
        radius=0.9,
        signed_mix=-0.3,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    original_direction = provider.direction.clone()
    source_direction[0] = 99.0
    assert torch.equal(provider.direction, original_direction)
    assert provider.radius == pytest.approx(0.9)
    assert provider.signed_mix == pytest.approx(-0.3)
    assert isinstance(provider, Gemma3L3L4CorrectionProvider)
    assert isinstance(
        provider, AutonomousCompleteH4FisherSoftPolaritySignedStackProvider
    )

    metadata = provider.metadata()
    assert metadata["base_provider_artifact_sha256"] == base.artifact_sha256
    assert metadata["proposal_provider_artifact_sha256"] == proposal.artifact_sha256
    assert metadata["direction_float64_scalar_count"] == 4
    assert metadata["response_float64_scalar_count"] == 2
    assert metadata["fitted_float64_scalar_count"] == 6
    assert metadata["signed_stack_polarity_fitted_float_scalar_count"] == 6
    assert metadata["constant_tensor_sha256s"] == (
        fisher_soft_polarity_signed_stack_constant_tensor_sha256s()
    )
    expected_incremental_scalars = (
        base.incremental_prepared_float_scalar_count
        + proposal.incremental_prepared_float_scalar_count
        + 6
    )
    assert provider.incremental_prepared_float_scalar_count == (
        expected_incremental_scalars
    )
    expected_incremental_macs = (
        2 * _RANK + 4 + 10 * _RANK * base.conditional_rank + 6
    )
    assert provider.incremental_logical_macs_per_token_upper_bound == (
        expected_incremental_macs
    )
    assert metadata["signed_stack_projection_dot_macs_per_token"] == 4
    assert metadata["signed_stack_calibrator_scalar_arithmetic_per_token"] == 8
    assert metadata["signed_stack_elementwise_scalar_arithmetic_per_token"] == 11
    assert metadata["signed_stack_nonlinear_scalar_ops_per_token"] == 2
    assert metadata["routing_control_flow"] == "none_validation_guards_only"
    assert metadata["global_gain_certificate"] == "absolute_gain_at_most_one"
    assert metadata["experimental_serving_status"].startswith("analysis_only")

    equivalent = build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
        base,
        proposal,
        direction=original_direction.clone(),
        radius=0.9,
        signed_mix=-0.3,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert equivalent.artifact_sha256 == provider.artifact_sha256
    changed = build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
        base,
        proposal,
        direction=original_direction.clone(),
        radius=0.9,
        signed_mix=-0.31,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert changed.artifact_sha256 != provider.artifact_sha256

    tampered_direction = copy.deepcopy(provider)
    tampered_direction.direction[0] += 0.01
    with pytest.raises((ValueError, RuntimeError), match="direction|payload drifted"):
        tampered_direction.validate_integrity()
    tampered_mix = copy.deepcopy(provider)
    object.__setattr__(tampered_mix, "signed_mix", -0.31)
    with pytest.raises(RuntimeError, match="payload drifted"):
        tampered_mix.validate_integrity()


def test_provider_evidence_codec_roundtrips_without_model_state(endpoints) -> None:
    provider, payload, metadata = _provider_evidence(endpoints)
    validated = validate_fisher_soft_polarity_signed_stack_provider_evidence(
        payload, metadata
    )
    assert validated.artifact_sha256 == provider.artifact_sha256
    assert (
        fisher_soft_polarity_signed_stack_provider_artifact_sha256(payload)
        == provider.artifact_sha256
    )
    assert validated.payload["runtime_inputs"] == [
        "one_pass_prefix",
        "realized_pre_correction_h4",
    ]
    serialized = json.loads(json.dumps({"payload": payload, "metadata": metadata}))
    replayed = validate_fisher_soft_polarity_signed_stack_provider_evidence(
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
        fisher_soft_polarity_signed_stack_constant_tensor_sha256s()
    )


@pytest.mark.parametrize("field", ("radius", "signed_mix"))
@pytest.mark.parametrize(
    "invalid",
    (True, "0.25", float("nan"), float("inf"), -0.0),
)
def test_provider_evidence_rejects_noncanonical_response_scalars(
    endpoints,
    field: str,
    invalid: object,
) -> None:
    _provider, payload, _metadata = _provider_evidence(endpoints)
    payload[field] = invalid
    with pytest.raises(ValueError):
        fisher_soft_polarity_signed_stack_provider_artifact_sha256(payload)


@pytest.mark.parametrize("valid", (-0.5, -0.499, -0.1, 0.0, 0.5))
def test_provider_accepts_signed_mix_across_closed_interval(endpoints, valid) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
        base,
        proposal,
        direction=_normalized_direction(),
        radius=0.7,
        signed_mix=valid,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert provider.signed_mix == valid


@pytest.mark.parametrize("invalid", (-0.5000001, 0.5000001, -2.0, 2.0))
def test_provider_rejects_signed_mix_off_range(endpoints, invalid: float) -> None:
    base, proposal = endpoints
    with pytest.raises(ValueError, match=r"inside \[-0.5,0.5\]"):
        build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
            base,
            proposal,
            direction=_normalized_direction(),
            radius=0.7,
            signed_mix=invalid,
            transfer_protocol_sha256=_PROTOCOL,
            transfer_evidence_sha256=_EVIDENCE,
        )


def test_provider_evidence_rejects_payload_and_metadata_tampering(endpoints) -> None:
    _provider, payload, metadata = _provider_evidence(endpoints)

    payload_attacks = []
    stale_radius = copy.deepcopy(payload)
    stale_radius["radius"] = 0.91
    payload_attacks.append(stale_radius)
    stale_mix = copy.deepcopy(payload)
    stale_mix["signed_mix"] = -0.31
    payload_attacks.append(stale_mix)
    changed_formula = copy.deepcopy(payload)
    changed_formula["gain_formula"] = "different"
    payload_attacks.append(changed_formula)
    false_serving = copy.deepcopy(payload)
    false_serving["experimental_serving_status"] = "serving_ready"
    payload_attacks.append(false_serving)
    raw_tensor = copy.deepcopy(payload)
    raw_tensor["direction_sha256"] = torch.zeros(4)
    payload_attacks.append(raw_tensor)
    for attacked in payload_attacks:
        with pytest.raises(ValueError):
            fisher_soft_polarity_signed_stack_provider_artifact_sha256(attacked)

    missing_payload = copy.deepcopy(payload)
    missing_payload.pop("gain_formula")
    with pytest.raises(ValueError, match="key set differs"):
        fisher_soft_polarity_signed_stack_provider_artifact_sha256(missing_payload)

    metadata_attacks = []
    wrong_artifact = copy.deepcopy(metadata)
    wrong_artifact["artifact_sha256"] = "0" * 64
    metadata_attacks.append(wrong_artifact)
    wrong_prepared = copy.deepcopy(metadata)
    wrong_prepared["incremental_prepared_float_scalar_count"] += 1
    wrong_prepared["incremental_runtime_parameter_bytes_float64"] += 8
    metadata_attacks.append(wrong_prepared)
    wrong_macs = copy.deepcopy(metadata)
    wrong_macs["incremental_logical_macs_per_token_upper_bound"] += 1
    metadata_attacks.append(wrong_macs)
    wrong_certificate = copy.deepcopy(metadata)
    wrong_certificate["box_certificate"]["signed_mix"] = -0.2
    metadata_attacks.append(wrong_certificate)
    wrong_shape_claim = copy.deepcopy(metadata)
    wrong_shape_claim["box_certificate"][
        "calibrator_oddness_claim_when_signed_mix_nonzero"
    ] = "odd"
    metadata_attacks.append(wrong_shape_claim)
    for attacked in metadata_attacks:
        with pytest.raises(ValueError):
            validate_fisher_soft_polarity_signed_stack_provider_evidence(
                payload, attacked
            )

    missing_metadata = copy.deepcopy(metadata)
    missing_metadata.pop("rank")
    with pytest.raises(ValueError, match="key set differs"):
        validate_fisher_soft_polarity_signed_stack_provider_evidence(
            payload, missing_metadata
        )


def test_provider_supports_batched_leading_dimensions(endpoints) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
        base,
        proposal,
        direction=_normalized_direction(),
        radius=0.7,
        signed_mix=0.4,
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
    ("direction", "radius", "signed_mix", "message"),
    (
        (torch.zeros(4), 1.0, 0.0, "nonzero"),
        (torch.zeros(3), 1.0, 0.0, "exactly four"),
        (_raw_direction(), 1.0, 0.0, "box-corner normalized"),
        (_normalized_direction(), -0.1, 0.0, "nonnegative"),
        (_normalized_direction(), 0.1, -0.51, r"inside \[-0.5,0.5\]"),
        (_normalized_direction(), 0.1, 0.51, r"inside \[-0.5,0.5\]"),
        (_normalized_direction(), True, 0.0, "floating scalar"),
        (_normalized_direction(), 0.1, True, "floating scalar"),
        (_normalized_direction(), float("nan"), 0.0, "finite"),
        (_normalized_direction(), 0.1, float("inf"), "finite"),
        (_normalized_direction(), -0.0, 0.0, "signed negative zero"),
        (_normalized_direction(), 0.1, -0.0, "signed negative zero"),
    ),
)
def test_provider_parameter_validation_fails_closed(
    endpoints, direction, radius, signed_mix, message
) -> None:
    base, proposal = endpoints
    with pytest.raises(ValueError, match=message):
        build_autonomous_complete_h4_fisher_soft_polarity_signed_stack(
            base,
            proposal,
            direction=direction,
            radius=radius,
            signed_mix=signed_mix,
            transfer_protocol_sha256=_PROTOCOL,
            transfer_evidence_sha256=_EVIDENCE,
        )


def test_pure_formula_rejects_bad_geometry_and_radius() -> None:
    coordinates = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="box-corner normalized"):
        fisher_soft_polarity_signed_stack_gain(
            coordinates, _raw_direction(), 1.0, 0.0
        )
    with pytest.raises(ValueError, match="numerical magnitude limit"):
        fisher_soft_polarity_signed_stack_gain(
            coordinates,
            _normalized_direction(),
            2.0 * FISHER_SOFT_POLARITY_SIGNED_STACK_RADIUS_MAX,
            0.0,
        )
    with pytest.raises(ValueError, match=r"inside \[-1,1\]"):
        fisher_soft_polarity_signed_stack_gain(
            torch.tensor(((0.0, 1.01),), dtype=torch.float64),
            _normalized_direction(),
            1.0,
            0.0,
        )


def test_exported_geometry_constants_are_pinned() -> None:
    assert FISHER_SOFT_POLARITY_SIGNED_STACK_DIRECTION_COUNT == 4
    assert FISHER_SOFT_POLARITY_SIGNED_STACK_FITTED_SCALAR_COUNT == 6
    assert FISHER_SOFT_POLARITY_SIGNED_STACK_SIGNED_MIX_MAX_ABS == 0.5
