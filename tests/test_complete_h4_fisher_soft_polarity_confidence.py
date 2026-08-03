from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4TrainingSequence,
    fit_autonomous_complete_h4_residual,
)
from fisher_graph.complete_h4_fisher_conditional_pedal import (
    fisher_xy_pointwise_bounded_direction,
    fit_autonomous_complete_h4_fisher_xy_pedal,
)
from fisher_graph.complete_h4_fisher_finite_joint_pedal import (
    AutonomousCompleteH4FisherFiniteJointPedalProvider,
    initialize_autonomous_complete_h4_fisher_finite_joint_pedal,
    refit_autonomous_complete_h4_fisher_finite_joint_pedal,
)
from fisher_graph.complete_h4_fisher_soft_polarity import (
    build_autonomous_complete_h4_fisher_soft_polarity,
    fisher_soft_polarity_envelope,
    fisher_soft_polarity_features,
    fisher_soft_polarity_gain,
)
from fisher_graph.complete_h4_fisher_soft_polarity_confidence import (
    FISHER_SOFT_POLARITY_CONFIDENCE_CALIBRATOR_MAX,
    FISHER_SOFT_POLARITY_CONFIDENCE_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_CONFIDENCE_FITTED_SCALAR_COUNT,
    AutonomousCompleteH4FisherSoftPolarityConfidenceProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_confidence,
    fisher_soft_polarity_confidence_box_certificate,
    fisher_soft_polarity_confidence_calibrator,
    fisher_soft_polarity_confidence_constant_tensor_sha256s,
    fisher_soft_polarity_confidence_direction_sha256,
    fisher_soft_polarity_confidence_gain,
    fisher_soft_polarity_confidence_modal_terms,
    fisher_soft_polarity_confidence_projection,
    fisher_soft_polarity_confidence_provider_artifact_sha256,
    fisher_soft_polarity_confidence_value,
    normalize_fisher_soft_polarity_confidence_direction,
    validate_fisher_soft_polarity_confidence_provider_evidence,
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
    generator = torch.Generator().manual_seed(95100 + index)
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
        example_id=f"confidence-polarity-example-{index}",
        family_id=f"confidence-polarity-family-{index // 2}",
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


def _raw_direction(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor((0.35, -0.55, 0.20, 0.45), dtype=dtype)


def _normalized_direction() -> torch.Tensor:
    return normalize_fisher_soft_polarity_confidence_direction(_raw_direction())


def _provider_evidence(endpoints):
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base,
        proposal,
        direction=_normalized_direction(),
        linear_coefficient=0.9,
        cubic_coefficient=1.4,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    return provider, provider.artifact_payload(), provider.metadata()


def test_direction_normalization_certifies_all_box_extrema() -> None:
    direction = _normalized_direction()
    corners = torch.tensor(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
        dtype=torch.float64,
    )
    corner_projection = fisher_soft_polarity_confidence_projection(
        corners, direction
    )
    assert float(corner_projection.abs().max()) == pytest.approx(1.0, abs=1.0e-15)

    grid = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    c1, c2 = torch.meshgrid(grid, grid, indexing="ij")
    dense = torch.stack((c1.reshape(-1), c2.reshape(-1)), dim=1)
    projection = fisher_soft_polarity_confidence_projection(dense, direction)
    assert float(projection.abs().max()) <= 1.0 + 1.0e-14

    scaled = normalize_fisher_soft_polarity_confidence_direction(
        7.0 * _raw_direction()
    )
    torch.testing.assert_close(scaled, direction, rtol=2.0e-16, atol=2.0e-16)


def test_public_direction_hash_matches_provider_payload(endpoints) -> None:
    provider, payload, _metadata = _provider_evidence(endpoints)
    assert fisher_soft_polarity_confidence_direction_sha256(
        provider.direction
    ) == payload["direction_sha256"]
    with pytest.raises(ValueError, match="box-corner normalized"):
        fisher_soft_polarity_confidence_direction_sha256(_raw_direction())


def test_exact_formula_and_analytic_gain_bound() -> None:
    generator = torch.Generator().manual_seed(6619)
    coordinates = 2.0 * torch.rand(
        20000, 2, generator=generator, dtype=torch.float64
    ) - 1.0
    direction = _normalized_direction()
    a = torch.tensor(1.4, dtype=torch.float64)
    b = torch.tensor(2.2, dtype=torch.float64)
    features = fisher_soft_polarity_features(coordinates)
    z = features @ direction
    expected_value = torch.tanh(a * z + b * z.square() * z)
    expected_gain = fisher_soft_polarity_envelope(coordinates) * expected_value

    torch.testing.assert_close(
        fisher_soft_polarity_confidence_projection(coordinates, direction),
        z,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fisher_soft_polarity_confidence_value(coordinates, direction, a, b),
        expected_value,
        rtol=0.0,
        atol=0.0,
    )
    observed = fisher_soft_polarity_confidence_gain(
        coordinates, direction, a, b
    )
    torch.testing.assert_close(observed, expected_gain, rtol=0.0, atol=0.0)
    assert bool(torch.isfinite(observed).all())
    assert float(observed.abs().max()) <= 1.0


@pytest.mark.parametrize(
    ("a_value", "b_value"),
    ((0.0, 0.0), (0.0, 2.0), (1.25, 0.0), (1.25, 2.0)),
)
def test_calibrator_is_odd_and_monotone(a_value: float, b_value: float) -> None:
    positive_z = torch.linspace(0.0, 1.0, 1001, dtype=torch.float64)
    negative_q = fisher_soft_polarity_confidence_calibrator(
        -positive_z,
        a_value,
        b_value,
    )
    positive_q = fisher_soft_polarity_confidence_calibrator(
        positive_z,
        a_value,
        b_value,
    )
    torch.testing.assert_close(negative_q, -positive_q, rtol=0.0, atol=0.0)

    z = torch.cat((-torch.flip(positive_z[1:], dims=(0,)), positive_z)).requires_grad_()
    a = torch.tensor(a_value, dtype=torch.float64)
    b = torch.tensor(b_value, dtype=torch.float64)
    q = fisher_soft_polarity_confidence_calibrator(z, a, b)
    assert q[1000] == 0.0
    assert bool((q[1:] >= q[:-1]).all())
    derivative = torch.autograd.grad(q, z, torch.ones_like(q))[0]
    assert bool(torch.isfinite(derivative).all())
    assert float(derivative.min()) >= -1.0e-15
    if a_value > 0.0 or b_value > 0.0:
        assert float(q[-1].detach()) > float(q[0].detach())
    else:
        assert torch.equal(q, torch.zeros_like(q))


def test_certificate_pins_bounds_oddness_monotonicity_and_trust() -> None:
    direction = _normalized_direction()
    a = torch.tensor(0.0, dtype=torch.float64)
    b = torch.tensor(3.0, dtype=torch.float64)
    certificate = fisher_soft_polarity_confidence_box_certificate(
        direction,
        linear_coefficient=float(a),
        cubic_coefficient=float(b),
    )
    assert certificate["projection_max_abs"] == 1.0
    assert certificate["calibrator_odd"] is True
    assert certificate["calibrator_monotone_nondecreasing"] is True
    assert certificate["calibrator_strictly_monotone"] is True
    assert certificate["calibrator_pre_tanh_derivative_min"] == 0.0
    assert certificate["calibrator_argument_max_abs_upper_bound"] == 3.0
    assert certificate["gain_max_abs"] == 1.0
    assert certificate["pointwise_trust_fraction"] == 0.25
    assert max(abs(item) for item in certificate["direction_box_corner_logits"]) == (
        pytest.approx(1.0)
    )


def test_b_zero_is_equivalent_to_existing_eta_a_times_d(endpoints) -> None:
    base, proposal = endpoints
    direction = _normalized_direction()
    a = torch.tensor(1.75, dtype=torch.float64)
    b = torch.tensor(0.0, dtype=torch.float64)
    eta = a * direction
    coordinates = torch.tensor(
        ((-0.8, 0.25), (-0.2, -0.6), (0.0, 0.4), (0.7, -0.1)),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        fisher_soft_polarity_confidence_gain(coordinates, direction, a, b),
        fisher_soft_polarity_gain(coordinates, eta),
        rtol=2.0e-16,
        atol=2.0e-16,
    )

    confidence = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base,
        proposal,
        direction=direction,
        linear_coefficient=float(a),
        cubic_coefficient=float(b),
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    linear = build_autonomous_complete_h4_fisher_soft_polarity(
        base,
        proposal,
        eta=eta,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    parent = torch.tensor(
        ((0.4, -0.2, 0.1, 0.3), (-0.1, 0.5, -0.3, 0.2)),
        dtype=torch.float64,
    )
    bounded = base.bounded_coordinates(parent)
    torch.testing.assert_close(
        confidence.response_gain(bounded),
        linear.response_gain(bounded),
        rtol=2.0e-16,
        atol=2.0e-16,
    )
    for confidence_term, linear_term in zip(
        confidence.terms_from_parent(parent, bounded),
        linear.terms_from_parent(parent, bounded),
        strict=True,
    ):
        torch.testing.assert_close(
            confidence_term, linear_term, rtol=5.0e-15, atol=5.0e-15
        )
    assert confidence.artifact_sha256 != linear.artifact_sha256


def test_modal_terms_preserve_gradients_and_pointwise_trust() -> None:
    generator = torch.Generator().manual_seed(1977)
    parent = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    coordinates = 0.8 * torch.tanh(
        torch.randn(7, 2, generator=generator, dtype=torch.float64)
    )
    left0 = 1.0e3 * torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right0 = 1.0e3 * torch.randn(3, 4, generator=generator, dtype=torch.float64)
    left1 = -0.7 * left0
    right1 = 1.1 * right0
    raw_direction = _raw_direction().requires_grad_(True)
    direction = normalize_fisher_soft_polarity_confidence_direction(raw_direction)
    a = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    gain, _q, bounded, _logit, _pedal, delta = (
        fisher_soft_polarity_confidence_modal_terms(
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
            a,
            b,
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
    for gradient in (raw_direction.grad, a.grad, b.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())


def test_provider_artifact_accounting_and_mutation_safety(endpoints) -> None:
    base, proposal = endpoints
    source_direction = _normalized_direction()
    provider = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base,
        proposal,
        direction=source_direction,
        linear_coefficient=0.9,
        cubic_coefficient=1.4,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    original_direction = provider.direction.clone()
    source_direction[0] = 99.0
    assert torch.equal(provider.direction, original_direction)
    assert provider.linear_coefficient == pytest.approx(0.9)
    assert provider.cubic_coefficient == pytest.approx(1.4)
    assert isinstance(provider, Gemma3L3L4CorrectionProvider)
    assert isinstance(
        provider, AutonomousCompleteH4FisherSoftPolarityConfidenceProvider
    )

    metadata = provider.metadata()
    assert metadata["base_provider_artifact_sha256"] == base.artifact_sha256
    assert metadata["proposal_provider_artifact_sha256"] == proposal.artifact_sha256
    assert metadata["direction_float64_scalar_count"] == 4
    assert metadata["calibrator_float64_scalar_count"] == 2
    assert metadata["fitted_float64_scalar_count"] == 6
    assert metadata["confidence_polarity_fitted_float_scalar_count"] == 6
    assert metadata["constant_tensor_sha256s"] == (
        fisher_soft_polarity_confidence_constant_tensor_sha256s()
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
    assert metadata["confidence_projection_dot_macs_per_token"] == 4
    assert metadata["confidence_calibrator_scalar_arithmetic_per_token"] == 5
    assert metadata["confidence_elementwise_scalar_arithmetic_per_token"] == 9
    assert metadata["confidence_nonlinear_scalar_ops_per_token"] == 2
    assert metadata["routing_control_flow"] == "none_validation_guards_only"
    assert metadata["global_gain_certificate"] == "absolute_gain_at_most_one"

    equivalent = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base,
        proposal,
        direction=original_direction.clone(),
        linear_coefficient=0.9,
        cubic_coefficient=1.4,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert equivalent.artifact_sha256 == provider.artifact_sha256
    changed_cubic = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base,
        proposal,
        direction=original_direction.clone(),
        linear_coefficient=0.9,
        cubic_coefficient=1.41,
        transfer_protocol_sha256=_PROTOCOL,
        transfer_evidence_sha256=_EVIDENCE,
    )
    assert changed_cubic.artifact_sha256 != provider.artifact_sha256

    tampered_direction = copy.deepcopy(provider)
    tampered_direction.direction[0] += 0.01
    with pytest.raises((ValueError, RuntimeError), match="direction|payload drifted"):
        tampered_direction.validate_integrity()
    tampered_b = copy.deepcopy(provider)
    object.__setattr__(tampered_b, "cubic_coefficient", 1.41)
    with pytest.raises(RuntimeError, match="payload drifted"):
        tampered_b.validate_integrity()


def test_provider_evidence_codec_roundtrips_without_model_state(endpoints) -> None:
    provider, payload, metadata = _provider_evidence(endpoints)
    validated = validate_fisher_soft_polarity_confidence_provider_evidence(
        payload,
        metadata,
    )
    assert validated.artifact_sha256 == provider.artifact_sha256
    assert (
        fisher_soft_polarity_confidence_provider_artifact_sha256(payload)
        == provider.artifact_sha256
    )
    assert validated.payload["runtime_inputs"] == [
        "one_pass_prefix",
        "realized_pre_correction_h4",
    ]
    assert validated.metadata["artifact_sha256"] == provider.artifact_sha256

    serialized = json.loads(json.dumps({"payload": payload, "metadata": metadata}))
    replayed = validate_fisher_soft_polarity_confidence_provider_evidence(
        serialized["payload"],
        serialized["metadata"],
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
        fisher_soft_polarity_confidence_constant_tensor_sha256s()
    )


@pytest.mark.parametrize("field", ("linear_coefficient", "cubic_coefficient"))
@pytest.mark.parametrize(
    "invalid",
    (
        True,
        "0.5",
        float("nan"),
        float("inf"),
        -0.1,
        -0.0,
    ),
)
def test_provider_evidence_rejects_noncanonical_calibrator_scalars(
    endpoints,
    field: str,
    invalid: object,
) -> None:
    _provider, payload, _metadata = _provider_evidence(endpoints)
    payload[field] = invalid
    with pytest.raises(ValueError):
        fisher_soft_polarity_confidence_provider_artifact_sha256(payload)


def test_provider_evidence_rejects_payload_forgery(endpoints) -> None:
    _provider, payload, _metadata = _provider_evidence(endpoints)

    attacks = []
    invalid_sha = copy.deepcopy(payload)
    invalid_sha["bridge_binding_sha256"] = "A" * 64
    attacks.append(invalid_sha)

    stale_coefficient_hash = copy.deepcopy(payload)
    stale_coefficient_hash["linear_coefficient"] = 0.91
    attacks.append(stale_coefficient_hash)

    frozen_constant = copy.deepcopy(payload)
    frozen_constant["constant_tensor_sha256s"]["box_corner_features"] = "0" * 64
    attacks.append(frozen_constant)

    frozen_formula = copy.deepcopy(payload)
    frozen_formula["gain_formula"] = "different"
    attacks.append(frozen_formula)

    frozen_status = copy.deepcopy(payload)
    frozen_status["experimental_serving_status"] = "serving_ready"
    attacks.append(frozen_status)

    raw_tensor = copy.deepcopy(payload)
    raw_tensor["direction_sha256"] = torch.zeros(4, dtype=torch.float64)
    attacks.append(raw_tensor)

    for attacked in attacks:
        with pytest.raises(ValueError):
            fisher_soft_polarity_confidence_provider_artifact_sha256(attacked)

    missing = copy.deepcopy(payload)
    missing.pop("gain_formula")
    with pytest.raises(ValueError, match="key set differs"):
        fisher_soft_polarity_confidence_provider_artifact_sha256(missing)
    extra = copy.deepcopy(payload)
    extra["unexpected"] = "field"
    with pytest.raises(ValueError, match="key set differs"):
        fisher_soft_polarity_confidence_provider_artifact_sha256(extra)


def test_provider_evidence_rejects_metadata_forgery(endpoints) -> None:
    _provider, payload, metadata = _provider_evidence(endpoints)

    attacks = []
    wrong_artifact = copy.deepcopy(metadata)
    wrong_artifact["artifact_sha256"] = "0" * 64
    attacks.append(wrong_artifact)

    payload_disagreement = copy.deepcopy(metadata)
    payload_disagreement["transfer_evidence_sha256"] = "1" * 64
    attacks.append(payload_disagreement)

    bool_rank = copy.deepcopy(metadata)
    bool_rank["rank"] = True
    attacks.append(bool_rank)

    wrong_prepared_formula = copy.deepcopy(metadata)
    wrong_prepared_formula["incremental_prepared_float_scalar_count"] += 1
    wrong_prepared_formula["incremental_runtime_parameter_bytes_float64"] += 8
    attacks.append(wrong_prepared_formula)

    wrong_bytes = copy.deepcopy(metadata)
    wrong_bytes["incremental_runtime_parameter_bytes_float64"] += 8
    attacks.append(wrong_bytes)

    wrong_macs = copy.deepcopy(metadata)
    wrong_macs["incremental_logical_macs_per_token_upper_bound"] += 1
    attacks.append(wrong_macs)

    wrong_certificate_hash = copy.deepcopy(metadata)
    wrong_certificate_hash["box_certificate"]["direction_sha256"] = "2" * 64
    attacks.append(wrong_certificate_hash)

    wrong_corner_certificate = copy.deepcopy(metadata)
    wrong_corner_certificate["box_certificate"]["direction_box_corner_logits"] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    attacks.append(wrong_corner_certificate)

    wrong_accounting_scope = copy.deepcopy(metadata)
    wrong_accounting_scope["logical_macs_accounting_scope"] = "different"
    attacks.append(wrong_accounting_scope)

    raw_tensor = copy.deepcopy(metadata)
    raw_tensor["box_certificate"]["direction_box_corner_logits"] = torch.zeros(4)
    attacks.append(raw_tensor)

    for attacked in attacks:
        with pytest.raises(ValueError):
            validate_fisher_soft_polarity_confidence_provider_evidence(
                payload,
                attacked,
            )

    missing = copy.deepcopy(metadata)
    missing.pop("rank")
    with pytest.raises(ValueError, match="key set differs"):
        validate_fisher_soft_polarity_confidence_provider_evidence(payload, missing)
    extra = copy.deepcopy(metadata)
    extra["unexpected"] = "field"
    with pytest.raises(ValueError, match="key set differs"):
        validate_fisher_soft_polarity_confidence_provider_evidence(payload, extra)


def test_provider_supports_batched_leading_dimensions(endpoints) -> None:
    base, proposal = endpoints
    provider = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base,
        proposal,
        direction=_normalized_direction(),
        linear_coefficient=0.7,
        cubic_coefficient=1.2,
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
    ("direction", "linear_coefficient", "cubic_coefficient", "message"),
    (
        (torch.zeros(4), 1.0, 0.0, "nonzero"),
        (torch.zeros(3), 1.0, 0.0, "exactly four"),
        (
            torch.tensor((1.0, 0.0, 0.0, float("nan"))),
            1.0,
            0.0,
            "finite",
        ),
        (_raw_direction(), 1.0, 0.0, "box-corner normalized"),
        (_normalized_direction(), -0.1, 0.0, "nonnegative"),
        (_normalized_direction(), 0.1, -0.1, "nonnegative"),
    ),
)
def test_provider_parameter_validation_fails_closed(
    endpoints, direction, linear_coefficient, cubic_coefficient, message
) -> None:
    base, proposal = endpoints
    with pytest.raises(ValueError, match=message):
        build_autonomous_complete_h4_fisher_soft_polarity_confidence(
            base,
            proposal,
            direction=direction,
            linear_coefficient=linear_coefficient,
            cubic_coefficient=cubic_coefficient,
            transfer_protocol_sha256=_PROTOCOL,
            transfer_evidence_sha256=_EVIDENCE,
        )


def test_pure_formula_rejects_unnormalized_direction_and_bad_coefficients() -> None:
    coordinates = torch.zeros(3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="box-corner normalized"):
        fisher_soft_polarity_confidence_gain(
            coordinates,
            _raw_direction(),
            torch.tensor(1.0),
            torch.tensor(0.0),
        )
    direction = _normalized_direction()
    with pytest.raises(ValueError, match="nonnegative"):
        fisher_soft_polarity_confidence_gain(
            coordinates,
            direction,
            torch.tensor(-1.0),
            torch.tensor(0.0),
        )
    with pytest.raises(ValueError, match="numerical magnitude limit"):
        fisher_soft_polarity_confidence_gain(
            coordinates,
            direction,
            torch.tensor(
                2.0 * FISHER_SOFT_POLARITY_CONFIDENCE_CALIBRATOR_MAX,
                dtype=torch.float64,
            ),
            torch.tensor(0.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match=r"inside \[-1,1\]"):
        fisher_soft_polarity_confidence_gain(
            torch.tensor(((0.0, 1.01),), dtype=torch.float64),
            direction,
            torch.tensor(1.0),
            torch.tensor(0.0),
        )


def test_public_constant_rebinding_cannot_change_provider_semantics(
    endpoints, monkeypatch
) -> None:
    import fisher_graph.complete_h4_fisher_soft_polarity_confidence as module

    base, proposal = endpoints
    kwargs = {
        "direction": _normalized_direction(),
        "linear_coefficient": 0.8,
        "cubic_coefficient": 1.3,
        "transfer_protocol_sha256": _PROTOCOL,
        "transfer_evidence_sha256": _EVIDENCE,
    }
    expected = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base, proposal, **kwargs
    )
    monkeypatch.setattr(module, "FISHER_SOFT_POLARITY_CONFIDENCE_DIRECTION_COUNT", 3)
    monkeypatch.setattr(
        module, "FISHER_SOFT_POLARITY_CONFIDENCE_FITTED_SCALAR_COUNT", 1
    )
    monkeypatch.setattr(
        module, "FISHER_SOFT_POLARITY_CONFIDENCE_CALIBRATOR_MAX", 0.01
    )
    observed = build_autonomous_complete_h4_fisher_soft_polarity_confidence(
        base, proposal, **kwargs
    )
    assert observed.artifact_sha256 == expected.artifact_sha256
    assert observed.metadata()["fitted_float64_scalar_count"] == 6


def test_exported_geometry_constants_are_pinned() -> None:
    assert FISHER_SOFT_POLARITY_CONFIDENCE_DIRECTION_COUNT == 4
    assert FISHER_SOFT_POLARITY_CONFIDENCE_FITTED_SCALAR_COUNT == 6
