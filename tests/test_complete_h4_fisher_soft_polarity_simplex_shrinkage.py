from __future__ import annotations

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
from fisher_graph.complete_h4_fisher_soft_polarity_simplex_response import (
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_response,
    fisher_soft_polarity_simplex_response_gain,
    normalize_fisher_soft_polarity_simplex_response_direction,
)
from fisher_graph.complete_h4_fisher_soft_polarity_simplex_shrinkage import (
    FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_LINEAGE_SCALAR_COUNT,
    FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256,
    FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_RUNTIME_FITTED_SCALAR_COUNT,
    AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_shrinkage,
    fisher_soft_polarity_simplex_shrinkage_box_certificate,
    fisher_soft_polarity_simplex_shrinkage_calibrator,
    fisher_soft_polarity_simplex_shrinkage_effective_parameters,
    fisher_soft_polarity_simplex_shrinkage_gain,
    fisher_soft_polarity_simplex_shrinkage_modal_terms,
    fisher_soft_polarity_simplex_shrinkage_provider_artifact_sha256,
    validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4CorrectionProvider,
)


_BRIDGE = "d" * 64
_V19_PROTOCOL = "a" * 64
_TRANSFER_PROTOCOL = "e" * 64
_TRANSFER_EVIDENCE = "f" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 8) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(118700 + index)
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
        example_id=f"simplex-shrinkage-example-{index}",
        family_id=f"simplex-shrinkage-family-{index // 2}",
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


def _direction() -> torch.Tensor:
    return normalize_fisher_soft_polarity_simplex_response_direction(
        torch.tensor((0.35, -0.55, 0.20, 0.45), dtype=torch.float64)
    )


def _provider(endpoints, lambda_: float = 0.375):
    base, proposal = endpoints
    return build_autonomous_complete_h4_fisher_soft_polarity_simplex_shrinkage(
        base,
        proposal,
        direction=_direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        lambda_=lambda_,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )


def test_protocol_and_scalar_accounting_are_frozen() -> None:
    assert FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_PROTOCOL_SHA256 == (
        "d348cc7160ef6237ead45aea8cd752be5bd35f880044831f26d106f00c225b35"
    )
    assert FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_DIRECTION_COUNT == 4
    assert FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_RUNTIME_FITTED_SCALAR_COUNT == 7
    assert FISHER_SOFT_POLARITY_SIMPLEX_SHRINKAGE_FIT_LINEAGE_SCALAR_COUNT == 8


@pytest.mark.parametrize(
    ("lambda_", "expected_u", "expected_v"),
    (
        (0.0, 0.0, 0.0),
        (0.25, 0.0625, -0.03125),
        (0.5, 0.125, -0.0625),
        (1.0, 0.25, -0.125),
    ),
)
def test_effective_parameters_are_exactly_materialized(
    lambda_: float, expected_u: float, expected_v: float
) -> None:
    assert fisher_soft_polarity_simplex_shrinkage_effective_parameters(
        0.25, 0.25, -0.125, lambda_
    ) == (0.25, expected_u, expected_v)


def test_lambda_zero_canonicalizes_negative_bias_product_to_positive_zero() -> None:
    _r, _u, effective_v = (
        fisher_soft_polarity_simplex_shrinkage_effective_parameters(
            0.25, 0.25, -0.125, 0.0
        )
    )
    assert effective_v.hex() == "0x0.0p+0"


@pytest.mark.parametrize(
    ("args", "match"),
    (
        ((0.25, 0.25, -0.125, float("nan")), "finite"),
        ((0.25, 0.25, -0.125, -0.1), "inside"),
        ((0.25, 0.25, -0.125, 1.1), "inside"),
        ((0.25, 0.125, 0.25, 0.5), r"abs\(polarity_bias\)"),
        ((0.25, 0.6, 0.0, 0.5), "inside"),
        ((-0.0, 0.0, 0.0, 0.0), "negative zero"),
    ),
)
def test_invalid_source_or_lambda_is_rejected(
    args: tuple[float, float, float, float], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        fisher_soft_polarity_simplex_shrinkage_effective_parameters(*args)


def test_formula_is_exact_and_bounded_for_interior_lambda() -> None:
    generator = torch.Generator().manual_seed(4401)
    coordinates = 2.0 * torch.rand(
        20000, 2, generator=generator, dtype=torch.float64
    ) - 1.0
    direction = _direction()
    corners = torch.stack(
        (
            torch.ones(20000, dtype=torch.float64),
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates.prod(dim=1),
        ),
        dim=1,
    )
    z = corners @ direction
    lambda_ = 0.375
    expected = (
        (1.0 - lambda_ * 0.25 * z.square()) * torch.tanh(0.25 * z)
        + lambda_ * -0.125 * z.square()
    )
    observed = fisher_soft_polarity_simplex_shrinkage_calibrator(
        z, 0.25, 0.25, -0.125, lambda_
    )
    assert torch.equal(observed, expected)
    assert float(observed.abs().max()) <= 1.0


def test_lambda_endpoints_are_bit_exact_v20m_controls() -> None:
    coordinates = torch.tensor(
        ((-0.8, 0.25), (-0.2, -0.6), (0.0, 0.4), (0.7, -0.1)),
        dtype=torch.float64,
    )
    direction = _direction()
    linear = fisher_soft_polarity_simplex_response_gain(
        coordinates, direction, 0.25, 0.0, 0.0
    )
    source = fisher_soft_polarity_simplex_response_gain(
        coordinates, direction, 0.25, 0.25, -0.125
    )
    assert torch.equal(
        fisher_soft_polarity_simplex_shrinkage_gain(
            coordinates, direction, 0.25, 0.25, -0.125, 0.0
        ),
        linear,
    )
    assert torch.equal(
        fisher_soft_polarity_simplex_shrinkage_gain(
            coordinates, direction, 0.25, 0.25, -0.125, 1.0
        ),
        source,
    )
    interior = fisher_soft_polarity_simplex_shrinkage_gain(
        coordinates, direction, 0.25, 0.25, -0.125, 0.5
    )
    assert not torch.equal(interior, linear)
    assert not torch.equal(interior, source)


def test_certificate_preserves_source_effective_and_inherited_proof() -> None:
    certificate = fisher_soft_polarity_simplex_shrinkage_box_certificate(
        _direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        lambda_=0.375,
    )
    assert certificate["source_shrink_mass"] == 0.25
    assert certificate["source_polarity_bias"] == -0.125
    assert certificate["shrinkage_lambda"] == 0.375
    assert certificate["effective_shrink_mass"] == 0.09375
    assert certificate["effective_polarity_bias"] == -0.046875
    assert certificate[
        "interior_effective_parameters_distinct_when_source_u_positive"
    ] is True
    assert certificate["activation_dependent_branch"] is False
    assert certificate["shrinkage_lambda_runtime_float_scalar_count"] == 0
    inherited = certificate["inherited_v20m_box_certificate"]
    assert inherited["simplex_weights_nonnegative"] is True
    assert inherited["simplex_weights_sum_to_one"] is True


def test_provider_materializes_one_v20m_runtime_without_extra_accounting(
    endpoints,
) -> None:
    provider = _provider(endpoints)
    assert isinstance(provider, Gemma3L3L4CorrectionProvider)
    assert isinstance(
        provider, AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider
    )
    assert provider.radius == 0.25
    assert provider.shrink_mass == 0.25
    assert provider.polarity_bias == -0.125
    assert provider.lambda_ == 0.375
    assert provider.effective_shrink_mass == 0.09375
    assert provider.runtime_provider is provider.compiled_provider
    assert provider.effective_polarity_bias == -0.046875
    assert (
        provider.prepared_float_scalar_count
        == provider.compiled_provider.prepared_float_scalar_count
    )
    assert (
        provider.logical_macs_per_token_upper_bound
        == provider.compiled_provider.logical_macs_per_token_upper_bound
    )
    metadata = provider.metadata()
    assert metadata["shrinkage_runtime_extra_scalar_count"] == 0
    assert metadata["source_coefficients_and_lambda_runtime_float_scalar_count"] == 0
    assert metadata["routing_control_flow"] == "none_validation_guards_only"


def test_provider_runtime_terms_are_bit_exact_materialized_v20m(endpoints) -> None:
    provider = _provider(endpoints, 0.5)
    base, proposal = endpoints
    v20m = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=_direction(),
        radius=0.25,
        shrink_mass=0.125,
        polarity_bias=-0.0625,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    generator = torch.Generator().manual_seed(9001)
    parent = torch.randn(9, provider.rank, generator=generator, dtype=torch.float64)
    coordinates = provider.bounded_coordinates(parent)
    assert torch.equal(provider.response_gain(coordinates), v20m.response_gain(coordinates))
    for observed, expected in zip(
        provider.terms_from_parent(parent, coordinates),
        v20m.terms_from_parent(parent, coordinates),
        strict=True,
    ):
        assert torch.equal(observed, expected)


def test_provider_endpoint_compilation_is_exact_v20m(endpoints) -> None:
    base, proposal = endpoints
    zero = _provider(endpoints, 0.0)
    one = _provider(endpoints, 1.0)
    linear = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=_direction(),
        radius=0.25,
        shrink_mass=0.0,
        polarity_bias=0.0,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    source = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=_direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    assert zero.compiled_provider.artifact_sha256 == linear.artifact_sha256
    assert one.compiled_provider.artifact_sha256 == source.artifact_sha256


def test_modal_terms_function_is_bit_exact_v20m_materialization(endpoints) -> None:
    base, proposal = endpoints
    generator = torch.Generator().manual_seed(9002)
    parent = torch.randn(7, base.rank, generator=generator, dtype=torch.float64)
    coordinates = base.bounded_coordinates(parent)
    observed = fisher_soft_polarity_simplex_shrinkage_modal_terms(
        parent,
        coordinates,
        base.direction_left,
        base.direction_right,
        proposal.direction_left,
        proposal.direction_right,
        base.pedal_weight,
        base.pedal_bias,
        proposal.pedal_weight,
        proposal.pedal_bias,
        _direction(),
        0.25,
        0.25,
        -0.125,
        0.5,
    )
    provider_terms = _provider(endpoints, 0.5).terms_from_parent(
        parent, coordinates
    )
    for left, right in zip(observed, provider_terms, strict=True):
        assert torch.equal(left, right)


def test_artifact_preserves_lineage_even_when_runtime_endpoint_matches(endpoints) -> None:
    zero = _provider(endpoints, 0.0)
    half = _provider(endpoints, 0.5)
    one = _provider(endpoints, 1.0)
    assert len({zero.artifact_sha256, half.artifact_sha256, one.artifact_sha256}) == 3
    assert (
        fisher_soft_polarity_simplex_shrinkage_provider_artifact_sha256(
            half.artifact_payload()
        )
        == half.artifact_sha256
    )


def test_provider_evidence_round_trip_and_nested_authentication(endpoints) -> None:
    provider = _provider(endpoints)
    validated = validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence(
        provider.artifact_payload(), provider.metadata()
    )
    assert validated.artifact_sha256 == provider.artifact_sha256
    assert validated.payload["source_shrink_mass"] == 0.25
    assert validated.payload["effective_shrink_mass"] == 0.09375
    assert validated.metadata["shrinkage_runtime_extra_scalar_count"] == 0

    json_validated = validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence(
        json.loads(json.dumps(provider.artifact_payload())),
        json.loads(json.dumps(provider.metadata())),
    )
    assert json_validated.artifact_sha256 == provider.artifact_sha256


@pytest.mark.parametrize(
    "attack",
    (
        "lambda",
        "effective",
        "compiled_payload",
        "certificate",
        "accounting",
        "extra_key",
    ),
)
def test_provider_evidence_rejects_forgery(endpoints, attack: str) -> None:
    provider = _provider(endpoints)
    payload = provider.artifact_payload()
    metadata = provider.metadata()
    if attack == "lambda":
        payload["shrinkage_lambda"] = 0.5
    elif attack == "effective":
        payload["effective_shrink_mass"] = 0.125
    elif attack == "compiled_payload":
        payload["compiled_simplex_response_provider_payload"]["shrink_mass"] = 0.125
    elif attack == "certificate":
        metadata["box_certificate"]["activation_dependent_branch"] = True
    elif attack == "accounting":
        metadata["prepared_float_scalar_count"] += 1
    else:
        payload["unexpected"] = False
    with pytest.raises((TypeError, ValueError)):
        validate_fisher_soft_polarity_simplex_shrinkage_provider_evidence(
            payload, metadata
        )


def test_provider_constructor_rejects_mismatched_materialization(endpoints) -> None:
    compiled = _provider(endpoints, 0.5).compiled_provider
    with pytest.raises(RuntimeError, match="materialized V20m coefficients"):
        AutonomousCompleteH4FisherSoftPolaritySimplexShrinkageProvider(
            compiled_provider=compiled,
            source_radius=0.25,
            source_shrink_mass=0.25,
            source_polarity_bias=-0.125,
            shrinkage_lambda=0.25,
        )


def test_artifact_payload_copy_cannot_mutate_provider(endpoints) -> None:
    provider = _provider(endpoints)
    payload = provider.artifact_payload()
    payload["source_radius"] = 0.125
    payload["compiled_simplex_response_provider_payload"]["radius"] = 0.125
    provider.validate_integrity()
    assert provider.source_radius == 0.25
    assert provider.compiled_provider.radius == 0.25
