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
    build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control,
)
from fisher_graph.complete_h4_fisher_soft_polarity_signed_continuum import (
    FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_FIT_LINEAGE_SCALAR_COUNT,
    FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256,
    FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_RUNTIME_FITTED_SCALAR_COUNT,
    AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider,
    build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum,
    fisher_soft_polarity_signed_continuum_box_certificate,
    fisher_soft_polarity_signed_continuum_gain,
    fisher_soft_polarity_signed_continuum_materialized_parameters,
    fisher_soft_polarity_signed_continuum_provider_artifact_sha256,
    validate_fisher_soft_polarity_signed_continuum_provider_evidence,
    validate_fisher_soft_polarity_signed_continuum_runtime_provider_evidence,
)
from fisher_graph.complete_h4_fisher_soft_polarity_simplex_response import (
    build_autonomous_complete_h4_fisher_soft_polarity_simplex_response,
    normalize_fisher_soft_polarity_simplex_response_direction,
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
    generator = torch.Generator().manual_seed(218700 + index)
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
        example_id=f"signed-continuum-example-{index}",
        family_id=f"signed-continuum-family-{index // 2}",
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


def _provider(endpoints, signed_scalar: float = 0.375):
    base, proposal = endpoints
    return build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum(
        base,
        proposal,
        direction=_direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        signed_scalar=signed_scalar,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )


def test_protocol_and_runtime_materialization_are_frozen() -> None:
    assert FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_PROTOCOL_SHA256 == (
        "2143b3652fa1d435bfb6d12574ad11b4dedf7c7a118d4de5b196834910dd02c9"
    )
    assert FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_DIRECTION_COUNT == 4
    assert FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_FIT_LINEAGE_SCALAR_COUNT == 8
    assert FISHER_SOFT_POLARITY_SIGNED_CONTINUUM_RUNTIME_FITTED_SCALAR_COUNT == 8
    assert fisher_soft_polarity_signed_continuum_materialized_parameters(-0.25) == (
        -1,
        0.25,
    )
    assert fisher_soft_polarity_signed_continuum_materialized_parameters(0.0) == (
        1,
        0.0,
    )


@pytest.mark.parametrize("signed", [-1.01, 1.01, float("nan"), True, -0.0])
def test_invalid_signed_scalar_fails_closed(endpoints, signed) -> None:
    with pytest.raises((TypeError, ValueError)):
        _provider(endpoints, signed)


def test_formula_is_bounded_and_uses_one_continuous_path() -> None:
    coordinates = torch.tensor(
        [[-1.0, -1.0], [-0.5, 0.25], [0.0, 0.0], [0.4, -0.8], [1.0, 1.0]],
        dtype=torch.float64,
    )
    direction = _direction()
    for signed in (-1.0, -0.63, 0.0, 0.22, 1.0):
        sign, mix = fisher_soft_polarity_signed_continuum_materialized_parameters(
            signed
        )
        compiled = direction if sign == 1 else -direction
        gain = fisher_soft_polarity_signed_continuum_gain(
            coordinates, compiled, 0.25, 0.25, -0.125, mix
        )
        assert torch.isfinite(gain).all()
        assert bool((gain.abs() <= 1.0).all())


def test_all_three_anchors_are_bit_exact_complete_modal_controls(endpoints) -> None:
    base, proposal = endpoints
    direction = _direction()
    reflected = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=direction,
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    mirror = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=-direction,
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    fixed_plus = build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
        base,
        proposal,
        polarity=1,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    parent = torch.randn(11, _RANK, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    coordinates = base.bounded_coordinates(parent)
    controls = ((-1.0, mirror), (0.0, fixed_plus), (1.0, reflected))
    for signed, control in controls:
        provider = _provider(endpoints, signed).runtime_provider
        observed = provider.terms_from_parent(parent, coordinates)
        expected = control.terms_from_parent(parent, coordinates)
        assert all(torch.equal(left, right) for left, right in zip(observed, expected, strict=True))


def test_real_v20m_direction_is_not_renormalized_at_signed_anchors(endpoints) -> None:
    base, proposal = endpoints
    # This authenticated V20m-shaped direction moves by one ULP if normalized
    # again, which is enough to break the exact endpoint receipt comparison.
    direction = torch.tensor(
        (
            0.38868548947918286,
            -0.3423603815999848,
            0.2721916569520773,
            0.5411457858729096,
        ),
        dtype=torch.float64,
    )
    assert not torch.equal(
        normalize_fisher_soft_polarity_simplex_response_direction(direction),
        direction,
    )
    reflected = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=direction,
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    mirror = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
        base,
        proposal,
        direction=-direction,
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    fixed_plus = build_autonomous_complete_h4_fisher_soft_polarity_fixed_envelope_control(
        base,
        proposal,
        polarity=1,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )
    parent = torch.randn(
        11,
        _RANK,
        generator=torch.Generator().manual_seed(9),
        dtype=torch.float64,
    )
    coordinates = base.bounded_coordinates(parent)
    for signed, control in ((-1.0, mirror), (0.0, fixed_plus), (1.0, reflected)):
        provider = build_autonomous_complete_h4_fisher_soft_polarity_signed_continuum(
            base,
            proposal,
            direction=direction,
            radius=0.25,
            shrink_mass=0.25,
            polarity_bias=-0.125,
            signed_scalar=signed,
            transfer_protocol_sha256=_TRANSFER_PROTOCOL,
            transfer_evidence_sha256=_TRANSFER_EVIDENCE,
        )
        assert torch.equal(provider.direction, direction)
        expected_runtime_direction = direction if signed >= 0.0 else -direction
        assert torch.equal(provider.runtime_provider.direction, expected_runtime_direction)
        assert fisher_soft_polarity_signed_continuum_box_certificate(
            direction,
            radius=0.25,
            shrink_mass=0.25,
            polarity_bias=-0.125,
            signed_scalar=signed,
        ) == provider.metadata()["box_certificate"]
        observed = provider.runtime_provider.terms_from_parent(parent, coordinates)
        expected = control.terms_from_parent(parent, coordinates)
        assert all(
            torch.equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )


def test_wrapper_exposes_only_compiled_runtime_for_execution(endpoints) -> None:
    provider = _provider(endpoints, -0.375)
    assert isinstance(
        provider, AutonomousCompleteH4FisherSoftPolaritySignedContinuumProvider
    )
    assert provider.runtime_provider is provider.compiled_provider
    assert provider.compiled_direction_sign == -1
    assert provider.compiled_mix == 0.375
    assert torch.equal(provider.direction, _direction())
    assert torch.equal(provider.runtime_provider.direction, -_direction())
    assert provider.runtime_provider.artifact_sha256 != provider.artifact_sha256
    assert provider.metadata()["routing_control_flow"] == "none_validation_guards_only"


def test_provider_and_runtime_evidence_survive_json_roundtrip(endpoints) -> None:
    provider = _provider(endpoints, -0.375)
    payload = json.loads(json.dumps(provider.artifact_payload()))
    metadata = json.loads(json.dumps(provider.metadata()))
    evidence = validate_fisher_soft_polarity_signed_continuum_provider_evidence(
        payload, metadata
    )
    assert evidence.artifact_sha256 == provider.artifact_sha256
    runtime_payload = payload["compiled_runtime_provider_payload"]
    runtime_metadata = metadata["compiled_runtime_provider_metadata"]
    runtime_evidence = (
        validate_fisher_soft_polarity_signed_continuum_runtime_provider_evidence(
            runtime_payload, runtime_metadata
        )
    )
    assert runtime_evidence.artifact_sha256 == provider.runtime_provider.artifact_sha256
    assert fisher_soft_polarity_signed_continuum_provider_artifact_sha256(
        payload
    ) == provider.artifact_sha256


@pytest.mark.parametrize(
    "attack",
    ["signed_scalar", "compiled_mix", "compiled_direction", "nested_runtime"],
)
def test_provider_evidence_rejects_nested_tampering(endpoints, attack: str) -> None:
    provider = _provider(endpoints, -0.375)
    payload = copy.deepcopy(provider.artifact_payload())
    metadata = copy.deepcopy(provider.metadata())
    if attack == "signed_scalar":
        payload["signed_scalar"] = -0.25
    elif attack == "compiled_mix":
        payload["compiled_mix"] = 0.25
    elif attack == "compiled_direction":
        payload["compiled_direction_sha256"] = "0" * 64
    else:
        payload["compiled_runtime_provider_payload"]["compiled_mix"] = 0.25
    with pytest.raises((TypeError, ValueError)):
        validate_fisher_soft_polarity_signed_continuum_provider_evidence(
            payload, metadata
        )


def test_box_certificate_freezes_anchor_and_branch_claims() -> None:
    certificate = fisher_soft_polarity_signed_continuum_box_certificate(
        _direction(),
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        signed_scalar=0.0,
    )
    assert certificate["zero_exact_fixed_plus"] is True
    assert certificate["minus_one_exact_v20m_mirror"] is False
    assert certificate["plus_one_exact_v20m_reflected"] is False
    assert certificate["runtime_activation_dependent_branch"] is False
    assert certificate["gain_max_abs"] == 1.0


def test_artifact_payload_copy_cannot_mutate_provider(endpoints) -> None:
    provider = _provider(endpoints)
    artifact = provider.artifact_sha256
    payload = provider.artifact_payload()
    payload["compiled_runtime_provider_payload"]["compiled_mix"] = 0.0
    assert provider.artifact_sha256 == artifact
    provider.validate_integrity()
