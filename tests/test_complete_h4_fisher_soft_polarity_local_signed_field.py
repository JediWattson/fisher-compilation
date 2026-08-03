from __future__ import annotations

import copy
import json

import pytest
import torch

import fisher_graph.complete_h4_fisher_soft_polarity_local_signed_field as local_field_module
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
from fisher_graph.complete_h4_fisher_soft_polarity_local_signed_field import (
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_DIRECTION_COUNT,
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_ID_COUNT,
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_LINEAGE_SCALAR_COUNT,
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256,
    FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_RUNTIME_FITTED_SCALAR_COUNT,
    build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field,
    fisher_soft_polarity_local_signed_field_box_certificate,
    fisher_soft_polarity_local_signed_field_calibrator,
    fisher_soft_polarity_local_signed_field_feature,
    fisher_soft_polarity_local_signed_field_gain,
    fisher_soft_polarity_local_signed_field_projection,
    fisher_soft_polarity_local_signed_field_provider_artifact_sha256,
    fisher_soft_polarity_local_signed_field_runtime_provider_artifact_sha256,
    fisher_soft_polarity_local_signed_field_signed_scalar,
    validate_fisher_soft_polarity_local_signed_field_provider_evidence,
    validate_fisher_soft_polarity_local_signed_field_runtime_provider_evidence,
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
_OBSIDIAN_DIRECTION = torch.tensor(
    (
        0.38868548947918286,
        -0.3423603815999848,
        0.2721916569520773,
        0.5411457858729096,
    ),
    dtype=torch.float64,
)


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 8) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(219700 + index)
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
        example_id=f"local-field-example-{index}",
        family_id=f"local-field-family-{index // 2}",
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


def _provider(
    endpoints,
    *,
    field_bias: float = 0.2,
    field_slope: float = 0.7,
    feature_id: int | str = "z",
    direction: torch.Tensor = _OBSIDIAN_DIRECTION,
):
    base, proposal = endpoints
    return build_autonomous_complete_h4_fisher_soft_polarity_local_signed_field(
        base,
        proposal,
        direction=direction,
        radius=0.25,
        shrink_mass=0.25,
        polarity_bias=-0.125,
        field_bias=field_bias,
        field_slope=field_slope,
        feature_id=feature_id,
        transfer_protocol_sha256=_TRANSFER_PROTOCOL,
        transfer_evidence_sha256=_TRANSFER_EVIDENCE,
    )


def test_protocol_shape_and_compact_runtime_accounting_are_frozen() -> None:
    assert FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_PROTOCOL_SHA256 == (
        "4f0503ee0f66680a0d9b8efd26b174afb7e81607d0dbd7947943086a4fdef4d5"
    )
    assert FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_DIRECTION_COUNT == 4
    assert FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FIT_LINEAGE_SCALAR_COUNT == 9
    assert FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_RUNTIME_FITTED_SCALAR_COUNT == 9
    assert FISHER_SOFT_POLARITY_LOCAL_SIGNED_FIELD_FEATURE_ID_COUNT == 1


@pytest.mark.parametrize(
    ("feature_id", "expected"),
    (
        ("c1", torch.tensor((-0.5, 0.3), dtype=torch.float64)),
        ("c2", torch.tensor((0.25, -0.7), dtype=torch.float64)),
        ("c1_times_c2", torch.tensor((-0.125, -0.21), dtype=torch.float64)),
    ),
)
def test_feature_ladder_uses_only_frozen_local_coordinates(feature_id, expected) -> None:
    coordinates = torch.tensor(((-0.5, 0.25), (0.3, -0.7)), dtype=torch.float64)
    projection = torch.tensor((0.1, -0.2), dtype=torch.float64)
    assert torch.equal(
        fisher_soft_polarity_local_signed_field_feature(
            coordinates, projection, feature_id
        ),
        expected,
    )
    assert torch.equal(
        fisher_soft_polarity_local_signed_field_feature(
            coordinates, projection, "z"
        ),
        projection,
    )


def test_formula_spellings_canonicalize_to_fitter_protocol_ids(endpoints) -> None:
    product = _provider(endpoints, feature_id="c1*c2")
    projection = _provider(endpoints, feature_id="z")
    assert (product.feature_id, product.feature_name) == (2, "c1_times_c2")
    assert (projection.feature_id, projection.feature_name) == (3, "source_z")


def test_local_field_is_continuous_clamped_and_branch_free() -> None:
    psi = torch.linspace(-1.0, 1.0, 101, dtype=torch.float64)
    signed = fisher_soft_polarity_local_signed_field_signed_scalar(
        psi, field_bias=0.15, field_slope=1.4
    )
    assert torch.isfinite(signed).all()
    assert bool((signed.abs() <= 1.0).all())
    assert bool((signed[1:] >= signed[:-1]).all())
    assert torch.equal(
        signed,
        torch.clamp(0.15 + 1.4 * psi, min=-1.0, max=1.0),
    )


def test_frozen_formula_is_bounded_for_nonconstant_local_fields() -> None:
    coordinates = torch.tensor(
        ((-1.0, -1.0), (-0.5, 0.25), (0.0, 0.0), (0.4, -0.8), (1.0, 1.0)),
        dtype=torch.float64,
    )
    direction = _OBSIDIAN_DIRECTION
    projection = fisher_soft_polarity_local_signed_field_projection(
        coordinates, direction
    )
    for feature_id in range(4):
        feature = fisher_soft_polarity_local_signed_field_feature(
            coordinates, projection, feature_id
        )
        value = fisher_soft_polarity_local_signed_field_calibrator(
            projection,
            feature,
            0.25,
            0.25,
            -0.125,
            0.2,
            0.9,
        )
        gain = fisher_soft_polarity_local_signed_field_gain(
            coordinates,
            direction,
            0.25,
            0.25,
            -0.125,
            0.2,
            0.9,
            feature_id,
        )
        assert torch.isfinite(value).all() and bool((value.abs() <= 1.0).all())
        assert torch.isfinite(gain).all() and bool((gain.abs() <= 1.0).all())


@pytest.mark.parametrize("feature_id", range(4))
def test_all_constant_anchors_are_bit_exact_for_every_feature(
    endpoints, feature_id: int
) -> None:
    base, proposal = endpoints
    direction = _OBSIDIAN_DIRECTION
    # This authenticated V20m direction is deliberately not normalization-
    # idempotent; another normalization moves it by one ULP.
    assert not torch.equal(
        normalize_fisher_soft_polarity_simplex_response_direction(direction),
        direction,
    )
    source = build_autonomous_complete_h4_fisher_soft_polarity_simplex_response(
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
        19,
        _RANK,
        generator=torch.Generator().manual_seed(31 + feature_id),
        dtype=torch.float64,
    )
    coordinates = base.bounded_coordinates(parent)
    for field_bias, control in ((-1.0, mirror), (0.0, fixed_plus), (1.0, source)):
        provider = _provider(
            endpoints,
            field_bias=field_bias,
            field_slope=0.0,
            feature_id=feature_id,
            direction=direction,
        )
        assert torch.equal(provider.runtime_provider.direction, direction)
        assert (
            fisher_soft_polarity_local_signed_field_box_certificate(
                direction,
                radius=0.25,
                shrink_mass=0.25,
                polarity_bias=-0.125,
                field_bias=field_bias,
                field_slope=0.0,
                feature_id=feature_id,
            )
            == provider.metadata()["box_certificate"]
        )
        observed = provider.runtime_provider.terms_from_parent(parent, coordinates)
        expected = control.terms_from_parent(parent, coordinates)
        assert all(
            torch.equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )


def test_evidence_round_trips_canonically_and_counts_two_field_scalars(endpoints) -> None:
    provider = _provider(endpoints, feature_id="c1*c2")
    payload = provider.artifact_payload()
    metadata = provider.metadata()
    validated = validate_fisher_soft_polarity_local_signed_field_provider_evidence(
        json.loads(json.dumps(payload)), json.loads(json.dumps(metadata))
    )
    assert validated.artifact_sha256 == provider.artifact_sha256
    assert (
        fisher_soft_polarity_local_signed_field_provider_artifact_sha256(payload)
        == provider.artifact_sha256
    )
    runtime = provider.runtime_provider
    runtime_validated = (
        validate_fisher_soft_polarity_local_signed_field_runtime_provider_evidence(
            runtime.artifact_payload(), runtime.metadata()
        )
    )
    assert runtime_validated.artifact_sha256 == runtime.artifact_sha256
    assert (
        fisher_soft_polarity_local_signed_field_runtime_provider_artifact_sha256(
            runtime.artifact_payload()
        )
        == runtime.artifact_sha256
    )
    nested = runtime.source_simplex_provider.metadata()
    assert metadata["runtime_fitted_float_scalar_count"] == 9
    assert metadata["feature_id_uint8_count"] == 1
    assert metadata["incremental_prepared_float_scalar_count"] == (
        nested["incremental_prepared_float_scalar_count"] + 2
    )
    assert metadata["runtime_parameter_bytes_uint8"] == 1


def test_payload_or_nested_evidence_tampering_fails_closed(endpoints) -> None:
    provider = _provider(endpoints)
    payload = provider.artifact_payload()
    tampered = copy.deepcopy(payload)
    tampered["compiled_runtime_provider_payload"]["field_slope"] = 0.8
    with pytest.raises(ValueError):
        fisher_soft_polarity_local_signed_field_provider_artifact_sha256(tampered)

    metadata = provider.metadata()
    tampered_metadata = copy.deepcopy(metadata)
    tampered_metadata["box_certificate"]["runtime_activation_dependent_router"] = True
    with pytest.raises(ValueError):
        validate_fisher_soft_polarity_local_signed_field_provider_evidence(
            payload, tampered_metadata
        )


@pytest.mark.parametrize(
    ("key", "noncanonical"),
    (
        ("feature_id", "z"),
        ("feature_id", "source_z"),
        ("field_bias", 1),
        ("field_slope", 0),
    ),
)
def test_rehashed_noncanonical_runtime_types_fail_full_evidence_validation(
    endpoints, key: str, noncanonical: object
) -> None:
    runtime = _provider(
        endpoints,
        field_bias=1.0,
        field_slope=0.0,
        feature_id="source_z",
    ).runtime_provider
    payload = runtime.artifact_payload()
    metadata = runtime.metadata()
    payload[key] = noncanonical
    metadata[key] = noncanonical
    # Simulate an adversary recomputing the outer artifact digest over the
    # altered canonical tree.  Validation must reject the type before trusting
    # that otherwise internally consistent hash.
    forged_artifact = local_field_module._sha256(
        local_field_module._RUNTIME_PROVIDER_DOMAIN, payload
    )
    metadata["artifact_sha256"] = forged_artifact
    with pytest.raises(ValueError, match="canonical"):
        validate_fisher_soft_polarity_local_signed_field_runtime_provider_evidence(
            payload, metadata
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"feature_id": "unknown"}, ValueError),
        ({"feature_id": True}, TypeError),
        ({"field_bias": float("nan")}, ValueError),
        ({"field_slope": float("inf")}, ValueError),
        ({"field_slope": -0.0}, ValueError),
    ),
)
def test_invalid_field_configuration_fails_closed(endpoints, kwargs, error) -> None:
    with pytest.raises(error):
        _provider(endpoints, **kwargs)
