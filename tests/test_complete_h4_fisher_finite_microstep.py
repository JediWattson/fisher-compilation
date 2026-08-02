from __future__ import annotations

import copy

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
    replay_autonomous_complete_h4_fisher_finite_joint_pedal,
)
from fisher_graph.complete_h4_fisher_finite_microstep import (
    FISHER_FINITE_MICROSTEP_PATHS,
    FisherFiniteMicrostepResult,
    autonomous_complete_h4_fisher_finite_microstep_from_state_dict,
    autonomous_complete_h4_fisher_finite_microstep_state_dict,
    build_autonomous_complete_h4_fisher_finite_microstep,
    fisher_finite_microstep_selected_tensor_sha256s,
    interpolate_fisher_finite_microstep_parameters,
)


_BRIDGE = "d" * 64
_V19_PROTOCOL = "a" * 64
_V20_PROTOCOL = "1" * 64
_V20_EVIDENCE = "2" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 10) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(62000 + index)
    source = torch.randn(
        length, _SOURCE_RANK, generator=generator, dtype=torch.float64
    )
    base = torch.randn(length, _WIDTH, generator=generator, dtype=torch.float64)
    parent = 0.12 * base[:, :_RANK]
    raw = torch.stack(
        (
            parent[:, 0] + 0.2 * parent[:, 2],
            -0.7 * parent[:, 1] + 0.3 * parent[:, 3],
        ),
        dim=1,
    )
    coordinates = raw / (0.4 + raw.abs())
    direction = torch.stack(
        (
            0.20 * coordinates[:, 0] * parent[:, 0],
            -0.16 * coordinates[:, 1] * parent[:, 1],
            0.14 * coordinates[:, 0] * parent[:, 2]
            + 0.05 * coordinates.prod(dim=1) * parent[:, 0],
            0.12 * coordinates[:, 1] * parent[:, 3],
        ),
        dim=1,
    )
    pedal = torch.sigmoid(
        0.3
        + 0.5 * coordinates[:, 0]
        - 0.3 * coordinates[:, 1]
        + 0.2 * coordinates.prod(dim=1)
    )
    native = base + (parent + pedal.unsqueeze(1) * direction) @ _decoder()
    gradients = torch.zeros_like(base)
    gradients[:, :_RANK] = torch.stack(
        (
            parent[:, 0] + 0.1 * parent[:, 2],
            parent[:, 1],
            parent[:, 2],
            parent[:, 3] - 0.1 * parent[:, 1],
        ),
        dim=1,
    )
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"finite-microstep-example-{index}",
        family_id=f"finite-microstep-family-{index // 2}",
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
def fitted() -> tuple[
    tuple[AutonomousCompleteH4TrainingSequence, ...],
    object,
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
            -0.015,
            0.025,
            base_provider.direction_left.numel(),
            dtype=torch.float64,
        ).reshape_as(base_provider.direction_left),
        direction_right=base_provider.direction_right
        + torch.linspace(
            0.02,
            -0.01,
            base_provider.direction_right.numel(),
            dtype=torch.float64,
        ).reshape_as(base_provider.direction_right),
        pedal_weight=torch.tensor((0.8, -0.5, 0.3), dtype=torch.float64),
        pedal_bias=torch.tensor((0.2,), dtype=torch.float64),
        fit_protocol_sha256=_V19_PROTOCOL,
        fit_evidence_sha256="c" * 64,
    )
    return sequences, start, base_provider, proposal_provider


def _build(
    base: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    proposal: AutonomousCompleteH4FisherFiniteJointPedalProvider,
    *,
    path: str,
    alpha: float,
    evidence: str = _V20_EVIDENCE,
) -> FisherFiniteMicrostepResult:
    return build_autonomous_complete_h4_fisher_finite_microstep(
        base,
        proposal,
        microstep_path=path,
        alpha=alpha,
        microstep_protocol_sha256=_V20_PROTOCOL,
        microstep_evidence_sha256=evidence,
    )


def test_alpha_zero_is_bitwise_base_identity_for_every_path(fitted) -> None:
    _sequences, _start, base, proposal = fitted
    for path in sorted(FISHER_FINITE_MICROSTEP_PATHS):
        parameters = interpolate_fisher_finite_microstep_parameters(
            base, proposal, microstep_path=path, alpha=0.0
        )
        result = _build(base, proposal, path=path, alpha=0.0)
        assert result.provider is base
        for name in (
            "direction_left",
            "direction_right",
            "pedal_weight",
            "pedal_bias",
        ):
            assert torch.equal(getattr(parameters, name), getattr(base, name))
            assert torch.equal(getattr(result.provider, name), getattr(base, name))
        assert result.receipt.alpha == 0.0
        assert result.receipt.microstep_path == path
        assert dict(result.receipt.selected_tensor_sha256s) == (
            fisher_finite_microstep_selected_tensor_sha256s(result.provider)
        )


def test_alpha_one_joint_is_bitwise_proposal_identity(fitted) -> None:
    _sequences, _start, base, proposal = fitted
    parameters = interpolate_fisher_finite_microstep_parameters(
        base, proposal, microstep_path="joint", alpha=1.0
    )
    result = _build(base, proposal, path="joint", alpha=1.0)
    assert result.provider is proposal
    for name in (
        "direction_left",
        "direction_right",
        "pedal_weight",
        "pedal_bias",
    ):
        assert torch.equal(getattr(parameters, name), getattr(proposal, name))
        assert torch.equal(getattr(result.provider, name), getattr(proposal, name))


def test_factor_space_interpolation_is_exact_deterministic_and_path_isolated(
    fitted,
) -> None:
    _sequences, _start, base, proposal = fitted
    alpha = 0.25
    direction = _build(base, proposal, path="direction_only", alpha=alpha)
    pedal = _build(base, proposal, path="pedal_only", alpha=alpha)
    joint = _build(base, proposal, path="joint", alpha=alpha)

    expected_left = base.direction_left + alpha * (
        proposal.direction_left - base.direction_left
    )
    expected_right = base.direction_right + alpha * (
        proposal.direction_right - base.direction_right
    )
    expected_weight = base.pedal_weight + alpha * (
        proposal.pedal_weight - base.pedal_weight
    )
    expected_bias = base.pedal_bias + alpha * (
        proposal.pedal_bias - base.pedal_bias
    )
    for selected in (direction.provider, joint.provider):
        assert torch.equal(selected.direction_left, expected_left)
        assert torch.equal(selected.direction_right, expected_right)
    for selected in (pedal.provider, joint.provider):
        assert torch.equal(selected.pedal_weight, expected_weight)
        assert torch.equal(selected.pedal_bias, expected_bias)
    assert torch.equal(direction.provider.pedal_weight, base.pedal_weight)
    assert torch.equal(direction.provider.pedal_bias, base.pedal_bias)
    assert torch.equal(pedal.provider.direction_left, base.direction_left)
    assert torch.equal(pedal.provider.direction_right, base.direction_right)

    repeated = _build(base, proposal, path="joint", alpha=alpha)
    assert repeated.provider.artifact_sha256 == joint.provider.artifact_sha256
    assert repeated.receipt.metadata() == joint.receipt.metadata()
    # The primitive follows the supplied factor-space direction rather than an
    # SVD retraction of an interpolated dense product.  The live V20a harness
    # separately authenticates that its supplied endpoint is V19 Adam step 1.
    assert joint.receipt.metadata()["runtime_provider_type"] == (
        "AutonomousCompleteH4FisherFiniteJointPedalProvider"
    )


def test_negative_alpha_uses_the_same_positive_proposal_affine_direction(
    fitted,
) -> None:
    _sequences, _start, base, proposal = fitted
    alpha = -0.25
    parameters = interpolate_fisher_finite_microstep_parameters(
        base,
        proposal,
        microstep_path="joint",
        alpha=alpha,
    )
    assert torch.equal(
        parameters.direction_left,
        base.direction_left
        + alpha * (proposal.direction_left - base.direction_left),
    )
    assert torch.equal(
        parameters.direction_right,
        base.direction_right
        + alpha * (proposal.direction_right - base.direction_right),
    )
    assert torch.equal(
        parameters.pedal_weight,
        base.pedal_weight
        + alpha * (proposal.pedal_weight - base.pedal_weight),
    )
    assert torch.equal(
        parameters.pedal_bias,
        base.pedal_bias + alpha * (proposal.pedal_bias - base.pedal_bias),
    )
    assert parameters.proposal_provider_artifact_sha256 == proposal.artifact_sha256
    result = _build(base, proposal, path="joint", alpha=alpha)
    assert (
        result.receipt.proposal_provider_artifact_sha256
        == proposal.artifact_sha256
    )
    assert result.receipt.alpha == alpha


@pytest.mark.parametrize("path", sorted(FISHER_FINITE_MICROSTEP_PATHS))
@pytest.mark.parametrize(
    "alpha", (-1.0, -0.5, -(2.0**-8), 2.0**-8, 2.0**-4, 0.5, 1.0)
)
def test_microsteps_remain_finite_pointwise_trusted_and_resource_matched(
    fitted, path: str, alpha: float
) -> None:
    sequences, _start, base, proposal = fitted
    result = _build(base, proposal, path=path, alpha=alpha)
    assert isinstance(result.provider, AutonomousCompleteH4FisherFiniteJointPedalProvider)
    assert result.provider.prepared_float_scalar_count == (
        base.prepared_float_scalar_count
    )
    assert result.provider.logical_macs_per_token_upper_bound == (
        base.logical_macs_per_token_upper_bound
    )
    assert result.receipt.prepared_float_scalar_count == (
        base.prepared_float_scalar_count
    )
    assert result.receipt.logical_macs_per_token_upper_bound == (
        base.logical_macs_per_token_upper_bound
    )
    replay = replay_autonomous_complete_h4_fisher_finite_joint_pedal(
        result.provider, sequences[0]
    )
    metadata = replay.metadata()
    assert metadata["pointwise_trust_certificate_passed"] is True
    assert metadata["max_bounded_direction_to_parent_norm_ratio"] <= 0.25 + 1e-14
    assert metadata["max_emitted_delta_to_parent_norm_ratio"] <= 0.25 + 1e-14
    for value in (
        replay.unbounded_direction,
        replay.bounded_direction,
        replay.pedal,
        replay.emitted_delta,
    ):
        assert bool(torch.isfinite(value).all())


@pytest.mark.parametrize(
    "alpha,error",
    (
        (-1.01, ValueError),
        (1.01, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
    ),
)
def test_microstep_alpha_is_strict(fitted, alpha: float, error: type[Exception]) -> None:
    _sequences, _start, base, proposal = fitted
    with pytest.raises(error, match="alpha"):
        interpolate_fisher_finite_microstep_parameters(
            base, proposal, microstep_path="joint", alpha=alpha
        )
    with pytest.raises(ValueError, match="path"):
        interpolate_fisher_finite_microstep_parameters(
            base, proposal, microstep_path="not-a-path", alpha=0.5
        )


def test_receipt_binds_path_alpha_endpoints_protocol_and_evidence(fitted) -> None:
    _sequences, _start, base, proposal = fitted
    reference = _build(base, proposal, path="joint", alpha=0.25)
    changed_path = _build(base, proposal, path="direction_only", alpha=0.25)
    changed_alpha = _build(base, proposal, path="joint", alpha=0.125)
    changed_evidence = _build(
        base, proposal, path="joint", alpha=0.25, evidence="3" * 64
    )
    artifacts = {
        reference.artifact_sha256,
        changed_path.artifact_sha256,
        changed_alpha.artifact_sha256,
        changed_evidence.artifact_sha256,
    }
    assert len(artifacts) == 4
    assert reference.receipt.base_provider_artifact_sha256 == base.artifact_sha256
    assert (
        reference.receipt.proposal_provider_artifact_sha256
        == proposal.artifact_sha256
    )
    metadata = reference.receipt.metadata()
    assert metadata["endpoint_tensors_gradients_or_optimizer_state_serialized"] is False
    assert all(not isinstance(value, torch.Tensor) for value in metadata.values())


def test_selected_v19_state_and_receipt_roundtrip_fail_closed(fitted) -> None:
    _sequences, start, base, proposal = fitted
    result = _build(base, proposal, path="joint", alpha=0.125)
    state = autonomous_complete_h4_fisher_finite_microstep_state_dict(result)
    assert set(state) == {
        "schema",
        "format_version",
        "receipt",
        "selected_provider_state",
    }
    assert "base_provider_state" not in state
    assert "proposal_provider_state" not in state
    assert "optimizer_state" not in state
    restored = autonomous_complete_h4_fisher_finite_microstep_from_state_dict(
        state,
        expected_artifact_sha256=result.artifact_sha256,
        expected_bridge_binding_sha256=_BRIDGE,
        expected_start_provider_artifact_sha256=start.artifact_sha256,
        expected_base_provider_artifact_sha256=base.artifact_sha256,
        expected_proposal_provider_artifact_sha256=proposal.artifact_sha256,
    )
    assert restored.receipt.metadata() == result.receipt.metadata()
    assert restored.provider.metadata() == result.provider.metadata()

    tampered = copy.deepcopy(state)
    tampered["receipt"]["alpha"] = 0.25
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        autonomous_complete_h4_fisher_finite_microstep_from_state_dict(
            tampered,
            expected_artifact_sha256=result.artifact_sha256,
        )
    tampered = copy.deepcopy(state)
    tampered["selected_provider_state"]["tensors"]["pedal_bias"][0] += 1.0
    with pytest.raises(ValueError, match="fit receipt hash mismatch"):
        autonomous_complete_h4_fisher_finite_microstep_from_state_dict(
            tampered,
            expected_artifact_sha256=result.artifact_sha256,
        )
    with pytest.raises(ValueError, match="proposal provider differs"):
        autonomous_complete_h4_fisher_finite_microstep_from_state_dict(
            state,
            expected_artifact_sha256=result.artifact_sha256,
            expected_proposal_provider_artifact_sha256="f" * 64,
        )
