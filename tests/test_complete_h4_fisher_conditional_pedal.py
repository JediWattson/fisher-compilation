from __future__ import annotations

import copy
from dataclasses import replace
import stat

import pytest
import torch

from fisher_graph.complete_h4_autonomous_residual import (
    AutonomousCompleteH4TrainingSequence,
    _bounded_vjp_multipliers,
    fit_autonomous_complete_h4_residual,
)
from fisher_graph.complete_h4_fisher_conditional_pedal import (
    FISHER_XY_PEDAL_TRUST_FRACTION,
    autonomous_complete_h4_fisher_xy_pedal_provider_from_state_dict,
    autonomous_complete_h4_fisher_xy_pedal_provider_state_dict,
    fisher_xy_pedal_fit_support_mask,
    fisher_xy_pointwise_bounded_direction,
    fit_autonomous_complete_h4_fisher_xy_pedal,
    load_autonomous_complete_h4_fisher_xy_pedal_provider,
    replay_autonomous_complete_h4_fisher_xy_pedal,
    save_autonomous_complete_h4_fisher_xy_pedal_provider,
    validate_fisher_xy_pedal_runtime_replay_metadata,
)
from fisher_graph.gemma3_l3_l4_graph_organized_svd_shadow_runtime import (
    Gemma3L3L4OnePassPrefix,
)
from fisher_graph.radial_finite_displacement_correction import (
    family_balanced_row_weights,
)


_BRIDGE = "d" * 64
_WIDTH = 640
_SOURCE_RANK = 64
_RANK = 4


def _decoder() -> torch.Tensor:
    return torch.eye(_WIDTH, dtype=torch.float64)[:_RANK].contiguous()


def _sequence(index: int, *, length: int = 18) -> AutonomousCompleteH4TrainingSequence:
    generator = torch.Generator().manual_seed(17300 + index)
    source = torch.randn(length, _SOURCE_RANK, generator=generator, dtype=torch.float64)
    base = torch.randn(length, _WIDTH, generator=generator, dtype=torch.float64)
    source_kernel = torch.randn(
        _SOURCE_RANK,
        _RANK,
        generator=generator,
        dtype=torch.float64,
    ) * 0.025
    state_kernel = torch.tensor(
        [
            [0.18, -0.03, 0.02, 0.01],
            [0.02, 0.15, -0.04, 0.00],
            [-0.01, 0.03, 0.14, 0.02],
            [0.00, -0.02, 0.04, 0.13],
        ],
        dtype=torch.float64,
    )
    parent_modal = source @ source_kernel + base[:, :_RANK] @ state_kernel
    raw = torch.stack(
        (
            1.1 * parent_modal[:, 0] + 0.35 * parent_modal[:, 2],
            -0.85 * parent_modal[:, 1] + 0.30 * parent_modal[:, 3],
        ),
        dim=1,
    )
    coordinates = raw / (0.65 + raw.abs())
    a_x = torch.tensor(
        [
            [0.55, 0.05, 0.00, 0.00],
            [0.00, -0.45, 0.06, 0.00],
            [0.05, 0.00, 0.40, 0.00],
            [0.00, 0.05, 0.00, -0.35],
        ],
        dtype=torch.float64,
    )
    a_y = torch.tensor(
        [
            [-0.42, 0.00, 0.05, 0.00],
            [0.05, 0.50, 0.00, 0.00],
            [0.00, 0.05, -0.38, 0.04],
            [0.00, 0.00, 0.05, 0.36],
        ],
        dtype=torch.float64,
    )
    a_xy = torch.tensor(
        [
            [0.30, 0.00, 0.00, 0.04],
            [0.00, -0.25, 0.04, 0.00],
            [0.04, 0.00, 0.22, 0.00],
            [0.00, 0.04, 0.00, -0.20],
        ],
        dtype=torch.float64,
    )
    direction = (
        coordinates[:, :1] * (parent_modal @ a_x)
        + coordinates[:, 1:] * (parent_modal @ a_y)
        + coordinates[:, :1] * coordinates[:, 1:] * (parent_modal @ a_xy)
    )
    pedal = (
        0.52
        + 0.32 * coordinates[:, 0]
        - 0.22 * coordinates[:, 1]
        + 0.18 * coordinates[:, 0] * coordinates[:, 1]
    ).clamp(0.0, 1.0)
    native = base + (parent_modal + pedal.unsqueeze(1) * direction) @ _decoder()

    modal_gradient = torch.stack(
        (
            parent_modal[:, 0] + 0.2 * parent_modal[:, 2],
            parent_modal[:, 1] - 0.3 * parent_modal[:, 3],
            0.2 * parent_modal[:, 0] + 0.1 * parent_modal[:, 2],
            -0.1 * parent_modal[:, 1] + 0.2 * parent_modal[:, 3],
        ),
        dim=1,
    )
    gradients = torch.zeros((length, _WIDTH), dtype=torch.float64)
    gradients[:, :_RANK] = modal_gradient
    mask = torch.ones(length, dtype=torch.bool)
    return AutonomousCompleteH4TrainingSequence(
        example_id=f"pedal-example-{index}",
        family_id=f"pedal-family-{index % 3}",
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
    providers = {
        (objective, mode): fit_autonomous_complete_h4_fisher_xy_pedal(
            sequences=sequences,
            parent_provider=parent,
            conditional_rank=_RANK,
            coordinate_objective=objective,
            pedal_mode=mode,
            router_ridge=1.0e-7,
            direction_ridge=1.0e-7,
            pedal_ridge=1.0e-7,
        )
        for objective, mode in (
            ("reverse_vjp_fisher", "conditional"),
            ("reverse_vjp_fisher", "constant_optimal"),
            ("reverse_vjp_fisher", "unit"),
            ("activation_pca", "conditional"),
        )
    }
    return sequences, parent, providers


def test_pointwise_clip_never_amplifies_and_handles_exact_zero_rows() -> None:
    parent = torch.tensor(
        [[4.0, 0.0], [4.0, 0.0], [0.0, 0.0], [4.0, 0.0]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [[0.2, 0.0], [3.0, 4.0], [3.0, 4.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    bounded = fisher_xy_pointwise_bounded_direction(parent, direction)
    torch.testing.assert_close(bounded[0], direction[0], rtol=0.0, atol=0.0)
    assert torch.linalg.vector_norm(bounded[1]).item() == pytest.approx(1.0)
    assert torch.count_nonzero(bounded[2:]) == 0
    parent_norm = torch.linalg.vector_norm(parent, dim=1)
    bounded_norm = torch.linalg.vector_norm(bounded, dim=1)
    assert bool(
        (
            bounded_norm
            <= FISHER_XY_PEDAL_TRUST_FRACTION * parent_norm + 1.0e-14
        ).all()
    )


def test_subnormal_fit_directions_are_ignored_and_nonfinite_energy_fails_closed() -> None:
    parent = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    direction = torch.tensor(
        [
            [1.0e-200, 0.0],
            [2.0 * torch.finfo(torch.float64).eps**0.5, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    support = fisher_xy_pedal_fit_support_mask(parent, direction)
    assert torch.equal(
        support,
        torch.tensor([False, True, False], dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="energy became nonfinite"):
        fisher_xy_pedal_fit_support_mask(
            torch.full((1, 2), 1.0e308, dtype=torch.float64),
            torch.ones((1, 2), dtype=torch.float64),
        )


def test_fit_and_runtime_replay_obey_pointwise_trust(fitted) -> None:
    sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    assert provider.fit_max_bounded_direction_ratio <= 0.25 + 1.0e-14
    assert provider.fit_max_emitted_delta_ratio <= 0.25 + 1.0e-14
    assert provider.weighted_bounded_target_rmse_after <= (
        provider.weighted_bounded_target_rmse_before + 1.0e-12
    )
    assert provider.weighted_residual_rmse_after <= (
        provider.weighted_residual_rmse_constant + 1.0e-12
    )
    assert 0.0 <= provider.pedal_min <= provider.pedal_weighted_mean
    assert provider.pedal_weighted_mean <= provider.pedal_max <= 1.0
    assert provider.pedal_weighted_std >= 0.0
    assert 0.0 <= provider.pedal_effective_min
    assert provider.pedal_effective_min <= provider.pedal_effective_weighted_mean
    assert provider.pedal_effective_weighted_mean <= provider.pedal_effective_max
    assert provider.pedal_effective_max <= 1.0
    assert provider.pedal_effective_weighted_std >= 0.0
    replay_zero_directions = 0
    for sequence in sequences:
        replay = replay_autonomous_complete_h4_fisher_xy_pedal(provider, sequence)
        replay.validate_integrity()
        metadata = replay.metadata()
        assert validate_fisher_xy_pedal_runtime_replay_metadata(metadata) == metadata
        assert metadata["max_bounded_direction_to_parent_norm_ratio"] <= 0.25 + 1e-14
        assert metadata["max_emitted_delta_to_parent_norm_ratio"] <= 0.25 + 1e-14
        assert metadata["pointwise_trust_certificate_passed"] is True
        replay_zero_directions += metadata["zero_direction_row_count"]
    assert replay_zero_directions == provider.zero_direction_row_count


def test_fisher_pca_and_all_pedal_controls_are_exactly_resource_matched(fitted) -> None:
    _sequences, parent, providers = fitted
    values = tuple(providers.values())
    assert len({value.incremental_prepared_float_scalar_count for value in values}) == 1
    assert len({value.incremental_logical_macs_per_token_upper_bound for value in values}) == 1
    expected_scalars = 2 * _RANK + 8 + 4 * _RANK * _RANK
    expected_macs = 2 * _RANK + 4 * _RANK * _RANK + 3
    for provider in values:
        assert provider.incremental_prepared_float_scalar_count == expected_scalars
        assert provider.incremental_logical_macs_per_token_upper_bound == expected_macs
        assert provider.prepared_float_scalar_count == (
            parent.prepared_float_scalar_count + expected_scalars
        )
        assert provider.logical_macs_per_token_upper_bound == (
            parent.logical_macs_per_token_upper_bound + expected_macs
        )
        assert provider.metadata()["runtime_state_float_scalars_per_sequence"] == 0

    unit = providers[("reverse_vjp_fisher", "unit")]
    assert unit.pedal_min == unit.pedal_max == 1.0
    assert unit.pedal_weighted_std == 0.0
    assert unit.pedal_effective_min == unit.pedal_effective_max == 1.0
    assert unit.pedal_effective_weighted_std == 0.0

    fisher_controls = tuple(
        providers[("reverse_vjp_fisher", mode)]
        for mode in ("conditional", "constant_optimal", "unit")
    )
    for field in (
        "router_weight",
        "router_bias",
        "coordinate_scales",
        "direction_left",
        "direction_right",
    ):
        reference = getattr(fisher_controls[0], field)
        for provider in fisher_controls[1:]:
            torch.testing.assert_close(
                getattr(provider, field),
                reference,
                rtol=0.0,
                atol=0.0,
            )


def test_mode_tensor_invariants_are_authenticated(fitted) -> None:
    _sequences, _parent, providers = fitted
    unit = providers[("reverse_vjp_fisher", "unit")]
    with pytest.raises(ValueError, match="unit.*tensors differ"):
        replace(
            unit,
            pedal_bias=torch.zeros_like(unit.pedal_bias),
            artifact_sha256="",
        )
    constant = providers[("reverse_vjp_fisher", "constant_optimal")]
    with pytest.raises(ValueError, match="constant-optimal.*tensors differ"):
        replace(
            constant,
            pedal_weight=torch.ones_like(constant.pedal_weight),
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="unit.*diagnostics differ"):
        replace(
            unit,
            pedal_effective_weighted_std=0.5,
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="constant-optimal.*diagnostics differ"):
        replace(
            constant,
            pedal_bias=0.5 * constant.pedal_bias,
            artifact_sha256="",
        )


@pytest.mark.parametrize("mode", ("unit", "constant_optimal", "conditional"))
def test_pedal_statistics_are_numerically_valid_with_variable_sequence_lengths(
    mode: str,
) -> None:
    lengths = (7, 11, 13, 17, 19, 23)
    sequences = tuple(
        _sequence(120 + index, length=length)
        for index, length in enumerate(lengths)
    )
    parent = fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=_decoder(),
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-7,
    )
    provider = fit_autonomous_complete_h4_fisher_xy_pedal(
        sequences=sequences,
        parent_provider=parent,
        conditional_rank=_RANK,
        coordinate_objective="reverse_vjp_fisher",
        pedal_mode=mode,
        router_ridge=1.0e-7,
        direction_ridge=1.0e-7,
        pedal_ridge=1.0e-7,
    )
    if mode == "conditional":
        assert (
            provider.pedal_min
            <= provider.pedal_weighted_mean
            <= provider.pedal_max
        )
        assert (
            provider.pedal_effective_min
            <= provider.pedal_effective_weighted_mean
            <= provider.pedal_effective_max
        )
        assert provider.pedal_weighted_std >= 0.0
        assert provider.pedal_effective_weighted_std >= 0.0
        # A legitimate constant conditional fit is valid core output; the
        # development qualification, rather than construction, rejects it.
        return
    expected = 1.0 if mode == "unit" else float(provider.pedal_bias[0])
    assert provider.pedal_weighted_mean == expected
    assert provider.pedal_weighted_std == 0.0
    assert provider.pedal_min == provider.pedal_max == expected
    assert provider.pedal_effective_weighted_mean == expected
    assert provider.pedal_effective_weighted_std == 0.0
    assert provider.pedal_effective_min == provider.pedal_effective_max == expected
    assert provider.pedal_zero_fraction == (1.0 if expected == 0.0 else 0.0)
    assert provider.pedal_one_fraction == (1.0 if expected == 1.0 else 0.0)


def test_constant_control_is_exact_global_constrained_sse_optimum(fitted) -> None:
    sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "constant_optimal")]
    ordered = tuple(sorted(sequences, key=lambda value: (value.family_id, value.example_id)))
    residual_rows = []
    bounded_rows = []
    gradient_rows = []
    families: list[str] = []
    examples: list[str] = []
    for sequence in ordered:
        replay = replay_autonomous_complete_h4_fisher_xy_pedal(provider, sequence)
        target = (
            sequence.native_h4 - sequence.base_h4
        ) @ provider.parent_provider.output_decoder.T
        residual_rows.append(target[sequence.support_mask] - replay.parent_modal)
        bounded_rows.append(replay.bounded_direction)
        gradient_rows.append(sequence.reverse_vjp_gradients[sequence.support_mask])
        count = int(sequence.support_mask.sum())
        families.extend([sequence.family_id] * count)
        examples.extend([sequence.example_id] * count)
    residual = torch.cat(residual_rows)
    bounded = torch.cat(bounded_rows)
    gradients = torch.cat(gradient_rows)
    base_weights = family_balanced_row_weights(tuple(families), tuple(examples))
    multipliers = _bounded_vjp_multipliers(
        gradients,
        tuple(examples),
        floor=0.5,
        ceiling=2.0,
    )
    fit_weights = base_weights * multipliers
    fit_weights = fit_weights / fit_weights.sum()
    expected = (
        (fit_weights * (residual * bounded).sum(dim=1)).sum()
        / (fit_weights * bounded.square().sum(dim=1)).sum()
    ).clamp(0.0, 1.0)
    assert provider.pedal_bias[0].item() == pytest.approx(float(expected), abs=1e-12)
    assert provider.pedal_bias[0].item() == pytest.approx(
        min(1.0, max(0.0, provider.pedal_unclipped_target_weighted_mean)),
        abs=1e-12,
    )
    assert torch.count_nonzero(provider.pedal_weight) == 0


def test_zero_pedal_abstains_exactly(fitted) -> None:
    sequences, parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    abstaining = replace(
        provider,
        pedal_weight=torch.zeros_like(provider.pedal_weight),
        pedal_bias=torch.zeros_like(provider.pedal_bias),
        pedal_mode="conditional",
        artifact_sha256="",
    )
    sequence = sequences[0]
    prefix = _prefix(sequence)
    realized = sequence.base_h4.unsqueeze(0)
    expected = parent.modal_correction(prefix, realized)
    actual = abstaining.modal_correction(prefix, realized)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    replay = replay_autonomous_complete_h4_fisher_xy_pedal(abstaining, sequence)
    assert torch.count_nonzero(replay.pedal) == 0
    assert torch.count_nonzero(replay.emitted_delta) == 0


def test_runtime_is_causal_supported_and_source_free(fitted) -> None:
    sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    sequence = sequences[1]
    prefix = _prefix(sequence)
    correction = provider.correction(prefix, sequence.base_h4.unsqueeze(0))
    support = prefix.complete_h4_causal_support_mask()
    assert torch.count_nonzero(correction[~support]) == 0
    assert provider.metadata()["runtime_forbidden_inputs"] == (
        "native_h4",
        "targets",
        "logits",
        "gradients",
        "coordinate_axes",
        "family_ids",
    )

    cutoff = 7
    changed_source = sequence.source_modes.clone()
    changed_source[cutoff + 1 :] += 500.0
    changed_base = sequence.base_h4.clone()
    changed_base[cutoff + 1 :] -= 500.0
    changed = AutonomousCompleteH4TrainingSequence(
        example_id="future-changed",
        family_id="held-family",
        source_modes=changed_source,
        logical_positions=sequence.logical_positions,
        valid_mask=sequence.valid_mask,
        source_mask=sequence.source_mask,
        support_mask=sequence.support_mask,
        base_h4=changed_base,
        native_h4=changed_base,
        reverse_vjp_gradients=sequence.reverse_vjp_gradients,
    )
    changed_correction = provider.correction(
        _prefix(changed),
        changed.base_h4.unsqueeze(0),
    )
    torch.testing.assert_close(
        changed_correction[:, : cutoff + 1],
        correction[:, : cutoff + 1],
        rtol=0.0,
        atol=0.0,
    )


def test_replay_ignores_native_h4_reverse_vjp_and_identity_fields(fitted) -> None:
    sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    sequence = sequences[2]
    reference = replay_autonomous_complete_h4_fisher_xy_pedal(provider, sequence)
    generator = torch.Generator().manual_seed(5543)
    changed = AutonomousCompleteH4TrainingSequence(
        example_id="forbidden-changed",
        family_id="unseen-family",
        source_modes=sequence.source_modes,
        logical_positions=sequence.logical_positions,
        valid_mask=sequence.valid_mask,
        source_mask=sequence.source_mask,
        support_mask=sequence.support_mask,
        base_h4=sequence.base_h4,
        native_h4=torch.randn(
            sequence.native_h4.shape,
            generator=generator,
            dtype=torch.float64,
        ),
        reverse_vjp_gradients=None,
    )
    actual = replay_autonomous_complete_h4_fisher_xy_pedal(provider, changed)
    assert actual.sequence_artifact_sha256 == changed.artifact_sha256
    assert reference.sequence_artifact_sha256 == sequence.artifact_sha256
    assert actual.sequence_artifact_sha256 != reference.sequence_artifact_sha256
    for name in (
        "parent_modal",
        "bounded_coordinates",
        "unbounded_direction",
        "bounded_direction",
        "pedal",
        "emitted_delta",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(reference, name),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    "field_name",
    ("base_h4", "native_h4", "reverse_vjp_gradients"),
)
def test_replay_rejects_in_place_training_sequence_payload_drift(
    fitted,
    field_name: str,
) -> None:
    _sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    sequence = _sequence(80)
    tensor = getattr(sequence, field_name)
    assert tensor is not None
    tensor[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="training sequence payload drifted"):
        replay_autonomous_complete_h4_fisher_xy_pedal(provider, sequence)


@pytest.mark.parametrize("field_name", ("native_h4", "reverse_vjp_gradients"))
def test_fit_rejects_in_place_training_sequence_payload_drift(
    field_name: str,
) -> None:
    sequences = tuple(_sequence(90 + index) for index in range(6))
    parent = fit_autonomous_complete_h4_residual(
        sequences=sequences,
        output_decoder=_decoder(),
        bridge_binding_sha256=_BRIDGE,
        lag_count=1,
        ridge=1.0e-7,
    )
    tensor = getattr(sequences[0], field_name)
    assert tensor is not None
    tensor[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="training sequence payload drifted"):
        fit_autonomous_complete_h4_fisher_xy_pedal(
            sequences=sequences,
            parent_provider=parent,
            conditional_rank=_RANK,
        )


def test_replay_binds_provider_and_rejects_a_forged_trust_escape(fitted) -> None:
    sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    replay = replay_autonomous_complete_h4_fisher_xy_pedal(provider, sequences[0])
    assert replay.provider_artifact_sha256 == provider.artifact_sha256
    assert replay.parent_provider_artifact_sha256 == (
        provider.parent_provider.artifact_sha256
    )
    assert replay.sequence_artifact_sha256 == sequences[0].artifact_sha256
    assert replay.trust_fraction == 0.25
    forged_bounded = replay.parent_modal.clone()
    forged_delta = replay.pedal.unsqueeze(1) * forged_bounded
    with pytest.raises(ValueError, match="escaped pointwise trust"):
        replace(
            replay,
            bounded_direction=forged_bounded,
            emitted_delta=forged_delta,
            artifact_sha256="",
        )
    with pytest.raises(ValueError, match="must be 0.25"):
        replace(replay, trust_fraction=0.5, artifact_sha256="")
    tampered_metadata = replay.metadata()
    tampered_metadata["pedal_mean"] = 0.0
    with pytest.raises(ValueError, match="range differs|artifact hash mismatch"):
        validate_fisher_xy_pedal_runtime_replay_metadata(tampered_metadata)


def test_fit_ownership_is_exact(fitted) -> None:
    sequences, parent, _providers = fitted
    with pytest.raises(ValueError, match="ownership"):
        fit_autonomous_complete_h4_fisher_xy_pedal(
            sequences=sequences[:-1],
            parent_provider=parent,
            conditional_rank=_RANK,
        )


def test_authenticated_state_tamper_and_secure_file_roundtrip(fitted, tmp_path) -> None:
    _sequences, _parent, providers = fitted
    provider = providers[("reverse_vjp_fisher", "conditional")]
    state = autonomous_complete_h4_fisher_xy_pedal_provider_state_dict(provider)
    restored = autonomous_complete_h4_fisher_xy_pedal_provider_from_state_dict(
        state,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_bridge_binding_sha256=_BRIDGE,
    )
    assert restored.metadata() == provider.metadata()

    tampered = copy.deepcopy(state)
    tampered["tensors"]["pedal_weight"][0] += 1.0
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        autonomous_complete_h4_fisher_xy_pedal_provider_from_state_dict(
            tampered,
            expected_artifact_sha256=provider.artifact_sha256,
            expected_bridge_binding_sha256=_BRIDGE,
        )

    path = tmp_path / "fisher-xy-pedal.pt"
    receipt = save_autonomous_complete_h4_fisher_xy_pedal_provider(provider, path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert receipt["file_mode_octal"] == "0600"
    loaded = load_autonomous_complete_h4_fisher_xy_pedal_provider(
        path,
        expected_artifact_sha256=provider.artifact_sha256,
        expected_file_sha256=receipt["file_sha256"],
        expected_bridge_binding_sha256=_BRIDGE,
    )
    assert loaded.metadata() == provider.metadata()
    with pytest.raises(FileExistsError, match="overwrite"):
        save_autonomous_complete_h4_fisher_xy_pedal_provider(provider, path)
