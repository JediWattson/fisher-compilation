from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import pytest
import torch
from torch import Tensor, nn

import fisher_graph.gemma3_l3_l4_manifold_lift as lift_module
from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.gemma3_l3_l4_basis_package import _build_package
from fisher_graph.gemma3_l3_l4_manifold_lift import (
    MANIFOLD_LIFT_FORMULA_VERSION,
    lift_synthetic_reference_batch_to_gemma3_manifold,
)
from fisher_graph.gemma3_l3_l4_spectral_mapping_experiment import (
    Gemma3L3L4SpectralReference,
)
from fisher_graph.gemma3_l3_l4_synthetic_materialization import (
    materialize_synthetic_reference_batches,
)
from fisher_graph.gemma3_l3_l4_synthetic_reference_protocol import (
    SyntheticReferenceProbe,
    default_synthetic_reference_protocol,
)


_WIDTH = 64
_NULL_INDEX = 13
_EPSILON = 1e-6


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gain() -> Tensor:
    gain = torch.linspace(1.0, 2.0, _WIDTH, dtype=torch.float64)
    gain[_NULL_INDEX] = 0.0
    return gain


def _basis():
    identity = torch.eye(_WIDTH, dtype=torch.float64)
    direction = torch.tensor(
        [0.2 if index % 2 == 0 else -0.2 for index in range(_WIDTH)],
        dtype=torch.float64,
    )
    direction[_NULL_INDEX] = 0.0
    x3_mean = _gain() * direction
    reference = Gemma3L3L4SpectralReference(
        hierarchy_artifact_sha256=_sha("hierarchy"),
        source_model_sha256=_sha("model"),
        base_artifact_file_sha256=_sha("base-file"),
        base_scientific_payload_sha256=_sha("base-payload"),
        refit_artifact_file_sha256=_sha("refit-file"),
        refit_scientific_payload_sha256=_sha("refit-payload"),
        generator_plan_sha256s=(_sha("plan-0"), _sha("plan-1")),
        layer3_factor_sha256=_sha("factor-3"),
        layer4_factor_sha256=_sha("factor-4"),
        x3_mean=x3_mean,
        y3_mean=torch.zeros(_WIDTH, dtype=torch.float64),
        x4_mean=torch.zeros(_WIDTH, dtype=torch.float64),
        y4_mean=torch.zeros(_WIDTH, dtype=torch.float64),
        R3=identity,
        P3=identity,
        R4=identity,
        P4=identity,
        S4=torch.linspace(
            float(_WIDTH),
            1.0,
            _WIDTH,
            dtype=torch.float64,
        ),
        x3_covariance=identity,
        upstream_mean_prompt_local_kernel=torch.zeros(
            (1, 1, 1),
            dtype=torch.float64,
        ),
    )
    return _build_package(reference)


class _UnitOffsetNorm(nn.Module):
    def __init__(
        self,
        *,
        gain: Tensor | None = None,
        epsilon: float = _EPSILON,
    ) -> None:
        super().__init__()
        value = _gain() if gain is None else gain
        self.weight = nn.Parameter(value - 1.0, requires_grad=False)
        self.variance_epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        normalized = values * torch.rsqrt(
            values.square().mean(dim=-1, keepdim=True)
            + self.variance_epsilon
        )
        return normalized * (1.0 + self.weight)


class _WrongUnitOffsetNorm(_UnitOffsetNorm):
    def forward(self, values: Tensor) -> Tensor:
        return super().forward(values) * 0.9


class _BFloatUnitOffsetNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            (_gain() - 1.0).to(dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.variance_epsilon = _EPSILON

    def forward(self, values: Tensor) -> Tensor:
        values32 = values.float()
        normalized = values32 * torch.rsqrt(
            values32.square().mean(dim=-1, keepdim=True)
            + self.variance_epsilon
        )
        return (
            normalized * (1.0 + self.weight.float())
        ).to(dtype=values.dtype)


def _batch_and_exact_specs(
    probes: tuple[SyntheticReferenceProbe, ...],
):
    protocol = default_synthetic_reference_protocol()
    (batch,) = materialize_synthetic_reference_batches(protocol, probes)
    by_id = {probe.probe_id: probe for probe in probes}
    exact = tuple(by_id[probe_id] for probe_id in batch.probe_ids)
    return batch, exact


def _lift(
    probes: tuple[SyntheticReferenceProbe, ...],
    *,
    basis=None,
    norm: nn.Module | None = None,
):
    batch, exact = _batch_and_exact_specs(probes)
    package = _basis() if basis is None else basis
    module = _UnitOffsetNorm().eval() if norm is None else norm
    result = lift_synthetic_reference_batch_to_gemma3_manifold(
        package,
        module,
        epsilon=_EPSILON,
        batch=batch,
        probes=exact,
    )
    return result, batch, exact, package, module


def test_option_b_lift_matches_exact_direction_and_live_norm_formula() -> None:
    protocol = default_synthetic_reference_protocol()
    probe = next(
        value
        for value in protocol.probes
        if value.family == "rademacher"
        and value.sequence_length == 32
        and value.source_offset > 0
    )
    result, batch, exact, basis, norm = _lift((probe,))
    repeated, *_ = _lift((probe,))

    assert result.formula_version == MANIFOLD_LIFT_FORMULA_VERSION
    assert result.null_gain_indices == (_NULL_INDEX,)
    assert result.requested_seeds is result.requested_standardized_modes
    assert result.requested_standardized_modes.shape == (1, 32, 64)
    assert result.hidden_states.shape == (1, 32, _WIDTH)
    assert result.actual_x3.shape == (1, 32, _WIDTH)
    assert result.absolute_realized_standardized_modes.shape == (
        1,
        32,
        64,
    )
    assert result.normalized_null_features.shape == (1, 32, 1)
    assert result.active_mask.dtype == torch.bool
    assert torch.equal(
        result.requested_standardized_modes,
        batch.values,
    )

    gain = _gain()
    neutral_pre_norm = torch.zeros(_WIDTH, dtype=torch.float64)
    nonnull = gain != 0.0
    neutral_pre_norm[nonnull] = basis.x3_mean[nonnull] / gain[nonnull]
    neutral = (
        neutral_pre_norm / neutral_pre_norm.square().mean().sqrt()
    )
    neutral[_NULL_INDEX] = 0.0
    torch.testing.assert_close(
        result.neutral_hidden_state,
        neutral,
        rtol=0.0,
        atol=0.0,
    )
    inactive = ~result.active_mask
    neutral_grid = neutral.view(1, 1, -1).expand_as(result.hidden_states)
    assert torch.equal(
        result.hidden_states[inactive],
        neutral_grid[inactive],
    )

    position = probe.source_offset
    requested = batch.values[0, position]
    target = basis.x3_mean + requested
    pre_norm = torch.zeros_like(target)
    pre_norm[nonnull] = target[nonnull] / gain[nonnull]
    direction = pre_norm / pre_norm.square().mean().sqrt()
    expected_hidden = direction * probe.radial_scale
    expected_hidden[_NULL_INDEX] = probe.null_coordinate
    torch.testing.assert_close(
        result.hidden_states[0, position],
        expected_hidden,
        rtol=0.0,
        atol=0.0,
    )
    with torch.no_grad():
        expected_x3 = norm(expected_hidden.view(1, 1, -1))[0, 0]
    torch.testing.assert_close(
        result.actual_x3[0, position],
        expected_x3,
        rtol=0.0,
        atol=0.0,
    )
    assert result.actual_x3[0, position, _NULL_INDEX] == 0.0
    assert result.discarded_requested_x3_null[0, position, 0] == (
        target[_NULL_INDEX]
    )
    expected_absolute = expected_x3 - basis.x3_mean
    expected_delta = expected_x3 - result.neutral_actual_x3
    torch.testing.assert_close(
        result.absolute_realized_standardized_modes[0, position],
        expected_absolute,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.neutral_delta_realized_standardized_modes[0, position],
        expected_delta,
        rtol=0.0,
        atol=0.0,
    )

    expected_rms = math.sqrt(
        probe.radial_scale**2
        + probe.null_coordinate**2 / _WIDTH
    )
    assert result.row_rms[0, position].item() == pytest.approx(
        expected_rms,
        abs=1e-14,
    )
    nonnull_hidden = result.hidden_states[0, position].clone()
    nonnull_hidden[_NULL_INDEX] = 0.0
    assert nonnull_hidden.square().mean().sqrt().item() == pytest.approx(
        probe.radial_scale,
        abs=1e-14,
    )
    expected_null = probe.null_coordinate / math.sqrt(
        expected_rms**2 + _EPSILON
    )
    assert result.normalized_null_features[
        0,
        position,
        0,
    ].item() == pytest.approx(expected_null, abs=1e-14)

    metadata = result.metadata()
    assert metadata["basis_payload_sha256"] == basis.basis_payload_sha256
    assert metadata["norm_module_sha256"] == module_state_fingerprint(norm)
    assert metadata["protocol_sha256"] == batch.protocol_sha256
    assert metadata["materialized_batch_artifact_sha256"] == (
        batch.artifact_sha256
    )
    assert metadata["probe_artifact_sha256s"] == list(
        value.artifact_sha256 for value in exact
    )
    assert result.artifact_sha256 == repeated.artifact_sha256
    assert result.tensor_sha256s == repeated.tensor_sha256s
    result.validate_integrity()


def test_sparse_inactive_rows_never_receive_probe_radial_or_null_state() -> None:
    protocol = default_synthetic_reference_protocol()
    probe = next(
        value
        for value in protocol.probes
        if value.family == "sparse"
        and value.radial_scale != 1.0
        and value.null_coordinate != 0.0
    )
    result, batch, _, _, _ = _lift((probe,))
    active_positions = {
        position for position, _, _ in probe.sparse_coordinates
    }
    expected_mask = torch.tensor(
        [
            position in active_positions
            for position in range(probe.sequence_length)
        ],
        dtype=torch.bool,
    )

    assert torch.equal(result.active_mask[0], expected_mask)
    neutral = result.neutral_hidden_state
    for position in range(probe.sequence_length):
        if position in active_positions:
            assert result.hidden_states[
                0,
                position,
                _NULL_INDEX,
            ] == probe.null_coordinate
            continue
        assert torch.equal(result.hidden_states[0, position], neutral)
        assert result.normalized_null_features[0, position, 0] == 0.0
        assert torch.count_nonzero(batch.values[0, position]).item() == 0
    assert result.diagnostics.inactive_hidden_maximum_abs_difference == 0.0
    assert result.diagnostics.inactive_actual_x3_maximum_abs_difference == 0.0
    assert result.diagnostics.inactive_delta_mode_maximum_abs == 0.0


def test_radial_collisions_share_x3_direction_but_not_hidden_radius() -> None:
    protocol = default_synthetic_reference_protocol()
    groups: dict[str, list[SyntheticReferenceProbe]] = defaultdict(list)
    for probe in protocol.probes:
        if probe.family == "radial_collision":
            assert probe.collision_group is not None
            groups[probe.collision_group].append(probe)
    probes = tuple(next(iter(groups.values())))
    result, _, exact, _, _ = _lift(probes)
    active_position = exact[0].source_offset
    index_by_radial = {
        probe.radial_scale: index for index, probe in enumerate(exact)
    }

    assert all(
        torch.equal(
            result.requested_standardized_modes[0],
            result.requested_standardized_modes[index],
        )
        for index in range(1, 3)
    )
    for radial, index in index_by_radial.items():
        hidden = result.hidden_states[index, active_position]
        assert hidden[_NULL_INDEX] == 0.0
        assert result.row_rms[index, active_position].item() == pytest.approx(
            radial,
            abs=1e-14,
        )
        expected_q = radial**2 / (radial**2 + _EPSILON)
        actual = result.actual_x3[index, active_position]
        quotient = torch.zeros_like(actual)
        quotient[_gain() != 0.0] = (
            actual[_gain() != 0.0] / _gain()[_gain() != 0.0]
        )
        assert quotient.square().mean().item() == pytest.approx(
            expected_q,
            abs=1e-14,
        )
    low = result.actual_x3[index_by_radial[0.5], active_position]
    high = result.actual_x3[index_by_radial[2.0], active_position]
    torch.testing.assert_close(low, high, rtol=2e-6, atol=2e-6)
    assert not torch.equal(
        result.hidden_states[index_by_radial[0.5], active_position],
        result.hidden_states[index_by_radial[2.0], active_position],
    )


def test_null_collisions_preserve_x3_for_opposite_hidden_null_signs() -> None:
    protocol = default_synthetic_reference_protocol()
    groups: dict[str, list[SyntheticReferenceProbe]] = defaultdict(list)
    for probe in protocol.probes:
        if probe.family == "null_collision":
            assert probe.collision_group is not None
            groups[probe.collision_group].append(probe)
    probes = tuple(next(iter(groups.values())))
    result, _, exact, _, _ = _lift(probes)
    position = exact[0].source_offset
    index_by_null = {
        probe.null_coordinate: index for index, probe in enumerate(exact)
    }
    negative = index_by_null[-1.0]
    zero = index_by_null[0.0]
    positive = index_by_null[1.0]

    assert torch.equal(
        result.requested_standardized_modes[negative],
        result.requested_standardized_modes[positive],
    )
    assert torch.equal(
        result.actual_x3[negative, position],
        result.actual_x3[positive, position],
    )
    assert result.hidden_states[negative, position, _NULL_INDEX] == -1.0
    assert result.hidden_states[zero, position, _NULL_INDEX] == 0.0
    assert result.hidden_states[positive, position, _NULL_INDEX] == 1.0
    assert result.normalized_null_features[
        negative,
        position,
        0,
    ] == -result.normalized_null_features[positive, position, 0]
    assert not torch.equal(
        result.hidden_states[negative, position],
        result.hidden_states[positive, position],
    )
    assert torch.count_nonzero(
        result.actual_x3[..., _NULL_INDEX]
    ).item() == 0
    expected_q = 1.0 / (1.0 + 1.0 / _WIDTH + _EPSILON)
    actual = result.actual_x3[positive, position]
    quotient = torch.zeros_like(actual)
    quotient[_gain() != 0.0] = (
        actual[_gain() != 0.0] / _gain()[_gain() != 0.0]
    )
    assert quotient.square().mean().item() == pytest.approx(
        expected_q,
        abs=1e-14,
    )


def test_bfloat16_live_rounding_is_repeatable_and_explicitly_measured() -> None:
    protocol = default_synthetic_reference_protocol()
    probe = next(
        value
        for value in protocol.probes
        if value.family == "rademacher" and value.source_offset > 0
    )
    norm = _BFloatUnitOffsetNorm().eval()
    first, *_ = _lift((probe,), norm=norm)
    second, *_ = _lift((probe,), norm=norm)

    assert first.hidden_states.dtype == torch.float64
    assert first.actual_x3.dtype == torch.float64
    assert torch.equal(first.hidden_states, second.hidden_states)
    assert torch.equal(first.actual_x3, second.actual_x3)
    assert first.tensor_sha256s == second.tensor_sha256s
    assert first.artifact_sha256 == second.artifact_sha256
    assert first.diagnostics.live_vs_analytic_relative_l2 < 5e-3
    assert first.diagnostics.inactive_hidden_maximum_abs_difference == 0.0
    first.validate_integrity()


def test_live_norm_basis_batch_and_probe_authentication_fail_closed() -> None:
    protocol = default_synthetic_reference_protocol()
    probes = tuple(
        probe
        for probe in protocol.probes
        if probe.family == "radial_collision"
        and probe.collision_group == "assessment.radial.mode_00"
    )
    batch, exact = _batch_and_exact_specs(probes)

    with pytest.raises(ValueError, match="exact batch order"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            _basis(),
            _UnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=batch,
            probes=tuple(reversed(exact)),
        )

    mutated_batch, mutated_exact = _batch_and_exact_specs((probes[0],))
    mutated_batch.values[0, probes[0].source_offset, 0] += 1.0
    with pytest.raises(ValueError, match="tensor was mutated"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            _basis(),
            _UnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=mutated_batch,
            probes=mutated_exact,
        )

    tampered_basis = _basis()
    tampered_basis.x3_mean[0] += 0.25
    with pytest.raises(ValueError, match="logical payload hash"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            tampered_basis,
            _UnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=batch,
            probes=exact,
        )

    with pytest.raises(ValueError, match="frozen and in evaluation"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            _basis(),
            _UnitOffsetNorm(),
            epsilon=_EPSILON,
            batch=batch,
            probes=exact,
        )
    wrong_epsilon = _UnitOffsetNorm(epsilon=1e-5).eval()
    with pytest.raises(ValueError, match="epsilon differs"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            _basis(),
            wrong_epsilon,
            epsilon=_EPSILON,
            batch=batch,
            probes=exact,
        )
    with pytest.raises(ValueError, match="not the declared"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            _basis(),
            _WrongUnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=batch,
            probes=exact,
        )
    no_null_gain = torch.ones(_WIDTH, dtype=torch.float64)
    with pytest.raises(ValueError, match="exactly one"):
        lift_synthetic_reference_batch_to_gemma3_manifold(
            _basis(),
            _UnitOffsetNorm(gain=no_null_gain).eval(),
            epsilon=_EPSILON,
            batch=batch,
            probes=exact,
        )


def test_realized_diagnostics_and_tensor_hashes_detect_mutation() -> None:
    protocol = default_synthetic_reference_protocol()
    probe = next(
        value
        for value in protocol.probes
        if value.family == "rademacher" and value.source_offset > 0
    )
    result, _, _, _, _ = _lift((probe,))

    assert result.diagnostics.active_row_count == int(
        result.active_mask.sum()
    )
    assert result.diagnostics.inactive_row_count == int(
        (~result.active_mask).sum()
    )
    assert result.diagnostics.discarded_null_maximum_abs > 0.0
    assert result.diagnostics.active_actual_pre_gain_q_minimum > 0.9
    assert result.diagnostics.live_vs_analytic_maximum_abs < 1e-12
    assert result.diagnostics.requested_vs_realized_relative_error_p90 >= 0.0
    assert result.diagnostics.absolute_realized_mode_l2_maximum > 0.0
    assert set(dict(result.tensor_sha256s)) == {
        "requested_standardized_modes",
        "hidden_states",
        "actual_x3",
        "absolute_realized_standardized_modes",
        "neutral_delta_realized_standardized_modes",
        "row_rms",
        "normalized_null_features",
        "discarded_requested_x3_null",
        "neutral_hidden_state",
        "neutral_actual_x3",
        "active_mask",
    }

    result.actual_x3[0, 0, 0] += 0.25
    with pytest.raises(ValueError, match="tensor was mutated"):
        result.validate_integrity()


def test_manifold_lift_has_no_prompt_model_or_artifact_loader() -> None:
    assert "transformers" not in lift_module.__dict__
    assert "AutoModelForCausalLM" not in lift_module.__dict__
    assert "AutoTokenizer" not in lift_module.__dict__
    assert "Path" not in lift_module.__dict__
    assert "torch_load" not in lift_module.__dict__
