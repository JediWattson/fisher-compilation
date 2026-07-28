from __future__ import annotations

import hashlib
import math

import pytest
import torch
from torch import Tensor, nn

from fisher_graph.adapters import module_state_fingerprint
from fisher_graph.gemma3_l3_l4_basis_package import _build_package
from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_lift import (
    FROZEN_PROVIDER_ASSESSMENT_V3_LIFT_FORMULA_VERSION,
    lift_frozen_provider_assessment_v3_batch,
)
from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_materialization import (
    MaterializedV3Batch,
    materialize_v3_panel,
)
from fisher_graph.gemma3_l3_l4_frozen_provider_assessment_v3_protocol import (
    V3ProbeSpec,
    default_v3_assessment_protocol,
)
from fisher_graph.gemma3_l3_l4_spectral_mapping_experiment import (
    Gemma3L3L4SpectralReference,
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
    def __init__(self, *, epsilon: float = _EPSILON) -> None:
        super().__init__()
        self.weight = nn.Parameter(_gain() - 1.0, requires_grad=False)
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


def _batch_with_length(
    length: int,
) -> tuple[MaterializedV3Batch, tuple[V3ProbeSpec, ...]]:
    protocol = default_v3_assessment_protocol()
    batch = next(
        value
        for value in materialize_v3_panel(protocol)
        if value.sequence_length == length
    )
    by_id = {probe.probe_id: probe for probe in protocol.probes}
    probes = tuple(by_id[probe_id] for probe_id in batch.probe_ids)
    return batch, probes


def _lift(length: int):
    batch, probes = _batch_with_length(length)
    basis = _basis()
    norm = _UnitOffsetNorm().eval()
    result = lift_frozen_provider_assessment_v3_batch(
        basis,
        norm,
        epsilon=_EPSILON,
        batch=batch,
        probes=probes,
    )
    return result, batch, probes, basis, norm


def test_v3_lift_matches_exact_v2_option_b_formula() -> None:
    result, batch, probes, basis, norm = _lift(48)
    repeated, *_ = _lift(48)

    assert (
        result.formula_version
        == FROZEN_PROVIDER_ASSESSMENT_V3_LIFT_FORMULA_VERSION
    )
    assert result.null_gain_indices == (_NULL_INDEX,)
    assert result.requested_seeds is result.requested_standardized_modes
    assert result.requested_standardized_modes.shape == (2, 48, 64)
    assert result.hidden_states.shape == (2, 48, 64)
    assert result.absolute_realized_standardized_modes.shape == (2, 48, 64)
    assert result.normalized_null_features.shape == (2, 48, 1)
    assert torch.equal(result.requested_standardized_modes, batch.values)

    gain = _gain()
    nonnull = gain != 0.0
    neutral_pre_norm = torch.zeros(_WIDTH, dtype=torch.float64)
    neutral_pre_norm[nonnull] = basis.x3_mean[nonnull] / gain[nonnull]
    neutral = neutral_pre_norm / neutral_pre_norm.square().mean().sqrt()
    neutral[_NULL_INDEX] = 0.0
    torch.testing.assert_close(
        result.neutral_hidden_state,
        neutral,
        rtol=0.0,
        atol=0.0,
    )

    probe_index = 0
    probe = probes[probe_index]
    position = probe.source_offset
    requested = batch.values[probe_index, position]
    target = basis.x3_mean + requested
    pre_norm = torch.zeros_like(target)
    pre_norm[nonnull] = target[nonnull] / gain[nonnull]
    direction = pre_norm / pre_norm.square().mean().sqrt()
    expected_hidden = direction * batch.radial_scales[probe_index]
    expected_hidden[_NULL_INDEX] = batch.null_coordinates[probe_index]
    torch.testing.assert_close(
        result.hidden_states[probe_index, position],
        expected_hidden,
        rtol=0.0,
        atol=0.0,
    )
    with torch.no_grad():
        expected_x3 = norm(expected_hidden.view(1, 1, -1))[0, 0]
    torch.testing.assert_close(
        result.actual_x3[probe_index, position],
        expected_x3,
        rtol=0.0,
        atol=0.0,
    )
    expected_absolute = expected_x3 - basis.x3_mean
    expected_delta = expected_x3 - result.neutral_actual_x3
    torch.testing.assert_close(
        result.absolute_realized_standardized_modes[probe_index, position],
        expected_absolute,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        result.neutral_delta_realized_standardized_modes[
            probe_index,
            position,
        ],
        expected_delta,
        rtol=0.0,
        atol=0.0,
    )

    expected_rms = math.sqrt(
        probe.radial_scale**2
        + probe.null_coordinate**2 / _WIDTH
    )
    assert result.row_rms[probe_index, position].item() == pytest.approx(
        expected_rms,
        abs=1e-14,
    )
    expected_null = probe.null_coordinate / math.sqrt(
        expected_rms**2 + _EPSILON
    )
    assert result.normalized_null_features[
        probe_index,
        position,
        0,
    ].item() == pytest.approx(expected_null, abs=1e-14)
    assert result.actual_x3[probe_index, position, _NULL_INDEX] == 0.0

    metadata = result.metadata()
    assert metadata["basis_payload_sha256"] == basis.basis_payload_sha256
    assert metadata["norm_module_sha256"] == module_state_fingerprint(norm)
    assert metadata["protocol_sha256"] == batch.protocol_sha256
    assert metadata["panel_spec_sha256"] == batch.panel_spec_sha256
    assert metadata["materialized_batch_artifact_sha256"] == (
        batch.artifact_sha256
    )
    assert result.artifact_sha256 == repeated.artifact_sha256
    assert result.tensor_sha256s == repeated.tensor_sha256s
    result.validate_integrity()


def test_v3_null_sign_variants_preserve_x3_and_expose_hidden_invariance() -> None:
    result, _, probes, _, _ = _lift(40)
    null_group = next(
        probe.contrast_group
        for probe in probes
        if probe.family == "null_single_invariance"
    )
    variants = {
        probe.null_coordinate: index
        for index, probe in enumerate(probes)
        if probe.contrast_group == null_group
    }
    negative = variants[-0.75]
    zero = variants[0.0]
    positive = variants[0.75]
    position = probes[negative].source_offset

    assert torch.equal(
        result.requested_standardized_modes[negative],
        result.requested_standardized_modes[positive],
    )
    assert torch.equal(
        result.actual_x3[negative, position],
        result.actual_x3[positive, position],
    )
    assert result.hidden_states[negative, position, _NULL_INDEX] == -0.75
    assert result.hidden_states[zero, position, _NULL_INDEX] == 0.0
    assert result.hidden_states[positive, position, _NULL_INDEX] == 0.75
    assert result.normalized_null_features[
        negative,
        position,
        0,
    ] == -result.normalized_null_features[positive, position, 0]
    assert torch.count_nonzero(
        result.actual_x3[..., _NULL_INDEX]
    ).item() == 0

    inactive = ~result.active_mask
    neutral = result.neutral_hidden_state.view(1, 1, -1)
    neutral = neutral.expand_as(result.hidden_states)
    assert torch.equal(result.hidden_states[inactive], neutral[inactive])
    assert result.diagnostics.inactive_hidden_maximum_abs_difference == 0.0
    assert result.diagnostics.inactive_actual_x3_maximum_abs_difference == 0.0
    assert result.diagnostics.inactive_delta_mode_maximum_abs == 0.0


def test_v3_lift_authentication_and_integrity_fail_closed() -> None:
    batch, probes = _batch_with_length(40)
    with pytest.raises(ValueError, match="exact batch order"):
        lift_frozen_provider_assessment_v3_batch(
            _basis(),
            _UnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=batch,
            probes=tuple(reversed(probes)),
        )

    tampered = MaterializedV3Batch.from_state_dict(batch.state_dict())
    tampered.values[0, probes[0].source_offset, 0] += 1.0
    with pytest.raises(ValueError, match="tensor hash mismatch"):
        lift_frozen_provider_assessment_v3_batch(
            _basis(),
            _UnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=tampered,
            probes=probes,
        )

    with pytest.raises(ValueError, match="not the declared"):
        lift_frozen_provider_assessment_v3_batch(
            _basis(),
            _WrongUnitOffsetNorm().eval(),
            epsilon=_EPSILON,
            batch=batch,
            probes=probes,
        )

    result, *_ = _lift(40)
    active = torch.nonzero(result.active_mask, as_tuple=False)[0]
    result.hidden_states[int(active[0]), int(active[1]), 0] += 0.25
    with pytest.raises(ValueError):
        result.validate_integrity()
