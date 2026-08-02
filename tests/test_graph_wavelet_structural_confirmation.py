from __future__ import annotations

import json
import math

import pytest
import torch

from fisher_graph.conditional_spectral_generator import (
    ConditionalSpectralGeneratorPlan,
)
from fisher_graph.graph_wavelet_random_partition_confirmation import (
    derive_balanced_random_partition_panel,
)
from fisher_graph.graph_wavelet_structural_confirmation import (
    MAXIMUM_NATIVE_RELATIVE_ERROR,
    evaluate_graph_wavelet_structural_confirmation,
)


_GROUPS = tuple(
    tuple(range(start, start + 8))
    for start in range(0, 64, 8)
)
_ORIGINS = (0, 1, 2)
_SCALES = torch.ones(64, dtype=torch.float64)


def _panel():
    return derive_balanced_random_partition_panel(
        candidate_artifact_sha256="a" * 64,
        parent_basis_sha256="b" * 64,
        native_partition_artifact_sha256="c" * 64,
        native_groups=_GROUPS,
    )


def _plan(
    left_multiplier: float,
    right_multiplier: float | None = None,
    *,
    source_scales: torch.Tensor = _SCALES,
) -> ConditionalSpectralGeneratorPlan:
    right = left_multiplier if right_multiplier is None else right_multiplier
    source_basis = torch.zeros((64, 8), dtype=torch.float64)
    for column, group in enumerate(_GROUPS):
        source_basis[group[0], column] = 1.0
    target_basis = torch.zeros((64, 1), dtype=torch.float64)
    target_basis[0, 0] = 1.0
    cores = torch.zeros((2, 32, 8, 1), dtype=torch.float64)
    amplitudes = torch.arange(1, 9, dtype=torch.float64)
    cores[0, 0, :, 0] = amplitudes * left_multiplier
    cores[1, 0, :, 0] = amplitudes * right
    total = float(cores.square().sum())
    source_singular = torch.zeros(8, dtype=torch.float64)
    source_singular[0] = math.sqrt(total)
    return ConditionalSpectralGeneratorPlan(
        response_binding_sha256="d" * 64,
        fit_weighted_kernels_sha256="e" * 64,
        fit_knot_origins=(0, 2),
        source_scales=source_scales,
        source_basis=source_basis,
        target_basis=target_basis,
        knot_cores=cores,
        source_singular_values=source_singular,
        target_singular_values=torch.tensor(
            [math.sqrt(total)],
            dtype=torch.float64,
        ),
        fft_length=32,
        input_transform="standardized_linear",
        weighted_total_energy=total,
        weighted_retained_energy=total,
        weighted_relative_error=0.0,
        source_parseval_relative_error=0.0,
        target_parseval_relative_error=0.0,
    )


def _responses() -> torch.Tensor:
    native = _plan(1.0)
    return torch.stack(
        tuple(native.weighted_kernel_at_origin(origin) for origin in _ORIGINS),
        dim=1,
    )


def _evaluate(
    *,
    native: ConditionalSpectralGeneratorPlan | None = None,
    controls: tuple[ConditionalSpectralGeneratorPlan, ...] | None = None,
    signed: ConditionalSpectralGeneratorPlan | None = None,
    global_svd: ConditionalSpectralGeneratorPlan | None = None,
    responses: torch.Tensor | None = None,
    scales: torch.Tensor = _SCALES,
    origins: tuple[int, ...] = _ORIGINS,
    groups: tuple[tuple[int, ...], ...] = _GROUPS,
):
    return evaluate_graph_wavelet_structural_confirmation(
        panel=_panel(),
        native_plan=native or _plan(1.0),
        control_plans=controls or (_plan(1.30),) * 63,
        signed_gfa_reference_plan=signed or _plan(1.10),
        global_svd_ceiling_plan=global_svd or _plan(1.0),
        fresh_central_responses=(
            _responses() if responses is None else responses
        ),
        source_scales=scales,
        origins=origins,
        native_groups=groups,
    )


def _assert_no_tensors(value: object) -> None:
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_no_tensors(child)


def test_exact_native_passes_full_primary_gate_and_returns_scalars_only() -> None:
    result = _evaluate(global_svd=_plan(2.0))
    replay = _evaluate(global_svd=_plan(2.0))

    assert result.passed is True
    assert result.native.pooled_relative_error == pytest.approx(0.0)
    assert result.native.pooled_cosine == pytest.approx(1.0)
    assert tuple(
        metric.relative_error for metric in result.native.per_origin
    ) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(
        metric.cosine for metric in result.native.per_origin
    ) == pytest.approx((1.0, 1.0, 1.0))
    assert math.fsum(result.native.native_group_residual_sses) == pytest.approx(
        result.native.pooled_residual_sse
    )
    assert all(
        math.fsum(metric.native_group_residual_sses)
        == pytest.approx(metric.pooled_residual_sse)
        for metric in result.controls
    )
    assert result.random_null_statistics.empirical_p_value == pytest.approx(
        1.0 / 64.0
    )
    assert result.random_null_statistics.family_win_count == 8
    assert result.global_svd_ceiling.pooled_relative_error == pytest.approx(1.0)
    assert result.global_svd_is_descriptive_only is True
    assert result.artifact_sha256 == replay.artifact_sha256
    metadata = result.metadata()
    _assert_no_tensors(metadata)
    json.dumps(metadata, allow_nan=False)


def test_every_origin_relative_gate_can_fail_when_pooled_metric_passes() -> None:
    result = _evaluate(
        native=_plan(1.25, 1.0),
        controls=(_plan(1.60),) * 63,
        signed=_plan(1.40),
    )

    assert result.native.pooled_relative_error < MAXIMUM_NATIVE_RELATIVE_ERROR
    assert result.native.per_origin[0].relative_error == pytest.approx(0.25)
    assert result.native_relative_error_gate_passed is False
    assert result.native_cosine_gate_passed is True
    assert result.signed_gfa_sse_gate_passed is True
    assert result.random_null_gate_passed is True
    assert result.passed is False


def test_native_must_not_lose_to_signed_gfa_even_if_other_gates_pass() -> None:
    result = _evaluate(
        native=_plan(1.10),
        controls=(_plan(1.50),) * 63,
        signed=_plan(1.05),
    )

    assert result.native_relative_error_gate_passed is True
    assert result.native_cosine_gate_passed is True
    assert result.random_null_gate_passed is True
    assert result.signed_gfa_sse_gate_passed is False
    assert result.passed is False


def test_control_metrics_follow_frozen_panel_order() -> None:
    controls = tuple(_plan(1.20 + ordinal / 1000.0) for ordinal in range(63))
    result = _evaluate(controls=controls)
    panel = _panel()

    assert tuple(metric.control_ordinal for metric in result.controls) == tuple(
        range(63)
    )
    assert tuple(
        metric.partition_artifact_sha256 for metric in result.controls
    ) == panel.control_artifact_sha256s
    assert tuple(
        metric.plan_artifact_sha256 for metric in result.controls
    ) == tuple(plan.artifact_sha256 for plan in controls)


def test_rejects_wrong_geometry_origins_groups_controls_or_scales() -> None:
    with pytest.raises(ValueError, match="exactly 63"):
        _evaluate(controls=(_plan(1.3),) * 62)
    with pytest.raises(ValueError, match="shape"):
        _evaluate(responses=torch.ones((64, 3, 31, 64)))
    with pytest.raises(ValueError, match="strictly increasing"):
        _evaluate(origins=(0, 2, 1))
    interleaved_groups = tuple(
        tuple(range(offset, 64, 8)) for offset in range(8)
    )
    with pytest.raises(ValueError, match="frozen null panel"):
        _evaluate(groups=interleaved_groups)
    changed_scales = torch.ones(64, dtype=torch.float64)
    changed_scales[0] = 2.0
    with pytest.raises(ValueError, match="source scales differ"):
        _evaluate(scales=changed_scales)


def test_rejects_zero_energy_or_non_plan_inputs() -> None:
    with pytest.raises(ValueError, match="positive at every origin"):
        _evaluate(responses=torch.zeros((64, 3, 32, 64)))
    with pytest.raises(TypeError, match="every plan"):
        evaluate_graph_wavelet_structural_confirmation(
            panel=_panel(),
            native_plan=object(),  # type: ignore[arg-type]
            control_plans=(_plan(1.3),) * 63,
            signed_gfa_reference_plan=_plan(1.1),
            global_svd_ceiling_plan=_plan(1.0),
            fresh_central_responses=_responses(),
            source_scales=_SCALES,
            origins=_ORIGINS,
            native_groups=_GROUPS,
        )
