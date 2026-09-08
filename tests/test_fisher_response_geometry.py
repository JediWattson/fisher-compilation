from __future__ import annotations

import math

import pytest
import torch

from fisher_graph.fisher_response_geometry import (
    AffineResponseChart,
    FisherProjection,
    central_difference_sweep,
    frozen_scale_control,
    jvp_actions,
    metric_operator_analysis,
    normalization_jvp,
    normalization_update,
)


DTYPE = torch.float64


def _tensor(values: object) -> torch.Tensor:
    return torch.tensor(values, dtype=DTYPE)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-11)


def test_normalization_jvp_matches_dense_analytic_autodiff_and_finite_differences() -> None:
    message = _tensor([0.7, -1.1, 0.3])
    direction = _tensor([0.4, 0.8, -0.6])
    projection = _tensor([[1.2, 0.1, -0.3], [0.2, 0.7, 0.4], [-0.1, 0.6, 1.1], [0.5, -0.2, 0.3]])
    weight = _tensor([0.2, -0.3, 0.4, 0.1])
    epsilon = 0.013
    fn = lambda m: normalization_update(m, projection, weight, epsilon)
    projected = projection @ message
    scale = torch.sqrt(projected.square().mean() + epsilon)
    dense_analytic = torch.diag(1.0 + weight) @ (
        torch.eye(4, dtype=DTYPE) / scale
        - torch.outer(projected, projected) / (4 * scale**3)
    ) @ projection

    dense_autodiff = torch.autograd.functional.jacobian(fn, message)
    actions = jvp_actions(fn, message, torch.eye(3, dtype=DTYPE))
    analytic_action = normalization_jvp(message, direction, projection, weight, epsilon)
    _assert_close(dense_autodiff, dense_analytic)
    _assert_close(actions, dense_analytic)
    _assert_close(analytic_action, dense_analytic @ direction)

    sweep = central_difference_sweep(
        fn, message, direction, (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5), analytic_action
    )
    assert [row["step"] for row in sweep] == [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5]
    assert all(math.isfinite(row["relative_error"]) for row in sweep)
    assert min(row["relative_error"] for row in sweep) < 1.0e-8
    assert sweep[2]["absolute_error"] < sweep[0]["absolute_error"] / 100.0


def test_normalization_radial_epsilon_response_catches_detached_scale_control() -> None:
    message = _tensor([0.8, -1.4, 0.5])
    projection = torch.eye(3, dtype=DTYPE)
    weight = _tensor([0.2, -0.1, 0.3])
    epsilon = 1.0e-3
    scale = torch.sqrt(message.square().mean() + epsilon)
    true_fn = lambda m: normalization_update(m, projection, weight, epsilon)
    wrong_fn = lambda m: frozen_scale_control(m, projection, weight, epsilon)
    _assert_close(true_fn(message), wrong_fn(message))

    true_radial = jvp_actions(true_fn, message, message[:, None])[:, 0]
    wrong_radial = jvp_actions(wrong_fn, message, message[:, None])[:, 0]
    expected = epsilon * (1.0 + weight) * message / scale**3
    _assert_close(true_radial, expected)
    _assert_close(wrong_radial, true_fn(message))
    assert torch.linalg.vector_norm(wrong_radial) > 100 * torch.linalg.vector_norm(true_radial)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gemma_semantics_preserve_projection_then_fp32_norm_then_native_cast(dtype: torch.dtype) -> None:
    message = torch.tensor([0.8, -1.4, 0.5], dtype=dtype)
    projection = torch.tensor([[0.8, 0.3, -0.2], [0.1, 1.2, 0.4], [-0.2, 0.5, 0.7]], dtype=dtype)
    weight = torch.tensor([0.2, -0.1, 0.3], dtype=dtype)
    projected_fp32 = (projection @ message).float()
    expected = (
        projected_fp32
        * torch.rsqrt(projected_fp32.square().mean() + 1.0e-6)
        * (1.0 + weight.float())
    ).to(dtype)
    actual = normalization_update(message, projection, weight, 1.0e-6, semantics="gemma")
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_constant_affine_map_recovers_nonzero_offset_and_constant_partial_jacobians() -> None:
    K = _tensor([[0.3, -0.2], [0.1, 0.6]])
    R = _tensor([[1.2, 0.4], [-0.5, 0.7]])
    offset = _tensor([2.0, -3.0])
    h_anchor = _tensor([0.4, -0.7])
    m_anchor = _tensor([-0.5, 0.8])
    identity = torch.eye(2, dtype=DTYPE)
    phi = lambda h, m: offset + K @ h + R @ m
    chart = AffineResponseChart(
        state_anchor=h_anchor,
        message_anchor=m_anchor,
        beta=phi(h_anchor, m_anchor),
        B=K,
        A=R,
        U=identity,
        Q=identity,
        G=identity,
        state_radius=10.0,
        message_radius=10.0,
    )
    h = _tensor([1.1, 0.2])
    m = _tensor([0.3, -0.4])
    _assert_close(chart.predict_supported(h, m), phi(h, m))
    for h_point, m_point in ((h_anchor, m_anchor), (h, m)):
        _assert_close(jvp_actions(lambda x: phi(x, m_point), h_point, identity), K)
        _assert_close(jvp_actions(lambda x: phi(h_point, x), m_point, identity), R)
    assert not torch.allclose(chart.predict(h, m), R @ m)


def test_bilinear_response_changes_with_state_and_has_known_finite_taylor_remainder() -> None:
    phi = lambda h, m: torch.stack((h[0] * m[0] + h[1], h[1] * m[1]))
    h_anchor = _tensor([0.4, -0.8])
    m_anchor = _tensor([1.2, 0.3])
    identity = torch.eye(2, dtype=DTYPE)
    K = _tensor([[1.2, 1.0], [0.0, 0.3]])
    R = torch.diag(h_anchor)
    _assert_close(jvp_actions(lambda h: phi(h, m_anchor), h_anchor, identity), K)
    _assert_close(jvp_actions(lambda m: phi(h_anchor, m), m_anchor, identity), R)
    h_other = h_anchor + _tensor([0.7, 0.2])
    R_other = jvp_actions(lambda m: phi(h_other, m), m_anchor, identity)
    _assert_close(R_other, torch.diag(h_other))
    assert not torch.allclose(R_other, R)

    chart = AffineResponseChart(
        state_anchor=h_anchor, message_anchor=m_anchor,
        beta=phi(h_anchor, m_anchor), B=K, A=R,
        U=identity, Q=identity, G=identity,
        state_radius=2.0, message_radius=2.0,
    )
    delta_h = _tensor([0.2, -0.1])
    delta_m = _tensor([-0.3, 0.4])
    errors = []
    for radius in (0.5, 1.0):
        h = h_anchor + radius * delta_h
        m = m_anchor + radius * delta_m
        error = phi(h, m) - chart.predict_supported(h, m)
        _assert_close(error, radius**2 * delta_h * delta_m)
        errors.append(torch.linalg.vector_norm(error))
    _assert_close(errors[1], 4 * errors[0])


def test_fisher_projection_uses_gram_dual_and_preserves_batch_coordinates() -> None:
    basis = _tensor([[1.0, 0.3], [0.2, 1.4], [-0.1, 0.5]])
    factor = torch.diag(_tensor([2.0, 1.0, 3.0]))
    metric = factor @ factor.T
    projection = FisherProjection(basis=basis, factor=factor)
    expected_dual = torch.linalg.solve(basis.T @ metric @ basis, basis.T @ metric)
    _assert_close(projection.coordinate_map, expected_dual)
    _assert_close(projection.coordinate_map @ basis, torch.eye(2, dtype=DTYPE))
    values = _tensor([[0.7, -0.4, 1.2], [-0.6, 0.8, 0.3]])
    _assert_close(projection.encode(values), values @ expected_dual.T)
    _assert_close(projection.decode(projection.encode(values)), projection.project(values))
    residual = values - projection.project(values)
    _assert_close(residual @ metric @ basis, torch.zeros((2, 2), dtype=DTYPE))
    _assert_close(projection.fisher_squared(values), (values @ factor).square().sum(dim=-1))


def test_singular_ambient_fisher_accepts_supported_basis_and_rejects_null_generator() -> None:
    factor = _tensor([[2.0], [0.0]])
    supported_basis = _tensor([[0.5], [0.0]])
    projection = FisherProjection(basis=supported_basis, factor=factor)
    _assert_close(projection.gram, torch.ones((1, 1), dtype=DTYPE))
    _assert_close(projection.project(_tensor([3.0, 8.0])), _tensor([3.0, 0.0]))
    _assert_close(projection.fisher_squared(_tensor([0.0, 8.0])), _tensor(0.0))
    with pytest.raises(ValueError):
        FisherProjection(basis=torch.eye(2, dtype=DTYPE), factor=factor)


def test_substantial_out_of_span_response_is_visible_despite_zero_projected_operator() -> None:
    projection = FisherProjection(
        basis=_tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        factor=torch.diag(_tensor([2.0, 1.0, 3.0])),
    )
    response = _tensor([0.0, 0.0, 2.0])
    _assert_close(projection.encode(response), torch.zeros(2, dtype=DTYPE))
    leakage = response - projection.project(response)
    _assert_close(projection.fisher_squared(leakage), _tensor(36.0))
    assert torch.linalg.vector_norm(leakage).item() == 2.0


@pytest.mark.parametrize(
    ("operator", "gains"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], [1.0, 1.0]),
        ([[0.0, -1.0], [1.0, 0.0]], [1.0, 1.0]),
        ([[2.0, 0.0], [0.0, 0.5]], [2.0, 0.5]),
        ([[1.0, 1.0], [0.0, 1.0]], [(1.0 + math.sqrt(5.0)) / 2.0, (math.sqrt(5.0) - 1.0) / 2.0]),
    ],
)
def test_finite_operator_gain_and_polar_fixtures(operator: list[list[float]], gains: list[float]) -> None:
    matrix = _tensor(operator)
    result = metric_operator_analysis(matrix, torch.eye(2, dtype=DTYPE))
    _assert_close(result["singular_values"], _tensor(gains))
    O, P = result["orthogonal_factor"], result["positive_factor"]
    _assert_close(O.T @ O, torch.eye(2, dtype=DTYPE))
    _assert_close(P, P.T)
    _assert_close(O @ P, result["whitened"])
    assert torch.linalg.eigvalsh(P).min().item() >= 0.0
    if operator == [[1.0, 0.0], [0.0, 1.0]]:
        _assert_close(O, torch.eye(2, dtype=DTYPE))
    if operator == [[1.0, 1.0], [0.0, 1.0]]:
        _assert_close(torch.linalg.eigvals(matrix).real, torch.ones(2, dtype=DTYPE))
        assert result["singular_values"][0].item() > 1.6


def test_fisher_gram_corrects_spurious_coordinate_gain() -> None:
    basis = torch.diag(_tensor([2.0, 0.5]))
    physical_rotation = _tensor([[0.0, -1.0], [1.0, 0.0]])
    coefficient_operator = torch.linalg.solve(basis, physical_rotation @ basis)
    assert torch.linalg.svdvals(coefficient_operator)[0].item() == 4.0
    result = metric_operator_analysis(coefficient_operator, basis.T @ basis)
    _assert_close(result["singular_values"], torch.ones(2, dtype=DTYPE))
    _assert_close(result["orthogonal_factor"] @ result["positive_factor"], result["whitened"])


def test_singular_operator_and_singular_metric_report_supported_gain_limits() -> None:
    singular_operator = metric_operator_analysis(
        torch.diag(_tensor([2.0, 0.0])), torch.eye(2, dtype=DTYPE)
    )
    assert singular_operator["rank_deficient"]
    _assert_close(singular_operator["singular_values"], _tensor([2.0, 0.0]))
    _assert_close(
        singular_operator["orthogonal_factor"] @ singular_operator["positive_factor"],
        singular_operator["whitened"],
    )

    metric = torch.diag(_tensor([1.0, 0.0]))
    identity = metric_operator_analysis(torch.eye(2, dtype=DTYPE), metric)
    assert identity["input_supported_rank"] == identity["output_supported_rank"] == 1
    _assert_close(identity["singular_values"], _tensor([1.0]))
    assert identity["full_semimetric_gain_defined"]
    assert identity["null_input_visible_response_norm"] == 0.0
    unsafe = metric_operator_analysis(_tensor([[1.0, 1.0], [0.0, 1.0]]), metric)
    _assert_close(unsafe["singular_values"], _tensor([1.0]))
    assert not unsafe["full_semimetric_gain_defined"]
    assert unsafe["null_input_visible_response_norm"] > 0.9


def test_rectangular_gain_uses_distinct_input_and_output_metrics() -> None:
    operator = _tensor([[1.0, 2.0], [0.0, 1.0], [5.0, 8.0]])
    input_metric = torch.diag(_tensor([4.0, 1.0]))
    output_metric = torch.diag(_tensor([1.0, 9.0, 0.0]))
    result = metric_operator_analysis(operator, output_metric, input_metric)
    expected_supported = torch.diag(_tensor([1.0, 3.0])) @ operator[:2] @ torch.diag(_tensor([0.5, 1.0]))
    _assert_close(result["singular_values"], torch.linalg.svdvals(expected_supported))
    assert result["input_supported_rank"] == result["output_supported_rank"] == 2
    assert result["full_semimetric_gain_defined"]
    assert result["null_input_visible_response_norm"] == 0.0
    assert result["reflection"] is None


@pytest.mark.parametrize("transpose", [False, True])
def test_full_rank_rectangular_polar_partial_isometry_is_unique(transpose: bool) -> None:
    operator = _tensor([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    if transpose:
        operator = operator.T
    result = metric_operator_analysis(
        operator,
        torch.eye(operator.shape[0], dtype=DTYPE),
        torch.eye(operator.shape[1], dtype=DTYPE),
    )
    assert result["polar_factor_unique"]
    assert result["reflection"] is None
    O = result["orthogonal_factor"]
    identity = O @ O.T if transpose else O.T @ O
    _assert_close(identity, torch.eye(2, dtype=DTYPE))
    _assert_close(O @ result["positive_factor"], result["whitened"])


def test_different_metric_coordinate_orientations_do_not_authorize_reflection_claim() -> None:
    result = metric_operator_analysis(
        torch.eye(2, dtype=DTYPE),
        torch.diag(_tensor([4.0, 1.0])),
        torch.diag(_tensor([1.0, 4.0])),
    )
    _assert_close(result["singular_values"], _tensor([2.0, 0.5]))
    assert result["reflection"] is None


def test_null_input_visibility_is_invariant_to_global_metric_scale() -> None:
    operator = _tensor([[1.0, 1.0], [0.0, 1.0]])
    for metric_scale in (1.0, 1.0e-30):
        metric = metric_scale * torch.diag(_tensor([1.0, 0.0]))
        result = metric_operator_analysis(operator, metric)
        _assert_close(result["singular_values"], _tensor([1.0]))
        assert not result["full_semimetric_gain_defined"]
        assert result["null_input_visible_response_norm"] > 0.0


def test_nonsymmetric_metric_is_rejected_independently_of_metric_scale() -> None:
    for metric_scale in (1.0, 1.0e-30):
        metric = metric_scale * _tensor([[1.0, 0.2], [0.0, 1.0]])
        with pytest.raises(ValueError):
            metric_operator_analysis(torch.eye(2, dtype=DTYPE), metric)


@pytest.mark.parametrize("coordinate_change", [[[1.0, 0.0], [0.0, -1.0]], [[0.6, -0.8], [0.8, 0.6]]])
def test_consistent_basis_sign_and_rotation_preserve_projection_and_fisher_gains(coordinate_change: list[list[float]]) -> None:
    rotation = _tensor(coordinate_change)
    basis = _tensor([[1.0, 0.3], [0.2, 1.4], [-0.1, 0.5]])
    factor = torch.diag(_tensor([2.0, 1.0, 3.0]))
    original = FisherProjection(basis=basis, factor=factor)
    changed = FisherProjection(basis=basis @ rotation, factor=factor)
    values = _tensor([[0.7, -0.4, 1.2], [-0.6, 0.8, 0.3]])
    _assert_close(changed.project(values), original.project(values))
    _assert_close(changed.encode(values), original.encode(values) @ rotation)
    operator = _tensor([[1.0, 0.8], [-0.2, 0.7]])
    before = metric_operator_analysis(operator, original.gram)
    after = metric_operator_analysis(rotation.T @ operator @ rotation, changed.gram)
    _assert_close(before["singular_values"], after["singular_values"])


def test_consistently_rotated_state_message_and_output_chart_bases_preserve_prediction() -> None:
    identity = torch.eye(2, dtype=DTYPE)
    output_rotation = _tensor([[0.6, -0.8], [0.8, 0.6]])
    state_rotation = _tensor([[0.0, -1.0], [1.0, 0.0]])
    message_rotation = torch.diag(_tensor([-1.0, 1.0]))
    beta = _tensor([2.0, -0.3])
    B = _tensor([[0.3, -0.2], [0.1, 0.6]])
    A = _tensor([[1.2, 0.4], [-0.5, 0.7]])
    anchors = dict(state_anchor=_tensor([0.2, -0.1]), message_anchor=_tensor([-0.3, 0.4]), state_radius=10.0, message_radius=10.0)
    original = AffineResponseChart(beta=beta, B=B, A=A, U=identity, Q=identity, G=identity, **anchors)
    changed = AffineResponseChart(
        beta=output_rotation.T @ beta,
        B=output_rotation.T @ B @ state_rotation,
        A=output_rotation.T @ A @ message_rotation,
        U=state_rotation, Q=message_rotation, G=output_rotation, **anchors,
    )
    h, m = _tensor([0.8, 0.2]), _tensor([0.4, -0.7])
    _assert_close(changed.predict_supported(h, m), original.predict_supported(h, m))


def test_identical_response_derivatives_do_not_hide_different_offsets_or_support_failure() -> None:
    phi = lambda h, m: m + 3.0 * h - h**3
    left_h, right_h, m = _tensor([-1.0]), _tensor([1.0]), _tensor([0.0])
    identity = torch.ones((1, 1), dtype=DTYPE)
    left_K = jvp_actions(lambda h: phi(h, m), left_h, identity)
    right_K = jvp_actions(lambda h: phi(h, m), right_h, identity)
    left_R = jvp_actions(lambda x: phi(left_h, x), m, identity)
    right_R = jvp_actions(lambda x: phi(right_h, x), m, identity)
    _assert_close(left_K, right_K)
    _assert_close(left_R, right_R)
    assert torch.linalg.vector_norm(phi(right_h, m) - phi(left_h, m)).item() == 4.0
    chart = AffineResponseChart(
        state_anchor=left_h, message_anchor=m, beta=phi(left_h, m),
        B=left_K, A=left_R, U=identity, Q=identity, G=identity,
        state_radius=0.25, message_radius=0.25,
    )
    assert chart.in_support(left_h, m)
    assert not chart.in_support(right_h, m)
    assert not chart.in_support(left_h, _tensor([0.3]))
    _assert_close(chart.predict(right_h, m), phi(left_h, m))
    with pytest.raises(ValueError):
        chart.predict_supported(right_h, m)


def test_chart_support_checks_directions_omitted_by_input_bases() -> None:
    anchor = torch.zeros(2, dtype=DTYPE)
    first_axis = _tensor([[1.0], [0.0]])
    chart = AffineResponseChart(
        state_anchor=anchor, message_anchor=anchor,
        beta=_tensor([0.0]), B=_tensor([[0.0]]), A=_tensor([[0.0]]),
        U=first_axis, Q=first_axis, G=first_axis,
        state_radius=0.25, message_radius=0.25,
    )
    orthogonal_shift = _tensor([0.0, 10.0])
    assert chart.in_support(anchor, anchor)
    assert not chart.in_support(orthogonal_shift, anchor)
    assert not chart.in_support(anchor, orthogonal_shift)
    with pytest.raises(ValueError):
        chart.predict_supported(orthogonal_shift, anchor)


def test_fixed_and_recomputed_message_derivatives_follow_distinct_chain_rules() -> None:
    state = _tensor([0.4, -0.7])
    K = _tensor([[0.2, -0.1], [0.4, 0.3]])
    R = _tensor([[0.7, 0.5], [-0.2, 0.8]])
    identity = torch.eye(2, dtype=DTYPE)
    message_fn = lambda h: h.square()
    phi = lambda h, m: _tensor([0.1, -0.2]) + K @ h + R @ m
    fixed_message = message_fn(state)
    fixed_fn = lambda h: h + phi(h, fixed_message)
    recomputed_fn = lambda h: h + phi(h, message_fn(h))
    fixed_J = jvp_actions(fixed_fn, state, identity)
    total_J = jvp_actions(recomputed_fn, state, identity)
    _assert_close(fixed_J, identity + K)
    _assert_close(total_J, identity + K + R @ torch.diag(2.0 * state))
    assert not torch.allclose(total_J, fixed_J)
