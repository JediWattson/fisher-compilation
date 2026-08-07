from __future__ import annotations

import pytest
import torch

from fisher_graph.downstream_affine_coordinate_solver import (
    DownstreamAffineSolverConfig,
    materialize_affine_candidate,
    solve_batched_downstream_sensitive_affine_coordinates,
    solve_downstream_sensitive_affine_coordinates,
)


def _squared_objective(target: torch.Tensor):
    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = (candidate - target).square().mean()
        return {"loss": loss, "kl": loss}

    return callback


def _coupled_nonlinear_row_metrics(
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Couple coordinates within each row, but never couple distinct rows."""

    candidate_features = torch.stack(
        (
            candidate[..., 0] * candidate[..., 1],
            torch.sin(candidate[..., 2]) + candidate[..., 0].square(),
            candidate[..., 1] * torch.tanh(candidate[..., 2]),
        ),
        dim=-1,
    )
    target_features = torch.stack(
        (
            target[..., 0] * target[..., 1],
            torch.sin(target[..., 2]) + target[..., 0].square(),
            target[..., 1] * torch.tanh(target[..., 2]),
        ),
        dim=-1,
    )
    loss = (candidate_features - target_features).square().mean(dim=-1)
    loss = loss + 0.07 * (candidate - target).pow(4).mean(dim=-1)
    kl = (torch.tanh(candidate) - torch.tanh(target)).square().mean(dim=-1)
    kl = kl + 0.03 * (
        candidate_features - target_features
    ).abs().mean(dim=-1)
    return {"loss": loss, "kl": kl}


def _batched_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    bias = torch.tensor(
        [
            [0.20, -0.10, 0.30],
            [-0.30, 0.25, 0.10],
            [0.15, 0.05, -0.20],
            [-0.05, -0.20, 0.35],
        ],
        dtype=torch.float64,
    )
    decoder = torch.tensor(
        [[1.0, -0.25, 0.40], [0.15, 0.80, -0.55]],
        dtype=torch.float64,
    )
    initial = torch.tensor(
        [[0.20, -0.10], [-0.35, 0.40], [0.05, 0.15], [0.50, -0.25]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [0.80, -0.40, 0.20],
            [-0.10, 0.70, -0.50],
            [0.30, -0.20, 0.65],
            [-0.55, 0.15, 0.45],
        ],
        dtype=torch.float64,
    )
    learning_rates = torch.tensor(
        [0.0125, 0.035, 0.061, 0.093], dtype=torch.float64
    )
    return bias, decoder, initial, target, learning_rates


def test_solve_reduces_loss_and_returns_exact_affine_member() -> None:
    bias = torch.tensor([1.0, -1.0], dtype=torch.float64)
    decoder = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    initial = torch.tensor([0.0], dtype=torch.float64)
    target = torch.tensor([3.0, -1.0], dtype=torch.float64)

    solution = solve_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        _squared_objective(target),
        config=DownstreamAffineSolverConfig(
            steps=80,
            learning_rate=0.1,
        ),
    )

    exact = bias + solution.coordinates @ decoder
    assert torch.equal(solution.candidate, exact)
    assert solution.candidate[1].item() == -1.0
    assert solution.receipt["affine_membership_by_construction"] is True
    assert solution.receipt["selected_loss_reduced_from_initial"] is True
    assert float(solution.receipt["selected_downstream_loss"]) < 1.0e-4


def test_kl_first_selection_is_derived_from_complete_trace() -> None:
    bias = torch.zeros(1, dtype=torch.float64)
    decoder = torch.ones((1, 1), dtype=torch.float64)
    initial = torch.zeros(1, dtype=torch.float64)

    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "loss": (candidate - 2.0).square().mean(),
            "kl": (candidate - 1.0).square().mean(),
        }

    solution = solve_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        callback,
        config=DownstreamAffineSolverConfig(
            steps=40,
            learning_rate=0.1,
        ),
    )
    trace = solution.receipt["evaluations"]
    assert isinstance(trace, list)
    expected = min(
        trace,
        key=lambda row: (
            float(row["kl"]),
            float(row["downstream_loss"]),
            int(row["step"]),
        ),
    )
    assert solution.receipt["selected_step"] == expected["step"]
    assert sum(bool(row["selected"]) for row in trace) == 1


def test_trust_region_is_per_row_and_ridge_is_receipted() -> None:
    bias = torch.zeros((2, 2), dtype=torch.float64)
    decoder = torch.eye(2, dtype=torch.float64)
    initial = torch.zeros((2, 2), dtype=torch.float64)
    target = torch.full((2, 2), 10.0, dtype=torch.float64)
    radius = 0.25

    solution = solve_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        _squared_objective(target),
        config=DownstreamAffineSolverConfig(
            steps=10,
            learning_rate=1.0,
            ridge=0.5,
            trust_radius=radius,
        ),
    )

    row_norms = torch.linalg.vector_norm(
        (solution.coordinates - initial).reshape(-1, 2), dim=-1
    )
    assert bool((row_norms <= radius + 1.0e-12).all().item())
    assert solution.receipt["ridge"] == 0.5
    assert solution.receipt["trust_projection_count"] > 0
    for row in solution.receipt["evaluations"]:
        assert (
            float(row["maximum_row_coordinate_displacement_l2"])
            <= radius + 1.0e-12
        )


def test_source_tensors_and_existing_gradients_are_never_mutated() -> None:
    bias = torch.tensor([0.5, -0.5], requires_grad=True)
    decoder = torch.tensor([[1.0, 2.0]], requires_grad=True)
    initial = torch.tensor([0.0], requires_grad=True)
    bias.grad = torch.tensor([7.0, 8.0])
    decoder.grad = torch.tensor([[9.0, 10.0]])
    initial.grad = torch.tensor([11.0])
    snapshots = {
        "bias": bias.detach().clone(),
        "decoder": decoder.detach().clone(),
        "initial": initial.detach().clone(),
        "bias_grad": bias.grad.clone(),
        "decoder_grad": decoder.grad.clone(),
        "initial_grad": initial.grad.clone(),
    }

    solution = solve_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        _squared_objective(torch.tensor([1.5, 1.5])),
        config=DownstreamAffineSolverConfig(steps=5, learning_rate=0.1),
    )

    assert torch.equal(bias, snapshots["bias"])
    assert torch.equal(decoder, snapshots["decoder"])
    assert torch.equal(initial, snapshots["initial"])
    assert torch.equal(bias.grad, snapshots["bias_grad"])
    assert torch.equal(decoder.grad, snapshots["decoder_grad"])
    assert torch.equal(initial.grad, snapshots["initial_grad"])
    assert solution.coordinates.data_ptr() != initial.data_ptr()
    assert solution.receipt["source_tensors_optimizer_owned"] is False


def test_callback_parameter_gradient_is_not_accumulated_or_mutated() -> None:
    callback_weight = torch.nn.Parameter(torch.tensor(2.0))
    callback_weight.grad = torch.tensor(7.0)

    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = (candidate * callback_weight - 1.0).square().mean()
        return {"loss": loss, "kl": loss}

    solve_downstream_sensitive_affine_coordinates(
        torch.zeros(1),
        torch.ones((1, 1)),
        torch.zeros(1),
        callback,
        config=DownstreamAffineSolverConfig(steps=3, learning_rate=0.1),
    )

    assert torch.equal(callback_weight.grad, torch.tensor(7.0))


def test_exact_metric_tie_selects_the_initial_step() -> None:
    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        connected_zero = candidate.sum() * 0.0
        return {"loss": connected_zero, "kl": connected_zero}

    solution = solve_downstream_sensitive_affine_coordinates(
        torch.zeros(1),
        torch.ones((1, 1)),
        torch.zeros(1),
        callback,
        config=DownstreamAffineSolverConfig(steps=3, learning_rate=0.1),
    )

    assert solution.receipt["selected_step"] == 0
    assert solution.receipt["selected_is_final_step"] is False


def test_receipt_is_deterministic_for_identical_inputs() -> None:
    bias = torch.tensor([0.0, 1.0], dtype=torch.float64)
    decoder = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    initial = torch.tensor([0.0], dtype=torch.float64)
    target = torch.tensor([2.0, 0.0], dtype=torch.float64)
    config = DownstreamAffineSolverConfig(
        steps=12,
        learning_rate=0.05,
        ridge=0.01,
        trust_radius=0.5,
    )

    first = solve_downstream_sensitive_affine_coordinates(
        bias, decoder, initial, _squared_objective(target), config=config
    )
    second = solve_downstream_sensitive_affine_coordinates(
        bias, decoder, initial, _squared_objective(target), config=config
    )

    assert first.receipt == second.receipt
    assert torch.equal(first.coordinates, second.coordinates)
    assert torch.equal(first.candidate, second.candidate)


def test_batched_solver_matches_independent_scalar_solves_with_row_lrs() -> None:
    bias, decoder, initial, target, learning_rates = _batched_fixture()
    config = DownstreamAffineSolverConfig(
        steps=28,
        learning_rate=0.5,
        ridge=0.037,
        trust_radius=0.42,
    )

    batched = solve_batched_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        lambda candidate: _coupled_nonlinear_row_metrics(candidate, target),
        config=config,
        learning_rate_by_row=learning_rates,
    )
    scalar_solutions = []
    for row_index in range(initial.shape[0]):
        scalar_solutions.append(
            solve_downstream_sensitive_affine_coordinates(
                bias[row_index],
                decoder,
                initial[row_index],
                lambda candidate, row_index=row_index: (
                    _coupled_nonlinear_row_metrics(
                        candidate, target[row_index]
                    )
                ),
                config=DownstreamAffineSolverConfig(
                    steps=config.steps,
                    learning_rate=float(learning_rates[row_index].item()),
                    ridge=config.ridge,
                    trust_radius=config.trust_radius,
                ),
            )
        )

    expected_coordinates = torch.stack(
        [solution.coordinates for solution in scalar_solutions]
    )
    expected_candidates = torch.stack(
        [solution.candidate for solution in scalar_solutions]
    )
    torch.testing.assert_close(
        batched.coordinates,
        expected_coordinates,
        rtol=0.0,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        batched.candidate,
        expected_candidates,
        rtol=0.0,
        atol=1.0e-15,
    )
    receipt = batched.receipt
    assert receipt["gradient_reduction"] == (
        "sum_of_per_row_regularized_objectives"
    )
    assert receipt["learning_rate_source"] == "explicit_per_row_vector"
    assert receipt["per_row_selected_steps"] == [
        solution.receipt["selected_step"] for solution in scalar_solutions
    ]
    assert receipt["per_row_initial_kl"] == [
        solution.receipt["initial_kl"] for solution in scalar_solutions
    ]
    torch.testing.assert_close(
        torch.tensor(receipt["per_row_selected_kl"]),
        torch.tensor(
            [
                solution.receipt["selected_kl"]
                for solution in scalar_solutions
            ]
        ),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_batched_solver_is_permutation_and_subset_invariant() -> None:
    bias, decoder, initial, target, learning_rates = _batched_fixture()
    config = DownstreamAffineSolverConfig(
        steps=17,
        learning_rate=0.1,
        ridge=0.011,
        trust_radius=0.35,
    )

    def solve(indices: torch.Tensor):
        selected_target = target[indices]
        return solve_batched_downstream_sensitive_affine_coordinates(
            bias[indices],
            decoder,
            initial[indices],
            lambda candidate: _coupled_nonlinear_row_metrics(
                candidate, selected_target
            ),
            config=config,
            learning_rate_by_row=learning_rates[indices],
        )

    identity = torch.arange(initial.shape[0])
    full = solve(identity)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = solve(permutation)
    inverse = torch.argsort(permutation)
    assert torch.equal(permuted.coordinates[inverse], full.coordinates)
    assert torch.equal(permuted.candidate[inverse], full.candidate)
    for key in (
        "per_row_selected_steps",
        "per_row_initial_downstream_loss",
        "per_row_selected_downstream_loss",
        "per_row_initial_kl",
        "per_row_selected_kl",
        "per_row_trust_projection_counts",
    ):
        permuted_values = permuted.receipt[key]
        assert isinstance(permuted_values, list)
        assert [permuted_values[index] for index in inverse.tolist()] == (
            full.receipt[key]
        )

    subset_indices = torch.tensor([3, 1])
    subset = solve(subset_indices)
    assert torch.equal(subset.coordinates, full.coordinates[subset_indices])
    assert torch.equal(subset.candidate, full.candidate[subset_indices])
    for key in (
        "per_row_selected_steps",
        "per_row_initial_downstream_loss",
        "per_row_selected_downstream_loss",
        "per_row_initial_kl",
        "per_row_selected_kl",
        "per_row_trust_projection_counts",
    ):
        full_values = full.receipt[key]
        subset_values = subset.receipt[key]
        assert isinstance(full_values, list)
        assert isinstance(subset_values, list)
        assert subset_values == [full_values[index] for index in subset_indices]


@pytest.mark.parametrize("shape", [(), (4, 1), (1,)])
@pytest.mark.parametrize("field", ["loss", "kl"])
def test_batched_solver_rejects_scalar_or_shared_callback_metrics(
    field: str,
    shape: tuple[int, ...],
) -> None:
    bias, decoder, initial, target, _ = _batched_fixture()

    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        values = _coupled_nonlinear_row_metrics(candidate, target)
        shared = candidate.sum() * 0.0
        if shape:
            shared = shared.expand(shape)
        values[field] = shared
        return values

    with pytest.raises(ValueError, match=rf"{field} must have shape \[rows\]"):
        solve_batched_downstream_sensitive_affine_coordinates(
            bias,
            decoder,
            initial,
            callback,
            config=DownstreamAffineSolverConfig(steps=1),
        )


def test_batched_solver_is_source_safe_and_receipt_has_no_tensors() -> None:
    bias, decoder, initial, target, learning_rates = _batched_fixture()
    bias.requires_grad_()
    decoder.requires_grad_()
    initial.requires_grad_()
    learning_rates.requires_grad_()
    bias.grad = torch.full_like(bias, 2.0)
    decoder.grad = torch.full_like(decoder, 3.0)
    initial.grad = torch.full_like(initial, 4.0)
    learning_rates.grad = torch.full_like(learning_rates, 5.0)
    callback_weight = torch.nn.Parameter(torch.tensor(0.9, dtype=torch.float64))
    callback_weight.grad = torch.tensor(6.0, dtype=torch.float64)
    tensors = (bias, decoder, initial, learning_rates)
    values_before = [value.detach().clone() for value in tensors]
    gradients_before = [value.grad.clone() for value in tensors]

    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        metrics = _coupled_nonlinear_row_metrics(
            candidate * callback_weight, target
        )
        return metrics

    first = solve_batched_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        callback,
        config=DownstreamAffineSolverConfig(steps=5, learning_rate=0.1),
        learning_rate_by_row=learning_rates,
    )
    second = solve_batched_downstream_sensitive_affine_coordinates(
        bias,
        decoder,
        initial,
        callback,
        config=DownstreamAffineSolverConfig(steps=5, learning_rate=0.1),
        learning_rate_by_row=learning_rates,
    )

    for value, expected, expected_gradient in zip(
        tensors, values_before, gradients_before, strict=True
    ):
        assert torch.equal(value, expected)
        assert torch.equal(value.grad, expected_gradient)
    assert torch.equal(callback_weight.grad, torch.tensor(6.0))
    assert first.receipt == second.receipt
    assert torch.equal(first.coordinates, second.coordinates)
    assert torch.equal(first.candidate, second.candidate)

    def assert_no_tensors(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for child in value.values():
                assert_no_tensors(child)
        elif isinstance(value, list):
            for child in value:
                assert_no_tensors(child)

    assert_no_tensors(first.receipt)
    assert "evaluations" not in first.receipt
    assert first.receipt["large_tensor_payloads_in_receipt"] is False
    hashes = first.receipt["hashes"]
    assert isinstance(hashes, dict)
    assert all(len(value) == 64 for value in hashes.values())


def test_batched_exact_ties_abstain_at_step_zero_per_row() -> None:
    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        connected = candidate.sum(dim=-1) * 0.0
        return {"loss": connected, "kl": connected}

    solution = solve_batched_downstream_sensitive_affine_coordinates(
        torch.zeros((3, 2), dtype=torch.float64),
        torch.eye(2, dtype=torch.float64),
        torch.zeros((3, 2), dtype=torch.float64),
        callback,
        config=DownstreamAffineSolverConfig(steps=3, learning_rate=0.1),
    )

    assert solution.receipt["per_row_selected_steps"] == [0, 0, 0]
    assert torch.equal(solution.coordinates, torch.zeros((3, 2)))


@pytest.mark.parametrize(
    ("learning_rates", "message"),
    [
        (torch.ones(3, dtype=torch.float64), r"shape \[rows\]"),
        (
            torch.tensor([0.1, 0.2, 0.3, float("nan")]),
            "finite values",
        ),
        (torch.tensor([0.1, 0.0, 0.3, 0.4]), "positive"),
    ],
)
def test_batched_solver_rejects_invalid_per_row_learning_rates(
    learning_rates: torch.Tensor,
    message: str,
) -> None:
    bias, decoder, initial, target, _ = _batched_fixture()
    with pytest.raises(ValueError, match=message):
        solve_batched_downstream_sensitive_affine_coordinates(
            bias,
            decoder,
            initial,
            lambda candidate: _coupled_nonlinear_row_metrics(
                candidate, target
            ),
            config=DownstreamAffineSolverConfig(steps=1),
            learning_rate_by_row=learning_rates,
        )


def test_batched_ridge_cannot_mask_disconnected_row_losses() -> None:
    bias, decoder, initial, _, _ = _batched_fixture()

    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        detached = candidate.detach().square().mean(dim=-1).requires_grad_()
        return {"loss": detached, "kl": detached.detach()}

    with pytest.raises(RuntimeError, match="not connected"):
        solve_batched_downstream_sensitive_affine_coordinates(
            bias,
            decoder,
            initial,
            callback,
            config=DownstreamAffineSolverConfig(steps=1, ridge=1.0),
        )


@pytest.mark.parametrize(
    ("bias", "decoder", "coordinates", "message"),
    [
        (
            torch.zeros(3),
            torch.zeros((2, 3)),
            torch.zeros(1),
            "trailing dimension",
        ),
        (
            torch.zeros(4),
            torch.zeros((2, 3)),
            torch.zeros(2),
            "bias must have shape",
        ),
        (
            torch.zeros(3),
            torch.zeros((2, 3, 1)),
            torch.zeros(2),
            "decoder must have shape",
        ),
        (
            torch.zeros(3),
            torch.tensor([[float("nan"), 0.0, 0.0]]),
            torch.zeros(1),
            "decoder must contain only finite",
        ),
    ],
)
def test_invalid_or_nonfinite_affine_inputs_fail(
    bias: torch.Tensor,
    decoder: torch.Tensor,
    coordinates: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        materialize_affine_candidate(bias, coordinates, decoder)


@pytest.mark.parametrize("field", ["loss", "kl"])
def test_nonfinite_callback_scalars_fail(field: str) -> None:
    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = candidate.square().mean()
        values = {"loss": loss, "kl": loss.detach().clone()}
        values[field] = loss * float("nan")
        return values

    with pytest.raises(RuntimeError, match=f"{field} is nonfinite"):
        solve_downstream_sensitive_affine_coordinates(
            torch.zeros(1),
            torch.ones((1, 1)),
            torch.zeros(1),
            callback,
            config=DownstreamAffineSolverConfig(steps=1),
        )


def test_invalid_callback_shapes_fail() -> None:
    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        values = candidate.square()
        return {"loss": values, "kl": values.mean()}

    with pytest.raises(ValueError, match="loss must be scalar"):
        solve_downstream_sensitive_affine_coordinates(
            torch.zeros(2),
            torch.eye(2),
            torch.zeros(2),
            callback,
            config=DownstreamAffineSolverConfig(steps=1),
        )


def test_ridge_cannot_mask_a_disconnected_downstream_loss() -> None:
    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        detached_leaf = candidate.detach().square().mean().requires_grad_()
        return {"loss": detached_leaf, "kl": detached_leaf.detach()}

    with pytest.raises(RuntimeError, match="not connected"):
        solve_downstream_sensitive_affine_coordinates(
            torch.zeros(1),
            torch.ones((1, 1)),
            torch.ones(1),
            callback,
            config=DownstreamAffineSolverConfig(
                steps=1,
                ridge=1.0,
            ),
        )


def test_invalid_config_fails_before_callback() -> None:
    callback_calls = 0

    def callback(candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        nonlocal callback_calls
        callback_calls += 1
        loss = candidate.square().mean()
        return {"loss": loss, "kl": loss}

    with pytest.raises(ValueError, match="trust_radius must be finite"):
        solve_downstream_sensitive_affine_coordinates(
            torch.zeros(1),
            torch.ones((1, 1)),
            torch.zeros(1),
            callback,
            config=DownstreamAffineSolverConfig(trust_radius=float("inf")),
        )
    assert callback_calls == 0
