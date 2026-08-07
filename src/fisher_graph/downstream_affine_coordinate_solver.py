"""Generic downstream-sensitive optimization inside a frozen affine span.

The solver in this module deliberately owns only coordinates.  A candidate is
materialized as ``bias + coordinates @ decoder`` on every evaluation, while
``bias`` and ``decoder`` are detached, cloned, and never passed to the
optimizer.  The downstream callback may therefore run an arbitrary
differentiable suffix without giving the solver permission to alter the source
geometry.

This is infrastructure, not a Gemma experiment protocol.  In particular, it
does not select calibration rows, build a decoder, or make a held-out or
compression claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import TypeAlias

import torch
from torch import Tensor


DownstreamLossCallback: TypeAlias = Callable[[Tensor], Mapping[str, Tensor]]
BatchedDownstreamLossCallback: TypeAlias = Callable[
    [Tensor], Mapping[str, Tensor]
]


@dataclass(frozen=True)
class DownstreamAffineSolverConfig:
    """Fixed optimization and regularization controls for one solve.

    ``ridge`` penalizes mean squared coordinate displacement from
    ``initial_coordinates``.  ``trust_radius``, when present, is a hard L2
    radius applied independently to every flattened coordinate row around the
    same initial point.
    """

    steps: int = 64
    learning_rate: float = 1.0e-2
    ridge: float = 0.0
    trust_radius: float | None = None


@dataclass(frozen=True)
class DownstreamAffineCoordinateSolution:
    """Detached selected coordinates, their exact candidate, and a receipt."""

    coordinates: Tensor
    candidate: Tensor
    receipt: dict[str, object]


def _finite_real(
    value: object,
    *,
    label: str,
    minimum: float,
    strict: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if (strict and parsed <= minimum) or (not strict and parsed < minimum):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{label} must be {relation} {minimum}")
    return parsed


def _validated_config(
    config: DownstreamAffineSolverConfig,
) -> DownstreamAffineSolverConfig:
    if not isinstance(config, DownstreamAffineSolverConfig):
        raise TypeError("config must be a DownstreamAffineSolverConfig")
    if isinstance(config.steps, bool) or not isinstance(config.steps, int):
        raise TypeError("steps must be an integer")
    if config.steps < 1:
        raise ValueError("steps must be at least 1")
    learning_rate = _finite_real(
        config.learning_rate,
        label="learning_rate",
        minimum=0.0,
        strict=True,
    )
    ridge = _finite_real(
        config.ridge,
        label="ridge",
        minimum=0.0,
        strict=False,
    )
    trust_radius = config.trust_radius
    if trust_radius is not None:
        trust_radius = _finite_real(
            trust_radius,
            label="trust_radius",
            minimum=0.0,
            strict=False,
        )
    return DownstreamAffineSolverConfig(
        steps=config.steps,
        learning_rate=learning_rate,
        ridge=ridge,
        trust_radius=trust_radius,
    )


def _validate_affine_inputs(
    bias: Tensor,
    decoder: Tensor,
    coordinates: Tensor,
) -> tuple[int, tuple[int, ...]]:
    values = {
        "bias": bias,
        "decoder": decoder,
        "coordinates": coordinates,
    }
    for label, value in values.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if value.layout is not torch.strided:
            raise ValueError(f"{label} must use strided tensor layout")
        if not value.is_floating_point():
            raise TypeError(f"{label} must have a floating-point dtype")
        if value.numel() == 0:
            raise ValueError(f"{label} must be nonempty")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise ValueError(f"{label} must contain only finite values")

    if decoder.ndim != 2:
        raise ValueError("decoder must have shape [rank, width]")
    if coordinates.ndim < 1:
        raise ValueError("coordinates must have shape [..., rank]")
    rank, width = decoder.shape
    if rank < 1 or width < 1:
        raise ValueError("decoder rank and width must be positive")
    if coordinates.shape[-1] != rank:
        raise ValueError(
            "coordinates trailing dimension must equal decoder rank"
        )
    expected_shape = (*coordinates.shape[:-1], width)
    if bias.shape not in ((width,), expected_shape):
        raise ValueError(
            "bias must have shape [width] or match the candidate shape"
        )
    if not (bias.dtype == decoder.dtype == coordinates.dtype):
        raise ValueError("bias, decoder, and coordinates must share a dtype")
    if not (bias.device == decoder.device == coordinates.device):
        raise ValueError("bias, decoder, and coordinates must share a device")
    return rank, expected_shape


def materialize_affine_candidate(
    bias: Tensor,
    coordinates: Tensor,
    decoder: Tensor,
) -> Tensor:
    """Return exactly ``bias + coordinates @ decoder`` after strict checks."""

    _validate_affine_inputs(bias, decoder, coordinates)
    candidate = bias + coordinates @ decoder
    if not bool(torch.isfinite(candidate.detach()).all().item()):
        raise RuntimeError("affine candidate is nonfinite")
    return candidate


def _callback_scalars(
    callback: DownstreamLossCallback,
    candidate: Tensor,
) -> tuple[Tensor, Tensor]:
    values = callback(candidate)
    if not isinstance(values, Mapping):
        raise TypeError("loss_callback must return a mapping")
    missing = {"loss", "kl"}.difference(values)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"loss_callback result is missing: {names}")
    loss = values["loss"]
    kl = values["kl"]
    for label, value in (("loss", loss), ("kl", kl)):
        if not isinstance(value, Tensor):
            raise TypeError(f"loss_callback {label} must be a torch.Tensor")
        if value.numel() != 1:
            raise ValueError(f"loss_callback {label} must be scalar")
        if not value.is_floating_point():
            raise TypeError(
                f"loss_callback {label} must have a floating-point dtype"
            )
        if value.device != candidate.device:
            raise ValueError(
                f"loss_callback {label} must be on the candidate device"
            )
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise RuntimeError(f"loss_callback {label} is nonfinite")
    if not loss.requires_grad:
        raise RuntimeError("loss_callback loss must be differentiable")
    return loss.reshape(()), kl.reshape(())


def _callback_row_vectors(
    callback: BatchedDownstreamLossCallback,
    candidate: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate the explicit one-loss-and-KL-per-affine-row contract."""

    values = callback(candidate)
    if not isinstance(values, Mapping):
        raise TypeError("loss_callback must return a mapping")
    missing = {"loss", "kl"}.difference(values)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"loss_callback result is missing: {names}")

    expected_shape = (candidate.shape[0],)
    loss = values["loss"]
    kl = values["kl"]
    for label, value in (("loss", loss), ("kl", kl)):
        if not isinstance(value, Tensor):
            raise TypeError(f"loss_callback {label} must be a torch.Tensor")
        if value.shape != expected_shape:
            raise ValueError(
                f"loss_callback {label} must have shape [rows] "
                f"({expected_shape}), not {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(
                f"loss_callback {label} must have a floating-point dtype"
            )
        if value.device != candidate.device:
            raise ValueError(
                f"loss_callback {label} must be on the candidate device"
            )
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise RuntimeError(f"loss_callback {label} is nonfinite")
    if not loss.requires_grad:
        raise RuntimeError("loss_callback loss must be differentiable")
    return loss, kl


def _tensor_sha256(value: Tensor) -> str:
    """Hash tensor metadata and exact bytes without serializing its values."""

    detached = value.detach().contiguous().cpu()
    metadata = json.dumps(
        {"dtype": str(detached.dtype), "shape": list(detached.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = detached.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _apply_batched_trust_region(
    coordinates: Tensor,
    initial_coordinates: Tensor,
    radius: float | None,
) -> Tensor:
    """Project each explicit coordinate row and return its projection mask."""

    row_count = coordinates.shape[0]
    if radius is None:
        return torch.zeros(
            row_count,
            dtype=torch.bool,
            device=coordinates.device,
        )
    with torch.no_grad():
        displacement = coordinates - initial_coordinates
        norms = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        projected = norms.squeeze(-1) > radius
        denominator = norms.clamp_min(torch.finfo(displacement.dtype).tiny)
        scales = torch.where(
            projected.unsqueeze(-1),
            torch.full_like(norms, radius) / denominator,
            torch.ones_like(norms),
        )
        coordinates.copy_(initial_coordinates + displacement * scales)
    return projected


def _coordinate_norms(
    coordinates: Tensor,
    initial_coordinates: Tensor,
) -> tuple[Tensor, Tensor]:
    displacement = coordinates - initial_coordinates
    rows = displacement.reshape(-1, displacement.shape[-1])
    row_norms = torch.linalg.vector_norm(rows, dim=-1)
    total_norm = torch.linalg.vector_norm(rows)
    return total_norm, row_norms


def _apply_trust_region(
    coordinates: Tensor,
    initial_coordinates: Tensor,
    radius: float | None,
) -> bool:
    if radius is None:
        return False
    with torch.no_grad():
        displacement = coordinates - initial_coordinates
        rows = displacement.reshape(-1, displacement.shape[-1])
        norms = torch.linalg.vector_norm(rows, dim=-1, keepdim=True)
        projected = bool((norms > radius).any().item())
        denominator = norms.clamp_min(torch.finfo(rows.dtype).tiny)
        scales = torch.where(
            norms > radius,
            torch.full_like(norms, radius) / denominator,
            torch.ones_like(norms),
        )
        coordinates.copy_(initial_coordinates + (rows * scales).reshape_as(
            displacement
        ))
    return projected


def solve_downstream_sensitive_affine_coordinates(
    bias: Tensor,
    decoder: Tensor,
    initial_coordinates: Tensor,
    loss_callback: DownstreamLossCallback,
    *,
    config: DownstreamAffineSolverConfig = DownstreamAffineSolverConfig(),
) -> DownstreamAffineCoordinateSolution:
    """Optimize coordinates through ``loss_callback`` and select KL-first.

    The callback must return a mapping containing differentiable scalar
    ``loss`` and finite scalar ``kl`` tensors.  Adam minimizes
    ``loss + ridge * mean((coordinates - initial_coordinates) ** 2)``.  Every
    state, including the initial state and the state after the final update,
    is evaluated.  Selection is lexicographic: lowest KL, then lowest callback
    loss, then earliest step.
    """

    parsed = _validated_config(config)
    _validate_affine_inputs(bias, decoder, initial_coordinates)
    if not callable(loss_callback):
        raise TypeError("loss_callback must be callable")

    # The optimizer receives only this new Parameter.  The source tensors are
    # detached and cloned so neither autograd nor in-place operations in this
    # routine can affect the caller's basis or bias.
    frozen_bias = bias.detach().clone()
    frozen_decoder = decoder.detach().clone()
    initial = initial_coordinates.detach().clone()
    coordinates = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        (coordinates,),
        lr=parsed.learning_rate,
        weight_decay=0.0,
    )

    evaluations: list[dict[str, object]] = []
    best_key: tuple[float, float, int] | None = None
    best_coordinates: Tensor | None = None
    best_candidate: Tensor | None = None
    selected_step = -1
    trust_projection_count = 0

    for step in range(parsed.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        candidate = materialize_affine_candidate(
            frozen_bias,
            coordinates,
            frozen_decoder,
        )
        downstream_loss, kl = _callback_scalars(loss_callback, candidate)
        displacement = coordinates - initial
        raw_ridge = displacement.square().mean()
        ridge_penalty = raw_ridge * parsed.ridge
        objective = downstream_loss + ridge_penalty
        if not bool(torch.isfinite(objective.detach()).all().item()):
            raise RuntimeError("regularized objective is nonfinite")

        total_norm, row_norms = _coordinate_norms(coordinates, initial)
        downstream_value = float(downstream_loss.detach().item())
        kl_value = float(kl.detach().item())
        row: dict[str, object] = {
            "step": step,
            "downstream_loss": downstream_value,
            "kl": kl_value,
            "ridge_mean_square_displacement": float(
                raw_ridge.detach().item()
            ),
            "ridge_penalty": float(ridge_penalty.detach().item()),
            "regularized_objective": float(objective.detach().item()),
            "coordinate_displacement_l2": float(total_norm.detach().item()),
            "maximum_row_coordinate_displacement_l2": float(
                row_norms.max().detach().item()
            ),
            "gradient_l2": None,
            "trust_projection_applied_to_next_step": False,
            "selected": False,
        }
        evaluations.append(row)

        key = (kl_value, downstream_value, step)
        if best_key is None or key < best_key:
            best_key = key
            selected_step = step
            best_coordinates = coordinates.detach().clone()
            best_candidate = candidate.detach().clone()

        if step == parsed.steps:
            continue

        downstream_gradient = torch.autograd.grad(
            downstream_loss,
            coordinates,
            allow_unused=True,
        )[0]
        if downstream_gradient is None:
            raise RuntimeError(
                "loss_callback loss is not connected to affine coordinates"
            )
        # Form the regularized-objective gradient explicitly.  Backpropagating
        # ``downstream_loss + ridge_penalty`` and merely checking
        # ``coordinates.grad`` would fail open: a disconnected callback could
        # still produce a gradient through the ridge term alone.
        gradient = downstream_gradient
        if parsed.ridge != 0.0:
            gradient = gradient + (
                (2.0 * parsed.ridge / displacement.numel()) * displacement
            )
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError("coordinate gradient is nonfinite")
        coordinates.grad = gradient.detach().clone()
        row["gradient_l2"] = float(
            torch.linalg.vector_norm(gradient).detach().item()
        )
        optimizer.step()
        if not bool(torch.isfinite(coordinates).all().item()):
            raise RuntimeError("optimizer produced nonfinite coordinates")
        projected = _apply_trust_region(
            coordinates,
            initial,
            parsed.trust_radius,
        )
        row["trust_projection_applied_to_next_step"] = projected
        trust_projection_count += int(projected)

    if best_coordinates is None or best_candidate is None or best_key is None:
        raise RuntimeError("affine coordinate solve produced no evaluations")
    evaluations[selected_step]["selected"] = True
    final = evaluations[-1]
    initial_evaluation = evaluations[0]
    selected = evaluations[selected_step]

    # Re-materialize instead of trusting a callback-owned tensor.  This is the
    # exact affine-membership guarantee of the returned solution.
    selected_candidate = materialize_affine_candidate(
        frozen_bias,
        best_coordinates,
        frozen_decoder,
    ).detach().clone()
    if not torch.equal(selected_candidate, best_candidate):
        raise RuntimeError("selected affine candidate failed exact replay")

    receipt: dict[str, object] = {
        "schema": "fisher_graph.downstream_affine_coordinate_solve.v1",
        "candidate_formula": "bias + coordinates @ decoder",
        "affine_membership_by_construction": True,
        "optimizer": "Adam",
        "steps": parsed.steps,
        "learning_rate": parsed.learning_rate,
        "ridge": parsed.ridge,
        "ridge_center": "initial_coordinates",
        "trust_radius": parsed.trust_radius,
        "trust_scope": "per_flattened_coordinate_row_l2_from_initial",
        "trust_projection_count": trust_projection_count,
        "selection": "minimum_kl_then_downstream_loss_then_earliest_step",
        "selected_step": selected_step,
        "selected_is_final_step": selected_step == parsed.steps,
        "initial_downstream_loss": initial_evaluation["downstream_loss"],
        "selected_downstream_loss": selected["downstream_loss"],
        "final_downstream_loss": final["downstream_loss"],
        "initial_kl": initial_evaluation["kl"],
        "selected_kl": selected["kl"],
        "final_kl": final["kl"],
        "selected_loss_reduced_from_initial": bool(
            float(selected["downstream_loss"])
            < float(initial_evaluation["downstream_loss"])
        ),
        "selected_kl_reduced_from_initial": bool(
            float(selected["kl"]) < float(initial_evaluation["kl"])
        ),
        "bias_and_decoder_detached_clones": True,
        "source_tensors_optimizer_owned": False,
        "evaluation_count": len(evaluations),
        "evaluations": evaluations,
    }
    return DownstreamAffineCoordinateSolution(
        coordinates=best_coordinates.detach().clone(),
        candidate=selected_candidate,
        receipt=receipt,
    )


def solve_batched_downstream_sensitive_affine_coordinates(
    bias: Tensor,
    decoder: Tensor,
    initial_coordinates: Tensor,
    loss_callback: BatchedDownstreamLossCallback,
    *,
    config: DownstreamAffineSolverConfig = DownstreamAffineSolverConfig(),
    learning_rate_by_row: Tensor | None = None,
) -> DownstreamAffineCoordinateSolution:
    """Optimize and select every affine row independently in one Adam solve.

    ``initial_coordinates`` must have shape ``[rows, rank]``.  The callback
    must return ``loss`` and ``kl`` vectors with shape ``[rows]``; scalar or
    broadcast/shared metrics are rejected.  Adam receives the gradient of the
    sum of the per-row regularized objectives, which is numerically equivalent
    to independent solves when the callback's rows are themselves independent.

    ``learning_rate_by_row``, when supplied, must be a positive ``[rows]``
    vector.  A single Adam optimizer owns one parameter group per row, so each
    row has both its own exact Adam state and its own learning rate while the
    expensive downstream callback remains batched.

    Checkpoint selection is also independent: each row chooses minimum KL,
    then minimum downstream loss, then earliest step, with step zero acting as
    a safe abstention.  The compact receipt stores scalar lists and hashes, not
    coordinate, candidate, or optimization-trace tensors.
    """

    parsed = _validated_config(config)
    _validate_affine_inputs(bias, decoder, initial_coordinates)
    if initial_coordinates.ndim != 2:
        raise ValueError(
            "initial_coordinates must have shape [rows, rank] for a "
            "batched solve"
        )
    if initial_coordinates.device.type != "cpu":
        raise ValueError("batched affine coordinate solves require CPU tensors")
    if not callable(loss_callback):
        raise TypeError("loss_callback must be callable")

    frozen_bias = bias.detach().clone()
    frozen_decoder = decoder.detach().clone()
    initial = initial_coordinates.detach().clone()
    row_count, rank = initial.shape
    if learning_rate_by_row is None:
        row_learning_rates = torch.full(
            (row_count,),
            parsed.learning_rate,
            dtype=initial.dtype,
            device=initial.device,
        )
        learning_rate_source = "config_scalar_broadcast_to_rows"
    else:
        if not isinstance(learning_rate_by_row, Tensor):
            raise TypeError("learning_rate_by_row must be a torch.Tensor")
        if learning_rate_by_row.shape != (row_count,):
            raise ValueError("learning_rate_by_row must have shape [rows]")
        if not learning_rate_by_row.is_floating_point():
            raise TypeError("learning_rate_by_row must have a floating dtype")
        if learning_rate_by_row.device != initial.device:
            raise ValueError(
                "learning_rate_by_row must be on the coordinates device"
            )
        if not bool(torch.isfinite(learning_rate_by_row.detach()).all().item()):
            raise ValueError("learning_rate_by_row must contain finite values")
        if not bool((learning_rate_by_row.detach() > 0.0).all().item()):
            raise ValueError("learning_rate_by_row values must be positive")
        row_learning_rates = learning_rate_by_row.detach().clone()
        learning_rate_source = "explicit_per_row_vector"

    # One Parameter and optimizer group per row preserves torch Adam's exact
    # independent moment and learning-rate semantics.  Stacking the parameters
    # only for candidate materialization keeps the costly callback vectorized.
    coordinate_rows = tuple(
        torch.nn.Parameter(row.clone()) for row in initial.unbind(dim=0)
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [coordinate_row],
                "lr": float(row_learning_rates[row_index].item()),
            }
            for row_index, coordinate_row in enumerate(coordinate_rows)
        ],
        lr=parsed.learning_rate,
        weight_decay=0.0,
    )

    best_keys: list[tuple[float, float, int] | None] = [None] * row_count
    selected_steps = [-1] * row_count
    best_coordinates = torch.empty_like(initial)
    best_candidate = torch.empty(
        (row_count, frozen_decoder.shape[1]),
        dtype=initial.dtype,
        device=initial.device,
    )
    initial_losses: list[float] | None = None
    initial_kls: list[float] | None = None
    selected_losses = [math.inf] * row_count
    selected_kls = [math.inf] * row_count
    final_losses: list[float] | None = None
    final_kls: list[float] | None = None
    trust_projection_counts = [0] * row_count
    trace_digest = hashlib.sha256()

    for step in range(parsed.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        coordinates = torch.stack(coordinate_rows, dim=0)
        candidate = materialize_affine_candidate(
            frozen_bias,
            coordinates,
            frozen_decoder,
        )
        downstream_losses, kls = _callback_row_vectors(
            loss_callback,
            candidate,
        )
        displacement = coordinates - initial
        ridge_mean_squares = displacement.square().mean(dim=-1)
        ridge_penalties = ridge_mean_squares * parsed.ridge
        per_row_objectives = downstream_losses + ridge_penalties
        objective_sum = per_row_objectives.sum()
        if not bool(torch.isfinite(objective_sum.detach()).all().item()):
            raise RuntimeError("summed regularized objective is nonfinite")

        loss_values = [
            float(value) for value in downstream_losses.detach().cpu().tolist()
        ]
        kl_values = [float(value) for value in kls.detach().cpu().tolist()]
        if step == 0:
            initial_losses = list(loss_values)
            initial_kls = list(kl_values)
        if step == parsed.steps:
            final_losses = list(loss_values)
            final_kls = list(kl_values)

        # Hash the full optimization evidence without putting row-by-step
        # vectors or coordinates into the returned receipt.
        trace_row = {
            "step": step,
            "coordinates_sha256": _tensor_sha256(coordinates),
            "downstream_loss_sha256": _tensor_sha256(downstream_losses),
            "kl_sha256": _tensor_sha256(kls),
            "ridge_mean_square_sha256": _tensor_sha256(ridge_mean_squares),
        }
        trace_digest.update(
            json.dumps(
                trace_row,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        trace_digest.update(b"\n")

        with torch.no_grad():
            for row_index, (loss_value, kl_value) in enumerate(
                zip(loss_values, kl_values, strict=True)
            ):
                key = (kl_value, loss_value, step)
                if (
                    best_keys[row_index] is None
                    or key < best_keys[row_index]
                ):
                    best_keys[row_index] = key
                    selected_steps[row_index] = step
                    selected_losses[row_index] = loss_value
                    selected_kls[row_index] = kl_value
                    best_coordinates[row_index].copy_(coordinates[row_index])
                    best_candidate[row_index].copy_(candidate[row_index])

        if step == parsed.steps:
            continue

        downstream_gradient_rows = torch.autograd.grad(
            downstream_losses.sum(),
            coordinate_rows,
            allow_unused=True,
        )
        if any(gradient is None for gradient in downstream_gradient_rows):
            raise RuntimeError(
                "loss_callback loss is not connected to affine coordinates"
            )
        gradient = torch.stack(
            [
                gradient
                for gradient in downstream_gradient_rows
                if gradient is not None
            ],
            dim=0,
        )
        if parsed.ridge != 0.0:
            # Sum_i ridge * mean_rank((z_i - z_i0)^2) gives the same
            # gradient for row i as its own scalar-solver objective.
            gradient = gradient + (
                (2.0 * parsed.ridge / rank) * displacement
            )
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError("coordinate gradient is nonfinite")
        for coordinate_row, gradient_row in zip(
            coordinate_rows, gradient.detach(), strict=True
        ):
            coordinate_row.grad = gradient_row.clone()
        optimizer.step()
        updated_coordinates = torch.stack(coordinate_rows, dim=0)
        if not bool(torch.isfinite(updated_coordinates).all().item()):
            raise RuntimeError("optimizer produced nonfinite coordinates")
        projected = _apply_batched_trust_region(
            updated_coordinates,
            initial,
            parsed.trust_radius,
        )
        with torch.no_grad():
            for coordinate_row, projected_row in zip(
                coordinate_rows, updated_coordinates, strict=True
            ):
                coordinate_row.copy_(projected_row)
        for row_index, was_projected in enumerate(
            projected.detach().cpu().tolist()
        ):
            trust_projection_counts[row_index] += int(was_projected)

    if (
        initial_losses is None
        or initial_kls is None
        or final_losses is None
        or final_kls is None
        or any(key is None for key in best_keys)
        or any(step < 0 for step in selected_steps)
    ):
        raise RuntimeError("batched affine coordinate solve was incomplete")

    selected_candidate = materialize_affine_candidate(
        frozen_bias,
        best_coordinates,
        frozen_decoder,
    ).detach().clone()
    if not torch.equal(selected_candidate, best_candidate):
        raise RuntimeError("selected affine candidates failed exact replay")

    def aggregate(values: list[float]) -> dict[str, float]:
        value_tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "sum": float(value_tensor.sum().item()),
            "mean": float(value_tensor.mean().item()),
            "minimum": float(value_tensor.min().item()),
            "maximum": float(value_tensor.max().item()),
        }

    receipt: dict[str, object] = {
        "schema": "fisher_graph.batched_downstream_affine_coordinate_solve.v1",
        "candidate_formula": "bias + coordinates @ decoder",
        "affine_membership_by_construction": True,
        "optimizer": "Adam",
        "gradient_reduction": "sum_of_per_row_regularized_objectives",
        "row_count": row_count,
        "rank": rank,
        "steps": parsed.steps,
        "learning_rate": parsed.learning_rate,
        "learning_rate_source": learning_rate_source,
        "per_row_learning_rate": [
            float(value) for value in row_learning_rates.cpu().tolist()
        ],
        "ridge": parsed.ridge,
        "ridge_center": "each_initial_coordinate_row",
        "trust_radius": parsed.trust_radius,
        "trust_scope": "per_coordinate_row_l2_from_its_initial_row",
        "selection": (
            "independent_per_row_minimum_kl_then_downstream_loss_then_"
            "earliest_step"
        ),
        "step_zero_is_abstention": True,
        "per_row_selected_steps": selected_steps,
        "per_row_initial_downstream_loss": initial_losses,
        "per_row_selected_downstream_loss": selected_losses,
        "per_row_final_downstream_loss": final_losses,
        "per_row_initial_kl": initial_kls,
        "per_row_selected_kl": selected_kls,
        "per_row_final_kl": final_kls,
        "per_row_trust_projection_counts": trust_projection_counts,
        "selected_loss_reduction_count": sum(
            selected < initial_value
            for selected, initial_value in zip(
                selected_losses, initial_losses, strict=True
            )
        ),
        "selected_kl_reduction_count": sum(
            selected < initial_value
            for selected, initial_value in zip(
                selected_kls, initial_kls, strict=True
            )
        ),
        "aggregates": {
            "initial_downstream_loss": aggregate(initial_losses),
            "selected_downstream_loss": aggregate(selected_losses),
            "final_downstream_loss": aggregate(final_losses),
            "initial_kl": aggregate(initial_kls),
            "selected_kl": aggregate(selected_kls),
            "final_kl": aggregate(final_kls),
            "selected_step": aggregate(
                [float(step) for step in selected_steps]
            ),
            "trust_projection_count": sum(trust_projection_counts),
        },
        "hashes": {
            "source_bias_sha256": _tensor_sha256(bias),
            "source_decoder_sha256": _tensor_sha256(decoder),
            "source_initial_coordinates_sha256": _tensor_sha256(
                initial_coordinates
            ),
            "learning_rate_by_row_sha256": _tensor_sha256(
                row_learning_rates
            ),
            "selected_coordinates_sha256": _tensor_sha256(best_coordinates),
            "selected_candidate_sha256": _tensor_sha256(selected_candidate),
            "optimization_trace_sha256": trace_digest.hexdigest(),
        },
        "evaluation_count_per_row": parsed.steps + 1,
        "determinism_scope": "fixed_cpu_inputs_and_deterministic_callback",
        "bias_and_decoder_detached_clones": True,
        "source_tensors_optimizer_owned": False,
        "large_tensor_payloads_in_receipt": False,
    }
    return DownstreamAffineCoordinateSolution(
        coordinates=best_coordinates.detach().clone(),
        candidate=selected_candidate,
        receipt=receipt,
    )


__all__ = [
    "BatchedDownstreamLossCallback",
    "DownstreamAffineCoordinateSolution",
    "DownstreamAffineSolverConfig",
    "DownstreamLossCallback",
    "materialize_affine_candidate",
    "solve_batched_downstream_sensitive_affine_coordinates",
    "solve_downstream_sensitive_affine_coordinates",
]
