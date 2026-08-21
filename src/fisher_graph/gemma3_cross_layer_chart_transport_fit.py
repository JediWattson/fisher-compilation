"""Deterministic numerical fits for adjacent chart-generator transport.

The helpers in this module operate only on already captured tensors.  They do
not load a model, tokenize prompts, mutate an executor, or open a data split.
Dense chart-to-mode coefficient banks are factored into the tensors consumed
by :class:`Gemma3FactorizedChartGeneratorLayer`.  Frozen source packets can
then be screened and fitted as square, adjacent-layer transport edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


_REFINEMENT_SWEEPS = 8


def _as_fit_tensor(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite floating CPU Tensor")
    return value.detach().to(dtype=torch.float64).contiguous()


def _normalized_row_fisher_weights(
    value: Tensor,
    *,
    row_count: int,
) -> Tensor:
    weights = _as_fit_tensor(value, label="row_fisher_weights")
    if weights.shape != (row_count,) or bool((weights <= 0.0).any()):
        raise ValueError("row_fisher_weights must be a positive [rows] Tensor")
    return weights / weights.mean()


def _validated_relative_ridge(value: float) -> float:
    if (
        not isinstance(value, float)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")
    return value


def _relative_rmse(actual: Tensor, expected: Tensor) -> float:
    numerator = float((actual - expected).square().sum().sqrt().item())
    denominator = float(expected.square().sum().sqrt().item())
    return numerator / max(denominator, torch.finfo(torch.float64).tiny)


def fit_fisher_weighted_affine_base(
    *,
    inputs: Tensor,
    modal_states: Tensor,
    row_fisher_weights: Tensor,
    relative_ridge: float,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Fit the dense affine modal base with a weighted float64 dual solve.

    The intercept is handled by Fisher-weighted centering and is therefore not
    ridge-penalized.  Returned executor tensors are always CPU float32.
    """

    features = _as_fit_tensor(inputs, label="inputs")
    targets = _as_fit_tensor(modal_states, label="modal_states")
    ridge_multiplier = _validated_relative_ridge(relative_ridge)
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError("inputs must have shape [rows, hidden_width]")
    rows, hidden_width = features.shape
    if (
        targets.ndim != 2
        or targets.shape[0] != rows
        or not targets.shape[1]
    ):
        raise ValueError("modal_states must have shape [rows, mode_count]")
    mode_count = targets.shape[1]
    weights = _normalized_row_fisher_weights(
        row_fisher_weights,
        row_count=rows,
    )
    weight_sum = weights.sum()
    input_mean = (weights.unsqueeze(1) * features).sum(dim=0) / weight_sum
    target_mean = (weights.unsqueeze(1) * targets).sum(dim=0) / weight_sum
    root_weight = weights.sqrt().unsqueeze(1)
    weighted_inputs = (features - input_mean) * root_weight
    weighted_targets = (targets - target_mean) * root_weight
    dual_gram = weighted_inputs @ weighted_inputs.T
    gram_scale = float(dual_gram.diag().mean().clamp_min(1.0e-12).item())
    absolute_ridge = ridge_multiplier * gram_scale
    dual_solution = torch.linalg.solve(
        dual_gram
        + absolute_ridge * torch.eye(rows, dtype=torch.float64),
        weighted_targets,
    )
    input_to_modes = weighted_inputs.T @ dual_solution
    bias = target_mean - input_mean @ input_to_modes
    affine_weight = input_to_modes.T.to(dtype=torch.float32).contiguous()
    affine_bias = bias.to(dtype=torch.float32).contiguous()
    prediction = features @ affine_weight.double().T + affine_bias.double()
    residual = targets - prediction
    weighted_target_energy = _weighted_squared_error(targets, weights)
    weighted_residual_error = _weighted_squared_error(residual, weights)
    metrics: dict[str, object] = {
        "row_count": rows,
        "hidden_width": hidden_width,
        "mode_count": mode_count,
        "fit_algorithm": "fisher_weighted_centered_dual_ridge_float64",
        "materialized_dtype": "float32",
        "relative_ridge": ridge_multiplier,
        "absolute_ridge": absolute_ridge,
        "normalized_row_weight_minimum": float(weights.min().item()),
        "normalized_row_weight_maximum": float(weights.max().item()),
        "weighted_target_squared_norm": weighted_target_energy,
        "weighted_squared_error": weighted_residual_error,
        "weighted_relative_rmse": math.sqrt(
            weighted_residual_error
            / max(weighted_target_energy, torch.finfo(torch.float64).tiny)
        ),
        "relative_rmse": _relative_rmse(prediction, targets),
    }
    return affine_weight, affine_bias, metrics


def fit_fisher_weighted_dense_chart_edge_bank(
    *,
    memberships: Tensor,
    coordinates: Tensor,
    base_modal_residual: Tensor,
    row_fisher_weights: Tensor,
    relative_ridge: float,
) -> tuple[Tensor, dict[str, object]]:
    """Jointly fit all dense chart-to-mode edges with weighted ridge.

    ``memberships`` and ``coordinates`` must be frozen outputs of the fit-only
    chart geometry.  The design is the flattened bank
    ``q_c * [1, z_c]``.  Returned edge coefficients are CPU float32.
    """

    chart_memberships = _as_fit_tensor(
        memberships,
        label="memberships",
    )
    chart_coordinates = _as_fit_tensor(
        coordinates,
        label="coordinates",
    )
    residual_target = _as_fit_tensor(
        base_modal_residual,
        label="base_modal_residual",
    )
    ridge_multiplier = _validated_relative_ridge(relative_ridge)
    if (
        chart_memberships.ndim != 2
        or not chart_memberships.shape[0]
        or not chart_memberships.shape[1]
    ):
        raise ValueError("memberships must have shape [rows, charts]")
    rows, chart_count = chart_memberships.shape
    if (
        bool((chart_memberships < 0.0).any())
        or not torch.allclose(
            chart_memberships.sum(dim=1),
            torch.ones(rows, dtype=torch.float64),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    ):
        raise ValueError("memberships must be nonnegative and sum to one")
    if (
        chart_coordinates.ndim != 3
        or chart_coordinates.shape[:2] != (rows, chart_count)
        or not chart_coordinates.shape[2]
    ):
        raise ValueError("coordinates must have shape [rows, charts, rank]")
    chart_rank = chart_coordinates.shape[2]
    if (
        residual_target.ndim != 2
        or residual_target.shape[0] != rows
        or not residual_target.shape[1]
    ):
        raise ValueError(
            "base_modal_residual must have shape [rows, mode_count]"
        )
    mode_count = residual_target.shape[1]
    weights = _normalized_row_fisher_weights(
        row_fisher_weights,
        row_count=rows,
    )
    augmented = torch.cat(
        (
            torch.ones(
                rows,
                chart_count,
                1,
                dtype=torch.float64,
            ),
            chart_coordinates,
        ),
        dim=2,
    )
    chart_feature_width = chart_rank + 1
    design = (chart_memberships.unsqueeze(2) * augmented).reshape(
        rows,
        chart_count * chart_feature_width,
    )
    root_weight = weights.sqrt().unsqueeze(1)
    weighted_design = design * root_weight
    weighted_target = residual_target * root_weight
    gram = weighted_design.T @ weighted_design
    gram_scale = float(gram.diag().mean().clamp_min(1.0e-12).item())
    absolute_ridge = ridge_multiplier * gram_scale
    coefficients = torch.linalg.solve(
        gram
        + absolute_ridge
        * torch.eye(gram.shape[0], dtype=torch.float64),
        weighted_design.T @ weighted_target,
    )
    chart_edge_weight = coefficients.reshape(
        chart_count,
        chart_feature_width,
        mode_count,
    ).to(dtype=torch.float32).contiguous()
    prediction = design @ chart_edge_weight.double().reshape(
        chart_count * chart_feature_width,
        mode_count,
    )
    fit_residual = residual_target - prediction
    baseline_error = _weighted_squared_error(residual_target, weights)
    fitted_error = _weighted_squared_error(fit_residual, weights)
    reduction = max(0.0, baseline_error - fitted_error)
    entropy = -(
        chart_memberships
        * chart_memberships.clamp_min(1.0e-12).log()
    ).sum(dim=1)
    metrics: dict[str, object] = {
        "row_count": rows,
        "chart_count": chart_count,
        "chart_rank": chart_rank,
        "chart_feature_width": chart_feature_width,
        "mode_count": mode_count,
        "fit_algorithm": "joint_fisher_weighted_chart_ridge_float64",
        "materialized_dtype": "float32",
        "relative_ridge": ridge_multiplier,
        "absolute_ridge": absolute_ridge,
        "dense_coefficient_count": chart_edge_weight.numel(),
        "normalized_row_weight_minimum": float(weights.min().item()),
        "normalized_row_weight_maximum": float(weights.max().item()),
        "baseline_weighted_squared_error": baseline_error,
        "fitted_weighted_squared_error": fitted_error,
        "weighted_squared_error_reduction": reduction,
        "weighted_explained_fraction": (
            0.0 if baseline_error == 0.0 else reduction / baseline_error
        ),
        "weighted_relative_rmse": math.sqrt(
            fitted_error
            / max(baseline_error, torch.finfo(torch.float64).tiny)
        ),
        "relative_rmse": _relative_rmse(prediction, residual_target),
        "mean_membership_entropy": float(entropy.mean().item()),
        "mean_max_membership": float(
            chart_memberships.max(dim=1).values.mean().item()
        ),
        "hard_chart_occupancy": torch.bincount(
            chart_memberships.argmax(dim=1),
            minlength=chart_count,
        ).tolist(),
    }
    return chart_edge_weight, metrics


def _canonicalize_svd_signs(
    left: Tensor,
    right: Tensor,
) -> tuple[Tensor, Tensor]:
    """Choose one deterministic sign for every retained SVD component."""

    canonical_left = left.clone()
    canonical_right = right.clone()
    for component in range(canonical_right.shape[0]):
        row = canonical_right[component]
        pivot = int(torch.argmax(row.abs()).item())
        if float(row[pivot]) < 0.0:
            canonical_left[:, component].neg_()
            canonical_right[component].neg_()
    return canonical_left, canonical_right


def factor_chart_edge_bank(
    chart_edge_weight: Tensor,
    generator_rank: int,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Factor ``[charts, features, modes]`` edges into local generators.

    For each chart, a truncated SVD gives ``E_c ~= L_c D_c``.  Singular
    values are absorbed into ``L_c`` and the rows of ``D_c`` remain
    orthonormal.  The returned tensors retain the input dtype and CPU device;
    the decomposition itself is performed in float64.
    """

    if (
        not isinstance(chart_edge_weight, Tensor)
        or chart_edge_weight.ndim != 3
        or not chart_edge_weight.is_floating_point()
        or chart_edge_weight.device.type != "cpu"
        or not bool(torch.isfinite(chart_edge_weight).all())
    ):
        raise ValueError(
            "chart_edge_weight must be a finite floating CPU "
            "[charts, features, modes] Tensor"
        )
    charts, features, modes = chart_edge_weight.shape
    if charts <= 0 or features <= 0 or modes <= 0:
        raise ValueError("chart_edge_weight dimensions must be positive")
    if (
        type(generator_rank) is not int
        or generator_rank <= 0
        or generator_rank > min(features, modes)
    ):
        raise ValueError("generator_rank exceeds the chart edge-bank rank")

    source_dtype = chart_edge_weight.dtype
    canonical = chart_edge_weight.detach().to(dtype=torch.float64)
    local_encoders: list[Tensor] = []
    mode_decoders: list[Tensor] = []
    for chart in range(charts):
        edge = canonical[chart]
        left, singular_values, right = torch.linalg.svd(
            edge,
            full_matrices=False,
        )
        local = left[:, :generator_rank] * singular_values[
            :generator_rank
        ].unsqueeze(0)
        decoder = right[:generator_rank]
        local, decoder = _canonicalize_svd_signs(local, decoder)
        reconstruction = local @ decoder
        local_encoders.append(local)
        mode_decoders.append(decoder)

    local_result = torch.stack(local_encoders).to(dtype=source_dtype)
    decoder_result = torch.stack(mode_decoders).to(dtype=source_dtype)
    reconstructed = torch.einsum(
        "csg,cgk->csk",
        local_result.double(),
        decoder_result.double(),
    )
    canonical_error = float(
        (canonical - reconstructed).square().sum().item()
    )
    canonical_norm = float(canonical.square().sum().item())
    chart_errors = (canonical - reconstructed).square().sum(dim=(1, 2))
    chart_norms = canonical.square().sum(dim=(1, 2))
    per_chart_energy = [
        (
            1.0
            if float(chart_norm) == 0.0
            else max(
                0.0,
                min(1.0, 1.0 - float(chart_error / chart_norm)),
            )
        )
        for chart_error, chart_norm in zip(chart_errors, chart_norms)
    ]
    identity = torch.eye(generator_rank, dtype=torch.float64)
    orthonormal_error = max(
        float(
            (
                decoder.double() @ decoder.double().T - identity
            ).abs().max().item()
        )
        for decoder in decoder_result
    )
    dense_count = chart_edge_weight.numel()
    factorized_count = local_result.numel() + decoder_result.numel()
    metrics: dict[str, object] = {
        "chart_count": charts,
        "chart_feature_width": features,
        "mode_count": modes,
        "generator_rank": generator_rank,
        "dense_coefficient_count": dense_count,
        "factorized_coefficient_count": factorized_count,
        "coefficient_reduction_fraction": (
            (dense_count - factorized_count) / dense_count
        ),
        "squared_frobenius_error": canonical_error,
        "relative_frobenius_error": math.sqrt(
            canonical_error / max(canonical_norm, torch.finfo(torch.float64).tiny)
        ),
        "captured_energy_fraction": (
            1.0
            if canonical_norm == 0.0
            else max(0.0, min(1.0, 1.0 - canonical_error / canonical_norm))
        ),
        "per_chart_captured_energy_fraction": per_chart_energy,
        "decoder_row_orthonormality_max_abs_error": orthonormal_error,
        "svd_sign_convention": "largest_absolute_decoder_entry_positive",
    }
    return local_result, decoder_result, metrics


@dataclass(frozen=True, slots=True)
class FittedAdjacentTransportEdge:
    """One fit-only square source-chart to target-chart transport edge."""

    source_chart_index: int
    target_chart_index: int
    transport_weight: Tensor
    greedy_step: int
    greedy_weighted_sse_gain: float
    absolute_ridge: float

    def __post_init__(self) -> None:
        if (
            type(self.source_chart_index) is not int
            or type(self.target_chart_index) is not int
            or self.source_chart_index < 0
            or self.target_chart_index < 0
        ):
            raise ValueError("transport chart indices must be nonnegative")
        if type(self.greedy_step) is not int or self.greedy_step < 0:
            raise ValueError("greedy_step must be nonnegative")
        if (
            not isinstance(self.transport_weight, Tensor)
            or self.transport_weight.ndim != 2
            or self.transport_weight.shape[0]
            != self.transport_weight.shape[1]
            or not self.transport_weight.shape[0]
            or self.transport_weight.device.type != "cpu"
            or not self.transport_weight.is_floating_point()
            or not bool(torch.isfinite(self.transport_weight).all())
        ):
            raise ValueError("transport_weight must be a finite square CPU Tensor")
        for label, value in (
            ("greedy_weighted_sse_gain", self.greedy_weighted_sse_gain),
            ("absolute_ridge", self.absolute_ridge),
        ):
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.greedy_weighted_sse_gain < 0.0 or self.absolute_ridge <= 0.0:
            raise ValueError("transport edge fit metrics are outside their domain")
        object.__setattr__(
            self,
            "transport_weight",
            self.transport_weight.detach().clone().contiguous(),
        )

    @property
    def coordinate(self) -> tuple[int, int]:
        return self.source_chart_index, self.target_chart_index

    def metadata(self) -> dict[str, object]:
        return {
            "source_chart_index": self.source_chart_index,
            "target_chart_index": self.target_chart_index,
            "generator_rank": int(self.transport_weight.shape[0]),
            "greedy_step": self.greedy_step,
            "greedy_weighted_sse_gain": self.greedy_weighted_sse_gain,
            "absolute_ridge": self.absolute_ridge,
            "transport_weight_frobenius_norm": float(
                self.transport_weight.double().square().sum().sqrt().item()
            ),
        }


@dataclass(slots=True)
class _MutableEdgeFit:
    source_chart_index: int
    target_chart_index: int
    transport_weight: Tensor
    prediction: Tensor
    design: Tensor
    decoder: Tensor
    gram: Tensor
    absolute_ridge: float
    greedy_step: int
    greedy_weighted_sse_gain: float


def _weighted_squared_error(residual: Tensor, row_weights: Tensor) -> float:
    return float(
        (row_weights.unsqueeze(1) * residual.square()).sum().item()
    )


def _fit_square_edge(
    *,
    design: Tensor,
    decoder: Tensor,
    residual: Tensor,
    row_weights: Tensor,
    relative_ridge: float,
) -> tuple[Tensor, Tensor, Tensor, float]:
    weighted_design = design * row_weights.sqrt().unsqueeze(1)
    gram = weighted_design.T @ weighted_design
    scale = float(gram.diag().mean().clamp_min(1.0e-12).item())
    absolute_ridge = relative_ridge * scale
    rhs = design.T @ (row_weights.unsqueeze(1) * residual) @ decoder.T
    transport = torch.linalg.solve(
        gram + absolute_ridge * torch.eye(gram.shape[0], dtype=torch.float64),
        rhs,
    )
    prediction = design @ transport @ decoder
    return transport, prediction, gram, absolute_ridge


def fit_greedy_adjacent_transport_edges(
    *,
    source_emitted_latents: Tensor,
    target_memberships: Tensor,
    target_mode_decoders: Tensor,
    target_modal_residual: Tensor,
    row_fisher_weights: Tensor,
    maximum_incoming_charts_per_target: int,
    relative_ridge: float,
) -> tuple[tuple[FittedAdjacentTransportEdge, ...], dict[str, object]]:
    """Greedily fit sparse square edges against a frozen local residual.

    Source packets must already include their source-chart membership.  This
    helper applies the target membership, so candidate ``c -> d`` uses design
    ``q_target[:, d] * source_packet[:, c]``.  Decoder rows must be
    orthonormal, as produced by :func:`factor_chart_edge_bank`.
    """

    source = _as_fit_tensor(
        source_emitted_latents,
        label="source_emitted_latents",
    )
    source_dtype = source_emitted_latents.dtype
    memberships = _as_fit_tensor(
        target_memberships,
        label="target_memberships",
    )
    decoders = _as_fit_tensor(
        target_mode_decoders,
        label="target_mode_decoders",
    )
    target = _as_fit_tensor(
        target_modal_residual,
        label="target_modal_residual",
    )
    weights = _as_fit_tensor(
        row_fisher_weights,
        label="row_fisher_weights",
    )
    if source.ndim != 3:
        raise ValueError("source_emitted_latents must have shape [rows, charts, rank]")
    rows, source_charts, generator_rank = source.shape
    if rows <= 0 or source_charts <= 0 or generator_rank <= 0:
        raise ValueError("source emitted-latent dimensions must be positive")
    if memberships.ndim != 2 or memberships.shape[0] != rows:
        raise ValueError("target_memberships must have shape [rows, charts]")
    target_charts = memberships.shape[1]
    if target_charts <= 0:
        raise ValueError("target chart count must be positive")
    if (
        decoders.ndim != 3
        or decoders.shape[:2] != (target_charts, generator_rank)
    ):
        raise ValueError(
            "target_mode_decoders must have shape "
            "[target_charts, generator_rank, modes]"
        )
    modes = decoders.shape[2]
    if modes <= 0:
        raise ValueError("target mode count must be positive")
    if target.shape != (rows, modes):
        raise ValueError("target_modal_residual must have shape [rows, modes]")
    if weights.shape != (rows,) or bool((weights <= 0.0).any()):
        raise ValueError("row_fisher_weights must be a positive [rows] Tensor")
    if (
        bool((memberships < 0.0).any())
        or not torch.allclose(
            memberships.sum(dim=1),
            torch.ones(rows, dtype=torch.float64),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    ):
        raise ValueError("target memberships must be nonnegative and sum to one")
    if generator_rank > modes:
        raise ValueError("generator rank cannot exceed mode count")
    identity = torch.eye(generator_rank, dtype=torch.float64)
    decoder_error = max(
        float((decoder @ decoder.T - identity).abs().max().item())
        for decoder in decoders
    )
    if decoder_error > 2.0e-5:
        raise ValueError("target mode-decoder rows must be orthonormal")
    if (
        type(maximum_incoming_charts_per_target) is not int
        or maximum_incoming_charts_per_target <= 0
        or maximum_incoming_charts_per_target > source_charts
    ):
        raise ValueError("maximum incoming charts per target is invalid")
    if (
        not isinstance(relative_ridge, float)
        or not math.isfinite(relative_ridge)
        or relative_ridge <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")

    weights = weights / weights.mean()
    baseline_sse = _weighted_squared_error(target, weights)
    remaining = target.clone()
    incoming_counts = [0 for _ in range(target_charts)]
    selected_coordinates: set[tuple[int, int]] = set()
    selected: list[_MutableEdgeFit] = []
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(
        1.0,
        baseline_sse,
    )
    maximum_edges = target_charts * maximum_incoming_charts_per_target
    for step in range(maximum_edges):
        current_sse = _weighted_squared_error(remaining, weights)
        best: _MutableEdgeFit | None = None
        best_gain = -math.inf
        for target_chart in range(target_charts):
            if (
                incoming_counts[target_chart]
                >= maximum_incoming_charts_per_target
            ):
                continue
            decoder = decoders[target_chart]
            target_membership = memberships[:, target_chart].unsqueeze(1)
            for source_chart in range(source_charts):
                coordinate = source_chart, target_chart
                if coordinate in selected_coordinates:
                    continue
                design = source[:, source_chart] * target_membership
                transport, prediction, gram, absolute_ridge = _fit_square_edge(
                    design=design,
                    decoder=decoder,
                    residual=remaining,
                    row_weights=weights,
                    relative_ridge=relative_ridge,
                )
                candidate_sse = _weighted_squared_error(
                    remaining - prediction,
                    weights,
                )
                gain = max(0.0, current_sse - candidate_sse)
                if gain > best_gain + tolerance or (
                    abs(gain - best_gain) <= tolerance
                    and best is not None
                    and coordinate
                    < (best.source_chart_index, best.target_chart_index)
                ):
                    best_gain = gain
                    best = _MutableEdgeFit(
                        source_chart_index=source_chart,
                        target_chart_index=target_chart,
                        transport_weight=transport,
                        prediction=prediction,
                        design=design,
                        decoder=decoder,
                        gram=gram,
                        absolute_ridge=absolute_ridge,
                        greedy_step=step,
                        greedy_weighted_sse_gain=gain,
                    )
        if best is None or best_gain <= tolerance:
            break
        selected.append(best)
        selected_coordinates.add(
            (best.source_chart_index, best.target_chart_index)
        )
        incoming_counts[best.target_chart_index] += 1
        remaining = remaining - best.prediction

    sweeps_completed = 0
    if selected:
        total_prediction = sum(
            (edge.prediction for edge in selected),
            torch.zeros_like(target),
        )
        previous_objective = math.inf
        for sweep in range(_REFINEMENT_SWEEPS):
            for edge in sorted(
                selected,
                key=lambda value: (
                    value.target_chart_index,
                    value.source_chart_index,
                ),
            ):
                residual_without_edge = target - (
                    total_prediction - edge.prediction
                )
                rhs = edge.design.T @ (
                    weights.unsqueeze(1) * residual_without_edge
                ) @ edge.decoder.T
                transport = torch.linalg.solve(
                    edge.gram
                    + edge.absolute_ridge
                    * torch.eye(generator_rank, dtype=torch.float64),
                    rhs,
                )
                prediction = edge.design @ transport @ edge.decoder
                total_prediction += prediction - edge.prediction
                edge.transport_weight = transport
                edge.prediction = prediction
            residual = target - total_prediction
            objective = _weighted_squared_error(residual, weights) + sum(
                edge.absolute_ridge
                * float(edge.transport_weight.square().sum().item())
                for edge in selected
            )
            if objective > previous_objective + tolerance:
                raise RuntimeError("transport ridge refinement was not monotone")
            sweeps_completed = sweep + 1
            if previous_objective - objective <= tolerance:
                break
            previous_objective = objective
        remaining = target - total_prediction

    fitted_sse = _weighted_squared_error(remaining, weights)
    reduction = max(0.0, baseline_sse - fitted_sse)
    edges = tuple(
        FittedAdjacentTransportEdge(
            source_chart_index=edge.source_chart_index,
            target_chart_index=edge.target_chart_index,
            transport_weight=edge.transport_weight.to(dtype=source_dtype),
            greedy_step=edge.greedy_step,
            greedy_weighted_sse_gain=float(
                edge.greedy_weighted_sse_gain
            ),
            absolute_ridge=float(edge.absolute_ridge),
        )
        for edge in sorted(
            selected,
            key=lambda value: (
                value.target_chart_index,
                value.source_chart_index,
            ),
        )
    )
    metrics: dict[str, object] = {
        "row_count": rows,
        "source_chart_count": source_charts,
        "target_chart_count": target_charts,
        "generator_rank": generator_rank,
        "mode_count": modes,
        "candidate_edge_count": source_charts * target_charts,
        "selected_edge_count": len(edges),
        "maximum_incoming_charts_per_target": (
            maximum_incoming_charts_per_target
        ),
        "incoming_edge_count_by_target": incoming_counts,
        "relative_ridge": relative_ridge,
        "baseline_weighted_squared_error": baseline_sse,
        "fitted_weighted_squared_error": fitted_sse,
        "weighted_squared_error_reduction": reduction,
        "transport_partial_explained_fraction": (
            0.0 if baseline_sse == 0.0 else reduction / baseline_sse
        ),
        "decoder_row_orthonormality_max_abs_error": decoder_error,
        "refinement_sweeps_completed": sweeps_completed,
        "edge_metadata": [edge.metadata() for edge in edges],
    }
    return edges, metrics


__all__ = [
    "FittedAdjacentTransportEdge",
    "factor_chart_edge_bank",
    "fit_fisher_weighted_affine_base",
    "fit_fisher_weighted_dense_chart_edge_bank",
    "fit_greedy_adjacent_transport_edges",
]
