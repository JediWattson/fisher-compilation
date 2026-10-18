"""Offline fitting for source-free state-conditioned modal interactions.

The teacher layer is permitted only in this module's offline inputs.  Teacher
flow can score candidate displacements and produce route responsibilities, but
the deployable artifacts contain only source-modal gate coefficients and
polynomial velocity factors.

The fit has two deliberately separate stages:

* ``teacher_flow_routing`` compares candidate hidden-space displacements with
  teacher direction and relative flow error, then returns fit-only
  responsibilities and hard labels.
* ``fit_state_conditioned_modal_interactions`` distils those labels into a
  current-state linear router, fits an affine modal message on each assigned
  route, and optionally adds a tangent-preserving factorized quadratic
  residual.

Router standardization is folded algebraically into each edge's gate weight
and bias.  Runtime therefore reads only the final source modal coordinate;
there is no teacher-flow, target-message, calibration-row, or callback field
in ``StateConditionedModalGeneratorInteraction``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .conditional_quadratic_edge import (
    TangentPreservingQuadraticSample,
    fit_tangent_preserving_quadratic_edge,
)
from .conditional_routing import (
    RouterClassificationMetrics,
    fit_pointwise_causal_router,
)
from .modal_generator_graph import (
    StateConditionedModalGeneratorInteraction,
)


__all__ = [
    "EWCStateConditionedEdgeDiagnostics",
    "EWCStateConditionedModalInteractionFit",
    "PullbackFisherSourceChart",
    "SharedPullbackFisherLayerChart",
    "StateConditionedEdgeFitMetrics",
    "StateConditionedModalInteractionFit",
    "TeacherFlowRouting",
    "fit_ewc_state_conditioned_modal_interactions",
    "fit_shared_pullback_fisher_layer_chart",
    "fit_state_conditioned_modal_interactions",
    "pullback_fisher_source_chart",
    "teacher_flow_routing",
]


def _positive_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite_float64(value: Tensor, *, label: str, ndim_min: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.ndim < ndim_min
        or any(int(width) <= 0 for width in value.shape)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty floating Tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()


@dataclass(frozen=True, slots=True)
class TeacherFlowRouting:
    """Fit-only route assignments derived from teacher-flow alignment."""

    responsibilities: Tensor
    route_labels: Tensor
    mean_best_alignment: float
    minimum_best_alignment: float
    mean_selected_relative_residual: float
    maximum_selected_relative_residual: float

    @property
    def routes(self) -> int:
        return int(self.responsibilities.shape[-1])

    @property
    def observations(self) -> int:
        return int(self.route_labels.numel())


def teacher_flow_routing(
    teacher_flow: Tensor,
    candidate_displacements: Tensor,
    *,
    prior_logits: Tensor | None = None,
    alignment_weight: float = 1.0,
    residual_weight: float = 1.0,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> TeacherFlowRouting:
    """Return teacher-only soft responsibilities and deterministic labels.

    ``teacher_flow`` has shape ``[..., d]`` and candidate displacements have
    shape ``[..., routes, d]``.  Equal scores resolve to the first candidate,
    so callers should place candidates in canonical target-node order. Scores
    combine cosine alignment with a scale-free squared flow residual, avoiding
    ties between correctly sized and arbitrarily oversized parallel vectors.
    """

    teacher = _finite_float64(
        teacher_flow,
        label="teacher_flow",
        ndim_min=2,
    )
    candidates = _finite_float64(
        candidate_displacements,
        label="candidate_displacements",
        ndim_min=3,
    )
    if (
        candidates.shape[:-2] != teacher.shape[:-1]
        or candidates.shape[-1] != teacher.shape[-1]
        or candidates.shape[-2] < 2
    ):
        raise ValueError(
            "candidate displacements must align with teacher flow and contain "
            "at least two routes"
        )
    weight = _nonnegative_float(
        alignment_weight,
        label="alignment_weight",
    )
    flow_weight = _nonnegative_float(
        residual_weight,
        label="residual_weight",
    )
    tau = _positive_float(temperature, label="temperature")
    eps = _positive_float(epsilon, label="epsilon")
    teacher_norm = torch.linalg.vector_norm(teacher, dim=-1, keepdim=True)
    candidate_norm = torch.linalg.vector_norm(
        candidates,
        dim=-1,
    )
    numerator = (candidates * teacher.unsqueeze(-2)).sum(dim=-1)
    denominator = teacher_norm * candidate_norm
    alignment = numerator / denominator.clamp_min(eps)
    relative_residual = (
        (candidates - teacher.unsqueeze(-2)).square().sum(dim=-1)
        / teacher.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    )
    scores = weight * alignment - flow_weight * relative_residual
    if prior_logits is not None:
        prior = _finite_float64(
            prior_logits,
            label="prior_logits",
            ndim_min=2,
        )
        if prior.shape != scores.shape:
            raise ValueError("prior_logits must match route score shape")
        scores = scores + prior
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("teacher flow route scores became non-finite")
    # Center before temperature scaling. With a tiny positive temperature,
    # scaling raw positive logits can overflow even though their differences
    # define a perfectly valid near-hard responsibility distribution.
    centered_scores = scores - scores.max(dim=-1, keepdim=True).values
    responsibilities = torch.softmax(centered_scores / tau, dim=-1)
    if not bool(torch.isfinite(responsibilities).all()):
        raise ValueError("teacher flow responsibilities became non-finite")
    labels = scores.argmax(dim=-1).to(dtype=torch.int64)
    best = alignment.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    selected_residual = relative_residual.gather(
        -1,
        labels.unsqueeze(-1),
    ).squeeze(-1)
    return TeacherFlowRouting(
        responsibilities=responsibilities.contiguous(),
        route_labels=labels.contiguous(),
        mean_best_alignment=float(best.mean().item()),
        minimum_best_alignment=float(best.min().item()),
        mean_selected_relative_residual=float(
            selected_residual.mean().item()
        ),
        maximum_selected_relative_residual=float(
            selected_residual.max().item()
        ),
    )


@dataclass(frozen=True, slots=True)
class StateConditionedEdgeFitMetrics:
    """Assigned-row fidelity for one distilled polynomial interaction."""

    target_node: str
    observations: int
    affine_mse: float
    polynomial_mse: float
    polynomial_improvement_fraction: float


@dataclass(frozen=True, slots=True)
class StateConditionedModalInteractionFit:
    """Inspectable result of one source routing-group distillation."""

    interactions: tuple[StateConditionedModalGeneratorInteraction, ...]
    router_metrics: RouterClassificationMetrics
    edge_metrics: tuple[StateConditionedEdgeFitMetrics, ...]
    route_counts: tuple[int, ...]
    router_fold_audit_observations: int
    router_fold_float32_max_abs_error: float
    router_fold_float32_tolerance: float

    @property
    def parameter_count(self) -> int:
        return sum(edge.parameter_count for edge in self.interactions)

    @property
    def dense_macs_per_token(self) -> int:
        return sum(edge.macs_per_token for edge in self.interactions)


def _fit_affine_message(
    source: Tensor,
    target: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    ones = torch.ones((source.shape[0], 1), dtype=torch.float64)
    design = torch.cat((source, ones), dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(
        design.T @ design + penalty,
        design.T @ target,
    )
    if not bool(torch.isfinite(coefficients).all()):
        raise RuntimeError("state-conditioned affine message fit became non-finite")
    return coefficients[:-1].contiguous(), coefficients[-1].contiguous()


def fit_state_conditioned_modal_interactions(
    source_states: Tensor,
    target_messages: Mapping[str, Tensor],
    route_labels: Tensor,
    *,
    source_node: str,
    routing_group: str,
    router_validation_states: Tensor | None = None,
    temperature: float = 1.0,
    top_k: int = 1,
    router_ridge: float = 1e-3,
    message_ridge: float = 1e-6,
    quadratic_rank: int = 0,
    quadratic_steps: int = 400,
    quadratic_learning_rate: float = 1e-2,
    quadratic_ridge: float = 1e-6,
    seed: int = 0,
) -> StateConditionedModalInteractionFit:
    """Distil hard teacher assignments into source-only routed edge artifacts.

    Route ids correspond to target nodes in sorted canonical order.  Message
    fits use only rows assigned to that route.  When ``quadratic_rank`` is
    positive, the affine map is fixed as the local tangent and a deterministic
    low-rank quadratic residual is fitted around it.
    """

    source = _finite_float64(
        source_states,
        label="source_states",
        ndim_min=2,
    )
    leading_shape = source.shape[:-1]
    rows = source.reshape(-1, source.shape[-1])
    if not isinstance(target_messages, Mapping) or len(target_messages) < 2:
        raise ValueError("target_messages must contain at least two routes")
    target_nodes = tuple(sorted(target_messages))
    if len(target_nodes) != len(set(target_nodes)):
        raise ValueError("target message names must be unique")
    targets: dict[str, Tensor] = {}
    for name in target_nodes:
        value = _finite_float64(
            target_messages[name],
            label=f"target_messages[{name!r}]",
            ndim_min=2,
        )
        if value.shape[:-1] != leading_shape:
            raise ValueError("target messages must share source leading shape")
        targets[name] = value.reshape(-1, value.shape[-1])
    if (
        not isinstance(route_labels, Tensor)
        or route_labels.dtype not in (torch.int32, torch.int64)
        or route_labels.shape != leading_shape
    ):
        raise ValueError("route_labels must be integer and match leading shape")
    labels = route_labels.detach().to(
        device="cpu",
        dtype=torch.int64,
    ).reshape(-1)
    routes = len(target_nodes)
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= routes:
        raise ValueError("route_labels exceed the target route range")
    counts_tensor = torch.bincount(labels, minlength=routes)
    if bool((counts_tensor == 0).any()):
        raise ValueError("every state-conditioned route needs assigned rows")
    if type(top_k) is not int or top_k != 1:
        raise ValueError(
            "hard-label state-conditioned fitting currently requires top_k=1"
        )
    tau = _positive_float(temperature, label="temperature")
    route_ridge = _positive_float(router_ridge, label="router_ridge")
    message_regularization = _nonnegative_float(
        message_ridge,
        label="message_ridge",
    )
    rank = _nonnegative_int(quadratic_rank, label="quadratic_rank")
    steps = _positive_int(quadratic_steps, label="quadratic_steps")
    learning_rate = _positive_float(
        quadratic_learning_rate,
        label="quadratic_learning_rate",
    )
    quadratic_regularization = _nonnegative_float(
        quadratic_ridge,
        label="quadratic_ridge",
    )
    fit_seed = _nonnegative_int(seed, label="seed")

    router, router_metrics = fit_pointwise_causal_router(
        rows,
        labels,
        route_count=routes,
        ridge=route_ridge,
    )
    # Fold ((x - mean) / scale) @ W + b into x @ W' + b'.
    folded_weight = router.weight / router.feature_scale[:, None]
    folded_bias = router.bias - (
        router.feature_mean / router.feature_scale
    ) @ router.weight
    # Serving stores one raw-coordinate affine gate per edge. Algebraically
    # folding centering/scaling into that gate can become cancellation-prone
    # for large-offset coordinates, even when the float64 formula is exact.
    # Refuse to emit an artifact unless the folded form reproduces the
    # normalized router in the float32 routing precision used by Gemma.
    audit_rows = rows
    if router_validation_states is not None:
        validation = _finite_float64(
            router_validation_states,
            label="router_validation_states",
            ndim_min=2,
        )
        if validation.shape[-1] != rows.shape[-1]:
            raise ValueError(
                "router_validation_states must match the source width"
            )
        audit_rows = torch.cat(
            (rows, validation.reshape(-1, validation.shape[-1])),
            dim=0,
        )
    runtime_rows = audit_rows.to(dtype=torch.float32)
    reference_logits = (
        (runtime_rows - router.feature_mean.to(dtype=torch.float32))
        / router.feature_scale.to(dtype=torch.float32)
    ) @ router.weight.to(dtype=torch.float32) + router.bias.to(
        dtype=torch.float32
    )
    folded_logits = (
        runtime_rows @ folded_weight.to(dtype=torch.float32)
        + folded_bias.to(dtype=torch.float32)
    )
    if not bool(torch.isfinite(folded_logits).all()):
        raise RuntimeError("folded state-conditioned router became non-finite")
    fold_error = float(
        (folded_logits - reference_logits).abs().max().item()
    )
    reference_scale = max(
        1.0,
        float(reference_logits.abs().max().item()),
    )
    fold_tolerance = 1e-6 + 1e-5 * reference_scale
    if (
        fold_error > fold_tolerance
        or not torch.equal(
            folded_logits.argmax(dim=-1),
            reference_logits.argmax(dim=-1),
        )
    ):
        raise RuntimeError(
            "folded state-conditioned router is not float32 route-stable; "
            "center or rescale the source modal coordinates"
        )

    interactions: list[StateConditionedModalGeneratorInteraction] = []
    metrics: list[StateConditionedEdgeFitMetrics] = []
    for route, target_node in enumerate(target_nodes):
        selected = torch.nonzero(labels == route, as_tuple=False).flatten()
        route_source = rows.index_select(0, selected)
        route_target = targets[target_node].index_select(0, selected)
        matrix, bias = _fit_affine_message(
            route_source,
            route_target,
            ridge=message_regularization,
        )
        affine = route_source @ matrix + bias
        left: Tensor | None = None
        right: Tensor | None = None
        output: Tensor | None = None
        prediction = affine
        if rank:
            sample = TangentPreservingQuadraticSample(
                source_modes=route_source,
                target_modes=route_target - bias,
                logical_positions=torch.arange(
                    route_source.shape[0],
                    dtype=torch.int64,
                ),
                valid_mask=torch.ones(
                    route_source.shape[0],
                    dtype=torch.bool,
                ),
            )
            quadratic = fit_tangent_preserving_quadratic_edge(
                (sample,),
                base_kernel=matrix.unsqueeze(0),
                hidden_width=rank,
                steps=steps,
                learning_rate=learning_rate,
                ridge=quadratic_regularization,
                seed=fit_seed + route,
            )
            left = quadratic.A
            right = quadratic.C
            output = quadratic.B
            prediction = affine + (
                (route_source @ left) * (route_source @ right)
            ) @ output
        affine_mse = float((affine - route_target).square().mean().item())
        polynomial_mse = float(
            (prediction - route_target).square().mean().item()
        )
        improvement = (
            0.0
            if affine_mse == 0.0
            else (affine_mse - polynomial_mse) / affine_mse
        )
        interactions.append(
            StateConditionedModalGeneratorInteraction(
                source_node=source_node,
                target_node=target_node,
                routing_group=routing_group,
                message_matrix=matrix,
                message_bias=bias,
                gate_weight=folded_weight[:, route],
                gate_bias=folded_bias[route : route + 1],
                quadratic_left=left,
                quadratic_right=right,
                quadratic_output=output,
                temperature=tau,
                top_k=top_k,
            )
        )
        metrics.append(
            StateConditionedEdgeFitMetrics(
                target_node=target_node,
                observations=int(selected.numel()),
                affine_mse=affine_mse,
                polynomial_mse=polynomial_mse,
                polynomial_improvement_fraction=improvement,
            )
        )
    return StateConditionedModalInteractionFit(
        interactions=tuple(interactions),
        router_metrics=router_metrics,
        edge_metrics=tuple(metrics),
        route_counts=tuple(int(value) for value in counts_tensor.tolist()),
        router_fold_audit_observations=int(audit_rows.shape[0]),
        router_fold_float32_max_abs_error=fold_error,
        router_fold_float32_tolerance=fold_tolerance,
    )


@dataclass(frozen=True, slots=True)
class PullbackFisherSourceChart:
    """Pullback-Fisher input chart basis projecting ambient hidden states to modal source coordinates."""

    origin: Tensor
    basis: Tensor
    eigenvalues: Tensor
    retained_pullback_energy_fraction: float

    def encode(self, hidden_states: Tensor) -> Tensor:
        states = _finite_float64(hidden_states, label="hidden_states", ndim_min=2)
        if states.shape[-1] != self.basis.shape[0]:
            raise ValueError("hidden_states width must match chart ambient width")
        return ((states - self.origin) @ self.basis).to(dtype=hidden_states.dtype).contiguous()


def pullback_fisher_source_chart(
    ambient_states: Tensor,
    target_messages: Mapping[str, Tensor],
    *,
    rank: int,
    target_fisher_weights: Mapping[str, Tensor] | None = None,
    ridge: float = 1e-6,
) -> PullbackFisherSourceChart:
    """Build a rank-``k`` source chart from the empirical pullback Fisher of target flows.

    Instead of choosing the source basis purely by source activation variance or
    codomain Fisher projection, this computes the linear pullback sensitivity
    ``J_r = (H_c^T H_c + ridge I)^{-1} H_c^T Y_r`` for each target route ``r`` and
    accumulates ``F_pb = sum_r J_r F_r J_r^T + ridge * Cov(H)`` so the retained
    ``rank`` coordinates capture the input directions that drive Fisher-weighted
    outgoing messages.
    """
    states = _finite_float64(ambient_states, label="ambient_states", ndim_min=2)
    rows = states.reshape(-1, states.shape[-1])
    n, d = rows.shape
    chart_rank = _positive_int(rank, label="rank")
    if chart_rank > d:
        raise ValueError("rank cannot exceed ambient_states width")
    reg = _positive_float(ridge, label="ridge")
    if not isinstance(target_messages, Mapping) or not target_messages:
        raise ValueError("target_messages must be a nonempty mapping")

    origin = rows.mean(dim=0)
    centered = rows - origin
    cov_h = (centered.T @ centered) / float(n)
    system = cov_h + reg * torch.eye(d, dtype=torch.float64)

    pullback_metric = torch.zeros((d, d), dtype=torch.float64)
    for name in sorted(target_messages):
        target = _finite_float64(target_messages[name], label=f"target_messages[{name!r}]", ndim_min=2)
        target_rows = target.reshape(-1, target.shape[-1])
        if target_rows.shape[0] != n:
            raise ValueError("target_messages must have the same number of observations as ambient_states")
        target_centered = target_rows - target_rows.mean(dim=0)
        cross = (centered.T @ target_centered) / float(n)
        jacobian_t = torch.linalg.solve(system, cross)  # [d, m_r]
        if target_fisher_weights is not None and name in target_fisher_weights:
            weights = _finite_float64(
                target_fisher_weights[name],
                label=f"target_fisher_weights[{name!r}]",
                ndim_min=1,
            )
            if weights.ndim == 1:
                if weights.shape[0] != target_rows.shape[1] or bool((weights < 0).any()):
                    raise ValueError("1D target_fisher_weights must be nonnegative and match target width")
                pullback_metric = pullback_metric + (jacobian_t * weights.unsqueeze(0)) @ jacobian_t.T
            elif weights.ndim == 2:
                if weights.shape != (target_rows.shape[1], target_rows.shape[1]):
                    raise ValueError("2D target_fisher_weights must be square and match target width")
                pullback_metric = pullback_metric + jacobian_t @ weights @ jacobian_t.T
            else:
                raise ValueError("target_fisher_weights must be 1D diagonal or 2D matrix")
        else:
            pullback_metric = pullback_metric + jacobian_t @ jacobian_t.T

    # Add a small state-covariance tie-breaker so null-response directions remain orthonormal and well-ordered.
    total_metric = 0.5 * (pullback_metric + pullback_metric.T) + reg * cov_h
    evals, evecs = torch.linalg.eigh(total_metric)
    basis = evecs[:, -chart_rank:].flip(1).contiguous()
    max_idx = basis.abs().argmax(dim=0)
    signs = torch.where(basis[max_idx, torch.arange(chart_rank)] < 0, -1.0, 1.0)
    basis = (basis * signs.unsqueeze(0)).contiguous()

    pb_evals = torch.linalg.eigvalsh(0.5 * (pullback_metric + pullback_metric.T)).clamp_min(0.0)
    total_pb = float(pb_evals.sum().item())
    retained_pb = (
        1.0
        if total_pb <= 1e-30
        else float(torch.trace(basis.T @ pullback_metric @ basis).clamp_min(0.0).item()) / total_pb
    )
    return PullbackFisherSourceChart(
        origin=origin.contiguous(),
        basis=basis,
        eigenvalues=evals.flip(0).contiguous(),
        retained_pullback_energy_fraction=min(1.0, max(0.0, retained_pb)),
    )


@dataclass(frozen=True, slots=True)
class SharedPullbackFisherLayerChart:
    """Shared hidden-state input chart pooled across all fragments in a layer."""

    basis: Tensor
    eigenvalues: Tensor
    retained_pullback_energy_fraction: float
    orthogonal_shrinkage: float

    @property
    def ambient_width(self) -> int:
        return int(self.basis.shape[0])

    @property
    def chart_rank(self) -> int:
        return int(self.basis.shape[1])

    def projector(self) -> Tensor:
        """Return the EWC subspace shrinkage operator Pi_chart + gamma_perp (I - Pi_chart)."""
        d = self.ambient_width
        pi = self.basis @ self.basis.T
        if self.orthogonal_shrinkage == 0.0:
            return pi.contiguous()
        eye = torch.eye(d, dtype=torch.float64)
        return (pi + self.orthogonal_shrinkage * (eye - pi)).contiguous()


def fit_shared_pullback_fisher_layer_chart(
    ambient_inputs: Tensor,
    fragment_coordinates: Mapping[str, Tensor],
    fragment_fisher_weights: Mapping[str, Tensor],
    *,
    chart_rank: int = 32,
    relative_ridge: float = 0.008,
    orthogonal_shrinkage: float = 0.25,
) -> SharedPullbackFisherLayerChart:
    """Fit a shared rank-``chart_rank`` Pullback-Fisher input chart across same-layer fragments."""
    inputs = _finite_float64(ambient_inputs, label="ambient_inputs", ndim_min=2)
    rows = inputs.reshape(-1, inputs.shape[-1])
    n, d = rows.shape
    rank = _positive_int(chart_rank, label="chart_rank")
    if rank > d:
        raise ValueError("chart_rank cannot exceed ambient input width")
    rel_ridge = _nonnegative_float(relative_ridge, label="relative_ridge")
    ortho = _nonnegative_float(orthogonal_shrinkage, label="orthogonal_shrinkage")
    if ortho > 1.0:
        raise ValueError("orthogonal_shrinkage must lie in [0, 1]")
    if not isinstance(fragment_coordinates, Mapping) or not fragment_coordinates:
        raise ValueError("fragment_coordinates must be a nonempty mapping")

    pullback_total = torch.zeros((d, d), dtype=torch.float64)
    for name in sorted(fragment_coordinates):
        coords = _finite_float64(
            fragment_coordinates[name],
            label=f"fragment_coordinates[{name!r}]",
            ndim_min=2,
        ).reshape(n, -1)
        weights = _finite_float64(
            fragment_fisher_weights[name],
            label=f"fragment_fisher_weights[{name!r}]",
            ndim_min=1,
        ).reshape(n)
        w_sum = float(weights.sum().clamp_min(1e-30).item())
        w_norm = weights * (float(n) / w_sum)
        x_mean = (rows * w_norm.unsqueeze(1)).sum(dim=0) / float(n)
        y_mean = (coords * w_norm.unsqueeze(1)).sum(dim=0) / float(n)
        x_c = rows - x_mean
        y_c = coords - y_mean
        root_w = w_norm.sqrt().unsqueeze(1)
        wx = x_c * root_w
        wy = y_c * root_w
        gram = (wx.T @ wx) / float(n)
        cross = (wx.T @ wy) / float(n)
        mean_diag = float(torch.diagonal(gram).mean().clamp_min(1e-12).item())
        eff_ridge = max(rel_ridge, 1e-6) * mean_diag
        coeff = torch.linalg.solve(
            gram + eff_ridge * torch.eye(d, dtype=torch.float64),
            cross,
        )
        # Weight each fragment's input pullback by its input covariance Gram @ coeff @ coeff^T @ Gram
        pullback_f = (gram @ coeff) @ (coeff.T @ gram)
        f_trace = float(torch.trace(pullback_f).clamp_min(1e-30).item())
        pullback_total = pullback_total + pullback_f / f_trace

    pullback_sym = 0.5 * (pullback_total + pullback_total.T)
    evals, evecs = torch.linalg.eigh(pullback_sym)
    basis = evecs[:, -rank:].flip(1).contiguous()
    max_idx = basis.abs().argmax(dim=0)
    signs = torch.where(basis[max_idx, torch.arange(rank)] < 0, -1.0, 1.0)
    basis = (basis * signs.unsqueeze(0)).contiguous()
    pos_evals = evals.clamp_min(0.0)
    total_e = float(pos_evals.sum().item())
    retained = 1.0 if total_e <= 1e-30 else float(pos_evals[-rank:].sum().item()) / total_e
    return SharedPullbackFisherLayerChart(
        basis=basis,
        eigenvalues=evals.flip(0).contiguous(),
        retained_pullback_energy_fraction=min(1.0, max(0.0, retained)),
        orthogonal_shrinkage=ortho,
    )


@dataclass(frozen=True, slots=True)
class EWCStateConditionedEdgeDiagnostics:
    """Per-route EWC consolidation and Wiener/Fisher shrinkage diagnostics."""

    target_node: str
    observations: int
    unshrunk_affine_mse: float
    ewc_shrunk_affine_mse: float
    polynomial_mse: float
    mean_mode_shrinkage_gain: float
    minimum_mode_shrinkage_gain: float
    maximum_mode_shrinkage_gain: float


@dataclass(frozen=True, slots=True)
class EWCStateConditionedModalInteractionFit:
    """Result of EWC-consolidated state-conditioned modal interaction fitting."""

    interactions: tuple[StateConditionedModalGeneratorInteraction, ...]
    router_metrics: RouterClassificationMetrics
    edge_metrics: tuple[StateConditionedEdgeFitMetrics, ...]
    ewc_diagnostics: tuple[EWCStateConditionedEdgeDiagnostics, ...]
    route_counts: tuple[int, ...]
    router_fold_audit_observations: int
    router_fold_float32_max_abs_error: float
    router_fold_float32_tolerance: float

    @property
    def parameter_count(self) -> int:
        return sum(edge.parameter_count for edge in self.interactions)

    @property
    def dense_macs_per_token(self) -> int:
        return sum(edge.macs_per_token for edge in self.interactions)


def _normalize_row_weights(
    row_weights: Tensor | None,
    count: int,
) -> Tensor:
    """Return mean-one positive token Fisher weights with robust outlier tempering."""
    if row_weights is None:
        return torch.ones(count, dtype=torch.float64)
    w = _finite_float64(row_weights, label="row_weights", ndim_min=1).reshape(-1)
    if w.shape[0] != count or bool((w < 0.0).any()):
        raise ValueError("row_weights must be nonnegative and match observation count")
    total = float(w.sum().item())
    if total <= 0.0:
        return torch.ones(count, dtype=torch.float64)
    mean_w = total / float(count)
    tempered = w.clamp_max(10.0 * mean_w) + 0.1 * mean_w
    return (tempered * (float(count) / float(tempered.sum().item()))).contiguous()


def _fit_ewc_raw_affine_message(
    route_source: Tensor,
    route_target: Tensor,
    global_source_covariance: Tensor,
    anchor_matrix: Tensor,
    anchor_bias: Tensor,
    *,
    ewc_prior_weight: float,
    route_row_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, float]:
    """Fit EWC covariance-anchored affine parameters under token Fisher weights."""
    n_r = route_source.shape[0]
    w = _normalize_row_weights(route_row_weights, n_r)
    w_col = w.unsqueeze(1)
    w_sum = float(w.sum().item())

    src_mean = (route_source * w_col).sum(dim=0) / w_sum
    tgt_mean = (route_target * w_col).sum(dim=0) / w_sum
    src_centered = route_source - src_mean
    tgt_centered = route_target - tgt_mean

    local_cov = ((src_centered * w_col).T @ src_centered) / w_sum
    cross_cov = ((src_centered * w_col).T @ tgt_centered) / w_sum

    system = local_cov + ewc_prior_weight * global_source_covariance
    rhs = cross_cov + ewc_prior_weight * (global_source_covariance @ anchor_matrix)
    raw_matrix = torch.linalg.solve(system, rhs).contiguous()
    raw_bias = (tgt_mean - src_mean @ raw_matrix).contiguous()

    intercept_trust = float(n_r) / (float(n_r) + ewc_prior_weight)
    anchored_bias = (
        intercept_trust * raw_bias
        + (1.0 - intercept_trust) * (anchor_bias + src_mean @ (anchor_matrix - raw_matrix))
    ).contiguous()

    unshrunk_pred = route_source @ raw_matrix + anchored_bias
    unshrunk_mse = float((unshrunk_pred - route_target).square().mean().item())
    return raw_matrix, anchored_bias, unshrunk_mse


def _compute_cross_block_ewc_gains(
    route_source: Tensor,
    route_target: Tensor,
    raw_prediction: Tensor,
    global_source_covariance: Tensor,
    anchor_matrix: Tensor,
    anchor_bias: Tensor,
    *,
    ewc_prior_weight: float,
    shrinkage_kappa: float,
    target_fisher_diag: Tensor | None = None,
    route_row_weights: Tensor | None = None,
    block_count: int = 8,
) -> Tensor:
    """Compute per-mode EWC Wiener shrinkage gains from Fisher-weighted cross-block OOF predictions."""
    n_r, d = route_source.shape
    m = route_target.shape[1]
    if shrinkage_kappa <= 0.0:
        return torch.ones(m, dtype=torch.float64)

    w = _normalize_row_weights(route_row_weights, n_r)
    w_col = w.unsqueeze(1)
    w_sum = float(w.sum().item())

    effective_blocks = min(block_count, n_r // max(2, d // 2))
    if effective_blocks >= 2:
        oof_pred = torch.zeros_like(route_target)
        boundaries = torch.linspace(0, n_r, steps=effective_blocks + 1, dtype=torch.int64)
        all_idx = torch.arange(n_r, dtype=torch.int64)
        for b in range(effective_blocks):
            start, end = int(boundaries[b].item()), int(boundaries[b + 1].item())
            val_mask = (all_idx >= start) & (all_idx < end)
            train_mask = ~val_mask
            if not bool(train_mask.any()) or not bool(val_mask.any()):
                continue
            mat_b, bias_b, _ = _fit_ewc_raw_affine_message(
                route_source[train_mask],
                route_target[train_mask],
                global_source_covariance,
                anchor_matrix,
                anchor_bias,
                ewc_prior_weight=ewc_prior_weight,
                route_row_weights=None if route_row_weights is None else route_row_weights[train_mask],
            )
            oof_pred[val_mask] = route_source[val_mask] @ mat_b + bias_b

        cross_dot = ((oof_pred * route_target) * w_col).sum(dim=0).div(w_sum).clamp_min(0.0)
        pred_energy = (oof_pred.square() * w_col).sum(dim=0).div(w_sum)
        error_energy = ((route_target - oof_pred).square() * w_col).sum(dim=0).div(w_sum)
        if target_fisher_diag is not None:
            f_scale = target_fisher_diag / target_fisher_diag.mean().clamp_min(1e-12)
            error_energy = error_energy * f_scale
        oof_gains = cross_dot / (pred_energy + shrinkage_kappa * error_energy + 1e-30)
        return oof_gains.clamp(0.0, 1.0).contiguous()

    signal_energy = (raw_prediction.square() * w_col).sum(dim=0).div(w_sum)
    error_energy = ((route_target - raw_prediction).square() * w_col).sum(dim=0).div(w_sum)
    if target_fisher_diag is not None:
        f_scale = target_fisher_diag / target_fisher_diag.mean().clamp_min(1e-12)
        error_energy = error_energy * f_scale
    gains = signal_energy / (signal_energy + shrinkage_kappa * error_energy + 1e-30)
    return gains.clamp(0.0, 1.0).contiguous()


def _fit_ewc_quadratic_output(
    quad_features: Tensor,
    residual_target: Tensor,
    *,
    ewc_prior_weight: float,
    shrinkage_kappa: float,
    Base_B: Tensor,
    target_fisher_diag: Tensor | None = None,
    route_row_weights: Tensor | None = None,
    block_count: int = 8,
) -> tuple[Tensor, Tensor]:
    """Refit and EWC-shrink the quadratic output factor B under token Fisher weights and 8-block OOF."""
    n_r, r = quad_features.shape
    m = residual_target.shape[1]
    w = _normalize_row_weights(route_row_weights, n_r)
    w_col = w.unsqueeze(1)
    w_sum = float(w.sum().item())

    q_gram_global = ((quad_features * w_col).T @ quad_features) / w_sum
    q_scale = max(float(torch.diagonal(q_gram_global).mean().item()), 1e-6)
    prior_cov = q_gram_global + (ewc_prior_weight * q_scale) * torch.eye(r, dtype=torch.float64)

    def solve_b(q_sub: Tensor, res_sub: Tensor, w_sub: Tensor) -> Tensor:
        ws = float(w_sub.sum().item())
        wc = w_sub.unsqueeze(1)
        g_loc = ((q_sub * wc).T @ q_sub) / ws
        c_loc = ((q_sub * wc).T @ res_sub) / ws
        sys_mat = g_loc + ewc_prior_weight * prior_cov
        rhs_mat = c_loc + ewc_prior_weight * (prior_cov @ Base_B)
        return torch.linalg.solve(sys_mat, rhs_mat).contiguous()

    raw_B = solve_b(quad_features, residual_target, w)
    if shrinkage_kappa <= 0.0:
        return raw_B, torch.ones(m, dtype=torch.float64)

    effective_blocks = min(block_count, n_r // max(2, r))
    if effective_blocks >= 2:
        oof_quad = torch.zeros_like(residual_target)
        boundaries = torch.linspace(0, n_r, steps=effective_blocks + 1, dtype=torch.int64)
        all_idx = torch.arange(n_r, dtype=torch.int64)
        for b in range(effective_blocks):
            start, end = int(boundaries[b].item()), int(boundaries[b + 1].item())
            val_mask = (all_idx >= start) & (all_idx < end)
            train_mask = ~val_mask
            if not bool(train_mask.any()) or not bool(val_mask.any()):
                continue
            b_fold = solve_b(quad_features[train_mask], residual_target[train_mask], w[train_mask])
            oof_quad[val_mask] = quad_features[val_mask] @ b_fold

        cross_dot = ((oof_quad * residual_target) * w_col).sum(dim=0).div(w_sum).clamp_min(0.0)
        pred_energy = (oof_quad.square() * w_col).sum(dim=0).div(w_sum)
        error_energy = ((residual_target - oof_quad).square() * w_col).sum(dim=0).div(w_sum)
        if target_fisher_diag is not None:
            f_scale = target_fisher_diag / target_fisher_diag.mean().clamp_min(1e-12)
            error_energy = error_energy * f_scale
        q_gains = cross_dot / (pred_energy + shrinkage_kappa * error_energy + 1e-30)
        q_gains = q_gains.clamp(0.0, 1.0).contiguous()
    else:
        raw_pred = quad_features @ raw_B
        sig_e = (raw_pred.square() * w_col).sum(dim=0).div(w_sum)
        err_e = ((residual_target - raw_pred).square() * w_col).sum(dim=0).div(w_sum)
        q_gains = (sig_e / (sig_e + shrinkage_kappa * err_e + 1e-30)).clamp(0.0, 1.0).contiguous()

    return (raw_B * q_gains.unsqueeze(0)).contiguous(), q_gains


def fit_ewc_state_conditioned_modal_interactions(
    source_states: Tensor,
    target_messages: Mapping[str, Tensor],
    route_labels: Tensor,
    *,
    source_node: str,
    routing_group: str,
    router_validation_states: Tensor | None = None,
    target_fisher_weights: Mapping[str, Tensor] | None = None,
    row_weights: Tensor | Mapping[str, Tensor] | None = None,
    temperature: float = 1.0,
    top_k: int = 1,
    router_ridge: float = 1e-3,
    message_ridge: float = 1e-4,
    ewc_prior_weight: float = 0.1,
    shrinkage_kappa: float = 0.5,
    anchor_to_global_fit: bool = False,
    orthogonalize_global_linear: bool = False,
    quadratic_rank: int = 0,
    quadratic_steps: int = 400,
    quadratic_learning_rate: float = 1e-2,
    quadratic_ridge: float = 1e-4,
    seed: int = 0,
) -> EWCStateConditionedModalInteractionFit:
    """Fit state-conditioned modal interactions with EWC covariance anchoring and per-mode Fisher shrinkage.

    Produces standard ``StateConditionedModalGeneratorInteraction`` artifacts with zero
    serving overhead: the EWC precision-shrunk gains are folded algebraically into
    ``message_matrix``, ``message_bias``, and ``quadratic_output``.
    """
    source = _finite_float64(source_states, label="source_states", ndim_min=2)
    leading_shape = source.shape[:-1]
    rows = source.reshape(-1, source.shape[-1])
    n_total, d = rows.shape
    if not isinstance(target_messages, Mapping) or len(target_messages) < 2:
        raise ValueError("target_messages must contain at least two routes")
    target_nodes = tuple(sorted(target_messages))
    targets: dict[str, Tensor] = {}
    for name in target_nodes:
        value = _finite_float64(
            target_messages[name],
            label=f"target_messages[{name!r}]",
            ndim_min=2,
        )
        if value.shape[:-1] != leading_shape:
            raise ValueError("target messages must share source leading shape")
        targets[name] = value.reshape(-1, value.shape[-1])

    if (
        not isinstance(route_labels, Tensor)
        or route_labels.dtype not in (torch.int32, torch.int64)
        or route_labels.shape != leading_shape
    ):
        raise ValueError("route_labels must be integer and match leading shape")
    labels = route_labels.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    routes = len(target_nodes)
    if int(labels.min().item()) < 0 or int(labels.max().item()) >= routes:
        raise ValueError("route_labels exceed the target route range")
    counts_tensor = torch.bincount(labels, minlength=routes)
    if bool((counts_tensor == 0).any()):
        raise ValueError("every state-conditioned route needs assigned rows")
    if type(top_k) is not int or top_k != 1:
        raise ValueError("hard-label state-conditioned fitting currently requires top_k=1")

    tau = _positive_float(temperature, label="temperature")
    route_ridge = _positive_float(router_ridge, label="router_ridge")
    msg_ridge = _positive_float(message_ridge, label="message_ridge")
    ewc_weight = _nonnegative_float(ewc_prior_weight, label="ewc_prior_weight")
    kappa = _nonnegative_float(shrinkage_kappa, label="shrinkage_kappa")
    rank = _nonnegative_int(quadratic_rank, label="quadratic_rank")
    steps = _positive_int(quadratic_steps, label="quadratic_steps")
    learning_rate = _positive_float(quadratic_learning_rate, label="quadratic_learning_rate")
    quad_ridge = _nonnegative_float(quadratic_ridge, label="quadratic_ridge")
    fit_seed = _nonnegative_int(seed, label="seed")

    router, router_metrics = fit_pointwise_causal_router(
        rows,
        labels,
        route_count=routes,
        ridge=route_ridge,
    )
    folded_weight = router.weight / router.feature_scale[:, None]
    folded_bias = router.bias - (router.feature_mean / router.feature_scale) @ router.weight

    audit_rows = rows
    if router_validation_states is not None:
        validation = _finite_float64(
            router_validation_states,
            label="router_validation_states",
            ndim_min=2,
        )
        if validation.shape[-1] != d:
            raise ValueError("router_validation_states must match the source width")
        audit_rows = torch.cat((rows, validation.reshape(-1, d)), dim=0)

    runtime_rows = audit_rows.to(dtype=torch.float32)
    reference_logits = (
        (runtime_rows - router.feature_mean.to(dtype=torch.float32))
        / router.feature_scale.to(dtype=torch.float32)
    ) @ router.weight.to(dtype=torch.float32) + router.bias.to(dtype=torch.float32)
    folded_logits = runtime_rows @ folded_weight.to(dtype=torch.float32) + folded_bias.to(dtype=torch.float32)
    if not bool(torch.isfinite(folded_logits).all()):
        raise RuntimeError("folded state-conditioned router became non-finite")
    fold_error = float((folded_logits - reference_logits).abs().max().item())
    reference_scale = max(1.0, float(reference_logits.abs().max().item()))
    fold_tolerance = 1e-6 + 1e-5 * reference_scale
    if fold_error > fold_tolerance or not torch.equal(
        folded_logits.argmax(dim=-1),
        reference_logits.argmax(dim=-1),
    ):
        raise RuntimeError(
            "folded state-conditioned router is not float32 route-stable; "
            "center or rescale the source modal coordinates"
        )

    rows_centered = rows - rows.mean(dim=0)
    global_cov = (rows_centered.T @ rows_centered) / float(n_total) + msg_ridge * torch.eye(d, dtype=torch.float64)

    interactions: list[StateConditionedModalGeneratorInteraction] = []
    metrics: list[StateConditionedEdgeFitMetrics] = []
    ewc_diagnostics: list[EWCStateConditionedEdgeDiagnostics] = []

    for route, target_node in enumerate(target_nodes):
        selected = torch.nonzero(labels == route, as_tuple=False).flatten()
        route_source = rows.index_select(0, selected)
        route_target = targets[target_node].index_select(0, selected)
        m_width = route_target.shape[1]

        if anchor_to_global_fit:
            anchor_matrix, anchor_bias = _fit_affine_message(
                rows,
                targets[target_node],
                ridge=msg_ridge,
            )
        else:
            anchor_matrix = torch.zeros((d, m_width), dtype=torch.float64)
            anchor_bias = torch.zeros(m_width, dtype=torch.float64)

        fisher_diag: Tensor | None = None
        if target_fisher_weights is not None and target_node in target_fisher_weights:
            fw = _finite_float64(
                target_fisher_weights[target_node],
                label=f"target_fisher_weights[{target_node!r}]",
                ndim_min=1,
            )
            fisher_diag = fw if fw.ndim == 1 else torch.diag(fw)

        route_w: Tensor | None = None
        full_w: Tensor | None = None
        if isinstance(row_weights, Mapping):
            if target_node in row_weights:
                full_w = _finite_float64(
                    row_weights[target_node],
                    label=f"row_weights[{target_node!r}]",
                    ndim_min=1,
                ).reshape(-1)
                if full_w.shape[0] != n_total:
                    raise ValueError("row_weights entry length must match source observations")
                route_w = full_w.index_select(0, selected)
        elif row_weights is not None:
            full_w = _finite_float64(row_weights, label="row_weights", ndim_min=1).reshape(-1)
            if full_w.shape[0] != n_total:
                raise ValueError("row_weights length must match source observations")
            route_w = full_w.index_select(0, selected)

        full_target = targets[target_node]
        if orthogonalize_global_linear:
            # Project out the global Fisher-weighted linear component on source_states
            # so routed edges fit only piecewise route specialization and quadratic curvature,
            # never undoing EWC/ridge regularization applied to the base node generators.
            gw = _normalize_row_weights(full_w, n_total).unsqueeze(1)
            gw_sum = float(gw.sum().item())
            g_src_mean = (rows * gw).sum(dim=0) / gw_sum
            g_tgt_mean = (full_target * gw).sum(dim=0) / gw_sum
            g_src_c = rows - g_src_mean
            g_tgt_c = full_target - g_tgt_mean
            g_cov = ((g_src_c * gw).T @ g_src_c) / gw_sum + msg_ridge * torch.eye(d, dtype=torch.float64)
            g_cross = ((g_src_c * gw).T @ g_tgt_c) / gw_sum
            g_mat = torch.linalg.solve(g_cov, g_cross)
            global_linear_fit = g_tgt_mean + g_src_c @ g_mat
            effective_route_target = (full_target - global_linear_fit).index_select(0, selected)
        else:
            effective_route_target = route_target

        raw_matrix, raw_bias, unshrunk_mse = _fit_ewc_raw_affine_message(
            route_source,
            effective_route_target,
            global_cov,
            anchor_matrix,
            anchor_bias,
            ewc_prior_weight=ewc_weight,
            route_row_weights=route_w,
        )
        raw_affine = route_source @ raw_matrix + raw_bias
        affine_gains = _compute_cross_block_ewc_gains(
            route_source,
            effective_route_target,
            raw_affine,
            global_cov,
            anchor_matrix,
            anchor_bias,
            ewc_prior_weight=ewc_weight,
            shrinkage_kappa=kappa,
            target_fisher_diag=fisher_diag,
            route_row_weights=route_w,
        )
        matrix = (raw_matrix * affine_gains.unsqueeze(0)).contiguous()
        bias = (raw_bias * affine_gains).contiguous()
        affine = route_source @ matrix + bias

        left: Tensor | None = None
        right: Tensor | None = None
        output: Tensor | None = None
        gains = affine_gains
        prediction = affine
        if rank:
            sample = TangentPreservingQuadraticSample(
                source_modes=route_source,
                target_modes=effective_route_target - raw_bias,
                logical_positions=torch.arange(route_source.shape[0], dtype=torch.int64),
                valid_mask=torch.ones(route_source.shape[0], dtype=torch.bool),
            )
            quadratic = fit_tangent_preserving_quadratic_edge(
                (sample,),
                base_kernel=raw_matrix.unsqueeze(0),
                hidden_width=rank,
                steps=steps,
                learning_rate=learning_rate,
                ridge=quad_ridge,
                seed=fit_seed + route,
            )
            left = quadratic.A
            right = quadratic.C
            quad_features = (route_source @ left) * (route_source @ right)
            residual_after_affine = effective_route_target - affine
            output, quad_gains = _fit_ewc_quadratic_output(
                quad_features,
                residual_after_affine,
                ewc_prior_weight=ewc_weight,
                shrinkage_kappa=kappa,
                Base_B=quadratic.B,
                target_fisher_diag=fisher_diag,
                route_row_weights=route_w,
            )
            gains = 0.5 * (affine_gains + quad_gains)
            prediction = affine + quad_features @ output

        affine_mse = float((affine - route_target).square().mean().item())
        polynomial_mse = float((prediction - route_target).square().mean().item())
        improvement = 0.0 if affine_mse == 0.0 else (affine_mse - polynomial_mse) / affine_mse

        interactions.append(
            StateConditionedModalGeneratorInteraction(
                source_node=source_node,
                target_node=target_node,
                routing_group=routing_group,
                message_matrix=matrix,
                message_bias=bias,
                gate_weight=folded_weight[:, route],
                gate_bias=folded_bias[route : route + 1],
                quadratic_left=left,
                quadratic_right=right,
                quadratic_output=output,
                temperature=tau,
                top_k=top_k,
            )
        )
        metrics.append(
            StateConditionedEdgeFitMetrics(
                target_node=target_node,
                observations=int(selected.numel()),
                affine_mse=affine_mse,
                polynomial_mse=polynomial_mse,
                polynomial_improvement_fraction=improvement,
            )
        )
        ewc_diagnostics.append(
            EWCStateConditionedEdgeDiagnostics(
                target_node=target_node,
                observations=int(selected.numel()),
                unshrunk_affine_mse=unshrunk_mse,
                ewc_shrunk_affine_mse=affine_mse,
                polynomial_mse=polynomial_mse,
                mean_mode_shrinkage_gain=float(gains.mean().item()),
                minimum_mode_shrinkage_gain=float(gains.min().item()),
                maximum_mode_shrinkage_gain=float(gains.max().item()),
            )
        )

    return EWCStateConditionedModalInteractionFit(
        interactions=tuple(interactions),
        router_metrics=router_metrics,
        edge_metrics=tuple(metrics),
        ewc_diagnostics=tuple(ewc_diagnostics),
        route_counts=tuple(int(value) for value in counts_tensor.tolist()),
        router_fold_audit_observations=int(audit_rows.shape[0]),
        router_fold_float32_max_abs_error=fold_error,
        router_fold_float32_tolerance=fold_tolerance,
    )

