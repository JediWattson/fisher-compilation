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
    "StateConditionedEdgeFitMetrics",
    "StateConditionedModalInteractionFit",
    "TeacherFlowRouting",
    "fit_state_conditioned_modal_interactions",
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
