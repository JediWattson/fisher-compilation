"""Read-only shape-and-flow evaluation for frozen modal interactions.

This module evaluates a state-conditioned routing group without loading or
calling its source model.  The caller supplies frozen source-modal states,
native correction coordinates for every candidate target, and the target
decoders that place those corrections in one shared residual boundary.

For target ``j``, the teacher-only candidate displacement is

``d_j = correction_coordinates[j] @ decoder_bases[j]``.

The correctable teacher flow is ``sum_j d_j``.  Oracle route labels are
derived in canonical sorted-target order with :func:`teacher_flow_routing`.
They are used only for assessment: routed predictions read the source state
and copied interaction coefficients, and this module never fits or mutates an
interaction.

Three source-free conditions are reported:

``routed_graph``
    The frozen top-1 router selects one polynomial proposal per row.
``dense_all_target``
    Every frozen proposal is decoded and summed.
``constant_oracle_majority``
    One route, chosen as the assessment oracle's majority class, is used for
    every row.  This is an explicitly diagnostic control, not a deployable
    selection rule.

All returned dataclasses contain only JSON-safe scalar, string, tuple, and
nested-dataclass values.  No activation, teacher, route-logit, or correction
tensor is retained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .modal_generator_graph import (
    StateConditionedModalGeneratorInteraction,
)
from .state_conditioned_modal_fitting import teacher_flow_routing


__all__ = [
    "StateConditionedFamilyFlowEvaluation",
    "StateConditionedFlowMetrics",
    "StateConditionedModalFlowEvaluation",
    "StateConditionedRouteMetrics",
    "evaluate_state_conditioned_modal_flow",
]


def _finite_rows(value: Tensor, *, label: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.ndim != 2
        or value.shape[0] <= 0
        or value.shape[1] <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{label} must be a finite nonempty rank-2 Tensor")
    return (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )


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


def _row_weights(
    value: Tensor | None,
    *,
    observations: int,
) -> Tensor:
    if value is None:
        return torch.ones(observations, dtype=torch.float64)
    if (
        not isinstance(value, Tensor)
        or not value.is_floating_point()
        or value.shape != (observations,)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            "row_weights must be a finite floating Tensor with one value "
            "per observation"
        )
    result = (
        value.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .clone()
    )
    if bool((result < 0.0).any()) or float(result.sum().item()) <= 0.0:
        raise ValueError(
            "row_weights must be nonnegative with positive total mass"
        )
    return result


def _nearest_rank(values: Tensor, fraction: float) -> float:
    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("percentile values must be a nonempty vector")
    ordered = values.sort().values
    index = max(
        0,
        min(
            ordered.numel() - 1,
            int(math.ceil(fraction * ordered.numel())) - 1,
        ),
    )
    return float(ordered[index].item())


def _lower_nearest_rank(values: Tensor, fraction: float) -> float:
    return -_nearest_rank(-values, 1.0 - fraction)


@dataclass(frozen=True, slots=True)
class StateConditionedFlowMetrics:
    """Finite flow-reconstruction metrics for one frozen condition."""

    observations: int
    residual_width: int
    weight_sum: float
    teacher_squared_l2: float
    residual_squared_l2: float
    nrmse: float
    weighted_teacher_squared_l2: float
    weighted_residual_squared_l2: float
    weighted_nrmse: float
    aggregate_cosine: float
    weighted_aggregate_cosine: float
    sse_improvement_over_edgeless: float
    weighted_sse_improvement_over_edgeless: float
    p90_relative_error: float
    p10_cosine: float
    teacher_signal_defined: bool
    weighted_teacher_signal_defined: bool

    def metadata(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "residual_width": self.residual_width,
            "weight_sum": self.weight_sum,
            "teacher_squared_l2": self.teacher_squared_l2,
            "residual_squared_l2": self.residual_squared_l2,
            "nrmse": self.nrmse,
            "weighted_teacher_squared_l2": (
                self.weighted_teacher_squared_l2
            ),
            "weighted_residual_squared_l2": (
                self.weighted_residual_squared_l2
            ),
            "weighted_nrmse": self.weighted_nrmse,
            "aggregate_cosine": self.aggregate_cosine,
            "weighted_aggregate_cosine": self.weighted_aggregate_cosine,
            "sse_improvement_over_edgeless": (
                self.sse_improvement_over_edgeless
            ),
            "weighted_sse_improvement_over_edgeless": (
                self.weighted_sse_improvement_over_edgeless
            ),
            "p90_relative_error": self.p90_relative_error,
            "p10_cosine": self.p10_cosine,
            "teacher_signal_defined": self.teacher_signal_defined,
            "weighted_teacher_signal_defined": (
                self.weighted_teacher_signal_defined
            ),
        }


@dataclass(frozen=True, slots=True)
class StateConditionedRouteMetrics:
    """Oracle-label agreement and route-occupancy diagnostics."""

    observations: int
    accuracy: float
    weighted_accuracy: float
    macro_recall: float
    majority_route_ordinal: int
    majority_route_target: str
    majority_baseline_accuracy: float
    weighted_majority_baseline_accuracy: float
    oracle_route_counts: tuple[int, ...]
    predicted_route_counts: tuple[int, ...]
    oracle_route_weight_mass: tuple[float, ...]
    predicted_route_weight_mass: tuple[float, ...]
    oracle_entropy_nats: float
    predicted_entropy_nats: float
    oracle_normalized_entropy: float
    predicted_normalized_entropy: float
    oracle_max_share: float
    predicted_max_share: float
    confusion_matrix: tuple[tuple[int, ...], ...]

    def metadata(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "accuracy": self.accuracy,
            "weighted_accuracy": self.weighted_accuracy,
            "macro_recall": self.macro_recall,
            "majority_route_ordinal": self.majority_route_ordinal,
            "majority_route_target": self.majority_route_target,
            "majority_baseline_accuracy": self.majority_baseline_accuracy,
            "weighted_majority_baseline_accuracy": (
                self.weighted_majority_baseline_accuracy
            ),
            "oracle_route_counts": self.oracle_route_counts,
            "predicted_route_counts": self.predicted_route_counts,
            "oracle_route_weight_mass": self.oracle_route_weight_mass,
            "predicted_route_weight_mass": self.predicted_route_weight_mass,
            "oracle_entropy_nats": self.oracle_entropy_nats,
            "predicted_entropy_nats": self.predicted_entropy_nats,
            "oracle_normalized_entropy": self.oracle_normalized_entropy,
            "predicted_normalized_entropy": self.predicted_normalized_entropy,
            "oracle_max_share": self.oracle_max_share,
            "predicted_max_share": self.predicted_max_share,
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass(frozen=True, slots=True)
class StateConditionedFamilyFlowEvaluation:
    """One declared family's read-only route and flow measurements."""

    family_id: str
    observations: int
    routes: StateConditionedRouteMetrics
    routed_graph: StateConditionedFlowMetrics
    dense_all_target: StateConditionedFlowMetrics
    constant_oracle_majority: StateConditionedFlowMetrics

    def metadata(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "observations": self.observations,
            "routes": self.routes.metadata(),
            "conditions": {
                "routed_graph": self.routed_graph.metadata(),
                "dense_all_target": self.dense_all_target.metadata(),
                "constant_oracle_majority": {
                    **self.constant_oracle_majority.metadata(),
                    "route_source": (
                        "family_assessment_oracle_majority_diagnostic_only"
                    ),
                    "route_ordinal": self.routes.majority_route_ordinal,
                    "route_target": self.routes.majority_route_target,
                },
            },
        }


@dataclass(frozen=True, slots=True)
class StateConditionedModalFlowEvaluation:
    """JSON-safe assessment receipt for one frozen routing group."""

    source_node: str
    routing_group: str
    target_nodes: tuple[str, ...]
    interaction_artifact_sha256s: tuple[str, ...]
    observations: int
    residual_width: int
    routes: StateConditionedRouteMetrics
    routed_graph: StateConditionedFlowMetrics
    dense_all_target: StateConditionedFlowMetrics
    constant_oracle_majority: StateConditionedFlowMetrics
    mean_oracle_alignment: float
    minimum_oracle_alignment: float
    mean_oracle_relative_residual: float
    maximum_oracle_relative_residual: float
    families: tuple[StateConditionedFamilyFlowEvaluation, ...]

    def metadata(self) -> dict[str, object]:
        return {
            "evaluation_kind": (
                "fisher_graph.state_conditioned_modal_flow_evaluation"
            ),
            "source_free": True,
            "assessment_read_only": True,
            "coefficients_fitted": False,
            "teacher_used_for_scoring_only": True,
            "routed_graph_uses_source_state_only": True,
            "canonical_target_order": self.target_nodes,
            "source_node": self.source_node,
            "routing_group": self.routing_group,
            "interaction_artifact_sha256s": (
                self.interaction_artifact_sha256s
            ),
            "observations": self.observations,
            "residual_width": self.residual_width,
            "teacher_route_summary": {
                "mean_best_alignment": self.mean_oracle_alignment,
                "minimum_best_alignment": self.minimum_oracle_alignment,
                "mean_selected_relative_residual": (
                    self.mean_oracle_relative_residual
                ),
                "maximum_selected_relative_residual": (
                    self.maximum_oracle_relative_residual
                ),
            },
            "routes": self.routes.metadata(),
            "conditions": {
                "routed_graph": self.routed_graph.metadata(),
                "dense_all_target": self.dense_all_target.metadata(),
                "constant_oracle_majority": {
                    **self.constant_oracle_majority.metadata(),
                    "route_source": (
                        "assessment_oracle_majority_diagnostic_only"
                    ),
                },
            },
            "families": tuple(family.metadata() for family in self.families),
        }


def _flow_metrics(
    teacher: Tensor,
    prediction: Tensor,
    weights: Tensor,
    *,
    epsilon: float,
) -> StateConditionedFlowMetrics:
    residual = prediction - teacher
    teacher_row_energy = teacher.square().sum(dim=-1)
    prediction_row_energy = prediction.square().sum(dim=-1)
    residual_row_energy = residual.square().sum(dim=-1)
    teacher_energy = float(teacher_row_energy.sum().item())
    prediction_energy = float(prediction_row_energy.sum().item())
    residual_energy = float(residual_row_energy.sum().item())
    weighted_teacher_energy = float(
        (weights * teacher_row_energy).sum().item()
    )
    weighted_prediction_energy = float(
        (weights * prediction_row_energy).sum().item()
    )
    weighted_residual_energy = float(
        (weights * residual_row_energy).sum().item()
    )
    dot_rows = (teacher * prediction).sum(dim=-1)
    dot = float(dot_rows.sum().item())
    weighted_dot = float((weights * dot_rows).sum().item())
    nrmse = math.sqrt(residual_energy / max(teacher_energy, epsilon))
    weighted_nrmse = math.sqrt(
        weighted_residual_energy / max(weighted_teacher_energy, epsilon)
    )
    aggregate_cosine = dot / max(
        math.sqrt(teacher_energy * prediction_energy),
        epsilon,
    )
    weighted_cosine = weighted_dot / max(
        math.sqrt(
            weighted_teacher_energy * weighted_prediction_energy
        ),
        epsilon,
    )
    teacher_norm = teacher_row_energy.sqrt()
    prediction_norm = prediction_row_energy.sqrt()
    relative_error = residual_row_energy.sqrt() / teacher_norm.clamp_min(
        epsilon
    )
    row_cosine = dot_rows / (
        teacher_norm * prediction_norm
    ).clamp_min(epsilon)
    values = (
        teacher_energy,
        residual_energy,
        nrmse,
        weighted_teacher_energy,
        weighted_residual_energy,
        weighted_nrmse,
        aggregate_cosine,
        weighted_cosine,
        1.0 - residual_energy / max(teacher_energy, epsilon),
        1.0
        - weighted_residual_energy
        / max(weighted_teacher_energy, epsilon),
        _nearest_rank(relative_error, 0.90),
        _lower_nearest_rank(row_cosine, 0.10),
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("flow metrics became non-finite")
    return StateConditionedFlowMetrics(
        observations=int(teacher.shape[0]),
        residual_width=int(teacher.shape[1]),
        weight_sum=float(weights.sum().item()),
        teacher_squared_l2=teacher_energy,
        residual_squared_l2=residual_energy,
        nrmse=nrmse,
        weighted_teacher_squared_l2=weighted_teacher_energy,
        weighted_residual_squared_l2=weighted_residual_energy,
        weighted_nrmse=weighted_nrmse,
        aggregate_cosine=aggregate_cosine,
        weighted_aggregate_cosine=weighted_cosine,
        sse_improvement_over_edgeless=values[8],
        weighted_sse_improvement_over_edgeless=values[9],
        p90_relative_error=values[10],
        p10_cosine=values[11],
        teacher_signal_defined=teacher_energy > epsilon,
        weighted_teacher_signal_defined=weighted_teacher_energy > epsilon,
    )


def _entropy(counts: Tensor) -> tuple[float, float, float]:
    total = float(counts.sum().item())
    probabilities = counts.to(dtype=torch.float64) / total
    positive = probabilities > 0.0
    entropy = float(
        -(probabilities[positive] * probabilities[positive].log()).sum().item()
    )
    normalized = (
        0.0
        if counts.numel() <= 1
        else entropy / math.log(int(counts.numel()))
    )
    return entropy, normalized, float(probabilities.max().item())


def _route_metrics(
    oracle: Tensor,
    predicted: Tensor,
    weights: Tensor,
    *,
    target_nodes: tuple[str, ...],
) -> StateConditionedRouteMetrics:
    routes = len(target_nodes)
    oracle_counts = torch.bincount(oracle, minlength=routes)
    predicted_counts = torch.bincount(predicted, minlength=routes)
    confusion = torch.zeros((routes, routes), dtype=torch.int64)
    for target, observed in zip(oracle, predicted, strict=True):
        confusion[int(target.item()), int(observed.item())] += 1
    observed_routes = oracle_counts > 0
    recalls = (
        confusion.diagonal()[observed_routes].to(dtype=torch.float64)
        / oracle_counts[observed_routes].to(dtype=torch.float64)
    )
    oracle_mass = torch.zeros(routes, dtype=torch.float64)
    predicted_mass = torch.zeros(routes, dtype=torch.float64)
    oracle_mass.scatter_add_(0, oracle, weights)
    predicted_mass.scatter_add_(0, predicted, weights)
    majority = int(oracle_counts.argmax().item())
    total_weight = float(weights.sum().item())
    oracle_entropy, oracle_normalized, oracle_max = _entropy(oracle_counts)
    predicted_entropy, predicted_normalized, predicted_max = _entropy(
        predicted_counts
    )
    return StateConditionedRouteMetrics(
        observations=int(oracle.numel()),
        accuracy=float((oracle == predicted).to(torch.float64).mean().item()),
        weighted_accuracy=float(
            (weights * (oracle == predicted).to(torch.float64)).sum().item()
            / total_weight
        ),
        macro_recall=float(recalls.mean().item()),
        majority_route_ordinal=majority,
        majority_route_target=target_nodes[majority],
        majority_baseline_accuracy=float(
            oracle_counts[majority].item() / oracle.numel()
        ),
        weighted_majority_baseline_accuracy=float(
            oracle_mass.max().item() / total_weight
        ),
        oracle_route_counts=tuple(int(value) for value in oracle_counts.tolist()),
        predicted_route_counts=tuple(
            int(value) for value in predicted_counts.tolist()
        ),
        oracle_route_weight_mass=tuple(
            float(value) for value in oracle_mass.tolist()
        ),
        predicted_route_weight_mass=tuple(
            float(value) for value in predicted_mass.tolist()
        ),
        oracle_entropy_nats=oracle_entropy,
        predicted_entropy_nats=predicted_entropy,
        oracle_normalized_entropy=oracle_normalized,
        predicted_normalized_entropy=predicted_normalized,
        oracle_max_share=oracle_max,
        predicted_max_share=predicted_max,
        confusion_matrix=tuple(
            tuple(int(value) for value in row)
            for row in confusion.tolist()
        ),
    )


def _family_vector(
    family_ids: Sequence[str] | None,
    *,
    observations: int,
) -> tuple[str, ...] | None:
    if family_ids is None:
        return None
    if isinstance(family_ids, (str, bytes)) or not isinstance(
        family_ids,
        Sequence,
    ):
        raise TypeError("family_ids must be a sequence or None")
    result = tuple(family_ids)
    if (
        len(result) != observations
        or any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in result
        )
    ):
        raise ValueError(
            "family_ids must contain one nonempty stripped string per row"
        )
    return result


def evaluate_state_conditioned_modal_flow(
    source_states: Tensor,
    correction_coordinates: Mapping[str, Tensor],
    decoder_bases: Mapping[str, Tensor],
    interactions: Sequence[StateConditionedModalGeneratorInteraction],
    *,
    family_ids: Sequence[str] | None = None,
    row_weights: Tensor | None = None,
    alignment_weight: float = 1.0,
    residual_weight: float = 1.0,
    oracle_temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> StateConditionedModalFlowEvaluation:
    """Evaluate frozen source-only routing against same-boundary teacher flow.

    ``correction_coordinates[target]`` has shape ``[rows, target_width]`` and
    ``decoder_bases[target]`` has shape
    ``[target_width, shared_residual_width]``.  Mapping insertion order is
    ignored: target order is always the sorted interaction target order.

    ``row_weights`` is optional nonnegative per-row importance, typically the
    sum of target Fisher weights.  It affects weighted metrics only; oracle
    labels, ordinary metrics, percentiles, and route counts remain exact
    unweighted observations.
    """

    source = _finite_rows(source_states, label="source_states")
    if isinstance(interactions, (str, bytes)) or not isinstance(
        interactions,
        Sequence,
    ):
        raise TypeError("interactions must be a sequence")
    supplied = tuple(interactions)
    if len(supplied) < 2 or any(
        not isinstance(value, StateConditionedModalGeneratorInteraction)
        for value in supplied
    ):
        raise ValueError(
            "interactions must contain at least two state-conditioned edges"
        )
    for edge in supplied:
        edge.validate_integrity()
    edges = tuple(sorted(supplied, key=lambda edge: edge.target_node))
    target_nodes = tuple(edge.target_node for edge in edges)
    if len(target_nodes) != len(set(target_nodes)):
        raise ValueError("interaction targets must be unique")
    first = edges[0]
    if (
        source.shape[1] != first.source_width
        or any(
            edge.source_node != first.source_node
            or edge.routing_group != first.routing_group
            or edge.source_width != first.source_width
            or edge.temperature != first.temperature
            or edge.top_k != 1
            for edge in edges
        )
    ):
        raise ValueError(
            "interactions must form one compatible top-1 routing group"
        )
    if not isinstance(correction_coordinates, Mapping) or set(
        correction_coordinates
    ) != set(target_nodes):
        raise ValueError(
            "correction_coordinates must exactly cover interaction targets"
        )
    if not isinstance(decoder_bases, Mapping) or set(decoder_bases) != set(
        target_nodes
    ):
        raise ValueError("decoder_bases must exactly cover interaction targets")

    corrections: list[Tensor] = []
    decoders: list[Tensor] = []
    residual_width: int | None = None
    for edge in edges:
        correction = _finite_rows(
            correction_coordinates[edge.target_node],
            label=f"correction_coordinates[{edge.target_node!r}]",
        )
        decoder = _finite_rows(
            decoder_bases[edge.target_node],
            label=f"decoder_bases[{edge.target_node!r}]",
        )
        if correction.shape != (source.shape[0], edge.target_width):
            raise ValueError(
                "correction coordinate rows or target width drifted"
            )
        if decoder.shape[0] != edge.target_width:
            raise ValueError("decoder input width differs from its target edge")
        if residual_width is None:
            residual_width = int(decoder.shape[1])
        elif decoder.shape[1] != residual_width:
            raise ValueError("decoder bases do not share one residual boundary")
        corrections.append(correction)
        decoders.append(decoder)
    assert residual_width is not None

    weights = _row_weights(row_weights, observations=source.shape[0])
    families = _family_vector(family_ids, observations=source.shape[0])
    alignment = _nonnegative_float(
        alignment_weight,
        label="alignment_weight",
    )
    residual = _nonnegative_float(
        residual_weight,
        label="residual_weight",
    )
    temperature = _positive_float(
        oracle_temperature,
        label="oracle_temperature",
    )
    eps = _positive_float(epsilon, label="epsilon")

    with torch.inference_mode():
        candidate_displacements = torch.stack(
            tuple(
                correction @ decoder
                for correction, decoder in zip(
                    corrections,
                    decoders,
                    strict=True,
                )
            ),
            dim=1,
        )
        teacher = candidate_displacements.sum(dim=1)
        oracle_routing = teacher_flow_routing(
            teacher,
            candidate_displacements,
            alignment_weight=alignment,
            residual_weight=residual,
            temperature=temperature,
            epsilon=eps,
        )
        oracle_labels = oracle_routing.route_labels.reshape(-1)

        logits = torch.stack(
            tuple(edge.routing_logit(source) for edge in edges),
            dim=-1,
        )
        predicted_labels = logits.argmax(dim=-1)
        proposals = torch.stack(
            tuple(
                edge.proposed_message(source) @ decoder
                for edge, decoder in zip(edges, decoders, strict=True)
            ),
            dim=1,
        )
        rows = torch.arange(source.shape[0], dtype=torch.int64)
        routed = proposals[rows, predicted_labels]
        dense = proposals.sum(dim=1)
        majority = int(
            torch.bincount(
                oracle_labels,
                minlength=len(target_nodes),
            ).argmax().item()
        )
        constant = proposals[:, majority, :]

    if float(teacher.square().sum().item()) <= eps:
        raise ValueError(
            "same-boundary teacher flow has no aggregate signal to evaluate"
        )
    route_summary = _route_metrics(
        oracle_labels,
        predicted_labels,
        weights,
        target_nodes=target_nodes,
    )
    routed_metrics = _flow_metrics(teacher, routed, weights, epsilon=eps)
    dense_metrics = _flow_metrics(teacher, dense, weights, epsilon=eps)
    constant_metrics = _flow_metrics(
        teacher,
        constant,
        weights,
        epsilon=eps,
    )

    family_results: list[StateConditionedFamilyFlowEvaluation] = []
    if families is not None:
        for family_id in sorted(set(families)):
            selected = torch.tensor(
                tuple(value == family_id for value in families),
                dtype=torch.bool,
            )
            if float(weights[selected].sum().item()) <= 0.0:
                raise ValueError(
                    f"family {family_id!r} has no positive row-weight mass"
                )
            family_routes = _route_metrics(
                oracle_labels[selected],
                predicted_labels[selected],
                weights[selected],
                target_nodes=target_nodes,
            )
            family_constant = proposals[
                selected,
                family_routes.majority_route_ordinal,
                :,
            ]
            family_results.append(
                StateConditionedFamilyFlowEvaluation(
                    family_id=family_id,
                    observations=int(selected.sum().item()),
                    routes=family_routes,
                    routed_graph=_flow_metrics(
                        teacher[selected],
                        routed[selected],
                        weights[selected],
                        epsilon=eps,
                    ),
                    dense_all_target=_flow_metrics(
                        teacher[selected],
                        dense[selected],
                        weights[selected],
                        epsilon=eps,
                    ),
                    constant_oracle_majority=_flow_metrics(
                        teacher[selected],
                        family_constant,
                        weights[selected],
                        epsilon=eps,
                    ),
                )
            )

    return StateConditionedModalFlowEvaluation(
        source_node=first.source_node,
        routing_group=first.routing_group,
        target_nodes=target_nodes,
        interaction_artifact_sha256s=tuple(
            edge.artifact_sha256 for edge in edges
        ),
        observations=int(source.shape[0]),
        residual_width=residual_width,
        routes=route_summary,
        routed_graph=routed_metrics,
        dense_all_target=dense_metrics,
        constant_oracle_majority=constant_metrics,
        mean_oracle_alignment=oracle_routing.mean_best_alignment,
        minimum_oracle_alignment=oracle_routing.minimum_best_alignment,
        mean_oracle_relative_residual=(
            oracle_routing.mean_selected_relative_residual
        ),
        maximum_oracle_relative_residual=(
            oracle_routing.maximum_selected_relative_residual
        ),
        families=tuple(family_results),
    )
