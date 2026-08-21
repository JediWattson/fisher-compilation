"""Causal confirmation for dense edges beside an exact Gemma execution.

This module never installs an edge in the model.  It perturbs one source chart
site only in an isolated exact shadow execution, measures the residual target
flow left after the frozen target-local baseline, and compares that response to
the dense edge's linear prediction.  The unperturbed execution remains the
authoritative output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import torch
from torch import Tensor

from .adapters import SequenceContext
from .gemma3_joint_cross_block_chart_executor import (
    Gemma3JointCrossBlockChartExecutor,
    JOINT_EXACT_CONTROL_ARM_ID,
)
from .gemma3_unified_event_chart_sidecar import (
    FittedUnifiedEventDenseEdge,
    FittedUnifiedEventLocalBaseline,
)


_DIRECTION_DOMAIN = b"fisher_graph:exact-edge-sidecar-direction:v1\0"


def _cosine(left: Tensor, right: Tensor) -> float:
    left_flat = left.double().reshape(-1)
    right_flat = right.double().reshape(-1)
    denominator = float(left_flat.norm() * right_flat.norm())
    if denominator <= 0.0:
        return 1.0 if bool(torch.equal(left_flat, right_flat)) else 0.0
    return float(torch.dot(left_flat, right_flat) / denominator)


def _relative_error(predicted: Tensor, observed: Tensor) -> float:
    denominator = max(float(observed.double().norm()), 1.0e-30)
    return float((predicted.double() - observed.double()).norm()) / denominator


def _require_epsilon_ladder(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("epsilon_ladder must be a sequence")
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not math.isfinite(value) or value <= 0.0 for value in result)
        or tuple(sorted(result)) != result
        or len(set(result)) != len(result)
    ):
        raise ValueError("epsilon_ladder must be strictly increasing and positive")
    return result


@dataclass(frozen=True, slots=True)
class ExactEdgeDirectionDiagnostic:
    """One deterministic direction's exact and finite-response comparison."""

    direction_index: int
    direction_sha256: str
    source_rms: float
    exact_jvp_cosine: float
    exact_jvp_relative_error: float
    finite_cosines: tuple[float, ...]
    finite_relative_errors: tuple[float, ...]
    adjacent_finite_derivative_disagreements: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.direction_index) is not int or self.direction_index < 0:
            raise ValueError("direction_index must be nonnegative")
        if (
            not isinstance(self.direction_sha256, str)
            or len(self.direction_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.direction_sha256
            )
        ):
            raise ValueError("direction_sha256 must be lowercase SHA-256")
        if not math.isfinite(self.source_rms) or self.source_rms <= 0.0:
            raise ValueError("source_rms must be finite and positive")
        cosines = (self.exact_jvp_cosine, *self.finite_cosines)
        errors = (
            self.exact_jvp_relative_error,
            *self.finite_relative_errors,
            *self.adjacent_finite_derivative_disagreements,
        )
        if any(
            not math.isfinite(value) or value < -1.0 or value > 1.0
            for value in cosines
        ):
            raise ValueError("direction cosines must be finite and in [-1, 1]")
        if any(not math.isfinite(value) or value < 0.0 for value in errors):
            raise ValueError("direction errors must be finite and nonnegative")
        if (
            not self.finite_cosines
            or len(self.finite_cosines) != len(self.finite_relative_errors)
            or len(self.adjacent_finite_derivative_disagreements)
            != len(self.finite_cosines) - 1
        ):
            raise ValueError("finite direction diagnostic inventory drifted")

    def metadata(self) -> dict[str, object]:
        return {
            "direction_index": self.direction_index,
            "direction_sha256": self.direction_sha256,
            "source_rms": self.source_rms,
            "exact_jvp_cosine": self.exact_jvp_cosine,
            "exact_jvp_relative_error": self.exact_jvp_relative_error,
            "finite_cosines": list(self.finite_cosines),
            "finite_relative_errors": list(self.finite_relative_errors),
            "adjacent_finite_derivative_disagreements": list(
                self.adjacent_finite_derivative_disagreements
            ),
        }


@dataclass(frozen=True, slots=True)
class ExactEdgeCausalEvaluation:
    """Frozen C-role JVP and symmetric finite-intervention receipt."""

    source_site_id: str
    target_site_id: str
    direction_count: int
    epsilon_ladder: tuple[float, ...]
    diagnostics: tuple[ExactEdgeDirectionDiagnostic, ...]
    baseline_vs_zero_shadow_max_absolute_error: float
    minimum_jvp_cosine: float
    maximum_jvp_relative_error: float
    minimum_finite_cosine: float
    maximum_finite_relative_error: float
    maximum_adjacent_finite_derivative_disagreement: float
    exact_forward_count: int
    exact_jvp_count: int
    backward_or_jvp_count: int
    finite_intervention_forward_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_site_id, str)
            or not self.source_site_id
            or not isinstance(self.target_site_id, str)
            or not self.target_site_id.endswith(".output")
        ):
            raise ValueError("causal evaluation endpoint ids are invalid")
        if type(self.direction_count) is not int or self.direction_count <= 0:
            raise ValueError("direction_count must be positive")
        epsilons = _require_epsilon_ladder(self.epsilon_ladder)
        if epsilons != self.epsilon_ladder:
            raise ValueError("epsilon_ladder must already be canonical")
        if (
            len(self.diagnostics) != self.direction_count
            or tuple(item.direction_index for item in self.diagnostics)
            != tuple(range(self.direction_count))
            or len({item.direction_sha256 for item in self.diagnostics})
            != self.direction_count
            or any(
                len(item.finite_cosines) != len(epsilons)
                for item in self.diagnostics
            )
        ):
            raise ValueError("causal direction inventory drifted")
        aggregates = {
            "minimum_jvp_cosine": min(
                item.exact_jvp_cosine for item in self.diagnostics
            ),
            "maximum_jvp_relative_error": max(
                item.exact_jvp_relative_error for item in self.diagnostics
            ),
            "minimum_finite_cosine": min(
                value
                for item in self.diagnostics
                for value in item.finite_cosines
            ),
            "maximum_finite_relative_error": max(
                value
                for item in self.diagnostics
                for value in item.finite_relative_errors
            ),
            "maximum_adjacent_finite_derivative_disagreement": max(
                (
                    value
                    for item in self.diagnostics
                    for value in item.adjacent_finite_derivative_disagreements
                ),
                default=0.0,
            ),
        }
        if any(
            getattr(self, name) != value for name, value in aggregates.items()
        ):
            raise ValueError("causal aggregate metrics do not recompute")
        if (
            not math.isfinite(self.baseline_vs_zero_shadow_max_absolute_error)
            or self.baseline_vs_zero_shadow_max_absolute_error < 0.0
        ):
            raise ValueError("zero-shadow error must be finite and nonnegative")
        expected_finite = 2 * self.direction_count * len(epsilons)
        expected_total = 2 + self.direction_count + expected_finite
        if (
            self.exact_jvp_count != self.direction_count
            or self.backward_or_jvp_count != self.direction_count
            or self.finite_intervention_forward_count != expected_finite
            or self.exact_forward_count != expected_total
        ):
            raise ValueError("causal execution-count ledger does not close")

    @property
    def absolute_jvp_thresholds_passed(self) -> bool:
        return (
            self.minimum_jvp_cosine >= 0.90
            and self.maximum_jvp_relative_error <= 0.25
        )

    @property
    def absolute_finite_thresholds_passed(self) -> bool:
        return (
            self.minimum_finite_cosine >= 0.80
            and self.maximum_finite_relative_error <= 0.35
            and self.maximum_adjacent_finite_derivative_disagreement <= 0.10
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source_site_id": self.source_site_id,
            "target_site_id": self.target_site_id,
            "direction_count": self.direction_count,
            "epsilon_ladder": list(self.epsilon_ladder),
            "diagnostics": [item.metadata() for item in self.diagnostics],
            "direction_normalization": (
                "valid_row_rademacher_RMS_one_then_scaled_by_source_RMS"
            ),
            "baseline_vs_zero_shadow_max_absolute_error": (
                self.baseline_vs_zero_shadow_max_absolute_error
            ),
            "minimum_jvp_cosine": self.minimum_jvp_cosine,
            "maximum_jvp_relative_error": self.maximum_jvp_relative_error,
            "minimum_finite_cosine": self.minimum_finite_cosine,
            "maximum_finite_relative_error": self.maximum_finite_relative_error,
            "maximum_adjacent_finite_derivative_disagreement": (
                self.maximum_adjacent_finite_derivative_disagreement
            ),
            "exact_forward_count": self.exact_forward_count,
            "exact_jvp_count": self.exact_jvp_count,
            "backward_or_jvp_count": self.backward_or_jvp_count,
            "finite_intervention_forward_count": (
                self.finite_intervention_forward_count
            ),
            "absolute_jvp_thresholds_passed": (
                self.absolute_jvp_thresholds_passed
            ),
            "absolute_finite_thresholds_passed": (
                self.absolute_finite_thresholds_passed
            ),
            "negative_control_margin_evaluated": False,
            "causal_gate_authorized": False,
            "baseline_model_output_replaced": False,
            "shadow_execution_only": True,
        }


@dataclass(frozen=True, slots=True)
class ExactEdgeCausalControlComparison:
    """Real-edge causal receipt compared with all frozen negative controls."""

    real: ExactEdgeCausalEvaluation
    row_shuffled: ExactEdgeCausalEvaluation
    family_shuffled: ExactEdgeCausalEvaluation
    provenance_permuted: ExactEdgeCausalEvaluation
    zero_edge_minimum_jvp_cosine: float = 0.0
    zero_edge_minimum_finite_cosine: float = 0.0

    def __post_init__(self) -> None:
        evaluations = (
            self.real,
            self.row_shuffled,
            self.family_shuffled,
            self.provenance_permuted,
        )
        if any(
            not isinstance(item, ExactEdgeCausalEvaluation)
            for item in evaluations
        ):
            raise TypeError("causal control comparison receipts are invalid")
        if any(
            item.target_site_id != self.real.target_site_id
            or item.direction_count != self.real.direction_count
            or item.epsilon_ladder != self.real.epsilon_ladder
            for item in evaluations[1:]
        ):
            raise ValueError("causal controls do not share target/probe contract")
        if (
            self.zero_edge_minimum_jvp_cosine != 0.0
            or self.zero_edge_minimum_finite_cosine != 0.0
        ):
            raise ValueError("zero-edge causal prediction must have zero cosine")

    @property
    def control_minimum_jvp_cosines(self) -> dict[str, float]:
        return {
            "zero_edge_control": self.zero_edge_minimum_jvp_cosine,
            "row_shuffled_control": self.row_shuffled.minimum_jvp_cosine,
            "family_shuffled_control": self.family_shuffled.minimum_jvp_cosine,
            "source_provenance_permutation_control": (
                self.provenance_permuted.minimum_jvp_cosine
            ),
        }

    @property
    def control_minimum_finite_cosines(self) -> dict[str, float]:
        return {
            "zero_edge_control": self.zero_edge_minimum_finite_cosine,
            "row_shuffled_control": self.row_shuffled.minimum_finite_cosine,
            "family_shuffled_control": self.family_shuffled.minimum_finite_cosine,
            "source_provenance_permutation_control": (
                self.provenance_permuted.minimum_finite_cosine
            ),
        }

    @property
    def minimum_jvp_cosine_margin(self) -> float:
        return min(
            self.real.minimum_jvp_cosine - value
            for value in self.control_minimum_jvp_cosines.values()
        )

    @property
    def minimum_finite_cosine_margin(self) -> float:
        return min(
            self.real.minimum_finite_cosine - value
            for value in self.control_minimum_finite_cosines.values()
        )

    @property
    def causal_gate_passed(self) -> bool:
        return (
            self.real.absolute_jvp_thresholds_passed
            and self.real.absolute_finite_thresholds_passed
            and self.minimum_jvp_cosine_margin >= 0.05
            and self.minimum_finite_cosine_margin >= 0.05
        )

    @property
    def exact_forward_count(self) -> int:
        return sum(
            item.exact_forward_count
            for item in (
                self.real,
                self.row_shuffled,
                self.family_shuffled,
                self.provenance_permuted,
            )
        )

    def metadata(self) -> dict[str, object]:
        return {
            "real": self.real.metadata(),
            "controls": {
                "zero_edge_control": {
                    "minimum_jvp_cosine": 0.0,
                    "minimum_finite_cosine": 0.0,
                    "extra_shadow_execution_required": False,
                },
                "row_shuffled_control": self.row_shuffled.metadata(),
                "family_shuffled_control": self.family_shuffled.metadata(),
                "source_provenance_permutation_control": (
                    self.provenance_permuted.metadata()
                ),
            },
            "minimum_jvp_cosine_margin_over_each_control": (
                self.minimum_jvp_cosine_margin
            ),
            "minimum_finite_cosine_margin_over_each_control": (
                self.minimum_finite_cosine_margin
            ),
            "required_margin": 0.05,
            "causal_gate_passed": self.causal_gate_passed,
            "exact_forward_count": self.exact_forward_count,
            "exact_jvp_count": sum(
                item.exact_jvp_count
                for item in (
                    self.real,
                    self.row_shuffled,
                    self.family_shuffled,
                    self.provenance_permuted,
                )
            ),
            "finite_intervention_forward_count": sum(
                item.finite_intervention_forward_count
                for item in (
                    self.real,
                    self.row_shuffled,
                    self.family_shuffled,
                    self.provenance_permuted,
                )
            ),
            "negative_control_margin_evaluated": True,
        }


def _target_residual(
    *,
    target_input: Tensor,
    target_flow: Tensor,
    valid_mask: Tensor,
    local_baseline: FittedUnifiedEventLocalBaseline,
) -> Tensor:
    input_rows = target_input[valid_mask]
    flow_rows = target_flow[valid_mask]
    weight = local_baseline.weight.to(
        device=input_rows.device,
        dtype=input_rows.dtype,
    )
    bias = local_baseline.bias.to(
        device=input_rows.device,
        dtype=input_rows.dtype,
    )
    return flow_rows - (input_rows @ weight.T + bias)


def _direction(
    *,
    edge: FittedUnifiedEventDenseEdge,
    direction_index: int,
    shape: torch.Size,
    valid_mask: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, str]:
    digest = hashlib.sha256(
        _DIRECTION_DOMAIN
        + edge.spec.source_site_id.encode("utf-8")
        + b"\0"
        + edge.spec.target_event_id.encode("utf-8")
        + b"\0"
        + str(direction_index).encode("ascii")
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    values = torch.randint(
        0,
        2,
        shape,
        generator=generator,
        dtype=torch.int8,
        device="cpu",
    ).to(dtype=torch.float32)
    values.mul_(2).sub_(1)
    values.masked_fill_(~valid_mask.detach().cpu().unsqueeze(-1), 0)
    direction_sha256 = hashlib.sha256(
        _DIRECTION_DOMAIN + values.numpy().tobytes(order="C")
    ).hexdigest()
    return values.to(device=device, dtype=dtype), direction_sha256


def evaluate_exact_dense_edge_causality(
    *,
    executor: Gemma3JointCrossBlockChartExecutor,
    input_ids: Tensor,
    sequence: SequenceContext,
    edge: FittedUnifiedEventDenseEdge,
    local_baseline: FittedUnifiedEventLocalBaseline,
    direction_count: int = 8,
    epsilon_ladder: Sequence[float] = (0.001, 0.003, 0.01),
) -> ExactEdgeCausalEvaluation:
    """Compare a dense edge to exact JVPs and finite shadow displacements."""

    if (
        not isinstance(executor, Gemma3JointCrossBlockChartExecutor)
        or executor.arm_id != JOINT_EXACT_CONTROL_ARM_ID
    ):
        raise TypeError("causal evaluation requires the exact 39-event executor")
    if not isinstance(edge, FittedUnifiedEventDenseEdge):
        raise TypeError("edge must be a fitted dense unified-event edge")
    if not isinstance(local_baseline, FittedUnifiedEventLocalBaseline):
        raise TypeError("local_baseline must be a frozen unified-event fit")
    if edge.spec.target_event_id != local_baseline.event_id:
        raise ValueError("edge and local baseline target events disagree")
    if (
        edge.source_fingerprint != local_baseline.source_fingerprint
        or edge.fit_execution_fingerprint
        != local_baseline.fit_execution_fingerprint
        or edge.fit_row_identity_sha256
        != local_baseline.fit_row_identity_sha256
        or edge.fit_row_keys != local_baseline.fit_row_keys
    ):
        raise ValueError("edge and local baseline fit lineage disagrees")
    if type(direction_count) is not int or direction_count <= 0:
        raise ValueError("direction_count must be positive")
    epsilons = _require_epsilon_ladder(epsilon_ladder)
    valid_mask = sequence.query_valid_mask
    perturbation_shape = torch.Size(
        (*input_ids.shape, executor.hidden_width)
    )
    zero = torch.zeros(
        perturbation_shape,
        dtype=next(executor.parameters()).dtype,
        device=input_ids.device,
    )

    baseline_sites: dict[str, Tensor] = {}

    def recorder(*, event_id: str, role: str, values: Tensor) -> None:
        site_id = f"{event_id}.{role}"
        if site_id in {
            edge.spec.source_site_id,
            local_baseline.input_site_id,
            local_baseline.output_site_id,
        }:
            baseline_sites[site_id] = values

    baseline_observed = executor.forward_components_with_observer(
        input_ids,
        sequence,
        observer=recorder,
    )
    source_values = baseline_sites.get(edge.spec.source_site_id)
    target_input = baseline_sites.get(local_baseline.input_site_id)
    target_flow = baseline_sites.get(local_baseline.output_site_id)
    if any(value is None for value in (source_values, target_input, target_flow)):
        raise RuntimeError("baseline shadow sites were not observed")
    assert isinstance(source_values, Tensor)
    assert isinstance(target_input, Tensor)
    assert isinstance(target_flow, Tensor)
    baseline_residual = _target_residual(
        target_input=target_input,
        target_flow=target_flow,
        valid_mask=valid_mask.cpu(),
        local_baseline=local_baseline,
    )

    def exact_residual(perturbation: Tensor) -> Tensor:
        observed = executor.forward_components_with_shadow_intervention(
            input_ids,
            sequence,
            source_site_id=edge.spec.source_site_id,
            target_event_id=edge.spec.target_event_id,
            perturbation=perturbation,
        )
        if (
            observed.shadow_target_input is None
            or observed.shadow_target_flow is None
        ):
            raise RuntimeError("shadow execution omitted its target tensors")
        return _target_residual(
            target_input=observed.shadow_target_input,
            target_flow=observed.shadow_target_flow,
            valid_mask=valid_mask,
            local_baseline=local_baseline,
        )

    zero_residual = exact_residual(zero)
    baseline_error = float(
        (zero_residual.detach().cpu() - baseline_residual).abs().max()
    )
    if baseline_error != 0.0:
        raise RuntimeError("zero shadow intervention changed exact target flow")
    source_rms = math.sqrt(
        float(source_values[valid_mask.cpu()].double().square().mean())
    )
    if not math.isfinite(source_rms) or source_rms <= 0.0:
        raise RuntimeError("source chart RMS is degenerate")

    coefficient = edge.coefficient.to(device=zero.device, dtype=zero.dtype)
    diagnostics: list[ExactEdgeDirectionDiagnostic] = []
    for direction_index in range(direction_count):
        unit_direction, direction_sha256 = _direction(
            edge=edge,
            direction_index=direction_index,
            shape=perturbation_shape,
            valid_mask=valid_mask,
            dtype=zero.dtype,
            device=zero.device,
        )
        tangent = unit_direction * source_rms
        predicted = tangent[valid_mask] @ coefficient
        _, exact_jvp = torch.autograd.functional.jvp(
            exact_residual,
            (zero,),
            (tangent,),
            create_graph=False,
            strict=True,
        )
        finite_derivatives: list[Tensor] = []
        finite_cosines: list[float] = []
        finite_errors: list[float] = []
        for epsilon in epsilons:
            plus = exact_residual(tangent * epsilon)
            minus = exact_residual(tangent * -epsilon)
            derivative = (plus - minus) / (2.0 * epsilon)
            finite_derivatives.append(derivative.detach())
            finite_cosines.append(_cosine(predicted, derivative))
            finite_errors.append(_relative_error(predicted, derivative))
        adjacent = tuple(
            _relative_error(right, left)
            for left, right in zip(
                finite_derivatives,
                finite_derivatives[1:],
                strict=False,
            )
        )
        diagnostics.append(
            ExactEdgeDirectionDiagnostic(
                direction_index=direction_index,
                direction_sha256=direction_sha256,
                source_rms=source_rms,
                exact_jvp_cosine=_cosine(predicted, exact_jvp),
                exact_jvp_relative_error=_relative_error(
                    predicted,
                    exact_jvp,
                ),
                finite_cosines=tuple(finite_cosines),
                finite_relative_errors=tuple(finite_errors),
                adjacent_finite_derivative_disagreements=adjacent,
            )
        )

    return ExactEdgeCausalEvaluation(
        source_site_id=edge.spec.source_site_id,
        target_site_id=f"{edge.spec.target_event_id}.output",
        direction_count=direction_count,
        epsilon_ladder=epsilons,
        diagnostics=tuple(diagnostics),
        baseline_vs_zero_shadow_max_absolute_error=baseline_error,
        minimum_jvp_cosine=min(item.exact_jvp_cosine for item in diagnostics),
        maximum_jvp_relative_error=max(
            item.exact_jvp_relative_error for item in diagnostics
        ),
        minimum_finite_cosine=min(
            value for item in diagnostics for value in item.finite_cosines
        ),
        maximum_finite_relative_error=max(
            value
            for item in diagnostics
            for value in item.finite_relative_errors
        ),
        maximum_adjacent_finite_derivative_disagreement=max(
            (
                value
                for item in diagnostics
                for value in item.adjacent_finite_derivative_disagreements
            ),
            default=0.0,
        ),
        exact_forward_count=(
            2 + direction_count + 2 * direction_count * len(epsilons)
        ),
        exact_jvp_count=direction_count,
        backward_or_jvp_count=direction_count,
        finite_intervention_forward_count=(
            2 * direction_count * len(epsilons)
        ),
    )


__all__ = [
    "ExactEdgeCausalControlComparison",
    "ExactEdgeCausalEvaluation",
    "ExactEdgeDirectionDiagnostic",
    "evaluate_exact_dense_edge_causality",
]
