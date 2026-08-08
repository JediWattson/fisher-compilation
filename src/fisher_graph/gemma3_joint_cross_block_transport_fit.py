"""Fit rectangular, strictly-forward generator transports.

The MLP-only transport rung assumed that both endpoints used the same latent
rank.  A joint Gemma graph cannot make that assumption: attention emits an
output-coordinate packet whose rank may differ from the chart rank consumed by
a later MLP generator.  This module fits the rectangular map used by the joint
executor without consulting layer identity as a numerical feature.

The target decoder rows are required to be orthonormal.  For one candidate
source packet ``X`` and target-chart decoder ``D``, the exact ridge block
minimizer is

``(X.T W X + lambda I) T = X.T W R D.T``.

Greedy selection measures partial Fisher-weighted residual gain beyond the
already-fitted local target generator.  Selected blocks are then refined in a
fixed canonical order while all local generator coefficients remain frozen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class FittedJointCrossBlockTransport:
    """One fit-only rectangular source-chart to target-chart map."""

    source_event_id: str
    source_chart_index: int
    target_chart_index: int
    transport_weight: Tensor
    greedy_step: int
    greedy_weighted_sse_gain: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_event_id, str) or not self.source_event_id:
            raise ValueError("source_event_id must be nonempty")
        if type(self.source_chart_index) is not int or self.source_chart_index < 0:
            raise ValueError("source_chart_index must be nonnegative")
        if type(self.target_chart_index) is not int or self.target_chart_index < 0:
            raise ValueError("target_chart_index must be nonnegative")
        if (
            not isinstance(self.transport_weight, Tensor)
            or self.transport_weight.ndim != 2
            or not self.transport_weight.is_floating_point()
            or not bool(torch.isfinite(self.transport_weight).all())
        ):
            raise ValueError("transport_weight must be a finite matrix")
        if type(self.greedy_step) is not int or self.greedy_step < 0:
            raise ValueError("greedy_step must be nonnegative")
        if (
            not isinstance(self.greedy_weighted_sse_gain, (int, float))
            or isinstance(self.greedy_weighted_sse_gain, bool)
            or not math.isfinite(float(self.greedy_weighted_sse_gain))
            or float(self.greedy_weighted_sse_gain) <= 0.0
        ):
            raise ValueError("greedy gain must be finite and positive")


def _normalized_weights(value: Tensor, *, row_count: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.shape[0] != row_count
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
        or bool((value <= 0).any())
    ):
        raise ValueError("row_fisher_weights must be finite and positive")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    return result / result.mean().clamp_min(1.0e-12)


def _weighted_sse(residual: Tensor, weights: Tensor) -> Tensor:
    return (weights[:, None] * residual.square()).sum()


def _fit_block(
    source: Tensor,
    residual: Tensor,
    decoder: Tensor,
    weights: Tensor,
    *,
    relative_ridge: float,
) -> tuple[Tensor, Tensor, float]:
    design = source.to(device="cpu", dtype=torch.float64)
    target = residual.to(device="cpu", dtype=torch.float64)
    decode = decoder.to(device="cpu", dtype=torch.float64)
    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design
    scale = float(torch.trace(gram) / max(1, gram.shape[0]))
    absolute_ridge = float(relative_ridge) * max(scale, 1.0e-12)
    right = design.T @ (weights[:, None] * target) @ decode.T
    identity = torch.eye(gram.shape[0], dtype=torch.float64)
    coefficient = torch.linalg.solve(
        gram + absolute_ridge * identity,
        right,
    )
    prediction = design @ coefficient @ decode
    return coefficient, prediction, absolute_ridge


def fit_greedy_joint_cross_block_transports(
    *,
    source_packets: Mapping[str, Tensor],
    target_memberships: Tensor,
    target_mode_decoders: Tensor,
    target_modal_residual: Tensor,
    row_fisher_weights: Tensor,
    maximum_incoming_sources_per_target_chart: int,
    relative_ridge: float,
    refinement_sweeps: int = 8,
) -> tuple[tuple[FittedJointCrossBlockTransport, ...], dict[str, object]]:
    """Fit sparse rectangular transports into one MLP chart generator.

    ``source_packets`` maps earlier event ids to tensors shaped ``[N,Cs,Rs]``.
    Source ranks may differ across events.  The target decoder has shape
    ``[Ct,Rt,K]`` and must have row-orthonormal ``[Rt,K]`` blocks.
    """

    if not isinstance(source_packets, Mapping) or not source_packets:
        raise ValueError("source_packets must contain an earlier event")
    if (
        not isinstance(target_modal_residual, Tensor)
        or target_modal_residual.ndim != 2
        or not target_modal_residual.is_floating_point()
        or not bool(torch.isfinite(target_modal_residual).all())
    ):
        raise ValueError("target_modal_residual must be a finite [N,K] tensor")
    rows, modes = target_modal_residual.shape
    if (
        not isinstance(target_memberships, Tensor)
        or target_memberships.ndim != 2
        or target_memberships.shape[0] != rows
        or not target_memberships.is_floating_point()
        or not bool(torch.isfinite(target_memberships).all())
        or bool((target_memberships < 0).any())
        or not torch.allclose(
            target_memberships.sum(dim=1),
            torch.ones(rows, dtype=target_memberships.dtype),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
    ):
        raise ValueError("target_memberships must be a finite row simplex")
    charts = target_memberships.shape[1]
    if (
        not isinstance(target_mode_decoders, Tensor)
        or target_mode_decoders.ndim != 3
        or target_mode_decoders.shape[0] != charts
        or target_mode_decoders.shape[2] != modes
        or not target_mode_decoders.is_floating_point()
        or not bool(torch.isfinite(target_mode_decoders).all())
    ):
        raise ValueError("target_mode_decoders shape is inconsistent")
    target_rank = target_mode_decoders.shape[1]
    identity = torch.eye(target_rank, dtype=torch.float64)
    for decoder in target_mode_decoders:
        gram = decoder.double() @ decoder.double().T
        if not torch.allclose(gram, identity, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError("target decoder rows must be orthonormal")
    if (
        type(maximum_incoming_sources_per_target_chart) is not int
        or maximum_incoming_sources_per_target_chart <= 0
    ):
        raise ValueError("maximum incoming sources must be positive")
    if (
        not isinstance(relative_ridge, (int, float))
        or isinstance(relative_ridge, bool)
        or not math.isfinite(float(relative_ridge))
        or float(relative_ridge) <= 0.0
    ):
        raise ValueError("relative_ridge must be finite and positive")
    if type(refinement_sweeps) is not int or refinement_sweeps < 0:
        raise ValueError("refinement_sweeps must be nonnegative")

    canonical_packets: dict[str, Tensor] = {}
    for event_id, packet in sorted(source_packets.items()):
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("source event ids must be nonempty")
        if (
            not isinstance(packet, Tensor)
            or packet.ndim != 3
            or packet.shape[0] != rows
            or not packet.shape[1]
            or not packet.shape[2]
            or not packet.is_floating_point()
            or not bool(torch.isfinite(packet).all())
        ):
            raise ValueError("source packets must be finite [N,C,R] tensors")
        canonical_packets[event_id] = packet.detach().cpu().float().contiguous()

    weights = _normalized_weights(row_fisher_weights, row_count=rows)
    residual = target_modal_residual.detach().cpu().double().contiguous()
    memberships = target_memberships.detach().cpu().double().contiguous()
    decoders = target_mode_decoders.detach().cpu().double().contiguous()
    baseline_sse = float(_weighted_sse(residual, weights))
    incoming_counts = [0] * charts
    selected: list[dict[str, object]] = []
    candidate_count = sum(
        packet.shape[1] * charts for packet in canonical_packets.values()
    )
    tolerance = 128.0 * torch.finfo(torch.float64).eps * max(1.0, baseline_sse)

    for step in range(charts * maximum_incoming_sources_per_target_chart):
        best: tuple[
            float,
            str,
            int,
            int,
            Tensor,
            Tensor,
            float,
        ] | None = None
        for event_id, packet in canonical_packets.items():
            for source_chart in range(packet.shape[1]):
                base_source = packet[:, source_chart, :].double()
                for target_chart in range(charts):
                    if incoming_counts[target_chart] >= (
                        maximum_incoming_sources_per_target_chart
                    ):
                        continue
                    if any(
                        item["source_event_id"] == event_id
                        and item["source_chart_index"] == source_chart
                        and item["target_chart_index"] == target_chart
                        for item in selected
                    ):
                        continue
                    design = memberships[:, target_chart, None] * base_source
                    coefficient, prediction, absolute_ridge = _fit_block(
                        design,
                        residual,
                        decoders[target_chart],
                        weights,
                        relative_ridge=float(relative_ridge),
                    )
                    gain = float(
                        _weighted_sse(residual, weights)
                        - _weighted_sse(residual - prediction, weights)
                    )
                    key = (event_id, source_chart, target_chart)
                    if gain <= tolerance:
                        continue
                    if best is None or gain > best[0] + tolerance or (
                        abs(gain - best[0]) <= tolerance
                        and key < (best[1], best[2], best[3])
                    ):
                        best = (
                            gain,
                            event_id,
                            source_chart,
                            target_chart,
                            coefficient,
                            prediction,
                            absolute_ridge,
                        )
        if best is None:
            break
        gain, event_id, source_chart, target_chart, coefficient, prediction, ridge = best
        selected.append(
            {
                "source_event_id": event_id,
                "source_chart_index": source_chart,
                "target_chart_index": target_chart,
                "transport_weight": coefficient,
                "greedy_step": step,
                "greedy_weighted_sse_gain": gain,
                "absolute_ridge": ridge,
            }
        )
        incoming_counts[target_chart] += 1
        residual = residual - prediction

    sweeps_completed = 0
    for _ in range(refinement_sweeps):
        if not selected:
            break
        before = float(_weighted_sse(residual, weights))
        for item in selected:
            event_id = str(item["source_event_id"])
            source_chart = int(item["source_chart_index"])
            target_chart = int(item["target_chart_index"])
            old = item["transport_weight"]
            assert isinstance(old, Tensor)
            design = (
                memberships[:, target_chart, None]
                * canonical_packets[event_id][:, source_chart, :].double()
            )
            old_prediction = design @ old @ decoders[target_chart]
            residual_without = residual + old_prediction
            coefficient, prediction, ridge = _fit_block(
                design,
                residual_without,
                decoders[target_chart],
                weights,
                relative_ridge=float(relative_ridge),
            )
            item["transport_weight"] = coefficient
            item["absolute_ridge"] = ridge
            residual = residual_without - prediction
        after = float(_weighted_sse(residual, weights))
        if after > before + tolerance:
            raise RuntimeError("transport block refinement increased fit loss")
        sweeps_completed += 1

    fitted = tuple(
        FittedJointCrossBlockTransport(
            source_event_id=str(item["source_event_id"]),
            source_chart_index=int(item["source_chart_index"]),
            target_chart_index=int(item["target_chart_index"]),
            transport_weight=(
                item["transport_weight"].to(
                    dtype=target_modal_residual.dtype,
                    device=target_modal_residual.device,
                ).contiguous()
            ),
            greedy_step=int(item["greedy_step"]),
            greedy_weighted_sse_gain=float(
                item["greedy_weighted_sse_gain"]
            ),
        )
        for item in selected
    )
    final_sse = float(_weighted_sse(residual, weights))
    return fitted, {
        "fit_algorithm": (
            "greedy_rectangular_fisher_weighted_ridge_then_"
            "canonical_block_refinement"
        ),
        "row_count": rows,
        "source_event_count": len(canonical_packets),
        "candidate_edge_count": candidate_count,
        "selected_edge_count": len(fitted),
        "target_chart_count": charts,
        "target_generator_rank": target_rank,
        "maximum_incoming_sources_per_target_chart": (
            maximum_incoming_sources_per_target_chart
        ),
        "relative_ridge": float(relative_ridge),
        "refinement_sweeps_completed": sweeps_completed,
        "baseline_weighted_squared_error": baseline_sse,
        "fitted_weighted_squared_error": final_sse,
        "weighted_squared_error_reduction": baseline_sse - final_sse,
        "transport_partial_explained_fraction": (
            0.0
            if baseline_sse <= 0.0
            else (baseline_sse - final_sse) / baseline_sse
        ),
        "incoming_edge_count_by_target": incoming_counts,
        "source_packet_ranks": {
            event_id: int(packet.shape[2])
            for event_id, packet in canonical_packets.items()
        },
    }


__all__ = [
    "FittedJointCrossBlockTransport",
    "fit_greedy_joint_cross_block_transports",
]
