"""Probe conservative decisions over a fitted pointwise route score vector.

This is intentionally a calibration-B-only diagnostic.  It never reads the
locked validation or reserved test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fisher_graph.conditional_routing import ConditionalModalRoutingPlan
from fisher_graph.modes import FisherModeBasis
from fisher_graph.variable_associative_training import (
    load_variable_associative_checkpoint,
    variable_associative_answer_logits,
)
from fisher_graph.variable_conditional_experiment import (
    _behavior_record,
    _collect,
    _context_role_splits,
    _projected_answer_logits,
    _route_accounting,
    _scatter_valid_rows,
)


def _candidate_rows(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    probabilities = logits.softmax(dim=-1)
    routes = logits.shape[-1]
    route_values = torch.arange(
        routes,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    candidates = {"argmax": logits.argmax(dim=-1)}
    expected = (probabilities * route_values).sum(dim=-1)
    candidates["expected_round"] = expected.round().to(torch.int64)
    candidates["expected_ceil"] = expected.ceil().to(torch.int64)
    cumulative = probabilities.cumsum(dim=-1)
    for quantile in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        candidates[f"posterior_q{int(100 * quantile)}"] = (
            cumulative < quantile
        ).sum(dim=-1).clamp_max(routes - 1)
    margin = logits.topk(2, dim=-1).values
    gap = margin[:, 0] - margin[:, 1]
    argmax = candidates["argmax"]
    for threshold in (0.05, 0.10, 0.20, 0.30, 0.50):
        candidates[f"uncertain_plus_one_{threshold:.2f}"] = torch.where(
            gap < threshold,
            (argmax + 1).clamp_max(routes - 1),
            argmax,
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".local-runs/variable-associative/checkpoint.pt"),
    )
    arguments = parser.parse_args()
    artifact = torch.load(
        arguments.artifact,
        map_location="cpu",
        weights_only=True,
    )
    basis = FisherModeBasis.from_state_dict(artifact["basis"])
    plan = ConditionalModalRoutingPlan.from_state_dict(
        artifact["routing_plan"]
    )
    static_rank = int(
        artifact["analysis"]["calibration_b"]["analysis"][
            "smallest_passing_static"
        ]["rank"]
    )
    layer_index = int(artifact["protocol"]["layer_index"])
    contexts_per_role = int(artifact["protocol"]["contexts_per_role"])
    model, splits, _ = load_variable_associative_checkpoint(
        arguments.checkpoint
    )
    roles = _context_role_splits(
        splits.train,
        contexts_per_role=contexts_per_role,
    )
    split = roles["calibration_b"]
    input_site = f"layer.{layer_index}.input"
    output_site = f"layer.{layer_index}.output"
    inputs = _collect(
        model,
        split,
        activation_names={input_site},
    )[input_site]
    logits = plan.router.logits(inputs.activations)
    baseline_logits = variable_associative_answer_logits(model, split)
    static_macs = (
        2
        * basis.width
        * static_rank
        * int(split.attention_mask.sum().item())
    )
    records = {}
    for name, route_rows in _candidate_rows(logits).items():
        grid = _scatter_valid_rows(split, route_rows)
        projected, _ = _projected_answer_logits(
            model,
            split,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            mode_table=plan.mode_table,
            route_grid=grid,
        )
        behavior = _behavior_record(projected, baseline_logits, split)
        accounting = _route_accounting(
            grid,
            split,
            plan.mode_table,
            router_input_width=plan.router.input_features,
        )
        records[name] = {
            "passed": behavior["passed"],
            "hard_nll": behavior["metrics"]["hard_nll"],
            "answer_accuracy": behavior["metrics"]["answer_accuracy"],
            "paired_context_accuracy": behavior["metrics"][
                "paired_context_accuracy"
            ],
            "minimum_layout_accuracy": behavior["metrics"][
                "minimum_layout_accuracy"
            ],
            "average_active_modes": accounting["average_active_modes"],
            "full_width_fallback_rate": accounting[
                "full_width_fallback_rate"
            ],
            "static_rank": static_rank,
            "router_projection_to_static_mac_ratio": (
                accounting["router_plus_projection_macs"] / static_macs
            ),
            "route_counts": accounting["route_counts"],
        }
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
