"""Explore conditional Fisher-mode allocation on associative recall.

This experiment deliberately stops one rung before claiming a compiled
transformer replacement.  It fits a Fisher basis on one context-grouped
training partition, learns a pointwise input-only router on another, and then
projects the *native* layer output through route-specific mode subsets.

The native layer therefore still executes.  The experiment answers whether
conditional modal allocation preserves representation quality with fewer
active coordinates than a static projection.  It does not establish source
layer parameter, FLOP, kernel, or latency savings.

The controls are designed to expose the most important confound in the
repository's fixed-format associative-recall task:

* a static equal-or-greater average-rank projection;
* the smallest static rank passing the same behavior gates;
* a budget-histogram-matched shuffled route assignment; and
* an A-fitted position-only route schedule.

If the input router does not materially beat the position-only control, the
result supports conditional allocation but not content-conditioned routing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .adapters import ToyTransformerAdapter, module_state_fingerprint
from .associative import (
    AssociativeRecallMetrics,
    AssociativeRecallSplit,
    AssociativeRecallTaskConfig,
    build_associative_recall_splits,
    evaluate_associative_recall,
)
from .conditional_executor import (
    ConditionalModalProjectionOracleExecutor,
    HardRoutedFisherProjection,
)
from .conditional_routing import (
    ConditionalModalRoutingPlan,
    ConditionalModeTable,
    ConditionalRoutingFit,
    fisher_projection_damage_profiles,
    fit_conditional_modal_routing,
)
from .config import TransformerConfig
from .gemma3_ablation_experiment import _update_payload_digest
from .modes import (
    FisherModeBasis,
    collect_activation_score_gradients,
    decompose_fisher_modes,
)
from .model import ToyTransformer


DEFAULT_ARTIFACT_DIR = Path("artifacts/associative_recall")
DEFAULT_LAYER_INDEX = 0
DEFAULT_BASIS_EXAMPLES = 768
DEFAULT_ROUTER_EXAMPLES = 768
DEFAULT_CALIBRATION_B_EXAMPLES = 768
DEFAULT_ROUTE_BUDGETS = (0, 6, 16, 32)
DEFAULT_ROUTER_RIDGE = 1e-2
DEFAULT_SHUFFLE_SEED = 52_021
DEFAULT_MINIMUM_ANSWER_ACCURACY = 0.995
DEFAULT_MINIMUM_PAIRED_ACCURACY = 0.99
DEFAULT_MAXIMUM_NLL_INCREASE = 0.05
DEFAULT_MINIMUM_STATE_NLL_ADVANTAGE = 0.005

_ARTIFACT_SCHEMA = "fisher_graph.associative_conditional_rank"
_ARTIFACT_FORMAT_VERSION = 1
_PAYLOAD_DOMAIN = b"fisher_graph.associative_conditional_rank_payload.v1\0"
_REPORT_DOMAIN = b"fisher_graph.associative_conditional_rank_report.v1\0"
_PROFILE_SEMANTICS = (
    "squared_first_order_score_change_from_independent_orthonormal_"
    "mode_suppression"
)
_ROUTE_ASSIGNMENT = "equal_frequency_total_first_order_need_bins"
_TEST_POLICY = "existing_test_split_not_model_evaluated_by_this_run"


def default_associative_conditional_rank_output(
    *,
    layer_index: int = DEFAULT_LAYER_INDEX,
) -> Path:
    """Return an ignored output path for the exploratory routing artifact."""

    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    return (
        Path(".local-runs")
        / "associative-recall"
        / f"layer-{layer_index}-conditional-rank.pt"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor, *, domain: bytes) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _scientific_payload_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(_PAYLOAD_DOMAIN)
    _update_payload_digest(digest, payload)
    return digest.hexdigest()


def _report_sha256(report: Mapping[str, object]) -> str:
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_REPORT_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _finite(
    value: object,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return result


def _normalize_budgets(
    values: Sequence[int],
    *,
    width: int,
) -> tuple[int, ...]:
    budgets = tuple(values)
    if (
        len(budgets) < 2
        or any(type(value) is not int for value in budgets)
        or tuple(sorted(set(budgets))) != budgets
        or budgets[0] < 0
        or budgets[-1] != width
    ):
        raise ValueError(
            "route_budgets must be unique ascending values ending at width"
        )
    return budgets


def _subset_split(
    split: AssociativeRecallSplit,
    *,
    start: int,
    stop: int,
    name: str,
) -> AssociativeRecallSplit:
    """Slice complete context pairs and remap their context row indices."""

    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop <= split.samples
    ):
        raise ValueError("split slice is outside the source split")
    old_context_rows = split.example_context_indices[start:stop]
    unique_rows, counts = torch.unique(
        old_context_rows,
        sorted=True,
        return_counts=True,
    )
    if unique_rows.numel() == 0 or not bool((counts == 2).all()):
        raise ValueError(
            "conditional-rank partitions must contain complete two-query "
            "contexts"
        )
    remap = {
        int(old): new
        for new, old in enumerate(unique_rows.tolist())
    }
    remapped = torch.tensor(
        [remap[int(value)] for value in old_context_rows.tolist()],
        dtype=torch.long,
    )
    return AssociativeRecallSplit(
        name=name,
        input_ids=split.input_ids[start:stop],
        targets=split.targets[start:stop],
        query_slots=split.query_slots[start:stop],
        answer_value_indices=split.answer_value_indices[start:stop],
        example_context_indices=remapped,
        context_ids=split.context_ids.index_select(0, unique_rows),
        n_values=split.n_values,
    )


def _split_provenance(split: AssociativeRecallSplit) -> dict[str, object]:
    return {
        "name": split.name,
        "examples": split.samples,
        "contexts": split.contexts,
        "input_ids_sha256": _tensor_sha256(
            split.input_ids,
            domain=b"fisher_graph.conditional_rank.input_ids.v1\0",
        ),
        "targets_sha256": _tensor_sha256(
            split.targets,
            domain=b"fisher_graph.conditional_rank.targets.v1\0",
        ),
        "context_ids_sha256": _tensor_sha256(
            split.context_ids,
            domain=b"fisher_graph.conditional_rank.context_ids.v1\0",
        ),
    }


def _behavior_gates(
    metrics: AssociativeRecallMetrics,
    baseline: AssociativeRecallMetrics,
    *,
    minimum_answer_accuracy: float,
    minimum_paired_accuracy: float,
    maximum_nll_increase: float,
) -> dict[str, bool]:
    return {
        "answer_accuracy": (
            metrics.answer_accuracy >= minimum_answer_accuracy
        ),
        "paired_context_accuracy": (
            metrics.paired_context_accuracy >= minimum_paired_accuracy
        ),
        "hard_nll": (
            metrics.hard_nll - baseline.hard_nll
            <= maximum_nll_increase
        ),
    }


def _metric_record(
    metrics: AssociativeRecallMetrics,
    baseline: AssociativeRecallMetrics,
    *,
    minimum_answer_accuracy: float,
    minimum_paired_accuracy: float,
    maximum_nll_increase: float,
) -> dict[str, object]:
    gates = _behavior_gates(
        metrics,
        baseline,
        minimum_answer_accuracy=minimum_answer_accuracy,
        minimum_paired_accuracy=minimum_paired_accuracy,
        maximum_nll_increase=maximum_nll_increase,
    )
    return {
        "metrics": asdict(metrics),
        "delta_hard_nll": metrics.hard_nll - baseline.hard_nll,
        "gates": gates,
        "passed": all(gates.values()),
    }


@torch.no_grad()
def _collect_site_activations(
    model: ToyTransformer,
    split: AssociativeRecallSplit,
    *,
    site: str,
    batch_size: int = 256,
) -> Tensor:
    rows = []
    for start in range(0, split.samples, batch_size):
        output = model(
            split.input_ids[start : start + batch_size],
            capture_activations=True,
            retain_activation_gradients=False,
        )
        if output.activations is None:
            raise RuntimeError("source model did not return activations")
        rows.append(output.activations[site].detach().cpu())
    return torch.cat(rows, dim=0)


def _dense_project_with_routes(
    activation: Tensor,
    *,
    basis: FisherModeBasis,
    table: ConditionalModeTable,
    routes: Tensor,
) -> Tensor:
    if routes.shape != activation.shape[:-1]:
        raise ValueError("route tensor does not match projected activations")
    compute = activation.to(dtype=torch.float64)
    coordinates = (compute - basis.mean) @ basis.vectors
    masked = table.mask_coordinates(coordinates, routes.to(coordinates.device))
    return (masked @ basis.vectors.T + basis.mean).to(
        dtype=activation.dtype
    )


def _static_projection(
    basis: FisherModeBasis,
    *,
    rank: int,
) -> Callable[[Tensor], Tensor]:
    if type(rank) is not int or not 0 <= rank <= basis.width:
        raise ValueError("static rank is outside the Fisher basis")
    vectors = basis.vectors[:, :rank]

    def project(activation: Tensor) -> Tensor:
        compute = activation.to(dtype=torch.float64)
        coordinates = (compute - basis.mean) @ vectors
        return (coordinates @ vectors.T + basis.mean).to(
            dtype=activation.dtype
        )

    return project


def _evaluate_static(
    model: ToyTransformer,
    split: AssociativeRecallSplit,
    *,
    output_site: str,
    basis: FisherModeBasis,
    rank: int,
) -> AssociativeRecallMetrics:
    return evaluate_associative_recall(
        model,
        split,
        activation_interventions={
            output_site: _static_projection(basis, rank=rank)
        },
    )


def _evaluate_learned_oracle(
    model: ToyTransformer,
    split: AssociativeRecallSplit,
    *,
    layer_index: int,
    basis: FisherModeBasis,
    plan: ConditionalModalRoutingPlan,
) -> tuple[AssociativeRecallMetrics, dict[str, object]]:
    adapter = ToyTransformerAdapter(model)
    try:
        segment = adapter.segments[layer_index]
    except IndexError:
        raise ValueError("layer_index is outside the toy model") from None
    source = adapter.source_module(segment.layer_ids[0])
    if not isinstance(source, torch.nn.Module):
        raise TypeError("source layer must be a torch module")
    oracle = ConditionalModalProjectionOracleExecutor(
        source,  # type: ignore[arg-type]
        HardRoutedFisherProjection(basis, plan),
    )
    with adapter.replaced_segments({segment.id: oracle}):
        metrics = evaluate_associative_recall(model, split)
    status = oracle.execution_status()
    return metrics, {
        **asdict(status),
        "logical_active_mode_ratio": status.logical_active_mode_ratio,
    }


def _evaluate_fixed_routes(
    model: ToyTransformer,
    split: AssociativeRecallSplit,
    *,
    input_site: str,
    output_site: str,
    basis: FisherModeBasis,
    table: ConditionalModeTable,
    route_ids: Tensor,
) -> AssociativeRecallMetrics:
    if route_ids.shape != split.input_ids.shape:
        raise ValueError("fixed routes must match the split token grid")
    cursor = 0
    current: Tensor | None = None

    def capture_input(activation: Tensor) -> Tensor:
        nonlocal cursor, current
        batch = activation.shape[0]
        current = route_ids[cursor : cursor + batch].to(
            device=activation.device
        )
        cursor += batch
        return activation

    def project_output(activation: Tensor) -> Tensor:
        nonlocal current
        if current is None:
            raise RuntimeError("output projection ran before input routing")
        projected = _dense_project_with_routes(
            activation,
            basis=basis,
            table=table,
            routes=current,
        )
        current = None
        return projected

    result = evaluate_associative_recall(
        model,
        split,
        activation_interventions={
            input_site: capture_input,
            output_site: project_output,
        },
    )
    if cursor != split.samples or current is not None:
        raise RuntimeError("fixed route replay did not consume exactly once")
    return result


def _majority_routes_by_position(
    teacher_routes: Tensor,
    locations: Tensor,
    *,
    route_count: int,
    sequence_length: int,
) -> tuple[Tensor, list[dict[str, object]]]:
    if teacher_routes.ndim != 1 or locations.shape != (
        teacher_routes.numel(),
        2,
    ):
        raise ValueError("teacher routes and locations are not aligned")
    majority = torch.empty(sequence_length, dtype=torch.long)
    rows = []
    for position in range(sequence_length):
        selected = teacher_routes[locations[:, 1] == position]
        if selected.numel() == 0:
            raise ValueError("router fit omitted a logical position")
        counts = torch.bincount(selected, minlength=route_count)
        route = int(counts.argmax().item())
        probabilities = counts.to(torch.float64) / selected.numel()
        nonzero = probabilities > 0
        entropy = float(
            -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            .item()
        )
        majority[position] = route
        rows.append(
            {
                "position": position,
                "teacher_route_counts": tuple(
                    int(value) for value in counts.tolist()
                ),
                "majority_route": route,
                "route_entropy_nats": entropy,
            }
        )
    return majority, rows


def _fixed_position_grid(
    majority_routes: Tensor,
    *,
    examples: int,
) -> Tensor:
    return majority_routes.unsqueeze(0).expand(examples, -1).clone()


def _shuffle_route_grid(route_ids: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    flat = route_ids.detach().cpu().reshape(-1)
    permutation = torch.randperm(flat.numel(), generator=generator)
    return flat[permutation].reshape_as(route_ids)


def _route_accounting(
    route_ids: Tensor,
    *,
    table: ConditionalModeTable,
    input_width: int,
) -> dict[str, object]:
    routes = route_ids.detach().cpu().to(dtype=torch.long)
    counts = torch.bincount(routes.reshape(-1), minlength=table.routes)
    budgets = torch.tensor(table.route_budgets, dtype=torch.float64)
    active = budgets[routes]
    tokens = routes.numel()
    active_applications = int(active.sum().item())
    router_macs = tokens * input_width * table.routes
    projection_macs = 2 * table.modes * active_applications
    return {
        "tokens": tokens,
        "route_counts": tuple(int(value) for value in counts.tolist()),
        "route_utilization": tuple(
            float(value) / tokens for value in counts.tolist()
        ),
        "active_mode_applications": active_applications,
        "average_active_modes": float(active.mean().item()),
        "p50_active_modes": float(torch.quantile(active, 0.5).item()),
        "p90_active_modes": float(torch.quantile(active, 0.9).item()),
        "maximum_active_modes": int(active.max().item()),
        "full_width_fallback_rate": (
            float((active == table.modes).sum().item()) / tokens
        ),
        "router_linear_macs": router_macs,
        "ideal_sparse_projection_macs": projection_macs,
        "router_plus_projection_macs": router_macs + projection_macs,
        "dense_full_width_projection_macs": (
            2 * table.modes * table.modes * tokens
        ),
        "resource_scope": (
            "router_and_modal_projection_only_native_layer_still_executes"
        ),
    }


def _route_counts_by_position(
    route_ids: Tensor,
    *,
    route_count: int,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "position": position,
            "route_counts": tuple(
                int(value)
                for value in torch.bincount(
                    route_ids[:, position],
                    minlength=route_count,
                ).tolist()
            ),
        }
        for position in range(route_ids.shape[1])
    )


def _paired_route_changes(
    split: AssociativeRecallSplit,
    route_ids: Tensor,
) -> dict[str, object]:
    changes = torch.zeros(route_ids.shape[1], dtype=torch.long)
    contexts = 0
    for context in range(split.contexts):
        indices = (
            split.example_context_indices == context
        ).nonzero(as_tuple=False).flatten()
        if indices.numel() != 2:
            raise ValueError("paired route audit requires two examples")
        changes += (
            route_ids[indices[0]] != route_ids[indices[1]]
        ).to(dtype=torch.long)
        contexts += 1
    return {
        "contexts": contexts,
        "position_change_counts": tuple(
            int(value) for value in changes.tolist()
        ),
        "position_change_fractions": tuple(
            float(value) / contexts for value in changes.tolist()
        ),
        "any_route_change_contexts": sum(
            int(
                bool(
                    (
                        route_ids[
                            (split.example_context_indices == context)
                            .nonzero(as_tuple=False)
                            .flatten()[0]
                        ]
                        != route_ids[
                            (split.example_context_indices == context)
                            .nonzero(as_tuple=False)
                            .flatten()[1]
                        ]
                    ).any()
                )
            )
            for context in range(split.contexts)
        ),
    }


def _fit_summary(fit: ConditionalRoutingFit) -> dict[str, object]:
    return {
        "route_assignment": _ROUTE_ASSIGNMENT,
        "route_counts": fit.clustering.route_counts,
        "clustering_objective": fit.clustering.objective,
        "router_classification": asdict(fit.router_metrics),
        "teacher_compute": asdict(fit.teacher_metrics),
        "routed_compute": asdict(fit.routed_metrics),
    }


def _build_report(
    payload: Mapping[str, object],
    *,
    output: Path,
    scientific_digest: str,
) -> dict[str, object]:
    return {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "scientific_status": copy.deepcopy(
            dict(payload["scientific_status"])  # type: ignore[arg-type]
        ),
        "protocol": copy.deepcopy(
            dict(payload["protocol"])  # type: ignore[arg-type]
        ),
        "analysis": copy.deepcopy(
            dict(payload["analysis"])  # type: ignore[arg-type]
        ),
        "artifact": {
            "tensor_output": output.name,
            "contains_source_model_weights": False,
            "scientific_payload_sha256": scientific_digest,
        },
    }


def _scientific_values_equal(left: object, right: object) -> bool:
    """Compare deterministic scientific state without tensor truth ambiguity."""

    if isinstance(left, Tensor) or isinstance(right, Tensor):
        return (
            isinstance(left, Tensor)
            and isinstance(right, Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(
                _scientific_values_equal(left[key], right[key])
                for key in left
            )
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(
                _scientific_values_equal(left_value, right_value)
                for left_value, right_value in zip(  # type: ignore[arg-type]
                    left,
                    right,
                    strict=True,
                )
            )
        )
    return type(left) is type(right) and left == right


def _metrics_from_mapping(
    value: object,
    *,
    label: str,
) -> AssociativeRecallMetrics:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} metrics are invalid")
    expected = {
        "samples",
        "contexts",
        "hard_nll",
        "answer_accuracy",
        "paired_context_accuracy",
        "query_accuracies",
        "minimum_query_accuracy",
        "value_accuracies",
        "minimum_value_accuracy",
        "mean_correct_probability",
    }
    if set(value) != expected:
        raise ValueError(f"{label} metric fields are invalid")
    try:
        metrics = AssociativeRecallMetrics(**dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} metrics are invalid") from error
    if (
        type(metrics.samples) is not int
        or metrics.samples <= 0
        or type(metrics.contexts) is not int
        or metrics.contexts <= 0
    ):
        raise ValueError(f"{label} metric counts are invalid")
    for name in (
        "hard_nll",
        "answer_accuracy",
        "paired_context_accuracy",
        "minimum_query_accuracy",
        "minimum_value_accuracy",
        "mean_correct_probability",
    ):
        _finite(getattr(metrics, name), label=f"{label}.{name}")
    for name in ("query_accuracies", "value_accuracies"):
        values = getattr(metrics, name)
        if type(values) is not tuple or not values:
            raise ValueError(f"{label}.{name} is invalid")
        for index, item in enumerate(values):
            _finite(item, label=f"{label}.{name}[{index}]")
    return metrics


def _validate_stored_metric_record(
    value: object,
    baseline: AssociativeRecallMetrics,
    *,
    thresholds: Mapping[str, float],
    label: str,
) -> tuple[AssociativeRecallMetrics, bool]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} record is invalid")
    metrics = _metrics_from_mapping(
        value.get("metrics"),
        label=label,
    )
    expected = _metric_record(
        metrics,
        baseline,
        minimum_answer_accuracy=thresholds[
            "minimum_answer_accuracy"
        ],
        minimum_paired_accuracy=thresholds[
            "minimum_paired_context_accuracy"
        ],
        maximum_nll_increase=thresholds[
            "maximum_hard_nll_increase"
        ],
    )
    for key, expected_value in expected.items():
        if key not in value or not _scientific_values_equal(
            value[key],
            expected_value,
        ):
            raise ValueError(f"{label} derived gates are invalid")
    return metrics, bool(expected["passed"])


def run_associative_conditional_rank_experiment(
    *,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    layer_index: int = DEFAULT_LAYER_INDEX,
    basis_examples: int = DEFAULT_BASIS_EXAMPLES,
    router_examples: int = DEFAULT_ROUTER_EXAMPLES,
    calibration_b_examples: int = DEFAULT_CALIBRATION_B_EXAMPLES,
    route_budgets: Sequence[int] = DEFAULT_ROUTE_BUDGETS,
    router_ridge: float = DEFAULT_ROUTER_RIDGE,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    minimum_answer_accuracy: float = DEFAULT_MINIMUM_ANSWER_ACCURACY,
    minimum_paired_accuracy: float = DEFAULT_MINIMUM_PAIRED_ACCURACY,
    maximum_nll_increase: float = DEFAULT_MAXIMUM_NLL_INCREASE,
    minimum_state_nll_advantage: float = (
        DEFAULT_MINIMUM_STATE_NLL_ADVANTAGE
    ),
    output: Path | str | None = None,
) -> dict[str, object]:
    """Fit on grouped A partitions, gate on B, then evaluate validation once."""

    source_dir = Path(artifact_dir)
    checkpoint_path = source_dir / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"associative checkpoint does not exist: {checkpoint_path}"
        )
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError("layer_index must be nonnegative")
    for label, value in (
        ("basis_examples", basis_examples),
        ("router_examples", router_examples),
        ("calibration_b_examples", calibration_b_examples),
    ):
        if type(value) is not int or value <= 0 or value % 2:
            raise ValueError(f"{label} must be a positive even integer")
    if type(shuffle_seed) is not int:
        raise ValueError("shuffle_seed must be an integer")
    ridge = _finite(
        router_ridge,
        label="router_ridge",
        minimum=torch.finfo(torch.float64).eps,
    )
    answer_gate = _finite(
        minimum_answer_accuracy,
        label="minimum_answer_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    paired_gate = _finite(
        minimum_paired_accuracy,
        label="minimum_paired_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    nll_gate = _finite(
        maximum_nll_increase,
        label="maximum_nll_increase",
        minimum=0.0,
    )
    state_advantage = _finite(
        minimum_state_nll_advantage,
        label="minimum_state_nll_advantage",
        minimum=0.0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(checkpoint.get("model_config"), Mapping)
        or not isinstance(checkpoint.get("task_config"), Mapping)
        or not isinstance(checkpoint.get("model_state_dict"), Mapping)
    ):
        raise ValueError("associative checkpoint is invalid")
    model_config = TransformerConfig(**checkpoint["model_config"])
    task_config = AssociativeRecallTaskConfig(**checkpoint["task_config"])
    model = ToyTransformer(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    adapter = ToyTransformerAdapter(model)
    if layer_index >= len(adapter.segments):
        raise ValueError("layer_index is outside the checkpoint")
    segment = adapter.segments[layer_index]
    input_site = segment.input_site
    output_site = segment.output_site
    width = segment.output_width
    if segment.input_width != width:
        raise ValueError("conditional rank requires equal residual widths")
    budgets = _normalize_budgets(route_budgets, width=width)
    if width == 32 and budgets != DEFAULT_ROUTE_BUDGETS:
        raise ValueError(
            "the canonical width-32 exploratory run uses fixed budgets "
            f"{DEFAULT_ROUTE_BUDGETS}"
        )

    splits = build_associative_recall_splits(task_config)
    basis_start = 0
    basis_stop = basis_examples
    router_start = basis_stop
    router_stop = router_start + router_examples
    selection_start = router_stop
    selection_stop = selection_start + calibration_b_examples
    if selection_stop > splits.train.samples:
        raise ValueError("requested grouped partitions exceed training data")
    basis_split = _subset_split(
        splits.train,
        start=basis_start,
        stop=basis_stop,
        name="calibration_a_basis",
    )
    router_split = _subset_split(
        splits.train,
        start=router_start,
        stop=router_stop,
        name="calibration_a_router",
    )
    calibration_b = _subset_split(
        splits.train,
        start=selection_start,
        stop=selection_stop,
        name="calibration_b",
    )
    context_sets = [
        set(split.context_ids.tolist())
        for split in (basis_split, router_split, calibration_b)
    ]
    if any(
        left & right
        for index, left in enumerate(context_sets)
        for right in context_sets[index + 1 :]
    ):
        raise ValueError("conditional-rank partitions overlap by context")

    resolved_output = (
        default_associative_conditional_rank_output(
            layer_index=layer_index
        )
        if output is None
        else Path(output)
    )
    if resolved_output.suffix != ".pt":
        raise ValueError("output must use a .pt suffix")
    if resolved_output.exists() or resolved_output.with_suffix(
        ".json"
    ).exists():
        raise FileExistsError(
            "refusing to overwrite a conditional-rank experiment"
        )

    model_fingerprint_before = module_state_fingerprint(model)
    basis_collection = collect_activation_score_gradients(
        model,
        basis_split.input_ids,
        basis_split.targets,
        activation_names=(output_site,),
    )
    basis_samples = basis_collection.samples[output_site]
    basis = decompose_fisher_modes(basis_samples)
    router_collection = collect_activation_score_gradients(
        model,
        router_split.input_ids,
        router_split.targets,
        activation_names=(input_site, output_site),
    )
    input_samples = router_collection.samples[input_site]
    output_samples = router_collection.samples[output_site]
    model.requires_grad_(False)
    if module_state_fingerprint(model) != model_fingerprint_before:
        raise RuntimeError("Fisher collection mutated the source model")

    need_profiles = fisher_projection_damage_profiles(
        output_samples.activations.to(dtype=torch.float64),
        output_samples.score_gradients.to(dtype=torch.float64),
        center=basis.mean,
        basis_vectors=basis.vectors,
    )
    fit = fit_conditional_modal_routing(
        need_profiles,
        input_samples.activations.to(dtype=torch.float64),
        route_count=len(budgets),
        route_budgets=budgets,
        ridge=ridge,
        route_assignment="total_need_bins",
        profile_semantics=_PROFILE_SEMANTICS,
    )
    sequence_length = router_split.input_ids.shape[1]
    teacher_routes = fit.teacher_route_ids.reshape(-1)
    majority_routes, position_fit_rows = _majority_routes_by_position(
        teacher_routes,
        output_samples.locations,
        route_count=len(budgets),
        sequence_length=sequence_length,
    )
    matched_static_rank = int(
        math.ceil(fit.routed_metrics.average_active_modes)
    )

    thresholds = {
        "minimum_answer_accuracy": answer_gate,
        "minimum_paired_context_accuracy": paired_gate,
        "maximum_hard_nll_increase": nll_gate,
        "minimum_state_router_nll_advantage_over_position": (
            state_advantage
        ),
    }

    baseline_b = evaluate_associative_recall(model, calibration_b)
    learned_b, learned_execution_b = _evaluate_learned_oracle(
        model,
        calibration_b,
        layer_index=layer_index,
        basis=basis,
        plan=fit.plan,
    )
    calibration_b_inputs = _collect_site_activations(
        model,
        calibration_b,
        site=input_site,
    )
    learned_routes_b = fit.plan.route(calibration_b_inputs).cpu()
    position_routes_b = _fixed_position_grid(
        majority_routes,
        examples=calibration_b.samples,
    )
    shuffled_routes_b = _shuffle_route_grid(
        learned_routes_b,
        seed=shuffle_seed,
    )
    position_b = _evaluate_fixed_routes(
        model,
        calibration_b,
        input_site=input_site,
        output_site=output_site,
        basis=basis,
        table=fit.plan.mode_table,
        route_ids=position_routes_b,
    )
    shuffled_b = _evaluate_fixed_routes(
        model,
        calibration_b,
        input_site=input_site,
        output_site=output_site,
        basis=basis,
        table=fit.plan.mode_table,
        route_ids=shuffled_routes_b,
    )

    static_curve_b = []
    smallest_passing_static_rank: int | None = None
    static_metrics_by_rank: dict[int, AssociativeRecallMetrics] = {}
    for rank in range(width + 1):
        metrics = _evaluate_static(
            model,
            calibration_b,
            output_site=output_site,
            basis=basis,
            rank=rank,
        )
        static_metrics_by_rank[rank] = metrics
        record = _metric_record(
            metrics,
            baseline_b,
            minimum_answer_accuracy=answer_gate,
            minimum_paired_accuracy=paired_gate,
            maximum_nll_increase=nll_gate,
        )
        static_curve_b.append({"rank": rank, **record})
        if record["passed"] is True and smallest_passing_static_rank is None:
            smallest_passing_static_rank = rank

    learned_record_b = _metric_record(
        learned_b,
        baseline_b,
        minimum_answer_accuracy=answer_gate,
        minimum_paired_accuracy=paired_gate,
        maximum_nll_increase=nll_gate,
    )
    position_record_b = _metric_record(
        position_b,
        baseline_b,
        minimum_answer_accuracy=answer_gate,
        minimum_paired_accuracy=paired_gate,
        maximum_nll_increase=nll_gate,
    )
    shuffled_record_b = _metric_record(
        shuffled_b,
        baseline_b,
        minimum_answer_accuracy=answer_gate,
        minimum_paired_accuracy=paired_gate,
        maximum_nll_increase=nll_gate,
    )
    identity_record_b = static_curve_b[-1]
    calibration_b_passed = (
        learned_record_b["passed"] is True
        and identity_record_b["passed"] is True
        and smallest_passing_static_rank is not None
    )

    validation_evaluated = calibration_b_passed
    validation_analysis: dict[str, object]
    if validation_evaluated:
        baseline_validation = evaluate_associative_recall(
            model,
            splits.validation,
        )
        learned_validation, learned_execution_validation = (
            _evaluate_learned_oracle(
                model,
                splits.validation,
                layer_index=layer_index,
                basis=basis,
                plan=fit.plan,
            )
        )
        validation_inputs = _collect_site_activations(
            model,
            splits.validation,
            site=input_site,
        )
        learned_routes_validation = fit.plan.route(
            validation_inputs
        ).cpu()
        position_routes_validation = _fixed_position_grid(
            majority_routes,
            examples=splits.validation.samples,
        )
        shuffled_routes_validation = _shuffle_route_grid(
            learned_routes_validation,
            seed=shuffle_seed + 1,
        )
        position_validation = _evaluate_fixed_routes(
            model,
            splits.validation,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            table=fit.plan.mode_table,
            route_ids=position_routes_validation,
        )
        shuffled_validation = _evaluate_fixed_routes(
            model,
            splits.validation,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            table=fit.plan.mode_table,
            route_ids=shuffled_routes_validation,
        )
        matched_static_validation = _evaluate_static(
            model,
            splits.validation,
            output_site=output_site,
            basis=basis,
            rank=matched_static_rank,
        )
        assert smallest_passing_static_rank is not None
        viable_static_validation = _evaluate_static(
            model,
            splits.validation,
            output_site=output_site,
            basis=basis,
            rank=smallest_passing_static_rank,
        )
        identity_validation = _evaluate_static(
            model,
            splits.validation,
            output_site=output_site,
            basis=basis,
            rank=width,
        )
        learned_validation_record = _metric_record(
            learned_validation,
            baseline_validation,
            minimum_answer_accuracy=answer_gate,
            minimum_paired_accuracy=paired_gate,
            maximum_nll_increase=nll_gate,
        )
        position_validation_record = _metric_record(
            position_validation,
            baseline_validation,
            minimum_answer_accuracy=answer_gate,
            minimum_paired_accuracy=paired_gate,
            maximum_nll_increase=nll_gate,
        )
        learned_accounting_validation = _route_accounting(
            learned_routes_validation,
            table=fit.plan.mode_table,
            input_width=segment.input_width,
        )
        position_accounting_validation = _route_accounting(
            position_routes_validation,
            table=fit.plan.mode_table,
            input_width=0,
        )
        state_nll_advantage_value = (
            position_validation.hard_nll
            - learned_validation.hard_nll
        )
        state_conditioning_supported = (
            learned_validation_record["passed"] is True
            and state_nll_advantage_value >= state_advantage
        )
        validation_analysis = {
            "evaluated": True,
            "baseline": asdict(baseline_validation),
            "learned_input_router": {
                **learned_validation_record,
                "execution_audit": learned_execution_validation,
                "route_accounting": learned_accounting_validation,
                "route_counts_by_position": (
                    _route_counts_by_position(
                        learned_routes_validation,
                        route_count=len(budgets),
                    )
                ),
                "paired_counterfactual_route_changes": (
                    _paired_route_changes(
                        splits.validation,
                        learned_routes_validation,
                    )
                ),
            },
            "position_only_control": {
                **position_validation_record,
                "route_accounting": position_accounting_validation,
            },
            "histogram_matched_shuffled_control": _metric_record(
                shuffled_validation,
                baseline_validation,
                minimum_answer_accuracy=answer_gate,
                minimum_paired_accuracy=paired_gate,
                maximum_nll_increase=nll_gate,
            ),
            "matched_static_rank": {
                "rank": matched_static_rank,
                **_metric_record(
                    matched_static_validation,
                    baseline_validation,
                    minimum_answer_accuracy=answer_gate,
                    minimum_paired_accuracy=paired_gate,
                    maximum_nll_increase=nll_gate,
                ),
            },
            "smallest_b_passing_static_rank": {
                "rank": smallest_passing_static_rank,
                **_metric_record(
                    viable_static_validation,
                    baseline_validation,
                    minimum_answer_accuracy=answer_gate,
                    minimum_paired_accuracy=paired_gate,
                    maximum_nll_increase=nll_gate,
                ),
            },
            "full_rank_identity_control": {
                "rank": width,
                **_metric_record(
                    identity_validation,
                    baseline_validation,
                    minimum_answer_accuracy=answer_gate,
                    minimum_paired_accuracy=paired_gate,
                    maximum_nll_increase=nll_gate,
                ),
            },
            "state_router_nll_advantage_over_position": (
                state_nll_advantage_value
            ),
            "state_conditioning_supported": state_conditioning_supported,
            "position_conditioning_explains_observed_gain": (
                position_validation_record["passed"] is True
                and not state_conditioning_supported
            ),
        }
    else:
        learned_validation_record = None
        state_conditioning_supported = False
        validation_analysis = {
            "evaluated": False,
            "reason": (
                "calibration_b_gate_failed_validation_not_evaluated"
            ),
        }

    source_fingerprint_after = module_state_fingerprint(model)
    if source_fingerprint_after != model_fingerprint_before:
        raise RuntimeError("conditional experiment mutated source weights")

    conditional_representation_viable = (
        validation_evaluated
        and isinstance(learned_validation_record, Mapping)
        and learned_validation_record.get("passed") is True
    )
    analysis = {
        "fit": {
            **_fit_summary(fit),
            "position_only_fit": position_fit_rows,
            "basis_mean_loss": basis_collection.mean_loss,
            "router_mean_loss": router_collection.mean_loss,
        },
        "calibration_b": {
            "baseline": asdict(baseline_b),
            "learned_input_router": {
                **learned_record_b,
                "execution_audit": learned_execution_b,
                "route_accounting": _route_accounting(
                    learned_routes_b,
                    table=fit.plan.mode_table,
                    input_width=segment.input_width,
                ),
                "route_counts_by_position": _route_counts_by_position(
                    learned_routes_b,
                    route_count=len(budgets),
                ),
                "paired_counterfactual_route_changes": (
                    _paired_route_changes(
                        calibration_b,
                        learned_routes_b,
                    )
                ),
            },
            "position_only_control": {
                **position_record_b,
                "route_accounting": _route_accounting(
                    position_routes_b,
                    table=fit.plan.mode_table,
                    input_width=0,
                ),
            },
            "histogram_matched_shuffled_control": shuffled_record_b,
            "matched_static_rank": {
                "rank": matched_static_rank,
                **static_curve_b[matched_static_rank],
            },
            "smallest_passing_static_rank": (
                smallest_passing_static_rank
            ),
            "static_rank_curve": static_curve_b,
            "full_rank_identity_control": identity_record_b,
            "passed": calibration_b_passed,
        },
        "validation": validation_analysis,
    }
    protocol = {
        "scope": (
            "exploratory_conditional_modal_projection_oracle_on_"
            "previously_studied_toy_checkpoint"
        ),
        "layer_index": layer_index,
        "input_site": input_site,
        "output_site": output_site,
        "residual_width": width,
        "route_budgets": budgets,
        "route_assignment": _ROUTE_ASSIGNMENT,
        "profile_semantics": _PROFILE_SEMANTICS,
        "router": (
            "closed_form_pointwise_linear_ridge_current_block_input_only"
        ),
        "router_ridge": ridge,
        "basis_split": "calibration_a_basis_only",
        "router_split": "calibration_a_router_only",
        "selection_split": "calibration_b_only",
        "validation_policy": (
            "evaluate_once_only_after_learned_router_and_identity_pass_b"
        ),
        "test_policy": _TEST_POLICY,
        "thresholds": thresholds,
        "shuffle_seed": shuffle_seed,
        "schedule_selected_after_exploratory_analysis": True,
        "native_layer_executes_before_projection": True,
        "hard_route_grouped_projection_implemented": True,
        "standalone_graph_executor_fitted": False,
        "source_layer_compute_savings_claim": False,
        "parameter_reduction_claim": False,
        "latency_claim": False,
        "split_provenance": {
            "calibration_a_basis": _split_provenance(basis_split),
            "calibration_a_router": _split_provenance(router_split),
            "calibration_b": _split_provenance(calibration_b),
            "validation": _split_provenance(splits.validation),
            "test": _split_provenance(splits.test),
        },
    }
    scientific_status = {
        "exploratory_not_confirmatory": True,
        "source_model_frozen": True,
        "calibration_a_basis_fitted": True,
        "calibration_a_router_fitted": True,
        "calibration_b_evaluated": True,
        "calibration_b_passed": calibration_b_passed,
        "validation_evaluated": validation_evaluated,
        "test_evaluated": False,
        "conditional_representation_viable": (
            conditional_representation_viable
        ),
        "state_conditioning_supported": state_conditioning_supported,
        "position_conditioning_explains_observed_gain": (
            bool(
                validation_analysis.get(
                    "position_conditioning_explains_observed_gain",
                    False,
                )
            )
        ),
        "native_layer_executed": True,
        "standalone_executor": False,
        "source_layer_compute_savings_supported": False,
        "compression_claim": False,
        "parameter_reduction_claim": False,
        "latency_or_kernel_speed_claim": False,
    }
    payload = {
        "schema": _ARTIFACT_SCHEMA,
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "contains_model_weights": False,
        "source": {
            "checkpoint_file": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "model_config": copy.deepcopy(dict(checkpoint["model_config"])),
            "task_config": copy.deepcopy(dict(checkpoint["task_config"])),
            "model_state_fingerprint": model_fingerprint_before,
        },
        "scientific_status": scientific_status,
        "protocol": protocol,
        "basis": basis.state_dict(),
        "routing_plan": fit.plan.state_dict(),
        "fit_rows": {
            "need_profiles": need_profiles.detach().cpu(),
            "block_inputs": input_samples.activations.detach().cpu(),
            "locations": input_samples.locations.detach().cpu(),
            "teacher_route_ids": fit.teacher_route_ids.detach().cpu(),
        },
        "analysis": analysis,
    }
    scientific_digest = _scientific_payload_sha256(payload)
    report = _build_report(
        payload,
        output=resolved_output,
        scientific_digest=scientific_digest,
    )
    report_digest = _report_sha256(report)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **payload,
            "scientific_payload_sha256": scientific_digest,
            "report_sha256": report_digest,
        },
        resolved_output,
    )
    resolved_output.with_suffix(".json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def load_associative_conditional_rank_artifact(
    path: Path | str,
) -> dict[str, object]:
    """Authenticate a weights-only conditional-rank artifact and report."""

    artifact_path = Path(path)
    raw = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    required = {
        "schema",
        "format_version",
        "contains_model_weights",
        "source",
        "scientific_status",
        "protocol",
        "basis",
        "routing_plan",
        "fit_rows",
        "analysis",
        "scientific_payload_sha256",
        "report_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise ValueError("conditional-rank artifact fields are invalid")
    if (
        raw["schema"] != _ARTIFACT_SCHEMA
        or raw["format_version"] != _ARTIFACT_FORMAT_VERSION
        or raw["contains_model_weights"] is not False
        or not isinstance(raw["scientific_payload_sha256"], str)
        or not isinstance(raw["report_sha256"], str)
    ):
        raise ValueError("unsupported conditional-rank artifact")
    payload = {
        key: value
        for key, value in raw.items()
        if key not in {"scientific_payload_sha256", "report_sha256"}
    }
    digest = _scientific_payload_sha256(payload)
    if digest != raw["scientific_payload_sha256"]:
        raise ValueError("conditional-rank scientific digest mismatch")
    basis = FisherModeBasis.from_state_dict(raw["basis"])  # type: ignore[arg-type]
    plan = ConditionalModalRoutingPlan.from_state_dict(
        raw["routing_plan"]  # type: ignore[arg-type]
    )
    protocol = raw["protocol"]
    source = raw["source"]
    status = raw["scientific_status"]
    analysis = raw["analysis"]
    fit_rows = raw["fit_rows"]
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, source, status, analysis, fit_rows)
    ):
        raise ValueError("conditional-rank payload mappings are invalid")
    if (
        protocol.get("residual_width") != basis.width
        or tuple(protocol.get("route_budgets", ()))
        != plan.mode_table.route_budgets
        or protocol.get("route_assignment") != _ROUTE_ASSIGNMENT
        or protocol.get("profile_semantics") != _PROFILE_SEMANTICS
        or protocol.get("native_layer_executes_before_projection") is not True
        or protocol.get("standalone_graph_executor_fitted") is not False
        or protocol.get("source_layer_compute_savings_claim") is not False
        or status.get("native_layer_executed") is not True
        or status.get("standalone_executor") is not False
        or status.get("test_evaluated") is not False
        or status.get("source_layer_compute_savings_supported") is not False
    ):
        raise ValueError("conditional-rank protocol or status is invalid")
    need_profiles = fit_rows.get("need_profiles")
    block_inputs = fit_rows.get("block_inputs")
    teacher_routes = fit_rows.get("teacher_route_ids")
    locations = fit_rows.get("locations")
    if (
        not isinstance(need_profiles, Tensor)
        or not isinstance(block_inputs, Tensor)
        or not isinstance(teacher_routes, Tensor)
        or not isinstance(locations, Tensor)
        or not need_profiles.is_floating_point()
        or not block_inputs.is_floating_point()
        or teacher_routes.dtype not in (torch.int32, torch.int64)
        or locations.dtype not in (torch.int32, torch.int64)
        or need_profiles.ndim != 2
        or block_inputs.ndim != 2
        or need_profiles.shape[0] != block_inputs.shape[0]
        or teacher_routes.shape != (need_profiles.shape[0],)
        or locations.shape != (need_profiles.shape[0], 2)
        or need_profiles.shape[1] != basis.width
        or block_inputs.shape[1] != plan.router.input_features
        or not torch.isfinite(need_profiles).all()
        or bool((need_profiles < 0).any())
        or not torch.isfinite(block_inputs).all()
        or bool((locations < 0).any())
    ):
        raise ValueError("conditional-rank fit rows are invalid")
    budgets = _normalize_budgets(
        tuple(protocol.get("route_budgets", ())),
        width=basis.width,
    )
    ridge = _finite(
        protocol.get("router_ridge"),
        label="protocol.router_ridge",
        minimum=torch.finfo(torch.float64).eps,
    )
    recomputed_fit = fit_conditional_modal_routing(
        need_profiles,
        block_inputs,
        route_count=len(budgets),
        route_budgets=budgets,
        ridge=ridge,
        route_assignment="total_need_bins",
        profile_semantics=_PROFILE_SEMANTICS,
    )
    if not _scientific_values_equal(
        plan.state_dict(),
        recomputed_fit.plan.state_dict(),
    ):
        raise ValueError(
            "conditional-rank routing plan does not match fit evidence"
        )
    if not torch.equal(
        teacher_routes.reshape(-1).to(dtype=torch.int64),
        recomputed_fit.teacher_route_ids.reshape(-1),
    ):
        raise ValueError(
            "conditional-rank teacher routes do not match fit evidence"
        )
    expected_routes = recomputed_fit.plan.route(block_inputs)
    fit_analysis = analysis.get("fit")
    if not isinstance(fit_analysis, Mapping):
        raise ValueError("conditional-rank fit analysis is invalid")
    expected_fit_summary = _fit_summary(recomputed_fit)
    for key, expected_value in expected_fit_summary.items():
        if key not in fit_analysis or not _scientific_values_equal(
            fit_analysis[key],
            expected_value,
        ):
            raise ValueError("conditional-rank fit summary is invalid")
    logical_positions = locations[:, 1].to(dtype=torch.long)
    sequence_length = int(logical_positions.max().item()) + 1
    if not torch.equal(
        torch.unique(logical_positions, sorted=True),
        torch.arange(sequence_length),
    ):
        raise ValueError("conditional-rank fit positions are incomplete")
    _, expected_position_fit = _majority_routes_by_position(
        recomputed_fit.teacher_route_ids.reshape(-1),
        locations,
        route_count=len(budgets),
        sequence_length=sequence_length,
    )
    if not _scientific_values_equal(
        fit_analysis.get("position_only_fit"),
        expected_position_fit,
    ):
        raise ValueError("conditional-rank position fit is invalid")
    for name in ("basis_mean_loss", "router_mean_loss"):
        _finite(
            fit_analysis.get(name),
            label=f"analysis.fit.{name}",
            minimum=0.0,
        )
    routed_compute = fit_analysis.get("routed_compute")
    if not isinstance(routed_compute, Mapping):
        raise ValueError("conditional-rank route accounting is invalid")
    expected_route_counts = tuple(
        int(value)
        for value in torch.bincount(
            expected_routes.reshape(-1),
            minlength=plan.mode_table.routes,
        ).tolist()
    )
    if routed_compute.get("route_counts") != expected_route_counts:
        raise ValueError("conditional-rank route accounting is invalid")

    raw_thresholds = protocol.get("thresholds")
    if not isinstance(raw_thresholds, Mapping) or set(raw_thresholds) != {
        "minimum_answer_accuracy",
        "minimum_paired_context_accuracy",
        "maximum_hard_nll_increase",
        "minimum_state_router_nll_advantage_over_position",
    }:
        raise ValueError("conditional-rank thresholds are invalid")
    thresholds = {
        "minimum_answer_accuracy": _finite(
            raw_thresholds["minimum_answer_accuracy"],
            label="thresholds.minimum_answer_accuracy",
            minimum=0.0,
            maximum=1.0,
        ),
        "minimum_paired_context_accuracy": _finite(
            raw_thresholds["minimum_paired_context_accuracy"],
            label="thresholds.minimum_paired_context_accuracy",
            minimum=0.0,
            maximum=1.0,
        ),
        "maximum_hard_nll_increase": _finite(
            raw_thresholds["maximum_hard_nll_increase"],
            label="thresholds.maximum_hard_nll_increase",
            minimum=0.0,
        ),
        "minimum_state_router_nll_advantage_over_position": _finite(
            raw_thresholds[
                "minimum_state_router_nll_advantage_over_position"
            ],
            label=(
                "thresholds."
                "minimum_state_router_nll_advantage_over_position"
            ),
            minimum=0.0,
        ),
    }
    calibration_b_analysis = analysis.get("calibration_b")
    if not isinstance(calibration_b_analysis, Mapping):
        raise ValueError("conditional-rank calibration B analysis is invalid")
    baseline_b = _metrics_from_mapping(
        calibration_b_analysis.get("baseline"),
        label="calibration_b.baseline",
    )
    _, learned_b_passed = _validate_stored_metric_record(
        calibration_b_analysis.get("learned_input_router"),
        baseline_b,
        thresholds=thresholds,
        label="calibration_b.learned_input_router",
    )
    _validate_stored_metric_record(
        calibration_b_analysis.get("position_only_control"),
        baseline_b,
        thresholds=thresholds,
        label="calibration_b.position_only_control",
    )
    _validate_stored_metric_record(
        calibration_b_analysis.get(
            "histogram_matched_shuffled_control"
        ),
        baseline_b,
        thresholds=thresholds,
        label="calibration_b.histogram_matched_shuffled_control",
    )
    static_curve = calibration_b_analysis.get("static_rank_curve")
    if (
        not isinstance(static_curve, list)
        or len(static_curve) != basis.width + 1
    ):
        raise ValueError("conditional-rank static curve is invalid")
    static_passes: list[bool] = []
    for rank, record in enumerate(static_curve):
        if not isinstance(record, Mapping) or record.get("rank") != rank:
            raise ValueError("conditional-rank static curve rank is invalid")
        _, passed = _validate_stored_metric_record(
            record,
            baseline_b,
            thresholds=thresholds,
            label=f"calibration_b.static_rank_curve[{rank}]",
        )
        static_passes.append(passed)
    smallest_passing = next(
        (
            rank
            for rank, passed in enumerate(static_passes)
            if passed
        ),
        None,
    )
    matched_static_rank = int(
        math.ceil(recomputed_fit.routed_metrics.average_active_modes)
    )
    if (
        calibration_b_analysis.get("smallest_passing_static_rank")
        != smallest_passing
        or not _scientific_values_equal(
            calibration_b_analysis.get("matched_static_rank"),
            static_curve[matched_static_rank],
        )
        or not _scientific_values_equal(
            calibration_b_analysis.get("full_rank_identity_control"),
            static_curve[-1],
        )
    ):
        raise ValueError("conditional-rank static controls are invalid")
    calibration_b_passed = (
        learned_b_passed
        and static_passes[-1]
        and smallest_passing is not None
    )
    if calibration_b_analysis.get("passed") is not calibration_b_passed:
        raise ValueError("conditional-rank calibration B status is invalid")

    validation_analysis = analysis.get("validation")
    if not isinstance(validation_analysis, Mapping):
        raise ValueError("conditional-rank validation analysis is invalid")
    if validation_analysis.get("evaluated") is not calibration_b_passed:
        raise ValueError("conditional-rank validation gate is invalid")
    learned_validation_passed = False
    state_conditioning_supported = False
    position_explains = False
    if calibration_b_passed:
        baseline_validation = _metrics_from_mapping(
            validation_analysis.get("baseline"),
            label="validation.baseline",
        )
        learned_validation, learned_validation_passed = (
            _validate_stored_metric_record(
                validation_analysis.get("learned_input_router"),
                baseline_validation,
                thresholds=thresholds,
                label="validation.learned_input_router",
            )
        )
        position_validation, position_validation_passed = (
            _validate_stored_metric_record(
                validation_analysis.get("position_only_control"),
                baseline_validation,
                thresholds=thresholds,
                label="validation.position_only_control",
            )
        )
        _validate_stored_metric_record(
            validation_analysis.get(
                "histogram_matched_shuffled_control"
            ),
            baseline_validation,
            thresholds=thresholds,
            label="validation.histogram_matched_shuffled_control",
        )
        for field, expected_rank in (
            ("matched_static_rank", matched_static_rank),
            ("smallest_b_passing_static_rank", smallest_passing),
            ("full_rank_identity_control", basis.width),
        ):
            record = validation_analysis.get(field)
            if (
                not isinstance(record, Mapping)
                or record.get("rank") != expected_rank
            ):
                raise ValueError(
                    f"conditional-rank validation {field} is invalid"
                )
            _validate_stored_metric_record(
                record,
                baseline_validation,
                thresholds=thresholds,
                label=f"validation.{field}",
            )
        state_advantage = (
            position_validation.hard_nll - learned_validation.hard_nll
        )
        state_conditioning_supported = (
            learned_validation_passed
            and state_advantage
            >= thresholds[
                "minimum_state_router_nll_advantage_over_position"
            ]
        )
        position_explains = (
            position_validation_passed
            and not state_conditioning_supported
        )
        if (
            not _scientific_values_equal(
                validation_analysis.get(
                    "state_router_nll_advantage_over_position"
                ),
                state_advantage,
            )
            or validation_analysis.get("state_conditioning_supported")
            is not state_conditioning_supported
            or validation_analysis.get(
                "position_conditioning_explains_observed_gain"
            )
            is not position_explains
        ):
            raise ValueError(
                "conditional-rank state-conditioning conclusion is invalid"
            )
    elif set(validation_analysis) != {"evaluated", "reason"} or (
        validation_analysis.get("reason")
        != "calibration_b_gate_failed_validation_not_evaluated"
    ):
        raise ValueError("conditional-rank failed validation branch is invalid")

    expected_status = {
        "exploratory_not_confirmatory": True,
        "source_model_frozen": True,
        "calibration_a_basis_fitted": True,
        "calibration_a_router_fitted": True,
        "calibration_b_evaluated": True,
        "calibration_b_passed": calibration_b_passed,
        "validation_evaluated": calibration_b_passed,
        "test_evaluated": False,
        "conditional_representation_viable": (
            calibration_b_passed and learned_validation_passed
        ),
        "state_conditioning_supported": state_conditioning_supported,
        "position_conditioning_explains_observed_gain": position_explains,
        "native_layer_executed": True,
        "standalone_executor": False,
        "source_layer_compute_savings_supported": False,
        "compression_claim": False,
        "parameter_reduction_claim": False,
        "latency_or_kernel_speed_claim": False,
    }
    if not _scientific_values_equal(status, expected_status):
        raise ValueError("conditional-rank scientific status is invalid")
    source_path = Path(str(source.get("checkpoint_file", "")))
    if (
        not source_path.is_file()
        or source.get("checkpoint_sha256") != _file_sha256(source_path)
    ):
        raise ValueError("conditional-rank checkpoint binding is invalid")
    expected_report = _build_report(
        payload,
        output=artifact_path,
        scientific_digest=digest,
    )
    report = json.loads(
        artifact_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(report, Mapping)
        or _report_sha256(report) != raw["report_sha256"]
        or report
        != json.loads(
            json.dumps(
                expected_report,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    ):
        raise ValueError("conditional-rank report does not match payload")
    return {
        "basis": basis,
        "routing_plan": plan,
        "scientific_status": copy.deepcopy(dict(status)),
        "protocol": copy.deepcopy(dict(protocol)),
        "analysis": copy.deepcopy(dict(analysis)),
        "metadata": {
            "scientific_payload_sha256": digest,
            "report_sha256": raw["report_sha256"],
            "source": copy.deepcopy(dict(source)),
        },
        "report": copy.deepcopy(dict(report)),
    }


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "route budgets must be comma-separated integers"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and evaluate a causal conditional Fisher-rank projection "
            "oracle on the associative-recall checkpoint."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=DEFAULT_LAYER_INDEX,
    )
    parser.add_argument(
        "--basis-examples",
        type=int,
        default=DEFAULT_BASIS_EXAMPLES,
    )
    parser.add_argument(
        "--router-examples",
        type=int,
        default=DEFAULT_ROUTER_EXAMPLES,
    )
    parser.add_argument(
        "--calibration-b-examples",
        type=int,
        default=DEFAULT_CALIBRATION_B_EXAMPLES,
    )
    parser.add_argument(
        "--route-budgets",
        default=",".join(
            str(value) for value in DEFAULT_ROUTE_BUDGETS
        ),
    )
    parser.add_argument(
        "--router-ridge",
        type=float,
        default=DEFAULT_ROUTER_RIDGE,
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_associative_conditional_rank_experiment(
        artifact_dir=arguments.artifact_dir,
        layer_index=arguments.layer_index,
        basis_examples=arguments.basis_examples,
        router_examples=arguments.router_examples,
        calibration_b_examples=arguments.calibration_b_examples,
        route_budgets=_parse_budgets(arguments.route_budgets),
        router_ridge=arguments.router_ridge,
        shuffle_seed=arguments.shuffle_seed,
        output=arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ROUTE_BUDGETS",
    "default_associative_conditional_rank_output",
    "load_associative_conditional_rank_artifact",
    "run_associative_conditional_rank_experiment",
]
