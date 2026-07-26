"""Full-span, query-sparse conditional Fisher routing experiment.

This experiment targets every transformer block in the variable associative
model at once, from ``layer.0.input`` to ``layer.2.output``.  The task has one
supervised answer row per sequence.  Fisher fitting, route labels, modal
projection, and static comparisons therefore use only that demanded output
row, while every causal prefix row remains available to the router as a key.

The experiment is intentionally fail-closed and gate-only.  Route schedules are selected on
``policy_a``.  Causal-router hyperparameters are selected on an internal
context holdout within ``router_a``.  ``calibration_b`` is evaluated only after
the schedule, router, ablation, and metadata-control choices are frozen.  If
its joint fidelity, content, and compute gate fails, validation and test stay
untouched.  A pass marks a future graph-fitting command eligible; this command
never fits a source-independent graph itself.

The native-output projection is a representation oracle: it still runs the
three source blocks.  Complete graph accounting is reported only as a
hypothetical capacity envelope and is explicitly not an achieved compression
result.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .adapters import module_state_fingerprint
from .causal_routing import (
    CausalExponentialStateRouter,
    fit_causal_exponential_state_router,
)
from .conditional_controls import (
    HierarchicalCategoricalRouteControl,
    fit_hierarchical_categorical_route_control,
    route_histograms_by_stratum,
    stratified_shuffle_routes,
)
from .conditional_routing import (
    ConditionalModeTable,
    PointwiseCausalRouter,
    TotalNeedRouteTeacher,
    build_conditional_mode_table,
    fit_pointwise_causal_router,
    fit_total_need_route_teacher,
    linear_codec_fisher_damage_profiles,
    partition_fisher_need_profiles_by_teacher,
)
from .full_span_accounting import (
    conditional_causal_graph_accounting,
    native_transformer_span_accounting,
)
from .modes import (
    ActivationGradientSamples,
    FisherModeBasis,
    decompose_fisher_modes,
)
from .model import ToyTransformer
from .variable_associative import (
    VariableAssociativeRecallSplit,
    subset_variable_associative_recall_split,
)
from .variable_associative_training import (
    DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    load_variable_associative_checkpoint,
    variable_associative_answer_logits,
    variable_associative_metrics_from_logits,
)
from .variable_conditional_experiment import (
    _behavior_record,
    _bootstrap_advantage,
    _collect,
    _file_sha256,
    _jsonable,
    _metadata_features,
)


DEFAULT_OUTPUT = Path(
    ".local-runs/variable-associative/full-span-conditional.pt"
)
DEFAULT_CONTEXTS_PER_ROLE = 24
DEFAULT_ROUTER_FIT_CONTEXTS = 16
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 91_302
DEFAULT_SHUFFLE_SEED = 91_303
DEFAULT_RANDOM_SEED = 91_301
DEFAULT_MINIMUM_NLL_ADVANTAGE = 0.005
DEFAULT_MAXIMUM_RELATIVE_STATIC_WORK = 0.90
DEFAULT_INPUT_SITE = "layer.0.input"
DEFAULT_OUTPUT_SITE = "layer.2.output"

ROLE_NAMES = (
    "basis_a",
    "mask_a",
    "policy_a",
    "router_a",
    "calibration_b",
)

# Frozen before calibration B.  Positive query budgets keep output-demand
# sparsity separate from mode selection: non-query outputs are skipped by the
# query mask, not represented as a synthetic rank-zero Fisher class.
ROUTE_SCHEDULES: tuple[dict[str, object], ...] = (
    {
        "name": "b4_8_16_32_q25_50_75",
        "budgets": (4, 8, 16, 32),
        "quantiles": (0.25, 0.50, 0.75),
    },
    {
        "name": "b4_8_16_32_q50_80_95",
        "budgets": (4, 8, 16, 32),
        "quantiles": (0.50, 0.80, 0.95),
    },
    {
        "name": "b8_12_20_32_q50_80_95",
        "budgets": (8, 12, 20, 32),
        "quantiles": (0.50, 0.80, 0.95),
    },
    {
        "name": "b8_16_24_32_q50_80_95",
        "budgets": (8, 16, 24, 32),
        "quantiles": (0.50, 0.80, 0.95),
    },
    {
        "name": "b12_16_24_32_q50_80_95",
        "budgets": (12, 16, 24, 32),
        "quantiles": (0.50, 0.80, 0.95),
    },
    {
        "name": "b16_20_28_32_q50_80_95",
        "budgets": (16, 20, 28, 32),
        "quantiles": (0.50, 0.80, 0.95),
    },
    {
        "name": "b16_24_28_32_q50_80_95",
        "budgets": (16, 24, 28, 32),
        "quantiles": (0.50, 0.80, 0.95),
    },
)

DECAY_GRIDS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("c1_sum", (0.0,)),
    ("c4", (0.0, 0.125, 0.5, 2.0)),
    (
        "c8",
        (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0),
    ),
    (
        "c12",
        (
            0.0,
            0.015625,
            0.03125,
            0.0625,
            0.125,
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
        ),
    ),
)
ROUTER_RIDGES = (1e-3, 1e-1)
ROUTER_CLASS_BALANCE_POWERS = (0.0, 1.0)
ROUTER_DECISIONS = (
    "argmax",
    "posterior_q70",
    "posterior_q85",
    "posterior_q95",
)

# These are accounting envelopes, not candidates fitted by this experiment.
HYPOTHETICAL_GRAPH_CAPACITIES = (
    (1, 32),
    (4, 32),
    (8, 32),
)

_SCHEMA = "fisher_graph.variable_full_span_conditional"
_FORMAT_VERSION = 1


def _context_role_splits(
    source: VariableAssociativeRecallSplit,
    *,
    contexts_per_role: int,
) -> dict[str, VariableAssociativeRecallSplit]:
    if type(contexts_per_role) is not int or contexts_per_role <= 0:
        raise ValueError("contexts_per_role must be positive")
    needed = contexts_per_role * len(ROLE_NAMES)
    if needed > source.contexts:
        raise ValueError("source split has too few contexts for compiler roles")
    result: dict[str, VariableAssociativeRecallSplit] = {}
    for role_index, name in enumerate(ROLE_NAMES):
        start = role_index * contexts_per_role
        result[name] = subset_variable_associative_recall_split(
            source,
            context_rows=torch.arange(start, start + contexts_per_role),
            name=name,
        )
    hashes = [
        set(result[name].semantic_context_hashes) for name in ROLE_NAMES
    ]
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if not hashes[left].isdisjoint(hashes[right]):
                raise RuntimeError("compiler role semantic contexts overlap")
    return result


def _assert_evaluation_contexts_are_disjoint(
    roles: Mapping[str, VariableAssociativeRecallSplit],
    validation: VariableAssociativeRecallSplit,
    test: VariableAssociativeRecallSplit,
) -> None:
    role_hashes = set().union(
        *(
            set(roles[name].semantic_context_hashes)
            for name in ROLE_NAMES
        )
    )
    validation_hashes = set(validation.semantic_context_hashes)
    test_hashes = set(test.semantic_context_hashes)
    if not role_hashes.isdisjoint(validation_hashes):
        raise RuntimeError("compiler roles overlap validation contexts")
    if not role_hashes.isdisjoint(test_hashes):
        raise RuntimeError("compiler roles overlap test contexts")
    if not validation_hashes.isdisjoint(test_hashes):
        raise RuntimeError("validation and test contexts overlap")


def _query_mask(split: VariableAssociativeRecallSplit) -> Tensor:
    positions = torch.arange(split.maximum_sequence_length).unsqueeze(0)
    return positions.eq(split.supervised_positions.unsqueeze(1))


def _key_mask(split: VariableAssociativeRecallSplit) -> Tensor:
    positions = torch.arange(split.maximum_sequence_length).unsqueeze(0)
    return split.attention_mask & positions.le(
        split.supervised_positions.unsqueeze(1)
    )


def _logical_positions(split: VariableAssociativeRecallSplit) -> Tensor:
    return torch.arange(
        split.maximum_sequence_length,
        dtype=torch.int64,
    ).unsqueeze(0).expand(split.samples, -1)


def _answer_rows(
    values: Tensor,
    split: VariableAssociativeRecallSplit,
) -> Tensor:
    if values.ndim < 2 or values.shape[0] != split.samples:
        raise ValueError("values do not align with split examples")
    rows = torch.arange(split.samples, device=values.device)
    return values[
        rows,
        split.supervised_positions.to(device=values.device),
    ]


def _assert_sample_alignment(
    split: VariableAssociativeRecallSplit,
    *samples: ActivationGradientSamples,
) -> None:
    metadata = split.valid_token_metadata()
    expected_locations = torch.stack(
        (metadata.example_indices, metadata.logical_positions),
        dim=1,
    )
    for sample in samples:
        if (
            sample.sequences != split.samples
            or not torch.equal(sample.locations, expected_locations)
        ):
            raise RuntimeError(
                "activation-gradient rows do not match variable-token metadata"
            )


def _collect_grids(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    names: tuple[str, ...],
) -> dict[str, tuple[Tensor, Tensor]]:
    samples = _collect(
        model,
        split,
        activation_names=set(names),
    )
    _assert_sample_alignment(
        split,
        *(samples[name] for name in names),
    )
    selected = split.valid_token_metadata().selected_flat_indices
    result: dict[str, tuple[Tensor, Tensor]] = {}
    for name in names:
        sample = samples[name]
        activations = torch.zeros(
            split.samples,
            split.maximum_sequence_length,
            sample.width,
            dtype=sample.activations.dtype,
        )
        gradients = torch.zeros_like(activations)
        activations.reshape(-1, sample.width).index_copy_(
            0,
            selected,
            sample.activations,
        )
        gradients.reshape(-1, sample.width).index_copy_(
            0,
            selected,
            sample.score_gradients,
        )
        result[name] = (activations, gradients)
    return result


def _basis_samples(
    split: VariableAssociativeRecallSplit,
    activation_grid: Tensor,
    gradient_grid: Tensor,
    *,
    activation_name: str,
) -> ActivationGradientSamples:
    positions = split.supervised_positions.to(torch.int64)
    locations = torch.stack(
        (torch.arange(split.samples, dtype=torch.int64), positions),
        dim=1,
    )
    return ActivationGradientSamples(
        name=activation_name,
        activations=_answer_rows(activation_grid, split),
        score_gradients=_answer_rows(gradient_grid, split),
        locations=locations,
        sequences=split.samples,
        sequence_ids=split.example_ids,
    )


def _answer_need(
    split: VariableAssociativeRecallSplit,
    collected: Mapping[str, tuple[Tensor, Tensor]],
    basis: FisherModeBasis,
    *,
    input_site: str,
    output_site: str,
) -> Tensor:
    incoming = _answer_rows(collected[input_site][0], split)
    outgoing = _answer_rows(collected[output_site][0], split)
    gradients = _answer_rows(collected[output_site][1], split)
    return linear_codec_fisher_damage_profiles(
        outgoing - incoming,
        gradients,
        encoder=basis.vectors,
        decoder=basis.vectors,
    )


@torch.no_grad()
def _projected_answer_logits(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    basis: FisherModeBasis,
    *,
    input_site: str,
    output_site: str,
    table: ConditionalModeTable | None = None,
    routes: Tensor | None = None,
    static_mask: Tensor | None = None,
    batch_size: int = 512,
) -> Tensor:
    if (static_mask is None) == (routes is None):
        raise ValueError("supply exactly one of static_mask or routes")
    if routes is not None:
        if table is None:
            raise ValueError("conditional projection requires a mode table")
        if routes.shape != (split.samples,):
            raise ValueError("routes must contain one answer route per example")
    if static_mask is not None and (
        static_mask.dtype is not torch.bool
        or static_mask.shape != (basis.width,)
    ):
        raise ValueError("static_mask must match the Fisher width")

    cursor = 0
    incoming: Tensor | None = None

    def capture_input(values: Tensor) -> Tensor:
        nonlocal incoming
        incoming = values
        return values

    def project_output(values: Tensor) -> Tensor:
        nonlocal incoming, cursor
        if incoming is None:
            raise RuntimeError("output projection ran before input capture")
        batch = values.shape[0]
        rows = torch.arange(batch, device=values.device)
        positions = split.supervised_positions[cursor : cursor + batch].to(
            device=values.device
        )
        vectors = basis.vectors.to(
            device=values.device,
            dtype=torch.float64,
        )
        incoming_query = incoming[rows, positions].to(torch.float64)
        outgoing_query = values[rows, positions].to(torch.float64)
        coordinates = (outgoing_query - incoming_query) @ vectors
        if static_mask is not None:
            masks = static_mask.to(
                device=values.device,
                dtype=torch.float64,
            ).expand(batch, -1)
        else:
            assert routes is not None and table is not None
            live_routes = routes[cursor : cursor + batch].to(
                device=values.device,
                dtype=torch.int64,
            )
            masks = table.mode_masks.to(device=values.device)[
                live_routes
            ].to(torch.float64)
        query_output = incoming_query + (coordinates * masks) @ vectors.T
        projected = values.clone()
        projected[rows, positions] = query_output.to(dtype=values.dtype)
        incoming = None
        cursor += batch
        return projected

    logits = variable_associative_answer_logits(
        model,
        split,
        batch_size=batch_size,
        activation_interventions={
            input_site: capture_input,
            output_site: project_output,
        },
    )
    if cursor != split.samples or incoming is not None:
        raise RuntimeError("full-span projection did not consume the split")
    return logits


def _compact_behavior(record: Mapping[str, object]) -> dict[str, object]:
    metrics = record["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("behavior metrics must be a mapping")
    minimum_stratum_accuracy = min(
        float(metrics["minimum_query_accuracy"]),
        float(metrics["minimum_pair_order_accuracy"]),
        float(metrics["minimum_layout_accuracy"]),
        float(metrics["minimum_length_accuracy"]),
    )
    return {
        "hard_nll": float(metrics["hard_nll"]),
        "delta_nll": float(record["delta_nll"]),
        "answer_accuracy": float(metrics["answer_accuracy"]),
        "paired_context_accuracy": float(
            metrics["paired_context_accuracy"]
        ),
        "minimum_stratum_accuracy": minimum_stratum_accuracy,
        "top1_agreement": float(record["top1_agreement"]),
        "native_teacher_kl": float(record["native_teacher_kl"]),
        "p90_absolute_delta_nll": float(
            record["p90_absolute_delta_nll"]
        ),
        "gates": copy.deepcopy(record["gates"]),
        "passed": record["passed"] is True,
    }


def _route_summary(
    routes: Tensor,
    table: ConditionalModeTable,
    *,
    width: int,
) -> dict[str, object]:
    if routes.ndim != 1 or routes.numel() == 0:
        raise ValueError("routes must be a nonempty answer-row vector")
    selected = routes.to(device="cpu", dtype=torch.int64)
    counts = torch.bincount(selected, minlength=table.routes)
    budgets = torch.tensor(table.route_budgets, dtype=torch.int64)
    active = budgets[selected]
    return {
        "route_counts": tuple(int(value) for value in counts.tolist()),
        "average_active_modes": float(
            active.to(torch.float64).mean().item()
        ),
        "full_width_fallback_rate": float(
            active.eq(width).to(torch.float64).mean().item()
        ),
        "active_mode_applications": int(active.sum().item()),
        "ideal_selective_projection_macs": int(
            2 * width * active.sum().item()
        ),
    }


def _decision(logits: Tensor, name: str) -> Tensor:
    if name == "argmax":
        return logits.argmax(dim=-1)
    if not name.startswith("posterior_q"):
        raise ValueError("unsupported router decision")
    try:
        quantile = int(name.removeprefix("posterior_q")) / 100.0
    except ValueError as error:
        raise ValueError("posterior decision must end in an integer") from error
    if not 0.0 < quantile < 1.0:
        raise ValueError("posterior decision quantile must lie in (0, 1)")
    cumulative = logits.softmax(dim=-1).cumsum(dim=-1)
    return (cumulative < quantile).sum(dim=-1).clamp_max(
        logits.shape[-1] - 1
    )


def _class_weights(
    labels: Tensor,
    *,
    route_count: int,
    power: float,
) -> Tensor:
    counts = torch.bincount(labels, minlength=route_count).to(torch.float64)
    if (counts == 0).any():
        raise ValueError("router labels leave an empty route")
    return (counts.mean() / counts).pow(power)[labels]


def _classifier_diagnostics(
    predicted: Tensor,
    targets: Tensor,
    *,
    route_count: int,
) -> dict[str, object]:
    confusion = torch.zeros(route_count, route_count, dtype=torch.int64)
    for target, prediction in zip(targets, predicted, strict=True):
        confusion[int(target), int(prediction)] += 1
    counts = torch.bincount(targets, minlength=route_count)
    recalls = confusion.diagonal().to(torch.float64) / counts.clamp_min(1)
    return {
        "accuracy": float(
            predicted.eq(targets).to(torch.float64).mean().item()
        ),
        "macro_recall": float(recalls.mean().item()),
        "target_counts": tuple(int(value) for value in counts.tolist()),
        "predicted_counts": tuple(
            int(value)
            for value in torch.bincount(
                predicted,
                minlength=route_count,
            ).tolist()
        ),
        "confusion_matrix": tuple(
            tuple(int(value) for value in row)
            for row in confusion.tolist()
        ),
    }


def _select_router_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not candidates:
        raise ValueError("router candidate set cannot be empty")
    passing = [
        candidate
        for candidate in candidates
        if isinstance(candidate["behavior"], Mapping)
        and candidate["behavior"]["passed"] is True
    ]
    if passing:
        return min(
            passing,
            key=lambda candidate: (
                int(
                    candidate[
                        "router_plus_ideal_selective_projection_macs"
                    ]
                ),
                float(candidate["behavior"]["hard_nll"]),  # type: ignore[index]
                str(candidate["name"]),
            ),
        )
    return min(
        candidates,
        key=lambda candidate: (
            -sum(
                bool(value)
                for value in candidate["behavior"]["gates"].values()  # type: ignore[index,union-attr]
            ),
            float(candidate["behavior"]["hard_nll"]),  # type: ignore[index]
            int(
                candidate[
                    "router_plus_ideal_selective_projection_macs"
                ]
            ),
            str(candidate["name"]),
        ),
    )


def _fit_metadata_controls(
    routes: Tensor,
    split: VariableAssociativeRecallSplit,
    *,
    route_count: int,
) -> dict[str, HierarchicalCategoricalRouteControl]:
    query = _query_mask(split)
    route_grid = torch.zeros_like(query, dtype=torch.int64)
    route_grid[
        torch.arange(split.samples),
        split.supervised_positions,
    ] = routes
    features = _metadata_features(split)
    specifications = {
        "position_length": (
            ("position", "length"),
            ("position",),
            ("length",),
        ),
        "token_id_position_length": (
            ("token_id", "position", "length"),
            ("token_id", "position"),
            ("position", "length"),
            ("token_id",),
            ("position",),
            ("length",),
        ),
        "token_role_position_length": (
            ("token_role", "position", "length"),
            ("token_role", "position"),
            ("position", "length"),
            ("token_role",),
            ("position",),
            ("length",),
        ),
    }
    return {
        name: fit_hierarchical_categorical_route_control(
            route_grid,
            features,
            valid_mask=query,
            route_count=route_count,
            levels=levels,
        )
        for name, levels in specifications.items()
    }


def _control_routes(
    control: HierarchicalCategoricalRouteControl,
    split: VariableAssociativeRecallSplit,
) -> Tensor:
    grid = control.predict(
        _metadata_features(split),
        valid_mask=_query_mask(split),
    )
    return _answer_rows(grid, split)


def _hypothetical_graph_envelopes(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    table: ConditionalModeTable,
    learned_routes: Tensor,
    static_rank: int,
) -> tuple[dict[str, object], ...]:
    key_mask = _key_mask(split)
    positions = _logical_positions(split)
    prefix_rows = int(key_mask.sum().item())
    query_rows = split.samples
    prefix_pairs = int(
        (
            key_mask.unsqueeze(2)
            & key_mask.unsqueeze(1)
            & positions.unsqueeze(2).ge(positions.unsqueeze(1))
        ).sum().item()
    )
    active = torch.tensor(table.route_budgets, dtype=torch.int64)[
        learned_routes
    ]
    final_head_macs = (
        query_rows * model.config.d_model * model.config.vocab_size
    )
    padded_rows = split.samples * split.maximum_sequence_length
    padded_allowed_pairs = (
        split.samples
        * split.maximum_sequence_length
        * (split.maximum_sequence_length + 1)
        // 2
    )
    native = native_transformer_span_accounting(
        valid_rows=prefix_rows,
        causal_pairs=prefix_pairs,
        width=model.config.d_model,
        feed_forward_width=model.config.d_ff,
        layer_count=model.config.n_layers,
        padded_rows=padded_rows,
        padded_causal_pairs=padded_allowed_pairs,
    )
    native_complete = native.logical_total_macs + final_head_macs
    records: list[dict[str, object]] = []
    for channels, hidden in HYPOTHETICAL_GRAPH_CAPACITIES:
        conditional = conditional_causal_graph_accounting(
            key_rows=prefix_rows,
            query_rows=query_rows,
            width=model.config.d_model,
            state_channels=channels,
            hidden_width=hidden,
            routes=table.routes,
            active_rank_applications=int(active.sum().item()),
            padded_key_rows=padded_rows,
            padded_query_rows=padded_rows,
            logical_causal_pairs=prefix_pairs,
            padded_causal_pairs=padded_allowed_pairs,
        )
        static = conditional_causal_graph_accounting(
            key_rows=prefix_rows,
            query_rows=query_rows,
            width=model.config.d_model,
            state_channels=channels,
            hidden_width=hidden,
            routes=table.routes,
            active_rank_applications=query_rows * static_rank,
            include_router=False,
            stored_output_modes=static_rank,
            padded_key_rows=padded_rows,
            padded_query_rows=padded_rows,
            logical_causal_pairs=prefix_pairs,
            padded_causal_pairs=padded_allowed_pairs,
        )
        conditional_complete = conditional.ideal_total_macs + final_head_macs
        static_complete = static.ideal_total_macs + final_head_macs
        records.append(
            {
                "state_channels": channels,
                "hidden_width": hidden,
                "conditional": asdict(conditional),
                "same_trunk_static": asdict(static),
                "logical_prefix_native_complete_macs": native_complete,
                "conditional_complete_macs": conditional_complete,
                "same_trunk_static_complete_macs": static_complete,
                "conditional_to_native_ratio": (
                    conditional_complete / native_complete
                ),
                "conditional_to_same_trunk_static_ratio": (
                    conditional_complete / static_complete
                ),
                "runtime_scalars_to_source_span_parameters": (
                    conditional.total_runtime_scalar_count
                    / native.total_parameter_count
                ),
                "fitted": False,
                "behavior_validated": False,
            }
        )
    return tuple(records)


def _required_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_full_span_result_contract(
    payload: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    if (
        payload.get("schema") != _SCHEMA
        or payload.get("format_version") != _FORMAT_VERSION
    ):
        raise ValueError("unsupported full-span artifact schema")
    if payload.get("contains_source_model_weights") is not False:
        raise ValueError("full-span artifact must exclude source weights")
    if payload.get("contains_compiled_executor_weights") is not False:
        raise ValueError("gate-only artifact must exclude executor weights")

    payload_status = _required_mapping(
        payload.get("scientific_status"),
        name="scientific_status",
    )
    payload_protocol = _required_mapping(
        payload.get("protocol"),
        name="protocol",
    )
    payload_analysis = _required_mapping(
        payload.get("analysis"),
        name="analysis",
    )
    if report.get("scientific_status") != _jsonable(payload_status):
        raise ValueError("JSON and tensor scientific status disagree")
    if report.get("protocol") != _jsonable(payload_protocol):
        raise ValueError("JSON and tensor protocols disagree")
    if report.get("analysis") != _jsonable(payload_analysis):
        raise ValueError("JSON and tensor analyses disagree")

    role_hashes_raw = _required_mapping(
        payload_protocol.get("role_context_hashes"),
        name="role_context_hashes",
    )
    role_hashes = set().union(
        *(
            set(role_hashes_raw[name])  # type: ignore[arg-type]
            for name in ROLE_NAMES
        )
    )
    validation_hashes = set(
        payload_protocol.get("validation_context_hashes", ())
    )
    test_hashes = set(payload_protocol.get("test_context_hashes", ()))
    if not role_hashes.isdisjoint(validation_hashes):
        raise ValueError("artifact roles overlap validation contexts")
    if not role_hashes.isdisjoint(test_hashes):
        raise ValueError("artifact roles overlap test contexts")
    if not validation_hashes.isdisjoint(test_hashes):
        raise ValueError("artifact validation and test contexts overlap")

    calibration = _required_mapping(
        payload_analysis.get("calibration_b"),
        name="calibration_b",
    )
    joint_gate = _required_mapping(
        calibration.get("joint_gate"),
        name="calibration_b.joint_gate",
    )
    joint_passed = joint_gate.get("passed") is True
    if payload_status.get("calibration_b_passed") is not joint_passed:
        raise ValueError("calibration status disagrees with joint gate")
    if payload_status.get("model_level_eligible") is not joint_passed:
        raise ValueError("eligibility status disagrees with joint gate")
    if payload_status.get("model_level_graph_fitted") is not False:
        raise ValueError("gate-only artifact cannot contain a fitted graph")
    if payload_status.get("gate_only_command_never_fits_graph") is not True:
        raise ValueError("artifact must declare the gate-only fit boundary")
    if payload_status.get("validation_evaluated") is not False:
        raise ValueError("full-span gate must leave validation untouched")
    if payload_status.get("test_evaluated") is not False:
        raise ValueError("full-span gate must leave test untouched")

    compute = _required_mapping(
        payload_analysis.get("compute"),
        name="compute",
    )
    logical_native = int(
        compute["logical_valid_native_complete_macs"]
    )
    logical_oracle = int(
        compute["representation_oracle_logical_ideal_complete_macs"]
    )
    ratio = float(
        compute["representation_oracle_logical_ideal_to_native_ratio"]
    )
    if not math.isclose(
        ratio,
        logical_oracle / logical_native,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError("representation-oracle ratio is inconsistent")

    envelopes = compute.get("hypothetical_graph_envelopes")
    if not isinstance(envelopes, (tuple, list)):
        raise ValueError("graph envelopes must be a sequence")
    static_rank = int(
        _required_mapping(
            calibration.get("smallest_passing_static"),
            name="smallest_passing_static",
        )["rank"]
    )
    for index, raw_record in enumerate(envelopes):
        record = _required_mapping(
            raw_record,
            name=f"hypothetical_graph_envelopes[{index}]",
        )
        conditional = _required_mapping(
            record.get("conditional"),
            name=f"conditional envelope {index}",
        )
        static = _required_mapping(
            record.get("same_trunk_static"),
            name=f"static envelope {index}",
        )
        for name in (
            "padded_key_rows",
            "padded_query_rows",
            "padded_causal_pairs",
        ):
            if conditional.get(name) != static.get(name):
                raise ValueError(
                    "conditional/static padding provenance disagrees"
                )
        if static.get("include_router") is not False:
            raise ValueError("static envelope must not include a router")
        if int(static["stored_output_modes"]) != static_rank:
            raise ValueError("static envelope is not structurally rank-pruned")
        conditional_complete = int(record["conditional_complete_macs"])
        static_complete = int(
            record["same_trunk_static_complete_macs"]
        )
        if not math.isclose(
            float(record["conditional_to_same_trunk_static_ratio"]),
            conditional_complete / static_complete,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("conditional/static envelope ratio is inconsistent")


def verify_variable_full_span_artifacts(
    artifact: Path,
    *,
    report: Path | None = None,
) -> dict[str, object]:
    """Strict-load and authenticate a saved gate artifact/report pair."""

    artifact_path = Path(artifact)
    report_path = (
        artifact_path.with_suffix(".json")
        if report is None
        else Path(report)
    )
    raw_payload = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    payload = _required_mapping(raw_payload, name="artifact")
    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    report_mapping = _required_mapping(raw_report, name="JSON report")

    FisherModeBasis.from_state_dict(
        _required_mapping(payload.get("basis"), name="basis")
    )
    TotalNeedRouteTeacher.from_state_dict(
        _required_mapping(payload.get("teacher"), name="teacher")
    )
    ConditionalModeTable.from_state_dict(
        _required_mapping(payload.get("mode_table"), name="mode_table")
    )
    CausalExponentialStateRouter.from_state_dict(
        _required_mapping(
            payload.get("causal_router"),
            name="causal_router",
        )
    )
    PointwiseCausalRouter.from_state_dict(
        _required_mapping(
            payload.get("pointwise_router"),
            name="pointwise_router",
        )
    )
    controls = _required_mapping(
        payload.get("metadata_controls"),
        name="metadata_controls",
    )
    for name, state in controls.items():
        HierarchicalCategoricalRouteControl.from_state_dict(
            _required_mapping(
                state,
                name=f"metadata_controls.{name}",
            )
        )
    _validate_full_span_result_contract(payload, report_mapping)
    return copy.deepcopy(dict(report_mapping))


def run_variable_full_span_experiment(
    *,
    checkpoint: Path = DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    output: Path = DEFAULT_OUTPUT,
    contexts_per_role: int = DEFAULT_CONTEXTS_PER_ROLE,
    router_fit_contexts: int = DEFAULT_ROUTER_FIT_CONTEXTS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    random_seed: int = DEFAULT_RANDOM_SEED,
    minimum_nll_advantage: float = DEFAULT_MINIMUM_NLL_ADVANTAGE,
    maximum_relative_static_work: float = (
        DEFAULT_MAXIMUM_RELATIVE_STATIC_WORK
    ),
) -> dict[str, object]:
    """Run the locked full-span gate and save a source-weight-free artifact."""

    if (
        type(router_fit_contexts) is not int
        or not 0 < router_fit_contexts < contexts_per_role
    ):
        raise ValueError(
            "router_fit_contexts must be between zero and contexts_per_role"
        )
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    for name, value in (
        ("minimum_nll_advantage", minimum_nll_advantage),
        ("maximum_relative_static_work", maximum_relative_static_work),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    torch.manual_seed(random_seed)
    model, splits, checkpoint_metadata = (
        load_variable_associative_checkpoint(checkpoint_path)
    )
    model.eval()
    if model.config.n_layers < 1:
        raise ValueError("source model must contain at least one layer")
    input_site = DEFAULT_INPUT_SITE
    output_site = f"layer.{model.config.n_layers - 1}.output"
    source_fingerprint = module_state_fingerprint(model)
    roles = _context_role_splits(
        splits.train,
        contexts_per_role=contexts_per_role,
    )
    _assert_evaluation_contexts_are_disjoint(
        roles,
        splits.validation,
        splits.test,
    )
    width = model.config.d_model
    route_count = len(ROUTE_SCHEDULES[0]["budgets"])  # type: ignore[arg-type]

    collected: dict[
        str,
        dict[str, tuple[Tensor, Tensor]],
    ] = {}
    # Calibration B is deliberately not collected until every schedule and
    # router choice below is frozen.
    for role_name in ROLE_NAMES[:-1]:
        activation_names = (
            (output_site,)
            if role_name == "basis_a"
            else (input_site, output_site)
        )
        collected[role_name] = _collect_grids(
            model,
            roles[role_name],
            activation_names,
        )

    basis = decompose_fisher_modes(
        _basis_samples(
            roles["basis_a"],
            collected["basis_a"][output_site][0],
            collected["basis_a"][output_site][1],
            activation_name=output_site,
        )
    )
    if basis.width != width:
        raise RuntimeError("full-span Fisher width does not match the model")
    needs = {
        role_name: _answer_need(
            roles[role_name],
            collected[role_name],
            basis,
            input_site=input_site,
            output_site=output_site,
        )
        for role_name in ("mask_a", "policy_a", "router_a")
    }
    global_need = needs["mask_a"].mean(dim=0)
    global_order = tuple(
        sorted(
            range(width),
            key=lambda index: (
                -float(global_need[index].item()),
                index,
            ),
        )
    )

    policy_baseline = variable_associative_answer_logits(
        model,
        roles["policy_a"],
    )
    schedule_records: list[dict[str, object]] = []
    schedule_objects: dict[
        str,
        tuple[TotalNeedRouteTeacher, ConditionalModeTable],
    ] = {}
    for specification in ROUTE_SCHEDULES:
        budgets = tuple(specification["budgets"])  # type: ignore[arg-type]
        quantiles = tuple(specification["quantiles"])  # type: ignore[arg-type]
        if budgets[-1] != width:
            raise ValueError("every route schedule must end at model width")
        teacher = fit_total_need_route_teacher(
            needs["mask_a"],
            route_count=route_count,
            quantiles=quantiles,
        )
        clustering = partition_fisher_need_profiles_by_teacher(
            needs["mask_a"],
            teacher,
        )
        table = build_conditional_mode_table(
            needs["mask_a"],
            clustering,
            route_budgets=budgets,
        )
        policy_routes = teacher.assign(needs["policy_a"])
        policy_logits = _projected_answer_logits(
            model,
            roles["policy_a"],
            basis,
            input_site=input_site,
            output_site=output_site,
            table=table,
            routes=policy_routes,
        )
        record = {
            "name": specification["name"],
            "budgets": budgets,
            "quantiles": quantiles,
            "thresholds": tuple(
                float(value) for value in teacher.thresholds.tolist()
            ),
            "mask_route_counts": clustering.route_counts,
            "policy_teacher_behavior": _compact_behavior(
                _behavior_record(
                    policy_logits,
                    policy_baseline,
                    roles["policy_a"],
                )
            ),
            "policy_teacher_routes": _route_summary(
                policy_routes,
                table,
                width=width,
            ),
        }
        schedule_records.append(record)
        schedule_objects[str(specification["name"])] = (teacher, table)
    passing_schedules = [
        record
        for record in schedule_records
        if record["policy_teacher_behavior"]["passed"] is True  # type: ignore[index]
    ]
    if passing_schedules:
        selected_schedule = min(
            passing_schedules,
            key=lambda record: (
                int(
                    record["policy_teacher_routes"][
                        "ideal_selective_projection_macs"
                    ]  # type: ignore[index]
                ),
                float(
                    record["policy_teacher_behavior"]["hard_nll"]  # type: ignore[index]
                ),
                str(record["name"]),
            ),
        )
    else:
        selected_schedule = min(
            schedule_records,
            key=lambda record: (
                -sum(
                    bool(value)
                    for value in record[
                        "policy_teacher_behavior"
                    ]["gates"].values()  # type: ignore[index,union-attr]
                ),
                float(
                    record["policy_teacher_behavior"]["hard_nll"]  # type: ignore[index]
                ),
                int(
                    record["policy_teacher_routes"][
                        "ideal_selective_projection_macs"
                    ]  # type: ignore[index]
                ),
                str(record["name"]),
            ),
        )
    teacher, table = schedule_objects[str(selected_schedule["name"])]

    router_split = roles["router_a"]
    router_labels = teacher.assign(needs["router_a"])
    fit_examples = (
        router_split.example_context_indices < router_fit_contexts
    )
    holdout_examples = ~fit_examples
    fit_context_rows = torch.arange(router_fit_contexts)
    holdout_context_rows = torch.arange(
        router_fit_contexts,
        contexts_per_role,
    )
    router_fit_split = subset_variable_associative_recall_split(
        router_split,
        context_rows=fit_context_rows,
        name="router_a_internal_fit",
    )
    router_holdout = subset_variable_associative_recall_split(
        router_split,
        context_rows=holdout_context_rows,
        name="router_a_internal_holdout",
    )
    if not torch.equal(
        router_split.input_ids[fit_examples],
        router_fit_split.input_ids,
    ) or not torch.equal(
        router_split.input_ids[holdout_examples],
        router_holdout.input_ids,
    ):
        raise RuntimeError("router internal split no longer aligns by context")

    router_inputs = collected["router_a"][input_site][0]
    fit_inputs = router_inputs[fit_examples]
    holdout_inputs = router_inputs[holdout_examples]
    fit_labels = router_labels[fit_examples]
    holdout_labels = router_labels[holdout_examples]
    fit_query_mask = _query_mask(router_split)[fit_examples]
    holdout_query_mask = _query_mask(router_split)[holdout_examples]
    fit_key_mask = _key_mask(router_split)[fit_examples]
    holdout_key_mask = _key_mask(router_split)[holdout_examples]
    fit_positions = _logical_positions(router_split)[fit_examples]
    holdout_positions = _logical_positions(router_split)[holdout_examples]
    fit_label_grid = torch.zeros_like(fit_query_mask, dtype=torch.int64)
    fit_rows = torch.arange(fit_inputs.shape[0])
    fit_query_positions = router_split.supervised_positions[fit_examples]
    fit_label_grid[fit_rows, fit_query_positions] = fit_labels
    holdout_baseline = variable_associative_answer_logits(
        model,
        router_holdout,
    )

    causal_candidates: list[dict[str, object]] = []
    for decay_name, decay_values in DECAY_GRIDS:
        for ridge in ROUTER_RIDGES:
            for balance_power in ROUTER_CLASS_BALANCE_POWERS:
                selected_weights = _class_weights(
                    fit_labels,
                    route_count=route_count,
                    power=balance_power,
                )
                weight_grid = torch.zeros_like(
                    fit_query_mask,
                    dtype=torch.float64,
                )
                weight_grid[fit_rows, fit_query_positions] = selected_weights
                causal_router, fit_metrics = (
                    fit_causal_exponential_state_router(
                        fit_inputs,
                        fit_label_grid,
                        decay_rates=torch.tensor(
                            decay_values,
                            dtype=torch.float64,
                        ),
                        route_count=route_count,
                        query_valid_mask=fit_query_mask,
                        key_valid_mask=fit_key_mask,
                        logical_positions=fit_positions,
                        key_logical_positions=fit_positions,
                        sample_weights=weight_grid,
                        ridge=ridge,
                    )
                )
                holdout_logits = _answer_rows(
                    causal_router.logits(
                        holdout_inputs,
                        query_valid_mask=holdout_query_mask,
                        key_valid_mask=holdout_key_mask,
                        logical_positions=holdout_positions,
                        key_logical_positions=holdout_positions,
                    ),
                    router_holdout,
                )
                accounting = causal_router.analytic_accounting(
                    holdout_inputs,
                    query_valid_mask=holdout_query_mask,
                    key_valid_mask=holdout_key_mask,
                    logical_positions=holdout_positions,
                    key_logical_positions=holdout_positions,
                )
                for decision in ROUTER_DECISIONS:
                    predicted = _decision(holdout_logits, decision)
                    projected = _projected_answer_logits(
                        model,
                        router_holdout,
                        basis,
                        input_site=input_site,
                        output_site=output_site,
                        table=table,
                        routes=predicted,
                    )
                    route_summary = _route_summary(
                        predicted,
                        table,
                        width=width,
                    )
                    causal_candidates.append(
                        {
                            "name": (
                                f"{decay_name}_ridge{ridge:g}_"
                                f"balance{balance_power:g}_{decision}"
                            ),
                            "decay_name": decay_name,
                            "decay_rates": decay_values,
                            "ridge": ridge,
                            "class_balance_power": balance_power,
                            "decision": decision,
                            "fit_argmax_metrics": asdict(fit_metrics),
                            "holdout_classification": (
                                _classifier_diagnostics(
                                    predicted,
                                    holdout_labels,
                                    route_count=route_count,
                                )
                            ),
                            "behavior": _compact_behavior(
                                _behavior_record(
                                    projected,
                                    holdout_baseline,
                                    router_holdout,
                                )
                            ),
                            "routes": route_summary,
                            "router_macs": accounting.total_macs,
                            "router_parameters": (
                                accounting.total_stored_parameters
                            ),
                            "router_plus_ideal_selective_projection_macs": (
                                accounting.total_macs
                                + int(
                                    route_summary[
                                        "ideal_selective_projection_macs"
                                    ]
                                )
                            ),
                        }
                    )
    selected_causal = _select_router_candidate(causal_candidates)

    pointwise_candidates: list[dict[str, object]] = []
    fit_pointwise_inputs = _answer_rows(fit_inputs, router_fit_split)
    holdout_pointwise_inputs = _answer_rows(
        holdout_inputs,
        router_holdout,
    )
    for ridge in ROUTER_RIDGES:
        for balance_power in ROUTER_CLASS_BALANCE_POWERS:
            pointwise_router, fit_metrics = fit_pointwise_causal_router(
                fit_pointwise_inputs,
                fit_labels,
                route_count=route_count,
                sample_weights=_class_weights(
                    fit_labels,
                    route_count=route_count,
                    power=balance_power,
                ),
                ridge=ridge,
            )
            holdout_logits = pointwise_router.logits(
                holdout_pointwise_inputs
            )
            for decision in ROUTER_DECISIONS:
                predicted = _decision(holdout_logits, decision)
                projected = _projected_answer_logits(
                    model,
                    router_holdout,
                    basis,
                    input_site=input_site,
                    output_site=output_site,
                    table=table,
                    routes=predicted,
                )
                route_summary = _route_summary(
                    predicted,
                    table,
                    width=width,
                )
                router_macs = predicted.numel() * width * route_count
                pointwise_candidates.append(
                    {
                        "name": (
                            f"pointwise_ridge{ridge:g}_"
                            f"balance{balance_power:g}_{decision}"
                        ),
                        "ridge": ridge,
                        "class_balance_power": balance_power,
                        "decision": decision,
                        "fit_argmax_metrics": asdict(fit_metrics),
                        "holdout_classification": (
                            _classifier_diagnostics(
                                predicted,
                                holdout_labels,
                                route_count=route_count,
                            )
                        ),
                        "behavior": _compact_behavior(
                            _behavior_record(
                                projected,
                                holdout_baseline,
                                router_holdout,
                            )
                        ),
                        "routes": route_summary,
                        "router_macs": router_macs,
                        "router_parameters": (
                            2 * width
                            + width * route_count
                            + route_count
                        ),
                        "router_plus_ideal_selective_projection_macs": (
                            router_macs
                            + int(
                                route_summary[
                                    "ideal_selective_projection_macs"
                                ]
                            )
                        ),
                    }
                )
    selected_pointwise = _select_router_candidate(pointwise_candidates)

    metadata_selection_controls = _fit_metadata_controls(
        fit_labels,
        router_fit_split,
        route_count=route_count,
    )
    metadata_selection_records: dict[str, dict[str, object]] = {}
    for name, control in metadata_selection_controls.items():
        predicted = _control_routes(control, router_holdout)
        projected = _projected_answer_logits(
            model,
            router_holdout,
            basis,
            input_site=input_site,
            output_site=output_site,
            table=table,
            routes=predicted,
        )
        metadata_selection_records[name] = {
            "behavior": _compact_behavior(
                _behavior_record(
                    projected,
                    holdout_baseline,
                    router_holdout,
                )
            ),
            "routes": _route_summary(
                predicted,
                table,
                width=width,
            ),
        }
    selected_metadata_name = min(
        metadata_selection_records,
        key=lambda name: (
            float(
                metadata_selection_records[name]["behavior"][
                    "hard_nll"
                ]  # type: ignore[index]
            ),
            name,
        ),
    )

    all_query_mask = _query_mask(router_split)
    all_key_mask = _key_mask(router_split)
    all_positions = _logical_positions(router_split)
    all_rows = torch.arange(router_split.samples)
    all_label_grid = torch.zeros_like(all_query_mask, dtype=torch.int64)
    all_label_grid[all_rows, router_split.supervised_positions] = (
        router_labels
    )
    all_weights = _class_weights(
        router_labels,
        route_count=route_count,
        power=float(selected_causal["class_balance_power"]),
    )
    all_weight_grid = torch.zeros_like(
        all_query_mask,
        dtype=torch.float64,
    )
    all_weight_grid[all_rows, router_split.supervised_positions] = (
        all_weights
    )
    final_causal, final_causal_fit_metrics = (
        fit_causal_exponential_state_router(
            router_inputs,
            all_label_grid,
            decay_rates=torch.tensor(
                selected_causal["decay_rates"],
                dtype=torch.float64,
            ),
            route_count=route_count,
            query_valid_mask=all_query_mask,
            key_valid_mask=all_key_mask,
            logical_positions=all_positions,
            key_logical_positions=all_positions,
            sample_weights=all_weight_grid,
            ridge=float(selected_causal["ridge"]),
        )
    )
    final_pointwise, final_pointwise_fit_metrics = (
        fit_pointwise_causal_router(
            _answer_rows(router_inputs, router_split),
            router_labels,
            route_count=route_count,
            sample_weights=_class_weights(
                router_labels,
                route_count=route_count,
                power=float(
                    selected_pointwise["class_balance_power"]
                ),
            ),
            ridge=float(selected_pointwise["ridge"]),
        )
    )
    metadata_controls = _fit_metadata_controls(
        router_labels,
        router_split,
        route_count=route_count,
    )

    # Every choice is frozen above.  Calibration B starts here.
    calibration = roles["calibration_b"]
    collected["calibration_b"] = _collect_grids(
        model,
        calibration,
        (input_site, output_site),
    )
    needs["calibration_b"] = _answer_need(
        calibration,
        collected["calibration_b"],
        basis,
        input_site=input_site,
        output_site=output_site,
    )
    baseline_logits = variable_associative_answer_logits(
        model,
        calibration,
    )
    teacher_routes = teacher.assign(needs["calibration_b"])
    query_mask = _query_mask(calibration)
    key_mask = _key_mask(calibration)
    positions = _logical_positions(calibration)
    calibration_inputs = collected["calibration_b"][input_site][0]
    causal_routes = _decision(
        _answer_rows(
            final_causal.logits(
                calibration_inputs,
                query_valid_mask=query_mask,
                key_valid_mask=key_mask,
                logical_positions=positions,
                key_logical_positions=positions,
            ),
            calibration,
        ),
        str(selected_causal["decision"]),
    )
    pointwise_routes = _decision(
        final_pointwise.logits(
            _answer_rows(calibration_inputs, calibration)
        ),
        str(selected_pointwise["decision"]),
    )
    policy_routes = {
        "teacher": teacher_routes,
        "causal_router": causal_routes,
        "pointwise_ablation": pointwise_routes,
        **{
            f"metadata.{name}": _control_routes(control, calibration)
            for name, control in metadata_controls.items()
        },
    }
    learned_grid = torch.zeros_like(query_mask, dtype=torch.int64)
    learned_grid[
        torch.arange(calibration.samples),
        calibration.supervised_positions,
    ] = causal_routes
    metadata = _metadata_features(calibration)
    shuffle_specifications = {
        "position_length": {
            "position": metadata["position"],
            "length": metadata["length"],
        },
        "token_id_position_length": {
            "token_id": metadata["token_id"],
            "position": metadata["position"],
            "length": metadata["length"],
        },
    }
    shuffle_histograms: dict[str, bool] = {}
    for index, (name, strata) in enumerate(
        shuffle_specifications.items()
    ):
        shuffled_grid = stratified_shuffle_routes(
            learned_grid,
            strata,
            valid_mask=query_mask,
            route_count=route_count,
            seed=shuffle_seed + index,
        )
        policy_routes[f"shuffle.{name}"] = _answer_rows(
            shuffled_grid,
            calibration,
        )
        shuffle_histograms[name] = (
            route_histograms_by_stratum(
                learned_grid,
                strata,
                valid_mask=query_mask,
                route_count=route_count,
            )
            == route_histograms_by_stratum(
                shuffled_grid,
                strata,
                valid_mask=query_mask,
                route_count=route_count,
            )
        )

    projected_logits = {
        name: _projected_answer_logits(
            model,
            calibration,
            basis,
            input_site=input_site,
            output_site=output_site,
            table=table,
            routes=routes,
        )
        for name, routes in policy_routes.items()
    }
    behavior = {
        name: _compact_behavior(
            _behavior_record(logits, baseline_logits, calibration)
        )
        for name, logits in projected_logits.items()
    }
    route_summaries = {
        name: _route_summary(routes, table, width=width)
        for name, routes in policy_routes.items()
    }

    static_curve: dict[str, dict[str, object]] = {}
    smallest_static: tuple[int, str] | None = None
    for rank in range(width + 1):
        prefix_mask = torch.zeros(width, dtype=torch.bool)
        prefix_mask[:rank] = True
        global_mask = torch.zeros(width, dtype=torch.bool)
        if rank:
            global_mask[list(global_order[:rank])] = True
        for kind, mask in (
            ("fisher_prefix", prefix_mask),
            ("global_need", global_mask),
        ):
            static_logits = _projected_answer_logits(
                model,
                calibration,
                basis,
                input_site=input_site,
                output_site=output_site,
                static_mask=mask,
            )
            record = _compact_behavior(
                _behavior_record(
                    static_logits,
                    baseline_logits,
                    calibration,
                )
            )
            static_curve[f"{kind}:{rank}"] = record
            if record["passed"] is True and (
                smallest_static is None
                or (rank, kind) < smallest_static
            ):
                smallest_static = (rank, kind)
    if smallest_static is None:
        raise RuntimeError("full-width static identity failed behavior gates")
    static_rank, static_kind = smallest_static
    static_ideal_selective_projection_macs = (
        2 * width * static_rank * calibration.samples
    )
    diagnostic_dense_projection_macs = (
        2 * width * width * calibration.samples
    )

    causal_accounting = final_causal.analytic_accounting(
        calibration_inputs,
        query_valid_mask=query_mask,
        key_valid_mask=key_mask,
        logical_positions=positions,
        key_logical_positions=positions,
    )
    causal_ideal_total_macs = (
        causal_accounting.total_macs
        + int(
            route_summaries["causal_router"][
                "ideal_selective_projection_macs"
            ]
        )
    )
    pointwise_router_macs = (
        calibration.samples * width * route_count
    )
    pointwise_ideal_total_macs = (
        pointwise_router_macs
        + int(
            route_summaries["pointwise_ablation"][
                "ideal_selective_projection_macs"
            ]
        )
    )
    average_rank_ratio = (
        float(
            route_summaries["causal_router"][
                "average_active_modes"
            ]
        )
        / static_rank
    )
    ideal_mac_ratio = (
        causal_ideal_total_macs
        / static_ideal_selective_projection_macs
    )

    selected_metadata_control = f"metadata.{selected_metadata_name}"
    metadata_bootstrap = _bootstrap_advantage(
        projected_logits["causal_router"],
        projected_logits[selected_metadata_control],
        calibration,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    pointwise_bootstrap = _bootstrap_advantage(
        projected_logits["causal_router"],
        projected_logits["pointwise_ablation"],
        calibration,
        seed=bootstrap_seed + 1,
        samples=bootstrap_samples,
    )
    shuffle_bootstraps = {
        name: _bootstrap_advantage(
            projected_logits["causal_router"],
            projected_logits[f"shuffle.{name}"],
            calibration,
            seed=bootstrap_seed + 2 + index,
            samples=bootstrap_samples,
        )
        for index, name in enumerate(shuffle_specifications)
    }

    joint_gates = {
        "teacher_behavior": behavior["teacher"]["passed"] is True,
        "teacher_mean_rank_at_most_90_percent_static": (
            float(
                route_summaries["teacher"]["average_active_modes"]
            )
            / static_rank
            <= maximum_relative_static_work
        ),
        "causal_behavior": (
            behavior["causal_router"]["passed"] is True
        ),
        "smallest_static_exists": True,
        "mean_rank_at_most_90_percent_static": (
            average_rank_ratio <= maximum_relative_static_work
        ),
        "router_plus_ideal_selective_projection_at_most_90_percent_static": (
            ideal_mac_ratio <= maximum_relative_static_work
        ),
        "causal_beats_pointwise_nll": (
            float(pointwise_bootstrap["mean_nll_advantage"])
            >= minimum_nll_advantage
        ),
        "causal_pointwise_bootstrap_lower_bound": (
            float(pointwise_bootstrap["lower_95_percent"]) > 0
        ),
        "causal_beats_a_selected_metadata_nll": (
            float(metadata_bootstrap["mean_nll_advantage"])
            >= minimum_nll_advantage
        ),
        "causal_a_selected_metadata_bootstrap_lower_bound": (
            float(metadata_bootstrap["lower_95_percent"]) > 0
        ),
        "shuffle_histograms_preserved": all(
            shuffle_histograms.values()
        ),
        **{
            f"causal_beats_{name}_shuffle_nll": (
                float(record["mean_nll_advantage"])
                >= minimum_nll_advantage
            )
            for name, record in shuffle_bootstraps.items()
        },
        **{
            f"causal_{name}_shuffle_bootstrap_lower_bound": (
                float(record["lower_95_percent"]) > 0
            )
            for name, record in shuffle_bootstraps.items()
        },
    }
    joint_passed = all(joint_gates.values())

    valid_rows = int(calibration.attention_mask.sum().item())
    prefix_rows = int(key_mask.sum().item())
    all_pairs = int(
        (
            calibration.attention_mask.unsqueeze(2)
            & calibration.attention_mask.unsqueeze(1)
            & positions.unsqueeze(2).ge(positions.unsqueeze(1))
        ).sum().item()
    )
    prefix_pairs = int(
        (
            key_mask.unsqueeze(2)
            & key_mask.unsqueeze(1)
            & positions.unsqueeze(2).ge(positions.unsqueeze(1))
        ).sum().item()
    )
    padded_rows = (
        calibration.samples * calibration.maximum_sequence_length
    )
    padded_allowed_pairs = (
        calibration.samples
        * calibration.maximum_sequence_length
        * (calibration.maximum_sequence_length + 1)
        // 2
    )
    native_valid = native_transformer_span_accounting(
        valid_rows=valid_rows,
        causal_pairs=all_pairs,
        width=width,
        feed_forward_width=model.config.d_ff,
        layer_count=model.config.n_layers,
        padded_rows=padded_rows,
        padded_causal_pairs=padded_allowed_pairs,
    )
    native_prefix = native_transformer_span_accounting(
        valid_rows=prefix_rows,
        causal_pairs=prefix_pairs,
        width=width,
        feed_forward_width=model.config.d_ff,
        layer_count=model.config.n_layers,
        padded_rows=padded_rows,
        padded_causal_pairs=padded_allowed_pairs,
    )
    logical_all_row_head_macs = (
        valid_rows * width * model.config.vocab_size
    )
    padded_all_row_head_macs = (
        padded_rows * width * model.config.vocab_size
    )
    query_head_macs = (
        calibration.samples * width * model.config.vocab_size
    )
    logical_valid_native_complete = (
        native_valid.logical_total_macs + logical_all_row_head_macs
    )
    padded_allowed_edge_native_complete = (
        native_valid.padded_total_macs + padded_all_row_head_macs
    )
    logical_prefix_native_complete = (
        native_prefix.logical_total_macs + query_head_macs
    )
    padded_allowed_edge_prefix_native_complete = (
        native_prefix.padded_total_macs + query_head_macs
    )
    oracle_logical_ideal_complete = (
        logical_valid_native_complete + causal_ideal_total_macs
    )
    oracle_logical_diagnostic_dense_complete = (
        logical_valid_native_complete
        + causal_accounting.total_macs
        + diagnostic_dense_projection_macs
    )
    graph_envelopes = _hypothetical_graph_envelopes(
        model,
        calibration,
        table=table,
        learned_routes=causal_routes,
        static_rank=static_rank,
    )

    calibration_record = {
        "baseline": asdict(
            variable_associative_metrics_from_logits(
                calibration,
                baseline_logits,
            )
        ),
        "policies": {
            name: {
                "behavior": behavior[name],
                "routes": route_summaries[name],
            }
            for name in policy_routes
        },
        "causal_router": {
            "classification": _classifier_diagnostics(
                causal_routes,
                teacher_routes,
                route_count=route_count,
            ),
            "state_channels": final_causal.state_channels,
            "decay_rates": tuple(
                float(value)
                for value in final_causal.decay_rates.tolist()
            ),
            "decision": selected_causal["decision"],
            "accounting": asdict(causal_accounting),
            "router_macs": causal_accounting.total_macs,
            "router_parameters": (
                causal_accounting.total_stored_parameters
            ),
            "router_plus_ideal_selective_projection_macs": (
                causal_ideal_total_macs
            ),
            "router_plus_diagnostic_dense_projection_macs": (
                causal_accounting.total_macs
                + diagnostic_dense_projection_macs
            ),
        },
        "pointwise_ablation": {
            "classification": _classifier_diagnostics(
                pointwise_routes,
                teacher_routes,
                route_count=route_count,
            ),
            "decision": selected_pointwise["decision"],
            "router_macs": pointwise_router_macs,
            "router_parameters": (
                2 * width + width * route_count + route_count
            ),
            "router_plus_ideal_selective_projection_macs": (
                pointwise_ideal_total_macs
            ),
            "router_plus_diagnostic_dense_projection_macs": (
                pointwise_router_macs + diagnostic_dense_projection_macs
            ),
        },
        "a_selected_metadata_control": selected_metadata_control,
        "causal_advantage_over_a_selected_metadata_bootstrap": (
            metadata_bootstrap
        ),
        "causal_advantage_over_pointwise_bootstrap": (
            pointwise_bootstrap
        ),
        "shuffle_histograms_preserved": shuffle_histograms,
        "shuffle_advantage_bootstraps": shuffle_bootstraps,
        "static_curve": static_curve,
        "smallest_passing_static": {
            "rank": static_rank,
            "kind": static_kind,
            "ideal_selective_projection_macs": (
                static_ideal_selective_projection_macs
            ),
            "diagnostic_dense_projection_macs": (
                diagnostic_dense_projection_macs
            ),
            "behavior": static_curve[f"{static_kind}:{static_rank}"],
        },
        "comparisons": {
            "causal_average_rank_to_static": average_rank_ratio,
            "causal_router_plus_ideal_selective_projection_to_static": (
                ideal_mac_ratio
            ),
            "pointwise_router_plus_ideal_selective_projection_to_static": (
                pointwise_ideal_total_macs
                / static_ideal_selective_projection_macs
            ),
        },
        "joint_gate": {
            "gates": joint_gates,
            "passed": joint_passed,
            "reason_if_failed": (
                None
                if joint_passed
                else (
                    "full-span representation fidelity, content advantage, "
                    "and compute reduction did not pass jointly"
                )
            ),
        },
    }
    scientific_status = {
        "source_model_frozen": True,
        "full_transformer_span_targeted": True,
        "query_sparse_task": True,
        "output_demand_sparsity_separated_from_fisher_routing": True,
        "calibration_b_evaluated": True,
        "calibration_b_passed": joint_passed,
        "validation_evaluated": False,
        "test_evaluated": False,
        "gate_only_command_never_fits_graph": True,
        "model_level_graph_fitted": False,
        "model_level_eligible": joint_passed,
        "source_layer_compute_savings_achieved": False,
        "hypothetical_graph_headroom_only": True,
    }
    if module_state_fingerprint(model) != source_fingerprint:
        raise RuntimeError("full-span experiment mutated source weights")

    protocol = {
        "input_site": input_site,
        "output_site": output_site,
        "source_layers": tuple(
            f"layer.{index}" for index in range(model.config.n_layers)
        ),
        "contexts_per_role": contexts_per_role,
        "role_context_hashes": {
            name: roles[name].semantic_context_hashes
            for name in ROLE_NAMES
        },
        "validation_context_hashes": (
            splits.validation.semantic_context_hashes
        ),
        "test_context_hashes": splits.test.semantic_context_hashes,
        "validation_policy": "untouched_after_calibration_b_failure",
        "test_policy": "hash_only_not_model_evaluated",
        "basis_and_need_rows": "supervised_answer_only",
        "router_queries": "supervised_answer_only",
        "router_keys": "causal_prefix_through_answer",
        "route_schedules": ROUTE_SCHEDULES,
        "decay_grids": DECAY_GRIDS,
        "router_ridges": ROUTER_RIDGES,
        "router_class_balance_powers": (
            ROUTER_CLASS_BALANCE_POWERS
        ),
        "router_decisions": ROUTER_DECISIONS,
        "router_internal_fit_contexts": router_fit_contexts,
        "router_internal_holdout_contexts": (
            contexts_per_role - router_fit_contexts
        ),
        "minimum_nll_advantage": minimum_nll_advantage,
        "maximum_relative_static_work": (
            maximum_relative_static_work
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "shuffle_seed": shuffle_seed,
        "random_seed": random_seed,
        "native_span_executes_in_representation_oracle": True,
        "model_fit_fail_closed": True,
        "gate_only_command_never_fits_graph": True,
    }
    analysis = {
        "basis": {
            "width": basis.width,
            "observations": basis.observations,
            "sequences": basis.sequences,
            "fisher_trace": basis.fisher_trace,
            "eigenvalues": tuple(
                float(value) for value in basis.eigenvalues.tolist()
            ),
        },
        "schedule_selection": {
            "selection_role": "policy_a",
            "selection_rule": (
                "lowest ideal selective answer-only teacher projection "
                "MACs among full behavior passes"
            ),
            "selected": selected_schedule,
            "candidates": schedule_records,
        },
        "router_selection": {
            "selection_role": "router_a_internal_context_holdout",
            "selection_rule": (
                "lowest router-plus-ideal-selective-projection MACs among "
                "full behavior passes; deterministic best-gates fallback"
            ),
            "selected_causal": copy.deepcopy(selected_causal),
            "selected_pointwise": copy.deepcopy(selected_pointwise),
            "selected_metadata_control": selected_metadata_name,
            "metadata_control_selection_rule": (
                "lowest hard NLL on router_a internal context holdout; "
                "lexical name tie-break"
            ),
            "metadata_control_candidates": copy.deepcopy(
                metadata_selection_records
            ),
            "causal_candidate_count": len(causal_candidates),
            "pointwise_candidate_count": len(pointwise_candidates),
            "final_causal_argmax_fit_metrics": asdict(
                final_causal_fit_metrics
            ),
            "final_pointwise_argmax_fit_metrics": asdict(
                final_pointwise_fit_metrics
            ),
        },
        "calibration_b": calibration_record,
        "compute": {
            "valid_rows_with_suffix": valid_rows,
            "prefix_rows_through_answer": prefix_rows,
            "all_valid_causal_pairs": all_pairs,
            "prefix_causal_pairs": prefix_pairs,
            "padded_rows": padded_rows,
            "padded_allowed_causal_pairs": padded_allowed_pairs,
            "logical_valid_native_span": asdict(native_valid),
            "logical_prefix_native_span": asdict(native_prefix),
            "logical_all_row_head_macs": logical_all_row_head_macs,
            "padded_all_row_head_macs": padded_all_row_head_macs,
            "query_sparse_head_macs": query_head_macs,
            "logical_valid_native_complete_macs": (
                logical_valid_native_complete
            ),
            "padded_allowed_edge_native_complete_macs": (
                padded_allowed_edge_native_complete
            ),
            "logical_prefix_native_complete_macs": (
                logical_prefix_native_complete
            ),
            "padded_allowed_edge_prefix_native_complete_macs": (
                padded_allowed_edge_prefix_native_complete
            ),
            "representation_oracle_logical_ideal_complete_macs": (
                oracle_logical_ideal_complete
            ),
            "representation_oracle_logical_diagnostic_dense_complete_macs": (
                oracle_logical_diagnostic_dense_complete
            ),
            "representation_oracle_logical_ideal_to_native_ratio": (
                oracle_logical_ideal_complete
                / logical_valid_native_complete
            ),
            "router_plus_ideal_selective_projection_only_to_logical_prefix_native_ratio": (
                causal_ideal_total_macs / logical_prefix_native_complete
            ),
            "router_plus_projection_only_is_not_complete_executor": True,
            "padded_allowed_edge_estimate_is_not_backend_issued_work": True,
            "hypothetical_graph_envelopes": graph_envelopes,
            "excluded_from_graph_mac_envelopes": (
                "normalization, nonlinear elementwise work, gathers, "
                "masking, memory traffic, and kernel launch overhead"
            ),
        },
    }
    report = {
        "artifact": str(Path(output)),
        "source_checkpoint": str(checkpoint_path),
        "scientific_status": scientific_status,
        "protocol": protocol,
        "analysis": analysis,
    }
    payload = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "contains_source_model_weights": False,
        "contains_compiled_executor_weights": False,
        "source": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "model_state_fingerprint": source_fingerprint,
        },
        "scientific_status": scientific_status,
        "protocol": protocol,
        "basis": basis.state_dict(),
        "teacher": teacher.state_dict(),
        "mode_table": table.state_dict(),
        "causal_router": final_causal.state_dict(),
        "pointwise_router": final_pointwise.state_dict(),
        "metadata_controls": {
            name: control.state_dict()
            for name, control in metadata_controls.items()
        },
        "analysis": analysis,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            _jsonable(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    verify_variable_full_span_artifacts(output_path)
    return copy.deepcopy(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the query-sparse full-transformer-span Fisher routing gate "
            "without fitting a graph; a pass only marks a later fit eligible."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contexts-per-role",
        type=int,
        default=DEFAULT_CONTEXTS_PER_ROLE,
    )
    parser.add_argument(
        "--router-fit-contexts",
        type=int,
        default=DEFAULT_ROUTER_FIT_CONTEXTS,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_variable_full_span_experiment(
        checkpoint=arguments.checkpoint,
        output=arguments.output,
        contexts_per_role=arguments.contexts_per_role,
        router_fit_contexts=arguments.router_fit_contexts,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
