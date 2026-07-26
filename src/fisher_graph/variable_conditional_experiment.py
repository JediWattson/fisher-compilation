"""Variable-format conditional Fisher routing with fail-closed escalation.

The earlier fixed-layout toy result could be explained by absolute position.
This experiment removes that shortcut, fits every policy on disjoint semantic
contexts, and compares a hidden-state router with metadata-only and
same-stratum shuffled controls.  Validation is evaluated only after every
calibration-B gate passes.

The representation oracle still executes the native middle block and then
projects its residual delta.  A separate source-independent executor may be
trained only after this module reports ``model_level_eligible=True``.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import module_state_fingerprint
from .conditional_controls import (
    HierarchicalCategoricalRouteControl,
    fit_hierarchical_categorical_route_control,
    route_histograms_by_stratum,
    stratified_shuffle_routes,
)
from .conditional_routing import (
    ConditionalModalRoutingPlan,
    ConditionalModeTable,
    TotalNeedRouteTeacher,
    build_conditional_mode_table,
    fit_pointwise_causal_router,
    fit_total_need_route_teacher,
    linear_codec_fisher_damage_profiles,
    partition_fisher_need_profiles_by_teacher,
)
from .modes import (
    ActivationGradientSamples,
    FisherModeBasis,
    collect_activation_score_gradients,
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


DEFAULT_OUTPUT = Path(
    ".local-runs/variable-associative/layer-1-variable-conditional.pt"
)
DEFAULT_LAYER_INDEX = 1
DEFAULT_CONTEXTS_PER_ROLE = 24
DEFAULT_ROUTE_BUDGETS = (0, 8, 16, 24, 32)
DEFAULT_ROUTE_QUANTILES: tuple[float, ...] | None = None
DEFAULT_ROUTER_RIDGE = 1e-2
DEFAULT_ROUTER_CLASS_BALANCE_POWER = 0.0
DEFAULT_SHUFFLE_SEED = 91_207
DEFAULT_BOOTSTRAP_SEED = 91_208
DEFAULT_BOOTSTRAP_SAMPLES = 2_000
DEFAULT_MINIMUM_NLL_ADVANTAGE = 0.005
DEFAULT_MAXIMUM_RELATIVE_STATIC_WORK = 0.90
DEFAULT_MAXIMUM_FULL_WIDTH_FALLBACK_RATE = 0.25

_SCHEMA = "fisher_graph.variable_conditional_routing"
_FORMAT_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_role_splits(
    source: VariableAssociativeRecallSplit,
    *,
    contexts_per_role: int,
) -> dict[str, VariableAssociativeRecallSplit]:
    if type(contexts_per_role) is not int or contexts_per_role <= 0:
        raise ValueError("contexts_per_role must be positive")
    names = ("basis_a", "mask_a", "router_a", "calibration_b")
    needed = contexts_per_role * len(names)
    if needed > source.contexts:
        raise ValueError("source split has too few contexts for compiler roles")
    result = {}
    for role_index, name in enumerate(names):
        start = role_index * contexts_per_role
        result[name] = subset_variable_associative_recall_split(
            source,
            context_rows=torch.arange(start, start + contexts_per_role),
            name=name,
        )
    hash_sets = [
        set(split.semantic_context_hashes)
        for split in result.values()
    ]
    for left in range(len(hash_sets)):
        for right in range(left + 1, len(hash_sets)):
            if not hash_sets[left].isdisjoint(hash_sets[right]):
                raise RuntimeError("compiler role semantic contexts overlap")
    return result


def _collect(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    activation_names: set[str],
) -> dict[str, ActivationGradientSamples]:
    collection = collect_activation_score_gradients(
        model,
        split.input_ids,
        split.targets,
        attention_mask=split.attention_mask,
        activation_names=activation_names,
        ignore_index=split.ignore_index,
    )
    return dict(collection.samples)


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


def _delta_need(
    inputs: ActivationGradientSamples,
    outputs: ActivationGradientSamples,
    basis: FisherModeBasis,
) -> Tensor:
    if inputs.locations.shape != outputs.locations.shape or not torch.equal(
        inputs.locations,
        outputs.locations,
    ):
        raise ValueError("input/output activation samples are not aligned")
    delta = outputs.activations - inputs.activations
    return linear_codec_fisher_damage_profiles(
        delta,
        outputs.score_gradients,
        encoder=basis.vectors,
        decoder=basis.vectors,
    )


def _scatter_valid_rows(
    split: VariableAssociativeRecallSplit,
    rows: Tensor,
    *,
    fill: int | float = 0,
) -> Tensor:
    metadata = split.valid_token_metadata()
    if rows.ndim == 0 or rows.shape[0] != metadata.observations:
        raise ValueError("rows do not align with valid split positions")
    trailing = rows.shape[1:]
    output = torch.full(
        (split.samples, split.maximum_sequence_length, *trailing),
        fill,
        dtype=rows.dtype,
        device=rows.device,
    )
    output.reshape(-1, *trailing).index_copy_(
        0,
        metadata.selected_flat_indices.to(device=rows.device),
        rows,
    )
    return output


def _metadata_features(
    split: VariableAssociativeRecallSplit,
) -> dict[str, Tensor]:
    positions = torch.arange(
        split.maximum_sequence_length,
        dtype=torch.int64,
    ).unsqueeze(0).expand(split.samples, -1)
    lengths = split.valid_lengths.unsqueeze(1).expand_as(positions)
    return {
        "position": positions,
        "length": lengths,
        "token_role": split.token_role_ids,
        "token_id": split.input_ids,
    }


def _fit_metadata_controls(
    route_grid: Tensor,
    split: VariableAssociativeRecallSplit,
    *,
    route_count: int,
) -> dict[str, HierarchicalCategoricalRouteControl]:
    features = _metadata_features(split)
    specifications = {
        "position_only": (("position",),),
        "length_only": (("length",),),
        "position_length": (
            ("position", "length"),
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
        "token_id_position_length": (
            ("token_id", "position", "length"),
            ("token_id", "position"),
            ("position", "length"),
            ("token_id",),
            ("position",),
            ("length",),
        ),
    }
    return {
        name: fit_hierarchical_categorical_route_control(
            route_grid,
            features,
            valid_mask=split.attention_mask,
            route_count=route_count,
            levels=levels,
        )
        for name, levels in specifications.items()
    }


def _projected_answer_logits(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    input_site: str,
    output_site: str,
    basis: FisherModeBasis,
    mode_table: ConditionalModeTable | None = None,
    route_grid: Tensor | None = None,
    static_mask: Tensor | None = None,
    routing_plan: ConditionalModalRoutingPlan | None = None,
    batch_size: int = 512,
) -> tuple[Tensor, Tensor | None]:
    policies = sum(
        value is not None
        for value in (static_mask, routing_plan, route_grid)
    )
    if policies != 1:
        raise ValueError("exactly one projection policy must be supplied")
    if (route_grid is not None or routing_plan is not None) and mode_table is None:
        raise ValueError("conditional projection requires a mode table")
    if route_grid is not None and route_grid.shape != split.input_ids.shape:
        raise ValueError("route_grid must match the split token grid")
    if static_mask is not None and (
        static_mask.dtype != torch.bool
        or static_mask.shape != (basis.width,)
    ):
        raise ValueError("static_mask must match the Fisher width")

    cursor = 0
    incoming: Tensor | None = None
    current_routes: Tensor | None = None
    observed_routes: list[Tensor] = []

    def capture_input(values: Tensor) -> Tensor:
        nonlocal cursor, incoming, current_routes
        incoming = values
        batch = values.shape[0]
        if routing_plan is not None:
            current_routes = routing_plan.route(values)
        elif route_grid is not None:
            current_routes = route_grid[cursor : cursor + batch].to(
                device=values.device
            )
        else:
            current_routes = None
        if current_routes is not None:
            observed_routes.append(current_routes.detach().cpu())
        cursor += batch
        return values

    def project_output(values: Tensor) -> Tensor:
        nonlocal incoming, current_routes
        if incoming is None:
            raise RuntimeError("output projection ran before input capture")
        vectors = basis.vectors.to(
            device=values.device,
            dtype=torch.float64,
        )
        delta = values.to(torch.float64) - incoming.to(torch.float64)
        coordinates = delta @ vectors
        if static_mask is not None:
            masked = coordinates * static_mask.to(
                device=values.device,
                dtype=torch.float64,
            )
        else:
            assert mode_table is not None and current_routes is not None
            masked = mode_table.mask_coordinates(
                coordinates,
                current_routes,
            )
        projected = incoming.to(torch.float64) + masked @ vectors.T
        valid = split.attention_mask[
            cursor - values.shape[0] : cursor
        ].to(device=values.device)
        projected = torch.where(
            valid.unsqueeze(-1),
            projected,
            values.to(torch.float64),
        )
        incoming = None
        current_routes = None
        return projected.to(dtype=values.dtype)

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
        raise RuntimeError("projection replay did not consume the split exactly")
    routes = torch.cat(observed_routes) if observed_routes else None
    return logits, routes


def _per_example_nll(
    logits: Tensor,
    split: VariableAssociativeRecallSplit,
) -> Tensor:
    return F.cross_entropy(
        logits,
        split.answer_token_ids,
        reduction="none",
    ).to(torch.float64)


def _behavior_record(
    logits: Tensor,
    baseline_logits: Tensor,
    split: VariableAssociativeRecallSplit,
) -> dict[str, object]:
    metrics = variable_associative_metrics_from_logits(split, logits)
    baseline_metrics = variable_associative_metrics_from_logits(
        split,
        baseline_logits,
    )
    nll = _per_example_nll(logits, split)
    baseline_nll = _per_example_nll(baseline_logits, split)
    top1 = logits.argmax(dim=-1).eq(baseline_logits.argmax(dim=-1))
    teacher_kl = F.kl_div(
        logits.log_softmax(dim=-1),
        baseline_logits.softmax(dim=-1),
        reduction="none",
    ).sum(dim=-1)
    delta_nll = metrics.hard_nll - baseline_metrics.hard_nll
    p90_absolute_delta = float(
        torch.quantile((nll - baseline_nll).abs(), 0.90).item()
    )
    top1_agreement = float(top1.to(torch.float64).mean().item())
    p10_top1 = float(
        torch.quantile(top1.to(torch.float64), 0.10).item()
    )
    kl = float(teacher_kl.mean().item())
    gates = {
        "absolute_delta_nll": abs(delta_nll) <= 0.05,
        "top1_agreement": top1_agreement >= 0.95,
        "native_teacher_kl": kl <= 0.05,
        "p90_absolute_delta_nll": p90_absolute_delta <= 0.10,
        "p10_top1_agreement": p10_top1 >= 0.90,
        "answer_accuracy": metrics.answer_accuracy >= 0.995,
        "paired_context_accuracy": metrics.paired_context_accuracy >= 0.99,
        "minimum_stratum_accuracy": metrics.minimum_stratum_accuracy >= 0.99,
    }
    return {
        "metrics": asdict(metrics),
        "delta_nll": delta_nll,
        "top1_agreement": top1_agreement,
        "native_teacher_kl": kl,
        "p90_absolute_delta_nll": p90_absolute_delta,
        "p10_top1_agreement": p10_top1,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _route_accounting(
    route_grid: Tensor,
    split: VariableAssociativeRecallSplit,
    table: ConditionalModeTable,
    *,
    router_input_width: int,
) -> dict[str, object]:
    selected = route_grid[split.attention_mask].to(torch.int64)
    counts = torch.bincount(selected, minlength=table.routes)
    budgets = torch.tensor(table.route_budgets, dtype=torch.float64)
    active = budgets[selected]
    tokens = selected.numel()
    active_applications = int(active.sum().item())
    router_macs = tokens * router_input_width * table.routes
    projection_macs = 2 * table.modes * active_applications
    return {
        "valid_tokens": tokens,
        "route_counts": tuple(int(value) for value in counts.tolist()),
        "average_active_modes": float(active.mean().item()),
        "active_mode_ratio": float(active.mean().item()) / table.modes,
        "full_width_fallback_rate": float(
            (active == table.modes).to(torch.float64).mean().item()
        ),
        "active_mode_applications": active_applications,
        "router_macs": router_macs,
        "projection_macs": projection_macs,
        "router_plus_projection_macs": router_macs + projection_macs,
    }


def _bootstrap_advantage(
    candidate_logits: Tensor,
    control_logits: Tensor,
    split: VariableAssociativeRecallSplit,
    *,
    seed: int,
    samples: int,
) -> dict[str, object]:
    if type(samples) is not int or samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    candidate_nll = _per_example_nll(candidate_logits, split)
    control_nll = _per_example_nll(control_logits, split)
    differences = (control_nll - candidate_nll).detach().to(device="cpu")
    context_indices = split.example_context_indices.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    if context_indices.shape != differences.shape:
        raise RuntimeError("context IDs do not align with per-example NLL")
    unique_contexts = context_indices.unique(sorted=True)
    context_means = torch.stack(
        [
            differences[context_indices == context].mean()
            for context in unique_contexts
        ]
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(samples, dtype=torch.float64)
    for index in range(samples):
        rows = torch.randint(
            context_means.numel(),
            (context_means.numel(),),
            generator=generator,
        )
        means[index] = context_means.index_select(0, rows).mean()
    return {
        "resampling_unit": "semantic_context",
        "resampling_units": int(context_means.numel()),
        "mean_nll_advantage": float(context_means.mean().item()),
        "lower_95_percent": float(torch.quantile(means, 0.025).item()),
        "upper_95_percent": float(torch.quantile(means, 0.975).item()),
    }


def _fit_policy(
    model: ToyTransformer,
    roles: Mapping[str, VariableAssociativeRecallSplit],
    *,
    input_site: str,
    output_site: str,
    route_budgets: tuple[int, ...],
    route_quantiles: tuple[float, ...] | None,
    router_ridge: float,
    router_class_balance_power: float,
) -> tuple[
    FisherModeBasis,
    TotalNeedRouteTeacher,
    ConditionalModalRoutingPlan,
    dict[str, HierarchicalCategoricalRouteControl],
    dict[str, object],
]:
    basis_samples = _collect(
        model,
        roles["basis_a"],
        activation_names={output_site},
    )[output_site]
    basis = decompose_fisher_modes(basis_samples)
    if route_budgets[-1] != basis.width:
        raise ValueError("route budgets must end at the activation width")

    mask_samples = _collect(
        model,
        roles["mask_a"],
        activation_names={input_site, output_site},
    )
    mask_inputs = mask_samples[input_site]
    mask_outputs = mask_samples[output_site]
    _assert_sample_alignment(
        roles["mask_a"],
        mask_inputs,
        mask_outputs,
    )
    mask_need = _delta_need(mask_inputs, mask_outputs, basis)
    teacher = fit_total_need_route_teacher(
        mask_need,
        route_count=len(route_budgets),
        quantiles=route_quantiles,
    )
    clustering = partition_fisher_need_profiles_by_teacher(
        mask_need,
        teacher,
    )
    table = build_conditional_mode_table(
        mask_need,
        clustering,
        route_budgets=route_budgets,
    )
    router_samples = _collect(
        model,
        roles["router_a"],
        activation_names={input_site, output_site},
    )
    router_inputs = router_samples[input_site]
    router_outputs = router_samples[output_site]
    _assert_sample_alignment(
        roles["router_a"],
        router_inputs,
        router_outputs,
    )
    router_need = _delta_need(router_inputs, router_outputs, basis)
    teacher_rows = teacher.assign(router_need)
    router_metadata = roles["router_a"].valid_token_metadata()
    row_weights = (
        1.0 / router_metadata.valid_lengths.to(torch.float64)
    )
    class_counts = torch.bincount(
        teacher_rows,
        minlength=len(route_budgets),
    ).to(torch.float64)
    if (class_counts == 0).any():
        raise ValueError("router A labels leave an empty route")
    class_scale = (
        class_counts.mean() / class_counts
    ).pow(router_class_balance_power)
    row_weights = row_weights * class_scale[teacher_rows]
    router, router_metrics = fit_pointwise_causal_router(
        router_inputs.activations,
        teacher_rows,
        route_count=len(route_budgets),
        sample_weights=row_weights,
        ridge=router_ridge,
    )
    plan = ConditionalModalRoutingPlan(
        mode_table=table,
        router=router,
        profile_semantics=(
            "squared_first_order_score_change_from_residual_delta_"
            "orthonormal_mode_suppression"
        ),
    )
    teacher_grid = _scatter_valid_rows(
        roles["router_a"],
        teacher_rows,
    )
    controls = _fit_metadata_controls(
        teacher_grid,
        roles["router_a"],
        route_count=table.routes,
    )
    global_need = mask_need.to(torch.float64).mean(dim=0)
    global_order = tuple(
        sorted(
            range(basis.width),
            key=lambda index: (-float(global_need[index].item()), index),
        )
    )
    fit_report = {
        "basis_observations": basis.observations,
        "basis_sequences": basis.sequences,
        "mask_observations": mask_need.shape[0],
        "router_observations": router_inputs.observations,
        "router_metrics": asdict(router_metrics),
        "router_class_counts": tuple(
            int(value) for value in class_counts.tolist()
        ),
        "router_class_balance_power": router_class_balance_power,
        "teacher_thresholds": tuple(
            float(value) for value in teacher.thresholds.tolist()
        ),
        "teacher_fit_quantiles": teacher.fit_quantiles,
        "mask_route_counts": clustering.route_counts,
        "global_need_order": global_order,
    }
    return basis, teacher, plan, controls, fit_report


def _evaluate_split(
    model: ToyTransformer,
    split: VariableAssociativeRecallSplit,
    *,
    input_site: str,
    output_site: str,
    basis: FisherModeBasis,
    teacher: TotalNeedRouteTeacher,
    plan: ConditionalModalRoutingPlan,
    controls: Mapping[str, HierarchicalCategoricalRouteControl],
    global_need_order: tuple[int, ...],
    shuffle_seed: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, object], dict[str, object]]:
    baseline_logits = variable_associative_answer_logits(model, split)
    samples = _collect(
        model,
        split,
        activation_names={input_site, output_site},
    )
    inputs = samples[input_site]
    outputs = samples[output_site]
    _assert_sample_alignment(split, inputs, outputs)
    need = _delta_need(inputs, outputs, basis)
    learned_rows = plan.route(inputs.activations)
    learned_grid = _scatter_valid_rows(split, learned_rows)
    teacher_grid = teacher.assign(
        _scatter_valid_rows(split, need),
        valid_mask=split.attention_mask,
    )
    metadata = _metadata_features(split)
    control_grids = {
        name: control.predict(
            metadata,
            valid_mask=split.attention_mask,
        )
        for name, control in controls.items()
    }
    shuffled_grid = stratified_shuffle_routes(
        learned_grid,
        {
            "position": metadata["position"],
            "length": metadata["length"],
        },
        valid_mask=split.attention_mask,
        route_count=plan.mode_table.routes,
        seed=shuffle_seed,
    )

    learned_logits, runtime_routes = _projected_answer_logits(
        model,
        split,
        input_site=input_site,
        output_site=output_site,
        basis=basis,
        mode_table=plan.mode_table,
        routing_plan=plan,
    )
    if runtime_routes is None or not torch.equal(
        runtime_routes[split.attention_mask],
        learned_grid[split.attention_mask],
    ):
        raise RuntimeError("runtime learned routes disagree with collected inputs")
    teacher_logits, _ = _projected_answer_logits(
        model,
        split,
        input_site=input_site,
        output_site=output_site,
        basis=basis,
        mode_table=plan.mode_table,
        route_grid=teacher_grid,
    )
    control_logits = {
        name: _projected_answer_logits(
            model,
            split,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            mode_table=plan.mode_table,
            route_grid=grid,
        )[0]
        for name, grid in control_grids.items()
    }
    shuffled_logits, _ = _projected_answer_logits(
        model,
        split,
        input_site=input_site,
        output_site=output_site,
        basis=basis,
        mode_table=plan.mode_table,
        route_grid=shuffled_grid,
    )

    static_curve: dict[int, dict[str, object]] = {}
    static_logits: dict[tuple[str, int], Tensor] = {}
    for rank in range(basis.width + 1):
        prefix_mask = torch.zeros(basis.width, dtype=torch.bool)
        prefix_mask[:rank] = True
        global_mask = torch.zeros(basis.width, dtype=torch.bool)
        if rank:
            global_mask[list(global_need_order[:rank])] = True
        prefix_logits, _ = _projected_answer_logits(
            model,
            split,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            static_mask=prefix_mask,
        )
        global_logits, _ = _projected_answer_logits(
            model,
            split,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            static_mask=global_mask,
        )
        static_logits[("fisher_prefix", rank)] = prefix_logits
        static_logits[("global_need_mask", rank)] = global_logits
        static_curve[rank] = {
            "fisher_prefix": _behavior_record(
                prefix_logits,
                baseline_logits,
                split,
            ),
            "global_need_mask": _behavior_record(
                global_logits,
                baseline_logits,
                split,
            ),
        }
    passing_static = [
        (rank, kind)
        for rank, records in static_curve.items()
        for kind, record in records.items()
        if record["passed"] is True
    ]
    smallest_static = min(passing_static) if passing_static else None

    learned_record = _behavior_record(
        learned_logits,
        baseline_logits,
        split,
    )
    teacher_record = _behavior_record(
        teacher_logits,
        baseline_logits,
        split,
    )
    control_records = {
        name: _behavior_record(logits, baseline_logits, split)
        for name, logits in control_logits.items()
    }
    shuffle_record = _behavior_record(
        shuffled_logits,
        baseline_logits,
        split,
    )
    strongest_metadata_name = min(
        control_records,
        key=lambda name: float(
            control_records[name]["metrics"]["hard_nll"]  # type: ignore[index]
        ),
    )
    strongest_metadata_logits = control_logits[strongest_metadata_name]
    metadata_bootstrap = _bootstrap_advantage(
        learned_logits,
        strongest_metadata_logits,
        split,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    shuffle_bootstrap = _bootstrap_advantage(
        learned_logits,
        shuffled_logits,
        split,
        seed=bootstrap_seed + 1,
        samples=bootstrap_samples,
    )
    accounting = _route_accounting(
        learned_grid,
        split,
        plan.mode_table,
        router_input_width=plan.router.input_features,
    )
    if smallest_static is None:
        static_rank = None
        static_kind = None
        static_macs = None
        rank_ratio = None
        mac_ratio = None
    else:
        static_rank, static_kind = smallest_static
        static_macs = (
            2
            * basis.width
            * static_rank
            * int(split.attention_mask.sum().item())
        )
        rank_ratio = (
            accounting["average_active_modes"] / static_rank
            if static_rank > 0
            else math.inf
        )
        mac_ratio = (
            accounting["router_plus_projection_macs"] / static_macs
            if static_macs > 0
            else math.inf
        )
    diagnostic = {
        "baseline": asdict(
            variable_associative_metrics_from_logits(split, baseline_logits)
        ),
        "teacher_route_oracle": teacher_record,
        "learned_hidden_state_router": {
            **learned_record,
            "route_accounting": accounting,
        },
        "metadata_controls": control_records,
        "strongest_metadata_control": strongest_metadata_name,
        "position_length_stratified_shuffle": {
            **shuffle_record,
            "histograms_preserved": (
                route_histograms_by_stratum(
                    learned_grid,
                    {
                        "position": metadata["position"],
                        "length": metadata["length"],
                    },
                    valid_mask=split.attention_mask,
                    route_count=plan.mode_table.routes,
                )
                == route_histograms_by_stratum(
                    shuffled_grid,
                    {
                        "position": metadata["position"],
                        "length": metadata["length"],
                    },
                    valid_mask=split.attention_mask,
                    route_count=plan.mode_table.routes,
                )
            ),
        },
        "metadata_advantage_bootstrap": metadata_bootstrap,
        "shuffle_advantage_bootstrap": shuffle_bootstrap,
        "static_curve": static_curve,
        "smallest_passing_static": {
            "rank": static_rank,
            "kind": static_kind,
            "projection_macs": static_macs,
        },
        "average_rank_to_static_ratio": rank_ratio,
        "router_projection_to_static_mac_ratio": mac_ratio,
    }
    retained = {
        "baseline_logits": baseline_logits,
        "learned_logits": learned_logits,
        "teacher_logits": teacher_logits,
        "control_logits": control_logits,
        "shuffle_logits": shuffled_logits,
        "learned_grid": learned_grid,
        "teacher_grid": teacher_grid,
        "control_grids": control_grids,
        "shuffled_grid": shuffled_grid,
    }
    return diagnostic, retained


def _gate(
    analysis: Mapping[str, object],
    *,
    minimum_nll_advantage: float,
    maximum_relative_static_work: float,
    maximum_full_width_fallback_rate: float,
) -> dict[str, bool]:
    learned = analysis["learned_hidden_state_router"]
    teacher = analysis["teacher_route_oracle"]
    shuffle = analysis["position_length_stratified_shuffle"]
    metadata_bootstrap = analysis["metadata_advantage_bootstrap"]
    shuffle_bootstrap = analysis["shuffle_advantage_bootstrap"]
    accounting = learned["route_accounting"]  # type: ignore[index]
    return {
        "teacher_behavior": teacher["passed"] is True,  # type: ignore[index]
        "learned_behavior": learned["passed"] is True,  # type: ignore[index]
        "stratified_shuffle_histograms": (
            shuffle["histograms_preserved"] is True  # type: ignore[index]
        ),
        "metadata_nll_advantage": (
            float(metadata_bootstrap["mean_nll_advantage"])  # type: ignore[index]
            >= minimum_nll_advantage
        ),
        "metadata_bootstrap_lower_bound": (
            float(metadata_bootstrap["lower_95_percent"]) > 0  # type: ignore[index]
        ),
        "shuffle_nll_advantage": (
            float(shuffle_bootstrap["mean_nll_advantage"])  # type: ignore[index]
            >= minimum_nll_advantage
        ),
        "shuffle_bootstrap_lower_bound": (
            float(shuffle_bootstrap["lower_95_percent"]) > 0  # type: ignore[index]
        ),
        "static_comparator_exists": (
            analysis["smallest_passing_static"]["rank"] is not None  # type: ignore[index]
        ),
        "mean_rank_reduction": (
            analysis["average_rank_to_static_ratio"] is not None
            and float(analysis["average_rank_to_static_ratio"])
            <= maximum_relative_static_work
        ),
        "router_projection_mac_reduction": (
            analysis["router_projection_to_static_mac_ratio"] is not None
            and float(analysis["router_projection_to_static_mac_ratio"])
            <= maximum_relative_static_work
        ),
        "full_width_fallback_rate": (
            float(accounting["full_width_fallback_rate"])  # type: ignore[index]
            <= maximum_full_width_fallback_rate
        ),
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Tensor):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def run_variable_conditional_routing_experiment(
    *,
    checkpoint: Path = DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    output: Path = DEFAULT_OUTPUT,
    layer_index: int = DEFAULT_LAYER_INDEX,
    contexts_per_role: int = DEFAULT_CONTEXTS_PER_ROLE,
    route_budgets: Sequence[int] = DEFAULT_ROUTE_BUDGETS,
    route_quantiles: Sequence[float] | None = DEFAULT_ROUTE_QUANTILES,
    router_ridge: float = DEFAULT_ROUTER_RIDGE,
    router_class_balance_power: float = (
        DEFAULT_ROUTER_CLASS_BALANCE_POWER
    ),
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    minimum_nll_advantage: float = DEFAULT_MINIMUM_NLL_ADVANTAGE,
    maximum_relative_static_work: float = (
        DEFAULT_MAXIMUM_RELATIVE_STATIC_WORK
    ),
    maximum_full_width_fallback_rate: float = (
        DEFAULT_MAXIMUM_FULL_WIDTH_FALLBACK_RATE
    ),
) -> dict[str, object]:
    """Fit on A, gate on B, and touch validation only after a complete pass."""

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model, splits, checkpoint_metadata = (
        load_variable_associative_checkpoint(checkpoint_path)
    )
    model.eval()
    if (
        type(layer_index) is not int
        or not 0 <= layer_index < model.config.n_layers
    ):
        raise ValueError("layer_index is outside the source model")
    budgets = tuple(route_budgets)
    if (
        len(budgets) < 2
        or any(type(value) is not int for value in budgets)
        or tuple(sorted(set(budgets))) != budgets
        or budgets[0] < 0
        or budgets[-1] != model.config.d_model
    ):
        raise ValueError(
            "route budgets must be unique ascending and end at d_model"
        )
    if router_ridge <= 0:
        raise ValueError("router_ridge must be positive")
    if (
        not math.isfinite(router_class_balance_power)
        or not 0 <= router_class_balance_power <= 1
    ):
        raise ValueError("router_class_balance_power must be in [0, 1]")
    quantiles = (
        None
        if route_quantiles is None
        else tuple(float(value) for value in route_quantiles)
    )
    roles = _context_role_splits(
        splits.train,
        contexts_per_role=contexts_per_role,
    )
    input_site = f"layer.{layer_index}.input"
    output_site = f"layer.{layer_index}.output"
    source_fingerprint = module_state_fingerprint(model)

    basis, teacher, plan, controls, fit_report = _fit_policy(
        model,
        roles,
        input_site=input_site,
        output_site=output_site,
        route_budgets=budgets,
        route_quantiles=quantiles,
        router_ridge=router_ridge,
        router_class_balance_power=router_class_balance_power,
    )
    global_order = tuple(fit_report["global_need_order"])  # type: ignore[arg-type]
    calibration_b, _ = _evaluate_split(
        model,
        roles["calibration_b"],
        input_site=input_site,
        output_site=output_site,
        basis=basis,
        teacher=teacher,
        plan=plan,
        controls=controls,
        global_need_order=global_order,
        shuffle_seed=shuffle_seed,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    calibration_b_gates = _gate(
        calibration_b,
        minimum_nll_advantage=minimum_nll_advantage,
        maximum_relative_static_work=maximum_relative_static_work,
        maximum_full_width_fallback_rate=(
            maximum_full_width_fallback_rate
        ),
    )
    calibration_b_passed = all(calibration_b_gates.values())
    if calibration_b_passed:
        validation, _ = _evaluate_split(
            model,
            splits.validation,
            input_site=input_site,
            output_site=output_site,
            basis=basis,
            teacher=teacher,
            plan=plan,
            controls=controls,
            global_need_order=global_order,
            shuffle_seed=shuffle_seed + 1,
            bootstrap_seed=bootstrap_seed + 10,
            bootstrap_samples=bootstrap_samples,
        )
        validation_gates = _gate(
            validation,
            minimum_nll_advantage=minimum_nll_advantage,
            maximum_relative_static_work=maximum_relative_static_work,
            maximum_full_width_fallback_rate=(
                maximum_full_width_fallback_rate
            ),
        )
        validation_passed = all(validation_gates.values())
        validation_record: dict[str, object] = {
            "evaluated": True,
            "analysis": validation,
            "gates": validation_gates,
            "passed": validation_passed,
        }
    else:
        validation_passed = False
        validation_record = {
            "evaluated": False,
            "reason": "calibration_b_gate_failed",
            "conditional_candidate_test_untouched": True,
        }
    if module_state_fingerprint(model) != source_fingerprint:
        raise RuntimeError("conditional experiment mutated source weights")

    model_level_eligible = calibration_b_passed and validation_passed
    protocol = {
        "layer_index": layer_index,
        "input_site": input_site,
        "output_site": output_site,
        "contexts_per_role": contexts_per_role,
        "role_context_hashes": {
            name: split.semantic_context_hashes
            for name, split in roles.items()
        },
        "validation_context_hashes": splits.validation.semantic_context_hashes,
        "test_context_hashes": splits.test.semantic_context_hashes,
        "test_policy": "hash_only_not_model_evaluated_by_compiler_experiment",
        "route_budgets": budgets,
        "route_quantiles": (
            teacher.fit_quantiles
        ),
        "router_ridge": router_ridge,
        "router_class_balance_power": router_class_balance_power,
        "shuffle_seed": shuffle_seed,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "thresholds": {
            "minimum_nll_advantage": minimum_nll_advantage,
            "maximum_relative_static_work": maximum_relative_static_work,
            "maximum_full_width_fallback_rate": (
                maximum_full_width_fallback_rate
            ),
        },
        "native_layer_executes_in_representation_oracle": True,
        "model_level_escalation_fail_closed": True,
    }
    analysis = {
        "fit": fit_report,
        "calibration_b": {
            "analysis": calibration_b,
            "gates": calibration_b_gates,
            "passed": calibration_b_passed,
        },
        "validation": validation_record,
    }
    scientific_status = {
        "source_model_frozen": True,
        "variable_lengths": True,
        "semantic_context_roles_disjoint": True,
        "calibration_b_passed": calibration_b_passed,
        "validation_evaluated": calibration_b_passed,
        "validation_passed": validation_passed,
        "test_evaluated": False,
        "content_conditioned_routing_supported": model_level_eligible,
        "model_level_eligible": model_level_eligible,
        "native_layer_executed": True,
        "source_layer_compute_savings_supported": False,
    }
    payload = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "contains_model_weights": False,
        "source": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "model_state_fingerprint": source_fingerprint,
        },
        "protocol": protocol,
        "scientific_status": scientific_status,
        "basis": basis.state_dict(),
        "teacher": teacher.state_dict(),
        "routing_plan": plan.state_dict(),
        "metadata_controls": {
            name: control.state_dict()
            for name, control in controls.items()
        },
        "analysis": analysis,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report = {
        "artifact": str(output_path),
        "source_checkpoint": str(checkpoint_path),
        "scientific_status": scientific_status,
        "protocol": protocol,
        "analysis": analysis,
    }
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
    return copy.deepcopy(report)


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "route budgets must be comma-separated integers"
        ) from error


def _parse_quantiles(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "route quantiles must be comma-separated floats"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run variable-format conditional Fisher routing and fail closed "
            "before model-level training unless every control gate passes."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_VARIABLE_ASSOCIATIVE_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER_INDEX)
    parser.add_argument(
        "--contexts-per-role",
        type=int,
        default=DEFAULT_CONTEXTS_PER_ROLE,
    )
    parser.add_argument(
        "--route-budgets",
        type=_parse_budgets,
        default=DEFAULT_ROUTE_BUDGETS,
    )
    parser.add_argument(
        "--route-quantiles",
        type=_parse_quantiles,
        default=DEFAULT_ROUTE_QUANTILES,
    )
    parser.add_argument(
        "--router-ridge",
        type=float,
        default=DEFAULT_ROUTER_RIDGE,
    )
    parser.add_argument(
        "--router-class-balance-power",
        type=float,
        default=DEFAULT_ROUTER_CLASS_BALANCE_POWER,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_variable_conditional_routing_experiment(
        checkpoint=arguments.checkpoint,
        output=arguments.output,
        layer_index=arguments.layer_index,
        contexts_per_role=arguments.contexts_per_role,
        route_budgets=arguments.route_budgets,
        route_quantiles=arguments.route_quantiles,
        router_ridge=arguments.router_ridge,
        router_class_balance_power=(
            arguments.router_class_balance_power
        ),
        bootstrap_samples=arguments.bootstrap_samples,
    )
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
