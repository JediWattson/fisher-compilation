"""Real-Gemma same-layer shape-and-flow conditional-routing experiment.

The experiment is deliberately split in two durable phases:

``fit``
    Reuse the authenticated fit-Fisher analysis, fit four disjoint fragments
    on one physical Gemma MLP boundary, distil a top-1 source-state router,
    select a frozen polynomial candidate on open development, and save a
    source-safe candidate artifact.  The guard role is not opened.

``assess``
    Strict-load a promoted candidate, atomically claim the declared
    Calibration-A guard, open it once, and measure state flow plus native-model
    NLL/KL/top-1 and exact executed graph work.

This is a development experiment.  Even the guard is Calibration A, not a
fresh externally bound validation or test set.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_l3_l4_graph_organized_svd_shadow_protocol import (
    default_gemma3_l3_l4_graph_organized_svd_shadow_protocol,
)
from .gemma3_l3_l4_progressive_a_corpus import (
    Gemma3L3L4ProgressiveACorpus,
    Gemma3L3L4ProgressiveARolePrompts,
    load_gemma3_l3_l4_progressive_a_corpus,
)
from .gemma3_l3_l4_progressive_guard_ledger import (
    Gemma3L3L4ProgressiveGuardAlreadyClaimedError,
    claim_gemma3_l3_l4_progressive_guard,
    load_gemma3_l3_l4_progressive_guard_claim,
)
from .gemma3_modal_generator_dev_experiment import (
    ActivationRowFactory,
    DEFAULT_FIT_EXPORT,
    DEFAULT_GENERATOR_RANKS,
    DEFAULT_MODE_RANKS,
    DEFAULT_SELECTED_GENERATOR_RANK,
    DEFAULT_SELECTED_MODE_RANK,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    FittedModalGeneratorPilot,
    _layer_runtime_sites,
    _safe_tokenized_stream_metadata,
    _select_row_sites,
    fit_layer_cluster_modal_generator,
    load_development_prompt_export,
    load_gemma3_modal_generator_dev_artifact,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    DEFAULT_BASE_ARTIFACT,
    _bind_batch_example_ids,
    _restore_upstream_analysis,
    _validate_upstream_bindings,
)
from .gemma3_same_layer_shape_flow import (
    EdgelessSameLayerGraphPlan,
    SameLayerFragmentSelection,
    SameLayerShapeFlowRows,
    build_edgeless_same_layer_graph,
    collect_aligned_same_layer_fragment_rows,
    collect_edgeless_same_layer_shape_flow_rows,
    select_top_fisher_same_layer_fragments,
)
from .gemma3_state_conditioned_modal_graph_artifact import (
    load_gemma3_state_conditioned_modal_graph_candidate,
    save_gemma3_state_conditioned_modal_graph_candidate,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_compiler_pipeline import (
    ModalCompilerPipeline,
    build_modal_compiler_pipeline,
    build_modal_source_replacement_accounting,
)
from .modal_generator_graph import (
    ModalGeneratorGraphPlan,
    StateConditionedModalGeneratorInteraction,
)
from .modal_generator_lowering import ModalGeneratorLowering
from .modal_graph_rung_evaluation import (
    evaluate_modal_graph_rung_conditions,
)
from .modal_interaction_promotion import (
    ModalInteractionGraphPromotion,
    build_modal_interaction_graph_promotion,
)
from .state_conditioned_modal_evaluation import (
    StateConditionedModalFlowEvaluation,
    evaluate_state_conditioned_modal_flow,
)
from .state_conditioned_modal_fitting import (
    StateConditionedModalInteractionFit,
    fit_state_conditioned_modal_interactions,
    teacher_flow_routing,
)
from .streaming_analysis import iter_activation_score_gradient_rows


__all__ = [
    "DEFAULT_CANDIDATE_OUTPUT",
    "DEFAULT_GAIN_CANDIDATE_OUTPUT",
    "assess_gemma3_state_conditioned_shape_flow_candidate",
    "build_parser",
    "calibrate_gemma3_state_conditioned_shape_flow_gain",
    "fit_gemma3_state_conditioned_shape_flow_candidate",
    "main",
    "restore_gemma3_state_conditioned_shape_flow_runtime",
]


_LOCAL_ROOT = Path(".local-runs/google--gemma-3-270m")
DEFAULT_CORPUS_ARTIFACT = _LOCAL_ROOT / "progressive-a-fit-expanded-v1.corpus.json"
DEFAULT_CORPUS_FIT = _LOCAL_ROOT / "progressive-a-fit-expanded-v1.fit.json"
DEFAULT_CORPUS_SELECTION = _LOCAL_ROOT / "progressive-a-pilot-v1.selection.json"
# The expanded corpus artifact retains the pilot corpus id but binds the
# rotated v2 guard bytes published by the loss-v3 campaign.  The original
# pilot-v1 guard file is a different, correctly rejected manifest.
DEFAULT_CORPUS_GUARD = _LOCAL_ROOT / "progressive-a-loss-v3.guard.json"
DEFAULT_CANDIDATE_OUTPUT = (
    _LOCAL_ROOT / "state-conditioned-shape-flow-dev-v1.pt"
)
DEFAULT_GAIN_CANDIDATE_OUTPUT = (
    _LOCAL_ROOT / "state-conditioned-shape-flow-gain-dev-v2.pt"
)
DEFAULT_FRAGMENT_COUNT = 4
DEFAULT_MINIMUM_FRAGMENT_MODES = 32
DEFAULT_QUADRATIC_RANKS = (0, 4, 8)
DEFAULT_ROUTER_RIDGE = 1e-3
DEFAULT_MESSAGE_RIDGE = 1e-6
DEFAULT_QUADRATIC_STEPS = 300
DEFAULT_MINIMUM_NLL_IMPROVEMENT = 0.0
DEFAULT_GAIN_GRID = (
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    -0.125,
    -0.0625,
    0.0,
    0.0625,
    0.125,
    0.25,
    0.5,
    0.75,
    1.0,
)
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _progress(message: str) -> None:
    print(f"[same-layer-shape-flow] {message}", file=sys.stderr, flush=True)


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_float(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _canonical_positive_ints(
    values: Sequence[int],
    *,
    label: str,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    minimum = 0 if allow_zero else 1
    if (
        not result
        or any(type(value) is not int or value < minimum for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(f"{label} must be unique and increasing")
    return result


def _canonical_finite_floats(
    values: Sequence[float],
    *,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not math.isfinite(value) for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(f"{label} must be finite, unique, and increasing")
    return result


def _tokenizer_contract() -> dict[str, object]:
    protocol = default_gemma3_l3_l4_graph_organized_svd_shadow_protocol()
    protocol.validate_integrity()
    metadata = protocol.metadata()
    tokenizer = metadata.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("frozen protocol tokenizer contract is unavailable")
    return dict(tokenizer)


def _load_corpus(
    *,
    corpus_artifact_path: Path | str,
    corpus_fit_path: Path | str,
    corpus_selection_path: Path | str,
    corpus_guard_path: Path | str,
) -> Gemma3L3L4ProgressiveACorpus:
    return load_gemma3_l3_l4_progressive_a_corpus(
        corpus_artifact_path,
        role_input_paths={
            "calibration_a_fit": corpus_fit_path,
            "calibration_a_selection": corpus_selection_path,
            "calibration_a_guard": corpus_guard_path,
        },
        tokenizer_contract=_tokenizer_contract(),
    )


def _materialize_role(
    tokenizer: object,
    role: Gemma3L3L4ProgressiveARolePrompts,
    *,
    split_name: str,
    max_length: int,
    tokenization_batch_size: int,
    device: torch.device,
) -> tuple[tuple[CalibrationBatch, ...], dict[str, object]]:
    batches, stream = _materialize_split(
        tokenizer,
        role.prompts,
        split_name=split_name,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    return (
        _bind_batch_example_ids(batches, role.ordered_prompt_sha256s),
        stream,
    )


def _collect_same_layer_native_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    selection: SameLayerFragmentSelection,
    leaf_activation_site: str,
    row_factory: ActivationRowFactory = iter_activation_score_gradient_rows,
):
    sites = tuple(
        dict.fromkeys(
            site
            for fragment in selection.execution_order
            for site in (fragment.input_site, fragment.activation_site)
        )
    )
    down_weights: dict[str, Tensor] = {}
    for fragment in selection.execution_order:
        input_site, output_site, _, down_weight = _layer_runtime_sites(
            adapter,
            fragment.layer_ordinal,
        )
        if input_site != fragment.input_site or output_site != fragment.output_site:
            raise ValueError("selected same-layer fragment runtime sites drifted")
        down_weights[fragment.fragment_id] = down_weight
    requested = tuple(dict.fromkeys((*sites, leaf_activation_site)))
    raw_rows = row_factory(
        adapter,
        batches,
        activation_names=requested,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_activation_site,
        accumulation_dtype=torch.float64,
    )
    return collect_aligned_same_layer_fragment_rows(
        _select_row_sites(raw_rows, sites),
        selection=selection,
        down_projection_weights=down_weights,
    )


def _family_ids_for_rows(
    row_keys: Sequence[tuple[str, int]],
    role: Gemma3L3L4ProgressiveARolePrompts,
) -> tuple[str, ...]:
    family_by_example = role.family_by_example
    try:
        return tuple(family_by_example[example_id] for example_id, _ in row_keys)
    except KeyError as error:
        raise ValueError("runtime row does not belong to the declared role") from error


def _row_keys_for_batches(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for batch in batches:
        if batch.example_ids is None:
            raise ValueError("row-key materialization requires example ids")
        context = adapter.prepare_sequence(batch.model_inputs)
        valid = batch.valid_positions.to(device=context.logical_positions.device)
        for index, example_id in enumerate(batch.example_ids):
            positions = context.logical_positions[index, valid[index]]
            result.extend(
                (example_id, int(position))
                for position in positions.detach().cpu().tolist()
            )
    return tuple(result)


def _flow_inputs(
    rows: SameLayerShapeFlowRows,
    plan: EdgelessSameLayerGraphPlan,
) -> tuple[
    str,
    tuple[str, ...],
    dict[str, Tensor],
    dict[str, Tensor],
]:
    source_node = plan.node_names[0]
    target_nodes = tuple(sorted(plan.node_names[1:]))
    corrections = {
        name: rows.teacher_coordinates[name] - rows.node_states[name]
        for name in target_nodes
    }
    decoders = {
        name: plan.lowerings_by_node[
            name
        ].computational_mode_basis.decoder_basis
        for name in target_nodes
    }
    return source_node, target_nodes, corrections, decoders


def _oracle_labels(
    rows: SameLayerShapeFlowRows,
    plan: EdgelessSameLayerGraphPlan,
) -> Tensor:
    _, target_nodes, corrections, decoders = _flow_inputs(rows, plan)
    candidates = torch.stack(
        tuple(corrections[name] @ decoders[name] for name in target_nodes),
        dim=1,
    )
    return teacher_flow_routing(
        candidates.sum(dim=1),
        candidates,
        alignment_weight=1.0,
        residual_weight=1.0,
        temperature=1.0,
    ).route_labels


def _target_fisher_weights(native_rows, plan: EdgelessSameLayerGraphPlan) -> Tensor:
    weights = tuple(
        native_rows.rows_by_fragment[plan.fragment_id_by_node[name]].fisher_weights
        for name in plan.node_names[1:]
    )
    return torch.stack(weights, dim=0).sum(dim=0)


def _evaluate_flow(
    fit: StateConditionedModalInteractionFit,
    *,
    rows: SameLayerShapeFlowRows,
    plan: EdgelessSameLayerGraphPlan,
    row_weights: Tensor | None,
    family_ids: Sequence[str] | None,
) -> StateConditionedModalFlowEvaluation:
    source_node, _, corrections, decoders = _flow_inputs(rows, plan)
    return evaluate_state_conditioned_modal_flow(
        rows.node_states[source_node],
        corrections,
        decoders,
        fit.interactions,
        row_weights=row_weights,
        family_ids=family_ids,
    )


def _fit_metadata(fit: StateConditionedModalInteractionFit) -> dict[str, object]:
    return {
        "parameter_count": fit.parameter_count,
        "dense_macs_per_token": fit.dense_macs_per_token,
        "route_counts": fit.route_counts,
        "router_metrics": asdict(fit.router_metrics),
        "edge_metrics": tuple(asdict(value) for value in fit.edge_metrics),
        "router_fold_audit_observations": fit.router_fold_audit_observations,
        "router_fold_float32_max_abs_error": (
            fit.router_fold_float32_max_abs_error
        ),
        "router_fold_float32_tolerance": fit.router_fold_float32_tolerance,
    }


def _dynamic_graph(
    edgeless: EdgelessSameLayerGraphPlan,
    fit: StateConditionedModalInteractionFit,
) -> ModalGeneratorGraphPlan:
    return ModalGeneratorGraphPlan(
        model_fingerprint=edgeless.graph_plan.model_fingerprint,
        parameter_cluster_plan_sha256=(
            edgeless.graph_plan.parameter_cluster_plan_sha256
        ),
        nodes=edgeless.graph_plan.nodes,
        interactions=tuple(
            sorted(
                fit.interactions,
                key=lambda edge: (edge.source_node, edge.target_node),
            )
        ),
    )


def _scaled_state_conditioned_graph(
    graph: ModalGeneratorGraphPlan,
    gain: float,
) -> ModalGeneratorGraphPlan:
    """Scale a frozen conditional velocity field without changing routing.

    Scaling the affine matrix, affine bias, and quadratic output factor by the
    same signed scalar multiplies every proposed modal displacement exactly.
    Gate coefficients and therefore tokenwise route choices remain frozen.
    """

    graph.validate_integrity()
    scale = float(gain)
    if not math.isfinite(scale):
        raise ValueError("state-conditioned gain must be finite")
    if not graph.interactions or any(
        not isinstance(edge, StateConditionedModalGeneratorInteraction)
        for edge in graph.interactions
    ):
        raise ValueError(
            "gain calibration requires an all-state-conditioned graph"
        )
    interactions = tuple(
        StateConditionedModalGeneratorInteraction(
            source_node=edge.source_node,
            target_node=edge.target_node,
            routing_group=edge.routing_group,
            message_matrix=edge.message_matrix * scale,
            message_bias=edge.message_bias * scale,
            gate_weight=edge.gate_weight,
            gate_bias=edge.gate_bias,
            quadratic_left=edge.quadratic_left,
            quadratic_right=edge.quadratic_right,
            quadratic_output=(
                None
                if edge.quadratic_output is None
                else edge.quadratic_output * scale
            ),
            temperature=edge.temperature,
            top_k=edge.top_k,
        )
        for edge in graph.interactions
    )
    return ModalGeneratorGraphPlan(
        model_fingerprint=graph.model_fingerprint,
        parameter_cluster_plan_sha256=graph.parameter_cluster_plan_sha256,
        nodes=graph.nodes,
        interactions=interactions,
    )


def _condition(rung: Mapping[str, object], name: str) -> Mapping[str, object]:
    conditions = rung.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("rung evaluation conditions are unavailable")
    value = conditions.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"rung condition {name!r} is unavailable")
    return value


def _stream_sha256(stream: Mapping[str, object], *, label: str) -> str:
    safe = _safe_tokenized_stream_metadata(stream)
    return _require_sha256(safe.get("serialized_sha256"), label=label)


def _source_safe_split_metadata(
    *,
    upstream: Mapping[str, object],
    corpus: Gemma3L3L4ProgressiveACorpus,
    fit_split_sha256: str,
    selection_split_sha256: str,
    fit_family_ids: Sequence[str],
) -> dict[str, object]:
    selection = corpus.preclaim_view("calibration_a_selection")
    guard = corpus.preclaim_view("calibration_a_guard")
    fit_families = tuple(sorted(set(fit_family_ids)))
    return {
        "fit_split_sha256": fit_split_sha256,
        "eval_split_sha256": selection_split_sha256,
        "selection_split_sha256": selection_split_sha256,
        "fit_example_count": len(fit_family_ids),
        "fit_family_ids": fit_families,
        "selection_role_manifest_sha256": selection.manifest_sha256,
        "selection_example_count": selection.example_count,
        "selection_family_ids": selection.family_ids,
        "guard_role_manifest_sha256": guard.manifest_sha256,
        "guard_example_count": guard.example_count,
        "guard_family_ids": guard.family_ids,
        "exact_role_identity_overlap_count": len(
            set(selection.ordered_prompt_sha256s)
            & set(guard.ordered_prompt_sha256s)
        ),
        "declared_family_overlap_count": len(
            set(fit_families)
            & (set(selection.family_ids) | set(guard.family_ids))
        ),
        "corpus_artifact_sha256": corpus.artifact.artifact_sha256,
        "tokenizer_contract_sha256": (
            corpus.artifact.tokenizer_contract_sha256
        ),
        "source_analysis_sha256": _require_sha256(
            upstream.get("scientific_payload_sha256"),
            label="source analysis",
        ),
        "guard_opened_during_fit": False,
    }


def fit_gemma3_state_conditioned_shape_flow_candidate(
    *,
    revision: str,
    output: Path | str = DEFAULT_CANDIDATE_OUTPUT,
    fit_export_path: Path | str = DEFAULT_FIT_EXPORT,
    base_artifact_path: Path | str = DEFAULT_BASE_ARTIFACT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_ARTIFACT,
    corpus_fit_path: Path | str = DEFAULT_CORPUS_FIT,
    corpus_selection_path: Path | str = DEFAULT_CORPUS_SELECTION,
    corpus_guard_path: Path | str = DEFAULT_CORPUS_GUARD,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    fragment_count: int = DEFAULT_FRAGMENT_COUNT,
    minimum_fragment_modes: int = DEFAULT_MINIMUM_FRAGMENT_MODES,
    layer_ordinal: int | None = None,
    mode_ranks: Sequence[int] = DEFAULT_MODE_RANKS,
    selected_mode_rank: int = DEFAULT_SELECTED_MODE_RANK,
    generator_ranks: Sequence[int] = DEFAULT_GENERATOR_RANKS,
    selected_generator_rank: int = DEFAULT_SELECTED_GENERATOR_RANK,
    generator_ridge: float = 0.0,
    quadratic_ranks: Sequence[int] = DEFAULT_QUADRATIC_RANKS,
    router_ridge: float = DEFAULT_ROUTER_RIDGE,
    message_ridge: float = DEFAULT_MESSAGE_RIDGE,
    quadratic_steps: int = DEFAULT_QUADRATIC_STEPS,
    minimum_nll_improvement: float = DEFAULT_MINIMUM_NLL_IMPROVEMENT,
) -> dict[str, object]:
    """Fit, select, and checkpoint one guard-unopened Gemma candidate."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    destination = Path(output)
    if destination.suffix != ".pt":
        raise ValueError("candidate output must use .pt")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite candidate output")
    _positive_int(tokenization_batch_size, label="tokenization_batch_size")
    _positive_int(fragment_count, label="fragment_count")
    if fragment_count < 3:
        raise ValueError("fragment_count must provide one source and two targets")
    _positive_int(minimum_fragment_modes, label="minimum_fragment_modes")
    _positive_int(quadratic_steps, label="quadratic_steps")
    modes = _canonical_positive_ints(mode_ranks, label="mode_ranks")
    generators = _canonical_positive_ints(
        generator_ranks,
        label="generator_ranks",
    )
    quadratic = _canonical_positive_ints(
        quadratic_ranks,
        label="quadratic_ranks",
        allow_zero=True,
    )
    if selected_mode_rank not in modes:
        raise ValueError("selected_mode_rank must be in mode_ranks")
    if selected_generator_rank not in generators:
        raise ValueError("selected_generator_rank must be in generator_ranks")
    if layer_ordinal is not None and (
        type(layer_ordinal) is not int or layer_ordinal < 0
    ):
        raise ValueError("layer_ordinal must be nonnegative")
    generator_ridge = _nonnegative_float(
        generator_ridge,
        label="generator_ridge",
    )
    router_ridge = _nonnegative_float(router_ridge, label="router_ridge")
    message_ridge = _nonnegative_float(message_ridge, label="message_ridge")
    if router_ridge <= 0.0:
        raise ValueError("router_ridge must be positive")
    minimum_nll_improvement = _nonnegative_float(
        minimum_nll_improvement,
        label="minimum_nll_improvement",
    )
    _progress("preflight: authenticate source analysis and sealed A roles")
    upstream = load_gemma3_modal_generator_dev_artifact(base_artifact_path)
    fit_export = load_development_prompt_export(fit_export_path)
    upstream_splits = upstream.get("splits")
    if (
        not isinstance(upstream_splits, Mapping)
        or upstream_splits.get("fit_export") != fit_export.metadata()
    ):
        raise ValueError("fit export does not match the source analysis")
    corpus = _load_corpus(
        corpus_artifact_path=corpus_artifact_path,
        corpus_fit_path=corpus_fit_path,
        corpus_selection_path=corpus_selection_path,
        corpus_guard_path=corpus_guard_path,
    )
    selection_role = corpus.open_development_role(
        "calibration_a_selection"
    )
    if corpus.guard_opened or corpus.guard_consumed:
        raise RuntimeError("candidate fitting opened the guard unexpectedly")

    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    same_layer = select_top_fisher_same_layer_fragments(
        fragment_plan,
        count=fragment_count,
        minimum_fragment_modes=minimum_fragment_modes,
        layer_ordinal=layer_ordinal,
    )
    if any(
        selected_mode_rank > fragment.mode_count
        for fragment in same_layer.execution_order
    ):
        raise ValueError("selected mode rank exceeds a selected fragment")

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma checkpoint from external cache")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    model_fingerprint = adapter.model_fingerprint()
    _validate_upstream_bindings(
        upstream,
        model_id=model_id,
        revision=revision,
        model_fingerprint=model_fingerprint,
    )
    tokenizer_contract = _tokenizer_contract()
    max_length = int(tokenizer_contract["max_length"])

    _progress("tokenize: fit40 plus four-family selection; guard remains sealed")
    fit_batches, fit_stream = _materialize_split(
        tokenizer,
        fit_export.prompts,
        split_name="modal_generator_development_fit",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fit_batches = _bind_batch_example_ids(
        fit_batches,
        fit_export.prompt_sha256s,
    )
    selection_batches, selection_stream = _materialize_role(
        tokenizer,
        selection_role,
        split_name="same_layer_shape_flow_selection",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fit_safe = _safe_tokenized_stream_metadata(fit_stream)
    upstream_fit = upstream_splits.get("fit_tokenized")
    if fit_safe != upstream_fit:
        raise ValueError("live fit tokenization differs from source analysis")
    fit_split_sha256 = _require_sha256(
        fit_safe.get("serialized_sha256"),
        label="fit split",
    )
    selection_split_sha256 = _stream_sha256(
        selection_stream,
        label="selection split",
    )
    if fit_split_sha256 == selection_split_sha256:
        raise ValueError("fit and selection tokenized streams overlap")
    live_layer_specs, leaf_activation_site, _ = _whole_model_layer_specs(
        adapter
    )
    if tuple(spec.layer_id for spec in live_layer_specs) != tuple(
        spec.layer_id for spec in fit_trace.layer_specs
    ):
        raise ValueError("live layer catalog differs from source analysis")

    _progress(
        "rows: replay real activation/gradient rows for four same-layer fragments"
    )
    fit_native = _collect_same_layer_native_rows(
        adapter,
        fit_batches,
        selection=same_layer,
        leaf_activation_site=leaf_activation_site,
    )
    selection_native = _collect_same_layer_native_rows(
        adapter,
        selection_batches,
        selection=same_layer,
        leaf_activation_site=leaf_activation_site,
    )

    _progress(
        f"nodes: fit {fragment_count} generators on layer {same_layer.layer_ordinal}"
    )
    pilots: dict[str, FittedModalGeneratorPilot] = {}
    for fragment in same_layer.execution_order:
        fragment_mode_ranks = tuple(
            rank for rank in modes if rank <= fragment.mode_count
        )
        if selected_mode_rank not in fragment_mode_ranks:
            raise ValueError("rank clipping removed selected_mode_rank")
        pilots[fragment.fragment_id] = fit_layer_cluster_modal_generator(
            fit_native.rows_by_fragment[fragment.fragment_id],
            selection_native.rows_by_fragment[fragment.fragment_id],
            selection=fragment,
            source_model_sha256=model_fingerprint,
            parameter_catalog_sha256=catalog.artifact_sha256,
            fisher_coupling_sha256=fisher.artifact_sha256,
            fragment_plan=fragment_plan,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=selection_split_sha256,
            input_site=fragment.input_site,
            output_site=fragment.output_site,
            mode_ranks=fragment_mode_ranks,
            selected_mode_rank=selected_mode_rank,
            generator_ranks=generators,
            selected_generator_rank=selected_generator_rank,
            ridge=generator_ridge,
        )
    edgeless = build_edgeless_same_layer_graph(
        same_layer,
        fragment_plan=fragment_plan,
        lowerings_by_fragment={
            fragment_id: pilot.lowering
            for fragment_id, pilot in pilots.items()
        },
    )
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless.graph_plan,
        edgeless.lowerings,
    )

    _progress("states: capture realized edgeless shape and native flow")
    fit_rows = collect_edgeless_same_layer_shape_flow_rows(
        adapter,
        edgeless_executor,
        fit_batches,
        plan=edgeless,
        expected_row_keys=fit_native.row_keys,
    )
    selection_rows = collect_edgeless_same_layer_shape_flow_rows(
        adapter,
        edgeless_executor,
        selection_batches,
        plan=edgeless,
        expected_row_keys=selection_native.row_keys,
    )
    source_node, target_nodes, fit_corrections, _ = _flow_inputs(
        fit_rows,
        edgeless,
    )
    labels = _oracle_labels(fit_rows, edgeless)
    counts = torch.bincount(labels, minlength=len(target_nodes))
    if bool((counts == 0).any()):
        raise RuntimeError(
            "same-layer oracle did not use every predeclared target route"
        )
    selection_families = _family_ids_for_rows(
        selection_rows.row_keys,
        selection_role,
    )
    selection_weights = _target_fisher_weights(selection_native, edgeless)

    _progress(
        "edges: fit affine and factorized-quadratic source-state variants"
    )
    variants: list[dict[str, object]] = []
    for quadratic_rank in quadratic:
        fitted = fit_state_conditioned_modal_interactions(
            fit_rows.node_states[source_node],
            fit_corrections,
            labels,
            source_node=source_node,
            routing_group=(
                f"layer-{same_layer.layer_ordinal}.shape-flow-v1"
            ),
            router_validation_states=selection_rows.node_states[source_node],
            temperature=1.0,
            top_k=1,
            router_ridge=router_ridge,
            message_ridge=message_ridge,
            quadratic_rank=quadratic_rank,
            quadratic_steps=quadratic_steps,
            quadratic_learning_rate=1e-2,
            quadratic_ridge=1e-6,
            seed=17_000 + quadratic_rank,
        )
        graph = _dynamic_graph(edgeless, fitted)
        flow = _evaluate_flow(
            fitted,
            rows=selection_rows,
            plan=edgeless,
            row_weights=selection_weights,
            family_ids=selection_families,
        )
        variants.append(
            {
                "quadratic_rank": quadratic_rank,
                "fit": fitted,
                "graph": graph,
                "flow": flow,
            }
        )

    _progress("selection: run end-to-end NLL/KL/top-1 for each frozen variant")
    baseline_nll: float | None = None
    for variant in variants:
        graph = variant["graph"]
        assert isinstance(graph, ModalGeneratorGraphPlan)
        executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            graph,
            edgeless.lowerings,
        )
        rung = evaluate_modal_graph_rung_conditions(
            adapter,
            executor,
            edgeless_executor,
            selection_batches,
            assessment_role="open_development_assessment",
            expected_example_ids=selection_role.ordered_prompt_sha256s,
        )
        edgeless_condition = _condition(rung, "edgeless_graph")
        dynamic_condition = _condition(rung, "interacting_graph")
        current_baseline = float(edgeless_condition["nll_per_token"])
        if baseline_nll is None:
            baseline_nll = current_baseline
        elif baseline_nll != current_baseline:
            raise RuntimeError("selection edgeless NLL changed across variants")
        variant["rung"] = rung
        variant["candidate_nll"] = float(dynamic_condition["nll_per_token"])
        variant["nll_improvement"] = (
            current_baseline - float(dynamic_condition["nll_per_token"])
        )
    assert baseline_nll is not None
    chosen = min(
        variants,
        key=lambda value: (
            float(value["candidate_nll"]),
            value["flow"].routed_graph.weighted_nrmse,  # type: ignore[union-attr]
            int(value["quadratic_rank"]),
            value["graph"].artifact_sha256,  # type: ignore[union-attr]
        ),
    )
    chosen_fit = chosen["fit"]
    chosen_graph = chosen["graph"]
    chosen_flow = chosen["flow"]
    chosen_rung = chosen["rung"]
    assert isinstance(chosen_fit, StateConditionedModalInteractionFit)
    assert isinstance(chosen_graph, ModalGeneratorGraphPlan)
    assert isinstance(chosen_flow, StateConditionedModalFlowEvaluation)
    assert isinstance(chosen_rung, Mapping)
    nll_improvement = float(chosen["nll_improvement"])
    promoted = nll_improvement > max(0.0, minimum_nll_improvement)

    promotion: ModalInteractionGraphPromotion | None = None
    pipeline: ModalCompilerPipeline | None = None
    if promoted:
        promotion = build_modal_interaction_graph_promotion(
            chosen_graph,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=selection_split_sha256,
            selection_metric="nll_per_token",
            baseline_metric_value=baseline_nll,
            candidate_metric_value=float(chosen["candidate_nll"]),
            minimum_heldout_improvement=minimum_nll_improvement,
            heldout_observations=int(chosen_rung["supervised_tokens"]),
        )
        accounting = build_modal_source_replacement_accounting(
            catalog,
            fragment_plan,
            same_layer.fragment_ids,
        )
        pipeline = build_modal_compiler_pipeline(
            source_prompt_trace=fit_trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragment_plan,
            lowerings_by_node=edgeless.lowerings_by_node,
            graph_plan=chosen_graph,
            interaction_selection=promotion,
            source_replacement_accounting=accounting,
        )

    split_metadata = _source_safe_split_metadata(
        upstream=upstream,
        corpus=corpus,
        fit_split_sha256=fit_split_sha256,
        selection_split_sha256=selection_split_sha256,
        fit_family_ids=fit_export.family_ids,
    )
    variant_receipts = tuple(
        {
            "quadratic_rank": int(value["quadratic_rank"]),
            "dynamic_graph_sha256": value["graph"].artifact_sha256,
            "fit": _fit_metadata(value["fit"]),
            "flow": value["flow"].metadata(),
            "behavior": {
                "native": value["rung"]["native"],
                "conditions": value["rung"]["conditions"],
                "graph_comparison": value["rung"]["graph_comparison"],
                "supervised_tokens": value["rung"]["supervised_tokens"],
                "logical_valid_tokens": value["rung"]["logical_valid_tokens"],
            },
            "nll_improvement_over_edgeless": float(
                value["nll_improvement"]
            ),
        }
        for value in variants
    )
    resources = {
        **dict(chosen_rung["resource_accounting"]),
        "edgeless_parameter_count": edgeless.graph_plan.parameter_count,
        "dynamic_parameter_count": chosen_graph.parameter_count,
        "edgeless_dense_macs_per_token": edgeless.graph_plan.macs_per_token,
        "dynamic_dense_macs_per_token": chosen_graph.macs_per_token,
        "conditional_routing_macs_per_token": (
            chosen_graph.conditional_routing_macs_per_token
        ),
        "conditional_dense_message_macs_per_token": (
            chosen_graph.conditional_dense_message_macs_per_token
        ),
        "conditional_selected_message_macs_per_token_upper_bound": (
            chosen_graph.conditional_selected_message_macs_per_token_upper_bound
        ),
        "source_parameter_count": (
            None if pipeline is None else pipeline.source_parameter_count
        ),
        "source_macs_per_token": (
            None if pipeline is None else pipeline.source_macs_per_token
        ),
        "net_parameter_savings": (
            None if pipeline is None else pipeline.net_parameter_savings
        ),
        "net_dense_macs_saved_per_token": (
            None if pipeline is None else pipeline.net_macs_saved_per_token
        ),
    }
    report = save_gemma3_state_conditioned_modal_graph_candidate(
        destination,
        experiment={
            "experiment_kind": "real_gemma_same_layer_shape_flow_v1",
            "scientific_role": "open_development_candidate_selection",
            "model_id": model_id,
            "requested_revision": revision,
            "adapter_model_fingerprint": model_fingerprint,
            "source_model_unchanged": True,
            "guard_status": "sealed_unopened",
            "heldout_confirmation": False,
        },
        config={
            "fragment_selection": same_layer.metadata(),
            "fragment_count": fragment_count,
            "minimum_fragment_modes": minimum_fragment_modes,
            "selected_mode_rank": selected_mode_rank,
            "selected_generator_rank": selected_generator_rank,
            "mode_ranks": modes,
            "generator_ranks": generators,
            "quadratic_ranks": quadratic,
            "router_ridge": router_ridge,
            "message_ridge": message_ridge,
            "quadratic_steps": quadratic_steps,
            "top_k": 1,
            "minimum_nll_improvement": minimum_nll_improvement,
        },
        splits=split_metadata,
        selection={
            "selection_split_sha256": selection_split_sha256,
            "dynamic_graph_sha256": chosen_graph.artifact_sha256,
            "edgeless_graph_sha256": edgeless.graph_plan.artifact_sha256,
            "compiler_pipeline_sha256": (
                None if pipeline is None else pipeline.artifact_sha256
            ),
            "interaction_promotion_sha256": (
                None if promotion is None else promotion.artifact_sha256
            ),
            "chosen_quadratic_rank": int(chosen["quadratic_rank"]),
            "edgeless_nll_per_token": baseline_nll,
            "candidate_nll_per_token": float(chosen["candidate_nll"]),
            "nll_improvement_over_edgeless": nll_improvement,
            "promotion_passed": promoted,
            "guard_opened": False,
            "variants": variant_receipts,
        },
        resources=resources,
        lowerings_by_node=edgeless.lowerings_by_node,
        edgeless_graph=edgeless.graph_plan,
        dynamic_graph=chosen_graph,
        compiler_pipeline=pipeline,
    )
    if adapter.model_fingerprint() != model_fingerprint:
        raise RuntimeError("shape-flow experiment mutated the source model")
    _progress(
        f"candidate: {'promoted' if promoted else 'not promoted'}; wrote "
        f"{destination}"
    )
    return report


def _restore_candidate_graphs(
    raw: Mapping[str, object],
) -> tuple[
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
]:
    dynamic_state = raw.get("dynamic_graph")
    edgeless_state = raw.get("edgeless_graph")
    if not isinstance(dynamic_state, Mapping) or not isinstance(
        edgeless_state,
        Mapping,
    ):
        raise TypeError("candidate graph states are invalid")
    dynamic = ModalGeneratorGraphPlan.from_state_dict(dynamic_state)
    edgeless = ModalGeneratorGraphPlan.from_state_dict(edgeless_state)
    records = raw.get("lowering_records")
    if type(records) is not tuple:
        raise TypeError("candidate lowering records are invalid")
    lowerings: dict[str, ModalGeneratorLowering] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("candidate lowering record is invalid")
        name = record.get("node_name")
        state = record.get("lowering")
        if not isinstance(name, str) or not isinstance(state, Mapping):
            raise TypeError("candidate lowering record fields are invalid")
        lowerings[name] = ModalGeneratorLowering.from_state_dict(state)
    if set(lowerings) != {node.name for node in dynamic.nodes}:
        raise ValueError("candidate lowering catalog differs from its graph")
    return edgeless, dynamic, lowerings


def _restore_candidate_runtime(
    raw: Mapping[str, object],
) -> tuple[
    ModalCompilerPipeline,
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
]:
    edgeless, dynamic, lowerings = _restore_candidate_graphs(raw)
    pipeline_state = raw.get("compiler_pipeline")
    if pipeline_state is None:
        raise ValueError("candidate did not pass compiler promotion")
    if not isinstance(pipeline_state, Mapping):
        raise TypeError("candidate compiler pipeline state is invalid")
    pipeline = ModalCompilerPipeline.from_state_dict(pipeline_state)
    if not isinstance(
        pipeline.interaction_selection,
        ModalInteractionGraphPromotion,
    ):
        raise ValueError("candidate lacks a conditional graph promotion")
    if pipeline.graph_plan.artifact_sha256 != dynamic.artifact_sha256:
        raise ValueError("candidate runtime lineage differs from its pipeline")
    return pipeline, edgeless, dynamic, lowerings


def restore_gemma3_state_conditioned_shape_flow_runtime(
    raw: Mapping[str, object],
) -> tuple[
    ModalCompilerPipeline,
    ModalGeneratorGraphPlan,
    ModalGeneratorGraphPlan,
    dict[str, ModalGeneratorLowering],
]:
    """Strictly restore a promoted candidate for read-only external evals."""

    return _restore_candidate_runtime(raw)


def calibrate_gemma3_state_conditioned_shape_flow_gain(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE_OUTPUT,
    output: Path | str = DEFAULT_GAIN_CANDIDATE_OUTPUT,
    base_artifact_path: Path | str = DEFAULT_BASE_ARTIFACT,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_ARTIFACT,
    corpus_fit_path: Path | str = DEFAULT_CORPUS_FIT,
    corpus_selection_path: Path | str = DEFAULT_CORPUS_SELECTION,
    corpus_guard_path: Path | str = DEFAULT_CORPUS_GUARD,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    gains: Sequence[float] = DEFAULT_GAIN_GRID,
    minimum_nll_improvement: float = DEFAULT_MINIMUM_NLL_IMPROVEMENT,
) -> dict[str, object]:
    """Select a signed trust-region gain on the already-frozen edge field.

    This phase reuses only the open-development selection role. It does not
    refit routes or messages and cannot open the declared guard.
    """

    destination = Path(output)
    if destination.suffix != ".pt":
        raise ValueError("gain candidate output must use .pt")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite gain candidate output")
    gain_grid = _canonical_finite_floats(gains, label="gains")
    if 0.0 not in gain_grid:
        raise ValueError("gain calibration must include the zero control")
    _positive_int(tokenization_batch_size, label="tokenization_batch_size")
    minimum_nll_improvement = _nonnegative_float(
        minimum_nll_improvement,
        label="minimum_nll_improvement",
    )

    _progress("gain preflight: strict-load failed candidate; guard stays sealed")
    raw = load_gemma3_state_conditioned_modal_graph_candidate(candidate_path)
    experiment = raw.get("experiment")
    config = raw.get("config")
    splits = raw.get("splits")
    parent_selection = raw.get("selection")
    if not all(
        isinstance(value, Mapping)
        for value in (experiment, config, splits, parent_selection)
    ):
        raise TypeError("candidate source-safe metadata is invalid")
    assert isinstance(experiment, Mapping)
    assert isinstance(config, Mapping)
    assert isinstance(splits, Mapping)
    assert isinstance(parent_selection, Mapping)
    if (
        raw.get("compiler_pipeline") is not None
        or parent_selection.get("promotion_passed") is not False
        or parent_selection.get("guard_opened") is not False
        or experiment.get("guard_status") != "sealed_unopened"
    ):
        raise ValueError(
            "gain calibration requires one unpromoted guard-unopened candidate"
        )
    revision = experiment.get("requested_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("candidate revision is invalid")
    if experiment.get("model_id") != model_id:
        raise ValueError("candidate model id differs from calibration model")
    edgeless_graph, parent_graph, lowerings = _restore_candidate_graphs(raw)

    corpus = _load_corpus(
        corpus_artifact_path=corpus_artifact_path,
        corpus_fit_path=corpus_fit_path,
        corpus_selection_path=corpus_selection_path,
        corpus_guard_path=corpus_guard_path,
    )
    selection_view = corpus.preclaim_view("calibration_a_selection")
    if (
        splits.get("corpus_artifact_sha256")
        != corpus.artifact.artifact_sha256
        or splits.get("selection_role_manifest_sha256")
        != selection_view.manifest_sha256
    ):
        raise ValueError("gain candidate does not bind this selection corpus")
    selection_role = corpus.open_development_role(
        "calibration_a_selection"
    )
    if corpus.guard_opened or corpus.guard_consumed:
        raise RuntimeError("gain calibration opened the guard unexpectedly")

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("gain model: load pinned Gemma and materialize selection only")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    model_fingerprint = adapter.model_fingerprint()
    if model_fingerprint != experiment.get("adapter_model_fingerprint"):
        raise ValueError("gain model fingerprint differs from candidate")
    upstream = load_gemma3_modal_generator_dev_artifact(base_artifact_path)
    if splits.get("source_analysis_sha256") != upstream.get(
        "scientific_payload_sha256"
    ):
        raise ValueError("parent candidate source analysis binding drifted")
    _validate_upstream_bindings(
        upstream,
        model_id=model_id,
        revision=revision,
        model_fingerprint=model_fingerprint,
    )
    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    selection_metadata = config.get("fragment_selection")
    if not isinstance(selection_metadata, Mapping):
        raise ValueError("parent fragment selection metadata is unavailable")
    same_layer = select_top_fisher_same_layer_fragments(
        fragment_plan,
        count=int(config["fragment_count"]),
        minimum_fragment_modes=int(config["minimum_fragment_modes"]),
        layer_ordinal=int(selection_metadata["layer_ordinal"]),
    )
    if (
        same_layer.fragment_ids
        != tuple(selection_metadata.get("fragment_ids", ()))
        or tuple(
            fragment.artifact_sha256
            for fragment in same_layer.execution_order
        )
        != tuple(selection_metadata.get("execution_order_sha256s", ()))
    ):
        raise ValueError("parent same-layer fragments cannot be reconstructed")
    max_length = int(_tokenizer_contract()["max_length"])
    selection_batches, selection_stream = _materialize_role(
        tokenizer,
        selection_role,
        split_name="same_layer_shape_flow_selection",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_split_sha256 = _stream_sha256(
        selection_stream,
        label="selection split",
    )
    if selection_split_sha256 != splits.get("selection_split_sha256"):
        raise ValueError("gain selection stream differs from parent candidate")

    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless_graph,
        tuple(lowerings[name] for name in edgeless_graph.traversal_order),
    )
    variants: list[dict[str, object]] = []
    baseline_nll: float | None = None
    _progress(
        f"gain selection: evaluate {len(gain_grid)} signed frozen-field gains"
    )
    for gain in gain_grid:
        graph = _scaled_state_conditioned_graph(parent_graph, gain)
        executor = Gemma3ModalGeneratorGraphExecutor(
            adapter,
            graph,
            tuple(lowerings[name] for name in graph.traversal_order),
        )
        rung = evaluate_modal_graph_rung_conditions(
            adapter,
            executor,
            edgeless_executor,
            selection_batches,
            assessment_role="open_development_assessment",
            expected_example_ids=selection_role.ordered_prompt_sha256s,
        )
        edgeless = _condition(rung, "edgeless_graph")
        candidate = _condition(rung, "interacting_graph")
        current_baseline = float(edgeless["nll_per_token"])
        if baseline_nll is None:
            baseline_nll = current_baseline
        elif baseline_nll != current_baseline:
            raise RuntimeError("gain scan edgeless NLL changed across variants")
        candidate_nll = float(candidate["nll_per_token"])
        variants.append(
            {
                "gain": gain,
                "graph": graph,
                "rung": rung,
                "candidate_nll": candidate_nll,
                "nll_improvement": current_baseline - candidate_nll,
            }
        )
    assert baseline_nll is not None
    chosen = min(
        variants,
        key=lambda value: (
            float(value["candidate_nll"]),
            abs(float(value["gain"])),
            float(value["gain"]),
            value["graph"].artifact_sha256,  # type: ignore[union-attr]
        ),
    )
    chosen_graph = chosen["graph"]
    chosen_rung = chosen["rung"]
    assert isinstance(chosen_graph, ModalGeneratorGraphPlan)
    assert isinstance(chosen_rung, Mapping)
    nll_improvement = float(chosen["nll_improvement"])
    promoted = nll_improvement > max(0.0, minimum_nll_improvement)

    promotion: ModalInteractionGraphPromotion | None = None
    pipeline: ModalCompilerPipeline | None = None
    if promoted:
        promotion = build_modal_interaction_graph_promotion(
            chosen_graph,
            fit_split_sha256=str(splits["fit_split_sha256"]),
            eval_split_sha256=selection_split_sha256,
            selection_metric="nll_per_token",
            baseline_metric_value=baseline_nll,
            candidate_metric_value=float(chosen["candidate_nll"]),
            minimum_heldout_improvement=minimum_nll_improvement,
            heldout_observations=int(chosen_rung["supervised_tokens"]),
        )
        accounting = build_modal_source_replacement_accounting(
            catalog,
            fragment_plan,
            same_layer.fragment_ids,
        )
        pipeline = build_modal_compiler_pipeline(
            source_prompt_trace=fit_trace,
            parameter_catalog=catalog,
            grouped_fisher=fisher,
            fisher_clusters=clusters,
            parameter_cluster_fragments=fragment_plan,
            lowerings_by_node=lowerings,
            graph_plan=chosen_graph,
            interaction_selection=promotion,
            source_replacement_accounting=accounting,
        )

    variant_receipts = tuple(
        {
            "gain": float(value["gain"]),
            "dynamic_graph_sha256": value["graph"].artifact_sha256,
            "behavior": {
                "native": value["rung"]["native"],
                "conditions": value["rung"]["conditions"],
                "graph_comparison": value["rung"]["graph_comparison"],
                "supervised_tokens": value["rung"]["supervised_tokens"],
                "logical_valid_tokens": value["rung"]["logical_valid_tokens"],
            },
            "nll_improvement_over_edgeless": float(
                value["nll_improvement"]
            ),
        }
        for value in variants
    )
    resources = {
        **dict(chosen_rung["resource_accounting"]),
        "edgeless_parameter_count": edgeless_graph.parameter_count,
        "dynamic_parameter_count": chosen_graph.parameter_count,
        "edgeless_dense_macs_per_token": edgeless_graph.macs_per_token,
        "dynamic_dense_macs_per_token": chosen_graph.macs_per_token,
        "conditional_routing_macs_per_token": (
            chosen_graph.conditional_routing_macs_per_token
        ),
        "conditional_dense_message_macs_per_token": (
            chosen_graph.conditional_dense_message_macs_per_token
        ),
        "conditional_selected_message_macs_per_token_upper_bound": (
            chosen_graph.conditional_selected_message_macs_per_token_upper_bound
        ),
        "source_parameter_count": (
            None if pipeline is None else pipeline.source_parameter_count
        ),
        "source_macs_per_token": (
            None if pipeline is None else pipeline.source_macs_per_token
        ),
        "net_parameter_savings": (
            None if pipeline is None else pipeline.net_parameter_savings
        ),
        "net_dense_macs_saved_per_token": (
            None if pipeline is None else pipeline.net_macs_saved_per_token
        ),
    }
    report = save_gemma3_state_conditioned_modal_graph_candidate(
        destination,
        experiment={
            **dict(experiment),
            "experiment_kind": (
                "real_gemma_same_layer_shape_flow_signed_gain_v2"
            ),
            "scientific_role": "open_development_gain_selection",
            "guard_status": "sealed_unopened",
            "heldout_confirmation": False,
        },
        config={
            **dict(config),
            "parent_candidate_scientific_sha256": raw[
                "scientific_payload_sha256"
            ],
            "gain_grid": gain_grid,
            "chosen_gain": float(chosen["gain"]),
            "minimum_nll_improvement": minimum_nll_improvement,
            "routes_refitted": False,
            "messages_refitted": False,
        },
        splits=dict(splits),
        selection={
            "selection_split_sha256": selection_split_sha256,
            "parent_dynamic_graph_sha256": parent_graph.artifact_sha256,
            "dynamic_graph_sha256": chosen_graph.artifact_sha256,
            "edgeless_graph_sha256": edgeless_graph.artifact_sha256,
            "compiler_pipeline_sha256": (
                None if pipeline is None else pipeline.artifact_sha256
            ),
            "interaction_promotion_sha256": (
                None if promotion is None else promotion.artifact_sha256
            ),
            "chosen_gain": float(chosen["gain"]),
            "edgeless_nll_per_token": baseline_nll,
            "candidate_nll_per_token": float(chosen["candidate_nll"]),
            "nll_improvement_over_edgeless": nll_improvement,
            "promotion_passed": promoted,
            "guard_opened": False,
            "variants": variant_receipts,
        },
        resources=resources,
        lowerings_by_node=lowerings,
        edgeless_graph=edgeless_graph,
        dynamic_graph=chosen_graph,
        compiler_pipeline=pipeline,
    )
    if corpus.guard_opened or corpus.guard_consumed:
        raise RuntimeError("gain selection consumed the guard unexpectedly")
    if adapter.model_fingerprint() != model_fingerprint:
        raise RuntimeError("gain calibration mutated the source model")
    _progress(
        f"gain candidate: {'promoted' if promoted else 'not promoted'}; "
        f"gain={float(chosen['gain']):g}; wrote {destination}"
    )
    return report


def _rebuild_same_layer_plan(
    pipeline: ModalCompilerPipeline,
    edgeless_graph: ModalGeneratorGraphPlan,
    lowerings_by_node: Mapping[str, ModalGeneratorLowering],
    config: Mapping[str, object],
) -> EdgelessSameLayerGraphPlan:
    selection_metadata = config.get("fragment_selection")
    if not isinstance(selection_metadata, Mapping):
        raise ValueError("candidate fragment selection metadata is missing")
    count = int(config["fragment_count"])
    minimum = int(config["minimum_fragment_modes"])
    layer = int(selection_metadata["layer_ordinal"])
    selection = select_top_fisher_same_layer_fragments(
        pipeline.parameter_cluster_fragments,
        count=count,
        minimum_fragment_modes=minimum,
        layer_ordinal=layer,
    )
    lowerings_by_fragment = {
        lowering.mode_set_id: lowering
        for lowering in lowerings_by_node.values()
    }
    rebuilt = build_edgeless_same_layer_graph(
        selection,
        fragment_plan=pipeline.parameter_cluster_fragments,
        lowerings_by_fragment=lowerings_by_fragment,
    )
    if rebuilt.graph_plan.artifact_sha256 != edgeless_graph.artifact_sha256:
        raise ValueError("candidate edgeless graph cannot be reconstructed")
    return rebuilt


def _routing_receipt(
    executor: Gemma3ModalGeneratorGraphExecutor,
    batches: Sequence[CalibrationBatch],
) -> dict[str, object]:
    evaluated = {edge.key: 0 for edge in executor.graph_plan.interactions}
    valid_tokens = 0
    for batch in batches:
        with torch.no_grad():
            execution = executor.run(
                batch.model_inputs,
                condition="generated",
                capture_routing=True,
            )
        rows = execution.graph_execution.evaluated_edge_rows
        if rows is None or set(rows) != set(evaluated):
            raise RuntimeError("guard routing instrumentation is incomplete")
        for key, count in rows.items():
            evaluated[key] += int(count)
        valid_tokens += execution.valid_tokens
    selected_rows = sum(evaluated.values())
    if selected_rows != valid_tokens:
        raise RuntimeError(
            "top-1 conditional routing did not select one edge per valid token"
        )
    return {
        "valid_tokens": valid_tokens,
        "selected_edge_rows": selected_rows,
        "selected_edge_rows_by_interaction": dict(sorted(evaluated.items())),
        "exactly_one_selected_edge_per_valid_token": True,
    }


def assess_gemma3_state_conditioned_shape_flow_candidate(
    *,
    candidate_path: Path | str = DEFAULT_CANDIDATE_OUTPUT,
    output: Path | str | None = None,
    corpus_artifact_path: Path | str = DEFAULT_CORPUS_ARTIFACT,
    corpus_fit_path: Path | str = DEFAULT_CORPUS_FIT,
    corpus_selection_path: Path | str = DEFAULT_CORPUS_SELECTION,
    corpus_guard_path: Path | str = DEFAULT_CORPUS_GUARD,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> dict[str, object]:
    """Claim and consume the A guard for one already-promoted candidate."""

    raw = load_gemma3_state_conditioned_modal_graph_candidate(candidate_path)
    pipeline, edgeless_graph, dynamic_graph, lowerings = (
        _restore_candidate_runtime(raw)
    )
    experiment = raw.get("experiment")
    config = raw.get("config")
    splits = raw.get("splits")
    if not all(
        isinstance(value, Mapping)
        for value in (experiment, config, splits)
    ):
        raise TypeError("candidate source-safe metadata is invalid")
    experiment = experiment  # type: ignore[assignment]
    config = config  # type: ignore[assignment]
    splits = splits  # type: ignore[assignment]
    revision = experiment.get("requested_revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("candidate revision is invalid")
    if experiment.get("model_id") != model_id:
        raise ValueError("candidate model id differs from assessment model")
    destination = (
        Path(output)
        if output is not None
        else Path(candidate_path).with_suffix(".assessment.json")
    )
    if destination.exists():
        raise FileExistsError("refusing to overwrite guard assessment")
    _positive_int(tokenization_batch_size, label="tokenization_batch_size")

    _progress("assessment preflight: strict candidate reload; guard still sealed")
    corpus = _load_corpus(
        corpus_artifact_path=corpus_artifact_path,
        corpus_fit_path=corpus_fit_path,
        corpus_selection_path=corpus_selection_path,
        corpus_guard_path=corpus_guard_path,
    )
    guard_view = corpus.preclaim_view("calibration_a_guard")
    if (
        splits.get("guard_role_manifest_sha256")
        != guard_view.manifest_sha256
        or splits.get("corpus_artifact_sha256")
        != corpus.artifact.artifact_sha256
    ):
        raise ValueError("candidate does not bind this unopened guard")
    promotion = pipeline.interaction_selection
    assert isinstance(promotion, ModalInteractionGraphPromotion)
    protocol_sha256 = pipeline.artifact_sha256
    challenger_sha256 = promotion.artifact_sha256

    _progress("guard: atomically claim frozen candidate, then open A-guard once")
    try:
        claim = claim_gemma3_l3_l4_progressive_guard(
            protocol_sha256=protocol_sha256,
            guard_manifest_sha256=guard_view.manifest_sha256,
            challenger_receipt_sha256=challenger_sha256,
        )
    except Gemma3L3L4ProgressiveGuardAlreadyClaimedError:
        claim = load_gemma3_l3_l4_progressive_guard_claim(
            protocol_sha256=protocol_sha256,
            guard_manifest_sha256=guard_view.manifest_sha256,
            challenger_receipt_sha256=challenger_sha256,
        )
    guard_role = corpus.open_guard_after_claim(claim)

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma for guard-only assessment")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype=dtype,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    adapter = Gemma3CausalLMAdapter(model)
    model_fingerprint = adapter.model_fingerprint()
    if model_fingerprint != experiment.get("adapter_model_fingerprint"):
        raise ValueError("guard model fingerprint differs from candidate")
    max_length = int(_tokenizer_contract()["max_length"])
    guard_batches, guard_stream = _materialize_role(
        tokenizer,
        guard_role,
        split_name="same_layer_shape_flow_guard",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    guard_split_sha256 = _stream_sha256(
        guard_stream,
        label="guard split",
    )
    if guard_split_sha256 in {
        splits.get("fit_split_sha256"),
        splits.get("selection_split_sha256"),
    }:
        raise ValueError("guard tokenized stream overlaps fit or selection")

    plan = _rebuild_same_layer_plan(
        pipeline,
        edgeless_graph,
        lowerings,
        config,
    )
    edgeless_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        edgeless_graph,
        plan.lowerings,
    )
    dynamic_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        dynamic_graph,
        plan.lowerings,
    )

    _progress("guard: capture same-boundary shape/flow and route decisions")
    guard_keys = _row_keys_for_batches(adapter, guard_batches)
    guard_rows = collect_edgeless_same_layer_shape_flow_rows(
        adapter,
        edgeless_executor,
        guard_batches,
        plan=plan,
        expected_row_keys=guard_keys,
    )
    source_node, _, corrections, decoders = _flow_inputs(guard_rows, plan)
    guard_families = _family_ids_for_rows(guard_rows.row_keys, guard_role)
    flow = evaluate_state_conditioned_modal_flow(
        guard_rows.node_states[source_node],
        corrections,
        decoders,
        tuple(dynamic_graph.interactions),
        family_ids=guard_families,
    )
    routing = _routing_receipt(dynamic_executor, guard_batches)

    _progress("guard: evaluate native, dynamic, edgeless, and deletion behavior")
    rung = evaluate_modal_graph_rung_conditions(
        adapter,
        dynamic_executor,
        edgeless_executor,
        guard_batches,
        assessment_role="open_development_assessment",
        expected_example_ids=guard_role.ordered_prompt_sha256s,
    )
    dynamic_condition = _condition(rung, "interacting_graph")
    edgeless_condition = _condition(rung, "edgeless_graph")
    guard_nll_improvement = float(
        edgeless_condition["nll_per_token"]
    ) - float(dynamic_condition["nll_per_token"])
    resources = rung["resource_accounting"]
    if not isinstance(resources, Mapping):
        raise TypeError("guard resource accounting is unavailable")
    dynamic_resources = resources.get("interacting_graph")
    if not isinstance(dynamic_resources, Mapping):
        raise TypeError("guard dynamic resource accounting is unavailable")

    result: dict[str, object] = {
        "schema": "fisher_graph.gemma3_state_conditioned_shape_flow_assessment",
        "format_version": 1,
        "scientific_status": {
            "role": "family_disjoint_calibration_a_guard",
            "open_development": True,
            "heldout_confirmation": False,
            "fresh_validation": False,
            "test_data_used": False,
            "guard_claimed_before_materialization": True,
        },
        "candidate": {
            "tensor_file": Path(candidate_path).name,
            "scientific_payload_sha256": raw["scientific_payload_sha256"],
            "compiler_pipeline_sha256": pipeline.artifact_sha256,
            "interaction_promotion_sha256": promotion.artifact_sha256,
            "dynamic_graph_sha256": dynamic_graph.artifact_sha256,
        },
        "guard": {
            "role_manifest_sha256": guard_view.manifest_sha256,
            "tokenized_split_sha256": guard_split_sha256,
            "claim_sha256": claim.claim_sha256,
            "example_count": len(guard_role.prompts),
            "family_ids": tuple(sorted(set(guard_role.family_ids))),
        },
        "flow": flow.metadata(),
        "routing_execution": routing,
        "behavior": rung,
        "guard_nll_improvement_over_edgeless": guard_nll_improvement,
        "resource_summary": {
            "source_whole_model_learned_parameters": dynamic_resources[
                "source_whole_model_learned_parameters"
            ],
            "candidate_whole_model_learned_parameters": dynamic_resources[
                "candidate_whole_model_learned_parameters"
            ],
            "native_removed_learned_parameters": dynamic_resources[
                "native_removed_learned_parameters"
            ],
            "modal_graph_learned_parameters": dynamic_resources[
                "modal_graph_learned_parameters"
            ],
            "logical_modal_graph_macs": dynamic_resources[
                "logical_modal_graph_macs"
            ],
            "logical_executed_modal_graph_macs": dynamic_resources[
                "logical_executed_modal_graph_macs"
            ],
            "logical_valid_tokens": rung["logical_valid_tokens"],
        },
        "source_model_unchanged": adapter.model_fingerprint()
        == model_fingerprint,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    _progress(f"guard: wrote {destination}")
    return result


def _int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integers"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    return result


def _float_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated floats"
        ) from error
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError(
            "float list must be nonempty and finite"
        )
    return result


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=DEFAULT_CORPUS_ARTIFACT,
    )
    parser.add_argument("--corpus-fit", type=Path, default=DEFAULT_CORPUS_FIT)
    parser.add_argument(
        "--corpus-selection",
        type=Path,
        default=DEFAULT_CORPUS_SELECTION,
    )
    parser.add_argument(
        "--corpus-guard",
        type=Path,
        default=DEFAULT_CORPUS_GUARD,
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit or assess a real-Gemma same-layer state-conditioned modal graph"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser(
        "fit",
        help="fit/select a candidate without opening the guard",
    )
    fit.add_argument("--revision", required=True)
    fit.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    fit.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    fit.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_BASE_ARTIFACT,
    )
    fit.add_argument("--fragment-count", type=int, default=DEFAULT_FRAGMENT_COUNT)
    fit.add_argument(
        "--minimum-fragment-modes",
        type=int,
        default=DEFAULT_MINIMUM_FRAGMENT_MODES,
    )
    fit.add_argument("--layer-ordinal", type=int)
    fit.add_argument("--mode-ranks", type=_int_tuple, default=DEFAULT_MODE_RANKS)
    fit.add_argument(
        "--selected-mode-rank",
        type=int,
        default=DEFAULT_SELECTED_MODE_RANK,
    )
    fit.add_argument(
        "--generator-ranks",
        type=_int_tuple,
        default=DEFAULT_GENERATOR_RANKS,
    )
    fit.add_argument(
        "--selected-generator-rank",
        type=int,
        default=DEFAULT_SELECTED_GENERATOR_RANK,
    )
    fit.add_argument("--generator-ridge", type=float, default=0.0)
    fit.add_argument(
        "--quadratic-ranks",
        type=_int_tuple,
        default=DEFAULT_QUADRATIC_RANKS,
    )
    fit.add_argument(
        "--router-ridge",
        type=float,
        default=DEFAULT_ROUTER_RIDGE,
    )
    fit.add_argument(
        "--message-ridge",
        type=float,
        default=DEFAULT_MESSAGE_RIDGE,
    )
    fit.add_argument(
        "--quadratic-steps",
        type=int,
        default=DEFAULT_QUADRATIC_STEPS,
    )
    fit.add_argument(
        "--minimum-nll-improvement",
        type=float,
        default=DEFAULT_MINIMUM_NLL_IMPROVEMENT,
    )
    _add_corpus_arguments(fit)
    _add_runtime_arguments(fit)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="select a signed gain on a frozen failed candidate",
    )
    calibrate.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_CANDIDATE_OUTPUT,
    )
    calibrate.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GAIN_CANDIDATE_OUTPUT,
    )
    calibrate.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_BASE_ARTIFACT,
    )
    calibrate.add_argument(
        "--gains",
        type=_float_tuple,
        default=DEFAULT_GAIN_GRID,
    )
    calibrate.add_argument(
        "--minimum-nll-improvement",
        type=float,
        default=DEFAULT_MINIMUM_NLL_IMPROVEMENT,
    )
    _add_corpus_arguments(calibrate)
    _add_runtime_arguments(calibrate)

    assess = subparsers.add_parser(
        "assess",
        help="claim and consume the guard for a promoted candidate",
    )
    assess.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_CANDIDATE_OUTPUT,
    )
    assess.add_argument("--output", type=Path)
    _add_corpus_arguments(assess)
    _add_runtime_arguments(assess)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    common = {
        "corpus_artifact_path": arguments.corpus_artifact,
        "corpus_fit_path": arguments.corpus_fit,
        "corpus_selection_path": arguments.corpus_selection,
        "corpus_guard_path": arguments.corpus_guard,
        "model_id": arguments.model_id,
        "cache_dir": arguments.cache_dir,
        "device_name": arguments.device,
        "dtype": arguments.dtype,
        "tokenization_batch_size": arguments.tokenization_batch_size,
    }
    if arguments.command == "fit":
        report = fit_gemma3_state_conditioned_shape_flow_candidate(
            revision=arguments.revision,
            output=arguments.output,
            fit_export_path=arguments.fit_export,
            base_artifact_path=arguments.base_artifact,
            fragment_count=arguments.fragment_count,
            minimum_fragment_modes=arguments.minimum_fragment_modes,
            layer_ordinal=arguments.layer_ordinal,
            mode_ranks=arguments.mode_ranks,
            selected_mode_rank=arguments.selected_mode_rank,
            generator_ranks=arguments.generator_ranks,
            selected_generator_rank=arguments.selected_generator_rank,
            generator_ridge=arguments.generator_ridge,
            quadratic_ranks=arguments.quadratic_ranks,
            router_ridge=arguments.router_ridge,
            message_ridge=arguments.message_ridge,
            quadratic_steps=arguments.quadratic_steps,
            minimum_nll_improvement=arguments.minimum_nll_improvement,
            **common,
        )
    elif arguments.command == "calibrate":
        report = calibrate_gemma3_state_conditioned_shape_flow_gain(
            candidate_path=arguments.candidate,
            output=arguments.output,
            base_artifact_path=arguments.base_artifact,
            gains=arguments.gains,
            minimum_nll_improvement=arguments.minimum_nll_improvement,
            **common,
        )
    else:
        report = assess_gemma3_state_conditioned_shape_flow_candidate(
            candidate_path=arguments.candidate,
            output=arguments.output,
            **common,
        )
    json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
