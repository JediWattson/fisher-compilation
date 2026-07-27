"""Development-only multi-fragment Gemma modal-generator fan-in rung.

This runner extends the strict single-fragment v3 artifact without recomputing
its whole-model prompt Fisher analysis.  It:

* strict-loads the v3 fit trace, parameter catalog, Fisher, clusters, and
  fragment plan;
* chooses a fixed number of top-Fisher fragments on distinct layers;
* fits one predeclared computational-mode/modal-generator ladder per fragment;
* observes the physical edgeless compiled trajectory;
* selects only causal edges from earlier nodes into the terminal node; and
* evaluates the frozen graph on a disjoint open-development assessment split.

The assessment is not calibration B, a guard, validation, or test data.  The
saved v1 artifact is source safe and contains executable generator/interaction
weights, but no prompt text, token ids, raw activation rows, raw gradient rows,
tokenizer state, or source-model weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
import sys

import torch

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .fisher_prompt_clustering import FisherPromptClusterPlan
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_FIT_EXPORT,
    DEFAULT_GENERATOR_RANKS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODE_RANKS,
    DEFAULT_SELECTED_GENERATOR_RANK,
    DEFAULT_SELECTED_MODE_RANK,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    DevelopmentPromptExport,
    FittedModalGeneratorPilot,
    _layer_runtime_sites,
    _safe_tokenized_stream_metadata,
    _select_row_sites,
    fit_layer_cluster_modal_generator,
    load_development_prompt_export,
    load_gemma3_modal_generator_dev_artifact,
    validate_development_split_pair,
)
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorExecutor,
    Gemma3ModalGeneratorReplacement,
)
from .gemma3_modal_generator_graph_executor import (
    Gemma3ModalGeneratorGraphExecutor,
)
from .gemma3_modal_generator_multifragment_artifact import (
    Gemma3ModalGeneratorMultifragmentNodeRecord,
    build_gemma3_modal_generator_multifragment_evaluation_from_rung,
    build_gemma3_modal_generator_multifragment_model_metadata,
    build_gemma3_modal_generator_multifragment_protocol,
    build_gemma3_modal_generator_multifragment_scientific_status,
    build_gemma3_modal_generator_multifragment_splits,
    build_gemma3_modal_generator_multifragment_upstream_metadata,
    load_gemma3_modal_generator_multifragment_artifact,
    save_gemma3_modal_generator_multifragment_artifact,
)
from .gemma3_modal_generator_terminal_fanin import (
    AlignedFragmentRows,
    collect_aligned_fragment_rows,
    collect_edgeless_terminal_fanin_rows,
    fit_terminal_fanin_compilation,
    build_edgeless_terminal_fanin_plan,
    select_top_distinct_layer_fragments,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_graph_rung_evaluation import (
    evaluate_modal_graph_rung_conditions,
    partition_development_export_for_interactions,
)
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from .parameter_fisher_coupling import (
    GroupedVirtualGateFisher,
    NaturalMLPParameterGroupCatalog,
)
from .prompt_mode_tracing import PromptModeTrace
from .streaming_analysis import iter_activation_score_gradient_rows


__all__ = [
    "DEFAULT_BASE_ARTIFACT",
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_modal_generator_multifragment_dev_experiment",
]


DEFAULT_BASE_ARTIFACT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-graph-dev-v3.pt"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-multifragment-fanin-dev-v1.pt"
)
DEFAULT_FRAGMENT_COUNT = 4
DEFAULT_MINIMUM_FRAGMENT_MODES = 32
DEFAULT_INTERACTION_SELECTION_COUNT = 20
DEFAULT_INTERACTION_RIDGES = (0.0, 1e-6, 1e-4, 1e-2)
DEFAULT_MINIMUM_INTERACTION_IMPROVEMENT = 1e-3
DEFAULT_DENSE_EQUIVALENCE_ATOL = 1e-4
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _progress(message: str) -> None:
    print(f"[multifragment-fanin] {message}", file=sys.stderr, flush=True)


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _stream_content_sha256s(
    stream: Mapping[str, object],
    *,
    label: str,
) -> tuple[str, ...]:
    return _content_sha256s_from_safe_metadata(
        _safe_tokenized_stream_metadata(stream),
        label=label,
    )


def _content_sha256s_from_safe_metadata(
    metadata: Mapping[str, object],
    *,
    label: str,
) -> tuple[str, ...]:
    values = metadata.get("content_sha256")
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
    ):
        raise ValueError(f"{label} content hashes are unavailable")
    result = tuple(
        _require_sha256(value, label=f"{label} content hash")
        for value in values
    )
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{label} content hashes are empty or duplicated")
    return result


def _upstream_stream_metadata(
    upstream: Mapping[str, object],
    *,
    key: str,
) -> Mapping[str, object]:
    splits = upstream.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("upstream artifact split metadata is missing")
    value = splits.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"upstream {key} metadata is missing")
    return value


def _validate_upstream_export_bindings(
    upstream: Mapping[str, object],
    *,
    fit_export: DevelopmentPromptExport,
    eval_export: DevelopmentPromptExport,
) -> None:
    """Require the caller's raw prompt declarations to match upstream v3."""

    splits = upstream.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("upstream artifact split metadata is missing")
    for key, export in (
        ("fit_export", fit_export),
        ("eval_export", eval_export),
    ):
        expected = splits.get(key)
        if not isinstance(expected, Mapping):
            raise ValueError(f"upstream {key} metadata is missing")
        if dict(expected) != export.metadata():
            raise ValueError(
                f"live {key} does not match the strict upstream v3 artifact"
            )


def _canonical_positive_ints(
    values: Sequence[int],
    *,
    label: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            f"{label} must contain unique, strictly increasing positive ints"
        )
    return result


def _canonical_nonnegative_floats(
    values: Sequence[float],
    *,
    label: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{label} must be a sequence")
    if not values:
        raise ValueError(f"{label} cannot be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError(f"{label} must contain finite nonnegative numbers")
    result = tuple(float(value) for value in values)
    if result != tuple(sorted(set(result))):
        raise ValueError(
            f"{label} must contain unique, strictly increasing values"
        )
    return result


def _validate_runner_preflight(
    *,
    revision: str,
    output: Path | str,
    model_id: str,
    device_name: str,
    dtype: str,
    max_length: int,
    tokenization_batch_size: int,
    fragment_count: int,
    minimum_fragment_modes: int,
    interaction_selection_count: int,
    mode_ranks: Sequence[int],
    selected_mode_rank: int,
    generator_ranks: Sequence[int],
    selected_generator_rank: int,
    generator_ridge: float,
    interaction_ridges: Sequence[float],
    minimum_interaction_improvement: float,
    dense_equivalence_atol: float,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    """Validate all pure configuration before loading artifacts or a model."""

    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be nonempty")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("device_name must be nonempty")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype is unsupported")
    for value, label, minimum in (
        (max_length, "max_length", 2),
        (tokenization_batch_size, "tokenization_batch_size", 1),
        (fragment_count, "fragment_count", 2),
        (minimum_fragment_modes, "minimum_fragment_modes", 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be at least {minimum}")
    if (
        type(interaction_selection_count) is not int
        or not 0 < interaction_selection_count < 40
    ):
        raise ValueError(
            "interaction_selection_count must leave nonempty "
            "selection and assessment partitions of eval40"
        )

    modes = _canonical_positive_ints(mode_ranks, label="mode_ranks")
    generators = _canonical_positive_ints(
        generator_ranks,
        label="generator_ranks",
    )
    ridges = _canonical_nonnegative_floats(
        interaction_ridges,
        label="interaction_ridges",
    )
    if type(selected_mode_rank) is not int or selected_mode_rank not in modes:
        raise ValueError("selected_mode_rank must be in mode_ranks")
    if (
        type(selected_generator_rank) is not int
        or selected_generator_rank not in generators
    ):
        raise ValueError(
            "selected_generator_rank must be in generator_ranks"
        )
    for value, label in (
        (generator_ridge, "generator_ridge"),
        (
            minimum_interaction_improvement,
            "minimum_interaction_improvement",
        ),
        (dense_equivalence_atol, "dense_equivalence_atol"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and nonnegative")

    path = Path(output)
    if path.suffix != ".pt":
        raise ValueError("multifragment artifact output must use .pt")
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite multifragment artifact")
    return modes, generators, ridges


def _bind_batch_example_ids(
    batches: Sequence[CalibrationBatch],
    example_ids: Sequence[str],
) -> tuple[CalibrationBatch, ...]:
    """Replace positional tokenizer ids with independently declared hashes."""

    declared = tuple(example_ids)
    if (
        not declared
        or len(declared) != len(set(declared))
        or any(not isinstance(value, str) or not value for value in declared)
    ):
        raise ValueError("declared example ids must be nonempty and unique")
    offset = 0
    rebound: list[CalibrationBatch] = []
    for batch in batches:
        stop = offset + batch.batch_size
        ids = declared[offset:stop]
        if len(ids) != batch.batch_size:
            raise ValueError("declared example ids do not cover the batches")
        rebound.append(
            CalibrationBatch(
                model_inputs=batch.model_inputs,
                targets=batch.targets,
                valid_positions=batch.valid_positions,
                shared_input_names=batch.shared_input_names,
                example_ids=ids,
            )
        )
        offset = stop
    if offset != len(declared):
        raise ValueError("declared example ids exceed the batches")
    return tuple(rebound)


def _restore_upstream_analysis(
    upstream: Mapping[str, object],
) -> tuple[
    PromptModeTrace,
    NaturalMLPParameterGroupCatalog,
    GroupedVirtualGateFisher,
    FisherPromptClusterPlan,
    ParameterClusterLayerFragmentPlan,
]:
    return (
        PromptModeTrace.from_state_dict(upstream["fit_prompt_trace"]),
        NaturalMLPParameterGroupCatalog.from_state_dict(
            upstream["parameter_catalog"]
        ),
        GroupedVirtualGateFisher.from_state_dict(
            upstream["fisher_coupling"]
        ),
        FisherPromptClusterPlan.from_state_dict(
            upstream["parameter_clusters"]
        ),
        ParameterClusterLayerFragmentPlan.from_state_dict(
            upstream["parameter_cluster_fragments"]
        ),
    )


def _collect_native_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[object],
    *,
    fragments: Sequence[ParameterClusterLayerFragment],
    leaf_activation_site: str,
) -> AlignedFragmentRows:
    sites = tuple(
        dict.fromkeys(
            site
            for fragment in fragments
            for site in (fragment.input_site, fragment.activation_site)
        )
    )
    down_weights: dict[str, torch.Tensor] = {}
    for fragment in fragments:
        input_site, output_site, _, down_weight = _layer_runtime_sites(
            adapter,
            fragment.layer_ordinal,
        )
        if (
            input_site != fragment.input_site
            or output_site != fragment.output_site
        ):
            raise ValueError("selected fragment runtime sites drifted")
        down_weights[fragment.fragment_id] = down_weight
    requested = tuple(dict.fromkeys((*sites, leaf_activation_site)))
    raw_rows = iter_activation_score_gradient_rows(
        adapter,
        batches,
        activation_names=requested,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_activation_site,
        accumulation_dtype=torch.float64,
    )
    return collect_aligned_fragment_rows(
        _select_row_sites(raw_rows, sites),
        fragments=fragments,
        down_projection_weights=down_weights,
    )


def _artifact_evaluation_from_rung(
    rung: Mapping[str, object],
    *,
    assessment_split_sha256: str,
    dense_equivalence_atol: float,
) -> dict[str, object]:
    result = build_gemma3_modal_generator_multifragment_evaluation_from_rung(
        assessment_split_sha256=assessment_split_sha256,
        rung_evaluation=rung,
    )
    equivalence = result.get("edgeless_dense_equivalence")
    if (
        not isinstance(equivalence, Mapping)
        or float(equivalence["absolute_tolerance"])
        != dense_equivalence_atol
    ):
        raise RuntimeError("dense-equivalence tolerance binding drifted")
    return result


def _validate_upstream_bindings(
    upstream: Mapping[str, object],
    *,
    model_id: str,
    revision: str,
    model_fingerprint: str,
) -> None:
    model = upstream.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("model_id") != model_id
        or model.get("requested_revision") != revision
        or model.get("resolved_commit") != revision
        or model.get("adapter_model_fingerprint") != model_fingerprint
    ):
        raise ValueError("upstream v3 artifact does not bind the live model")


def run_gemma3_modal_generator_multifragment_dev_experiment(
    *,
    fit_export_path: Path | str,
    eval_export_path: Path | str,
    revision: str,
    base_artifact_path: Path | str = DEFAULT_BASE_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    fragment_count: int = DEFAULT_FRAGMENT_COUNT,
    minimum_fragment_modes: int = DEFAULT_MINIMUM_FRAGMENT_MODES,
    interaction_selection_count: int = (
        DEFAULT_INTERACTION_SELECTION_COUNT
    ),
    mode_ranks: Sequence[int] = DEFAULT_MODE_RANKS,
    selected_mode_rank: int = DEFAULT_SELECTED_MODE_RANK,
    generator_ranks: Sequence[int] = DEFAULT_GENERATOR_RANKS,
    selected_generator_rank: int = DEFAULT_SELECTED_GENERATOR_RANK,
    generator_ridge: float = 0.0,
    interaction_ridges: Sequence[float] = DEFAULT_INTERACTION_RIDGES,
    minimum_interaction_improvement: float = (
        DEFAULT_MINIMUM_INTERACTION_IMPROVEMENT
    ),
    dense_equivalence_atol: float = DEFAULT_DENSE_EQUIVALENCE_ATOL,
) -> dict[str, object]:
    """Run the four-node terminal-fan-in open-development rung."""

    mode_ranks, generator_ranks, interaction_ridges = (
        _validate_runner_preflight(
            revision=revision,
            output=output,
            model_id=model_id,
            device_name=device_name,
            dtype=dtype,
            max_length=max_length,
            tokenization_batch_size=tokenization_batch_size,
            fragment_count=fragment_count,
            minimum_fragment_modes=minimum_fragment_modes,
            interaction_selection_count=interaction_selection_count,
            mode_ranks=mode_ranks,
            selected_mode_rank=selected_mode_rank,
            generator_ranks=generator_ranks,
            selected_generator_rank=selected_generator_rank,
            generator_ridge=generator_ridge,
            interaction_ridges=interaction_ridges,
            minimum_interaction_improvement=(
                minimum_interaction_improvement
            ),
            dense_equivalence_atol=dense_equivalence_atol,
        )
    )

    _progress("preflight: strict-load v3 analysis and partition eval40")
    upstream = load_gemma3_modal_generator_dev_artifact(base_artifact_path)
    fit_export = load_development_prompt_export(fit_export_path)
    eval_export = load_development_prompt_export(eval_export_path)
    validate_development_split_pair(fit_export, eval_export)
    _validate_upstream_export_bindings(
        upstream,
        fit_export=fit_export,
        eval_export=eval_export,
    )
    partition = partition_development_export_for_interactions(
        eval_export,
        selection_count=interaction_selection_count,
        expected_prompt_count=40,
    )
    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    selected = select_top_distinct_layer_fragments(
        fragment_plan,
        count=fragment_count,
        minimum_fragment_modes=minimum_fragment_modes,
    )
    if any(
        selected_mode_rank > fragment.mode_count
        for fragment in selected.causal_order
    ):
        raise ValueError(
            "selected_mode_rank exceeds a selected fragment's intrinsic rank"
        )

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned Gemma checkpoint from local cache")
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
    live_model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    if live_model_metadata.get("resolved_commit") != revision:
        raise ValueError("loaded Gemma model does not bind the pinned revision")

    _progress("tokenize: materialize fit40 and selection20 only")
    fit_batches, fit_stream = _materialize_split(
        tokenizer,
        fit_export.prompts,
        split_name="modal_generator_development_fit",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    selection_batches, selection_stream = _materialize_split(
        tokenizer,
        partition.selection.prompts,
        split_name="modal_generator_interaction_selection",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    fit_batches = _bind_batch_example_ids(
        fit_batches,
        fit_export.prompt_sha256s,
    )
    selection_batches = _bind_batch_example_ids(
        selection_batches,
        partition.selection.prompt_sha256s,
    )
    fit_safe = _safe_tokenized_stream_metadata(fit_stream)
    upstream_fit = _upstream_stream_metadata(
        upstream,
        key="fit_tokenized",
    )
    if fit_safe != upstream_fit:
        raise ValueError("live fit tokenization differs from the v3 artifact")
    fit_split_sha256 = _require_sha256(
        fit_safe.get("serialized_sha256"),
        label="fit split sha256",
    )
    selection_safe = _safe_tokenized_stream_metadata(selection_stream)
    selection_split_sha256 = _require_sha256(
        selection_safe.get("serialized_sha256"),
        label="selection split sha256",
    )
    upstream_eval = _upstream_stream_metadata(
        upstream,
        key="eval_tokenized",
    )
    upstream_eval_content = _content_sha256s_from_safe_metadata(
        upstream_eval,
        label="upstream evaluation",
    )
    if set(_stream_content_sha256s(selection_stream, label="selection")) - set(
        upstream_eval_content
    ):
        raise ValueError("selection tokenization is not from upstream eval40")

    live_layer_specs, leaf_activation_site, _ = _whole_model_layer_specs(
        adapter
    )
    if tuple(spec.layer_id for spec in live_layer_specs) != tuple(
        spec.layer_id for spec in fit_trace.layer_specs
    ):
        raise ValueError("live layer catalog differs from the upstream trace")
    _progress("rows: one native gradient replay per fit/selection split")
    fit_native_rows = _collect_native_rows(
        adapter,
        fit_batches,
        fragments=selected.causal_order,
        leaf_activation_site=leaf_activation_site,
    )
    selection_native_rows = _collect_native_rows(
        adapter,
        selection_batches,
        fragments=selected.causal_order,
        leaf_activation_site=leaf_activation_site,
    )

    _progress("nodes: fit four fixed mode/generator ladders")
    pilots: dict[str, FittedModalGeneratorPilot] = {}
    for fragment in selected.causal_order:
        fragment_mode_ranks = tuple(
            rank for rank in mode_ranks if rank <= fragment.mode_count
        )
        if selected_mode_rank not in fragment_mode_ranks:
            raise RuntimeError("fragment rank clipping removed selected rank")
        _progress(
            "nodes: fit "
            f"layer {fragment.layer_ordinal} / cluster {fragment.cluster_id} "
            f"({fragment.mode_count} native channels; "
            f"mode ladder {fragment_mode_ranks})"
        )
        pilots[fragment.fragment_id] = fit_layer_cluster_modal_generator(
            fit_native_rows.rows_by_fragment[fragment.fragment_id],
            selection_native_rows.rows_by_fragment[fragment.fragment_id],
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
            generator_ranks=generator_ranks,
            selected_generator_rank=selected_generator_rank,
            ridge=generator_ridge,
        )
    edgeless = build_edgeless_terminal_fanin_plan(
        selected,
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

    _progress("states: capture physical edgeless trajectories and teachers")
    fit_runtime_rows = collect_edgeless_terminal_fanin_rows(
        adapter,
        edgeless_executor,
        fit_batches,
        plan=edgeless,
        expected_row_keys=fit_native_rows.row_keys,
    )
    selection_runtime_rows = collect_edgeless_terminal_fanin_rows(
        adapter,
        edgeless_executor,
        selection_batches,
        plan=edgeless,
        expected_row_keys=selection_native_rows.row_keys,
    )
    terminal_fragment_id = selected.terminal_fragment.fragment_id
    target_fisher_fit = fit_native_rows.rows_by_fragment[
        terminal_fragment_id
    ].fisher_weights
    target_fisher_selection = selection_native_rows.rows_by_fragment[
        terminal_fragment_id
    ].fisher_weights

    _progress("edges: fit and freeze terminal-only causal fan-in")
    compilation = fit_terminal_fanin_compilation(
        edgeless=edgeless,
        fit_rows=fit_runtime_rows,
        eval_rows=selection_runtime_rows,
        target_fisher_weights_fit=target_fisher_fit,
        target_fisher_weights_eval=target_fisher_selection,
        fit_prompt_trace=fit_trace,
        parameter_catalog=catalog,
        fisher_coupling=fisher,
        parameter_clusters=clusters,
        fragment_plan=fragment_plan,
        fit_split_sha256=fit_split_sha256,
        eval_split_sha256=selection_split_sha256,
        ridges=interaction_ridges,
        minimum_heldout_improvement=(
            minimum_interaction_improvement
        ),
        fit_intercept=False,
    )
    _progress(
        "edges: selected "
        f"{len(compilation.interaction_selection.interactions)} of "
        f"{len(compilation.interaction_selection.candidate_edges)} "
        "terminal fan-in candidates"
    )
    interacting_executor = Gemma3ModalGeneratorGraphExecutor(
        adapter,
        compilation.graph_plan,
        edgeless.lowerings,
    )
    dense_executor = Gemma3ModalGeneratorExecutor(
        adapter,
        tuple(
            Gemma3ModalGeneratorReplacement.from_lowering(
                pilots[fragment.fragment_id].lowering
            )
            for fragment in selected.causal_order
        ),
    )

    _progress("assessment: tokenize untouched assessment20 after graph freeze")
    assessment_batches, assessment_stream = _materialize_split(
        tokenizer,
        partition.assessment.prompts,
        split_name="modal_generator_open_development_assessment",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    assessment_batches = _bind_batch_example_ids(
        assessment_batches,
        partition.assessment.prompt_sha256s,
    )
    assessment_safe = _safe_tokenized_stream_metadata(assessment_stream)
    assessment_split_sha256 = _require_sha256(
        assessment_safe.get("serialized_sha256"),
        label="assessment split sha256",
    )
    fit_content = _stream_content_sha256s(fit_stream, label="fit")
    selection_content = _stream_content_sha256s(
        selection_stream,
        label="selection",
    )
    assessment_content = _stream_content_sha256s(
        assessment_stream,
        label="assessment",
    )
    if set(selection_content) | set(assessment_content) != set(
        upstream_eval_content
    ):
        raise ValueError(
            "selection20 and assessment20 do not reconstruct upstream eval40"
        )

    _progress("evaluate: native, interacting, edgeless, deletion, dense")
    rung_evaluation = evaluate_modal_graph_rung_conditions(
        adapter,
        interacting_executor,
        edgeless_executor,
        assessment_batches,
        nodewise_dense_executor=dense_executor,
        dense_equivalence_atol=dense_equivalence_atol,
        dense_equivalence_rtol=0.0,
        assessment_role="open_development_assessment",
        expected_example_ids=partition.assessment.prompt_sha256s,
    )
    evaluation = _artifact_evaluation_from_rung(
        rung_evaluation,
        assessment_split_sha256=assessment_split_sha256,
        dense_equivalence_atol=dense_equivalence_atol,
    )

    node_records = tuple(
        Gemma3ModalGeneratorMultifragmentNodeRecord(
            node_name=node.name,
            computational_modes=pilots[
                edgeless.fragment_id_by_node[node.name]
            ].computational_modes,
            modal_generators=pilots[
                edgeless.fragment_id_by_node[node.name]
            ].modal_generators,
            lowering=edgeless.lowerings_by_node[node.name],
        )
        for node in compilation.graph_plan.nodes
    )
    model_metadata = (
        build_gemma3_modal_generator_multifragment_model_metadata(
            model_id=model_id,
            requested_revision=revision,
            resolved_commit=revision,
            adapter_model_fingerprint=model_fingerprint,
            source_whole_model_learned_parameters=sum(
                parameter.numel() for parameter in model.parameters()
            ),
        )
    )
    splits = build_gemma3_modal_generator_multifragment_splits(
        fit_split_sha256=fit_split_sha256,
        upstream_evaluation_split_sha256=_require_sha256(
            upstream_eval.get("serialized_sha256"),
            label="upstream evaluation split sha256",
        ),
        selection_split_sha256=selection_split_sha256,
        assessment_split_sha256=assessment_split_sha256,
        source_evaluation_export_sha256=eval_export.artifact_sha256,
        raw_partition_plan_sha256=partition.artifact_sha256,
        selection_partition_sha256=partition.selection.artifact_sha256,
        assessment_partition_sha256=partition.assessment.artifact_sha256,
        fit_content_sha256s=fit_content,
        upstream_evaluation_content_sha256s=tuple(upstream_eval_content),
        selection_content_sha256s=selection_content,
        assessment_content_sha256s=assessment_content,
    )
    protocol = build_gemma3_modal_generator_multifragment_protocol(
        compiler_pipeline=compilation.compiler_pipeline,
        fragment_selection_rule="predeclared_fit_fisher_multifragment",
        interaction_weighting="native_reference_target_fragment_fisher",
    )
    upstream_metadata = (
        build_gemma3_modal_generator_multifragment_upstream_metadata(
            source_scientific_payload_sha256=_require_sha256(
                upstream.get("scientific_payload_sha256"),
                label="upstream scientific payload sha256",
            ),
            source_evaluation_export_sha256=eval_export.artifact_sha256,
            fit_prompt_trace=fit_trace,
            parameter_catalog=catalog,
            fisher_coupling=fisher,
            parameter_clusters=clusters,
            parameter_cluster_fragments=fragment_plan,
        )
    )
    _progress("artifact: save strict source-free v1 result")
    report = save_gemma3_modal_generator_multifragment_artifact(
        output,
        scientific_status=(
            build_gemma3_modal_generator_multifragment_scientific_status()
        ),
        model=model_metadata,
        protocol=protocol,
        splits=splits,
        upstream_metadata=upstream_metadata,
        fit_prompt_trace=fit_trace,
        parameter_catalog=catalog,
        fisher_coupling=fisher,
        parameter_clusters=clusters,
        parameter_cluster_fragments=fragment_plan,
        node_records=node_records,
        interaction_selection=compilation.interaction_selection,
        edgeless_graph=edgeless.graph_plan,
        compiler_pipeline=compilation.compiler_pipeline,
        evaluation=evaluation,
    )
    load_gemma3_modal_generator_multifragment_artifact(output)
    if adapter.model_fingerprint() != model_fingerprint:
        raise RuntimeError("multifragment experiment mutated the source model")
    _progress(f"wrote {Path(output)} and {Path(output).with_suffix('.json')}")
    return report


def _int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ranks must be comma-separated integers"
        ) from error
    if (
        not result
        or any(item <= 0 for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise argparse.ArgumentTypeError(
            "ranks must be unique, increasing, and positive"
        )
    return result


def _float_list(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ridges must be comma-separated numbers"
        ) from error
    if (
        not result
        or any(not math.isfinite(item) or item < 0.0 for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise argparse.ArgumentTypeError(
            "ridges must be unique, increasing, finite, and nonnegative"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-only Gemma multi-fragment terminal-fan-in rung."
        )
    )
    parser.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    parser.add_argument("--eval-export", type=Path, default=DEFAULT_EVAL_EXPORT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_BASE_ARTIFACT,
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=DEFAULT_TOKENIZATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--fragment-count",
        type=int,
        default=DEFAULT_FRAGMENT_COUNT,
    )
    parser.add_argument(
        "--minimum-fragment-modes",
        type=int,
        default=DEFAULT_MINIMUM_FRAGMENT_MODES,
    )
    parser.add_argument(
        "--interaction-selection-count",
        type=int,
        default=DEFAULT_INTERACTION_SELECTION_COUNT,
    )
    parser.add_argument(
        "--mode-ranks",
        type=_int_list,
        default=DEFAULT_MODE_RANKS,
    )
    parser.add_argument(
        "--selected-mode-rank",
        type=int,
        default=DEFAULT_SELECTED_MODE_RANK,
    )
    parser.add_argument(
        "--generator-ranks",
        type=_int_list,
        default=DEFAULT_GENERATOR_RANKS,
    )
    parser.add_argument(
        "--selected-generator-rank",
        type=int,
        default=DEFAULT_SELECTED_GENERATOR_RANK,
    )
    parser.add_argument("--generator-ridge", type=float, default=0.0)
    parser.add_argument(
        "--interaction-ridges",
        type=_float_list,
        default=DEFAULT_INTERACTION_RIDGES,
    )
    parser.add_argument(
        "--minimum-interaction-improvement",
        type=float,
        default=DEFAULT_MINIMUM_INTERACTION_IMPROVEMENT,
    )
    parser.add_argument(
        "--dense-equivalence-atol",
        type=float,
        default=DEFAULT_DENSE_EQUIVALENCE_ATOL,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_modal_generator_multifragment_dev_experiment(
        fit_export_path=arguments.fit_export,
        eval_export_path=arguments.eval_export,
        revision=arguments.revision,
        base_artifact_path=arguments.base_artifact,
        output=arguments.output,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        fragment_count=arguments.fragment_count,
        minimum_fragment_modes=arguments.minimum_fragment_modes,
        interaction_selection_count=(
            arguments.interaction_selection_count
        ),
        mode_ranks=arguments.mode_ranks,
        selected_mode_rank=arguments.selected_mode_rank,
        generator_ranks=arguments.generator_ranks,
        selected_generator_rank=arguments.selected_generator_rank,
        generator_ridge=arguments.generator_ridge,
        interaction_ridges=arguments.interaction_ridges,
        minimum_interaction_improvement=(
            arguments.minimum_interaction_improvement
        ),
        dense_equivalence_atol=arguments.dense_equivalence_atol,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
