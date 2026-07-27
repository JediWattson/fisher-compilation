"""Run the exhaustive all-block Gemma MLP modal-generator rung.

This development-only runner replaces every native MLP channel in every Gemma
transformer block with one Fisher-weighted, full-residual-width modal
generator.  Attention, embeddings, normalization, and the language-model head
remain native.  The first intended run uses rank 640 for both the
computational-mode basis and generator so it tests the replacement
architecture without a rank bottleneck.

The fit40 and a deterministic selection20 partition are used only to fit the
frozen generators.  The remaining assessment20 partition is materialized only
after the complete executor has been built.  It remains open development data,
not a held-out confirmation, guard, validation, or test split.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import json
import math
from pathlib import Path
import re
import sys

import torch

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .full_mlp_stack_evaluation import (
    evaluate_full_mlp_stack_conditions,
)
from .full_mlp_stack_generators import (
    FullMLPStackGeneratorFit,
    fit_full_mlp_stack_generators,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_executor import Gemma3FullMLPStackExecutor
from .gemma3_full_mlp_stack_rows import (
    FullMLPStackLayerRows,
    collect_full_mlp_stack_rows,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_FIT_EXPORT,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _layer_runtime_sites,
    _safe_tokenized_stream_metadata,
    _select_row_sites,
    load_development_prompt_export,
    load_gemma3_modal_generator_dev_artifact,
    validate_development_split_pair,
)
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    DEFAULT_BASE_ARTIFACT,
    _bind_batch_example_ids,
    _content_sha256s_from_safe_metadata,
    _require_sha256,
    _restore_upstream_analysis,
    _stream_content_sha256s,
    _upstream_stream_metadata,
    _validate_upstream_bindings,
    _validate_upstream_export_bindings,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_graph_rung_evaluation import (
    partition_development_export_for_interactions,
)
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragmentPlan,
)
from .parameter_layer_superfragments import (
    ParameterLayerSuperfragmentPlan,
    build_parameter_layer_superfragments,
)
from .streaming_analysis import iter_activation_score_gradient_rows


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_full_mlp_stack_dev_experiment",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-full-mlp-stack-dev-v1.pt"
)
DEFAULT_SELECTION_COUNT = 20
DEFAULT_MODE_RANKS = (640,)
DEFAULT_SELECTED_MODE_RANK = 640
DEFAULT_GENERATOR_RANKS = (640,)
DEFAULT_SELECTED_GENERATOR_RANK = 640
DEFAULT_GENERATOR_RIDGE = 1e-6
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _progress(message: str) -> None:
    print(f"[full-mlp-stack] {message}", file=sys.stderr, flush=True)


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
            f"{label} must contain unique, increasing positive integers"
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
    selection_count: int,
    mode_ranks: Sequence[int],
    selected_mode_rank: int,
    generator_ranks: Sequence[int],
    selected_generator_rank: int,
    generator_ridge: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("revision must be an exact lowercase commit hash")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be nonempty")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("device_name must be nonempty")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype is unsupported")
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be at least 2")
    if (
        type(tokenization_batch_size) is not int
        or tokenization_batch_size <= 0
    ):
        raise ValueError("tokenization_batch_size must be positive")
    if type(selection_count) is not int or not 0 < selection_count < 40:
        raise ValueError(
            "selection_count must leave nonempty selection and assessment "
            "partitions of eval40"
        )
    modes = _canonical_positive_ints(mode_ranks, label="mode_ranks")
    generators = _canonical_positive_ints(
        generator_ranks,
        label="generator_ranks",
    )
    if (
        type(selected_mode_rank) is not int
        or selected_mode_rank not in modes
    ):
        raise ValueError("selected_mode_rank must be in mode_ranks")
    if (
        type(selected_generator_rank) is not int
        or selected_generator_rank not in generators
    ):
        raise ValueError(
            "selected_generator_rank must be in generator_ranks"
        )
    if generators[-1] > selected_mode_rank:
        raise ValueError(
            "generator ranks cannot exceed the selected mode rank"
        )
    if (
        isinstance(generator_ridge, bool)
        or not isinstance(generator_ridge, (int, float))
        or not math.isfinite(float(generator_ridge))
        or float(generator_ridge) < 0.0
    ):
        raise ValueError("generator_ridge must be finite and nonnegative")
    path = Path(output)
    if path.suffix != ".pt":
        raise ValueError("full-stack artifact output must use .pt")
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite full-stack artifact")
    return modes, generators


def _validate_live_layers(
    adapter: Gemma3CausalLMAdapter,
    *,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    superfragment_plan: ParameterLayerSuperfragmentPlan,
) -> tuple[str, dict[int, torch.Tensor]]:
    live_layer_specs, leaf_activation_site, _ = _whole_model_layer_specs(
        adapter
    )
    expected_ordinals = tuple(range(len(live_layer_specs)))
    if (
        tuple(spec.layer_ordinal for spec in live_layer_specs)
        != expected_ordinals
        or tuple(
            value.layer_ordinal
            for value in superfragment_plan.superfragments
        )
        != expected_ordinals
    ):
        raise ValueError(
            "superfragments must exactly cover the live Gemma layer stack"
        )
    down_weights: dict[int, torch.Tensor] = {}
    for ordinal, spec in enumerate(live_layer_specs):
        fragments = fragment_plan.for_layer(ordinal)
        superfragment = superfragment_plan.for_layer(ordinal)
        if not fragments or any(
            fragment.layer_id != spec.layer_id for fragment in fragments
        ):
            raise ValueError("fragment layer catalog differs from live Gemma")
        input_site, output_site, _, down_weight = _layer_runtime_sites(
            adapter,
            ordinal,
        )
        if (
            input_site != superfragment.input_site
            or output_site != superfragment.output_site
            or superfragment.activation_site
            != fragments[0].activation_site
            or tuple(superfragment.channel_indices)
            != tuple(range(down_weight.shape[1]))
        ):
            raise ValueError(
                "full-layer superfragment differs from live Gemma MLP"
            )
        down_weights[ordinal] = down_weight
    return leaf_activation_site, down_weights


def _collect_rows(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    leaf_activation_site: str,
    down_projection_weights: Mapping[int, torch.Tensor],
) -> tuple[FullMLPStackLayerRows, ...]:
    fragments_by_layer = {
        ordinal: fragment_plan.for_layer(ordinal)
        for ordinal in sorted(down_projection_weights)
    }
    sites = tuple(
        dict.fromkeys(
            site
            for ordinal in sorted(fragments_by_layer)
            for site in (
                fragments_by_layer[ordinal][0].input_site,
                fragments_by_layer[ordinal][0].activation_site,
            )
        )
    )
    requested = tuple(dict.fromkeys((*sites, leaf_activation_site)))
    raw_rows = iter_activation_score_gradient_rows(
        adapter,
        batches,
        activation_names=requested,
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_activation_site,
        accumulation_dtype=torch.float64,
    )
    return collect_full_mlp_stack_rows(
        _select_row_sites(raw_rows, sites),
        fragments_by_layer=fragments_by_layer,
        down_projection_weights=down_projection_weights,
    )


def _split_metadata(
    *,
    fit_export: object,
    eval_export: object,
    partition: object,
    fit_safe: Mapping[str, object],
    upstream_eval: Mapping[str, object],
    selection_safe: Mapping[str, object],
    assessment_safe: Mapping[str, object],
    fit_content: tuple[str, ...],
    upstream_eval_content: tuple[str, ...],
    selection_content: tuple[str, ...],
    assessment_content: tuple[str, ...],
) -> dict[str, object]:
    return {
        "fit_export": fit_export.metadata(),
        "eval_export": eval_export.metadata(),
        "fit": {
            **dict(fit_safe),
            "role": "generator_fit",
            "content_sha256": fit_content,
        },
        "upstream_evaluation": {
            **dict(upstream_eval),
            "role": "development_partition_source",
            "content_sha256": upstream_eval_content,
        },
        "selection": {
            **dict(selection_safe),
            "role": "generator_selection",
            "content_sha256": selection_content,
        },
        "assessment": {
            **dict(assessment_safe),
            "role": "open_development_assessment",
            "content_sha256": assessment_content,
        },
        "partition": partition.metadata(),
        "provenance": {
            "assurance": "caller_declared_self_attested",
            "externally_authenticated": False,
            "selection_assessment_disjoint": True,
            "heldout_confirmation": False,
        },
    }


def run_gemma3_full_mlp_stack_dev_experiment(
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
    selection_count: int = DEFAULT_SELECTION_COUNT,
    mode_ranks: Sequence[int] = DEFAULT_MODE_RANKS,
    selected_mode_rank: int = DEFAULT_SELECTED_MODE_RANK,
    generator_ranks: Sequence[int] = DEFAULT_GENERATOR_RANKS,
    selected_generator_rank: int = DEFAULT_SELECTED_GENERATOR_RANK,
    generator_ridge: float = DEFAULT_GENERATOR_RIDGE,
) -> dict[str, object]:
    """Fit, execute, evaluate, and save one exhaustive all-MLP rung."""

    mode_ranks, generator_ranks = _validate_runner_preflight(
        revision=revision,
        output=output,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        selection_count=selection_count,
        mode_ranks=mode_ranks,
        selected_mode_rank=selected_mode_rank,
        generator_ranks=generator_ranks,
        selected_generator_rank=selected_generator_rank,
        generator_ridge=generator_ridge,
    )

    _progress("preflight: strict-load v3 Fisher analysis and partition eval40")
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
        selection_count=selection_count,
        expected_prompt_count=40,
    )
    fit_trace, catalog, fisher, clusters, fragment_plan = (
        _restore_upstream_analysis(upstream)
    )
    superfragment_plan = build_parameter_layer_superfragments(fragment_plan)

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
    if tuple(spec.layer_id for spec in fit_trace.layer_specs) != tuple(
        spec.id for spec in adapter.layers
    ):
        raise ValueError("live Gemma layers differ from the upstream trace")
    leaf_activation_site, down_weights = _validate_live_layers(
        adapter,
        fragment_plan=fragment_plan,
        superfragment_plan=superfragment_plan,
    )
    if (
        selected_mode_rank
        > min(value.output_width for value in superfragment_plan.superfragments)
        or selected_generator_rank
        > min(value.input_width for value in superfragment_plan.superfragments)
    ):
        raise ValueError("selected rank exceeds the live residual width")

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
        split_name="full_mlp_stack_generator_selection",
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
    upstream_fit = _upstream_stream_metadata(upstream, key="fit_tokenized")
    if fit_safe != upstream_fit:
        raise ValueError("live fit tokenization differs from upstream v3")
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

    _progress("rows: one native gradient replay per fit/selection split")
    fit_rows = _collect_rows(
        adapter,
        fit_batches,
        fragment_plan=fragment_plan,
        leaf_activation_site=leaf_activation_site,
        down_projection_weights=down_weights,
    )
    selection_rows = _collect_rows(
        adapter,
        selection_batches,
        fragment_plan=fragment_plan,
        leaf_activation_site=leaf_activation_site,
        down_projection_weights=down_weights,
    )
    fit_by_layer = {value.layer_ordinal: value for value in fit_rows}
    selection_by_layer = {
        value.layer_ordinal: value for value in selection_rows
    }
    del fit_rows, selection_rows

    _progress(
        "generators: fit one full-residual generator for every native MLP"
    )
    generator_fits: list[FullMLPStackGeneratorFit] = []
    for superfragment in superfragment_plan.superfragments:
        ordinal = superfragment.layer_ordinal
        _progress(
            f"generators: layer {ordinal + 1}/{superfragment_plan.layer_count} "
            f"({superfragment.mode_count} native channels, "
            f"mode rank {selected_mode_rank}, "
            f"generator rank {selected_generator_rank})"
        )
        generator_fits.append(
            fit_full_mlp_stack_generators(
                fit_by_layer.pop(ordinal),
                selection_by_layer.pop(ordinal),
                superfragment=superfragment,
                source_model_sha256=model_fingerprint,
                parameter_catalog_sha256=catalog.artifact_sha256,
                fisher_coupling_sha256=fisher.artifact_sha256,
                superfragment_plan_sha256=(
                    superfragment_plan.artifact_sha256
                ),
                fit_split_sha256=fit_split_sha256,
                eval_split_sha256=selection_split_sha256,
                mode_ranks=mode_ranks,
                selected_mode_rank=selected_mode_rank,
                generator_ranks=generator_ranks,
                selected_generator_rank=selected_generator_rank,
                ridge=generator_ridge,
            )
        )
        gc.collect()
    if fit_by_layer or selection_by_layer:
        raise RuntimeError("full-stack row tables were not consumed exactly")
    frozen_fits = tuple(generator_fits)
    replacements = tuple(
        Gemma3ModalGeneratorReplacement(
            layer_ordinal=fit.superfragment.layer_ordinal,
            removed_mode_indices=fit.superfragment.channel_indices,
            generator_plan=fit.executable_plan,
        )
        for fit in frozen_fits
    )
    executor = Gemma3FullMLPStackExecutor(adapter, replacements)
    if adapter.model_fingerprint() != model_fingerprint:
        raise RuntimeError("full-stack compilation mutated the source model")

    del fit_batches, selection_batches, down_weights, generator_fits
    gc.collect()
    _progress("assessment: tokenize assessment20 after all generators freeze")
    assessment_batches, assessment_stream = _materialize_split(
        tokenizer,
        partition.assessment.prompts,
        split_name="full_mlp_stack_open_development_assessment",
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
    if (
        set(selection_content) & set(assessment_content)
        or set(selection_content) | set(assessment_content)
        != set(upstream_eval_content)
    ):
        raise ValueError(
            "selection20 and assessment20 do not exactly partition eval40"
        )

    _progress("evaluate: native, full-stack generated, and matched deletion")
    evaluation = evaluate_full_mlp_stack_conditions(
        adapter,
        executor,
        assessment_batches,
        expected_example_ids=partition.assessment.prompt_sha256s,
        expected_mode_counts_by_layer=tuple(
            value.mode_count for value in superfragment_plan.superfragments
        ),
        assessment_role="open_development_assessment",
    )
    evaluation["assessment_split_sha256"] = assessment_split_sha256

    # Imported late so the numerical runner remains independently testable
    # while the strict serialization boundary stays in its own module.
    from .gemma3_full_mlp_stack_artifact import (
        load_gemma3_full_mlp_stack_artifact,
        save_gemma3_full_mlp_stack_artifact,
    )

    splits = _split_metadata(
        fit_export=fit_export,
        eval_export=eval_export,
        partition=partition,
        fit_safe=fit_safe,
        upstream_eval=upstream_eval,
        selection_safe=selection_safe,
        assessment_safe=assessment_safe,
        fit_content=fit_content,
        upstream_eval_content=upstream_eval_content,
        selection_content=selection_content,
        assessment_content=assessment_content,
    )
    _progress("artifact: save strict source-safe exhaustive result")
    report = save_gemma3_full_mlp_stack_artifact(
        output,
        model={
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_commit": revision,
            "adapter_model_fingerprint": model_fingerprint,
            "source_whole_model_learned_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "local_files_only": True,
        },
        protocol={
            "scope": "full_native_mlp_stack_replacement",
            "transformer_layer_count": superfragment_plan.layer_count,
            "source_fragment_count": (
                superfragment_plan.source_fragment_count
            ),
            "removed_mode_count": superfragment_plan.assigned_group_count,
            "mode_ranks": mode_ranks,
            "selected_mode_rank": selected_mode_rank,
            "generator_ranks": generator_ranks,
            "selected_generator_rank": selected_generator_rank,
            "generator_ridge": float(generator_ridge),
            "fit_rule": "fisher_weighted_full_layer_before_modes",
            "execution_path": "edgeless_dense_fused_residual_generators",
            "native_components_retained": (
                "embeddings",
                "attention",
                "normalization",
                "language_model_head",
            ),
            "local_files_only": True,
        },
        splits=splits,
        upstream_metadata={
            "source_schema": upstream["schema"],
            "source_format_version": upstream["format_version"],
            "source_scientific_payload_sha256": _require_sha256(
                upstream.get("scientific_payload_sha256"),
                label="upstream scientific payload sha256",
            ),
            "fit_prompt_trace_sha256": fit_trace.artifact_sha256,
            "parameter_catalog_sha256": catalog.artifact_sha256,
            "fisher_coupling_sha256": fisher.artifact_sha256,
            "parameter_clusters_sha256": clusters.artifact_sha256,
            "parameter_cluster_fragments_sha256": (
                fragment_plan.artifact_sha256
            ),
        },
        superfragment_plan=superfragment_plan,
        generator_fits=frozen_fits,
        evaluation=evaluation,
    )
    load_gemma3_full_mlp_stack_artifact(output)
    if adapter.model_fingerprint() != model_fingerprint:
        raise RuntimeError("full-stack experiment mutated the source model")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local-only exhaustive all-block Gemma MLP "
            "modal-generator rung."
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
        "--selection-count",
        type=int,
        default=DEFAULT_SELECTION_COUNT,
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
    parser.add_argument(
        "--generator-ridge",
        type=float,
        default=DEFAULT_GENERATOR_RIDGE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_full_mlp_stack_dev_experiment(
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
        selection_count=arguments.selection_count,
        mode_ranks=arguments.mode_ranks,
        selected_mode_rank=arguments.selected_mode_rank,
        generator_ranks=arguments.generator_ranks,
        selected_generator_rank=arguments.selected_generator_rank,
        generator_ridge=arguments.generator_ridge,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
