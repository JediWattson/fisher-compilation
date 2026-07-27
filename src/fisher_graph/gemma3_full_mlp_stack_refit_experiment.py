"""Sequentially refit the late Gemma MLP generators on compiled trajectories.

The frozen full-stack artifact was fitted on native layer inputs.  Its prefix
trajectory shows a marked interaction cliff after generated layers 0 through
9.  This development rung keeps those first ten generators fixed, then refits
layers 10 through 17 one at a time on the distribution produced by the
already-compiled prefix:

``frozen 0:9 -> refit 10 -> refit 11 -> ... -> refit 17``.

For each layer, fit40 and selection20 are replayed exactly once under a single
authenticated prefix overlay.  The current layer and its native suffix remain
native while rows are collected.  The old and refitted plans are measured on
the same ephemeral rows, the replacement is frozen, and only then may the next
layer be fitted.  Assessment20 is not materialized until all eight refits have
finished.

This remains an open-development experiment.  It performs no rank selection,
does not mutate source weights, and makes no held-out, compression, latency,
or kernel-speed claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CalibrationBatch, CausalLanguageModelNLL
from .full_mlp_stack_evaluation import evaluate_full_mlp_stack_conditions
from .full_mlp_stack_generators import (
    FullMLPStackGeneratorFit,
    fit_full_mlp_stack_generators,
)
from .full_mlp_stack_trajectory import (
    FrozenFullMLPStackTrajectoryExecutor,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    _model_provenance,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_artifact import (
    load_gemma3_full_mlp_stack_artifact,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_GENERATOR_RIDGE,
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
    _validate_live_layers,
)
from .gemma3_full_mlp_stack_executor import Gemma3FullMLPStackExecutor
from .gemma3_full_mlp_stack_refit_artifact import (
    compiled_prefix_catalog_sha256,
    frozen_baseline_conditions_sha256,
    load_gemma3_full_mlp_stack_refit_artifact,
    save_gemma3_full_mlp_stack_refit_artifact,
    trajectory_breakpoint_row_sha256,
)
from .gemma3_full_mlp_stack_rows import (
    FullMLPStackLayerRows,
    collect_full_mlp_stack_layer_rows,
)
from .gemma3_full_mlp_stack_trajectory_artifact import (
    load_gemma3_full_mlp_stack_trajectory_artifact,
)
from .gemma3_full_mlp_stack_trajectory_experiment import (
    DEFAULT_OUTPUT as DEFAULT_TRAJECTORY_ARTIFACT,
    DEFAULT_VOCABULARY_CHUNK_SIZE,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_FIT_EXPORT,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _safe_tokenized_stream_metadata,
    _select_row_sites,
    load_development_prompt_export,
    validate_development_split_pair,
)
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    _bind_batch_example_ids,
    _stream_content_sha256s,
)
from .modal_generators import ModalGeneratorPlan, _metrics
from .modal_graph_rung_evaluation import (
    partition_development_export_for_interactions,
)
from .parameter_cluster_fragments import (
    ParameterClusterLayerFragment,
    ParameterClusterLayerFragmentPlan,
)
from .parameter_layer_superfragments import (
    ParameterLayerSuperfragmentPlan,
)
from .streaming_analysis import (
    ActivationScoreGradientRows,
    iter_activation_score_gradient_rows,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_full_mlp_stack_refit_experiment",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-full-mlp-stack-refit-dev-v1.pt"
)
REFIT_START_LAYER = 10
EXPECTED_LAYER_COUNT = 18
REFIT_LAYER_ORDINALS = tuple(range(REFIT_START_LAYER, EXPECTED_LAYER_COUNT))
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
def _progress(message: str) -> None:
    print(f"[full-mlp-refit] {message}", file=sys.stderr, flush=True)


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_runner_preflight(
    *,
    revision: str,
    output: Path | str,
    base_artifact_path: Path | str,
    trajectory_artifact_path: Path | str,
    model_id: str,
    device_name: str,
    dtype: str,
    max_length: int,
    tokenization_batch_size: int,
    vocabulary_chunk_size: int,
    refit_start_layer: int,
) -> None:
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
    if (
        type(vocabulary_chunk_size) is not int
        or vocabulary_chunk_size <= 0
    ):
        raise ValueError("vocabulary_chunk_size must be positive")
    if refit_start_layer != REFIT_START_LAYER:
        raise ValueError(
            "this frozen protocol refits exactly zero-based layers 10 through 17"
        )
    if Path(base_artifact_path).suffix != ".pt":
        raise ValueError("frozen full-stack source artifact must use .pt")
    if Path(trajectory_artifact_path).suffix != ".json":
        raise ValueError("frozen trajectory artifact must use .json")
    destination = Path(output)
    if destination.suffix != ".pt":
        raise ValueError("refit output must use .pt")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError("refusing to overwrite refit artifact")


def _metric_mapping_close(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    fields = {
        "nll_per_token",
        "delta_nll_per_token",
        "native_to_candidate_kl_per_token",
        "top1_agreement_to_native",
    }
    if set(actual) != fields or set(expected) != fields:
        raise ValueError(f"{label} metric fields differ")
    for field in sorted(fields):
        left = actual[field]
        right = expected[field]
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
            or not math.isclose(
                float(left),
                float(right),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{label} {field} differs from frozen baseline")


def _split_without_annotations(
    value: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: child
        for key, child in value.items()
        if key not in {"role", "content_sha256"}
    }


def _validate_frozen_source_bindings(
    source: Mapping[str, object],
    trajectory: Mapping[str, object],
    *,
    source_file_sha256: str,
    revision: str,
    model_id: str,
    fit_export_metadata: Mapping[str, object],
    eval_export_metadata: Mapping[str, object],
    partition_metadata: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    """Cross-authenticate both frozen artifacts and the requested exports."""

    source_model = _require_mapping(source.get("model"), label="source model")
    source_protocol = _require_mapping(
        source.get("protocol"),
        label="source protocol",
    )
    source_splits = _require_mapping(
        source.get("splits"),
        label="source splits",
    )
    source_evaluation = _require_mapping(
        source.get("evaluation"),
        label="source evaluation",
    )
    source_conditions = _require_mapping(
        source_evaluation.get("conditions"),
        label="source conditions",
    )
    if (
        source_model.get("model_id") != model_id
        or source_model.get("requested_revision") != revision
        or source_model.get("resolved_commit") != revision
        or source_model.get("local_files_only") is not True
    ):
        raise ValueError("frozen source model binding differs from the request")
    if (
        source_protocol.get("scope")
        != "full_native_mlp_stack_replacement"
        or source_protocol.get("transformer_layer_count")
        != EXPECTED_LAYER_COUNT
        or source_protocol.get("mode_ranks")
        != (source_protocol.get("selected_mode_rank"),)
        or source_protocol.get("generator_ranks")
        != (source_protocol.get("selected_generator_rank"),)
        or source_protocol.get("selected_mode_rank")
        != source_protocol.get("selected_generator_rank")
        or source_protocol.get("local_files_only") is not True
    ):
        raise ValueError(
            "frozen source is not the fixed full-rank exhaustive rung"
        )
    if (
        source_splits.get("fit_export") != dict(fit_export_metadata)
        or source_splits.get("eval_export") != dict(eval_export_metadata)
        or source_splits.get("partition") != dict(partition_metadata)
    ):
        raise ValueError("requested exports or partition differ from source")

    trajectory_source = _require_mapping(
        trajectory.get("frozen_source_artifact"),
        label="trajectory frozen source",
    )
    trajectory_model = _require_mapping(
        trajectory.get("model"),
        label="trajectory model",
    )
    trajectory_protocol = _require_mapping(
        trajectory.get("protocol"),
        label="trajectory protocol",
    )
    trajectory_splits = _require_mapping(
        trajectory.get("splits"),
        label="trajectory splits",
    )
    trajectory_evaluation = _require_mapping(
        trajectory.get("evaluation"),
        label="trajectory evaluation",
    )
    source_payload_sha256 = _require_sha256(
        source.get("scientific_payload_sha256"),
        label="source scientific payload sha256",
    )
    if (
        trajectory_source.get("source_schema") != source.get("schema")
        or trajectory_source.get("source_format_version")
        != source.get("format_version")
        or trajectory_source.get("source_scope")
        != source_protocol.get("scope")
        or trajectory_source.get("artifact_file_sha256")
        != source_file_sha256
        or trajectory_source.get("scientific_payload_sha256")
        != source_payload_sha256
        or trajectory_source.get("frozen_before_trajectory") is not True
    ):
        raise ValueError("trajectory does not bind the exact frozen source")
    if (
        trajectory_model.get("model_id") != model_id
        or trajectory_model.get("requested_revision") != revision
        or trajectory_model.get("resolved_commit") != revision
        or trajectory_model.get("adapter_model_fingerprint")
        != source_model.get("adapter_model_fingerprint")
        or trajectory_model.get("source_whole_model_learned_parameters")
        != source_model.get("source_whole_model_learned_parameters")
        or trajectory_model.get("local_files_only") is not True
    ):
        raise ValueError("trajectory and source model bindings differ")
    if (
        trajectory_protocol.get("scope")
        != "frozen_full_native_mlp_stack_trajectory_ladder"
        or trajectory_protocol.get("transformer_layer_count")
        != EXPECTED_LAYER_COUNT
        or trajectory_protocol.get("generators_frozen") is not True
        or trajectory_protocol.get("generator_refit_performed") is not False
        or trajectory_protocol.get("generator_rank_selection_performed")
        is not False
    ):
        raise ValueError("trajectory is not the frozen no-refit diagnostic")

    source_assessment = _require_mapping(
        source_splits.get("assessment"),
        label="source assessment",
    )
    trajectory_assessment = _require_mapping(
        trajectory_splits.get("assessment"),
        label="trajectory assessment",
    )
    source_assessment_content = source_assessment.get("content_sha256")
    if (
        trajectory_assessment.get("role")
        != source_assessment.get("role")
        or trajectory_assessment.get("serialized_sha256")
        != source_assessment.get("serialized_sha256")
        or tuple(trajectory_assessment.get("content_sha256", ()))
        != tuple(source_assessment_content or ())
        or trajectory_assessment.get("example_count")
        != source_assessment.get("sequences")
        or trajectory_assessment.get("logical_valid_tokens")
        != _require_mapping(
            source_assessment.get("valid_tokens"),
            label="source assessment valid tokens",
        ).get("total")
        or trajectory_assessment.get("supervised_tokens")
        != _require_mapping(
            source_assessment.get("supervised_positions"),
            label="source assessment supervised positions",
        ).get("total")
    ):
        raise ValueError("trajectory and source assessment bindings differ")

    _metric_mapping_close(
        _require_mapping(
            trajectory_evaluation.get("native"),
            label="trajectory native",
        ),
        _require_mapping(source_conditions.get("native"), label="source native"),
        label="trajectory native",
    )
    prefix_ladder = trajectory_evaluation.get("prefix_ladder")
    if (
        isinstance(prefix_ladder, (str, bytes))
        or not isinstance(prefix_ladder, Sequence)
        or len(prefix_ladder) != EXPECTED_LAYER_COUNT
    ):
        raise ValueError("trajectory prefix ladder must have exactly 18 depths")
    breakpoint_row = _require_mapping(
        prefix_ladder[REFIT_START_LAYER - 1],
        label="trajectory breakpoint row",
    )
    breakpoint_resources = _require_mapping(
        breakpoint_row.get("resources"),
        label="trajectory breakpoint resources",
    )
    if (
        breakpoint_row.get("depth") != REFIT_START_LAYER
        or tuple(breakpoint_resources.get("replaced_layer_ordinals", ()))
        != tuple(range(REFIT_START_LAYER))
    ):
        raise ValueError("trajectory does not bind the layer-10 breakpoint")
    trajectory_endpoint = _require_mapping(
        prefix_ladder[-1],
        label="trajectory full-stack endpoint",
    )
    _metric_mapping_close(
        _require_mapping(
            trajectory_endpoint.get("metrics"),
            label="trajectory full-stack endpoint metrics",
        ),
        _require_mapping(
            source_conditions.get("generated_full_stack"),
            label="source generated full stack",
        ),
        label="trajectory full-stack endpoint",
    )
    return (
        source_model,
        source_protocol,
        source_splits,
        source_conditions,
        breakpoint_row,
    )


def _prefix_catalog_sha256(plan_sha256s: Sequence[str]) -> str:
    """Compatibility wrapper around the strict artifact catalog helper."""

    values = tuple(plan_sha256s)
    return compiled_prefix_catalog_sha256(
        tuple(range(len(values))),
        values,
    )


def _plan_metrics(
    rows: FullMLPStackLayerRows,
    plan: ModalGeneratorPlan,
) -> dict[str, object]:
    """Measure one plan against an immutable row target."""

    if not isinstance(rows, FullMLPStackLayerRows):
        raise TypeError("rows must be FullMLPStackLayerRows")
    if not isinstance(plan, ModalGeneratorPlan):
        raise TypeError("plan must be ModalGeneratorPlan")
    before = rows.row_key_sha256
    prediction = plan.apply(rows.inputs)
    result = _metrics(
        rows.contributions,
        prediction,
        rows.fisher_weights,
    ).metadata()
    if rows.row_key_sha256 != before:
        raise RuntimeError("plan evaluation mutated row identity")
    return result


def _collect_layer_rows_under_prefix(
    adapter: Gemma3CausalLMAdapter,
    batches: Sequence[CalibrationBatch],
    *,
    fragments: Sequence[ParameterClusterLayerFragment],
    down_projection_weight: Tensor,
    executor: FrozenFullMLPStackTrajectoryExecutor,
    generated_layer_ordinals: Sequence[int],
    row_factory: Callable[..., Iterable[ActivationScoreGradientRows]] = (
        iter_activation_score_gradient_rows
    ),
    row_collector: Callable[..., FullMLPStackLayerRows] = (
        collect_full_mlp_stack_layer_rows
    ),
) -> FullMLPStackLayerRows:
    """Consume a whole split in one authenticated compiled-prefix overlay."""

    materialized = tuple(batches)
    if not materialized:
        raise ValueError("batches must be nonempty")
    selected_fragments = tuple(fragments)
    if not selected_fragments:
        raise ValueError("fragments must be nonempty")
    input_site = selected_fragments[0].input_site
    activation_site = selected_fragments[0].activation_site
    sites = (input_site, activation_site)

    def consume() -> FullMLPStackLayerRows:
        raw_rows = row_factory(
            adapter,
            materialized,
            activation_names=sites,
            score_objective=CausalLanguageModelNLL(),
            # The prefix still executes and supplies this exact value.  Making
            # the current normalized input the detached leaf only discards the
            # graph before the teacher layer; all current-layer and suffix
            # gradients, including the down-input Fisher gradient, are exact.
            leaf_activation_name=input_site,
            accumulation_dtype=torch.float64,
        )
        return row_collector(
            _select_row_sites(raw_rows, sites),
            fragments=selected_fragments,
            down_projection_weight=down_projection_weight,
        )

    result = executor.run_with_subset_overlay(
        generated_layer_ordinals=tuple(generated_layer_ordinals),
        callback=consume,
        expected_forward_calls=sum(batch.batch_size for batch in materialized),
    )
    if not isinstance(result, FullMLPStackLayerRows):
        raise TypeError("compiled-prefix row collector returned invalid rows")
    if result.layer_ordinal != selected_fragments[0].layer_ordinal:
        raise ValueError("compiled-prefix rows describe the wrong layer")
    return result


def _source_fit_record(
    fit: FullMLPStackGeneratorFit,
) -> dict[str, object]:
    fit.validate_integrity()
    resources = fit.resource_metadata
    return {
        "layer_ordinal": fit.superfragment.layer_ordinal,
        "layer_id": fit.superfragment.layer_id,
        "input_site": fit.superfragment.input_site,
        "output_site": fit.superfragment.output_site,
        "input_width": fit.superfragment.input_width,
        "intermediate_width": fit.superfragment.mode_count,
        "residual_width": fit.superfragment.output_width,
        "source_fit_sha256": fit.artifact_sha256,
        "superfragment_sha256": fit.superfragment.artifact_sha256,
        "superfragment_plan_sha256": fit.superfragment_plan_sha256,
        "source_model_sha256": fit.superfragment.source_model_sha256,
        "parameter_catalog_sha256": (
            fit.superfragment.parameter_catalog_sha256
        ),
        "source_fisher_coupling_sha256": (
            fit.superfragment.source_fisher_coupling_sha256
        ),
        "source_fragment_plan_sha256": (
            fit.superfragment.source_fragment_plan_sha256
        ),
        "source_cluster_plan_sha256": (
            fit.superfragment.source_cluster_plan_sha256
        ),
        "dense_plan_sha256": fit.executable_plan.artifact_sha256,
        "selected_mode_rank": fit.selected_basis.rank,
        "selected_generator_rank": fit.executable_plan.rank,
        "native_mlp_parameter_count": resources[
            "native_mlp_parameter_count"
        ],
        "dense_fused_parameter_count": resources[
            "dense_fused_parameter_count"
        ],
        "dense_fused_macs_per_token": resources[
            "dense_fused_macs_per_token"
        ],
    }


def _restore_source_generator_catalog(
    generator_states: object,
    *,
    restore_fit: Callable[
        [Mapping[str, object]], FullMLPStackGeneratorFit
    ] = FullMLPStackGeneratorFit.from_state_dict,
) -> tuple[
    tuple[Gemma3ModalGeneratorReplacement, ...],
    tuple[dict[str, object], ...],
]:
    """Authenticate one fit at a time and release its analysis tensors."""

    if (
        isinstance(generator_states, (str, bytes))
        or not isinstance(generator_states, Sequence)
        or len(generator_states) != EXPECTED_LAYER_COUNT
    ):
        raise ValueError("source must contain exactly 18 generator fits")
    pending = (
        generator_states
        if isinstance(generator_states, list)
        else list(generator_states)
    )
    replacements: list[Gemma3ModalGeneratorReplacement] = []
    records: list[dict[str, object]] = []
    for expected_ordinal in range(EXPECTED_LAYER_COUNT):
        state = pending[expected_ordinal]
        if not isinstance(state, Mapping):
            raise TypeError("generator fit state must be a mapping")
        fit = restore_fit(state)
        if fit.superfragment.layer_ordinal != expected_ordinal:
            raise ValueError("source generator fits are not in exact layer order")
        records.append(_source_fit_record(fit))
        replacements.append(
            Gemma3ModalGeneratorReplacement(
                layer_ordinal=expected_ordinal,
                removed_mode_indices=fit.superfragment.channel_indices,
                generator_plan=fit.executable_plan,
            )
        )
        if isinstance(pending, list):
            pending[expected_ordinal] = None
        del fit
        gc.collect()
    return tuple(replacements), tuple(records)


def _fit_resource_signature(fit: FullMLPStackGeneratorFit) -> dict[str, object]:
    return dict(fit.resource_metadata)


def _replacement_plan_sha256(replacement: object) -> str:
    plan = getattr(replacement, "generator_plan", None)
    return _require_sha256(
        getattr(plan, "artifact_sha256", None),
        label="replacement generator plan sha256",
    )


def _sequentially_refit_layers(
    adapter: Gemma3CausalLMAdapter,
    fit_batches: Sequence[CalibrationBatch],
    selection_batches: Sequence[CalibrationBatch],
    *,
    fragment_plan: ParameterClusterLayerFragmentPlan,
    superfragment_plan: ParameterLayerSuperfragmentPlan,
    down_projection_weights: Mapping[int, Tensor],
    source_replacements: Sequence[Gemma3ModalGeneratorReplacement],
    source_fit_records: Sequence[Mapping[str, object]],
    source_model_sha256: str,
    parameter_catalog_sha256: str,
    fisher_coupling_sha256: str,
    fit_split_sha256: str,
    selection_split_sha256: str,
    mode_ranks: Sequence[int],
    selected_mode_rank: int,
    generator_ranks: Sequence[int],
    selected_generator_rank: int,
    generator_ridge: float,
    refit_start_layer: int = REFIT_START_LAYER,
    trajectory_executor_factory: Callable[..., Any] = (
        FrozenFullMLPStackTrajectoryExecutor
    ),
    collect_rows: Callable[..., FullMLPStackLayerRows] = (
        _collect_layer_rows_under_prefix
    ),
    fit_layer: Callable[..., FullMLPStackGeneratorFit] = (
        fit_full_mlp_stack_generators
    ),
    plan_metrics: Callable[
        [FullMLPStackLayerRows, ModalGeneratorPlan],
        Mapping[str, object],
    ] = _plan_metrics,
    replacement_factory: Callable[..., Gemma3ModalGeneratorReplacement] = (
        Gemma3ModalGeneratorReplacement
    ),
) -> tuple[
    tuple[Gemma3ModalGeneratorReplacement, ...],
    tuple[FullMLPStackGeneratorFit, ...],
    tuple[dict[str, object], ...],
]:
    """Refit exact layers 10..17 and freeze each result before advancing."""

    if refit_start_layer != REFIT_START_LAYER:
        raise ValueError("sequential refit must start at zero-based layer 10")
    current = list(source_replacements)
    source_records = tuple(source_fit_records)
    if (
        len(current) != EXPECTED_LAYER_COUNT
        or len(source_records) != EXPECTED_LAYER_COUNT
        or set(down_projection_weights) != set(range(EXPECTED_LAYER_COUNT))
        or superfragment_plan.layer_count != EXPECTED_LAYER_COUNT
    ):
        raise ValueError("sequential refit inputs must exactly cover 18 layers")
    if tuple(
        getattr(value, "layer_ordinal", None) for value in current
    ) != tuple(range(EXPECTED_LAYER_COUNT)):
        raise ValueError("source replacements are not in exact layer order")
    if tuple(
        value.get("layer_ordinal") for value in source_records
    ) != tuple(range(EXPECTED_LAYER_COUNT)):
        raise ValueError("source fit records are not in exact layer order")

    updated_fits: list[FullMLPStackGeneratorFit] = []
    layer_refits: list[dict[str, object]] = []
    for ordinal in REFIT_LAYER_ORDINALS:
        prefix_ordinals = tuple(range(ordinal))
        prefix_plan_sha256s = tuple(
            _replacement_plan_sha256(current[index])
            for index in prefix_ordinals
        )
        executor = trajectory_executor_factory(adapter, tuple(current))
        fragments = fragment_plan.for_layer(ordinal)
        fit_rows = collect_rows(
            adapter,
            fit_batches,
            fragments=fragments,
            down_projection_weight=down_projection_weights[ordinal],
            executor=executor,
            generated_layer_ordinals=prefix_ordinals,
        )
        selection_rows = collect_rows(
            adapter,
            selection_batches,
            fragments=fragments,
            down_projection_weight=down_projection_weights[ordinal],
            executor=executor,
            generated_layer_ordinals=prefix_ordinals,
        )
        del executor
        gc.collect()

        old_plan = current[ordinal].generator_plan
        old_metrics = {
            "fit": dict(plan_metrics(fit_rows, old_plan)),
            "selection": dict(plan_metrics(selection_rows, old_plan)),
        }
        superfragment = superfragment_plan.for_layer(ordinal)
        fitted = fit_layer(
            fit_rows,
            selection_rows,
            superfragment=superfragment,
            source_model_sha256=source_model_sha256,
            parameter_catalog_sha256=parameter_catalog_sha256,
            fisher_coupling_sha256=fisher_coupling_sha256,
            superfragment_plan_sha256=superfragment_plan.artifact_sha256,
            fit_split_sha256=fit_split_sha256,
            eval_split_sha256=selection_split_sha256,
            mode_ranks=tuple(mode_ranks),
            selected_mode_rank=selected_mode_rank,
            generator_ranks=tuple(generator_ranks),
            selected_generator_rank=selected_generator_rank,
            ridge=float(generator_ridge),
        )
        if fitted.superfragment.layer_ordinal != ordinal:
            raise ValueError("refitted generator describes the wrong layer")
        refit_resource = _fit_resource_signature(fitted)
        source_resource_signature = {
            name: source_records[ordinal].get(name)
            for name in (
                "native_mlp_parameter_count",
                "dense_fused_parameter_count",
                "dense_fused_macs_per_token",
            )
        }
        refit_resource_signature = {
            name: refit_resource.get(name)
            for name in source_resource_signature
        }
        if source_resource_signature != refit_resource_signature:
            raise ValueError(
                "refit changed layer rank, parameter, or compute resources"
            )
        if (
            fitted.selected_basis.rank != selected_mode_rank
            or fitted.executable_plan.rank != selected_generator_rank
            or fitted.executable_plan.parameter_count
            != old_plan.parameter_count
            or fitted.executable_plan.macs_per_token
            != old_plan.macs_per_token
        ):
            raise ValueError("refit changed fixed executable capacity")

        new_metrics = {
            "fit": dict(plan_metrics(fit_rows, fitted.executable_plan)),
            "selection": dict(
                plan_metrics(selection_rows, fitted.executable_plan)
            ),
        }
        replacement = replacement_factory(
            layer_ordinal=ordinal,
            removed_mode_indices=superfragment.channel_indices,
            generator_plan=fitted.executable_plan,
        )
        current[ordinal] = replacement
        layer_refits.append(
            {
                "layer_ordinal": ordinal,
                "generated_prefix_ordinals": prefix_ordinals,
                "generated_prefix_plan_sha256s": prefix_plan_sha256s,
                "generated_prefix_catalog_sha256": (
                    compiled_prefix_catalog_sha256(
                        prefix_ordinals,
                        prefix_plan_sha256s,
                    )
                ),
                "source_fit_sha256": source_records[ordinal][
                    "source_fit_sha256"
                ],
                "refit_fit_sha256": fitted.artifact_sha256,
                "fit_row_key_sha256": fit_rows.row_key_sha256,
                "selection_row_key_sha256": selection_rows.row_key_sha256,
                "fit_observations": fit_rows.observations,
                "fit_sequences": fit_rows.sequences,
                "selection_observations": selection_rows.observations,
                "selection_sequences": selection_rows.sequences,
                "old_selected_mode_rank": source_records[ordinal][
                    "selected_mode_rank"
                ],
                "old_selected_generator_rank": source_records[ordinal][
                    "selected_generator_rank"
                ],
                "refit_selected_mode_rank": selected_mode_rank,
                "refit_selected_generator_rank": selected_generator_rank,
                "old_plan_fit_metrics": old_metrics["fit"],
                "old_plan_selection_metrics": old_metrics["selection"],
                "refit_plan_fit_metrics": new_metrics["fit"],
                "refit_plan_selection_metrics": new_metrics["selection"],
                "refit_resource_metadata": refit_resource,
            }
        )
        updated_fits.append(fitted)
        del fit_rows, selection_rows
        gc.collect()
    if tuple(
        value["layer_ordinal"] for value in layer_refits
    ) != REFIT_LAYER_ORDINALS:
        raise RuntimeError("sequential refit did not consume exact layer order")
    return tuple(current), tuple(updated_fits), tuple(layer_refits)


def _live_split_matches_frozen(
    live: Mapping[str, object],
    frozen: Mapping[str, object],
    *,
    label: str,
) -> None:
    # Content membership is authenticated separately after all three streams
    # are available.  The safe live metadata now carries that derived field
    # itself, while older frozen rows add both content and a semantic role.
    if _split_without_annotations(live) != _split_without_annotations(frozen):
        raise ValueError(f"live {label} tokenization differs from frozen source")


def _artifact_split_entry(
    frozen: Mapping[str, object],
) -> dict[str, object]:
    valid_tokens = _require_mapping(
        frozen.get("valid_tokens"),
        label="frozen split valid tokens",
    ).get("total")
    supervised_tokens = _require_mapping(
        frozen.get("supervised_positions"),
        label="frozen split supervised positions",
    ).get("total")
    example_count = frozen.get("sequences")
    content = frozen.get("content_sha256")
    if (
        type(example_count) is not int
        or example_count <= 0
        or type(valid_tokens) is not int
        or valid_tokens <= 0
        or type(supervised_tokens) is not int
        or supervised_tokens <= 0
        or isinstance(content, (str, bytes))
        or not isinstance(content, Sequence)
    ):
        raise ValueError("frozen split token or membership totals are invalid")
    return {
        "role": frozen["role"],
        "serialized_sha256": frozen["serialized_sha256"],
        "content_sha256": tuple(content),
        "example_count": example_count,
        "logical_valid_tokens": valid_tokens,
        "supervised_tokens": supervised_tokens,
    }


def _publish_verified_refit(
    *,
    output: Path | str,
    adapter: Gemma3CausalLMAdapter,
    expected_model_fingerprint: str,
    source_path: Path,
    expected_source_file_sha256: str,
    trajectory_path: Path,
    expected_trajectory_file_sha256: str,
    save: Callable[..., dict[str, object]],
    load: Callable[[Path | str], Mapping[str, object]],
    save_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Save, strict-load, and remove only owned outputs on postcondition drift."""

    destination = Path(output)
    sidecar = destination.with_suffix(".json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite refit artifact")
    try:
        report = save(destination, **dict(save_kwargs))
        load(destination)
        if adapter.model_fingerprint() != expected_model_fingerprint:
            raise RuntimeError("refit publication mutated the source model")
        if _file_sha256(source_path) != expected_source_file_sha256:
            raise RuntimeError("frozen source artifact changed during refit")
        if _file_sha256(trajectory_path) != expected_trajectory_file_sha256:
            raise RuntimeError("frozen trajectory artifact changed during refit")
        return report
    except BaseException:
        for owned in (destination, sidecar):
            try:
                owned.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def run_gemma3_full_mlp_stack_refit_experiment(
    *,
    fit_export_path: Path | str = DEFAULT_FIT_EXPORT,
    eval_export_path: Path | str = DEFAULT_EVAL_EXPORT,
    revision: str,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    trajectory_artifact_path: Path | str = DEFAULT_TRAJECTORY_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    vocabulary_chunk_size: int = DEFAULT_VOCABULARY_CHUNK_SIZE,
    refit_start_layer: int = REFIT_START_LAYER,
) -> dict[str, object]:
    """Run and save the fixed-capacity sequential compiled-trajectory refit."""

    _validate_runner_preflight(
        revision=revision,
        output=output,
        base_artifact_path=base_artifact_path,
        trajectory_artifact_path=trajectory_artifact_path,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        vocabulary_chunk_size=vocabulary_chunk_size,
        refit_start_layer=refit_start_layer,
    )
    source_path = Path(base_artifact_path)
    trajectory_path = Path(trajectory_artifact_path)
    _progress("source: hash and strict-load full-stack plus trajectory")
    source_file_sha256 = _file_sha256(source_path)
    trajectory_file_sha256 = _file_sha256(trajectory_path)
    source = load_gemma3_full_mlp_stack_artifact(source_path)
    trajectory = load_gemma3_full_mlp_stack_trajectory_artifact(
        trajectory_path
    )

    fit_export = load_development_prompt_export(fit_export_path)
    eval_export = load_development_prompt_export(eval_export_path)
    validate_development_split_pair(fit_export, eval_export)
    source_splits_raw = _require_mapping(
        source.get("splits"),
        label="source splits",
    )
    partition_source = _require_mapping(
        source_splits_raw.get("partition"),
        label="source partition",
    )
    selection_count = partition_source.get("selection_prompt_count")
    expected_prompt_count = partition_source.get("expected_prompt_count")
    if type(selection_count) is not int or type(expected_prompt_count) is not int:
        raise TypeError("source partition counts must be exact integers")
    partition = partition_development_export_for_interactions(
        eval_export,
        selection_count=selection_count,
        expected_prompt_count=expected_prompt_count,
    )
    (
        source_model,
        source_protocol,
        source_splits,
        source_conditions,
        trajectory_breakpoint,
    ) = _validate_frozen_source_bindings(
        source,
        trajectory,
        source_file_sha256=source_file_sha256,
        revision=revision,
        model_id=model_id,
        fit_export_metadata=fit_export.metadata(),
        eval_export_metadata=eval_export.metadata(),
        partition_metadata=partition.metadata(),
    )

    raw_superfragment_plan = _require_mapping(
        source.get("superfragment_plan"),
        label="source superfragment plan",
    )
    superfragment_plan = ParameterLayerSuperfragmentPlan.from_state_dict(
        raw_superfragment_plan
    )
    fragment_plan = superfragment_plan.source_fragment_plan
    upstream = _require_mapping(
        source.get("upstream_metadata"),
        label="source upstream metadata",
    )
    parameter_catalog_sha256 = _require_sha256(
        upstream.get("parameter_catalog_sha256"),
        label="parameter catalog sha256",
    )
    fisher_coupling_sha256 = _require_sha256(
        upstream.get("fisher_coupling_sha256"),
        label="Fisher coupling sha256",
    )

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load pinned local Gemma checkpoint")
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
    if model_fingerprint != source_model.get("adapter_model_fingerprint"):
        raise ValueError("live model fingerprint differs from frozen source")
    live_model_metadata = _model_provenance(
        model,
        model_id=model_id,
        requested_revision=revision,
    )
    if live_model_metadata.get("resolved_commit") != revision:
        raise ValueError("loaded Gemma model does not bind the pinned revision")
    _, down_weights = _validate_live_layers(
        adapter,
        fragment_plan=fragment_plan,
        superfragment_plan=superfragment_plan,
    )

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
    selection_safe = _safe_tokenized_stream_metadata(selection_stream)
    frozen_fit = _require_mapping(
        source_splits.get("fit"),
        label="source fit split",
    )
    frozen_selection = _require_mapping(
        source_splits.get("selection"),
        label="source selection split",
    )
    _live_split_matches_frozen(fit_safe, frozen_fit, label="fit")
    _live_split_matches_frozen(
        selection_safe,
        frozen_selection,
        label="selection",
    )
    fit_split_sha256 = _require_sha256(
        fit_safe.get("serialized_sha256"),
        label="fit split sha256",
    )
    selection_split_sha256 = _require_sha256(
        selection_safe.get("serialized_sha256"),
        label="selection split sha256",
    )

    raw_generator_states = source.pop("generator_fits", None)
    if (
        isinstance(raw_generator_states, (str, bytes))
        or not isinstance(raw_generator_states, Sequence)
    ):
        raise TypeError("source generator fits must be a sequence")
    generator_states: list[object | None] = list(raw_generator_states)
    del raw_generator_states
    _progress("executor: authenticate 18 source fits memory-safely")
    source_replacements, source_fit_records = (
        _restore_source_generator_catalog(generator_states)
    )
    del generator_states
    gc.collect()

    mode_ranks = source_protocol["mode_ranks"]
    generator_ranks = source_protocol["generator_ranks"]
    selected_mode_rank = source_protocol["selected_mode_rank"]
    selected_generator_rank = source_protocol["selected_generator_rank"]
    generator_ridge = source_protocol["generator_ridge"]
    assert isinstance(mode_ranks, Sequence)
    assert isinstance(generator_ranks, Sequence)
    assert type(selected_mode_rank) is int
    assert type(selected_generator_rank) is int
    assert isinstance(generator_ridge, (int, float))

    _progress("refit: sequential compiled-prefix layers 10 through 17")
    final_replacements, refit_fits, layer_refits = (
        _sequentially_refit_layers(
            adapter,
            fit_batches,
            selection_batches,
            fragment_plan=fragment_plan,
            superfragment_plan=superfragment_plan,
            down_projection_weights=down_weights,
            source_replacements=source_replacements,
            source_fit_records=source_fit_records,
            source_model_sha256=model_fingerprint,
            parameter_catalog_sha256=parameter_catalog_sha256,
            fisher_coupling_sha256=fisher_coupling_sha256,
            fit_split_sha256=fit_split_sha256,
            selection_split_sha256=selection_split_sha256,
            mode_ranks=mode_ranks,
            selected_mode_rank=selected_mode_rank,
            generator_ranks=generator_ranks,
            selected_generator_rank=selected_generator_rank,
            generator_ridge=float(generator_ridge),
            refit_start_layer=refit_start_layer,
        )
    )
    executor = Gemma3FullMLPStackExecutor(adapter, final_replacements)
    if adapter.model_fingerprint() != model_fingerprint:
        raise RuntimeError("sequential refit mutated the source model")

    del fit_batches, selection_batches, down_weights, source_replacements
    gc.collect()
    _progress("assessment: materialize frozen assessment20 after all refits")
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
    frozen_assessment = _require_mapping(
        source_splits.get("assessment"),
        label="source assessment split",
    )
    _live_split_matches_frozen(
        assessment_safe,
        frozen_assessment,
        label="assessment",
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
        or tuple(frozen_fit.get("content_sha256", ())) != fit_content
        or tuple(frozen_selection.get("content_sha256", ()))
        != selection_content
        or tuple(frozen_assessment.get("content_sha256", ()))
        != assessment_content
    ):
        raise ValueError("live split content membership differs from source")

    _progress("evaluate: native, refitted full stack, and matched deletion")
    evaluation = evaluate_full_mlp_stack_conditions(
        adapter,
        executor,
        assessment_batches,
        expected_example_ids=partition.assessment.prompt_sha256s,
        expected_mode_counts_by_layer=tuple(
            value.mode_count for value in superfragment_plan.superfragments
        ),
        vocabulary_chunk_size=vocabulary_chunk_size,
        assessment_role="open_development_assessment",
    )
    evaluation["assessment_split_sha256"] = assessment_safe[
        "serialized_sha256"
    ]
    conditions = _require_mapping(
        evaluation.get("conditions"),
        label="refit conditions",
    )
    _metric_mapping_close(
        _require_mapping(conditions.get("native"), label="refit native"),
        _require_mapping(source_conditions.get("native"), label="source native"),
        label="native endpoint",
    )
    _metric_mapping_close(
        _require_mapping(
            conditions.get("matched_deletion"),
            label="refit deletion",
        ),
        _require_mapping(
            source_conditions.get("matched_deletion"),
            label="source deletion",
        ),
        label="matched-deletion endpoint",
    )

    baseline_sha256 = frozen_baseline_conditions_sha256(source_conditions)
    artifact_evaluation = {
        "execution_path": "sequential_refit_full_mlp_stack_rung",
        "assessment_role": "open_development_assessment",
        "heldout_confirmation": False,
        "assessment_membership_exact": True,
        "refit_frozen_before_assessment": True,
        "fit_and_selection_used_for_refit": True,
        "assessment_used_for_refit": False,
        "generator_rank_selection_performed": False,
        "latency_or_kernel_speed_claim": False,
        "supervised_tokens": evaluation["supervised_tokens"],
        "logical_valid_tokens": evaluation["logical_valid_tokens"],
        "assessment_split_sha256": evaluation[
            "assessment_split_sha256"
        ],
        "frozen_baseline_conditions_sha256": baseline_sha256,
        "conditions": {
            "native": dict(
                _require_mapping(
                    source_conditions.get("native"),
                    label="source native",
                )
            ),
            "frozen_generated_full_stack": dict(
                _require_mapping(
                    source_conditions.get("generated_full_stack"),
                    label="source generated full stack",
                )
            ),
            "sequential_refit_full_stack": dict(
                _require_mapping(
                    conditions.get("generated_full_stack"),
                    label="refit generated full stack",
                )
            ),
            "matched_deletion": dict(
                _require_mapping(
                    source_conditions.get("matched_deletion"),
                    label="source matched deletion",
                )
            ),
        },
        "control_validation": {
            "native_matches_frozen_full_stack_artifact": True,
            "frozen_generated_matches_frozen_full_stack_artifact": True,
            "matched_deletion_matches_frozen_full_stack_artifact": True,
            "frozen_generated_matches_trajectory_full_stack_endpoint": True,
            "physical_scope_identical": True,
            "refit_generator_compute_executed": True,
            "matched_deletion_compute_zero": True,
        },
    }
    save_kwargs: dict[str, object] = {
        "model": {
            **dict(source_model),
            "local_files_only": True,
        },
        "frozen_sources": {
            "full_stack": {
                "schema": source["schema"],
                "format_version": source["format_version"],
                "artifact_file_sha256": source_file_sha256,
                "scientific_payload_sha256": source[
                    "scientific_payload_sha256"
                ],
                "baseline_conditions_sha256": baseline_sha256,
            },
            "trajectory": {
                "schema": trajectory["schema"],
                "format_version": trajectory["format_version"],
                "artifact_file_sha256": trajectory_file_sha256,
                "scientific_payload_sha256": trajectory[
                    "scientific_payload_sha256"
                ],
                "breakpoint_direction": "prefix",
                "breakpoint_depth": REFIT_START_LAYER,
                "breakpoint_row_sha256": (
                    trajectory_breakpoint_row_sha256(
                        trajectory_breakpoint
                    )
                ),
            },
        },
        "splits": {
            "fit": _artifact_split_entry(frozen_fit),
            "selection": _artifact_split_entry(frozen_selection),
            "assessment": _artifact_split_entry(frozen_assessment),
            "provenance": {
                "assurance": "caller_declared_self_attested",
                "externally_authenticated": False,
                "fit_used_for_generator_refit": True,
                "selection_used_for_generator_refit": True,
                "assessment_used_for_generator_refit": False,
                "assessment_used_for_generator_rank_selection": False,
                "fit_selection_assessment_disjoint": True,
                "assessment_evaluated_only_after_refit_freeze": True,
                "heldout_confirmation": False,
            },
        },
        "protocol": {
            "scope": "sequential_compiled_trajectory_full_mlp_stack_refit",
            "transformer_layer_count": EXPECTED_LAYER_COUNT,
            "refit_start_layer": REFIT_START_LAYER,
            "unchanged_layer_ordinals": tuple(range(REFIT_START_LAYER)),
            "refit_layer_order": REFIT_LAYER_ORDINALS,
            "refit_rule": (
                "sequential_teacher_on_actual_compiled_prefix"
            ),
            "fisher_weighting": "current_compiled_prefix_rows",
            "jacobian_policy": (
                "no_explicit_jacobian_correction_in_direct_refit_rung"
            ),
            "rank_policy": "fixed_from_frozen_full_stack",
            "resource_budget_policy": (
                "exact_per_layer_equality_to_frozen_full_stack"
            ),
            "execution_path": (
                "frozen_prefix_then_sequential_refit_dense_generators"
            ),
            "source_model_weights_mutated": False,
            "assessment_role": "open_development_assessment",
            "assessment_after_refit_freeze": True,
            "generator_rank_selection_performed": False,
            "heldout_confirmation": False,
            "compression_claim": False,
            "latency_or_kernel_speed_claim": False,
            "local_files_only": True,
        },
        "source_layer_summaries": source_fit_records,
        "refit_generator_fits": refit_fits,
        "layer_refits": layer_refits,
        "evaluation": artifact_evaluation,
    }
    _progress("artifact: save and strict-load sequential refit result")
    report = _publish_verified_refit(
        output=output,
        adapter=adapter,
        expected_model_fingerprint=model_fingerprint,
        source_path=source_path,
        expected_source_file_sha256=source_file_sha256,
        trajectory_path=trajectory_path,
        expected_trajectory_file_sha256=trajectory_file_sha256,
        save=save_gemma3_full_mlp_stack_refit_artifact,
        load=load_gemma3_full_mlp_stack_refit_artifact,
        save_kwargs=save_kwargs,
    )
    _progress(f"wrote {Path(output)} and {Path(output).with_suffix('.json')}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially refit Gemma MLP generators 10..17 under the "
            "already-compiled prefix distribution."
        )
    )
    parser.add_argument("--fit-export", type=Path, default=DEFAULT_FIT_EXPORT)
    parser.add_argument("--eval-export", type=Path, default=DEFAULT_EVAL_EXPORT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--trajectory-artifact",
        type=Path,
        default=DEFAULT_TRAJECTORY_ARTIFACT,
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
        "--vocabulary-chunk-size",
        type=int,
        default=DEFAULT_VOCABULARY_CHUNK_SIZE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_gemma3_full_mlp_stack_refit_experiment(
        fit_export_path=arguments.fit_export,
        eval_export_path=arguments.eval_export,
        revision=arguments.revision,
        base_artifact_path=arguments.base_artifact,
        trajectory_artifact_path=arguments.trajectory_artifact,
        output=arguments.output,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
        vocabulary_chunk_size=arguments.vocabulary_chunk_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
