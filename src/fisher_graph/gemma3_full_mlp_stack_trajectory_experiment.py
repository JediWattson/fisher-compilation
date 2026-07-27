"""Replay the frozen Gemma full-MLP artifact as prefix/suffix trajectories.

This runner performs no fitting and no rank selection.  It strict-loads the
already frozen exhaustive MLP-stack artifact, reconstructs its 18 executable
generator plans, and evaluates cumulative generated prefixes and suffixes on
the exact open-development assessment membership recorded by that artifact.

The output is a tensor-free JSON diagnostic.  It is not a held-out result, a
compression claim, or a latency measurement.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from .adapters import Gemma3CausalLMAdapter
from .full_mlp_stack_generators import FullMLPStackGeneratorFit
from .full_mlp_stack_trajectory import (
    FrozenFullMLPStackTrajectoryExecutor,
    evaluate_full_mlp_stack_trajectory,
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
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_trajectory_artifact import (
    load_gemma3_full_mlp_stack_trajectory_artifact,
    save_gemma3_full_mlp_stack_trajectory_artifact,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _safe_tokenized_stream_metadata,
    load_development_prompt_export,
)
from .gemma3_modal_generator_executor import (
    Gemma3ModalGeneratorReplacement,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    _bind_batch_example_ids,
)
from .modal_graph_rung_evaluation import (
    partition_development_export_for_interactions,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_full_mlp_stack_trajectory_experiment",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-full-mlp-stack-trajectory-dev-v1.json"
)
DEFAULT_VOCABULARY_CHUNK_SIZE = 16384
_EXPECTED_LAYER_COUNT = 18
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _progress(message: str) -> None:
    print(f"[full-mlp-trajectory] {message}", file=sys.stderr, flush=True)


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
    model_id: str,
    device_name: str,
    dtype: str,
    max_length: int,
    tokenization_batch_size: int,
    vocabulary_chunk_size: int,
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
    source = Path(base_artifact_path)
    if source.suffix != ".pt":
        raise ValueError("frozen full-stack source artifact must use .pt")
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("trajectory output must use .json")
    if destination.exists():
        raise FileExistsError("refusing to overwrite trajectory artifact")


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _validate_source_bindings(
    source: Mapping[str, object],
    *,
    revision: str,
    model_id: str,
    eval_export_metadata: Mapping[str, object],
    partition_metadata: Mapping[str, object],
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    model = _require_mapping(source.get("model"), label="source model")
    protocol = _require_mapping(
        source.get("protocol"),
        label="source protocol",
    )
    splits = _require_mapping(source.get("splits"), label="source splits")
    resources = _require_mapping(
        source.get("resource_accounting"),
        label="source resources",
    )
    if (
        model.get("model_id") != model_id
        or model.get("requested_revision") != revision
        or model.get("resolved_commit") != revision
        or model.get("local_files_only") is not True
    ):
        raise ValueError("frozen source model binding differs from the request")
    if (
        protocol.get("scope") != "full_native_mlp_stack_replacement"
        or protocol.get("transformer_layer_count") != _EXPECTED_LAYER_COUNT
        or protocol.get("local_files_only") is not True
    ):
        raise ValueError("frozen source is not the exhaustive 18-layer rung")
    if splits.get("eval_export") != dict(eval_export_metadata):
        raise ValueError("evaluation export differs from the frozen source")
    if splits.get("partition") != dict(partition_metadata):
        raise ValueError("development partition differs from the frozen source")
    assessment = _require_mapping(
        splits.get("assessment"),
        label="source assessment",
    )
    if assessment.get("role") != "open_development_assessment":
        raise ValueError("source assessment role is not open development")
    return model, protocol, splits, resources


def _restore_replacements(
    generator_states: object,
) -> tuple[Gemma3ModalGeneratorReplacement, ...]:
    if (
        isinstance(generator_states, (str, bytes))
        or not isinstance(generator_states, Sequence)
        or len(generator_states) != _EXPECTED_LAYER_COUNT
    ):
        raise ValueError("source must contain exactly 18 generator fits")
    # Release each large analysis state after authenticating it.  The caller
    # passes an owning list, and the replacement retains only the dense
    # executable plan needed at runtime.
    pending: list[object | None]
    if isinstance(generator_states, list):
        pending = generator_states
    else:
        pending = list(generator_states)
    replacements: list[Gemma3ModalGeneratorReplacement] = []
    for expected_ordinal in range(_EXPECTED_LAYER_COUNT):
        state = pending[expected_ordinal]
        if not isinstance(state, Mapping):
            raise TypeError("generator fit state must be a mapping")
        fit = FullMLPStackGeneratorFit.from_state_dict(state)
        ordinal = fit.superfragment.layer_ordinal
        if ordinal != expected_ordinal:
            raise ValueError("generator fits are not in exact layer order")
        replacements.append(
            Gemma3ModalGeneratorReplacement(
                layer_ordinal=ordinal,
                removed_mode_indices=fit.superfragment.channel_indices,
                generator_plan=fit.executable_plan,
            )
        )
        pending[expected_ordinal] = None
        del fit
        gc.collect()
    return tuple(replacements)


def _metrics_close(
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
    for field in fields:
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


def _artifact_resources(
    raw: Mapping[str, object],
    *,
    layer_count: int,
    logical_valid_tokens: int,
) -> dict[str, object]:
    if logical_valid_tokens <= 0:
        raise ValueError("logical_valid_tokens must be positive")
    if raw.get("valid_tokens") != logical_valid_tokens:
        raise ValueError(
            "trajectory resource valid tokens differ from evaluation"
        )
    ordinals_raw = raw.get("generated_layer_ordinals")
    if (
        isinstance(ordinals_raw, (str, bytes))
        or not isinstance(ordinals_raw, Sequence)
    ):
        raise TypeError("generated_layer_ordinals must be a sequence")
    ordinals = tuple(ordinals_raw)
    generated_macs = raw.get("logical_generator_macs")
    generated_bias = raw.get("logical_generator_bias_additions")
    native_removed_macs = raw.get("logical_native_mlp_macs_removed")
    if (
        type(generated_macs) is not int
        or generated_macs % logical_valid_tokens
        or type(generated_bias) is not int
        or generated_bias % logical_valid_tokens
        or type(native_removed_macs) is not int
        or native_removed_macs % logical_valid_tokens
    ):
        raise ValueError("trajectory compute totals are not per-token exact")
    native_replaced = raw.get("logical_native_mlp_parameters_removed")
    generated_parameters = raw.get(
        "logical_generator_subset_learned_parameters"
    )
    if type(native_replaced) is not int or type(generated_parameters) is not int:
        raise TypeError("trajectory parameter accounting must be exact")
    generator_macs_per_token = generated_macs // logical_valid_tokens
    generator_bias_per_token = generated_bias // logical_valid_tokens
    native_macs_per_token = native_removed_macs // logical_valid_tokens
    net_macs = raw.get("net_logical_macs_saved")
    if (
        type(net_macs) is not int
        or net_macs != native_removed_macs - generated_macs
    ):
        raise ValueError("trajectory net compute accounting is inconsistent")
    return {
        "replacement_scope": (
            "full_native_mlp_stack_replacement"
            if len(ordinals) == layer_count
            else "partial_native_mlp_stack_replacement"
        ),
        "replaced_layer_count": len(ordinals),
        "replaced_layer_ordinals": ordinals,
        "removed_mode_count": raw["removed_mode_count"],
        "source_whole_model_learned_parameters": (
            raw["source_whole_model_learned_parameters"]
        ),
        "native_replaced_mlp_learned_parameters": native_replaced,
        "generator_replacement_learned_parameters": generated_parameters,
        "logical_candidate_learned_parameters": (
            raw["logical_candidate_learned_parameters"]
        ),
        "net_stored_parameter_savings": (
            raw["logical_net_stored_parameter_savings"]
        ),
        "native_replaced_mlp_linear_macs_per_token": native_macs_per_token,
        "generator_replacement_macs_per_token": generator_macs_per_token,
        "generator_replacement_bias_additions_per_token": (
            generator_bias_per_token
        ),
        "net_linear_macs_saved_per_token": (
            native_macs_per_token - generator_macs_per_token
        ),
        "logical_candidate_excludes_replaced_native_mlps": True,
        "whole_transformer_replaced": False,
    }


def _artifact_evaluation(
    raw: Mapping[str, object],
    *,
    assessment_split_sha256: str,
) -> dict[str, object]:
    conditions = _require_mapping(
        raw.get("conditions"),
        label="trajectory conditions",
    )
    paths = _require_mapping(
        raw.get("trajectory_condition_ids"),
        label="trajectory paths",
    )
    resources = _require_mapping(
        raw.get("resource_accounting"),
        label="trajectory resources",
    )
    layer_count = _require_mapping(
        raw.get("declared_scope"),
        label="trajectory declared scope",
    ).get("layer_count")
    logical_valid_tokens = raw.get("logical_valid_tokens")
    supervised_tokens = raw.get("supervised_tokens")
    if (
        type(layer_count) is not int
        or layer_count != _EXPECTED_LAYER_COUNT
        or type(logical_valid_tokens) is not int
        or logical_valid_tokens <= 0
        or type(supervised_tokens) is not int
        or supervised_tokens <= 0
    ):
        raise ValueError("trajectory aggregate counts are invalid")

    ladders: dict[str, list[dict[str, object]]] = {}
    for direction in ("prefix", "suffix"):
        condition_ids = paths.get(direction)
        if (
            isinstance(condition_ids, (str, bytes))
            or not isinstance(condition_ids, Sequence)
            or len(condition_ids) != layer_count
        ):
            raise ValueError(f"{direction} path does not cover every depth")
        rows: list[dict[str, object]] = []
        for depth, condition_id in enumerate(condition_ids, start=1):
            if not isinstance(condition_id, str):
                raise TypeError("trajectory condition IDs must be strings")
            metrics = _require_mapping(
                conditions.get(condition_id),
                label=f"{condition_id} metrics",
            )
            accounting = _require_mapping(
                resources.get(condition_id),
                label=f"{condition_id} resources",
            )
            rows.append(
                {
                    "depth": depth,
                    "metrics": dict(metrics),
                    "resources": _artifact_resources(
                        accounting,
                        layer_count=layer_count,
                        logical_valid_tokens=logical_valid_tokens,
                    ),
                }
            )
        ladders[direction] = rows
    if ladders["prefix"][-1] != ladders["suffix"][-1]:
        raise ValueError("trajectory paths do not share one full-stack endpoint")
    # Reuse the same immutable logical endpoint before JSON canonicalization.
    ladders["suffix"][-1] = ladders["prefix"][-1]
    native = _require_mapping(
        conditions.get("native"),
        label="native metrics",
    )
    return {
        "execution_path": "frozen_prefix_suffix_full_mlp_stack_ladder",
        "assessment_role": "open_development_assessment",
        "heldout_confirmation": False,
        "assessment_membership_exact": True,
        "frozen_before_assessment": True,
        "generator_refit_performed": False,
        "generator_rank_selection_performed": False,
        "latency_or_kernel_speed_claim": False,
        "supervised_tokens": supervised_tokens,
        "logical_valid_tokens": logical_valid_tokens,
        "assessment_split_sha256": assessment_split_sha256,
        "native": dict(native),
        "prefix_ladder": ladders["prefix"],
        "suffix_ladder": ladders["suffix"],
    }


def _verify_immutable_sources(
    *,
    adapter: Gemma3CausalLMAdapter,
    expected_model_fingerprint: str,
    source_path: Path,
    expected_source_file_sha256: str,
) -> None:
    if adapter.model_fingerprint() != expected_model_fingerprint:
        raise RuntimeError("trajectory experiment mutated the source model")
    if _file_sha256(source_path) != expected_source_file_sha256:
        raise RuntimeError("frozen source artifact changed during trajectory")


def _publish_verified_trajectory(
    *,
    output: Path | str,
    adapter: Gemma3CausalLMAdapter,
    expected_model_fingerprint: str,
    source_path: Path,
    expected_source_file_sha256: str,
    model: Mapping[str, object],
    frozen_source_artifact: Mapping[str, object],
    splits: Mapping[str, object],
    protocol: Mapping[str, object],
    evaluation: Mapping[str, object],
) -> dict[str, object]:
    """Publish only while both frozen inputs remain exactly authenticated."""

    _verify_immutable_sources(
        adapter=adapter,
        expected_model_fingerprint=expected_model_fingerprint,
        source_path=source_path,
        expected_source_file_sha256=expected_source_file_sha256,
    )
    published = False
    try:
        payload = save_gemma3_full_mlp_stack_trajectory_artifact(
            output,
            model=model,
            frozen_source_artifact=frozen_source_artifact,
            splits=splits,
            protocol=protocol,
            evaluation=evaluation,
        )
        published = True
        load_gemma3_full_mlp_stack_trajectory_artifact(output)
        _verify_immutable_sources(
            adapter=adapter,
            expected_model_fingerprint=expected_model_fingerprint,
            source_path=source_path,
            expected_source_file_sha256=expected_source_file_sha256,
        )
    except BaseException:
        # Preflight established that the destination did not exist.  Delete
        # only after our exclusive saver returned successfully, so a failed
        # concurrent publish can never remove another writer's file.
        if published:
            Path(output).unlink(missing_ok=True)
        raise
    return payload


def run_gemma3_full_mlp_stack_trajectory_experiment(
    *,
    eval_export_path: Path | str = DEFAULT_EVAL_EXPORT,
    revision: str,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
    vocabulary_chunk_size: int = DEFAULT_VOCABULARY_CHUNK_SIZE,
) -> dict[str, object]:
    """Evaluate and save the no-refit frozen prefix/suffix diagnostic."""

    _validate_runner_preflight(
        revision=revision,
        output=output,
        base_artifact_path=base_artifact_path,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        vocabulary_chunk_size=vocabulary_chunk_size,
    )
    source_path = Path(base_artifact_path)
    _progress("source: hash and strict-load frozen exhaustive artifact")
    source_file_sha256 = _file_sha256(source_path)
    source = load_gemma3_full_mlp_stack_artifact(source_path)
    eval_export = load_development_prompt_export(eval_export_path)
    source_splits = _require_mapping(
        source.get("splits"),
        label="source splits",
    )
    source_partition = _require_mapping(
        source_splits.get("partition"),
        label="source partition",
    )
    selection_count = source_partition.get("selection_prompt_count")
    expected_prompt_count = source_partition.get("expected_prompt_count")
    if type(selection_count) is not int or type(expected_prompt_count) is not int:
        raise TypeError("source partition counts must be exact integers")
    partition = partition_development_export_for_interactions(
        eval_export,
        selection_count=selection_count,
        expected_prompt_count=expected_prompt_count,
    )
    source_model, source_protocol, source_splits, source_resources = (
        _validate_source_bindings(
            source,
            revision=revision,
            model_id=model_id,
            eval_export_metadata=eval_export.metadata(),
            partition_metadata=partition.metadata(),
        )
    )
    source_assessment = _require_mapping(
        source_splits.get("assessment"),
        label="source assessment",
    )
    source_evaluation = _require_mapping(
        source.get("evaluation"),
        label="source evaluation",
    )
    source_conditions = _require_mapping(
        source_evaluation.get("conditions"),
        label="source conditions",
    )
    raw_generator_states = source.pop("generator_fits", None)
    if (
        isinstance(raw_generator_states, (str, bytes))
        or not isinstance(raw_generator_states, Sequence)
    ):
        raise TypeError("source generator fits must be a sequence")
    generator_states: list[object | None] = list(raw_generator_states)
    del raw_generator_states

    device = resolve_torch_device(device_name)
    cache = resolve_gemma3_huggingface_paths(cache_dir)["hub_cache"]
    _progress("model: load the pinned local Gemma checkpoint")
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

    _progress("executor: authenticate 18 frozen fits and compile once")
    replacements = _restore_replacements(generator_states)
    del generator_states
    executor = FrozenFullMLPStackTrajectoryExecutor(adapter, replacements)
    del replacements
    gc.collect()
    if (
        executor.replaced_layer_count != source_protocol.get(
            "transformer_layer_count"
        )
        or executor.removed_mode_count
        != source_protocol.get("removed_mode_count")
    ):
        raise ValueError("runtime executor scope differs from frozen source")

    _progress("assessment: materialize only the recorded assessment20")
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
    expected_assessment_safe = {
        key: value
        for key, value in source_assessment.items()
        if key != "role"
    }
    if assessment_safe != expected_assessment_safe:
        raise ValueError(
            "live assessment tokenization differs from frozen source"
        )
    assessment_split_sha256 = _require_sha256(
        assessment_safe.get("serialized_sha256"),
        label="assessment split sha256",
    )

    _progress(
        "evaluate: native plus 35 unique frozen prefix/suffix conditions"
    )
    raw_evaluation = evaluate_full_mlp_stack_trajectory(
        adapter,
        executor,
        assessment_batches,
        expected_example_ids=partition.assessment.prompt_sha256s,
        expected_mode_counts_by_layer=executor.mode_counts_by_layer,
        vocabulary_chunk_size=vocabulary_chunk_size,
    )
    raw_conditions = _require_mapping(
        raw_evaluation.get("conditions"),
        label="trajectory conditions",
    )
    _metrics_close(
        _require_mapping(raw_conditions.get("native"), label="native"),
        _require_mapping(
            source_conditions.get("native"),
            label="source native",
        ),
        label="native endpoint",
    )
    _metrics_close(
        _require_mapping(
            raw_conditions.get("full_stack"),
            label="full stack",
        ),
        _require_mapping(
            source_conditions.get("generated_full_stack"),
            label="source generated full stack",
        ),
        label="generated full-stack endpoint",
    )
    evaluation = _artifact_evaluation(
        raw_evaluation,
        assessment_split_sha256=assessment_split_sha256,
    )

    native_mlp_parameters = source_resources.get(
        "native_mlp_stack_learned_parameters"
    )
    native_mlp_macs = source_resources.get(
        "native_mlp_stack_linear_macs_per_token"
    )
    if type(native_mlp_parameters) is not int or type(native_mlp_macs) is not int:
        raise TypeError("source MLP resources must use exact integers")
    assessment_content = source_assessment.get("content_sha256")
    assessment_ids = source_assessment.get("source_prompt_sha256")
    if (
        isinstance(assessment_content, (str, bytes))
        or not isinstance(assessment_content, Sequence)
        or isinstance(assessment_ids, (str, bytes))
        or not isinstance(assessment_ids, Sequence)
    ):
        raise TypeError("source assessment membership must be a sequence")
    assessment_valid = _require_mapping(
        source_assessment.get("valid_tokens"),
        label="source assessment valid tokens",
    ).get("total")
    assessment_supervised = _require_mapping(
        source_assessment.get("supervised_positions"),
        label="source assessment supervised positions",
    ).get("total")
    if (
        type(assessment_valid) is not int
        or assessment_valid <= 0
        or type(assessment_supervised) is not int
        or assessment_supervised <= 0
    ):
        raise ValueError("source assessment token totals are invalid")
    source_payload_sha256 = _require_sha256(
        source.get("scientific_payload_sha256"),
        label="source scientific payload sha256",
    )
    _progress("artifact: save strict tensor-free trajectory result")
    payload = _publish_verified_trajectory(
        output=output,
        adapter=adapter,
        expected_model_fingerprint=model_fingerprint,
        source_path=source_path,
        expected_source_file_sha256=source_file_sha256,
        model={
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_commit": revision,
            "adapter_model_fingerprint": model_fingerprint,
            "source_whole_model_learned_parameters": (
                source_model["source_whole_model_learned_parameters"]
            ),
            "native_mlp_stack_learned_parameters": native_mlp_parameters,
            "native_mlp_stack_linear_macs_per_token": native_mlp_macs,
            "local_files_only": True,
        },
        frozen_source_artifact={
            "source_schema": source["schema"],
            "source_format_version": source["format_version"],
            "artifact_file_sha256": source_file_sha256,
            "scientific_payload_sha256": source_payload_sha256,
            "source_scope": source_protocol["scope"],
            "frozen_before_trajectory": True,
        },
        splits={
            "assessment": {
                "role": "open_development_assessment",
                "serialized_sha256": assessment_split_sha256,
                "content_sha256": tuple(assessment_content),
                "example_count": len(tuple(assessment_ids)),
                "logical_valid_tokens": assessment_valid,
                "supervised_tokens": assessment_supervised,
            },
            "provenance": {
                "assurance": "caller_declared_self_attested",
                "externally_authenticated": False,
                "heldout_confirmation": False,
                "assessment_used_for_generator_refit": False,
                "assessment_used_for_generator_rank_selection": False,
            },
        },
        protocol={
            "scope": "frozen_full_native_mlp_stack_trajectory_ladder",
            "transformer_layer_count": _EXPECTED_LAYER_COUNT,
            "removed_mode_count": source_protocol["removed_mode_count"],
            "prefix_depths": tuple(range(1, _EXPECTED_LAYER_COUNT + 1)),
            "suffix_depths": tuple(range(1, _EXPECTED_LAYER_COUNT + 1)),
            "prefix_rule": "generated_layers_0_through_depth_minus_1",
            "suffix_rule": "generated_layers_18_minus_depth_through_17",
            "depth_18_endpoint_rule": (
                "canonical_exact_prefix_suffix_equality"
            ),
            "execution_path": "frozen_mixed_native_generated_mlp_stack",
            "generators_frozen": True,
            "generator_refit_performed": False,
            "generator_rank_selection_performed": False,
            "source_model_weights_mutated": False,
            "assessment_role": "open_development_assessment",
            "heldout_confirmation": False,
            "latency_or_kernel_speed_claim": False,
            "local_files_only": True,
        },
        evaluation=evaluation,
    )
    _progress(f"wrote {Path(output)}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Gemma full-MLP prefix/suffix trajectories "
            "without fitting or rank selection."
        )
    )
    parser.add_argument("--eval-export", type=Path, default=DEFAULT_EVAL_EXPORT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
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
    payload = run_gemma3_full_mlp_stack_trajectory_experiment(
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
        vocabulary_chunk_size=arguments.vocabulary_chunk_size,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
