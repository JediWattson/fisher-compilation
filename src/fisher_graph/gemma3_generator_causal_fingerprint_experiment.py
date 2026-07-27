"""Trace prompt-conditioned causal identities of the frozen Gemma generators.

This development runner restores the exact base-plus-refit generator catalog,
executes one generated baseline and one exact singleton suppression per layer,
and records only source-safe prompt scalars plus bounded shared-output
similarities.  It does not fit, mutate, merge, route, prune, or lower anything.
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
from typing import Any

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .generator_causal_fingerprints import (
    GeneratorCausalFingerprintAccumulator,
    GeneratorCausalFingerprintAnalysis,
    GeneratorCausalFingerprintProvenance,
    generator_fingerprint_example_id_sha256,
)
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
    resolve_torch_device,
)
from .gemma3_full_mlp_stack_artifact import (
    GEMMA3_FULL_MLP_STACK_FORMAT_VERSION,
    GEMMA3_FULL_MLP_STACK_SCHEMA,
)
from .gemma3_full_mlp_stack_dev_experiment import (
    DEFAULT_OUTPUT as DEFAULT_FULL_MLP_STACK_ARTIFACT,
)
from .gemma3_full_mlp_stack_executor import Gemma3FullMLPStackExecutor
from .gemma3_full_mlp_stack_refit_artifact import (
    GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION,
    GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA,
)
from .gemma3_full_mlp_stack_refit_experiment import (
    DEFAULT_OUTPUT as DEFAULT_REFIT_ARTIFACT,
)
from .gemma3_full_mlp_stack_refit_runtime import (
    Gemma3RefitRuntimeCatalog,
    restore_gemma3_full_mlp_stack_refit_runtime,
)
from .gemma3_gated_executor_experiment import _materialize_split
from .gemma3_generator_causal_fingerprint_artifact import (
    gemma3_generator_prompt_fingerprint_sha256,
    save_gemma3_generator_causal_fingerprint_artifact,
)
from .gemma3_generator_causal_intervention import (
    FrozenGemma3GeneratorCausalInterventionExecutor,
    Gemma3GeneratorCausalIntervention,
)
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _objective_sha256,
    _safe_tokenized_stream_metadata,
    load_development_prompt_export,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    _bind_batch_example_ids,
    _stream_content_sha256s,
)
from .modal_graph_rung_evaluation import (
    partition_development_export_for_interactions,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_generator_causal_fingerprint_experiment",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-causal-fingerprint-dev-v1.json"
)
DEFAULT_ANCHOR_COUNT = 8
DEFAULT_TOP_IMPORTANCE_COUNT = 5
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_CATALOG_DOMAIN = (
    b"fisher_graph.gemma3.generator_causal_fingerprint.catalog.v1\0"
)


def _progress(message: str) -> None:
    print(
        f"[gemma-generator-causal-fingerprint] {message}",
        file=sys.stderr,
        flush=True,
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _model_logits(output: object) -> Tensor:
    logits = (
        output.get("logits")
        if isinstance(output, Mapping)
        else getattr(output, "logits", None)
    )
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or not logits.dtype.is_floating_point
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError(
            "intervention output must expose finite "
            "[batch, sequence, vocabulary] logits"
        )
    return logits


def _live_split_matches_normalized_refit(
    live: Mapping[str, object],
    frozen: Mapping[str, object],
) -> None:
    valid_tokens = _require_mapping(
        live.get("valid_tokens"),
        label="live split valid tokens",
    ).get("total")
    supervised_tokens = _require_mapping(
        live.get("supervised_positions"),
        label="live split supervised positions",
    ).get("total")
    live_content = live.get("content_sha256")
    frozen_content = frozen.get("content_sha256")
    if (
        live.get("serialized_sha256") != frozen.get("serialized_sha256")
        or live.get("sequences") != frozen.get("example_count")
        or valid_tokens != frozen.get("logical_valid_tokens")
        or supervised_tokens != frozen.get("supervised_tokens")
        or tuple(live_content or ()) != tuple(frozen_content or ())
    ):
        raise ValueError(
            "live tokenization differs from normalized frozen refit split"
        )


def _catalog_sha256(runtime: Gemma3RefitRuntimeCatalog) -> str:
    rows = tuple(
        {
            "layer_ordinal": ordinal,
            "source_fit_sha256": runtime.source_fit_sha256s[ordinal],
            "deployed_fit_sha256": runtime.deployed_fit_sha256s[ordinal],
            "deployed_plan_sha256": (
                runtime.generator_plan_sha256s[ordinal]
            ),
        }
        for ordinal in range(len(runtime.replacements))
    )
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_CATALOG_DOMAIN)
    digest.update(encoded)
    return digest.hexdigest()


def _prompts_in_frozen_order(
    export: object,
    prompt_sha256s: Sequence[str],
) -> tuple[str, ...]:
    prompts = getattr(export, "prompts", None)
    hashes = getattr(export, "prompt_sha256s", None)
    if (
        not isinstance(prompts, tuple)
        or not isinstance(hashes, tuple)
        or len(prompts) != len(hashes)
    ):
        raise TypeError("development export does not expose exact prompt rows")
    by_hash = dict(zip(hashes, prompts, strict=True))
    ordered = tuple(prompt_sha256s)
    if (
        not ordered
        or len(ordered) != len(set(ordered))
        or any(value not in by_hash for value in ordered)
    ):
        raise ValueError(
            "frozen analysis membership is absent from the development export"
        )
    return tuple(by_hash[value] for value in ordered)


def _causal_analysis_lineage(
    analysis: GeneratorCausalFingerprintAnalysis,
) -> dict[str, object]:
    metadata = analysis.metadata()
    provenance = _require_mapping(
        metadata.get("provenance"),
        label="core causal provenance",
    )
    tensor_hashes = _require_mapping(
        metadata.get("tensor_sha256s"),
        label="core causal tensor hashes",
    )
    policy = _require_mapping(
        metadata.get("observational_family_policy"),
        label="core observational family policy",
    )
    return {
        "artifact_kind": metadata["artifact_kind"],
        "format_version": metadata["format_version"],
        "artifact_sha256": metadata["artifact_sha256"],
        "source_model_sha256": provenance["source_model_sha256"],
        "generator_catalog_sha256": provenance[
            "generator_catalog_sha256"
        ],
        "evaluation_split_sha256": provenance[
            "evaluation_split_sha256"
        ],
        "objective_sha256": provenance["objective_sha256"],
        "intervention": provenance["intervention"],
        "generator_count": metadata["generator_count"],
        "prompt_count": metadata["prompt_count"],
        "anchor_count": metadata["anchor_count"],
        "anchor_frame_width": metadata["anchor_frame_width"],
        "shared_frame": metadata["shared_frame"],
        "effect_centering": metadata["effect_centering"],
        "gram_weighting": metadata["gram_weighting"],
        "top_importance_count": metadata["top_importance_count"],
        "observational_family_policy": dict(policy),
        "tensor_sha256s": dict(tensor_hashes),
    }


def _generator_fit_lineage(
    runtime: Gemma3RefitRuntimeCatalog,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "layer_ordinal": ordinal,
            "layer_id": runtime.layer_lineage[ordinal]["layer_id"],
            "deployment_source": (
                "frozen_full_stack"
                if ordinal < runtime.refit_start_layer
                else "sequential_refit_overlay"
            ),
            "source_artifact_scientific_payload_sha256": (
                runtime.base_scientific_payload_sha256
                if ordinal < runtime.refit_start_layer
                else runtime.refit_scientific_payload_sha256
            ),
            "base_fit_sha256": runtime.source_fit_sha256s[ordinal],
            "deployed_fit_sha256": runtime.deployed_fit_sha256s[ordinal],
            "deployed_generator_plan_sha256": (
                runtime.generator_plan_sha256s[ordinal]
            ),
        }
        for ordinal in range(len(runtime.replacements))
    )


def _generator_causal_summaries(
    analysis: GeneratorCausalFingerprintAnalysis,
    runtime: Gemma3RefitRuntimeCatalog,
    *,
    prompt_content_sha256s: Sequence[str],
    example_ids: Sequence[str],
) -> tuple[dict[str, object], ...]:
    prompt_hashes = tuple(prompt_content_sha256s)
    ordered_example_ids = tuple(example_ids)
    if analysis.generator_count != len(runtime.replacements):
        raise ValueError("causal analysis and runtime generator counts differ")
    expected_example_hashes = tuple(
        generator_fingerprint_example_id_sha256(value)
        for value in ordered_example_ids
    )
    if (
        len(prompt_hashes) != len(ordered_example_ids)
        or analysis.example_id_sha256s != expected_example_hashes
    ):
        raise ValueError(
            "causal analysis prompt identities differ from frozen membership"
        )
    split_sha256 = str(runtime.analysis_split["serialized_sha256"])
    rows: list[dict[str, object]] = []
    for ordinal, generator_id in enumerate(analysis.generator_ids):
        signature = analysis.generator_signature(generator_id)
        deltas = tuple(
            float(value)
            for value in signature.muted_minus_baseline_nll.tolist()
        )
        kls = tuple(
            float(value) for value in signature.baseline_to_muted_kl.tolist()
        )
        agreements = tuple(
            float(value) for value in signature.top1_agreement.tolist()
        )
        rms_values = tuple(
            float(value)
            for value in signature.centered_anchor_logit_effect_rms.tolist()
        )
        prompt_signatures = tuple(
            {
                "prompt_content_sha256": prompt_sha256,
                "muted_minus_baseline_nll_per_token": delta,
                "baseline_to_muted_kl_per_token": kl,
                "top1_agreement_to_baseline": agreement,
                "centered_anchor_logit_effect_rms": rms,
            }
            for prompt_sha256, delta, kl, agreement, rms in zip(
                prompt_hashes,
                deltas,
                kls,
                agreements,
                rms_values,
                strict=True,
            )
        )
        count = len(prompt_signatures)
        if count <= 0:
            raise ValueError("causal prompt signature cannot be empty")
        rows.append(
            {
                "layer_ordinal": ordinal,
                "deployed_generator_plan_sha256": (
                    runtime.generator_plan_sha256s[ordinal]
                ),
                "deployed_fit_sha256": (
                    runtime.deployed_fit_sha256s[ordinal]
                ),
                "analysis_split_sha256": split_sha256,
                "prompt_observation_count": count,
                "fingerprint_sha256": (
                    gemma3_generator_prompt_fingerprint_sha256(
                        layer_ordinal=ordinal,
                        analysis_split_sha256=split_sha256,
                        prompt_signatures=prompt_signatures,
                    )
                ),
                "prompt_signatures": prompt_signatures,
                "mean_muted_minus_baseline_nll_per_token": (
                    math.fsum(deltas) / count
                ),
                "rms_muted_minus_baseline_nll_per_token": math.sqrt(
                    math.fsum(value * value for value in deltas) / count
                ),
                "mean_absolute_muted_minus_baseline_nll_per_token": (
                    math.fsum(abs(value) for value in deltas) / count
                ),
                "maximum_absolute_muted_minus_baseline_nll_per_token": (
                    max(abs(value) for value in deltas)
                ),
                "mean_baseline_to_muted_kl_per_token": (
                    math.fsum(kls) / count
                ),
                "mean_top1_agreement_to_baseline": (
                    math.fsum(agreements) / count
                ),
                "mean_centered_anchor_logit_effect_rms": (
                    math.fsum(rms_values) / count
                ),
                "positive_delta_fraction": (
                    sum(value > 0.0 for value in deltas) / count
                ),
            }
        )
    return tuple(rows)


def _pairwise_similarities(
    analysis: GeneratorCausalFingerprintAnalysis,
    runtime: Gemma3RefitRuntimeCatalog,
    summaries: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    index_by_id = {
        generator_id: ordinal
        for ordinal, generator_id in enumerate(analysis.generator_ids)
    }
    split_sha256 = runtime.analysis_split["serialized_sha256"]
    rows: list[dict[str, object]] = []
    for pair in analysis.pair_similarities:
        left = index_by_id[pair.generator_a]
        right = index_by_id[pair.generator_b]
        if not left < right:
            raise ValueError("core generator pair order is not canonical")
        pair_metadata = pair.metadata()
        rows.append(
            {
                "left_layer_ordinal": left,
                "right_layer_ordinal": right,
                "left_generator_plan_sha256": (
                    runtime.generator_plan_sha256s[left]
                ),
                "right_generator_plan_sha256": (
                    runtime.generator_plan_sha256s[right]
                ),
                "left_fingerprint_sha256": summaries[left][
                    "fingerprint_sha256"
                ],
                "right_fingerprint_sha256": summaries[right][
                    "fingerprint_sha256"
                ],
                "analysis_split_sha256": split_sha256,
                "shared_prompt_count": analysis.prompt_count,
                **{
                    name: pair_metadata[name]
                    for name in (
                        "centered_shared_logit_effect_cosine",
                        "prompt_nll_effect_spearman",
                        "top_importance_overlap",
                        "top_importance_sign_agreement",
                        "top_importance_intersection_count",
                        "sufficient_causal_variation",
                        "observational_hypothesis",
                        "observational_only",
                        "authorizes_merge",
                        "authorizes_pruning",
                        "authorizes_routing",
                        "authorizes_mutation",
                    )
                },
            }
        )
    return tuple(rows)


def _validate_preflight(
    *,
    revision: str,
    output: Path | str,
    base_artifact_path: Path | str,
    refit_artifact_path: Path | str,
    model_id: str,
    device_name: str,
    dtype: str,
    max_length: int,
    tokenization_batch_size: int,
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
    output_path = Path(output)
    if output_path.suffix != ".json":
        raise ValueError("causal fingerprint output must use .json")
    if output_path.exists():
        raise FileExistsError("refusing to overwrite causal fingerprint output")
    base_path = Path(base_artifact_path)
    refit_path = Path(refit_artifact_path)
    if (
        not base_path.is_file()
        or not refit_path.is_file()
        or base_path.resolve() == refit_path.resolve()
    ):
        raise FileNotFoundError(
            "base and refit artifacts must be distinct existing files"
        )


def run_gemma3_generator_causal_fingerprint_experiment(
    *,
    eval_export_path: Path | str = DEFAULT_EVAL_EXPORT,
    revision: str,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_REFIT_ARTIFACT,
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> dict[str, object]:
    """Run and publish the frozen singleton-suppression fingerprint graph."""

    _validate_preflight(
        revision=revision,
        output=output,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
    )
    _progress("sources: authenticate and restore base plus refit generators")
    runtime = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    model_metadata = runtime.model_metadata
    if (
        model_metadata.get("model_id") != model_id
        or model_metadata.get("requested_revision") != revision
        or model_metadata.get("resolved_commit") != revision
    ):
        raise ValueError("requested model/revision differs from frozen runtime")

    analysis_content = tuple(
        runtime.analysis_split["content_sha256"]  # type: ignore[arg-type]
    )
    eval_export = load_development_prompt_export(eval_export_path)
    partition_metadata = runtime.partition_metadata
    selection_count = partition_metadata.get("selection_prompt_count")
    expected_prompt_count = partition_metadata.get("expected_prompt_count")
    nested_selection = _require_mapping(
        partition_metadata.get("selection"),
        label="frozen selection partition",
    )
    partition_salt = nested_selection.get("partition_salt")
    if (
        type(selection_count) is not int
        or type(expected_prompt_count) is not int
        or not isinstance(partition_salt, str)
    ):
        raise TypeError("frozen partition recipe is incomplete")
    partition = partition_development_export_for_interactions(
        eval_export,
        selection_count=selection_count,
        expected_prompt_count=expected_prompt_count,
        partition_salt=partition_salt,
    )
    if partition.metadata() != partition_metadata:
        raise ValueError("live development partition differs from frozen source")
    prompts = partition.assessment.prompts
    analysis_example_ids = partition.assessment.prompt_sha256s

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
    if adapter.model_fingerprint() != runtime.source_model_sha256:
        raise ValueError("live model fingerprint differs from frozen runtime")

    _progress("split: tokenize exact frozen open-development membership")
    batches, stream = _materialize_split(
        tokenizer,
        prompts,
        split_name="full_mlp_stack_open_development_assessment",
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
        device=device,
    )
    batches = _bind_batch_example_ids(batches, analysis_example_ids)
    safe_stream = _safe_tokenized_stream_metadata(stream)
    _live_split_matches_normalized_refit(
        safe_stream,
        runtime.analysis_split,
    )
    if (
        _stream_content_sha256s(stream, label="causal fingerprint analysis")
        != analysis_content
    ):
        raise ValueError("live analysis prompt membership differs from frozen")

    full_executor = Gemma3FullMLPStackExecutor(
        adapter,
        runtime.replacements,
    )
    intervention_executor = (
        FrozenGemma3GeneratorCausalInterventionExecutor(full_executor)
    )
    if (
        intervention_executor.generator_plan_sha256s
        != runtime.generator_plan_sha256s
    ):
        raise RuntimeError("physical generator catalog differs from runtime")
    provenance = GeneratorCausalFingerprintProvenance(
        source_model_sha256=runtime.source_model_sha256,
        generator_catalog_sha256=_catalog_sha256(runtime),
        evaluation_split_sha256=str(
            runtime.analysis_split["serialized_sha256"]
        ),
        objective_sha256=_objective_sha256(),
    )
    accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=intervention_executor.generator_ids,
        provenance=provenance,
        anchor_count=DEFAULT_ANCHOR_COUNT,
        top_importance_count=DEFAULT_TOP_IMPORTANCE_COUNT,
    )

    _progress(
        "trace: generated baseline plus 18 singleton suppressions per batch"
    )
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                if batch.example_ids is None:
                    raise ValueError("analysis batch lacks exact example ids")
                expected_order: list[int | None] = [
                    None,
                    *range(intervention_executor.layer_count),
                ]
                observed_order: list[int | None] = []

                def visit(
                    execution: Gemma3GeneratorCausalIntervention,
                ) -> None:
                    muted = execution.muted_layer_ordinal
                    expected = expected_order[len(observed_order)]
                    if muted != expected:
                        raise RuntimeError(
                            "intervention sweep order differs from protocol"
                        )
                    observed_order.append(muted)
                    if (
                        execution.generator_plan_sha256s
                        != runtime.generator_plan_sha256s
                        or execution.valid_tokens
                        != int(batch.valid_positions.sum().item())
                    ):
                        raise RuntimeError(
                            "intervention execution binding/accounting drifted"
                        )
                    logits = _model_logits(execution.model_output)
                    if muted is None:
                        accumulator.begin_batch(
                            example_ids=batch.example_ids or (),
                            baseline_logits=logits,
                            targets=batch.targets,
                            supervised_mask=batch.targets != -100,
                        )
                    else:
                        accumulator.add_muted_generator(
                            intervention_executor.generator_ids[muted],
                            logits,
                        )

                intervention_executor.visit_baseline_and_single_suppressions(
                    batch.model_inputs,
                    visitor=visit,
                )
                if observed_order != expected_order:
                    raise RuntimeError("intervention sweep was incomplete")
                accumulator.finish_batch()
                _progress(
                    f"trace: completed batch {batch_index + 1}/{len(batches)}"
                )
                gc.collect()
        analysis = accumulator.finalize()
    finally:
        accumulator.close()

    if (
        analysis.example_id_sha256s
        != tuple(
            generator_fingerprint_example_id_sha256(value)
            for value in analysis_example_ids
        )
        or analysis.generator_ids != intervention_executor.generator_ids
    ):
        raise RuntimeError("final causal analysis identity drifted")

    lineage = _generator_fit_lineage(runtime)
    summaries = _generator_causal_summaries(
        analysis,
        runtime,
        prompt_content_sha256s=analysis_content,
        example_ids=analysis_example_ids,
    )
    pairs = _pairwise_similarities(analysis, runtime, summaries)
    analysis_split = {
        "role": "adaptive_open_development_generator_family_discovery",
        "serialized_sha256": runtime.analysis_split["serialized_sha256"],
        "content_sha256": analysis_content,
        "example_count": runtime.analysis_split["example_count"],
        "logical_valid_tokens": runtime.analysis_split[
            "logical_valid_tokens"
        ],
        "supervised_tokens": runtime.analysis_split["supervised_tokens"],
        "membership_exact": True,
        "assurance": "caller_declared_self_attested",
        "externally_authenticated": False,
        "heldout_confirmation": False,
        "used_for_adaptive_analysis": True,
        "used_for_generator_fit": False,
        "used_for_generator_selection": False,
    }
    artifact_model = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit": revision,
        "adapter_model_fingerprint": runtime.source_model_sha256,
        "local_files_only": True,
    }
    frozen_sources = {
        "base_full_stack": {
            "schema": GEMMA3_FULL_MLP_STACK_SCHEMA,
            "format_version": GEMMA3_FULL_MLP_STACK_FORMAT_VERSION,
            "artifact_file_sha256": runtime.base_artifact_file_sha256,
            "scientific_payload_sha256": (
                runtime.base_scientific_payload_sha256
            ),
            "frozen_before_analysis": True,
        },
        "sequential_refit": {
            "schema": GEMMA3_FULL_MLP_STACK_REFIT_SCHEMA,
            "format_version": GEMMA3_FULL_MLP_STACK_REFIT_FORMAT_VERSION,
            "artifact_file_sha256": runtime.refit_artifact_file_sha256,
            "scientific_payload_sha256": (
                runtime.refit_scientific_payload_sha256
            ),
            "frozen_before_analysis": True,
        },
    }
    _progress("artifact: publish tensor-free observational graph")
    return save_gemma3_generator_causal_fingerprint_artifact(
        output,
        model=artifact_model,
        frozen_sources=frozen_sources,
        analysis_split=analysis_split,
        causal_analysis_lineage=_causal_analysis_lineage(analysis),
        generator_fit_lineage=lineage,
        generator_causal_summaries=summaries,
        pairwise_similarities=pairs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace non-destructive prompt-conditioned causal fingerprints "
            "for the frozen full Gemma generator stack."
        )
    )
    parser.add_argument("--eval-export", type=Path, default=DEFAULT_EVAL_EXPORT)
    parser.add_argument(
        "--base-artifact",
        type=Path,
        default=DEFAULT_FULL_MLP_STACK_ARTIFACT,
    )
    parser.add_argument(
        "--refit-artifact",
        type=Path,
        default=DEFAULT_REFIT_ARTIFACT,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_gemma3_generator_causal_fingerprint_experiment(
        eval_export_path=arguments.eval_export,
        revision=arguments.revision,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        output=arguments.output,
        model_id=arguments.model,
        cache_dir=arguments.cache_dir,
        device_name=arguments.device,
        dtype=arguments.dtype,
        max_length=arguments.max_length,
        tokenization_batch_size=arguments.tokenization_batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
