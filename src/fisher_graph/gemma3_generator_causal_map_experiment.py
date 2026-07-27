"""Build a frozen, source-safe causal interaction map for Gemma generators.

The runner restores the authenticated base-plus-refit generator stack, binds
the already-published singleton causal-fingerprint artifact, and replays one
generated baseline, all 18 singleton suppressions, and all 153 canonical
two-generator suppressions for every open-development assessment batch.

The result is observational only.  It contains node identities, copied
singleton similarities, finite-difference joint interactions, directed local
responses, and hashed declared-family cohort summaries.  It never proposes or
executes a merge, prune, route, compilation, or model mutation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gc
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
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
    GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION,
    GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA,
    load_gemma3_generator_causal_fingerprint_artifact,
)
from .gemma3_generator_causal_fingerprint_experiment import (
    DEFAULT_ANCHOR_COUNT,
    DEFAULT_OUTPUT as DEFAULT_CAUSAL_FINGERPRINT_ARTIFACT,
    _catalog_sha256,
    _generator_causal_summaries,
    _generator_fit_lineage,
    _live_split_matches_normalized_refit,
    _model_logits,
    _pairwise_similarities,
    _require_mapping,
)
from .gemma3_generator_causal_intervention import (
    FrozenGemma3GeneratorCausalInterventionExecutor,
    Gemma3GeneratorCausalIntervention,
)
from .gemma3_modal_generator_dev_experiment import (
    DEFAULT_EVAL_EXPORT,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZATION_BATCH_SIZE,
    _safe_tokenized_stream_metadata,
    load_development_prompt_export,
)
from .gemma3_modal_generator_multifragment_dev_experiment import (
    _bind_batch_example_ids,
    _stream_content_sha256s,
)
from .generator_causal_fingerprints import (
    GeneratorCausalFingerprintAccumulator,
    GeneratorCausalFingerprintAnalysis,
    GeneratorCausalFingerprintProvenance,
    ObservationalFamilyPolicy,
)
from .generator_interaction_map import (
    GeneratorInteractionMapAccumulator,
    GeneratorInteractionMapAnalysis,
    GeneratorInteractionMapProvenance,
    interaction_map_example_id_sha256,
)
from .modal_graph_rung_evaluation import (
    DevelopmentInteractionPartition,
    partition_development_export_for_interactions,
)

from .gemma3_generator_causal_map_artifact import (
    save_gemma3_generator_causal_map_artifact,
)


__all__ = [
    "DEFAULT_OUTPUT",
    "build_parser",
    "main",
    "run_gemma3_generator_causal_map_experiment",
]


DEFAULT_OUTPUT = Path(
    ".local-runs/google--gemma-3-270m/"
    "modal-generator-causal-map-dev-v1.json"
)
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_FAMILY_ID_DOMAIN = b"fisher_graph.gemma3.causal_map.family_id.v1\0"
_COHORT_MEMBERSHIP_DOMAIN = (
    b"fisher_graph.gemma3.causal_map.cohort_membership.v1\0"
)
_SINGLETON_REPLAY_DOMAIN = (
    b"fisher_graph.gemma3.causal_map.singleton_replay.v1\0"
)
_PAIR_COUNT = 153


def _progress(message: str) -> None:
    print(
        f"[gemma-generator-causal-map] {message}",
        file=sys.stderr,
        flush=True,
    )


def _file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object, *, domain: bytes) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(encoded)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty metric sequence")
    return math.fsum(values) / len(values)


def _rms(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty metric sequence")
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def _mean_defined(
    values: Sequence[float],
    defined: Sequence[bool],
) -> float:
    selected = tuple(
        value
        for value, is_defined in zip(values, defined, strict=True)
        if is_defined
    )
    return 0.0 if not selected else _mean(selected)


def _family_id_sha256(family_id: str) -> str:
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("family_id must be a nonempty string")
    digest = hashlib.sha256()
    digest.update(_FAMILY_ID_DOMAIN)
    digest.update(family_id.encode("utf-8"))
    return digest.hexdigest()


def _cohort_membership_sha256(
    *,
    cohort_id: str,
    prompt_content_sha256s: Sequence[str],
) -> str:
    return _json_digest(
        {
            "cohort_id": cohort_id,
            "prompt_content_sha256s": tuple(prompt_content_sha256s),
        },
        domain=_COHORT_MEMBERSHIP_DOMAIN,
    )


def _active_generator_output_mapping(
    execution: Gemma3GeneratorCausalIntervention | object,
    generator_ids: Sequence[str],
) -> dict[str, Tensor]:
    """Convert an ephemeral residual tuple to the exact active-id mapping."""

    ids = tuple(generator_ids)
    traces = getattr(execution, "generated_residuals", None)
    suppressed = getattr(execution, "suppressed_layer_ordinals", None)
    if (
        type(traces) is not tuple
        or type(suppressed) is not tuple
        or len(traces) != len(ids)
    ):
        raise ValueError(
            "baseline/singleton execution lacks an exact residual trace"
        )
    suppressed_set = set(suppressed)
    result: dict[str, Tensor] = {}
    for ordinal, (generator_id, value) in enumerate(
        zip(ids, traces, strict=True)
    ):
        if ordinal in suppressed_set:
            if value is not None:
                raise ValueError("suppressed generator exposed a residual")
        elif not isinstance(value, Tensor):
            raise ValueError("active generator lacks a residual tensor")
        else:
            result[generator_id] = value
    if set(result) != {
        generator_id
        for ordinal, generator_id in enumerate(ids)
        if ordinal not in suppressed_set
    }:
        raise RuntimeError("active residual mapping is incomplete")
    return result


def _fingerprint_policy(
    fingerprint: Mapping[str, object],
) -> ObservationalFamilyPolicy:
    lineage = _require_mapping(
        fingerprint.get("causal_analysis_lineage"),
        label="fingerprint causal analysis lineage",
    )
    raw = _require_mapping(
        lineage.get("observational_family_policy"),
        label="fingerprint observational family policy",
    )
    return ObservationalFamilyPolicy(
        minimum_centered_effect_cosine=float(
            raw["minimum_centered_effect_cosine"]
        ),
        minimum_prompt_nll_spearman=float(
            raw["minimum_prompt_nll_spearman"]
        ),
        minimum_top_importance_overlap=float(
            raw["minimum_top_importance_overlap"]
        ),
        minimum_top_importance_sign_agreement=float(
            raw["minimum_top_importance_sign_agreement"]
        ),
        minimum_prompt_count=int(raw["minimum_prompt_count"]),
    )


def _authenticate_fingerprint_source(
    fingerprint: Mapping[str, object],
    runtime: Gemma3RefitRuntimeCatalog | object,
    *,
    model_id: str,
    revision: str,
) -> None:
    """Bind a strict-loaded fingerprint to the exact restored runtime."""

    model = _require_mapping(
        fingerprint.get("model"),
        label="fingerprint model",
    )
    if model != {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit": revision,
        "adapter_model_fingerprint": runtime.source_model_sha256,
        "local_files_only": True,
    }:
        raise ValueError("fingerprint model differs from frozen runtime")

    split = _require_mapping(
        fingerprint.get("analysis_split"),
        label="fingerprint analysis split",
    )
    for field in (
        "serialized_sha256",
        "example_count",
        "logical_valid_tokens",
        "supervised_tokens",
    ):
        if split.get(field) != runtime.analysis_split[field]:
            raise ValueError(
                "fingerprint split differs from frozen runtime "
                f"at {field}"
            )
    if tuple(split.get("content_sha256", ())) != tuple(
        runtime.analysis_split["content_sha256"]
    ):
        raise ValueError("fingerprint prompt membership differs from runtime")

    plans = tuple(fingerprint.get("deployed_generator_plan_sha256s", ()))
    if plans != tuple(runtime.generator_plan_sha256s):
        raise ValueError("fingerprint generator plans differ from runtime")
    raw_lineage = fingerprint.get("generator_fit_lineage")
    if isinstance(raw_lineage, (str, bytes)) or not isinstance(
        raw_lineage,
        Sequence,
    ):
        raise TypeError("fingerprint generator lineage must be a sequence")
    if tuple(dict(row) for row in raw_lineage) != _generator_fit_lineage(
        runtime
    ):
        raise ValueError("fingerprint generator lineage differs from runtime")

    sources = _require_mapping(
        fingerprint.get("frozen_sources"),
        label="fingerprint frozen sources",
    )
    expected_sources = {
        "base_full_stack": (
            runtime.base_artifact_file_sha256,
            runtime.base_scientific_payload_sha256,
        ),
        "sequential_refit": (
            runtime.refit_artifact_file_sha256,
            runtime.refit_scientific_payload_sha256,
        ),
    }
    for name, (file_sha, scientific_sha) in expected_sources.items():
        source = _require_mapping(
            sources.get(name),
            label=f"fingerprint source {name}",
        )
        if (
            source.get("artifact_file_sha256") != file_sha
            or source.get("scientific_payload_sha256") != scientific_sha
            or source.get("frozen_before_analysis") is not True
        ):
            raise ValueError(
                f"fingerprint {name} source differs from runtime"
            )

    lineage = _require_mapping(
        fingerprint.get("causal_analysis_lineage"),
        label="fingerprint causal analysis lineage",
    )
    if (
        lineage.get("source_model_sha256") != runtime.source_model_sha256
        or lineage.get("generator_catalog_sha256")
        != _catalog_sha256(runtime)
        or lineage.get("evaluation_split_sha256")
        != runtime.analysis_split["serialized_sha256"]
        or lineage.get("generator_count") != len(runtime.replacements)
        or lineage.get("prompt_count") != runtime.analysis_split[
            "example_count"
        ]
    ):
        raise ValueError("fingerprint causal lineage differs from runtime")


def _verify_replayed_singleton_identity(
    replay: GeneratorCausalFingerprintAnalysis,
    fingerprint: Mapping[str, object],
    runtime: Gemma3RefitRuntimeCatalog | object,
    *,
    prompt_content_sha256s: Sequence[str],
    example_ids: Sequence[str],
) -> None:
    """Require exact core and published singleton identity before mapping."""

    lineage = _require_mapping(
        fingerprint.get("causal_analysis_lineage"),
        label="fingerprint causal analysis lineage",
    )
    if replay.artifact_sha256 != lineage.get("artifact_sha256"):
        raise ValueError(
            "replayed singleton causal analysis differs from fingerprint"
        )
    summaries = _generator_causal_summaries(
        replay,
        runtime,
        prompt_content_sha256s=prompt_content_sha256s,
        example_ids=example_ids,
    )
    raw_summaries = fingerprint.get("generator_causal_summaries")
    raw_pairs = fingerprint.get("pairwise_similarities")
    if isinstance(raw_summaries, (str, bytes)) or not isinstance(
        raw_summaries,
        Sequence,
    ):
        raise TypeError("frozen singleton summaries must be a sequence")
    if _json_digest(
        tuple(raw_summaries),
        domain=_SINGLETON_REPLAY_DOMAIN,
    ) != _json_digest(
        summaries,
        domain=_SINGLETON_REPLAY_DOMAIN,
    ):
        raise ValueError(
            "replayed per-generator fingerprints differ from frozen source"
        )
    pairs = _pairwise_similarities(replay, runtime, summaries)
    if isinstance(raw_pairs, (str, bytes)) or not isinstance(
        raw_pairs,
        Sequence,
    ):
        raise TypeError("frozen singleton similarities must be a sequence")
    if _json_digest(
        tuple(raw_pairs),
        domain=_SINGLETON_REPLAY_DOMAIN,
    ) != _json_digest(
        pairs,
        domain=_SINGLETON_REPLAY_DOMAIN,
    ):
        raise ValueError(
            "replayed singleton pair similarities differ from frozen source"
        )


def _generator_nodes(
    fingerprint: Mapping[str, object],
    *,
    generator_ids: Sequence[str],
) -> tuple[dict[str, object], ...]:
    lineage_rows = fingerprint["generator_fit_lineage"]
    summary_rows = fingerprint["generator_causal_summaries"]
    ids = tuple(generator_ids)
    if (
        isinstance(lineage_rows, (str, bytes))
        or not isinstance(lineage_rows, Sequence)
        or isinstance(summary_rows, (str, bytes))
        or not isinstance(summary_rows, Sequence)
        or len(lineage_rows) != len(summary_rows)
        or len(ids) != len(summary_rows)
        or len(ids) != len(set(ids))
        or any(not isinstance(value, str) or not value for value in ids)
    ):
        raise TypeError("fingerprint node sources are invalid")
    rows: list[dict[str, object]] = []
    for ordinal, (raw_lineage, raw_summary) in enumerate(
        zip(lineage_rows, summary_rows, strict=True)
    ):
        lineage = _require_mapping(
            raw_lineage,
            label=f"generator lineage {ordinal}",
        )
        summary = _require_mapping(
            raw_summary,
            label=f"generator summary {ordinal}",
        )
        rows.append(
            {
                "layer_ordinal": ordinal,
                "layer_id": lineage["layer_id"],
                "generator_id": ids[ordinal],
                "deployment_source": lineage["deployment_source"],
                "deployed_generator_plan_sha256": summary[
                    "deployed_generator_plan_sha256"
                ],
                "deployed_fit_sha256": summary["deployed_fit_sha256"],
                "singleton_fingerprint_sha256": summary[
                    "fingerprint_sha256"
                ],
                "mean_muted_minus_baseline_nll_per_token": summary[
                    "mean_muted_minus_baseline_nll_per_token"
                ],
                "rms_muted_minus_baseline_nll_per_token": summary[
                    "rms_muted_minus_baseline_nll_per_token"
                ],
                "mean_absolute_muted_minus_baseline_nll_per_token": summary[
                    "mean_absolute_muted_minus_baseline_nll_per_token"
                ],
                "maximum_absolute_muted_minus_baseline_nll_per_token": summary[
                    "maximum_absolute_muted_minus_baseline_nll_per_token"
                ],
                "mean_baseline_to_muted_kl_per_token": summary[
                    "mean_baseline_to_muted_kl_per_token"
                ],
                "mean_top1_agreement_to_baseline": summary[
                    "mean_top1_agreement_to_baseline"
                ],
                "mean_centered_anchor_logit_effect_rms": summary[
                    "mean_centered_anchor_logit_effect_rms"
                ],
                "positive_delta_fraction": summary[
                    "positive_delta_fraction"
                ],
                "observational_only": True,
                "authorizes_merge": False,
                "authorizes_pruning": False,
                "authorizes_routing": False,
                "authorizes_compilation": False,
                "authorizes_execution": False,
                "authorizes_mutation": False,
            }
        )
    return tuple(rows)


def _copied_pairwise_similarities(
    fingerprint: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    raw = fingerprint.get("pairwise_similarities")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("fingerprint pairwise similarities must be a sequence")
    rows = tuple(dict(row) for row in raw)
    if len(rows) != _PAIR_COUNT:
        raise ValueError("fingerprint lacks the complete 153-pair catalog")
    return rows


def _joint_interactions(
    analysis: GeneratorInteractionMapAnalysis | object,
    runtime: Gemma3RefitRuntimeCatalog | object,
    fingerprint: Mapping[str, object],
    *,
    prompt_content_sha256s: Sequence[str],
) -> tuple[dict[str, object], ...]:
    prompts = tuple(prompt_content_sha256s)
    summaries = tuple(fingerprint["generator_causal_summaries"])
    rows: list[dict[str, object]] = []
    for pair_index, (left_id, right_id) in enumerate(analysis.pair_catalog):
        left = analysis.generator_ids.index(left_id)
        right = analysis.generator_ids.index(right_id)
        nll = tuple(
            float(value)
            for value in analysis.prompt_nll_second_differences[
                pair_index
            ].tolist()
        )
        kl = tuple(
            float(value)
            for value in analysis.prompt_joint_baseline_to_condition_kls[
                pair_index
            ].tolist()
        )
        top1 = tuple(
            float(value)
            for value in analysis.prompt_joint_top1_agreements[
                pair_index
            ].tolist()
        )
        residual = tuple(
            float(value)
            for value in analysis.prompt_centered_anchor_interaction_residual_rms[
                pair_index
            ].tolist()
        )
        denominator = tuple(
            float(value)
            for value in analysis.prompt_relative_interaction_denominator_rms[
                pair_index
            ].tolist()
        )
        ratio = tuple(
            float(value)
            for value in analysis.prompt_relative_interaction_ratios[
                pair_index
            ].tolist()
        )
        defined = tuple(
            bool(value)
            for value in analysis.prompt_relative_interaction_defined[
                pair_index
            ].tolist()
        )
        prompt_rows = tuple(
            {
                "prompt_content_sha256": prompt,
                "nll_second_difference_per_token": nll_value,
                "joint_baseline_to_condition_kl_per_token": kl_value,
                "joint_top1_agreement_to_baseline": top1_value,
                "centered_anchor_interaction_residual_rms": residual_value,
                "relative_interaction_denominator_rms": denominator_value,
                "relative_interaction_ratio": ratio_value,
                "relative_interaction_defined": defined_value,
            }
            for (
                prompt,
                nll_value,
                kl_value,
                top1_value,
                residual_value,
                denominator_value,
                ratio_value,
                defined_value,
            ) in zip(
                prompts,
                nll,
                kl,
                top1,
                residual,
                denominator,
                ratio,
                defined,
                strict=True,
            )
        )
        left_summary = _require_mapping(
            summaries[left],
            label=f"left singleton summary {left}",
        )
        right_summary = _require_mapping(
            summaries[right],
            label=f"right singleton summary {right}",
        )
        rows.append(
            {
                "left_layer_ordinal": left,
                "right_layer_ordinal": right,
                "left_generator_id": left_id,
                "right_generator_id": right_id,
                "left_generator_plan_sha256": runtime.generator_plan_sha256s[
                    left
                ],
                "right_generator_plan_sha256": runtime.generator_plan_sha256s[
                    right
                ],
                "left_singleton_fingerprint_sha256": left_summary[
                    "fingerprint_sha256"
                ],
                "right_singleton_fingerprint_sha256": right_summary[
                    "fingerprint_sha256"
                ],
                "analysis_split_sha256": runtime.analysis_split[
                    "serialized_sha256"
                ],
                "prompt_count": len(prompts),
                "prompt_interactions": prompt_rows,
                "mean_nll_second_difference_per_token": _mean(nll),
                "rms_nll_second_difference_per_token": _rms(nll),
                "mean_absolute_nll_second_difference_per_token": _mean(
                    tuple(abs(value) for value in nll)
                ),
                "maximum_absolute_nll_second_difference_per_token": max(
                    abs(value) for value in nll
                ),
                "mean_joint_baseline_to_condition_kl_per_token": _mean(kl),
                "mean_joint_top1_agreement_to_baseline": _mean(top1),
                "mean_centered_anchor_interaction_residual_rms": _mean(
                    residual
                ),
                "mean_relative_interaction_denominator_rms": _mean(
                    denominator
                ),
                "relative_interaction_defined_fraction": (
                    sum(defined) / len(defined)
                ),
                "mean_relative_interaction_ratio_over_defined": _mean_defined(
                    ratio,
                    defined,
                ),
                "maximum_relative_interaction_ratio_over_defined": (
                    max(
                        (
                            value
                            for value, is_defined in zip(
                                ratio,
                                defined,
                                strict=True,
                            )
                            if is_defined
                        ),
                        default=0.0,
                    )
                ),
                "observational_only": True,
                "authorizes_merge": False,
                "authorizes_pruning": False,
                "authorizes_routing": False,
                "authorizes_compilation": False,
                "authorizes_execution": False,
                "authorizes_mutation": False,
            }
        )
    return tuple(rows)


def _directed_edges(
    analysis: GeneratorInteractionMapAnalysis | object,
    runtime: Gemma3RefitRuntimeCatalog | object,
    fingerprint: Mapping[str, object],
    *,
    prompt_content_sha256s: Sequence[str],
) -> tuple[dict[str, object], ...]:
    prompts = tuple(prompt_content_sha256s)
    summaries = tuple(fingerprint["generator_causal_summaries"])
    rows: list[dict[str, object]] = []
    for edge_index, (upstream_id, downstream_id) in enumerate(
        analysis.directed_edge_catalog
    ):
        upstream = analysis.generator_ids.index(upstream_id)
        downstream = analysis.generator_ids.index(downstream_id)
        response = tuple(
            float(value)
            for value in analysis.prompt_directed_response_rms[
                edge_index
            ].tolist()
        )
        baseline = tuple(
            float(value)
            for value in analysis.prompt_directed_baseline_output_rms[
                edge_index
            ].tolist()
        )
        cosine = tuple(
            float(value)
            for value in analysis.prompt_directed_response_cosines[
                edge_index
            ].tolist()
        )
        cosine_defined = tuple(
            bool(value)
            for value in analysis.prompt_directed_response_cosine_defined[
                edge_index
            ].tolist()
        )
        ratio = tuple(
            float(value)
            for value in analysis.prompt_directed_response_ratios[
                edge_index
            ].tolist()
        )
        ratio_defined = tuple(
            bool(value)
            for value in analysis.prompt_directed_response_ratio_defined[
                edge_index
            ].tolist()
        )
        prompt_rows = tuple(
            {
                "prompt_content_sha256": prompt,
                "directed_response_rms": response_value,
                "baseline_generator_output_rms": baseline_value,
                "directed_response_cosine": cosine_value,
                "directed_response_cosine_defined": cosine_is_defined,
                "directed_response_ratio": ratio_value,
                "directed_response_ratio_defined": ratio_is_defined,
            }
            for (
                prompt,
                response_value,
                baseline_value,
                cosine_value,
                cosine_is_defined,
                ratio_value,
                ratio_is_defined,
            ) in zip(
                prompts,
                response,
                baseline,
                cosine,
                cosine_defined,
                ratio,
                ratio_defined,
                strict=True,
            )
        )
        upstream_summary = _require_mapping(
            summaries[upstream],
            label=f"upstream singleton summary {upstream}",
        )
        downstream_summary = _require_mapping(
            summaries[downstream],
            label=f"downstream singleton summary {downstream}",
        )
        rows.append(
            {
                "upstream_layer_ordinal": upstream,
                "downstream_layer_ordinal": downstream,
                "upstream_generator_id": upstream_id,
                "downstream_generator_id": downstream_id,
                "upstream_generator_plan_sha256": (
                    runtime.generator_plan_sha256s[upstream]
                ),
                "downstream_generator_plan_sha256": (
                    runtime.generator_plan_sha256s[downstream]
                ),
                "upstream_singleton_fingerprint_sha256": upstream_summary[
                    "fingerprint_sha256"
                ],
                "downstream_singleton_fingerprint_sha256": downstream_summary[
                    "fingerprint_sha256"
                ],
                "analysis_split_sha256": runtime.analysis_split[
                    "serialized_sha256"
                ],
                "prompt_count": len(prompts),
                "prompt_responses": prompt_rows,
                "mean_directed_response_rms": _mean(response),
                "maximum_directed_response_rms": max(response),
                "mean_baseline_generator_output_rms": _mean(baseline),
                "directed_response_cosine_defined_fraction": (
                    sum(cosine_defined) / len(cosine_defined)
                ),
                "mean_directed_response_cosine_over_defined": _mean_defined(
                    cosine,
                    cosine_defined,
                ),
                "directed_response_ratio_defined_fraction": (
                    sum(ratio_defined) / len(ratio_defined)
                ),
                "mean_directed_response_ratio_over_defined": _mean_defined(
                    ratio,
                    ratio_defined,
                ),
                "maximum_directed_response_ratio_over_defined": max(
                    (
                        value
                        for value, is_defined in zip(
                            ratio,
                            ratio_defined,
                            strict=True,
                        )
                        if is_defined
                    ),
                    default=0.0,
                ),
                "strict_upstream_invariance_confirmed": True,
                "observational_only": True,
                "authorizes_merge": False,
                "authorizes_pruning": False,
                "authorizes_routing": False,
                "authorizes_compilation": False,
                "authorizes_execution": False,
                "authorizes_mutation": False,
            }
        )
    return tuple(rows)


def _prompt_cohorts(
    assessment: DevelopmentInteractionPartition | object,
    *,
    prompt_content_sha256s: Sequence[str],
) -> tuple[dict[str, object], ...]:
    raw_hashes = tuple(assessment.prompt_sha256s)
    families = tuple(assessment.family_ids)
    content_hashes = tuple(prompt_content_sha256s)
    if (
        len(raw_hashes) != len(families)
        or len(raw_hashes) != len(content_hashes)
        or len(raw_hashes) == 0
    ):
        raise ValueError("assessment cohort columns are inconsistent")
    ordered_families = tuple(
        family_id
        for _, family_id in sorted(
            (
                (_family_id_sha256(family_id), family_id)
                for family_id in set(families)
            ),
            key=lambda item: item[0],
        )
    )
    rows: list[dict[str, object]] = []
    for cohort_ordinal, family_id in enumerate(ordered_families):
        member_indices = tuple(
            index
            for index, value in enumerate(families)
            if value == family_id
        )
        members = tuple(content_hashes[index] for index in member_indices)
        cohort_id = _family_id_sha256(family_id)
        membership = _cohort_membership_sha256(
            cohort_id=cohort_id,
            prompt_content_sha256s=members,
        )
        rows.append(
            {
                "cohort_ordinal": cohort_ordinal,
                "cohort_id": cohort_id,
                "source_partition_sha256": assessment.artifact_sha256,
                "prompt_content_sha256s": members,
                "prompt_count": len(members),
                "membership_sha256": membership,
                "membership_exact": True,
                "stability_status": (
                    "descriptive_singleton_insufficient_for_stability"
                    if len(members) == 1
                    else "descriptive_multi_prompt_open_development"
                ),
            }
        )
    return tuple(rows)


def _generator_cohort_affinities(
    fingerprint: Mapping[str, object],
    runtime: Gemma3RefitRuntimeCatalog | object,
    cohorts: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    prompts = tuple(fingerprint["analysis_split"]["content_sha256"])
    prompt_index = {value: index for index, value in enumerate(prompts)}
    summaries = tuple(fingerprint["generator_causal_summaries"])
    prompt_deltas = tuple(
        tuple(
            float(signature["muted_minus_baseline_nll_per_token"])
            for signature in summary["prompt_signatures"]
        )
        for summary in summaries
    )
    importance_ranks: list[list[int]] = [
        [0 for _ in prompts] for _ in summaries
    ]
    for prompt_ordinal in range(len(prompts)):
        order = sorted(
            range(len(summaries)),
            key=lambda layer: (
                -abs(prompt_deltas[layer][prompt_ordinal]),
                layer,
            ),
        )
        for rank, layer in enumerate(order, start=1):
            importance_ranks[layer][prompt_ordinal] = rank

    rows: list[dict[str, object]] = []
    for layer_ordinal, summary in enumerate(summaries):
        for cohort in cohorts:
            indices = tuple(
                prompt_index[value]
                for value in cohort["prompt_content_sha256s"]
            )
            signatures = tuple(summary["prompt_signatures"])
            selected = tuple(signatures[index] for index in indices)
            deltas = tuple(
                float(value["muted_minus_baseline_nll_per_token"])
                for value in selected
            )
            kls = tuple(
                float(value["baseline_to_muted_kl_per_token"])
                for value in selected
            )
            top1 = tuple(
                float(value["top1_agreement_to_baseline"])
                for value in selected
            )
            effect = tuple(
                float(value["centered_anchor_logit_effect_rms"])
                for value in selected
            )
            positive_count = sum(value > 0.0 for value in deltas)
            rows.append(
                {
                    "layer_ordinal": layer_ordinal,
                    "cohort_ordinal": cohort["cohort_ordinal"],
                    "cohort_id": cohort["cohort_id"],
                    "membership_sha256": cohort["membership_sha256"],
                    "deployed_generator_plan_sha256": (
                        runtime.generator_plan_sha256s[layer_ordinal]
                    ),
                    "singleton_fingerprint_sha256": summary[
                        "fingerprint_sha256"
                    ],
                    "prompt_count": len(indices),
                    "mean_muted_minus_baseline_nll_per_token": _mean(deltas),
                    "rms_muted_minus_baseline_nll_per_token": _rms(deltas),
                    "mean_absolute_muted_minus_baseline_nll_per_token": _mean(
                        tuple(abs(value) for value in deltas)
                    ),
                    "maximum_absolute_muted_minus_baseline_nll_per_token": max(
                        abs(value) for value in deltas
                    ),
                    "mean_baseline_to_muted_kl_per_token": _mean(kls),
                    "mean_top1_agreement_to_baseline": _mean(top1),
                    "mean_centered_anchor_logit_effect_rms": _mean(effect),
                    "positive_delta_count": positive_count,
                    "positive_delta_fraction": positive_count / len(indices),
                    "mean_absolute_nll_importance_rank": _mean(
                        tuple(
                            float(importance_ranks[layer_ordinal][index])
                            for index in indices
                        )
                    ),
                    "descriptive_only": True,
                    "authorizes_merge": False,
                    "authorizes_pruning": False,
                    "authorizes_routing": False,
                    "authorizes_compilation": False,
                    "authorizes_execution": False,
                    "authorizes_mutation": False,
                }
            )
    return tuple(rows)


def _validate_preflight(
    *,
    revision: str,
    output: Path | str,
    base_artifact_path: Path | str,
    refit_artifact_path: Path | str,
    fingerprint_artifact_path: Path | str,
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
        raise ValueError("causal map output must use .json")
    if output_path.exists():
        raise FileExistsError("refusing to overwrite causal map output")
    sources = tuple(
        Path(value)
        for value in (
            base_artifact_path,
            refit_artifact_path,
            fingerprint_artifact_path,
        )
    )
    if (
        any(not path.is_file() for path in sources)
        or len({path.resolve() for path in sources}) != len(sources)
        or output_path.resolve() in {path.resolve() for path in sources}
    ):
        raise FileNotFoundError(
            "base, refit, and fingerprint sources must be distinct files"
        )


def run_gemma3_generator_causal_map_experiment(
    *,
    eval_export_path: Path | str = DEFAULT_EVAL_EXPORT,
    revision: str,
    base_artifact_path: Path | str = DEFAULT_FULL_MLP_STACK_ARTIFACT,
    refit_artifact_path: Path | str = DEFAULT_REFIT_ARTIFACT,
    fingerprint_artifact_path: Path | str = (
        DEFAULT_CAUSAL_FINGERPRINT_ARTIFACT
    ),
    output: Path | str = DEFAULT_OUTPUT,
    model_id: str = DEFAULT_MODEL_ID,
    cache_dir: Path | str | None = None,
    device_name: str = "cpu",
    dtype: str = "float32",
    max_length: int = DEFAULT_MAX_LENGTH,
    tokenization_batch_size: int = DEFAULT_TOKENIZATION_BATCH_SIZE,
) -> dict[str, object]:
    """Run and publish the frozen full-generator causal interaction map."""

    _validate_preflight(
        revision=revision,
        output=output,
        base_artifact_path=base_artifact_path,
        refit_artifact_path=refit_artifact_path,
        fingerprint_artifact_path=fingerprint_artifact_path,
        model_id=model_id,
        device_name=device_name,
        dtype=dtype,
        max_length=max_length,
        tokenization_batch_size=tokenization_batch_size,
    )
    _progress("sources: restore base plus refit generators")
    runtime = restore_gemma3_full_mlp_stack_refit_runtime(
        base_artifact_path,
        refit_artifact_path,
    )
    if (
        runtime.model_metadata.get("model_id") != model_id
        or runtime.model_metadata.get("requested_revision") != revision
        or runtime.model_metadata.get("resolved_commit") != revision
    ):
        raise ValueError("requested model/revision differs from frozen runtime")

    _progress("sources: strict-load and bind singleton fingerprint")
    fingerprint = load_gemma3_generator_causal_fingerprint_artifact(
        fingerprint_artifact_path
    )
    _authenticate_fingerprint_source(
        fingerprint,
        runtime,
        model_id=model_id,
        revision=revision,
    )
    causal_lineage = _require_mapping(
        fingerprint["causal_analysis_lineage"],
        label="fingerprint causal analysis lineage",
    )
    objective_sha256 = str(causal_lineage["objective_sha256"])
    anchor_count = int(causal_lineage["anchor_count"])
    top_importance_count = int(causal_lineage["top_importance_count"])

    eval_export = load_development_prompt_export(eval_export_path)
    partition_metadata = runtime.partition_metadata
    nested_selection = _require_mapping(
        partition_metadata.get("selection"),
        label="frozen selection partition",
    )
    selection_count = partition_metadata.get("selection_prompt_count")
    expected_prompt_count = partition_metadata.get("expected_prompt_count")
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
    assessment = partition.assessment
    prompts = assessment.prompts
    analysis_example_ids = assessment.prompt_sha256s
    analysis_content = tuple(runtime.analysis_split["content_sha256"])

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

    _progress("split: tokenize exact frozen assessment membership")
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
        _stream_content_sha256s(stream, label="causal map analysis")
        != analysis_content
    ):
        raise ValueError("live causal-map membership differs from frozen split")

    full_executor = Gemma3FullMLPStackExecutor(adapter, runtime.replacements)
    executor = FrozenGemma3GeneratorCausalInterventionExecutor(full_executor)
    if executor.generator_plan_sha256s != runtime.generator_plan_sha256s:
        raise RuntimeError("physical generator catalog differs from runtime")
    fingerprint_provenance = GeneratorCausalFingerprintProvenance(
        source_model_sha256=runtime.source_model_sha256,
        generator_catalog_sha256=_catalog_sha256(runtime),
        evaluation_split_sha256=str(
            runtime.analysis_split["serialized_sha256"]
        ),
        objective_sha256=objective_sha256,
    )
    map_provenance = GeneratorInteractionMapProvenance(
        source_model_sha256=runtime.source_model_sha256,
        generator_catalog_sha256=_catalog_sha256(runtime),
        evaluation_split_sha256=str(
            runtime.analysis_split["serialized_sha256"]
        ),
        objective_sha256=objective_sha256,
    )
    fingerprint_accumulator = GeneratorCausalFingerprintAccumulator(
        generator_ids=executor.generator_ids,
        provenance=fingerprint_provenance,
        anchor_count=anchor_count,
        top_importance_count=top_importance_count,
        policy=_fingerprint_policy(fingerprint),
    )
    map_accumulator = GeneratorInteractionMapAccumulator(
        generator_ids=executor.generator_ids,
        provenance=map_provenance,
        anchor_count=anchor_count,
    )
    pair_schedule = tuple(combinations(range(executor.layer_count), 2))
    if len(pair_schedule) != _PAIR_COUNT:
        raise RuntimeError("physical generator catalog is not the 18-node stack")

    _progress(
        "trace: baseline + 18 singletons + 153 pairs per batch "
        f"({len(batches)} batches)"
    )
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                if batch.example_ids is None:
                    raise ValueError("analysis batch lacks exact example ids")
                expected_schedule = (
                    (),
                    *((ordinal,) for ordinal in range(executor.layer_count)),
                    *pair_schedule,
                )
                observed = 0
                pair_observed = 0

                def visit(
                    execution: Gemma3GeneratorCausalIntervention,
                ) -> None:
                    nonlocal observed, pair_observed
                    expected = expected_schedule[observed]
                    if execution.suppressed_layer_ordinals != expected:
                        raise RuntimeError(
                            "interaction sweep order differs from protocol"
                        )
                    observed += 1
                    if (
                        execution.generator_plan_sha256s
                        != runtime.generator_plan_sha256s
                        or execution.valid_tokens
                        != int(batch.valid_positions.sum().item())
                        or execution.observational_only is not True
                        or execution.mutation_authority is not False
                    ):
                        raise RuntimeError(
                            "interaction execution binding/accounting drifted"
                        )
                    logits = _model_logits(execution.model_output)
                    suppressed = execution.suppressed_layer_ordinals
                    if not suppressed:
                        outputs = _active_generator_output_mapping(
                            execution,
                            executor.generator_ids,
                        )
                        map_accumulator.begin_batch(
                            example_ids=batch.example_ids or (),
                            baseline_logits=logits,
                            targets=batch.targets,
                            supervised_mask=batch.targets != -100,
                            valid_mask=batch.valid_positions,
                            baseline_generator_outputs=outputs,
                        )
                        fingerprint_accumulator.begin_batch(
                            example_ids=batch.example_ids or (),
                            baseline_logits=logits,
                            targets=batch.targets,
                            supervised_mask=batch.targets != -100,
                        )
                        _progress(
                            f"trace: batch {batch_index + 1}/{len(batches)} "
                            "baseline complete"
                        )
                    elif len(suppressed) == 1:
                        ordinal = suppressed[0]
                        outputs = _active_generator_output_mapping(
                            execution,
                            executor.generator_ids,
                        )
                        generator_id = executor.generator_ids[ordinal]
                        map_accumulator.add_singleton(
                            generator_id,
                            logits,
                            outputs,
                        )
                        fingerprint_accumulator.add_muted_generator(
                            generator_id,
                            logits,
                        )
                        if ordinal == executor.layer_count - 1:
                            # Release the shadow full-vocabulary baseline before
                            # the much longer pair schedule begins.
                            fingerprint_accumulator.finish_batch()
                            _progress(
                                f"trace: batch {batch_index + 1}/"
                                f"{len(batches)} singletons 18/18"
                            )
                    else:
                        if fingerprint_accumulator.has_active_batch:
                            raise RuntimeError(
                                "shadow singleton baseline survived into pairs"
                            )
                        left, right = suppressed
                        map_accumulator.add_joint(
                            executor.generator_ids[left],
                            executor.generator_ids[right],
                            logits,
                        )
                        pair_observed += 1
                        if (
                            pair_observed == 1
                            or pair_observed % 10 == 0
                            or pair_observed == len(pair_schedule)
                        ):
                            _progress(
                                f"trace: batch {batch_index + 1}/"
                                f"{len(batches)} pairs "
                                f"{pair_observed}/{len(pair_schedule)}"
                            )

                executor.visit_generator_interaction_map(
                    batch.model_inputs,
                    visitor=visit,
                    joint_pairs=pair_schedule,
                )
                if observed != len(expected_schedule):
                    raise RuntimeError("interaction sweep was incomplete")
                map_accumulator.finish_batch()
                gc.collect()
        singleton_replay = fingerprint_accumulator.finalize()
        interaction_analysis = map_accumulator.finalize()
    finally:
        fingerprint_accumulator.close()
        map_accumulator.close()

    expected_map_ids = tuple(
        interaction_map_example_id_sha256(value)
        for value in analysis_example_ids
    )
    if (
        interaction_analysis.example_id_sha256s != expected_map_ids
        or interaction_analysis.generator_ids != executor.generator_ids
    ):
        raise RuntimeError("final interaction-map identity drifted")
    _verify_replayed_singleton_identity(
        singleton_replay,
        fingerprint,
        runtime,
        prompt_content_sha256s=analysis_content,
        example_ids=analysis_example_ids,
    )
    _progress("identity: exact singleton fingerprint replay confirmed")

    nodes = _generator_nodes(
        fingerprint,
        generator_ids=interaction_analysis.generator_ids,
    )
    copied_pairs = _copied_pairwise_similarities(fingerprint)
    joints = _joint_interactions(
        interaction_analysis,
        runtime,
        fingerprint,
        prompt_content_sha256s=analysis_content,
    )
    directed = _directed_edges(
        interaction_analysis,
        runtime,
        fingerprint,
        prompt_content_sha256s=analysis_content,
    )
    cohorts = _prompt_cohorts(
        assessment,
        prompt_content_sha256s=analysis_content,
    )
    cohort_partition_lineage = {
        "partition_plan_sha256": partition.artifact_sha256,
        "assessment_partition_sha256": assessment.artifact_sha256,
        "source_export_sha256": assessment.source_export_sha256,
        "source_fit_prompt_index_sha256": (
            assessment.source_fit_prompt_index_sha256
        ),
        "role": assessment.role,
        "assessment_status": "open_development_not_closed_guard",
        "membership_provenance": "caller_declared_self_attested",
        "membership_externally_authenticated": False,
        "serialized_contains_prompt_text": False,
        "prompt_count": assessment.prompt_count,
        "family_count": len(set(assessment.family_ids)),
        "family_id_storage": "domain_separated_sha256_only",
        "exact_declared_family_membership": True,
    }
    affinities = _generator_cohort_affinities(
        fingerprint,
        runtime,
        cohorts,
    )
    artifact_model = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_commit": revision,
        "adapter_model_fingerprint": runtime.source_model_sha256,
        "local_files_only": True,
    }
    fingerprint_file_sha256 = _file_sha256(fingerprint_artifact_path)
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
        "singleton_causal_fingerprint": {
            "schema": GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_SCHEMA,
            "format_version": (
                GEMMA3_GENERATOR_CAUSAL_FINGERPRINT_FORMAT_VERSION
            ),
            "artifact_file_sha256": fingerprint_file_sha256,
            "scientific_payload_sha256": fingerprint[
                "scientific_payload_sha256"
            ],
            "frozen_before_analysis": True,
        },
    }
    _progress(
        "artifact: publish tensor-free 18-node/153-pair observational map"
    )
    return save_gemma3_generator_causal_map_artifact(
        output,
        model=artifact_model,
        frozen_sources=frozen_sources,
        analysis_split=dict(fingerprint["analysis_split"]),
        interaction_analysis_lineage=interaction_analysis.metadata(),
        cohort_partition_lineage=cohort_partition_lineage,
        generator_fit_lineage=_generator_fit_lineage(runtime),
        generator_nodes=nodes,
        copied_pairwise_similarities=copied_pairs,
        joint_interactions=joints,
        directed_edges=directed,
        prompt_cohorts=cohorts,
        generator_cohort_affinities=affinities,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map frozen Gemma generator interactions with exhaustive exact "
            "singleton and pair suppressions."
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
    parser.add_argument(
        "--fingerprint-artifact",
        type=Path,
        default=DEFAULT_CAUSAL_FINGERPRINT_ARTIFACT,
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
    result = run_gemma3_generator_causal_map_experiment(
        eval_export_path=arguments.eval_export,
        revision=arguments.revision,
        base_artifact_path=arguments.base_artifact,
        refit_artifact_path=arguments.refit_artifact,
        fingerprint_artifact_path=arguments.fingerprint_artifact,
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
