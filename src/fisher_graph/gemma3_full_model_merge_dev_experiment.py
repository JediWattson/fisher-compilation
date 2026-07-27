"""Exploratory uncapped cross-block merging across the complete Gemma model.

The fit and evaluation partitions are both taken from calibration-A fit
development exports.  Calibration-A guard, calibration-B, validation, and
test are never opened by this runner.  All MLP coordinates are eligible, but a
recorded sparse nearest-neighbor proxy search is used instead of materializing
the complete 641-million-edge cross-layer graph.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import json
from pathlib import Path
import sys
import time

import torch
from torch import Tensor

from .adapters import Gemma3CausalLMAdapter
from .compiler.calibration import CausalLanguageModelNLL
from .gemma3_experiment import (
    DEFAULT_MODEL_ID,
    load_gemma3,
    resolve_gemma3_huggingface_paths,
)
from .gemma3_gated_executor_experiment import (
    _behavior_aggregate,
    _behavior_examples,
    _materialize_split,
)
from .gemma3_global_cross_block_merge_executor import (
    Gemma3GlobalCrossBlockMergeExecutor,
)
from .gemma3_whole_model_mode_graph_discovery import (
    _whole_model_layer_specs,
)
from .modal_ablation import (
    _causal_lm_batch_scores,
    _example_ids,
)
from .streaming_analysis import iter_activation_score_gradient_rows
from .structured_mlp_cross_block_bundling import (
    CrossBlockDiscoveryResult,
    CrossBlockDiscoverySketch,
    CrossBlockExactCriteria,
    CrossBlockSketchConfig,
    replay_cross_block_discovery_shortlist,
    rescreen_cross_block_discovery_sketch,
)
from .structured_mlp_global_cross_block_merge import (
    GlobalCrossBlockMergePlan,
    plan_global_cross_block_merges,
)


DEFAULT_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"
DEFAULT_MAX_LENGTH = 256
DEFAULT_TOKENIZATION_BATCH_SIZE = 4
DEFAULT_PROXY_NEIGHBORS = 8
DEFAULT_PROXY_MIN_CORRELATION = 0.8
_EXPORT_SCHEMA = "fisher_graph.local_v9_a_fit_development_export"
_REPORT_SCHEMA = "fisher_graph.gemma3_full_model_cross_block_merge_development"


def _progress(message: str) -> None:
    print(
        f"[gemma-full-model-merge-dev] {message}",
        file=sys.stderr,
        flush=True,
    )


def _load_export(
    path: Path,
    *,
    expected_positions: tuple[int, ...],
) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": _EXPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": "development_only",
        "source_corpus_id": "structured-strong-v9",
        "source_role": "calibration_a_fit_only",
        "guard_exported": False,
        "calibration_b_exported": False,
        "validation_exported": False,
        "test_exported": False,
        "model_or_tokenizer_accessed": False,
    }
    if not isinstance(raw, dict) or any(
        raw.get(key) != value for key, value in required.items()
    ):
        raise ValueError(f"development export contract drifted: {path}")
    prompts = raw.get("prompts")
    positions = raw.get("fit_positions")
    source_indices = raw.get("source_prompt_indices")
    families = raw.get("family_ids")
    if (
        not isinstance(prompts, list)
        or not isinstance(positions, list)
        or not isinstance(source_indices, list)
        or not isinstance(families, list)
        or tuple(positions) != expected_positions
        or not (
            len(prompts)
            == len(positions)
            == len(source_indices)
            == len(families)
            == 40
        )
        or any(not isinstance(prompt, str) for prompt in prompts)
        or any(not isinstance(family, str) for family in families)
        or len(set(source_indices)) != 40
        or len(set(families)) != 8
        or any(count != 5 for count in Counter(families).values())
    ):
        raise ValueError(f"development export shape drifted: {path}")
    return raw


def _family_fold_assignment(
    families: Sequence[str],
) -> dict[str, int]:
    unique = tuple(sorted(set(families)))
    if len(unique) != 8:
        raise ValueError("fit development export must contain eight families")
    family_to_fold = {
        family: index % 4 for index, family in enumerate(unique)
    }
    return {
        f"prompt.{index:06d}": family_to_fold[family]
        for index, family in enumerate(families)
    }


class _ProgressRows:
    def __init__(self, rows: object, total: int) -> None:
        self._rows = rows
        self._total = total

    def __iter__(self):
        for index, row in enumerate(self._rows, start=1):  # type: ignore[arg-type]
            _progress(f"exact replay sequence {index}/{self._total}")
            yield row


class _ConditionAccumulator:
    def __init__(
        self,
        *,
        family_by_example: Mapping[str, str],
    ) -> None:
        self.family_by_example = dict(family_by_example)
        self.examples: list[dict[str, object]] = []
        self.kl_sum = 0.0
        self.square_error = 0.0
        self.reference_square = 0.0
        self.candidate_square = 0.0
        self.dot = 0.0
        self.maximum_absolute_error = 0.0
        self.supervised_tokens = 0

    def update(
        self,
        *,
        batch: object,
        example_ids: tuple[str, ...],
        native_logits: Tensor,
        candidate_logits: Tensor,
        objective: CausalLanguageModelNLL,
    ) -> None:
        baseline = _causal_lm_batch_scores(
            native_logits,
            batch,  # type: ignore[arg-type]
            objective=objective,
        )
        candidate = _causal_lm_batch_scores(
            candidate_logits,
            batch,  # type: ignore[arg-type]
            objective=objective,
        )
        self.examples.extend(
            _behavior_examples(
                batch=batch,  # type: ignore[arg-type]
                example_ids=example_ids,
                baseline=baseline,
                predicted=candidate,
            )
        )
        supervised = baseline.supervised_mask.to(
            device=native_logits.device
        )
        native = native_logits.float()[supervised].double()
        predicted = candidate_logits.float()[supervised].double()
        difference = predicted - native
        self.square_error += float(difference.square().sum().item())
        self.reference_square += float(native.square().sum().item())
        self.candidate_square += float(predicted.square().sum().item())
        self.dot += float((native * predicted).sum().item())
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(difference.abs().max().item()),
        )
        log_native = native.log_softmax(dim=-1)
        log_candidate = predicted.log_softmax(dim=-1)
        probability = log_native.exp()
        self.kl_sum += float(
            (probability * (log_native - log_candidate)).sum().item()
        )
        self.supervised_tokens += int(supervised.sum().item())

    def summary(self) -> dict[str, object]:
        if not self.examples or self.supervised_tokens <= 0:
            raise ValueError("cannot summarize an empty condition")
        behavior = _behavior_aggregate(self.examples)
        denominator = max(self.reference_square, torch.finfo(torch.float64).tiny)
        cosine_denominator = (
            max(self.reference_square, 0.0)
            * max(self.candidate_square, 0.0)
        ) ** 0.5
        family_rows: dict[str, list[Mapping[str, object]]] = {}
        for row in self.examples:
            example_id = str(row["example_id"])
            try:
                family = self.family_by_example[example_id]
            except KeyError as error:
                raise ValueError(
                    "evaluation family map does not cover every example"
                ) from error
            family_rows.setdefault(family, []).append(row)
        by_family = {
            family: _behavior_aggregate(rows)
            for family, rows in sorted(family_rows.items())
        }
        return {
            "behavior": behavior,
            "teacher_kl_per_token": self.kl_sum / self.supervised_tokens,
            "final_logits": {
                "nrmse": (self.square_error / denominator) ** 0.5,
                "cosine": (
                    self.dot / cosine_denominator
                    if cosine_denominator > 0.0
                    else 0.0
                ),
                "maximum_absolute_error": self.maximum_absolute_error,
                "square_error": self.square_error,
                "reference_square": self.reference_square,
            },
            "by_family": by_family,
            "worst_family_top1_agreement": min(
                float(row["top1_agreement_to_baseline"])
                for row in by_family.values()
            ),
            "worst_family_absolute_delta_nll_per_token": max(
                abs(float(row["delta_nll_per_token"]))
                for row in by_family.values()
            ),
        }


def _run_evaluation(
    adapter: Gemma3CausalLMAdapter,
    executor: Gemma3GlobalCrossBlockMergeExecutor,
    batches: Sequence[object],
    *,
    families: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    example_ids = tuple(
        f"prompt.{index:06d}" for index in range(len(families))
    )
    family_by_example = dict(zip(example_ids, families, strict=True))
    accumulators = {
        condition: _ConditionAccumulator(
            family_by_example=family_by_example
        )
        for condition in ("merged", "deletion")
    }
    objective = CausalLanguageModelNLL()
    sequence_offset = 0
    resource: dict[str, object] | None = None
    with torch.no_grad():
        for batch_index, batch in enumerate(batches, start=1):
            _progress(f"evaluation batch {batch_index}/{len(batches)}")
            call_inputs: dict[str, object] = dict(
                batch.model_inputs  # type: ignore[attr-defined]
            )
            call_inputs["use_cache"] = False
            call_inputs["return_dict"] = True
            native_output = adapter.module(**call_inputs)
            native_logits = native_output.logits
            ids = _example_ids(
                batch,  # type: ignore[arg-type]
                sequence_offset=sequence_offset,
            )
            sequence_offset += batch.batch_size  # type: ignore[attr-defined]
            for condition in ("merged", "deletion"):
                execution = executor.run(
                    batch.model_inputs,  # type: ignore[attr-defined]
                    condition=condition,
                )
                candidate_logits = execution.model_output.logits
                accumulators[condition].update(
                    batch=batch,
                    example_ids=ids,
                    native_logits=native_logits,
                    candidate_logits=candidate_logits,
                    objective=objective,
                )
                current_resource = {
                    "merge_count": execution.merge_count,
                    "native_root_count": execution.native_root_count,
                    "affected_layer_count": execution.affected_layer_count,
                    "source_whole_model_learned_parameters": (
                        execution.source_whole_model_learned_parameters
                    ),
                    "candidate_whole_model_learned_parameters": (
                        execution.candidate_whole_model_learned_parameters
                    ),
                    "removed_learned_parameters": (
                        execution.removed_learned_parameters
                    ),
                    "fixed_scale_coefficients": (
                        executor.merge_count
                    ),
                    "net_stored_coefficient_savings": (
                        execution.removed_learned_parameters
                        - executor.merge_count
                    ),
                    "peak_live_root_scalars_per_token": (
                        execution.peak_live_root_scalars_per_token
                    ),
                    "linear_macs_removed_per_valid_token": (
                        execution.removed_learned_parameters
                    ),
                    "merge_scale_macs_per_valid_token": (
                        executor.merge_count
                    ),
                    "net_linear_macs_saved_per_valid_token": (
                        execution.removed_learned_parameters
                        - executor.merge_count
                    ),
                }
                if resource is None:
                    resource = current_resource
                elif resource != current_resource:
                    raise RuntimeError(
                        "resource accounting drifted across conditions/batches"
                    )
                del candidate_logits, execution
            del native_logits, native_output
    if resource is None:
        raise RuntimeError("development evaluation produced no resource report")
    return (
        {
            condition: accumulator.summary()
            for condition, accumulator in accumulators.items()
        },
        resource,
    )


def run_gemma3_full_model_merge_development(
    *,
    source_discovery_path: Path | str,
    fit_export_path: Path | str,
    evaluation_export_path: Path | str,
    output_path: Path | str,
    plan_output_path: Path | str,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    proxy_neighbors: int = DEFAULT_PROXY_NEIGHBORS,
    proxy_min_correlation: float = DEFAULT_PROXY_MIN_CORRELATION,
) -> dict[str, object]:
    source_path = Path(source_discovery_path)
    fit_path = Path(fit_export_path)
    evaluation_path = Path(evaluation_export_path)
    output = Path(output_path)
    plan_output = Path(plan_output_path)
    if output.exists() or plan_output.exists():
        raise FileExistsError("refusing to overwrite a development result")
    fit_export = _load_export(
        fit_path,
        expected_positions=tuple(range(40)),
    )
    evaluation_export = _load_export(
        evaluation_path,
        expected_positions=tuple(range(40, 80)),
    )
    fit_source_indices = set(fit_export["source_prompt_indices"])
    evaluation_source_indices = set(
        evaluation_export["source_prompt_indices"]
    )
    if fit_source_indices & evaluation_source_indices:
        raise ValueError("fit and development-evaluation prompts overlap")
    source_artifact = torch.load(
        source_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(source_artifact, Mapping):
        raise ValueError("source discovery artifact is invalid")
    source_sketch_state = source_artifact.get("sketch_state")
    if not isinstance(source_sketch_state, Mapping):
        raise ValueError("source discovery sketch state is missing")
    source_sketch = CrossBlockDiscoverySketch.from_state_dict(
        source_sketch_state
    )
    layer_widths = {spec.width for spec in source_sketch.layer_specs}
    if layer_widths != {2_048} or len(source_sketch.layer_specs) != 18:
        raise ValueError("source discovery is not the full Gemma MLP stack")
    config = CrossBlockSketchConfig(
        sketch_size=source_sketch.config.sketch_size,
        sketch_seed=source_sketch.config.sketch_seed,
        per_layer_pool_size=2_048,
        neighbors_per_mode=proxy_neighbors,
        proxy_min_signed_correlation=proxy_min_correlation,
    )
    _progress("rescreen all 36,864 modes across every layer pair")
    rescreen_started = time.monotonic()
    broad_sketch = rescreen_cross_block_discovery_sketch(
        source_sketch,
        config=config,
    )
    rescreen_seconds = time.monotonic() - rescreen_started

    device = torch.device("cpu")
    cache = resolve_gemma3_huggingface_paths()["hub_cache"]
    _progress("load pinned local Gemma source")
    tokenizer, model = load_gemma3(
        model_id=model_id,
        revision=revision,
        cache_dir=cache,
        device=device,
        dtype="float32",
        local_files_only=True,
    )
    adapter = Gemma3CausalLMAdapter(model)
    if adapter.model_fingerprint() != broad_sketch.provenance.model_fingerprint:
        raise ValueError("source sketch does not bind the pinned Gemma model")
    layer_specs, leaf_site, mlp_sites = _whole_model_layer_specs(adapter)
    if layer_specs != broad_sketch.layer_specs:
        raise ValueError("source sketch layer catalog drifted")
    fit_batches, fit_stream = _materialize_split(
        tokenizer,
        tuple(fit_export["prompts"]),
        split_name="calibration_a_fit_development_first40",
        max_length=DEFAULT_MAX_LENGTH,
        tokenization_batch_size=DEFAULT_TOKENIZATION_BATCH_SIZE,
        device=device,
    )
    fold_assignment = _family_fold_assignment(
        fit_export["family_ids"]  # type: ignore[arg-type]
    )
    criteria = CrossBlockExactCriteria(
        min_row_signed_correlation=0.9,
        min_sequence_signed_correlation=0.9,
        min_energy_balance=0.1,
        min_absolute_activation_correlation=0.9,
        max_activation_rank_one_tail_fraction=0.05,
        min_coactivity=0.5,
        max_endpoint_activation_density=0.5,
        max_endpoint_influence_density=0.5,
        fold_count=4,
        min_fold_signed_correlation=0.8,
    )
    _progress(
        f"exact replay {len(broad_sketch.proxy_edges):,} proxy edges"
    )
    replay_started = time.monotonic()
    row_stream = iter_activation_score_gradient_rows(
        adapter,
        fit_batches,
        activation_names=(leaf_site, *mlp_sites),
        score_objective=CausalLanguageModelNLL(),
        leaf_activation_name=leaf_site,
    )
    try:
        discovery = replay_cross_block_discovery_shortlist(
            _ProgressRows(row_stream, 40),
            sketch=broad_sketch,
            criteria=criteria,
            fold_assignment=fold_assignment,
        )
    finally:
        row_stream.close()
    replay_seconds = time.monotonic() - replay_started
    plan = plan_global_cross_block_merges(discovery)
    classification_counts = Counter(
        evidence.classification for evidence in discovery.evidence
    )
    _progress(
        f"qualified={plan.qualified_hypothesis_count:,} "
        f"compiled={plan.merge_count:,}"
    )
    if plan.merge_count == 0:
        evaluation: dict[str, object] = {}
        resource: dict[str, object] = {
            "merge_count": 0,
            "reason": "no_exact_hypothesis_survived_the_unrestricted_search",
        }
    else:
        executor = Gemma3GlobalCrossBlockMergeExecutor(adapter, plan)
        evaluation_batches, evaluation_stream = _materialize_split(
            tokenizer,
            tuple(evaluation_export["prompts"]),
            split_name="calibration_a_fit_development_positions_40_79",
            max_length=DEFAULT_MAX_LENGTH,
            tokenization_batch_size=DEFAULT_TOKENIZATION_BATCH_SIZE,
            device=device,
        )
        evaluation, resource = _run_evaluation(
            adapter,
            executor,
            evaluation_batches,
            families=evaluation_export["family_ids"],  # type: ignore[arg-type]
        )
        resource["whole_model_parameter_reduction_fraction"] = (
            int(resource["removed_learned_parameters"])
            / int(resource["source_whole_model_learned_parameters"])
        )
        source_mlp_parameters = 18 * 3 * 640 * 2_048
        resource["source_all_mlp_learned_parameters"] = source_mlp_parameters
        resource["all_mlp_parameter_reduction_fraction"] = (
            int(resource["removed_learned_parameters"])
            / source_mlp_parameters
        )
        resource["all_mlp_linear_mac_reduction_fraction"] = (
            int(resource["net_linear_macs_saved_per_valid_token"])
            / source_mlp_parameters
        )

    report: dict[str, object] = {
        "schema": _REPORT_SCHEMA,
        "format_version": 1,
        "scientific_status": {
            "outcome": (
                "exploratory_full_model_merge_evaluated"
                if plan.merge_count
                else "exploratory_no_qualifying_full_model_merges"
            ),
            "development_only": True,
            "scientific_compression_success": False,
            "all_18_layers_executed": bool(plan.merge_count),
            "all_36_864_modes_eligible": True,
            "maximum_accepted_merges": None,
            "calibration_a_guard_opened": False,
            "calibration_b_opened": False,
            "validation_opened": False,
            "test_opened": False,
            "authorizes_calibration_b": False,
            "latency_or_kernel_speed_claim": False,
        },
        "model": {
            "model_id": model_id,
            "resolved_revision": revision,
            "model_fingerprint": adapter.model_fingerprint(),
            "execution_fingerprint": adapter.execution_fingerprint(),
            "layer_count": 18,
            "modes_per_layer": 2_048,
            "total_mode_count": 36_864,
        },
        "protocol": {
            "fit_role": "calibration_a_fit_positions_0_39",
            "evaluation_role": "calibration_a_fit_positions_40_79",
            "fit_evaluation_prompt_disjoint": True,
            "fit_evaluation_family_names_reused": True,
            "evaluation_is_development_not_guard": True,
            "source_discovery_path": source_path.name,
            "fit_export_path": fit_path.name,
            "evaluation_export_path": evaluation_path.name,
            "fit_tokenized_stream": fit_stream,
            "proxy_search": {
                **config.metadata(),
                "all_modes_eligible": True,
                "all_layer_pairs_eligible": True,
                "search_approximation": (
                    "top_k_proxy_neighbors_per_mode_after_all_mode_pool"
                ),
                "proxy_recall_known": False,
                "merge_count_quota": None,
            },
            "exact_criteria": criteria.metadata(),
            "selection": {
                "consumer_indegree_maximum": 1,
                "anchor_fanout_limit": None,
                "removed_consumer_may_anchor": False,
                "accepted_merge_limit": None,
                "scale": "unweighted_activation_least_squares_no_intercept",
            },
            "calibration_a_guard_tokenized": False,
            "calibration_a_guard_evaluated": False,
            "calibration_b_tokenized": False,
            "calibration_b_evaluated": False,
            "validation_tokenized": False,
            "validation_evaluated": False,
            "test_tokenized": False,
            "test_evaluated": False,
        },
        "discovery": {
            "source_sketch_artifact_sha256": (
                source_sketch.artifact_sha256
            ),
            "rescreened_sketch_artifact_sha256": (
                broad_sketch.artifact_sha256
            ),
            "eligible_mode_count": len(broad_sketch.pool_mode_keys),
            "proxy_edge_count": len(broad_sketch.proxy_edges),
            "exact_evidence_count": len(discovery.evidence),
            "classification_counts": dict(
                sorted(classification_counts.items())
            ),
            "qualified_hypothesis_count": (
                plan.qualified_hypothesis_count
            ),
            "compiled_merge_count": plan.merge_count,
            "rescreen_elapsed_seconds": rescreen_seconds,
            "exact_replay_elapsed_seconds": replay_seconds,
        },
        "plan": plan.metadata(),
        "evaluation": evaluation,
        "resource": resource,
        "artifact": {
            "contains_source_model_weights": False,
            "contains_candidate_model_weights": False,
            "contains_prompt_text": False,
            "plan_tensor_file": plan_output.name,
            "json_report": output.name,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(plan.state_dict(), plan_output)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return copy.deepcopy(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exploratory uncapped full-model Gemma merging."
    )
    parser.add_argument(
        "--source-discovery",
        type=Path,
        default=Path(
            ".local-runs/google--gemma-3-270m/"
            "dev-whole-model-cross-block-v3.pt"
        ),
    )
    parser.add_argument(
        "--fit-export",
        type=Path,
        default=Path(
            ".local-runs/google--gemma-3-270m/"
            "dev-v9-a-fit-first40-export.json"
        ),
    )
    parser.add_argument(
        "--evaluation-export",
        type=Path,
        default=Path(
            ".local-runs/google--gemma-3-270m/"
            "dev-v9-a-fit-40-79-export.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".local-runs/google--gemma-3-270m/"
            "full-model-unbounded-cross-block-merge-dev-v1.json"
        ),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path(
            ".local-runs/google--gemma-3-270m/"
            "full-model-unbounded-cross-block-merge-dev-v1.plan.pt"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--proxy-neighbors",
        type=int,
        default=DEFAULT_PROXY_NEIGHBORS,
    )
    parser.add_argument(
        "--proxy-min-correlation",
        type=float,
        default=DEFAULT_PROXY_MIN_CORRELATION,
    )
    args = parser.parse_args()
    report = run_gemma3_full_model_merge_development(
        source_discovery_path=args.source_discovery,
        fit_export_path=args.fit_export,
        evaluation_export_path=args.evaluation_export,
        output_path=args.output,
        plan_output_path=args.plan_output,
        model_id=args.model,
        revision=args.revision,
        proxy_neighbors=args.proxy_neighbors,
        proxy_min_correlation=args.proxy_min_correlation,
    )
    print(
        json.dumps(
            {
                "outcome": report["scientific_status"]["outcome"],  # type: ignore[index]
                "eligible_modes": report["discovery"]["eligible_mode_count"],  # type: ignore[index]
                "proxy_edges": report["discovery"]["proxy_edge_count"],  # type: ignore[index]
                "qualified": report["discovery"][  # type: ignore[index]
                    "qualified_hypothesis_count"
                ],
                "compiled_merges": report["discovery"][  # type: ignore[index]
                    "compiled_merge_count"
                ],
                "output": str(args.output),
                "plan": str(args.plan_output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
