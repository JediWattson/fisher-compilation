"""Clean expanded-task replication of the source-free full-span executor.

The first static full-span experiment established the mechanism, but an
earlier interactive probe overlapped its nominal calibration panel.  This V2
protocol is a clean confirmation:

* train a new 10-key by 10-value source model;
* exclude every semantic mapping in the original 8-by-8 task;
* estimate a new answer-row Fisher basis on a dedicated role;
* select rank and seed on development-only roles across five seeds;
* open calibration only after rank, seed, checkpoint, scale, and decoder freeze;
* evaluate fresh-only validation once only after calibration passes;
* retain fresh-only test mappings as hashes, never executor evaluations.

The compiled executor receives token-plus-position embeddings and returns the
demanded answer row without calling any source transformer block.  Logical MAC
accounting remains an ideal sparse-work statement: the PyTorch reference
executor uses dense prefix kernels and makes no latency claim.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .adapters import module_state_fingerprint
from .modes import FisherModeBasis, decompose_fisher_modes
from .model import ToyTransformer
from .static_transformer_span_executor import StaticTransformerSpanExecutor
from .variable_associative import VariableAssociativeRecallSplit
from .variable_associative_training import (
    load_variable_associative_checkpoint,
    variable_associative_answer_logits,
    variable_associative_metrics_from_logits,
)
from .variable_conditional_experiment import (
    _behavior_record,
    _file_sha256,
    _jsonable,
)
from .variable_full_span_experiment import (
    _answer_rows,
    _basis_samples,
    _collect_grids,
    _compact_behavior,
    _projected_answer_logits,
)
from .variable_static_full_span_experiment import (
    GraphBoundaryCorpus,
    NativeBoundaryCorpus,
    StaticGraphCandidate,
    StaticGraphTrainingConfig,
    _bootstrap_nll_degradation,
    _collect_native_boundaries,
    _compute_accounting,
    _coordinate_scale,
    _direct_replacement_answer_logits,
    _future_invariance_audit,
    _make_executor,
    _no_op_answer_logits,
    _sequence_context,
    _span_membership_audit,
    _standardize_corpus,
    _student_outputs,
    verify_variable_static_full_span_artifacts,
)
from .variable_static_full_span_v2_protocol import (
    V2_ROLE_NAMES,
    V2_ROLE_SALT,
    V2_ROLE_SIZES,
    V2_TASK_CONFIG,
    build_variable_static_full_span_v2_protocol,
    variable_static_full_span_v2_novelty_accuracy,
)


DEFAULT_CHECKPOINT = Path(
    ".local-runs/variable-associative-v2/checkpoint.pt"
)
DEFAULT_HYPOTHESIS_ARTIFACT = Path(
    ".local-runs/variable-associative/static-transformer-full-span.pt"
)
DEFAULT_OUTPUT = Path(
    ".local-runs/variable-associative-v2/static-transformer-full-span-v2.pt"
)
DEFAULT_INPUT_SITE = "layer.0.input"
DEFAULT_OUTPUT_SITE = "layer.2.output"
DEFAULT_RETAINED_RANKS = (24,)
DEFAULT_PROJECTION_LADDER_RANKS = (14, 18, 24)
DEFAULT_CANDIDATE = StaticGraphCandidate("l3_h24", 24, 3, 4, 48)
DEFAULT_SEEDS = (129_101, 129_102, 129_103, 129_104, 129_105)
DEFAULT_REQUIRED_STRONG_SEEDS = 4
DEFAULT_EMA_DECAY = 0.995
DEFAULT_BOOTSTRAP_SEED = 129_106
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_MAXIMUM_RELATIVE_WORK = 0.90
DEFAULT_MAXIMUM_RELATIVE_STORAGE = 0.90
DEFAULT_TRAINING = StaticGraphTrainingConfig(
    max_steps=3_200,
    modal_mse_weight=0.05,
    cross_entropy_weight=0.25,
    teacher_kl_weight=4.0,
    label_smoothing=0.0,
)

_SCHEMA = "fisher_graph.variable_static_transformer_full_span.v2"
_FORMAT_VERSION = 1
_PROTECTED_ACCESS_SCHEMA = (
    "fisher_graph.variable_static_transformer_full_span.v2.protected_access"
)


def _validate_frozen_v2_recipe(
    *,
    retained_ranks: Sequence[int],
    candidate: StaticGraphCandidate,
    seeds: Sequence[int],
    required_strong_seeds: int,
    training: StaticGraphTrainingConfig,
    ema_decay: float,
    maximum_relative_work: float,
    maximum_relative_storage: float,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> None:
    """Reject any relaxation of the confirmatory V2 recipe."""

    checks = (
        (
            tuple(retained_ranks) == DEFAULT_RETAINED_RANKS,
            "retained ranks",
        ),
        (
            candidate == DEFAULT_CANDIDATE,
            "executor candidate",
        ),
        (
            tuple(seeds) == DEFAULT_SEEDS,
            "seed panel",
        ),
        (
            required_strong_seeds == DEFAULT_REQUIRED_STRONG_SEEDS,
            "required strong seeds",
        ),
        (
            training == DEFAULT_TRAINING,
            "training recipe",
        ),
        (
            ema_decay == DEFAULT_EMA_DECAY,
            "EMA decay",
        ),
        (
            bootstrap_seed == DEFAULT_BOOTSTRAP_SEED,
            "bootstrap seed",
        ),
        (
            bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES,
            "bootstrap sample count",
        ),
        (
            maximum_relative_work == DEFAULT_MAXIMUM_RELATIVE_WORK,
            "relative-work gate",
        ),
        (
            maximum_relative_storage
            == DEFAULT_MAXIMUM_RELATIVE_STORAGE,
            "relative-storage gate",
        ),
    )
    for passed, name in checks:
        if not passed:
            raise ValueError(f"V2 {name} is frozen by the protocol")


def _v2_strong_behavior_gates(
    record: Mapping[str, object],
    novelty: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, bool]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("behavior metrics must be a mapping")
    minimum_stratum = min(
        float(metrics["minimum_query_accuracy"]),
        float(metrics["minimum_pair_order_accuracy"]),
        float(metrics["minimum_layout_accuracy"]),
        float(metrics["minimum_length_accuracy"]),
    )
    def novelty_exact(name: str) -> bool:
        item = novelty.get(name)
        return (
            isinstance(item, Mapping)
            and int(item["contexts"]) > 0
            and float(item["accuracy"]) == 1.0
        )

    queried_key = _required_mapping(
        identity.get("queried_key"),
        name="queried-key identity accuracy",
    )
    answer_value = _required_mapping(
        identity.get("answer_value"),
        name="answer-value identity accuracy",
    )
    return {
        "delta_nll_at_most_0.007": float(record["delta_nll"]) <= 0.007,
        "answer_accuracy_exact": float(metrics["answer_accuracy"]) == 1.0,
        "paired_context_accuracy_exact": (
            float(metrics["paired_context_accuracy"]) == 1.0
        ),
        "minimum_layout_query_order_length_accuracy_exact": (
            minimum_stratum == 1.0
        ),
        "top1_agreement_exact": float(record["top1_agreement"]) == 1.0,
        "native_teacher_kl_at_most_0.007": (
            float(record["native_teacher_kl"]) <= 0.007
        ),
        "p90_absolute_delta_nll_at_most_0.020": (
            float(record["p90_absolute_delta_nll"]) <= 0.020
        ),
        "new_key_accuracy_exact": novelty_exact("new_key"),
        "new_value_accuracy_exact": novelty_exact("new_value"),
        "new_key_only_accuracy_exact": novelty_exact("key_only"),
        "new_value_only_accuracy_exact": novelty_exact("value_only"),
        "new_key_and_value_accuracy_exact": novelty_exact("both"),
        "minimum_queried_key_accuracy_exact": (
            float(queried_key["minimum_accuracy"]) == 1.0
        ),
        "minimum_answer_value_accuracy_exact": (
            float(answer_value["minimum_accuracy"]) == 1.0
        ),
    }


def _identity_accuracy_summary(
    logits: Tensor,
    split: VariableAssociativeRecallSplit,
) -> dict[str, object]:
    correct = logits.argmax(dim=-1).cpu().eq(split.answer_token_ids)

    def grouped(values: Tensor) -> dict[str, object]:
        records: list[dict[str, object]] = []
        for value in values.unique(sorted=True):
            mask = values == value
            samples = int(mask.sum().item())
            correct_samples = int(correct[mask].sum().item())
            records.append(
                {
                    "id": int(value.item()),
                    "samples": samples,
                    "correct_samples": correct_samples,
                    "accuracy": correct_samples / samples,
                }
            )
        return {
            "strata": records,
            "minimum_accuracy": min(
                float(record["accuracy"]) for record in records
            ),
        }

    return {
        "queried_key": grouped(split.queried_key_ids),
        "answer_value": grouped(split.answer_value_indices),
    }


def _v2_behavior_summary(
    logits: Tensor,
    baseline_logits: Tensor,
    split: VariableAssociativeRecallSplit,
) -> dict[str, object]:
    record = _behavior_record(logits, baseline_logits, split)
    novelty = asdict(
        variable_static_full_span_v2_novelty_accuracy(
            split,
            logits,
        )
    )
    identity = _identity_accuracy_summary(logits, split)
    gates = _v2_strong_behavior_gates(record, novelty, identity)
    return {
        **_compact_behavior(record),
        "minimum_viability_passed": record["passed"] is True,
        "novelty": novelty,
        "identity_accuracy": identity,
        "strong_gates": gates,
        "strong_passed": all(gates.values()),
    }


@torch.no_grad()
def _evaluate_executor_v2(
    executor: StaticTransformerSpanExecutor,
    model: ToyTransformer,
    corpus: GraphBoundaryCorpus,
    *,
    batch_size: int = 512,
) -> tuple[dict[str, object], Tensor]:
    was_training = executor.training
    executor.eval()
    coordinate_chunks: list[Tensor] = []
    logit_chunks: list[Tensor] = []
    try:
        for start in range(0, corpus.split.samples, batch_size):
            stop = min(start + batch_size, corpus.split.samples)
            rows = torch.arange(start, stop, dtype=torch.int64)
            coordinates, logits = _student_outputs(
                executor,
                model,
                corpus,
                rows,
            )
            coordinate_chunks.append(coordinates.cpu())
            logit_chunks.append(logits.cpu())
    finally:
        executor.train(was_training)
    predicted_coordinates = torch.cat(coordinate_chunks)
    logits = torch.cat(logit_chunks)
    summary = _v2_behavior_summary(
        logits,
        corpus.teacher_logits,
        corpus.split,
    )
    summary["standardized_modal_mse"] = float(
        F.mse_loss(
            predicted_coordinates,
            corpus.standardized_coordinates,
        ).item()
    )
    return summary, logits


def _checkpoint_score_v2(
    behavior: Mapping[str, object],
    *,
    step: int,
) -> tuple[int, int, float, float, float, int]:
    return (
        0 if behavior["strong_passed"] is True else 1,
        0 if behavior["minimum_viability_passed"] is True else 1,
        float(behavior["native_teacher_kl"]),
        float(behavior["p90_absolute_delta_nll"]),
        abs(float(behavior["delta_nll"])),
        -step,
    )


def _train_rank_seed(
    candidate: StaticGraphCandidate,
    *,
    retained_rank: int,
    seed: int,
    model: ToyTransformer,
    fit: GraphBoundaryCorpus,
    stop: GraphBoundaryCorpus,
    select: GraphBoundaryCorpus,
    decoder: Tensor,
    training: StaticGraphTrainingConfig,
    ema_decay: float,
    progress: Callable[[str], None] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if not math.isfinite(ema_decay) or not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    executor = _make_executor(
        candidate,
        residual_width=model.config.d_model,
        retained_rank=retained_rank,
        decoder=decoder,
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        executor.parameters(),
        lr=training.learning_rate,
        betas=training.betas,
        eps=training.epsilon,
        weight_decay=training.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000_003)
    ema_parameters = {
        name: parameter.detach().clone()
        for name, parameter in executor.named_parameters()
    }
    best_score: tuple[int, int, float, float, float, int] | None = None
    best_artifact: dict[str, object] | None = None
    best_step = 0
    history: list[dict[str, object]] = []

    for step in range(1, training.max_steps + 1):
        executor.train()
        rows = torch.randint(
            fit.split.samples,
            (training.batch_size,),
            generator=generator,
        )
        predicted_coordinates, student_logits = _student_outputs(
            executor,
            model,
            fit,
            rows,
        )
        target_coordinates = fit.standardized_coordinates.index_select(
            0,
            rows,
        )
        teacher_logits = fit.teacher_logits.index_select(0, rows)
        answer_ids = fit.split.answer_token_ids.index_select(0, rows)
        modal_loss = F.mse_loss(
            predicted_coordinates,
            target_coordinates,
        )
        cross_entropy = F.cross_entropy(
            student_logits,
            answer_ids,
            label_smoothing=training.label_smoothing,
        )
        teacher_kl = F.kl_div(
            student_logits.log_softmax(dim=-1),
            teacher_logits.softmax(dim=-1),
            reduction="batchmean",
        )
        loss = (
            training.modal_mse_weight * modal_loss
            + training.cross_entropy_weight * cross_entropy
            + training.teacher_kl_weight * teacher_kl
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            executor.parameters(),
            training.gradient_clip_norm,
        )
        optimizer.step()
        with torch.no_grad():
            for name, parameter in executor.named_parameters():
                ema_parameters[name].lerp_(
                    parameter.detach(),
                    1.0 - ema_decay,
                )

        if step % training.evaluation_interval:
            continue
        ema_executor = copy.deepcopy(executor)
        with torch.no_grad():
            for name, parameter in ema_executor.named_parameters():
                parameter.copy_(ema_parameters[name])
        ema_executor.eval()
        stop_behavior, _ = _evaluate_executor_v2(
            ema_executor,
            model,
            stop,
        )
        evaluation = {
            "step": step,
            "training_loss": float(loss.detach().item()),
            "training_modal_mse": float(modal_loss.detach().item()),
            "training_cross_entropy": float(
                cross_entropy.detach().item()
            ),
            "training_teacher_kl": float(teacher_kl.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
            "graph_stop_a": stop_behavior,
        }
        history.append(evaluation)
        score = _checkpoint_score_v2(stop_behavior, step=step)
        if best_score is None or score < best_score:
            best_score = score
            best_step = step
            best_artifact = ema_executor.artifact_state_dict()
        if progress is not None:
            progress(
                f"rank={retained_rank} seed={seed} step={step}: "
                f"stop_delta={float(stop_behavior['delta_nll']):+.6f}, "
                f"KL={float(stop_behavior['native_teacher_kl']):.6f}, "
                f"p90={float(stop_behavior['p90_absolute_delta_nll']):.6f}, "
                f"strong={stop_behavior['strong_passed']}"
            )

    if best_artifact is None or best_score is None:
        raise RuntimeError("V2 graph training produced no checkpoint")
    best_executor = StaticTransformerSpanExecutor.from_artifact_state_dict(
        best_artifact
    )
    select_behavior, _ = _evaluate_executor_v2(
        best_executor,
        model,
        select,
    )
    stop_behavior = next(
        item["graph_stop_a"]
        for item in history
        if item["step"] == best_step
    )
    return (
        {
            "candidate": asdict(candidate),
            "retained_rank": retained_rank,
            "seed": seed,
            "best_step": best_step,
            "graph_stop_a": stop_behavior,
            "graph_select_a": select_behavior,
            "history": history,
            "executor_fingerprint": best_executor.execution_fingerprint(),
            "ema_decay": ema_decay,
            "runtime_stored_coefficient_count": (
                best_executor.total_runtime_coefficient_count
            ),
        },
        best_artifact,
    )


def _select_rank_and_seed(
    records: Sequence[Mapping[str, object]],
    *,
    required_strong_seeds: int = DEFAULT_REQUIRED_STRONG_SEEDS,
    expected_seed_count: int = len(DEFAULT_SEEDS),
) -> dict[str, object]:
    """Require seed stability, then choose the cheapest passing rank."""

    if type(required_strong_seeds) is not int or required_strong_seeds <= 0:
        raise ValueError("required_strong_seeds must be positive")
    if type(expected_seed_count) is not int or expected_seed_count <= 0:
        raise ValueError("expected_seed_count must be positive")
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for record in records:
        candidate = record.get("candidate")
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate record is invalid")
        key = (str(candidate["name"]), int(record["retained_rank"]))
        groups.setdefault(key, []).append(record)

    summaries: list[dict[str, object]] = []
    for (name, rank), group in groups.items():
        if len(group) != expected_seed_count:
            raise ValueError(
                f"rank {rank} must contain exactly "
                f"{expected_seed_count} seed records"
            )
        if len({int(record["seed"]) for record in group}) != len(group):
            raise ValueError("rank seeds must be unique")
        passing = [
            record
            for record in group
            if isinstance(record.get("graph_select_a"), Mapping)
            and record["graph_select_a"]["strong_passed"] is True  # type: ignore[index]
        ]
        eligible = len(passing) >= required_strong_seeds
        seed_pool = passing if eligible else group
        ordered = sorted(
            seed_pool,
            key=lambda record: (
                float(record["graph_select_a"]["hard_nll"]),  # type: ignore[index]
                int(record["seed"]),
            ),
        )
        selected_seed = ordered[len(ordered) // 2]
        accounting = selected_seed.get("accounting")
        if not isinstance(accounting, Mapping):
            raise TypeError("rank accounting is missing")
        summaries.append(
            {
                "candidate": copy.deepcopy(dict(group[0]["candidate"])),  # type: ignore[arg-type]
                "retained_rank": rank,
                "seed_count": len(group),
                "strong_seed_count": len(passing),
                "required_strong_seeds": required_strong_seeds,
                "rank_passed": eligible,
                "selected_median_strong_seed": int(
                    selected_seed["seed"]
                ),
                "selected_seed_graph_select_a": copy.deepcopy(
                    selected_seed["graph_select_a"]
                ),
                "accounting": copy.deepcopy(dict(accounting)),
            }
        )

    passing_ranks = [
        summary for summary in summaries if summary["rank_passed"] is True
    ]
    selected: dict[str, object] | None = None
    runtime_key: tuple[str, int, int] | None = None
    if passing_ranks:
        selected = min(
            passing_ranks,
            key=lambda summary: (
                int(
                    summary["accounting"]["static_graph"]["complete_macs"]  # type: ignore[index]
                ),
                int(
                    summary["accounting"]["static_graph"][  # type: ignore[index]
                        "runtime_stored_coefficients"
                    ]
                ),
                float(
                    summary["selected_seed_graph_select_a"]["hard_nll"]  # type: ignore[index]
                ),
                int(summary["retained_rank"]),
            ),
        )
        runtime_key = (
            str(selected["candidate"]["name"]),  # type: ignore[index]
            int(selected["retained_rank"]),
            int(selected["selected_median_strong_seed"]),
        )
    return {
        "ranks": sorted(
            summaries,
            key=lambda summary: int(summary["retained_rank"]),
        ),
        "selected": copy.deepcopy(selected),
        "selected_runtime_key": runtime_key,
    }


def _slice_native_corpus(
    corpus: NativeBoundaryCorpus,
    retained_rank: int,
) -> NativeBoundaryCorpus:
    if not 1 <= retained_rank <= corpus.raw_delta_coordinates.shape[1]:
        raise ValueError("retained rank is outside the collected basis span")
    return NativeBoundaryCorpus(
        split=corpus.split,
        incoming=corpus.incoming,
        raw_delta_coordinates=corpus.raw_delta_coordinates[
            :, :retained_rank
        ],
        teacher_logits=corpus.teacher_logits,
    )


@torch.no_grad()
def _native_projection_logits_from_corpus(
    model: ToyTransformer,
    corpus: NativeBoundaryCorpus,
    basis: FisherModeBasis,
    retained_rank: int,
) -> Tensor:
    if not 1 <= retained_rank <= corpus.raw_delta_coordinates.shape[1]:
        raise ValueError("projection rank is outside the collected span")
    incoming = _answer_rows(corpus.incoming, corpus.split).to(torch.float32)
    vectors = basis.vectors[:, :retained_rank].to(torch.float32)
    hidden = (
        incoming
        + corpus.raw_delta_coordinates[:, :retained_rank].to(torch.float32)
        @ vectors.T
    )
    return model.lm_head(model.final_norm(hidden)).cpu()


def _required_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_protocol_hash_partition(
    protocol: Mapping[str, object],
) -> None:
    canonical_manifest = (
        build_variable_static_full_span_v2_protocol().manifest()
    )
    for field, expected in canonical_manifest.items():
        if field not in protocol or _jsonable(protocol[field]) != _jsonable(
            expected
        ):
            raise ValueError(
                f"V2 protocol field {field!r} is not canonical"
            )

    role_hashes = _required_mapping(
        protocol["role_context_hashes"],
        name="role hashes",
    )
    if tuple(role_hashes) != V2_ROLE_NAMES:
        raise ValueError("V2 role order is invalid")
    sets = [set(role_hashes[name]) for name in V2_ROLE_NAMES]
    reserve = set(protocol["reserve_context_hashes"])  # type: ignore[arg-type]
    validation = set(
        protocol["fresh_validation_context_hashes"]  # type: ignore[arg-type]
    )
    test = set(protocol["fresh_test_context_hashes"])  # type: ignore[arg-type]
    excluded = set(
        protocol["excluded_baseline_context_hashes"]  # type: ignore[arg-type]
    )
    all_sets = [*sets, reserve, validation, test]
    for left in range(len(all_sets)):
        for right in range(left + 1, len(all_sets)):
            if not all_sets[left].isdisjoint(all_sets[right]):
                raise ValueError("V2 semantic context roles overlap")
    if any(not role.isdisjoint(excluded) for role in all_sets):
        raise ValueError("V2 fresh roles overlap the original task")
    declared_sizes = _required_mapping(
        protocol["role_sizes"],
        name="role sizes",
    )
    if any(
        len(set(role_hashes[name])) != int(declared_sizes[name])
        for name in V2_ROLE_NAMES
    ):
        raise ValueError("V2 role hash count is invalid")

    strong_thresholds = {
        "delta_nll": 0.007,
        "answer_accuracy": 1.0,
        "paired_context_accuracy": 1.0,
        "minimum_layout_query_order_length_accuracy": 1.0,
        "top1_agreement": 1.0,
        "native_teacher_kl": 0.007,
        "p90_absolute_delta_nll": 0.020,
        "new_key_accuracy": 1.0,
        "new_value_accuracy": 1.0,
        "new_key_only_accuracy": 1.0,
        "new_value_only_accuracy": 1.0,
        "new_key_and_value_accuracy": 1.0,
        "minimum_queried_key_accuracy": 1.0,
        "minimum_answer_value_accuracy": 1.0,
    }
    expected_recipe = {
        "input_site": DEFAULT_INPUT_SITE,
        "output_site": DEFAULT_OUTPUT_SITE,
        "source_layers": 3,
        "fisher_scope": "supervised answer row at output boundary",
        "retained_rank_candidates": DEFAULT_RETAINED_RANKS,
        "candidate": asdict(DEFAULT_CANDIDATE),
        "seeds": DEFAULT_SEEDS,
        "rank_pass_rule": (
            "at least 4 of 5 graph_select_a seeds pass every strong gate"
        ),
        "training": asdict(DEFAULT_TRAINING),
        "executor_parameter_ema_decay": DEFAULT_EMA_DECAY,
        "deterministic_algorithms": True,
        "strong_behavior_thresholds": strong_thresholds,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
        "bootstrap_upper_degradation_threshold": 0.0085,
        "maximum_relative_work": DEFAULT_MAXIMUM_RELATIVE_WORK,
        "maximum_relative_storage": DEFAULT_MAXIMUM_RELATIVE_STORAGE,
    }
    for field, expected in expected_recipe.items():
        if field not in protocol or _jsonable(protocol[field]) != _jsonable(
            expected
        ):
            raise ValueError(
                f"V2 protocol recipe field {field!r} is not frozen"
            )

    expected_digests = {
        "excluded_baseline": _hash_list_digest(
            canonical_manifest[
                "excluded_baseline_context_hashes"
            ]  # type: ignore[arg-type]
        ),
        **{
            name: _hash_list_digest(hashes)  # type: ignore[arg-type]
            for name, hashes in _required_mapping(
                canonical_manifest["role_context_hashes"],
                name="canonical role hashes",
            ).items()
        },
        "reserve": _hash_list_digest(
            canonical_manifest[
                "reserve_context_hashes"
            ]  # type: ignore[arg-type]
        ),
        "fresh_validation": _hash_list_digest(
            canonical_manifest[
                "fresh_validation_context_hashes"
            ]  # type: ignore[arg-type]
        ),
        "fresh_test": _hash_list_digest(
            canonical_manifest[
                "fresh_test_context_hashes"
            ]  # type: ignore[arg-type]
        ),
    }
    if _jsonable(protocol.get("hash_set_digests")) != _jsonable(
        expected_digests
    ):
        raise ValueError("V2 protocol hash-set digests are invalid")


def _boolean_gates(
    value: object,
    *,
    name: str,
) -> dict[str, bool]:
    gates = _required_mapping(value, name=name)
    if any(type(result) is not bool for result in gates.values()):
        raise ValueError(f"{name} must contain only booleans")
    return {str(key): bool(result) for key, result in gates.items()}


def _validate_behavior_receipt(
    value: object,
    *,
    name: str,
) -> Mapping[str, object]:
    behavior = _required_mapping(value, name=name)
    gates = _boolean_gates(
        behavior.get("gates"),
        name=f"{name} minimum gates",
    )
    expected_minimum = {
        "absolute_delta_nll": (
            abs(float(behavior["delta_nll"])) <= 0.05
        ),
        "top1_agreement": float(behavior["top1_agreement"]) >= 0.95,
        "native_teacher_kl": (
            float(behavior["native_teacher_kl"]) <= 0.05
        ),
        "p90_absolute_delta_nll": (
            float(behavior["p90_absolute_delta_nll"]) <= 0.10
        ),
        # The compact receipt omits the raw p10 quantile. Exact top-1
        # behavior does, however, determine this gate.
        "p10_top1_agreement": (
            True
            if float(behavior["top1_agreement"]) == 1.0
            else gates.get("p10_top1_agreement", False)
        ),
        "answer_accuracy": float(behavior["answer_accuracy"]) >= 0.995,
        "paired_context_accuracy": (
            float(behavior["paired_context_accuracy"]) >= 0.99
        ),
        "minimum_stratum_accuracy": (
            float(behavior["minimum_stratum_accuracy"]) >= 0.99
        ),
    }
    if gates != expected_minimum:
        raise ValueError(f"{name} minimum gates are inconsistent")
    minimum_passed = all(gates.values())
    if (
        behavior.get("passed") is not minimum_passed
        or behavior.get("minimum_viability_passed") is not minimum_passed
    ):
        raise ValueError(f"{name} minimum-pass claim is inconsistent")

    novelty = _required_mapping(
        behavior.get("novelty"),
        name=f"{name} novelty",
    )
    identity = _required_mapping(
        behavior.get("identity_accuracy"),
        name=f"{name} identity accuracy",
    )

    def novelty_exact(stratum_name: str) -> bool:
        stratum = _required_mapping(
            novelty.get(stratum_name),
            name=f"{name} novelty {stratum_name}",
        )
        return (
            int(stratum["contexts"]) > 0
            and float(stratum["accuracy"]) == 1.0
        )

    queried_key = _required_mapping(
        identity.get("queried_key"),
        name=f"{name} queried-key identity",
    )
    answer_value = _required_mapping(
        identity.get("answer_value"),
        name=f"{name} answer-value identity",
    )
    expected_strong = {
        "delta_nll_at_most_0.007": (
            float(behavior["delta_nll"]) <= 0.007
        ),
        "answer_accuracy_exact": (
            float(behavior["answer_accuracy"]) == 1.0
        ),
        "paired_context_accuracy_exact": (
            float(behavior["paired_context_accuracy"]) == 1.0
        ),
        "minimum_layout_query_order_length_accuracy_exact": (
            float(behavior["minimum_stratum_accuracy"]) == 1.0
        ),
        "top1_agreement_exact": (
            float(behavior["top1_agreement"]) == 1.0
        ),
        "native_teacher_kl_at_most_0.007": (
            float(behavior["native_teacher_kl"]) <= 0.007
        ),
        "p90_absolute_delta_nll_at_most_0.020": (
            float(behavior["p90_absolute_delta_nll"]) <= 0.020
        ),
        "new_key_accuracy_exact": novelty_exact("new_key"),
        "new_value_accuracy_exact": novelty_exact("new_value"),
        "new_key_only_accuracy_exact": novelty_exact("key_only"),
        "new_value_only_accuracy_exact": novelty_exact("value_only"),
        "new_key_and_value_accuracy_exact": novelty_exact("both"),
        "minimum_queried_key_accuracy_exact": (
            float(queried_key["minimum_accuracy"]) == 1.0
        ),
        "minimum_answer_value_accuracy_exact": (
            float(answer_value["minimum_accuracy"]) == 1.0
        ),
    }
    strong_gates = _boolean_gates(
        behavior.get("strong_gates"),
        name=f"{name} strong gates",
    )
    if strong_gates != expected_strong:
        raise ValueError(f"{name} strong gates are inconsistent")
    if behavior.get("strong_passed") is not all(strong_gates.values()):
        raise ValueError(f"{name} strong-pass claim is inconsistent")
    return behavior


def _panel_passed(
    value: object,
    *,
    name: str,
) -> tuple[Mapping[str, object] | None, bool]:
    if value is None:
        return None, False
    panel = _required_mapping(value, name=name)
    gates = _boolean_gates(panel.get("gates"), name=f"{name} gates")
    passed = all(gates.values())
    if panel.get("passed") is not passed:
        raise ValueError(f"{name} pass claim is inconsistent")
    return panel, passed


def _validate_scientific_receipt(
    payload: Mapping[str, object],
    *,
    executor: StaticTransformerSpanExecutor | None,
) -> None:
    """Cross-check stored conclusions against their stored evidence."""

    analysis = _required_mapping(payload["analysis"], name="analysis")
    status = _required_mapping(
        payload["scientific_status"],
        name="scientific status",
    )
    selection = _required_mapping(
        analysis.get("selection_a"),
        name="selection",
    )
    runs = selection.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise ValueError("V2 selection runs must be a sequence")
    claims_five_seed_stability = (
        status.get("five_seed_stability_confirmation") is True
    )
    if claims_five_seed_stability and len(runs) != len(DEFAULT_SEEDS):
        raise ValueError("V2 selection must contain the frozen five runs")
    run_seeds: set[int] = set()
    strong_run_count = 0
    for index, raw_run in enumerate(runs):
        run = _required_mapping(raw_run, name=f"selection run {index}")
        if claims_five_seed_stability and (
            _jsonable(run.get("candidate"))
            != _jsonable(asdict(DEFAULT_CANDIDATE))
            or int(run["retained_rank"]) != DEFAULT_RETAINED_RANKS[0]
        ):
            raise ValueError("V2 selection run recipe is inconsistent")
        run_seeds.add(int(run["seed"]))
        raw_behavior = run.get("graph_select_a")
        if isinstance(raw_behavior, Mapping):
            behavior = (
                _validate_behavior_receipt(
                    raw_behavior,
                    name=f"selection run {index}",
                )
                if "strong_gates" in raw_behavior
                else raw_behavior
            )
            strong_run_count += int(
                behavior.get("strong_passed") is True
            )
    if claims_five_seed_stability and run_seeds != set(DEFAULT_SEEDS):
        raise ValueError("V2 selection run seeds are not canonical")

    selected_raw = selection.get("selected")
    selected = (
        None
        if selected_raw is None
        else _required_mapping(selected_raw, name="selected rank")
    )
    five_seed_stability = False
    if selected is not None:
        if (
            int(selected["retained_rank"]) != DEFAULT_RETAINED_RANKS[0]
            or _jsonable(selected.get("candidate"))
            != _jsonable(asdict(DEFAULT_CANDIDATE))
            or int(selected["selected_median_strong_seed"])
            not in DEFAULT_SEEDS
        ):
            raise ValueError("V2 selected recipe is inconsistent")
        if all(
            field in selected
            for field in (
                "seed_count",
                "strong_seed_count",
                "required_strong_seeds",
                "rank_passed",
            )
        ):
            five_seed_stability = (
                int(selected["seed_count"]) == len(DEFAULT_SEEDS)
                and int(selected["strong_seed_count"]) == strong_run_count
                and strong_run_count >= DEFAULT_REQUIRED_STRONG_SEEDS
                and int(selected["required_strong_seeds"])
                == DEFAULT_REQUIRED_STRONG_SEEDS
                and selected["rank_passed"] is True
            )
        selected_behavior = selected.get(
            "selected_seed_graph_select_a"
        )
        if isinstance(selected_behavior, Mapping):
            _validate_behavior_receipt(
                selected_behavior,
                name="selected seed behavior",
            )
    if (executor is not None) is not (selected is not None):
        raise ValueError("V2 executor and selection presence disagree")

    calibration, calibration_passed = _panel_passed(
        analysis.get("calibration_b"),
        name="calibration B",
    )
    validation, validation_passed = _panel_passed(
        analysis.get("fresh_validation"),
        name="fresh validation",
    )
    if validation is not None and not calibration_passed:
        raise ValueError("V2 validation was opened before calibration passed")

    if calibration is not None:
        replacement = _validate_behavior_receipt(
            calibration.get("replacement"),
            name="calibration replacement",
        )
        calibration_gates = _boolean_gates(
            calibration.get("gates"),
            name="calibration B gates",
        )
        bootstrap = _required_mapping(
            calibration.get("context_bootstrap"),
            name="calibration bootstrap",
        )
        accounting = _required_mapping(
            calibration.get("accounting"),
            name="calibration accounting",
        )
        deployment = _required_mapping(
            accounting.get("deployment_parameters"),
            name="deployment accounting",
        )
        expected = {
            "fresh_context_protocol": True,
            "strong_behavior": replacement["strong_passed"] is True,
            "context_bootstrap_upper_at_most_0.0085": (
                float(bootstrap["upper_95_percent"]) <= 0.0085
            ),
            "zero_source_layer_calls": tuple(
                calibration["source_layer_call_counts"]  # type: ignore[arg-type]
            )
            == (0, 0, 0),
            "source_free_executor_state": (
                executor is not None
                and executor.executor_local_source_free
                and not hasattr(executor, "source")
                and not hasattr(executor, "fallback")
                and all(
                    "source" not in name and "fallback" not in name
                    for name in executor.state_dict()
                )
            ),
            "boundary_and_direct_paths_match": (
                float(
                    calibration[
                        "direct_boundary_maximum_absolute_difference"
                    ]
                )
                == 0.0
            ),
            "artifact_fingerprint_matches": executor is not None,
            "artifact_outputs_match": (
                float(
                    calibration[
                        "artifact_reload_maximum_absolute_difference"
                    ]
                )
                == 0.0
            ),
            "reloaded_zero_source_layer_calls": tuple(
                calibration[
                    "reloaded_source_layer_call_counts"
                ]  # type: ignore[arg-type]
            )
            == (0, 0, 0),
            "future_invariance": (
                _required_mapping(
                    calibration.get("future_invariance"),
                    name="future invariance",
                ).get("passed")
                is True
            ),
            "delta_confined_to_fisher_span": (
                _required_mapping(
                    calibration.get("span_membership"),
                    name="span membership",
                ).get("passed")
                is True
            ),
            "ideal_complete_macs_at_most_90_percent_native": (
                float(
                    accounting["graph_to_native_complete_mac_ratio"]
                )
                <= DEFAULT_MAXIMUM_RELATIVE_WORK
            ),
            "stored_coefficients_at_most_90_percent_native_span": (
                float(
                    accounting[
                        "graph_to_native_source_storage_ratio"
                    ]
                )
                <= DEFAULT_MAXIMUM_RELATIVE_STORAGE
            ),
            "total_deployed_parameters_at_most_90_percent_source": (
                float(
                    deployment[
                        "compiled_to_source_total_parameter_ratio"
                    ]
                )
                <= DEFAULT_MAXIMUM_RELATIVE_STORAGE
            ),
        }
        if calibration_gates != expected:
            raise ValueError("calibration B gates are inconsistent")
        if calibration.get("passed") is not all(expected.values()):
            raise ValueError("calibration B conclusion is inconsistent")

    if validation is not None:
        replacement = _validate_behavior_receipt(
            validation.get("replacement"),
            name="validation replacement",
        )
        bootstrap = _required_mapping(
            validation.get("context_bootstrap"),
            name="validation bootstrap",
        )
        expected = {
            "strong_behavior": replacement["strong_passed"] is True,
            "context_bootstrap_upper_at_most_0.0085": (
                float(bootstrap["upper_95_percent"]) <= 0.0085
            ),
            "zero_source_layer_calls": tuple(
                validation["source_layer_call_counts"]  # type: ignore[arg-type]
            )
            == (0, 0, 0),
        }
        if _boolean_gates(
            validation.get("gates"),
            name="fresh validation gates",
        ) != expected:
            raise ValueError("fresh-validation gates are inconsistent")
        if validation.get("passed") is not all(expected.values()):
            raise ValueError("fresh-validation conclusion is inconsistent")

    rank_14_rejected = False
    if (
        "rank_14_v1_hypothesis_rejected_by_v2_development_ceiling"
        in status
    ):
        projection_ladder = analysis.get(
            "development_native_projection_ladder"
        )
        if not isinstance(projection_ladder, Sequence) or isinstance(
            projection_ladder,
            (str, bytes),
        ):
            raise ValueError("V2 projection ladder is invalid")
        rank_14 = next(
            (
                _required_mapping(item, name="rank-14 projection")
                for item in projection_ladder
                if isinstance(item, Mapping)
                and int(item.get("retained_rank", -1)) == 14
            ),
            None,
        )
        if rank_14 is None:
            raise ValueError("V2 projection ladder omits rank 14")
        rank_14_behavior = _required_mapping(
            rank_14.get("behavior"),
            name="rank-14 projection behavior",
        )
        rank_14_rejected = (
            rank_14_behavior.get("strong_passed") is False
        )
    calibration_gates = (
        {}
        if calibration is None
        else _boolean_gates(
            calibration.get("gates"),
            name="calibration B gates",
        )
    )
    expected_status = {
        "source_model_frozen": True,
        "new_expanded_source_checkpoint": True,
        "new_v2_fisher_basis": True,
        "all_original_task_semantic_contexts_excluded": True,
        "rank_14_v1_hypothesis_rejected_by_v2_development_ceiling": (
            rank_14_rejected
        ),
        "rank_24_frozen_before_calibration": True,
        "deeper_narrower_graph_frozen_before_calibration": True,
        "five_seed_stability_confirmation": five_seed_stability,
        "full_transformer_span_replaced": executor is not None,
        "source_independent_graph_fitted": executor is not None,
        "calibration_b_evaluated": calibration is not None,
        "calibration_b_passed": calibration_passed,
        "calibration_b_confirmatory": calibration_passed,
        "fresh_validation_evaluated": validation is not None,
        "fresh_validation_passed": validation_passed,
        "executor_test_evaluated": False,
        "fresh_test_contexts_hash_only": True,
        "source_training_checkpoint_contains_native_test_metrics": (
            payload["source"].get("checkpoint_native_test_metrics")  # type: ignore[union-attr]
            is not None
        ),
        "zero_source_layer_calls_in_replacement": (
            calibration_gates.get("zero_source_layer_calls") is True
        ),
        "ideal_compute_storage_and_total_parameter_reduction_achieved": all(
            calibration_gates.get(name) is True
            for name in (
                "ideal_complete_macs_at_most_90_percent_native",
                "stored_coefficients_at_most_90_percent_native_span",
                "total_deployed_parameters_at_most_90_percent_source",
            )
        ),
        "reference_kernel_wall_clock_speed_claim": False,
        "scope_is_query_sparse_associative_recall_not_general_lm": True,
        "model_level_viable": validation_passed,
    }
    unknown_status = set(status) - set(expected_status)
    if unknown_status:
        raise ValueError(
            "V2 scientific status contains unsupported fields"
        )
    if any(
        _jsonable(value) != _jsonable(expected_status[key])
        for key, value in status.items()
    ):
        raise ValueError("V2 scientific status is inconsistent")


def _validate_artifact_contract(
    payload: Mapping[str, object],
    report: Mapping[str, object],
) -> StaticTransformerSpanExecutor | None:
    report_fields = {
        "schema",
        "format_version",
        "source",
        "hypothesis",
        "protocol",
        "analysis",
        "scientific_status",
        "contains_source_model_weights",
        "contains_compiled_executor_weights",
    }
    payload_fields = report_fields | {
        "basis",
        "coordinate_scale",
        "executor",
    }
    if set(payload) != payload_fields or set(report) != report_fields:
        raise ValueError("V2 artifact fields are invalid")
    if (
        payload["schema"] != _SCHEMA
        or report["schema"] != _SCHEMA
        or payload["format_version"] != _FORMAT_VERSION
        or report["format_version"] != _FORMAT_VERSION
    ):
        raise ValueError("unsupported V2 artifact")
    if payload["contains_source_model_weights"] is not False:
        raise ValueError("V2 artifact must not contain source model weights")
    if (
        _jsonable({key: payload[key] for key in report_fields})
        != _jsonable(dict(report))
    ):
        raise ValueError("V2 artifact and JSON report disagree")

    protocol = _required_mapping(payload["protocol"], name="protocol")
    _validate_protocol_hash_partition(protocol)
    source = _required_mapping(payload["source"], name="source")
    canonical_protocol = build_variable_static_full_span_v2_protocol()
    if (
        source.get("dataset_sha256")
        != canonical_protocol.source_dataset_sha256
        or source.get("task_fingerprint") != V2_TASK_CONFIG.fingerprint
        or _jsonable(source.get("task_config"))
        != _jsonable(asdict(V2_TASK_CONFIG))
    ):
        raise ValueError("V2 source provenance is not canonical")
    basis = FisherModeBasis.from_state_dict(
        _required_mapping(payload["basis"], name="basis")
    )
    if basis.activation_name != DEFAULT_OUTPUT_SITE:
        raise ValueError("V2 Fisher basis is bound to the wrong site")
    scale = payload["coordinate_scale"]
    if (
        not isinstance(scale, Tensor)
        or scale.device.type != "cpu"
        or scale.ndim != 1
        or not scale.is_floating_point()
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError("V2 coordinate scale is invalid")

    selection = _required_mapping(
        _required_mapping(
            payload["analysis"],
            name="analysis",
        )["selection_a"],
        name="selection",
    )
    selected = selection.get("selected")
    raw_executor = payload["executor"]
    if raw_executor is None:
        if payload["contains_compiled_executor_weights"] is not False:
            raise ValueError("V2 executor flag is inconsistent")
        if selected is not None:
            raise ValueError("V2 selected executor is missing")
        _validate_scientific_receipt(payload, executor=None)
        return None
    if payload["contains_compiled_executor_weights"] is not True:
        raise ValueError("V2 executor flag is inconsistent")

    selected_mapping = _required_mapping(
        selected,
        name="selected rank",
    )
    candidate = _required_mapping(
        selected_mapping["candidate"],
        name="selected candidate",
    )
    rank = int(selected_mapping["retained_rank"])
    seed = int(selected_mapping["selected_median_strong_seed"])
    runs = selection.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise ValueError("V2 selection runs must be a sequence")
    matches = [
        _required_mapping(run, name="selection run")
        for run in runs
        if isinstance(run, Mapping)
        and isinstance(run.get("candidate"), Mapping)
        and run["candidate"].get("name") == candidate.get("name")
        and int(run.get("retained_rank", -1)) == rank
        and int(run.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError("V2 selected executor binding is ambiguous")
    selected_run = matches[0]
    executor = StaticTransformerSpanExecutor.from_artifact_state_dict(
        _required_mapping(raw_executor, name="executor")
    )
    expected_config = {
        "residual_width": basis.width,
        "hidden_width": int(candidate["hidden_width"]),
        "layer_count": int(candidate["layer_count"]),
        "head_count": int(candidate["head_count"]),
        "feed_forward_width": int(candidate["feed_forward_width"]),
        "retained_rank": rank,
    }
    if asdict(executor.config) != expected_config:
        raise ValueError("V2 executor config binding is invalid")
    if scale.numel() != rank:
        raise ValueError("V2 scale and selected rank disagree")
    if (
        executor.execution_fingerprint()
        != selected_run.get("executor_fingerprint")
    ):
        raise ValueError("V2 executor fingerprint binding is invalid")
    if executor.total_runtime_coefficient_count != int(
        selected_run["runtime_stored_coefficient_count"]
    ):
        raise ValueError("V2 executor storage binding is invalid")
    expected_decoder = (
        basis.vectors[:, :rank].to(torch.float32)
        * scale.to(torch.float32).unsqueeze(0)
    )
    try:
        torch.testing.assert_close(
            executor.decoder.cpu(),
            expected_decoder,
            rtol=0.0,
            atol=0.0,
        )
    except AssertionError as error:
        raise ValueError("V2 executor decoder binding is invalid") from error
    _validate_scientific_receipt(payload, executor=executor)
    return executor


def verify_variable_static_full_span_v2_artifacts(
    artifact: Path,
    *,
    report: Path | None = None,
) -> dict[str, object]:
    """Strict-load the source-free V2 artifact and authenticate its report."""

    artifact_path = Path(artifact)
    report_path = (
        artifact_path.with_suffix(".json")
        if report is None
        else Path(report)
    )
    payload = _required_mapping(
        torch.load(
            artifact_path,
            map_location="cpu",
            weights_only=True,
        ),
        name="artifact",
    )
    report_mapping = _required_mapping(
        json.loads(report_path.read_text(encoding="utf-8")),
        name="JSON report",
    )
    _validate_artifact_contract(payload, report_mapping)
    return copy.deepcopy(dict(report_mapping))


def _save_result(
    *,
    output: Path,
    source: dict[str, object],
    hypothesis: dict[str, object],
    protocol: dict[str, object],
    basis: FisherModeBasis,
    coordinate_scale: Tensor,
    executor_artifact: dict[str, object] | None,
    analysis: dict[str, object],
    scientific_status: dict[str, object],
) -> dict[str, object]:
    contains_executor = executor_artifact is not None
    common = {
        "schema": _SCHEMA,
        "format_version": _FORMAT_VERSION,
        "source": source,
        "hypothesis": hypothesis,
        "protocol": protocol,
        "analysis": analysis,
        "scientific_status": scientific_status,
        "contains_source_model_weights": False,
        "contains_compiled_executor_weights": contains_executor,
    }
    payload = {
        **common,
        "basis": basis.state_dict(),
        "coordinate_scale": coordinate_scale.detach().cpu().clone(),
        "executor": copy.deepcopy(executor_artifact),
    }
    report = copy.deepcopy(common)
    destination = Path(output)
    report_destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or report_destination.exists():
        raise FileExistsError(
            "refusing to overwrite an existing V2 artifact or report"
        )
    with destination.open("xb") as artifact_stream:
        torch.save(payload, artifact_stream)
    with report_destination.open("x", encoding="utf-8") as report_stream:
        json.dump(
            _jsonable(report),
            report_stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        report_stream.write("\n")
    verify_variable_static_full_span_v2_artifacts(destination)
    return report


def _deployment_parameter_accounting(
    model: ToyTransformer,
    executor: StaticTransformerSpanExecutor,
) -> dict[str, object]:
    """Count the shared shell plus either source blocks or graph runtime."""

    source_total = sum(parameter.numel() for parameter in model.parameters())
    source_span = sum(
        parameter.numel()
        for layer in model.layers
        for parameter in layer.parameters()
    )
    shared_shell = source_total - source_span
    compiled_total = (
        shared_shell + executor.total_runtime_coefficient_count
    )
    return {
        "source_model_parameters": source_total,
        "source_transformer_span_parameters": source_span,
        "shared_embedding_position_norm_head_parameters": shared_shell,
        "compiled_executor_runtime_coefficients": (
            executor.total_runtime_coefficient_count
        ),
        "compiled_total_deployed_parameters": compiled_total,
        "compiled_to_source_total_parameter_ratio": (
            compiled_total / source_total
        ),
    }


def _hash_list_digest(values: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(values),
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _protected_access_receipt_path(
    checkpoint: Path,
    *,
    panel: str,
) -> Path:
    if panel not in {"calibration_b", "fresh_validation"}:
        raise ValueError("unsupported protected V2 panel")
    checkpoint_path = Path(checkpoint)
    return checkpoint_path.parent / (
        f".{checkpoint_path.stem}.static-full-span-v2."
        f"{panel}.access.json"
    )


def _claim_protected_panel(
    checkpoint: Path,
    *,
    panel: str,
    context_hashes: Sequence[str],
    protocol_manifest: Mapping[str, object],
) -> Path:
    """Atomically consume one protected panel for this source checkpoint."""

    destination = _protected_access_receipt_path(
        checkpoint,
        panel=panel,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": _PROTECTED_ACCESS_SCHEMA,
        "format_version": 1,
        "panel": panel,
        "policy": (
            "exclusive fail-closed receipt created before panel tensors "
            "or source logits are materialized"
        ),
        "source_checkpoint_sha256": _file_sha256(Path(checkpoint)),
        "protocol_manifest_sha256": _json_sha256(protocol_manifest),
        "context_set_sha256": _hash_list_digest(context_hashes),
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(
            receipt,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return destination


def run_variable_static_full_span_v2_experiment(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    hypothesis_artifact: Path = DEFAULT_HYPOTHESIS_ARTIFACT,
    output: Path = DEFAULT_OUTPUT,
    retained_ranks: Sequence[int] = DEFAULT_RETAINED_RANKS,
    candidate: StaticGraphCandidate = DEFAULT_CANDIDATE,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    required_strong_seeds: int = DEFAULT_REQUIRED_STRONG_SEEDS,
    training: StaticGraphTrainingConfig = DEFAULT_TRAINING,
    ema_decay: float = DEFAULT_EMA_DECAY,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    maximum_relative_work: float = DEFAULT_MAXIMUM_RELATIVE_WORK,
    maximum_relative_storage: float = DEFAULT_MAXIMUM_RELATIVE_STORAGE,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Run the clean V2 selection, calibration, and gated validation."""

    ranks = tuple(retained_ranks)
    seed_values = tuple(seeds)
    _validate_frozen_v2_recipe(
        retained_ranks=ranks,
        candidate=candidate,
        seeds=seed_values,
        required_strong_seeds=required_strong_seeds,
        training=training,
        ema_decay=ema_decay,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        maximum_relative_work=maximum_relative_work,
        maximum_relative_storage=maximum_relative_storage,
    )

    checkpoint_path = Path(checkpoint)
    hypothesis_path = Path(hypothesis_artifact)
    output_path = Path(output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not hypothesis_path.is_file():
        raise FileNotFoundError(hypothesis_path)
    conventional_output = checkpoint_path.parent / DEFAULT_OUTPUT.name
    protected_paths = tuple(
        dict.fromkeys(
            (
                output_path,
                output_path.with_suffix(".json"),
                conventional_output,
                conventional_output.with_suffix(".json"),
                _protected_access_receipt_path(
                    checkpoint_path,
                    panel="calibration_b",
                ),
                _protected_access_receipt_path(
                    checkpoint_path,
                    panel="fresh_validation",
                ),
            )
        )
    )
    existing = [path for path in protected_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "V2 output or protected-panel receipt already exists: "
            + ", ".join(str(path) for path in existing)
        )

    torch.use_deterministic_algorithms(True)
    hypothesis_report = verify_variable_static_full_span_artifacts(
        hypothesis_path
    )
    clean_protocol = build_variable_static_full_span_v2_protocol()
    if not clean_protocol.audit.all_overlap_checks_pass:
        raise RuntimeError("V2 protocol overlap audit failed")
    protocol_manifest = clean_protocol.manifest()

    model, splits, checkpoint_metadata = (
        load_variable_associative_checkpoint(checkpoint_path)
    )
    model.eval()
    if (
        splits.task_config.fingerprint != V2_TASK_CONFIG.fingerprint
        or splits.dataset_sha256 != clean_protocol.source_dataset_sha256
    ):
        raise ValueError("source checkpoint does not match the frozen V2 task")
    if checkpoint_metadata.get("converged") is not True:
        raise ValueError("source checkpoint did not meet its convergence gate")
    checkpoint_validation = _required_mapping(
        checkpoint_metadata["validation_metrics"],
        name="source validation metrics",
    )
    if (
        float(checkpoint_validation["answer_accuracy"]) != 1.0
        or float(checkpoint_validation["paired_context_accuracy"]) != 1.0
    ):
        raise ValueError("source checkpoint is not exact on source validation")
    if max(ranks) > model.config.d_model:
        raise ValueError("retained rank exceeds source residual width")
    output_site = f"layer.{model.config.n_layers - 1}.output"
    if output_site != DEFAULT_OUTPUT_SITE:
        raise ValueError("V2 protocol expects a three-layer source model")

    roles = clean_protocol.roles
    if progress is not None:
        progress(
            "estimating the new V2 Fisher basis on basis_fit_a; "
            "calibration_b remains unopened"
        )
    basis_grid = _collect_grids(
        model,
        roles["basis_fit_a"],
        (output_site,),
    )
    basis = decompose_fisher_modes(
        _basis_samples(
            roles["basis_fit_a"],
            basis_grid[output_site][0],
            basis_grid[output_site][1],
            activation_name=output_site,
        )
    )
    if basis.width != model.config.d_model:
        raise RuntimeError("V2 Fisher width does not match the source")
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    maximum_rank = max(max(ranks), max(DEFAULT_PROJECTION_LADDER_RANKS))
    maximum_vectors = basis.vectors[:, :maximum_rank].to(torch.float32)

    if progress is not None:
        progress(
            "collecting development-only source boundaries for graph fitting, "
            "checkpoint stopping, and rank selection"
        )
    fit_native_full = _collect_native_boundaries(
        model,
        roles["graph_fit_a"],
        basis_vectors=maximum_vectors,
        input_site=DEFAULT_INPUT_SITE,
        output_site=output_site,
    )
    stop_native_full = _collect_native_boundaries(
        model,
        roles["graph_stop_a"],
        basis_vectors=maximum_vectors,
        input_site=DEFAULT_INPUT_SITE,
        output_site=output_site,
    )
    select_native_full = _collect_native_boundaries(
        model,
        roles["graph_select_a"],
        basis_vectors=maximum_vectors,
        input_site=DEFAULT_INPUT_SITE,
        output_site=output_site,
    )
    full_scale = _coordinate_scale(fit_native_full)
    development_projection_ladder = [
        {
            "retained_rank": rank,
            "behavior": _v2_behavior_summary(
                _native_projection_logits_from_corpus(
                    model,
                    select_native_full,
                    basis,
                    rank,
                ),
                select_native_full.teacher_logits,
                roles["graph_select_a"],
            ),
        }
        for rank in DEFAULT_PROJECTION_LADDER_RANKS
    ]
    if any(
        next(
            item["behavior"]["strong_passed"]  # type: ignore[index]
            for item in development_projection_ladder
            if item["retained_rank"] == rank
        )
        is not True
        for rank in ranks
    ):
        raise ValueError(
            "a requested training rank fails its exact native projection "
            "ceiling on graph_select_a"
        )

    run_records: list[dict[str, object]] = []
    run_artifacts: dict[
        tuple[str, int, int],
        dict[str, object],
    ] = {}
    rank_scales: dict[int, Tensor] = {}
    for rank in ranks:
        scale = full_scale[:rank].clone()
        rank_scales[rank] = scale
        fit = _standardize_corpus(
            _slice_native_corpus(fit_native_full, rank),
            scale,
        )
        stop = _standardize_corpus(
            _slice_native_corpus(stop_native_full, rank),
            scale,
        )
        select = _standardize_corpus(
            _slice_native_corpus(select_native_full, rank),
            scale,
        )
        decoder = (
            basis.vectors[:, :rank].to(torch.float32)
            * scale.unsqueeze(0)
        )
        for seed in seed_values:
            record, artifact = _train_rank_seed(
                candidate,
                retained_rank=rank,
                seed=seed,
                model=model,
                fit=fit,
                stop=stop,
                select=select,
                decoder=decoder,
                training=training,
                ema_decay=ema_decay,
                progress=progress,
            )
            executor = (
                StaticTransformerSpanExecutor.from_artifact_state_dict(
                    artifact
                )
            )
            accounting = _compute_accounting(
                model,
                roles["graph_select_a"],
                executor,
            )
            accounting["deployment_parameters"] = (
                _deployment_parameter_accounting(model, executor)
            )
            record["accounting"] = accounting
            run_records.append(record)
            run_artifacts[(candidate.name, rank, seed)] = artifact

    selection = _select_rank_and_seed(
        run_records,
        required_strong_seeds=required_strong_seeds,
    )
    selected_key = selection.pop("selected_runtime_key")
    selected_artifact: dict[str, object] | None = None
    selected_executor: StaticTransformerSpanExecutor | None = None
    selected_scale = torch.ones(1, dtype=torch.float32)
    calibration: dict[str, object] | None = None
    validation: dict[str, object] | None = None

    if selected_key is not None:
        selected_artifact = run_artifacts[selected_key]
        selected_executor = (
            StaticTransformerSpanExecutor.from_artifact_state_dict(
                selected_artifact
            )
        )
        selected_rank = selected_key[1]
        selected_scale = rank_scales[selected_rank]
        basis_vectors = basis.vectors[:, :selected_rank].to(torch.float32)
        if progress is not None:
            progress(
                f"selection froze rank={selected_rank}, "
                f"seed={selected_key[2]}; opening clean calibration_b"
            )
        _claim_protected_panel(
            checkpoint_path,
            panel="calibration_b",
            context_hashes=roles[
                "calibration_b"
            ].semantic_context_hashes,
            protocol_manifest=protocol_manifest,
        )
        calibration_native = _collect_native_boundaries(
            model,
            roles["calibration_b"],
            basis_vectors=basis_vectors,
            input_site=DEFAULT_INPUT_SITE,
            output_site=output_site,
        )
        calibration_corpus = _standardize_corpus(
            calibration_native,
            selected_scale,
        )
        boundary_behavior, boundary_logits = _evaluate_executor_v2(
            selected_executor,
            model,
            calibration_corpus,
        )
        direct_logits, source_layer_calls = (
            _direct_replacement_answer_logits(
                model,
                selected_executor,
                roles["calibration_b"],
            )
        )
        direct_behavior = _v2_behavior_summary(
            direct_logits,
            calibration_native.teacher_logits,
            roles["calibration_b"],
        )
        direct_boundary_maximum = float(
            (direct_logits - boundary_logits).abs().max().item()
        )
        static_mask = torch.zeros(
            model.config.d_model,
            dtype=torch.bool,
        )
        static_mask[:selected_rank] = True
        oracle_logits = _projected_answer_logits(
            model,
            roles["calibration_b"],
            basis,
            input_site=DEFAULT_INPUT_SITE,
            output_site=output_site,
            static_mask=static_mask,
        )
        oracle_behavior = _v2_behavior_summary(
            oracle_logits,
            calibration_native.teacher_logits,
            roles["calibration_b"],
        )
        no_op_logits = _no_op_answer_logits(model, calibration_native)
        no_op_behavior = _v2_behavior_summary(
            no_op_logits,
            calibration_native.teacher_logits,
            roles["calibration_b"],
        )
        bootstrap = _bootstrap_nll_degradation(
            direct_logits,
            calibration_native.teacher_logits,
            roles["calibration_b"],
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )
        accounting = _compute_accounting(
            model,
            roles["calibration_b"],
            selected_executor,
        )
        deployment = _deployment_parameter_accounting(
            model,
            selected_executor,
        )
        accounting["deployment_parameters"] = deployment
        reloaded = StaticTransformerSpanExecutor.from_artifact_state_dict(
            selected_artifact
        )
        reloaded_logits, reloaded_source_calls = (
            _direct_replacement_answer_logits(
                model,
                reloaded,
                roles["calibration_b"],
            )
        )
        reload_maximum = float(
            (reloaded_logits - direct_logits).abs().max().item()
        )
        future_invariance = _future_invariance_audit(
            selected_executor,
            calibration_corpus,
        )
        span_membership = _span_membership_audit(
            selected_executor,
            calibration_corpus,
            basis_vectors,
        )
        source_free_state = (
            selected_executor.executor_local_source_free
            and not hasattr(selected_executor, "source")
            and not hasattr(selected_executor, "fallback")
            and all(
                "source" not in name and "fallback" not in name
                for name in selected_executor.state_dict()
            )
        )
        calibration_gates = {
            "fresh_context_protocol": (
                clean_protocol.audit.all_overlap_checks_pass
            ),
            "strong_behavior": direct_behavior["strong_passed"] is True,
            "context_bootstrap_upper_at_most_0.0085": (
                float(bootstrap["upper_95_percent"]) <= 0.0085
            ),
            "zero_source_layer_calls": (
                source_layer_calls == (0,) * model.config.n_layers
            ),
            "source_free_executor_state": source_free_state,
            "boundary_and_direct_paths_match": (
                direct_boundary_maximum == 0.0
            ),
            "artifact_fingerprint_matches": (
                reloaded.execution_fingerprint()
                == selected_executor.execution_fingerprint()
            ),
            "artifact_outputs_match": reload_maximum == 0.0,
            "reloaded_zero_source_layer_calls": (
                reloaded_source_calls == (0,) * model.config.n_layers
            ),
            "future_invariance": future_invariance["passed"] is True,
            "delta_confined_to_fisher_span": (
                span_membership["passed"] is True
            ),
            "ideal_complete_macs_at_most_90_percent_native": (
                float(accounting["graph_to_native_complete_mac_ratio"])
                <= maximum_relative_work
            ),
            "stored_coefficients_at_most_90_percent_native_span": (
                float(
                    accounting["graph_to_native_source_storage_ratio"]
                )
                <= maximum_relative_storage
            ),
            "total_deployed_parameters_at_most_90_percent_source": (
                float(
                    deployment[
                        "compiled_to_source_total_parameter_ratio"
                    ]
                )
                <= maximum_relative_storage
            ),
        }
        calibration = {
            "baseline": asdict(
                variable_associative_metrics_from_logits(
                    roles["calibration_b"],
                    calibration_native.teacher_logits,
                )
            ),
            "replacement": direct_behavior,
            "boundary_replay": boundary_behavior,
            "native_output_projection_oracle": oracle_behavior,
            "no_op_source_span_control": no_op_behavior,
            "context_bootstrap": bootstrap,
            "accounting": accounting,
            "source_layer_call_counts": source_layer_calls,
            "reloaded_source_layer_call_counts": reloaded_source_calls,
            "direct_boundary_maximum_absolute_difference": (
                direct_boundary_maximum
            ),
            "artifact_reload_maximum_absolute_difference": reload_maximum,
            "future_invariance": future_invariance,
            "span_membership": span_membership,
            "gates": calibration_gates,
            "passed": all(calibration_gates.values()),
        }

        if calibration["passed"] is True:
            if progress is not None:
                progress(
                    "clean calibration_b passed; evaluating fresh-only "
                    "validation exactly once"
                )
            validation_split = clean_protocol.fresh_validation
            _claim_protected_panel(
                checkpoint_path,
                panel="fresh_validation",
                context_hashes=validation_split.semantic_context_hashes,
                protocol_manifest=protocol_manifest,
            )
            native_validation_logits = variable_associative_answer_logits(
                model,
                validation_split,
            )
            validation_logits, validation_source_calls = (
                _direct_replacement_answer_logits(
                    model,
                    selected_executor,
                    validation_split,
                )
            )
            validation_behavior = _v2_behavior_summary(
                validation_logits,
                native_validation_logits,
                validation_split,
            )
            validation_bootstrap = _bootstrap_nll_degradation(
                validation_logits,
                native_validation_logits,
                validation_split,
                seed=bootstrap_seed + 1,
                samples=bootstrap_samples,
            )
            validation_gates = {
                "strong_behavior": (
                    validation_behavior["strong_passed"] is True
                ),
                "context_bootstrap_upper_at_most_0.0085": (
                    float(validation_bootstrap["upper_95_percent"])
                    <= 0.0085
                ),
                "zero_source_layer_calls": (
                    validation_source_calls
                    == (0,) * model.config.n_layers
                ),
            }
            validation = {
                "baseline": asdict(
                    variable_associative_metrics_from_logits(
                        validation_split,
                        native_validation_logits,
                    )
                ),
                "replacement": validation_behavior,
                "context_bootstrap": validation_bootstrap,
                "source_layer_call_counts": validation_source_calls,
                "gates": validation_gates,
                "passed": all(validation_gates.values()),
            }

    calibration_passed = (
        calibration is not None and calibration["passed"] is True
    )
    validation_passed = (
        validation is not None and validation["passed"] is True
    )
    selected_summary = selection.get("selected")
    five_seed_stability = (
        isinstance(selected_summary, Mapping)
        and int(selected_summary["seed_count"]) == len(DEFAULT_SEEDS)
        and int(selected_summary["strong_seed_count"])
        >= DEFAULT_REQUIRED_STRONG_SEEDS
        and selected_summary["rank_passed"] is True
    )
    analysis = {
        "development_native_projection_ladder": (
            development_projection_ladder
        ),
        "selection_a": {
            "runs": run_records,
            **selection,
        },
        "calibration_b": calibration,
        "fresh_validation": validation,
    }
    scientific_status = {
        "source_model_frozen": True,
        "new_expanded_source_checkpoint": True,
        "new_v2_fisher_basis": True,
        "all_original_task_semantic_contexts_excluded": True,
        "rank_14_v1_hypothesis_rejected_by_v2_development_ceiling": (
            development_projection_ladder[0]["behavior"]["strong_passed"]  # type: ignore[index]
            is False
        ),
        "rank_24_frozen_before_calibration": ranks == (24,),
        "deeper_narrower_graph_frozen_before_calibration": (
            asdict(candidate) == asdict(DEFAULT_CANDIDATE)
        ),
        "five_seed_stability_confirmation": five_seed_stability,
        "full_transformer_span_replaced": selected_executor is not None,
        "source_independent_graph_fitted": selected_executor is not None,
        "calibration_b_evaluated": calibration is not None,
        "calibration_b_passed": calibration_passed,
        "calibration_b_confirmatory": calibration_passed,
        "fresh_validation_evaluated": validation is not None,
        "fresh_validation_passed": validation_passed,
        "executor_test_evaluated": False,
        "fresh_test_contexts_hash_only": True,
        "source_training_checkpoint_contains_native_test_metrics": True,
        "zero_source_layer_calls_in_replacement": (
            calibration is not None
            and calibration["gates"]["zero_source_layer_calls"] is True  # type: ignore[index]
        ),
        "ideal_compute_storage_and_total_parameter_reduction_achieved": (
            calibration is not None
            and calibration["gates"][  # type: ignore[index]
                "ideal_complete_macs_at_most_90_percent_native"
            ]
            is True
            and calibration["gates"][  # type: ignore[index]
                "stored_coefficients_at_most_90_percent_native_span"
            ]
            is True
            and calibration["gates"][  # type: ignore[index]
                "total_deployed_parameters_at_most_90_percent_source"
            ]
            is True
        ),
        "reference_kernel_wall_clock_speed_claim": False,
        "scope_is_query_sparse_associative_recall_not_general_lm": True,
        "model_level_viable": validation_passed,
    }

    checkpoint_hash = _file_sha256(checkpoint_path)
    source = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model_state_fingerprint": module_state_fingerprint(model),
        "dataset_sha256": splits.dataset_sha256,
        "task_fingerprint": splits.task_config.fingerprint,
        "model_config": asdict(model.config),
        "task_config": asdict(splits.task_config),
        "checkpoint_best_step": checkpoint_metadata["best_step"],
        "checkpoint_final_step": checkpoint_metadata["final_step"],
        "checkpoint_converged": checkpoint_metadata["converged"],
        "checkpoint_train_metrics": checkpoint_metadata["train_metrics"],
        "checkpoint_validation_metrics": (
            checkpoint_metadata["validation_metrics"]
        ),
        "checkpoint_native_test_metrics": checkpoint_metadata[
            "test_metrics"
        ],
    }
    hypothesis_selection = _required_mapping(
        _required_mapping(
            hypothesis_report["analysis"],
            name="V1 hypothesis analysis",
        )["selection_a"],
        name="V1 hypothesis selection",
    )
    hypothesis = {
        "artifact": str(hypothesis_path),
        "artifact_sha256": _file_sha256(hypothesis_path),
        "report_sha256": _file_sha256(
            hypothesis_path.with_suffix(".json")
        ),
        "schema": hypothesis_report["schema"],
        "selected": hypothesis_selection["selected"],
        "scientific_status": hypothesis_report["scientific_status"],
        "role": (
            "development-only hypothesis provenance; no V1 context is "
            "eligible for V2 compiler fitting or evaluation"
        ),
    }
    manifest = protocol_manifest
    role_hashes = _required_mapping(
        manifest["role_context_hashes"],
        name="V2 manifest role hashes",
    )
    protocol = {
        **manifest,
        "input_site": DEFAULT_INPUT_SITE,
        "output_site": output_site,
        "source_layers": model.config.n_layers,
        "fisher_scope": "supervised answer row at output boundary",
        "retained_rank_candidates": ranks,
        "rank_development_policy": (
            "V1 development proposed rank 14; the exact native V2 projection "
            "ceiling on graph_select_a rejected ranks 14 and 18, while rank "
            "24 passed every strong gate; rank 24 then froze before calibration"
        ),
        "development_exposure_ledger": {
            "protected_panels_evaluated": False,
            "accessed_roles": (
                "basis_fit_a",
                "graph_fit_a",
                "graph_stop_a",
                "graph_select_a",
            ),
            "rank_14_initial_recipe": (
                "failed three stop-panel seeds; 4-of-5 became impossible"
            ),
            "native_projection_ladder": (
                "rank 14 and 18 ceilings failed; rank 24 passed"
            ),
            "rank_24_v1_loss": (
                "failed two stop-panel seeds; 4-of-5 became impossible"
            ),
            "moderate_distillation": (
                "modal .05, CE 1, KL 2 failed the p90 tail gate"
            ),
            "heavy_distillation": (
                "modal .025, CE .5, KL 4 passed two of three probe seeds"
            ),
            "selected_teacher_dominant_recipe": (
                "modal .05, CE .25, KL 4 retained after the authenticated "
                "L2 rerun reached only two of five selection passes"
            ),
            "architecture_sweep": (
                "L3/H24/FF48 was the best same-budget development geometry; "
                "L4/H20 and L3/H24/FF64 remained seed-unstable"
            ),
            "checkpoint_rule_audit": (
                "KL-first stop selection recovered late strong checkpoints "
                "that absolute-NLL-first selection discarded"
            ),
            "weight_averaging_sweep": (
                "EMA decay .99 missed exact selection; decay .995 passed "
                "both declared hard seeds and froze before calibration"
            ),
            "calibration_b_evaluated": False,
            "reserve_evaluated": False,
            "fresh_validation_evaluated": False,
            "fresh_test_evaluated": False,
        },
        "candidate": asdict(candidate),
        "seeds": seed_values,
        "rank_pass_rule": (
            f"at least {required_strong_seeds} of {len(seed_values)} "
            "graph_select_a seeds pass every strong gate"
        ),
        "checkpoint_rule": (
            "strong pass, minimum pass, teacher KL, p90 absolute NLL "
            "degradation, absolute mean NLL degradation, later step"
        ),
        "seed_rule": (
            "upper-median hard-NLL strong seed after the rank passes"
        ),
        "rank_rule": (
            "rank 24 is fixed; generic fail-closed selector would use lowest "
            "ideal MACs, runtime storage, median NLL, and rank"
        ),
        "training": asdict(training),
        "executor_parameter_ema_decay": ema_decay,
        "deterministic_algorithms": True,
        "strong_behavior_thresholds": {
            "delta_nll": 0.007,
            "answer_accuracy": 1.0,
            "paired_context_accuracy": 1.0,
            "minimum_layout_query_order_length_accuracy": 1.0,
            "top1_agreement": 1.0,
            "native_teacher_kl": 0.007,
            "p90_absolute_delta_nll": 0.020,
            "new_key_accuracy": 1.0,
            "new_value_accuracy": 1.0,
            "new_key_only_accuracy": 1.0,
            "new_value_only_accuracy": 1.0,
            "new_key_and_value_accuracy": 1.0,
            "minimum_queried_key_accuracy": 1.0,
            "minimum_answer_value_accuracy": 1.0,
        },
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_upper_degradation_threshold": 0.0085,
        "maximum_relative_work": maximum_relative_work,
        "maximum_relative_storage": maximum_relative_storage,
        "hash_set_digests": {
            "excluded_baseline": _hash_list_digest(
                manifest["excluded_baseline_context_hashes"]  # type: ignore[arg-type]
            ),
            **{
                name: _hash_list_digest(hashes)  # type: ignore[arg-type]
                for name, hashes in role_hashes.items()
            },
            "reserve": _hash_list_digest(
                manifest["reserve_context_hashes"]  # type: ignore[arg-type]
            ),
            "fresh_validation": _hash_list_digest(
                manifest["fresh_validation_context_hashes"]  # type: ignore[arg-type]
            ),
            "fresh_test": _hash_list_digest(
                manifest["fresh_test_context_hashes"]  # type: ignore[arg-type]
            ),
        },
        "calibration_policy": (
            "unopened until basis, loss, rank panel, seed, checkpoint, "
            "coordinate scale, and decoder are frozen"
        ),
        "validation_policy": (
            "fresh-only validation evaluated once after joint calibration pass"
        ),
        "executor_test_policy": (
            "fresh-only test retained as hashes and never evaluated here"
        ),
        "reference_kernel_policy": (
            "dense PyTorch correctness reference; ideal logical MACs only"
        ),
    }
    return _save_result(
        output=Path(output),
        source=source,
        hypothesis=hypothesis,
        protocol=protocol,
        basis=basis,
        coordinate_scale=selected_scale,
        executor_artifact=selected_artifact,
        analysis=analysis,
        scientific_status=scientific_status,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the clean expanded-task V2 replication of the source-free "
            "full-span transformer graph."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--hypothesis-artifact",
        type=Path,
        default=DEFAULT_HYPOTHESIS_ARTIFACT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_variable_static_full_span_v2_experiment(
        checkpoint=arguments.checkpoint,
        hypothesis_artifact=arguments.hypothesis_artifact,
        output=arguments.output,
        progress=None if arguments.quiet else print,
    )
    selection = report["analysis"]["selection_a"]  # type: ignore[index]
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "selection": {
                    "ranks": selection["ranks"],  # type: ignore[index]
                    "selected": selection["selected"],  # type: ignore[index]
                },
                "calibration_b": report["analysis"]["calibration_b"],  # type: ignore[index]
                "fresh_validation": report["analysis"]["fresh_validation"],  # type: ignore[index]
                "scientific_status": report["scientific_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
